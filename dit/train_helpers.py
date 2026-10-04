from __future__ import annotations

import argparse
import importlib
import importlib.util
import math
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm.auto import tqdm

try:
    from accelerate import Accelerator, DeepSpeedPlugin
    from accelerate.utils import ProjectConfiguration, set_seed
    from accelerate.utils.deepspeed import DummyOptim, DummyScheduler
except ImportError:
    Accelerator = None
    DeepSpeedPlugin = None
    ProjectConfiguration = None
    DummyOptim = None
    DummyScheduler = None
    set_seed = None

from dataset import FirstFrameGenSuperpointChunkDataset
from flow import add_flow_noise, position_lead_timestep, sample_flow_timesteps
from vae_bridge import load_json_config, load_model as load_frozen_vae_model


def load_config(path: Path):
    spec = importlib.util.spec_from_file_location("dit_v4_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model(cfg):
    module_name = str(getattr(cfg, "model_module", "models.stage1"))
    class_name = str(getattr(cfg, "model_class", "TwoStagePositionDiTV18V3"))
    module = importlib.import_module(module_name)
    model_cls = getattr(module, class_name)
    return model_cls(**dict(cfg.model))


def build_dataset(kwargs: dict[str, Any]) -> Dataset:
    datasets_cfg = kwargs.pop("datasets", None)
    if datasets_cfg is None:
        return FirstFrameGenSuperpointChunkDataset(**kwargs)
    if not datasets_cfg:
        raise ValueError("dataset.datasets must contain at least one dataset config")
    common = dict(kwargs)
    datasets = []
    for ds_cfg in datasets_cfg:
        merged = dict(common)
        merged.update(dict(ds_cfg))
        datasets.append(FirstFrameGenSuperpointChunkDataset(**merged))
    return ConcatDataset(datasets)


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    err = (pred - target).pow(2).mean(dim=-1)
    if weight is not None:
        err = err * weight.view(-1, 1)
    m = mask.float()
    return (err * m).sum() / m.sum().clamp_min(1.0)


def masked_mse_per_sample(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    err = (pred - target).pow(2).mean(dim=-1)
    if weight is not None:
        err = err * weight.view(-1, 1)
    m = mask.float()
    return (err * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


def fixed_flow_noise_by_sample_id(
    x0_pos: torch.Tensor,
    x0_con: torch.Tensor,
    mask: torch.Tensor,
    sample_ids: torch.Tensor,
    flow_cfg: dict[str, Any],
    seed: int,
    timestep_mode: str = "independent",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    t_eps = float(flow_cfg.get("t_eps", 1e-4))
    t_pos_vals = []
    t_con_vals = []
    eps_pos_vals = []
    eps_con_vals = []
    mode = str(timestep_mode).lower()
    beta = float(flow_cfg.get("position_lead_beta", 0.5))
    for sample_id in sample_ids.detach().cpu().tolist():
        generator = torch.Generator(device=x0_pos.device)
        generator.manual_seed(int(seed) + int(sample_id))
        if mode == "position_leading":
            t_con = torch.rand((), device=x0_pos.device, generator=generator).clamp(t_eps, 1.0)
            t_pos = position_lead_timestep(t_con, beta).clamp(t_eps, 1.0)
        elif mode == "independent":
            t_pos = torch.rand((), device=x0_pos.device, generator=generator).clamp(t_eps, 1.0)
            t_con = torch.rand((), device=x0_pos.device, generator=generator).clamp(t_eps, 1.0)
        else:
            raise ValueError(f"Unsupported validation timestep mode={timestep_mode!r}")
        t_pos_vals.append(t_pos)
        t_con_vals.append(t_con)
        eps_pos_vals.append(torch.randn(x0_pos[0].shape, device=x0_pos.device, dtype=x0_pos.dtype, generator=generator))
        eps_con_vals.append(torch.randn(x0_con[0].shape, device=x0_con.device, dtype=x0_con.dtype, generator=generator))
    t_pos = torch.stack(t_pos_vals)
    t_con = torch.stack(t_con_vals)
    eps_pos = torch.stack(eps_pos_vals)
    eps_con = torch.stack(eps_con_vals)
    t_pos_view = t_pos.view(-1, 1, 1)
    t_con_view = t_con.view(-1, 1, 1)
    xt_pos = (1.0 - t_pos_view) * x0_pos + t_pos_view * eps_pos
    xt_con = (1.0 - t_con_view) * x0_con + t_con_view * eps_con
    v_pos = eps_pos - x0_pos
    v_con = eps_con - x0_con
    xt_pos = xt_pos.masked_fill(~mask.unsqueeze(-1), 0.0)
    xt_con = xt_con.masked_fill(~mask.unsqueeze(-1), 0.0)
    return t_pos, t_con, xt_pos, xt_con, v_pos, v_con


def cosine_lr(step: int, max_steps: int, lr: float, warmup: int) -> float:
    if warmup > 0 and step < warmup:
        return lr * float(step + 1) / float(warmup)
    p = float(step - warmup) / float(max(max_steps - warmup, 1))
    return lr * 0.5 * (1.0 + math.cos(math.pi * min(max(p, 0.0), 1.0)))


def scheduled_lr(step: int, max_steps: int, train_cfg: dict[str, Any]) -> float:
    lr = float(train_cfg["lr"])
    warmup = int(train_cfg.get("warmup_steps", 0))
    if warmup > 0 and step < warmup:
        return lr * float(step + 1) / float(warmup)
    schedule_type = str(train_cfg.get("schedule_type", "cosine")).lower()
    if schedule_type in ("constant", "none"):
        return lr
    if schedule_type == "cosine":
        return cosine_lr(step, max_steps, lr, warmup)
    raise ValueError(f"Unsupported schedule_type={schedule_type!r}")


def chamfer_loss(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    vals = []
    for xb, yb, mb in zip(x, y, mask, strict=True):
        xb = xb[mb]
        yb = yb[mb]
        if xb.numel() == 0 or yb.numel() == 0:
            continue
        d = torch.cdist(xb.float(), yb.float())
        vals.append(d.min(dim=1).values.mean() + d.min(dim=0).values.mean())
    if not vals:
        return x.new_zeros(())
    return torch.stack(vals).mean()


def repulsion_loss(x: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    vals = []
    for xb, mb in zip(x, mask, strict=True):
        xb = xb[mb]
        if xb.shape[0] <= k:
            continue
        d = torch.cdist(xb.float(), xb.float())
        knn = d.topk(k + 1, largest=False).values[:, 1:]
        h = knn.mean(dim=1, keepdim=True).detach()
        vals.append(F.relu(h - knn).pow(2).mean())
    if not vals:
        return x.new_zeros(())
    return torch.stack(vals).mean()


def build_frozen_vae_head(cfg, device: torch.device):
    reg_cfg = cfg.regularization
    if float(reg_cfg.get("vae_head_weight", 0.0)) <= 0:
        return None
    vae_cfg_path = Path(str(reg_cfg["vae_head_config"]))
    vae_cfg = load_json_config(vae_cfg_path).get("vae", {})
    vae_cfg["device"] = str(device)
    if bool(reg_cfg.get("vae_head_lightweight", True)):
        try:
            vae = load_lightweight_vae_heads(vae_cfg, device)
            vae.eval().requires_grad_(False)
            return vae
        except Exception as exc:
            print(f"[warn] lightweight VAE head load failed, falling back to full VAE: {type(exc).__name__}: {exc}")
    vae = load_frozen_vae_model(vae_cfg, device)
    vae.eval().requires_grad_(False)
    return vae


class LightweightVAEHeads(torch.nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, vae_latent_channels: int = 16) -> None:
        super().__init__()
        self.sp_vae_texture_head = torch.nn.Sequential(torch.nn.Linear(latent_dim, hidden_dim), torch.nn.GELU(), torch.nn.Linear(hidden_dim, vae_latent_channels))
        self.geometry_radius_head = torch.nn.Sequential(torch.nn.Linear(latent_dim, hidden_dim), torch.nn.GELU(), torch.nn.Linear(hidden_dim, 1))
        self.geometry_eigvals_head = torch.nn.Sequential(torch.nn.Linear(latent_dim, hidden_dim), torch.nn.GELU(), torch.nn.Linear(hidden_dim, 3))
        self.geometry_normal_head = torch.nn.Sequential(torch.nn.Linear(latent_dim, hidden_dim), torch.nn.GELU(), torch.nn.Linear(hidden_dim, 3))
        self.geometry_scalar_head = torch.nn.Sequential(torch.nn.Linear(latent_dim, hidden_dim), torch.nn.GELU(), torch.nn.Linear(hidden_dim, 6))


def load_lightweight_vae_heads(vae_cfg: dict[str, Any], device: torch.device) -> LightweightVAEHeads:
    ckpt_path = Path(str(vae_cfg["checkpoint"]))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    first_key = "sp_vae_texture_head.0.weight"
    if first_key not in state:
        raise KeyError(f"{ckpt_path} missing {first_key}")
    hidden_dim, latent_dim = map(int, state[first_key].shape)
    vae_latent_channels = int(state["sp_vae_texture_head.2.weight"].shape[0])
    model = LightweightVAEHeads(latent_dim=latent_dim, hidden_dim=hidden_dim, vae_latent_channels=vae_latent_channels)
    head_prefixes = (
        "sp_vae_texture_head",
        "geometry_radius_head",
        "geometry_eigvals_head",
        "geometry_normal_head",
        "geometry_scalar_head",
    )
    head_state = {k: v for k, v in state.items() if any(k.startswith(prefix + ".") for prefix in head_prefixes)}
    missing, unexpected = model.load_state_dict(head_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"lightweight VAE head load mismatch: missing={missing} unexpected={unexpected}")
    print(f"[model] loaded lightweight VAE heads from {ckpt_path} latent_dim={latent_dim} hidden_dim={hidden_dim}")
    return model.to(device)


def vae_head_consistency_loss(vae, x0hat_con_std: torch.Tensor, batch: dict[str, torch.Tensor], reg_cfg: dict[str, float]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = x0hat_con_std.new_zeros(())
    logs: dict[str, torch.Tensor] = {
        "head": zero,
        "head_texture": zero,
        "head_radius": zero,
        "head_eigvals": zero,
        "head_normal": zero,
        "head_scalar": zero,
    }
    if vae is None or float(reg_cfg.get("vae_head_weight", 0.0)) <= 0:
        return zero, logs

    z_sp = x0hat_con_std * batch["zsp_std"].unsqueeze(1) + batch["zsp_mean"].unsqueeze(1)
    mask = batch["target_mask"].bool()
    if "target_geom_valid" in batch:
        mask = mask & batch["target_geom_valid"].bool()
    if not mask.any():
        return zero, logs

    losses = []
    dtype = z_sp.dtype

    tex_w = float(reg_cfg.get("vae_head_texture_weight", 1.0))
    if tex_w > 0 and "target_sp_vae" in batch:
        pred = vae.sp_vae_texture_head(z_sp)
        gt = torch.nan_to_num(batch["target_sp_vae"].to(dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0)
        tex_mask = mask & torch.isfinite(gt).all(dim=-1) & (gt.abs().sum(dim=-1) > 0)
        if tex_mask.any():
            tex_loss = z_sp.new_zeros(())
            cos_w = float(reg_cfg.get("vae_head_texture_cos_weight", 1.0))
            mse_w = float(reg_cfg.get("vae_head_texture_mse_weight", 0.0))
            if cos_w > 0:
                tex_loss = tex_loss + cos_w * (1.0 - F.cosine_similarity(F.normalize(pred[tex_mask], dim=-1, eps=1e-6), F.normalize(gt[tex_mask], dim=-1, eps=1e-6), dim=-1)).mean()
            if mse_w > 0:
                tex_loss = tex_loss + mse_w * F.mse_loss(pred[tex_mask], gt[tex_mask])
            losses.append(tex_w * tex_loss)
            logs["head_texture"] = tex_loss.detach()

    radius_w = float(reg_cfg.get("vae_head_radius_weight", 1.0))
    if radius_w > 0 and "target_sp_radius_p90_norm" in batch:
        pred = vae.geometry_radius_head(z_sp)
        gt = batch["target_sp_radius_p90_norm"].to(dtype=dtype).unsqueeze(-1)
        m = mask & torch.isfinite(gt.squeeze(-1))
        if m.any():
            l = F.smooth_l1_loss(pred[m], gt[m])
            losses.append(radius_w * l)
            logs["head_radius"] = l.detach()

    eig_w = float(reg_cfg.get("vae_head_eigvals_weight", 1.0))
    if eig_w > 0 and "target_sp_cov_eigvals_norm" in batch:
        pred = F.softplus(vae.geometry_eigvals_head(z_sp))
        gt = batch["target_sp_cov_eigvals_norm"].to(dtype=dtype).clamp_min(0.0)
        m = mask & torch.isfinite(gt).all(dim=-1)
        if m.any():
            l = F.smooth_l1_loss(pred[m], gt[m])
            losses.append(eig_w * l)
            logs["head_eigvals"] = l.detach()

    normal_w = float(reg_cfg.get("vae_head_normal_weight", 0.5))
    if normal_w > 0 and "target_sp_normal" in batch:
        pred = F.normalize(vae.geometry_normal_head(z_sp), dim=-1, eps=1e-6)
        gt = F.normalize(batch["target_sp_normal"].to(dtype=dtype), dim=-1, eps=1e-6)
        m = mask & torch.isfinite(gt).all(dim=-1)
        if m.any():
            l = (1.0 - (pred[m] * gt[m]).sum(dim=-1).abs()).mean()
            losses.append(normal_w * l)
            logs["head_normal"] = l.detach()

    scalar_w = float(reg_cfg.get("vae_head_scalar_weight", 0.25))
    scalar_fields = [
        "target_sp_linearity",
        "target_sp_planarity",
        "target_sp_scattering",
        "target_sp_anisotropy",
        "target_sp_curvature",
        "target_sp_omnivariance_norm",
    ]
    if scalar_w > 0 and all(k in batch for k in scalar_fields):
        pred = vae.geometry_scalar_head(z_sp)
        gt = torch.stack([batch[k].to(dtype=dtype) for k in scalar_fields], dim=-1)
        m = mask & torch.isfinite(gt).all(dim=-1)
        if m.any():
            l = F.smooth_l1_loss(pred[m], gt[m])
            losses.append(scalar_w * l)
            logs["head_scalar"] = l.detach()

    if not losses:
        return zero, logs
    total = torch.stack(losses).sum()
    logs["head"] = total.detach()
    return total, logs


def scheduled_vae_head_weight(base_weight: float, reg_cfg: dict[str, Any], global_step: int | None) -> tuple[float, float]:
    if base_weight <= 0:
        return 0.0, 0.0
    if global_step is None:
        return base_weight, 1.0
    start_steps = int(reg_cfg.get("vae_head_warmup_start_steps", 0) or 0)
    warmup_steps = int(reg_cfg.get("vae_head_warmup_steps", 0) or 0)
    if int(global_step) < start_steps:
        return 0.0, 0.0
    if warmup_steps <= 0:
        return base_weight, 1.0
    progress = min(1.0, max(0.0, float(int(global_step) - start_steps) / float(warmup_steps)))
    return base_weight * progress, progress


def model_coordinate_kwargs(batch: dict[str, Any], cfg) -> dict[str, torch.Tensor]:
    if not bool(getattr(cfg, "model_requires_xyz_stats", False)):
        return {}
    missing = [key for key in ("xyz_mean", "xyz_std") if key not in batch]
    if missing:
        raise KeyError(f"Model requires coordinate statistics missing from batch: {missing}")
    return {"xyz_mean": batch["xyz_mean"], "xyz_std": batch["xyz_std"]}


def model_step(
    model,
    batch,
    cfg,
    selfcond: bool = True,
    frozen_vae_head=None,
    global_step: int | None = None,
    deterministic_seed: int | None = None,
    deterministic_sample_ids: torch.Tensor | None = None,
    validation_timestep_mode: str = "independent",
    force_drop_cond: bool | None = None,
    force_drop_cam: bool | None = None,
    return_per_sample: bool = False,
):
    flow_cfg = cfg.flow
    x0_pos = batch["target_pos"]
    x0_con = batch["target_con"]
    mask = batch["target_mask"].bool()
    b = x0_pos.shape[0]
    generator = None
    if deterministic_sample_ids is not None:
        seed = 12345 if deterministic_seed is None else int(deterministic_seed)
        t_pos, t_con, xt_pos, xt_con, v_pos_tgt, v_con_tgt = fixed_flow_noise_by_sample_id(
            x0_pos, x0_con, mask, deterministic_sample_ids, flow_cfg, seed, validation_timestep_mode
        )
    else:
        if deterministic_seed is not None:
            generator = torch.Generator(device=x0_pos.device)
            generator.manual_seed(int(deterministic_seed))
        t_pos, t_con = sample_flow_timesteps(b, x0_pos.device, flow_cfg, generator=generator)
        xt_pos, xt_con, v_pos_tgt, v_con_tgt = add_flow_noise(x0_pos, x0_con, mask, t_pos, t_con, generator=generator)

    drop_cond = bool(force_drop_cond) if force_drop_cond is not None else random.random() < float(flow_cfg.get("cond_drop_prob", 0.0))
    drop_cam = bool(force_drop_cam) if force_drop_cam is not None else random.random() < float(flow_cfg.get("cam_drop_prob", 0.0))
    coordinate_kwargs = model_coordinate_kwargs(batch, cfg)
    x0hat = None
    if selfcond and random.random() < float(flow_cfg.get("selfcond_prob", 0.0)):
        with torch.no_grad():
            sc = model(
                xt_pos,
                xt_con,
                t_pos,
                t_con,
                mask,
                batch["cond_pos"],
                batch["cond_con"],
                batch["cond_mask"].bool(),
                batch["norm_camtoworlds"],
                batch["Ks"],
                batch["cam_mask"].bool(),
                x0hat_pos=None,
                drop_cond=drop_cond,
                drop_cam=drop_cam,
                **coordinate_kwargs,
            )
            x0hat = (xt_pos - t_pos.view(-1, 1, 1) * sc["v_pos"]).detach()

    out = model(
        xt_pos,
        xt_con,
        t_pos,
        t_con,
        mask,
        batch["cond_pos"],
        batch["cond_con"],
        batch["cond_mask"].bool(),
        batch["norm_camtoworlds"],
        batch["Ks"],
        batch["cam_mask"].bool(),
        x0hat_pos=x0hat,
        drop_cond=drop_cond,
        drop_cam=drop_cam,
        **coordinate_kwargs,
    )
    pos_weight = t_pos.pow(2) if bool(flow_cfg.get("pos_t2_weight", False)) else None
    loss_pos = masked_mse(out["v_pos"], v_pos_tgt, mask, pos_weight)
    loss_con = masked_mse(out["v_con"], v_con_tgt, mask)
    loss = float(flow_cfg.get("lambda_pos", 1.0)) * loss_pos + float(flow_cfg.get("lambda_con", 1.0)) * loss_con

    reg_cfg = cfg.regularization
    logs = {
        "loss_pos": loss_pos.detach(),
        "loss_con": loss_con.detach(),
        "loss_flow": loss.detach(),
        "t_pos_mean": t_pos.mean().detach(),
        "t_con_mean": t_con.mean().detach(),
        "t_pos_leads_fraction": (t_pos < t_con).float().mean().detach(),
    }
    if return_per_sample:
        sample_loss_pos = masked_mse_per_sample(out["v_pos"], v_pos_tgt, mask, pos_weight)
        sample_loss_con = masked_mse_per_sample(out["v_con"], v_con_tgt, mask)
        sample_loss_flow = float(flow_cfg.get("lambda_pos", 1.0)) * sample_loss_pos + float(flow_cfg.get("lambda_con", 1.0)) * sample_loss_con
        logs["sample_loss_pos"] = sample_loss_pos.detach()
        logs["sample_loss_con"] = sample_loss_con.detach()
        logs["sample_loss_flow"] = sample_loss_flow.detach()
        logs["sample_t_pos"] = t_pos.detach()
        logs["sample_t_con"] = t_con.detach()
    c2p_gates = [parameter for name, parameter in model.named_parameters() if name.endswith("c2p_gate")]
    if c2p_gates:
        effective_c2p_gates = torch.stack([torch.tanh(gate.float()) for gate in c2p_gates])
        logs["c2p_gate_mean"] = effective_c2p_gates.mean().detach()
        logs["c2p_gate_abs_mean"] = effective_c2p_gates.abs().mean().detach()
    need_x0hat_pos = float(reg_cfg.get("chamfer_weight", 0.0)) > 0 or float(reg_cfg.get("repulsion_weight", 0.0)) > 0
    vae_head_base_weight = float(reg_cfg.get("vae_head_weight", 0.0))
    vae_head_weight, vae_head_warmup = scheduled_vae_head_weight(vae_head_base_weight, reg_cfg, global_step)
    vae_head_source = str(reg_cfg.get("vae_head_source", "final")).lower()
    need_x0hat_con = vae_head_base_weight > 0
    low_t_power = float(reg_cfg.get("low_t_power", 1.0))
    low_t_pos_w = (1.0 - t_pos).pow(low_t_power).mean() if need_x0hat_pos else None
    low_t_con_w = (1.0 - t_con).pow(low_t_power).mean() if need_x0hat_con else None
    if need_x0hat_pos:
        x0hat_pos = xt_pos - t_pos.view(-1, 1, 1) * out["v_pos"]
        if float(reg_cfg.get("chamfer_weight", 0.0)) > 0:
            cd = chamfer_loss(x0hat_pos, x0_pos, mask)
            loss = loss + float(reg_cfg.get("chamfer_weight", 0.0)) * low_t_pos_w * cd
            logs["loss_cd"] = cd.detach()
        if float(reg_cfg.get("repulsion_weight", 0.0)) > 0:
            rep = repulsion_loss(x0hat_pos, mask, int(reg_cfg.get("repulsion_k", 8)))
            loss = loss + float(reg_cfg.get("repulsion_weight", 0.0)) * low_t_pos_w * rep
            logs["loss_repulsion"] = rep.detach()
    if need_x0hat_con:
        if vae_head_source == "aux":
            if "aux_x0_con" not in out:
                raise RuntimeError("regularization.vae_head_source='aux' requires model.aux_con_layer > 0")
            x0hat_con = out["aux_x0_con"]
        elif vae_head_source == "final":
            x0hat_con = xt_con - t_con.view(-1, 1, 1) * out["v_con"]
        else:
            raise ValueError(f"Unsupported regularization.vae_head_source={vae_head_source!r}")
        if vae_head_weight > 0:
            head_loss, head_logs = vae_head_consistency_loss(frozen_vae_head, x0hat_con, batch, reg_cfg)
            weighted_head_loss = vae_head_weight * low_t_con_w * head_loss
            loss = loss + weighted_head_loss
            logs.update({f"loss_{k}": v for k, v in head_logs.items()})
        else:
            weighted_head_loss = x0hat_con.new_zeros(())
            zero = weighted_head_loss.detach()
            logs.update(
                {
                    "loss_head": zero,
                    "loss_head_texture": zero,
                    "loss_head_radius": zero,
                    "loss_head_eigvals": zero,
                    "loss_head_normal": zero,
                    "loss_head_scalar": zero,
                }
            )
        logs["loss_head_weighted"] = weighted_head_loss.detach()
        logs["loss_head_low_t_w"] = low_t_con_w.detach()
        logs["loss_head_weight"] = x0hat_con.new_tensor(vae_head_weight)
        logs["loss_head_warmup"] = x0hat_con.new_tensor(vae_head_warmup)
    logs["loss"] = loss.detach()
    return loss, logs


def format_loss_extras(values: dict[str, float], denom: float) -> str:
    names = {
        "loss_flow": "flow",
        "loss_head_weighted": "head_final",
        "loss_head_low_t_w": "head_low_t_w",
        "loss_head_weight": "head_w",
        "loss_head_warmup": "head_warmup",
        "loss_head": "head_raw",
        "loss_head_texture": "head_tex",
        "loss_head_radius": "head_radius",
        "loss_head_eigvals": "head_eig",
        "loss_head_normal": "head_normal",
        "loss_head_scalar": "head_scalar",
        "loss_cd": "cd",
        "loss_repulsion": "repulsion",
    }
    ordered = [
        "loss_flow",
        "loss_head_weighted",
        "loss_head_low_t_w",
        "loss_head_weight",
        "loss_head_warmup",
        "loss_head",
        "loss_head_texture",
        "loss_head_radius",
        "loss_head_eigvals",
        "loss_head_normal",
        "loss_head_scalar",
        "loss_cd",
        "loss_repulsion",
    ]
    parts = []
    emitted = set()
    for key in ordered:
        if key in values:
            parts.append(f"{names[key]}={values[key] / denom:.5f}")
            emitted.add(key)
    for key in sorted(values):
        if key in emitted or key in ("loss", "loss_pos", "loss_con"):
            continue
        name = names.get(key, key[5:] if key.startswith("loss_") else key)
        parts.append(f"{name}={values[key] / denom:.5f}")
    return " ".join(parts)


class ModelEma:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            k: v.detach().clone().float().cpu()
            for k, v in model.state_dict().items()
            if torch.is_floating_point(v)
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        for k, v in state.items():
            if k not in self.shadow:
                continue
            self.shadow[k].mul_(self.decay).add_(v.detach().float().cpu(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.decay = float(state.get("decay", self.decay))
        self.shadow = dict(state["shadow"])

    def model_state_dict(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        current = model.state_dict()
        out = {}
        for k, v in current.items():
            if k in self.shadow:
                out[k] = self.shadow[k].to(device=v.device, dtype=v.dtype)
            else:
                out[k] = v.detach().clone()
        return out


def save_model_ckpt(path: Path, model, opt, step: int, cfg_path: Path, accelerator: Accelerator, ema: ModelEma | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = accelerator.unwrap_model(model)
    model_state = raw_model.state_dict()
    if ema is not None:
        model_state = ema.model_state_dict(raw_model)
    payload = {
        "model": model_state,
        "step": step,
        "config": str(cfg_path),
    }
    if ema is not None:
        payload["ema"] = ema.state_dict()
    try:
        payload["optimizer"] = opt.state_dict()
    except Exception:
        pass
    accelerator.save(payload, path)


def latest_accelerate_checkpoint(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.is_dir():
        return None
    candidates = [p for p in checkpoint_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: int(p.name.rsplit("-", 1)[-1]))[-1]


def prune_checkpoints(checkpoint_dir: Path, total_limit: int | None) -> None:
    if total_limit is None or total_limit <= 0 or not checkpoint_dir.is_dir():
        return
    checkpoints = [p for p in checkpoint_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    checkpoints = sorted(checkpoints, key=lambda p: int(p.name.rsplit("-", 1)[-1]))
    excess = len(checkpoints) - int(total_limit) + 1
    for ckpt in checkpoints[: max(excess, 0)]:
        shutil.rmtree(ckpt)


def build_loader(ds: Dataset, train_cfg: dict[str, Any], *, shuffle: bool, drop_last: bool, num_workers: int) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": int(train_cfg["batch_size"]),
        "shuffle": shuffle,
        "num_workers": int(num_workers),
        "pin_memory": bool(train_cfg.get("pin_memory", True)),
        "drop_last": drop_last,
        "persistent_workers": bool(train_cfg.get("persistent_workers", True)) and int(num_workers) > 0,
    }
    if int(num_workers) > 0:
        kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 2))
    return DataLoader(ds, **kwargs)


def env_flag(name: str) -> bool:
    return os.environ.get(name, "false").lower() in ("1", "true", "yes", "on")


def build_deepspeed_config(train_cfg: dict[str, Any]) -> dict[str, Any]:
    mixed_precision = str(train_cfg.get("mixed_precision", "no")).lower()
    zero_stage = int(train_cfg.get("deepspeed_zero_stage", 2))
    offload_optimizer = bool(train_cfg.get("deepspeed_offload_optimizer", True))
    zero_optimization: dict[str, Any] = {
        "stage": zero_stage,
        "overlap_comm": True,
        "contiguous_gradients": True,
    }
    if offload_optimizer:
        zero_optimization["offload_optimizer"] = {
            "device": str(train_cfg.get("deepspeed_offload_optimizer_device", "cpu")),
            "pin_memory": True,
        }
    if zero_stage >= 3 and train_cfg.get("deepspeed_offload_param_device"):
        zero_optimization["offload_param"] = {
            "device": str(train_cfg["deepspeed_offload_param_device"]),
            "pin_memory": True,
        }
    schedule_type = str(train_cfg.get("schedule_type", "cosine")).lower()
    if schedule_type in ("constant", "none"):
        scheduler = {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto",
                "warmup_type": "linear",
            },
        }
    elif schedule_type == "cosine":
        scheduler = {
            "type": "WarmupCosineLR",
            "params": {
                "warmup_num_steps": "auto",
                "warmup_type": "linear",
                "total_num_steps": "auto",
            },
        }
    else:
        raise ValueError(f"Unsupported schedule_type={schedule_type!r}")
    return {
        "bf16": {"enabled": mixed_precision == "bf16"},
        "fp16": {"enabled": mixed_precision == "fp16"},
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto",
            },
        },
        "scheduler": scheduler,
        "train_micro_batch_size_per_gpu": int(train_cfg["batch_size"]),
        "gradient_accumulation_steps": int(train_cfg.get("gradient_accumulation_steps", 1)),
        "gradient_clipping": 0.0,
        "zero_optimization": zero_optimization,
    }


def run_training(
    config_path: Path,
    *,
    resume: Path | None = None,
    resume_from_checkpoint: str | None = None,
    train_overrides: dict[str, Any] | None = None,
    work_dir: Path | None = None,
    seed: int | None = None,
    use_deepspeed: bool | None = None,
) -> None:
    if Accelerator is None:
        raise ImportError("Direct train_dit_v4.py execution requires the 'accelerate' package; use the DDP entrypoint otherwise")
    cfg = load_config(config_path)
    if work_dir is not None:
        cfg.work_dir = str(work_dir)
    if seed is not None:
        cfg.seed = int(seed)
    if train_overrides:
        flow_overrides = train_overrides.pop("_flow_overrides", None)
        cfg.train.update({k: v for k, v in train_overrides.items() if v is not None})
        if flow_overrides:
            cfg.flow.update({k: v for k, v in flow_overrides.items() if v is not None})
    train_cfg = dict(cfg.train)
    work_dir = Path(cfg.work_dir)
    logging_dir = work_dir / str(train_cfg.get("logging_dir", "logs"))
    use_deepspeed = bool(use_deepspeed) or env_flag("ACCELERATE_USE_DEEPSPEED")
    deepspeed_plugin = None
    if use_deepspeed:
        try:
            import deepspeed  # noqa: F401
        except ImportError as exc:
            raise ImportError("DeepSpeed training requested, but the 'deepspeed' package is not installed.") from exc
        deepspeed_plugin = DeepSpeedPlugin(
            hf_ds_config=build_deepspeed_config(train_cfg),
            gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
            gradient_clipping=float(train_cfg["grad_clip"]),
            zero_stage=int(train_cfg.get("deepspeed_zero_stage", 2)),
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
        mixed_precision=str(train_cfg.get("mixed_precision", "fp16" if bool(train_cfg.get("amp", True)) else "no")),
        log_with=train_cfg.get("report_to", None),
        project_config=ProjectConfiguration(project_dir=str(work_dir), logging_dir=str(logging_dir)),
        deepspeed_plugin=deepspeed_plugin,
    )
    if bool(train_cfg.get("allow_tf32", True)) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    set_seed(int(cfg.seed) + accelerator.process_index)
    np.random.seed(int(cfg.seed) + accelerator.process_index)
    random.seed(int(cfg.seed) + accelerator.process_index)
    device = accelerator.device
    if accelerator.is_main_process:
        work_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    train_ds = build_dataset(dict(cfg.dataset))
    train_loader = build_loader(train_ds, train_cfg, shuffle=True, drop_last=True, num_workers=int(train_cfg["num_workers"]))
    val_ds = build_dataset(dict(cfg.val_dataset))
    val_loader = (
        build_loader(
            val_ds,
            train_cfg,
            shuffle=False,
            drop_last=False,
            num_workers=max(0, min(int(train_cfg.get("val_num_workers", 2)), int(train_cfg["num_workers"]))),
        )
        if len(val_ds)
        else None
    )

    model = build_model(cfg).to(device)
    ema_decay = float(train_cfg.get("ema_decay", 0.0) or 0.0)
    ema = ModelEma(model, ema_decay) if ema_decay > 0.0 else None
    if ema is not None:
        accelerator.register_for_checkpointing(ema)
    frozen_vae_head = build_frozen_vae_head(cfg, device)
    if use_deepspeed:
        opt = DummyOptim(
            model.parameters(),
            lr=float(train_cfg["lr"]),
            betas=tuple(train_cfg.get("adam_betas", (0.9, 0.999))),
            eps=float(train_cfg.get("adam_eps", 1e-8)),
            weight_decay=float(train_cfg["weight_decay"]),
        )
        lr_scheduler = DummyScheduler(
            opt,
            warmup_num_steps=int(train_cfg["warmup_steps"]),
            total_num_steps=int(train_cfg["max_steps"]),
        )
    else:
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg["lr"]),
            betas=tuple(train_cfg.get("adam_betas", (0.9, 0.999))),
            eps=float(train_cfg.get("adam_eps", 1e-8)),
            weight_decay=float(train_cfg["weight_decay"]),
        )
        lr_scheduler = None
    if val_loader is None:
        if use_deepspeed:
            model, opt, train_loader, lr_scheduler = accelerator.prepare(model, opt, train_loader, lr_scheduler)
        else:
            model, opt, train_loader = accelerator.prepare(model, opt, train_loader)
    else:
        if use_deepspeed:
            model, opt, train_loader, val_loader, lr_scheduler = accelerator.prepare(model, opt, train_loader, val_loader, lr_scheduler)
        else:
            model, opt, train_loader, val_loader = accelerator.prepare(model, opt, train_loader, val_loader)
    if frozen_vae_head is not None:
        frozen_vae_head.to(device)

    start_step = 0
    resume_from_checkpoint = resume_from_checkpoint or train_cfg.get("resume_from_checkpoint", None)
    if resume_from_checkpoint:
        if str(resume_from_checkpoint) == "latest":
            resume_path = latest_accelerate_checkpoint(work_dir / "checkpoints")
        else:
            resume_path = Path(str(resume_from_checkpoint))
        if resume_path is not None and resume_path.is_dir():
            accelerator.print(f"Resuming training state from {resume_path}")
            accelerator.load_state(str(resume_path), map_location="cpu")
            start_step = int(resume_path.name.rsplit("-", 1)[-1])
        else:
            accelerator.print(f"Accelerate checkpoint {resume_from_checkpoint!r} not found; starting normally.")

    resume = resume or (Path(train_cfg["resume"]) if train_cfg.get("resume") else None)
    if start_step == 0 and resume and Path(resume).is_file():
        ckpt = torch.load(resume, map_location="cpu")
        accelerator.unwrap_model(model).load_state_dict(ckpt["model"], strict=True)
        if ema is not None and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        if "optimizer" in ckpt and not use_deepspeed:
            opt.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))

    step = start_step
    loader_iter = iter(train_loader)
    model.train()
    total_batch = int(train_cfg["batch_size"]) * accelerator.num_processes * int(train_cfg.get("gradient_accumulation_steps", 1))
    if accelerator.is_main_process:
        print("***** Running training *****")
        print(f"  Num examples = {len(train_ds)}")
        print(f"  Num validation examples = {len(val_ds)}")
        print(f"  Per-device batch size = {int(train_cfg['batch_size'])}")
        print(f"  Total batch size = {total_batch}")
        print(f"  Gradient accumulation steps = {int(train_cfg.get('gradient_accumulation_steps', 1))}")
        print(f"  Max steps = {int(train_cfg['max_steps'])}")
        print(f"  Mixed precision = {accelerator.mixed_precision}")
        print(f"  DeepSpeed = {use_deepspeed}")

    progress = tqdm(
        range(int(train_cfg["max_steps"])),
        initial=step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    running_loss = 0.0
    running_pos = 0.0
    running_con = 0.0
    accum_logs: dict[str, float] = {}
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
    while step < int(train_cfg["max_steps"]):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)
        batch = to_device(batch, device)

        with accelerator.accumulate(model):
            if use_deepspeed:
                try:
                    lr = float(lr_scheduler.get_last_lr()[0]) if lr_scheduler is not None else float(train_cfg["lr"])
                except Exception:
                    lr = float(train_cfg["lr"])
            else:
                lr = scheduled_lr(step, int(train_cfg["max_steps"]), train_cfg)
                for group in opt.param_groups:
                    group["lr"] = lr
            with autocast(device_type=device.type, enabled=bool(train_cfg.get("amp", True)) and device.type == "cuda"):
                loss, logs = model_step(model, batch, cfg, selfcond=True, frozen_vae_head=frozen_vae_head, global_step=step)
            avg_loss = accelerator.reduce(loss.detach(), reduction="mean")
            if use_deepspeed and accelerator.sync_gradients:
                max_grad_norm = float(train_cfg["grad_clip"])
                engine = accelerator.deepspeed_engine_wrapped.engine
                engine._config.gradient_clipping = max_grad_norm
                if hasattr(engine, "optimizer"):
                    engine.optimizer.clip_grad = max_grad_norm
            accelerator.backward(loss)
            if accelerator.sync_gradients and not use_deepspeed:
                accelerator.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
            opt.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            if accelerator.sync_gradients and ema is not None:
                ema.update(accelerator.unwrap_model(model))
            opt.zero_grad(set_to_none=True)

        running_loss += avg_loss.item() / grad_accum
        running_pos += accelerator.reduce(logs["loss_pos"], reduction="mean").item() / grad_accum
        running_con += accelerator.reduce(logs["loss_con"], reduction="mean").item() / grad_accum
        for k in sorted(logs):
            if k in ("loss", "loss_pos", "loss_con"):
                continue
            v = logs[k]
            accum_logs[k] = accum_logs.get(k, 0.0) + accelerator.reduce(v.detach(), reduction="mean").item() / grad_accum

        if accelerator.sync_gradients:
            step += 1
            progress.update(1)
            if step % int(train_cfg["log_every"]) == 0:
                denom = float(int(train_cfg["log_every"]))
                extras = format_loss_extras(accum_logs, denom)
                extras_s = " " + extras if extras else ""
                accelerator.print(
                    f"step={step} lr={lr:.3e} loss={running_loss / denom:.5f} "
                    f"pos={running_pos / denom:.5f} con={running_con / denom:.5f}{extras_s}"
                )
                running_loss = 0.0
                running_pos = 0.0
                running_con = 0.0
                accum_logs = {}

            if step % int(train_cfg["save_every"]) == 0:
                accelerator.wait_for_everyone()
                checkpoint_dir = work_dir / "checkpoints"
                if accelerator.is_main_process:
                    prune_checkpoints(checkpoint_dir, train_cfg.get("checkpoints_total_limit", None))
                save_path = checkpoint_dir / f"checkpoint-{step}"
                accelerator.save_state(str(save_path))
                if accelerator.is_main_process:
                    save_model_ckpt(work_dir / f"step_{step:07d}.pt", model, opt, step, config_path, accelerator, ema=ema)
                    save_model_ckpt(work_dir / "latest.pt", model, opt, step, config_path, accelerator, ema=ema)

            if val_loader is not None and step % int(train_cfg["val_every"]) == 0:
                model.eval()
                vals = []
                with torch.no_grad():
                    for i, vb in enumerate(val_loader):
                        if i >= int(train_cfg.get("max_val_batches", 8)):
                            break
                        vb = to_device(vb, device)
                        val_seed = int(train_cfg.get("validation_seed", 12345)) + i
                        vloss, _ = model_step(
                            model,
                            vb,
                            cfg,
                            selfcond=False,
                            frozen_vae_head=frozen_vae_head,
                            global_step=step,
                            deterministic_seed=val_seed,
                            force_drop_cond=bool(train_cfg.get("validation_drop_cond", False)),
                            force_drop_cam=bool(train_cfg.get("validation_drop_cam", False)),
                        )
                        vals.append(accelerator.reduce(vloss.detach(), reduction="mean"))
                if vals:
                    accelerator.print(f"val step={step} loss={torch.stack(vals).mean().item():.5f}")
                model.train()

        if step >= int(train_cfg["max_steps"]):
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_model_ckpt(work_dir / "latest.pt", model, opt, step, config_path, accelerator, ema=ema)
    accelerator.end_training()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/stage1.py"))
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--resume-from-checkpoint", type=str, default=None, help="Accelerate checkpoint dir or 'latest'.")
    args = ap.parse_args()
    run_training(args.config, resume=args.resume, resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
