"""GLD/DDP training entrypoint for the Wan-style content DiT v26.

Independent copy of the v23 trainer.  Residual / content losses stay
ungated (``v = stage1 - gt``, ``res = t_pos * v``).  v26 adds a detached
confidence BCE: ``y = exp(-||stage1-gt|| / gate_sigma)`` vs
``sigmoid(conf_logit)``.  The gate is applied only at inference.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import train_common as gld
from flow import add_flow_noise, sample_flow_timesteps
from train_helpers import (
    chamfer_loss,
    fixed_flow_noise_by_sample_id,
    masked_mse,
    masked_mse_per_sample,
    model_coordinate_kwargs,
    repulsion_loss,
    scheduled_vae_head_weight,
    vae_head_consistency_loss,
)


def _canonical_target_positions(model, batch, cfg) -> torch.Tensor:
    core_model = model.module if hasattr(model, "module") else model
    return core_model.world_standardized_to_canonical(
        batch["target_pos"],
        batch["norm_camtoworlds"],
        batch["xyz_mean"],
        batch["xyz_std"],
    ).masked_fill(~batch["target_mask"].bool().unsqueeze(-1), 0.0)


def _canonical_stage1_positions(model, batch, mask: torch.Tensor) -> torch.Tensor:
    if "stage1_pos" not in batch:
        raise KeyError(
            "v26 requires batch['stage1_pos'] (stage-1 DiT xyz sidecar). "
            "Set dataset.stage1_xyz_cache_root and run tools/cache_stage1_xyz.py."
        )
    core_model = model.module if hasattr(model, "module") else model
    return core_model.world_standardized_to_canonical(
        batch["stage1_pos"],
        batch["norm_camtoworlds"],
        batch["xyz_mean"],
        batch["xyz_std"],
    ).masked_fill(~mask.unsqueeze(-1), 0.0)


def _geometry_quality_mix(
    x0_pos: torch.Tensor,
    x1_source: torch.Tensor,
    mask: torch.Tensor,
    t_pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample the geometry-quality state and its residual regression target.

    x_geo = (1 - t_pos) * gt + t_pos * stage1 is the fixed geometry input
    (t_pos = 1 reproduces the pure stage-1 anchor seen at inference start).
    The residual target is x_geo - gt = t_pos * d with d = stage1 - gt; the
    model predicts it as res = t_pos * v_head (explicit multiplication), so
    the head itself learns the t-independent velocity d and the loss below is
    the residual/x0 MSE, numerically comparable to the v20-v22 runs.
    """
    t_view = t_pos.view(-1, 1, 1)
    x_geo = ((1.0 - t_view) * x0_pos + t_view * x1_source).masked_fill(~mask.unsqueeze(-1), 0.0)
    res_tgt = (x_geo - x0_pos).masked_fill(~mask.unsqueeze(-1), 0.0)
    return x_geo, res_tgt


def _masked_bce_with_logits(
    logit: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    err = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    if err.ndim == mask.ndim + 1:
        err = err.mean(dim=-1)
    m = mask.float()
    return (err * m).sum() / m.sum().clamp_min(1.0)


def _aux_position_weights(reg_cfg, num_refine_groups: int) -> list[float]:
    configured = reg_cfg.get("aux_position_loss_weights", 0.0)
    if isinstance(configured, (int, float)):
        return [float(configured)] * num_refine_groups
    weights = [float(value) for value in configured]
    if len(weights) != num_refine_groups:
        raise ValueError(
            "regularization.aux_position_loss_weights must have one value per refine group "
            f"(num_groups - refine_start_group): expected {num_refine_groups}, got {len(weights)}"
        )
    if any(weight < 0.0 for weight in weights):
        raise ValueError("auxiliary position loss weights must be non-negative")
    return weights


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
    reg_cfg = cfg.regularization
    # Geometry lives in training-stat standardized camera-0 coordinates.  The
    # content stream keeps the Gaussian source; geometry is a fixed
    # quality-simulated mix between GT and the stage-1 anchor.
    x0_pos = _canonical_target_positions(model, batch, cfg)
    mask = batch["target_mask"].bool()
    x1_pos = _canonical_stage1_positions(model, batch, mask)
    x0_con = batch["target_con"]
    batch_size = x0_pos.shape[0]
    generator = None
    if deterministic_sample_ids is not None:
        seed = 12345 if deterministic_seed is None else int(deterministic_seed)
        # Reuse the deterministic sampler for t_pos/t_con and the content noise;
        # its position noise is discarded (geometry comes from the quality mix).
        t_pos, t_con, _xt_pos, xt_con, _v_pos_tgt, v_con_tgt = fixed_flow_noise_by_sample_id(
            x0_pos,
            x0_con,
            mask,
            deterministic_sample_ids,
            flow_cfg,
            seed,
            validation_timestep_mode,
        )
    else:
        if deterministic_seed is not None:
            generator = torch.Generator(device=x0_pos.device)
            generator.manual_seed(int(deterministic_seed))
        t_pos, t_con = sample_flow_timesteps(batch_size, x0_pos.device, flow_cfg, generator=generator)
        _xt_pos, xt_con, _v_pos_tgt, v_con_tgt = add_flow_noise(
            x0_pos, x0_con, mask, t_pos, t_con, generator=generator
        )
    x_geo, res_tgt = _geometry_quality_mix(x0_pos, x1_pos, mask, t_pos)

    drop_cond = (
        bool(force_drop_cond)
        if force_drop_cond is not None
        else random.random() < float(flow_cfg.get("cond_drop_prob", 0.0))
    )
    # v26 is single-stream: no camera conditioning (force_drop_cam is accepted for
    # driver compatibility and ignored) and no self-conditioning pass.
    del force_drop_cam
    if selfcond and float(flow_cfg.get("selfcond_prob", 0.0)) > 0.0:
        raise ValueError("v26 single-stream architecture does not support self-conditioning")
    coordinate_kwargs = model_coordinate_kwargs(batch, cfg)

    out = model(
        x_geo,
        xt_con,
        t_pos,
        t_con,
        mask,
        batch["cond_pos"],
        batch["cond_con"],
        batch["cond_mask"].bool(),
        batch["norm_camtoworlds"],
        drop_cond=drop_cond,
        **coordinate_kwargs,
    )
    if "aux_res_pos" not in out:
        raise RuntimeError("DiT v26 must return aux_res_pos with shape [refine_groups, B, N, 3]")
    if "conf_logit" not in out:
        raise RuntimeError("DiT v26 must return conf_logit with shape [B, N, 1]")

    loss_pos = masked_mse(out["res_pos"], res_tgt, mask)
    loss_con = masked_mse(out["v_con"], v_con_tgt, mask)
    loss_flow = (
        float(flow_cfg.get("lambda_pos", 1.0)) * loss_pos
        + float(flow_cfg.get("lambda_con", 1.0)) * loss_con
    )

    aux_predictions = out["aux_res_pos"]
    aux_losses = [
        masked_mse(aux_predictions[index], res_tgt, mask)
        for index in range(aux_predictions.shape[0])
    ]
    weights = _aux_position_weights(reg_cfg, int(aux_predictions.shape[0]))
    loss_aux_pos = torch.stack(
        [loss_value * weight for loss_value, weight in zip(aux_losses, weights)]
    ).sum()
    # Auxiliary heads shape training, while validation/best-checkpoint selection
    # remains comparable through the final res_pos/v_con objective.
    gate_sigma = max(float(flow_cfg.get("gate_sigma", 0.08)), 1e-6)
    conf_tgt = torch.exp(-(x1_pos - x0_pos).norm(dim=-1, keepdim=True) / gate_sigma)
    conf_tgt = conf_tgt.masked_fill(~mask.unsqueeze(-1), 0.0)
    loss_gate = _masked_bce_with_logits(out["conf_logit"], conf_tgt, mask)

    loss = loss_flow
    if model.training:
        loss = loss + float(flow_cfg.get("lambda_pos", 1.0)) * loss_aux_pos
        loss = loss + float(flow_cfg.get("lambda_gate", 1.0)) * loss_gate

    stage1_residual = (x1_pos - x0_pos).norm(dim=-1)
    stage1_pos_mse = masked_mse(x1_pos, x0_pos, mask)
    x0hat_pos_mse = masked_mse(out["x0hat_pos"], x0_pos, mask)
    conf = out["conf"].squeeze(-1)
    mask_f = mask.float()
    mask_denom = mask_f.sum().clamp_min(1.0)
    # Static keys (aux weights, source flag) are intentionally omitted: the
    # trainer pays one .item() + all_reduce per key per step.
    logs = {
        "loss_pos": loss_pos.detach(),
        "loss_con": loss_con.detach(),
        "loss_flow": loss_flow.detach(),
        "loss_aux_pos": loss_aux_pos.detach(),
        "t_pos_mean": t_pos.mean().detach(),
        "t_con_mean": t_con.mean().detach(),
        "t_pos_leads_fraction": (t_pos < t_con).float().mean().detach(),
        "stage1_residual_mean": (stage1_residual * mask.float()).sum() / mask.float().sum().clamp_min(1.0),
        # Do-nothing baseline vs refined-geometry error, same units: the gap is
        # the refinement gain at the sampled quality levels.
        "stage1_pos_mse": stage1_pos_mse.detach(),
        "x0hat_pos_mse": x0hat_pos_mse.detach(),
        "loss_gate": loss_gate.detach(),
        "conf_mean": ((conf * mask_f).sum() / mask_denom).detach(),
        "conf_tgt_mean": ((conf_tgt.squeeze(-1) * mask_f).sum() / mask_denom).detach(),
    }
    for index, aux_loss in enumerate(aux_losses, start=1):
        logs[f"loss_aux_pos_g{index}"] = aux_loss.detach()

    if return_per_sample:
        sample_loss_pos = masked_mse_per_sample(out["res_pos"], res_tgt, mask)
        sample_loss_con = masked_mse_per_sample(out["v_con"], v_con_tgt, mask)
        logs["sample_loss_pos"] = sample_loss_pos.detach()
        logs["sample_loss_con"] = sample_loss_con.detach()
        logs["sample_loss_flow"] = (
            float(flow_cfg.get("lambda_pos", 1.0)) * sample_loss_pos
            + float(flow_cfg.get("lambda_con", 1.0)) * sample_loss_con
        ).detach()
        logs["sample_t_pos"] = t_pos.detach()
        logs["sample_t_con"] = t_con.detach()

    need_x0hat_pos = (
        float(reg_cfg.get("chamfer_weight", 0.0)) > 0
        or float(reg_cfg.get("repulsion_weight", 0.0)) > 0
    )
    vae_head_base_weight = float(reg_cfg.get("vae_head_weight", 0.0))
    vae_head_weight, vae_head_warmup = scheduled_vae_head_weight(
        vae_head_base_weight, reg_cfg, global_step
    )
    vae_head_source = str(reg_cfg.get("vae_head_source", "final")).lower()
    need_x0hat_con = vae_head_base_weight > 0
    low_t_power = float(reg_cfg.get("low_t_power", 1.0))

    if need_x0hat_pos:
        low_t_pos_w = (1.0 - t_pos).pow(low_t_power).mean()
        x0hat_pos = out["x0hat_pos"]
        if float(reg_cfg.get("chamfer_weight", 0.0)) > 0:
            cd = chamfer_loss(x0hat_pos, x0_pos, mask)
            loss = loss + float(reg_cfg["chamfer_weight"]) * low_t_pos_w * cd
            logs["loss_cd"] = cd.detach()
        if float(reg_cfg.get("repulsion_weight", 0.0)) > 0:
            rep = repulsion_loss(x0hat_pos, mask, int(reg_cfg.get("repulsion_k", 8)))
            loss = loss + float(reg_cfg["repulsion_weight"]) * low_t_pos_w * rep
            logs["loss_repulsion"] = rep.detach()

    if need_x0hat_con:
        low_t_con_w = (1.0 - t_con).pow(low_t_power).mean()
        if vae_head_source == "aux":
            if "aux_x0_con" not in out:
                raise RuntimeError("vae_head_source='aux' requires model.aux_con_layer > 0")
            x0hat_con = out["aux_x0_con"]
        elif vae_head_source == "final":
            x0hat_con = xt_con - t_con[:, None, None] * out["v_con"]
        else:
            raise ValueError(f"Unsupported regularization.vae_head_source={vae_head_source!r}")
        if vae_head_weight > 0:
            head_loss, head_logs = vae_head_consistency_loss(
                frozen_vae_head, x0hat_con, batch, reg_cfg
            )
            weighted_head_loss = vae_head_weight * low_t_con_w * head_loss
            loss = loss + weighted_head_loss
            logs.update({f"loss_{key}": value for key, value in head_logs.items()})
        else:
            weighted_head_loss = x0hat_con.new_zeros(())
        logs["loss_head_weighted"] = weighted_head_loss.detach()
        logs["loss_head_low_t_w"] = low_t_con_w.detach()
        logs["loss_head_weight"] = x0hat_con.new_tensor(vae_head_weight)
        logs["loss_head_warmup"] = x0hat_con.new_tensor(vae_head_warmup)

    logs["loss"] = loss.detach()
    return loss, logs


def main() -> None:
    args = gld.parse_args()
    if "--config" not in sys.argv:
        args.config = Path("configs/stage2.py")
    gld.model_step = model_step
    gld.train(args)


if __name__ == "__main__":
    main()
