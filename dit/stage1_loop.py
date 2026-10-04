"""GLD/DDP training entrypoint for v16_v3 stage-1 position DiT.

Same flow/loss as v16_v2 stage-1 (see dit_v16_v3_primitives docstring for the
model-side A+B+C optimizations); v16_v3 only shrinks the backbone to 768d x 8
layers via config (num_groups=4). Kept from v16_v2:

  - ``_sample_first_frame_source`` training path is vectorized and host-sync free
    (cumsum + searchsorted gather instead of per-sample boolean indexing). The
    deterministic validation path keeps the original per-sample loop.
  - static log keys (aux_pos_weight_g*, aux_position_loss_mode_x0, t_con_mean,
    t_pos_leads_fraction, first_frame_source_prior) are no longer emitted: the
    shared trainer calls ``.item()`` + all_reduce on every log key every step, so
    each constant key cost a host sync and a collective per step.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import torch

import train_common as gld
from flow_stage1 import add_flow_noise, fixed_flow_noise_by_sample_id, sample_flow_timesteps
from flow import position_lead_timestep
from train_helpers import (
    chamfer_loss,
    masked_mse,
    masked_mse_per_sample,
    model_coordinate_kwargs,
    repulsion_loss,
)


def _aux_position_weights(reg_cfg, num_groups: int) -> list[float]:
    configured = reg_cfg.get("aux_position_loss_weights", 0.0)
    if isinstance(configured, (int, float)):
        return [float(configured)] * num_groups
    weights = [float(value) for value in configured]
    if len(weights) != num_groups:
        raise ValueError(
            "regularization.aux_position_loss_weights must have one value per group: "
            f"expected {num_groups}, got {len(weights)}"
        )
    if any(weight < 0.0 for weight in weights):
        raise ValueError("auxiliary position loss weights must be non-negative")
    return weights


def _canonical_target_positions(model, batch) -> torch.Tensor:
    core_model = model.module if hasattr(model, "module") else model
    return core_model.world_standardized_to_canonical(
        batch["target_pos"],
        batch["norm_camtoworlds"],
        batch["xyz_mean"],
        batch["xyz_std"],
    ).masked_fill(~batch["target_mask"].bool().unsqueeze(-1), 0.0)


def _canonical_cond_positions(model, batch) -> torch.Tensor:
    core_model = model.module if hasattr(model, "module") else model
    return core_model.world_standardized_to_canonical(
        batch["cond_pos"],
        batch["norm_camtoworlds"],
        batch["xyz_mean"],
        batch["xyz_std"],
    ).masked_fill(~batch["cond_mask"].bool().unsqueeze(-1), 0.0)


def _sample_first_frame_source(
    cond_pos: torch.Tensor,
    cond_mask: torch.Tensor,
    target_shape: tuple[int, int, int],
    source_noise: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Uniform sampling with replacement over each row's valid cond points.

    Sync-free: cumsum ranks + searchsorted gather; rows without any valid point get
    pure Gaussian noise, exactly like the v16 reference implementation.
    """
    if not 0.0 <= source_noise <= 1.0:
        raise ValueError("flow.first_frame_source_noise must be in [0, 1]")
    b, n, c = target_shape
    mask = cond_mask.bool()
    m = mask.shape[1]
    cumsum = mask.to(torch.int64).cumsum(dim=1)
    count = cumsum[:, -1:]
    u = torch.rand((b, n), device=cond_pos.device, generator=generator)
    rank = (u * count.clamp_min(1).to(u.dtype)).long()
    idx = torch.searchsorted(cumsum, rank + 1).clamp_max(m - 1)
    source = torch.gather(cond_pos, 1, idx.unsqueeze(-1).expand(b, n, c))
    gaussian = torch.randn(source.shape, device=source.device, dtype=source.dtype, generator=generator)
    source = torch.where((count > 0)[..., None], source, gaussian)
    if source_noise > 0.0:
        noise = torch.randn(source.shape, device=source.device, dtype=source.dtype, generator=generator)
        source = (1.0 - source_noise) * source + source_noise * noise
    return source


def _sample_first_frame_source_deterministic(
    cond_pos: torch.Tensor,
    cond_mask: torch.Tensor,
    target_shape: tuple[int, int, int],
    source_noise: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Reference per-sample loop, kept for the deterministic validation path."""
    if not 0.0 <= source_noise <= 1.0:
        raise ValueError("flow.first_frame_source_noise must be in [0, 1]")
    b, n, c = target_shape
    source = torch.empty((b, n, c), device=cond_pos.device, dtype=cond_pos.dtype)
    for batch_index in range(b):
        valid_points = cond_pos[batch_index, cond_mask[batch_index].bool()]
        if valid_points.numel() == 0:
            source[batch_index] = torch.randn((n, c), device=cond_pos.device, dtype=cond_pos.dtype, generator=generator)
            continue
        indices = torch.randint(
            valid_points.shape[0],
            (n,),
            device=cond_pos.device,
            generator=generator,
        )
        source[batch_index] = valid_points[indices]
    if source_noise > 0.0:
        gaussian = torch.randn(source.shape, device=source.device, dtype=source.dtype, generator=generator)
        source = (1.0 - source_noise) * source + source_noise * gaussian
    return source


def _flow_uses_first_frame_source(flow_cfg) -> bool:
    return str(flow_cfg.get("source_prior", "gaussian")).lower() == "first_frame_partial_noise"


def _deterministic_timesteps(
    batch_size: int,
    device: torch.device,
    flow_cfg,
    sample_ids: torch.Tensor,
    seed: int,
    validation_timestep_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Generator]]:
    t_pos_values, t_con_values, generators = [], [], []
    for sample_id in sample_ids.detach().cpu().tolist():
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed) + int(sample_id))
        t_pos, t_con = sample_flow_timesteps(1, device, flow_cfg, generator=generator)
        if str(validation_timestep_mode).lower() == "position_leading":
            t_pos = position_lead_timestep(
                t_con[0], float(flow_cfg.get("position_lead_beta", 0.5))
            ).clamp(float(flow_cfg.get("t_eps", 1e-4)), 1.0).view(1)
        elif str(validation_timestep_mode).lower() != "independent":
            raise ValueError(f"Unsupported validation timestep mode={validation_timestep_mode!r}")
        t_pos_values.append(t_pos[0])
        t_con_values.append(t_con[0])
        generators.append(generator)
    if len(generators) != batch_size:
        raise ValueError(f"expected {batch_size} deterministic sample ids, got {len(generators)}")
    return torch.stack(t_pos_values), torch.stack(t_con_values), generators


def _stage1_flow(
    x0_pos: torch.Tensor,
    source_pos: torch.Tensor | None,
    mask: torch.Tensor,
    flow_cfg,
    generator: torch.Generator | None,
    deterministic_sample_ids: torch.Tensor | None,
    deterministic_seed: int | None,
    validation_timestep_mode: str,
):
    dummy_con = x0_pos.new_zeros((*x0_pos.shape[:2], 1))
    if source_pos is not None:
        if deterministic_sample_ids is not None:
            seed = 12345 if deterministic_seed is None else int(deterministic_seed)
            t_pos, t_con, _generators = _deterministic_timesteps(
                x0_pos.shape[0],
                x0_pos.device,
                flow_cfg,
                deterministic_sample_ids,
                seed,
                validation_timestep_mode,
            )
        else:
            t_pos, t_con = sample_flow_timesteps(x0_pos.shape[0], x0_pos.device, flow_cfg, generator=generator)
        xt_pos, _xt_con, v_pos_tgt, _v_con_tgt, paired_pos, _paired_con = add_flow_noise(
            x0_pos,
            dummy_con,
            mask,
            t_pos,
            t_con,
            flow_cfg,
            generator=generator,
            eps_pos=source_pos,
        )
        return t_pos, xt_pos, v_pos_tgt, paired_pos
    if deterministic_sample_ids is not None:
        seed = 12345 if deterministic_seed is None else int(deterministic_seed)
        (
            t_pos,
            _t_con,
            xt_pos,
            _xt_con,
            v_pos_tgt,
            _v_con_tgt,
            paired_pos,
            _paired_con,
        ) = fixed_flow_noise_by_sample_id(
            x0_pos,
            dummy_con,
            mask,
            deterministic_sample_ids,
            flow_cfg,
            seed,
            validation_timestep_mode,
        )
        return t_pos, xt_pos, v_pos_tgt, paired_pos
    t_pos, t_con = sample_flow_timesteps(x0_pos.shape[0], x0_pos.device, flow_cfg, generator=generator)
    xt_pos, _xt_con, v_pos_tgt, _v_con_tgt, paired_pos, _paired_con = add_flow_noise(
        x0_pos,
        dummy_con,
        mask,
        t_pos,
        t_con,
        flow_cfg,
        generator=generator,
    )
    return t_pos, xt_pos, v_pos_tgt, paired_pos


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
    del frozen_vae_head
    flow_cfg = cfg.flow
    reg_cfg = cfg.regularization
    x0_pos = _canonical_target_positions(model, batch)
    mask = batch["target_mask"].bool()
    generator = None
    if deterministic_seed is not None and deterministic_sample_ids is None:
        generator = torch.Generator(device=x0_pos.device)
        generator.manual_seed(int(deterministic_seed))
    source_pos = None
    if _flow_uses_first_frame_source(flow_cfg):
        cond_pos = _canonical_cond_positions(model, batch)
        if deterministic_sample_ids is not None:
            seed = 12345 if deterministic_seed is None else int(deterministic_seed)
            source_parts = []
            for batch_index, sample_id in enumerate(deterministic_sample_ids.detach().cpu().tolist()):
                source_generator = torch.Generator(device=x0_pos.device)
                source_generator.manual_seed(int(seed) + int(sample_id) + 1000003)
                source_parts.append(
                    _sample_first_frame_source_deterministic(
                        cond_pos[batch_index : batch_index + 1],
                        batch["cond_mask"][batch_index : batch_index + 1].bool(),
                        (1, x0_pos.shape[1], x0_pos.shape[2]),
                        float(flow_cfg.get("first_frame_source_noise", 0.5)),
                        source_generator,
                    )
                )
            source_pos = torch.cat(source_parts, dim=0)
        else:
            source_pos = _sample_first_frame_source(
                cond_pos,
                batch["cond_mask"].bool(),
                tuple(x0_pos.shape),
                float(flow_cfg.get("first_frame_source_noise", 0.5)),
                generator,
            )
        source_pos = source_pos.masked_fill(~mask.unsqueeze(-1), 0.0)
    t_pos, xt_pos, v_pos_tgt, x0_pos = _stage1_flow(
        x0_pos,
        source_pos,
        mask,
        flow_cfg,
        generator,
        deterministic_sample_ids,
        deterministic_seed,
        validation_timestep_mode,
    )
    drop_cond = (
        bool(force_drop_cond)
        if force_drop_cond is not None
        else random.random() < float(flow_cfg.get("cond_drop_prob", 0.0))
    )
    drop_cam = (
        bool(force_drop_cam)
        if force_drop_cam is not None
        else random.random() < float(flow_cfg.get("cam_drop_prob", 0.0))
    )
    coordinate_kwargs = model_coordinate_kwargs(batch, cfg)
    x0hat = None
    if selfcond and random.random() < float(flow_cfg.get("selfcond_prob", 0.0)):
        with torch.no_grad():
            sc = model(
                xt_pos,
                t_pos,
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
            x0hat = sc["x0_pos"].detach()
        # The no_grad forward populates the autocast weight cache with detached bf16
        # casts; without clearing it, the training forward below reuses them and every
        # Linear weight silently receives no gradient on selfcond steps.
        torch.clear_autocast_cache()
    out = model(
        xt_pos,
        t_pos,
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
    if "x0_pos" not in out or "aux_x0hat_pos" not in out:
        raise RuntimeError("v16_v3 stage1 model must return x0_pos and aux_x0hat_pos")
    loss_pos = masked_mse(out["x0_pos"], x0_pos, mask)
    aux_predictions = out["aux_x0hat_pos"]
    aux_losses = [masked_mse(aux_predictions[index], x0_pos, mask) for index in range(aux_predictions.shape[0])]
    weights = _aux_position_weights(reg_cfg, int(aux_predictions.shape[0]))
    loss_aux_pos = torch.stack([value * weight for value, weight in zip(aux_losses, weights)]).sum()
    loss = float(flow_cfg.get("lambda_pos", 1.0)) * loss_pos
    if model.training:
        loss = loss + float(flow_cfg.get("lambda_pos", 1.0)) * loss_aux_pos

    need_x0hat_pos = (
        float(reg_cfg.get("chamfer_weight", 0.0)) > 0
        or float(reg_cfg.get("repulsion_weight", 0.0)) > 0
    )
    if need_x0hat_pos:
        low_t_w = (1.0 - t_pos).pow(float(reg_cfg.get("low_t_power", 1.0))).mean()
        if float(reg_cfg.get("chamfer_weight", 0.0)) > 0:
            cd = chamfer_loss(out["x0_pos"], x0_pos, mask)
            loss = loss + float(reg_cfg["chamfer_weight"]) * low_t_w * cd
        if float(reg_cfg.get("repulsion_weight", 0.0)) > 0:
            rep = repulsion_loss(out["x0_pos"], mask, int(reg_cfg.get("repulsion_k", 8)))
            loss = loss + float(reg_cfg["repulsion_weight"]) * low_t_w * rep

    # Only dynamic scalars are logged: the shared trainer syncs + all_reduces every
    # key every step, so static config-known keys are omitted on purpose.
    logs = {
        "loss": loss.detach(),
        "loss_pos": loss_pos.detach(),
        "loss_con": loss_pos.new_zeros(()),
        "loss_flow": (float(flow_cfg.get("lambda_pos", 1.0)) * loss_pos).detach(),
        "loss_aux_pos": loss_aux_pos.detach(),
        "t_pos_mean": t_pos.mean().detach(),
        "ot_path_cost_pos": masked_mse(v_pos_tgt, torch.zeros_like(v_pos_tgt), mask).detach(),
    }
    for index, aux_loss in enumerate(aux_losses, start=1):
        logs[f"loss_aux_x0_pos_g{index}"] = aux_loss.detach()
    for key in ("spatial_rope_gate", "real_count_mean", "virt_count_mean", "virtual_type_bias", "desc_bias_abs_mean"):
        if key in out:
            value = out[key]
            logs[key if key != "spatial_rope_gate" else "spatial_rope_gate_mean"] = (
                value.float().mean().detach() if torch.is_tensor(value) else loss.new_tensor(float(value))
            )
    if return_per_sample:
        logs["sample_loss_pos"] = masked_mse_per_sample(out["x0_pos"], x0_pos, mask).detach()
        logs["sample_loss_con"] = torch.zeros_like(logs["sample_loss_pos"])
        logs["sample_loss_flow"] = float(flow_cfg.get("lambda_pos", 1.0)) * logs["sample_loss_pos"]
        logs["sample_t_pos"] = t_pos.detach()
        logs["sample_t_con"] = t_pos.detach()
    return loss, logs


def main() -> None:
    args = gld.parse_args()
    if "--config" not in sys.argv:
        args.config = Path("configs/stage1.py")
    gld.model_step = model_step
    gld.train(args)


if __name__ == "__main__":
    main()
