"""Stage-1 position sampling and stage-2 content sampling with gated refinement."""

from __future__ import annotations

from typing import Any

import torch

from dataset import first_view_source_mask_from_batch, stage1_view_kwargs


def to_device_sample(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        out[key] = value.unsqueeze(0).to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def coordinate_kwargs(model, batch):
    if bool(getattr(model, "requires_xyz_stats", False)):
        return {"xyz_mean": batch["xyz_mean"], "xyz_std": batch["xyz_std"]}
    return {}


def sample_first_frame_source(model, batch, n_points: int, source_noise: float, generator: torch.Generator):
    if not 0.0 <= source_noise <= 1.0:
        raise ValueError("source_noise must be in [0, 1]")
    cond_pos = model.world_standardized_to_canonical(
        batch["cond_pos"], batch["norm_camtoworlds"], batch["xyz_mean"], batch["xyz_std"]
    )
    cond_mask = first_view_source_mask_from_batch(batch)
    source = torch.empty((1, n_points, 3), device=cond_pos.device, dtype=cond_pos.dtype)
    valid_points = cond_pos[0, cond_mask[0]]
    if valid_points.numel() == 0:
        source[0] = torch.randn((n_points, 3), device=cond_pos.device, dtype=cond_pos.dtype, generator=generator)
    else:
        indices = torch.randint(valid_points.shape[0], (n_points,), device=cond_pos.device, generator=generator)
        source[0] = valid_points[indices]
    if source_noise > 0.0:
        gaussian = torch.randn(source.shape, device=source.device, dtype=source.dtype, generator=generator)
        source = (1.0 - source_noise) * source + source_noise * gaussian
    return source


@torch.no_grad()
def _predict_stage1(model, x_pos, mask, t, batch, guidance, guidance_mode, guidance_scale_t):
    tt = torch.full((x_pos.shape[0],), float(t), device=x_pos.device)
    kwargs = dict(
        xt_pos=x_pos,
        t_pos=tt,
        target_mask=mask,
        cond_pos=batch["cond_pos"],
        cond_con=batch["cond_con"],
        cond_mask=batch["cond_mask"].bool(),
        cam_c2w=batch["norm_camtoworlds"],
        cam_K=batch["Ks"],
        cam_mask=batch["cam_mask"].bool(),
        x0hat_pos=None,
    )
    kwargs.update(coordinate_kwargs(model, batch))
    kwargs.update(stage1_view_kwargs(model, batch))
    if guidance_mode in ("none", "conditional"):
        return model(**kwargs)
    if guidance_mode != "camera":
        raise ValueError(f"Unsupported stage1 guidance_mode={guidance_mode!r}")
    cond = model(**kwargs, drop_cond=False, drop_cam=False)
    scale = float(guidance) * float(guidance_scale_t)
    if scale == 1.0:
        return cond
    uncond = model(**kwargs, drop_cond=False, drop_cam=True)
    return {"v_pos": uncond["v_pos"] + scale * (cond["v_pos"] - uncond["v_pos"])}


@torch.no_grad()
def sample_stage1_xyz(
    model,
    batch,
    n_points: int,
    steps: int,
    seed: int,
    source_noise: float,
    guidance: float = 1.0,
    guidance_mode: str = "camera",
    guidance_start: float = 1.0,
    guidance_end: float = 1.0,
):
    device = next(model.parameters()).device
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    x_pos = sample_first_frame_source(model, batch, n_points, source_noise, generator)
    mask = torch.ones((1, n_points), dtype=torch.bool, device=device)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for step_idx in range(steps):
        t = float(ts[step_idx].item())
        t_next = float(ts[step_idx + 1].item())
        progress = 0.0 if steps <= 1 else step_idx / float(steps - 1)
        scale_t = float(guidance_start) + (float(guidance_end) - float(guidance_start)) * progress
        out = _predict_stage1(model, x_pos, mask, t, batch, guidance, guidance_mode, scale_t)
        x_pos = x_pos - (t - t_next) * out["v_pos"]
        x_pos = torch.nan_to_num(x_pos, nan=0.0, posinf=0.0, neginf=0.0)
    return x_pos.detach(), mask


@torch.no_grad()
def _forward_stage2(model, x_geo, x_con, t_pos, t_con, mask, batch, *, drop_cond: bool):
    return model(
        x_geo,
        x_con,
        t_pos,
        t_con,
        mask,
        batch["cond_pos"],
        batch["cond_con"],
        batch["cond_mask"].bool(),
        batch["norm_camtoworlds"],
        drop_cond=drop_cond,
        xyz_mean=batch["xyz_mean"],
        xyz_std=batch["xyz_std"],
        **stage1_view_kwargs(model, batch),
    )


@torch.no_grad()
def sample_stage2(
    model,
    batch,
    x_geo: torch.Tensor,
    mask: torch.Tensor,
    steps: int,
    seed: int,
    guidance_cond: float = 1.5,
    refine_steps: int = 10,
    refine_gate: bool = True,
    refine_t_pos_mode: str = "schedule",
):
    """Content Euler loop. The last ``refine_steps`` also integrate position velocity."""
    device = next(model.parameters()).device
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    x_con = torch.randn((x_geo.shape[0], x_geo.shape[1], model.target_dim), device=device, generator=generator)
    use_cfg = float(guidance_cond) != 1.0
    k_total = min(int(refine_steps), steps)
    refine_start = steps - k_total
    t_pos = torch.ones((x_geo.shape[0],), device=device)
    res0_norm = None
    prev_res_norm = None
    frozen_conf = None
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for step_idx in range(steps):
        t = float(ts[step_idx].item())
        t_next = float(ts[step_idx + 1].item())
        t_con = torch.full((x_geo.shape[0],), t, device=device)
        refining = step_idx >= refine_start and k_total > 0
        if refining:
            k = step_idx - refine_start
            if refine_t_pos_mode == "schedule":
                t_pos = torch.full_like(t_pos, (k_total - k) / k_total)
            elif refine_t_pos_mode == "adaptive" and prev_res_norm is not None and res0_norm:
                t_pos = torch.full_like(t_pos, min(max(prev_res_norm / res0_norm, 0.0), 1.0))
        out_cond = _forward_stage2(model, x_geo, x_con, t_pos, t_con, mask, batch, drop_cond=False)
        if use_cfg:
            out_uncond = _forward_stage2(model, x_geo, x_con, t_pos, t_con, mask, batch, drop_cond=True)
            v_con = out_uncond["v_con"] + float(guidance_cond) * (out_cond["v_con"] - out_uncond["v_con"])
        else:
            v_con = out_cond["v_con"]
        x_con = torch.nan_to_num(x_con - (t - t_next) * v_con, nan=0.0, posinf=0.0, neginf=0.0)
        if refining:
            v_pos = out_cond["v_pos"]
            if refine_gate:
                if frozen_conf is None:
                    frozen_conf = out_cond["conf"]
                v_pos = (1.0 - frozen_conf) * v_pos
            prev_res_norm = float(out_cond["res_pos"].norm(dim=-1).mean().item())
            if res0_norm is None:
                res0_norm = max(prev_res_norm, 1e-8)
            x_geo = torch.nan_to_num(x_geo - v_pos / k_total, nan=0.0, posinf=0.0, neginf=0.0)
    conf = frozen_conf.detach() if frozen_conf is not None else None
    return x_con.detach(), x_geo.detach(), conf
