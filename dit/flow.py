from __future__ import annotations

import torch


def position_lead_timestep(t, beta: float):
    if not 0.0 <= float(beta) < 1.0:
        raise ValueError("position lead beta must be in [0, 1)")
    return t - float(beta) * t * (1.0 - t)


def sample_flow_timesteps(
    batch_size: int,
    device: torch.device,
    flow_cfg,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    mode = str(flow_cfg.get("timestep_mode", "independent")).lower()
    if mode != "independent":
        raise ValueError(f"DiT v4 requires flow.timestep_mode='independent', got {mode!r}")
    t_eps = float(flow_cfg.get("t_eps", 1e-4))
    t_pos = torch.rand((batch_size,), device=device, generator=generator).clamp(t_eps, 1.0)
    # Inference spends ~all non-refine steps at t_pos = 1 (geometry = stage-1
    # anchor), a point with zero density under U(0,1).  Anchor a fraction of
    # samples exactly at 1.0 so the content stream is trained in the regime
    # that dominates inference; the uniform half keeps (0,1) coverage for the
    # Euler refinement phase.  Default 0 => other versions are unaffected.
    anchor_prob = float(flow_cfg.get("t_pos_anchor_prob", 0.0))
    if anchor_prob > 0.0:
        anchor = torch.rand((batch_size,), device=device, generator=generator) < anchor_prob
        t_pos = torch.where(anchor, torch.ones_like(t_pos), t_pos)
    t_con = torch.rand((batch_size,), device=device, generator=generator).clamp(t_eps, 1.0)
    return t_pos, t_con


def add_flow_noise(
    x0_pos: torch.Tensor,
    x0_con: torch.Tensor,
    mask: torch.Tensor,
    t_pos: torch.Tensor,
    t_con: torch.Tensor,
    generator: torch.Generator | None = None,
):
    eps_pos = torch.randn(x0_pos.shape, device=x0_pos.device, dtype=x0_pos.dtype, generator=generator)
    eps_con = torch.randn(x0_con.shape, device=x0_con.device, dtype=x0_con.dtype, generator=generator)
    t_pos_view = t_pos.view(-1, 1, 1)
    t_con_view = t_con.view(-1, 1, 1)
    xt_pos = (1.0 - t_pos_view) * x0_pos + t_pos_view * eps_pos
    xt_con = (1.0 - t_con_view) * x0_con + t_con_view * eps_con
    v_pos = eps_pos - x0_pos
    v_con = eps_con - x0_con
    xt_pos = xt_pos.masked_fill(~mask.unsqueeze(-1), 0.0)
    xt_con = xt_con.masked_fill(~mask.unsqueeze(-1), 0.0)
    return xt_pos, xt_con, v_pos, v_con
