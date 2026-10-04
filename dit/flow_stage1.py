from __future__ import annotations

from typing import Any

import torch

from flow import position_lead_timestep


def _sample_logit_normal(
    shape: tuple[int, ...],
    device: torch.device,
    flow_cfg: Any,
    generator: torch.Generator | None,
) -> torch.Tensor:
    mean = float(flow_cfg.get("logit_normal_mean", 0.0))
    std = float(flow_cfg.get("logit_normal_std", 1.0))
    if std <= 0.0:
        raise ValueError("flow.logit_normal_std must be positive")
    logits = torch.randn(shape, device=device, generator=generator) * std + mean
    return logits.sigmoid().clamp(float(flow_cfg.get("t_eps", 1e-4)), 1.0)


def _sample_timestep(
    shape: tuple[int, ...],
    device: torch.device,
    flow_cfg: Any,
    generator: torch.Generator | None,
) -> torch.Tensor:
    distribution = str(flow_cfg.get("timestep_distribution", "logit_normal")).lower()
    if distribution == "logit_normal":
        return _sample_logit_normal(shape, device, flow_cfg, generator)
    if distribution == "uniform":
        return torch.rand(shape, device=device, generator=generator).clamp(
            float(flow_cfg.get("t_eps", 1e-4)), 1.0
        )
    raise ValueError(f"Unsupported flow.timestep_distribution={distribution!r}")


def sample_flow_timesteps(
    batch_size: int,
    device: torch.device,
    flow_cfg: Any,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if str(flow_cfg.get("timestep_mode", "independent")).lower() != "independent":
        raise ValueError("DiT v13 requires flow.timestep_mode='independent'")
    t_pos = _sample_timestep((batch_size,), device, flow_cfg, generator)
    t_con = _sample_timestep((batch_size,), device, flow_cfg, generator)
    return t_pos, t_con


def _sinkhorn_permutation(
    noise: torch.Tensor,
    target: torch.Tensor,
    iterations: int,
    temperature: float,
) -> torch.Tensor:
    """Return target indices paired one-to-one with noise rows."""
    cost = torch.cdist(noise.float(), target.float()).square()
    log_plan = -cost / temperature
    for _ in range(iterations):
        log_plan = log_plan - torch.logsumexp(log_plan, dim=1, keepdim=True)
        log_plan = log_plan - torch.logsumexp(log_plan, dim=0, keepdim=True)

    n = noise.shape[0]
    preferences = log_plan.argsort(dim=1, descending=True)
    permutation = torch.full((n,), -1, dtype=torch.long, device=noise.device)
    available = torch.ones((n,), dtype=torch.bool, device=noise.device)
    preference_index = torch.zeros((n,), dtype=torch.long, device=noise.device)
    row_ids = torch.arange(n, device=noise.device)
    while (permutation < 0).any():
        pending = permutation < 0
        choices = preferences[row_ids, preference_index]
        rejected = pending & ~available[choices]
        preference_index[rejected] += 1
        proposing = pending & ~rejected
        chosen_scores = log_plan[row_ids, choices].masked_fill(~proposing, -torch.inf)
        winners = torch.full((n,), -torch.inf, device=noise.device)
        winners.scatter_reduce_(0, choices, chosen_scores, reduce="amax", include_self=True)
        candidates = proposing & (chosen_scores == winners[choices])
        winning_rows = torch.full((n,), n, dtype=torch.long, device=noise.device)
        winning_rows.scatter_reduce_(
            0,
            choices,
            row_ids.masked_fill(~candidates, n),
            reduce="amin",
            include_self=True,
        )
        accepted = candidates & (row_ids == winning_rows[choices])
        permutation[accepted] = choices[accepted]
        available[choices[accepted]] = False
        preference_index[proposing & ~accepted] += 1
    return permutation


def add_flow_noise(
    x0_pos: torch.Tensor,
    x0_con: torch.Tensor,
    mask: torch.Tensor,
    t_pos: torch.Tensor,
    t_con: torch.Tensor,
    flow_cfg: Any,
    generator: torch.Generator | None = None,
    eps_pos: torch.Tensor | None = None,
    eps_con: torch.Tensor | None = None,
):
    eps_pos = eps_pos if eps_pos is not None else torch.randn(
        x0_pos.shape, device=x0_pos.device, dtype=x0_pos.dtype, generator=generator
    )
    eps_con = eps_con if eps_con is not None else torch.randn(
        x0_con.shape, device=x0_con.device, dtype=x0_con.dtype, generator=generator
    )
    paired_pos = torch.zeros_like(x0_pos)
    paired_con = torch.zeros_like(x0_con)
    coupling = str(flow_cfg.get("ot_coupling", "sinkhorn_permutation")).lower()
    if coupling not in {"sinkhorn_permutation", "none"}:
        raise ValueError(f"Unsupported flow.ot_coupling={coupling!r}")
    iterations = int(flow_cfg.get("ot_sinkhorn_iterations", 20))
    temperature = float(flow_cfg.get("ot_sinkhorn_temperature", 0.1))
    if iterations <= 0 or temperature <= 0.0:
        raise ValueError("OT Sinkhorn iterations and temperature must be positive")
    for batch_index in range(x0_pos.shape[0]):
        valid = mask[batch_index].bool()
        if coupling == "sinkhorn_permutation":
            permutation = _sinkhorn_permutation(
                eps_pos[batch_index, valid], x0_pos[batch_index, valid], iterations, temperature
            )
        else:
            permutation = torch.arange(valid.sum(), device=x0_pos.device)
        paired_pos[batch_index, valid] = x0_pos[batch_index, valid][permutation]
        paired_con[batch_index, valid] = x0_con[batch_index, valid][permutation]

    t_pos_view = t_pos.view(-1, 1, 1)
    t_con_view = t_con.view(-1, 1, 1)
    xt_pos = (1.0 - t_pos_view) * paired_pos + t_pos_view * eps_pos
    xt_con = (1.0 - t_con_view) * paired_con + t_con_view * eps_con
    v_pos = eps_pos - paired_pos
    v_con = eps_con - paired_con
    valid = mask.unsqueeze(-1)
    return (
        xt_pos.masked_fill(~valid, 0.0),
        xt_con.masked_fill(~valid, 0.0),
        v_pos.masked_fill(~valid, 0.0),
        v_con.masked_fill(~valid, 0.0),
        paired_pos,
        paired_con,
    )


def fixed_flow_noise_by_sample_id(
    x0_pos: torch.Tensor,
    x0_con: torch.Tensor,
    mask: torch.Tensor,
    sample_ids: torch.Tensor,
    flow_cfg: Any,
    seed: int,
    timestep_mode: str = "independent",
):
    t_pos_values, t_con_values, eps_pos_values, eps_con_values = [], [], [], []
    for sample_id in sample_ids.detach().cpu().tolist():
        generator = torch.Generator(device=x0_pos.device)
        generator.manual_seed(int(seed) + int(sample_id))
        t_con = _sample_timestep((), x0_pos.device, flow_cfg, generator)
        if str(timestep_mode).lower() == "position_leading":
            t_pos = position_lead_timestep(
                t_con, float(flow_cfg.get("position_lead_beta", 0.5))
            ).clamp(float(flow_cfg.get("t_eps", 1e-4)), 1.0)
        elif str(timestep_mode).lower() == "independent":
            t_pos = _sample_timestep((), x0_pos.device, flow_cfg, generator)
        else:
            raise ValueError(f"Unsupported validation timestep mode={timestep_mode!r}")
        t_pos_values.append(t_pos)
        t_con_values.append(t_con)
        eps_pos_values.append(
            torch.randn(
                x0_pos[0].shape,
                device=x0_pos.device,
                dtype=x0_pos.dtype,
                generator=generator,
            )
        )
        eps_con_values.append(
            torch.randn(
                x0_con[0].shape,
                device=x0_con.device,
                dtype=x0_con.dtype,
                generator=generator,
            )
        )
    t_pos = torch.stack(t_pos_values)
    t_con = torch.stack(t_con_values)
    outputs = add_flow_noise(
        x0_pos,
        x0_con,
        mask,
        t_pos,
        t_con,
        flow_cfg,
        eps_pos=torch.stack(eps_pos_values),
        eps_con=torch.stack(eps_con_values),
    )
    return t_pos, t_con, *outputs
