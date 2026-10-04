"""GLD/DDP training entrypoint for v18_v3_2view stage-1 position DiT.

Isolated from 1-view trainers. Flow/loss matches v16_v3 / v18_v3 except:

  - source is sampled from first-view tokens only (``cond_view_index == 0``)
  - model condition stays ``cat(first, last)``
  - ``cond_view_index`` / ``cond_frame_index`` are passed into the 2-view model
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

import stage1_loop as v16_v3
import train_stage1 as v18_v3
import train_common as gld
from dataset import build_first_last_dataset, first_view_source_mask_from_batch, stage1_view_kwargs
from train_helpers import (
    chamfer_loss,
    masked_mse,
    masked_mse_per_sample,
    model_coordinate_kwargs,
    repulsion_loss,
)

RAYMAP_LOG_KEYS = v18_v3.RAYMAP_LOG_KEYS


def build_dataset(kwargs: dict[str, Any]) -> Dataset:
    return build_first_last_dataset(kwargs)


def _sample_source_pos(model, batch, x0_pos, mask, flow_cfg, generator, deterministic_sample_ids, deterministic_seed):
    cond_pos = v16_v3._canonical_cond_positions(model, batch)
    source_mask = first_view_source_mask_from_batch(batch)
    if deterministic_sample_ids is not None:
        seed = 12345 if deterministic_seed is None else int(deterministic_seed)
        source_parts = []
        for batch_index, sample_id in enumerate(deterministic_sample_ids.detach().cpu().tolist()):
            source_generator = torch.Generator(device=x0_pos.device)
            source_generator.manual_seed(int(seed) + int(sample_id) + 1000003)
            source_parts.append(
                v16_v3._sample_first_frame_source_deterministic(
                    cond_pos[batch_index : batch_index + 1],
                    source_mask[batch_index : batch_index + 1],
                    (1, x0_pos.shape[1], x0_pos.shape[2]),
                    float(flow_cfg.get("first_frame_source_noise", 0.5)),
                    source_generator,
                )
            )
        source_pos = torch.cat(source_parts, dim=0)
    else:
        source_pos = v16_v3._sample_first_frame_source(
            cond_pos,
            source_mask,
            tuple(x0_pos.shape),
            float(flow_cfg.get("first_frame_source_noise", 0.5)),
            generator,
        )
    return source_pos.masked_fill(~mask.unsqueeze(-1), 0.0)


def _forward_model(model, *args, batch, **kwargs):
    return model(*args, **kwargs, **stage1_view_kwargs(model, batch))


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
    x0_pos = v16_v3._canonical_target_positions(model, batch)
    mask = batch["target_mask"].bool()
    generator = None
    if deterministic_seed is not None and deterministic_sample_ids is None:
        generator = torch.Generator(device=x0_pos.device)
        generator.manual_seed(int(deterministic_seed))
    source_pos = None
    if v16_v3._flow_uses_first_frame_source(flow_cfg):
        if "cond_view_index" not in batch:
            raise KeyError("v18_v3_2view source sampling requires batch['cond_view_index']")
        source_pos = _sample_source_pos(
            model,
            batch,
            x0_pos,
            mask,
            flow_cfg,
            generator,
            deterministic_sample_ids,
            deterministic_seed,
        )
    t_pos, xt_pos, v_pos_tgt, x0_pos = v16_v3._stage1_flow(
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
            sc = _forward_model(
                model,
                xt_pos,
                t_pos,
                mask,
                batch["cond_pos"],
                batch["cond_con"],
                batch["cond_mask"].bool(),
                batch["norm_camtoworlds"],
                batch["Ks"],
                batch["cam_mask"].bool(),
                batch=batch,
                x0hat_pos=None,
                drop_cond=drop_cond,
                drop_cam=drop_cam,
                **coordinate_kwargs,
            )
            x0hat = sc["x0_pos"].detach()
        torch.clear_autocast_cache()
    out = _forward_model(
        model,
        xt_pos,
        t_pos,
        mask,
        batch["cond_pos"],
        batch["cond_con"],
        batch["cond_mask"].bool(),
        batch["norm_camtoworlds"],
        batch["Ks"],
        batch["cam_mask"].bool(),
        batch=batch,
        x0hat_pos=x0hat,
        drop_cond=drop_cond,
        drop_cam=drop_cam,
        **coordinate_kwargs,
    )
    if "x0_pos" not in out or "aux_x0hat_pos" not in out:
        raise RuntimeError("v18_v3_2view stage1 model must return x0_pos and aux_x0hat_pos")
    loss_pos = masked_mse(out["x0_pos"], x0_pos, mask)
    aux_predictions = out["aux_x0hat_pos"]
    aux_losses = [masked_mse(aux_predictions[index], x0_pos, mask) for index in range(aux_predictions.shape[0])]
    weights = v16_v3._aux_position_weights(reg_cfg, int(aux_predictions.shape[0]))
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
    core_model = model.module if hasattr(model, "module") else model
    pop = getattr(core_model, "pop_diagnostic", None)
    if pop is not None:
        for key in RAYMAP_LOG_KEYS:
            value = pop(key)
            if value is not None:
                logs[key] = value
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
        args.config = Path("configs/stage1_2view.py")
    gld.build_dataset = build_dataset
    gld.model_step = model_step
    gld.train(args)


if __name__ == "__main__":
    main()
