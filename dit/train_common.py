#!/usr/bin/env python3
"""GLD-aligned DDP training entrypoint for dual-timestep DiT v4.

This intentionally avoids Accelerate and DeepSpeed.  It follows the training
shape used by GLD: torchrun/DDP, constant LR by default, EMA, bf16 autocast,
manual gradient clipping, periodic validation, and .pt checkpoints.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import importlib
import importlib.util
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from dataset import FirstFrameGenSuperpointChunkDataset
from train_helpers import ModelEma, format_loss_extras, model_step, scheduled_lr, to_device
from vae_bridge import load_json_config, load_model as load_frozen_vae_model


def load_config(path: Path):
    spec = importlib.util.spec_from_file_location("dit_v4_gld_config", path)
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


class FixedValidationDataset(Dataset):
    def __init__(self, dataset: Dataset, num_samples: int) -> None:
        self.dataset = dataset
        self.num_samples = int(num_samples)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.dataset[int(index)])
        item["sample_id"] = torch.tensor(int(index), dtype=torch.long)
        return item


def init_distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        timeout_seconds = int(os.environ.get("DDP_TIMEOUT_SECONDS", "3600"))
        dist.init_process_group(
            "nccl" if torch.cuda.is_available() else "gloo",
            timeout=datetime.timedelta(seconds=timeout_seconds),
        )
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, world_size, local_rank, device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def reduce_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    value = value.detach()
    if world_size <= 1:
        return value
    value = value.clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    value /= float(world_size)
    return value


def reduce_sum(value: torch.Tensor, world_size: int) -> torch.Tensor:
    value = value.detach()
    if world_size <= 1:
        return value
    value = value.clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def set_all_seeds(seed: int, rank: int) -> None:
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loader(
    ds: Dataset,
    train_cfg: dict[str, Any],
    *,
    rank: int,
    world_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last)
    kwargs: dict[str, Any] = {
        "batch_size": int(train_cfg["batch_size"]),
        "shuffle": shuffle and sampler is None,
        "sampler": sampler,
        "num_workers": int(num_workers),
        "pin_memory": bool(train_cfg.get("pin_memory", True)),
        "drop_last": drop_last,
        "persistent_workers": bool(train_cfg.get("persistent_workers", True)) and int(num_workers) > 0,
    }
    if int(num_workers) > 0:
        kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 2))
    return DataLoader(ds, **kwargs), sampler


def build_validation_loader(
    ds: Dataset,
    train_cfg: dict[str, Any],
    *,
    rank: int,
    world_size: int,
) -> tuple[DataLoader | None, DistributedSampler | None, int]:
    if len(ds) == 0:
        return None, None, 0
    val_batch_size = int(train_cfg.get("val_batch_size", train_cfg["batch_size"]))
    default_samples = int(train_cfg.get("max_val_batches", 8)) * val_batch_size * world_size
    requested = int(train_cfg.get("validation_num_samples", default_samples))
    num_samples = min(len(ds), requested)
    if world_size > 1:
        num_samples -= num_samples % world_size
    if num_samples <= 0:
        return None, None, 0
    val_cfg = dict(train_cfg)
    val_cfg["batch_size"] = val_batch_size
    loader, sampler = build_loader(
        FixedValidationDataset(ds, num_samples),
        val_cfg,
        rank=rank,
        world_size=world_size,
        shuffle=False,
        drop_last=False,
        num_workers=max(0, min(int(train_cfg.get("val_num_workers", 2)), int(train_cfg["num_workers"]))),
    )
    return loader, sampler, num_samples


def build_frozen_vae_head(cfg, device: torch.device):
    reg_cfg = cfg.regularization
    if float(reg_cfg.get("vae_head_weight", 0.0)) <= 0:
        return None
    vae_cfg_path = Path(str(reg_cfg["vae_head_config"]))
    vae_cfg = load_json_config(vae_cfg_path).get("vae", {})
    vae_cfg["device"] = str(device)
    vae = load_frozen_vae_model(vae_cfg, device)
    vae.eval().requires_grad_(False)
    return vae


def freeze_structurally_unused_params(model: torch.nn.Module, enabled: bool = True) -> list[str]:
    if not enabled:
        return []
    frozen = []
    for name, param in model.named_parameters():
        if ".pos_delta." in name or name.endswith(".pos_gate"):
            param.requires_grad_(False)
            frozen.append(name)
    return frozen


def freeze_unused_null_condition_params(model: torch.nn.Module, *, cond_drop_prob: float) -> list[str]:
    if float(cond_drop_prob) > 0.0:
        return []
    frozen = []
    for name in ("null_cond_con", "null_cond_pos"):
        param = getattr(model, name, None)
        if isinstance(param, torch.nn.Parameter):
            param.requires_grad_(False)
            frozen.append(name)
    return frozen


def freeze_unused_aux_head(model: torch.nn.Module, *, vae_head_weight: float, vae_head_source: str) -> list[str]:
    if float(vae_head_weight) > 0.0 and str(vae_head_source).lower() == "aux":
        return []
    aux_head = getattr(model, "aux_con_head", None)
    if aux_head is None:
        return []
    frozen = []
    for name, param in aux_head.named_parameters(prefix="aux_con_head"):
        param.requires_grad_(False)
        frozen.append(name)
    return frozen


def append_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def format_val_metrics(metrics: dict[str, float], prefix: str = "") -> str:
    parts = [
        f"{prefix}loss={metrics['loss']:.5f}",
        f"{prefix}pos={metrics['loss_pos']:.5f}",
        f"{prefix}con={metrics['loss_con']:.5f}",
        f"{prefix}flow={metrics['loss_flow']:.5f}",
    ]
    for i in range(int(metrics.get("num_t_bins", 0))):
        count = int(metrics.get(f"tbin{i}_count", 0))
        if count <= 0:
            continue
        lo = metrics[f"tbin{i}_lo"]
        hi = metrics[f"tbin{i}_hi"]
        parts.append(
            f"{prefix}tbin{i}[{lo:.1f},{hi:.1f})={metrics[f'tbin{i}_loss']:.5f}/"
            f"{metrics[f'tbin{i}_pos']:.5f}/{metrics[f'tbin{i}_con']:.5f}:n={count}"
        )
    return " ".join(parts)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    config_path: Path,
    ema: ModelEma | None,
    best_val_loss: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if isinstance(model, DDP) else model
    raw_state = raw_model.state_dict()
    payload = {
        "model": ema.model_state_dict(raw_model) if ema is not None else raw_state,
        "model_raw": raw_state,
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "epoch": int(epoch),
        "config": str(config_path),
    }
    if best_val_loss is not None and math.isfinite(float(best_val_loss)):
        payload["best_val_loss"] = float(best_val_loss)
    if ema is not None:
        payload["ema"] = ema.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: ModelEma | None,
    *,
    resume_optimizer: bool = True,
) -> tuple[int, int, float]:
    ckpt = torch.load(path, map_location="cpu")
    raw_model = model.module if isinstance(model, DDP) else model
    state = ckpt.get("model_raw", ckpt["model"])
    raw_model.load_state_dict(state, strict=True)
    if resume_optimizer and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except ValueError as exc:
            print(f"[warn] skipped optimizer state restore: {exc}")
    if ema is not None and "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
    return int(ckpt.get("step", 0)), int(ckpt.get("epoch", 0)), float(ckpt.get("best_val_loss", math.inf))


def apply_overrides(cfg, args: argparse.Namespace, world_size: int) -> None:
    if args.output_dir is not None:
        cfg.work_dir = str(args.output_dir)
    if args.seed is not None:
        cfg.seed = int(args.seed)

    train_cfg = cfg.train
    if args.global_batch_size is not None:
        denom = world_size * int(args.gradient_accumulation_steps or train_cfg.get("gradient_accumulation_steps", 1))
        if int(args.global_batch_size) % denom != 0:
            raise ValueError(f"global_batch_size={args.global_batch_size} must be divisible by world_size*grad_accum={denom}")
        train_cfg["batch_size"] = int(args.global_batch_size) // denom

    mappings = {
        "train_batch_size": "batch_size",
        "gradient_accumulation_steps": "gradient_accumulation_steps",
        "dataloader_num_workers": "num_workers",
        "val_dataloader_num_workers": "val_num_workers",
        "max_train_steps": "max_steps",
        "learning_rate": "lr",
        "lr_warmup_steps": "warmup_steps",
        "schedule_type": "schedule_type",
        "adam_weight_decay": "weight_decay",
        "adam_epsilon": "adam_eps",
        "max_grad_norm": "grad_clip",
        "ema_decay": "ema_decay",
        "mixed_precision": "mixed_precision",
        "checkpointing_steps": "save_every",
        "validation_steps": "val_every",
        "checkpointing_and_validation_steps": None,
        "max_val_batches": "max_val_batches",
        "val_batch_size": "val_batch_size",
        "validation_num_samples": "validation_num_samples",
        "validation_timestep_bins": "validation_timestep_bins",
        "logging_dir": "logging_dir",
    }
    if args.checkpointing_and_validation_steps is not None:
        train_cfg["save_every"] = int(args.checkpointing_and_validation_steps)
        train_cfg["val_every"] = int(args.checkpointing_and_validation_steps)
    for arg_name, cfg_name in mappings.items():
        if cfg_name is None:
            continue
        value = getattr(args, arg_name)
        if value is not None:
            train_cfg[cfg_name] = value

    if args.cam_drop_prob is not None:
        cfg.flow["cam_drop_prob"] = float(args.cam_drop_prob)
    if args.cond_drop_prob is not None:
        cfg.flow["cond_drop_prob"] = float(args.cond_drop_prob)
    if args.selfcond_prob is not None:
        cfg.flow["selfcond_prob"] = float(args.selfcond_prob)
    if args.vae_head_weight is not None:
        cfg.regularization["vae_head_weight"] = float(args.vae_head_weight)
    if args.vae_head_warmup_start_steps is not None:
        cfg.regularization["vae_head_warmup_start_steps"] = int(args.vae_head_warmup_start_steps)
    if args.vae_head_warmup_steps is not None:
        cfg.regularization["vae_head_warmup_steps"] = int(args.vae_head_warmup_steps)
    if args.find_unused_parameters:
        cfg.train["find_unused_parameters"] = True
    if args.no_find_unused_parameters:
        cfg.train["find_unused_parameters"] = False

    for ds_section in ("dataset", "val_dataset"):
        ds_cfg = getattr(cfg, ds_section, None)
        if ds_cfg is None:
            continue
        if args.no_index_cache:
            ds_cfg["use_index_cache"] = False
        if args.rebuild_index_cache:
            ds_cfg["use_index_cache"] = True
            ds_cfg["rebuild_index_cache"] = True
        if args.index_cache_dir is not None:
            ds_cfg["use_index_cache"] = True
            ds_cfg["index_cache_dir"] = str(args.index_cache_dir)


@torch.no_grad()
def run_validation(
    model,
    val_loader,
    cfg,
    device: torch.device,
    world_size: int,
    validation_timestep_mode: str = "independent",
    frozen_vae_head=None,
    global_step: int | None = None,
) -> dict[str, float] | None:
    if val_loader is None:
        return None
    model.eval()
    train_cfg = cfg.train
    num_bins = int(train_cfg.get("validation_timestep_bins", 5))
    bin_edges = torch.linspace(0.0, 1.0, num_bins + 1, device=device)
    totals = torch.zeros(4, device=device, dtype=torch.float64)
    bin_sums = torch.zeros(num_bins, 3, device=device, dtype=torch.float64)
    bin_counts = torch.zeros(num_bins, device=device, dtype=torch.float64)
    for i, batch in enumerate(val_loader):
        batch = to_device(batch, device)
        loss, logs = model_step(
            model,
            batch,
            cfg,
            selfcond=False,
            frozen_vae_head=frozen_vae_head,
            global_step=global_step,
            deterministic_seed=int(train_cfg.get("validation_seed", 12345)),
            deterministic_sample_ids=batch["sample_id"],
            validation_timestep_mode=validation_timestep_mode,
            force_drop_cond=bool(train_cfg.get("validation_drop_cond", False)),
            force_drop_cam=bool(train_cfg.get("validation_drop_cam", False)),
            return_per_sample=True,
        )
        batch_count = float(batch["sample_id"].numel())
        totals += torch.tensor(
            [
                float(loss.detach().item()) * batch_count,
                float(logs["loss_pos"].detach().item()) * batch_count,
                float(logs["loss_con"].detach().item()) * batch_count,
                float(logs["loss_flow"].detach().item()) * batch_count,
            ],
            device=device,
            dtype=torch.float64,
        )
        bin_ids = torch.bucketize(logs["sample_t_con"], bin_edges[1:-1])
        for bin_id in range(num_bins):
            mask = bin_ids == bin_id
            if not mask.any():
                continue
            bin_counts[bin_id] += mask.sum().to(torch.float64)
            bin_sums[bin_id, 0] += logs["sample_loss_flow"][mask].double().sum()
            bin_sums[bin_id, 1] += logs["sample_loss_pos"][mask].double().sum()
            bin_sums[bin_id, 2] += logs["sample_loss_con"][mask].double().sum()
    model.train()
    totals = reduce_sum(totals, world_size)
    bin_sums = reduce_sum(bin_sums, world_size)
    bin_counts = reduce_sum(bin_counts, world_size)
    count = totals.new_tensor(float(len(val_loader.dataset)))
    if count.item() <= 0:
        return None
    metrics = {
        "loss": float((totals[0] / count).item()),
        "loss_pos": float((totals[1] / count).item()),
        "loss_con": float((totals[2] / count).item()),
        "loss_flow": float((totals[3] / count).item()),
        "num_t_bins": float(num_bins),
    }
    for bin_id in range(num_bins):
        bin_count = bin_counts[bin_id].clamp_min(1.0)
        metrics[f"tbin{bin_id}_lo"] = float(bin_edges[bin_id].item())
        metrics[f"tbin{bin_id}_hi"] = float(bin_edges[bin_id + 1].item())
        metrics[f"tbin{bin_id}_count"] = float(bin_counts[bin_id].item())
        metrics[f"tbin{bin_id}_loss"] = float((bin_sums[bin_id, 0] / bin_count).item())
        metrics[f"tbin{bin_id}_pos"] = float((bin_sums[bin_id, 1] / bin_count).item())
        metrics[f"tbin{bin_id}_con"] = float((bin_sums[bin_id, 2] / bin_count).item())
    return metrics


def train(args: argparse.Namespace) -> None:
    rank, world_size, local_rank, device = init_distributed()
    try:
        cfg = load_config(args.config)
        apply_overrides(cfg, args, world_size)
        train_cfg = dict(cfg.train)
        work_dir = Path(cfg.work_dir)
        train_log_path = work_dir / "train.log"
        val_log_path = work_dir / "val.log"

        if bool(train_cfg.get("allow_tf32", True)) and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        set_all_seeds(int(cfg.seed), rank)
        if is_main_process(rank):
            work_dir.mkdir(parents=True, exist_ok=True)

        train_ds = build_dataset(dict(cfg.dataset))
        val_ds = build_dataset(dict(cfg.val_dataset))
        train_loader, train_sampler = build_loader(
            train_ds,
            train_cfg,
            rank=rank,
            world_size=world_size,
            shuffle=True,
            drop_last=True,
            num_workers=int(train_cfg["num_workers"]),
        )
        val_loader, _, effective_val_samples = build_validation_loader(
            val_ds,
            train_cfg,
            rank=rank,
            world_size=world_size,
        )

        model = build_model(cfg).to(device)
        freeze_pos_delta = str(getattr(cfg, "model", {}).get("content_rope_impl", "pointrope")) == "pointrope"
        frozen_param_names = freeze_structurally_unused_params(model, enabled=freeze_pos_delta)
        frozen_null_names = freeze_unused_null_condition_params(model, cond_drop_prob=float(cfg.flow.get("cond_drop_prob", 0.0)))
        frozen_aux_names = freeze_unused_aux_head(
            model,
            vae_head_weight=float(cfg.regularization.get("vae_head_weight", 0.0)),
            vae_head_source=str(cfg.regularization.get("vae_head_source", "final")),
        )
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(train_cfg["lr"]),
            betas=tuple(train_cfg.get("adam_betas", (0.9, 0.95))),
            eps=float(train_cfg.get("adam_eps", 1e-8)),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )
        ema_decay = float(train_cfg.get("ema_decay", 0.0) or 0.0)
        ema = ModelEma(model, ema_decay) if ema_decay > 0 else None
        frozen_vae_head = build_frozen_vae_head(cfg, device)

        if world_size > 1:
            ddp_kwargs = {"device_ids": [local_rank], "output_device": local_rank} if device.type == "cuda" else {}
            ddp_kwargs["find_unused_parameters"] = bool(train_cfg.get("find_unused_parameters", True))
            model = DDP(model, **ddp_kwargs)

        start_step = 0
        start_epoch = 0
        best_val_loss = math.inf
        resume_path = args.resume
        if resume_path is None and train_cfg.get("resume"):
            resume_path = Path(str(train_cfg["resume"]))
        if resume_path is not None and Path(resume_path).is_file():
            start_step, start_epoch, best_val_loss = load_checkpoint(
                Path(resume_path),
                model,
                optimizer,
                ema,
                resume_optimizer=not args.no_resume_optimizer,
            )
            if is_main_process(rank):
                print(f"Resumed from {resume_path} at step={start_step}, epoch={start_epoch}, best_val_loss={best_val_loss:.5f}")

        grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
        max_steps = int(train_cfg["max_steps"])
        save_every = int(train_cfg.get("save_every", 5000))
        val_every = int(train_cfg.get("val_every", save_every))
        log_every = int(train_cfg.get("log_every", 1))
        mixed_precision = str(train_cfg.get("mixed_precision", "bf16")).lower()
        amp_dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
        amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda" and mixed_precision != "no"

        if is_main_process(rank):
            total_batch = int(train_cfg["batch_size"]) * world_size * grad_accum
            print("***** Running GLD-style DDP training *****")
            print(f"  Num examples = {len(train_ds)}")
            print(f"  Num validation examples = {len(val_ds)}")
            print(f"  Fixed validation samples = {effective_val_samples}")
            print(f"  Per-device batch size = {int(train_cfg['batch_size'])}")
            print(f"  Per-device validation batch size = {int(train_cfg.get('val_batch_size', train_cfg['batch_size']))}")
            print(f"  Global batch size = {total_batch}")
            print(f"  Gradient accumulation steps = {grad_accum}")
            print(f"  Max steps = {max_steps}")
            print(f"  LR schedule = {train_cfg.get('schedule_type', 'constant')}")
            print(f"  EMA decay = {ema_decay}")
            print(f"  cam_drop_prob = {cfg.flow.get('cam_drop_prob', 0.0)}")
            print(f"  cond_drop_prob = {cfg.flow.get('cond_drop_prob', 0.0)}")
            print(f"  DDP find_unused_parameters = {bool(train_cfg.get('find_unused_parameters', True))}")
            print(f"  Frozen structurally unused params = {len(frozen_param_names)}")
            print(f"  Frozen unused null condition params = {frozen_null_names}")
            print(f"  Frozen unused aux head params = {frozen_aux_names}")
            append_log(
                train_log_path,
                (
                    f"# run start step={start_step} epoch={start_epoch} max_steps={max_steps} "
                    f"global_batch={total_batch} lr={float(train_cfg['lr']):.6e}"
                ),
            )
            append_log(val_log_path, f"# run start step={start_step} epoch={start_epoch} best_val_loss={best_val_loss}")

        step = start_step
        epoch = start_epoch
        running: dict[str, float] = {}
        model.train()
        progress = tqdm(total=max_steps, initial=step, disable=not is_main_process(rank), desc="Steps")

        while step < max_steps:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            for batch_idx, batch in enumerate(train_loader):
                batch = to_device(batch, device)
                sync_now = ((batch_idx + 1) % grad_accum == 0)
                no_sync = model.no_sync if isinstance(model, DDP) and not sync_now else contextlib.nullcontext
                with no_sync():
                    lr = scheduled_lr(step, max_steps, train_cfg)
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                        loss, logs = model_step(model, batch, cfg, selfcond=True, frozen_vae_head=frozen_vae_head, global_step=step)
                        loss = loss / float(grad_accum)
                    loss.backward()

                for k in sorted(logs):
                    v = logs[k]
                    running[k] = running.get(k, 0.0) + float(reduce_mean(v, world_size).item()) / float(grad_accum)

                if not sync_now:
                    continue

                if float(train_cfg.get("grad_clip", 0.0)) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    raw_model = model.module if isinstance(model, DDP) else model
                    ema.update(raw_model)

                step += 1
                progress.update(1)

                if is_main_process(rank) and step % log_every == 0:
                    denom = float(log_every)
                    loss_s = running.get("loss", 0.0) / denom
                    pos_s = running.get("loss_pos", 0.0) / denom
                    con_s = running.get("loss_con", 0.0) / denom
                    extras = format_loss_extras(running, denom)
                    extras_s = " " + extras if extras else ""
                    train_line = f"step={step} lr={lr:.3e} loss={loss_s:.5f} pos={pos_s:.5f} con={con_s:.5f}{extras_s}"
                    print(train_line)
                    append_log(train_log_path, train_line)
                    running = {}

                if step % save_every == 0:
                    if world_size > 1:
                        dist.barrier()
                    if is_main_process(rank):
                        save_checkpoint(work_dir / "latest.pt", model, optimizer, step, epoch, args.config, ema, best_val_loss)
                    if world_size > 1:
                        dist.barrier()

                if val_loader is not None and step % val_every == 0:
                    val_metrics = run_validation(
                        model,
                        val_loader,
                        cfg,
                        device,
                        world_size,
                        "independent",
                        frozen_vae_head=frozen_vae_head,
                        global_step=step,
                    )
                    path_metrics = None
                    if bool(train_cfg.get("validation_position_leading", True)):
                        path_metrics = run_validation(
                            model,
                            val_loader,
                            cfg,
                            device,
                            world_size,
                            "position_leading",
                            frozen_vae_head=frozen_vae_head,
                            global_step=step,
                        )
                    if is_main_process(rank) and val_metrics is not None:
                        val_loss = val_metrics["loss"]
                        improved = val_loss < best_val_loss
                        if improved:
                            best_val_loss = val_loss
                            save_checkpoint(work_dir / "best.pt", model, optimizer, step, epoch, args.config, ema, best_val_loss)
                        val_line = f"step={step} {format_val_metrics(val_metrics)}"
                        if path_metrics is not None:
                            val_line += " " + format_val_metrics(path_metrics, prefix="path_")
                        val_line += f" best={best_val_loss:.5f}"
                        if improved:
                            val_line += " saved_best=1"
                        print(f"val {val_line}")
                        append_log(val_log_path, val_line)
                    if world_size > 1:
                        dist.barrier()

                if step >= max_steps:
                    break
            epoch += 1

        if world_size > 1:
            dist.barrier()
        if is_main_process(rank):
            save_checkpoint(work_dir / "latest.pt", model, optimizer, step, epoch, args.config, ema, best_val_loss)
        progress.close()
    finally:
        cleanup_distributed()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GLD-aligned torchrun/DDP training for dual-timestep DiT v4.")
    parser.add_argument("--config", type=Path, default=Path("configs/stage1.py"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--global_batch_size", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--dataloader_num_workers", type=int, default=None)
    parser.add_argument("--val_dataloader_num_workers", type=int, default=None)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--lr_warmup_steps", type=int, default=None)
    parser.add_argument("--schedule_type", choices=("constant", "cosine"), default=None)
    parser.add_argument("--adam_weight_decay", type=float, default=None)
    parser.add_argument("--adam_epsilon", type=float, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--ema_decay", type=float, default=None)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default=None)
    parser.add_argument("--cam_drop_prob", type=float, default=None)
    parser.add_argument("--cond_drop_prob", type=float, default=None)
    parser.add_argument("--selfcond_prob", type=float, default=None)
    parser.add_argument("--vae_head_weight", type=float, default=None)
    parser.add_argument("--vae_head_warmup_start_steps", type=int, default=None)
    parser.add_argument("--vae_head_warmup_steps", type=int, default=None)
    parser.add_argument("--checkpointing_and_validation_steps", type=int, default=None)
    parser.add_argument("--checkpointing_steps", type=int, default=None)
    parser.add_argument("--validation_steps", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--val_batch_size", type=int, default=None)
    parser.add_argument("--validation_num_samples", type=int, default=None)
    parser.add_argument("--validation_timestep_bins", type=int, default=None)
    parser.add_argument("--logging_dir", type=str, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--no_resume_optimizer", action="store_true")
    parser.add_argument("--index_cache_dir", type=Path, default=None)
    parser.add_argument("--rebuild_index_cache", action="store_true")
    parser.add_argument("--no_index_cache", action="store_true")
    parser.add_argument("--find_unused_parameters", action="store_true")
    parser.add_argument("--no_find_unused_parameters", action="store_true")
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
