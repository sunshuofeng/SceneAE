import argparse
import datetime
import inspect
import importlib
import importlib.machinery
import math
import os
import os.path as osp
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from mmengine import Config
from mmengine.logging import MMLogger
from mmengine.optim import build_optim_wrapper
from mmengine.runner import set_random_seed
from mmengine.utils import symlink
from timm.scheduler import CosineLRScheduler, MultiStepLRScheduler
from torch.amp import autocast


def install_optional_dependency_stubs() -> None:
    moviepy_stub = types.ModuleType("moviepy")
    moviepy_editor_stub = types.ModuleType("moviepy.editor")
    moviepy_stub.__spec__ = importlib.machinery.ModuleSpec("moviepy", loader=None, is_package=True)
    moviepy_editor_stub.__spec__ = importlib.machinery.ModuleSpec("moviepy.editor", loader=None)
    moviepy_stub.editor = moviepy_editor_stub
    sys.modules.setdefault("moviepy", moviepy_stub)
    sys.modules.setdefault("moviepy.editor", moviepy_editor_stub)


install_optional_dependency_stubs()

from utils import collate_points
import torch.nn.functional as F


def load_model_class(cfg: Config) -> type:
    module_name = str(cfg.get("model_module", "models.scene_vae"))
    class_name = str(cfg.get("model_class", "SuperpointVAEGridV30"))
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _instantiate_model(model_cls: type, **kwargs):
    sig = inspect.signature(model_cls.__init__)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return model_cls(**kwargs)
    allowed = {
        key
        for key, param in params.items()
        if key != "self"
        and param.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return model_cls(**{key: value for key, value in kwargs.items() if key in allowed})


def maybe_resolve_feature_dims(cfg: Config) -> None:
    if str(cfg.get("dataset_name", "")).lower() not in ("scannet_chunkv2", "scannet_chunkv2_c15"):
        return
    if not bool(cfg.get("auto_feature_dims", True)):
        return
    dataset_kwargs = dict(cfg.get("dataset_kwargs", {}))
    superpoint_root = dataset_kwargs.get("superpoint_root", None)
    if superpoint_root is None:
        return
    from dataset.scannet import infer_chunk_feature_dims

    dims = infer_chunk_feature_dims(Path(str(superpoint_root)))
    cfg.encoder_img_feat_in_dim = int(dims["encoder_img_feat_in_dim"])
    cfg.dino_feat_dim = int(dims["dino_dim"])
    cfg.da3_feat_dim = int(dims["da3_dim"])
    cfg.vae_latent_channels = int(dims["vae_dim"])


def pass_print(*args, **kwargs):
    pass


def batch_to_device(batch: dict, non_blocking: bool = True) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.cuda(non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v
    return out


def _debug_assert_named_tensor_finite(kind: str, name: str, value: torch.Tensor) -> None:
    if value is None or value.numel() == 0:
        return
    finite = torch.isfinite(value)
    if bool(finite.all()):
        return
    value_detached = value.detach()
    finite_values = value_detached[finite]
    if finite_values.numel() > 0:
        min_val = float(finite_values.float().min().cpu())
        max_val = float(finite_values.float().max().cpu())
        mean_val = float(finite_values.float().mean().cpu())
    else:
        min_val = float("nan")
        max_val = float("nan")
        mean_val = float("nan")
    nan_ratio = float(torch.isnan(value_detached).float().mean().cpu())
    posinf_ratio = float(torch.isposinf(value_detached).float().mean().cpu())
    neginf_ratio = float(torch.isneginf(value_detached).float().mean().cpu())
    raise FloatingPointError(
        f"[NaNCheck] {kind}.{name}: shape={tuple(value.shape)} dtype={value.dtype} "
        f"nan_ratio={nan_ratio:.6g} posinf_ratio={posinf_ratio:.6g} "
        f"neginf_ratio={neginf_ratio:.6g} finite_min={min_val:.6g} "
        f"finite_max={max_val:.6g} finite_mean={mean_val:.6g}"
    )


def debug_assert_model_finite(model: torch.nn.Module, kind: str) -> None:
    for name, param in model.named_parameters():
        if kind == "param":
            _debug_assert_named_tensor_finite(kind, name, param)
        elif kind == "grad" and param.grad is not None:
            _debug_assert_named_tensor_finite(kind, name, param.grad)


def _reduce_metrics(metrics: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not dist.is_available() or not dist.is_initialized():
        return metrics
    out = {}
    world = dist.get_world_size()
    for key, val in metrics.items():
        t = val.detach().clone()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        out[key] = t / world
    return out


def _reduce_metric_sums(metrics: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not dist.is_available() or not dist.is_initialized():
        return metrics
    out = {}
    for key, val in metrics.items():
        t = val.detach().clone()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        out[key] = t
    return out


def _finalize_eval_metrics(sums: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    rgb_mse = sums["rgb_sse"] / sums["rgb_count"].clamp_min(1.0)
    depth_mse = sums["depth_sse"] / sums["depth_count"].clamp_min(1.0)
    depth_valid_ratio = sums["depth_valid_sum"] / sums["depth_valid_count"].clamp_min(1.0)
    rgb_psnr = -10.0 * torch.log10(rgb_mse.clamp_min(1e-10))
    rgb_ssim = sums["rgb_ssim_sum"] / sums["rgb_ssim_count"].clamp_min(1.0)
    return {
        "loss": rgb_mse + depth_mse,
        "rgb_mse": rgb_mse,
        "rgb_psnr": rgb_psnr,
        "rgb_ssim": rgb_ssim,
        "depth_mse": depth_mse,
        "depth_valid_ratio": depth_valid_ratio,
    }


def _build_per_dataset_val_loaders(cfg: Config, distributed: bool, collate_fn):
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from dataset.dl3dv import DL3DVSuperpointChunkDatasetV2C15
    from dataset.re10k import RE10KSuperpointChunkDatasetV2C15
    from dataset.scannet import ScanNetSuperpointChunkDatasetV2C15

    # V2 cached-latent training: training reads first-frame-gen chunks + z_sp
    # sidecars (cfg.dataset_kwargs), while validation keeps the original C15
    # chunks with the frozen online encoder for v20-comparable metrics.
    dataset_kwargs = dict(cfg.get("eval_dataset_kwargs", None) or cfg.get("dataset_kwargs", {}))
    child_specs = dict(dataset_kwargs.get("datasets", {}))
    eval_names = list(cfg.get("eval_datasets", dataset_kwargs.get("val_datasets", child_specs.keys())))
    loaders = {}

    shared_kwargs = dict(dataset_kwargs)
    for key in ("datasets", "mix_ratio", "equal_ratio", "val_datasets", "val_mix_ratio"):
        shared_kwargs.pop(key, None)

    for name in eval_names:
        if name not in child_specs:
            raise KeyError(f"eval dataset {name!r} not found in dataset_kwargs.datasets")
        child_kwargs = dict(shared_kwargs)
        child_kwargs.update(dict(child_specs[name]))
        dataset_name = str(cfg.get("eval_dataset_name_map", {}).get(name, _default_eval_dataset_name(name)))
        dataset_cls = {
            "scannet_chunkv2_c15": ScanNetSuperpointChunkDatasetV2C15,
            "dl3dv_chunkv2_c15": DL3DVSuperpointChunkDatasetV2C15,
            "re10k_chunkv2_c15": RE10KSuperpointChunkDatasetV2C15,
        }.get(dataset_name.lower())
        if dataset_cls is None:
            raise KeyError(f"Unsupported eval dataset_name={dataset_name!r} for eval dataset {name!r}")

        child_kwargs = _filter_kwargs_for_class(dataset_cls, child_kwargs)
        val_dataset = dataset_cls(split="test", **child_kwargs)
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False) if distributed else None
        num_workers = int(cfg.get("val_num_workers", cfg.get("num_workers", 4)))
        persistent_workers = bool(cfg.get("persistent_workers", True)) and num_workers > 0
        loader_kwargs = dict(
            pin_memory=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=int(cfg.get("val_batch_size", cfg.get("train_batch_size", 2))),
            shuffle=False,
            sampler=val_sampler,
            **loader_kwargs,
        )
        loaders[str(name)] = val_loader
    return loaders


def _build_ffg_cached_val_loaders(cfg: Config, distributed: bool, collate_fn):
    """Per-dataset first-frame-gen CACHED val loaders (split='test').

    These feed the decoder through the cached-z_sp path (DA3 + encoder bypassed),
    i.e. the SAME input distribution as training.  Comparing these metrics against
    the C15 online val disentangles in-distribution jitter-blur (this val) from
    the FFG->C15 domain shift (the gap between the two vals).

    Enabled via cfg.ffg_val_enabled; capped per dataset via cfg.ffg_val_max_per_dataset
    (0 = no cap) to keep the extra val cheap.  Returns {} when disabled.
    """
    from torch.utils.data import DataLoader, Subset
    from torch.utils.data.distributed import DistributedSampler
    from dataset.ffg_cached import FFGCachedChunkDataset

    if not bool(cfg.get("ffg_val_enabled", False)):
        return {}
    dataset_kwargs = dict(cfg.get("dataset_kwargs", {}) or {})
    child_specs = dict(dataset_kwargs.get("datasets", {}))
    eval_names = list(cfg.get("eval_datasets", dataset_kwargs.get("val_datasets", child_specs.keys())))
    max_per = int(cfg.get("ffg_val_max_per_dataset", 0))
    loaders = {}

    shared_kwargs = dict(dataset_kwargs)
    for key in ("datasets", "mix_ratio", "equal_ratio", "val_datasets", "val_mix_ratio"):
        shared_kwargs.pop(key, None)

    for name in eval_names:
        if name not in child_specs:
            continue
        child_kwargs = dict(shared_kwargs)
        child_kwargs.update(dict(child_specs[name]))
        child_kwargs = _filter_kwargs_for_class(FFGCachedChunkDataset, child_kwargs)
        try:
            val_dataset = FFGCachedChunkDataset(split="test", **child_kwargs)
        except Exception as exc:
            print(f"[ffg_val] failed to build val dataset for {name!r}: {exc}")
            continue
        if len(val_dataset) == 0:
            print(f"[ffg_val] {name!r}: 0 cached val chunks (cache may not cover val scenes); skipped.")
            continue
        if max_per > 0 and len(val_dataset) > max_per:
            # Deterministic evenly-spaced subset -> fast, stable metric.
            idx = torch.linspace(0, len(val_dataset) - 1, steps=max_per).round().long().tolist()
            val_dataset = Subset(val_dataset, idx)
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False) if distributed else None
        num_workers = int(cfg.get("val_num_workers", cfg.get("num_workers", 4)))
        persistent_workers = bool(cfg.get("persistent_workers", True)) and num_workers > 0
        loader_kwargs = dict(
            pin_memory=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=int(cfg.get("val_batch_size", cfg.get("train_batch_size", 2))),
            shuffle=False,
            sampler=val_sampler,
            **loader_kwargs,
        )
        loaders[f"ffg_{name}"] = val_loader
    return loaders


def _filter_kwargs_for_class(cls, kwargs: dict) -> dict:
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    allowed = {
        key
        for key, param in params.items()
        if key != "self"
        and param.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return {key: value for key, value in kwargs.items() if key in allowed}


def _default_eval_dataset_name(name: str) -> str:
    name = str(name).lower()
    if name == "scannet":
        return "scannet_chunkv2_c15"
    if name == "dl3dv":
        return "dl3dv_chunkv2_c15"
    if name == "re10k":
        return "re10k_chunkv2_c15"
    return name


def _format_metrics(metrics: dict[str, torch.Tensor], keys: list[str], aliases: dict[str, str] | None = None) -> str:
    aliases = aliases or {}
    parts = []
    for key in keys:
        name = aliases.get(key, key)
        value = float(metrics[key].item())
        if abs(value) < 1e-4 and value != 0.0:
            parts.append(f"{name}={value:.8e}")
        else:
            parts.append(f"{name}={value:.8f}")
    return " ".join(parts)


def backup_training_sources(work_dir: str, py_config: str, model_cls: type) -> list[str]:
    backup_dir = Path(work_dir) / "source_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parent

    source_paths: list[Path] = []

    train_path = Path(__file__).resolve()
    source_paths.append(train_path)

    cfg_path = Path(py_config).resolve()
    if cfg_path.is_file():
        source_paths.append(cfg_path)

    model_src = inspect.getsourcefile(model_cls)
    if model_src:
        model_path = Path(model_src).resolve()
        if model_path.is_file():
            source_paths.append(model_path)

    copied: list[str] = []
    seen: set[Path] = set()
    for src in source_paths:
        if src in seen or not src.is_file():
            continue
        seen.add(src)
        try:
            rel = src.relative_to(repo_root)
            dst = backup_dir / rel
        except ValueError:
            dst = backup_dir / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(str(dst))

    manifest_path = backup_dir / "manifest.txt"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for path in copied:
            f.write(path + "\n")
    copied.append(str(manifest_path))
    return copied


def filter_compatible_state_dict(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], list[str]]:
    current = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in current:
            skipped.append(f"{key}: missing in current model")
            continue
        if current[key].shape != value.shape:
            skipped.append(f"{key}: checkpoint {tuple(value.shape)} != current {tuple(current[key].shape)}")
            continue
        filtered[key] = value
    return filtered, skipped


@torch.no_grad()
def _gather_selected_gt_rgb_full(
    image: torch.Tensor,
    view_idx_rgb: torch.Tensor | None,
    v_rgb: int,
) -> torch.Tensor:
    b, v, h, w, c = image.shape
    if view_idx_rgb is None:
        return image[:, :v_rgb]
    view_idx_e = view_idx_rgb.long()[..., None, None, None].expand(b, v_rgb, h, w, c)
    return torch.gather(image, 1, view_idx_e)


def _gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    kernel_2d = torch.outer(g, g)
    window = kernel_2d.view(1, 1, window_size, window_size)
    return window.expand(channels, 1, window_size, window_size).contiguous()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    pred = pred.float().clamp(0.0, 1.0)
    target = target.float().clamp(0.0, 1.0)
    c = pred.shape[1]
    window = _gaussian_window(window_size, sigma, c, pred.device, pred.dtype)
    padding = window_size // 2

    mu_x = F.conv2d(pred, window, padding=padding, groups=c)
    mu_y = F.conv2d(target, window, padding=padding, groups=c)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, window, padding=padding, groups=c) - mu_x2
    sigma_y2 = F.conv2d(target * target, window, padding=padding, groups=c) - mu_y2
    sigma_xy = F.conv2d(pred * target, window, padding=padding, groups=c) - mu_xy

    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-12)
    return ssim_map.mean(dim=(1, 2, 3))


@torch.no_grad()
def evaluate(model, val_loader, decode_rgb: bool, use_fps_spxyz: bool = False, cached: bool = False) -> dict[str, torch.Tensor]:
    model.eval()
    prev_infer_use_fps_spxyz = getattr(model, "infer_use_fps_spxyz", False)
    model.infer_use_fps_spxyz = bool(use_fps_spxyz)
    device = next(model.parameters()).device
    sums = {
        "rgb_sse": torch.zeros((), device=device, dtype=torch.float64),
        "rgb_count": torch.zeros((), device=device, dtype=torch.float64),
        "rgb_ssim_sum": torch.zeros((), device=device, dtype=torch.float64),
        "rgb_ssim_count": torch.zeros((), device=device, dtype=torch.float64),
        "depth_sse": torch.zeros((), device=device, dtype=torch.float64),
        "depth_count": torch.zeros((), device=device, dtype=torch.float64),
        "depth_valid_sum": torch.zeros((), device=device, dtype=torch.float64),
        "depth_valid_count": torch.zeros((), device=device, dtype=torch.float64),
    }
    for batch in val_loader:
        batch = batch_to_device(batch)
        with autocast("cuda", dtype=AMP_DTYPE):
            if cached:
                # FFG cached-latent path: feed the DiT z_sp sidecar straight to
                # the decoder (DA3 + encoder bypassed), matching the training
                # input distribution so this val is directly comparable to train.
                result = model(
                    sp_feat=batch["sp_feat"],
                    camtoworlds=batch["norm_camtoworlds"],
                    Ks=batch["Ks"],
                    image=batch["image"],
                    sp_xyz=batch["norm_sp_xyz"],
                    valid_mask=batch.get("valid_mask", None),
                    sp_valid_mask=batch.get("superpoint_valid_mask", None),
                    decode_rgb=decode_rgb,
                    cached_z_sp=batch["cached_z_sp"],
                )
            else:
                result = model(
                    sp_feat=batch["sp_feat"],
                    camtoworlds=batch["norm_camtoworlds"],
                    Ks=batch["Ks"],
                    image=batch["image"],
                    sp_xyz=batch["norm_sp_xyz"],
                    da3_feature_map=batch.get("da3_feature_map", None),
                    da3_feature_maps=batch.get("da3_feature_maps", None),
                    norm_depth=batch.get("norm_depth", None),
                    valid_mask=batch.get("valid_mask", None),
                    norm_pw=batch.get("norm_pw", None),
                    norm_pw_offsets=batch.get("norm_pw_offsets", None),
                    decode_rgb=decode_rgb,
                )

        pred_rgb = result["rendered_rgb"]
        pred_depth = result["rendered_depths"][..., 0]
        v_rgb = pred_rgb.shape[1]
        v_depth = pred_depth.shape[1]

        gt_rgb = _gather_selected_gt_rgb_full(batch["image"], result.get("selected_view_idx_rgb", None), v_rgb).to(pred_rgb.dtype)
        gt_depth = batch["norm_depth"]
        if gt_depth.dim() == 3:
            gt_depth = gt_depth.unsqueeze(1).expand(gt_depth.shape[0], v_depth, *gt_depth.shape[1:])
        valid = batch["valid_mask"].bool()
        if valid.dim() == 3:
            valid = valid.unsqueeze(1).expand(valid.shape[0], v_depth, *valid.shape[1:])

        rgb_diff2 = (pred_rgb.float() - gt_rgb.float()).pow(2)
        sums["rgb_sse"] += rgb_diff2.sum().double()
        sums["rgb_count"] += torch.tensor(float(rgb_diff2.numel()), device=device, dtype=torch.float64)
        pred_rgb_nchw = pred_rgb.float().clamp(0.0, 1.0).reshape(-1, *pred_rgb.shape[2:]).permute(0, 3, 1, 2).contiguous()
        gt_rgb_nchw = gt_rgb.float().clamp(0.0, 1.0).reshape(-1, *gt_rgb.shape[2:]).permute(0, 3, 1, 2).contiguous()
        rgb_ssim_per = compute_ssim(pred_rgb_nchw, gt_rgb_nchw)
        sums["rgb_ssim_sum"] += rgb_ssim_per.sum().double()
        sums["rgb_ssim_count"] += torch.tensor(float(rgb_ssim_per.numel()), device=device, dtype=torch.float64)

        # Depth is produced under autocast during validation.  Compute metrics in
        # fp32 so small but real DA-depth errors are not rounded away by fp16.
        depth_diff2 = (pred_depth.float() - gt_depth.float()).pow(2)
        valid_f = valid.float()
        sums["depth_sse"] += (depth_diff2 * valid_f).sum().double()
        sums["depth_count"] += valid_f.sum().double()
        sums["depth_valid_sum"] += valid_f.sum().double()
        sums["depth_valid_count"] += torch.tensor(float(valid_f.numel()), device=device, dtype=torch.float64)
    model.infer_use_fps_spxyz = prev_infer_use_fps_spxyz
    return sums


# Autocast dtype for train + eval forwards. Set from cfg.amp_dtype in main().
# bfloat16 avoids fp16 overflow, so no GradScaler is needed.
AMP_DTYPE = torch.bfloat16


class EMA:
    """Exponential moving average of model weights (params + buffers).

    Validation and the best checkpoint use the EMA weights, matching GLD.  The
    live training weights are swapped out via store()/copy_to()/restore() around
    evaluation so training is unaffected.
    """

    def __init__(self, model, decay: float):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self._backup = None

    @torch.no_grad()
    def update(self, model, num_updates: int | None = None) -> None:
        d = self.decay
        if num_updates is not None:
            # Ramp the decay early so the EMA is not stuck near init.
            d = min(d, (1.0 + num_updates) / (10.0 + num_updates))
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                s.copy_(v)  # integer buffers (e.g. num_batches_tracked)

    def store(self, model) -> None:
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def copy_to(self, model) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model) -> None:
        if self._backup is not None:
            model.load_state_dict(self._backup, strict=True)
            self._backup = None

    def state_dict(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: dict, device) -> None:
        for k in self.shadow:
            if k in sd:
                self.shadow[k].copy_(sd[k].to(device))


def _clip_grad_groups(model, max_norm: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Per-parameter-group gradient clipping (design §6.3).

    Groups: {frontend + shared trunk} / {rgb blocks + rgb head} /
    {depth branch + depth head}.  Prevents a collapsing branch (e.g. depth) from
    shrinking the others' effective LR the way a single global clip would.
    Returns (total_norm, per_group_pre_clip_norms).
    """
    groups: dict[str, list] = {"frontend_trunk": [], "rgb": [], "depth": [], "other": []}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if "decoder.frontend." in name or "decoder.shared_blocks." in name:
            groups["frontend_trunk"].append(p)
        elif "decoder.rgb_blocks." in name or "decoder.rgb_head." in name:
            groups["rgb"].append(p)
        elif (
            "decoder.depth_proj" in name
            or "decoder.depth_prep_blocks." in name
            or "decoder.depth_blocks." in name
            or "decoder.depth_head." in name
        ):
            groups["depth"].append(p)
        else:
            groups["other"].append(p)
    norms: dict[str, torch.Tensor] = {}
    for gname, params in groups.items():
        if params:
            norms[gname] = torch.nn.utils.clip_grad_norm_(params, max_norm)
    if norms:
        total = torch.sqrt(sum(n.float() ** 2 for n in norms.values()))
    else:
        total = torch.zeros((), device=next(model.parameters()).device)
    return total, norms


def _jitter_sigma_for_epoch(epoch: int, sigma_max: float, start: int, end: int) -> float:
    """Decode-side jitter sigma curriculum (design §6.2): sigma=0 for
    epoch < start, linear ramp to sigma_max over [start, end], hold after."""
    if sigma_max <= 0:
        return 0.0
    if epoch < start:
        return 0.0
    if epoch >= end:
        return float(sigma_max)
    return float(sigma_max) * float(epoch - start) / float(max(end - start, 1))


def build_model_from_cfg(cfg, image_hw=None, for_eval: bool = False):
    """Instantiate the v27 (or compatible) model from a training config.

    ``for_eval=True`` zeros LPIPS so test scripts do not download/load VGG.
    ``image_hw`` overrides ``grid_image_size`` when the eval resolution differs.
    """
    model_cls = load_model_class(cfg)
    grid_image_size = tuple(image_hw) if image_hw is not None else tuple(cfg.get("grid_image_size", (378, 518)))
    lpips_w = 0.0 if for_eval else float(cfg.get("lpips_loss_weight", 0.0))
    rgb_lpips_w = 0.0 if for_eval else float(cfg.get("rgb_lpips_weight", 0.3))
    return _instantiate_model(
        model_cls,
        latent_dim=int(cfg.get("latent_dim", 128)),
        encoder_img_feat_in_dim=int(cfg.get("encoder_img_feat_in_dim", 1024)),
        num_da3_levels=int(cfg.get("num_da3_levels", 4)),
        dino_dim=int(cfg.get("dino_feat_dim", 24)),
        da3_dim=int(cfg.get("da3_feat_dim", 24)),
        vae_latent_channels=int(cfg.get("vae_latent_channels", 16)),
        encoder_hidden_dim=int(cfg.get("encoder_hidden_dim", 256)),
        encoder_pooled_level_dim=int(cfg.get("encoder_pooled_level_dim", 128)),
        encoder_layers=int(cfg.get("encoder_layers", 8)),
        encoder_heads=int(cfg.get("encoder_heads", 8)),
        encoder_mlp_ratio=float(cfg.get("encoder_mlp_ratio", 4.0)),
        encoder_max_views=int(cfg.get("encoder_max_views", 4)),
        encoder_view_capacity=int(cfg.get("encoder_view_capacity", cfg.get("dataset_kwargs", {}).get("num_views", cfg.get("encoder_max_views", 4)))),
        encoder_query_view_topk=int(cfg.get("encoder_query_view_topk", 0)),
        encoder_query_init_topk=int(cfg.get("encoder_query_init_topk", cfg.get("encoder_query_view_topk", 0))),
        encoder_query_attn_topk=int(cfg.get("encoder_query_attn_topk", cfg.get("encoder_query_view_topk", 0))),
        encoder_query_attn_topk_schedule=tuple(cfg.get("encoder_query_attn_topk_schedule", ())),
        encoder_mv_residual_topk_schedule=tuple(cfg.get("encoder_mv_residual_topk_schedule", ())),
        encoder_mv_residual_deform_points=int(cfg.get("encoder_mv_residual_deform_points", cfg.get("encoder_deform_points", 4))),
        encoder_level_logits_tau_main=float(cfg.get("encoder_level_logits_tau_main", 0.3)),
        encoder_level_logits_tau_mv=float(cfg.get("encoder_level_logits_tau_mv", 1.0)),
        encoder_unified_fuse_mlp_ratio=float(cfg.get("encoder_unified_fuse_mlp_ratio", 4.0)),
        online_da3=bool(cfg.get("online_da3", True)),
        da3_weights=cfg.get("da3_weights", "/home/shuofeng/DA3-BASE"),
        da3_feat_layers=tuple(cfg.get("da3_feat_layers", (11, 15, 19, 23))),
        da3_process_res=int(cfg.get("da3_process_res", 320)),
        da3_process_res_method=str(cfg.get("da3_process_res_method", "upper_bound_resize")),
        da3_use_camera_tokens=bool(cfg.get("da3_use_camera_tokens", False)),
        da3_normalization_stat_path=cfg.get("da3_normalization_stat_path", None),
        da3_normalization_eps=float(cfg.get("da3_normalization_eps", 1e-5)),
        decoder_cross_layers=int(cfg.get("decoder_cross_layers", 6)),
        decoder_rgb_grid_channels=int(cfg.get("decoder_rgb_grid_channels", 64)),
        decoder_depth_grid_channels=int(cfg.get("decoder_depth_grid_channels", cfg.get("decoder_rgb_grid_channels", 64))),
        decoder_rgb_cross_layers=int(cfg.get("decoder_rgb_cross_layers", cfg.get("decoder_cross_layers", 6))),
        decoder_depth_cross_layers=int(cfg.get("decoder_depth_cross_layers", cfg.get("decoder_cross_layers", 6))),
        decoder_rgb_num_views=int(cfg.get("decoder_rgb_num_views", 2)),
        decoder_depth_num_views=int(cfg.get("decoder_depth_num_views", cfg.get("decoder_rgb_num_views", 2))),
        decoder_cross_heads=int(cfg.get("decoder_cross_heads", 4)),
        decoder_rgb_cross_heads=cfg.get("decoder_rgb_cross_heads", None),
        decoder_depth_cross_heads=cfg.get("decoder_depth_cross_heads", None),
        decoder_knn_k=int(cfg.get("decoder_knn_k", 20)),
        vae_scale_factor=int(cfg.get("vae_scale_factor", 8)),
        rgb_loss_weight=float(cfg.get("rgb_loss_weight", 1.0)),
        depth_loss_weight=float(cfg.get("depth_loss_weight", 1.0)),
        lpips_loss_weight=lpips_w,
        latent_reg_weight=float(cfg.get("latent_reg_weight", 0.01)),
        sp_feat_loss_weight=float(cfg.get("sp_feat_loss_weight", 0.3)),
        vae_distill_weight=float(cfg.get("vae_distill_weight", 1.0)),
        rgb_fullres_weight=float(cfg.get("rgb_fullres_weight", 0.5)),
        rgb_grad_weight=float(cfg.get("rgb_grad_weight", 0.2)),
        dino_cos_weight=float(cfg.get("dino_cos_weight", 0.5)),
        dino_mse_weight=float(cfg.get("dino_mse_weight", 0.5)),
        da3_teacher_weight=float(cfg.get("da3_teacher_weight", 0.2)),
        vae_teacher_weight=float(cfg.get("vae_teacher_weight", 1.0)),
        sp_vae_loss_weight=float(cfg.get("sp_vae_loss_weight", 0.0)),
        sp_vae_cos_weight=float(cfg.get("sp_vae_cos_weight", 1.0)),
        sp_vae_mse_weight=float(cfg.get("sp_vae_mse_weight", 0.0)),
        pixel_cls_weight=float(cfg.get("pixel_cls_weight", 0.0)),
        pixel_cls_num_samples=int(cfg.get("pixel_cls_num_samples", 1024)),
        pixel_cls_temperature=float(cfg.get("pixel_cls_temperature", 0.07)),
        pixel_cls_embed_dim=int(cfg.get("pixel_cls_embed_dim", 64)),
        coarse_pull_weight=float(cfg.get("coarse_pull_weight", 0.0)),
        depth_da3_distill_weight=float(cfg.get("depth_da3_distill_weight", 0.05)),
        depth_grad_weight=float(cfg.get("depth_grad_weight", 0.0)),
        rgb_to_depth_attn_weight=float(cfg.get("rgb_to_depth_attn_weight", 0.0)),
        depth_to_rgb_attn_weight=float(cfg.get("depth_to_rgb_attn_weight", 0.0)),
        rgb_decoder_attn_sp_weight=float(cfg.get("rgb_decoder_attn_sp_weight", 0.0)),
        depth_decoder_attn_sp_weight=float(cfg.get("depth_decoder_attn_sp_weight", 0.0)),
        depth_smooth_weight=float(cfg.get("depth_smooth_weight", 0.02)),
        depth_smooth_edge_alpha=float(cfg.get("depth_smooth_edge_alpha", 10.0)),
        depth_smooth_use_da3_edge=bool(cfg.get("depth_smooth_use_da3_edge", True)),
        train_depth_patch_size=int(cfg.get("train_depth_patch_size", 2)),
        geometry_loss_weight=float(cfg.get("geometry_loss_weight", 0.0)),
        geometry_radius_weight=float(cfg.get("geometry_radius_weight", 1.0)),
        geometry_eigvals_weight=float(cfg.get("geometry_eigvals_weight", 1.0)),
        geometry_normal_weight=float(cfg.get("geometry_normal_weight", 0.5)),
        geometry_scalar_weight=float(cfg.get("geometry_scalar_weight", 0.25)),
        eval_use_all_encoder_views=bool(cfg.get("eval_use_all_encoder_views", True)),
        encoder_vae_local_window=int(cfg.get("encoder_vae_local_window", 5)),
        encoder_deform_points=int(cfg.get("encoder_deform_points", 4)),
        encoder_deform_im2col_step=int(cfg.get("encoder_deform_im2col_step", 64)),
        encoder_self_window=int(cfg.get("encoder_self_window", cfg.get("encoder_self_knn", 64))),
        encoder_self_attn_type=str(cfg.get("encoder_self_attn_type", "local")),
        encoder_self_attn_use_rope=bool(cfg.get("encoder_self_attn_use_rope", False)),
        encoder_self_attn_rope_base=float(cfg.get("encoder_self_attn_rope_base", 10000.0)),
        encoder_self_attn_rope_fwd=float(cfg.get("encoder_self_attn_rope_fwd", 1.0)),
        encoder_self_attn_rope_bits=int(cfg.get("encoder_self_attn_rope_bits", 10)),
        sp_xyz_jitter_sigma=float(cfg.get("sp_xyz_jitter_sigma", 0.01)),
        sp_xyz_jitter_prob=float(cfg.get("sp_xyz_jitter_prob", 0.8)),
        use_fps_spxyz=bool(cfg.get("use_fps_spxyz", True)),
        infer_use_fps_spxyz=bool(cfg.get("infer_use_fps_spxyz", False)),
        infer_fps_voxel_size=float(cfg.get("infer_fps_voxel_size", 0.03)),
        enable_sp_feat_supervision=bool(cfg.get("enable_sp_feat_supervision", False)),
        use_sp_field_distill=bool(cfg.get("use_sp_field_distill", False)),
        sp_field_knn=int(cfg.get("sp_field_knn", 8)),
        sp_field_sigma_fallback=float(cfg.get("sp_field_sigma_fallback", 0.05)),
        debug_nan_checks=bool(cfg.get("debug_nan_checks", False)),
        decode_sp_xyz_jitter_sigma=float(cfg.get("decode_sp_xyz_jitter_sigma", 0.0)),
        decode_sp_xyz_jitter_prob=float(cfg.get("decode_sp_xyz_jitter_prob", 0.0)),
        decode_sp_cluster_drop_prob=float(cfg.get("decode_sp_cluster_drop_prob", 0.0)),
        decode_sp_cluster_n=int(cfg.get("decode_sp_cluster_n", 0)),
        decode_sp_cluster_k=int(cfg.get("decode_sp_cluster_k", 0)),
        decode_sp_fly_prob=float(cfg.get("decode_sp_fly_prob", 0.0)),
        decode_sp_fly_count_min=int(cfg.get("decode_sp_fly_count_min", 8)),
        decode_sp_fly_count_max=int(cfg.get("decode_sp_fly_count_max", 16)),
        decode_sp_fly_offset_min=float(cfg.get("decode_sp_fly_offset_min", 0.05)),
        decode_sp_fly_offset_max=float(cfg.get("decode_sp_fly_offset_max", 0.2)),
        decode_sp_fly_spread=float(cfg.get("decode_sp_fly_spread", 0.01)),
        decode_sp_fly_feat_sigma=float(cfg.get("decode_sp_fly_feat_sigma", 0.01)),
        warp_loss_weight=float(cfg.get("warp_loss_weight", 0.0)),
        warp_tol=float(cfg.get("warp_tol", 0.02)),
        warp_use_rendered_depth=bool(cfg.get("warp_use_rendered_depth", False)),
        encoder_frozen_no_grad=bool(cfg.get("encoder_frozen_no_grad", False)),
        decoder_use_geometry_aniso=bool(cfg.get("decoder_use_geometry_aniso", False)),
        depth_inverse_param=bool(cfg.get("depth_inverse_param", False)),
        depth_min_norm=float(cfg.get("depth_min_norm", 0.02)),
        depth_max_norm=float(cfg.get("depth_max_norm", 50.0)),
        depth_pct_lo=float(cfg.get("depth_pct_lo", 1.0)),
        depth_pct_hi=float(cfg.get("depth_pct_hi", 99.0)),
        grid_token_dim=int(cfg.get("grid_token_dim", 792)),
        grid_patch=int(cfg.get("grid_patch", 14)),
        grid_image_size=grid_image_size,
        grid_k_init=int(cfg.get("grid_k_init", 8)),
        grid_k_xattn=int(cfg.get("grid_k_xattn", 64)),
        grid_n_groups=int(cfg.get("grid_n_groups", 4)),
        grid_shared_layers=int(cfg.get("grid_shared_layers", 4)),
        grid_rgb_layers=int(cfg.get("grid_rgb_layers", 8)),
        grid_depth_layers=int(cfg.get("grid_depth_layers", 3)),
        grid_depth_prep_layers=int(cfg.get("grid_depth_prep_layers", 0)),
        grid_depth_dim=int(cfg.get("grid_depth_dim", 384)),
        grid_n_heads=int(cfg.get("grid_n_heads", 12)),
        grid_depth_heads=int(cfg.get("grid_depth_heads", 8)),
        grid_mlp_ratio=float(cfg.get("grid_mlp_ratio", 4.0)),
        grid_rgb_num_views=int(cfg.get("grid_rgb_num_views", 3)),
        grid_depth_num_views=int(cfg.get("grid_depth_num_views", 3)),
        grid_alpha_grad_gate=float(cfg.get("grid_alpha_grad_gate", 0.5)),
        grid_beta_init=float(cfg.get("grid_beta_init", 3.0)),
        grid_gamma_init=float(cfg.get("grid_gamma_init", 1.0)),
        grid_depth_min_norm=float(cfg.get("grid_depth_min_norm", 0.02)),
        grid_depth_max_norm=float(cfg.get("grid_depth_max_norm", 50.0)),
        grid_depth_pct_lo=float(cfg.get("grid_depth_pct_lo", 1.0)),
        grid_depth_pct_hi=float(cfg.get("grid_depth_pct_hi", 99.0)),
        rgb_l1_weight=float(cfg.get("rgb_l1_weight", 1.0)),
        rgb_grad_weight_v27=float(cfg.get("rgb_grad_weight_v27", 0.5)),
        rgb_lpips_weight=rgb_lpips_w,
        depth_l1_weight=float(cfg.get("depth_l1_weight", 1.0)),
        depth_grad_weight_v27=float(cfg.get("depth_grad_weight_v27", 0.2)),
    )


def main(local_rank, args):
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = Config.fromfile(args.py_config)
    maybe_resolve_feature_dims(cfg)
    cfg.work_dir = args.work_dir
    model_cls = load_model_class(cfg)

    global AMP_DTYPE
    AMP_DTYPE = torch.bfloat16 if str(cfg.get("amp_dtype", "bfloat16")) == "bfloat16" else torch.float16

    if args.gpus > 1:
        distributed = True
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", cfg.get("port", 27500))
        hosts = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        gpus = torch.cuda.device_count()
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{ip}:{port}",
            world_size=hosts * gpus,
            rank=rank * gpus + local_rank,
            timeout=datetime.timedelta(minutes=int(cfg.get("dist_timeout_minutes", 60))),
        )
        world_size = dist.get_world_size()
        cfg.gpu_ids = range(world_size)
        torch.cuda.set_device(local_rank)
        if local_rank != 0:
            import builtins

            builtins.print = pass_print
    else:
        distributed = False

    if local_rank == 0:
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))
        backup_training_sources(args.work_dir, args.py_config, model_cls)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    train_log_file = osp.join(args.work_dir, f"{timestamp}_train_detail.log")
    val_log_file = osp.join(args.work_dir, f"{timestamp}_val_summary.log")
    logger = MMLogger("train_superpoint_vae2_c_train", log_file=train_log_file)
    val_logger = MMLogger("train_superpoint_vae2_c_val", log_file=val_log_file)
    MMLogger._instance_dict["train_superpoint_vae2_c_train"] = logger
    MMLogger._instance_dict["train_superpoint_vae2_c_val"] = val_logger
    logger.info(f"Config:\n{cfg.pretty_text}")
    logger.info(
        f"Resolved feature dims: encoder_img_feat_in_dim={int(cfg.get('encoder_img_feat_in_dim', -1))} "
        f"dino_feat_dim={int(cfg.get('dino_feat_dim', -1))} "
        f"da3_feat_dim={int(cfg.get('da3_feat_dim', -1))} "
        f"vae_latent_channels={int(cfg.get('vae_latent_channels', -1))}"
    )
    val_logger.info(
        f"summary_log_start timestamp={timestamp} work_dir={args.work_dir} "
        f"eval_every_epochs={int(cfg.get('eval_every_epochs', 1))} "
        f"encoder_query_init_topk={int(cfg.get('encoder_query_init_topk', cfg.get('encoder_query_view_topk', 0)))} "
        f"encoder_query_attn_topk={int(cfg.get('encoder_query_attn_topk', cfg.get('encoder_query_view_topk', 0)))} "
        f"encoder_query_attn_topk_schedule={tuple(cfg.get('encoder_query_attn_topk_schedule', ()))} "
        f"encoder_mv_residual_topk_schedule={tuple(cfg.get('encoder_mv_residual_topk_schedule', ()))} "
        f"encoder_level_logits_tau_main={float(cfg.get('encoder_level_logits_tau_main', 0.3))} "
        f"encoder_level_logits_tau_mv={float(cfg.get('encoder_level_logits_tau_mv', 1.0))} "
        f"encoder_unified_fuse_mlp_ratio={float(cfg.get('encoder_unified_fuse_mlp_ratio', 4.0))} "
        f"encoder_self_attn_type={str(cfg.get('encoder_self_attn_type', 'local'))} "
        f"encoder_self_window={int(cfg.get('encoder_self_window', cfg.get('encoder_self_knn', 64)))} "
        f"encoder_self_attn_use_rope={int(bool(cfg.get('encoder_self_attn_use_rope', False)))} "
        f"val_has_norm_depth=1"
    )

    from dataset import get_dataloader

    my_model = build_model_from_cfg(cfg)

    logger.info(f"Number of params: {sum(p.numel() for p in my_model.parameters() if p.requires_grad)}")

    # V2: freeze everything except the decoder BEFORE wrapping with DDP so
    # frozen params are excluded from both the optimizer and grad reduction.
    freeze_encoder = bool(cfg.get("freeze_encoder", False))
    if freeze_encoder:
        trainable_prefixes = tuple(cfg.get("trainable_prefixes", ("decoder",)))
        for name, param in my_model.named_parameters():
            param.requires_grad = any(name.startswith(pre) for pre in trainable_prefixes)
        n_train = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in my_model.parameters())
        logger.info(
            f"freeze_encoder=True trainable_prefixes={trainable_prefixes} "
            f"trainable_params={n_train} / {n_total} ({100.0 * n_train / max(n_total, 1):.1f}%)"
        )

    if distributed:
        if cfg.get("syncBN", False):
            my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
        my_model = torch.nn.parallel.DistributedDataParallel(
            my_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            # Aux heads are not in the render loss. With the encoder unfrozen
            # they require grad but receive none.
            find_unused_parameters=not freeze_encoder,
        )
        raw_model = my_model.module
    else:
        my_model = my_model.cuda()
        raw_model = my_model

    train_loader, _mixed_val_loader = get_dataloader(
        batch_size=int(cfg.get("train_batch_size", 2)),
        dataset_name=str(cfg.get("dataset_name", "scannet_chunkv2")),
        dataset_kwargs=cfg.get("dataset_kwargs", None),
        dist=distributed,
        iter_resume=args.iter_resume,
        collate_fn=collate_points,
        num_workers=int(cfg.get("num_workers", 4)),
        prefetch_factor=int(cfg.get("prefetch_factor", 2)),
        persistent_workers=bool(cfg.get("persistent_workers", True)),
    )
    val_loaders = _build_per_dataset_val_loaders(cfg, distributed, collate_points)
    logger.info("Per-dataset validation loaders: " + ", ".join(f"{k}={len(v.dataset)}" for k, v in val_loaders.items()))
    ffg_val_loaders = _build_ffg_cached_val_loaders(cfg, distributed, collate_points)
    if ffg_val_loaders:
        logger.info("FFG-cached validation loaders: " + ", ".join(f"{k}={len(v.dataset)}" for k, v in ffg_val_loaders.items()))

    optimizer = build_optim_wrapper(my_model, cfg.optimizer)
    max_num_epochs = int(cfg.max_epochs)
    scheduler = (
        MultiStepLRScheduler(optimizer, **cfg.multisteplr_config)
        if cfg.get("multisteplr", False)
        else CosineLRScheduler(
            optimizer,
            t_initial=max_num_epochs,
            lr_min=float(cfg.get("lr_min", 1e-6)),
            warmup_t=int(cfg.get("warmup_t", 5)),
            warmup_lr_init=1e-6,
            t_in_epochs=True,
        )
    )

    epoch = 0
    global_iter = 0
    best_loss = float("inf")
    # v27: separate best checkpoints (design §6.5) selected on the FFG cached val
    # (training distribution).  best_rgb by rgb_psnr (max), best_depth by depth_mse (min).
    best_rgb_psnr = float("-inf")
    best_depth_mse = float("inf")
    # bf16 autocast needs no GradScaler; non-finite grads are skipped explicitly.
    use_ema = bool(cfg.get("use_ema", True))
    ema_decay = float(cfg.get("ema_decay", 0.9995))
    loaded_ema_state = None
    print_freq = int(cfg.get("print_freq", 10))
    decode_rgb = bool(cfg.get("decode_rgb", True))
    debug_nan_checks = bool(cfg.get("debug_nan_checks", False))
    val_use_fps_spxyz = bool(cfg.get("val_use_fps_spxyz", True))

    cfg_resume_from = ""
    if osp.exists(osp.join(args.work_dir, "latest.pth")):
        cfg_resume_from = osp.join(args.work_dir, "latest.pth")
    if args.resume_from:
        cfg_resume_from = args.resume_from

    # V2: fresh decoder retraining initializes weights-only from a pretrained
    # checkpoint (no optimizer/scheduler/epoch). Only used when we are NOT
    # resuming an existing run.
    init_from = str(cfg.get("init_from", ""))

    if cfg_resume_from and osp.exists(cfg_resume_from):
        ckpt = torch.load(cfg_resume_from, map_location="cpu")
        loaded_ema_state = ckpt.get("ema_state_dict", None)
        compatible_state, skipped_keys = filter_compatible_state_dict(raw_model, ckpt["state_dict"])
        missing, unexpected = raw_model.load_state_dict(compatible_state, strict=False)
        epoch = int(ckpt.get("epoch", 0))
        global_iter = int(ckpt.get("global_iter", 0))
        best_loss = float(ckpt.get("best_loss", best_loss))
        best_rgb_psnr = float(ckpt.get("best_rgb_psnr", best_rgb_psnr))
        best_depth_mse = float(ckpt.get("best_depth_mse", best_depth_mse))
        if skipped_keys or missing or unexpected:
            logger.warning(
                f"Loaded compatible model weights from {cfg_resume_from}, skipped {len(skipped_keys)} incompatible keys, "
                f"missing {len(missing)} keys, unexpected {len(unexpected)} keys. "
                "Optimizer/scheduler state was not restored because the model structure changed."
            )
            for msg in skipped_keys[:20]:
                logger.warning(f"Skipped checkpoint key: {msg}")
            if len(skipped_keys) > 20:
                logger.warning(f"... and {len(skipped_keys) - 20} more skipped keys")
            for key in list(missing)[:20]:
                logger.warning(f"Missing checkpoint key: {key}")
            if len(missing) > 20:
                logger.warning(f"... and {len(missing) - 20} more missing keys")
            for key in list(unexpected)[:20]:
                logger.warning(f"Unexpected checkpoint key: {key}")
            if len(unexpected) > 20:
                logger.warning(f"... and {len(unexpected) - 20} more unexpected keys")
            val_logger.info(
                f"partial_resume checkpoint={cfg_resume_from} skipped_keys={len(skipped_keys)} "
                f"missing={len(missing)} unexpected={len(unexpected)} epoch={epoch} iter={global_iter}"
            )
        else:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
                scheduler.load_state_dict(ckpt["scheduler"])
                logger.info(f"Resumed from {cfg_resume_from} at epoch={epoch} iter={global_iter}")
                val_logger.info(f"resume checkpoint={cfg_resume_from} epoch={epoch} iter={global_iter}")
            except ValueError as exc:
                ckpt_groups = [len(g.get("params", [])) for g in ckpt.get("optimizer", {}).get("param_groups", [])]
                cur_groups = [len(g.get("params", [])) for g in optimizer.optimizer.param_groups]
                logger.warning(
                    f"Loaded model weights from {cfg_resume_from}, but optimizer/scheduler state was not restored: {exc}. "
                    f"checkpoint_param_groups={ckpt_groups} current_param_groups={cur_groups}"
                )
                val_logger.info(
                    f"partial_resume checkpoint={cfg_resume_from} optimizer_mismatch=1 "
                    f"checkpoint_param_groups={ckpt_groups} current_param_groups={cur_groups} "
                    f"epoch={epoch} iter={global_iter}"
                )

    elif init_from:
        # V2: weights-only initialization (frozen-encoder decoder retraining).
        # Optimizer/scheduler/epoch/best_loss all start fresh.
        if osp.exists(init_from):
            ckpt = torch.load(init_from, map_location="cpu")
            init_sd = ckpt.get("state_dict", ckpt)
            compatible_state, skipped_keys = filter_compatible_state_dict(raw_model, init_sd)
            missing, unexpected = raw_model.load_state_dict(compatible_state, strict=False)
            logger.info(
                f"init_from={init_from}: loaded {len(compatible_state)} compatible tensors, "
                f"skipped {len(skipped_keys)} incompatible, missing {len(missing)} (kept at init), "
                f"unexpected {len(unexpected)}. Optimizer/scheduler start fresh."
            )
            for msg in skipped_keys[:10]:
                logger.info(f"init_from skipped: {msg}")
            for key in list(missing)[:10]:
                logger.info(f"init_from missing (kept at init): {key}")
            val_logger.info(
                f"init_from checkpoint={init_from} loaded={len(compatible_state)} "
                f"skipped={len(skipped_keys)} missing={len(missing)} unexpected={len(unexpected)}"
            )
            del ckpt, init_sd, compatible_state
        else:
            logger.warning(f"init_from={init_from} does not exist; training from scratch.")

    # EMA is built after (any) resume so its shadow starts from the resumed
    # weights; if the checkpoint carried an EMA state we restore it exactly.
    ema = EMA(raw_model, ema_decay) if use_ema else None
    if ema is not None and loaded_ema_state is not None:
        ema.load_state_dict(loaded_ema_state, next(raw_model.parameters()).device)
        logger.info(f"Restored EMA weights from checkpoint (decay={ema_decay}).")

    while epoch < max_num_epochs:
        # val_metrics = evaluate(raw_model, val_loader, decode_rgb=decode_rgb, use_fps_spxyz=val_use_fps_spxyz)
        # raise Exception('ss')
        my_model.train()
        if freeze_encoder:
            # my_model.train() recursively re-enables train mode; pin the
            # frozen encoder back to eval (dropout/BN safety).
            raw_model.encoder.eval()
        if getattr(raw_model, "_online_da3_extractor", None) is not None:
            raw_model._online_da3_extractor.eval()
        # v27 decode-side jitter curriculum (design §6.2): sigma=0 early so the
        # model first anchors sharp rendering on clean positions, then ramps up.
        if hasattr(raw_model, "set_jitter_sigma"):
            _js = _jitter_sigma_for_epoch(
                epoch,
                float(cfg.get("jitter_sigma_max", 0.01)),
                int(cfg.get("jitter_curriculum_start", 2)),
                int(cfg.get("jitter_curriculum_end", 6)),
            )
            raw_model.set_jitter_sigma(_js)
            if local_rank == 0:
                _cn = int(cfg.get("decode_sp_cluster_n", 0))
                _ck = int(cfg.get("decode_sp_cluster_k", 0))
                _cp = float(cfg.get("decode_sp_cluster_drop_prob", 0.0))
                _hole = f" cluster_n={_cn} cluster_k={_ck} cluster_prob={_cp:.2f}" if _cn > 0 and _ck > 0 else ""
                _fp = float(cfg.get("decode_sp_fly_prob", 0.0))
                _fly = f" fly_prob={_fp:.2f}" if _fp > 0 else ""
                logger.info(
                    f"[JITTER] epoch={epoch} decode_sp_xyz_jitter_sigma={_js:.5f} "
                    f"prob={float(cfg.get('decode_sp_xyz_jitter_prob', 0.5)):.2f}{_hole}{_fly}"
                )
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        running = []
        skipped_steps = 0

        train_iter = iter(train_loader)
        i_iter = 0
        while True:
            try:
                current_iter = i_iter
                data = next(train_iter)
                data = batch_to_device(data)
                if not freeze_encoder:
                    # The FFG dataset still ships the frozen-encoder sidecar.
                    # Drop it so this step encodes the views and trains the encoder.
                    data.pop("cached_z_sp", None)

                optimizer.zero_grad()
                if debug_nan_checks:
                    debug_assert_model_finite(raw_model, "param")
                with autocast("cuda", dtype=AMP_DTYPE):
                    _, losses = raw_model.forward_with_loss(data, decode_rgb=decode_rgb)
                    loss = losses["loss"]
                loss.backward()
                if debug_nan_checks:
                    debug_assert_model_finite(raw_model, "grad")
                # v27: per-parameter-group clipping (design §6.3) so a collapsing
                # branch cannot steal the others' effective LR.
                grad_norm, group_grad_norms = _clip_grad_groups(my_model, max_norm=float(cfg.get("grad_max_norm", 1.0)))
                # bf16 rarely overflows, but guard against a non-finite step and
                # keep the EMA/params clean when it happens.
                if torch.isfinite(grad_norm):
                    optimizer.step()
                    if ema is not None:
                        ema.update(raw_model, num_updates=global_iter)
                else:
                    skipped_steps += 1
                if debug_nan_checks:
                    debug_assert_model_finite(raw_model, "param")
                running.append(loss.detach().item())
                global_iter += 1
                i_iter += 1
            except StopIteration:
                break
            except Exception:
                logger.exception(f"[TRAIN] rank={local_rank} failed at epoch={epoch} iter={i_iter}")
                if distributed and dist.is_initialized():
                    dist.destroy_process_group()
                raise

            if current_iter % print_freq == 0 and local_rank == 0:
                # per-group pre-clip gradient norms (design §6.4 branch-contention signal)
                ggn = "".join(
                    f" gn_{name}={float(val):.4f}" for name, val in group_grad_norms.items()
                )
                logger.info(
                    f"[TRAIN] epoch={epoch} iter={current_iter} global_iter={global_iter} "
                    + _format_metrics(
                        losses,
                        [
                            "loss",
                            "rgb_loss",
                            "rgb_l1",
                            "rgb_grad",
                            "rgb_lpips",
                            "depth_loss",
                            "depth_l1",
                            "depth_grad",
                            "frontend_w_null_mean",
                            "frontend_w_entropy",
                            "frontend_k_valid_mean",
                            "frontend_beta_z",
                            "frontend_beta_r",
                            "frontend_gamma",
                            "frontend_wp_over_wv",
                        ],
                        aliases={
                            "rgb_grad": "rgb_grad",
                            "depth_grad": "dep_grad",
                            "frontend_w_null_mean": "w_null",
                            "frontend_w_entropy": "w_ent",
                            "frontend_k_valid_mean": "k_val",
                            "frontend_beta_z": "b_z",
                            "frontend_beta_r": "b_r",
                            "frontend_gamma": "gamma",
                            "frontend_wp_over_wv": "wp_wv",
                        },
                    )
                    + f" grad_norm={float(grad_norm):.4f}{ggn} skips={skipped_steps}"
                )

        epoch += 1
        scheduler.step(epoch)
        if local_rank == 0 and running:
            logger.info(f"[TRAIN] Epoch {epoch} mean loss: {np.mean(running):.4f}")

        if epoch % int(cfg.get("eval_every_epochs", 1)) != 0:
            continue

        # Evaluate the EMA weights (GLD-style): swap them in for validation, then
        # restore the live training weights.  Runs on every rank so metric reduce
        # stays consistent.
        if ema is not None:
            ema.store(raw_model)
            ema.copy_to(raw_model)
        val_results = {}
        for val_name, val_loader in val_loaders.items():
            val_sums = evaluate(raw_model, val_loader, decode_rgb=decode_rgb, use_fps_spxyz=val_use_fps_spxyz)
            if distributed:
                val_sums = _reduce_metric_sums(val_sums)
            val_results[val_name] = (val_sums, _finalize_eval_metrics(val_sums))
        # FFG val uses the same chunks as training. With the encoder unfrozen it
        # re-encodes; the cached path is only for a frozen-encoder run.
        ffg_val_results = {}
        for val_name, val_loader in ffg_val_loaders.items():
            val_sums = evaluate(
                raw_model,
                val_loader,
                decode_rgb=decode_rgb,
                use_fps_spxyz=val_use_fps_spxyz,
                cached=freeze_encoder,
            )
            if distributed:
                val_sums = _reduce_metric_sums(val_sums)
            ffg_val_results[val_name] = (val_sums, _finalize_eval_metrics(val_sums))
        if ema is not None:
            ema.restore(raw_model)

        metric_keys = ["rgb_sse", "rgb_count", "rgb_ssim_sum", "rgb_ssim_count", "depth_sse", "depth_count", "depth_valid_sum", "depth_valid_count"]

        def _mean_metrics(results):
            mean_sums = {
                key: sum((sums[key] for sums, _ in results.values()), next(iter(results.values()))[0][key].new_zeros(()))
                for key in metric_keys
            }
            return mean_sums, _finalize_eval_metrics(mean_sums)

        # C15 online val -> secondary metric (comparable to the v20 run).
        c15_primary_sums, c15_primary_metrics = _mean_metrics(val_results)
        # FFG cached val (training distribution) -> PRIMARY for checkpoint selection
        # (design §6.6).  Falls back to C15 if FFG val is disabled.
        if ffg_val_results:
            ffg_primary_sums, select_metrics = _mean_metrics(ffg_val_results)
            select_mode = "ffg_cached"
        else:
            ffg_primary_sums, select_metrics = c15_primary_sums, c15_primary_metrics
            select_mode = "c15_online"

        if local_rank == 0:
            ema_state = ema.state_dict() if ema is not None else None
            ckpt = {
                # latest.pth keeps the live training weights (for exact resume);
                # the EMA weights ride along under ema_state_dict.
                "state_dict": raw_model.state_dict(),
                "ema_state_dict": ema_state,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "global_iter": global_iter,
                "best_loss": best_loss,
                "best_rgb_psnr": best_rgb_psnr,
                "best_depth_mse": best_depth_mse,
            }
            latest_path = osp.join(args.work_dir, "latest.pth")
            torch.save(ckpt, latest_path)
            symlink(latest_path, osp.join(args.work_dir, "latest_link.pth"))

            def _ema_ckpt():
                c = dict(ckpt)
                if ema_state is not None:
                    c["state_dict"] = ema_state
                return c

            # Separate best checkpoints (design §6.5): RGB by psnr, depth by mse.
            cur_rgb_psnr = float(select_metrics["rgb_psnr"].item())
            cur_depth_mse = float(select_metrics["depth_mse"].item())
            is_best_rgb = cur_rgb_psnr > best_rgb_psnr
            if is_best_rgb:
                best_rgb_psnr = cur_rgb_psnr
                torch.save(_ema_ckpt(), osp.join(args.work_dir, "best_rgb.pth"))
            is_best_depth = cur_depth_mse < best_depth_mse
            if is_best_depth:
                best_depth_mse = cur_depth_mse
                torch.save(_ema_ckpt(), osp.join(args.work_dir, "best_depth.pth"))

            logger.info(
                f"[VAL] epoch={epoch} global_iter={global_iter} select={select_mode} "
                f"best_rgb_psnr={best_rgb_psnr:.4f} best_depth_mse={best_depth_mse:.6f} "
                f"is_best_rgb={int(is_best_rgb)} is_best_depth={int(is_best_depth)} "
                + _format_metrics(select_metrics, ["loss", "rgb_mse", "rgb_psnr", "rgb_ssim", "depth_mse", "depth_valid_ratio"])
            )
            # C15 online per-dataset (secondary).
            for val_name, (val_sums, val_metrics) in val_results.items():
                val_logger.info(
                    f"epoch={epoch} global_iter={global_iter} dataset={val_name} mode=c15_online status=latest "
                    + _format_metrics(
                        val_metrics,
                        ["loss", "rgb_mse", "rgb_psnr", "rgb_ssim", "depth_mse", "depth_valid_ratio"],
                    )
                    + f" depth_sse={float(val_sums['depth_sse'].item()):.8e}"
                    + f" depth_count={float(val_sums['depth_count'].item()):.0f}"
                )
            # FFG cached-latent per-dataset (primary, training distribution).
            for val_name, (val_sums, val_metrics) in ffg_val_results.items():
                val_logger.info(
                    f"epoch={epoch} global_iter={global_iter} dataset={val_name} mode=ffg_cached "
                    f"status={'best_rgb' if is_best_rgb else ''}{'/best_depth' if is_best_depth else ''} "
                    f"best_rgb_psnr={best_rgb_psnr:.4f} best_depth_mse={best_depth_mse:.6f} "
                    + _format_metrics(
                        val_metrics,
                        ["loss", "rgb_mse", "rgb_psnr", "rgb_ssim", "depth_mse", "depth_valid_ratio"],
                    )
                    + f" depth_sse={float(val_sums['depth_sse'].item()):.8e}"
                    + f" depth_count={float(val_sums['depth_count'].item()):.0f}"
                )
            # summary lines (mean over datasets) for both modes
            val_logger.info(
                f"epoch={epoch} global_iter={global_iter} dataset=mean mode=c15_online status=latest "
                + _format_metrics(c15_primary_metrics, ["loss", "rgb_mse", "rgb_psnr", "rgb_ssim", "depth_mse", "depth_valid_ratio"])
            )
            if ffg_val_results:
                val_logger.info(
                    f"epoch={epoch} global_iter={global_iter} dataset=mean mode=ffg_cached "
                    f"status={'best_rgb' if is_best_rgb else ''}{'/best_depth' if is_best_depth else ''} "
                    + _format_metrics(select_metrics, ["loss", "rgb_mse", "rgb_psnr", "rgb_ssim", "depth_mse", "depth_valid_ratio"])
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the superpoint encoder and grid decoder with multi-dataset validation")
    parser.add_argument("--py-config", default="config.py")
    parser.add_argument("--work-dir", type=str, required=True)
    parser.add_argument("--resume-from", type=str, default="")
    parser.add_argument("--iter-resume", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()

    ngpus = torch.cuda.device_count()
    args.gpus=ngpus
    if args.gpus > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    else:
        main(0, args)
