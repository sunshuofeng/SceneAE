#!/usr/bin/env python3
"""
Cache first/rest z_sp latents for first-frame generation chunks.

The script runs a frozen SuperpointVAE2C encoder twice per chunk:
  - first_* superpoints with frame 0 only -> first_superpoint_z_sp
  - rest_* superpoints with all frames -> rest_superpoint_z_sp

By default it writes sidecar files next to each chunk:
  chunk_xxxxxx_*.zsp.npz
If cache_root is set, sidecars are written under that directory while
preserving the chunk path relative to chunk_root.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import multiprocessing as mp
import shutil
import sys
import types
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


THIS_FILE = Path(__file__).resolve()
VAE_ROOT = THIS_FILE.parent
PROJECT_ROOT = VAE_ROOT.parent

for p in (
    VAE_ROOT,
    PROJECT_ROOT / "third_party" / "dinov3",
    PROJECT_ROOT / "third_party" / "Depth-Anything-3" / "da3",
):
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)


def install_optional_dependency_stubs() -> None:
    moviepy_stub = types.ModuleType("moviepy")
    moviepy_editor_stub = types.ModuleType("moviepy.editor")
    moviepy_stub.editor = moviepy_editor_stub
    sys.modules.setdefault("moviepy", moviepy_stub)
    sys.modules.setdefault("moviepy.editor", moviepy_editor_stub)


install_optional_dependency_stubs()


def load_py_config(path: Path) -> dict[str, Any]:
    namespace: dict[str, Any] = {"__file__": str(path)}
    code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
    exec(code, namespace)
    return {k: v for k, v in namespace.items() if not k.startswith("__")}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_json_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def instantiate_model(model_cls: type, **kwargs):
    sig = inspect.signature(model_cls.__init__)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return model_cls(**kwargs)
    allowed = {
        key
        for key, param in params.items()
        if key != "self"
        and param.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return model_cls(**{key: value for key, value in kwargs.items() if key in allowed})


def filter_compatible_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], list[str]]:
    current = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in state_dict.items():
        clean_key = key[7:] if key.startswith("module.") else key
        if clean_key not in current:
            skipped.append(f"{key}: missing in current model")
            continue
        if current[clean_key].shape != value.shape:
            skipped.append(f"{key}: checkpoint {tuple(value.shape)} != current {tuple(current[clean_key].shape)}")
            continue
        filtered[clean_key] = value
    return filtered, skipped


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_model_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    dataset_kwargs = dict(cfg.get("dataset_kwargs", {}))
    kwargs = dict(
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
        encoder_view_capacity=int(cfg.get("encoder_view_capacity", dataset_kwargs.get("num_views", cfg.get("encoder_max_views", 4)))),
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
        lpips_loss_weight=0.0,
        latent_reg_weight=float(cfg.get("latent_reg_weight", 0.01)),
        sp_feat_loss_weight=float(cfg.get("sp_feat_loss_weight", 0.0)),
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
        sp_xyz_jitter_sigma=0.0,
        sp_xyz_jitter_prob=0.0,
        use_fps_spxyz=bool(cfg.get("use_fps_spxyz", True)),
        infer_use_fps_spxyz=False,
        infer_fps_voxel_size=0.0,
        enable_sp_feat_supervision=bool(cfg.get("enable_sp_feat_supervision", False)),
        use_sp_field_distill=bool(cfg.get("use_sp_field_distill", False)),
        sp_field_knn=int(cfg.get("sp_field_knn", 8)),
        sp_field_sigma_fallback=float(cfg.get("sp_field_sigma_fallback", 0.05)),
        debug_nan_checks=bool(cfg.get("debug_nan_checks", False)),
    )
    # Grid-decoder versions (v27+) read architecture from grid_* in the py
    # config.  Class defaults match v27 (shared=4 / rgb=8); v30 is shared=0 /
    # rgb=24 / depth-prep=8 and will silently mismatch the checkpoint if these
    # are dropped.
    for key, value in cfg.items():
        if key.startswith("grid_"):
            kwargs[key] = value
    return kwargs


def load_model(vae_cfg: dict[str, Any], device: torch.device) -> nn.Module:
    cfg_path = Path(str(vae_cfg["config"]))
    cfg = load_py_config(cfg_path)
    if vae_cfg.get("model_module"):
        cfg["model_module"] = vae_cfg["model_module"]
    if vae_cfg.get("model_class"):
        cfg["model_class"] = vae_cfg["model_class"]

    module_name = str(cfg.get("model_module", "models.scene_vae"))
    class_name = str(cfg.get("model_class", "SuperpointVAEGridV30"))
    module_candidates = [module_name]
    last_exc: Exception | None = None
    for candidate in module_candidates:
        try:
            model_cls = getattr(importlib.import_module(candidate), class_name)
            break
        except ModuleNotFoundError as exc:
            last_exc = exc
            top_name = candidate.split(".", 1)[0]
            if exc.name not in (candidate, top_name, module_name):
                raise
        except Exception as exc:
            last_exc = exc
            raise
    else:
        raise ModuleNotFoundError(f"Could not import any of {module_candidates}") from last_exc
    model = instantiate_model(model_cls, **build_model_kwargs(cfg))

    ckpt_path = Path(str(vae_cfg["checkpoint"]))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    compatible, skipped = filter_compatible_state_dict(model, state)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(
        f"[model] loaded {ckpt_path} compatible={len(compatible)} "
        f"skipped={len(skipped)} missing={len(missing)} unexpected={len(unexpected)}"
    )
    if skipped:
        print(f"[model] first skipped key: {skipped[0]}")
    model.to(device).eval()
    model.requires_grad_(False)
    if hasattr(model, "infer_use_fps_spxyz"):
        model.infer_use_fps_spxyz = False
    dataset_kwargs = cfg.get("dataset_kwargs", {})
    if isinstance(dataset_kwargs, dict) and "num_fps_sp_xyz" in dataset_kwargs:
        model.cache_num_fps_sp_xyz = int(dataset_kwargs["num_fps_sp_xyz"])
    return model


def apply_cache_fps_count(model: nn.Module, data_cfg: dict[str, Any]) -> None:
    """v30 train dataset_kwargs omits num_fps_sp_xyz; the cache JSON can supply it."""
    if "num_fps_sp_xyz" not in data_cfg:
        return
    model.cache_num_fps_sp_xyz = int(data_cfg["num_fps_sp_xyz"])


def iter_chunk_npz(root: Path) -> list[Path]:
    if root.is_file() and root.suffix == ".npz":
        return [root]
    return sorted(p for p in root.rglob("*.npz") if p.name.endswith(".npz") and not p.name.endswith(".zsp.npz"))


def sidecar_path(chunk_path: Path, chunk_root: Path | None = None, cache_root: Path | None = None) -> Path:
    if cache_root is None:
        return chunk_path.with_suffix(".zsp.npz")
    if chunk_root is not None:
        base_root = chunk_root.parent if chunk_root.suffix == ".npz" else chunk_root
        try:
            rel = chunk_path.relative_to(base_root)
        except ValueError:
            rel = Path(chunk_path.name)
    else:
        rel = Path(chunk_path.name)
    return cache_root / rel.with_suffix(".zsp.npz")


class RunningStats:
    def __init__(self, dim: int) -> None:
        self.dim = int(dim)
        self.count = 0
        self.mean = np.zeros((dim,), dtype=np.float64)
        self.m2 = np.zeros((dim,), dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[0] == 0:
            return
        batch_count = x.shape[0]
        batch_mean = x.mean(axis=0)
        batch_m2 = ((x - batch_mean) ** 2).sum(axis=0)
        if self.count == 0:
            self.count = int(batch_count)
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        self.m2 = self.m2 + batch_m2 + delta**2 * self.count * batch_count / total
        self.count = int(total)

    def merge(self, other_count: int, other_mean: np.ndarray, other_m2: np.ndarray) -> None:
        other_count = int(other_count)
        if other_count <= 0:
            return
        other_mean = np.asarray(other_mean, dtype=np.float64)
        other_m2 = np.asarray(other_m2, dtype=np.float64)
        if self.count == 0:
            self.count = other_count
            self.mean = other_mean.copy()
            self.m2 = other_m2.copy()
            return
        delta = other_mean - self.mean
        total = self.count + other_count
        self.mean = self.mean + delta * other_count / total
        self.m2 = self.m2 + other_m2 + delta**2 * self.count * other_count / total
        self.count = int(total)

    def state_dict(self) -> dict[str, Any]:
        return {
            "count": int(self.count),
            "mean": self.mean.astype(np.float64),
            "m2": self.m2.astype(np.float64),
        }

    @property
    def var(self) -> np.ndarray:
        if self.count <= 1:
            return np.ones((self.dim,), dtype=np.float64)
        return np.maximum(self.m2 / max(self.count - 1, 1), 1e-12)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)


def select_scene_split(
    chunks: list[Path],
    chunk_root: Path,
    scene_split: str,
    split_ratio: float,
) -> list[Path]:
    """Select a deterministic scene-level split without leaking scenes."""
    if scene_split == "all" or chunk_root.is_file():
        return chunks
    if not 0.0 < split_ratio < 1.0:
        raise ValueError(f"split_ratio must be in (0, 1), got {split_ratio}")

    scene_to_chunks: dict[str, list[Path]] = {}
    for chunk in chunks:
        try:
            relative = chunk.relative_to(chunk_root)
        except ValueError as exc:
            raise ValueError(f"Chunk {chunk} is outside chunk root {chunk_root}") from exc
        if len(relative.parts) < 2:
            raise ValueError(
                f"Scene split requires chunks below scene directories, got {relative}"
            )
        scene_to_chunks.setdefault(relative.parts[0], []).append(chunk)

    scenes = sorted(scene_to_chunks)
    split_index = int(len(scenes) * split_ratio)
    selected_scenes = scenes[:split_index] if scene_split == "train" else scenes[split_index:]
    selected = []
    for scene in selected_scenes:
        selected.extend(scene_to_chunks[scene])
    print(
        f"[split] mode={scene_split} ratio={split_ratio:g} "
        f"scenes={len(selected_scenes)}/{len(scenes)} chunks={len(selected)}"
    )
    return selected


def validate_chunk_contract(
    chunks: list[Path],
    required_schema: str | None,
    required_frames: int | None,
) -> None:
    if required_schema is None and required_frames is None:
        return
    for chunk_path in chunks:
        with np.load(chunk_path, allow_pickle=True) as chunk:
            if required_schema is not None:
                actual_schema = (
                    str(chunk["chunk_schema_version"].item())
                    if "chunk_schema_version" in chunk.files
                    else "missing"
                )
                if actual_schema != required_schema:
                    raise ValueError(
                        f"Stale chunk schema in {chunk_path}: expected "
                        f"{required_schema!r}, got {actual_schema!r}"
                    )
            if required_frames is not None:
                if "frame_indices" not in chunk.files:
                    raise KeyError(f"{chunk_path} missing frame_indices")
                actual_frames = len(chunk["frame_indices"])
                if actual_frames != required_frames:
                    raise ValueError(
                        f"Wrong frame count in {chunk_path}: expected "
                        f"{required_frames}, got {actual_frames}"
                    )
    print(
        f"[contract] validated chunks={len(chunks)} "
        f"schema={required_schema or 'any'} frames={required_frames or 'any'}"
    )


def chunk_source_split(z: np.lib.npyio.NpzFile) -> str:
    if "image_paths" not in z.files:
        return "unknown"
    paths = z["image_paths"].tolist()
    paths = paths if isinstance(paths, list) else [paths]
    detected: set[str] = set()
    for value in paths:
        parts = {part.lower() for part in str(value).replace("\\", "/").split("/")}
        if "train" in parts or "training" in parts:
            detected.add("train")
        if parts.intersection({"test", "testing", "val", "validation", "eval"}):
            detected.add("val")
    if len(detected) > 1:
        raise ValueError(f"Chunk mixes train and validation image paths: {paths[:2]}")
    return next(iter(detected), "unknown")


def include_chunk_in_stats(z: np.lib.npyio.NpzFile, data_cfg: dict[str, Any]) -> bool:
    requested = str(data_cfg.get("stats_data_split", "all")).lower()
    if requested == "all":
        return True
    actual = chunk_source_split(z)
    if actual == "unknown":
        raise ValueError(
            f"Cannot infer chunk source split from image_paths for stats_data_split={requested}"
        )
    return actual == requested


@dataclass
class ImageCache:
    re10k_files: OrderedDict[Path, Any]
    max_files: int

    def __init__(self, max_files: int = 1) -> None:
        self.re10k_files = OrderedDict()
        self.max_files = max(1, int(max_files))

    def read(self, stored_path: str) -> np.ndarray:
        s = str(stored_path)
        if "::" in s:
            return self._read_re10k(s)
        img = cv2.imread(s, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {s}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _read_re10k(self, spec: str) -> np.ndarray:
        source, scene_key, frame_name = spec.split("::", 2)
        source_path = Path(source)
        if source_path not in self.re10k_files:
            try:
                self.re10k_files[source_path] = torch.load(source_path, map_location="cpu", weights_only=True)
            except TypeError:
                self.re10k_files[source_path] = torch.load(source_path, map_location="cpu")
            while len(self.re10k_files) > self.max_files:
                self.re10k_files.popitem(last=False)
        else:
            self.re10k_files.move_to_end(source_path)
        frame_idx = int(frame_name.rsplit("_", 1)[-1])
        for scene in self.re10k_files[source_path]:
            if str(scene.get("key", source_path.stem)) != scene_key:
                continue
            image_bytes = scene["images"][frame_idx]
            pil = Image.open(BytesIO(image_bytes.numpy().tobytes())).convert("RGB")
            return np.asarray(pil)
        raise KeyError(f"RE10K scene {scene_key} not found in {source_path}")


def load_chunk_inputs(z: np.lib.npyio.NpzFile, data_cfg: dict[str, Any], image_cache: ImageCache) -> dict[str, torch.Tensor]:
    da3_depths = z["da3_depths"].astype(np.float32)
    camera_poses = z["camera_poses_c2w"].astype(np.float32)
    camera_intrinsics = z["camera_intrinsics"].astype(np.float32)
    paths = z["image_paths"].tolist() if hasattr(z["image_paths"], "tolist") else list(z["image_paths"])

    norm_factor = float(z["norm_scale"]) if "norm_scale" in z.files else 1.0
    if (not np.isfinite(norm_factor)) or norm_factor < 1e-6:
        norm_factor = 1.0

    resize_w = data_cfg.get("resize_w", None)
    resize_h = data_cfg.get("resize_h", None)
    depth_near = float(data_cfg.get("depth_near", 1e-3))
    depth_far = data_cfg.get("depth_far", None)
    norm_depth_clip = float(data_cfg.get("norm_depth_clip", 100.0))

    images: list[torch.Tensor] = []
    Ks: list[torch.Tensor] = []
    camtoworlds: list[torch.Tensor] = []
    valid_masks: list[torch.Tensor] = []
    norm_depths: list[torch.Tensor] = []

    for view_idx, path in enumerate(paths):
        img = image_cache.read(str(path))
        depth = da3_depths[view_idx].astype(np.float32)
        h_da3, w_da3 = depth.shape
        target_w = int(resize_w) if resize_w is not None else w_da3
        target_h = int(resize_h) if resize_h is not None else h_da3

        if img.shape[:2] != (target_h, target_w):
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        if depth.shape != (target_h, target_w):
            depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        k0 = camera_intrinsics[view_idx]
        sw = target_w / float(w_da3)
        sh = target_h / float(h_da3)
        k = np.array(
            [
                [float(k0[0, 0]) * sw, 0.0, float(k0[0, 2]) * sw],
                [0.0, float(k0[1, 1]) * sh, float(k0[1, 2]) * sh],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        valid = np.isfinite(depth) & (depth > depth_near)
        if depth_far is not None:
            valid &= depth < float(depth_far)
        pose = camera_poses[view_idx].copy()
        pose[:3, 3] = pose[:3, 3] / norm_factor

        images.append(torch.from_numpy((img.astype(np.float32) / 255.0)))
        Ks.append(torch.from_numpy(k))
        camtoworlds.append(torch.from_numpy(pose.astype(np.float32)))
        valid_masks.append(torch.from_numpy(valid.astype(bool)))
        norm_depths.append(torch.from_numpy((np.clip(depth, 0.0, norm_depth_clip) / norm_factor).astype(np.float32)))

    return {
        "image": torch.stack(images, dim=0),
        "Ks": torch.stack(Ks, dim=0),
        "camtoworlds": torch.stack(camtoworlds, dim=0),
        "valid_mask": torch.stack(valid_masks, dim=0),
        "norm_depth": torch.stack(norm_depths, dim=0),
        "norm_scale": torch.tensor(norm_factor, dtype=torch.float32),
    }


def load_group(z: np.lib.npyio.NpzFile, prefix: str, norm_factor: float) -> tuple[torch.Tensor, torch.Tensor]:
    xyz = z[f"{prefix}superpoint_xyz"].astype(np.float32) / float(norm_factor)
    valid_key = f"{prefix}superpoint_valid_mask"
    if valid_key in z.files:
        valid = z[valid_key].astype(bool)
    else:
        valid = np.ones((xyz.shape[0],), dtype=bool)
    return torch.from_numpy(xyz), torch.from_numpy(valid)


def _encoder_helper(model: nn.Module, name: str):
    """Resolve an encoder helper from the model class or an ancestor module.

    v30 lives in its own module and inherits the C20 encoder. The pool helpers
    stay on the ancestor module, not on ``superpoint_da3_groups_v30``.
    """
    for cls in type(model).mro():
        module = sys.modules.get(cls.__module__)
        if module is not None and hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"{type(model).__name__} has no encoder helper {name}")


def encode_z_sp_only(
    model: nn.Module,
    *,
    image: torch.Tensor,
    Ks: torch.Tensor,
    camtoworlds: torch.Tensor,
    sp_xyz: torch.Tensor,
    norm_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    sp_valid_mask: torch.Tensor,
    norm_pw: torch.Tensor | None = None,
    norm_pw_offsets: torch.Tensor | None = None,
    use_fps_queries: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run only the SuperpointVAE encoder path needed for cached z_sp."""
    if image.dim() != 5:
        raise ValueError(f"image expected [B,V,H,W,3], got {tuple(image.shape)}")
    projective_pool_grid = _encoder_helper(model, "_projective_pool_grid")
    projective_pool_sample = _encoder_helper(model, "_projective_pool_sample")
    select_query_topk_views_by_depth = _encoder_helper(model, "_select_query_topk_views_by_depth")

    b_img, v_img, h_img, w_img, _ = image.shape
    image_encoder = image
    Ks_encoder = Ks
    camtoworlds_encoder = camtoworlds
    norm_depth_encoder = norm_depth
    valid_mask_encoder = valid_mask
    max_enc_views = max(1, min(int(model.encoder_max_views), int(v_img)))
    use_view_subset = model.training or (not bool(model.eval_use_all_encoder_views))
    if use_view_subset and v_img > max_enc_views:
        if model.training:
            encoder_view_idx = torch.rand(b_img, v_img, device=image.device).argsort(dim=-1)[:, :max_enc_views]
            encoder_view_idx, _ = encoder_view_idx.sort(dim=-1)
        else:
            encoder_view_idx = torch.arange(max_enc_views, device=image.device).unsqueeze(0).expand(b_img, max_enc_views).contiguous()
        image_encoder = torch.gather(
            image,
            1,
            encoder_view_idx[:, :, None, None, None].expand(b_img, max_enc_views, h_img, w_img, image.shape[-1]),
        )
        Ks_encoder = torch.gather(
            Ks,
            1,
            encoder_view_idx[:, :, None, None].expand(b_img, max_enc_views, Ks.shape[-2], Ks.shape[-1]),
        )
        camtoworlds_encoder = torch.gather(
            camtoworlds,
            1,
            encoder_view_idx[:, :, None, None].expand(b_img, max_enc_views, camtoworlds.shape[-2], camtoworlds.shape[-1]),
        )
        valid_mask_encoder = torch.gather(
            valid_mask,
            1,
            encoder_view_idx[:, :, None, None].expand(b_img, max_enc_views, valid_mask.shape[-2], valid_mask.shape[-1]),
        )
        norm_depth_encoder = torch.gather(
            norm_depth,
            1,
            encoder_view_idx[:, :, None, None].expand(b_img, max_enc_views, norm_depth.shape[-2], norm_depth.shape[-1]),
        )

    sp_xyz = torch.nan_to_num(sp_xyz, nan=0.0, posinf=0.0, neginf=0.0)
    if use_fps_queries and norm_pw is not None:
        sp_xyz = model._infer_sp_xyz_from_point_cloud(norm_pw, norm_pw_offsets, sp_xyz.shape[1], sp_xyz)
    if hasattr(model, "_maybe_jitter_sp_xyz"):
        sp_xyz = model._maybe_jitter_sp_xyz(sp_xyz)
    sp_valid_mask = sp_valid_mask.to(device=sp_xyz.device).bool()

    da3_levels = model._resolve_da3_levels(
        None,
        None,
        image=image_encoder,
        Ks=Ks_encoder,
        camtoworlds=camtoworlds_encoder,
    )
    img_feat_list, plucker, _view_ids = model._build_encoder_tokens(
        da3_levels, Ks_encoder, camtoworlds_encoder, (h_img, w_img)
    )
    feat_hw = (da3_levels[0].shape[2], da3_levels[0].shape[3])
    pool_grid, pool_valid, vis_ratio = projective_pool_grid(
        sp_xyz, Ks_encoder, camtoworlds_encoder, (h_img, w_img), feat_hw
    )
    pool_valid = pool_valid & sp_valid_mask[:, None, :]
    pool_valid_for_init = select_query_topk_views_by_depth(
        sp_xyz,
        Ks_encoder,
        camtoworlds_encoder,
        norm_depth_encoder,
        pool_valid,
        (h_img, w_img),
        model.encoder_query_init_topk,
        valid_mask=valid_mask_encoder,
    )
    attn_schedule = model.encoder_query_attn_topk_schedule
    if not attn_schedule:
        attn_schedule = (model.encoder_query_attn_topk,)
    pool_valid_layers = [
        select_query_topk_views_by_depth(
            sp_xyz,
            Ks_encoder,
            camtoworlds_encoder,
            norm_depth_encoder,
            pool_valid,
            (h_img, w_img),
            int(attn_schedule[min(layer_idx, len(attn_schedule) - 1)]),
            valid_mask=valid_mask_encoder,
        )
        for layer_idx in range(len(model.encoder.blocks))
    ]
    mv_schedule = model.encoder_mv_residual_topk_schedule
    mv_pool_valid_layers = None
    if mv_schedule:
        mv_pool_valid_layers = []
        for layer_idx in range(len(model.encoder.blocks)):
            topk = int(mv_schedule[min(layer_idx, len(mv_schedule) - 1)])
            if topk <= 0:
                mv_pool_valid_layers.append(torch.zeros_like(pool_valid))
            else:
                mv_pool_valid_layers.append(
                    select_query_topk_views_by_depth(
                        sp_xyz,
                        Ks_encoder,
                        camtoworlds_encoder,
                        norm_depth_encoder,
                        pool_valid,
                        (h_img, w_img),
                        topk,
                        valid_mask=valid_mask_encoder,
                    )
                )
    pooled_feats = [projective_pool_sample(lvl, pool_grid, pool_valid_for_init) for lvl in da3_levels]
    z_sp, _sp_xyz_refined, _anchor_aux = model.encoder(
        da3_levels=img_feat_list,
        pooled_feats=pooled_feats,
        vis_ratio=vis_ratio,
        sp_xyz=sp_xyz,
        plucker=plucker,
        da3_hw=feat_hw,
        pool_grid=pool_grid,
        pool_valid_layers=pool_valid_layers,
        mv_pool_valid_layers=mv_pool_valid_layers,
        sp_valid_mask=sp_valid_mask,
    )
    return z_sp, sp_xyz


@torch.no_grad()
def encode_group(
    model: nn.Module,
    common: dict[str, torch.Tensor],
    sp_xyz: torch.Tensor,
    sp_valid: torch.Tensor,
    view_indices: list[int],
    device: torch.device,
    amp: bool,
    query_mode: str = "raw",
    num_queries: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = torch.tensor(view_indices, dtype=torch.long)
    image = common["image"].index_select(0, index).unsqueeze(0).to(device, non_blocking=True)
    Ks = common["Ks"].index_select(0, index).unsqueeze(0).to(device, non_blocking=True)
    camtoworlds = common["camtoworlds"].index_select(0, index).unsqueeze(0).to(device, non_blocking=True)
    valid_mask = common["valid_mask"].index_select(0, index).unsqueeze(0).to(device, non_blocking=True)
    norm_depth = common["norm_depth"].index_select(0, index).unsqueeze(0).to(device, non_blocking=True)
    query_mode = str(query_mode)
    use_fps_queries = query_mode == "fps"
    if query_mode not in ("raw", "fps"):
        raise ValueError(f"Unsupported query_mode={query_mode}")
    source_xyz = sp_xyz[sp_valid.bool()]
    if use_fps_queries:
        if source_xyz.shape[0] == 0:
            raise ValueError("Cannot build FPS queries from an empty superpoint set.")
        if num_queries is None:
            num_queries = int(getattr(model, "cache_num_fps_sp_xyz", source_xyz.shape[0]))
        sp_xyz_in = torch.zeros((int(num_queries), 3), dtype=sp_xyz.dtype)
        sp_valid_in = torch.ones((int(num_queries),), dtype=torch.bool)
    else:
        sp_xyz_in = sp_xyz
        sp_valid_in = sp_valid
    sp_xyz_b = sp_xyz_in.unsqueeze(0).to(device, non_blocking=True)
    sp_valid_b = sp_valid_in.unsqueeze(0).to(device, non_blocking=True)
    norm_pw = source_xyz.to(device, non_blocking=True) if use_fps_queries else None
    norm_pw_offsets = (
        torch.tensor([int(source_xyz.shape[0])], dtype=torch.long, device=device)
        if use_fps_queries
        else None
    )

    use_amp = amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        z_sp_t, query_xyz_t = encode_z_sp_only(
            model,
            image=image,
            Ks=Ks,
            camtoworlds=camtoworlds,
            sp_xyz=sp_xyz_b,
            norm_depth=norm_depth,
            valid_mask=valid_mask,
            sp_valid_mask=sp_valid_b,
            norm_pw=norm_pw,
            norm_pw_offsets=norm_pw_offsets,
            use_fps_queries=use_fps_queries,
        )
    z_sp = z_sp_t[0].detach().float().cpu().numpy().astype(np.float32)
    query_xyz = query_xyz_t[0].detach().float().cpu().numpy().astype(np.float32)
    query_valid = sp_valid_b[0].detach().cpu().numpy().astype(bool)
    if not np.isfinite(z_sp).all():
        raise FloatingPointError("non-finite z_sp produced")
    if not np.isfinite(query_xyz).all():
        raise FloatingPointError("non-finite query xyz produced")
    return z_sp, query_xyz, query_valid


def atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.savez_compressed(tmp, **arrays)
    actual = tmp if tmp.is_file() else Path(str(tmp) + ".npz")
    actual.replace(path)


def write_inplace(chunk_path: Path, z: np.lib.npyio.NpzFile, extra: dict[str, Any]) -> None:
    tmp = chunk_path.with_suffix(".npz.tmp")
    payload = {k: z[k] for k in z.files}
    payload.update(extra)
    np.savez_compressed(tmp, **payload)
    actual = tmp if tmp.is_file() else Path(str(tmp) + ".npz")
    backup = chunk_path.with_suffix(".npz.bak")
    shutil.copy2(chunk_path, backup)
    actual.replace(chunk_path)


def update_stats_from_sidecar(
    sidecar: Path,
    chunk_path: Path,
    data_cfg: dict[str, Any],
    z_stats: RunningStats,
    xyz_stats: RunningStats,
) -> bool:
    cond_prefix = str(data_cfg.get("cond_prefix", "first_"))
    target_prefix = str(data_cfg.get("target_prefix", "rest_"))
    try:
        with np.load(chunk_path, allow_pickle=True) as z:
            norm_factor = float(z["norm_scale"]) if "norm_scale" in z.files else 1.0
            include_stats = include_chunk_in_stats(z, data_cfg)
        if (not np.isfinite(norm_factor)) or norm_factor < 1e-6:
            norm_factor = 1.0
        with np.load(sidecar, allow_pickle=True) as cached:
            for prefix in (cond_prefix, target_prefix):
                z_key = f"{prefix}superpoint_z_sp"
                xyz_key = f"{prefix}superpoint_xyz"
                valid_key = f"{prefix}superpoint_valid_mask"
                if z_key not in cached.files or xyz_key not in cached.files:
                    return False
                z_sp = cached[z_key].astype(np.float32)
                xyz = cached[xyz_key].astype(np.float32) / norm_factor
                valid = cached[valid_key].astype(bool) if valid_key in cached.files else np.ones((z_sp.shape[0],), dtype=bool)
                if include_stats:
                    z_stats.update(z_sp[valid])
                    xyz_stats.update(xyz[valid])
        return True
    except Exception as exc:
        print(f"[warn] failed to read existing cache {sidecar}: {type(exc).__name__}: {exc}")
        return False


def process_chunk(
    chunk_path: Path,
    chunk_root: Path,
    model: nn.Module,
    data_cfg: dict[str, Any],
    out_mode: str,
    cache_root: Path | None,
    overwrite: bool,
    device: torch.device,
    amp: bool,
    image_cache: ImageCache,
    z_stats: RunningStats,
    xyz_stats: RunningStats,
) -> bool:
    out_path = sidecar_path(chunk_path, chunk_root=chunk_root, cache_root=cache_root)
    if out_mode == "sidecar" and out_path.is_file() and not overwrite:
        if update_stats_from_sidecar(out_path, chunk_path, data_cfg, z_stats, xyz_stats):
            return False
        print(f"[warn] recomputing unreadable/incomplete cache {out_path}")

    z = np.load(chunk_path, allow_pickle=True)
    cond_prefix = str(data_cfg.get("cond_prefix", "first_"))
    target_prefix = str(data_cfg.get("target_prefix", "rest_"))
    required = (
        "da3_depths",
        "camera_poses_c2w",
        "camera_intrinsics",
        "image_paths",
        f"{cond_prefix}superpoint_xyz",
        f"{target_prefix}superpoint_xyz",
    )
    missing = [k for k in required if k not in z.files]
    if missing:
        raise KeyError(f"{chunk_path} missing keys: {missing}")

    common = load_chunk_inputs(z, data_cfg, image_cache)
    include_stats = include_chunk_in_stats(z, data_cfg)
    norm_factor = float(common["norm_scale"])
    cond_xyz, cond_valid = load_group(z, cond_prefix, norm_factor)
    target_xyz, target_valid = load_group(z, target_prefix, norm_factor)

    cond_views = [0]
    target_views = list(range(int(common["image"].shape[0])))
    max_target_views = data_cfg.get("max_target_views", None)
    if max_target_views is None:
        model_capacity = int(max(getattr(model, "encoder_max_views", len(target_views)), getattr(model, "encoder_view_capacity", len(target_views))))
        if bool(getattr(model, "eval_use_all_encoder_views", True)) and len(target_views) > model_capacity:
            policy = str(data_cfg.get("on_too_many_target_views", "skip"))
            if policy == "truncate":
                target_views = target_views[:model_capacity]
            elif policy == "skip":
                print(
                    f"[skip] {chunk_path} has {len(target_views)} target views "
                    f"but model encoder capacity is {model_capacity}"
                )
                return False
            else:
                raise ValueError(
                    f"chunk has {len(target_views)} target views but model encoder capacity is {model_capacity}. "
                    "Set data.max_target_views or data.on_too_many_target_views='truncate'/'skip' if this is intentional."
                )
    else:
        target_views = target_views[: int(max_target_views)]

    cond_query_mode = str(data_cfg.get("cond_query_mode", "raw"))
    target_query_mode = str(data_cfg.get("target_query_mode", "fps"))
    if target_query_mode == "fps" and not hasattr(model, "cache_num_fps_sp_xyz"):
        raise KeyError("target_query_mode='fps' requires vae config dataset_kwargs.num_fps_sp_xyz")
    target_num_queries = int(getattr(model, "cache_num_fps_sp_xyz")) if target_query_mode == "fps" else None

    cond_z, cond_query_xyz, cond_query_valid = encode_group(
        model,
        common,
        cond_xyz,
        cond_valid,
        cond_views,
        device,
        amp,
        query_mode=cond_query_mode,
        num_queries=None,
    )
    target_z, target_query_xyz, target_query_valid = encode_group(
        model,
        common,
        target_xyz,
        target_valid,
        target_views,
        device,
        amp,
        query_mode=target_query_mode,
        num_queries=target_num_queries,
    )

    if include_stats:
        z_stats.update(cond_z[cond_query_valid])
        z_stats.update(target_z[target_query_valid])
        xyz_stats.update(cond_query_xyz[cond_query_valid])
        xyz_stats.update(target_query_xyz[target_query_valid])

    extra = {
        f"{cond_prefix}superpoint_xyz": (cond_query_xyz * norm_factor).astype(np.float32),
        f"{cond_prefix}superpoint_valid_mask": cond_query_valid.astype(bool),
        f"{cond_prefix}superpoint_z_sp": cond_z.astype(np.float16),
        f"{target_prefix}superpoint_xyz": (target_query_xyz * norm_factor).astype(np.float32),
        f"{target_prefix}superpoint_valid_mask": target_query_valid.astype(bool),
        f"{target_prefix}superpoint_z_sp": target_z.astype(np.float16),
        "cond_zsp_view_indices": np.array(cond_views, dtype=np.int32),
        "target_zsp_view_indices": np.array(target_views, dtype=np.int32),
        "cond_zsp_query_mode": np.array(cond_query_mode),
        "target_zsp_query_mode": np.array(target_query_mode),
        f"{cond_prefix}zsp_query_mode": np.array(cond_query_mode),
        f"{target_prefix}zsp_query_mode": np.array(target_query_mode),
        "zsp_schema_version": np.array("zsp_cache_v1"),
    }
    if out_mode == "sidecar":
        atomic_savez(out_path, **extra)
    elif out_mode == "inplace":
        write_inplace(chunk_path, z, extra)
    else:
        raise ValueError(f"Unsupported out_mode={out_mode}")
    return True


def parse_devices(devices_cfg: Any, default_device: str) -> list[str]:
    if devices_cfg is None:
        if str(default_device).startswith("cuda") and torch.cuda.is_available():
            return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        return [default_device]
    if isinstance(devices_cfg, str):
        items = [x.strip() for x in devices_cfg.split(",") if x.strip()]
    else:
        items = [str(x).strip() for x in devices_cfg if str(x).strip()]
    if len(items) == 1 and items[0].lower() == "auto":
        if torch.cuda.is_available():
            return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        return ["cpu"]
    out: list[str] = []
    for item in items:
        if item.isdigit():
            out.append(f"cuda:{item}")
        elif item.startswith("cuda") or item == "cpu":
            out.append(item)
        else:
            raise ValueError(f"Unsupported device spec: {item}")
    return out or [default_device]


def run_worker(
    worker_id: int,
    device_name: str,
    chunk_paths: list[str],
    chunk_root: str,
    vae_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    out_mode: str,
    cache_root: str | None,
    overwrite: bool,
    amp: bool,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = load_model(vae_cfg, device)
    apply_cache_fps_count(model, data_cfg)
    image_cache = ImageCache(max_files=int(data_cfg.get("image_cache_max_files", 1)))
    z_stats = RunningStats(64)
    xyz_stats = RunningStats(3)
    done = 0
    skipped = 0
    root = Path(chunk_root)
    cache = Path(cache_root) if cache_root is not None else None
    total = len(chunk_paths)
    print(f"[worker {worker_id}] start device={device} chunks={total}", flush=True)
    for i, chunk_str in enumerate(chunk_paths):
        chunk_path = Path(chunk_str)
        try:
            wrote = process_chunk(
                chunk_path=chunk_path,
                chunk_root=root,
                model=model,
                data_cfg=data_cfg,
                out_mode=out_mode,
                cache_root=cache,
                overwrite=overwrite,
                device=device,
                amp=amp,
                image_cache=image_cache,
                z_stats=z_stats,
                xyz_stats=xyz_stats,
            )
            if wrote:
                done += 1
                print(f"[worker {worker_id} {i + 1}/{total}] wrote {chunk_path}", flush=True)
            else:
                skipped += 1
        except Exception as exc:
            print(f"[worker {worker_id}] failed {chunk_path}: {type(exc).__name__}: {exc}", flush=True)
            raise
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[worker {worker_id}] done wrote={done} skipped={skipped}", flush=True)
    return {
        "done": done,
        "skipped": skipped,
        "z_stats": z_stats.state_dict(),
        "xyz_stats": xyz_stats.state_dict(),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache SuperpointVAE2C z_sp latents for first/rest generation chunks.")
    p.add_argument("--config", type=Path, default=VAE_ROOT / "cache_latent.json")
    p.add_argument("--chunk-root", type=Path, default=None)
    p.add_argument("--vae-ckpt", type=Path, default=None)
    p.add_argument("--vae-config", type=Path, default=None)
    p.add_argument("--out-mode", choices=("sidecar", "inplace"), default=None)
    p.add_argument("--cache-root", type=Path, default=None)
    p.add_argument("--stats-path", type=Path, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--devices", type=str, default=None, help="Comma-separated devices, e.g. '0,1' or 'cuda:0,cuda:1'. Use 'auto' for all GPUs.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--scene-split", choices=("all", "train", "val"), default="all")
    p.add_argument("--split-ratio", type=float, default=0.9)
    p.add_argument("--required-schema", type=str, default=None)
    p.add_argument("--required-frames", type=int, default=None)
    p.add_argument("--stats-data-split", choices=("all", "train", "val"), default=None)
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_json_config(args.config)
    vae_cfg = dict(cfg.get("vae", {}))
    data_cfg = dict(cfg.get("data", {}))
    if args.chunk_root is not None:
        cfg["chunk_root"] = str(args.chunk_root)
    if args.vae_ckpt is not None:
        vae_cfg["checkpoint"] = str(args.vae_ckpt)
    if args.vae_config is not None:
        vae_cfg["config"] = str(args.vae_config)
    if args.out_mode is not None:
        cfg["out_mode"] = args.out_mode
    if args.cache_root is not None:
        cfg["cache_root"] = str(args.cache_root)
    if args.stats_path is not None:
        cfg["stats_path"] = str(args.stats_path)
    if args.device is not None:
        vae_cfg["device"] = args.device
    if args.devices is not None:
        vae_cfg["devices"] = args.devices
    if args.limit is not None:
        cfg["limit"] = args.limit
    if args.overwrite:
        cfg["overwrite"] = True
    if args.no_amp:
        vae_cfg["amp"] = False
    if args.stats_data_split is not None:
        data_cfg["stats_data_split"] = args.stats_data_split

    devices = parse_devices(vae_cfg.get("devices", None), str(vae_cfg.get("device", "cuda")))
    amp = bool(vae_cfg.get("amp", True))
    out_mode = str(cfg.get("out_mode", "sidecar"))
    overwrite = bool(cfg.get("overwrite", False))

    chunk_root = Path(str(cfg["chunk_root"]))
    cache_root = Path(str(cfg["cache_root"])) if cfg.get("cache_root") is not None else None
    chunks = iter_chunk_npz(chunk_root)
    chunks = select_scene_split(chunks, chunk_root, args.scene_split, args.split_ratio)
    limit = cfg.get("limit", None)
    if limit is not None:
        chunks = chunks[: int(limit)]
    validate_chunk_contract(chunks, args.required_schema, args.required_frames)
    cache_msg = f" cache_root={cache_root}" if cache_root is not None else ""
    print(f"[init] chunks={len(chunks)} out_mode={out_mode} devices={devices}{cache_msg}")

    z_stats = RunningStats(64)
    xyz_stats = RunningStats(3)
    done = 0
    skipped = 0
    if len(devices) <= 1:
        device = resolve_device(devices[0])
        model = load_model(vae_cfg, device)
        apply_cache_fps_count(model, data_cfg)
        image_cache = ImageCache(max_files=int(data_cfg.get("image_cache_max_files", 1)))
        for i, chunk_path in enumerate(chunks):
            try:
                wrote = process_chunk(
                    chunk_path=chunk_path,
                    chunk_root=chunk_root,
                    model=model,
                    data_cfg=data_cfg,
                    out_mode=out_mode,
                    cache_root=cache_root,
                    overwrite=overwrite,
                    device=device,
                    amp=amp,
                    image_cache=image_cache,
                    z_stats=z_stats,
                    xyz_stats=xyz_stats,
                )
                if wrote:
                    done += 1
                    print(f"[{i + 1}/{len(chunks)}] wrote {chunk_path}")
                else:
                    skipped += 1
            except Exception as exc:
                print(f"[warn] failed {chunk_path}: {type(exc).__name__}: {exc}")
                raise
    else:
        if any(d == "cpu" for d in devices):
            raise ValueError("Multi-worker cache only supports CUDA devices. Use one CPU worker or specify CUDA devices.")
        chunk_splits = [[str(p) for p in chunks[i:: len(devices)]] for i in range(len(devices))]
        worker_args = [
            (
                i,
                devices[i],
                chunk_splits[i],
                str(chunk_root),
                vae_cfg,
                data_cfg,
                out_mode,
                str(cache_root) if cache_root is not None else None,
                overwrite,
                amp,
            )
            for i in range(len(devices))
            if chunk_splits[i]
        ]
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(worker_args)) as pool:
            results = pool.starmap(run_worker, worker_args)
        for result in results:
            done += int(result["done"])
            skipped += int(result["skipped"])
            zs = result["z_stats"]
            xs = result["xyz_stats"]
            z_stats.merge(zs["count"], zs["mean"], zs["m2"])
            xyz_stats.merge(xs["count"], xs["mean"], xs["m2"])

    stats_path = Path(str(cfg.get("stats_path", VAE_ROOT / "stats" / "zsp_xyz_stats.npz")))
    atomic_savez(
        stats_path,
        zsp_mean=z_stats.mean.astype(np.float32),
        zsp_m2=z_stats.m2.astype(np.float64),
        zsp_var=z_stats.var.astype(np.float32),
        zsp_std=z_stats.std.astype(np.float32),
        xyz_mean=xyz_stats.mean.astype(np.float32),
        xyz_m2=xyz_stats.m2.astype(np.float64),
        xyz_var=xyz_stats.var.astype(np.float32),
        xyz_std=xyz_stats.std.astype(np.float32),
        zsp_count=np.int64(z_stats.count),
        xyz_count=np.int64(xyz_stats.count),
    )
    print(f"[done] wrote={done} skipped={skipped} stats={stats_path} z_count={z_stats.count} xyz_count={xyz_stats.count}")


if __name__ == "__main__":
    main()
