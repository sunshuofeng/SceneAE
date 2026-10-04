"""
Build chunk-level RealEstate10K data with 3D superpoint geometry statistics.

This variant reads pixelSplat-style RE10K .torch chunks, uses RGB frames,
RE10K cameras, and Depth Anything 3 Base predicted depth. DINO/DA3/VAE
features are used for superpoint generation and aggregation; saved fields
include DA3 depth, superpoint VAE features, and per-superpoint geometry.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import json
import math
import sys
import types
from io import BytesIO
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from diffusers import AutoencoderKL
from PIL import Image
from torch_scatter import scatter_add, scatter_max, scatter_mean, scatter_std

ROOT = Path(__file__).resolve().parent.parent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "third_party" / "dinov3"))
sys.path.append(str(PROJECT_ROOT / "third_party" / "superpoint_transformer"))
sys.path.append(str(PROJECT_ROOT / "third_party" / "Depth-Anything-3" / "da3"))


def install_optional_dependency_stubs() -> None:
    moviepy_stub = types.ModuleType("moviepy")
    moviepy_editor_stub = types.ModuleType("moviepy.editor")
    moviepy_stub.__spec__ = importlib.machinery.ModuleSpec("moviepy", loader=None, is_package=True)
    moviepy_editor_stub.__spec__ = importlib.machinery.ModuleSpec("moviepy.editor", loader=None)
    moviepy_stub.editor = moviepy_editor_stub
    sys.modules.setdefault("moviepy", moviepy_stub)
    sys.modules.setdefault("moviepy.editor", moviepy_editor_stub)

    wandb_stub = types.ModuleType("wandb")
    wandb_stub.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)
    wandb_stub.run = None
    wandb_stub.finish = lambda *args, **kwargs: None
    wandb_stub.Table = lambda *args, **kwargs: None
    wandb_stub.plot_table = lambda *args, **kwargs: None
    sys.modules.setdefault("wandb", wandb_stub)


install_optional_dependency_stubs()

from depth_anything_3.api import DepthAnything3

FEATURE_DIM = 24
INNER_FRAME_STEP = 3
DEFAULT_FRAMES_PER_CHUNK = 9
FRAMES_PER_CHUNK = DEFAULT_FRAMES_PER_CHUNK
CHUNK_INDEX_STRIDE = INNER_FRAME_STEP * FRAMES_PER_CHUNK
KNN_K = 8
DEFAULT_VOXEL_SIZE = 0.01
TARGET_NUM_SUPERPOINTS = 4000
CHUNK_SCHEMA_VERSION = "geometry_sp_re10k_da3base_vae_v1"


def clear_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "cuda" in msg and "out of memory" in msg


class DinoFeatureExtractor(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)["x_norm_patchtokens"]


def pca_torch(x: torch.Tensor, k: int) -> torch.Tensor:
    x_centered = x - x.mean(dim=0, keepdim=True)
    cov = (x_centered.T @ x_centered) / max(x.shape[0] - 1, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    idx = torch.argsort(eigenvalues, descending=True)[:k]
    return x_centered @ eigenvectors[:, idx]


def process_feat(tokens: torch.Tensor, h: int, w: int, out_dim: int = FEATURE_DIM) -> torch.Tensor:
    feat = tokens.squeeze(0)
    feat = pca_torch(feat, out_dim)
    num_tokens = int(feat.shape[0])
    target_ratio = float(h) / float(max(w, 1))
    best_hw = None
    best_score = float("inf")
    for hp in range(1, int(math.sqrt(num_tokens)) + 1):
        if num_tokens % hp != 0:
            continue
        wp = num_tokens // hp
        for gh, gw in ((hp, wp), (wp, hp)):
            score = abs((float(gh) / float(max(gw, 1))) - target_ratio)
            if score < best_score:
                best_score = score
                best_hw = (gh, gw)
    if best_hw is None:
        raise ValueError(f"Cannot infer patch grid from token count {num_tokens}.")
    gh, gw = best_hw
    feat = feat.permute(1, 0).contiguous().view(1, out_dim, gh, gw)
    feat = F.interpolate(feat, size=(h, w), mode="bilinear", align_corners=False)
    return feat.squeeze(0).permute(1, 2, 0).contiguous()


def process_feat_batch(tokens: torch.Tensor, h: int, w: int, out_dim: int = FEATURE_DIM) -> list[torch.Tensor]:
    return [process_feat(tokens[i : i + 1], h, w, out_dim) for i in range(tokens.shape[0])]


def load_intrinsic(scene_dir: Path) -> dict:
    intr_path = scene_dir / "intrinsic" / "intrinsic_depth.txt"
    if not intr_path.is_file():
        raise FileNotFoundError(f"Missing intrinsic file: {intr_path}")
    k = np.loadtxt(intr_path, dtype=np.float64)
    if k.shape == (4, 4):
        k = k[:3, :3]
    if k.shape != (3, 3):
        raise ValueError(f"Unexpected intrinsic shape {k.shape} in {intr_path}")
    return {"fx": float(k[0, 0]), "fy": float(k[1, 1]), "cx": float(k[0, 2]), "cy": float(k[1, 2])}


def intr_dict_to_3x3(intr: dict) -> np.ndarray:
    k = np.eye(3, dtype=np.float64)
    k[0, 0] = intr["fx"]
    k[1, 1] = intr["fy"]
    k[0, 2] = intr["cx"]
    k[1, 2] = intr["cy"]
    return k


def load_pose_matrix(pose_path: Path) -> np.ndarray:
    pose = np.loadtxt(pose_path, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"Unexpected pose shape {pose.shape} in {pose_path}")
    return pose


def pose_is_finite(pose: np.ndarray) -> bool:
    return bool(np.isfinite(pose).all())


def depth_to_point_cloud(depth: np.ndarray, intr: dict, valid_mask: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    z = depth[valid_mask]
    x = (u[valid_mask] - intr["cx"]) * z / intr["fx"]
    y = (v[valid_mask] - intr["cy"]) * z / intr["fy"]
    return np.stack([x, y, z], axis=1).astype(np.float32)


def transform_points(points: np.ndarray, pose_c2w: np.ndarray) -> np.ndarray:
    pts_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    out = (pose_c2w @ pts_h.T).T
    return out[:, :3]


def make_sp_id_map_from_depth(
    depth: np.ndarray,
    pose_c2w: np.ndarray,
    intr: dict,
    voxel_labels: np.ndarray,
    raw_to_fixed: np.ndarray,
    voxel_size: float,
    min_depth: float,
    max_depth: float,
    voxel_lookup: dict[tuple[int, int, int], int],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape
    valid = (depth > min_depth) & (depth < max_depth)
    id_full = np.full((h, w), -1, dtype=np.int32)
    valid_full = np.zeros((h, w), dtype=bool)
    if valid.sum() == 0 or len(voxel_lookup) == 0:
        return id_full, valid_full

    pts_cam = depth_to_point_cloud(depth, intr, valid)
    pts_world = transform_points(pts_cam, pose_c2w).astype(np.float32)
    finite = np.isfinite(pts_world).all(axis=1)
    if finite.sum() == 0:
        return id_full, valid_full

    yy, xx = np.where(valid)
    yy = yy[finite]
    xx = xx[finite]
    coords = np.floor(pts_world[finite].astype(np.float64) / max(voxel_size, 1e-6)).astype(np.int64)
    fixed = np.full((coords.shape[0],), -1, dtype=np.int32)
    for i, coord in enumerate(coords):
        voxel_id = voxel_lookup.get((int(coord[0]), int(coord[1]), int(coord[2])))
        if voxel_id is None:
            continue
        raw_id = int(voxel_labels[voxel_id])
        if 0 <= raw_id < raw_to_fixed.shape[0]:
            fixed[i] = raw_to_fixed[raw_id]

    ok = fixed >= 0

    id_full[yy[ok], xx[ok]] = fixed[ok]
    valid_full[yy[ok], xx[ok]] = True
    valid_full &= id_full >= 0
    return id_full, valid_full


def fix_superpoint_slots(
    sp_xyz_raw: np.ndarray,
    target_n: int = TARGET_NUM_SUPERPOINTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_raw = sp_xyz_raw.shape[0]
    if n_raw == 0:
        raise RuntimeError("empty superpoint set before padding")

    if n_raw <= target_n:
        source_idx = np.arange(n_raw, dtype=np.int32)
        valid_mask = np.ones(n_raw, dtype=bool)
        pad_idx = np.random.randint(0, n_raw, size=(target_n - n_raw)).astype(np.int32)
        sp_xyz_fixed = np.concatenate([sp_xyz_raw, sp_xyz_raw[pad_idx]], axis=0)
        source_idx_fixed = np.concatenate([source_idx, pad_idx], axis=0)
        valid_mask_fixed = np.concatenate(
            [valid_mask, np.zeros(target_n - n_raw, dtype=bool)],
            axis=0,
        )
    else:
        keep_idx = np.random.choice(n_raw, size=target_n, replace=False).astype(np.int32)
        sp_xyz_fixed = sp_xyz_raw[keep_idx]
        source_idx_fixed = keep_idx
        valid_mask_fixed = np.ones(target_n, dtype=bool)

    raw_to_fixed = np.full((n_raw,), -1, dtype=np.int32)
    for fixed_id, raw_id in enumerate(source_idx_fixed):
        if valid_mask_fixed[fixed_id] and raw_to_fixed[raw_id] < 0:
            raw_to_fixed[raw_id] = fixed_id

    return (
        sp_xyz_fixed.astype(np.float32),
        valid_mask_fixed.astype(bool),
        source_idx_fixed.astype(np.int32),
        raw_to_fixed,
    )


def raw_field_to_fixed(
    raw_field: np.ndarray,
    source_idx_fixed: np.ndarray,
    valid_mask_fixed: np.ndarray,
    fill_value=0,
) -> np.ndarray:
    raw_field = np.asarray(raw_field)
    out_shape = (source_idx_fixed.shape[0],) + raw_field.shape[1:]
    fixed = np.full(out_shape, fill_value, dtype=raw_field.dtype)
    if raw_field.shape[0] == 0:
        return fixed
    clipped = np.clip(source_idx_fixed.astype(np.int64), 0, raw_field.shape[0] - 1)
    fixed[:] = raw_field[clipped]
    fixed[~valid_mask_fixed] = fill_value
    return fixed


@torch.no_grad()
def compute_superpoint_geometry_stats(
    points_world: np.ndarray,
    sp_idx: np.ndarray,
    sp_xyz_raw: np.ndarray,
    norm_factor: float,
    device: torch.device,
    eps: float = 1e-8,
) -> dict[str, np.ndarray]:
    n_raw = int(sp_xyz_raw.shape[0])

    pts_np = np.nan_to_num(points_world, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    pts = torch.from_numpy(pts_np).to(device)
    sp_idx_t = torch.from_numpy(sp_idx.astype(np.int64)).to(device)
    center = torch.from_numpy(sp_xyz_raw.astype(np.float32)).to(device)

    point_count = torch.zeros(n_raw, dtype=torch.long, device=device)
    point_count.scatter_add_(0, sp_idx_t, torch.ones_like(sp_idx_t))
    geometry_valid = point_count >= 3
    valid_f = geometry_valid.float()

    delta = pts - center[sp_idx_t]
    dist = delta.norm(dim=1)

    radius_mean = scatter_mean(dist, sp_idx_t, dim=0, dim_size=n_raw)
    radius_std = scatter_std(dist, sp_idx_t, dim=0, dim_size=n_raw, unbiased=False)
    radius_max = scatter_max(dist, sp_idx_t, dim=0, dim_size=n_raw)[0]
    radius_max = torch.where(point_count > 0, radius_max, torch.zeros_like(radius_max))

    order_inner = torch.argsort(dist)
    sp_sorted = sp_idx_t[order_inner]
    dist_sorted_inner = dist[order_inner]
    order_outer = torch.argsort(sp_sorted, stable=True)
    dist_sorted = dist_sorted_inner[order_outer]
    seg_ptr = torch.zeros(n_raw + 1, dtype=torch.long, device=device)
    seg_ptr[1:] = torch.cumsum(point_count, dim=0)

    def seg_quantile(q: float) -> torch.Tensor:
        out_q = torch.zeros(n_raw, dtype=dist.dtype, device=device)
        valid = point_count > 0
        if not valid.any():
            return out_q
        k = (q * point_count.float()).long()
        k = torch.minimum(k, (point_count - 1).clamp(min=0))
        gather_idx = seg_ptr[:-1][valid] + k[valid]
        out_q[valid] = dist_sorted[gather_idx]
        return out_q

    radius_p50 = seg_quantile(0.50)
    radius_p90 = seg_quantile(0.90)
    radius_p95 = seg_quantile(0.95)

    outer = delta.unsqueeze(2) * delta.unsqueeze(1)
    cov_sum = scatter_add(outer, sp_idx_t, dim=0, dim_size=n_raw)
    denom = (point_count - 1).clamp(min=1).to(cov_sum.dtype).view(-1, 1, 1)
    cov = cov_sum / denom
    cov = torch.where(geometry_valid.view(-1, 1, 1), cov, torch.zeros_like(cov))
    cov = torch.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)

    vals_asc, vecs_asc = torch.linalg.eigh(cov.double())
    vals_asc = vals_asc.clamp(min=0.0).float()
    vecs_asc = vecs_asc.float()
    vals = vals_asc.flip(dims=[1])
    vecs = vecs_asc.flip(dims=[2])

    normal = vecs_asc[:, :, 0].clone()
    normal = normal / normal.norm(dim=1, keepdim=True).clamp(min=eps)
    normal = normal * valid_f.view(-1, 1)
    vals = vals * valid_f.view(-1, 1)
    vecs = vecs * valid_f.view(-1, 1, 1)

    l1, l2, l3 = vals[:, 0], vals[:, 1], vals[:, 2]
    denom_l1 = l1.clamp(min=eps)
    lsum = (l1 + l2 + l3).clamp(min=eps)
    linearity = ((l1 - l2) / denom_l1) * valid_f
    planarity = ((l2 - l3) / denom_l1) * valid_f
    scattering = (l3 / denom_l1) * valid_f
    anisotropy = ((l1 - l3) / denom_l1) * valid_f
    curvature = (l3 / lsum) * valid_f
    omnivariance = (l1 * l2 * l3).clamp(min=0.0).pow(1.0 / 3.0) * valid_f
    probs = vals / lsum.unsqueeze(1)
    eigenentropy = -(probs * torch.log(probs.clamp(min=eps))).sum(dim=1) * valid_f

    vol = (4.0 / 3.0) * math.pi * radius_p90.clamp(min=eps).pow(3)
    density_p90 = (point_count.float() / vol) * valid_f
    log_point_count = torch.log1p(point_count.float())
    nf = max(float(norm_factor), eps)

    def to_np(t: torch.Tensor, dtype=np.float32) -> np.ndarray:
        return t.detach().cpu().numpy().astype(dtype)

    return {
        "sp_center_from_points": to_np(center),
        "sp_point_count": to_np(point_count, np.int32),
        "sp_log_point_count": to_np(log_point_count),
        "sp_radius_mean": to_np(radius_mean),
        "sp_radius_std": to_np(radius_std),
        "sp_radius_max": to_np(radius_max),
        "sp_radius_p50": to_np(radius_p50),
        "sp_radius_p90": to_np(radius_p90),
        "sp_radius_p95": to_np(radius_p95),
        "sp_radius_mean_norm": to_np(radius_mean / nf),
        "sp_radius_std_norm": to_np(radius_std / nf),
        "sp_radius_max_norm": to_np(radius_max / nf),
        "sp_radius_p50_norm": to_np(radius_p50 / nf),
        "sp_radius_p90_norm": to_np(radius_p90 / nf),
        "sp_radius_p95_norm": to_np(radius_p95 / nf),
        "sp_cov": to_np(cov),
        "sp_cov_norm": to_np(cov / (nf * nf)),
        "sp_cov_eigvals": to_np(vals),
        "sp_cov_eigvals_norm": to_np(vals / (nf * nf)),
        "sp_cov_eigvecs": to_np(vecs),
        "sp_normal": to_np(normal),
        "sp_linearity": to_np(linearity),
        "sp_planarity": to_np(planarity),
        "sp_scattering": to_np(scattering),
        "sp_anisotropy": to_np(anisotropy),
        "sp_curvature": to_np(curvature),
        "sp_omnivariance": to_np(omnivariance),
        "sp_omnivariance_norm": to_np(omnivariance / (nf * nf)),
        "sp_eigenentropy": to_np(eigenentropy),
        "sp_density_p90": to_np(density_p90),
        "sp_density_p90_norm": to_np(density_p90 * (nf ** 3)),
        "sp_geometry_valid_mask": to_np(geometry_valid, bool),
    }


def _voxel_fallback_labels(points_world: np.ndarray, voxel_size: float = 0.05) -> np.ndarray:
    coords = np.floor(points_world / max(voxel_size, 1e-6)).astype(np.int64)
    _, inv = np.unique(coords, axis=0, return_inverse=True)
    return inv.astype(np.int64)


def voxel_aggregate_fused_cloud(
    points_world: np.ndarray,
    dino_feats: torch.Tensor,
    full_feats: torch.Tensor,
    voxel_size: float,
    device: torch.device,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    if voxel_size <= 0:
        return points_world, dino_feats, full_feats

    vs = float(voxel_size)
    coords = np.floor(points_world.astype(np.float64) / vs).astype(np.int64)
    _, inverse = np.unique(coords, axis=0, return_inverse=True)

    inv_t = torch.from_numpy(inverse).long().to(device)
    pts_t = torch.from_numpy(points_world.astype(np.float32)).to(device)
    voxel_xyz = scatter_mean(pts_t, inv_t, dim=0)
    voxel_dino = scatter_mean(dino_feats.float(), inv_t, dim=0)
    voxel_full = scatter_mean(full_feats.float(), inv_t, dim=0)

    return (
        voxel_xyz.detach().cpu().numpy().astype(np.float32),
        voxel_dino,
        voxel_full,
    )


def voxel_aggregate_partition_cloud(
    points_world: np.ndarray,
    partition_feats: torch.Tensor,
    vae_feats: torch.Tensor,
    voxel_size: float,
    device: torch.device,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    if voxel_size <= 0:
        coords = np.arange(points_world.shape[0], dtype=np.int64)[:, None]
        coords = np.pad(coords, ((0, 0), (0, 2)), mode="constant")
        inverse = np.arange(points_world.shape[0], dtype=np.int64)
        return points_world, partition_feats, vae_feats, coords, inverse

    vs = float(voxel_size)
    coords = np.floor(points_world.astype(np.float64) / vs).astype(np.int64)
    voxel_coords, inverse = np.unique(coords, axis=0, return_inverse=True)

    inv_t = torch.from_numpy(inverse).long().to(device)
    pts_t = torch.from_numpy(points_world.astype(np.float32)).to(device)
    voxel_xyz = scatter_mean(pts_t, inv_t, dim=0)
    voxel_feat = scatter_mean(partition_feats.float(), inv_t, dim=0)
    voxel_vae = scatter_mean(vae_feats.float(), inv_t, dim=0)

    return (
        voxel_xyz.detach().cpu().numpy().astype(np.float32),
        voxel_feat,
        voxel_vae,
        voxel_coords.astype(np.int64),
        inverse.astype(np.int64),
    )


def nearest_prevoxel_points_to_superpoints(
    prevoxel_points_world: np.ndarray,
    prevoxel_sp_idx: np.ndarray,
    sp_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_sp = int(sp_xyz.shape[0])
    nearest_points = np.zeros((n_sp, 3), dtype=np.float32)
    nearest_indices = np.full((n_sp,), -1, dtype=np.int64)
    nearest_dists = np.full((n_sp,), np.inf, dtype=np.float32)
    valid = np.zeros((n_sp,), dtype=bool)
    if prevoxel_points_world.shape[0] == 0 or n_sp == 0:
        return nearest_points, nearest_indices, nearest_dists, valid

    sp_idx = prevoxel_sp_idx.astype(np.int64, copy=False)
    ok = (sp_idx >= 0) & (sp_idx < n_sp)
    if not ok.any():
        return nearest_points, nearest_indices, nearest_dists, valid

    raw_indices = np.nonzero(ok)[0]
    pts = prevoxel_points_world[ok].astype(np.float32, copy=False)
    assigned_sp = sp_idx[ok]
    delta = pts - sp_xyz[assigned_sp].astype(np.float32, copy=False)
    dist2 = np.einsum("ij,ij->i", delta, delta).astype(np.float32)

    order = np.lexsort((dist2, assigned_sp))
    assigned_sorted = assigned_sp[order]
    first = np.r_[True, assigned_sorted[1:] != assigned_sorted[:-1]]
    best_order_idx = order[first]
    best_sp = assigned_sp[best_order_idx]
    best_raw_idx = raw_indices[best_order_idx]

    nearest_points[best_sp] = prevoxel_points_world[best_raw_idx].astype(np.float32, copy=False)
    nearest_indices[best_sp] = best_raw_idx.astype(np.int64, copy=False)
    nearest_dists[best_sp] = np.sqrt(dist2[best_order_idx]).astype(np.float32, copy=False)
    valid[best_sp] = True
    return nearest_points, nearest_indices, nearest_dists, valid


@torch.no_grad()
def compute_ezsp_hierarchy(
    points_world: np.ndarray,
    features: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    points_world = np.nan_to_num(points_world, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    center = points_world.mean(axis=0, keepdims=True)
    points_centered = points_world - center
    scale = np.linalg.norm(points_centered, axis=1).mean()
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    points_norm = points_centered / scale
    fallback_voxel_size = max(0.02, 0.06 * float(scale))

    try:
        from src.data import Data as SPTData
        from src.transforms.graph import AdjacencyGraph
        from src.transforms.neighbors import KNN
        from src.transforms.partition import GreedyContourPriorPartition

        k_local = min(KNN_K, max(points_norm.shape[0] - 1, 1))
        data = SPTData(pos=torch.from_numpy(points_norm).to(device), x=features.to(device))
        data = KNN(k=k_local, r_max=0.12, self_is_neighbor=False)(data)
        data = AdjacencyGraph(k=k_local, w=0.0)(data)
        partition = GreedyContourPriorPartition(
            reg=[2e-2, 5e-2, 1e-1],
            min_size=[5, 30, 90],
            edge_weight_mode="unit",
            edge_reduce="add",
            k=0,
            w_adjacency=0.0,
            max_iterations=-1,
            verbose=False,
            sharding=None,
        )
        nag = partition(data)
       
        
        fine = nag.get_super_index(3, low=0).detach().cpu().numpy().astype(np.int64)
       
        return fine
    except Exception as exc:
        if is_cuda_oom(exc):
            raise
        print(
            f"[warn] EZSP failed ({type(exc).__name__}: {exc}); "
            f"fallback to voxel clustering."
        )
        fine = _voxel_fallback_labels(points_world, voxel_size=fallback_voxel_size).astype(np.int64)
        return fine


def iter_re10k_files(data_root: Path, split: str) -> Iterable[Path]:
    split_root = data_root / split
    if not split_root.is_dir() and data_root.name == split:
        split_root = data_root
    if not split_root.is_dir():
        raise FileNotFoundError(f"Missing RE10K split directory: {split_root}")
    yield from sorted(split_root.glob("*.torch"))


def safe_path_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name)


def build_sequential_chunks(frames: list[dict], frames_per_chunk: int, frame_stride: int) -> list[list[dict]]:
    chunks: list[list[dict]] = []
    if not frames:
        return chunks
    span = 1 + frame_stride * (frames_per_chunk - 1)
    for start in range(0, len(frames) - span + 1, span):
        chunks.append([frames[start + frame_stride * j] for j in range(frames_per_chunk)])
    return chunks


def _camera_center_and_forward_score(poses: list[np.ndarray], opencv: bool) -> tuple[np.ndarray, np.ndarray, float, float]:
    centers = np.stack([p[:3, 3] for p in poses], axis=0).astype(np.float64)
    center_mean = centers.mean(axis=0)
    dirs_to_center = center_mean[None, :] - centers
    denom = np.linalg.norm(dirs_to_center, axis=1)
    ok = denom > 1e-8
    cam_forward = np.array([0.0, 0.0, 1.0 if opencv else -1.0], dtype=np.float64)
    forwards = np.stack([p[:3, :3] @ cam_forward for p in poses], axis=0)
    forward_norm = np.linalg.norm(forwards, axis=1)
    ok &= forward_norm > 1e-8
    if not ok.any():
        return centers, forwards, float("nan"), float("nan")
    cos = np.einsum("ij,ij->i", forwards[ok], dirs_to_center[ok]) / (forward_norm[ok] * denom[ok])
    return centers, forwards, float(np.nanmean(cos)), float(np.nanmedian(cos))


def print_chunk_diagnostics(
    scene_name: str,
    out_name: str,
    stems: list[str],
    raw_depths: list[np.ndarray],
    sanitized_depths: list[np.ndarray],
    poses_c2w: list[np.ndarray],
    raw_poses_c2w: list[np.ndarray],
    intrs: list[dict],
    min_depth: float,
    max_depth: float,
) -> None:
    print(f"[debug] chunk has no valid points: {scene_name} {out_name} frames={stems[0]}..{stems[-1]}")
    print(f"[debug] depth filter: min_depth={min_depth:g}, max_depth={max_depth:g}")
    for i, (raw_d, clean_d, pose, intr) in enumerate(zip(raw_depths, sanitized_depths, poses_c2w, intrs, strict=True)):
        raw_d = np.asarray(raw_d, dtype=np.float32)
        finite_raw = np.isfinite(raw_d)
        raw_vals = raw_d[finite_raw]
        if raw_vals.size > 0:
            q = np.percentile(raw_vals, [0, 1, 5, 50, 95, 99, 100])
            raw_msg = (
                f"raw finite={raw_vals.size}/{raw_d.size} "
                f"q0/1/5/50/95/99/100="
                f"{q[0]:.4g}/{q[1]:.4g}/{q[2]:.4g}/{q[3]:.4g}/{q[4]:.4g}/{q[5]:.4g}/{q[6]:.4g}"
            )
        else:
            raw_msg = f"raw finite=0/{raw_d.size}"

        valid = (clean_d > min_depth) & (clean_d < max_depth)
        finite_world = 0
        if valid.sum() > 0:
            pts_cam = depth_to_point_cloud(clean_d, intr, valid)
            pts_w = transform_points(pts_cam, pose)
            finite_world = int(np.isfinite(pts_w).all(axis=1).sum())
        print(
            f"[debug]   {i:02d} {stems[i]}: {raw_msg}; "
            f"valid_after_filter={int(valid.sum())}/{clean_d.size}; "
            f"finite_world={finite_world}; "
            f"K=({intr['fx']:.2f},{intr['fy']:.2f},{intr['cx']:.2f},{intr['cy']:.2f})"
        )

    centers_cv, _, cv_mean, cv_median = _camera_center_and_forward_score(poses_c2w, opencv=True)
    centers_raw_gl, _, raw_gl_mean, raw_gl_median = _camera_center_and_forward_score(raw_poses_c2w, opencv=False)
    _, _, raw_cv_mean, raw_cv_median = _camera_center_and_forward_score(raw_poses_c2w, opencv=True)
    cmin = centers_cv.min(axis=0)
    cmax = centers_cv.max(axis=0)
    print(
        "[debug] camera centers xyz min/max: "
        f"min=({cmin[0]:.4g},{cmin[1]:.4g},{cmin[2]:.4g}) "
        f"max=({cmax[0]:.4g},{cmax[1]:.4g},{cmax[2]:.4g})"
    )
    print(
        "[debug] pose forward-to-center cosine, larger positive is more plausible: "
        f"raw_as_nerf_opengl mean/median={raw_gl_mean:.4f}/{raw_gl_median:.4f}; "
        f"raw_as_opencv mean/median={raw_cv_mean:.4f}/{raw_cv_median:.4f}; "
        f"converted_opencv mean/median={cv_mean:.4f}/{cv_median:.4f}"
    )


def decode_re10k_image(image_bytes: torch.Tensor) -> np.ndarray:
    pil = Image.open(BytesIO(image_bytes.numpy().tobytes())).convert("RGB")
    return np.asarray(pil)


def re10k_cameras_to_poses_intrinsics(cameras: torch.Tensor, image_hw: tuple[int, int]) -> tuple[list[np.ndarray], list[dict]]:
    cam = cameras.detach().cpu().float()
    n = int(cam.shape[0])
    h, w = image_hw

    intrs: list[dict] = []
    for fx, fy, cx, cy in cam[:, :4].numpy():
        intrs.append(
            {
                "fx": float(fx) * float(w),
                "fy": float(fy) * float(h),
                "cx": float(cx) * float(w),
                "cy": float(cy) * float(h),
            }
        )

    w2c = torch.eye(4, dtype=torch.float32).repeat(n, 1, 1)
    w2c[:, :3] = cam[:, 6:].reshape(n, 3, 4)
    c2w = torch.linalg.inv(w2c).numpy().astype(np.float64)
    return [c2w[i] for i in range(n)], intrs


def build_re10k_frames(scene: dict, min_num_frames: int = 1) -> list[dict]:
    images = scene["images"]
    cameras = scene["cameras"]
    timestamps = scene.get("timestamps")
    if len(images) < min_num_frames:
        return []
    first_rgb = decode_re10k_image(images[0])
    orig_h, orig_w = first_rgb.shape[:2]
    poses, intrs = re10k_cameras_to_poses_intrinsics(cameras, image_hw=(orig_h, orig_w))

    frames: list[dict] = []
    for i, (image_bytes, pose, intr) in enumerate(zip(images, poses, intrs, strict=True)):
        if not pose_is_finite(pose):
            continue
        ts = int(timestamps[i]) if timestamps is not None else i
        frames.append(
            {
                "image_bytes": image_bytes,
                "pose_c2w": pose,
                "raw_pose_c2w": pose,
                "intrinsic": intr,
                "frame_index": i,
                "timestamp": ts,
            }
        )
    frames.sort(key=lambda x: x["frame_index"])
    return frames


@torch.no_grad()
def dino_dense_batch(
    rgb_images_u8_hw3: list[np.ndarray],
    norm: transforms.Compose,
    dino_model: DinoFeatureExtractor,
    device: torch.device,
) -> list[torch.Tensor]:
    imgs = torch.stack([norm(rgb) for rgb in rgb_images_u8_hw3], dim=0).to(device)
    tokens = dino_model(imgs)
    h, w = rgb_images_u8_hw3[0].shape[:2]
    feats = process_feat_batch(tokens, h, w, FEATURE_DIM)
    del imgs, tokens
    return feats


@torch.no_grad()
def vae_latent_dense_batch(
    rgb_images_u8_hw3: list[np.ndarray],
    vae: AutoencoderKL,
    device: torch.device,
) -> list[torch.Tensor]:
    x = torch.stack([transforms.ToTensor()(rgb) for rgb in rgb_images_u8_hw3], dim=0).to(device)
    x = 2.0 * x - 1.0
    latent = vae.encode(x).latent_dist.sample()
    sf = getattr(vae.config, "scaling_factor", 1.0)
    latent = latent * float(sf)
    h, w = rgb_images_u8_hw3[0].shape[:2]
    latent = F.interpolate(latent.float(), size=(h, w), mode="bilinear", align_corners=False)
    latent = latent.permute(0, 2, 3, 1).contiguous()
    out = [latent[i].contiguous() for i in range(latent.shape[0])]
    del x, latent
    return out


def run_da3_depth_and_features(
    da3_model: DepthAnything3,
    rgb_images_u8_320: list[np.ndarray],
    poses_c2w: np.ndarray,
    intrinsics_3x3_np: np.ndarray,
    resize_hw: tuple[int, int],
    feat_layer_id: int,
    process_res: int,
    process_res_method: str,
    device: torch.device,
) -> tuple[np.ndarray, torch.Tensor]:
    prediction = da3_model.inference(
        rgb_images_u8_320,
        extrinsics=poses_c2w,
        intrinsics=intrinsics_3x3_np,
        align_to_input_ext_scale=True,
        infer_gs=False,
        use_ray_pose=False,
        process_res=process_res,
        process_res_method=process_res_method,
        export_feat_layers=[feat_layer_id],
    )
    if not hasattr(prediction, "depth") or prediction.depth is None:
        raise RuntimeError("DepthAnything3 prediction has no depth field.")
    depths = np.asarray(prediction.depth, dtype=np.float32)  # [V, H0, W0]
    aux = prediction.aux
    key = f"feat_layer_{feat_layer_id}"
    if aux is None or key not in aux:
        keys = list(aux.keys()) if aux else []
        raise KeyError(f"DA3 auxiliary key {key} missing after inference. Aux keys={keys}")
    feats = aux[key].astype(np.float32)  # [V, Hf, Wf, C]

    tgt_h, tgt_w = resize_hw
    depth_t = torch.from_numpy(depths).unsqueeze(1).to(device=device, dtype=torch.float32)
    depth_t = F.interpolate(depth_t, size=(tgt_h, tgt_w), mode="bilinear", align_corners=False)
    depths_up = depth_t[:, 0].contiguous().cpu().numpy()

    da3_pca_maps: list[torch.Tensor] = []
    for vi in range(feats.shape[0]):
        feat_i = torch.from_numpy(feats[vi]).to(device=device, dtype=torch.float32)
        hf, wf, _ = feat_i.shape
        feat_i = pca_torch(feat_i.view(hf * wf, -1).contiguous(), FEATURE_DIM)
        feat_i = feat_i.view(hf, wf, FEATURE_DIM).permute(2, 0, 1).unsqueeze(0)
        feat_i = F.interpolate(feat_i, size=(tgt_h, tgt_w), mode="bilinear", align_corners=False)
        da3_pca_maps.append(feat_i[0].permute(1, 2, 0).contiguous())
    feat_up = torch.stack(da3_pca_maps, dim=0)
    del prediction, aux, feats, depth_t, da3_pca_maps
    return depths_up, feat_up


def sanitize_pred_depth(depth: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    invalid = (~np.isfinite(depth)) | (depth <= min_depth) | (depth >= max_depth)
    depth[invalid] = 0.0
    return depth


def fuse_and_superpoint_chunk(
    depths_list: list[np.ndarray],
    poses_list: list[np.ndarray],
    intr_used_list: list[dict],
    partition_features_list: list[torch.Tensor],
    vae_features_list: list[torch.Tensor],
    device: torch.device,
    voxel_size: float,
    min_depth: float,
    max_depth: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    all_xyz: list[np.ndarray] = []
    all_features: list[torch.Tensor] = []
    all_vae: list[torch.Tensor] = []

    for depth, pose, intr, feat, vae_feat in zip(
        depths_list,
        poses_list,
        intr_used_list,
        partition_features_list,
        vae_features_list,
        strict=True,
    ):
        valid_mask = (depth > min_depth) & (depth < max_depth)
        if valid_mask.sum() < 32:
            continue
        pts_cam = depth_to_point_cloud(depth, intr, valid_mask)
        pts_w = transform_points(pts_cam, pose)
        finite = np.isfinite(pts_w).all(axis=1)
        if finite.sum() < 32:
            continue
        feat_nv = feat[valid_mask].float()
        vae_nv = vae_feat[valid_mask].float()
        pts_w = pts_w[finite]
        feat_part = feat_nv[finite]
        vae_part = vae_nv[finite]
        all_xyz.append(pts_w)
        all_features.append(feat_part)
        all_vae.append(vae_part)

    if not all_xyz:
        raise RuntimeError("No valid points in chunk (DA3 depth / pose).")

    points_world = np.concatenate(all_xyz, axis=0)
    prevoxel_points_world = points_world.astype(np.float32, copy=True)
    feat_cat = torch.cat(all_features, dim=0)
    vae_cat = torch.cat(all_vae, dim=0)

    points_world, feat_cat, vae_cat, voxel_coords, prevoxel_to_voxel = voxel_aggregate_partition_cloud(
        points_world,
        feat_cat,
        vae_cat,
        voxel_size,
        device,
    )
    if points_world.shape[0] < 32:
        raise RuntimeError(f"Too few voxels after merge ({points_world.shape[0]} < 32).")

    world_dist = np.linalg.norm(points_world, ord=2, axis=-1)
    norm_factor = float(np.nanmean(world_dist)) if world_dist.size > 0 else 1.0
    if (not np.isfinite(norm_factor)) or norm_factor < 1e-6:
        norm_factor = 1.0

    sp_idx = compute_ezsp_hierarchy(points_world, feat_cat, device=device)
    pts_w_t = torch.from_numpy(points_world).float().to(device)
    sp_idx_t = torch.from_numpy(sp_idx).long().to(device)
    sp_xyz = scatter_mean(pts_w_t, sp_idx_t, dim=0)
    sp_vae = scatter_mean(vae_cat.float(), sp_idx_t, dim=0)
    sp_xyz_np = sp_xyz.detach().cpu().numpy().astype(np.float32)
    prevoxel_sp_idx = sp_idx[prevoxel_to_voxel].astype(np.int64)
    (
        sp_nearest_prevoxel_xyz,
        sp_nearest_prevoxel_index,
        sp_nearest_prevoxel_dist,
        sp_nearest_prevoxel_valid,
    ) = nearest_prevoxel_points_to_superpoints(
        prevoxel_points_world,
        prevoxel_sp_idx,
        sp_xyz_np,
    )
    return (
        sp_xyz_np,
        sp_idx.astype(np.int64),
        points_world.astype(np.float32),
        voxel_coords.astype(np.int64),
        sp_vae.detach().cpu().numpy().astype(np.float32),
        prevoxel_points_world.astype(np.float32),
        prevoxel_sp_idx.astype(np.int64),
        sp_nearest_prevoxel_xyz,
        sp_nearest_prevoxel_index,
        sp_nearest_prevoxel_dist,
        sp_nearest_prevoxel_valid,
        norm_factor,
    )


def process_scene(
    scene: dict,
    source_file: Path,
    split: str,
    out_root: Path,
    dino_model: DinoFeatureExtractor,
    da3_model: DepthAnything3,
    vae: AutoencoderKL,
    device: torch.device,
    resize_w: int,
    resize_h: int,
    da3_feat_layer: int,
    da3_process_res: int,
    da3_process_res_method: str,
    norm: transforms.Compose,
    skip_existing: bool,
    voxel_size: float,
    min_depth: float,
    max_depth: float,
    target_num_superpoints: int,
    frames_per_chunk: int,
    frame_stride: int,
    skip_bad_chunks: bool,
    only_chunk_ids: set[int] | None = None,
    only_frame_groups: list[tuple[int, list[int]]] | None = None,
) -> None:
    scene_name = str(scene.get("key", source_file.stem))
    scene_dir_name = safe_path_name(scene_name)
    try:
        frames = build_re10k_frames(scene, min_num_frames=1 if only_frame_groups else frames_per_chunk)
    except Exception as exc:
        if not skip_bad_chunks:
            raise
        print(f"[skip] bad RE10K scene metadata in {scene_name}: {type(exc).__name__}: {exc}")
        return

    if only_frame_groups is not None:
        by_index = {int(frame["frame_index"]): frame for frame in frames}
        enumerated: list[tuple[int, list[dict]]] = []
        for chunk_id, indices in only_frame_groups:
            missing = [int(index) for index in indices if int(index) not in by_index]
            if missing:
                raise KeyError(f"{scene_name} is missing frame indices {missing}")
            enumerated.append((int(chunk_id), [by_index[int(index)] for index in indices]))
    else:
        enumerated = list(
            enumerate(
                build_sequential_chunks(
                    frames,
                    frames_per_chunk=frames_per_chunk,
                    frame_stride=frame_stride,
                )
            )
        )
    if len(enumerated) == 0:
        print(f"[skip] no chunks in {scene_name}")
        return

    out_scene = out_root / split / scene_dir_name / "chunks"
    out_scene.mkdir(parents=True, exist_ok=True)

    for this_chunk_id, frames_chunk in enumerated:
        stems = [f"frame_{int(f['frame_index']):06d}" for f in frames_chunk]
        frame_ids = [int(f["frame_index"]) for f in frames_chunk]
        out_npz = out_scene / f"chunk_{this_chunk_id:06d}_{stems[0]}_{stems[-1]}.npz"

        if only_chunk_ids is not None and only_frame_groups is None and this_chunk_id not in only_chunk_ids:
            continue
        if skip_existing and out_npz.is_file():
            continue

        try:
            poses: list[np.ndarray] = [f["pose_c2w"] for f in frames_chunk]
            raw_poses: list[np.ndarray] = [f["raw_pose_c2w"] for f in frames_chunk]
            skip_chunk = False

            colors: list[np.ndarray] = []
            intrs: list[dict] = []
            paths_str: list[str] = [
                f"{source_file.resolve()}::{scene_name}::{stem}"
                for stem in stems
            ]

            for frame in frames_chunk:
                try:
                    rgb = decode_re10k_image(frame["image_bytes"])
                except Exception as exc:
                    if not skip_bad_chunks:
                        raise RuntimeError(f"Failed to decode RE10K image {scene_name} frame {frame['frame_index']}") from exc
                    skip_chunk = True
                    break
                orig_h, orig_w = rgb.shape[:2]
                rgb = cv2.resize(rgb, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
                sw = resize_w / max(orig_w, 1)
                sh = resize_h / max(orig_h, 1)
                intr = frame["intrinsic"]
                intr_u = {
                    "fx": intr["fx"] * sw,
                    "fy": intr["fy"] * sh,
                    "cx": intr["cx"] * sw,
                    "cy": intr["cy"] * sh,
                }
                colors.append(rgb)
                intrs.append(intr_u)
            if skip_chunk:
                continue

            poses_np = np.stack(poses, axis=0)
            intr_stack = np.stack([intr_dict_to_3x3(k) for k in intrs], axis=0)
            pred_depths, da3_maps = run_da3_depth_and_features(
                da3_model=da3_model,
                rgb_images_u8_320=colors,
                poses_c2w=poses_np,
                intrinsics_3x3_np=intr_stack,
                resize_hw=(resize_h, resize_w),
                feat_layer_id=da3_feat_layer,
                process_res=da3_process_res,
                process_res_method=da3_process_res_method,
                device=device,
            )

            raw_pred_depths = [np.asarray(d, dtype=np.float32).copy() for d in pred_depths]
            pred_depths = [sanitize_pred_depth(d, min_depth=min_depth, max_depth=max_depth) for d in pred_depths]
            dino_maps = dino_dense_batch(colors, norm, dino_model, device)
            vae_maps_t = vae_latent_dense_batch(colors, vae, device)

            partition_features: list[torch.Tensor] = []
            for vi, _rgb in enumerate(colors):
                dino_h = dino_maps[vi]
                da3_h = da3_maps[vi]
                vae_h = vae_maps_t[vi]
                fused = torch.cat([dino_h, da3_h, vae_h], dim=-1)
                partition_features.append(fused)

            try:
                (
                    sp_xyz_raw,
                    sp_idx,
                    _voxel_points_world,
                    voxel_coords,
                    sp_vae_features,
                    _prevoxel_points_world,
                    prevoxel_sp_idx,
                    sp_nearest_prevoxel_xyz,
                    sp_nearest_prevoxel_index,
                    _sp_nearest_prevoxel_dist,
                    sp_nearest_prevoxel_valid,
                    norm_factor,
                ) = fuse_and_superpoint_chunk(
                    pred_depths,
                    poses,
                    intrs,
                    partition_features,
                    vae_maps_t,
                    device,
                    voxel_size,
                    min_depth=min_depth,
                    max_depth=max_depth,
                )
            except RuntimeError as exc:
                if "No valid points in chunk" in str(exc):
                    print_chunk_diagnostics(
                        scene_name=scene_name,
                        out_name=out_npz.name,
                        stems=stems,
                        raw_depths=raw_pred_depths,
                        sanitized_depths=pred_depths,
                        poses_c2w=poses,
                        raw_poses_c2w=raw_poses,
                        intrs=intrs,
                        min_depth=min_depth,
                        max_depth=max_depth,
                    )
                raise

            n_raw = sp_xyz_raw.shape[0]
            geometry_raw = compute_superpoint_geometry_stats(
                points_world=_voxel_points_world,
                sp_idx=sp_idx,
                sp_xyz_raw=sp_xyz_raw,
                norm_factor=norm_factor,
                device=device,
            )
            valid_mask_raw = np.ones((n_raw,), dtype=bool)

            np.savez_compressed(
                out_npz,
                superpoint_xyz=sp_xyz_raw.astype(np.float32),
                superpoint_valid_mask=valid_mask_raw.astype(bool),
                superpoint_source_idx=np.arange(n_raw, dtype=np.int32),
                superpoint_vae_features=sp_vae_features.astype(np.float16),
                prevoxel_superpoint_idx=prevoxel_sp_idx.astype(np.int32),
                superpoint_nearest_prevoxel_xyz=sp_nearest_prevoxel_xyz.astype(np.float32),
                superpoint_nearest_prevoxel_index=sp_nearest_prevoxel_index.astype(np.int64),
                superpoint_nearest_prevoxel_valid=sp_nearest_prevoxel_valid.astype(bool),
                da3_depths=np.stack(pred_depths, axis=0).astype(np.float16),
                camera_poses_c2w=np.stack(poses, axis=0).astype(np.float32),
                camera_intrinsics=np.stack([intr_dict_to_3x3(k) for k in intrs], axis=0).astype(np.float32),
                **geometry_raw,
                norm_scale=np.float32(norm_factor),
                image_paths=np.array(paths_str, dtype=object),
                source_file=np.array(str(source_file.resolve())),
                scene_key=np.array(scene_name),
                timestamps=np.array([int(f["timestamp"]) for f in frames_chunk], dtype=np.int64),
                frame_indices=np.array(frame_ids, dtype=np.int32),
                chunk_base_index=np.int32(frame_ids[0]),
                num_superpoints_raw=np.int32(n_raw),
                num_superpoints_fixed=np.int32(n_raw),
                geometry_dim=np.int32(geometry_raw["sp_cov_eigvals"].shape[-1]),
                chunk_schema_version=np.array(CHUNK_SCHEMA_VERSION),
            )
            print(
                f"[{scene_name}] wrote {out_npz.name}  "
                f"(Sraw={n_raw}, "
                f"valid_geom={float(geometry_raw['sp_geometry_valid_mask'].mean()):.3f})"
            )
            clear_cuda_cache(device)
        except Exception as exc:
            clear_cuda_cache(device)
            if is_cuda_oom(exc) or not skip_bad_chunks:
                raise
            print(
                f"[warn] skip chunk {scene_name} {out_npz.name} "
                f"(base={frame_ids[0]}, frames={stems[0]}..{stems[-1]}): {type(exc).__name__}: {exc}"
            )
            continue

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chunk RE10K 3D superpoint geometry data using DA3 Base predicted depth.")
    p.add_argument("--data-root", type=Path, default=Path("/mnt/share/codes/kimi/ssf_dataset/RE10K/re10k"))
    p.add_argument("--split", type=str, default="train", choices=("train", "test"))
    p.add_argument("--out-root", type=Path, default=Path("/mnt/share/codes/kimi/ssf_dataset/RE10K/re10k_chunk_geometry_sp_da3base_vae"))
    p.add_argument("--resize", type=int, default=None, help="Deprecated square resize fallback. Prefer --resize-w/--resize-h.")
    p.add_argument("--resize-w", type=int, default=518)
    p.add_argument("--resize-h", type=int, default=378)
    p.add_argument("--da3-weights", type=Path, default=Path("/mnt/share/codes/kimi/ssf/DA3-BASE"))
    p.add_argument("--da3-feat-layer", type=int, default=11)
    p.add_argument("--da3-process-res", type=int, default=518)
    p.add_argument("--da3-process-res-method", type=str, default="upper_bound_resize")
    p.add_argument(
        "--vae-path",
        type=Path,
        default=Path("/mnt/share/codes/kimi/ssf/stable-diffusion-3.5-large"),
        help="Stable Diffusion model directory containing the VAE subfolder.",
    )
    p.add_argument("--dinov3-repo", type=Path, default=Path(__file__).resolve().parents[1] / "third_party" / "dinov3")
    p.add_argument(
        "--dinov3-weights",
        type=Path,
        default=Path("/mnt/share/codes/kimi/ssf/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
    )
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument(
        "--skip-bad-chunks",
        action="store_true",
        help="Skip chunks that fail with non-OOM errors. Leave disabled while debugging to get full tracebacks.",
    )
    p.add_argument("--frames-per-chunk", type=int, default=DEFAULT_FRAMES_PER_CHUNK)
    p.add_argument("--frame-stride", type=int, default=INNER_FRAME_STEP)
    p.add_argument("--voxel-size", type=float, default=DEFAULT_VOXEL_SIZE)
    p.add_argument("--min-depth", type=float, default=0.1)
    p.add_argument("--max-depth", type=float, default=50.0)
    p.add_argument(
        "--target-num-superpoints",
        type=int,
        default=TARGET_NUM_SUPERPOINTS,
        help="Deprecated compatibility option; chunks now store raw superpoint counts.",
    )
    args = p.parse_args()
    if args.resize is not None:
        args.resize_w = int(args.resize)
        args.resize_h = int(args.resize)
    if args.resize_w <= 0 or args.resize_h <= 0:
        raise ValueError(f"resize_w/resize_h must be positive, got {args.resize_w}x{args.resize_h}")
    if args.frames_per_chunk <= 0:
        raise ValueError(f"frames_per_chunk must be positive, got {args.frames_per_chunk}")
    if args.frame_stride <= 0:
        raise ValueError(f"frame_stride must be positive, got {args.frame_stride}")
    da3_config_path = args.da3_weights / "config.json"
    if da3_config_path.is_file():
        with da3_config_path.open("r", encoding="utf-8") as f:
            da3_config = json.load(f)
        da3_out_layers = da3_config.get("config", {}).get("net", {}).get("out_layers")
        if da3_out_layers and args.da3_feat_layer not in da3_out_layers:
            raise ValueError(
                f"--da3-feat-layer={args.da3_feat_layer} is not in "
                f"{args.da3_weights} out_layers={da3_out_layers}"
            )
    return args


def run_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")
    print(f"[worker {rank}] device={device}")

    base_model = torch.hub.load(
        str(args.dinov3_repo),
        "dinov3_vitl16",
        source="local",
        weights=str(args.dinov3_weights),
    )
    base_model.eval()
    dino_model = DinoFeatureExtractor(base_model).to(device).eval()

    da3_model = DepthAnything3.from_pretrained(str(args.da3_weights)).to(device).eval()
    vae = AutoencoderKL.from_pretrained(str(args.vae_path), subfolder="vae").to(device).eval()

    norm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    re10k_files = list(iter_re10k_files(args.data_root, split=args.split))
    worker_files = re10k_files[rank::world_size] if world_size > 1 else re10k_files
    print(f"[worker {rank}] files={len(worker_files)}")

    for source_file in worker_files:
        print(f"[worker {rank}] loading {source_file.name}")
        try:
            scenes = torch.load(source_file, map_location="cpu", weights_only=True)
        except TypeError:
            scenes = torch.load(source_file, map_location="cpu")
        for scene in scenes:
            print(f"[worker {rank}] processing {source_file.name}::{scene.get('key', '<missing-key>')}")
            process_scene(
                scene=scene,
                source_file=source_file,
                split=args.split,
                out_root=args.out_root,
                dino_model=dino_model,
                da3_model=da3_model,
                vae=vae,
                device=device,
                resize_w=args.resize_w,
                resize_h=args.resize_h,
                da3_feat_layer=args.da3_feat_layer,
                da3_process_res=args.da3_process_res,
                da3_process_res_method=args.da3_process_res_method,
                norm=norm,
                skip_existing=args.skip_existing,
                voxel_size=args.voxel_size,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                target_num_superpoints=args.target_num_superpoints,
                frames_per_chunk=args.frames_per_chunk,
                frame_stride=args.frame_stride,
                skip_bad_chunks=args.skip_bad_chunks,
            )


if __name__ == "__main__":
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    world_size = gpu_count if gpu_count > 0 else 1
    print(f"[init] gpus={gpu_count}, world_size={world_size}")

    if world_size > 1:
        mp.spawn(run_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run_worker(0, world_size, args)
