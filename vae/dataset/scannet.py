"""
PyTorch dataset for ScanNet chunk samples built from RGB + known pose using
`make_superpoint_chunk_da3depth_vae_scannet_strict.py`.

Compared with scannet_chunk.py, this dataset supports two chunk schemas.
Legacy feature-teacher chunks contain:
  - superpoint_xyz
  - superpoint_feature
  - da3_depths               (preferred supervision depth)
  - vae_feature_maps         (optional low-resolution VAE latent maps)
  - dino_dim / da3_dim / vae_dim (optional feature-dimension metadata)
  - image_paths
  - frame_indices            (optional)
  - chunk_base_index         (optional)
  - norm_scale               (optional but preferred)

The chunk does NOT need to store:
  - da3_feature_maps

Pixel-classification chunks contain no superpoint_feature.  Instead they store:
  - superpoint_xyz
  - superpoint_valid_mask
  - vae_feature_maps
  - sp_id_maps
  - sp_mask_valid

If raw depth files are available in `data_root/scene/depth`, they are loaded as
additional real-depth supervision only. If not, zero placeholders are returned.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

try:
    from .manifest_cache import (
        default_manifest_path,
        load_manifest,
        path_from_manifest,
        path_to_manifest,
        save_manifest,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from manifest_cache import (
        default_manifest_path,
        load_manifest,
        path_from_manifest,
        path_to_manifest,
        save_manifest,
    )


CHUNK_NUM_VIEWS = 4


GEOMETRY_FIELDS = (
    "sp_center_from_points",
    "sp_point_count",
    "sp_log_point_count",
    "sp_radius_mean",
    "sp_radius_std",
    "sp_radius_max",
    "sp_radius_p50",
    "sp_radius_p90",
    "sp_radius_p95",
    "sp_radius_mean_norm",
    "sp_radius_std_norm",
    "sp_radius_max_norm",
    "sp_radius_p50_norm",
    "sp_radius_p90_norm",
    "sp_radius_p95_norm",
    "sp_cov",
    "sp_cov_norm",
    "sp_cov_eigvals",
    "sp_cov_eigvals_norm",
    "sp_cov_eigvecs",
    "sp_normal",
    "sp_linearity",
    "sp_planarity",
    "sp_scattering",
    "sp_anisotropy",
    "sp_curvature",
    "sp_omnivariance",
    "sp_omnivariance_norm",
    "sp_eigenentropy",
    "sp_density_p90",
    "sp_density_p90_norm",
)


def _farthest_point_sample_np(points: np.ndarray, num_samples: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    points = points[np.isfinite(points).all(axis=-1)]
    if points.shape[0] == 0:
        return np.zeros((int(num_samples), 3), dtype=np.float32)
    if points.shape[0] < int(num_samples):
        extra = np.random.choice(points.shape[0], size=int(num_samples) - points.shape[0], replace=True)
        return np.concatenate([points, points[extra]], axis=0).astype(np.float32)
    if points.shape[0] == int(num_samples):
        return points.astype(np.float32)

    pts = points.astype(np.float32, copy=False)
    chosen = np.empty((int(num_samples),), dtype=np.int64)
    dists = np.full((pts.shape[0],), np.inf, dtype=np.float32)
    centroid = pts.mean(axis=0, keepdims=True)
    farthest = int(np.argmax(((pts - centroid) ** 2).sum(axis=-1)))
    for i in range(int(num_samples)):
        chosen[i] = farthest
        dist = ((pts - pts[farthest : farthest + 1]) ** 2).sum(axis=-1)
        dists = np.minimum(dists, dist)
        farthest = int(np.argmax(dists))
    return pts[chosen].astype(np.float32)


def _voxel_downsample_np(points: np.ndarray, voxel_size: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if voxel_size <= 0 or points.shape[0] == 0:
        return points
    coords = np.floor(points / max(float(voxel_size), 1e-6)).astype(np.int64)
    _, first_idx = np.unique(coords, axis=0, return_index=True)
    return points[np.sort(first_idx)]


def _pad_np_first_dim(arr: np.ndarray, target_n: int, fill_value=0) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.shape[0] == target_n:
        return arr
    if arr.shape[0] > target_n:
        return arr[:target_n]
    out = np.full((target_n,) + arr.shape[1:], fill_value, dtype=arr.dtype)
    out[: arr.shape[0]] = arr
    return out


def _resolve_scene_image_path(scene_dir: Path, stored_path: Path) -> Path:
    """Resolve chunk image_paths against the current ScanNet scene directory.

    Older chunks may store absolute paths from another data_root.  Use the
    stored path only to recover the frame filename/subdirectory, then read from
    scene_dir so RGB, pose, and depth come from the same relocated dataset.
    """
    candidates = [
        scene_dir / stored_path.parent.name / stored_path.name,
        scene_dir / "color" / stored_path.name,
        scene_dir / "rgb" / stored_path.name,
        scene_dir / stored_path.name,
    ]
    for p in candidates:
        if p.is_file():
            return p

    for subdir in ("color", "rgb"):
        root = scene_dir / subdir
        if not root.is_dir():
            continue
        for suffix in (stored_path.suffix, ".jpg", ".png", ".jpeg"):
            if not suffix:
                continue
            p = root / f"{stored_path.stem}{suffix}"
            if p.is_file():
                return p
    return candidates[0]


def _load_intrinsic(scene_dir: Path) -> Dict[str, float]:
    intr_path = scene_dir / "intrinsic" / "intrinsic_depth.txt"
    if not intr_path.is_file():
        raise FileNotFoundError(f"Missing intrinsic file: {intr_path}")
    k = np.loadtxt(intr_path, dtype=np.float32)
    if k.shape == (4, 4):
        k = k[:3, :3]
    if k.shape != (3, 3):
        raise ValueError(f"Unexpected intrinsic shape {k.shape} in {intr_path}")
    return {"fx": float(k[0, 0]), "fy": float(k[1, 1]), "cx": float(k[0, 2]), "cy": float(k[1, 2])}


def _iter_scene_dirs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    split_dirs = [d for d in root.iterdir() if d.is_dir() and d.name in ("train", "test", "val")]
    if split_dirs:
        scenes: List[Path] = []
        for sd in sorted(split_dirs):
            scenes.extend([x for x in sorted(sd.iterdir()) if x.is_dir() and x.name.startswith("scene")])
        return scenes
    return [x for x in sorted(root.iterdir()) if x.is_dir() and x.name.startswith("scene")]


def infer_chunk_feature_dims(superpoint_root: Path) -> Dict[str, int]:
    superpoint_root = Path(superpoint_root)
    if not superpoint_root.exists():
        raise FileNotFoundError(f"superpoint_root does not exist: {superpoint_root}")

    for chunk_npz in sorted(superpoint_root.glob("**/chunks/*.npz")):
        try:
            z = np.load(chunk_npz, allow_pickle=True)
            if "superpoint_feature" in z.files:
                sp_feat = z["superpoint_feature"]
                if sp_feat.ndim != 2 or sp_feat.shape[-1] <= 0:
                    continue
                sp_feat_dim = int(sp_feat.shape[-1])
            elif "superpoint_xyz" in z.files and (
                "vae_feature_maps" in z.files or "sp_geometry_valid_mask" in z.files
            ):
                sp_feat_dim = 0
            else:
                continue
            if "vae_dim" in z.files:
                vae_dim = int(z["vae_dim"])
            elif "vae_feature_maps" in z.files:
                vae_dim = int(z["vae_feature_maps"].shape[-1])
            elif "vae_feature_map" in z.files:
                vae_dim = int(z["vae_feature_map"].shape[-1])
            else:
                vae_dim = 16

            if sp_feat_dim == 0:
                dino_dim = int(z["dino_dim"]) if "dino_dim" in z.files else 1024
                da3_dim = int(z["da3_dim"]) if "da3_dim" in z.files else 1024
            elif "dino_dim" in z.files and "da3_dim" in z.files:
                dino_dim = int(z["dino_dim"])
                da3_dim = int(z["da3_dim"])
            else:
                remain = sp_feat_dim - vae_dim
                if remain <= 0 or remain % 2 != 0:
                    raise ValueError(
                        f"Cannot infer dino/da3 dims from sp_feat_dim={sp_feat_dim}, vae_dim={vae_dim} in {chunk_npz}"
                    )
                dino_dim = remain // 2
                da3_dim = remain // 2

            return {
                "sp_feat_dim": sp_feat_dim,
                "dino_dim": dino_dim,
                "da3_dim": da3_dim,
                "vae_dim": vae_dim,
                "encoder_img_feat_in_dim": da3_dim,
            }
        except Exception:
            continue
    raise RuntimeError(f"Failed to infer feature dims from chunk npz under: {superpoint_root}")


def depth_to_point_cloud(depth: np.ndarray, intr: dict, valid_mask: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    z = depth[valid_mask]
    x = (u[valid_mask] - intr["cx"]) * z / intr["fx"]
    y = (v[valid_mask] - intr["cy"]) * z / intr["fy"]
    return np.stack([x, y, z], axis=1).astype(np.float32)


def transform_points(points: np.ndarray, pose_c2w: np.ndarray) -> np.ndarray:
    points = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
    pose_c2w = np.nan_to_num(pose_c2w, nan=0.0, posinf=0.0, neginf=0.0)
    pts_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    out = (pose_c2w @ pts_h.T).T
    out = out[:, :3]
    finite = np.isfinite(out).all(axis=1)
    return out[finite]


class ScanNetSuperpointChunkDatasetV2C15(Dataset):
    """
    One sample = one chunk NPZ + four RGB frames (+ optional real-depth supervision).
    """

    def __init__(
        self,
        data_root: Path = Path("/data1/ssf_dataset/scannet_rgb_depth"),
        superpoint_root: Path = Path("/data2/ssf_dataset/scannet_superpoint_chunk_da3depth_vae_nopca"),
        split: str = "train",
        split_ratio: float = 0.9,
        depth_scale: float = 1000.0,
        resize_w: int = 320,
        resize_h: int = 320,
        image_transform: Optional[torch.nn.Module] = None,
        allow_missing_depth: bool = True,
        max_superpoints: Optional[int] = None,
        superpoint_pad_extra: int = 200,
        num_views: Optional[int] = None,
        num_fps_sp_xyz: int = 4096,
        fps_voxel_size: float = 0.0,
        use_manifest_cache: bool = True,
        rebuild_manifest: bool = False,
        manifest_path: Optional[Path] = None,
        eval_manifest_path: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.superpoint_root = Path(superpoint_root)
        self.split = split
        self.split_ratio = split_ratio
        self.depth_scale = depth_scale
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.image_transform = image_transform or T.ToTensor()
        self.allow_missing_depth = bool(allow_missing_depth)
        self.num_views = int(num_views) if num_views is not None else CHUNK_NUM_VIEWS
        self.max_superpoints_override = int(max_superpoints) if max_superpoints is not None else None
        self.superpoint_pad_extra = int(superpoint_pad_extra)
        self.num_fps_sp_xyz = int(num_fps_sp_xyz)
        self.fps_voxel_size = float(fps_voxel_size)
        self.use_manifest_cache = bool(use_manifest_cache)
        self.rebuild_manifest = bool(rebuild_manifest)
        self.manifest_path = Path(manifest_path) if manifest_path is not None else None
        self.eval_manifest_path = Path(eval_manifest_path) if eval_manifest_path is not None else None
        self.max_superpoints = self.max_superpoints_override or self.num_fps_sp_xyz
        self.items: List[Dict] = []
        self._build_index()

    def _build_index(self) -> None:
        manifest_root = self.superpoint_root
        manifest_path = self._manifest_path(manifest_root)
        if self.use_manifest_cache and not self.rebuild_manifest:
            cached = load_manifest(manifest_path, manifest_root, self.split, self.num_views, "scannet_chunkv2_c15")
            if cached is not None:
                if cached or self.split == "train":
                    self.items = [
                        {
                            "scene_dir": path_from_manifest(x["scene_dir"]),
                            "scene_name": str(x["scene_name"]),
                            "chunk_npz": path_from_manifest(x["chunk_npz"]),
                            "num_superpoints": int(x["num_superpoints"]),
                        }
                        for x in cached
                    ]
                    if self.max_superpoints_override is None:
                        self.max_superpoints = self.num_fps_sp_xyz
                    return

        scenes = _iter_scene_dirs(self.data_root)
        if not scenes:
            self.items = []
            return

        has_train_split = any(s.parent.name == "train" for s in scenes)
        has_eval_split = any(s.parent.name in ("test", "val") for s in scenes)
        has_explicit_split = has_train_split and has_eval_split
        if has_explicit_split:
            if self.split == "train":
                target = [s for s in scenes if s.parent.name == "train"]
            else:
                target = [s for s in scenes if s.parent.name in ("test", "val")]
        else:
            scenes = sorted(scenes, key=lambda x: x.name)
            split_idx = int(len(scenes) * self.split_ratio)
            target = scenes[:split_idx] if self.split == "train" else scenes[split_idx:]

        items: List[Dict] = []
        max_raw_superpoints = 0
        for scene_dir in target:
            chunks_dir = self.superpoint_root / scene_dir.name / "chunks"
            if not chunks_dir.is_dir():
                continue

            for chunk_npz in sorted(chunks_dir.glob("*.npz")):
                try:
                    z = np.load(chunk_npz, allow_pickle=True)
                    has_legacy = "superpoint_feature" in z.files
                    has_pixel_cls = (
                        "superpoint_xyz" in z.files
                        and "superpoint_valid_mask" in z.files
                        and "sp_id_maps" in z.files
                        and "sp_mask_valid" in z.files
                        and "vae_feature_maps" in z.files
                    )
                    has_geometry = (
                        "superpoint_xyz" in z.files
                        and "superpoint_valid_mask" in z.files
                        and "sp_geometry_valid_mask" in z.files
                    )
                    if (
                        not has_legacy
                        and not has_pixel_cls
                        and not has_geometry
                    ) or "superpoint_xyz" not in z.files or "image_paths" not in z.files:
                        continue
                    paths = z["image_paths"]
                    if hasattr(paths, "tolist"):
                        paths = paths.tolist()
                    if len(paths) != self.num_views:
                        continue
                    n_sp = int(z["superpoint_xyz"].shape[0])
                except Exception:
                    continue

                pose_ok = True
                for p in paths:
                    cp = Path(str(p))
                    pose_path = scene_dir / "pose" / f"{cp.stem}.txt"
                    if not pose_path.is_file():
                        pose_ok = False
                        break
                    try:
                        pose_probe = np.loadtxt(pose_path, dtype=np.float32)
                        if pose_probe.shape != (4, 4) or (not np.isfinite(pose_probe).all()):
                            pose_ok = False
                            break
                    except Exception:
                        pose_ok = False
                        break
                if not pose_ok:
                    continue

                items.append(
                    {
                        "scene_dir": scene_dir,
                        "scene_name": scene_dir.name,
                        "chunk_npz": chunk_npz,
                        "num_superpoints": n_sp,
                    }
                )
                max_raw_superpoints = max(max_raw_superpoints, n_sp)
        self.items = items
        if self.use_manifest_cache:
            save_manifest(
                manifest_path,
                manifest_root,
                self.split,
                self.num_views,
                "scannet_chunkv2_c15",
                [
                    {
                        "scene_dir": path_to_manifest(x["scene_dir"]),
                        "scene_name": x["scene_name"],
                        "chunk_npz": path_to_manifest(x["chunk_npz"]),
                        "num_superpoints": int(x["num_superpoints"]),
                    }
                    for x in items
                ],
            )
        if self.max_superpoints_override is None:
            self.max_superpoints = self.num_fps_sp_xyz

    def _manifest_path(self, root: Path) -> Path:
        if self.split == "train" and self.manifest_path is not None:
            return self.manifest_path
        if self.split != "train" and self.eval_manifest_path is not None:
            return self.eval_manifest_path
        return default_manifest_path(root, self.split, self.num_views, "scannet_chunkv2_c15")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        e = self.items[idx]
        scene_dir: Path = e["scene_dir"]
        scene_name = e["scene_name"]
        z = np.load(e["chunk_npz"], allow_pickle=True)

        sp_xyz = z["superpoint_xyz"].astype(np.float32)
        sp_xyz_raw = sp_xyz.copy()
        raw_num_superpoints = int(sp_xyz.shape[0])
        target_num_superpoints = int(self.max_superpoints or raw_num_superpoints)
        if "superpoint_feature" in z.files:
            sp_feat = z["superpoint_feature"].astype(np.float32)
        else:
            sp_feat = np.zeros((sp_xyz.shape[0], 0), dtype=np.float32)
        sp_xyz = _pad_np_first_dim(sp_xyz, target_num_superpoints, fill_value=0).astype(np.float32)
        sp_feat = _pad_np_first_dim(sp_feat, target_num_superpoints, fill_value=0).astype(np.float32)
        dino_dim = int(z["dino_dim"]) if "dino_dim" in z.files else None
        da3_dim = int(z["da3_dim"]) if "da3_dim" in z.files else None
        vae_dim = int(z["vae_dim"]) if "vae_dim" in z.files else None
        da3_depths = z["da3_depths"].astype(np.float32) if "da3_depths" in z.files else None
        vae_feat_mv = None
        if "vae_feature_maps" in z.files:
            vae_feat_mv = z["vae_feature_maps"].astype(np.float32)
        elif "vae_feature_map" in z.files:
            vae_feat_mv = z["vae_feature_map"].astype(np.float32)
        sp_vae = None
        if "sp_vae" in z.files:
            sp_vae = z["sp_vae"].astype(np.float32)
        elif "superpoint_vae_features" in z.files:
            sp_vae = z["superpoint_vae_features"].astype(np.float32)
        elif "superpoint_vae_feature" in z.files:
            sp_vae = z["superpoint_vae_feature"].astype(np.float32)
        elif "sp_vae_path" in z.files:
            sp_vae_path_raw = np.asarray(z["sp_vae_path"]).reshape(-1)[0].item()
            sp_vae_path = Path(str(sp_vae_path_raw))
            if not sp_vae_path.is_absolute():
                sp_vae_path = e["chunk_npz"].parent / sp_vae_path
            if sp_vae_path.is_file():
                sp_vae = np.load(sp_vae_path).astype(np.float32)
        if vae_dim is None and vae_feat_mv is not None:
            vae_dim = int(vae_feat_mv.shape[-1])
        if vae_dim is None and sp_vae is not None:
            vae_dim = int(sp_vae.shape[-1])
        saved_norm_factor = float(z["norm_scale"]) if "norm_scale" in z.files else None

        paths = z["image_paths"]
        if hasattr(paths, "tolist"):
            paths = paths.tolist()
        paths = [Path(str(p)) for p in paths]
        if len(paths) != self.num_views:
            raise RuntimeError(f"Chunk {e['chunk_npz']} has {len(paths)} views, expected {self.num_views}.")

        frame_indices = z["frame_indices"].astype(np.int64) if "frame_indices" in z.files else None
        if "chunk_base_index" in z.files:
            chunk_base = int(z["chunk_base_index"])
        elif frame_indices is not None and frame_indices.size > 0:
            chunk_base = int(frame_indices[0])
        else:
            chunk_base = int(paths[0].stem)

        intr = _load_intrinsic(scene_dir)
        imgs_hw3: List[np.ndarray] = []
        depths: List[np.ndarray] = []
        real_depths: List[np.ndarray] = []
        poses: List[np.ndarray] = []
        Ks: List[torch.Tensor] = []
        valid_masks: List[torch.Tensor] = []
        real_valid_masks: List[torch.Tensor] = []
        norm_depths: List[torch.Tensor] = []

        for view_idx, stored_color_path in enumerate(paths):
            color_path = _resolve_scene_image_path(scene_dir, stored_color_path)
            stem = stored_color_path.stem
            pose_path = scene_dir / "pose" / f"{stem}.txt"
            img = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(
                    f"Failed to read image resolved from chunk path {stored_color_path}: {color_path}"
                )
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            pose_mat = np.loadtxt(pose_path, dtype=np.float32)
            if pose_mat.shape != (4, 4) or (not np.isfinite(pose_mat).all()):
                raise RuntimeError(f"Invalid pose: {pose_path}")

            orig_h, orig_w = img.shape[:2]
            intr_used = intr.copy()
            if self.resize_w is not None and self.resize_h is not None:
                img = cv2.resize(img, (self.resize_w, self.resize_h), interpolation=cv2.INTER_LINEAR)
                sw = self.resize_w / float(orig_w)
                sh = self.resize_h / float(orig_h)
                intr_used["fx"] *= sw
                intr_used["fy"] *= sh
                intr_used["cx"] *= sw
                intr_used["cy"] *= sh

            depth_path = scene_dir / "depth" / f"{stem}.png"
            if depth_path.is_file():
                depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                if depth_raw is None:
                    raise RuntimeError(f"Failed to read depth: {depth_path}")
                real_depth = depth_raw.astype(np.float32) / float(self.depth_scale)
                if self.resize_w is not None and self.resize_h is not None:
                    real_depth = cv2.resize(real_depth, (self.resize_w, self.resize_h), interpolation=cv2.INTER_NEAREST)
                real_valid_mask = (real_depth > 0.1) & (real_depth < 10.0)
            else:
                if not self.allow_missing_depth:
                    raise RuntimeError(f"Missing depth: {depth_path}")
                real_depth = np.zeros((self.resize_h, self.resize_w), dtype=np.float32)
                real_valid_mask = np.zeros((self.resize_h, self.resize_w), dtype=bool)

            if da3_depths is not None:
                depth = da3_depths[view_idx].astype(np.float32)
                if self.resize_w is not None and self.resize_h is not None and depth.shape != (self.resize_h, self.resize_w):
                    depth = cv2.resize(depth, (self.resize_w, self.resize_h), interpolation=cv2.INTER_NEAREST)
                valid_mask = (depth > 0.1) & (depth < 10.0)
            else:
                if depth_path.is_file():
                    depth = real_depth.copy()
                    if self.resize_w is not None and self.resize_h is not None:
                        depth = cv2.resize(depth, (self.resize_w, self.resize_h), interpolation=cv2.INTER_NEAREST)
                    valid_mask = real_valid_mask.copy()
                else:
                    if not self.allow_missing_depth:
                        raise RuntimeError(f"Missing depth: {depth_path}")
                    depth = np.zeros((self.resize_h, self.resize_w), dtype=np.float32)
                    valid_mask = np.zeros((self.resize_h, self.resize_w), dtype=bool)

            imgs_hw3.append(img)
            depths.append(depth)
            real_depths.append(real_depth)
            poses.append(pose_mat)
            Ks.append(
                torch.tensor(
                    [
                        [intr_used["fx"], 0.0, intr_used["cx"]],
                        [0.0, intr_used["fy"], intr_used["cy"]],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=torch.float32,
                )
            )
            valid_masks.append(torch.from_numpy(valid_mask))
            real_valid_masks.append(torch.from_numpy(real_valid_mask))

        if saved_norm_factor is not None:
            norm_factor = saved_norm_factor
        else:
            sp_dist = np.linalg.norm(sp_xyz, ord=2, axis=-1)
            norm_factor = float(np.nanmean(sp_dist)) if sp_dist.size > 0 else 1.0
        if (not np.isfinite(norm_factor)) or norm_factor < 1e-6:
            norm_factor = 1.0

        norm_sp_center_xyz = sp_xyz_raw / norm_factor
        if "superpoint_nearest_prevoxel_xyz" in z.files:
            sp_anchor_xyz = z["superpoint_nearest_prevoxel_xyz"].astype(np.float32)
            if "superpoint_nearest_prevoxel_valid" in z.files:
                anchor_valid = z["superpoint_nearest_prevoxel_valid"].astype(bool)
            else:
                anchor_valid = np.isfinite(sp_anchor_xyz).all(axis=-1)
            anchor_valid = anchor_valid & np.isfinite(sp_anchor_xyz).all(axis=-1)
            sp_anchor_xyz = np.where(anchor_valid[:, None], sp_anchor_xyz, sp_xyz_raw)
            norm_sp_xyz = sp_anchor_xyz / norm_factor
        else:
            norm_sp_xyz = norm_sp_center_xyz
        norm_sp_center_xyz = _pad_np_first_dim(norm_sp_center_xyz, target_num_superpoints, fill_value=0).astype(np.float32)
        norm_sp_xyz = _pad_np_first_dim(norm_sp_xyz, target_num_superpoints, fill_value=0).astype(np.float32)
        img_tensors: List[torch.Tensor] = []
        norm_c2w_list: List[torch.Tensor] = []
        for pose_mat, img in zip(poses, imgs_hw3, strict=True):
            norm_c2w = pose_mat.copy()
            norm_c2w[:3, 3] = norm_c2w[:3, 3] / norm_factor
            norm_c2w_list.append(torch.from_numpy(norm_c2w).to(torch.float32))
            img_tensors.append(self.image_transform(img).permute(1, 2, 0).contiguous().to(torch.float32))

        for depth in depths:
            nd = np.clip(depth, 0.0, 100.0) / norm_factor
            norm_depths.append(torch.from_numpy(nd.astype(np.float32)))

        # C15 uses original superpoint centers as the FPS source.  Keep this
        # variable-length tensor in norm_pw so collate_points provides offsets,
        # and avoid dense per-pixel depth backprojection in the dataloader.
        norm_pw = (sp_xyz_raw / norm_factor).astype(np.float32)

        # C15 deliberately avoids CPU FPS in the dataloader.  The model forward
        # receives superpoint-center norm_pw after batch_to_device(), then uses
        # Pointcept pointops FPS on GPU to produce the actual token xyz.  This
        # placeholder only fixes the query count and provides a fallback.
        fps_sp_valid = np.ones((self.num_fps_sp_xyz,), dtype=bool)
        fps_sp_xyz = np.zeros((self.num_fps_sp_xyz, 3), dtype=np.float32)

        H, W = depths[0].shape
        out: Dict = {
            "seq": scene_name,
            "frame_id": int(frame_indices[0]) if frame_indices is not None and frame_indices.size > 0 else int(chunk_base),
            "chunk_stem": e["chunk_npz"].stem,
            "num_views": self.num_views,
            "image": torch.stack(img_tensors, dim=0),
            "superpoint_latent": torch.from_numpy(sp_feat).to(torch.float32),
            "pose": torch.stack([torch.from_numpy(p).to(torch.float32) for p in poses], dim=0),
            "sp_feat": torch.from_numpy(sp_feat).to(torch.float32),
            "Ks": torch.stack(Ks, dim=0),
            "image_size": (H, W),
            "valid_mask": torch.stack(valid_masks, dim=0),
            "norm_depth": torch.stack(norm_depths, dim=0),
            "real_depth": torch.stack([torch.from_numpy(d.astype(np.float32)) for d in real_depths], dim=0),
            "real_depth_valid_mask": torch.stack(real_valid_masks, dim=0),
            "norm_sp_xyz": torch.from_numpy(fps_sp_xyz).to(torch.float32),
            "norm_sp_center_xyz": torch.from_numpy(fps_sp_xyz).to(torch.float32),
            "source_superpoint_xyz": torch.from_numpy(norm_sp_xyz).to(torch.float32),
            "source_superpoint_center_xyz": torch.from_numpy(norm_sp_center_xyz).to(torch.float32),
            "norm_camtoworlds": torch.stack(norm_c2w_list, dim=0),
            "norm_scale": torch.tensor(norm_factor, dtype=torch.float32),
            "num_superpoints_raw": torch.tensor(raw_num_superpoints, dtype=torch.int32),
            "num_superpoints_fixed": torch.tensor(target_num_superpoints, dtype=torch.int32),
            "norm_pw": torch.from_numpy(norm_pw).to(torch.float32),
        }
        if "superpoint_valid_mask" in z.files:
            sp_valid = z["superpoint_valid_mask"].astype(bool)
        else:
            sp_valid = np.ones((raw_num_superpoints,), dtype=bool)
        sp_valid = _pad_np_first_dim(sp_valid, target_num_superpoints, fill_value=False).astype(bool)
        if raw_num_superpoints > target_num_superpoints:
            sp_valid[target_num_superpoints:] = False
        out["source_superpoint_valid_mask"] = torch.from_numpy(sp_valid)
        out["superpoint_valid_mask"] = torch.from_numpy(fps_sp_valid)
        if "superpoint_source_idx" in z.files:
            source_idx = z["superpoint_source_idx"].astype(np.int64)
            out["superpoint_source_idx"] = torch.from_numpy(
                _pad_np_first_dim(source_idx, target_num_superpoints, fill_value=-1).astype(np.int64)
            )
        if "sp_id_maps" in z.files:
            out["sp_id_maps"] = torch.from_numpy(z["sp_id_maps"].astype(np.int64))
        if "sp_mask_valid" in z.files:
            out["sp_mask_valid"] = torch.from_numpy(z["sp_mask_valid"].astype(bool))
        for field in GEOMETRY_FIELDS:
            if field in z.files:
                out[field] = torch.from_numpy(
                    _pad_np_first_dim(z[field].astype(np.float32), target_num_superpoints, fill_value=0).astype(np.float32)
                )
        if "sp_geometry_valid_mask" in z.files:
            geom_valid = z["sp_geometry_valid_mask"].astype(bool)
            geom_valid = _pad_np_first_dim(geom_valid, target_num_superpoints, fill_value=False).astype(bool)
            out["sp_geometry_valid_mask"] = torch.from_numpy(geom_valid & sp_valid)
        if dino_dim is not None:
            out["dino_dim"] = torch.tensor(dino_dim, dtype=torch.int32)
        if da3_dim is not None:
            out["da3_dim"] = torch.tensor(da3_dim, dtype=torch.int32)
        if vae_dim is not None:
            out["vae_dim"] = torch.tensor(vae_dim, dtype=torch.int32)
        if frame_indices is not None:
            out["frame_indices"] = torch.from_numpy(frame_indices.copy())
        if vae_feat_mv is not None and vae_feat_mv.shape[0] == self.num_views:
            out["vae_feature_map"] = torch.from_numpy(vae_feat_mv).to(torch.float32)
        if sp_vae is not None:
            sp_vae = _pad_np_first_dim(sp_vae, target_num_superpoints, fill_value=0).astype(np.float32)
            out["sp_vae"] = torch.from_numpy(sp_vae).to(torch.float32)
        return out


ScanNetSuperpointChunkDatasetV2 = ScanNetSuperpointChunkDatasetV2C15
