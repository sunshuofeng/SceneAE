"""
PyTorch dataset for RealEstate10K (RE10K) chunk samples, mirroring the C15
ScanNet reader (`scannet_chunkv2_c15.py`) so the model forward can consume both
without change.

Like the DL3DV reader, RE10K geometry chunks are fully self-contained: camera
poses, intrinsics (on the da3_depths grid), DA3-predicted depth, superpoint
geometry and VAE latents all live inside the chunk NPZ.  There is NO real sensor
depth, so `real_depth` is filled with the DA3 depth.

The only difference from DL3DV is RGB access: RE10K frames are packed inside
`.torch` archives, and `image_paths` encode their location as
    "<file>.torch::<scene_key>::frame_<index>"
where <index> is the position of the frame in that scene's image list.  A small
one-slot cache avoids reloading the same archive across a chunk's views.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
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


CHUNK_NUM_VIEWS = 9


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


def _pad_np_first_dim(arr: np.ndarray, target_n: int, fill_value=0) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.shape[0] == target_n:
        return arr
    if arr.shape[0] > target_n:
        return arr[:target_n]
    out = np.full((target_n,) + arr.shape[1:], fill_value, dtype=arr.dtype)
    out[: arr.shape[0]] = arr
    return out


def infer_chunk_feature_dims(superpoint_root: Path) -> Dict[str, int]:
    """Infer feature dims from geometry chunks (no legacy superpoint_feature).

    RE10K geometry chunks carry only VAE latents per superpoint, so sp_feat_dim is
    0 and dino/da3 dims fall back to their defaults, matching the C15 behaviour.
    """
    superpoint_root = Path(superpoint_root)
    if not superpoint_root.exists():
        raise FileNotFoundError(f"superpoint_root does not exist: {superpoint_root}")

    for chunk_npz in sorted(superpoint_root.glob("**/chunks/*.npz")):
        try:
            z = np.load(chunk_npz, allow_pickle=True)
            if "superpoint_xyz" not in z.files:
                continue
            if "vae_dim" in z.files:
                vae_dim = int(z["vae_dim"])
            elif "superpoint_vae_features" in z.files:
                vae_dim = int(z["superpoint_vae_features"].shape[-1])
            else:
                vae_dim = 16
            dino_dim = int(z["dino_dim"]) if "dino_dim" in z.files else 1024
            da3_dim = int(z["da3_dim"]) if "da3_dim" in z.files else 1024
            return {
                "sp_feat_dim": 0,
                "dino_dim": dino_dim,
                "da3_dim": da3_dim,
                "vae_dim": vae_dim,
                "encoder_img_feat_in_dim": da3_dim,
            }
        except Exception:
            continue
    raise RuntimeError(f"Failed to infer feature dims from chunk npz under: {superpoint_root}")


class RE10KSuperpointChunkDatasetV2C15(Dataset):
    """One sample = one RE10K chunk NPZ + its RGB frames (DA3 depth supervision)."""

    def __init__(
        self,
        data_root: Path = Path("/mnt/share/codes/kimi/ssf_dataset/RE10K/re10k"),
        superpoint_root: Path = Path(
            "/mnt/share/codes/kimi/ssf_dataset/RE10K/re10k_chunk_geometry_sp_da3base_vae"
        ),
        split: str = "train",
        split_ratio: float = 0.9,
        resize_w: int = 320,
        resize_h: int = 320,
        image_transform: Optional[torch.nn.Module] = None,
        depth_min: Optional[float] = None,
        depth_max: Optional[float] = None,
        depth_near: float = 1e-3,
        depth_far: Optional[float] = None,
        norm_depth_clip: float = 100.0,
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
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.image_transform = image_transform or T.ToTensor()
        self.depth_near = float(depth_min) if depth_min is not None else float(depth_near)
        self.depth_far = float(depth_max) if depth_max is not None else (float(depth_far) if depth_far is not None else None)
        self.norm_depth_clip = float(norm_depth_clip)
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
        # One-slot cache: {scene_key: images_list} for the most recently loaded archive.
        self._archive_cache: Optional[Tuple[str, Dict[str, list]]] = None
        self.items: List[Dict] = []
        self._build_index()

    def _split_root(self) -> Path:
        subdir = "train" if self.split == "train" else "test"
        return self.superpoint_root / subdir

    def _build_index(self) -> None:
        required = ("superpoint_xyz", "image_paths", "da3_depths", "camera_poses_c2w", "camera_intrinsics")
        root = self._split_root()
        items: List[Dict] = []
        if not root.exists():
            self.items = items
            return
        manifest_path = self._manifest_path(root)
        if self.use_manifest_cache and not self.rebuild_manifest:
            cached = load_manifest(manifest_path, root, self.split, self.num_views, "re10k_chunkv2_c15")
            if cached is not None:
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
        for scene_dir in sorted(root.iterdir()):
            chunks_dir = scene_dir / "chunks"
            if not chunks_dir.is_dir():
                continue
            for chunk_npz in sorted(chunks_dir.glob("*.npz")):
                try:
                    z = np.load(chunk_npz, allow_pickle=True)
                    if any(k not in z.files for k in required):
                        continue
                    paths = z["image_paths"]
                    if hasattr(paths, "tolist"):
                        paths = paths.tolist()
                    if len(paths) != self.num_views:
                        continue
                    n_sp = int(z["superpoint_xyz"].shape[0])
                except Exception:
                    continue
                items.append(
                    {
                        "scene_dir": scene_dir,
                        "scene_name": scene_dir.name,
                        "chunk_npz": chunk_npz,
                        "num_superpoints": n_sp,
                    }
                )
        self.items = items
        if self.use_manifest_cache:
            save_manifest(
                manifest_path,
                root,
                self.split,
                self.num_views,
                "re10k_chunkv2_c15",
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
        return default_manifest_path(root, self.split, self.num_views, "re10k_chunkv2_c15")

    def _archive_images(self, torch_path: str) -> Dict[str, list]:
        """Return {scene_key: encoded-image list} for a `.torch` archive (cached)."""
        if self._archive_cache is not None and self._archive_cache[0] == torch_path:
            return self._archive_cache[1]
        obj = torch.load(torch_path, map_location="cpu", weights_only=False)
        key_to_images = {str(d["key"]): d["images"] for d in obj}
        self._archive_cache = (torch_path, key_to_images)
        return key_to_images

    def _read_view_image(self, stored_path: str) -> np.ndarray:
        """Decode one RGB frame (HxWx3, RGB) from a `file.torch::key::frame_idx` ref."""
        torch_path, scene_key, frame_tag = str(stored_path).rsplit("::", 2)
        frame_idx = int(frame_tag.split("_")[-1])
        images = self._archive_images(torch_path)
        if scene_key not in images:
            raise RuntimeError(f"Scene key {scene_key} not found in archive {torch_path}")
        encoded = images[scene_key][frame_idx]
        arr = np.frombuffer(encoded.numpy().tobytes(), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to decode image {scene_key}:{frame_idx} from {torch_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        e = self.items[idx]
        scene_name = e["scene_name"]
        z = np.load(e["chunk_npz"], allow_pickle=True)

        sp_xyz = z["superpoint_xyz"].astype(np.float32)
        sp_xyz_raw = sp_xyz.copy()
        raw_num_superpoints = int(sp_xyz.shape[0])
        target_num_superpoints = int(self.max_superpoints or raw_num_superpoints)
        sp_feat = np.zeros((sp_xyz.shape[0], 0), dtype=np.float32)
        sp_xyz = _pad_np_first_dim(sp_xyz, target_num_superpoints, fill_value=0).astype(np.float32)
        sp_feat = _pad_np_first_dim(sp_feat, target_num_superpoints, fill_value=0).astype(np.float32)

        da3_depths = z["da3_depths"].astype(np.float32)
        camera_poses_c2w = z["camera_poses_c2w"].astype(np.float32)
        camera_intrinsics = z["camera_intrinsics"].astype(np.float32)

        sp_vae = None
        if "superpoint_vae_features" in z.files:
            sp_vae = z["superpoint_vae_features"].astype(np.float32)
        elif "sp_vae" in z.files:
            sp_vae = z["sp_vae"].astype(np.float32)
        vae_dim = int(z["vae_dim"]) if "vae_dim" in z.files else (int(sp_vae.shape[-1]) if sp_vae is not None else None)
        saved_norm_factor = float(z["norm_scale"]) if "norm_scale" in z.files else None

        paths = z["image_paths"]
        if hasattr(paths, "tolist"):
            paths = paths.tolist()
        if len(paths) != self.num_views:
            raise RuntimeError(f"Chunk {e['chunk_npz']} has {len(paths)} views, expected {self.num_views}.")

        frame_indices = z["frame_indices"].astype(np.int64) if "frame_indices" in z.files else None
        if "chunk_base_index" in z.files:
            chunk_base = int(z["chunk_base_index"])
        elif frame_indices is not None and frame_indices.size > 0:
            chunk_base = int(frame_indices[0])
        else:
            chunk_base = 0

        imgs_hw3: List[np.ndarray] = []
        depths: List[np.ndarray] = []
        real_depths: List[np.ndarray] = []
        poses: List[np.ndarray] = []
        Ks: List[torch.Tensor] = []
        valid_masks: List[torch.Tensor] = []
        real_valid_masks: List[torch.Tensor] = []
        norm_depths: List[torch.Tensor] = []

        for view_idx, stored_path in enumerate(paths):
            img = self._read_view_image(stored_path)

            depth = da3_depths[view_idx].astype(np.float32)
            h_da3, w_da3 = depth.shape
            target_w = int(self.resize_w) if self.resize_w is not None else w_da3
            target_h = int(self.resize_h) if self.resize_h is not None else h_da3

            if img.shape[:2] != (target_h, target_w):
                img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            if depth.shape != (target_h, target_w):
                depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

            k0 = camera_intrinsics[view_idx]
            sw = target_w / float(w_da3)
            sh = target_h / float(h_da3)
            fx, fy = float(k0[0, 0]) * sw, float(k0[1, 1]) * sh
            cx, cy = float(k0[0, 2]) * sw, float(k0[1, 2]) * sh

            valid_mask = np.isfinite(depth) & (depth > self.depth_near)
            if self.depth_far is not None:
                valid_mask &= depth < self.depth_far
            # No real sensor depth for RE10K: fill real depth with the DA3 depth.
            real_depth = depth.copy()
            real_valid_mask = valid_mask.copy()

            pose_mat = camera_poses_c2w[view_idx].astype(np.float32)
            if pose_mat.shape != (4, 4) or (not np.isfinite(pose_mat).all()):
                raise RuntimeError(f"Invalid pose in chunk {e['chunk_npz']} view {view_idx}")

            imgs_hw3.append(img)
            depths.append(depth)
            real_depths.append(real_depth)
            poses.append(pose_mat)
            Ks.append(
                torch.tensor(
                    [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32
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
            nd = np.clip(depth, 0.0, self.norm_depth_clip) / norm_factor
            norm_depths.append(torch.from_numpy(nd.astype(np.float32)))

        # Same contract as C15: norm_pw carries variable-length superpoint centers
        # for collate_points offsets; the model runs FPS on GPU after batching, so
        # the dataloader only fixes the query count with a placeholder.
        norm_pw = (sp_xyz_raw / norm_factor).astype(np.float32)
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
        for field in GEOMETRY_FIELDS:
            if field in z.files:
                out[field] = torch.from_numpy(
                    _pad_np_first_dim(z[field].astype(np.float32), target_num_superpoints, fill_value=0).astype(np.float32)
                )
        if "sp_geometry_valid_mask" in z.files:
            geom_valid = z["sp_geometry_valid_mask"].astype(bool)
            geom_valid = _pad_np_first_dim(geom_valid, target_num_superpoints, fill_value=False).astype(bool)
            out["sp_geometry_valid_mask"] = torch.from_numpy(geom_valid & sp_valid)
        if vae_dim is not None:
            out["vae_dim"] = torch.tensor(vae_dim, dtype=torch.int32)
        if frame_indices is not None:
            out["frame_indices"] = torch.from_numpy(frame_indices.copy())
        if sp_vae is not None:
            sp_vae = _pad_np_first_dim(sp_vae, target_num_superpoints, fill_value=0).astype(np.float32)
            out["sp_vae"] = torch.from_numpy(sp_vae).to(torch.float32)
        return out
