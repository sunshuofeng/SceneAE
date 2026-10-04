"""
PyTorch dataset for the first-frame -> subsequent-frames generation setup used by
the DiT generative model.

The chunk format mirrors the decoder's C15 chunks (DL3DV/RE10K style, self-
contained: camera poses, intrinsics on the da3_depths grid, and DA3-predicted
depth all embedded in the NPZ), but each chunk additionally stores TWO superpoint
groups built at different times:

  - ``first_*``  : superpoints built from the FIRST frame only  (the CONDITION)
  - ``rest_*``   : superpoints built from all SUBSEQUENT frames  (the TARGET)

The DiT is trained to generate the ``rest_`` (target) superpoints given the
``first_`` (condition) superpoints.  Both groups are homogeneous: each superpoint
carries a center ``xyz``, a VAE latent, and the geometry descriptor fields.

NOTE: the concrete dataset is not built yet, so the root paths below are
PLACEHOLDERS.  The chunk key schema documented here is the contract the data
builder should follow; adjust the ``_PLACEHOLDER_*`` roots once data exists.

Per-chunk NPZ keys (V = frames per chunk, frame 0 is the condition frame)
------------------------------------------------------------------------
RGBD / camera (all V frames):
  image_paths          (V,)          object   absolute RGB paths on disk
  da3_depths           (V, Hda3, Wda3) float   DA3-predicted depth (only depth)
  camera_poses_c2w     (V, 4, 4)     float32  camera-to-world
  camera_intrinsics    (V, 3, 3)     float32  intrinsics on the da3_depths grid
  norm_scale           ()            float32  shared metric normalizer
  frame_indices        (V,)          int      (optional)
  chunk_base_index     ()            int      (optional)

Condition group (prefix ``first_``) and target group (prefix ``rest_``), same schema:
  {p}superpoint_xyz            (N, 3)   float32  superpoint centers (world)
  {p}superpoint_vae_features   (N, 16)  float    VAE latent per superpoint
  {p}superpoint_z_sp           (N, 64)  float    cached SuperpointVAE encoder latent
  {p}superpoint_valid_mask     (N,)     bool     (optional; default all True)
  {p}sp_geometry_valid_mask    (N,)     bool     (optional)
  {p}<geometry field>          (N, ...) float32  see GEOMETRY_FIELDS (optional)
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset


# Frame 0 of every chunk is the condition frame; frames 1..V-1 are the target frames.
COND_FRAME_INDEX = 0

# Placeholder roots -- replace once the generation dataset is built.
_PLACEHOLDER_DATA_ROOT = Path("/mnt/share/codes/kimi/ssf_dataset/GEN/first_frame_gen")
_PLACEHOLDER_SUPERPOINT_ROOT = Path("/mnt/share/codes/kimi/ssf_dataset/GEN/first_frame_gen_chunk")


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

GEOMETRY_FIELD_SHAPES = {
    "sp_center_from_points": (3,),
    "sp_point_count": (),
    "sp_log_point_count": (),
    "sp_radius_mean": (),
    "sp_radius_std": (),
    "sp_radius_max": (),
    "sp_radius_p50": (),
    "sp_radius_p90": (),
    "sp_radius_p95": (),
    "sp_radius_mean_norm": (),
    "sp_radius_std_norm": (),
    "sp_radius_max_norm": (),
    "sp_radius_p50_norm": (),
    "sp_radius_p90_norm": (),
    "sp_radius_p95_norm": (),
    "sp_cov": (3, 3),
    "sp_cov_norm": (3, 3),
    "sp_cov_eigvals": (3,),
    "sp_cov_eigvals_norm": (3,),
    "sp_cov_eigvecs": (3, 3),
    "sp_normal": (3,),
    "sp_linearity": (),
    "sp_planarity": (),
    "sp_scattering": (),
    "sp_anisotropy": (),
    "sp_curvature": (),
    "sp_omnivariance": (),
    "sp_omnivariance_norm": (),
    "sp_eigenentropy": (),
    "sp_density_p90": (),
    "sp_density_p90_norm": (),
}

V1_GEOMETRY_FIELDS = (
    "sp_radius_mean_norm",
    "sp_radius_std_norm",
    "sp_radius_max_norm",
    "sp_radius_p50_norm",
    "sp_radius_p90_norm",
    "sp_radius_p95_norm",
    "sp_cov_eigvals_norm",
    "sp_normal",
    "sp_linearity",
    "sp_planarity",
    "sp_scattering",
    "sp_anisotropy",
    "sp_curvature",
    "sp_omnivariance_norm",
    "sp_eigenentropy",
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


def _iter_scene_dirs(root: Path) -> List[Path]:
    """Scene chunk directories under ``root``.

    Supports both a flat layout (``root/<scene>/chunks``) and a split layout
    (``root/{train,test,val}/<scene>/chunks``).
    """
    if not root.exists():
        return []
    split_dirs = [d for d in sorted(root.iterdir()) if d.is_dir() and d.name in ("train", "test", "val")]
    roots = split_dirs if split_dirs else [root]
    scenes: List[Path] = []
    for r in roots:
        scenes.extend([d for d in sorted(r.iterdir()) if d.is_dir() and (d / "chunks").is_dir()])
    return scenes


class FirstFrameGenSuperpointChunkDataset(Dataset):
    """One sample = one chunk: all-frames RGBD + first-frame (cond) and
    subsequent-frames (target) superpoint groups."""

    def __init__(
        self,
        data_root: Path = _PLACEHOLDER_DATA_ROOT,
        superpoint_root: Path = _PLACEHOLDER_SUPERPOINT_ROOT,
        split: str = "train",
        split_ratio: float = 0.9,
        source_split: Optional[str] = None,
        resize_w: int = 518,
        resize_h: int = 378,
        image_transform: Optional[torch.nn.Module] = None,
        depth_min: Optional[float] = None,
        depth_max: Optional[float] = None,
        depth_near: float = 1e-3,
        depth_far: Optional[float] = None,
        norm_depth_clip: float = 100.0,
        allow_missing_depth: bool = True,
        num_views: Optional[int] = None,
        cond_max_superpoints: int = 3000,
        target_max_superpoints: int = 3000,
        include_geometry: bool = True,
        zsp_stats_path: Optional[Path] = None,
        zsp_sidecar: bool = True,
        zsp_cache_root: Optional[Path] = None,
        stage1_xyz_cache_root: Optional[Path] = None,
        stage1_xyz_required: bool = True,
        target_perm_cache_root: Optional[Path] = None,
        target_perm_required: bool = False,
        return_v1: bool = False,
        return_v2: bool = False,
        load_images: bool = True,
        max_views: Optional[int] = None,
        use_index_cache: bool = False,
        rebuild_index_cache: bool = False,
        index_cache_path: Optional[Path] = None,
        index_cache_dir: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.superpoint_root = Path(superpoint_root)
        self.split = split
        self.split_ratio = split_ratio
        self.source_split = "all" if source_split is None else str(source_split).lower()
        if self.source_split not in ("all", "train", "val"):
            raise ValueError(f"source_split must be one of all/train/val, got {source_split!r}")
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.image_transform = image_transform or T.ToTensor()
        self.depth_near = float(depth_min) if depth_min is not None else float(depth_near)
        self.depth_far = float(depth_max) if depth_max is not None else (float(depth_far) if depth_far is not None else None)
        self.norm_depth_clip = float(norm_depth_clip)
        self.allow_missing_depth = bool(allow_missing_depth)
        self.num_views = int(num_views) if num_views is not None else None
        self.cond_max_superpoints = int(cond_max_superpoints)
        self.target_max_superpoints = int(target_max_superpoints)
        self.include_geometry = bool(include_geometry)
        self.zsp_stats_path = Path(zsp_stats_path) if zsp_stats_path is not None else None
        self.zsp_sidecar = bool(zsp_sidecar)
        self.zsp_cache_root = Path(zsp_cache_root) if zsp_cache_root is not None else None
        self.stage1_xyz_cache_root = Path(stage1_xyz_cache_root) if stage1_xyz_cache_root is not None else None
        self.stage1_xyz_required = bool(stage1_xyz_required)
        self.target_perm_cache_root = Path(target_perm_cache_root) if target_perm_cache_root is not None else None
        self.target_perm_required = bool(target_perm_required)
        self.return_v1 = bool(return_v1)
        self.return_v2 = bool(return_v2)
        self.load_images = bool(load_images)
        self.max_views = int(max_views) if max_views is not None else None
        self.use_index_cache = bool(use_index_cache)
        self.rebuild_index_cache = bool(rebuild_index_cache)
        self.index_cache_path = Path(index_cache_path) if index_cache_path is not None else None
        self.index_cache_dir = Path(index_cache_dir) if index_cache_dir is not None else None
        self.zsp_mean = np.zeros((64,), dtype=np.float32)
        self.zsp_std = np.ones((64,), dtype=np.float32)
        self.xyz_mean = np.zeros((3,), dtype=np.float32)
        self.xyz_std = np.ones((3,), dtype=np.float32)
        if self.zsp_stats_path is not None and self.zsp_stats_path.is_file():
            s = np.load(self.zsp_stats_path)
            self.zsp_mean = s["zsp_mean"].astype(np.float32)
            self.zsp_std = np.maximum(s["zsp_std"].astype(np.float32), 1e-6)
            self.xyz_mean = s["xyz_mean"].astype(np.float32)
            self.xyz_std = np.maximum(s["xyz_std"].astype(np.float32), 1e-6)
        self.items: List[Dict] = []
        self._re10k_files: OrderedDict[Path, List[Dict]] = OrderedDict()
        self._build_index()

    def _scenes_for_split(self) -> List[Path]:
        scenes = _iter_scene_dirs(self.superpoint_root)
        if not scenes:
            return []
        has_split = any(s.parent != self.superpoint_root and s.parent.name in ("train", "test", "val") for s in scenes)
        if has_split:
            if self.split == "train":
                return [s for s in scenes if s.parent.name == "train"]
            return [s for s in scenes if s.parent.name in ("test", "val")]
        split_idx = int(len(scenes) * self.split_ratio)
        return scenes[:split_idx] if self.split == "train" else scenes[split_idx:]

    def _default_index_cache_path(self) -> Path:
        if self.index_cache_path is not None:
            return self.index_cache_path
        root_id = hashlib.sha1(str(self.superpoint_root.resolve()).encode("utf-8")).hexdigest()[:12]
        views = "all" if self.num_views is None else str(self.num_views)
        name = f"index_cache_{root_id}_{self.split}_ratio{self.split_ratio:g}_views{views}.json"
        if self.source_split != "all":
            name = name.replace(".json", f"_source{self.source_split}.json")
        if self.index_cache_dir is not None:
            return self.index_cache_dir / name
        return self.superpoint_root / name

    def _index_cache_meta(self) -> dict:
        return {
            "schema_version": 3,
            "superpoint_root": str(self.superpoint_root),
            "zsp_cache_root": str(self.zsp_cache_root) if self.zsp_cache_root is not None else None,
            "zsp_sidecar": bool(self.zsp_sidecar),
            "return_v1": bool(self.return_v1),
            "return_v2": bool(self.return_v2),
            "root_name": self.superpoint_root.name,
            "root_is_split_dir": self.superpoint_root.name in ("train", "test", "val"),
            "split": str(self.split),
            "split_ratio": float(self.split_ratio),
            "source_split": self.source_split,
            "num_views": self.num_views,
            "max_views": self.max_views,
        }

    def _load_index_cache(self, path: Path) -> bool:
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False
        if payload.get("meta") != self._index_cache_meta():
            return False
        items = payload.get("items", None)
        if not isinstance(items, list):
            return False
        loaded = []
        for item in items:
            try:
                chunk_npz = Path(item["chunk_npz"])
                scene_dir = Path(item["scene_dir"])
                scene_name = str(item["scene_name"])
            except (KeyError, TypeError):
                return False
            loaded.append({"scene_dir": scene_dir, "scene_name": scene_name, "chunk_npz": chunk_npz})
        self.items = loaded
        return True

    def _save_index_cache(self, path: Path, items: List[Dict]) -> None:
        payload = {
            "meta": self._index_cache_meta(),
            "count": len(items),
            "items": [
                {
                    "scene_dir": str(item["scene_dir"]),
                    "scene_name": str(item["scene_name"]),
                    "chunk_npz": str(item["chunk_npz"]),
                }
                for item in items
            ],
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp, path)
        except OSError:
            try:
                if "tmp" in locals() and tmp.exists():
                    tmp.unlink()
            except OSError:
                pass

    def _chunk_has_required_zsp(self, z, chunk_npz: Path) -> bool:
        if not (self.return_v1 or self.return_v2):
            return True
        required_zsp = ("first_superpoint_z_sp", "rest_superpoint_z_sp")
        if all(k in z.files for k in required_zsp):
            return True
        if not self.zsp_sidecar:
            return False
        sidecar = self._sidecar_path(chunk_npz)
        if not sidecar.is_file():
            return False
        try:
            with np.load(sidecar, allow_pickle=True) as zsp:
                return all((k in z.files) or (k in zsp.files) for k in required_zsp)
        except Exception:
            return False

    def _build_index(self) -> None:
        cache_path = self._default_index_cache_path()
        if self.use_index_cache and not self.rebuild_index_cache and cache_path.is_file():
            if self._load_index_cache(cache_path):
                return

        required = (
            "image_paths",
            "da3_depths",
            "camera_poses_c2w",
            "camera_intrinsics",
            "first_superpoint_xyz",
            "rest_superpoint_xyz",
        )
        items: List[Dict] = []
        for scene_dir in self._scenes_for_split():
            for chunk_npz in sorted((scene_dir / "chunks").glob("*.npz")):
                try:
                    with np.load(chunk_npz, allow_pickle=True) as z:
                        if any(k not in z.files for k in required):
                            continue
                        if not self._chunk_has_required_zsp(z, chunk_npz):
                            continue
                        paths = z["image_paths"]
                        if hasattr(paths, "tolist"):
                            paths = paths.tolist()
                        if self.source_split != "all" and self._source_split_from_paths(paths) != self.source_split:
                            continue
                        if self.num_views is not None and len(paths) != self.num_views:
                            continue
                except Exception:
                    continue
                items.append({"scene_dir": scene_dir, "scene_name": scene_dir.name, "chunk_npz": chunk_npz})
        self.items = items
        if self.use_index_cache:
            self._save_index_cache(cache_path, items)

    @staticmethod
    def _source_split_from_paths(paths) -> str:
        paths = paths if isinstance(paths, list) else [paths]
        detected = set()
        for value in paths:
            parts = {part.lower() for part in str(value).replace("\\", "/").split("/")}
            if parts.intersection({"train", "training"}):
                detected.add("train")
            if parts.intersection({"test", "testing", "val", "validation", "eval"}):
                detected.add("val")
        if len(detected) > 1:
            raise ValueError(f"Chunk mixes train and validation image paths: {paths[:2]}")
        return next(iter(detected), "unknown")

    def _read_view_image(self, stored_path: str) -> np.ndarray:
        """Read one RGB frame (HxWx3, RGB) from an absolute image path on disk."""
        path_str = str(stored_path)
        if "::" in path_str:
            return self._read_re10k_image(path_str)
        color_path = Path(path_str)
        img = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {color_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _read_re10k_image(self, spec: str) -> np.ndarray:
        source, scene_key, frame_name = spec.split("::", 2)
        source_path = Path(source)
        if source_path not in self._re10k_files:
            try:
                self._re10k_files[source_path] = torch.load(source_path, map_location="cpu", weights_only=True)
            except TypeError:
                self._re10k_files[source_path] = torch.load(source_path, map_location="cpu")
            while len(self._re10k_files) > 2:
                self._re10k_files.popitem(last=False)
        else:
            self._re10k_files.move_to_end(source_path)
        frame_idx = int(frame_name.rsplit("_", 1)[-1])
        for scene in self._re10k_files[source_path]:
            if str(scene.get("key", source_path.stem)) != scene_key:
                continue
            image_bytes = scene["images"][frame_idx]
            return np.asarray(Image.open(BytesIO(image_bytes.numpy().tobytes())).convert("RGB"), dtype=np.uint8).copy()
        raise KeyError(f"RE10K scene {scene_key} not found in {source_path}")

    def _sidecar_path(self, chunk_npz: Path) -> Path:
        if self.zsp_cache_root is None:
            return chunk_npz.with_suffix(".zsp.npz")
        try:
            rel = chunk_npz.relative_to(self.superpoint_root)
        except ValueError:
            rel = Path(chunk_npz.name)
        return self.zsp_cache_root / rel.with_suffix(".zsp.npz")

    def _stage1_xyz_path(self, chunk_npz: Path) -> Path:
        if self.stage1_xyz_cache_root is None:
            return chunk_npz.with_suffix(".s1xyz.npz")
        try:
            rel = chunk_npz.relative_to(self.superpoint_root)
        except ValueError:
            rel = Path(chunk_npz.name)
        return self.stage1_xyz_cache_root / rel.with_suffix(".s1xyz.npz")

    def _target_perm_path(self, chunk_npz: Path) -> Path:
        if self.target_perm_cache_root is None:
            return chunk_npz.with_suffix(".s1perm.npz")
        try:
            rel = chunk_npz.relative_to(self.superpoint_root)
        except ValueError:
            rel = Path(chunk_npz.name)
        return self.target_perm_cache_root / rel.with_suffix(".s1perm.npz")

    @staticmethod
    def _has_key(z, zsp, key: str) -> bool:
        return key in z.files or (zsp is not None and key in zsp.files)

    @staticmethod
    def _read_key(z, zsp, key: str):
        if zsp is not None and key in zsp.files:
            return zsp[key]
        if key in z.files:
            return z[key]
        raise KeyError(key)

    @staticmethod
    def _zsp_query_mode(zsp, prefix: str) -> str:
        if zsp is None:
            return "raw"
        key = f"{prefix}zsp_query_mode"
        if key not in zsp.files:
            return "raw"
        value = zsp[key]
        if hasattr(value, "item"):
            value = value.item()
        return str(value)

    def _load_superpoint_group(
        self, z, zsp, prefix: str, role: str, target_n: int, norm_factor: float
    ) -> Dict[str, torch.Tensor]:
        """Load one superpoint group (``first_``/``rest_``) into ``role``-prefixed tensors."""
        xyz_key = f"{prefix}superpoint_xyz"
        if not self._has_key(z, zsp, xyz_key):
            raise RuntimeError(f"Chunk missing superpoint group key: {xyz_key}")
        sp_xyz = self._read_key(z, zsp, xyz_key).astype(np.float32)
        raw_n = int(sp_xyz.shape[0])
        query_mode = self._zsp_query_mode(zsp, prefix)
        aligned_source_fields = query_mode == "raw"

        # VAE latent (accept both explicit-features and short aliases).
        sp_vae = None
        if aligned_source_fields:
            for k in (f"{prefix}superpoint_vae_features", f"{prefix}sp_vae"):
                if self._has_key(z, zsp, k):
                    sp_vae = self._read_key(z, zsp, k).astype(np.float32)
                    break

        sp_z = None
        z_key = f"{prefix}superpoint_z_sp"
        if self._has_key(z, zsp, z_key):
            sp_z = self._read_key(z, zsp, z_key).astype(np.float32)

        valid_key = f"{prefix}superpoint_valid_mask"
        if self._has_key(z, zsp, valid_key):
            sp_valid = self._read_key(z, zsp, valid_key).astype(bool)
        else:
            sp_valid = np.ones((raw_n,), dtype=bool)

        norm_xyz = (sp_xyz / norm_factor).astype(np.float32)
        norm_xyz = _pad_np_first_dim(norm_xyz, target_n, fill_value=0).astype(np.float32)
        sp_valid = _pad_np_first_dim(sp_valid, target_n, fill_value=False).astype(bool)
        if raw_n > target_n:
            sp_valid[target_n:] = False

        out: Dict[str, torch.Tensor] = {
            f"{role}_sp_xyz": torch.from_numpy(norm_xyz),
            f"{role}_sp_valid_mask": torch.from_numpy(sp_valid),
            f"num_{role}_superpoints_raw": torch.tensor(raw_n, dtype=torch.int32),
            f"num_{role}_superpoints_fixed": torch.tensor(target_n, dtype=torch.int32),
        }
        if sp_vae is not None:
            sp_vae = _pad_np_first_dim(sp_vae, target_n, fill_value=0).astype(np.float32)
            out[f"{role}_sp_vae"] = torch.from_numpy(sp_vae)
        if sp_z is not None:
            sp_z = _pad_np_first_dim(sp_z, target_n, fill_value=0).astype(np.float32)
            out[f"{role}_sp_z"] = torch.from_numpy(sp_z)
            out[f"{role}_con"] = torch.from_numpy(((sp_z - self.zsp_mean) / self.zsp_std).astype(np.float32))
        out[f"{role}_pos"] = torch.from_numpy(((norm_xyz - self.xyz_mean) / self.xyz_std).astype(np.float32))
        out[f"{role}_mask"] = torch.from_numpy(sp_valid)
        if f"{role}_con" in out:
            out[f"{role}_67"] = torch.cat([out[f"{role}_con"], out[f"{role}_pos"]], dim=-1)
        if self.include_geometry and aligned_source_fields:
            for field in GEOMETRY_FIELDS:
                fk = f"{prefix}{field}"
                if self._has_key(z, zsp, fk):
                    padded = _pad_np_first_dim(self._read_key(z, zsp, fk).astype(np.float32), target_n, fill_value=0).astype(np.float32)
                else:
                    padded = np.zeros((target_n,) + GEOMETRY_FIELD_SHAPES[field], dtype=np.float32)
                out[f"{role}_{field}"] = torch.from_numpy(padded)
            gk = f"{prefix}sp_geometry_valid_mask"
            if self._has_key(z, zsp, gk):
                geom_valid = _pad_np_first_dim(self._read_key(z, zsp, gk).astype(bool), target_n, fill_value=False).astype(bool)
            else:
                geom_valid = np.zeros((target_n,), dtype=bool)
            out[f"{role}_sp_geometry_valid_mask"] = torch.from_numpy(geom_valid & sp_valid)
        elif self.include_geometry:
            for field in GEOMETRY_FIELDS:
                padded = np.zeros((target_n,) + GEOMETRY_FIELD_SHAPES[field], dtype=np.float32)
                out[f"{role}_{field}"] = torch.from_numpy(padded)
            out[f"{role}_sp_geometry_valid_mask"] = torch.zeros((target_n,), dtype=torch.bool)
        return out

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        e = self.items[idx]
        scene_name = e["scene_name"]
        chunk_npz = e["chunk_npz"]
        z = np.load(chunk_npz, allow_pickle=True)
        zsp = None
        if self.zsp_sidecar:
            sidecar = self._sidecar_path(chunk_npz)
            if sidecar.is_file():
                zsp = np.load(sidecar, allow_pickle=True)

        da3_depths = z["da3_depths"].astype(np.float32)
        camera_poses_c2w = z["camera_poses_c2w"].astype(np.float32)
        camera_intrinsics = z["camera_intrinsics"].astype(np.float32)

        paths = z["image_paths"]
        if hasattr(paths, "tolist"):
            paths = paths.tolist()
        if self.max_views is not None:
            paths = paths[: self.max_views]
        num_views = len(paths)

        frame_indices = z["frame_indices"].astype(np.int64) if "frame_indices" in z.files else None
        if frame_indices is not None and self.max_views is not None:
            frame_indices = frame_indices[: self.max_views]
        if "chunk_base_index" in z.files:
            chunk_base = int(z["chunk_base_index"])
        elif frame_indices is not None and frame_indices.size > 0:
            chunk_base = int(frame_indices[0])
        else:
            chunk_base = 0

        # Shared metric normalizer: both superpoint groups and camera translations
        # are divided by the same factor so they share one coordinate frame.
        if "norm_scale" in z.files:
            norm_factor = float(z["norm_scale"])
        else:
            all_xyz = np.concatenate(
                [z[k].astype(np.float32) for k in ("first_superpoint_xyz", "rest_superpoint_xyz") if k in z.files],
                axis=0,
            )
            sp_dist = np.linalg.norm(all_xyz, ord=2, axis=-1)
            norm_factor = float(np.nanmean(sp_dist)) if sp_dist.size > 0 else 1.0
        if (not np.isfinite(norm_factor)) or norm_factor < 1e-6:
            norm_factor = 1.0

        depths: List[np.ndarray] = []
        poses: List[np.ndarray] = []
        Ks: List[torch.Tensor] = []
        valid_masks: List[torch.Tensor] = []
        norm_depths: List[torch.Tensor] = []
        norm_c2w_list: List[torch.Tensor] = []
        img_tensors: List[torch.Tensor] = []

        for view_idx, stored_path in enumerate(paths):
            depth = da3_depths[view_idx].astype(np.float32)
            h_da3, w_da3 = depth.shape
            target_w = int(self.resize_w) if self.resize_w is not None else w_da3
            target_h = int(self.resize_h) if self.resize_h is not None else h_da3
            if self.load_images:
                img = self._read_view_image(stored_path)
                if img.shape[:2] != (target_h, target_w):
                    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                img_t = self.image_transform(img).permute(1, 2, 0).contiguous().to(torch.float32)
            else:
                img_t = torch.empty((target_h, target_w, 3), dtype=torch.float32)
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

            pose_mat = camera_poses_c2w[view_idx].astype(np.float32)
            if pose_mat.shape != (4, 4) or (not np.isfinite(pose_mat).all()):
                raise RuntimeError(f"Invalid pose in chunk {e['chunk_npz']} view {view_idx}")
            norm_c2w = pose_mat.copy()
            norm_c2w[:3, 3] = norm_c2w[:3, 3] / norm_factor

            depths.append(depth)
            poses.append(pose_mat)
            Ks.append(torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32))
            valid_masks.append(torch.from_numpy(valid_mask))
            norm_depths.append(torch.from_numpy((np.clip(depth, 0.0, self.norm_depth_clip) / norm_factor).astype(np.float32)))
            norm_c2w_list.append(torch.from_numpy(norm_c2w).to(torch.float32))
            img_tensors.append(img_t)

        H, W = depths[0].shape
        valid_num_views = num_views
        cam_mask = torch.ones((valid_num_views,), dtype=torch.bool)
        if self.max_views is not None and valid_num_views < self.max_views:
            pad_v = self.max_views - valid_num_views
            img_tensors.extend([torch.zeros_like(img_tensors[0]) for _ in range(pad_v)])
            depths.extend([np.zeros_like(depths[0]) for _ in range(pad_v)])
            poses.extend([np.eye(4, dtype=np.float32) for _ in range(pad_v)])
            Ks.extend([torch.eye(3, dtype=torch.float32) for _ in range(pad_v)])
            valid_masks.extend([torch.zeros_like(valid_masks[0]) for _ in range(pad_v)])
            norm_depths.extend([torch.zeros_like(norm_depths[0]) for _ in range(pad_v)])
            norm_c2w_list.extend([torch.eye(4, dtype=torch.float32) for _ in range(pad_v)])
            cam_mask = torch.cat([cam_mask, torch.zeros((pad_v,), dtype=torch.bool)], dim=0)
            num_views = self.max_views
        out: Dict = {
            "seq": scene_name,
            "frame_id": int(frame_indices[0]) if frame_indices is not None and frame_indices.size > 0 else int(chunk_base),
            "chunk_stem": e["chunk_npz"].stem,
            "num_views": num_views,
            "cond_frame_index": COND_FRAME_INDEX,
            "image": torch.stack(img_tensors, dim=0),
            "pose": torch.stack([torch.from_numpy(p).to(torch.float32) for p in poses], dim=0),
            "Ks": torch.stack(Ks, dim=0),
            "image_size": (H, W),
            "valid_mask": torch.stack(valid_masks, dim=0),
            "norm_depth": torch.stack(norm_depths, dim=0),
            "real_depth": torch.stack([torch.from_numpy(d.astype(np.float32)) for d in depths], dim=0),
            "real_depth_valid_mask": torch.stack(valid_masks, dim=0),
            "norm_camtoworlds": torch.stack(norm_c2w_list, dim=0),
            "norm_scale": torch.tensor(norm_factor, dtype=torch.float32),
            "cam_mask": cam_mask,
            "xyz_mean": torch.from_numpy(self.xyz_mean.copy()),
            "xyz_std": torch.from_numpy(self.xyz_std.copy()),
            "zsp_mean": torch.from_numpy(self.zsp_mean.copy()),
            "zsp_std": torch.from_numpy(self.zsp_std.copy()),
        }
        if frame_indices is not None:
            if self.max_views is not None and frame_indices.shape[0] < self.max_views:
                frame_indices = np.pad(frame_indices, (0, self.max_views - frame_indices.shape[0]), constant_values=-1)
            out["frame_indices"] = torch.from_numpy(frame_indices.copy())

        # Condition = superpoints from the first frame; target = superpoints from
        # the subsequent frames.  Both normalized by the shared norm_factor.
        out.update(self._load_superpoint_group(z, zsp, "first_", "cond", self.cond_max_superpoints, norm_factor))
        out.update(self._load_superpoint_group(z, zsp, "rest_", "target", self.target_max_superpoints, norm_factor))
        if self.stage1_xyz_cache_root is not None:
            # Optional two-stage sidecar: stage-1 DiT predicted xyz in the same
            # world-standardized frame as target_pos (see tools/cache_stage1_xyz.py).
            s1_path = self._stage1_xyz_path(chunk_npz)
            if s1_path.is_file():
                with np.load(s1_path) as s1:
                    s1_xyz = s1["stage1_pos"].astype(np.float32)
                s1_xyz = _pad_np_first_dim(s1_xyz, self.target_max_superpoints, fill_value=0).astype(np.float32)
                out["stage1_pos"] = torch.from_numpy(s1_xyz)
            elif self.stage1_xyz_required:
                raise FileNotFoundError(
                    f"Missing stage-1 xyz sidecar: {s1_path}. "
                    "Generate the cache with tools/cache_stage1_xyz.py first."
                )
            else:
                out["stage1_pos"] = out["target_pos"].clone()
        if self.target_perm_cache_root is not None:
            # Hungarian rebinding (tools/build_hungarian.py): stage-1
            # slot i takes over the matched GT point's per-point target fields,
            # so position/content targets are defined by where the token sits,
            # not by the GT storage order.  Unmatched slots are masked out.
            perm_path = self._target_perm_path(chunk_npz)
            if perm_path.is_file():
                with np.load(perm_path) as perm_npz:
                    perm = torch.from_numpy(np.asarray(perm_npz["perm"], dtype=np.int64))
                    perm_dist = torch.from_numpy(np.asarray(perm_npz["perm_dist"], dtype=np.float32))
                n_slots = int(perm.shape[0])
                matched = perm >= 0
                perm_index = perm.clamp_min(0)
                for key in list(out.keys()):
                    value = out[key]
                    if (
                        key.startswith("target_")
                        and torch.is_tensor(value)
                        and value.dim() >= 1
                        and value.shape[0] == n_slots
                    ):
                        out[key] = value[perm_index]
                out["target_mask"] = out["target_mask"] & matched
                out["target_sp_valid_mask"] = out["target_sp_valid_mask"] & matched
                if "target_sp_geometry_valid_mask" in out:
                    out["target_sp_geometry_valid_mask"] = out["target_sp_geometry_valid_mask"] & matched
                out["target_perm_dist"] = perm_dist
                out["target_perm_matched"] = matched
            elif self.target_perm_required:
                raise FileNotFoundError(
                    f"Missing target permutation sidecar: {perm_path}. "
                    "Generate the cache with tools/build_hungarian.py first."
                )
        if self.return_v1 or self.return_v2:
            if "cond_con" not in out or "target_con" not in out:
                sidecar = self._sidecar_path(chunk_npz) if self.zsp_sidecar else None
                raise RuntimeError(
                    "Missing cached z_sp for v1 sample: "
                    f"{chunk_npz} sidecar={sidecar} sidecar_exists={bool(sidecar and sidecar.is_file())}"
                )
            geom_parts = []
            for field in V1_GEOMETRY_FIELDS:
                key = f"target_{field}"
                if key in out:
                    arr = out[key].to(torch.float32)
                    geom_parts.append(arr.reshape(arr.shape[0], -1))
            if geom_parts:
                out["target_geom"] = torch.cat(geom_parts, dim=-1)
            else:
                out["target_geom"] = torch.zeros((self.target_max_superpoints, 0), dtype=torch.float32)
            out["target_geom_valid"] = out.get("target_sp_geometry_valid_mask", out["target_mask"]).bool() & out["target_mask"].bool()
            if "target_sp_vae" not in out:
                out["target_sp_vae"] = torch.zeros((self.target_max_superpoints, 16), dtype=torch.float32)
        return out
