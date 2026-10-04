"""Cached-latent first-frame-gen chunk dataset for C20_v2 decoder retraining.

Each sample pairs a first-frame-gen chunk (self-contained NPZ: image paths,
DA3 depths, camera poses/intrinsics, norm_scale) with the z_sp sidecar cache
produced by ``vae/cache_latent.py`` (frozen encoder latents).

Only the ``rest_`` group is served: it is encoded from ALL views with FPS
queries, i.e. exactly the distribution the DiT is trained to generate, so the
decoder trains on the same (xyz, z_sp) pairs it will receive at generation
time.  The ``first_`` group (DiT condition) is ignored here.

The model detects ``cached_z_sp`` in the batch and skips DA3 + encoder
entirely, so training only pays for the decoder forward/backward.

Per-chunk NPZ keys consumed (V = frames per chunk):
  image_paths, da3_depths, camera_poses_c2w, camera_intrinsics, norm_scale,
  frame_indices (optional), chunk_base_index (optional)
Sidecar keys consumed:
  rest_superpoint_xyz        (N, 3)  float32  world-frame (NOT normalized)
  rest_superpoint_valid_mask (N,)    bool
  rest_superpoint_z_sp       (N, 64) float16
"""

from __future__ import annotations

import json
from collections import OrderedDict
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .combined import (
    ConcatDatasetWithEpoch,
    ResizedDataset,
    _filter_kwargs_for_class,
    _merge_kwargs,
    _normalize_name,
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
    """Scene directories with a ``chunks/`` subfolder under ``root``."""
    if not root.exists():
        return []
    split_dirs = [d for d in sorted(root.iterdir()) if d.is_dir() and d.name in ("train", "test", "val")]
    roots = split_dirs if split_dirs else [root]
    scenes: List[Path] = []
    for r in roots:
        scenes.extend([d for d in sorted(r.iterdir()) if d.is_dir() and (d / "chunks").is_dir()])
    return scenes


class FFGCachedChunkDataset(Dataset):
    """One sample = one first-frame-gen chunk + its cached rest_-group z_sp.

    Args:
        chunk_root: first-frame-gen chunk root (``<scene>/chunks/*.npz``).
        cache_root: z_sp sidecar root mirroring the chunk tree.
        eval_chunk_root / eval_cache_root: optional separate roots used when
            ``split != "train"`` (DL3DV/RE10K style physical splits).
        split_reference_root: optional c15-style chunk root whose sorted scene
            list defines the train/val split (ScanNet: guarantees the exact
            same scene partition as the C15 reconstruction training/eval).
        split_ratio: fallback ratio when no physical split and no reference
            root are available.
        depth_near / depth_far: supervision valid-mask range in metric units
            (ScanNet: 0.1~10, DL3DV/RE10K: 0.1~50, matching the C15 datasets).
        norm_depth_clip: clip applied to depth before dividing by norm_scale.
    """

    def __init__(
        self,
        chunk_root: str,
        cache_root: str,
        split: str = "train",
        eval_chunk_root: Optional[str] = None,
        eval_cache_root: Optional[str] = None,
        split_reference_root: Optional[str] = None,
        split_ratio: float = 0.9,
        resize_w: int = 518,
        resize_h: int = 378,
        num_views: int = 9,
        target_max_superpoints: int = 3000,
        depth_near: float = 0.1,
        depth_far: Optional[float] = None,
        norm_depth_clip: float = 100.0,
        image_cache_max_files: int = 1,
        require_cache: bool = True,
        min_depth_valid_ratio: float = 0.0,
        depth_valid_index: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.chunk_root = Path(chunk_root)
        self.cache_root = Path(cache_root)
        self.eval_chunk_root = Path(eval_chunk_root) if eval_chunk_root else None
        self.eval_cache_root = Path(eval_cache_root) if eval_cache_root else None
        self.split = split
        self.split_ratio = float(split_ratio)
        self.split_reference_root = Path(split_reference_root) if split_reference_root else None
        self.resize_w = int(resize_w)
        self.resize_h = int(resize_h)
        self.num_views = int(num_views)
        self.target_max_superpoints = int(target_max_superpoints)
        self.depth_near = float(depth_near)
        self.depth_far = float(depth_far) if depth_far is not None else None
        self.norm_depth_clip = float(norm_depth_clip)
        self.require_cache = bool(require_cache)
        self.min_depth_valid_ratio = float(min_depth_valid_ratio)
        self._depth_valid_index_arg = depth_valid_index
        self._re10k_files: OrderedDict[Path, list] = OrderedDict()
        self._re10k_max_files = max(1, int(image_cache_max_files))

        self.active_chunk_root, self.active_cache_root = self._resolve_roots()
        self._valid_index = self._load_depth_valid_index()
        self.items: List[Dict] = []
        self._build_index()

    # ------------------------------------------------------------------
    # Index / split
    # ------------------------------------------------------------------
    def _resolve_roots(self) -> tuple[Path, Path]:
        if self.split != "train" and self.eval_chunk_root is not None:
            cache = self.eval_cache_root if self.eval_cache_root is not None else self.cache_root
            return self.eval_chunk_root, cache
        return self.chunk_root, self.cache_root

    def _load_depth_valid_index(self) -> Optional[Dict[str, float]]:
        """Load the optional per-chunk depth-validity index for filtering.

        Only consulted when ``min_depth_valid_ratio > 0``.  Defaults to
        ``<active_chunk_root>/depth_valid_index.json`` (built by
        ``process/build_depth_valid_index.py``).
        """
        if self.min_depth_valid_ratio <= 0:
            return None
        cand = (
            Path(self._depth_valid_index_arg)
            if self._depth_valid_index_arg
            else (self.active_chunk_root / "depth_valid_index.json")
        )
        if not cand.is_file():
            print(
                f"[ffg_cached] min_depth_valid_ratio={self.min_depth_valid_ratio} set but "
                f"index not found: {cand}; NO filtering applied."
            )
            return None
        try:
            with cand.open("r", encoding="utf-8") as f:
                idx = json.load(f)
        except Exception as exc:
            print(f"[ffg_cached] failed to load depth-valid index {cand}: {exc}; NO filtering.")
            return None
        print(
            f"[ffg_cached] loaded depth-valid index {cand} ({len(idx)} entries); "
            f"dropping chunks with valid_ratio < {self.min_depth_valid_ratio}"
        )
        return idx

    def _reference_split_names(self) -> Optional[tuple[set, set]]:
        """Train/val scene-name sets from the c15 reference root (ScanNet)."""
        if self.split_reference_root is None or not self.split_reference_root.exists():
            return None
        names = sorted(
            p.name for p in self.split_reference_root.iterdir() if p.is_dir() and p.name.startswith("scene")
        )
        if not names:
            return None
        split_idx = int(len(names) * self.split_ratio)
        return set(names[:split_idx]), set(names[split_idx:])

    def _scenes_for_split(self) -> List[Path]:
        scenes = _iter_scene_dirs(self.active_chunk_root)
        if not scenes:
            return []
        ref = self._reference_split_names()
        if ref is not None:
            train_names, val_names = ref
            if self.split == "train":
                # Scenes absent from the reference root cannot leak into the
                # c15 eval set, so they are safe to keep for training.
                return [s for s in scenes if s.name not in val_names]
            return [s for s in scenes if s.name in val_names]
        has_split = any(
            s.parent != self.active_chunk_root and s.parent.name in ("train", "test", "val") for s in scenes
        )
        if has_split:
            if self.split == "train":
                return [s for s in scenes if s.parent.name == "train"]
            return [s for s in scenes if s.parent.name in ("test", "val")]
        scenes = sorted(scenes, key=lambda x: x.name)
        split_idx = int(len(scenes) * self.split_ratio)
        return scenes[:split_idx] if self.split == "train" else scenes[split_idx:]

    def _sidecar_path(self, chunk_npz: Path) -> Path:
        try:
            rel = chunk_npz.relative_to(self.active_chunk_root)
        except ValueError:
            rel = Path(chunk_npz.name)
        return self.active_cache_root / rel.with_suffix(".zsp.npz")

    def _build_index(self) -> None:
        required = (
            "image_paths",
            "da3_depths",
            "camera_poses_c2w",
            "camera_intrinsics",
        )
        n_low_valid = 0
        for scene_dir in self._scenes_for_split():
            for chunk_npz in sorted((scene_dir / "chunks").glob("*.npz")):
                if chunk_npz.name.endswith(".zsp.npz"):
                    continue
                if self._valid_index is not None:
                    try:
                        rel = str(chunk_npz.relative_to(self.active_chunk_root))
                    except ValueError:
                        rel = chunk_npz.name
                    vr = self._valid_index.get(rel)
                    if vr is not None and vr < self.min_depth_valid_ratio:
                        n_low_valid += 1
                        continue
                sidecar = self._sidecar_path(chunk_npz)
                if self.require_cache and not sidecar.is_file():
                    continue
                try:
                    with np.load(chunk_npz, allow_pickle=True) as z:
                        if any(k not in z.files for k in required):
                            continue
                        paths = z["image_paths"]
                        if hasattr(paths, "tolist"):
                            paths = paths.tolist()
                        if len(paths) != self.num_views:
                            continue
                except Exception:
                    continue
                self.items.append(
                    {
                        "scene_dir": scene_dir,
                        "scene_name": scene_dir.name,
                        "chunk_npz": chunk_npz,
                        "sidecar": sidecar,
                    }
                )
        if self._valid_index is not None:
            print(
                f"[ffg_cached] depth-valid filter: dropped {n_low_valid} sparse chunks, "
                f"kept {len(self.items)} (split={self.split})"
            )

    def __len__(self) -> int:
        return len(self.items)

    # ------------------------------------------------------------------
    # Image reading (plain paths + RE10K embedded archives)
    # ------------------------------------------------------------------
    def _read_view_image(self, stored_path: str) -> np.ndarray:
        path_str = str(stored_path)
        if "::" in path_str:
            return self._read_re10k_image(path_str)
        img = cv2.imread(path_str, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path_str}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _read_re10k_image(self, spec: str) -> np.ndarray:
        source, scene_key, frame_name = spec.split("::", 2)
        source_path = Path(source)
        if source_path not in self._re10k_files:
            try:
                self._re10k_files[source_path] = torch.load(source_path, map_location="cpu", weights_only=True)
            except TypeError:
                self._re10k_files[source_path] = torch.load(source_path, map_location="cpu")
            while len(self._re10k_files) > self._re10k_max_files:
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

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict:
        e = self.items[idx]
        chunk_npz: Path = e["chunk_npz"]
        z = np.load(chunk_npz, allow_pickle=True)
        sidecar_path: Path = e["sidecar"]
        if not sidecar_path.is_file():
            raise FileNotFoundError(f"Missing z_sp sidecar: {sidecar_path}")
        zsp = np.load(sidecar_path, allow_pickle=True)

        da3_depths = z["da3_depths"].astype(np.float32)
        camera_poses_c2w = z["camera_poses_c2w"].astype(np.float32)
        camera_intrinsics = z["camera_intrinsics"].astype(np.float32)
        paths = z["image_paths"]
        if hasattr(paths, "tolist"):
            paths = paths.tolist()
        num_views = len(paths)

        frame_indices = z["frame_indices"].astype(np.int64) if "frame_indices" in z.files else None
        if "chunk_base_index" in z.files:
            chunk_base = int(z["chunk_base_index"])
        elif frame_indices is not None and frame_indices.size > 0:
            chunk_base = int(frame_indices[0])
        else:
            chunk_base = 0

        norm_factor = float(z["norm_scale"]) if "norm_scale" in z.files else 1.0
        if (not np.isfinite(norm_factor)) or norm_factor < 1e-6:
            norm_factor = 1.0

        img_tensors: List[torch.Tensor] = []
        Ks: List[torch.Tensor] = []
        norm_c2w_list: List[torch.Tensor] = []
        valid_masks: List[torch.Tensor] = []
        norm_depths: List[torch.Tensor] = []

        for view_idx, stored_path in enumerate(paths):
            depth = da3_depths[view_idx].astype(np.float32)
            h_da3, w_da3 = depth.shape
            target_w, target_h = self.resize_w, self.resize_h

            img = self._read_view_image(stored_path)
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

            valid_mask = np.isfinite(depth) & (depth > self.depth_near)
            if self.depth_far is not None:
                valid_mask &= depth < self.depth_far
            pose_mat = camera_poses_c2w[view_idx].astype(np.float32)
            if pose_mat.shape != (4, 4) or (not np.isfinite(pose_mat).all()):
                raise RuntimeError(f"Invalid pose in chunk {chunk_npz} view {view_idx}")
            norm_c2w = pose_mat.copy()
            norm_c2w[:3, 3] = norm_c2w[:3, 3] / norm_factor

            img_tensors.append(torch.from_numpy((img.astype(np.float32) / 255.0)))
            Ks.append(torch.from_numpy(k))
            norm_c2w_list.append(torch.from_numpy(norm_c2w).to(torch.float32))
            valid_masks.append(torch.from_numpy(valid_mask.astype(bool)))
            norm_depths.append(
                torch.from_numpy((np.clip(depth, 0.0, self.norm_depth_clip) / norm_factor).astype(np.float32))
            )

        # Cached rest_-group (all-views FPS queries) = the DiT target
        # distribution.  xyz is stored un-normalized; z_sp is stored fp16.
        rest_xyz = zsp["rest_superpoint_xyz"].astype(np.float32) / norm_factor
        rest_valid = (
            zsp["rest_superpoint_valid_mask"].astype(bool)
            if "rest_superpoint_valid_mask" in zsp.files
            else np.ones((rest_xyz.shape[0],), dtype=bool)
        )
        rest_z = zsp["rest_superpoint_z_sp"].astype(np.float32)
        raw_n = int(rest_xyz.shape[0])
        if rest_z.shape[0] != raw_n:
            raise RuntimeError(f"{sidecar_path}: z_sp N={rest_z.shape[0]} != xyz N={raw_n}")

        target_n = self.target_max_superpoints
        rest_xyz = _pad_np_first_dim(rest_xyz, target_n, fill_value=0).astype(np.float32)
        rest_z = _pad_np_first_dim(rest_z, target_n, fill_value=0).astype(np.float32)
        rest_valid = _pad_np_first_dim(rest_valid, target_n, fill_value=False).astype(bool)
        if raw_n > target_n:
            rest_valid[target_n:] = False

        H, W = self.resize_h, self.resize_w
        out: Dict = {
            "seq": e["scene_name"],
            "frame_id": int(frame_indices[0]) if frame_indices is not None and frame_indices.size > 0 else int(chunk_base),
            "chunk_stem": chunk_npz.stem,
            "num_views": num_views,
            "image": torch.stack(img_tensors, dim=0),
            "Ks": torch.stack(Ks, dim=0),
            "norm_camtoworlds": torch.stack(norm_c2w_list, dim=0),
            "valid_mask": torch.stack(valid_masks, dim=0),
            "norm_depth": torch.stack(norm_depths, dim=0),
            "norm_scale": torch.tensor(norm_factor, dtype=torch.float32),
            # Decoder inputs (cached latent path).
            "norm_sp_xyz": torch.from_numpy(rest_xyz),
            "superpoint_valid_mask": torch.from_numpy(rest_valid),
            "cached_z_sp": torch.from_numpy(rest_z),
            "num_superpoints_raw": torch.tensor(raw_n, dtype=torch.int32),
            "num_superpoints_fixed": torch.tensor(target_n, dtype=torch.int32),
            # Placeholder: per-point teacher features are not part of the
            # cache; the last-dim-0 tensor keeps sp_feat supervision disabled.
            "sp_feat": torch.zeros((target_n, 0), dtype=torch.float32),
        }
        if frame_indices is not None:
            out["frame_indices"] = torch.from_numpy(frame_indices.copy())
        return out


class CombinedFFGCachedDataset(Dataset):
    """Multi-dataset cached first-frame-gen chunks with GLD-style ratio resize.

    Mirrors CombinedSuperpointChunkDatasetV2C15 but every child is an
    FFGCachedChunkDataset serving cached (xyz, z_sp) decoder inputs.
    """

    def __init__(
        self,
        split: str = "train",
        datasets: Optional[Dict[str, Dict]] = None,
        mix_ratio: Optional[Dict[str, int]] = None,
        equal_ratio: bool = False,
        val_datasets: Optional[List[str]] = None,
        val_mix_ratio: Optional[Dict[str, int]] = None,
        **shared_kwargs,
    ) -> None:
        super().__init__()
        if not datasets:
            raise ValueError("CombinedFFGCachedDataset requires non-empty `datasets` config.")
        self.split = split
        self.mix_ratio = mix_ratio or {}
        self.equal_ratio = bool(equal_ratio)
        self.val_mix_ratio = val_mix_ratio

        names = list(datasets.keys())
        if split != "train" and val_datasets is not None:
            names = [n for n in val_datasets if n in datasets]
        if not names:
            raise ValueError(f"No dataset names selected for split={split}.")

        children = []
        for name in names:
            child_kwargs = _merge_kwargs(shared_kwargs, datasets[name])
            child_kwargs = _filter_kwargs_for_class(FFGCachedChunkDataset, child_kwargs)
            child = FFGCachedChunkDataset(split=split, **child_kwargs)
            if len(child) == 0:
                continue
            children.append((name, child))
        if not children:
            raise ValueError(f"No non-empty child datasets for split={split}.")

        if split == "train":
            if self.equal_ratio:
                target = max(len(child) for _, child in children)
                self.dataset = ConcatDatasetWithEpoch(ResizedDataset(child, target) for _, child in children)
            else:
                resized = []
                for name, child in children:
                    target = self.mix_ratio.get(name, self.mix_ratio.get(_normalize_name(name), len(child)))
                    resized.append(ResizedDataset(child, int(target)))
                self.dataset = ConcatDatasetWithEpoch(resized)
        else:
            if self.val_mix_ratio:
                resized = []
                for name, child in children:
                    target = self.val_mix_ratio.get(name, self.val_mix_ratio.get(_normalize_name(name), len(child)))
                    resized.append(ResizedDataset(child, int(target)))
                self.dataset = ConcatDatasetWithEpoch(resized)
            else:
                self.dataset = ConcatDatasetWithEpoch(child for _, child in children)

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def __getitem__(self, idx: int):
        return self.dataset[idx]
