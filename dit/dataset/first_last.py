"""2-view dataset: first-frame + last-frame condition, same target as 1-view.

This class is independent of the 1-view default path. Existing configs that
construct ``FirstFrameGenSuperpointChunkDataset`` are unchanged.

Condition tokens are ``cat(first, last)`` along the superpoint axis. Target
tokens, stage-1 xyz, Hungarian perms, and the 9-view cameras/RGBD all come
from the 1-view chunk unchanged. Missing last superpoints or last z_sp raise;
there is no silent fallback to 1-view.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

import inspect

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from .first_frame import (
    FirstFrameGenSuperpointChunkDataset,
    _PLACEHOLDER_SUPERPOINT_ROOT,
)

_DATASET_ROOT = Path("/mnt/share/codes/kimi/ssf_dataset")
_LAST_ZSP_ROOT = _DATASET_ROOT / "first_frame_gen_cache_last_zsp"

# 1-view superpoint_root -> (last superpoint root, last z_sp cache root)
DEFAULT_LAST_ROOTS: dict[Path, tuple[Path, Path]] = {
    _DATASET_ROOT / "DL3DV" / "first_frame_gen_chunk_new": (
        _DATASET_ROOT / "DL3DV" / "first_frame_gen_chunk_last_new",
        _LAST_ZSP_ROOT / "dl3dv" / "train",
    ),
    _DATASET_ROOT / "DL3DV" / "first_frame_gen_chunk_eval_new": (
        _DATASET_ROOT / "DL3DV" / "first_frame_gen_chunk_last_eval_new",
        _LAST_ZSP_ROOT / "dl3dv" / "val",
    ),
    _DATASET_ROOT / "RE10K" / "first_frame_gen_chunk" / "train": (
        _DATASET_ROOT / "RE10K" / "first_frame_gen_chunk_last" / "train",
        _LAST_ZSP_ROOT / "re10k" / "train",
    ),
    _DATASET_ROOT / "RE10K" / "first_frame_gen_chunk" / "test": (
        _DATASET_ROOT / "RE10K" / "first_frame_gen_chunk_last" / "test",
        _LAST_ZSP_ROOT / "re10k" / "val",
    ),
    _DATASET_ROOT / "scannet_first_frame_gen_chunk_new": (
        _DATASET_ROOT / "scannet_first_frame_gen_chunk_last_new",
        _LAST_ZSP_ROOT / "scannet",
    ),
}


def infer_last_cache_roots(superpoint_root: Path) -> tuple[Path, Path]:
    root = Path(superpoint_root).resolve()
    for source, mapped in DEFAULT_LAST_ROOTS.items():
        if root == source.resolve():
            return mapped
    raise FileNotFoundError(
        "Cannot infer 2-view last roots from superpoint_root="
        f"{superpoint_root}. Pass last_superpoint_root and last_zsp_cache_root."
    )


def build_first_last_dataset(kwargs: dict[str, Any]) -> Dataset:
    """Concat-aware factory. Does not replace the 1-view trainer factory."""
    kwargs = dict(kwargs)
    datasets_cfg = kwargs.pop("datasets", None)
    if datasets_cfg is None:
        return FirstLastFrameGenSuperpointChunkDataset(**kwargs)
    if not datasets_cfg:
        raise ValueError("dataset.datasets must contain at least one dataset config")
    datasets = []
    for ds_cfg in datasets_cfg:
        merged = dict(kwargs)
        merged.update(dict(ds_cfg))
        datasets.append(FirstLastFrameGenSuperpointChunkDataset(**merged))
    return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]


def first_view_source_mask(
    cond_mask: torch.Tensor,
    cond_view_index: torch.Tensor | None,
) -> torch.Tensor:
    """Valid first-view tokens only. Does not use a fixed array slice."""
    mask = cond_mask.bool()
    if cond_view_index is None:
        return mask
    view = cond_view_index.to(device=mask.device)
    if view.shape != mask.shape:
        raise ValueError(
            f"cond_view_index shape {tuple(view.shape)} does not match cond_mask {tuple(mask.shape)}"
        )
    return mask & (view == 0)


def first_view_source_mask_from_batch(batch: dict[str, Any]) -> torch.Tensor:
    if "cond_mask" not in batch:
        raise KeyError("batch is missing cond_mask")
    return first_view_source_mask(batch["cond_mask"], batch.get("cond_view_index"))


def stage1_view_kwargs(model: Any, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Pass cond_view_index / cond_frame_index only if the model accepts them."""
    core = model.module if hasattr(model, "module") else model
    params = inspect.signature(core.forward).parameters
    extra: dict[str, torch.Tensor] = {}
    if "cond_view_index" in params and "cond_view_index" in batch:
        extra["cond_view_index"] = batch["cond_view_index"]
    if "cond_frame_index" in params and "cond_frame_index" in batch:
        extra["cond_frame_index"] = batch["cond_frame_index"]
    return extra


class FirstLastFrameGenSuperpointChunkDataset(FirstFrameGenSuperpointChunkDataset):
    """1-view chunk + last-frame sidecar -> cond = first ⊕ last."""

    def __init__(
        self,
        *args,
        last_superpoint_root: Optional[Path] = None,
        last_zsp_cache_root: Optional[Path] = None,
        cond_max_superpoints_per_view: Optional[int] = None,
        **kwargs,
    ) -> None:
        per_view = int(
            cond_max_superpoints_per_view
            if cond_max_superpoints_per_view is not None
            else kwargs.get("cond_max_superpoints", 3000)
        )
        if per_view <= 0:
            raise ValueError(f"cond_max_superpoints_per_view must be > 0, got {per_view}")
        kwargs["cond_max_superpoints"] = per_view

        if "superpoint_root" in kwargs:
            superpoint_root = Path(kwargs["superpoint_root"])
        elif len(args) >= 2:
            superpoint_root = Path(args[1])
        else:
            superpoint_root = _PLACEHOLDER_SUPERPOINT_ROOT

        if last_superpoint_root is None or last_zsp_cache_root is None:
            inferred_last, inferred_zsp = infer_last_cache_roots(superpoint_root)
            last_superpoint_root = last_superpoint_root or inferred_last
            last_zsp_cache_root = last_zsp_cache_root or inferred_zsp

        self.last_superpoint_root = Path(last_superpoint_root)
        self.last_zsp_cache_root = Path(last_zsp_cache_root)
        self.cond_max_superpoints_per_view = per_view
        self.cond_max_superpoints_total = 2 * per_view
        self.cond_protocol = "first_last"
        super().__init__(*args, **kwargs)

    def _last_superpoint_path(self, chunk_npz: Path) -> Path:
        try:
            rel = chunk_npz.relative_to(self.superpoint_root)
        except ValueError:
            rel = Path(chunk_npz.name)
        return self.last_superpoint_root / rel

    def _last_zsp_path(self, chunk_npz: Path) -> Path:
        last_npz = self._last_superpoint_path(chunk_npz)
        try:
            rel = last_npz.relative_to(self.last_superpoint_root)
        except ValueError:
            rel = Path(last_npz.name)
        return self.last_zsp_cache_root / rel.with_suffix(".last.zsp.npz")

    def _default_index_cache_path(self) -> Path:
        if self.index_cache_path is not None:
            return Path(self.index_cache_path)
        path = super()._default_index_cache_path()
        last_id = hashlib.sha1(
            f"{self.last_superpoint_root.resolve()}|{self.last_zsp_cache_root.resolve()}".encode("utf-8")
        ).hexdigest()[:8]
        return path.with_name(path.stem + f"_2view_{last_id}" + path.suffix)

    def _index_cache_meta(self) -> dict:
        meta = super()._index_cache_meta()
        meta.update(
            {
                "cond_protocol": self.cond_protocol,
                "last_superpoint_root": str(self.last_superpoint_root.resolve()),
                "last_zsp_cache_root": str(self.last_zsp_cache_root.resolve()),
                "cond_max_superpoints_per_view": int(self.cond_max_superpoints_per_view),
            }
        )
        return meta

    def _chunk_has_required_zsp(self, z, chunk_npz: Path) -> bool:
        if not super()._chunk_has_required_zsp(z, chunk_npz):
            return False
        last_npz = self._last_superpoint_path(chunk_npz)
        if not last_npz.is_file():
            return False
        try:
            with np.load(last_npz, allow_pickle=True) as last_z:
                if "last_superpoint_xyz" not in last_z.files:
                    return False
        except Exception:
            return False
        if not (self.return_v1 or self.return_v2):
            return True
        last_zsp = self._last_zsp_path(chunk_npz)
        if not last_zsp.is_file():
            return False
        try:
            with np.load(last_zsp, allow_pickle=True) as last_cache:
                return "last_superpoint_z_sp" in last_cache.files
        except Exception:
            return False

    def _require_last_files(self, chunk_npz: Path) -> tuple[Path, Path]:
        last_npz = self._last_superpoint_path(chunk_npz)
        last_zsp_path = self._last_zsp_path(chunk_npz)
        if not last_npz.is_file():
            raise FileNotFoundError(
                f"Missing last-frame superpoints for 2-view sample: {last_npz}. "
                "Refusing to fall back to 1-view."
            )
        if (self.return_v1 or self.return_v2) and not last_zsp_path.is_file():
            raise FileNotFoundError(
                f"Missing last-frame z_sp for 2-view sample: {last_zsp_path}. "
                "Refusing to fall back to 1-view."
            )
        return last_npz, last_zsp_path

    def __getitem__(self, idx: int) -> Dict:
        out = super().__getitem__(idx)
        chunk_npz = self.items[idx]["chunk_npz"]
        last_npz, last_zsp_path = self._require_last_files(chunk_npz)

        norm_factor = float(out["norm_scale"])
        last_zsp = None
        with np.load(last_npz, allow_pickle=True) as last_z:
            if last_zsp_path.is_file():
                last_zsp = np.load(last_zsp_path, allow_pickle=True)
            try:
                last_group = self._load_superpoint_group(
                    last_z,
                    last_zsp,
                    "last_",
                    "last",
                    self.cond_max_superpoints_per_view,
                    norm_factor,
                )
            finally:
                if last_zsp is not None:
                    last_zsp.close()
        if (self.return_v1 or self.return_v2) and "last_con" not in last_group:
            raise RuntimeError(f"last z_sp sidecar has no last_superpoint_z_sp: {last_zsp_path}")
        if (self.return_v1 or self.return_v2) and "cond_con" not in out:
            raise RuntimeError(f"1-view sample is missing first-frame z_sp: {chunk_npz}")

        per_view = self.cond_max_superpoints_per_view
        first_raw = int(out["num_cond_superpoints_raw"])
        last_raw = int(last_group["num_last_superpoints_raw"])
        for key, last_value in last_group.items():
            if not key.startswith("last_") or not torch.is_tensor(last_value) or last_value.dim() < 1:
                continue
            if last_value.shape[0] != per_view:
                continue
            cond_key = "cond_" + key[len("last_") :]
            first_value = out.get(cond_key)
            if (
                torch.is_tensor(first_value)
                and first_value.dim() >= 1
                and first_value.shape[0] == per_view
            ):
                out[cond_key] = torch.cat([first_value, last_value], dim=0)
        out.update(last_group)
        out["num_cond_superpoints_raw"] = torch.tensor(first_raw + last_raw, dtype=torch.int32)
        out["num_cond_superpoints_fixed"] = torch.tensor(2 * per_view, dtype=torch.int32)
        out["cond_view_index"] = torch.cat(
            [
                torch.zeros((per_view,), dtype=torch.int64),
                torch.ones((per_view,), dtype=torch.int64),
            ],
            dim=0,
        )
        num_views = int(out["num_views"])
        last_frame = max(num_views - 1, 0)
        if "cam_mask" in out:
            valid_views = int(out["cam_mask"].to(torch.bool).sum().item())
            if valid_views > 0:
                last_frame = valid_views - 1
        out["cond_frame_index"] = torch.tensor([0, last_frame], dtype=torch.int64)
        return out
