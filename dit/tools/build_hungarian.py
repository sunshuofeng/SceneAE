#!/usr/bin/env python3
"""Build Hungarian permutation sidecars that rebind GT superpoints to stage-1 slots.

Why this exists
---------------
v18 stage-1 denoises 3000 slots from pure noise; the GT chunk's storage order is
spatially random, so at inference slot i lands on an essentially random scene
point (measured: ~0% identity preservation, identity-L2 5-7x chamfer).  Stage-2
training, however, binds token i to GT point i, which pollutes both the position
residual target and the content target.

This tool fixes the binding WITHOUT retraining stage-1: for every chunk it runs
a one-to-one Hungarian assignment between the valid GT superpoints (rows) and
the stage-1 predicted slots (columns), then stores, per stage-1 slot i, the
matched original GT index ``perm[i]`` (or -1 when the slot is unmatched) and the
match distance ``perm_dist[i]``.  The dataset hook
``FirstFrameGenSuperpointChunkDataset(target_perm_cache_root=...)`` applies the
permutation to every per-point ``target_*`` field at load time, so token i's
position/content targets become "the GT point it actually sits next to" -- a
position-defined, index-free rule that is identical at training and inference.

Matching runs in the world-standardized frame (the frame of ``target_pos`` and
of the ``.s1xyz.npz`` sidecars), reusing the dataset's own
``_load_superpoint_group`` so normalization/masking are identical to training.

Output layout mirrors the stage-1 xyz cache.  Each sidecar is written to the
config entry's ``target_perm_cache_root`` (the same key the training loader
reads), i.e. for the v23/v24 configs:

    /mnt/share/codes/kimi/ssf_dataset/first_frame_gen_cache_new_3000/
        v18_stage1_perm_hungarian/<dataset>[/<split>]/<rel>.s1perm.npz
    keys: perm (int32 [N]), perm_dist (float32 [N])

Run under torchrun to shard across processes/GPUs, e.g.:

    torchrun --nproc_per_node=8 tools/build_hungarian.py \
        --dataset_config configs/stage2.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sample import to_device_sample
from train_common import build_dataset, cleanup_distributed, init_distributed, load_config

# 2-view configs add last-frame roots. Matching still only uses rest_ GT +
# the stage-1 xyz sidecar, but the *index* must match training: 2-view
# drops chunks that have no last-frame. Walking the 1-view index hits those
# extra chunks and FileNotFoundError on .s1xyz.npz.
_TWOVIEW_ONLY_KEYS = (
    "last_superpoint_root",
    "last_zsp_cache_root",
    "cond_max_superpoints_per_view",
)


def _is_twoview_entry(kwargs: dict) -> bool:
    return bool(
        kwargs.get("last_superpoint_root")
        or kwargs.get("last_zsp_cache_root")
        or kwargs.get("cond_max_superpoints_per_view")
    )


def _build_perm_dataset(ds_kwargs: dict):
    kwargs = dict(ds_kwargs)
    if _is_twoview_entry(kwargs):
        from dataset import build_first_last_dataset

        return build_first_last_dataset(kwargs), "first_last"
    for key in _TWOVIEW_ONLY_KEYS:
        kwargs.pop(key, None)
    return build_dataset(kwargs), "first_only"


def _iter_dataset_entries(cfg, splits: set[str]):
    for split_name, cfg_key in (("train", "dataset"), ("val", "val_dataset")):
        if split_name not in splits:
            continue
        root_cfg = dict(getattr(cfg, cfg_key))
        entries = root_cfg.pop("datasets", None)
        if not entries:
            raise ValueError(f"cfg.{cfg_key}.datasets is required for permutation cache generation")
        for entry in entries:
            merged = dict(root_cfg)
            merged.update(dict(entry))
            if not merged.get("stage1_xyz_cache_root"):
                raise ValueError(f"cfg.{cfg_key} entry {entry.get('superpoint_root')!r} lacks stage1_xyz_cache_root")
            merged["load_images"] = False
            # Only indexing + _load_superpoint_group are used; the stage-1
            # sidecar is read directly, so don't gate construction on it.
            merged["stage1_xyz_required"] = False
            # The permutation sidecars may not exist yet (that is what this
            # tool is building), so never gate on them here.
            merged["target_perm_required"] = False
            yield split_name, merged


def _save_atomic(path: Path, perm: np.ndarray, perm_dist: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, perm=perm, perm_dist=perm_dist)
    os.replace(tmp, path)


def _norm_factor(z) -> float:
    """Mirror FirstFrameGenSuperpointChunkDataset.__getitem__ exactly."""
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
    return norm_factor


def _chunk_perm(ds, chunk_npz: Path, device: torch.device) -> tuple[np.ndarray, np.ndarray, dict]:
    """Hungarian-match valid GT points (rows) to stage-1 slots (columns)."""
    with np.load(chunk_npz, allow_pickle=True) as z:
        zsp = None
        if ds.zsp_sidecar:
            sidecar = ds._sidecar_path(chunk_npz)
            if sidecar.is_file():
                zsp = np.load(sidecar, allow_pickle=True)
        try:
            group = ds._load_superpoint_group(
                z, zsp, "rest_", "target", ds.target_max_superpoints, _norm_factor(z)
            )
        finally:
            if zsp is not None:
                zsp.close()
    gt_pos = group["target_pos"].numpy().astype(np.float32)
    gt_mask = group["target_mask"].numpy().astype(bool)
    valid_index = np.where(gt_mask)[0]

    s1_path = ds._stage1_xyz_path(chunk_npz)
    with np.load(s1_path) as s1:
        s1_pos = s1["stage1_pos"].astype(np.float32)
    n_slots = int(s1_pos.shape[0])

    perm = np.full((n_slots,), -1, dtype=np.int32)
    perm_dist = np.full((n_slots,), -1.0, dtype=np.float32)
    stats = {"n_gt": int(valid_index.size), "n_slots": n_slots, "match_dist_sum": 0.0}
    if valid_index.size == 0:
        return perm, perm_dist, stats

    gt_t = torch.from_numpy(gt_pos[valid_index]).to(device)
    s1_t = torch.from_numpy(s1_pos).to(device)
    cost = torch.cdist(gt_t.float(), s1_t.float()).cpu().numpy()
    rows, cols = linear_sum_assignment(cost)  # rows: valid-GT order, cols: stage-1 slots
    perm[cols] = valid_index[rows].astype(np.int32)
    perm_dist[cols] = cost[rows, cols].astype(np.float32)
    stats["match_dist_sum"] = float(cost[rows, cols].sum())
    return perm, perm_dist, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_config", type=Path,
                        default=Path("configs/stage2.py"))
    parser.add_argument("--splits", type=str, default="train,val")
    parser.add_argument("--out_name", type=str, default="v18_stage1_perm_hungarian",
                        help="Fallback only (config entries without target_perm_cache_root): "
                             "name of the sibling collection replacing --s1_collection")
    parser.add_argument("--s1_collection", type=str, default="v18_stage1_xyz",
                        help="Fallback only: stage-1 xyz collection directory name to replace")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Debug: cap chunks per dataset entry (per rank shard).")
    args = parser.parse_args()

    rank, world_size, _, device = init_distributed()
    try:
        dataset_cfg = load_config(args.dataset_config)
        splits = {s.strip() for s in str(args.splits).split(",") if s.strip()}
        if rank == 0:
            print("***** Building Hungarian permutation sidecars (stage-2 GT rebinding) *****")
            print(f"  Config = {args.dataset_config}")
            print("  Output root = each entry's target_perm_cache_root "
                  f"(fallback collection name = {args.out_name})")
            print(f"  World size = {world_size}")

        for split_name, ds_kwargs in _iter_dataset_entries(dataset_cfg, splits):
            ds, index_kind = _build_perm_dataset(ds_kwargs)
            if rank == 0:
                print(f"  [{split_name}] index={index_kind} n={len(ds)} "
                      f"superpoint_root={ds_kwargs.get('superpoint_root')}")
            # Single source of truth: the config's target_perm_cache_root (the
            # training loader reads the same key).  Fall back to replacing the
            # stage-1 xyz collection directory only for configs lacking the key.
            if ds_kwargs.get("target_perm_cache_root"):
                out_root = Path(ds_kwargs["target_perm_cache_root"])
            else:
                s1_root = Path(ds_kwargs["stage1_xyz_cache_root"])
                parts = list(s1_root.parts)
                if args.s1_collection not in parts:
                    raise ValueError(
                        f"stage1_xyz_cache_root {s1_root} lacks collection dir "
                        f"{args.s1_collection!r}; set target_perm_cache_root in the config"
                    )
                parts[parts.index(args.s1_collection)] = args.out_name
                out_root = Path(*parts)
            indices = list(range(rank, len(ds), world_size))
            if args.limit > 0:
                indices = indices[: args.limit]
            written = skipped = missing_s1 = 0
            dist_sum = matched = 0
            iterator = tqdm(indices, desc=f"{split_name}:{out_root.name} rank={rank}", disable=rank != 0)
            for idx in iterator:
                chunk_npz = Path(ds.items[idx]["chunk_npz"])
                try:
                    rel = chunk_npz.relative_to(ds.superpoint_root)
                except ValueError:
                    rel = Path(chunk_npz.name)
                out_path = out_root / rel.with_suffix(".s1perm.npz")
                if out_path.is_file() and not args.overwrite:
                    skipped += 1
                    continue
                s1_path = ds._stage1_xyz_path(chunk_npz)
                if not s1_path.is_file():
                    missing_s1 += 1
                    if missing_s1 <= 5:
                        print(f"[rank {rank}] missing stage-1 xyz, skip: {s1_path}")
                    continue
                perm, perm_dist, stats = _chunk_perm(ds, chunk_npz, device)
                _save_atomic(out_path, perm, perm_dist)
                written += 1
                dist_sum += stats["match_dist_sum"]
                matched += stats["n_gt"]
                if rank == 0:
                    mean_d = dist_sum / max(matched, 1)
                    iterator.set_postfix_str(
                        f"written={written} skipped={skipped} missing_s1={missing_s1} "
                        f"mean_dist={mean_d:.4f}"
                    )
            mean_d = dist_sum / max(matched, 1)
            print(f"[rank {rank}] {split_name}:{out_root} written={written} skipped={skipped} "
                  f"missing_s1={missing_s1} mean_match_dist={mean_d:.4f}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
