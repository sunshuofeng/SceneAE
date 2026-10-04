#!/usr/bin/env python3
"""Cache stage-1 predicted xyz sidecars for stage-2 training."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sample import sample_stage1_xyz, to_device_sample
from train_common import build_dataset, build_model, cleanup_distributed, init_distributed, load_config


def _chunk_seed(rel_path: str, base_seed: int) -> int:
    digest = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) + int(base_seed)) % (2**31 - 1)


def _save_atomic(path: Path, stage1_pos: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, stage1_pos=stage1_pos)
    os.replace(tmp, path)


def _iter_dataset_entries(cfg, splits: set[str]):
    for split_name, cfg_key in (("train", "dataset"), ("val", "val_dataset")):
        if split_name not in splits:
            continue
        root_cfg = dict(getattr(cfg, cfg_key))
        entries = root_cfg.pop("datasets", None)
        if not entries:
            raise ValueError(f"cfg.{cfg_key}.datasets is required")
        for entry in entries:
            merged = dict(root_cfg)
            merged.update(dict(entry))
            if not merged.get("stage1_xyz_cache_root"):
                raise ValueError("dataset entry lacks stage1_xyz_cache_root")
            merged["load_images"] = False
            merged["stage1_xyz_required"] = False
            yield split_name, merged


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1_config", type=Path, default=Path("configs/stage1.py"))
    parser.add_argument("--stage1_ckpt", type=Path, default=None)
    parser.add_argument("--dataset_config", type=Path, default=Path("configs/stage2.py"))
    parser.add_argument("--splits", type=str, default="train,val")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--source_noise", type=float, default=None)
    parser.add_argument("--n_points", type=int, default=3000)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--guidance_mode", choices=("none", "conditional", "camera"), default="camera")
    parser.add_argument("--base_seed", type=int, default=123)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--two_view", action="store_true")
    args = parser.parse_args()

    rank, world_size, _, device = init_distributed()
    try:
        stage1_cfg = load_config(args.stage1_config)
        dataset_cfg = load_config(args.dataset_config)
        ckpt_path = args.stage1_ckpt or Path(stage1_cfg.sample["ckpt"])
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = build_model(stage1_cfg).to(device).eval()
        model.load_state_dict(ckpt["model"], strict=True)
        source_noise = float(
            args.source_noise
            if args.source_noise is not None
            else stage1_cfg.sample.get("first_frame_source_noise", stage1_cfg.flow.get("first_frame_source_noise", 0.5))
        )
        if args.two_view:
            from dataset import build_first_last_dataset

            def _build(kwargs):
                return build_first_last_dataset(kwargs)
        else:
            _build = build_dataset
        splits = {s.strip() for s in str(args.splits).split(",") if s.strip()}
        for split_name, ds_kwargs in _iter_dataset_entries(dataset_cfg, splits):
            ds = _build(dict(ds_kwargs))
            cache_root = Path(ds_kwargs["stage1_xyz_cache_root"])
            indices = list(range(rank, len(ds), world_size))
            if args.limit > 0:
                indices = indices[: args.limit]
            written = skipped = 0
            iterator = tqdm(indices, desc=f"{split_name}:{cache_root.name} rank={rank}", disable=rank != 0)
            for idx in iterator:
                item = ds.items[idx]
                chunk_npz = Path(item["chunk_npz"])
                sidecar = ds._stage1_xyz_path(chunk_npz)
                if sidecar.is_file() and not args.overwrite:
                    skipped += 1
                    continue
                batch = to_device_sample(ds[idx], device)
                rel = str(chunk_npz.relative_to(ds.superpoint_root))
                x_pos, _ = sample_stage1_xyz(
                    model,
                    batch,
                    n_points=args.n_points,
                    steps=args.steps,
                    seed=_chunk_seed(rel, args.base_seed),
                    source_noise=source_noise,
                    guidance=args.guidance,
                    guidance_mode=args.guidance_mode,
                )
                pos_world = model.canonical_to_world_standardized(
                    x_pos, batch["norm_camtoworlds"], batch["xyz_mean"], batch["xyz_std"]
                )[0]
                pos_world = torch.nan_to_num(pos_world, nan=0.0, posinf=0.0, neginf=0.0)
                _save_atomic(sidecar, pos_world.detach().cpu().numpy().astype(np.float32))
                written += 1
            print(f"[rank {rank}] {split_name}:{cache_root} written={written} skipped={skipped}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
