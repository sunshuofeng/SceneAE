#!/usr/bin/env python3
"""Compute target-coordinate statistics in the first-camera coordinate frame."""

from __future__ import annotations

import argparse
import importlib.util
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    chunk_root: Path
    cache_root: Path | None
    split: str
    split_ratio: float
    source_split: str


class VectorStats:
    def __init__(self, dim: int) -> None:
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)
        self.cross_m2 = np.zeros((dim, dim), dtype=np.float64)
        self.minimum = np.full(dim, np.inf, dtype=np.float64)
        self.maximum = np.full(dim, -np.inf, dtype=np.float64)

    def merge(self, state: dict[str, Any]) -> None:
        count = int(state["count"])
        if count <= 0:
            return
        mean = np.asarray(state["mean"], dtype=np.float64)
        m2 = np.asarray(state["m2"], dtype=np.float64)
        cross_m2 = np.asarray(state["cross_m2"], dtype=np.float64)
        if self.count == 0:
            self.count = count
            self.mean = mean.copy()
            self.m2 = m2.copy()
            self.cross_m2 = cross_m2.copy()
            self.minimum = np.asarray(state["minimum"], dtype=np.float64).copy()
            self.maximum = np.asarray(state["maximum"], dtype=np.float64).copy()
            return
        delta = mean - self.mean
        total = self.count + count
        correction = np.outer(delta, delta) * self.count * count / total
        self.mean += delta * count / total
        self.m2 += m2 + np.square(delta) * self.count * count / total
        self.cross_m2 += cross_m2 + correction
        self.minimum = np.minimum(self.minimum, state["minimum"])
        self.maximum = np.maximum(self.maximum, state["maximum"])
        self.count = total

    def payload(self, prefix: str = "canonical_xyz") -> dict[str, Any]:
        denom = max(self.count - 1, 1)
        var = np.maximum(self.m2 / denom, 1e-12)
        covariance = self.cross_m2 / denom
        return {
            f"{prefix}_count": np.int64(self.count),
            f"{prefix}_mean": self.mean.astype(np.float32),
            f"{prefix}_m2": self.m2.astype(np.float64),
            f"{prefix}_var": var.astype(np.float32),
            f"{prefix}_std": np.sqrt(var).astype(np.float32),
            f"{prefix}_covariance": covariance.astype(np.float32),
            f"{prefix}_min": self.minimum.astype(np.float32),
            f"{prefix}_max": self.maximum.astype(np.float32),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute train target statistics after conversion to camera-0 coordinates."
    )
    parser.add_argument("config", type=Path, help="DiT Python config, e.g. configs/stage1.py")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Joint output .npz. Defaults beside config.dataset.zsp_stats_path.",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=None, help="Maximum chunks per dataset for a quick check.")
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def load_config(path: Path) -> ModuleType:
    path = path.resolve()
    spec = importlib.util.spec_from_file_location("canonical_target_stats_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dataset_name(root: Path) -> str:
    text = str(root).lower()
    for name in ("dl3dv", "re10k", "scannet"):
        if name in text:
            return name
    return root.name


def train_specs(cfg: ModuleType) -> list[DatasetSpec]:
    section = dict(cfg.dataset)
    children = section.pop("datasets", None)
    if not children:
        raise ValueError("config.dataset.datasets must contain the training datasets")
    specs = []
    used_names: dict[str, int] = {}
    for child in children:
        merged = dict(section)
        merged.update(dict(child))
        root = Path(str(merged["superpoint_root"])).expanduser().resolve()
        name = dataset_name(root)
        used_names[name] = used_names.get(name, 0) + 1
        if used_names[name] > 1:
            name = f"{name}_{used_names[name]}"
        cache_value = merged.get("zsp_cache_root")
        specs.append(
            DatasetSpec(
                name=name,
                chunk_root=root,
                cache_root=Path(str(cache_value)).expanduser().resolve() if cache_value else None,
                split=str(merged.get("split", "train")),
                split_ratio=float(merged.get("split_ratio", 0.9)),
                source_split=str(merged.get("source_split", "all")).lower(),
            )
        )
    return specs


def iter_scene_dirs(root: Path) -> list[Path]:
    split_dirs = [root / name for name in ("train", "test", "val") if (root / name).is_dir()]
    search_roots = split_dirs or [root]
    return [
        scene
        for parent in search_roots
        for scene in sorted(parent.iterdir())
        if scene.is_dir() and (scene / "chunks").is_dir()
    ]


def selected_scenes(spec: DatasetSpec) -> list[Path]:
    scenes = iter_scene_dirs(spec.chunk_root)
    has_split = any(
        scene.parent != spec.chunk_root and scene.parent.name in ("train", "test", "val")
        for scene in scenes
    )
    if has_split:
        if spec.split == "train":
            return [scene for scene in scenes if scene.parent.name == "train"]
        return [scene for scene in scenes if scene.parent.name in ("test", "val")]
    split_index = int(len(scenes) * spec.split_ratio)
    return scenes[:split_index] if spec.split == "train" else scenes[split_index:]


def source_split(paths: Any) -> str:
    paths = paths.tolist() if hasattr(paths, "tolist") else paths
    paths = paths if isinstance(paths, list) else [paths]
    detected: set[str] = set()
    for value in paths:
        parts = {part.lower() for part in str(value).replace("\\", "/").split("/")}
        if parts.intersection({"train", "training"}):
            detected.add("train")
        if parts.intersection({"test", "testing", "val", "validation", "eval"}):
            detected.add("val")
    if len(detected) > 1:
        raise ValueError(f"Chunk mixes train and validation paths: {paths[:2]}")
    return next(iter(detected), "unknown")


def sidecar_path(spec: DatasetSpec, chunk: Path) -> Path | None:
    if spec.cache_root is None:
        candidate = chunk.with_suffix(".zsp.npz")
    else:
        candidate = spec.cache_root / chunk.relative_to(spec.chunk_root).with_suffix(".zsp.npz")
    return candidate if candidate.is_file() else None


def selected_chunks(spec: DatasetSpec, limit: int | None) -> list[tuple[Path, Path | None, str]]:
    selected = []
    for scene in selected_scenes(spec):
        for chunk in sorted((scene / "chunks").glob("*.npz")):
            if spec.source_split != "all":
                with np.load(chunk, allow_pickle=True) as data:
                    actual = source_split(data["image_paths"])
                if actual != spec.source_split:
                    continue
            selected.append((chunk, sidecar_path(spec, chunk), spec.name))
            if limit is not None and len(selected) >= limit:
                return selected
    return selected


def chunk_state(task: tuple[Path, Path | None, str, float]) -> tuple[str, dict[str, Any]]:
    chunk_path, cache_path, name, canonical_scale = task
    with np.load(chunk_path, allow_pickle=True) as chunk:
        source = np.load(cache_path, allow_pickle=True) if cache_path is not None else chunk
        try:
            xyz = np.asarray(source["rest_superpoint_xyz"], dtype=np.float64)
            valid = (
                np.asarray(source["rest_superpoint_valid_mask"], dtype=bool)
                if "rest_superpoint_valid_mask" in source.files
                else np.ones(xyz.shape[0], dtype=bool)
            )
        finally:
            if source is not chunk:
                source.close()
        norm_scale = float(chunk["norm_scale"]) if "norm_scale" in chunk.files else 1.0
        pose0 = np.asarray(chunk["camera_poses_c2w"][0], dtype=np.float64).copy()
    if not np.isfinite(norm_scale) or norm_scale <= 1e-8:
        raise ValueError(f"Invalid norm_scale={norm_scale} in {chunk_path}")
    if canonical_scale <= 0:
        raise ValueError(f"canonical_position_scale must be positive, got {canonical_scale}")
    valid &= np.isfinite(xyz).all(axis=-1)
    world_normalized = xyz[valid] / norm_scale
    rotation = pose0[:3, :3]
    translation = pose0[:3, 3] / norm_scale
    canonical = ((world_normalized - translation) @ rotation) / canonical_scale
    if canonical.shape[0] == 0:
        return name, {"count": 0}
    mean = canonical.mean(axis=0)
    centered = canonical - mean
    return name, {
        "count": canonical.shape[0],
        "mean": mean,
        "m2": np.square(centered).sum(axis=0),
        "cross_m2": centered.T @ centered,
        "minimum": canonical.min(axis=0),
        "maximum": canonical.max(axis=0),
        "radius_sum": float(np.linalg.norm(canonical, axis=-1).sum()),
        "radius_sq_sum": float(np.square(np.linalg.norm(canonical, axis=-1)).sum()),
    }


def save_stats(path: Path, stats: VectorStats, radius_sum: float, radius_sq_sum: float, **meta: Any) -> None:
    payload = stats.payload()
    count = max(stats.count, 1)
    radius_mean = radius_sum / count
    radius_var = max((radius_sq_sum - radius_sum * radius_sum / count) / max(count - 1, 1), 0.0)
    payload.update(
        canonical_radius_mean=np.float32(radius_mean),
        canonical_radius_std=np.float32(np.sqrt(radius_var)),
        **{key: np.array(value) for key, value in meta.items()},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    specs = train_specs(cfg)
    canonical_scale = float(dict(cfg.model).get("canonical_position_scale", 1.0))
    stats_path = Path(str(cfg.dataset["zsp_stats_path"])).expanduser().resolve()
    output = args.output or stats_path.with_name("canonical_target_stats_train_joint.npz")
    output = output.expanduser().resolve()
    per_dataset_dir = output.parent / f"{output.stem}_datasets"

    tasks: list[tuple[Path, Path | None, str, float]] = []
    for spec in specs:
        if not spec.chunk_root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {spec.chunk_root}")
        chunks = selected_chunks(spec, args.limit)
        missing = sum(cache is None for _, cache, _ in chunks)
        if missing:
            raise FileNotFoundError(
                f"{spec.name}: {missing}/{len(chunks)} selected chunks have no z_sp sidecar under {spec.cache_root}"
            )
        tasks.extend((chunk, cache, name, canonical_scale) for chunk, cache, name in chunks)
        print(f"[dataset] {spec.name}: chunks={len(chunks)} root={spec.chunk_root} cache={spec.cache_root}")

    states = {spec.name: VectorStats(3) for spec in specs}
    radius_sums = {spec.name: 0.0 for spec in specs}
    radius_sq_sums = {spec.name: 0.0 for spec in specs}
    workers = max(1, int(args.workers))
    iterator: Iterable[tuple[str, dict[str, Any]]]
    if workers == 1:
        iterator = map(chunk_state, tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(chunk_state, tasks, chunksize=8)
    try:
        for index, (name, state) in enumerate(iterator, 1):
            if int(state["count"]) > 0:
                states[name].merge(state)
                radius_sums[name] += float(state["radius_sum"])
                radius_sq_sums[name] += float(state["radius_sq_sum"])
            if args.progress_every > 0 and index % args.progress_every == 0:
                print(f"[progress] chunks={index}/{len(tasks)}")
    finally:
        if workers != 1:
            executor.shutdown()

    joint = VectorStats(3)
    joint_radius_sum = 0.0
    joint_radius_sq_sum = 0.0
    for spec in specs:
        state = states[spec.name]
        joint.merge(
            {
                "count": state.count,
                "mean": state.mean,
                "m2": state.m2,
                "cross_m2": state.cross_m2,
                "minimum": state.minimum,
                "maximum": state.maximum,
            }
        )
        joint_radius_sum += radius_sums[spec.name]
        joint_radius_sq_sum += radius_sq_sums[spec.name]
        dataset_output = per_dataset_dir / f"{spec.name}.npz"
        save_stats(
            dataset_output,
            state,
            radius_sums[spec.name],
            radius_sq_sums[spec.name],
            dataset=spec.name,
            coordinate_frame="camera0_canonical",
            canonical_position_scale=canonical_scale,
            config=str(args.config.resolve()),
        )
        print(f"[result] {spec.name}: count={state.count} mean={state.mean} std={np.sqrt(state.m2 / max(state.count - 1, 1))}")

    save_stats(
        output,
        joint,
        joint_radius_sum,
        joint_radius_sq_sum,
        datasets=np.array([spec.name for spec in specs]),
        coordinate_frame="camera0_canonical",
        canonical_position_scale=canonical_scale,
        config=str(args.config.resolve()),
    )
    print(f"[done] joint count={joint.count} mean={joint.mean} std={np.sqrt(joint.m2 / max(joint.count - 1, 1))}")
    print(f"[done] wrote {output}")
    print(f"[done] per-dataset stats {per_dataset_dir}")


if __name__ == "__main__":
    main()
