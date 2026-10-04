"""
Build ScanNet chunks for first-frame conditioned DiT generation.  (v2)

v2 fix vs make_gen_chunk_scannet.py
------------------------------------
The v1 FFG ScanNet chunks have ~92% zeroed DA3 depth (sparse, far-biased
superpoints).  Root cause (verified empirically on scene0001_00, 9 frames):

  DA3-BASE       -> raw depth median 24.3m, 92.4% >= 10m  -> sanitized valid 7.6%
  DA3-LARGE-1.1  -> raw depth median  8.8m, 36.1% >= 10m  -> sanitized valid 63.9%

The v1 generation shell overrode --da3-weights to DA3-BASE, which miscalibrates
indoor ScanNet metric depth.  The dense C15 chunks were produced with
DA3-LARGE-1.1.  v2 therefore:

  * defaults --da3-weights to the local DA3-LARGE-1.1 and --da3-feat-layer to 23
  * fixes stale /home/shuofeng and /data1,/data2 paths to this machine
  * writes to a separate out-root (scannet_first_frame_gen_chunk_v2)
  * adds a per-chunk depth-valid-ratio guardrail + logging so the fix is
    verifiable and degenerate chunks can be skipped
  * bumps CHUNK_SCHEMA_VERSION so stale v2 chunks are rebuilt

Each output chunk stores two superpoint groups:
  - first_*: superpoints fused from frame 0 only (condition)
  - rest_*: superpoints fused from all frames in the chunk (target scene)

Input layout follows make_superpoint_chunk_da3depth_geometry_scannet.py:
scene*/color, scene*/pose, scene*/intrinsic/intrinsic_color.txt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from diffusers import AutoencoderKL

from make_superpoint_chunk_da3depth_geometry_scannet import (
    DEFAULT_VOXEL_SIZE,
    DinoFeatureExtractor,
    clear_cuda_cache,
    compute_superpoint_geometry_stats,
    dino_dense_batch,
    fuse_and_superpoint_chunk,
    intr_dict_to_3x3,
    is_cuda_oom,
    load_intrinsic,
    load_pose_matrix,
    pose_is_finite,
    run_da3_depth_and_features,
    sanitize_pred_depth,
    scale_intrinsics,
    vae_latent_dense_batch,
)
from depth_anything_3.api import DepthAnything3


INNER_FRAME_STEP = 10
DEFAULT_FRAMES_PER_CHUNK = 9
CHUNK_SCHEMA_VERSION = "first_frame_gen_sp_scannet_da3_vae_v3_da3large"


def chunk_has_current_schema(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            return str(data["chunk_schema_version"].item()) == CHUNK_SCHEMA_VERSION
    except (KeyError, OSError, ValueError):
        return False


def iter_scenes(data_root: Path) -> Iterable[Path]:
    for d in sorted(data_root.iterdir()):
        if d.is_dir() and d.name.startswith("scene"):
            yield d


def build_strict_chunks(color_files: list[Path], frames_per_chunk: int) -> list[list[Path]]:
    id_to_path: dict[int, Path] = {}
    for p in color_files:
        try:
            frame_id = int(p.stem)
        except ValueError:
            continue
        id_to_path[frame_id] = p

    chunks: list[list[Path]] = []
    used: set[int] = set()
    frame_ids = sorted(id_to_path.keys())
    if not frame_ids:
        return chunks

    first_base = ((frame_ids[0] + INNER_FRAME_STEP - 1) // INNER_FRAME_STEP) * INNER_FRAME_STEP
    last_base = frame_ids[-1]
    for base_id in range(first_base, last_base + 1, INNER_FRAME_STEP):
        if base_id in used:
            continue
        candidate = [base_id + INNER_FRAME_STEP * j for j in range(frames_per_chunk)]
        if any(fid not in id_to_path for fid in candidate):
            continue
        if any(fid in used for fid in candidate):
            continue
        chunks.append([id_to_path[fid] for fid in candidate])
        used.update(candidate)
    return chunks


def prefixed_geometry(prefix: str, geometry: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {f"{prefix}{k}": v for k, v in geometry.items()}


def build_group(
    prefix: str,
    depths: list[np.ndarray],
    poses: list[np.ndarray],
    intrs: list[dict],
    partition_features: list[torch.Tensor],
    vae_maps: list[torch.Tensor],
    device: torch.device,
    voxel_size: float,
    min_depth: float,
    max_depth: float,
    norm_factor_for_geometry: float | None = None,
) -> tuple[dict[str, np.ndarray], float]:
    (
        sp_xyz,
        sp_idx,
        points_world,
        _voxel_coords,
        sp_vae_features,
        _prevoxel_points_world,
        _prevoxel_sp_idx,
        _sp_nearest_prevoxel_xyz,
        _sp_nearest_prevoxel_index,
        _sp_nearest_prevoxel_dist,
        _sp_nearest_prevoxel_valid,
        group_norm_factor,
    ) = fuse_and_superpoint_chunk(
        depths,
        poses,
        intrs,
        partition_features,
        vae_maps,
        device,
        voxel_size,
        min_depth=min_depth,
        max_depth=max_depth,
    )

    geom_norm = group_norm_factor if norm_factor_for_geometry is None else float(norm_factor_for_geometry)
    geometry = compute_superpoint_geometry_stats(
        points_world=points_world,
        sp_idx=sp_idx,
        sp_xyz_raw=sp_xyz,
        norm_factor=geom_norm,
        device=device,
    )
    n_sp = int(sp_xyz.shape[0])
    out = {
        f"{prefix}superpoint_xyz": sp_xyz.astype(np.float32),
        f"{prefix}superpoint_valid_mask": np.ones((n_sp,), dtype=bool),
        f"{prefix}superpoint_vae_features": sp_vae_features.astype(np.float16),
        **prefixed_geometry(prefix, geometry),
    }
    return out, group_norm_factor


def process_scene(
    scene_dir: Path,
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
    skip_bad_chunks: bool,
    voxel_size: float,
    min_depth: float,
    max_depth: float,
    frames_per_chunk: int,
    min_depth_valid_ratio: float,
) -> list[float]:
    color_dir = scene_dir / "color"
    pose_dir = scene_dir / "pose"
    if not (color_dir.is_dir() and pose_dir.is_dir()):
        print(f"[skip] missing dirs in {scene_dir}")
        return []

    try:
        intr = load_intrinsic(scene_dir, intrinsic_file="intrinsic_color.txt")
    except Exception as exc:
        if not skip_bad_chunks:
            raise
        print(f"[skip] bad intrinsic in {scene_dir.name}: {type(exc).__name__}: {exc}")
        return []

    color_files = sorted(color_dir.glob("*.jpg"))
    if not color_files:
        color_files = sorted(color_dir.glob("*.png"))
    if len(color_files) == 0:
        return []
    chunk_color_paths = build_strict_chunks(color_files, frames_per_chunk=frames_per_chunk)
    if len(chunk_color_paths) == 0:
        print(f"[skip] no strict chunks in {scene_dir.name}")
        return []

    out_scene = out_root / scene_dir.name / "chunks"
    out_scene.mkdir(parents=True, exist_ok=True)

    valid_ratios: list[float] = []
    chunk_id = 0
    for color_paths in chunk_color_paths:
        stems = [p.stem for p in color_paths]
        frame_ids = [int(s) for s in stems]
        out_npz = out_scene / f"chunk_{chunk_id:06d}_{stems[0]}_{stems[-1]}.npz"
        chunk_id += 1

        if skip_existing and out_npz.is_file():
            if chunk_has_current_schema(out_npz):
                continue
            print(f"[{scene_dir.name}] rebuilding stale chunk schema: {out_npz.name}")

        try:
            poses: list[np.ndarray] = []
            skip_chunk = False
            for cp in color_paths:
                pp = pose_dir / f"{cp.stem}.txt"
                if not pp.is_file():
                    skip_chunk = True
                    break
                p = load_pose_matrix(pp)
                if not pose_is_finite(p):
                    skip_chunk = True
                    break
                poses.append(p)
            if skip_chunk:
                continue

            colors: list[np.ndarray] = []
            intrs: list[dict] = []
            paths_str: list[str] = [str(p.resolve()) for p in color_paths]
            source_hw: tuple[int, int] | None = None

            for cp in color_paths:
                bgr = cv2.imread(str(cp), cv2.IMREAD_COLOR)
                if bgr is None:
                    skip_chunk = True
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                orig_h, orig_w = rgb.shape[:2]
                if source_hw is None:
                    source_hw = (orig_h, orig_w)
                elif source_hw != (orig_h, orig_w):
                    raise ValueError(
                        f"Mixed color resolutions in one chunk: {source_hw} vs {(orig_h, orig_w)}"
                    )
                rgb = cv2.resize(rgb, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
                intr_u = scale_intrinsics(intr, orig_w, orig_h, resize_w, resize_h)
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

            pred_depths = [sanitize_pred_depth(d, min_depth=min_depth, max_depth=max_depth) for d in pred_depths]

            # --- v2 guardrail: per-chunk depth validity ---------------------
            depth_valid_ratio = float(np.mean([float(np.mean(d > 0)) for d in pred_depths]))
            valid_ratios.append(depth_valid_ratio)
            if depth_valid_ratio < min_depth_valid_ratio:
                print(
                    f"[warn] skip {scene_dir.name} {out_npz.name}: "
                    f"depth_valid_ratio={depth_valid_ratio:.3f} < {min_depth_valid_ratio:.3f}"
                )
                continue
            if depth_valid_ratio < 0.30:
                print(f"[warn] low depth validity {scene_dir.name} {out_npz.name}: {depth_valid_ratio:.3f}")
            # ----------------------------------------------------------------

            dino_maps = dino_dense_batch(colors, norm, dino_model, device)
            vae_maps = vae_latent_dense_batch(colors, vae, device)

            partition_features: list[torch.Tensor] = []
            for vi, _rgb in enumerate(colors):
                partition_features.append(torch.cat([dino_maps[vi], da3_maps[vi], vae_maps[vi]], dim=-1))

            rest_group, norm_factor = build_group(
                prefix="rest_",
                depths=pred_depths,
                poses=poses,
                intrs=intrs,
                partition_features=partition_features,
                vae_maps=vae_maps,
                device=device,
                voxel_size=voxel_size,
                min_depth=min_depth,
                max_depth=max_depth,
            )
            first_group, _first_norm_factor = build_group(
                prefix="first_",
                depths=pred_depths[:1],
                poses=poses[:1],
                intrs=intrs[:1],
                partition_features=partition_features[:1],
                vae_maps=vae_maps[:1],
                device=device,
                voxel_size=voxel_size,
                min_depth=min_depth,
                max_depth=max_depth,
                norm_factor_for_geometry=norm_factor,
            )

            first_n = int(first_group["first_superpoint_xyz"].shape[0])
            rest_n = int(rest_group["rest_superpoint_xyz"].shape[0])
            np.savez_compressed(
                out_npz,
                da3_depths=np.stack(pred_depths, axis=0).astype(np.float16),
                camera_poses_c2w=np.stack(poses, axis=0).astype(np.float32),
                camera_intrinsics=np.stack([intr_dict_to_3x3(k) for k in intrs], axis=0).astype(np.float32),
                camera_intrinsics_source=np.array(intr["source"]),
                camera_intrinsics_source_hw=np.array(source_hw, dtype=np.int32),
                camera_intrinsics_target_hw=np.array([resize_h, resize_w], dtype=np.int32),
                norm_scale=np.float32(norm_factor),
                image_paths=np.array(paths_str, dtype=object),
                frame_indices=np.array(frame_ids, dtype=np.int32),
                chunk_base_index=np.int32(frame_ids[0]),
                num_first_superpoints_raw=np.int32(first_n),
                num_rest_superpoints_raw=np.int32(rest_n),
                depth_valid_ratio=np.float32(depth_valid_ratio),
                chunk_schema_version=np.array(CHUNK_SCHEMA_VERSION),
                **first_group,
                **rest_group,
            )
            print(
                f"[{scene_dir.name}] wrote {out_npz.name}  "
                f"(first={first_n}, rest={rest_n}, depth_valid={depth_valid_ratio:.3f}, "
                f"first_geom={float(first_group['first_sp_geometry_valid_mask'].mean()):.3f}, "
                f"rest_geom={float(rest_group['rest_sp_geometry_valid_mask'].mean()):.3f})"
            )
            clear_cuda_cache(device)
        except Exception as exc:
            clear_cuda_cache(device)
            if is_cuda_oom(exc) or not skip_bad_chunks:
                raise
            print(
                f"[warn] skip chunk {scene_dir.name} {out_npz.name} "
                f"(base={frame_ids[0]}, frames={stems[0]}..{stems[-1]}): {type(exc).__name__}: {exc}"
            )
            continue

    if valid_ratios:
        arr = np.asarray(valid_ratios, dtype=np.float64)
        print(
            f"[{scene_dir.name}] depth_valid_ratio over {len(valid_ratios)} chunks: "
            f"mean={arr.mean():.3f} min={arr.min():.3f} p10={np.percentile(arr,10):.3f}"
        )
    return valid_ratios


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ScanNet first-frame generation superpoint chunks (v2, DA3-LARGE).")
    p.add_argument("--data-root", type=Path, default=Path("/mnt/share/codes/kimi/ssf_dataset/scannet_rgb_depth/train"))
    p.add_argument("--out-root", type=Path, default=Path("/mnt/share/codes/kimi/ssf_dataset/scannet_first_frame_gen_chunk_v2"))
    p.add_argument("--resize", type=int, default=None, help="Deprecated square resize fallback. Prefer --resize-w/--resize-h.")
    p.add_argument("--resize-w", type=int, default=518)
    p.add_argument("--resize-h", type=int, default=378)
    p.add_argument("--da3-weights", type=Path, default=Path("/mnt/share/codes/kimi/ssf/DA3-LARGE-1.1"))
    p.add_argument("--da3-feat-layer", type=int, default=23)
    p.add_argument("--da3-process-res", type=int, default=518)
    p.add_argument("--da3-process-res-method", type=str, default="upper_bound_resize")
    p.add_argument("--vae-path", type=Path, default=Path("/mnt/share/codes/kimi/ssf/stable-diffusion-3.5-large"))
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
    p.add_argument("--voxel-size", type=float, default=DEFAULT_VOXEL_SIZE)
    p.add_argument("--min-depth", type=float, default=0.1)
    p.add_argument("--max-depth", type=float, default=10.0)
    p.add_argument(
        "--min-depth-valid-ratio",
        type=float,
        default=0.0,
        help="Skip chunks whose sanitized DA3 depth valid ratio is below this. "
             "0 disables skipping (warn only). DA3-LARGE typically gives ~0.6-0.9.",
    )
    p.add_argument("--limit-scenes", type=int, default=0, help="Debug: only process first N scenes per worker.")
    args = p.parse_args()
    if args.resize is not None:
        args.resize_w = int(args.resize)
        args.resize_h = int(args.resize)
    if args.resize_w <= 0 or args.resize_h <= 0:
        raise ValueError(f"resize_w/resize_h must be positive, got {args.resize_w}x{args.resize_h}")
    if args.frames_per_chunk <= 0:
        raise ValueError(f"frames_per_chunk must be positive, got {args.frames_per_chunk}")
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

    scenes = list(iter_scenes(args.data_root))
    worker_scenes = scenes[rank::world_size] if world_size > 1 else scenes
    if args.limit_scenes > 0:
        worker_scenes = worker_scenes[: args.limit_scenes]
    print(f"[worker {rank}] scenes={len(worker_scenes)}")

    all_ratios: list[float] = []
    for scene_dir in worker_scenes:
        print(f"[worker {rank}] processing {scene_dir.name}")
        all_ratios.extend(
            process_scene(
                scene_dir=scene_dir,
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
                skip_bad_chunks=args.skip_bad_chunks,
                voxel_size=args.voxel_size,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                frames_per_chunk=args.frames_per_chunk,
                min_depth_valid_ratio=args.min_depth_valid_ratio,
            )
        )

    if all_ratios:
        arr = np.asarray(all_ratios, dtype=np.float64)
        print(
            f"[worker {rank}] GLOBAL depth_valid_ratio over {len(all_ratios)} chunks: "
            f"mean={arr.mean():.3f} min={arr.min():.3f} p10={np.percentile(arr,10):.3f} "
            f"frac<0.3={float((arr<0.3).mean()):.3f}"
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
