"""
Build DL3DV chunks for first-frame conditioned DiT generation.

Each output chunk stores two superpoint groups:
  - first_*: superpoints fused from frame 0 only (condition)
  - rest_*: superpoints fused from all frames in the chunk (target scene)

Frame-level DA3 depth, camera poses, intrinsics, and image paths are shared.
The schema is aligned with dit/dataset/first_frame_gen_chunkv1.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from diffusers import AutoencoderKL

from make_superpoint_chunk_da3depth_geometry_dl3dv import (
    DEFAULT_VOXEL_SIZE,
    DinoFeatureExtractor,
    clear_cuda_cache,
    compute_superpoint_geometry_stats,
    dino_dense_batch,
    fuse_and_superpoint_chunk,
    intr_dict_to_3x3,
    is_cuda_oom,
    iter_scenes,
    load_dl3dv_metadata,
    print_chunk_diagnostics,
    run_da3_depth_and_features,
    sanitize_pred_depth,
    scale_intrinsics,
    vae_latent_dense_batch,
)
from depth_anything_3.api import DepthAnything3


DEFAULT_FRAMES_PER_CHUNK = 9
DEFAULT_FRAME_STRIDE = 3
CHUNK_SCHEMA_VERSION = "first_frame_gen_sp_dl3dv_da3base_vae_v2_intrinsics"


def chunk_has_current_schema(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            return str(data["chunk_schema_version"].item()) == CHUNK_SCHEMA_VERSION
    except (KeyError, OSError, ValueError):
        return False


def build_sequential_chunks(frames: list[dict], frames_per_chunk: int, frame_stride: int) -> list[list[dict]]:
    chunks: list[list[dict]] = []
    if not frames:
        return chunks
    span = 1 + frame_stride * (frames_per_chunk - 1)
    for start in range(0, len(frames) - span + 1, span):
        chunks.append([frames[start + frame_stride * j] for j in range(frames_per_chunk)])
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
        voxel_points_world,
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
        points_world=voxel_points_world,
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
    voxel_size: float,
    min_depth: float,
    max_depth: float,
    frames_per_chunk: int,
    frame_stride: int,
    image_subdir: str,
    skip_bad_chunks: bool,
) -> None:
    try:
        frames, intr = load_dl3dv_metadata(scene_dir, image_subdir=image_subdir)
    except Exception as exc:
        if not skip_bad_chunks:
            raise
        print(f"[skip] bad DL3DV metadata in {scene_dir.name}: {type(exc).__name__}: {exc}")
        return

    chunk_frames = build_sequential_chunks(
        frames,
        frames_per_chunk=frames_per_chunk,
        frame_stride=frame_stride,
    )
    if len(chunk_frames) == 0:
        print(f"[skip] no chunks in {scene_dir.name}")
        return

    out_scene = out_root / scene_dir.name / "chunks"
    out_scene.mkdir(parents=True, exist_ok=True)

    chunk_id = 0
    for frames_chunk in chunk_frames:
        color_paths = [f["image_path"] for f in frames_chunk]
        stems = [p.stem for p in color_paths]
        frame_ids = [int(f["frame_index"]) for f in frames_chunk]
        out_npz = out_scene / f"chunk_{chunk_id:06d}_{stems[0]}_{stems[-1]}.npz"
        chunk_id += 1

        if skip_existing and out_npz.is_file():
            if chunk_has_current_schema(out_npz):
                continue
            print(f"[{scene_dir.name}] rebuilding stale chunk schema: {out_npz.name}")

        try:
            poses: list[np.ndarray] = [f["pose_c2w"] for f in frames_chunk]
            raw_poses: list[np.ndarray] = [f["raw_pose_c2w"] for f in frames_chunk]
            skip_chunk = False

            colors: list[np.ndarray] = []
            intrs: list[dict] = []
            paths_str: list[str] = [str(p.resolve()) for p in color_paths]

            for cp in color_paths:
                bgr = cv2.imread(str(cp), cv2.IMREAD_COLOR)
                if bgr is None:
                    if not skip_bad_chunks:
                        raise FileNotFoundError(f"Failed to read image: {cp}")
                    skip_chunk = True
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                orig_h, orig_w = rgb.shape[:2]
                calibration_aspect = float(intr["width"]) / float(intr["height"])
                decoded_aspect = float(orig_w) / float(orig_h)
                if abs(decoded_aspect / calibration_aspect - 1.0) > 1e-3:
                    raise ValueError(
                        f"Image/calibration aspect mismatch for {cp}: "
                        f"image={orig_w}x{orig_h}, calibration={intr['width']}x{intr['height']}"
                    )
                rgb = cv2.resize(rgb, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
                intr_u = scale_intrinsics(intr, resize_w, resize_h)
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
            vae_maps = vae_latent_dense_batch(colors, vae, device)

            partition_features: list[torch.Tensor] = []
            for vi, _rgb in enumerate(colors):
                partition_features.append(torch.cat([dino_maps[vi], da3_maps[vi], vae_maps[vi]], dim=-1))

            try:
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
            except RuntimeError as exc:
                if "No valid points in chunk" in str(exc):
                    print_chunk_diagnostics(
                        scene_name=scene_dir.name,
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

            first_n = int(first_group["first_superpoint_xyz"].shape[0])
            rest_n = int(rest_group["rest_superpoint_xyz"].shape[0])
            np.savez_compressed(
                out_npz,
                da3_depths=np.stack(pred_depths, axis=0).astype(np.float16),
                camera_poses_c2w=np.stack(poses, axis=0).astype(np.float32),
                camera_intrinsics=np.stack([intr_dict_to_3x3(k) for k in intrs], axis=0).astype(np.float32),
                camera_intrinsics_source=np.array(intr["source"]),
                camera_intrinsics_source_hw=np.array([intr["height"], intr["width"]], dtype=np.int32),
                camera_intrinsics_target_hw=np.array([resize_h, resize_w], dtype=np.int32),
                norm_scale=np.float32(norm_factor),
                image_paths=np.array(paths_str, dtype=object),
                frame_indices=np.array(frame_ids, dtype=np.int32),
                chunk_base_index=np.int32(frame_ids[0]),
                num_first_superpoints_raw=np.int32(first_n),
                num_rest_superpoints_raw=np.int32(rest_n),
                chunk_schema_version=np.array(CHUNK_SCHEMA_VERSION),
                **first_group,
                **rest_group,
            )
            print(
                f"[{scene_dir.name}] wrote {out_npz.name}  "
                f"(first={first_n}, rest={rest_n}, "
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build DL3DV first-frame generation superpoint chunks.")
    p.add_argument("--data-root", type=Path, default=Path("/mnt/share/codes/kimi/ssf_dataset/DL3DV/DL3DV/10K"))
    p.add_argument("--out-root", type=Path, default=Path("/mnt/share/codes/kimi/ssf_dataset/GEN/first_frame_gen_chunk"))
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
    p.add_argument("--frame-stride", type=int, default=DEFAULT_FRAME_STRIDE)
    p.add_argument("--image-subdir", type=str, default="images_8")
    p.add_argument("--voxel-size", type=float, default=DEFAULT_VOXEL_SIZE)
    p.add_argument("--min-depth", type=float, default=0.1)
    p.add_argument("--max-depth", type=float, default=50.0)
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

    scenes = list(iter_scenes(args.data_root))
    worker_scenes = scenes[rank::world_size] if world_size > 1 else scenes
    print(f"[worker {rank}] scenes={len(worker_scenes)}")

    for scene_dir in worker_scenes:
        print(f"[worker {rank}] processing {scene_dir.name}")
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
            voxel_size=args.voxel_size,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            frames_per_chunk=args.frames_per_chunk,
            frame_stride=args.frame_stride,
            image_subdir=args.image_subdir,
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
