"""
Build RealEstate10K chunks for first-frame conditioned DiT generation.

Input is the pixelSplat-style RE10K .torch format used by
make_superpoint_chunk_da3depth_geometry_re10k.py. Each output chunk stores:
  - first_*: superpoints fused from frame 0 only (condition)
  - rest_*: superpoints fused from all frames in the chunk (target scene)
"""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from diffusers import AutoencoderKL
from PIL import Image

from make_superpoint_chunk_da3depth_geometry_re10k import (
    DEFAULT_VOXEL_SIZE,
    DinoFeatureExtractor,
    clear_cuda_cache,
    compute_superpoint_geometry_stats,
    dino_dense_batch,
    fuse_and_superpoint_chunk,
    intr_dict_to_3x3,
    is_cuda_oom,
    pose_is_finite,
    print_chunk_diagnostics,
    run_da3_depth_and_features,
    safe_path_name,
    sanitize_pred_depth,
    vae_latent_dense_batch,
)
from depth_anything_3.api import DepthAnything3


DEFAULT_FRAMES_PER_CHUNK = 25
DEFAULT_FRAME_STRIDE = 16
CHUNK_SCHEMA_VERSION = "first_frame_gen_sp_re10k_da3base_vae_v1"


def iter_re10k_files(data_root: Path, split: str) -> Iterable[Path]:
    split_root = data_root / split
    if not split_root.is_dir() and data_root.name == split:
        split_root = data_root
    if not split_root.is_dir():
        raise FileNotFoundError(f"Missing RE10K split directory: {split_root}")
    yield from sorted(split_root.glob("*.torch"))


def build_sequential_chunks(frames: list[dict], frames_per_chunk: int, frame_stride: int) -> list[list[dict]]:
    chunks: list[list[dict]] = []
    if not frames:
        return chunks
    span = 1 + frame_stride * (frames_per_chunk - 1)
    for start in range(0, len(frames) - span + 1, span):
        chunks.append([frames[start + frame_stride * j] for j in range(frames_per_chunk)])
    return chunks


def decode_re10k_image(image_bytes: torch.Tensor) -> np.ndarray:
    pil = Image.open(BytesIO(image_bytes.numpy().tobytes())).convert("RGB")
    return np.asarray(pil)


def re10k_cameras_to_poses_intrinsics(cameras: torch.Tensor, image_hw: tuple[int, int]) -> tuple[list[np.ndarray], list[dict]]:
    cam = cameras.detach().cpu().float()
    n = int(cam.shape[0])
    h, w = image_hw

    intrs: list[dict] = []
    for fx, fy, cx, cy in cam[:, :4].numpy():
        intrs.append(
            {
                "fx": float(fx) * float(w),
                "fy": float(fy) * float(h),
                "cx": float(cx) * float(w),
                "cy": float(cy) * float(h),
            }
        )

    w2c = torch.eye(4, dtype=torch.float32).repeat(n, 1, 1)
    w2c[:, :3] = cam[:, 6:].reshape(n, 3, 4)
    c2w = torch.linalg.inv(w2c).numpy().astype(np.float64)
    return [c2w[i] for i in range(n)], intrs


def build_re10k_frames(scene: dict, min_num_frames: int = 1) -> list[dict]:
    images = scene["images"]
    cameras = scene["cameras"]
    timestamps = scene.get("timestamps")
    if len(images) < min_num_frames:
        return []
    first_rgb = decode_re10k_image(images[0])
    orig_h, orig_w = first_rgb.shape[:2]
    poses, intrs = re10k_cameras_to_poses_intrinsics(cameras, image_hw=(orig_h, orig_w))

    frames: list[dict] = []
    for i, (image_bytes, pose, intr) in enumerate(zip(images, poses, intrs, strict=True)):
        if not pose_is_finite(pose):
            continue
        ts = int(timestamps[i]) if timestamps is not None else i
        frames.append(
            {
                "image_bytes": image_bytes,
                "pose_c2w": pose,
                "raw_pose_c2w": pose,
                "intrinsic": intr,
                "frame_index": i,
                "timestamp": ts,
            }
        )
    frames.sort(key=lambda x: x["frame_index"])
    return frames


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
    scene: dict,
    source_file: Path,
    split: str,
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
    skip_bad_chunks: bool,
    only_frame_groups: list[tuple[int, list[int]]] | None = None,
) -> None:
    scene_name = str(scene.get("key", source_file.stem))
    scene_dir_name = safe_path_name(scene_name)
    try:
        frames = build_re10k_frames(
            scene,
            min_num_frames=1 if only_frame_groups is not None else frames_per_chunk,
        )
    except Exception as exc:
        if not skip_bad_chunks:
            raise
        print(f"[skip] bad RE10K scene metadata in {scene_name}: {type(exc).__name__}: {exc}")
        return

    if only_frame_groups is not None:
        by_index = {int(frame["frame_index"]): frame for frame in frames}
        enumerated: list[tuple[int, list[dict]]] = []
        for chunk_id, indices in only_frame_groups:
            missing = [int(index) for index in indices if int(index) not in by_index]
            if missing:
                raise KeyError(f"{scene_name} is missing frame indices {missing}")
            enumerated.append((int(chunk_id), [by_index[int(index)] for index in indices]))
    else:
        enumerated = list(
            enumerate(
                build_sequential_chunks(
                    frames,
                    frames_per_chunk=frames_per_chunk,
                    frame_stride=frame_stride,
                )
            )
        )
    if len(enumerated) == 0:
        print(f"[skip] no chunks in {scene_name}")
        return

    out_scene = out_root / split / scene_dir_name / "chunks"
    out_scene.mkdir(parents=True, exist_ok=True)

    for this_chunk_id, frames_chunk in enumerated:
        stems = [f"frame_{int(f['frame_index']):06d}" for f in frames_chunk]
        frame_ids = [int(f["frame_index"]) for f in frames_chunk]
        out_npz = out_scene / f"chunk_{this_chunk_id:06d}_{stems[0]}_{stems[-1]}.npz"

        if skip_existing and out_npz.is_file():
            continue

        try:
            poses: list[np.ndarray] = [f["pose_c2w"] for f in frames_chunk]
            raw_poses: list[np.ndarray] = [f["raw_pose_c2w"] for f in frames_chunk]
            skip_chunk = False

            colors: list[np.ndarray] = []
            intrs: list[dict] = []
            paths_str: list[str] = [
                f"{source_file.resolve()}::{scene_name}::{stem}"
                for stem in stems
            ]

            for frame in frames_chunk:
                try:
                    rgb = decode_re10k_image(frame["image_bytes"])
                except Exception as exc:
                    if not skip_bad_chunks:
                        raise RuntimeError(f"Failed to decode RE10K image {scene_name} frame {frame['frame_index']}") from exc
                    skip_chunk = True
                    break
                orig_h, orig_w = rgb.shape[:2]
                rgb = cv2.resize(rgb, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
                sw = resize_w / max(orig_w, 1)
                sh = resize_h / max(orig_h, 1)
                intr = frame["intrinsic"]
                intr_u = {
                    "fx": intr["fx"] * sw,
                    "fy": intr["fy"] * sh,
                    "cx": intr["cx"] * sw,
                    "cy": intr["cy"] * sh,
                }
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
                        scene_name=scene_name,
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
                norm_scale=np.float32(norm_factor),
                image_paths=np.array(paths_str, dtype=object),
                source_file=np.array(str(source_file.resolve())),
                scene_key=np.array(scene_name),
                timestamps=np.array([int(f["timestamp"]) for f in frames_chunk], dtype=np.int64),
                frame_indices=np.array(frame_ids, dtype=np.int32),
                chunk_base_index=np.int32(frame_ids[0]),
                num_first_superpoints_raw=np.int32(first_n),
                num_rest_superpoints_raw=np.int32(rest_n),
                chunk_schema_version=np.array(CHUNK_SCHEMA_VERSION),
                **first_group,
                **rest_group,
            )
            print(
                f"[{scene_name}] wrote {out_npz.name}  "
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
                f"[warn] skip chunk {scene_name} {out_npz.name} "
                f"(base={frame_ids[0]}, frames={stems[0]}..{stems[-1]}): {type(exc).__name__}: {exc}"
            )
            continue


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build RE10K first-frame generation superpoint chunks.")
    p.add_argument("--data-root", type=Path, default=Path("/mnt/share/codes/kimi/ssf_dataset/RE10K/re10k"))
    p.add_argument("--split", type=str, default="train", choices=("train", "test"))
    p.add_argument("--out-root", type=Path, default=Path("/mnt/share/codes/kimi/ssf_dataset/RE10K/first_frame_gen_chunk"))
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

    re10k_files = list(iter_re10k_files(args.data_root, split=args.split))
    worker_files = re10k_files[rank::world_size] if world_size > 1 else re10k_files
    print(f"[worker {rank}] files={len(worker_files)}")

    for source_file in worker_files:
        print(f"[worker {rank}] loading {source_file.name}")
        try:
            scenes = torch.load(source_file, map_location="cpu", weights_only=True)
        except TypeError:
            scenes = torch.load(source_file, map_location="cpu")
        for scene in scenes:
            print(f"[worker {rank}] processing {source_file.name}::{scene.get('key', '<missing-key>')}")
            process_scene(
                scene=scene,
                source_file=source_file,
                split=args.split,
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
