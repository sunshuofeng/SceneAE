"""Build last-frame superpoints from existing first-frame-gen chunks.

Reads 1-view chunks (first_* + rest_ + shared RGBD/cameras) and writes a
separate last_* sidecar tree. The source NPZ is never modified.

last_* is fused from the last stored frame only, using the chunk's existing
DA3 depth / pose / K / shared norm_scale. DINO + DA3 features + SD3.5 VAE
are re-extracted on that last RGB so partition features exist; geometry
still uses the stored last-frame depth so it stays on the 1-view metric.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from diffusers import AutoencoderKL
from PIL import Image

from make_gen_chunk_dl3dv import build_group
from make_superpoint_chunk_da3depth_geometry_dl3dv import (
    DEFAULT_VOXEL_SIZE,
    DinoFeatureExtractor,
    clear_cuda_cache,
    dino_dense_batch,
    is_cuda_oom,
    run_da3_depth_and_features,
    sanitize_pred_depth,
    vae_latent_dense_batch,
)
from depth_anything_3.api import DepthAnything3


CHUNK_SCHEMA_VERSION = "last_frame_gen_sp_from_existing_chunk_v1"
_SKIP_SUFFIXES = (".zsp.npz", ".s1xyz.npz", ".s1perm.npz", ".last.npz")
_REQUIRED_SOURCE_KEYS = (
    "da3_depths",
    "camera_poses_c2w",
    "camera_intrinsics",
    "norm_scale",
    "image_paths",
    "first_superpoint_xyz",
    "rest_superpoint_xyz",
)


def k_to_intr_dict(k: np.ndarray) -> dict[str, float]:
    return {
        "fx": float(k[0, 0]),
        "fy": float(k[1, 1]),
        "cx": float(k[0, 2]),
        "cy": float(k[1, 2]),
    }


def is_source_chunk_name(path: Path) -> bool:
    name = path.name
    if not name.startswith("chunk_") or not name.endswith(".npz"):
        return False
    return not name.endswith(_SKIP_SUFFIXES)


def iter_source_chunks(chunk_root: Path) -> list[Path]:
    return sorted(path for path in chunk_root.rglob("chunk_*.npz") if is_source_chunk_name(path))


def output_path_for(source_npz: Path, chunk_root: Path, out_root: Path) -> Path:
    try:
        rel = source_npz.relative_to(chunk_root)
    except ValueError:
        rel = Path(source_npz.name)
    return out_root / rel


def chunk_has_current_schema(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            if "last_superpoint_xyz" not in data.files:
                return False
            return str(data["chunk_schema_version"].item()) == CHUNK_SCHEMA_VERSION
    except (KeyError, OSError, ValueError):
        return False


class ImageReader:
    """Load last-frame RGB from a filesystem path or a RE10K `file::scene::frame` spec."""

    def __init__(self) -> None:
        self._re10k_files: OrderedDict[Path, object] = OrderedDict()

    def read(self, stored_path: str) -> np.ndarray:
        path_str = str(stored_path)
        if "::" in path_str:
            return self._read_re10k(path_str)
        color_path = Path(path_str)
        image = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {color_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _read_re10k(self, spec: str) -> np.ndarray:
        source, scene_key, frame_name = spec.split("::", 2)
        source_path = Path(source)
        if source_path not in self._re10k_files:
            try:
                payload = torch.load(source_path, map_location="cpu", weights_only=True)
            except TypeError:
                payload = torch.load(source_path, map_location="cpu")
            self._re10k_files[source_path] = payload
            while len(self._re10k_files) > 2:
                self._re10k_files.popitem(last=False)
        else:
            self._re10k_files.move_to_end(source_path)
        frame_idx = int(frame_name.rsplit("_", 1)[-1])
        for scene in self._re10k_files[source_path]:
            if str(scene.get("key", source_path.stem)) != scene_key:
                continue
            image_bytes = scene["images"][frame_idx]
            return np.asarray(
                Image.open(BytesIO(image_bytes.numpy().tobytes())).convert("RGB"),
                dtype=np.uint8,
            ).copy()
        raise KeyError(f"RE10K scene {scene_key} not found in {source_path}")


def process_chunk(
    source_npz: Path,
    out_npz: Path,
    *,
    image_reader: ImageReader,
    dino_model: DinoFeatureExtractor,
    da3_model: DepthAnything3,
    vae: AutoencoderKL,
    device: torch.device,
    norm: transforms.Compose,
    da3_feat_layer: int,
    da3_process_res: int,
    da3_process_res_method: str,
    voxel_size: float,
    min_depth: float,
    max_depth: float,
    skip_existing: bool,
    chunk_root: Path,
) -> str:
    if skip_existing and out_npz.is_file() and chunk_has_current_schema(out_npz):
        return "skipped"

    with np.load(source_npz, allow_pickle=True) as src:
        missing = [key for key in _REQUIRED_SOURCE_KEYS if key not in src.files]
        if missing:
            raise KeyError(f"Source chunk missing {missing}: {source_npz}")
        depths = src["da3_depths"].astype(np.float32)
        poses = src["camera_poses_c2w"].astype(np.float32)
        intrinsics = src["camera_intrinsics"].astype(np.float32)
        norm_scale = float(src["norm_scale"])
        paths = src["image_paths"]
        if hasattr(paths, "tolist"):
            paths = paths.tolist()
        frame_indices = (
            src["frame_indices"].astype(np.int32) if "frame_indices" in src.files else None
        )
        first_n = int(src["first_superpoint_xyz"].shape[0])
        rest_n = int(src["rest_superpoint_xyz"].shape[0])

    if depths.ndim != 3 or poses.shape[0] != depths.shape[0]:
        raise ValueError(
            f"Unexpected chunk layout depths={depths.shape} poses={poses.shape}: {source_npz}"
        )
    last_idx = int(depths.shape[0] - 1)
    last_depth = sanitize_pred_depth(depths[last_idx], min_depth=min_depth, max_depth=max_depth)
    last_pose = poses[last_idx]
    last_k = intrinsics[last_idx]
    last_path = str(paths[last_idx])
    last_frame_index = (
        int(frame_indices[last_idx]) if frame_indices is not None else last_idx
    )
    height, width = int(last_depth.shape[0]), int(last_depth.shape[1])

    rgb = image_reader.read(last_path)
    if rgb.shape[:2] != (height, width):
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)

    last_intr = k_to_intr_dict(last_k)
    # V=1 cannot Umeyama-align to the input path. Depth is taken from the
    # stored chunk; DA3 is only needed for partition features.
    _pred_depths, da3_maps = run_da3_depth_and_features(
        da3_model=da3_model,
        rgb_images_u8_320=[rgb],
        poses_c2w=last_pose[None],
        intrinsics_3x3_np=last_k[None],
        resize_hw=(height, width),
        feat_layer_id=da3_feat_layer,
        process_res=da3_process_res,
        process_res_method=da3_process_res_method,
        device=device,
        align_to_input_ext_scale=False,
    )
    del _pred_depths
    dino_maps = dino_dense_batch([rgb], norm, dino_model, device)
    vae_maps = vae_latent_dense_batch([rgb], vae, device)
    partition_features = [torch.cat([dino_maps[0], da3_maps[0], vae_maps[0]], dim=-1)]

    last_group, _last_norm = build_group(
        prefix="last_",
        depths=[last_depth],
        poses=[last_pose],
        intrs=[last_intr],
        partition_features=partition_features,
        vae_maps=vae_maps,
        device=device,
        voxel_size=voxel_size,
        min_depth=min_depth,
        max_depth=max_depth,
        norm_factor_for_geometry=norm_scale,
    )
    last_n = int(last_group["last_superpoint_xyz"].shape[0])
    try:
        rel = str(source_npz.relative_to(chunk_root))
    except ValueError:
        rel = source_npz.name

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    # numpy savez appends .npz unless the path already ends with it.
    tmp_npz = out_npz.with_name(out_npz.stem + ".writing.npz")
    np.savez_compressed(
        tmp_npz,
        last_frame_index=np.int32(last_frame_index),
        last_image_path=np.array(last_path),
        source_chunk=np.array(str(source_npz.resolve())),
        source_chunk_relpath=np.array(rel),
        source_num_views=np.int32(depths.shape[0]),
        source_num_first_superpoints=np.int32(first_n),
        source_num_rest_superpoints=np.int32(rest_n),
        num_last_superpoints_raw=np.int32(last_n),
        norm_scale=np.float32(norm_scale),
        camera_pose_c2w=last_pose.astype(np.float32),
        camera_intrinsics=last_k.astype(np.float32),
        chunk_schema_version=np.array(CHUNK_SCHEMA_VERSION),
        **last_group,
    )
    tmp_npz.replace(out_npz)
    print(
        f"[ok] {rel}  last={last_n} first={first_n} rest={rest_n}  "
        f"geom={float(last_group['last_sp_geometry_valid_mask'].mean()):.3f}  "
        f"-> {out_npz}"
    )
    return "wrote"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build last-frame superpoints from existing 1-view gen chunks."
    )
    parser.add_argument(
        "--chunk-root",
        type=Path,
        required=True,
        help="Existing 1-view first_frame_gen_chunk root. Never written.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="New directory for last_* NPZs. Mirrors the source relative layout.",
    )
    parser.add_argument("--da3-weights", type=Path, default=Path("/mnt/share/codes/kimi/ssf/DA3-BASE"))
    parser.add_argument("--da3-feat-layer", type=int, default=11)
    parser.add_argument("--da3-process-res", type=int, default=518)
    parser.add_argument("--da3-process-res-method", type=str, default="upper_bound_resize")
    parser.add_argument(
        "--vae-path",
        type=Path,
        default=Path("/mnt/share/codes/kimi/ssf/stable-diffusion-3.5-large"),
    )
    parser.add_argument("--dinov3-repo", type=Path, default=Path(__file__).resolve().parents[1] / "third_party" / "dinov3")
    parser.add_argument(
        "--dinov3-weights",
        type=Path,
        default=Path("/mnt/share/codes/kimi/ssf/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-bad-chunks", action="store_true")
    parser.add_argument("--voxel-size", type=float, default=DEFAULT_VOXEL_SIZE)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=50.0)
    parser.add_argument("--limit", type=int, default=0, help="Process at most this many source chunks (0=all).")
    parser.add_argument("--start-index", type=int, default=0, help="Skip the first N source chunks after sort.")
    args = parser.parse_args()
    if args.voxel_size <= 0:
        raise ValueError(f"voxel_size must be positive, got {args.voxel_size}")
    if args.min_depth <= 0 or args.max_depth <= args.min_depth:
        raise ValueError(f"invalid depth range: {args.min_depth}..{args.max_depth}")
    if args.limit < 0 or args.start_index < 0:
        raise ValueError("--limit and --start-index must be >= 0")
    da3_config_path = args.da3_weights / "config.json"
    if da3_config_path.is_file():
        with da3_config_path.open("r", encoding="utf-8") as handle:
            da3_config = json.load(handle)
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
    image_reader = ImageReader()

    chunks = iter_source_chunks(args.chunk_root)
    if args.start_index:
        chunks = chunks[args.start_index :]
    if args.limit > 0:
        chunks = chunks[: args.limit]
    worker_chunks = chunks[rank::world_size] if world_size > 1 else chunks
    print(f"[worker {rank}] chunks={len(worker_chunks)} / {len(chunks)}")

    wrote = skipped = failed = 0
    for source_npz in worker_chunks:
        out_npz = output_path_for(source_npz, args.chunk_root, args.out_root)
        try:
            status = process_chunk(
                source_npz,
                out_npz,
                image_reader=image_reader,
                dino_model=dino_model,
                da3_model=da3_model,
                vae=vae,
                device=device,
                norm=norm,
                da3_feat_layer=args.da3_feat_layer,
                da3_process_res=args.da3_process_res,
                da3_process_res_method=args.da3_process_res_method,
                voxel_size=args.voxel_size,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                skip_existing=args.skip_existing,
                chunk_root=args.chunk_root,
            )
            if status == "skipped":
                skipped += 1
            else:
                wrote += 1
            clear_cuda_cache(device)
        except Exception as exc:
            clear_cuda_cache(device)
            if is_cuda_oom(exc) or not args.skip_bad_chunks:
                raise
            failed += 1
            print(f"[warn] skip {source_npz}: {type(exc).__name__}: {exc}")
    print(f"[worker {rank}] done wrote={wrote} skipped={skipped} failed={failed}")


def main() -> None:
    args = parse_args()
    if not args.chunk_root.is_dir():
        raise FileNotFoundError(f"Missing --chunk-root: {args.chunk_root}")
    if args.out_root.resolve() == args.chunk_root.resolve():
        raise ValueError("--out-root must be a new directory; refusing to write into --chunk-root")
    args.out_root.mkdir(parents=True, exist_ok=True)
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    world_size = gpu_count if gpu_count > 0 else 1
    print(
        f"[init] gpus={gpu_count} world_size={world_size} "
        f"chunk_root={args.chunk_root} out_root={args.out_root}"
    )
    if world_size > 1:
        mp.spawn(run_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run_worker(0, world_size, args)


if __name__ == "__main__":
    main()
