from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

seed = 42
work_dir = str(ROOT / "out" / "dit_v18_v3_2view_stage1_position_gld_ddp")
model_module = "models.stage1_2view"
model_class = "TwoStagePositionDiTV18V3TwoView"
model_requires_xyz_stats = True

CACHE_ROOT = Path("/mnt/share/codes/kimi/ssf_dataset/first_frame_gen_cache_new_3000")
LAST_ZSP_ROOT = Path("/mnt/share/codes/kimi/ssf_dataset/first_frame_gen_cache_last_zsp")
JOINT_ZSP_STATS = str(CACHE_ROOT / "stats" / "zsp_xyz_stats_train_joint.npz")
CANONICAL_XYZ_STATS = CACHE_ROOT / "stats" / "canonical_target_stats_train_joint.npz"
if not CANONICAL_XYZ_STATS.is_file():
    raise FileNotFoundError(
        f"Missing v18_v3_2view canonical statistics: {CANONICAL_XYZ_STATS}. Run: "
        "python tools/compute_canonical_stats.py configs/stage1.py"
    )
with np.load(CANONICAL_XYZ_STATS) as _canonical_stats:
    CANONICAL_XYZ_MEAN = tuple(float(value) for value in _canonical_stats["canonical_xyz_mean"])
    CANONICAL_XYZ_STD = tuple(float(value) for value in _canonical_stats["canonical_xyz_std"])

# 2-view cond = cat(first, last). Each side pads to per_view, so cond length is 6000.
# 1-view zsp / xyz stats stay the same so target space does not fork.
dataset = dict(
    split="train",
    split_ratio=1.0,
    resize_w=518,
    resize_h=378,
    num_views=None,
    max_views=9,
    cond_max_superpoints=3000,
    cond_max_superpoints_per_view=3000,
    target_max_superpoints=3000,
    include_geometry=True,
    zsp_stats_path=JOINT_ZSP_STATS,
    zsp_sidecar=True,
    use_index_cache=True,
    index_cache_dir=str(ROOT / "out" / "index_cache"),
    return_v1=True,
    load_images=False,
    datasets=[
        dict(
            superpoint_root="/mnt/share/codes/kimi/ssf_dataset/DL3DV/first_frame_gen_chunk_new",
            zsp_cache_root=str(CACHE_ROOT / "dl3dv" / "train"),
            last_superpoint_root="/mnt/share/codes/kimi/ssf_dataset/DL3DV/first_frame_gen_chunk_last_new",
            last_zsp_cache_root=str(LAST_ZSP_ROOT / "dl3dv" / "train"),
        ),
        dict(
            superpoint_root="/mnt/share/codes/kimi/ssf_dataset/RE10K/first_frame_gen_chunk/train",
            zsp_cache_root=str(CACHE_ROOT / "re10k" / "train"),
            last_superpoint_root="/mnt/share/codes/kimi/ssf_dataset/RE10K/first_frame_gen_chunk_last/train",
            last_zsp_cache_root=str(LAST_ZSP_ROOT / "re10k" / "train"),
            split="train",
            split_ratio=1.0,
        ),
        dict(
            superpoint_root="/mnt/share/codes/kimi/ssf_dataset/scannet_first_frame_gen_chunk_new",
            zsp_cache_root=str(CACHE_ROOT / "scannet"),
            last_superpoint_root="/mnt/share/codes/kimi/ssf_dataset/scannet_first_frame_gen_chunk_last_new",
            last_zsp_cache_root=str(LAST_ZSP_ROOT / "scannet"),
            source_split="train",
        ),
    ],
)

val_dataset = dict(
    dataset,
    split="val",
    datasets=[
        dict(
            superpoint_root="/mnt/share/codes/kimi/ssf_dataset/DL3DV/first_frame_gen_chunk_eval_new",
            zsp_cache_root=str(CACHE_ROOT / "dl3dv" / "val"),
            last_superpoint_root="/mnt/share/codes/kimi/ssf_dataset/DL3DV/first_frame_gen_chunk_last_eval_new",
            last_zsp_cache_root=str(LAST_ZSP_ROOT / "dl3dv" / "val"),
            split="train",
            split_ratio=1.0,
        ),
        dict(
            superpoint_root="/mnt/share/codes/kimi/ssf_dataset/RE10K/first_frame_gen_chunk/test",
            zsp_cache_root=str(CACHE_ROOT / "re10k" / "val"),
            last_superpoint_root="/mnt/share/codes/kimi/ssf_dataset/RE10K/first_frame_gen_chunk_last/test",
            last_zsp_cache_root=str(LAST_ZSP_ROOT / "re10k" / "val"),
            split="train",
            split_ratio=1.0,
        ),
        dict(
            superpoint_root="/mnt/share/codes/kimi/ssf_dataset/scannet_first_frame_gen_chunk_new",
            zsp_cache_root=str(CACHE_ROOT / "scannet"),
            last_superpoint_root="/mnt/share/codes/kimi/ssf_dataset/scannet_first_frame_gen_chunk_last_new",
            last_zsp_cache_root=str(LAST_ZSP_ROOT / "scannet"),
            split="train",
            split_ratio=1.0,
            source_split="val",
        ),
    ],
)

model = dict(
    cond_dim=64,
    pos_hidden_dim=768,
    # Same lightweight backbone as 1-view v18_v3. Cond length doubles; the
    # extra cost is only memory cross-attn / raymap known-depth splat.
    num_groups=4,
    layers_per_group=2,
    pos_heads=8,
    mlp_ratio=4.0,
    time_dim=256,
    xyz_fourier_bands=16,
    max_target_points=3000,
    max_views=32,
    content_rope_base=10000.0,
    content_rope_fwd=1.0,
    content_rope_bits=10,
    content_rope_impl="continuous3d",
    content_rope_t_mid=0.5,
    content_rope_t_scale=8.0,
    use_flash_varlen=True,
    camera_image_width=518,
    camera_image_height=378,
    canonical_position_scale=1.0,
    canonical_xyz_mean=CANONICAL_XYZ_MEAN,
    canonical_xyz_std=CANONICAL_XYZ_STD,
    route_gradient_scale=0.25,
    raymap_token_grid=37,
    raymap_stride=5,  # -> 8x8 = 64 cells/view; 8 future views = 512 ray tokens
    raymap_dim=256,
    raymap_heads=4,
    raymap_rope_base=100.0,
    raymap_rope_fwd=8.0,
    raymap_uv_clamp=3.0,
)

train = dict(
    batch_size=1,
    num_workers=4,
    val_num_workers=2,
    persistent_workers=True,
    prefetch_factor=2,
    pin_memory=True,
    max_steps=200000,
    gradient_accumulation_steps=1,
    lr=5e-5,
    weight_decay=0.0,
    adam_betas=(0.9, 0.95),
    adam_eps=1e-8,
    warmup_steps=1000,
    schedule_type="constant",
    grad_clip=1.0,
    ema_decay=0.9995,
    amp=True,
    mixed_precision="bf16",
    allow_tf32=True,
    log_every=1,
    save_every=2000,
    val_every=1000,
    max_val_batches=8,
    val_batch_size=4,
    validation_num_samples=256,
    validation_timestep_bins=5,
    validation_position_leading=False,
    validation_seed=12345,
    validation_drop_cond=False,
    validation_drop_cam=False,
    find_unused_parameters=True,
    checkpoints_total_limit=5,
    resume_from_checkpoint="",
    logging_dir="logs",
    report_to=None,
    resume="",
    deepspeed_zero_stage=2,
    deepspeed_offload_optimizer=True,
    deepspeed_offload_optimizer_device="cpu",
)

flow = dict(
    lambda_pos=1.0,
    lambda_con=0.0,
    t_eps=1e-4,
    selfcond_prob=0,
    # Trainer key is unchanged. 2-view samples source from first-view tokens
    # only (cond_view_index == 0); condition memory still uses first+last.
    source_prior="first_frame_partial_noise",
    first_frame_source_noise=0.5,
    cond_drop_prob=0.0,
    cam_drop_prob=0.1,
    timestep_mode="independent",
    timestep_distribution="logit_normal",
    logit_normal_mean=0.0,
    logit_normal_std=1.0,
    position_lead_beta=0.5,
    ot_coupling="none",
    ot_sinkhorn_iterations=20,
    ot_sinkhorn_temperature=0.1,
)

regularization = dict(
    aux_position_loss_mode="x0",
    aux_position_loss_weights=(0.05, 0.07, 0.10, 0.10),
    chamfer_weight=0.0,
    repulsion_weight=0.0,
    repulsion_k=8,
    low_t_power=1.0,
    vae_head_weight=0,
    vae_head_config="",
)

sample = dict(
    ckpt=str(Path(work_dir) / "latest.pt"),
    num_steps=50,
    num_target_points=3000,
    first_frame_source_noise=0.5,
    guidance=1.0,
    guidance_mode="camera",
    guidance_cond=0.0,
    guidance_cam=1.0,
    guidance_start=1.0,
    guidance_end=1.0,
    seed=123,
    out_dir=str(ROOT / "out" / "dit_v18_v3_2view_stage1_position_gld_samples"),
)
