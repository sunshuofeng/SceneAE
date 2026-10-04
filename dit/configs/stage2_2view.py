from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

seed = 42
work_dir = str(ROOT / "out" / "dit_v26_2view_wan_style_content_refine_gld_ddp")
model_module = "models.stage2_2view"
model_class = "PointDiTV26TwoView"
model_requires_xyz_stats = True

CACHE_ROOT = Path("/mnt/share/codes/kimi/ssf_dataset/first_frame_gen_cache_new_3000")
LAST_ZSP_ROOT = Path("/mnt/share/codes/kimi/ssf_dataset/first_frame_gen_cache_last_zsp")
JOINT_ZSP_STATS = str(CACHE_ROOT / "stats" / "zsp_xyz_stats_train_joint.npz")
CANONICAL_XYZ_STATS = CACHE_ROOT / "stats" / "canonical_target_stats_train_joint.npz"
# 2-view stage-1 xyz sidecar (v18_v3_2view). Isolated from the 1-view
# v18_stage1_xyz tree used by dit_v26.
STAGE1_XYZ_CACHE = CACHE_ROOT / "v18_v3_2view_stage1_xyz"
# Hungarian sidecars from tools/build_hungarian.py --dataset_config configs/stage2_2view.py.
# Reuse that tree; do not rebuild a separate perm cache.
STAGE2_PERM_CACHE = CACHE_ROOT / "v18_v3_2view_stage1_perm_hungarian"
if not CANONICAL_XYZ_STATS.is_file():
    raise FileNotFoundError(
        f"Missing v26_2view canonical statistics: {CANONICAL_XYZ_STATS}. Run: "
        "python tools/compute_canonical_stats.py configs/stage1.py"
    )
with np.load(CANONICAL_XYZ_STATS) as _canonical_stats:
    CANONICAL_XYZ_MEAN = tuple(float(value) for value in _canonical_stats["canonical_xyz_mean"])
    CANONICAL_XYZ_STD = tuple(float(value) for value in _canonical_stats["canonical_xyz_std"])

# 2-view cond = cat(first, last). Each side pads to per_view, so cond length is 6000.
# Target tokens, 9-view cameras, and xyz/zsp stats stay the same as 1-view v26.
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
    stage1_xyz_required=True,
    target_perm_required=True,
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
            stage1_xyz_cache_root=str(STAGE1_XYZ_CACHE / "dl3dv" / "train"),
            target_perm_cache_root=str(STAGE2_PERM_CACHE / "dl3dv" / "train"),
        ),
        dict(
            superpoint_root="/mnt/share/codes/kimi/ssf_dataset/RE10K/first_frame_gen_chunk/train",
            zsp_cache_root=str(CACHE_ROOT / "re10k" / "train"),
            last_superpoint_root="/mnt/share/codes/kimi/ssf_dataset/RE10K/first_frame_gen_chunk_last/train",
            last_zsp_cache_root=str(LAST_ZSP_ROOT / "re10k" / "train"),
            stage1_xyz_cache_root=str(STAGE1_XYZ_CACHE / "re10k" / "train"),
            target_perm_cache_root=str(STAGE2_PERM_CACHE / "re10k" / "train"),
            split="train",
            split_ratio=1.0,
        ),
        dict(
            superpoint_root="/mnt/share/codes/kimi/ssf_dataset/scannet_first_frame_gen_chunk_new",
            zsp_cache_root=str(CACHE_ROOT / "scannet"),
            last_superpoint_root="/mnt/share/codes/kimi/ssf_dataset/scannet_first_frame_gen_chunk_last_new",
            last_zsp_cache_root=str(LAST_ZSP_ROOT / "scannet"),
            stage1_xyz_cache_root=str(STAGE1_XYZ_CACHE / "scannet"),
            target_perm_cache_root=str(STAGE2_PERM_CACHE / "scannet"),
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
            stage1_xyz_cache_root=str(STAGE1_XYZ_CACHE / "dl3dv" / "val"),
            target_perm_cache_root=str(STAGE2_PERM_CACHE / "dl3dv" / "val"),
            split="train",
            split_ratio=1.0,
        ),
        dict(
            superpoint_root="/mnt/share/codes/kimi/ssf_dataset/RE10K/first_frame_gen_chunk/test",
            zsp_cache_root=str(CACHE_ROOT / "re10k" / "val"),
            last_superpoint_root="/mnt/share/codes/kimi/ssf_dataset/RE10K/first_frame_gen_chunk_last/test",
            last_zsp_cache_root=str(LAST_ZSP_ROOT / "re10k" / "val"),
            stage1_xyz_cache_root=str(STAGE1_XYZ_CACHE / "re10k" / "val"),
            target_perm_cache_root=str(STAGE2_PERM_CACHE / "re10k" / "val"),
            split="train",
            split_ratio=1.0,
        ),
        dict(
            superpoint_root="/mnt/share/codes/kimi/ssf_dataset/scannet_first_frame_gen_chunk_new",
            zsp_cache_root=str(CACHE_ROOT / "scannet"),
            last_superpoint_root="/mnt/share/codes/kimi/ssf_dataset/scannet_first_frame_gen_chunk_last_new",
            last_zsp_cache_root=str(LAST_ZSP_ROOT / "scannet"),
            stage1_xyz_cache_root=str(STAGE1_XYZ_CACHE / "scannet"),
            target_perm_cache_root=str(STAGE2_PERM_CACHE / "scannet"),
            split="train",
            split_ratio=1.0,
            source_split="val",
        ),
    ],
)

model = dict(
    target_dim=64,
    cond_dim=64,
    # Same Wan + conf_head capacity as 1-view v26. Cond length doubles; extra
    # cost is cross-attn over 6000 tokens plus a zero-init first/last embedding.
    hidden_dim=1536,
    num_layers=22,
    num_heads=12,
    mlp_ratio=4.0,
    time_dim=256,
    xyz_fourier_bands=16,
    content_rope_base=10000.0,
    content_rope_fwd=1.0,
    content_rope_bits=10,
    content_rope_impl="continuous3d",
    aux_con_layer=16,
    use_flash_varlen=True,
    canonical_position_scale=1.0,
    canonical_xyz_mean=CANONICAL_XYZ_MEAN,
    canonical_xyz_std=CANONICAL_XYZ_STD,
    refine_xyz_dim=256,
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
    val_batch_size=6,
    validation_num_samples=384,
    validation_timestep_bins=5,
    validation_position_leading=True,
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
    # Same v26 gated residual: v = stage1 - gt, res = t_pos * v, plus
    # detached s = exp(-||stage1-gt|| / gate_sigma) BCE. Inference freezes
    # s at the first refine step and uses v_used = (1-s) * v.
    lambda_pos=1.0,
    lambda_con=1.0,
    lambda_gate=1.0,
    gate_sigma=0.08,
    t_eps=1e-4,
    t_pos_anchor_prob=0.5,
    selfcond_prob=0.0,
    cond_drop_prob=0.1,
    timestep_mode="independent",
    position_lead_beta=0.5,
)

regularization = dict(
    aux_position_loss_weights=(0.0,),
    chamfer_weight=0.0,
    repulsion_weight=0.0,
    repulsion_k=8,
    low_t_power=1.0,
    vae_head_weight=0,
    vae_head_warmup_start_steps=1000,
    vae_head_warmup_steps=9000,
    vae_head_source="aux",
    vae_head_lightweight=True,
    vae_head_config="",
    vae_head_texture_weight=1.0,
    vae_head_texture_cos_weight=1.0,
    vae_head_texture_mse_weight=0.0,
    vae_head_radius_weight=1.0,
    vae_head_eigvals_weight=1.0,
    vae_head_normal_weight=0.5,
    vae_head_scalar_weight=0.25,
)

sample = dict(
    ckpt=str(Path(work_dir) / "latest.pt"),
    num_steps=50,
    num_target_points=3000,
    refine_gate=True,
    guidance=1.5,
    guidance_mode="cond",
    guidance_cond=1.5,
    guidance_cam=0.0,
    guidance_start=1.0,
    guidance_end=1.0,
    seed=123,
    position_lead_beta=0.5,
    out_dir=str(ROOT / "out" / "dit_v26_2view_wan_style_content_refine_gld_samples"),
)
