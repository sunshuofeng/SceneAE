# v30 grid decoder, with the superpoint encoder trained jointly.
#
# The internal v30 runs froze the encoder and trained the decoder on cached
# z_sp so the decoder could be iterated quickly. This release trains both.
# DA3 stays a frozen feature extractor. init_from still loads the pretrained
# encoder; the grid decoder has no matching keys and starts from scratch.
#
# Decoder split after the frontend:
#   * shared_layers = 0 (RGB no longer shares ViT tokens with depth).
#   * RGB path = 24 layers.
#   * Depth: 8 isolated token-dim prep layers, then the depth decoder
#     (proj + 4 depth blocks + residual inverse-depth head).  Prep tokens are
#     detached from the frontend so depth cannot tilt appearance features.
#   * Cross-attn every 2 layers on RGB and depth-prep (injected by the v30 wrapper).
#
# Training reads FFG chunks (images, cameras, depth, FPS anchors). The encoder
# runs every step. Cached z_sp is not the training target.
#
# Training changes vs v20_v2 (design §6):
#   * from-scratch LR 2e-4 (cosine, warmup 2 ep, final 2e-5), betas (0.9,0.95).
#   * per-parameter-group gradient clipping (frontend+trunk / rgb / depth).
#   * decode-side jitter: per-sample Bernoulli(prob=0.5) + sigma curriculum
#     (0 for epoch<2, linear to 0.01 over epoch 2..6, then hold).
#   * separate best_rgb.pth / best_depth.pth selected on the FFG val
#     (same chunks as training, encoder included); C15 online val is secondary.
init_from = "/mnt/share/codes/kimi/ssf/E3D/decoder/out/v4_8_combined_multieval_resume/best.pth"
freeze_encoder = False
# Used only when freeze_encoder is True.
trainable_prefixes = ["decoder"]
grad_max_norm = 1.0  # per-group max_norm (design §6.3)
print_freq = 10
max_epochs = 100
eval_every_epochs = 1
# cosine schedule: warmup 2 epochs then decay to lr_min (design §6.3).
warmup_t = 2
lr_min = 2e-5
amp_dtype = "bfloat16"
use_ema = True
ema_decay = 0.9995
syncBN = False
dist_timeout_minutes = 60
num_workers = 16
prefetch_factor = 2
persistent_workers = True

multisteplr = False
multisteplr_config = dict(
    decay_t=[25, 35],
    decay_rate=0.1,
    t_in_epochs=False,
)

optimizer = dict(
    optimizer=dict(
        type="AdamW",
        lr=2e-4,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    ),
)

model_module = "models.scene_vae"
model_class = "SuperpointVAEGridV30"

# ---------------------------------------------------------------------------
# Data: identical to v20_v2 (FFG cached z_sp sidecars).
# ---------------------------------------------------------------------------
dataset_name = "ffg_cached"
train_batch_size = 2

dataset_kwargs = dict(
    resize_w=518,
    resize_h=378,
    num_views=9,
    target_max_superpoints=3000,

    equal_ratio=False,
    mix_ratio=dict(
        scannet=10000,
        dl3dv=30000,
        re10k=30000,
    ),

    val_datasets=["scannet","dl3dv", "re10k"],
    datasets=dict(
        scannet=dict(
            chunk_root="/mnt/share/codes/kimi/ssf_dataset/scannet_first_frame_gen_chunk_new",
            cache_root="/mnt/share/codes/kimi/ssf_dataset/first_frame_gen_cache_new_3000/scannet",
            split_reference_root="/mnt/share/codes/kimi/ssf_dataset/scannet_chunk_geometry_sp2",
            split_ratio=0.9,
            depth_near=0.1,
            depth_far=10.0,
            norm_depth_clip=100.0,
            min_depth_valid_ratio=0.15,
        ),
        dl3dv=dict(
            chunk_root="/mnt/share/codes/kimi/ssf_dataset/DL3DV/first_frame_gen_chunk_new",
            cache_root="/mnt/share/codes/kimi/ssf_dataset/first_frame_gen_cache_new_3000/dl3dv/train",
            eval_chunk_root="/mnt/share/codes/kimi/ssf_dataset/DL3DV/first_frame_gen_chunk_eval_new",
            eval_cache_root="/mnt/share/codes/kimi/ssf_dataset/first_frame_gen_cache_new_3000/dl3dv/val",
            depth_near=0.1,
            depth_far=50.0,
            norm_depth_clip=50.0,
        ),
        re10k=dict(
            chunk_root="/mnt/share/codes/kimi/ssf_dataset/RE10K/first_frame_gen_chunk/train",
            cache_root="/mnt/share/codes/kimi/ssf_dataset/first_frame_gen_cache_new_3000/re10k/train",
            eval_chunk_root="/mnt/share/codes/kimi/ssf_dataset/RE10K/first_frame_gen_chunk/test",
            eval_cache_root="/mnt/share/codes/kimi/ssf_dataset/first_frame_gen_cache_new_3000/re10k/val",
            depth_near=0.1,
            depth_far=50.0,
            norm_depth_clip=50.0,
        ),
    ),
)
auto_feature_dims = False

# C15 online val (secondary metric, comparable to the v20 run).  Not used for
# checkpoint selection in v30 (FFG cached val is primary).
eval_dataset_kwargs = dict(
    resize_w=518,
    resize_h=378,
    num_views=9,
    num_fps_sp_xyz=3000,
    fps_voxel_size=0.0,
    allow_missing_depth=True,
    use_manifest_cache=True,
    rebuild_manifest=False,

    equal_ratio=False,
    mix_ratio=dict(
        scannet=10000,
        dl3dv=30000,
        re10k=30000,
    ),

    val_datasets=["scannet","dl3dv", "re10k"],
    datasets=dict(
        scannet=dict(
            data_root="/mnt/share/codes/kimi/ssf_dataset/scannet_rgb_depth",
            superpoint_root="/mnt/share/codes/kimi/ssf_dataset/scannet_chunk_geometry_sp2",
            eval_manifest_path="/mnt/share/codes/kimi/ssf/E3D/decoder/out/manifest_cache/scannet_chunkv2_c15_index_test_full_views9.json",
            split_ratio=0.9,
            depth_scale=1000.0,
        ),
        dl3dv=dict(
            data_root="/mnt/share/codes/kimi/ssf_dataset/DL3DV/DL3DV",
            eval_data_root="/mnt/share/codes/kimi/ssf_dataset/DL3DV/DL3DV-Eval",
            superpoint_root="/mnt/share/codes/kimi/ssf_dataset/DL3DV/10K_chunk_geometry_sp_da3base_vae",
            eval_superpoint_root="/mnt/share/codes/kimi/ssf_dataset/DL3DV/10K_eval_chunk_geometry_sp_da3base_vae",
            depth_min=0.1,
            depth_max=50.0,
            norm_depth_clip=50.0,
        ),
        re10k=dict(
            data_root="/mnt/share/codes/kimi/ssf_dataset/RE10K/re10k",
            superpoint_root="/mnt/share/codes/kimi/ssf_dataset/RE10K/re10k_chunk_geometry_sp_da3base_vae",
            split_ratio=0.9,
            depth_min=0.1,
            depth_max=50.0,
            norm_depth_clip=50.0,
        ),
    ),
)

# ---------------------------------------------------------------------------
# Encoder (trained; initialised from init_from).  DA3-Base stays frozen.
# ---------------------------------------------------------------------------
latent_dim = 64
encoder_img_feat_in_dim = 768
num_da3_levels = 12
dino_feat_dim = 768
da3_feat_dim = 768
vae_latent_channels = 16

encoder_hidden_dim = 792
encoder_pooled_level_dim = 128
encoder_layers = 8
encoder_heads = 12
encoder_mlp_ratio = 4.0
encoder_max_views = 9
encoder_view_capacity = 9
encoder_query_view_topk = 2
encoder_query_init_topk = 1
encoder_query_attn_topk = 1
encoder_query_attn_topk_schedule = (1, 1, 1, 1, 1, 1, 1, 1)
encoder_mv_residual_topk_schedule = (2, 2, 4, 4, 6, 6, 8, 8)
encoder_mv_residual_deform_points = 4
encoder_level_logits_tau_main = 1.0
encoder_level_logits_tau_mv = 1.0
encoder_unified_fuse_mlp_ratio = 2.0
eval_use_all_encoder_views = True
encoder_deform_points = 4
encoder_deform_im2col_step = 64
encoder_self_attn_type = "global"
encoder_self_window = 128
encoder_self_attn_use_rope = True
encoder_self_attn_rope_base = 10000.0
encoder_self_attn_rope_fwd = 1.0
encoder_self_attn_rope_bits = 10

online_da3 = True
da3_weights = "/mnt/share/codes/kimi/ssf/DA3-BASE"
da3_feat_layers = tuple(range(12))
da3_process_res = 518
da3_process_res_method = "upper_bound_resize"
da3_use_camera_tokens = True

# Legacy decoder_* params: only used to build the (immediately replaced) base
# decoder, so keep them tiny to minimise the throwaway construction cost.
decoder_cross_layers = 2
decoder_rgb_cross_layers = 2
decoder_depth_cross_layers = 2
decoder_rgb_grid_channels = 64
decoder_depth_grid_channels = 64
decoder_rgb_num_views = 9
decoder_depth_num_views = 9
decoder_cross_heads = 4
decoder_rgb_cross_heads = 4
decoder_depth_cross_heads = 4
decoder_knn_k = 4
vae_scale_factor = 8

# ---------------------------------------------------------------------------
# v30 grid decoder: split RGB / depth, residual compose kept.
# ---------------------------------------------------------------------------
grid_token_dim = 792
grid_patch = 14
grid_image_size = (378, 518)
grid_k_init = 8
grid_k_xattn = 64
grid_n_groups = 4
grid_shared_layers = 0
grid_rgb_layers = 24
grid_depth_prep_layers = 8
grid_depth_layers = 4
grid_depth_dim = 384
grid_n_heads = 12
grid_depth_heads = 8
grid_mlp_ratio = 4.0
grid_rgb_num_views = 6
grid_depth_num_views = 6
grid_alpha_grad_gate = 0.0
grid_beta_init = 3.0
grid_gamma_init = 1.0
grid_depth_min_norm = 0.02
grid_depth_max_norm = 50.0
grid_depth_pct_lo = 1.0
grid_depth_pct_hi = 99.0

# ---------------------------------------------------------------------------
# v27 loss weights (design §5).  LPIPS weight locked at 0.3 (not a sharpness knob).
# ---------------------------------------------------------------------------
rgb_l1_weight = 1.0
rgb_grad_weight_v27 = 0.5
rgb_lpips_weight = 0.3
depth_l1_weight = 1.0
depth_grad_weight_v27 = 0.2

# Kept for the base-class __init__ (it builds the frozen LPIPS model when >0);
# the v27 compute_losses uses rgb_lpips_weight.  Also disables v20 losses.
lpips_loss_weight = 0.3
rgb_loss_weight = 1.0
depth_loss_weight = 1.0
rgb_fullres_weight = 0.5
rgb_grad_weight = 0.5
depth_grad_weight = 0.2
depth_da3_distill_weight = 0.0
warp_loss_weight = 0.0  # design §0 discipline 3: no warp consistency loss
latent_reg_weight = 0.0
use_sp_field_distill = False
sp_field_knn = 8
sp_field_sigma_fallback = 0.05
sp_vae_loss_weight = 0.0
sp_vae_cos_weight = 1.0
sp_vae_mse_weight = 0.0
geometry_loss_weight = 0.0
geometry_radius_weight = 1.0
geometry_eigvals_weight = 1.0
geometry_normal_weight = 0.5
geometry_scalar_weight = 0.25
depth_smooth_weight = 0.0
debug_nan_checks = False

# V2 decoder switches (unused by the grid decoder; forced off in the wrapper).
decoder_use_geometry_aniso = False
depth_inverse_param = False
depth_min_norm = 0.02
depth_max_norm = 50.0
depth_pct_lo = 1.0
depth_pct_hi = 99.0

# v27 decode-side position jitter: per-sample Bernoulli(prob) + sigma curriculum.
# sigma starts at 0 and is set per-epoch by the train loop (design §6.2).
decode_sp_xyz_jitter_sigma = 0.0
decode_sp_xyz_jitter_prob = 0.5
jitter_sigma_max = 0.01
jitter_curriculum_start = 2
jitter_curriculum_end = 6
encoder_frozen_no_grad = False

sp_xyz_jitter_sigma = 0
sp_xyz_jitter_prob = 0
infer_use_fps_spxyz = False
use_fps_spxyz = True
infer_fps_voxel_size = 0.0
val_use_fps_spxyz = True

# Validation: per-dataset.  Checkpoint selection uses the FFG cached val (primary).
eval_datasets = ["scannet", "dl3dv", "re10k"]
primary_eval_dataset = "mean"
val_batch_size = 1
val_num_workers = 4

# FFG cached-latent validation -- in v27 this is the PRIMARY metric (training
# distribution).  C15 online val above is secondary / diagnostic only.
ffg_val_enabled = True
ffg_val_max_per_dataset = 256

enable_sp_feat_supervision = False
sp_feat_loss_weight = 0.0

decode_rgb = True
