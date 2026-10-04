"""C20_v2 (v20_v2): decoder retraining on top of a frozen C19/C20 encoder.

Built on models/superpoint_da3_groups_c19.py (the C20 config's model). The
encoder is frozen and initialized from the v20 checkpoint; only the decoder
is retrained with the following changes:

  1. Decode-side position jitter (stage-0.1): the encoder always sees clean
     FPS superpoint xyz; only the decoder input positions are perturbed during
     training. This simulates DiT-generated anchor noise without corrupting
     the encoder's projection pooling. Jitter is masked to valid superpoints.
  2. Geometry-driven anisotropy (stage-1.5): the decoder's per-point 3D
     covariance is constructed from the frozen, directly-supervised geometry
     heads (radius as principal scale, eigval ratios for anisotropy, normal
     for the frame + a zero-init per-point modulation head for residual
     freedom), replacing the free-running aniso_param_head.
  3. Multi-view warp consistency loss (stage-1.4): rendered RGB of one view is
     warped into another rendered view using GT depth (default; optionally the
     rendered depth) and compared against that view's own render, with
     z-buffer occlusion masking.
  4. Inverse-depth output parameterization with per-view adaptive range from
     superpoint z percentiles (available at generation time, robust to noisy
     DiT positions); the depth grid carries a normalized inverse-depth prior
     channel instead of the raw z_prior channel (same channel count, so the
     v20 depth upsampler weights remain loadable).
  5. Encoder + frozen heads run under no_grad when `encoder_frozen_no_grad`
     is set, so backward only traverses the decoder.
"""

from __future__ import annotations

import contextlib
import inspect
import math
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import (
    SuperpointVAE2C,
    _charbonnier,
    _debug_check_finite,
    _projective_pool_grid,
    _projective_pool_sample,
    _select_query_topk_views_by_depth,
    _visible_query_mask_from_projection,
)


class SuperpointVAE2CV2(SuperpointVAE2C):
    """C20_v2 model: frozen C19 encoder + retrained V2 decoder.

    Extra args vs SuperpointVAE2C:
        decode_sp_xyz_jitter_sigma/prob: position noise applied ONLY to the
            decoder input (training time), simulating DiT-generated anchors.
        warp_loss_weight / warp_tol / warp_use_rendered_depth: multi-view warp
            consistency loss settings.
        encoder_frozen_no_grad: run encoder + frozen heads under no_grad.
        decoder_use_geometry_aniso / depth_inverse_param / depth_*: V2 decoder
            switches.
    """

    def __init__(
        self,
        *args,
        decode_sp_xyz_jitter_sigma: float = 0.01,
        decode_sp_xyz_jitter_prob: float = 0.9,
        warp_loss_weight: float = 0.5,
        warp_tol: float = 0.02,
        warp_use_rendered_depth: bool = False,
        encoder_frozen_no_grad: bool = True,
        decoder_use_geometry_aniso: bool = True,
        depth_inverse_param: bool = True,
        depth_min_norm: float = 0.02,
        depth_max_norm: float = 50.0,
        depth_pct_lo: float = 1.0,
        depth_pct_hi: float = 99.0,
        **kwargs,
    ) -> None:
        # The frozen C19 base does not accept every kwarg that newer configs
        # carry (e.g. da3_normalization_stat_path / da3_normalization_eps).
        # Forward only what the base __init__ understands.
        base_params = inspect.signature(SuperpointVAE2C.__init__).parameters
        if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in base_params.values()):
            kwargs = {k: v for k, v in kwargs.items() if k in base_params}
        super().__init__(*args, **kwargs)
        # Grid decoder is installed by SceneLatentVAE. These switches stay
        # accepted so older configs still construct the wrapper.
        del decoder_use_geometry_aniso, depth_inverse_param, depth_min_norm, depth_max_norm, depth_pct_lo, depth_pct_hi
        self.decode_sp_xyz_jitter_sigma = float(decode_sp_xyz_jitter_sigma)
        self.decode_sp_xyz_jitter_prob = float(decode_sp_xyz_jitter_prob)
        self.warp_loss_weight = float(warp_loss_weight)
        self.warp_tol = float(warp_tol)
        self.warp_use_rendered_depth = bool(warp_use_rendered_depth)
        self.encoder_frozen_no_grad = bool(encoder_frozen_no_grad)

    def _maybe_jitter_decode_sp_xyz(
        self,
        sp_xyz: torch.Tensor,
        sp_valid_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Position noise for the DECODER input only (stage-0.1)."""
        if (not self.training) or self.decode_sp_xyz_jitter_sigma <= 0:
            return sp_xyz
        if torch.rand((), device=sp_xyz.device) > self.decode_sp_xyz_jitter_prob:
            return sp_xyz
        noise = torch.randn_like(sp_xyz) * self.decode_sp_xyz_jitter_sigma
        if sp_valid_mask is not None:
            noise = noise * sp_valid_mask.to(device=sp_xyz.device, dtype=noise.dtype).unsqueeze(-1)
        return sp_xyz + noise

    def _forward_cached_zsp(
        self,
        *,
        sp_xyz: torch.Tensor,
        camtoworlds: torch.Tensor,
        Ks: torch.Tensor,
        image: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
        sp_valid_mask: Optional[torch.Tensor],
        cached_z_sp: torch.Tensor,
        decode_rgb: bool,
    ) -> Dict[str, torch.Tensor]:
        """Frozen-encoder fast path over the DiT z_sp sidecar cache.

        (sp_xyz, cached_z_sp) are the cached rest_-group FPS anchors and
        latents produced by vae/cache_latent.py with the frozen C20
        encoder — exactly the distribution the DiT is trained to generate, so
        the decoder trains on its generation-time input distribution.  DA3 and
        the encoder are not touched; the frozen prediction heads still run on
        the cached latent so the V2 decoder receives its geometry-driven
        anisotropy inputs.  Encoder-side view subsetting / FPS / jitter are
        all bypassed (the cache already encodes all views at FPS anchors).
        """
        if image.dim() != 5:
            raise ValueError(f"image expected [B,V,H,W,3], got {tuple(image.shape)}")
        sp_xyz = torch.nan_to_num(sp_xyz, nan=0.0, posinf=0.0, neginf=0.0)
        z_sp = torch.nan_to_num(
            cached_z_sp.to(device=sp_xyz.device, dtype=torch.float32),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        if z_sp.shape[:2] != sp_xyz.shape[:2]:
            raise ValueError(f"cached_z_sp shape {tuple(z_sp.shape)} vs sp_xyz {tuple(sp_xyz.shape)}")
        if sp_valid_mask is not None:
            sp_valid_mask = sp_valid_mask.to(device=sp_xyz.device).bool()

        head_ctx = torch.no_grad() if self.encoder_frozen_no_grad else contextlib.nullcontext()
        with head_ctx:
            if self.enable_sp_feat_supervision:
                pred_sp_dino = self.sp_feat_head_dino(z_sp)
                pred_sp_da3 = self.sp_feat_head_da3(z_sp)
                pred_sp_vae = self.sp_feat_head_vae(z_sp)
                pred_sp_feat = torch.cat([pred_sp_dino, pred_sp_da3, pred_sp_vae], dim=-1)
            else:
                pred_sp_dino = z_sp.new_zeros((*z_sp.shape[:2], 0))
                pred_sp_da3 = z_sp.new_zeros((*z_sp.shape[:2], 0))
                pred_sp_vae = z_sp.new_zeros((*z_sp.shape[:2], 0))
                pred_sp_feat = z_sp.new_zeros((*z_sp.shape[:2], 0))
            pred_geometry_radius = self.geometry_radius_head(z_sp)
            pred_geometry_eigvals = F.softplus(self.geometry_eigvals_head(z_sp))
            pred_geometry_normal = F.normalize(self.geometry_normal_head(z_sp), dim=-1, eps=1e-6)
            pred_geometry_scalar = self.geometry_scalar_head(z_sp)
            pred_sp_vae_texture = self.sp_vae_texture_head(z_sp)
        anchor_aux = {
            "anchor_offsets": z_sp.new_zeros((*z_sp.shape[:2], 0, 3)),
            "anchor_offset_total": z_sp.new_zeros((*z_sp.shape[:2], 3)),
        }

        # Decode-side jitter still applies: cached anchors are the clean
        # targets, jitter simulates DiT-generated anchor noise on top.
        sp_xyz_dec = self._maybe_jitter_decode_sp_xyz(sp_xyz, sp_valid_mask)
        decode_out = self.decoder(
            sp_feat=z_sp,
            camtoworlds=camtoworlds,
            Ks=Ks,
            image=image,
            sp_xyz=sp_xyz_dec,
            valid_mask=valid_mask,
            sp_valid_mask=sp_valid_mask,
            decode_rgb=decode_rgb,
            geom_radius=pred_geometry_radius,
            geom_eigvals=pred_geometry_eigvals,
            geom_normal=pred_geometry_normal,
        )
        # No encoder view subset in this path: local == global view indices.
        if decode_out.get("selected_view_idx_rgb", None) is not None:
            decode_out["selected_view_idx_rgb_local"] = decode_out["selected_view_idx_rgb"]
        if decode_out.get("selected_view_idx_depth", None) is not None:
            decode_out["selected_view_idx_depth_local"] = decode_out["selected_view_idx_depth"]

        out: Dict[str, torch.Tensor] = {
            "free_latent_sp": z_sp,
            "sp_xyz_initial": sp_xyz,
            "sp_xyz_refined": sp_xyz,
            "sp_xyz_decode": sp_xyz_dec,
            **anchor_aux,
            "pred_sp_dino": pred_sp_dino,
            "pred_sp_da3": pred_sp_da3,
            "pred_sp_vae": pred_sp_vae,
            "pred_sp_feat": pred_sp_feat,
            "pred_geometry_radius": pred_geometry_radius,
            "pred_geometry_eigvals": pred_geometry_eigvals,
            "pred_geometry_normal": pred_geometry_normal,
            "pred_geometry_scalar": pred_geometry_scalar,
            "pred_sp_vae_texture": pred_sp_vae_texture,
            "pred_sp_field_vae": None,
            "pred_sp_field_geom_radius": None,
            "pred_sp_field_geom_eigvals": None,
            "pred_sp_field_geom_normal": None,
            "pred_sp_field_geom_scalar": None,
            "sp_field_valid_mask": None,
            **decode_out,
        }
        return out

    def forward_with_loss(
        self,
        batch: dict,
        decode_rgb: bool = False,
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        result = self(
            sp_feat=batch["sp_feat"],
            camtoworlds=batch["norm_camtoworlds"],
            Ks=batch["Ks"],
            image=batch["image"],
            sp_xyz=batch["norm_sp_xyz"],
            da3_feature_map=batch.get("da3_feature_map", None),
            da3_feature_maps=batch.get("da3_feature_maps", None),
            norm_depth=batch.get("norm_depth", None),
            valid_mask=batch.get("valid_mask", None),
            sp_valid_mask=batch.get("superpoint_valid_mask", None),
            sp_radius_p90_norm=batch.get("sp_radius_p90_norm", None),
            source_superpoint_xyz=batch.get("source_superpoint_xyz", None),
            source_superpoint_valid_mask=batch.get("source_superpoint_valid_mask", None),
            norm_pw=batch.get("norm_pw", None),
            norm_pw_offsets=batch.get("norm_pw_offsets", None),
            decode_rgb=decode_rgb,
            cached_z_sp=batch.get("cached_z_sp", None),
        )
        losses = self.compute_losses(result, batch)
        return result, losses

    def _encode_online(
        self,
        *,
        camtoworlds: torch.Tensor,
        Ks: torch.Tensor,
        image: torch.Tensor,
        sp_xyz: torch.Tensor,
        da3_feature_map: Optional[torch.Tensor] = None,
        da3_feature_maps: Optional[Sequence[torch.Tensor]] = None,
        norm_depth: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        sp_valid_mask: Optional[torch.Tensor] = None,
        norm_pw: Optional[torch.Tensor] = None,
        norm_pw_offsets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """DA3 + encoder only.  Used by v28-v2 (no C20 decoder / aux heads)."""
        if image.dim() != 5:
            raise ValueError(f"image expected [B,V,H,W,3], got {tuple(image.shape)}")
        b_img, v_img, h_img, w_img, _ = image.shape
        encoder_view_idx = None
        image_encoder = image
        Ks_encoder = Ks
        camtoworlds_encoder = camtoworlds
        norm_depth_encoder = norm_depth
        valid_mask_encoder = valid_mask
        max_enc_views = max(1, min(int(self.encoder_max_views), int(v_img)))
        use_view_subset = self.training or (not self.eval_use_all_encoder_views)
        if use_view_subset and v_img > max_enc_views:
            if self.training:
                encoder_view_idx = torch.rand(b_img, v_img, device=image.device).argsort(dim=-1)[:, :max_enc_views]
                encoder_view_idx, _ = encoder_view_idx.sort(dim=-1)
            else:
                encoder_view_idx = torch.arange(max_enc_views, device=image.device).unsqueeze(0).expand(b_img, max_enc_views).contiguous()
            image_encoder = torch.gather(
                image, 1,
                encoder_view_idx[:, :, None, None, None].expand(b_img, max_enc_views, h_img, w_img, image.shape[-1]),
            )
            Ks_encoder = torch.gather(
                Ks, 1,
                encoder_view_idx[:, :, None, None].expand(b_img, max_enc_views, Ks.shape[-2], Ks.shape[-1]),
            )
            camtoworlds_encoder = torch.gather(
                camtoworlds, 1,
                encoder_view_idx[:, :, None, None].expand(b_img, max_enc_views, camtoworlds.shape[-2], camtoworlds.shape[-1]),
            )
            if valid_mask is not None:
                valid_mask_encoder = torch.gather(
                    valid_mask, 1,
                    encoder_view_idx[:, :, None, None].expand(b_img, max_enc_views, valid_mask.shape[-2], valid_mask.shape[-1]),
                )
            if norm_depth is not None:
                norm_depth_encoder = torch.gather(
                    norm_depth, 1,
                    encoder_view_idx[:, :, None, None].expand(b_img, max_enc_views, norm_depth.shape[-2], norm_depth.shape[-1]),
                )
        sp_xyz = torch.nan_to_num(sp_xyz, nan=0.0, posinf=0.0, neginf=0.0)
        if (self.use_fps_spxyz or ((not self.training) and self.infer_use_fps_spxyz)) and norm_pw is not None:
            sp_xyz = self._infer_sp_xyz_from_point_cloud(norm_pw, norm_pw_offsets, sp_xyz.shape[1], sp_xyz)
        sp_xyz = self._maybe_jitter_sp_xyz(sp_xyz)
        if sp_valid_mask is not None:
            sp_valid_mask = sp_valid_mask.to(device=sp_xyz.device).bool()

        da3_levels = self._resolve_da3_levels(
            da3_feature_map, da3_feature_maps,
            image=image_encoder, Ks=Ks_encoder, camtoworlds=camtoworlds_encoder,
        )
        img_feat_list, plucker, view_ids = self._build_encoder_tokens(
            da3_levels, Ks_encoder, camtoworlds_encoder, (h_img, w_img)
        )
        feat_hw = (da3_levels[0].shape[2], da3_levels[0].shape[3])
        for lvl in da3_levels[1:]:
            if (lvl.shape[2], lvl.shape[3]) != feat_hw:
                raise ValueError(
                    "Dedup path assumes all DA3 levels share (Hf, Wf); got "
                    f"{(lvl.shape[2], lvl.shape[3])} vs {feat_hw}"
                )
        pool_grid, pool_valid, vis_ratio = _projective_pool_grid(
            sp_xyz, Ks_encoder, camtoworlds_encoder, (h_img, w_img), feat_hw
        )
        if sp_valid_mask is not None:
            pool_valid = pool_valid & sp_valid_mask[:, None, :]
        pool_valid_for_init = _select_query_topk_views_by_depth(
            sp_xyz, Ks_encoder, camtoworlds_encoder, norm_depth_encoder, pool_valid,
            (h_img, w_img), self.encoder_query_init_topk, valid_mask=valid_mask_encoder,
        )
        attn_schedule = self.encoder_query_attn_topk_schedule
        if not attn_schedule:
            attn_schedule = (self.encoder_query_attn_topk,)
        pool_valid_layers = [
            _select_query_topk_views_by_depth(
                sp_xyz, Ks_encoder, camtoworlds_encoder, norm_depth_encoder, pool_valid,
                (h_img, w_img), int(attn_schedule[min(layer_idx, len(attn_schedule) - 1)]),
                valid_mask=valid_mask_encoder,
            )
            for layer_idx in range(len(self.encoder.blocks))
        ]
        mv_schedule = self.encoder_mv_residual_topk_schedule
        mv_pool_valid_layers = None
        if mv_schedule:
            mv_pool_valid_layers = []
            for layer_idx in range(len(self.encoder.blocks)):
                topk = int(mv_schedule[min(layer_idx, len(mv_schedule) - 1)])
                if topk <= 0:
                    mv_pool_valid_layers.append(torch.zeros_like(pool_valid))
                else:
                    mv_pool_valid_layers.append(
                        _select_query_topk_views_by_depth(
                            sp_xyz, Ks_encoder, camtoworlds_encoder, norm_depth_encoder, pool_valid,
                            (h_img, w_img), topk, valid_mask=valid_mask_encoder,
                        )
                    )
        pooled_feats = [
            _projective_pool_sample(lvl, pool_grid, pool_valid_for_init) for lvl in da3_levels
        ]
        enc_ctx = torch.no_grad() if self.encoder_frozen_no_grad else contextlib.nullcontext()
        with enc_ctx:
            z_sp, sp_xyz_refined, anchor_aux = self.encoder(
                da3_levels=img_feat_list,
                pooled_feats=pooled_feats,
                vis_ratio=vis_ratio,
                sp_xyz=sp_xyz,
                plucker=plucker,
                da3_hw=feat_hw,
                pool_grid=pool_grid,
                pool_valid_layers=pool_valid_layers,
                mv_pool_valid_layers=mv_pool_valid_layers,
                sp_valid_mask=sp_valid_mask,
            )
        return {
            "z_sp": z_sp,
            "sp_xyz": sp_xyz,
            "sp_xyz_refined": sp_xyz_refined,
            "anchor_aux": anchor_aux,
            "image": image,
            "camtoworlds": camtoworlds,
            "Ks": Ks,
            "sp_valid_mask": sp_valid_mask,
            "valid_mask": valid_mask,
            "encoder_view_idx": encoder_view_idx,
        }

    def forward(
        self,
        sp_feat: torch.Tensor,
        camtoworlds: torch.Tensor,
        Ks: torch.Tensor,
        image: torch.Tensor,
        sp_xyz: torch.Tensor,
        da3_feature_map: Optional[torch.Tensor] = None,
        da3_feature_maps: Optional[Sequence[torch.Tensor]] = None,
        norm_depth: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        sp_valid_mask: Optional[torch.Tensor] = None,
        sp_radius_p90_norm: Optional[torch.Tensor] = None,
        source_superpoint_xyz: Optional[torch.Tensor] = None,
        source_superpoint_valid_mask: Optional[torch.Tensor] = None,
        norm_pw: Optional[torch.Tensor] = None,
        norm_pw_offsets: Optional[torch.Tensor] = None,
        decode_rgb: bool = False,
        cached_z_sp: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Cached-latent fast path: (sp_xyz, cached_z_sp) come from the DiT
        # sidecar cache, so DA3 + encoder are skipped entirely.
        if cached_z_sp is not None:
            return self._forward_cached_zsp(
                sp_xyz=sp_xyz,
                camtoworlds=camtoworlds,
                Ks=Ks,
                image=image,
                valid_mask=valid_mask,
                sp_valid_mask=sp_valid_mask,
                cached_z_sp=cached_z_sp,
                decode_rgb=decode_rgb,
            )
        # Same orchestration as the C19 forward; V2 changes are marked.
        if image.dim() != 5:
            raise ValueError(f"image expected [B,V,H,W,3], got {tuple(image.shape)}")
        b_img, v_img, h_img, w_img, _ = image.shape
        encoder_view_idx = None
        image_encoder = image
        Ks_encoder = Ks
        camtoworlds_encoder = camtoworlds
        norm_depth_encoder = norm_depth
        valid_mask_encoder = valid_mask
        max_enc_views = max(1, min(int(self.encoder_max_views), int(v_img)))
        use_view_subset = self.training or (not self.eval_use_all_encoder_views)
        if use_view_subset and v_img > max_enc_views:
            if self.training:
                encoder_view_idx = torch.rand(b_img, v_img, device=image.device).argsort(dim=-1)[:, :max_enc_views]
                encoder_view_idx, _ = encoder_view_idx.sort(dim=-1)
            else:
                encoder_view_idx = torch.arange(max_enc_views, device=image.device).unsqueeze(0).expand(b_img, max_enc_views).contiguous()
            image_encoder = torch.gather(
                image, 1,
                encoder_view_idx[:, :, None, None, None].expand(b_img, max_enc_views, h_img, w_img, image.shape[-1]),
            )
            Ks_encoder = torch.gather(
                Ks, 1,
                encoder_view_idx[:, :, None, None].expand(b_img, max_enc_views, Ks.shape[-2], Ks.shape[-1]),
            )
            camtoworlds_encoder = torch.gather(
                camtoworlds, 1,
                encoder_view_idx[:, :, None, None].expand(b_img, max_enc_views, camtoworlds.shape[-2], camtoworlds.shape[-1]),
            )
            if valid_mask is not None:
                valid_mask_encoder = torch.gather(
                    valid_mask, 1,
                    encoder_view_idx[:, :, None, None].expand(b_img, max_enc_views, valid_mask.shape[-2], valid_mask.shape[-1]),
                )
            if norm_depth is not None:
                norm_depth_encoder = torch.gather(
                    norm_depth, 1,
                    encoder_view_idx[:, :, None, None].expand(b_img, max_enc_views, norm_depth.shape[-2], norm_depth.shape[-1]),
                )
        sp_xyz = torch.nan_to_num(sp_xyz, nan=0.0, posinf=0.0, neginf=0.0)

        if (self.use_fps_spxyz or ((not self.training) and self.infer_use_fps_spxyz)) and norm_pw is not None:
            sp_xyz = self._infer_sp_xyz_from_point_cloud(norm_pw, norm_pw_offsets, sp_xyz.shape[1], sp_xyz)
        sp_xyz = self._maybe_jitter_sp_xyz(sp_xyz)  # encoder-side jitter (off in v2 config)
        if sp_valid_mask is not None:
            sp_valid_mask = sp_valid_mask.to(device=sp_xyz.device).bool()

        da3_levels = self._resolve_da3_levels(
            da3_feature_map,
            da3_feature_maps,
            image=image_encoder,
            Ks=Ks_encoder,
            camtoworlds=camtoworlds_encoder,
        )
        img_feat_list, plucker, view_ids = self._build_encoder_tokens(
            da3_levels, Ks_encoder, camtoworlds_encoder, (h_img, w_img)
        )
        da3_mean_feature_map = torch.stack(da3_levels, dim=0).mean(dim=0)

        feat_hw = (da3_levels[0].shape[2], da3_levels[0].shape[3])
        for lvl in da3_levels[1:]:
            if (lvl.shape[2], lvl.shape[3]) != feat_hw:
                raise ValueError(
                    "Dedup path assumes all DA3 levels share (Hf, Wf); got "
                    f"{(lvl.shape[2], lvl.shape[3])} vs {feat_hw}"
                )
        pool_grid, pool_valid, vis_ratio = _projective_pool_grid(
            sp_xyz, Ks_encoder, camtoworlds_encoder, (h_img, w_img), feat_hw
        )
        if sp_valid_mask is not None:
            pool_valid = pool_valid & sp_valid_mask[:, None, :]
        pool_valid_for_init = _select_query_topk_views_by_depth(
            sp_xyz, Ks_encoder, camtoworlds_encoder, norm_depth_encoder, pool_valid,
            (h_img, w_img), self.encoder_query_init_topk, valid_mask=valid_mask_encoder,
        )
        attn_schedule = self.encoder_query_attn_topk_schedule
        if not attn_schedule:
            attn_schedule = (self.encoder_query_attn_topk,)
        pool_valid_layers = [
            _select_query_topk_views_by_depth(
                sp_xyz, Ks_encoder, camtoworlds_encoder, norm_depth_encoder, pool_valid,
                (h_img, w_img), int(attn_schedule[min(layer_idx, len(attn_schedule) - 1)]),
                valid_mask=valid_mask_encoder,
            )
            for layer_idx in range(len(self.encoder.blocks))
        ]
        mv_schedule = self.encoder_mv_residual_topk_schedule
        mv_pool_valid_layers = None
        if mv_schedule:
            mv_pool_valid_layers = []
            for layer_idx in range(len(self.encoder.blocks)):
                topk = int(mv_schedule[min(layer_idx, len(mv_schedule) - 1)])
                if topk <= 0:
                    mv_pool_valid_layers.append(torch.zeros_like(pool_valid))
                else:
                    mv_pool_valid_layers.append(
                        _select_query_topk_views_by_depth(
                            sp_xyz, Ks_encoder, camtoworlds_encoder, norm_depth_encoder, pool_valid,
                            (h_img, w_img), topk, valid_mask=valid_mask_encoder,
                        )
                    )
        pool_valid_for_attn = pool_valid_layers[-1] if pool_valid_layers else pool_valid
        visible_query_mask, visible_query_ratio = _visible_query_mask_from_projection(
            sp_xyz, Ks_encoder, camtoworlds_encoder, (h_img, w_img), valid_mask=valid_mask_encoder,
        )
        pooled_feats = [
            _projective_pool_sample(lvl, pool_grid, pool_valid_for_init) for lvl in da3_levels
        ]

        # V2: frozen encoder + frozen heads run under no_grad so backward only
        # traverses the decoder.
        enc_ctx = torch.no_grad() if self.encoder_frozen_no_grad else contextlib.nullcontext()
        with enc_ctx:
            z_sp, sp_xyz_refined, anchor_aux = self.encoder(
                da3_levels=img_feat_list,
                pooled_feats=pooled_feats,
                vis_ratio=vis_ratio,
                sp_xyz=sp_xyz,
                plucker=plucker,
                da3_hw=feat_hw,
                pool_grid=pool_grid,
                pool_valid_layers=pool_valid_layers,
                mv_pool_valid_layers=mv_pool_valid_layers,
                sp_valid_mask=sp_valid_mask,
            )
            if getattr(self, "use_aux_prediction_heads", True):
                if self.enable_sp_feat_supervision:
                    pred_sp_dino = self.sp_feat_head_dino(z_sp)
                    pred_sp_da3 = self.sp_feat_head_da3(z_sp)
                    pred_sp_vae = self.sp_feat_head_vae(z_sp)
                    pred_sp_feat = torch.cat([pred_sp_dino, pred_sp_da3, pred_sp_vae], dim=-1)
                else:
                    pred_sp_dino = z_sp.new_zeros((*z_sp.shape[:2], 0))
                    pred_sp_da3 = z_sp.new_zeros((*z_sp.shape[:2], 0))
                    pred_sp_vae = z_sp.new_zeros((*z_sp.shape[:2], 0))
                    pred_sp_feat = z_sp.new_zeros((*z_sp.shape[:2], 0))
                pred_geometry_radius = self.geometry_radius_head(z_sp)
                pred_geometry_eigvals = F.softplus(self.geometry_eigvals_head(z_sp))
                pred_geometry_normal = F.normalize(self.geometry_normal_head(z_sp), dim=-1, eps=1e-6)
                pred_geometry_scalar = self.geometry_scalar_head(z_sp)
                pred_sp_vae_texture = self.sp_vae_texture_head(z_sp)
                sp_field, sp_field_valid = self._aggregate_query_to_superpoint(
                    z_sp=z_sp,
                    query_xyz=sp_xyz_refined,
                    sp_xyz=source_superpoint_xyz,
                    sp_radius=sp_radius_p90_norm,
                    sp_valid=source_superpoint_valid_mask,
                )
                if sp_field is not None:
                    pred_sp_field_vae = self.sp_vae_texture_head(sp_field)
                    pred_sp_field_geom_radius = self.geometry_radius_head(sp_field)
                    pred_sp_field_geom_eigvals = F.softplus(self.geometry_eigvals_head(sp_field))
                    pred_sp_field_geom_normal = F.normalize(self.geometry_normal_head(sp_field), dim=-1, eps=1e-6)
                    pred_sp_field_geom_scalar = self.geometry_scalar_head(sp_field)
                else:
                    pred_sp_field_vae = None
                    pred_sp_field_geom_radius = None
                    pred_sp_field_geom_eigvals = None
                    pred_sp_field_geom_normal = None
                    pred_sp_field_geom_scalar = None
            else:
                pred_sp_dino = z_sp.new_zeros((*z_sp.shape[:2], 0))
                pred_sp_da3 = z_sp.new_zeros((*z_sp.shape[:2], 0))
                pred_sp_vae = z_sp.new_zeros((*z_sp.shape[:2], 0))
                pred_sp_feat = z_sp.new_zeros((*z_sp.shape[:2], 0))
                pred_geometry_radius = None
                pred_geometry_eigvals = None
                pred_geometry_normal = None
                pred_geometry_scalar = None
                pred_sp_vae_texture = None
                pred_sp_field_vae = None
                pred_sp_field_geom_radius = None
                pred_sp_field_geom_eigvals = None
                pred_sp_field_geom_normal = None
                pred_sp_field_geom_scalar = None
                sp_field_valid = None

        # V2: decoder sees jittered positions (clean features), simulating
        # DiT-generated anchor noise.
        sp_xyz_dec = self._maybe_jitter_decode_sp_xyz(sp_xyz_refined, sp_valid_mask)
        decode_geom = {}
        if pred_geometry_radius is not None:
            decode_geom = dict(
                geom_radius=pred_geometry_radius,
                geom_eigvals=pred_geometry_eigvals,
                geom_normal=pred_geometry_normal,
            )
        decode_out = self.decoder(
            sp_feat=z_sp,
            camtoworlds=camtoworlds_encoder,
            Ks=Ks_encoder,
            image=image_encoder,
            sp_xyz=sp_xyz_dec,
            valid_mask=valid_mask_encoder,
            sp_valid_mask=sp_valid_mask,
            decode_rgb=decode_rgb,
            **decode_geom,
        )
        if decode_out.get("selected_view_idx_rgb", None) is not None:
            decode_out["selected_view_idx_rgb_local"] = decode_out["selected_view_idx_rgb"]
        if decode_out.get("selected_view_idx_depth", None) is not None:
            decode_out["selected_view_idx_depth_local"] = decode_out["selected_view_idx_depth"]
        if encoder_view_idx is not None:
            for key in ("selected_view_idx_rgb", "selected_view_idx_depth"):
                if key in decode_out and decode_out[key] is not None:
                    idx_local = decode_out[key].long()
                    decode_out[key] = torch.gather(encoder_view_idx, 1, idx_local.clamp(0, encoder_view_idx.shape[1] - 1))

        out: Dict[str, torch.Tensor] = {
            "free_latent_sp": z_sp,
            "sp_xyz_initial": sp_xyz,
            "sp_xyz_refined": sp_xyz_refined,
            "sp_xyz_decode": sp_xyz_dec,
            **anchor_aux,
            "pred_sp_dino": pred_sp_dino,
            "pred_sp_da3": pred_sp_da3,
            "pred_sp_vae": pred_sp_vae,
            "pred_sp_feat": pred_sp_feat,
            "pred_geometry_radius": pred_geometry_radius,
            "pred_geometry_eigvals": pred_geometry_eigvals,
            "pred_geometry_normal": pred_geometry_normal,
            "pred_geometry_scalar": pred_geometry_scalar,
            "pred_sp_vae_texture": pred_sp_vae_texture,
            "pred_sp_field_vae": pred_sp_field_vae,
            "pred_sp_field_geom_radius": pred_sp_field_geom_radius,
            "pred_sp_field_geom_eigvals": pred_sp_field_geom_eigvals,
            "pred_sp_field_geom_normal": pred_sp_field_geom_normal,
            "pred_sp_field_geom_scalar": pred_sp_field_geom_scalar,
            "sp_field_valid_mask": sp_field_valid,
            "pooled_local_feat": pooled_feats[-1],
            "pooled_local_feats": pooled_feats,
            "da3_feature_map_unified": da3_mean_feature_map,
            "da3_texture_feature_map": da3_mean_feature_map,
            "da3_geometry_feature_map": da3_mean_feature_map,
            "da3_last_feature_map": da3_mean_feature_map,
            "pooled_vis_ratio": vis_ratio,
            "attn_vis_ratio": pool_valid_for_attn.float().mean(dim=1).unsqueeze(-1),
            "init_vis_ratio": pool_valid_for_init.float().mean(dim=1).unsqueeze(-1),
            "context_vis_ratio": pool_valid_for_init.float().mean(dim=1).unsqueeze(-1),
            "visible_query_mask": visible_query_mask,
            "visible_query_ratio": visible_query_ratio,
            "img_view_ids": view_ids,
            "img_plucker": plucker,
            **decode_out,
        }
        return out

    # ------------------------------------------------------------------
    # Multi-view warp consistency loss (stage-1.4)
    # ------------------------------------------------------------------
    def _compute_warp_consistency_loss(self, result_dict: dict, batch: dict) -> torch.Tensor:
        rgb = result_dict.get("rendered_rgb", None)
        zero = rgb.new_zeros(()) if torch.is_tensor(rgb) else torch.zeros(())
        if self.warp_loss_weight <= 0 or not torch.is_tensor(rgb) or rgb.dim() != 5:
            return zero
        b, v_rgb, h, w, _ = rgb.shape
        if v_rgb < 2:
            return zero
        depth = batch.get("norm_depth", None)
        valid = batch.get("valid_mask", None)
        if depth is None or valid is None or depth.dim() != 4:
            # Warp needs per-view GT depth [B,V,H,W].
            return zero
        Ks = batch["Ks"].float()
        c2w = batch["norm_camtoworlds"].float()
        depth = depth.float()
        valid = valid.bool()

        view_idx = result_dict.get("selected_view_idx_rgb", None)
        if view_idx is not None:
            gi = view_idx.long()
            K_r = torch.gather(Ks, 1, gi[:, :, None, None].expand(b, v_rgb, 3, 3))
            c2w_r = torch.gather(c2w, 1, gi[:, :, None, None].expand(b, v_rgb, 4, 4))
            depth_r = torch.gather(depth, 1, gi[:, :, None, None].expand(b, v_rgb, h, w))
            valid_r = torch.gather(valid, 1, gi[:, :, None, None].expand(b, v_rgb, h, w))
        else:
            K_r, c2w_r, depth_r, valid_r = Ks[:, :v_rgb], c2w[:, :v_rgb], depth[:, :v_rgb], valid[:, :v_rgb]
        w2c_r = torch.linalg.inv(c2w_r)

        if self.warp_use_rendered_depth and result_dict.get("rendered_depths", None) is not None:
            rd = result_dict["rendered_depths"]
            if rd.shape[1] == v_rgb:
                depth_src = rd[..., 0].float()
            else:
                depth_src = depth_r
        else:
            depth_src = depth_r

        device = rgb.device
        ys, xs = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        rgb_f = rgb.permute(0, 1, 4, 2, 3).float()  # [B,Vr,3,H,W]
        losses: List[torch.Tensor] = []
        for j in range(v_rgb):
            Kj = K_r[:, j]
            fxj = Kj[:, 0, 0].view(b, 1, 1)
            fyj = Kj[:, 1, 1].view(b, 1, 1)
            cxj = Kj[:, 0, 2].view(b, 1, 1)
            cyj = Kj[:, 1, 2].view(b, 1, 1)
            d_j = depth_src[:, j]  # [B,H,W]
            xn = (xs.unsqueeze(0) + 0.5 - cxj) / fxj
            yn = (ys.unsqueeze(0) + 0.5 - cyj) / fyj
            Xc = torch.stack([xn * d_j, yn * d_j, d_j], dim=-1)  # [B,H,W,3]
            Xw = torch.einsum("bij,bhwj->bhwi", c2w_r[:, j, :3, :3], Xc) + c2w_r[:, j, :3, 3].view(b, 1, 1, 3)
            for i in range(v_rgb):
                if i == j:
                    continue
                Xi = torch.einsum("bij,bhwj->bhwi", w2c_r[:, i, :3, :3], Xw) + w2c_r[:, i, :3, 3].view(b, 1, 1, 3)
                zi = Xi[..., 2]
                Ki = K_r[:, i]
                fxi = Ki[:, 0, 0].view(b, 1, 1)
                fyi = Ki[:, 1, 1].view(b, 1, 1)
                cxi = Ki[:, 0, 2].view(b, 1, 1)
                cyi = Ki[:, 1, 2].view(b, 1, 1)
                zi_safe = zi.clamp_min(1e-4)
                # ui/vi are in the half-integer pixel-center convention
                # (unprojection used xs+0.5); grid_sample with
                # align_corners=True wants integer pixel centers -> shift by 0.5.
                ui = fxi * Xi[..., 0] / zi_safe + cxi - 0.5
                vi = fyi * Xi[..., 1] / zi_safe + cyi - 0.5
                in_bounds = (zi > 1e-3) & (ui >= 0.0) & (ui <= float(w - 1)) & (vi >= 0.0) & (vi <= float(h - 1))
                grid = torch.stack(
                    [ui / max(float(w - 1), 1.0) * 2.0 - 1.0, vi / max(float(h - 1), 1.0) * 2.0 - 1.0],
                    dim=-1,
                )  # align_corners=True convention
                d_i_px = F.grid_sample(
                    depth_r[:, i].unsqueeze(1), grid, mode="bilinear", padding_mode="zeros", align_corners=True
                )[:, 0]
                v_i_px = F.grid_sample(
                    valid_r[:, i].float().unsqueeze(1), grid, mode="nearest", padding_mode="zeros", align_corners=True
                )[:, 0] > 0.5
                # z-buffer occlusion test: the transferred point must not sit
                # behind the observed surface in view i.
                visible = zi <= d_i_px * (1.0 + self.warp_tol) + 1e-4
                mask = in_bounds & valid_r[:, j] & v_i_px & visible
                if not bool(mask.any()):
                    continue
                sampled_i = F.grid_sample(rgb_f[:, i], grid, mode="bilinear", padding_mode="zeros", align_corners=True)
                sampled_i = sampled_i.permute(0, 2, 3, 1)  # [B,H,W,3]
                # elementwise charbonnier (the module-level _charbonnier reduces
                # to a scalar, which cannot be masked).
                d = sampled_i - rgb[:, j].float()
                diff = torch.sqrt(d * d + 1e-6).mean(dim=-1)  # [B,H,W]
                losses.append(diff[mask].mean())
        if not losses:
            return zero
        return torch.stack(losses).mean()

    def compute_losses(self, result_dict: dict, batch: dict) -> dict[str, torch.Tensor]:
        losses = super().compute_losses(result_dict, batch)
        warp_loss = self._compute_warp_consistency_loss(result_dict, batch)
        if self.warp_loss_weight > 0:
            losses["loss"] = losses["loss"] + self.warp_loss_weight * warp_loss
        losses["warp_loss"] = warp_loss.detach()
        return losses
