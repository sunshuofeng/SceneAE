"""SceneLatent VAE: trainable superpoint encoder and the v30 grid decoder.

RGB and depth do not share a ViT. Depth is a residual on the splatted
inverse-depth prior. Training encodes each batch. A batch that still carries
``cached_z_sp`` skips the encoder; the train loop drops that field when the
encoder is unfrozen. DA3 stays frozen.
"""



from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import _expand_to_views, _image_gradient_loss
from .grid_decoder import SuperpointGridDecoder
from .vae_frozen import SuperpointVAE2CV2


class SuperpointVAEGridV30(SuperpointVAE2CV2):
    def __init__(
        self,
        *args,
        # ---- v27 grid decoder params ----
        grid_token_dim: int = 792,
        grid_patch: int = 14,
        grid_image_size=(378, 518),
        grid_k_init: int = 8,
        grid_k_xattn: int = 64,
        grid_n_groups: int = 4,
        grid_shared_layers: int = 0,
        grid_rgb_layers: int = 24,
        grid_depth_layers: int = 3,
        grid_depth_prep_layers: int = 8,
        grid_depth_dim: int = 384,
        grid_n_heads: int = 12,
        grid_depth_heads: int = 8,
        grid_mlp_ratio: float = 4.0,
        grid_rgb_num_views: int = 3,
        grid_depth_num_views: int = 3,
        grid_alpha_grad_gate: float = 0.5,
        grid_beta_init: float = 3.0,
        grid_gamma_init: float = 1.0,
        grid_depth_min_norm: float = 0.02,
        grid_depth_max_norm: float = 50.0,
        grid_depth_pct_lo: float = 1.0,
        grid_depth_pct_hi: float = 99.0,
        # ---- v27 loss weights (design §5) ----
        rgb_l1_weight: float = 1.0,
        rgb_grad_weight_v27: float = 0.5,
        rgb_lpips_weight: float = 0.3,
        depth_l1_weight: float = 1.0,
        depth_grad_weight_v27: float = 0.2,
        **kwargs,
    ) -> None:
        # v27 loss weights (stored before super().__init__; base names differ)
        self.rgb_l1_weight = float(rgb_l1_weight)
        self.rgb_grad_weight_v27 = float(rgb_grad_weight_v27)
        self.rgb_lpips_weight = float(rgb_lpips_weight)
        self.depth_l1_weight = float(depth_l1_weight)
        self.depth_grad_weight_v27 = float(depth_grad_weight_v27)

        # The grid decoder replaces the aniso / warp / inverse-depth machinery of
        # V2; force those off.  lpips_loss_weight is forwarded so the base builds
        # the (frozen) LPIPS model that our compute_losses uses.
        kwargs["warp_loss_weight"] = 0.0
        kwargs["decoder_use_geometry_aniso"] = False
        kwargs["depth_inverse_param"] = False
        kwargs["lpips_loss_weight"] = float(rgb_lpips_weight)
        # decode-side jitter sigma starts at 0; the train loop ramps it (§6.2).
        kwargs["decode_sp_xyz_jitter_sigma"] = 0.0

        super().__init__(*args, **kwargs)

        latent_dim = int(self.decoder.feat_dim)
        old = self.decoder
        self.decoder = self._build_grid_decoder(
            latent_dim,
            token_dim=grid_token_dim,
            patch=grid_patch,
            image_size=tuple(grid_image_size),
            k_init=grid_k_init,
            k_xattn=grid_k_xattn,
            n_groups=grid_n_groups,
            shared_layers=grid_shared_layers,
            rgb_layers=grid_rgb_layers,
            depth_layers=grid_depth_layers,
            depth_prep_layers=grid_depth_prep_layers,
            depth_dim=grid_depth_dim,
            n_heads=grid_n_heads,
            depth_heads=grid_depth_heads,
            mlp_ratio=grid_mlp_ratio,
            rgb_num_views=grid_rgb_num_views,
            depth_num_views=grid_depth_num_views,
            alpha_grad_gate=grid_alpha_grad_gate,
            beta_init=grid_beta_init,
            gamma_init=grid_gamma_init,
            depth_min_norm=grid_depth_min_norm,
            depth_max_norm=grid_depth_max_norm,
            depth_pct_lo=grid_depth_pct_lo,
            depth_pct_hi=grid_depth_pct_hi,
        )
        del old
        self.grid_image_size = tuple(grid_image_size)

    @staticmethod
    def _build_grid_decoder(latent_dim: int, **grid_kwargs):
        rgb = int(grid_kwargs.get("rgb_layers", 24))
        prep = int(grid_kwargs.get("depth_prep_layers", 8))
        grid_kwargs.setdefault("xattn_after_shared", ())
        grid_kwargs.setdefault("xattn_after_rgb", tuple(range(0, rgb, 2)))
        grid_kwargs.setdefault("xattn_after_depth_prep", tuple(range(0, prep, 2)))
        return SuperpointGridDecoder(superpoint_dim=latent_dim, **grid_kwargs)

    # ------------------------------------------------------------------
    # Decode-side jitter: per-sample Bernoulli (design §6.2 fix #1).
    # Sigma is driven externally by the curriculum (set_jitter_sigma).
    # ------------------------------------------------------------------
    def set_jitter_sigma(self, sigma: float) -> None:
        self.decode_sp_xyz_jitter_sigma = float(sigma)

    def _maybe_jitter_decode_sp_xyz(
        self,
        sp_xyz: torch.Tensor,
        sp_valid_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if (not self.training) or self.decode_sp_xyz_jitter_sigma <= 0:
            return sp_xyz
        B = sp_xyz.shape[0]
        apply = (torch.rand(B, device=sp_xyz.device) < self.decode_sp_xyz_jitter_prob).to(sp_xyz.dtype)
        noise = torch.randn_like(sp_xyz) * self.decode_sp_xyz_jitter_sigma
        noise = noise * apply[:, None, None]
        if sp_valid_mask is not None:
            noise = noise * sp_valid_mask.to(device=sp_xyz.device, dtype=noise.dtype).unsqueeze(-1)
        return sp_xyz + noise

    # ------------------------------------------------------------------
    # Cached-latent path: no frozen geometry heads (grid decoder does not use
    # aniso), straight to the grid decoder.
    # ------------------------------------------------------------------
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
        if image.dim() != 5:
            raise ValueError(f"image expected [B,V,H,W,3], got {tuple(image.shape)}")
        sp_xyz = torch.nan_to_num(sp_xyz, nan=0.0, posinf=0.0, neginf=0.0)
        z_sp = torch.nan_to_num(
            cached_z_sp.to(device=sp_xyz.device, dtype=torch.float32),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        if z_sp.shape[:2] != sp_xyz.shape[:2]:
            raise ValueError(f"cached_z_sp {tuple(z_sp.shape)} vs sp_xyz {tuple(sp_xyz.shape)}")
        if sp_valid_mask is not None:
            sp_valid_mask = sp_valid_mask.to(device=sp_xyz.device).bool()

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
        )
        if decode_out.get("selected_view_idx_rgb", None) is not None:
            decode_out["selected_view_idx_rgb_local"] = decode_out["selected_view_idx_rgb"]
        if decode_out.get("selected_view_idx_depth", None) is not None:
            decode_out["selected_view_idx_depth_local"] = decode_out["selected_view_idx_depth"]

        out: Dict[str, torch.Tensor] = {
            "free_latent_sp": z_sp,
            "sp_xyz_initial": sp_xyz,
            "sp_xyz_refined": sp_xyz,
            "sp_xyz_decode": sp_xyz_dec,
            **decode_out,
        }
        return out

    # ------------------------------------------------------------------
    # v27 loss set (design §5): rgb_l1 / rgb_grad / rgb_lpips + depth_l1 / depth_grad.
    # ------------------------------------------------------------------
    def compute_losses(self, result_dict: dict, batch: dict) -> dict[str, torch.Tensor]:
        pred_rgb = result_dict["rendered_rgb"]               # [B,Vr,H,W,3]
        b, v_rgb, h, w, _ = pred_rgb.shape
        v_all = batch["image"].shape[1]
        pred_depth = result_dict["rendered_depths"][..., 0]  # [B,Vd,H,W]

        # ---- RGB ----
        rgb_idx = result_dict.get("selected_view_idx_rgb", None)
        gt_rgb = self._gather_selected_gt_rgb_full(batch["image"], rgb_idx, v_rgb).to(pred_rgb.dtype)
        pred_rgb_nchw = pred_rgb.permute(0, 1, 4, 2, 3).reshape(b * v_rgb, 3, h, w)
        gt_rgb_nchw = gt_rgb.permute(0, 1, 4, 2, 3).reshape(b * v_rgb, 3, h, w)
        rgb_l1 = (pred_rgb.float() - gt_rgb.float()).abs().mean()
        rgb_grad = _image_gradient_loss(pred_rgb_nchw, gt_rgb_nchw)
        rgb_lpips = pred_rgb.new_zeros(())
        if self.rgb_lpips_weight > 0 and getattr(self, "_lpips_model", None) is not None:
            self._lpips_model.to(pred_rgb.device)
            rgb_lpips = self._lpips_model(
                pred_rgb_nchw * 2.0 - 1.0, gt_rgb_nchw * 2.0 - 1.0
            ).mean()
        rgb_loss = (
            self.rgb_l1_weight * rgb_l1
            + self.rgb_grad_weight_v27 * rgb_grad
            + self.rgb_lpips_weight * rgb_lpips
        )

        # ---- depth ----
        gt_depth = batch["norm_depth"]
        if gt_depth.dim() == 3:
            gt_depth = _expand_to_views(gt_depth, v_all)
        valid = batch["valid_mask"].bool()
        if valid.dim() == 3:
            valid = _expand_to_views(valid, v_all)
        depth_idx = result_dict.get("selected_view_idx_depth", None)
        if depth_idx is not None:
            depth_idx = depth_idx.long()
            v_depth = depth_idx.shape[1]
            gather = depth_idx[:, :, None, None].expand(b, v_depth, h, w)
            gt_depth = torch.gather(gt_depth, 1, gather)
            valid = torch.gather(valid, 1, gather)
        pred_depth = torch.nan_to_num(pred_depth, nan=0.0, posinf=1e3, neginf=-1e3)
        dep_diff = (pred_depth - gt_depth).abs()
        depth_l1 = dep_diff[valid].mean() if bool(valid.any()) else dep_diff.mean() * 0.0
        depth_grad = self._compute_dense_depth_grad_loss(pred_depth, gt_depth, valid)
        depth_loss = self.depth_l1_weight * depth_l1 + self.depth_grad_weight_v27 * depth_grad

        loss = rgb_loss + depth_loss

        losses: dict[str, torch.Tensor] = {
            "loss": loss,
            "rgb_loss": rgb_loss.detach(),
            "rgb_l1": rgb_l1.detach(),
            "rgb_full": rgb_l1.detach(),  # alias so the train loop's rgb_full key works
            "rgb_grad": rgb_grad.detach(),
            "rgb_lpips": rgb_lpips.detach(),
            "depth_loss": depth_loss.detach(),
            "depth_l1": depth_l1.detach(),
            "depth_grad": depth_grad.detach(),
        }
        # frontend monitoring stats (design §6.4) ride along for logging
        for key in (
            "frontend_w_null_mean",
            "frontend_w_entropy",
            "frontend_k_valid_mean",
            "frontend_beta_z",
            "frontend_beta_r",
            "frontend_wp_over_wv",
            "frontend_gamma",
            "depth_residual_abs",
            "depth_u_prior_mean",
            "cluster_drop_frac",
        ):
            val = result_dict.get(key, None)
            if torch.is_tensor(val):
                losses[key] = val.detach()
        return losses
