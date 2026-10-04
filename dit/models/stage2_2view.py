"""dit_v26_2view: v26 gated refine DiT with first+last condition.

Independent copy of ``dit_v26``. Task, losses, Wan blocks, and the detached
per-point gate are unchanged. The only model-side addition is a zero-init
first/last embedding so the 6000-token ``cat(first, last)`` condition can
mark which tokens are last-frame.

v26 gate kept: ``s`` from ``stopgrad(h)``, train with
``exp(-||stage1-gt|| / σ)``, infer ``v_used = (1-s) * v`` frozen at the
first refine step.

v23/v26 block structure kept for isolation:

- **Wan-style flat blocks** (no groups): every block is
  ``adaLN self-attn -> cross-attn -> adaLN FFN``.  Cross-attention to the
  first+last reference happens in *every* block (Wan ratio 1:1) instead of
  every 4th layer (v21/v22 ratio 1:3).  Modulation is Wan's shared 6-way
  scheme: one ``time_projection`` maps the time embedding to ``[B, 6, dim]``
  and each block adds its own learnable ``scale_shift_table`` to get
  (shift, scale, gate) for self-attn and FFN; cross-attn is unmodulated,
  exactly as in Wan.  The modulated time embedding is
  ``content_time_mlp(temb(t_con)) + pos_time_mlp(temb(t_pos))``: with 22
  geometry-routed injections per forward (vs 7 in the group structure), the
  trunk must know how dirty x_geo is (t_pos = 1 <=> pure stage-1 anchor).
  ``pos_time_mlp``'s last layer is zero-init so the trunk starts identical
  to a t_con-only modulation.
- **No camera control**: no trajectory encoder / scene planner / camera
  adapter (same as v19+; cameras only define the canonical coordinate frame).
- **3D xyz RoPE instead of Wan's (t, h, w) grid RoPE**: continuous3d RoPE on
  the superpoint coordinates, applied in self-attention (target xyz) and in
  cross-attention (q = target xyz, k = cond xyz) so reference reads stay
  geometry-aligned.
- **Final position refinement head**: a single zero-init nonlinear MLP head on
  the last block predicts the t_pos-independent path velocity
  ``v = stage1 - gt`` from content features + ``x_geo`` + ``t_pos``; the model
  forms the residual explicitly as ``res = t_pos * v`` (the regression target
  ``x_geo - gt = t_pos * d`` is a content-driven term TIMES t_pos, so the
  multiplicative structure is built in, not left to a linear head to
  discover).  Inference Euler integrates the velocity directly
  (``x <- x - v / K``), with no division by ``t_pos``.
- **Losses**: content v-MSE + ungated position residual MSE (same v23
 口径) + detached confidence BCE.  ``cond_drop_prob=0.1`` keeps first-frame
  cond CFG available at inference.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .layers import (
    ExplicitAttention,
    FourierXYZ,
    begin_rope_cache,
    masks_full_cached,
    timestep_embedding,
)


def _transform_points(transform: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    return torch.einsum("bij,bnj->bni", transform, points_h)[..., :3]


def world_standardized_to_camera0(
    points: torch.Tensor,
    cam_c2w: torch.Tensor,
    xyz_mean: torch.Tensor,
    xyz_std: torch.Tensor,
    canonical_scale: float,
) -> torch.Tensor:
    """Convert dataset world-standardized points to camera-0 canonical flow coordinates."""
    world_metric = points * xyz_std[:, None, :].to(dtype=points.dtype)
    world_metric = world_metric + xyz_mean[:, None, :].to(dtype=points.dtype)
    reference_w2c = torch.linalg.inv(cam_c2w[:, 0].float())
    canonical_metric = _transform_points(reference_w2c, world_metric.float())
    return (canonical_metric / float(canonical_scale)).to(dtype=points.dtype)


def normalize_canonical(
    points: torch.Tensor,
    canonical_xyz_mean: torch.Tensor,
    canonical_xyz_std: torch.Tensor,
) -> torch.Tensor:
    """Standardize camera-0 coordinates with training-set canonical statistics."""
    mean = canonical_xyz_mean[:, None, :].to(device=points.device, dtype=points.dtype)
    std = canonical_xyz_std[:, None, :].to(device=points.device, dtype=points.dtype).clamp_min(1e-8)
    return (points - mean) / std


def denormalize_canonical(
    points: torch.Tensor,
    canonical_xyz_mean: torch.Tensor,
    canonical_xyz_std: torch.Tensor,
) -> torch.Tensor:
    """Undo the training-stat standardization in canonical Flow space."""
    mean = canonical_xyz_mean[:, None, :].to(device=points.device, dtype=points.dtype)
    std = canonical_xyz_std[:, None, :].to(device=points.device, dtype=points.dtype).clamp_min(1e-8)
    return points * std + mean


def world_standardized_to_camera0_normalized(
    points: torch.Tensor,
    cam_c2w: torch.Tensor,
    xyz_mean: torch.Tensor,
    xyz_std: torch.Tensor,
    canonical_scale: float,
    canonical_xyz_mean: torch.Tensor,
    canonical_xyz_std: torch.Tensor,
) -> torch.Tensor:
    canonical = world_standardized_to_camera0(
        points, cam_c2w, xyz_mean, xyz_std, canonical_scale
    )
    return normalize_canonical(canonical, canonical_xyz_mean, canonical_xyz_std)


def camera0_to_world_standardized(
    points: torch.Tensor,
    cam_c2w: torch.Tensor,
    xyz_mean: torch.Tensor,
    xyz_std: torch.Tensor,
    canonical_scale: float,
) -> torch.Tensor:
    """Convert camera-0 canonical flow coordinates back to dataset coordinates."""
    canonical_metric = points.float() * float(canonical_scale)
    world_metric = _transform_points(cam_c2w[:, 0].float(), canonical_metric)
    world_standardized = (
        world_metric - xyz_mean[:, None, :].float()
    ) / xyz_std[:, None, :].float().clamp_min(1e-8)
    return world_standardized.to(dtype=points.dtype)


class _WanStyleBlock(nn.Module):
    """Wan2.1-style block: adaLN self-attn + plain-norm cross-attn + adaLN FFN.

    Both attentions use continuous 3D xyz RoPE (self: target xyz; cross:
    q = target xyz, k = cond xyz).  ``mod`` is the shared [B, 6, dim] time
    modulation; the block's learnable ``scale_shift_table`` is added on top,
    exactly like Wan.  Cross-attention is unmodulated (Wan keeps it a plain
    affine LayerNorm read of the context).
    """

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        mlp_ratio: float,
        rope_base: float,
        rope_fwd: float,
        rope_bits: int,
        rope_impl: str,
        use_flash: bool,
    ) -> None:
        super().__init__()
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 6, hidden_dim) / hidden_dim**0.5
        )
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.self_attn = ExplicitAttention(
            hidden_dim,
            heads,
            use_rope=True,
            qk_norm=True,
            use_flash_varlen=use_flash,
            rope_base=rope_base,
            rope_fwd=rope_fwd,
            rope_bits=rope_bits,
            rope_impl=rope_impl,
        )
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.cross_attn = ExplicitAttention(
            hidden_dim,
            heads,
            use_rope=True,
            qk_norm=True,
            use_flash_varlen=use_flash,
            rope_base=rope_base,
            rope_fwd=rope_fwd,
            rope_bits=rope_bits,
            rope_impl=rope_impl,
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        ffn_dim = int(hidden_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )

    def forward(
        self,
        h: torch.Tensor,
        cond_h: torch.Tensor,
        x_geo: torch.Tensor,
        cond_xyz: torch.Tensor,
        mod: torch.Tensor,
        target_mask: torch.Tensor,
        cond_mask: torch.Tensor,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table + mod
        ).chunk(6, dim=1)
        x = self.norm1(h) * (1.0 + scale_msa) + shift_msa
        h = h + gate_msa * self.self_attn(
            x, x, x,
            q_mask=target_mask,
            k_mask=target_mask,
            q_pos=x_geo,
            k_pos=x_geo,
        )
        h = h + self.cross_attn(
            self.norm3(h),
            cond_h,
            cond_h,
            q_mask=target_mask,
            k_mask=cond_mask,
            q_pos=x_geo,
            k_pos=cond_xyz,
        )
        x = self.norm2(h) * (1.0 + scale_mlp) + shift_mlp
        h = h + gate_mlp * self.ffn(x)
        return h


class _ContentPosRefineHead(nn.Module):
    """Final head: predict the t_pos-INDEPENDENT path velocity ``v = stage1 - gt``.

    Structural fix for "the head cannot express its regression target": the
    residual target is ``res = t_pos * d`` (content-driven term TIMES t_pos),
    so the multiplicative structure is built in explicitly -- this MLP predicts
    the velocity ``v = d`` and the model multiplies by ``t_pos`` outside to
    form the residual (see PointDiTV26TwoView.forward).  Consequences: idempotency at
    ``t_pos = 0`` by construction, the head's task is t-independent ("estimate
    d", large-t supervision transfers to small t), and inference Euler reads
    the velocity directly (``x <- x - v / K``, no division by t_pos).

    Nonlinear MLP over [content features, fourier xyz, t_pos embedding]; the
    last layer is zero-init so the initial prediction is ``v = 0``
    (``x0hat = x_geo``, identity).
    """

    def __init__(
        self,
        hidden_dim: int,
        xyz_pe_dim: int,
        refine_dim: int,
        time_dim: int,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.con_norm = nn.LayerNorm(hidden_dim)
        self.xyz_in = nn.Linear(xyz_pe_dim, refine_dim)
        self.time_in = nn.Linear(time_dim, refine_dim)
        self.xyz_norm = nn.LayerNorm(refine_dim)
        mlp_dim = refine_dim * int(mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + refine_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.SiLU(),
        )
        self.fc = nn.Linear(mlp_dim, 3)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(
        self,
        h: torch.Tensor,
        xyz_features: torch.Tensor,
        t_pos_embedding: torch.Tensor,
    ) -> torch.Tensor:
        xyz_feat = self.xyz_norm(self.xyz_in(xyz_features) + self.time_in(t_pos_embedding)[:, None])
        return self.fc(self.mlp(torch.cat([self.con_norm(h), xyz_feat], dim=-1)))


class _PointConfidenceHead(nn.Module):
    """Detached side head: per-point confidence ``s ∈ (0,1)``.

    Target is ``y = exp(-||stage1-gt|| / σ)`` (t-independent).  Reads
    ``stopgrad(h)`` so this loss cannot steer the trunk or the velocity
    head.  Last-layer bias is -4 so ``s ≈ 0.018`` at init (almost full
    refine, v23-like) until the head learns to raise ``s`` on near points.
    """

    def __init__(
        self,
        hidden_dim: int,
        xyz_pe_dim: int,
        refine_dim: int,
        time_dim: int,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.con_norm = nn.LayerNorm(hidden_dim)
        self.xyz_in = nn.Linear(xyz_pe_dim, refine_dim)
        self.time_in = nn.Linear(time_dim, refine_dim)
        self.xyz_norm = nn.LayerNorm(refine_dim)
        mlp_dim = refine_dim * int(mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + refine_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.SiLU(),
        )
        self.fc = nn.Linear(mlp_dim, 1)
        nn.init.zeros_(self.fc.weight)
        nn.init.constant_(self.fc.bias, -4.0)

    def forward(
        self,
        h: torch.Tensor,
        xyz_features: torch.Tensor,
        t_pos_embedding: torch.Tensor,
    ) -> torch.Tensor:
        xyz_feat = self.xyz_norm(self.xyz_in(xyz_features) + self.time_in(t_pos_embedding)[:, None])
        return self.fc(self.mlp(torch.cat([self.con_norm(h), xyz_feat], dim=-1)))


class PointDiTV26TwoView(nn.Module):
    """v26_2view: v26 gated refine + first/last cond view embedding."""

    requires_xyz_stats = True
    position_flow_frame = "camera0_canonical_normalized"

    def __init__(
        self,
        *,
        target_dim: int,
        cond_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        time_dim: int,
        xyz_fourier_bands: int,
        content_rope_base: float,
        content_rope_fwd: float,
        content_rope_bits: int,
        content_rope_impl: str,
        aux_con_layer: int,
        use_flash_varlen: bool,
        canonical_position_scale: float,
        canonical_xyz_mean: tuple[float, float, float] | list[float],
        canonical_xyz_std: tuple[float, float, float] | list[float],
        refine_xyz_dim: int = 128,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if canonical_position_scale <= 0.0:
            raise ValueError("canonical_position_scale must be positive")
        if len(canonical_xyz_mean) != 3 or len(canonical_xyz_std) != 3:
            raise ValueError("canonical_xyz_mean/std must each contain exactly 3 values")
        if any(float(value) <= 0.0 for value in canonical_xyz_std):
            raise ValueError("canonical_xyz_std values must all be positive")

        self.target_dim = int(target_dim)
        self.cond_dim = int(cond_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.time_dim = int(time_dim)
        self.canonical_position_scale = float(canonical_position_scale)
        self.register_buffer(
            "canonical_xyz_mean",
            torch.tensor(canonical_xyz_mean, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "canonical_xyz_std",
            torch.tensor(canonical_xyz_std, dtype=torch.float32),
            persistent=True,
        )
        self.aux_con_layer_index = int(aux_con_layer)

        self.xyz_pe = FourierXYZ(xyz_fourier_bands)
        self.con_in = nn.Linear(target_dim, hidden_dim)
        self.cond_con_in = nn.Linear(cond_dim, hidden_dim)
        self.content_time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4), nn.SiLU(), nn.Linear(time_dim * 4, time_dim)
        )
        # Geometry-quality time enters the trunk too: every block routes via
        # 3D RoPE on x_geo, so it must know how dirty that geometry is
        # (t_pos = 1 <=> pure stage-1 anchor).  Zero-init last layer => at
        # init the trunk is identical to the t_con-only modulation.
        self.pos_time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4), nn.SiLU(), nn.Linear(time_dim * 4, time_dim)
        )
        nn.init.zeros_(self.pos_time_mlp[-1].weight)
        nn.init.zeros_(self.pos_time_mlp[-1].bias)
        # Wan-style shared 6-way modulation: one projection from the time
        # embedding, each block adds its own scale_shift_table.
        self.time_projection = nn.Linear(time_dim, 6 * hidden_dim)
        self.null_cond_con = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        # Auxiliary first/last flag. Geometry already differs; this only helps
        # the model mark which memory tokens came from the last frame.
        self.view_embed = nn.Embedding(2, hidden_dim)
        nn.init.zeros_(self.view_embed.weight)

        self.blocks = nn.ModuleList(
            _WanStyleBlock(
                hidden_dim,
                num_heads,
                mlp_ratio,
                content_rope_base,
                content_rope_fwd,
                content_rope_bits,
                content_rope_impl,
                use_flash_varlen,
            )
            for _ in range(self.num_layers)
        )

        # Single final refinement head: a pure side-output whose prediction
        # never feeds back into RoPE inside the forward pass.
        self.refine_head = _ContentPosRefineHead(
            hidden_dim, self.xyz_pe.out_dim, refine_xyz_dim, time_dim, mlp_ratio=4
        )
        self.conf_head = _PointConfidenceHead(
            hidden_dim, self.xyz_pe.out_dim, refine_xyz_dim, time_dim, mlp_ratio=4
        )
        self.out_con = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, target_dim))
        nn.init.zeros_(self.out_con[-1].weight)
        nn.init.zeros_(self.out_con[-1].bias)

        if self.aux_con_layer_index > self.num_layers:
            raise ValueError(f"aux_con_layer={aux_con_layer} exceeds total layer count={self.num_layers}")
        self.aux_con_head = (
            nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, target_dim))
            if self.aux_con_layer_index > 0
            else None
        )

    def world_standardized_to_canonical(
        self,
        points: torch.Tensor,
        cam_c2w: torch.Tensor,
        xyz_mean: torch.Tensor,
        xyz_std: torch.Tensor,
    ) -> torch.Tensor:
        return world_standardized_to_camera0_normalized(
            points,
            cam_c2w,
            xyz_mean,
            xyz_std,
            self.canonical_position_scale,
            self.canonical_xyz_mean[None].expand(points.shape[0], -1),
            self.canonical_xyz_std[None].expand(points.shape[0], -1),
        )

    def canonical_to_world_standardized(
        self,
        points: torch.Tensor,
        cam_c2w: torch.Tensor,
        xyz_mean: torch.Tensor,
        xyz_std: torch.Tensor,
    ) -> torch.Tensor:
        canonical = denormalize_canonical(
            points,
            self.canonical_xyz_mean[None].expand(points.shape[0], -1),
            self.canonical_xyz_std[None].expand(points.shape[0], -1),
        )
        return camera0_to_world_standardized(
            canonical, cam_c2w, xyz_mean, xyz_std, self.canonical_position_scale
        )

    def forward(
        self,
        xt_pos: torch.Tensor,
        xt_con: torch.Tensor,
        t_pos: torch.Tensor,
        t_con: torch.Tensor,
        target_mask: torch.Tensor,
        cond_pos: torch.Tensor,
        cond_con: torch.Tensor,
        cond_mask: torch.Tensor,
        cam_c2w: torch.Tensor,
        drop_cond: bool = False,
        xyz_mean: Optional[torch.Tensor] = None,
        xyz_std: Optional[torch.Tensor] = None,
        cond_view_index: Optional[torch.Tensor] = None,
        cond_frame_index: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        begin_rope_cache()
        del cond_frame_index
        b = xt_pos.shape[0]
        if xyz_mean is None or xyz_std is None:
            raise ValueError("v26_2view requires xyz_mean and xyz_std for camera-relative geometry")
        if t_pos.shape != (b,) or t_con.shape != (b,):
            raise ValueError(f"t_pos and t_con must both have shape ({b},)")

        target_mask = target_mask.bool()
        cond_mask = cond_mask.bool()
        target_full = masks_full_cached(target_mask)
        if drop_cond:
            cond_pos_input = cond_pos.new_zeros((b, 1, 3))
            cond_con_input = cond_con.new_zeros((b, 1, self.cond_dim))
            cond_mask = torch.ones((b, 1), dtype=torch.bool, device=cond_pos.device)
            cond_view_index = torch.zeros((b, 1), dtype=torch.int64, device=cond_pos.device)
        else:
            cond_pos_input = cond_pos
            cond_con_input = cond_con

        cond_canonical = self.world_standardized_to_canonical(
            cond_pos_input, cam_c2w, xyz_mean, xyz_std
        )
        if drop_cond:
            cond_canonical = torch.zeros_like(cond_canonical)
        cond_h = self.cond_con_in(cond_con_input)
        if drop_cond:
            cond_h = self.null_cond_con.expand(b, 1, -1)
        else:
            if cond_view_index is None:
                view = torch.zeros(cond_mask.shape, dtype=torch.int64, device=cond_mask.device)
            else:
                view = cond_view_index.to(device=cond_mask.device, dtype=torch.int64)
                if view.dim() == 1:
                    view = view.unsqueeze(0).expand(cond_mask.shape[0], -1)
                if view.shape != cond_mask.shape:
                    raise ValueError(
                        f"cond_view_index shape {tuple(view.shape)} does not match "
                        f"cond tokens {tuple(cond_mask.shape)}"
                    )
            cond_h = cond_h + self.view_embed(view.clamp(0, 1))

        pos_temb = timestep_embedding(t_pos, self.time_dim)
        con_te = self.content_time_mlp(timestep_embedding(t_con, self.time_dim)) + self.pos_time_mlp(
            pos_temb
        )
        mod = self.time_projection(con_te).view(b, 6, self.hidden_dim)
        h = self.con_in(xt_con)
        # Geometry is fixed for the whole forward pass: xt_pos is the sampled
        # geometry-quality state x_geo, and every block's 3D RoPE reads it as-is.
        x_geo = xt_pos

        def _mask(x: torch.Tensor) -> torch.Tensor:
            if target_full:
                return x
            return x.masked_fill(~target_mask.unsqueeze(-1), 0.0)

        aux_con = None
        for layer_index, block in enumerate(self.blocks, start=1):
            h = block(h, cond_h, x_geo, cond_canonical, mod, target_mask, cond_mask)
            if self.aux_con_head is not None and layer_index == self.aux_con_layer_index:
                aux_con = _mask(self.aux_con_head(h))

        h = _mask(h)
        # The head predicts the t_pos-independent velocity v = stage1 - gt; the
        # multiplicative structure res = t_pos * v is explicit (the residual
        # target IS t_pos * d), so idempotency at t_pos = 0 holds by
        # construction and the loss stays the residual/x0 MSE of the v20+ runs.
        # Zero-init => v = 0 => res = 0 and x0hat = x_geo (identity).
        xyz_features = self.xyz_pe(x_geo)
        v_pos = _mask(self.refine_head(h, xyz_features, pos_temb))
        # Gate reads a detached trunk so its BCE cannot move content/refine.
        conf_logit = _mask(self.conf_head(h.detach(), xyz_features, pos_temb))
        conf = _mask(torch.sigmoid(conf_logit))
        res_pos = _mask(t_pos.view(b, 1, 1) * v_pos)
        x0hat_pos = _mask(x_geo - res_pos)
        out = {
            # v_pos is the inference Euler velocity; res_pos is the loss target
            # side (residual MSE == x0 MSE, comparable to the v20-v22 runs).
            # conf is not applied here: training keeps the ungated residual.
            "v_pos": v_pos,
            "res_pos": res_pos,
            "x0hat_pos": x0hat_pos,
            "conf": conf,
            "conf_logit": conf_logit,
            "v_con": _mask(self.out_con(h)),
            # Single head: stack of one, keeping the v20+ trainer/eval contract.
            "aux_res_pos": res_pos.unsqueeze(0),
            "aux_x0hat_pos": x0hat_pos.unsqueeze(0),
        }
        if aux_con is not None:
            out["aux_x0_con"] = aux_con
        return out
