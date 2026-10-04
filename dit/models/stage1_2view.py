"""v18_v3_2view stage-1 position DiT: 768d x 8L, first+last cond.

Standalone copy of ``TwoStagePositionDiTV18V3``. Condition memory consumes
``cat(first, last)``. xyz / Fourier / 3D RoPE stay in camera-0 canonical.

Unlike the 1-view copy, last-token descriptors use the last camera's pose/K:
ray, depth, and lattice uv are computed in that source camera, then the true
observation ray is expressed in camera-0 as a Plücker coordinate. A small
first/last embedding is added on top of that geometry; it does not replace it.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .layers import (
    AdaLN,
    ExplicitAttention,
    FourierXYZ,
    HeadRMSNorm,
    _apply_1d_continuous_rope,
    begin_rope_cache,
    camera0_to_world_standardized,
    denormalize_canonical,
    masks_full_cached,
    timestep_embedding,
    world_standardized_to_camera0_normalized,
)

# [plucker(6), lattice_u(1), lattice_v(1), traj_time(2), known_log_depth(1), known_mask(1)]
RAY_FEATURE_DIM = 12
# [plucker_dir(3), plucker_moment(3), log_depth(1), valid(1), src_lattice_u(1), src_lattice_v(1)]
POINT_DESC_DIM = 10
_BIG_DEPTH = 1.0e4


def _scale_intrinsics(Ks: torch.Tensor, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> torch.Tensor:
    """Mirror of E3D ``plucker_camera.scale_intrinsics`` (kept local so v18_v3 has no cross-repo import)."""
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    out = Ks.clone()
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    out[..., 0, 0] = out[..., 0, 0] * sx
    out[..., 0, 2] = out[..., 0, 2] * sx
    out[..., 1, 1] = out[..., 1, 1] * sy
    out[..., 1, 2] = out[..., 1, 2] * sy
    return out


def _apply_2d_rope(tokens: torch.Tensor, uv: torch.Tensor, base: float, fwd: float) -> torch.Tensor:
    """Rotate the first half of each head by u and the second half by v.

    tokens: [B, L, H, Dh], uv: [B, L, 2].
    """
    head_dim = tokens.shape[-1]
    rope_dim = (head_dim // 4) * 4
    if rope_dim == 0:
        return tokens
    axis_dim = rope_dim // 2
    rope_part = tokens[..., :rope_dim]
    pass_part = tokens[..., rope_dim:]
    u_part, v_part = rope_part.split(axis_dim, dim=-1)
    rotated = torch.cat(
        [
            _apply_1d_continuous_rope(u_part, uv[..., 0], base, fwd),
            _apply_1d_continuous_rope(v_part, uv[..., 1], base, fwd),
        ],
        dim=-1,
    )
    if pass_part.shape[-1] == 0:
        return rotated
    return torch.cat([rotated, pass_part], dim=-1)


class _RayMapCrossAttention(nn.Module):
    """Per-view cross-attention from target queries to future-camera ray tokens.

    The query is projected once and then given a per-view 2D lattice coordinate, so the
    attention logit depends on "where do I land in this view's image plane" rather than
    on a 3D distance to the camera centre. Per-view outputs are summed (no softmax over
    views) and gated by whether the query is actually in front of that camera.
    """

    def __init__(self, dim: int, ray_dim: int, heads: int, rope_base: float, rope_fwd: float) -> None:
        super().__init__()
        if ray_dim % heads != 0:
            raise ValueError(f"raymap_dim={ray_dim} must be divisible by raymap_heads={heads}")
        self.ray_dim = int(ray_dim)
        self.heads = int(heads)
        self.head_dim = self.ray_dim // self.heads
        self.rope_base = float(rope_base)
        self.rope_fwd = float(rope_fwd)
        self.q_proj = nn.Linear(dim, ray_dim)
        self.k_proj = nn.Linear(ray_dim, ray_dim)
        self.v_proj = nn.Linear(ray_dim, ray_dim)
        self.q_norm = HeadRMSNorm(self.head_dim)
        self.k_norm = HeadRMSNorm(self.head_dim)
        # per-view query modulation from [log depth, in-front flag]; zero-init so it opens gradually
        self.depth_proj = nn.Linear(2, ray_dim)
        nn.init.zeros_(self.depth_proj.weight)
        nn.init.zeros_(self.depth_proj.bias)
        self.out_proj = nn.Linear(ray_dim, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        q_x: torch.Tensor,
        ray_tokens: torch.Tensor,
        ray_uv: torch.Tensor,
        ray_mask: torch.Tensor,
        q_uv: torch.Tensor,
        q_log_depth: torch.Tensor,
        q_gate: torch.Tensor,
        q_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, views, m, _ = ray_tokens.shape
        n = q_x.shape[1]
        if views == 0 or m == 0:
            return q_x.new_zeros(q_x.shape)

        full_rays = masks_full_cached(ray_mask)
        q = self.q_proj(q_x)[:, None].expand(b, views, n, self.ray_dim)
        q = q + self.depth_proj(torch.stack([q_log_depth, q_gate], dim=-1).to(q.dtype))
        q = q.reshape(b * views, n, self.heads, self.head_dim)
        k = self.k_proj(ray_tokens).reshape(b * views, m, self.heads, self.head_dim)
        v = self.v_proj(ray_tokens).reshape(b * views, m, self.heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = _apply_2d_rope(q, q_uv.reshape(b * views, n, 2), self.rope_base, self.rope_fwd)
        k = _apply_2d_rope(k, ray_uv.reshape(b * views, m, 2), self.rope_base, self.rope_fwd)

        key_mask = ray_mask.reshape(b * views, m)
        if full_rays:
            # no attn_mask: SDPA can take its flash backend, and empty-row handling is
            # unnecessary because every row has all keys valid
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
            ).transpose(1, 2)
        else:
            # additive mask: all-invalid rows (dropped views) produce finite outputs that
            # are zeroed by view_available below; no host sync required
            attn_mask = q.new_zeros((b * views, 1, 1, m)).masked_fill(~key_mask[:, None, None, :], -10000.0)
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=attn_mask,
            ).transpose(1, 2)
        out = out.reshape(b, views, n, self.ray_dim)

        view_available = ray_mask.any(dim=-1).to(q_gate.dtype)[:, :, None]
        gate = (q_gate * view_available).to(out.dtype).unsqueeze(-1)
        out = (out * gate).sum(dim=1)  # sum over views; deliberately NOT a softmax over views
        return self.out_proj(out).masked_fill(~q_mask.unsqueeze(-1), 0.0)


class _Stage1PositionBlock(nn.Module):
    """Self-attention over target points + 3D-RoPE cross-attention to cond points."""

    def __init__(
        self,
        dim: int,
        time_dim: int,
        heads: int,
        mlp_ratio: float,
        rope_cfg: dict,
        use_flash: bool,
    ) -> None:
        super().__init__()
        self.self_norm = AdaLN(dim, time_dim)
        self.self_attn = ExplicitAttention(dim, heads, use_rope=True, use_flash_varlen=use_flash, **rope_cfg)
        self.mem_norm = AdaLN(dim, time_dim)
        self.mem_kv_norm = nn.LayerNorm(dim)
        self.mem_cross = ExplicitAttention(dim, heads, use_rope=True, use_flash_varlen=use_flash, **rope_cfg)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim))

    def forward(
        self,
        q: torch.Tensor,
        route: torch.Tensor,
        memory: torch.Tensor,
        memory_pos: torch.Tensor,
        memory_mask: torch.Tensor,
        memory_bias: torch.Tensor,
        te: torch.Tensor,
        q_mask: torch.Tensor,
        rope_gate: torch.Tensor,
    ) -> torch.Tensor:
        qn = self.self_norm(q, te)
        q = q + self.self_attn(qn, qn, qn, q_mask=q_mask, k_mask=q_mask, q_pos=route, k_pos=route, rope_gate=rope_gate)
        qn = self.mem_norm(q, te)
        mem = self.mem_kv_norm(memory)
        q = q + self.mem_cross(
            qn,
            mem,
            mem,
            q_mask=q_mask,
            k_mask=memory_mask,
            q_pos=route,
            k_pos=memory_pos,
            rope_gate=rope_gate,
            attn_bias=memory_bias[:, None, None, :],
        )
        out = q + self.mlp(q)
        if masks_full_cached(q_mask):
            return out
        return out.masked_fill(~q_mask.unsqueeze(-1), 0.0)


class TwoStagePositionDiTV18V3TwoView(nn.Module):
    """Stage-1 position DiT for 2-view v18_v3: first+last point memory + future-camera ray map."""

    requires_xyz_stats = True
    position_flow_frame = "camera0_canonical_normalized"

    def __init__(
        self,
        *,
        cond_dim: int,
        pos_hidden_dim: int,
        num_groups: int,
        layers_per_group: int,
        pos_heads: int,
        mlp_ratio: float,
        time_dim: int,
        xyz_fourier_bands: int,
        max_target_points: int,
        max_views: int,
        content_rope_base: float,
        content_rope_fwd: float,
        content_rope_bits: int,
        content_rope_impl: str,
        content_rope_t_mid: float,
        content_rope_t_scale: float,
        use_flash_varlen: bool,
        camera_image_width: int,
        camera_image_height: int,
        canonical_position_scale: float,
        canonical_xyz_mean: tuple[float, float, float] | list[float],
        canonical_xyz_std: tuple[float, float, float] | list[float],
        route_gradient_scale: float,
        raymap_token_grid: int,
        raymap_stride: int,
        raymap_dim: int,
        raymap_heads: int,
        raymap_rope_base: float,
        raymap_rope_fwd: float,
        raymap_uv_clamp: float,
    ) -> None:
        super().__init__()
        if num_groups < 1 or layers_per_group < 1:
            raise ValueError("num_groups and layers_per_group must be positive")
        if max_target_points < 1:
            raise ValueError("max_target_points must be positive")
        if len(canonical_xyz_mean) != 3 or len(canonical_xyz_std) != 3:
            raise ValueError("canonical_xyz_mean/std must each contain exactly 3 values")
        if raymap_token_grid < 2:
            raise ValueError("raymap_token_grid must be at least 2")
        if raymap_stride < 1 or raymap_stride >= raymap_token_grid:
            raise ValueError("raymap_stride must be in [1, raymap_token_grid)")

        self.cond_dim = int(cond_dim)
        self.pos_hidden_dim = int(pos_hidden_dim)
        self.num_groups = int(num_groups)
        self.layers_per_group = int(layers_per_group)
        self.time_dim = int(time_dim)
        self.max_target_points = int(max_target_points)
        self.max_views = int(max_views)
        self.content_rope_t_mid = float(content_rope_t_mid)
        self.content_rope_t_scale = float(content_rope_t_scale)
        self.camera_image_width = int(camera_image_width)
        self.camera_image_height = int(camera_image_height)
        self.canonical_position_scale = float(canonical_position_scale)
        self.route_gradient_scale = float(route_gradient_scale)
        self.raymap_token_grid = int(raymap_token_grid)
        self.raymap_stride = int(raymap_stride)
        self.raymap_uv_clamp = float(raymap_uv_clamp)
        self.register_buffer("canonical_xyz_mean", torch.tensor(canonical_xyz_mean, dtype=torch.float32), persistent=True)
        self.register_buffer("canonical_xyz_std", torch.tensor(canonical_xyz_std, dtype=torch.float32), persistent=True)
        cells = torch.arange(0, self.raymap_token_grid, self.raymap_stride, dtype=torch.float32)
        self.register_buffer("raymap_cells", cells, persistent=False)

        self.xyz_pe = FourierXYZ(xyz_fourier_bands)
        self.point_in = nn.Sequential(
            nn.Linear(self.xyz_pe.out_dim + self.cond_dim + POINT_DESC_DIM, pos_hidden_dim),
            nn.SiLU(),
            nn.Linear(pos_hidden_dim, pos_hidden_dim),
        )
        self.ray_in = nn.Sequential(
            nn.Linear(RAY_FEATURE_DIM, raymap_dim),
            nn.SiLU(),
            nn.Linear(raymap_dim, raymap_dim),
        )
        self.query_in = nn.Linear(self.xyz_pe.out_dim, pos_hidden_dim)
        self.query_time_to_pos = nn.Linear(time_dim, pos_hidden_dim)
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim * 4), nn.SiLU(), nn.Linear(time_dim * 4, time_dim))
        self.target_order_embed = nn.Parameter(torch.zeros(1, self.max_target_points, pos_hidden_dim))
        # cond points (first and last, already concatenated) share one memory encoder
        self.desc_bias = nn.Linear(POINT_DESC_DIM, 1)
        nn.init.zeros_(self.desc_bias.weight)
        nn.init.zeros_(self.desc_bias.bias)
        self.view_embed = nn.Embedding(2, pos_hidden_dim)
        nn.init.zeros_(self.view_embed.weight)

        rope_cfg = dict(
            rope_base=content_rope_base,
            rope_fwd=content_rope_fwd,
            rope_bits=content_rope_bits,
            rope_impl=content_rope_impl,
        )
        self.blocks = nn.ModuleList(
            _Stage1PositionBlock(pos_hidden_dim, time_dim, pos_heads, mlp_ratio, rope_cfg, use_flash_varlen)
            for _ in range(num_groups * layers_per_group)
        )
        # ray-map attention runs once per group: the route it projects is constant within a group
        self.ray_norms = nn.ModuleList(AdaLN(pos_hidden_dim, time_dim) for _ in range(num_groups))
        self.ray_cross = nn.ModuleList(
            _RayMapCrossAttention(pos_hidden_dim, raymap_dim, raymap_heads, raymap_rope_base, raymap_rope_fwd)
            for _ in range(num_groups)
        )
        self.group_heads = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(pos_hidden_dim), nn.Linear(pos_hidden_dim, 3))
            for _ in range(num_groups)
        )
        for head in self.group_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self._diagnostics: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ diagnostics

    def pop_diagnostic(self, key: str) -> Optional[torch.Tensor]:
        return self._diagnostics.pop(key, None)

    # ------------------------------------------------------------------ coordinates

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
        return camera0_to_world_standardized(canonical, cam_c2w, xyz_mean, xyz_std, self.canonical_position_scale)

    def _to_metric(self, standardized: torch.Tensor) -> torch.Tensor:
        """Standardized canonical -> camera-0 metric, for real geometry (v12 contract)."""
        b = standardized.shape[0]
        raw = denormalize_canonical(
            standardized,
            self.canonical_xyz_mean[None].expand(b, -1),
            self.canonical_xyz_std[None].expand(b, -1),
        )
        return raw * self.canonical_position_scale

    def _relative_cameras(self, cam_c2w: torch.Tensor) -> torch.Tensor:
        camera = cam_c2w.float()
        reference_w2c = torch.linalg.inv(camera[:, 0])
        return torch.matmul(reference_w2c[:, None], camera)

    def _lattice_intrinsics(self, cam_k: torch.Tensor) -> torch.Tensor:
        """Full-resolution K -> the VAE's G x G token-lattice K."""
        grid = self.raymap_token_grid
        return _scale_intrinsics(
            cam_k.float(),
            src_hw=(self.camera_image_height, self.camera_image_width),
            dst_hw=(grid, grid),
        )

    def _project_to_views(
        self,
        points_metric: torch.Tensor,
        relative: torch.Tensor,
        k_lattice: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project camera-0 metric points into each supplied view's token lattice.

        Returns (u, v, u_norm, v_norm, log_depth, in_front), all [B, V, N] except the
        lattice coords which are also [B, V, N].
        """
        view_from_c0 = torch.linalg.inv(relative.float())
        homogeneous = torch.cat([points_metric.float(), torch.ones_like(points_metric[..., :1])], dim=-1)
        p_view = torch.einsum("bvij,bnj->bvni", view_from_c0[:, :, :3, :], homogeneous)
        z_raw = p_view[..., 2]
        z = z_raw.clamp_min(1e-4)
        fx = k_lattice[:, :, 0, 0][..., None]
        fy = k_lattice[:, :, 1, 1][..., None]
        cx = k_lattice[:, :, 0, 2][..., None]
        cy = k_lattice[:, :, 1, 2][..., None]
        u = fx * p_view[..., 0] / z + cx
        v = fy * p_view[..., 1] / z + cy
        denom = float(max(self.raymap_token_grid - 1, 1))
        u_norm = 2.0 * u / denom - 1.0
        v_norm = 2.0 * v / denom - 1.0
        return u, v, u_norm, v_norm, z.log().clamp(-8.0, 8.0), z_raw > 1e-4

    # ------------------------------------------------------------------ condition A

    def _source_camera_index(
        self,
        cond_view_index: Optional[torch.Tensor],
        cond_frame_index: Optional[torch.Tensor],
        cam_mask: torch.Tensor,
        n_cond: int,
    ) -> torch.Tensor:
        """Per-token source camera. first -> 0; last -> last valid / cond_frame_index[1]."""
        b, n_views = cam_mask.shape
        device = cam_mask.device
        if cond_frame_index is not None:
            last = cond_frame_index.to(device=device, dtype=torch.int64)
            if last.dim() == 1:
                last = last.unsqueeze(0).expand(b, -1)
            last_cam = last[:, 1].clamp(0, n_views - 1)
        else:
            last_cam = cam_mask.to(torch.int64).sum(dim=1).clamp_min(1) - 1
        if cond_view_index is None:
            return torch.zeros((b, n_cond), dtype=torch.int64, device=device)
        view = cond_view_index.to(device=device, dtype=torch.int64)
        if view.dim() == 1:
            view = view.unsqueeze(0).expand(b, -1)
        if view.shape != (b, n_cond):
            raise ValueError(
                f"cond_view_index shape {tuple(view.shape)} does not match cond tokens {(b, n_cond)}"
            )
        first_cam = torch.zeros((b, n_cond), dtype=torch.int64, device=device)
        return torch.where(view == 0, first_cam, last_cam[:, None].expand(b, n_cond))

    def _point_geometry(
        self,
        points_metric_c0: torch.Tensor,
        relative: torch.Tensor,
        k_lattice: torch.Tensor,
        source_cam: torch.Tensor,
    ) -> torch.Tensor:
        """Source-camera depth/uv plus the observation ray as camera-0 Plücker.

        ``points_metric_c0`` stays in camera-0 metric xyz. Each token is then
        transformed into its true source camera to compute ray / depth / uv.
        That observation ray is rotated back to camera-0 and stored as Plücker
        ``(dir, origin x dir)``.
        """
        b, n, _ = points_metric_c0.shape
        n_views = relative.shape[1]
        src = source_cam.clamp(0, n_views - 1)
        gather_rel = src[:, :, None, None].expand(b, n, 4, 4)
        gather_k = src[:, :, None, None].expand(b, n, 3, 3)
        rel_src = torch.gather(relative.float(), 1, gather_rel)
        k_src = torch.gather(k_lattice.float(), 1, gather_k)

        view_from_c0 = torch.linalg.inv(rel_src)
        homogeneous = torch.cat(
            [points_metric_c0.float(), torch.ones_like(points_metric_c0[..., :1])],
            dim=-1,
        )
        p_src = torch.einsum("bnij,bnj->bni", view_from_c0[:, :, :3, :], homogeneous)
        z_raw = p_src[..., 2]
        z = z_raw.clamp_min(1e-4)
        valid = (z_raw > 1e-4).to(torch.float32)
        fx = k_src[:, :, 0, 0]
        fy = k_src[:, :, 1, 1]
        cx = k_src[:, :, 0, 2]
        cy = k_src[:, :, 1, 2]
        u = fx * p_src[..., 0] / z + cx
        v = fy * p_src[..., 1] / z + cy
        denom = float(max(self.raymap_token_grid - 1, 1))
        clamp = self.raymap_uv_clamp
        u_norm = (2.0 * u / denom - 1.0).clamp(-clamp, clamp)
        v_norm = (2.0 * v / denom - 1.0).clamp(-clamp, clamp)

        dir_src = F.normalize(p_src, dim=-1, eps=1e-6)
        dir_c0 = F.normalize(torch.einsum("bnij,bnj->bni", rel_src[:, :, :3, :3], dir_src), dim=-1, eps=1e-6)
        origin_c0 = rel_src[:, :, :3, 3]
        plucker = torch.cat([dir_c0, torch.cross(origin_c0, dir_c0, dim=-1)], dim=-1)
        feat = torch.cat(
            [
                plucker,
                z.log().clamp(-8.0, 8.0).unsqueeze(-1),
                valid.unsqueeze(-1),
                u_norm.unsqueeze(-1),
                v_norm.unsqueeze(-1),
            ],
            dim=-1,
        )
        return torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).to(points_metric_c0.dtype)

    def _build_memory(
        self,
        cond_canonical: torch.Tensor,
        cond_metric: torch.Tensor,
        cond_con: torch.Tensor,
        cond_mask: torch.Tensor,
        relative: torch.Tensor,
        k_lattice: torch.Tensor,
        cond_view_index: Optional[torch.Tensor],
        cond_frame_index: Optional[torch.Tensor],
        cam_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        source_cam = self._source_camera_index(
            cond_view_index, cond_frame_index, cam_mask, cond_metric.shape[1]
        )
        point_desc = self._point_geometry(cond_metric, relative, k_lattice, source_cam)
        point_feat = torch.cat([self.xyz_pe(cond_canonical), cond_con, point_desc.to(cond_con.dtype)], dim=-1)
        memory = self.point_in(point_feat)
        view = cond_view_index
        if view is None:
            view = torch.zeros(cond_mask.shape, dtype=torch.int64, device=cond_mask.device)
        else:
            view = view.to(device=cond_mask.device, dtype=torch.int64)
            if view.dim() == 1:
                view = view.unsqueeze(0).expand(cond_mask.shape[0], -1)
        memory = memory + self.view_embed(view.clamp(0, 1))
        bias = self.desc_bias(point_desc.to(dtype=memory.dtype)).squeeze(-1)
        bias = bias.masked_fill(~cond_mask, -10000.0)
        memory = memory.masked_fill(~cond_mask.unsqueeze(-1), 0.0)
        return memory, cond_canonical, cond_mask, bias

    # ------------------------------------------------------------------ condition B

    def _ray_map(
        self,
        cond_metric: torch.Tensor,
        cond_mask: torch.Tensor,
        relative: torch.Tensor,
        k_lattice: torch.Tensor,
        cam_mask: torch.Tensor,
        drop_cam: bool,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dense ray map over a strided subset of the DA3 token lattice, per future view."""
        b, views = relative.shape[:2]
        device = relative.device
        n_future = max(views - 1, 0)
        cells = self.raymap_cells
        n_cells = int(cells.numel())
        m = n_cells * n_cells
        if n_future == 0 or m == 0:
            empty_tokens = cond_metric.new_zeros((b, 0, m, self.ray_in[-1].out_features))
            empty_uv = cond_metric.new_zeros((b, 0, m, 2))
            empty_mask = torch.zeros((b, 0, m), dtype=torch.bool, device=device)
            return empty_tokens, empty_uv, empty_mask, cond_metric.new_zeros(())

        rel_future = relative[:, 1:].float()
        k_future = k_lattice[:, 1:].float()

        # lattice cell centres, integer convention (matches _projective_pool_grid)
        row, col = torch.meshgrid(cells, cells, indexing="ij")
        u_cell = col.reshape(-1)
        v_cell = row.reshape(-1)
        pix = torch.stack([u_cell, v_cell, torch.ones_like(u_cell)], dim=-1)

        # ray directions in camera-0 metric frame, then Plucker (dir, origin x dir)
        k_inv = torch.linalg.inv(k_future)
        dir_cam = F.normalize(torch.einsum("bvij,mj->bvmi", k_inv, pix), dim=-1, eps=1e-6)
        dir_c0 = F.normalize(torch.einsum("bvij,bvmj->bvmi", rel_future[:, :, :3, :3], dir_cam), dim=-1, eps=1e-6)
        origin_c0 = rel_future[:, :, :3, 3][:, :, None, :].expand_as(dir_c0)
        plucker = torch.cat([dir_c0, torch.cross(origin_c0, dir_c0, dim=-1)], dim=-1)

        # known surface depth along each ray from all cond points (first + last)
        u_pt, v_pt, u_pt_norm, v_pt_norm, log_z_pt, in_front = self._project_to_views(
            cond_metric, rel_future, k_future
        )
        in_bounds = in_front & (u_pt_norm.abs() <= 1.0) & (v_pt_norm.abs() <= 1.0) & cond_mask[:, None, :]
        cell_u = (u_pt / float(self.raymap_stride)).round().clamp(0, n_cells - 1).long()
        cell_v = (v_pt / float(self.raymap_stride)).round().clamp(0, n_cells - 1).long()
        flat = cell_v * n_cells + cell_u
        known = torch.full((b, n_future, m), _BIG_DEPTH, device=device, dtype=torch.float32)
        known.scatter_reduce_(2, flat, log_z_pt.masked_fill(~in_bounds, _BIG_DEPTH), reduce="amin", include_self=True)
        known_mask = known < (_BIG_DEPTH * 0.5)
        known = known.masked_fill(~known_mask, 0.0) / 4.0

        traj_time = torch.arange(n_future, device=device, dtype=torch.float32) / float(max(n_future - 1, 1))
        time_feat = torch.stack([traj_time, torch.sin(math.pi * traj_time)], dim=-1)

        denom = float(max(self.raymap_token_grid - 1, 1))
        uv_cell = torch.stack([2.0 * u_cell / denom - 1.0, 2.0 * v_cell / denom - 1.0], dim=-1)

        feat = torch.cat(
            [
                plucker,
                uv_cell.view(1, 1, m, 2).expand(b, n_future, m, 2),
                time_feat.view(1, n_future, 1, 2).expand(b, n_future, m, 2),
                known.unsqueeze(-1),
                known_mask.to(torch.float32).unsqueeze(-1),
            ],
            dim=-1,
        )
        feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

        available = cam_mask[:, 1:].bool()
        if drop_cam:
            available = torch.zeros_like(available)
        ray_mask = available[:, :, None].expand(b, n_future, m)
        tokens = self.ray_in(feat.to(dtype)).masked_fill(~ray_mask.unsqueeze(-1), 0.0)
        ray_uv = uv_cell.view(1, 1, m, 2).expand(b, n_future, m, 2).to(dtype)
        known_frac = (known_mask & ray_mask).float().sum() / ray_mask.float().sum().clamp_min(1.0)
        return tokens, ray_uv, ray_mask, known_frac.detach()

    def _query_view_coords(
        self,
        route_standardized: torch.Tensor,
        relative_future: torch.Tensor,
        k_lattice_future: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        route_metric = self._to_metric(route_standardized)
        _, _, u_norm, v_norm, log_depth, in_front = self._project_to_views(
            route_metric, relative_future, k_lattice_future
        )
        clamp = self.raymap_uv_clamp
        uv = torch.stack([u_norm.clamp(-clamp, clamp), v_norm.clamp(-clamp, clamp)], dim=-1)
        uv = torch.nan_to_num(uv, nan=0.0, posinf=0.0, neginf=0.0)
        log_depth = torch.nan_to_num(log_depth, nan=0.0, posinf=0.0, neginf=0.0)
        return uv.to(dtype), log_depth.to(dtype), in_front.to(dtype)

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        xt_pos: torch.Tensor,
        t_pos: torch.Tensor,
        target_mask: torch.Tensor,
        cond_pos: torch.Tensor,
        cond_con: torch.Tensor,
        cond_mask: torch.Tensor,
        cam_c2w: torch.Tensor,
        cam_K: torch.Tensor,
        cam_mask: torch.Tensor,
        x0hat_pos: Optional[torch.Tensor] = None,
        drop_cond: bool = False,
        drop_cam: bool = False,
        xyz_mean: Optional[torch.Tensor] = None,
        xyz_std: Optional[torch.Tensor] = None,
        cond_view_index: Optional[torch.Tensor] = None,
        cond_frame_index: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        begin_rope_cache()
        b = xt_pos.shape[0]
        if xyz_mean is None or xyz_std is None:
            raise ValueError("v18_v3_2view stage1 requires xyz_mean and xyz_std")
        if t_pos.shape != (b,):
            raise ValueError(f"t_pos must have shape ({b},)")
        if xt_pos.shape[1] > self.max_target_points:
            raise ValueError(f"target points={xt_pos.shape[1]} exceeds max_target_points={self.max_target_points}")
        if cam_c2w.shape[1] > self.max_views:
            raise ValueError(f"camera views={cam_c2w.shape[1]} exceeds max_views={self.max_views}")

        target_mask = target_mask.bool()
        cond_mask = cond_mask.bool()
        cam_mask = cam_mask.bool()

        if drop_cond:
            cond_pos = cond_pos.new_zeros((b, 1, 3))
            cond_con = cond_con.new_zeros((b, 1, self.cond_dim))
            cond_mask = torch.ones((b, 1), dtype=torch.bool, device=cond_pos.device)
            cond_view_index = torch.zeros((b, 1), dtype=torch.int64, device=cond_pos.device)

        relative = self._relative_cameras(cam_c2w)
        k_lattice = self._lattice_intrinsics(cam_K)

        cond_canonical = self.world_standardized_to_canonical(cond_pos, cam_c2w, xyz_mean, xyz_std)
        cond_metric = self._to_metric(cond_canonical)
        if drop_cond:
            cond_canonical = torch.zeros_like(cond_canonical)
            cond_metric = torch.zeros_like(cond_metric)

        # Condition A: concatenated first+last points. xyz stays camera-0 canonical;
        # descriptors use each token's source camera. drop_cam still only removes
        # the future trajectory (condition B).
        memory, memory_pos, memory_mask, memory_bias = self._build_memory(
            cond_canonical,
            cond_metric,
            cond_con,
            cond_mask,
            relative,
            k_lattice,
            cond_view_index,
            cond_frame_index,
            cam_mask,
        )
        # Condition B: future ray map. Depends only on conditioning, so it is built once.
        ray_tokens, ray_uv, ray_mask, known_depth_frac = self._ray_map(
            cond_metric, cond_mask, relative, k_lattice, cam_mask, drop_cam, xt_pos.dtype
        )
        relative_future = relative[:, 1:]
        k_lattice_future = k_lattice[:, 1:]

        te = self.time_mlp(timestep_embedding(t_pos, self.time_dim))
        q = self.query_in(self.xyz_pe(xt_pos))
        q = q + self.query_time_to_pos(te)[:, None]
        q = q + self.target_order_embed[:, : xt_pos.shape[1]].to(dtype=q.dtype)
        route = xt_pos if x0hat_pos is None else x0hat_pos
        rope_gate = torch.sigmoid((self.content_rope_t_mid - t_pos.float()) * self.content_rope_t_scale)

        aux_x0: list[torch.Tensor] = []
        aux_v: list[torch.Tensor] = []
        ray_norm_total = xt_pos.new_zeros(())
        q_valid_total = xt_pos.new_zeros(())
        block_index = 0
        for group_index, head in enumerate(self.group_heads):
            if ray_tokens.shape[1] > 0:
                q_uv, q_log_depth, q_gate = self._query_view_coords(
                    route, relative_future, k_lattice_future, xt_pos.dtype
                )
                qn = self.ray_norms[group_index](q, te)
                ray_out = self.ray_cross[group_index](
                    qn, ray_tokens, ray_uv, ray_mask, q_uv, q_log_depth, q_gate, target_mask
                )
                q = q + ray_out
                ray_norm_total = ray_norm_total + ray_out.detach().float().pow(2).mean().sqrt()
                q_valid_total = q_valid_total + q_gate.detach().float().mean()
            for _ in range(self.layers_per_group):
                q = self.blocks[block_index](
                    q,
                    route,
                    memory,
                    memory_pos,
                    memory_mask,
                    memory_bias,
                    te,
                    target_mask,
                    rope_gate,
                )
                block_index += 1
            x0 = head(q).masked_fill(~target_mask.unsqueeze(-1), 0.0)
            v = ((xt_pos - x0) / t_pos[:, None, None].clamp_min(1e-5)).masked_fill(~target_mask.unsqueeze(-1), 0.0)
            aux_x0.append(x0)
            aux_v.append(v)
            route = (x0.detach() + self.route_gradient_scale * (x0 - x0.detach())).masked_fill(
                ~target_mask.unsqueeze(-1), 0.0
            )

        groups = float(self.num_groups)
        self._diagnostics = {
            "raymap_out_norm": (ray_norm_total / groups).detach(),
            "raymap_q_valid_frac": (q_valid_total / groups).detach(),
            "raymap_known_depth_frac": known_depth_frac.detach(),
        }
        return {
            "x0_pos": aux_x0[-1],
            "v_pos": aux_v[-1],
            "aux_x0hat_pos": torch.stack(aux_x0, dim=0),
            "aux_v_pos": torch.stack(aux_v, dim=0),
            "x0hat_pos_internal": route,
            "spatial_rope_gate": rope_gate,
            "real_count_mean": cond_mask.float().sum(dim=1).mean(),
            "virt_count_mean": ray_mask.float().sum(dim=(1, 2)).mean() if ray_mask.numel() else cond_mask.new_zeros(()).float(),
            "desc_bias_abs_mean": self.desc_bias.weight.detach().abs().mean(),
        }


TwoStreamPointDiTV18V3TwoView = TwoStagePositionDiTV18V3TwoView
