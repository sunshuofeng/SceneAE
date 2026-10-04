"""Shared primitives for the v16_v3 stage-1 position DiT.

Copy of ``dit_v16_v2_primitives`` (itself ``dit_v16_primitives`` plus the A+B+C
training-speed optimizations profiled 2026-08-25, see
plans/dit_version_evolution.md). v16_v3 keeps these primitives unchanged and only
shrinks the backbone (768d x 8 layers instead of 12) in ``dit_v16_v3_stage1``;

  A. flash-attn fast path. In v1, ``HeadRMSNorm``'s fp32 weight silently promotes

  A. flash-attn fast path. In v1, ``HeadRMSNorm``'s fp32 weight silently promotes
     q/k/v to float32, so ``_flash_attn_varlen`` rejected every call and all
     attention fell back to the SDPA mem-efficient (cutlass) kernel whose backward
     is ~2.8x its forward. v2 casts q/k/v to bf16 before flash whenever autocast is
     active (numerically identical: SDPA autocast was already casting the same
     fp32 tensors to bf16 at the kernel boundary), and calls ``flash_attn_func``
     directly without varlen packing when both masks are entirely valid - which is
     always the case for this dataset (fixed 3000/3000 points per sample).

  B. RoPE cache. continuous3d position normalization and cos/sin depend only on
     the position tensors, which are identical across all 12 layers, yet v1
     recomputed them per attention call (~38x per forward). v2 caches both per
     forward pass. Cache entries are keyed by tensor identity and verified with
     weakrefs, so a missed ``begin_rope_cache()`` can waste memory but can never
     return stale values.

  C. Host-sync removal. The SDPA fallback no longer calls ``bool(empty_k.any())``
     per attention (an all-invalid row under the additive -10000 mask yields a
     finite zero-vector output instead of NaN, which is masked downstream anyway);
     ``_flash_attn_varlen`` passes padded lengths as the max_seqlen upper bounds
     instead of ``.item()``; mask fullness is queried once per unique mask set per
     forward through the same cache.

Also hosts the coordinate helpers and ``_TrajectoryEncoder`` (copied from
``dit_v16_stage2``) so v16_v3 does not import any v1 module.
"""

from __future__ import annotations

import math
import sys
import weakref
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POINTROPE_BUILD_ROOT = PROJECT_ROOT.parent / "Pointcept/libs/pointrope/build/lib.linux-x86_64-cpython-310"
if POINTROPE_BUILD_ROOT.is_dir() and str(POINTROPE_BUILD_ROOT) not in sys.path:
    sys.path.append(str(POINTROPE_BUILD_ROOT))
try:
    import pointrope as _pointrope_ext
except Exception as exc:  # pragma: no cover - depends on local CUDA extension build.
    _pointrope_ext = None
    _POINTROPE_IMPORT_ERROR = exc
else:
    _POINTROPE_IMPORT_ERROR = None

try:
    import flash_attn
except Exception as exc:  # pragma: no cover - optional CUDA dependency.
    flash_attn = None
    _FLASH_ATTN_IMPORT_ERROR = exc
else:
    _FLASH_ATTN_IMPORT_ERROR = None


# --------------------------------------------------------------------------- per-forward cache

_ROPE_CACHE: dict = {}


def begin_rope_cache() -> None:
    """Clear the per-forward RoPE / mask caches. Call once at the top of every model forward."""
    _ROPE_CACHE.clear()


def _cache_lookup(key, tensors):
    hit = _ROPE_CACHE.get(key)
    if hit is None:
        return None
    refs, value = hit
    if len(refs) != len(tensors):
        return None
    for ref, tensor in zip(refs, tensors):
        if ref() is not tensor:
            return None
    return value


def _cache_store(key, tensors, value) -> None:
    _ROPE_CACHE[key] = (tuple(weakref.ref(t) for t in tensors), value)


def masks_full_cached(*masks: torch.Tensor) -> bool:
    """True when every mask is entirely valid; syncs at most once per unique mask set per forward."""
    key = ("masks_full",) + tuple(id(m) for m in masks)
    hit = _cache_lookup(key, masks)
    if hit is None:
        hit = all(bool(m.all()) for m in masks)
        _cache_store(key, masks, hit)
    return hit


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class AdaLN(nn.Module):
    def __init__(self, dim: int, time_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.to_scale_shift = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, dim * 2))

    def forward(self, x: torch.Tensor, te: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_scale_shift(te).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale[:, None]) + shift[:, None]


class FourierXYZ(nn.Module):
    def __init__(self, num_bands: int = 16) -> None:
        super().__init__()
        self.num_bands = int(num_bands)
        freq = 2.0 ** torch.arange(self.num_bands, dtype=torch.float32)
        self.register_buffer("freq", freq, persistent=False)
        self.out_dim = 3 + 3 * 2 * self.num_bands

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        x = xyz.unsqueeze(-1) * self.freq
        return torch.cat([xyz, x.sin().flatten(-2), x.cos().flatten(-2)], dim=-1)


def _quantize_rope_xyz(xyz: torch.Tensor, bits: int = 10) -> torch.Tensor:
    xyz_f = xyz.detach().float()
    xyz_min = xyz_f.amin(dim=1, keepdim=True)
    xyz_max = xyz_f.amax(dim=1, keepdim=True)
    xyz_n = (xyz_f - xyz_min) / (xyz_max - xyz_min).clamp_min(1e-6)
    max_coord = (1 << int(bits)) - 1
    return (xyz_n.clamp(0.0, 1.0) * max_coord).round().to(torch.int64).contiguous()


class PointRopeFunction(Function):
    @staticmethod
    def forward(ctx, tokens: torch.Tensor, positions: torch.Tensor, base: float, fwd: float) -> torch.Tensor:
        if _pointrope_ext is None:
            raise ImportError(f"pointrope extension is unavailable: {_POINTROPE_IMPORT_ERROR}")
        out = tokens.contiguous().clone()
        pos = positions.contiguous().to(torch.int64)
        _pointrope_ext.pointrope(out, pos, float(base), float(fwd))
        ctx.save_for_backward(pos)
        ctx.base = float(base)
        ctx.fwd = float(fwd)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        (pos,) = ctx.saved_tensors
        grad = grad_output.contiguous().clone()
        _pointrope_ext.pointrope(grad, pos, ctx.base, -ctx.fwd)
        return grad, None, None, None


def _apply_point_rope(
    tokens: torch.Tensor,
    positions: torch.Tensor,
    *,
    enabled: bool,
    base: float,
    fwd: float,
) -> torch.Tensor:
    if not enabled:
        return tokens
    head_dim = tokens.shape[-1]
    rope_dim = (head_dim // 6) * 6
    if rope_dim == 0:
        return tokens
    rope_part = PointRopeFunction.apply(tokens[..., :rope_dim], positions, float(base), float(fwd))
    if rope_dim == head_dim:
        return rope_part
    return torch.cat([rope_part, tokens[..., rope_dim:]], dim=-1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def _apply_1d_continuous_rope(tokens: torch.Tensor, coord: torch.Tensor, base: float, fwd: float) -> torch.Tensor:
    dim = tokens.shape[-1]
    if dim == 0:
        return tokens
    if dim % 2 != 0:
        raise ValueError(f"continuous RoPE axis dim must be even, got {dim}")
    half = dim // 2
    freqs = torch.exp(-math.log(float(base)) * torch.arange(half, device=tokens.device, dtype=torch.float32) / max(half - 1, 1))
    angles = coord.float().unsqueeze(-1) * freqs * float(fwd)
    cos = angles.cos().to(dtype=tokens.dtype).unsqueeze(2).repeat_interleave(2, dim=-1)
    sin = angles.sin().to(dtype=tokens.dtype).unsqueeze(2).repeat_interleave(2, dim=-1)
    return tokens * cos + _rotate_half(tokens) * sin


def _continuous_3d_cos_sin(
    xyz: torch.Tensor,
    rope_dim: int,
    base: float,
    fwd: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin for all three axes in one shot; freqs are shared across axes.

    xyz: [..., 3] (already normalized). Returns ([..., rope_dim], [..., rope_dim]).
    """
    axis_dim = rope_dim // 3
    half = axis_dim // 2
    freqs = torch.exp(
        -math.log(float(base)) * torch.arange(half, device=xyz.device, dtype=torch.float32) / max(half - 1, 1)
    )
    angles = xyz.float().unsqueeze(-1) * freqs * float(fwd)  # [..., 3, half]
    cos = angles.cos().to(dtype).repeat_interleave(2, dim=-1)  # [..., 3, axis_dim]
    sin = angles.sin().to(dtype).repeat_interleave(2, dim=-1)
    return cos.flatten(-2), sin.flatten(-2)


def _cached_continuous_3d_cos_sin(
    xyz: torch.Tensor,
    rope_dim: int,
    base: float,
    fwd: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = ("rope3d_cs", id(xyz), int(rope_dim), float(base), float(fwd), dtype)
    hit = _cache_lookup(key, (xyz,))
    if hit is None:
        hit = _continuous_3d_cos_sin(xyz, rope_dim, base, fwd, dtype)
        _cache_store(key, (xyz,), hit)
    return hit


def _apply_continuous_3d_rope(
    tokens: torch.Tensor,
    xyz: torch.Tensor,
    base: float,
    fwd: float,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    head_dim = tokens.shape[-1]
    rope_dim = (head_dim // 6) * 6
    if rope_dim == 0:
        return tokens
    rope_part = tokens[..., :rope_dim]
    pass_part = tokens[..., rope_dim:]
    xyz_n = xyz.float()
    if normalize:
        xyz_center = xyz_n.mean(dim=1, keepdim=True)
        xyz_scale = (xyz_n - xyz_center).pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        xyz_n = (xyz_n - xyz_center) / xyz_scale
    cos, sin = _cached_continuous_3d_cos_sin(xyz_n, rope_dim, base, fwd, tokens.dtype)
    # cos/sin: [B, L, rope_dim] -> broadcast over heads; per-axis rotate_half pairing
    # is preserved because axis_dim is even and axes are concatenated in x/y/z order.
    rope_out = rope_part * cos.unsqueeze(-2) + _rotate_half(rope_part) * sin.unsqueeze(-2)
    if pass_part.shape[-1] == 0:
        return rope_out
    return torch.cat([rope_out, pass_part], dim=-1)


class HeadRMSNorm(nn.Module):
    def __init__(self, head_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


def _pack_by_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return x[mask.bool()].contiguous()


def _unpack_by_mask(x: torch.Tensor, mask: torch.Tensor, length: int) -> torch.Tensor:
    b = mask.shape[0]
    out = x.new_zeros((b, length, *x.shape[1:]))
    out[mask.bool()] = x
    return out


def _flash_attn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_mask: torch.Tensor,
    k_mask: torch.Tensor,
) -> Optional[torch.Tensor]:
    if flash_attn is None or not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16):
        return None
    q_lens = q_mask.sum(dim=1, dtype=torch.int32)
    k_lens = k_mask.sum(dim=1, dtype=torch.int32)
    if bool((q_lens == 0).any()) or bool((k_lens == 0).any()):
        return None

    q_pack = _pack_by_mask(q, q_mask)
    k_pack = _pack_by_mask(k, k_mask)
    v_pack = _pack_by_mask(v, k_mask)
    cu_q = F.pad(q_lens.cumsum(dim=0), (1, 0)).to(device=q.device, dtype=torch.int32)
    cu_k = F.pad(k_lens.cumsum(dim=0), (1, 0)).to(device=q.device, dtype=torch.int32)
    out_pack = flash_attn.flash_attn_varlen_func(
        q_pack,
        k_pack,
        v_pack,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        # padded lengths are valid upper bounds; avoids two host syncs per call
        max_seqlen_q=q.shape[1],
        max_seqlen_k=k.shape[1],
        dropout_p=0.0,
        causal=False,
    )
    return _unpack_by_mask(out_pack, q_mask, q.shape[1])


def _autocast_castable_to_half(q: torch.Tensor) -> bool:
    """True when casting q/k/v to bf16 before flash matches what SDPA autocast would do."""
    if q.dtype in (torch.float16, torch.bfloat16):
        return True
    if q.dtype != torch.float32:
        return False
    try:
        return bool(torch.is_autocast_enabled(q.device.type))
    except TypeError:  # torch < 2.4 signature
        return bool(torch.is_autocast_enabled())


class ExplicitAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        use_rope: bool = True,
        qk_norm: bool = True,
        use_flash_varlen: bool = True,
        rope_base: float = 10000.0,
        rope_fwd: float = 1.0,
        rope_bits: int = 10,
        rope_impl: str = "pointrope",
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.use_rope = bool(use_rope)
        self.use_flash_varlen = bool(use_flash_varlen)
        self.rope_base = float(rope_base)
        self.rope_fwd = float(rope_fwd)
        self.rope_bits = int(rope_bits)
        self.rope_impl = str(rope_impl)
        if self.rope_impl not in ("pointrope", "continuous3d"):
            raise ValueError(f"Unsupported rope_impl={self.rope_impl!r}")
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.q_norm = HeadRMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = HeadRMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.out_proj = nn.Linear(dim, dim)

    def _apply_rope_with_gate(
        self,
        tokens: torch.Tensor,
        xyz: torch.Tensor,
        rope_gate,
        *,
        positions_are_normalized: bool = False,
    ) -> torch.Tensor:
        if self.rope_impl == "pointrope":
            pos = _quantize_rope_xyz(xyz, bits=self.rope_bits)
            rope_tokens = _apply_point_rope(tokens, pos, enabled=self.use_rope, base=self.rope_base, fwd=self.rope_fwd)
        else:
            rope_tokens = _apply_continuous_3d_rope(
                tokens,
                xyz,
                base=self.rope_base,
                fwd=self.rope_fwd,
                normalize=not positions_are_normalized,
            )
        if rope_gate is None:
            return rope_tokens
        return tokens + rope_gate[:, None, None, None].to(tokens.dtype) * (rope_tokens - tokens)

    def _normalized_positions(
        self,
        q_pos: torch.Tensor,
        k_pos: torch.Tensor,
        q_mask: torch.Tensor,
        k_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Masked continuous3d position normalization, cached per (q_pos, k_pos, masks) per forward."""
        key = ("rope3d_norm", id(q_pos), id(k_pos), id(q_mask), id(k_mask))
        hit = _cache_lookup(key, (q_pos, k_pos, q_mask, k_mask))
        if hit is not None:
            return hit
        q_weight = q_mask.unsqueeze(-1).to(dtype=torch.float32)
        k_weight = k_mask.unsqueeze(-1).to(dtype=torch.float32)
        count = (q_weight.sum(dim=1, keepdim=True) + k_weight.sum(dim=1, keepdim=True)).clamp_min(1.0)
        center = (
            (q_pos.float() * q_weight).sum(dim=1, keepdim=True)
            + (k_pos.float() * k_weight).sum(dim=1, keepdim=True)
        ) / count
        variance = (
            ((q_pos.float() - center).pow(2) * q_weight).sum(dim=(1, 2), keepdim=True)
            + ((k_pos.float() - center).pow(2) * k_weight).sum(dim=(1, 2), keepdim=True)
        ) / (count * q_pos.shape[-1])
        scale = variance.clamp_min(1e-12).sqrt()
        hit = ((q_pos.float() - center) / scale, (k_pos.float() - center) / scale)
        _cache_store(key, (q_pos, k_pos, q_mask, k_mask), hit)
        return hit

    def forward(
        self,
        q_x: torch.Tensor,
        k_x: torch.Tensor,
        v_x: torch.Tensor,
        *,
        q_mask: torch.Tensor,
        k_mask: torch.Tensor,
        q_pos: Optional[torch.Tensor] = None,
        k_pos: Optional[torch.Tensor] = None,
        rope_gate: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, lq, _ = q_x.shape
        lk = k_x.shape[1]
        if lk == 0:
            return q_x.new_zeros(q_x.shape)
        q_mask = q_mask.bool()
        k_mask = k_mask.bool()
        full_masks = masks_full_cached(q_mask, k_mask)
        q = self.q_proj(q_x).view(b, lq, self.num_heads, self.head_dim)
        k = self.k_proj(k_x).view(b, lk, self.num_heads, self.head_dim)
        v = self.v_proj(v_x).view(b, lk, self.num_heads, self.head_dim)

        valid_q = valid_k = None
        if not full_masks:
            valid_q = q_mask[:, :, None, None].to(q.dtype)
            valid_k = k_mask[:, :, None, None].to(k.dtype)
        q = self.q_norm(q) if valid_q is None else self.q_norm(q) * valid_q
        k = self.k_norm(k) if valid_k is None else self.k_norm(k) * valid_k
        if valid_k is not None:
            v = v * valid_k
        if self.use_rope:
            if q_pos is None or k_pos is None:
                raise ValueError("q_pos and k_pos are required when RoPE is enabled")
            positions_are_normalized = False
            if self.rope_impl == "continuous3d":
                q_pos, k_pos = self._normalized_positions(q_pos, k_pos, q_mask, k_mask)
                positions_are_normalized = True
            q = self._apply_rope_with_gate(q, q_pos, rope_gate, positions_are_normalized=positions_are_normalized)
            k = self._apply_rope_with_gate(k, k_pos, rope_gate, positions_are_normalized=positions_are_normalized)
            if valid_q is not None:
                q = q * valid_q
                k = k * valid_k

        out = None
        if self.use_flash_varlen and attn_bias is None and flash_attn is not None and q.is_cuda:
            if _autocast_castable_to_half(q):
                qh, kh, vh = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
                if full_masks:
                    out = flash_attn.flash_attn_func(qh, kh, vh, dropout_p=0.0, causal=False)
                else:
                    out = _flash_attn_varlen(qh, kh, vh, q_mask, k_mask)
        if out is None:
            # Additive mask: an all-invalid row softmaxes over equally -10000-shifted logits
            # and, since padding values are zeroed above, yields a finite zero vector, so no
            # per-call empty-row host sync is needed.
            if attn_bias is not None:
                attn_mask = attn_bias.to(dtype=q.dtype).masked_fill(~k_mask[:, None, None, :], -10000.0)
            else:
                attn_mask = q.new_zeros((b, 1, 1, lk)).masked_fill(~k_mask[:, None, None, :], -10000.0)
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=attn_mask,
                dropout_p=0.0,
            ).transpose(1, 2)

        out = out.reshape(b, lq, self.dim)
        out = self.out_proj(out)
        if full_masks:
            return out
        return out.masked_fill(~q_mask.unsqueeze(-1), 0.0)


# --------------------------------------------------------------------------- coordinates (from dit_v16_stage2)


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
    """Restore metric-scaled camera-0 coordinates from the v16 stage2 Flow space."""
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


