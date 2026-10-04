from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _pixel_grid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([xs + 0.5, ys + 0.5, torch.ones_like(xs)], dim=-1)


def scale_intrinsics(Ks: torch.Tensor, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> torch.Tensor:
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    out = Ks.clone()
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    out[..., 0, 0] *= sx
    out[..., 0, 2] *= sx
    out[..., 1, 1] *= sy
    out[..., 1, 2] *= sy
    return out


def plucker_rays(camtoworlds: torch.Tensor, Ks: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Args:
        camtoworlds: [B, V, 4, 4]
        Ks: [B, V, 3, 3]
    Returns:
        [B, V, H, W, 6] = [dir_world, origin x dir_world]
    """
    if camtoworlds.dim() != 4 or Ks.dim() != 4:
        raise ValueError("camtoworlds and Ks must be [B,V,...]")
    b, v = camtoworlds.shape[:2]
    device = camtoworlds.device
    dtype = camtoworlds.dtype

    pix = _pixel_grid(height, width, device=device, dtype=dtype)
    pix = pix.view(1, 1, height, width, 3).expand(b, v, height, width, 3)

    K_inv = torch.linalg.inv(Ks)
    rays_cam = torch.einsum("bvij,bvhwj->bvhwi", K_inv, pix)
    rays_cam = F.normalize(rays_cam, dim=-1)

    rot = camtoworlds[..., :3, :3]
    origin = camtoworlds[..., :3, 3]
    rays_world = torch.einsum("bvij,bvhwj->bvhwi", rot, rays_cam)
    rays_world = F.normalize(rays_world, dim=-1)
    origin = origin[:, :, None, None, :].expand_as(rays_world)
    moment = torch.cross(origin, rays_world, dim=-1)
    return torch.cat([rays_world, moment], dim=-1)


def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half, 1))
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb
