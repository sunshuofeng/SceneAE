from __future__ import annotations

import importlib.machinery
import sys
import types
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norms import RMSNorm, SwiGLU
from .plucker_camera import plucker_rays, scale_intrinsics
# ProjectiveLocalFeaturePool from 2B is superseded by the deduplicated helpers below
# (_projective_pool_grid / _projective_pool_sample), which compute the projection grid
# once and reuse it across DA3 levels.

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DA3_PY_ROOT = PROJECT_ROOT / "third_party" / "Depth-Anything-3" / "da3"
if str(DA3_PY_ROOT) not in sys.path:
    sys.path.append(str(DA3_PY_ROOT))
LOCAL_TORCHVISION_VGG16 = PROJECT_ROOT / "weights" / "vgg16-397923af.pth"
# POINTOPS_BUILD_ROOT = PROJECT_ROOT.parent / "Pointcept/libs/pointops/build/lib.linux-x86_64-3.10"
# if POINTOPS_BUILD_ROOT.is_dir() and str(POINTOPS_BUILD_ROOT) not in sys.path:
#     sys.path.append(str(POINTOPS_BUILD_ROOT))


def prefer_local_torchvision_vgg16() -> None:
    if not LOCAL_TORCHVISION_VGG16.is_file():
        return
    try:
        import torchvision.models._api as tv_api
    except Exception:
        return
    current_loader = tv_api.load_state_dict_from_url
    if getattr(current_loader, "_e3d_local_vgg16", False):
        return

    def load_state_dict_from_url(url, *args, **kwargs):
        if str(url).endswith("/vgg16-397923af.pth"):
            return torch.load(LOCAL_TORCHVISION_VGG16, map_location=kwargs.get("map_location", "cpu"))
        return current_loader(url, *args, **kwargs)

    load_state_dict_from_url._e3d_local_vgg16 = True
    tv_api.load_state_dict_from_url = load_state_dict_from_url


def install_optional_dependency_stubs() -> None:
    moviepy_stub = types.ModuleType("moviepy")
    moviepy_editor_stub = types.ModuleType("moviepy.editor")
    moviepy_stub.__spec__ = importlib.machinery.ModuleSpec("moviepy", loader=None, is_package=True)
    moviepy_editor_stub.__spec__ = importlib.machinery.ModuleSpec("moviepy.editor", loader=None)
    moviepy_stub.editor = moviepy_editor_stub
    sys.modules.setdefault("moviepy", moviepy_stub)
    sys.modules.setdefault("moviepy.editor", moviepy_editor_stub)


install_optional_dependency_stubs()


from depth_anything_3.api import DepthAnything3
import torch
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.autograd.function import Function, once_differentiable
from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])

try:
    from flash_attn import flash_attn_func
except Exception as exc:  # pragma: no cover - depends on local CUDA extension build.
    flash_attn_func = None
    _FLASH_ATTN_IMPORT_ERROR = exc
else:
    _FLASH_ATTN_IMPORT_ERROR = None


def _flash_attn_maybe_cast(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Cast q/k/v to fp16/bf16.  ``flash_attn_func`` is a custom op, so autocast
    does not convert its inputs; LayerNorm / PointRoPE often leave them fp32."""
    src_dtype = q.dtype
    if src_dtype in (torch.float16, torch.bfloat16):
        return q, k, v, src_dtype
    target = torch.bfloat16
    if torch.is_autocast_enabled():
        try:
            amp = torch.get_autocast_dtype("cuda")
        except (TypeError, AttributeError):
            amp = torch.get_autocast_gpu_dtype()
        if amp in (torch.float16, torch.bfloat16):
            target = amp
    return q.to(target), k.to(target), v.to(target), src_dtype

POINTROPE_BUILD_ROOT = PROJECT_ROOT / "third_party" / "pointrope" / "build" / "lib.linux-x86_64-cpython-310"
if POINTROPE_BUILD_ROOT.is_dir() and str(POINTROPE_BUILD_ROOT) not in sys.path:
    sys.path.append(str(POINTROPE_BUILD_ROOT))
try:
    import pointrope as _pointrope_ext
except Exception as exc:  # pragma: no cover - depends on local CUDA extension build.
    _pointrope_ext = None
    _POINTROPE_IMPORT_ERROR = exc
else:
    _POINTROPE_IMPORT_ERROR = None


def _quantize_rope_xyz(sp_xyz: torch.Tensor, bits: int = 10) -> torch.Tensor:
    xyz = sp_xyz.detach().float()
    xyz_min = xyz.amin(dim=1, keepdim=True)
    xyz_max = xyz.amax(dim=1, keepdim=True)
    xyz_n = (xyz - xyz_min) / (xyz_max - xyz_min).clamp_min(1e-6)
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


class MultiScaleDeformableAttnFunction_fp16(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        """GPU version of multi-scale deformable attention.

        Args:
            value (Tensor): The value has shape
                (bs, num_keys, mum_heads, embed_dims//num_heads)
            value_spatial_shapes (Tensor): Spatial shape of
                each feature map, has shape (num_levels, 2),
                last dimension 2 represent (h, w)
            sampling_locations (Tensor): The location of sampling points,
                has shape
                (bs ,num_queries, num_heads, num_levels, num_points, 2),
                the last dimension 2 represent (x, y).
            attention_weights (Tensor): The weight of sampling points used
                when calculate the attention, has shape
                (bs ,num_queries, num_heads, num_levels, num_points),
            im2col_step (Tensor): The step used in image to column.

        Returns:
            Tensor: has shape (bs, num_queries, embed_dims)
        """
        ctx.im2col_step = im2col_step
        output = ext_module.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step=ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations,
                              attention_weights)
        return output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        """GPU version of backward function.

        Args:
            grad_output (Tensor): Gradient
                of output tensor of forward.

        Returns:
             Tuple[Tensor]: Gradient
                of input tensors in forward.
        """
        value, value_spatial_shapes, value_level_start_index, \
            sampling_locations, attention_weights = ctx.saved_tensors
        grad_value = torch.zeros_like(value)
        grad_sampling_loc = torch.zeros_like(sampling_locations)
        grad_attn_weight = torch.zeros_like(attention_weights)

        ext_module.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output.contiguous(),
            grad_value,
            grad_sampling_loc,
            grad_attn_weight,
            im2col_step=ctx.im2col_step)

        return grad_value, None, None, \
            grad_sampling_loc, grad_attn_weight, None


class MultiScaleDeformableAttnFunction_fp32(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        """GPU version of multi-scale deformable attention.

        Args:
            value (Tensor): The value has shape
                (bs, num_keys, mum_heads, embed_dims//num_heads)
            value_spatial_shapes (Tensor): Spatial shape of
                each feature map, has shape (num_levels, 2),
                last dimension 2 represent (h, w)
            sampling_locations (Tensor): The location of sampling points,
                has shape
                (bs ,num_queries, num_heads, num_levels, num_points, 2),
                the last dimension 2 represent (x, y).
            attention_weights (Tensor): The weight of sampling points used
                when calculate the attention, has shape
                (bs ,num_queries, num_heads, num_levels, num_points),
            im2col_step (Tensor): The step used in image to column.

        Returns:
            Tensor: has shape (bs, num_queries, embed_dims)
        """

        ctx.im2col_step = im2col_step
        output = ext_module.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step=ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations,
                              attention_weights)
        return output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        """GPU version of backward function.

        Args:
            grad_output (Tensor): Gradient
                of output tensor of forward.

        Returns:
             Tuple[Tensor]: Gradient
                of input tensors in forward.
        """
        value, value_spatial_shapes, value_level_start_index, \
            sampling_locations, attention_weights = ctx.saved_tensors
        grad_value = torch.zeros_like(value)
        grad_sampling_loc = torch.zeros_like(sampling_locations)
        grad_attn_weight = torch.zeros_like(attention_weights)

        ext_module.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output.contiguous(),
            grad_value,
            grad_sampling_loc,
            grad_attn_weight,
            im2col_step=ctx.im2col_step)

        return grad_value, None, None, \
            grad_sampling_loc, grad_attn_weight, None



POINTOPS_BUILD_ROOT = PROJECT_ROOT.parent / "Pointcept/libs/pointops/build/lib.linux-x86_64-3.10"
if POINTOPS_BUILD_ROOT.is_dir() and str(POINTOPS_BUILD_ROOT) not in sys.path:
    sys.path.append(str(POINTOPS_BUILD_ROOT))
try:
    from pointops import farthest_point_sampling as pointops_farthest_point_sampling
except Exception:
    pointops_farthest_point_sampling = None



def _charbonnier(diff: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(diff * diff + eps * eps).mean()


def _normalized_mse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_n = F.normalize(pred, dim=-1, eps=eps)
    target_n = F.normalize(target, dim=-1, eps=eps)
    return F.mse_loss(pred_n, target_n)


def _expand_to_views(x: torch.Tensor, v: int) -> torch.Tensor:
    if x.dim() in (3, 4):
        return x.unsqueeze(1).expand(x.shape[0], v, *x.shape[1:])
    return x


def _debug_check_finite(name: str, value: object) -> None:
    if value is None or not torch.is_tensor(value):
        return
    if value.numel() == 0:
        return
    finite = torch.isfinite(value)
    if bool(finite.all()):
        return
    value_detached = value.detach()
    finite_values = value_detached[finite]
    if finite_values.numel() > 0:
        min_val = float(finite_values.float().min().cpu())
        max_val = float(finite_values.float().max().cpu())
        mean_val = float(finite_values.float().mean().cpu())
    else:
        min_val = float("nan")
        max_val = float("nan")
        mean_val = float("nan")
    nan_ratio = float(torch.isnan(value_detached).float().mean().cpu())
    posinf_ratio = float(torch.isposinf(value_detached).float().mean().cpu())
    neginf_ratio = float(torch.isneginf(value_detached).float().mean().cpu())
    raise FloatingPointError(
        f"[NaNCheck] {name}: shape={tuple(value.shape)} dtype={value.dtype} "
        f"nan_ratio={nan_ratio:.6g} posinf_ratio={posinf_ratio:.6g} "
        f"neginf_ratio={neginf_ratio:.6g} finite_min={min_val:.6g} "
        f"finite_max={max_val:.6g} finite_mean={mean_val:.6g}"
    )


def _projective_pool_grid(
    sp_xyz: torch.Tensor,
    Ks: torch.Tensor,
    camtoworlds: torch.Tensor,
    image_hw: tuple[int, int],
    feat_hw: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the per-view grid_sample coordinates and validity mask used by
    ProjectiveLocalFeaturePool, once, so callers can sample multiple feature
    levels of the same spatial resolution without redoing the projection math.

    Args:
        sp_xyz: [B, N, 3] world-frame superpoint coords
        Ks: [B, V, 3, 3] at full image resolution
        camtoworlds: [B, V, 4, 4]
        image_hw: (H, W) full image resolution
        feat_hw: (Hf, Wf) feature map resolution (same across all levels reusing this grid)

    Returns:
        grid: [B*V, N, 1, 2] grid_sample coords in [-1, 1]
        valid: [B, V, N] bool (in-frustum + in-bounds + z>0)
        vis_ratio: [B, N, 1] mean validity across views
    """
    b = sp_xyz.shape[0]
    n = sp_xyz.shape[1]
    v = camtoworlds.shape[1]
    src_h, src_w = image_hw
    hf, wf = feat_hw
    Ks_f = scale_intrinsics(Ks, src_hw=(src_h, src_w), dst_hw=(hf, wf))

    sp_xyz_safe = torch.nan_to_num(sp_xyz, nan=0.0, posinf=0.0, neginf=0.0)
    ones = torch.ones((b, n, 1), device=sp_xyz.device, dtype=sp_xyz.dtype)
    sp_h = torch.cat([sp_xyz_safe, ones], dim=-1)
    sp_h_bv = sp_h[:, None].expand(b, v, n, 4).reshape(b * v, n, 4)
    w2c = torch.linalg.inv(camtoworlds).reshape(b * v, 4, 4).to(sp_xyz.dtype)
    cam_pts = torch.matmul(sp_h_bv, w2c.transpose(1, 2))
    cam_pts = torch.nan_to_num(cam_pts, nan=0.0, posinf=0.0, neginf=0.0)
    z = cam_pts[..., 2]
    z_safe = z.clamp_min(1e-4)
    x = cam_pts[..., 0] / z_safe
    y = cam_pts[..., 1] / z_safe

    Kf = Ks_f.reshape(b * v, 3, 3).to(sp_xyz.dtype)
    u = Kf[:, None, 0, 0] * x + Kf[:, None, 0, 2]
    vv = Kf[:, None, 1, 1] * y + Kf[:, None, 1, 2]
    grid_x = 2.0 * (u / max(wf - 1, 1)) - 1.0
    grid_y = 2.0 * (vv / max(hf - 1, 1)) - 1.0
    finite = torch.isfinite(grid_x) & torch.isfinite(grid_y) & torch.isfinite(z)
    grid_x = torch.nan_to_num(grid_x, nan=2.0, posinf=2.0, neginf=-2.0)
    grid_y = torch.nan_to_num(grid_y, nan=2.0, posinf=2.0, neginf=-2.0)
    grid = torch.stack([grid_x, grid_y], dim=-1).reshape(b * v, n, 1, 2)

    valid = ((z > 1e-4) & finite).reshape(b, v, n)
    valid = valid & (grid_x.reshape(b, v, n).abs() <= 1.0) & (grid_y.reshape(b, v, n).abs() <= 1.0)
    vis_ratio = valid.float().mean(dim=1).unsqueeze(-1)
    return grid, valid, vis_ratio


def _projective_pool_sample(
    feat_map: torch.Tensor,
    grid: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Sample a single feature level using a precomputed grid + validity mask.

    Args:
        feat_map: [B, V, Hf, Wf, C]
        grid:     [B*V, N, 1, 2] (from _projective_pool_grid)
        valid:    [B, V, N] bool
    Returns:
        pooled:   [B, N, C] (mean over valid views; zero where none valid)
    """
    b, v, hf, wf, c = feat_map.shape
    feat_chw = feat_map.permute(0, 1, 4, 2, 3).reshape(b * v, c, hf, wf).to(grid.dtype)
    sampled = F.grid_sample(feat_chw, grid, mode="bilinear", align_corners=True)
    n = grid.shape[1]
    sampled = sampled.squeeze(-1).permute(0, 2, 1).reshape(b, v, n, c)
    valid_f = valid.float().unsqueeze(-1)
    pooled = (sampled * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp_min(1.0)
    return pooled


@torch.no_grad()
def _visible_query_mask_from_projection(
    sp_xyz: torch.Tensor,
    Ks: torch.Tensor,
    camtoworlds: torch.Tensor,
    image_hw: tuple[int, int],
    valid_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-query visibility under the current encoder views.

    Visibility is intentionally tied to the sampled encoder views, not the full
    chunk, so geometry supervision only uses queries supported by current image
    evidence.
    """
    h, w = image_hw
    grid, in_frustum, _ = _projective_pool_grid(sp_xyz, Ks, camtoworlds, image_hw, (h, w))
    b, v, n = in_frustum.shape
    visible_view = in_frustum
    if valid_mask is not None:
        mask = valid_mask.bool()
        if mask.dim() == 3:
            mask = mask.unsqueeze(1).expand(b, v, h, w)
        if mask.shape[:2] != (b, v):
            raise ValueError(f"valid_mask shape {tuple(valid_mask.shape)} incompatible with B={b}, V={v}")
        mask_f = mask.float().reshape(b * v, 1, h, w)
        sampled = F.grid_sample(mask_f, grid, mode="nearest", padding_mode="zeros", align_corners=True)
        sampled = sampled.squeeze(1).squeeze(-1).reshape(b, v, n) > 0.5
        visible_view = visible_view & sampled
    visible_query = visible_view.any(dim=1)
    visible_ratio = visible_view.float().mean(dim=1).unsqueeze(-1)
    return visible_query, visible_ratio


@torch.no_grad()
def _select_query_topk_views(
    sp_xyz: torch.Tensor,
    camtoworlds: torch.Tensor,
    pool_valid: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Select nearest valid camera views independently for each query.

    The returned mask keeps the full view dimension but restricts each query's
    deformable attention to at most topk projected-valid views.
    """
    if topk <= 0:
        return pool_valid
    b, v, n = pool_valid.shape
    k = min(int(topk), int(v))
    if k >= v:
        return pool_valid

    cam_centers = camtoworlds[..., :3, 3].to(dtype=sp_xyz.dtype)
    dist = torch.linalg.norm(sp_xyz[:, None, :, :] - cam_centers[:, :, None, :], dim=-1)
    score = dist.masked_fill(~pool_valid, torch.finfo(dist.dtype).max)
    _, idx = score.topk(k=k, dim=1, largest=False)
    selected = torch.zeros_like(pool_valid)
    selected.scatter_(1, idx, True)
    selected = selected & pool_valid

    # Fallback: if a query has no valid projected views, keep all invalid so the
    # attention output is zeroed later rather than forced to an unrelated view.
    return selected


def _select_query_topk_views_by_depth(
    sp_xyz: torch.Tensor,
    Ks: torch.Tensor,
    camtoworlds: torch.Tensor,
    depth: Optional[torch.Tensor],
    pool_valid: torch.Tensor,
    image_hw: tuple[int, int],
    topk: int,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if topk <= 0:
        return pool_valid
    b, v, n = pool_valid.shape
    k = min(int(topk), int(v))
    if k >= v or depth is None:
        return _select_query_topk_views(sp_xyz, camtoworlds, pool_valid, topk)
    if depth.dim() == 3:
        depth = _expand_to_views(depth, v)
    if depth.shape[:2] != (b, v):
        return _select_query_topk_views(sp_xyz, camtoworlds, pool_valid, topk)

    h, w = depth.shape[-2:]
    grid, proj_valid, _ = _projective_pool_grid(sp_xyz, Ks, camtoworlds, image_hw, (h, w))
    depth_f = depth.to(device=sp_xyz.device, dtype=torch.float32)
    sampled_depth = F.grid_sample(
        depth_f.reshape(b * v, 1, h, w),
        grid.to(torch.float32),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(b, v, n)

    ones = torch.ones((b, n, 1), device=sp_xyz.device, dtype=torch.float32)
    pts_h = torch.cat([sp_xyz.to(torch.float32), ones], dim=-1)
    w2c = torch.linalg.inv(camtoworlds.to(torch.float32))
    cam = torch.matmul(
        pts_h[:, None].expand(b, v, n, 4).unsqueeze(-2),
        w2c.transpose(-1, -2)[:, :, None],
    ).squeeze(-2)[..., :3]
    z_cam = cam[..., 2]

    valid = pool_valid & proj_valid & (sampled_depth > 1e-4) & torch.isfinite(sampled_depth) & torch.isfinite(z_cam)
    if valid_mask is not None:
        vm = valid_mask.to(device=sp_xyz.device).bool()
        if vm.dim() == 3:
            vm = _expand_to_views(vm, v)
        if vm.shape[:2] == (b, v):
            sampled_valid = F.grid_sample(
                vm.float().reshape(b * v, 1, vm.shape[-2], vm.shape[-1]),
                grid.to(torch.float32),
                mode="nearest",
                padding_mode="zeros",
                align_corners=True,
            ).reshape(b, v, n) > 0.5
            valid = valid & sampled_valid

    err = (sampled_depth - z_cam).abs()
    err = err.masked_fill(~valid, torch.finfo(err.dtype).max)
    _, idx = err.topk(k=k, dim=1, largest=False)
    selected = torch.zeros_like(pool_valid)
    selected.scatter_(1, idx, True)
    selected = selected & valid
    empty = ~selected.any(dim=1, keepdim=True)
    selected = torch.where(empty, _select_query_topk_views(sp_xyz, camtoworlds, pool_valid, topk), selected)
    return selected & pool_valid


def _image_gradient_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[..., 1:, :] - pred[..., :-1, :]
    gt_dx = gt[..., 1:, :] - gt[..., :-1, :]
    pred_dy = pred[..., :, 1:] - pred[..., :, :-1]
    gt_dy = gt[..., :, 1:] - gt[..., :, :-1]
    return _charbonnier(pred_dx - gt_dx) + _charbonnier(pred_dy - gt_dy)


def _voxel_downsample_points(points: torch.Tensor, voxel_size: float) -> torch.Tensor:
    if voxel_size <= 0 or points.numel() == 0:
        return points
    pts = torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
    coords = torch.floor(pts / max(voxel_size, 1e-6)).to(torch.int64)
    _, inverse = torch.unique(coords, dim=0, return_inverse=True)
    num_unique = int(inverse.max().item()) + 1 if inverse.numel() > 0 else 0
    if num_unique == 0:
        return pts[:0]
    point_idx = torch.arange(coords.shape[0], device=coords.device, dtype=torch.long)
    uniq_idx = torch.full((num_unique,), coords.shape[0], device=coords.device, dtype=torch.long)
    uniq_idx.scatter_reduce_(0, inverse, point_idx, reduce="amin", include_self=True)
    uniq_idx = uniq_idx.sort().values
    return pts.index_select(0, uniq_idx)


def _farthest_point_sample(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    n = points.shape[0]
    if n == 0:
        raise ValueError("Cannot run FPS on an empty point set.")
    if n == num_samples:
        return points
    if n < num_samples:
        extra_idx = torch.randint(0, n, (num_samples - n,), device=points.device)
        return torch.cat([points, points.index_select(0, extra_idx)], dim=0)

    pts = torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
    if pointops_farthest_point_sampling is not None and pts.is_cuda:
        try:
            xyz = pts.to(torch.float32).contiguous()
            offset = torch.tensor([xyz.shape[0]], device=xyz.device, dtype=torch.int32)
            new_offset = torch.tensor([num_samples], device=xyz.device, dtype=torch.int32)
            fps_idx = pointops_farthest_point_sampling(xyz, offset, new_offset).long()
            return pts.index_select(0, fps_idx.clamp(0, n - 1))
        except Exception:
            pass

    chosen = torch.empty((num_samples,), device=points.device, dtype=torch.long)
    dists = torch.full((n,), float("inf"), device=points.device, dtype=torch.float32)
    centroid = pts.mean(dim=0, keepdim=True)
    first = torch.argmax(((pts - centroid) ** 2).sum(dim=-1))
    farthest = first
    for i in range(num_samples):
        chosen[i] = farthest
        current = pts[farthest : farthest + 1]
        dist = ((pts - current) ** 2).sum(dim=-1)
        dists = torch.minimum(dists, dist)
        farthest = torch.argmax(dists)
    return pts.index_select(0, chosen)


class DA3OnlineMultiLevelExtractor(nn.Module):
    """Online multi-level DA3 feature extractor, GPU-only, no numpy round-trip.

    Calls DA3's inner `model.forward(x, extrinsics, intrinsics, feat_layers, ...)`
    directly on a torch tensor, returning the requested aux feature levels as
    a list of [B, V, Hf, Wf, C] float tensors. Depth is also returned for
    optional auxiliary supervision."""

    NORMALIZE_MEAN = (0.485, 0.456, 0.406)
    NORMALIZE_STD = (0.229, 0.224, 0.225)
    PATCH_SIZE = 14

    def __init__(
        self,
        da3_weights: Union[str, Path],
        feat_layers: Sequence[int],
        process_res: int = 320,
        process_res_method: str = "upper_bound_resize",
        use_camera_tokens: bool = False,
    ) -> None:
        super().__init__()
        if DepthAnything3 is None:
            raise ImportError("DepthAnything3 is unavailable. Check Depth-Anything-3/da3 import path.")
        self.feat_layers = tuple(int(x) for x in feat_layers)
        self.process_res = int(process_res)
        self.process_res_method = str(process_res_method)
        self.use_camera_tokens = bool(use_camera_tokens)
        if self.process_res <= 0:
            raise ValueError(f"process_res={process_res} too small for patch size {self.PATCH_SIZE}")
        if self.process_res_method not in {
            "upper_bound_resize",
            "lower_bound_resize",
            "upper_bound_crop",
            "lower_bound_crop",
        }:
            raise ValueError(f"Unsupported DA3 process_res_method: {self.process_res_method}")

        self.api = DepthAnything3.from_pretrained(str(da3_weights))
        self.api.eval()
        for p in self.api.parameters():
            p.requires_grad = False

        self.register_buffer(
            "_img_mean",
            torch.tensor(self.NORMALIZE_MEAN, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_img_std",
            torch.tensor(self.NORMALIZE_STD, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )

    @staticmethod
    def _nearest_multiple(x: int, p: int) -> int:
        down = (x // p) * p
        up = down + p
        return max(1, up if abs(up - x) <= abs(x - down) else down)

    @classmethod
    def _compute_processed_hw(cls, h: int, w: int, process_res: int, method: str) -> tuple[int, int]:
        if method in ("upper_bound_resize", "upper_bound_crop"):
            side = max(h, w)
        elif method in ("lower_bound_resize", "lower_bound_crop"):
            side = min(h, w)
        else:
            raise ValueError(f"Unsupported DA3 process_res_method: {method}")
        if side != int(process_res):
            scale = float(process_res) / float(max(side, 1))
            h = max(1, int(round(h * scale)))
            w = max(1, int(round(w * scale)))
        if method.endswith("resize"):
            h = cls._nearest_multiple(h, cls.PATCH_SIZE)
            w = cls._nearest_multiple(w, cls.PATCH_SIZE)
        else:
            h = max(cls.PATCH_SIZE, (h // cls.PATCH_SIZE) * cls.PATCH_SIZE)
            w = max(cls.PATCH_SIZE, (w // cls.PATCH_SIZE) * cls.PATCH_SIZE)
        return h, w

    def _normalize_extrinsics(self, ex: torch.Tensor) -> torch.Tensor:
        # Mirror DepthAnything3._normalize_extrinsics on a torch tensor.
        # ex: [B, V, 4, 4] camera-to-world (per DA3.inference convention).
        with torch.autocast(device_type=ex.device.type, enabled=False):
            ex_f = ex.float()
            ref = ex_f[:, :1]
            transform = torch.linalg.inv(ref)
            ex_norm = ex_f @ transform
            c2ws = torch.linalg.inv(ex_norm)
            translations = c2ws[..., :3, 3]
            dists = translations.norm(dim=-1)
            median_dist = dists.median(dim=1).values.clamp_min(0.1)
            ex_norm = ex_norm.clone()
            ex_norm[..., :3, 3] = ex_norm[..., :3, 3] / median_dist[:, None, None]
        return ex_norm

    def forward(
        self,
        image: torch.Tensor,
        Ks: torch.Tensor,
        camtoworlds: torch.Tensor,
    ) -> tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            image: [B, V, H, W, 3] in [0, 1]
            Ks: [B, V, 3, 3]
            camtoworlds: [B, V, 4, 4] (c2w)
        Returns:
            da3_depths: [B, V, H, W] (resized back to input H, W)
            da3_levels: list of len L, each [B, V, Hf, Wf, C]
        """
        b, v, h, w, _ = image.shape
        device = image.device
        proc_h, proc_w = self._compute_processed_hw(
            h,
            w,
            self.process_res,
            self.process_res_method,
        )

        x = image.detach().to(torch.float32).permute(0, 1, 4, 2, 3).contiguous()
        if h != proc_h or w != proc_w:
            x = x.reshape(b * v, 3, h, w)
            x = F.interpolate(x, size=(proc_h, proc_w), mode="bilinear", align_corners=False)
            x = x.reshape(b, v, 3, proc_h, proc_w)
        x = (x - self._img_mean.to(device)) / self._img_std.to(device)

        ex_in: Optional[torch.Tensor] = None
        in_in: Optional[torch.Tensor] = None
        if self.use_camera_tokens:
            Ks_proc = Ks.detach().to(torch.float32).clone()
            Ks_proc[..., 0, :] *= float(proc_w) / float(w)
            Ks_proc[..., 1, :] *= float(proc_h) / float(h)
            in_in = Ks_proc
            ex_in = self._normalize_extrinsics(camtoworlds.detach().to(torch.float32))

        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        # flash_attn custom ops ignore autocast and reject fp32; feed bf16/fp16.
        # DA3 LayerNorm stays fp32 under AMP, so also disable the flash SDPA
        # backend in case torch routes SDPA through the flash_attn package.
        x = x.to(dtype=autocast_dtype)
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            _sdpa_ctx = sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION])
        except Exception:
            _sdpa_ctx = torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_mem_efficient=True, enable_math=True
            )
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                with _sdpa_ctx:
                    output = self.api.model(
                        x,
                        ex_in,
                        in_in,
                        list(self.feat_layers),
                        False,  # infer_gs
                        False,  # use_ray_pose
                        "saddle_balanced",
                    )

        depth_raw = output.get("depth", None) if hasattr(output, "get") else getattr(output, "depth", None)
        if depth_raw is None:
            da3_depths = torch.zeros((b, v, h, w), device=device, dtype=torch.float32)
        else:
            d = depth_raw.detach().float()
            if d.dim() == 5:  # [B, V, 1, Hd, Wd] or similar
                d = d.squeeze(-3)
            if d.dim() == 4 and d.shape[0] == b and d.shape[1] == v:
                d_flat = d.reshape(b * v, 1, d.shape[-2], d.shape[-1])
            else:  # fall back: treat first dim as B*V
                d_flat = d.reshape(-1, 1, d.shape[-2], d.shape[-1])
            d_flat = F.interpolate(d_flat, size=(h, w), mode="bilinear", align_corners=False)
            da3_depths = d_flat.reshape(b, v, h, w)

        aux = getattr(output, "aux", None)
        if aux is None:
            raise RuntimeError("DA3 output missing `aux`; check export_feat_layers wiring.")
        levels: List[torch.Tensor] = []
        for layer_id in self.feat_layers:
            key = f"feat_layer_{layer_id}"
            if key not in aux:
                raise KeyError(
                    f"Missing DA3 aux feature {key}. Available: {list(aux.keys())}"
                )
            t = aux[key]  # [B, V, Hf, Wf, C]
            t = t.detach().to(torch.float32).contiguous()
            levels.append(t)

        return da3_depths, levels


class SuperpointDeformableAttention(nn.Module):
    """Query-to-DA3 deformable attention using superpoint projections as reference points."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        num_points: int = 4,
        im2col_step: int = 64,
        max_views: int = 8,
    ) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by n_heads={n_heads}")
        self.dim = int(dim)
        self.n_heads = int(n_heads)
        self.num_points = int(num_points)
        self.im2col_step = int(im2col_step)
        self.max_views = int(max_views)
        self.value_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        self.sampling_offsets = nn.Linear(dim, n_heads * max_views * num_points * 2)
        self.attention_weights = nn.Linear(dim, n_heads * max_views * num_points)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * torch.pi / self.n_heads)
        grid = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid = grid / grid.abs().max(dim=-1, keepdim=True).values.clamp_min(1e-6)
        grid = grid.view(self.n_heads, 1, 1, 2).repeat(1, self.max_views, self.num_points, 1)
        for i in range(self.num_points):
            grid[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid.reshape(-1))

    @staticmethod
    def _reference_from_pool_grid(pool_grid: torch.Tensor, b: int, v: int, n: int) -> torch.Tensor:
        if pool_grid.shape != (b * v, n, 1, 2):
            raise ValueError(f"pool_grid shape {tuple(pool_grid.shape)} incompatible with B={b}, V={v}, N={n}")
        ref = pool_grid.reshape(b, v, n, 2).permute(0, 2, 1, 3).contiguous()
        return ((ref + 1.0) * 0.5).clamp(0.0, 1.0)

    def forward(
        self,
        query: torch.Tensor,
        value_tokens: torch.Tensor,
        da3_hw: tuple[int, int],
        pool_grid: torch.Tensor,
        pool_valid: torch.Tensor,
        query_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if MultiScaleDeformableAttnFunction_fp32 is None:
            raise ImportError(
                "mmcv multi-scale deformable attention extension is unavailable. "
                f"Original import error: {_MS_DEFORM_ATTN_IMPORT_ERROR}"
            )
        b, n, c = query.shape
        hd, wd = da3_hw
        if value_tokens.shape[0] != b or value_tokens.shape[-1] != c:
            raise ValueError(f"value_tokens shape {tuple(value_tokens.shape)} incompatible with query {tuple(query.shape)}")
        if value_tokens.shape[1] % (hd * wd) != 0:
            raise ValueError(f"value token count {value_tokens.shape[1]} does not match da3_hw={da3_hw}")
        v = value_tokens.shape[1] // (hd * wd)
        if v > self.max_views:
            raise ValueError(f"num_views={v} exceeds max_views={self.max_views}")
        if pool_valid.shape != (b, v, n):
            raise ValueError(f"pool_valid shape {tuple(pool_valid.shape)} incompatible with B={b}, V={v}, N={n}")

        value = self.value_proj(value_tokens).view(b, value_tokens.shape[1], self.n_heads, c // self.n_heads)
        spatial_shapes = torch.tensor([[hd, wd]] * v, dtype=torch.long, device=query.device)
        level_start_index = torch.cat([
            spatial_shapes.new_zeros(1),
            (spatial_shapes[:, 0] * spatial_shapes[:, 1]).cumsum(0)[:-1],
        ])
        reference_points = self._reference_from_pool_grid(pool_grid, b, v, n).to(query.dtype)
        query_for_sampling = query if query_pos is None else query + query_pos
        sampling_offsets = self.sampling_offsets(query_for_sampling).view(
            b, n, self.n_heads, self.max_views, self.num_points, 2
        )[:, :, :, :v]
        attention_weights = self.attention_weights(query_for_sampling).view(
            b, n, self.n_heads, self.max_views, self.num_points
        )[:, :, :, :v]
        view_valid = pool_valid.permute(0, 2, 1)[:, :, None, :, None]
        attention_weights = attention_weights.masked_fill(~view_valid, -1e4)
        attention_weights = attention_weights.reshape(b, n, self.n_heads, v * self.num_points).softmax(-1)
        attention_weights = attention_weights.view(b, n, self.n_heads, v, self.num_points)
        has_valid = pool_valid.any(dim=1)
        attention_weights = attention_weights * has_valid[:, :, None, None, None].to(attention_weights.dtype)

        offset_normalizer = torch.stack([spatial_shapes[:, 1], spatial_shapes[:, 0]], dim=-1).to(query.dtype)
        sampling_locations = reference_points[:, :, None, :, None, :] + (
            sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        )
        output = MultiScaleDeformableAttnFunction_fp32.apply(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations.to(value.dtype),
            attention_weights.to(value.dtype),
            self.im2col_step,
        )
        output = output * has_valid.to(output.dtype).unsqueeze(-1)
        return self.output_proj(output)


class GlobalQuerySelfAttention(nn.Module):
    """Full self-attention over all valid superpoint queries."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        use_rope: bool = False,
        rope_base: float = 10000.0,
        rope_fwd: float = 1.0,
        rope_bits: int = 10,
    ) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by n_heads={n_heads}")
        self.dim = int(dim)
        self.n_heads = int(n_heads)
        self.head_dim = int(dim) // int(n_heads)
        self.use_rope = bool(use_rope)
        self.rope_base = float(rope_base)
        self.rope_fwd = float(rope_fwd)
        self.rope_bits = int(rope_bits)
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        sp_xyz: Optional[torch.Tensor] = None,
        sp_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv_proj(x).view(b, n, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        if sp_padding_mask is not None:
            valid = (~sp_padding_mask)[:, :, None, None].to(q.dtype)
            q = q * valid
            k = k * valid
            v = v * valid
        if self.use_rope:
            if sp_xyz is None:
                raise ValueError("sp_xyz is required when query self-attention RoPE is enabled.")
            pos = _quantize_rope_xyz(sp_xyz, bits=self.rope_bits)
            q = _apply_point_rope(q, pos, enabled=True, base=self.rope_base, fwd=self.rope_fwd)
            k = _apply_point_rope(k, pos, enabled=True, base=self.rope_base, fwd=self.rope_fwd)
        if flash_attn_func is None:
            raise ImportError(
                "flash_attn is unavailable for GlobalQuerySelfAttention. "
                f"Original import error: {_FLASH_ATTN_IMPORT_ERROR}"
            )
        q, k, v, src_dtype = _flash_attn_maybe_cast(q, k, v)
        out = flash_attn_func(q, k, v, dropout_p=0.0, causal=False).reshape(b, n, c)
        if out.dtype != src_dtype:
            out = out.to(src_dtype)
        out = self.out_proj(out)
        if sp_padding_mask is not None:
            out = out.masked_fill(sp_padding_mask[:, :, None], 0.0)
        return out


class MortonFlashLocalSelfAttention(nn.Module):
    """FlashAttention local-window self-attention after Morton sorting xyz queries."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        window_size: int = 64,
        use_rope: bool = False,
        rope_base: float = 10000.0,
        rope_fwd: float = 1.0,
        rope_bits: int = 10,
    ) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by n_heads={n_heads}")
        self.dim = int(dim)
        self.n_heads = int(n_heads)
        self.window_size = max(int(window_size), 1)
        self.use_rope = bool(use_rope)
        self.rope_base = float(rope_base)
        self.rope_fwd = float(rope_fwd)
        self.rope_bits = int(rope_bits)
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)

    @staticmethod
    @torch.no_grad()
    def _quantize_xyz(sp_xyz: torch.Tensor, bits: int = 10) -> torch.Tensor:
        b, n, _ = sp_xyz.shape
        xyz = sp_xyz.float()
        xyz_min = xyz.amin(dim=1, keepdim=True)
        xyz_max = xyz.amax(dim=1, keepdim=True)
        xyz_n = (xyz - xyz_min) / (xyz_max - xyz_min).clamp_min(1e-6)
        max_coord = (1 << bits) - 1
        return (xyz_n.clamp(0.0, 1.0) * max_coord).long()

    @staticmethod
    def _morton_code(q: torch.Tensor, bits: int = 10) -> torch.Tensor:
        b, n, _ = q.shape
        code = torch.zeros((b, n), device=q.device, dtype=torch.long)
        for i in range(bits):
            code = code | (((q[..., 0] >> i) & 1) << (3 * i))
            code = code | (((q[..., 1] >> i) & 1) << (3 * i + 1))
            code = code | (((q[..., 2] >> i) & 1) << (3 * i + 2))
        return code

    @staticmethod
    def _right_shift(binary: torch.Tensor, k: int = 1, axis: int = -1) -> torch.Tensor:
        if binary.shape[axis] <= k:
            return torch.zeros_like(binary)
        slicing = [slice(None)] * len(binary.shape)
        slicing[axis] = slice(None, -k)
        return F.pad(binary[tuple(slicing)], (k, 0), mode="constant", value=0)

    @staticmethod
    def _gray_to_binary(gray: torch.Tensor, axis: int = -1) -> torch.Tensor:
        shift = 2 ** (torch.tensor([gray.shape[axis]], device=gray.device).log2().ceil().int() - 1)
        while int(shift.item()) > 0:
            gray = torch.logical_xor(gray, MortonFlashLocalSelfAttention._right_shift(gray, int(shift.item()), axis=axis))
            shift = torch.div(shift, 2, rounding_mode="floor")
        return gray

    @staticmethod
    def _hilbert_code(q: torch.Tensor, bits: int = 10) -> torch.Tensor:
        orig_shape = q.shape
        if orig_shape[-1] != 3:
            raise ValueError(f"Hilbert encoding expects [...,3] coords, got {tuple(q.shape)}")
        bitpack_mask = 1 << torch.arange(0, 8, device=q.device)
        bitpack_mask_rev = bitpack_mask.flip(-1)
        locs_uint8 = q.long().reshape(-1, 3).view(torch.uint8).reshape((-1, 3, 8)).flip(-1)
        gray = (
            locs_uint8.unsqueeze(-1)
            .bitwise_and(bitpack_mask_rev)
            .ne(0)
            .byte()
            .flatten(-2, -1)[..., -bits:]
        )
        for bit in range(bits):
            for dim in range(3):
                mask = gray[:, dim, bit]
                gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], mask[:, None])
                to_flip = torch.logical_and(
                    torch.logical_not(mask[:, None]).repeat(1, gray.shape[2] - bit - 1),
                    torch.logical_xor(gray[:, 0, bit + 1 :], gray[:, dim, bit + 1 :]),
                )
                gray[:, dim, bit + 1 :] = torch.logical_xor(gray[:, dim, bit + 1 :], to_flip)
                gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], to_flip)
        gray = gray.swapaxes(1, 2).reshape((-1, bits * 3))
        hh_bin = MortonFlashLocalSelfAttention._gray_to_binary(gray)
        padded = F.pad(hh_bin, (64 - bits * 3, 0), "constant", 0)
        hh_uint8 = (
            (padded.flip(-1).reshape((-1, 8, 8)) * bitpack_mask)
            .sum(2)
            .squeeze()
            .type(torch.uint8)
        )
        return hh_uint8.view(torch.int64).reshape(orig_shape[:-1])

    @staticmethod
    def _order_from_code(code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, n = code.shape
        order = code.argsort(dim=1)
        inv_order = torch.empty_like(order)
        arange = torch.arange(n, device=code.device).view(1, n).expand(b, n)
        inv_order.scatter_(1, order, arange)
        return order, inv_order

    @staticmethod
    @torch.no_grad()
    def serialization_orders(
        sp_xyz: torch.Tensor,
        order_names: Sequence[str] = ("z", "z-trans", "hilbert", "hilbert-trans"),
        bits: int = 10,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        q_base = MortonFlashLocalSelfAttention._quantize_xyz(sp_xyz, bits=bits)
        orders: list[torch.Tensor] = []
        inv_orders: list[torch.Tensor] = []
        for name in order_names:
            if name == "z":
                code = MortonFlashLocalSelfAttention._morton_code(q_base, bits=bits)
            elif name == "z-trans":
                code = MortonFlashLocalSelfAttention._morton_code(q_base[..., [1, 0, 2]], bits=bits)
            elif name == "hilbert":
                code = MortonFlashLocalSelfAttention._hilbert_code(q_base, bits=bits)
            elif name == "hilbert-trans":
                code = MortonFlashLocalSelfAttention._hilbert_code(q_base[..., [1, 0, 2]], bits=bits)
            else:
                raise ValueError(f"Unsupported serialization order: {name}")
            order, inv_order = MortonFlashLocalSelfAttention._order_from_code(code)
            orders.append(order)
            inv_orders.append(inv_order)
        return orders, inv_orders

    @staticmethod
    def _gather_tokens(x: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        return torch.gather(x, 1, order[:, :, None].expand(-1, -1, x.shape[-1]))

    def forward(
        self,
        x: torch.Tensor,
        order: torch.Tensor,
        inv_order: torch.Tensor,
        sp_xyz: Optional[torch.Tensor] = None,
        sp_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if flash_attn_func is None:
            raise ImportError(
                "flash_attn is unavailable for MortonFlashLocalSelfAttention. "
                f"Original import error: {_FLASH_ATTN_IMPORT_ERROR}"
            )
        b, n, c = x.shape
        x_sorted = self._gather_tokens(x, order)
        qkv = self.qkv_proj(x_sorted).view(b, n, 3, self.n_heads, c // self.n_heads)
        q, k, v = qkv.unbind(dim=2)
        if self.use_rope:
            if sp_xyz is None:
                raise ValueError("sp_xyz is required when query self-attention RoPE is enabled.")
            pos = _quantize_rope_xyz(sp_xyz, bits=self.rope_bits)
            pos_sorted = torch.gather(pos, 1, order[:, :, None].expand(-1, -1, 3)).contiguous()
            q = _apply_point_rope(q, pos_sorted, enabled=True, base=self.rope_base, fwd=self.rope_fwd)
            k = _apply_point_rope(k, pos_sorted, enabled=True, base=self.rope_base, fwd=self.rope_fwd)
        half_window = max(self.window_size // 2, 1)
        q, k, v, src_dtype = _flash_attn_maybe_cast(q, k, v)
        out_sorted = flash_attn_func(
            q,
            k,
            v,
            dropout_p=0.0,
            causal=False,
            window_size=(half_window, half_window),
        ).reshape(b, n, c)
        if out_sorted.dtype != src_dtype:
            out_sorted = out_sorted.to(src_dtype)
        out_sorted = self.out_proj(out_sorted)
        out = self._gather_tokens(out_sorted, inv_order)
        if sp_padding_mask is not None:
            out = out.masked_fill(sp_padding_mask[:, :, None], 0.0)
        return out


class UnifiedDA3FusionBlock(nn.Module):
    """C14 fusion block.

    Differences from C13:
      - cross_fuse + mv_residual_fuse + scalar gate replaced by a single
        FFN-style `unified_fuse` over concat(main_update, mv_update). The query
        state x is no longer fed back into the fuse; the residual `x + ...` is
        the only place x is reused, matching the standard transformer cross-
        attention residual pattern.
      - Removes the mv_residual_gate entirely. Since 4a now specializes the
        main / mv paths into texture-from-few-views vs geometry-from-many-views,
        the mv branch is reliable enough that a sigmoid suppression is no
        longer warranted. Removing it also unlocks 50x stronger gradient flow
        to `level_logits_mv` (was bottlenecked by sigmoid(-4)≈0.018 in v3_43).
      - 4a is retained; temperatures are tuned in the encoder, not here.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: float,
        max_views: int,
        deform_points: int,
        deform_im2col_step: int,
        mv_residual_deform_points: int,
        unified_fuse_mlp_ratio: float,
        self_window: int,
        self_attn_type: str,
        self_attn_use_rope: bool,
        self_attn_rope_base: float,
        self_attn_rope_fwd: float,
        self_attn_rope_bits: int,
        order_index: int,
    ) -> None:
        super().__init__()
        self.order_index = int(order_index)
        self.self_attn_type = str(self_attn_type).lower()
        if self.self_attn_type not in {"local", "global"}:
            raise ValueError(f"Unsupported query self-attention type: {self_attn_type}")

        self.norm_q_cross = RMSNorm(dim)
        self.norm_da3_for_q = RMSNorm(dim)
        self.norm_pos_for_q = RMSNorm(dim)
        self.query_da3_attn = SuperpointDeformableAttention(
            dim=dim,
            n_heads=n_heads,
            num_points=deform_points,
            im2col_step=deform_im2col_step,
            max_views=max_views,
        )
        self.mv_residual_attn = SuperpointDeformableAttention(
            dim=dim,
            n_heads=n_heads,
            num_points=mv_residual_deform_points,
            im2col_step=deform_im2col_step,
            max_views=max_views,
        )
        # FFN-style unified fuse: concat(main_update, mv_update) → hidden → dim.
        # Hidden dim follows the same expansion ratio as the block's main FFN
        # (typically 4x), giving the texture+geometry combination enough
        # capacity. No bottleneck via dim*2 → dim.
        hidden = max(int(dim * float(unified_fuse_mlp_ratio)), dim)
        self.unified_fuse = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

        self.norm_self = RMSNorm(dim)
        if self.self_attn_type == "global":
            self.self_attn = GlobalQuerySelfAttention(
                dim,
                n_heads,
                use_rope=self_attn_use_rope,
                rope_base=self_attn_rope_base,
                rope_fwd=self_attn_rope_fwd,
                rope_bits=self_attn_rope_bits,
            )
        else:
            self.self_attn = MortonFlashLocalSelfAttention(
                dim,
                n_heads,
                window_size=self_window,
                use_rope=self_attn_use_rope,
                rope_base=self_attn_rope_base,
                rope_fwd=self_attn_rope_fwd,
                rope_bits=self_attn_rope_bits,
            )
        self.norm_ffn = RMSNorm(dim)
        self.ffn = SwiGLU(dim, int(dim * mlp_ratio))

    def forward(
        self,
        x: torch.Tensor,
        da3_tokens: torch.Tensor,
        da3_pos: torch.Tensor,
        da3_hw: tuple[int, int],
        pool_grid: torch.Tensor,
        pool_valid: torch.Tensor,
        mv_pool_valid: Optional[torch.Tensor],
        self_orders: Optional[Sequence[torch.Tensor]],
        self_inv_orders: Optional[Sequence[torch.Tensor]],
        sp_xyz: torch.Tensor,
        sp_padding_mask: Optional[torch.Tensor] = None,
        da3_tokens_mv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.norm_q_cross(x)
        kv = self.norm_da3_for_q(da3_tokens)
        q_pos = self.norm_pos_for_q(da3_pos)
        da3_update = self.query_da3_attn(
            q, kv, da3_hw=da3_hw, pool_grid=pool_grid, pool_valid=pool_valid, query_pos=q_pos
        )
        if mv_pool_valid is not None and bool(mv_pool_valid.any()):
            kv_mv = kv if da3_tokens_mv is None else self.norm_da3_for_q(da3_tokens_mv)
            mv_update = self.mv_residual_attn(
                q,
                kv_mv,
                da3_hw=da3_hw,
                pool_grid=pool_grid,
                pool_valid=mv_pool_valid,
                query_pos=q_pos,
            )
        else:
            mv_update = torch.zeros_like(da3_update)
        x = x + self.unified_fuse(torch.cat([da3_update, mv_update], dim=-1))

        q_self = self.norm_self(x)
        if self.self_attn_type == "global":
            h = self.self_attn(q_self, sp_xyz=sp_xyz, sp_padding_mask=sp_padding_mask)
        else:
            if self_orders is None or self_inv_orders is None or len(self_orders) == 0:
                raise ValueError("self_orders and self_inv_orders are required for local query self-attention.")
            order_idx = self.order_index % len(self_orders)
            h = self.self_attn(
                q_self,
                self_orders[order_idx],
                self_inv_orders[order_idx],
                sp_xyz=sp_xyz,
                sp_padding_mask=sp_padding_mask,
            )
        x = x + h
        x = x + self.ffn(self.norm_ffn(x))
        return x


class MultiLevelSuperpointEncoderC(nn.Module):
    """C14 encoder.

    Drops the mv_residual gate entirely (no longer needed; texture/geometry
    specialization through 4a is reliable). Adds per-path temperatures on the
    4a layer-mix softmax so the main (texture) path can be sharpened to a
    narrow band of DA3 layers while the mv (geometry) path stays broad.

    `level_logits_main` / `level_logits_mv` are zero-initialized; with tau != 1
    the softmax still degenerates to uniform at step 0. Training from scratch.
    """

    def __init__(
        self,
        img_feat_in_dim: int = 1024,
        num_da3_levels: int = 24,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        pooled_level_dim: int = 128,
        n_layers: int = 8,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        max_views: int = 8,
        plucker_dim: int = 6,
        deform_points: int = 4,
        mv_residual_deform_points: int = 4,
        level_logits_tau_main: float = 0.3,
        level_logits_tau_mv: float = 1.0,
        unified_fuse_mlp_ratio: float = 4.0,
        deform_im2col_step: int = 64,
        self_window: int = 64,
        self_attn_type: str = "local",
        self_attn_use_rope: bool = False,
        self_attn_rope_base: float = 10000.0,
        self_attn_rope_fwd: float = 1.0,
        self_attn_rope_bits: int = 10,
        debug_nan_checks: bool = False,
    ) -> None:
        super().__init__()
        self.num_levels = int(num_da3_levels)
        self.hidden_dim = int(hidden_dim)
        self.debug_nan_checks = bool(debug_nan_checks)
        self.level_logits_tau_main = float(level_logits_tau_main)
        self.level_logits_tau_mv = float(level_logits_tau_mv)
        if self.level_logits_tau_main <= 0 or self.level_logits_tau_mv <= 0:
            raise ValueError("level_logits_tau_main / level_logits_tau_mv must be > 0")
        self.level_logits_main = nn.Parameter(torch.zeros(self.num_levels))
        self.level_logits_mv = nn.Parameter(torch.zeros(self.num_levels))
        self.xyz_fourier = FourierFeatureEncoder(in_dim=3, num_bands=8, include_input=True)

        self.query_content_proj = nn.Sequential(
            nn.Linear(img_feat_in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.xyz_pos_proj = nn.Sequential(
            nn.Linear(self.xyz_fourier.out_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.da3_feat_proj = nn.Linear(img_feat_in_dim, hidden_dim)
        self.da3_ray_proj = nn.Linear(plucker_dim, hidden_dim)

        self.blocks = nn.ModuleList(
            [
                UnifiedDA3FusionBlock(
                    hidden_dim,
                    n_heads,
                    mlp_ratio,
                    max_views=max_views,
                    deform_points=deform_points,
                    deform_im2col_step=deform_im2col_step,
                    mv_residual_deform_points=mv_residual_deform_points,
                    unified_fuse_mlp_ratio=float(unified_fuse_mlp_ratio),
                    self_window=self_window,
                    self_attn_type=self_attn_type,
                    self_attn_use_rope=self_attn_use_rope,
                    self_attn_rope_base=self_attn_rope_base,
                    self_attn_rope_fwd=self_attn_rope_fwd,
                    self_attn_rope_bits=self_attn_rope_bits,
                    order_index=i,
                )
                for i in range(n_layers)
            ]
        )
        self.out_norm = RMSNorm(hidden_dim)
        self.latent_head = nn.Linear(hidden_dim, latent_dim)

    def _weighted_levels(
        self, levels: List[torch.Tensor], logits: torch.Tensor, tau: float
    ) -> torch.Tensor:
        if len(levels) == 0:
            raise ValueError("DA3 levels must be non-empty.")
        if int(logits.numel()) != len(levels):
            raise ValueError(
                f"level logits dim {int(logits.numel())} must match num_levels {len(levels)}"
            )
        stacked = torch.stack(levels, dim=0)
        weight_shape = (-1,) + (1,) * (stacked.dim() - 1)
        w = (logits / float(tau)).softmax(dim=0).view(*weight_shape).to(stacked.dtype)
        return (stacked * w).sum(dim=0)

    def _build_da3_tokens(
        self, levels: List[torch.Tensor], logits: torch.Tensor, tau: float
    ) -> torch.Tensor:
        return self.da3_feat_proj(self._weighted_levels(levels, logits, tau))

    def _build_da3_pos_tokens(self, plucker: torch.Tensor) -> torch.Tensor:
        return self.da3_ray_proj(plucker)

    def forward(
        self,
        da3_levels: List[torch.Tensor],
        pooled_feats: List[torch.Tensor],
        vis_ratio: torch.Tensor,
        sp_xyz: torch.Tensor,
        plucker: torch.Tensor,
        da3_hw: tuple[int, int],
        pool_grid: torch.Tensor,
        pool_valid_layers: Sequence[torch.Tensor],
        mv_pool_valid_layers: Optional[Sequence[torch.Tensor]] = None,
        sp_valid_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        sp_padding_mask = ~sp_valid_mask.bool() if sp_valid_mask is not None else None
        pooled = torch.stack(pooled_feats, dim=0).mean(dim=0)
        x_content = self.query_content_proj(pooled)
        x_pos = self.xyz_pos_proj(self.xyz_fourier(sp_xyz))
        x = x_content + x_pos
        if sp_padding_mask is not None:
            x = x.masked_fill(sp_padding_mask[:, :, None], 0.0)

        self_orders, self_inv_orders = MortonFlashLocalSelfAttention.serialization_orders(sp_xyz.detach())
        da3_tokens_main = self._build_da3_tokens(
            da3_levels, self.level_logits_main, self.level_logits_tau_main
        )
        da3_tokens_mv = self._build_da3_tokens(
            da3_levels, self.level_logits_mv, self.level_logits_tau_mv
        )
        da3_pos_tokens = self._build_da3_pos_tokens(plucker)
        b = sp_xyz.shape[0]
        v = pool_valid_layers[0].shape[1]
        hd, wd = da3_hw
        da3_pos_map = da3_pos_tokens.reshape(b, v, hd, wd, self.hidden_dim)

        valid_ratios = []
        mv_valid_ratios = []
        for block_idx, block in enumerate(self.blocks):
            pool_valid_l = pool_valid_layers[min(block_idx, len(pool_valid_layers) - 1)]
            mv_pool_valid_l = None
            if mv_pool_valid_layers is not None and len(mv_pool_valid_layers) > 0:
                mv_pool_valid_l = mv_pool_valid_layers[min(block_idx, len(mv_pool_valid_layers) - 1)]
                mv_valid_ratios.append(mv_pool_valid_l.float().mean(dim=1).unsqueeze(-1))
            valid_ratios.append(pool_valid_l.float().mean(dim=1).unsqueeze(-1))
            da3_pos = _projective_pool_sample(da3_pos_map, pool_grid, pool_valid_l)
            x = block(
                x,
                da3_tokens=da3_tokens_main,
                da3_pos=da3_pos,
                da3_hw=da3_hw,
                pool_grid=pool_grid,
                pool_valid=pool_valid_l,
                mv_pool_valid=mv_pool_valid_l,
                self_orders=self_orders,
                self_inv_orders=self_inv_orders,
                sp_xyz=sp_xyz,
                sp_padding_mask=sp_padding_mask,
                da3_tokens_mv=da3_tokens_mv,
            )

        z = self.latent_head(self.out_norm(x))
        if sp_padding_mask is not None:
            z = z.masked_fill(sp_padding_mask[:, :, None], 0.0)
        aux = {
            "anchor_offsets": z.new_zeros((*z.shape[:2], 0, 3)),
            "anchor_offset_total": z.new_zeros((*z.shape[:2], 3)),
            "anchor_attn_vis_ratio": torch.stack(valid_ratios, dim=2).mean(dim=2),
            "mv_residual_vis_ratio": (
                torch.stack(mv_valid_ratios, dim=2).mean(dim=2)
                if mv_valid_ratios
                else z.new_zeros((*z.shape[:2], 1))
            ),
            "level_weights_main": (
                (self.level_logits_main / self.level_logits_tau_main)
                .softmax(dim=0)
                .detach()
                .to(z.dtype)
            ),
            "level_weights_mv": (
                (self.level_logits_mv / self.level_logits_tau_mv)
                .softmax(dim=0)
                .detach()
                .to(z.dtype)
            ),
        }
        return z, sp_xyz, aux



class FourierFeatureEncoder(nn.Module):
    def __init__(self, in_dim: int, num_bands: int = 6, include_input: bool = True):
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_bands = int(num_bands)
        self.include_input = bool(include_input)
        freq_bands = 2.0 ** torch.arange(self.num_bands, dtype=torch.float32)
        self.register_buffer("freq_bands", freq_bands, persistent=False)
        self.out_dim = self.in_dim * (2 * self.num_bands + (1 if self.include_input else 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import math
        out = [x] if self.include_input else []
        xb = x.unsqueeze(-1) * self.freq_bands
        xb = xb * math.pi
        out.append(torch.sin(xb).flatten(start_dim=-2))
        out.append(torch.cos(xb).flatten(start_dim=-2))
        return torch.cat(out, dim=-1)


class _LatentSlot(nn.Module):
    """Placeholder until SceneLatentVAE installs the grid decoder."""

    def __init__(self, feat_dim: int) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)


class SuperpointVAE2C(nn.Module):
    """Scheme C:
      - encoder: Mask2Former-style multi-level cross-attention over DA3 features
                 (layers [11, 15, 19, 23] by default via dataset side; placeholder
                 replicates the single available layer for Phase-1 smoke tests);
      - free latent z_sp [B, N, latent_dim], no channel partition;
      - decoder: superpoint_decoder.SuperpointDecoderPipeline with a freshly
                 trained RGB CNN upsampler (no frozen VAE)."""

    def __init__(
        self,
        latent_dim: int = 128,
        encoder_img_feat_in_dim: int = 1024,
        num_da3_levels: int = 4,
        dino_dim: int = 24,
        da3_dim: int = 24,
        vae_latent_channels: int = 16,
        encoder_hidden_dim: int = 256,
        encoder_pooled_level_dim: int = 128,
        encoder_layers: int = 8,
        encoder_heads: int = 8,
        encoder_mlp_ratio: float = 4.0,
        encoder_max_views: int = 4,
        encoder_view_capacity: Optional[int] = None,
        encoder_query_view_topk: int = 0,
        encoder_query_init_topk: Optional[int] = None,
        encoder_query_attn_topk: Optional[int] = None,
        encoder_query_attn_topk_schedule: Optional[Sequence[int]] = None,
        encoder_mv_residual_topk_schedule: Optional[Sequence[int]] = None,
        encoder_mv_residual_deform_points: int = 4,
        encoder_level_logits_tau_main: float = 0.3,
        encoder_level_logits_tau_mv: float = 1.0,
        encoder_unified_fuse_mlp_ratio: float = 4.0,
        online_da3: bool = True,
        da3_weights: Union[str, Path] = "/mnt/share/codes/kimi/ssf/DA3-BASE",
        da3_feat_layers: Sequence[int] = (11, 15, 19, 23),
        da3_process_res: int = 322,
        da3_process_res_method: str = "upper_bound_resize",
        da3_use_camera_tokens: bool = False,
        decoder_cross_layers: int = 6,
        decoder_rgb_grid_channels: int = 64,
        decoder_depth_grid_channels: Optional[int] = None,
        decoder_rgb_cross_layers: Optional[int] = None,
        decoder_depth_cross_layers: Optional[int] = None,
        decoder_rgb_num_views: int = 2,
        decoder_depth_num_views: int = 2,
        decoder_cross_heads: int = 4,
        decoder_rgb_cross_heads: Optional[int] = None,
        decoder_depth_cross_heads: Optional[int] = None,
        decoder_knn_k: int = 10,
        vae_scale_factor: int = 8,
        rgb_loss_weight: float = 1.0,
        depth_loss_weight: float = 1.0,
        lpips_loss_weight: float = 0.0,
        latent_reg_weight: float = 0.01,
        sp_feat_loss_weight: float = 0.3,
        vae_distill_weight: float = 0.0,
        rgb_fullres_weight: float = 0.5,
        rgb_grad_weight: float = 0.2,
        dino_cos_weight: float = 0.5,
        dino_mse_weight: float = 0.5,
        da3_teacher_weight: float = 0.2,
        vae_teacher_weight: float = 1.0,
        sp_vae_loss_weight: float = 0.0,
        sp_vae_cos_weight: float = 1.0,
        sp_vae_mse_weight: float = 0.0,
        pixel_cls_weight: float = 0.0,
        pixel_cls_num_samples: int = 1024,
        pixel_cls_temperature: float = 0.07,
        pixel_cls_embed_dim: int = 64,
        coarse_pull_weight: float = 0.0,
        depth_da3_distill_weight: float = 0.05,
        depth_grad_weight: float = 0.0,
        rgb_to_depth_attn_weight: float = 0.0,
        depth_to_rgb_attn_weight: float = 0.0,
        rgb_decoder_attn_sp_weight: float = 0.0,
        depth_decoder_attn_sp_weight: float = 0.0,
        depth_smooth_weight: float = 0.02,
        depth_smooth_edge_alpha: float = 10.0,
        depth_smooth_use_da3_edge: bool = True,
        train_depth_patch_size: int = 2,
        geometry_loss_weight: float = 0.0,
        geometry_radius_weight: float = 1.0,
        geometry_eigvals_weight: float = 1.0,
        geometry_normal_weight: float = 0.5,
        geometry_scalar_weight: float = 0.25,
        eval_use_all_encoder_views: bool = True,
        encoder_vae_local_window: int = 5,
        encoder_deform_points: int = 4,
        encoder_deform_im2col_step: int = 64,
        encoder_self_window: int = 64,
        encoder_self_attn_type: str = "local",
        encoder_self_attn_use_rope: bool = False,
        encoder_self_attn_rope_base: float = 10000.0,
        encoder_self_attn_rope_fwd: float = 1.0,
        encoder_self_attn_rope_bits: int = 10,
        sp_xyz_jitter_sigma: float = 0.01,
        sp_xyz_jitter_prob: float = 0.8,
        use_fps_spxyz: bool = True,
        infer_use_fps_spxyz: bool = False,
        infer_fps_voxel_size: float = 0.03,
        enable_sp_feat_supervision: bool = True,
        use_sp_field_distill: bool = False,
        sp_field_knn: int = 8,
        sp_field_sigma_fallback: float = 0.05,
        debug_nan_checks: bool = False,
    ) -> None:
        super().__init__()
        self.rgb_loss_weight = float(rgb_loss_weight)
        self.depth_loss_weight = float(depth_loss_weight)
        self.lpips_loss_weight = float(lpips_loss_weight)
        self.latent_reg_weight = float(latent_reg_weight)
        self.sp_feat_loss_weight = float(sp_feat_loss_weight)
        self.vae_distill_weight = 0.0
        self.rgb_fullres_weight = float(rgb_fullres_weight)
        self.rgb_grad_weight = float(rgb_grad_weight)
        self.dino_cos_weight = float(dino_cos_weight)
        self.dino_mse_weight = float(dino_mse_weight)
        self.da3_teacher_weight = float(da3_teacher_weight)
        self.vae_teacher_weight = float(vae_teacher_weight)
        self.sp_vae_loss_weight = float(sp_vae_loss_weight)
        self.sp_vae_cos_weight = float(sp_vae_cos_weight)
        self.sp_vae_mse_weight = float(sp_vae_mse_weight)
        self.pixel_cls_weight = float(pixel_cls_weight)
        self.pixel_cls_num_samples = int(pixel_cls_num_samples)
        self.pixel_cls_temperature = float(pixel_cls_temperature)
        self.coarse_pull_weight = float(coarse_pull_weight)
        self.depth_da3_distill_weight = float(depth_da3_distill_weight)
        self.depth_grad_weight = float(depth_grad_weight)
        self.rgb_to_depth_attn_weight = 0.0
        self.depth_to_rgb_attn_weight = 0.0
        self.rgb_decoder_attn_sp_weight = float(rgb_decoder_attn_sp_weight)
        self.depth_decoder_attn_sp_weight = float(depth_decoder_attn_sp_weight)
        self.depth_smooth_weight = float(depth_smooth_weight)
        self.depth_smooth_edge_alpha = float(depth_smooth_edge_alpha)
        self.depth_smooth_use_da3_edge = bool(depth_smooth_use_da3_edge)
        self.geometry_loss_weight = float(geometry_loss_weight)
        self.geometry_radius_weight = float(geometry_radius_weight)
        self.geometry_eigvals_weight = float(geometry_eigvals_weight)
        self.geometry_normal_weight = float(geometry_normal_weight)
        self.geometry_scalar_weight = float(geometry_scalar_weight)
        self.eval_use_all_encoder_views = bool(eval_use_all_encoder_views)
        self.sp_xyz_jitter_sigma = float(sp_xyz_jitter_sigma)
        self.sp_xyz_jitter_prob = float(sp_xyz_jitter_prob)
        self.use_fps_spxyz = bool(use_fps_spxyz)
        self.infer_use_fps_spxyz = bool(infer_use_fps_spxyz)
        self.infer_fps_voxel_size = float(infer_fps_voxel_size)
        self.enable_sp_feat_supervision = bool(enable_sp_feat_supervision)
        self.use_sp_field_distill = bool(use_sp_field_distill)
        self.sp_field_knn = int(sp_field_knn)
        self.sp_field_sigma_fallback = float(sp_field_sigma_fallback)
        self.debug_nan_checks = bool(debug_nan_checks)
        self.encoder_max_views = int(encoder_max_views)
        self.encoder_view_capacity = int(encoder_view_capacity) if encoder_view_capacity is not None else int(encoder_max_views)
        self.encoder_query_view_topk = int(encoder_query_view_topk)
        self.encoder_query_init_topk = (
            int(encoder_query_init_topk) if encoder_query_init_topk is not None else int(encoder_query_view_topk)
        )
        self.encoder_query_attn_topk = (
            int(encoder_query_attn_topk) if encoder_query_attn_topk is not None else int(encoder_query_view_topk)
        )
        self.encoder_query_attn_topk_schedule = (
            tuple(int(x) for x in encoder_query_attn_topk_schedule)
            if encoder_query_attn_topk_schedule is not None
            else tuple()
        )
        self.encoder_mv_residual_topk_schedule = (
            tuple(int(x) for x in encoder_mv_residual_topk_schedule)
            if encoder_mv_residual_topk_schedule is not None
            else tuple()
        )
        self.num_da3_levels = int(num_da3_levels)
        self.online_da3 = bool(online_da3)
        self.dino_dim = int(dino_dim)
        self.da3_dim = int(da3_dim)
        self.vae_latent_channels = int(vae_latent_channels)
        decoder_depth_grid_channels = (
            int(decoder_depth_grid_channels)
            if decoder_depth_grid_channels is not None
            else int(decoder_rgb_grid_channels)
        )
        self.da3_feat_layers = tuple(int(x) for x in da3_feat_layers)
        if len(self.da3_feat_layers) != self.num_da3_levels:
            raise ValueError(
                f"num_da3_levels={self.num_da3_levels} but da3_feat_layers={self.da3_feat_layers}"
            )

        self.encoder = MultiLevelSuperpointEncoderC(
            img_feat_in_dim=encoder_img_feat_in_dim,
            num_da3_levels=num_da3_levels,
            latent_dim=latent_dim,
            hidden_dim=encoder_hidden_dim,
            pooled_level_dim=encoder_pooled_level_dim,
            n_layers=encoder_layers,
            n_heads=encoder_heads,
            mlp_ratio=encoder_mlp_ratio,
            max_views=max(self.encoder_max_views, self.encoder_view_capacity),
            deform_points=encoder_deform_points,
            mv_residual_deform_points=encoder_mv_residual_deform_points,
            level_logits_tau_main=float(encoder_level_logits_tau_main),
            level_logits_tau_mv=float(encoder_level_logits_tau_mv),
            unified_fuse_mlp_ratio=float(encoder_unified_fuse_mlp_ratio),
            deform_im2col_step=encoder_deform_im2col_step,
            self_window=encoder_self_window,
            self_attn_type=encoder_self_attn_type,
            self_attn_use_rope=encoder_self_attn_use_rope,
            self_attn_rope_base=encoder_self_attn_rope_base,
            self_attn_rope_fwd=encoder_self_attn_rope_fwd,
            self_attn_rope_bits=encoder_self_attn_rope_bits,
            debug_nan_checks=debug_nan_checks,
        )
        if self.enable_sp_feat_supervision:
            self.sp_feat_head_dino = nn.Sequential(
                nn.Linear(latent_dim, encoder_hidden_dim),
                nn.GELU(),
                nn.Linear(encoder_hidden_dim, self.dino_dim),
            )
            self.sp_feat_head_da3 = nn.Sequential(
                nn.Linear(latent_dim, encoder_hidden_dim),
                nn.GELU(),
                nn.Linear(encoder_hidden_dim, self.da3_dim),
            )
            self.sp_feat_head_vae = nn.Sequential(
                nn.Linear(latent_dim, encoder_hidden_dim),
                nn.GELU(),
                nn.Linear(encoder_hidden_dim, self.vae_latent_channels),
            )
        else:
            self.sp_feat_head_dino = None
            self.sp_feat_head_da3 = None
            self.sp_feat_head_vae = None
        self.query_da3_mask_proj = nn.Sequential(
            nn.Linear(latent_dim, encoder_hidden_dim),
            nn.GELU(),
            nn.Linear(encoder_hidden_dim, self.da3_dim),
        )
        self.sp_vae_texture_head = nn.Sequential(
            nn.Linear(latent_dim, encoder_hidden_dim),
            nn.GELU(),
            nn.Linear(encoder_hidden_dim, self.vae_latent_channels),
        )
        self.geometry_radius_head = nn.Sequential(
            nn.Linear(latent_dim, encoder_hidden_dim),
            nn.GELU(),
            nn.Linear(encoder_hidden_dim, 1),
        )
        self.geometry_eigvals_head = nn.Sequential(
            nn.Linear(latent_dim, encoder_hidden_dim),
            nn.GELU(),
            nn.Linear(encoder_hidden_dim, 3),
        )
        self.geometry_normal_head = nn.Sequential(
            nn.Linear(latent_dim, encoder_hidden_dim),
            nn.GELU(),
            nn.Linear(encoder_hidden_dim, 3),
        )
        self.geometry_scalar_head = nn.Sequential(
            nn.Linear(latent_dim, encoder_hidden_dim),
            nn.GELU(),
            nn.Linear(encoder_hidden_dim, 6),
        )
        # The grid decoder replaces this slot immediately after init.
        self.decoder = _LatentSlot(latent_dim)
        self.depth_da3_distill_proj = nn.Linear(decoder_depth_grid_channels, self.da3_dim)
        self._online_da3_extractor = None
        if self.online_da3:
            self._online_da3_extractor = DA3OnlineMultiLevelExtractor(
                da3_weights=da3_weights,
                feat_layers=self.da3_feat_layers,
                process_res=da3_process_res,
                process_res_method=da3_process_res_method,
                use_camera_tokens=da3_use_camera_tokens,
            )

        self._lpips_model = None
        if self.lpips_loss_weight > 0:
            try:
                prefer_local_torchvision_vgg16()
                import lpips  # type: ignore

                lpips_model = lpips.LPIPS(net="vgg")
                for p in lpips_model.parameters():
                    p.requires_grad = False
                self.__dict__["_lpips_model"] = lpips_model
            except Exception as exc:  # pragma: no cover - optional dep
                warnings.warn(f"LPIPS requested (weight={self.lpips_loss_weight}) but unavailable: {exc}. Disabling.")
                self.lpips_loss_weight = 0.0
                self._lpips_model = None

    def _aggregate_query_to_superpoint(
        self,
        z_sp: torch.Tensor,
        query_xyz: torch.Tensor,
        sp_xyz: Optional[torch.Tensor],
        sp_radius: Optional[torch.Tensor],
        sp_valid: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.use_sp_field_distill or self.sp_field_knn <= 0 or sp_xyz is None:
            return None, None
        if z_sp.dim() != 3 or query_xyz.dim() != 3 or sp_xyz.dim() != 3:
            raise ValueError(
                "field distill expects z_sp/query_xyz/sp_xyz shapes [B,N,C], [B,N,3], [B,Nsp,3]; "
                f"got {tuple(z_sp.shape)}, {tuple(query_xyz.shape)}, {tuple(sp_xyz.shape)}"
            )
        b, n_q, c = z_sp.shape
        if sp_xyz.shape[0] != b or query_xyz.shape[:2] != (b, n_q):
            raise ValueError(
                f"field distill batch/query shape mismatch: z_sp={tuple(z_sp.shape)} "
                f"query_xyz={tuple(query_xyz.shape)} sp_xyz={tuple(sp_xyz.shape)}"
            )
        if n_q == 0 or sp_xyz.shape[1] == 0:
            return None, None

        sp_xyz = torch.nan_to_num(sp_xyz.to(device=z_sp.device, dtype=query_xyz.dtype), nan=0.0, posinf=0.0, neginf=0.0)
        query_xyz = torch.nan_to_num(query_xyz.to(dtype=sp_xyz.dtype), nan=0.0, posinf=0.0, neginf=0.0)
        k = min(int(self.sp_field_knn), int(n_q))
        knn_d, knn_i = torch.cdist(sp_xyz, query_xyz).topk(k=k, dim=-1, largest=False)
        gathered = torch.gather(
            z_sp.unsqueeze(1).expand(-1, sp_xyz.shape[1], -1, -1),
            2,
            knn_i.unsqueeze(-1).expand(-1, -1, -1, c),
        )

        if sp_radius is not None and sp_radius.shape[:2] == sp_xyz.shape[:2]:
            sigma = sp_radius.to(device=z_sp.device, dtype=knn_d.dtype).unsqueeze(-1)
            sigma = torch.nan_to_num(
                sigma,
                nan=self.sp_field_sigma_fallback,
                posinf=self.sp_field_sigma_fallback,
                neginf=self.sp_field_sigma_fallback,
            ).clamp_min(1e-4)
        else:
            sigma = knn_d.new_full((*knn_d.shape[:2], 1), self.sp_field_sigma_fallback)
        weights = torch.exp(-(knn_d**2) / (sigma**2 + 1e-6))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        sp_field = (gathered * weights.unsqueeze(-1).to(gathered.dtype)).sum(dim=2)

        if sp_valid is None:
            field_valid = torch.ones(sp_field.shape[:2], device=z_sp.device, dtype=torch.bool)
        else:
            field_valid = sp_valid.bool().to(z_sp.device)
            sp_field = sp_field.masked_fill(~field_valid.unsqueeze(-1), 0.0)
        return sp_field, field_valid

    def _maybe_jitter_sp_xyz(self, sp_xyz: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.sp_xyz_jitter_sigma <= 0:
            return sp_xyz
        if self.sp_xyz_jitter_prob < 1.0:
            if torch.rand((), device=sp_xyz.device) > self.sp_xyz_jitter_prob:
                return sp_xyz
        noise = torch.randn_like(sp_xyz) * self.sp_xyz_jitter_sigma
        return sp_xyz + noise

    def _infer_sp_xyz_from_point_cloud(
        self,
        norm_pw: torch.Tensor,
        norm_pw_offsets: Optional[torch.Tensor],
        num_queries: int,
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        out = []
        if norm_pw_offsets is not None:
            offsets = norm_pw_offsets.to(device=norm_pw.device, dtype=torch.long).view(-1)
            if (
                pointops_farthest_point_sampling is not None
                and norm_pw.is_cuda
                and self.infer_fps_voxel_size <= 0
                and offsets.numel() == fallback.shape[0]
                and torch.isfinite(norm_pw).all()
            ):
                counts = offsets - torch.cat([offsets.new_zeros(1), offsets[:-1]])
                if bool((counts >= int(num_queries)).all()):
                    xyz = torch.nan_to_num(norm_pw, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32).contiguous()
                    offset_i32 = offsets.to(torch.int32).contiguous()
                    new_offset = (
                        torch.arange(1, offsets.numel() + 1, device=norm_pw.device, dtype=torch.int32)
                        * int(num_queries)
                    )
                    fps_idx = pointops_farthest_point_sampling(xyz, offset_i32, new_offset).long()
                    return norm_pw.index_select(0, fps_idx.clamp(0, norm_pw.shape[0] - 1)).reshape(
                        offsets.numel(), int(num_queries), 3
                    ).to(fallback.dtype)
            start = 0
            for bidx, end in enumerate(offsets.tolist()):
                pts = norm_pw[start:end]
                start = end
                pts = pts[torch.isfinite(pts).all(dim=-1)]
                if pts.numel() == 0:
                    out.append(fallback[bidx])
                    continue
                pts = _voxel_downsample_points(pts, self.infer_fps_voxel_size)
                if pts.numel() == 0:
                    out.append(fallback[bidx])
                    continue
                out.append(_farthest_point_sample(pts, num_queries).to(fallback.dtype))
            if out:
                return torch.stack(out, dim=0)
            return fallback

        if norm_pw.dim() != 3:
            return fallback
        for bidx in range(norm_pw.shape[0]):
            pts = norm_pw[bidx]
            pts = pts[torch.isfinite(pts).all(dim=-1)]
            if pts.numel() == 0:
                out.append(fallback[bidx])
                continue
            pts = _voxel_downsample_points(pts, self.infer_fps_voxel_size)
            if pts.numel() == 0:
                out.append(fallback[bidx])
                continue
            out.append(_farthest_point_sample(pts, num_queries).to(fallback.dtype))
        return torch.stack(out, dim=0)

    def _resolve_da3_levels(
        self,
        da3_feature_map: Optional[torch.Tensor],
        da3_feature_maps: Optional[Sequence[torch.Tensor]],
        image: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
        camtoworlds: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        if self.online_da3:
            if self._online_da3_extractor is None:
                raise RuntimeError("online_da3=True but extractor is not initialized.")
            if image is None or Ks is None or camtoworlds is None:
                raise KeyError("online_da3 requires image, Ks, camtoworlds.")
            _, da3_levels = self._online_da3_extractor(image, Ks, camtoworlds)
            return da3_levels
        if da3_feature_maps is not None:
            if len(da3_feature_maps) != self.num_da3_levels:
                raise ValueError(
                    f"da3_feature_maps must contain {self.num_da3_levels} levels, got {len(da3_feature_maps)}"
                )
            return list(da3_feature_maps)
        if da3_feature_map is None:
            raise KeyError("Need da3_feature_maps (list of tensors) or da3_feature_map (single tensor).")
        warnings.warn(
            "SuperpointVAE2C: only single-layer da3_feature_map provided; replicating it "
            f"{self.num_da3_levels}x as a Phase-1 placeholder. Update preprocessing to cache real "
            "multi-layer DA3 features."
        )
        return [da3_feature_map] * self.num_da3_levels

    def _build_encoder_tokens(
        self,
        da3_levels: List[torch.Tensor],
        Ks: torch.Tensor,
        camtoworlds: torch.Tensor,
        image_hw: tuple[int, int],
    ) -> tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        ref = da3_levels[0]
        b, v, h_da3, w_da3, _ = ref.shape
        flat = []
        for lvl in da3_levels:
            if lvl.shape[:4] != ref.shape[:4]:
                raise ValueError(
                    "All DA3 levels must share [B,V,H,W]; got " f"{lvl.shape} vs {ref.shape}"
                )
            x = torch.nan_to_num(lvl, nan=0.0, posinf=0.0, neginf=0.0)
            flat.append(x.reshape(b, v * h_da3 * w_da3, x.shape[-1]))

        src_h, src_w = image_hw
        Ks_lat = scale_intrinsics(Ks, src_hw=(src_h, src_w), dst_hw=(h_da3, w_da3))
        plucker = plucker_rays(camtoworlds, Ks_lat, h_da3, w_da3).reshape(b, v * h_da3 * w_da3, 6)
        view_ids = torch.arange(v, device=ref.device, dtype=torch.long).view(1, v, 1, 1)
        view_ids = view_ids.expand(b, v, h_da3, w_da3).reshape(b, v * h_da3 * w_da3)
        return flat, plucker, view_ids

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
    ) -> Dict[str, torch.Tensor]:
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
                image,
                1,
                encoder_view_idx[:, :, None, None, None].expand(b_img, max_enc_views, h_img, w_img, image.shape[-1]),
            )
            Ks_encoder = torch.gather(
                Ks,
                1,
                encoder_view_idx[:, :, None, None].expand(b_img, max_enc_views, Ks.shape[-2], Ks.shape[-1]),
            )
            camtoworlds_encoder = torch.gather(
                camtoworlds,
                1,
                encoder_view_idx[:, :, None, None].expand(b_img, max_enc_views, camtoworlds.shape[-2], camtoworlds.shape[-1]),
            )
            if valid_mask is not None:
                valid_mask_encoder = torch.gather(
                    valid_mask,
                    1,
                    encoder_view_idx[:, :, None, None].expand(b_img, max_enc_views, valid_mask.shape[-2], valid_mask.shape[-1]),
                )
            if norm_depth is not None:
                norm_depth_encoder = torch.gather(
                    norm_depth,
                    1,
                    encoder_view_idx[:, :, None, None].expand(b_img, max_enc_views, norm_depth.shape[-2], norm_depth.shape[-1]),
                )
        sp_xyz = torch.nan_to_num(sp_xyz, nan=0.0, posinf=0.0, neginf=0.0)

        if (self.use_fps_spxyz or ((not self.training) and self.infer_use_fps_spxyz)) and norm_pw is not None:
            sp_xyz = self._infer_sp_xyz_from_point_cloud(norm_pw, norm_pw_offsets, sp_xyz.shape[1], sp_xyz)
        sp_xyz = self._maybe_jitter_sp_xyz(sp_xyz)
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

        # All DA3 levels share the same (Hf, Wf) under the current setup; compute the
        # projection grid + validity once and sample each level with it.
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
        if self.debug_nan_checks:
            _debug_check_finite("forward.sp_xyz", sp_xyz)
            _debug_check_finite("forward.Ks_encoder", Ks_encoder)
            _debug_check_finite("forward.camtoworlds_encoder", camtoworlds_encoder)
            _debug_check_finite("forward.pool_grid.initial", pool_grid)
            for lvl_idx, lvl in enumerate(da3_levels):
                _debug_check_finite(f"forward.da3_level{lvl_idx}", lvl)
        if sp_valid_mask is not None:
            pool_valid = pool_valid & sp_valid_mask[:, None, :]
        pool_valid_for_init = _select_query_topk_views_by_depth(
            sp_xyz,
            Ks_encoder,
            camtoworlds_encoder,
            norm_depth_encoder,
            pool_valid,
            (h_img, w_img),
            self.encoder_query_init_topk,
            valid_mask=valid_mask_encoder,
        )
        attn_schedule = self.encoder_query_attn_topk_schedule
        if not attn_schedule:
            attn_schedule = (self.encoder_query_attn_topk,)
        pool_valid_layers = [
            _select_query_topk_views_by_depth(
                sp_xyz,
                Ks_encoder,
                camtoworlds_encoder,
                norm_depth_encoder,
                pool_valid,
                (h_img, w_img),
                int(attn_schedule[min(layer_idx, len(attn_schedule) - 1)]),
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
                            sp_xyz,
                            Ks_encoder,
                            camtoworlds_encoder,
                            norm_depth_encoder,
                            pool_valid,
                            (h_img, w_img),
                            topk,
                            valid_mask=valid_mask_encoder,
                        )
                    )
        pool_valid_for_attn = pool_valid_layers[-1] if pool_valid_layers else pool_valid
        visible_query_mask, visible_query_ratio = _visible_query_mask_from_projection(
            sp_xyz,
            Ks_encoder,
            camtoworlds_encoder,
            (h_img, w_img),
            valid_mask=valid_mask_encoder,
        )
        pooled_feats = [
            _projective_pool_sample(lvl, pool_grid, pool_valid_for_init) for lvl in da3_levels
        ]
        if self.debug_nan_checks:
            for lvl_idx, pooled in enumerate(pooled_feats):
                _debug_check_finite(f"forward.pooled_feat{lvl_idx}", pooled)
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
        if self.debug_nan_checks:
            _debug_check_finite("forward.z_sp", z_sp)
            _debug_check_finite("forward.sp_xyz_refined", sp_xyz_refined)
            for key, value in anchor_aux.items():
                _debug_check_finite(f"forward.{key}", value)
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
        if self.debug_nan_checks:
            _debug_check_finite("forward.pred_geometry_radius", pred_geometry_radius)
            _debug_check_finite("forward.pred_geometry_eigvals", pred_geometry_eigvals)
            _debug_check_finite("forward.pred_geometry_normal", pred_geometry_normal)
            _debug_check_finite("forward.pred_geometry_scalar", pred_geometry_scalar)
            _debug_check_finite("forward.pred_sp_vae_texture", pred_sp_vae_texture)
            _debug_check_finite("forward.sp_field", sp_field)
            _debug_check_finite("forward.pred_sp_field_vae", pred_sp_field_vae)

        decode_out = self.decoder(
            sp_feat=z_sp,
            camtoworlds=camtoworlds_encoder,
            Ks=Ks_encoder,
            image=image_encoder,
            sp_xyz=sp_xyz_refined,
            valid_mask=valid_mask_encoder,
            sp_valid_mask=sp_valid_mask,
            decode_rgb=decode_rgb,
        )
        if self.debug_nan_checks:
            for key in ("rendered_rgb", "rendered_depths", "rgb_grid", "depth_grid"):
                _debug_check_finite(f"forward.decode_out.{key}", decode_out.get(key, None))
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

    @staticmethod
    def _gather_rgb_pixel_targets(
        image: torch.Tensor,
        idx_rgb_lat: torch.Tensor,
        h_lat: int,
        w_lat: int,
        view_idx_rgb: Optional[torch.Tensor],
    ) -> torch.Tensor:
        b, v, h, w, c = image.shape
        gt = image.permute(0, 1, 4, 2, 3).reshape(b * v, c, h, w)
        gt_lat = F.interpolate(gt, size=(h_lat, w_lat), mode="bilinear", align_corners=False)
        gt_lat = gt_lat.reshape(b, v, c, h_lat, w_lat).permute(0, 1, 3, 4, 2).contiguous()
        if view_idx_rgb is not None:
            v_rgb = view_idx_rgb.shape[1]
            view_idx_e = view_idx_rgb.long()[..., None, None, None].expand(b, v_rgb, h_lat, w_lat, c)
            gt_lat = torch.gather(gt_lat, 1, view_idx_e)
            v = v_rgb
        gt_flat = gt_lat.reshape(b * v, h_lat * w_lat, c)
        idx = idx_rgb_lat.long().clamp(0, h_lat * w_lat - 1)
        return torch.gather(gt_flat, 1, idx.unsqueeze(-1).expand(-1, -1, c))

    @staticmethod
    def _gather_rendered_rgb_sampled(
        rgb_feat_grid: torch.Tensor,
        idx_rgb_lat: torch.Tensor,
    ) -> torch.Tensor:
        # rgb_feat_grid: [B, V_rgb, H_lat, W_lat, C]
        b, v_rgb, h_lat, w_lat, c = rgb_feat_grid.shape
        flat = rgb_feat_grid.reshape(b * v_rgb, h_lat * w_lat, c)
        idx = idx_rgb_lat.long().clamp(0, h_lat * w_lat - 1)
        return torch.gather(flat, 1, idx.unsqueeze(-1).expand(-1, -1, c))

    @staticmethod
    def _gather_selected_gt_rgb_full(
        image: torch.Tensor,
        view_idx_rgb: Optional[torch.Tensor],
        v_rgb: int,
    ) -> torch.Tensor:
        b, v, h, w, c = image.shape
        if view_idx_rgb is None:
            return image[:, :v_rgb]
        view_idx_e = view_idx_rgb.long()[..., None, None, None].expand(b, v_rgb, h, w, c)
        return torch.gather(image, 1, view_idx_e)

    def _pixel_cls_ce_for_feature_map(
        self,
        query_embed: torch.Tensor,
        feature_map: torch.Tensor,
        batch: dict,
        device: torch.device,
    ) -> torch.Tensor:
        sp_id_maps = batch.get("sp_id_maps", None)
        sp_mask_valid = batch.get("sp_mask_valid", None)
        sp_valid = batch.get("superpoint_valid_mask", None)
        if sp_id_maps is None or sp_mask_valid is None or sp_valid is None:
            return query_embed.new_zeros(())

        b, n, c = query_embed.shape
        _, v, hf, wf, cf = feature_map.shape
        if c != cf:
            raise ValueError(f"query dim {c} must match feature map dim {cf}")
        labels = sp_id_maps.long().to(device)
        valid = sp_mask_valid.bool().to(device)
        if labels.shape[-2:] != (hf, wf):
            labels_f = labels.reshape(b * v, 1, labels.shape[-2], labels.shape[-1]).float()
            labels = F.interpolate(labels_f, size=(hf, wf), mode="nearest").reshape(b, v, hf, wf).long()
            valid_f = valid.reshape(b * v, 1, valid.shape[-2], valid.shape[-1]).float()
            valid = F.interpolate(valid_f, size=(hf, wf), mode="nearest").reshape(b, v, hf, wf).bool()
        valid = valid & (labels >= 0) & (labels < n)

        q = F.normalize(query_embed, dim=-1)
        pix_all = F.normalize(feature_map.to(query_embed.dtype), dim=-1)
        sp_valid = sp_valid.bool().to(device)
        losses = []
        max_samples = max(int(self.pixel_cls_num_samples), 1)
        temp = max(float(self.pixel_cls_temperature), 1e-6)
        for bi in range(b):
            for vi in range(v):
                valid_idx = torch.nonzero(valid[bi, vi].reshape(-1), as_tuple=False).squeeze(-1)
                if valid_idx.numel() == 0:
                    continue
                if valid_idx.numel() > max_samples:
                    perm = torch.randperm(valid_idx.numel(), device=device)[:max_samples]
                    valid_idx = valid_idx.index_select(0, perm)
                pix = pix_all[bi, vi].reshape(hf * wf, -1).index_select(0, valid_idx)
                target = labels[bi, vi].reshape(-1).index_select(0, valid_idx)
                ok = sp_valid[bi].index_select(0, target)
                if not ok.any():
                    continue
                pix = pix[ok]
                target = target[ok]
                logits = torch.matmul(pix, q[bi].transpose(0, 1)) / temp
                logits = logits.masked_fill(~sp_valid[bi].view(1, n), -1e4)
                losses.append(F.cross_entropy(logits.float(), target))
        if not losses:
            return query_embed.new_zeros(())
        return torch.stack(losses).mean()

    def _compute_pixel_cls_loss(self, result_dict: dict, batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_sp = result_dict["free_latent_sp"]
        zero = z_sp.new_zeros(())
        if self.pixel_cls_weight <= 0:
            return zero, zero, zero

        da3_map = result_dict.get("da3_feature_map_unified", result_dict.get("da3_last_feature_map", None))
        if da3_map is None:
            return zero, zero, zero
        if da3_map.shape[-1] != self.da3_dim:
            raise ValueError(f"da3_feature_map_unified last dim {da3_map.shape[-1]} != {self.da3_dim}")
        q_da3 = self.query_da3_mask_proj(z_sp)
        da3_loss = self._pixel_cls_ce_for_feature_map(q_da3, da3_map, batch, device)
        return da3_loss, da3_loss, zero

    @staticmethod
    def _gather_feature_map_at_fullres_idx(feature_map: torch.Tensor, idx_full: torch.Tensor, full_hw: tuple[int, int]) -> torch.Tensor:
        bv, ps = idx_full.shape
        b, v, hf, wf, c = feature_map.shape
        if bv != b * v:
            raise ValueError(f"idx batch {bv} does not match feature map B*V {b*v}")
        h, w = full_hw
        y = idx_full // w
        x = idx_full % w
        yf = torch.round(y.float() * float(max(hf - 1, 0)) / float(max(h - 1, 1))).long().clamp(0, hf - 1)
        xf = torch.round(x.float() * float(max(wf - 1, 0)) / float(max(w - 1, 1))).long().clamp(0, wf - 1)
        idx_feat = yf * wf + xf
        flat = feature_map.reshape(b * v, hf * wf, c)
        return torch.gather(flat, 1, idx_feat.unsqueeze(-1).expand(-1, -1, c))

    def _compute_depth_da3_distill_loss(self, result_dict: dict, batch: dict) -> torch.Tensor:
        depth_feat = result_dict.get("depth_feat_sampled", None)
        da3_map = result_dict.get("da3_last_feature_map", None)
        idx = result_dict.get("sampled_linear_idx_depth", None)
        if depth_feat is None or da3_map is None or idx is None or self.depth_da3_distill_weight <= 0:
            return result_dict["free_latent_sp"].new_zeros(())
        _, v_all, h, w, _ = batch["image"].shape
        view_idx_depth = result_dict.get("selected_view_idx_depth_local", result_dict.get("selected_view_idx_depth", None))
        if view_idx_depth is not None:
            view_idx_depth = view_idx_depth.long()
            b, v_depth = view_idx_depth.shape
            b_all, _, hf, wf, c = da3_map.shape
            if b_all != b:
                raise ValueError(f"DA3 batch {b_all} does not match selected depth batch {b}")
            gather_idx = view_idx_depth[:, :, None, None, None].expand(b, v_depth, hf, wf, c)
            da3_map = torch.gather(da3_map, 1, gather_idx)
        else:
            v_depth = da3_map.shape[1]
        grid_hw = result_dict.get("depth_grid_hw", None)
        if grid_hw is not None:
            hq = int(grid_hw[0].item()) if torch.is_tensor(grid_hw) else int(grid_hw[0])
            wq = int(grid_hw[1].item()) if torch.is_tensor(grid_hw) else int(grid_hw[1])
            ys = idx.long() // max(wq, 1)
            xs = idx.long() % max(wq, 1)
            full_y = ((ys.float() + 0.5) * (float(h) / float(max(hq, 1)))).floor().long().clamp(0, h - 1)
            full_x = ((xs.float() + 0.5) * (float(w) / float(max(wq, 1)))).floor().long().clamp(0, w - 1)
            idx = full_y * w + full_x
        pred = self.depth_da3_distill_proj(depth_feat)
        target = self._gather_feature_map_at_fullres_idx(da3_map.detach(), idx.long(), (h, w)).to(pred.dtype)
        pred = F.normalize(pred, dim=-1, eps=1e-6)
        target = F.normalize(target, dim=-1, eps=1e-6)
        return (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()

    def _compute_depth_patch_smooth_loss(self, result_dict: dict, batch: dict) -> torch.Tensor:
        patch_idx = result_dict.get("sampled_patch_idx_depth", None)
        patch_size_t = result_dict.get("sampled_depth_patch_size", None)
        delta = result_dict.get("depth_delta_sampled", None)
        if patch_idx is None or delta is None or self.depth_smooth_weight <= 0:
            return result_dict["free_latent_sp"].new_zeros(())
        patch_size = int(patch_size_t.item()) if torch.is_tensor(patch_size_t) else int(patch_size_t or 1)
        if patch_size <= 1 or patch_idx.numel() == 0:
            return delta.new_zeros(())
        bv, num_patches, patch_area = patch_idx.shape
        if patch_area != patch_size * patch_size:
            raise ValueError(f"patch_area={patch_area} inconsistent with patch_size={patch_size}")
        delta_patch = delta.reshape(bv, num_patches, patch_area).reshape(bv, num_patches, patch_size, patch_size)
        dx = delta_patch[:, :, :, 1:] - delta_patch[:, :, :, :-1]
        dy = delta_patch[:, :, 1:, :] - delta_patch[:, :, :-1, :]

        _, _, h, w, _ = batch["image"].shape
        edge_source = None
        if self.depth_smooth_use_da3_edge and result_dict.get("da3_last_feature_map", None) is not None:
            edge_source = self._gather_feature_map_at_fullres_idx(
                result_dict["da3_last_feature_map"].detach(), patch_idx.reshape(bv, -1), (h, w)
            )
            edge_source = F.normalize(edge_source, dim=-1, eps=1e-6)
        else:
            image_flat = batch["image"].reshape(bv, h * w, 3)
            edge_source = torch.gather(
                image_flat,
                1,
                patch_idx.reshape(bv, -1).unsqueeze(-1).expand(-1, -1, 3),
            ).to(delta.dtype)
        c = edge_source.shape[-1]
        edge_patch = edge_source.reshape(bv, num_patches, patch_size, patch_size, c)
        if self.depth_smooth_use_da3_edge and result_dict.get("da3_last_feature_map", None) is not None:
            edge_x = 1.0 - (edge_patch[:, :, :, 1:] * edge_patch[:, :, :, :-1]).sum(dim=-1)
            edge_y = 1.0 - (edge_patch[:, :, 1:, :] * edge_patch[:, :, :-1, :]).sum(dim=-1)
        else:
            edge_x = (edge_patch[:, :, :, 1:] - edge_patch[:, :, :, :-1]).abs().mean(dim=-1)
            edge_y = (edge_patch[:, :, 1:, :] - edge_patch[:, :, :-1, :]).abs().mean(dim=-1)
        wx = torch.exp(-self.depth_smooth_edge_alpha * edge_x.float()).to(delta.dtype)
        wy = torch.exp(-self.depth_smooth_edge_alpha * edge_y.float()).to(delta.dtype)
        return (dx.abs() * wx).mean() + (dy.abs() * wy).mean()

    @staticmethod
    def _compute_dense_depth_grad_loss(
        pred_depth: torch.Tensor,
        gt_depth: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        if pred_depth.numel() == 0:
            return pred_depth.new_zeros(())
        pred_depth = pred_depth.float()
        gt_depth = gt_depth.to(device=pred_depth.device, dtype=pred_depth.dtype)
        valid = valid.to(device=pred_depth.device).bool()

        pred_dx = pred_depth[..., :, 1:] - pred_depth[..., :, :-1]
        gt_dx = gt_depth[..., :, 1:] - gt_depth[..., :, :-1]
        valid_x = valid[..., :, 1:] & valid[..., :, :-1]

        pred_dy = pred_depth[..., 1:, :] - pred_depth[..., :-1, :]
        gt_dy = gt_depth[..., 1:, :] - gt_depth[..., :-1, :]
        valid_y = valid[..., 1:, :] & valid[..., :-1, :]

        losses = []
        if valid_x.any():
            losses.append(_charbonnier(pred_dx[valid_x] - gt_dx[valid_x]))
        if valid_y.any():
            losses.append(_charbonnier(pred_dy[valid_y] - gt_dy[valid_y]))
        if not losses:
            return pred_depth.new_zeros(())
        return torch.stack(losses).mean()

    def _compute_coarse_pull_loss(self, z_sp: torch.Tensor, batch: dict, device: torch.device) -> torch.Tensor:
        if self.coarse_pull_weight <= 0:
            return z_sp.new_zeros(())
        coarse = batch.get("coarse_sp_group", None)
        sp_valid = batch.get("superpoint_valid_mask", None)
        if coarse is None or sp_valid is None:
            return z_sp.new_zeros(())
        coarse = coarse.long().to(device)
        sp_valid = sp_valid.bool().to(device)
        losses = []
        for bi in range(z_sp.shape[0]):
            valid = sp_valid[bi] & (coarse[bi] >= 0)
            if not valid.any():
                continue
            gids = coarse[bi, valid]
            z = F.normalize(z_sp[bi, valid], dim=-1)
            for gid in gids.unique():
                m = gids == gid
                if int(m.sum().item()) <= 1:
                    continue
                center = F.normalize(z[m].mean(dim=0, keepdim=True).detach(), dim=-1)
                losses.append(1.0 - F.cosine_similarity(z[m], center, dim=-1).mean())
        if not losses:
            return z_sp.new_zeros(())
        return torch.stack(losses).mean()

    def _compute_geometry_loss(
        self,
        result_dict: dict,
        batch: dict,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_sp = result_dict["free_latent_sp"]
        zero = z_sp.new_zeros(())
        sp_valid = batch.get("superpoint_valid_mask", None)
        if self.use_sp_field_distill and result_dict.get("sp_field_valid_mask", None) is not None:
            sp_valid = batch.get("source_superpoint_valid_mask", sp_valid)
        if self.geometry_loss_weight <= 0:
            visible = result_dict.get("visible_query_mask", None)
            if self.use_sp_field_distill and result_dict.get("sp_field_valid_mask", None) is not None:
                visible = result_dict.get("sp_field_valid_mask", None)
            if sp_valid is not None and visible is not None:
                sp_valid_b = sp_valid.bool().to(device)
                ratio = (
                    (visible.bool().to(device) & sp_valid_b).float().sum()
                    / sp_valid_b.float().sum().clamp_min(1.0)
                ).to(z_sp.dtype)
            else:
                ratio = zero
            return zero, zero, zero, zero, zero, ratio.detach()
        geom_valid = batch.get("sp_geometry_valid_mask", None)
        use_field = self.use_sp_field_distill and result_dict.get("sp_field_valid_mask", None) is not None
        visible = result_dict.get("sp_field_valid_mask", None) if use_field else result_dict.get("visible_query_mask", None)
        if use_field:
            sp_valid = batch.get("source_superpoint_valid_mask", sp_valid)
        if sp_valid is None or geom_valid is None or visible is None:
            return zero, zero, zero, zero, zero, zero

        mask = sp_valid.bool().to(device) & geom_valid.bool().to(device) & visible.bool().to(device)
        visible_ratio = (
            mask.float().sum() / sp_valid.bool().to(device).float().sum().clamp_min(1.0)
        ).to(z_sp.dtype)
        if not mask.any():
            return zero, zero, zero, zero, zero, visible_ratio.detach()

        losses = []
        radius_loss = zero
        if "sp_radius_p90_norm" in batch:
            target_radius = batch["sp_radius_p90_norm"].to(device=device, dtype=z_sp.dtype).unsqueeze(-1)
            m = mask & torch.isfinite(target_radius.squeeze(-1))
            if m.any():
                pred_radius = result_dict["pred_sp_field_geom_radius"] if use_field else result_dict["pred_geometry_radius"]
                radius_loss = F.smooth_l1_loss(pred_radius[m], target_radius[m])
                losses.append(self.geometry_radius_weight * radius_loss)

        eigvals_loss = zero
        if "sp_cov_eigvals_norm" in batch:
            target_eig = batch["sp_cov_eigvals_norm"].to(device=device, dtype=z_sp.dtype).clamp_min(0.0)
            m = mask & torch.isfinite(target_eig).all(dim=-1)
            if m.any():
                pred_eig = result_dict["pred_sp_field_geom_eigvals"] if use_field else result_dict["pred_geometry_eigvals"]
                eigvals_loss = F.smooth_l1_loss(pred_eig[m], target_eig[m])
                losses.append(self.geometry_eigvals_weight * eigvals_loss)

        normal_loss = zero
        if "sp_normal" in batch:
            target_normal = F.normalize(batch["sp_normal"].to(device=device, dtype=z_sp.dtype), dim=-1, eps=1e-6)
            pred_normal = result_dict["pred_sp_field_geom_normal"] if use_field else result_dict["pred_geometry_normal"]
            m = mask & torch.isfinite(target_normal).all(dim=-1)
            if m.any():
                cos = (pred_normal[m] * target_normal[m]).sum(dim=-1).abs()
                normal_loss = (1.0 - cos).mean()
                losses.append(self.geometry_normal_weight * normal_loss)

        scalar_loss = zero
        scalar_fields = [
            "sp_linearity",
            "sp_planarity",
            "sp_scattering",
            "sp_anisotropy",
            "sp_curvature",
            "sp_omnivariance_norm",
        ]
        if all(name in batch for name in scalar_fields):
            target_scalar = torch.stack(
                [batch[name].to(device=device, dtype=z_sp.dtype) for name in scalar_fields],
                dim=-1,
            )
            m = mask & torch.isfinite(target_scalar).all(dim=-1)
            if m.any():
                pred_scalar = result_dict["pred_sp_field_geom_scalar"] if use_field else result_dict["pred_geometry_scalar"]
                scalar_loss = F.smooth_l1_loss(pred_scalar[m], target_scalar[m])
                losses.append(self.geometry_scalar_weight * scalar_loss)

        if not losses:
            return zero, radius_loss, eigvals_loss, normal_loss, scalar_loss, visible_ratio.detach()
        return torch.stack(losses).sum(), radius_loss, eigvals_loss, normal_loss, scalar_loss, visible_ratio.detach()

    def _aggregate_sp_vae_from_maps(
        self,
        batch: dict,
        device: torch.device,
        num_queries: int,
        dtype: torch.dtype,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        vae_map = batch.get("vae_feature_map", None)
        sp_id_maps = batch.get("sp_id_maps", None)
        sp_mask_valid = batch.get("sp_mask_valid", None)
        if vae_map is None or sp_id_maps is None or sp_mask_valid is None:
            return None, None

        vae_map = vae_map.to(device=device, dtype=dtype)
        labels = sp_id_maps.long().to(device)
        valid = sp_mask_valid.bool().to(device)
        b, v, hv, wv, c = vae_map.shape
        if c != self.vae_latent_channels:
            raise ValueError(f"vae_feature_map last dim {c} != {self.vae_latent_channels}")
        if labels.shape[:2] != (b, v):
            raise ValueError(f"sp_id_maps shape {tuple(labels.shape)} incompatible with vae_feature_map {tuple(vae_map.shape)}")
        if labels.shape[-2:] != (hv, wv):
            labels_f = labels.reshape(b * v, 1, labels.shape[-2], labels.shape[-1]).float()
            valid_f = valid.reshape(b * v, 1, valid.shape[-2], valid.shape[-1]).float()
            labels = F.interpolate(labels_f, size=(hv, wv), mode="nearest").reshape(b, v, hv, wv).long()
            valid = F.interpolate(valid_f, size=(hv, wv), mode="nearest").reshape(b, v, hv, wv).bool()

        labels_flat = labels.reshape(b, -1)
        valid_flat = valid.reshape(b, -1) & (labels_flat >= 0) & (labels_flat < num_queries)
        feat_flat = torch.nan_to_num(vae_map, nan=0.0, posinf=0.0, neginf=0.0).reshape(b, -1, c)
        safe_labels = labels_flat.clamp(0, max(num_queries - 1, 0))

        teacher = feat_flat.new_zeros((b, num_queries, c))
        counts = feat_flat.new_zeros((b, num_queries, 1))
        weighted_feat = feat_flat * valid_flat.unsqueeze(-1).to(feat_flat.dtype)
        teacher.scatter_add_(1, safe_labels.unsqueeze(-1).expand(-1, -1, c), weighted_feat)
        counts.scatter_add_(1, safe_labels.unsqueeze(-1), valid_flat.unsqueeze(-1).to(feat_flat.dtype))
        teacher = teacher / counts.clamp_min(1.0)
        return teacher, counts.squeeze(-1) > 0

    def _compute_sp_vae_texture_loss(
        self,
        result_dict: dict,
        batch: dict,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        use_field = self.use_sp_field_distill and result_dict.get("pred_sp_field_vae", None) is not None
        pred = result_dict["pred_sp_field_vae"] if use_field else result_dict["pred_sp_vae_texture"]
        zero = pred.new_zeros(())
        if self.sp_vae_loss_weight <= 0 or (self.sp_vae_cos_weight <= 0 and self.sp_vae_mse_weight <= 0):
            return zero, zero, zero, zero

        b, n, c = pred.shape
        if c != self.vae_latent_channels:
            raise ValueError(f"pred_sp_vae_texture last dim {c} != {self.vae_latent_channels}")

        teacher = batch.get("sp_vae", None)
        teacher_mask = None
        if teacher is not None:
            teacher = teacher.to(device=device, dtype=pred.dtype)
            if teacher.shape[:2] != (b, n) or teacher.shape[-1] != c:
                raise ValueError(f"sp_vae shape {tuple(teacher.shape)} incompatible with pred {tuple(pred.shape)}")
            teacher = torch.nan_to_num(teacher, nan=0.0, posinf=0.0, neginf=0.0)
            teacher_mask = torch.isfinite(teacher).all(dim=-1) & (teacher.abs().sum(dim=-1) > 0)
        else:
            teacher, teacher_mask = self._aggregate_sp_vae_from_maps(batch, device, n, pred.dtype)
            if teacher is None or teacher_mask is None:
                return zero, zero, zero, zero

        sp_valid = batch.get("source_superpoint_valid_mask", None) if use_field else batch.get("superpoint_valid_mask", None)
        mask = teacher_mask.bool()
        if sp_valid is not None:
            mask = mask & sp_valid.bool().to(device)
        visible = result_dict.get("sp_field_valid_mask", None) if use_field else result_dict.get("visible_query_mask", None)
        if visible is not None:
            mask = mask & visible.bool().to(device)
        if not mask.any():
            return zero, zero, zero, zero

        pred_sel = pred[mask]
        teacher_sel = teacher.detach()[mask]
        cos_loss = zero
        if self.sp_vae_cos_weight > 0:
            pred_n = F.normalize(pred_sel, dim=-1, eps=1e-6)
            teacher_n = F.normalize(teacher_sel, dim=-1, eps=1e-6)
            cos_loss = (1.0 - F.cosine_similarity(pred_n, teacher_n, dim=-1)).mean()
        mse_loss = zero
        if self.sp_vae_mse_weight > 0:
            mse_loss = F.mse_loss(pred_sel, teacher_sel)
        total = self.sp_vae_cos_weight * cos_loss + self.sp_vae_mse_weight * mse_loss
        valid_ratio = (mask.float().sum() / mask.numel()).to(pred.dtype)
        return total, cos_loss, mse_loss, valid_ratio.detach()

    @staticmethod
    def _gather_selected_sp_maps(
        sp_id_maps: torch.Tensor,
        sp_mask_valid: torch.Tensor,
        selected_view_idx: Optional[torch.Tensor],
        num_views: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = sp_id_maps.shape
        if selected_view_idx is None:
            return sp_id_maps[:, :num_views], sp_mask_valid[:, :num_views]
        view_idx = selected_view_idx.long()
        gather_idx = view_idx[:, :, None, None].expand(b, num_views, h, w)
        return torch.gather(sp_id_maps, 1, gather_idx), torch.gather(sp_mask_valid, 1, gather_idx)

    def _compute_decoder_attn_sp_loss(
        self,
        result_dict: dict,
        batch: dict,
        attn_name: str,
        idx_name: str,
        selected_view_name: str,
        num_queries: int,
    ) -> torch.Tensor:
        attn = result_dict.get(attn_name, None)
        attn_idx = result_dict.get(idx_name, None)
        sp_id_maps = batch.get("sp_id_maps", None)
        sp_mask_valid = batch.get("sp_mask_valid", None)
        sp_valid = batch.get("superpoint_valid_mask", None)
        z_sp = result_dict["free_latent_sp"]
        zero = z_sp.new_zeros(())
        if attn is None or attn_idx is None or sp_id_maps is None or sp_mask_valid is None or sp_valid is None:
            return zero

        b, v_sel, h_low, w_low, k = attn.shape
        if b != z_sp.shape[0] or num_queries != z_sp.shape[1]:
            return zero
        device = attn.device
        labels, valid = self._gather_selected_sp_maps(
            sp_id_maps.long().to(device),
            sp_mask_valid.bool().to(device),
            result_dict.get(selected_view_name, None),
            v_sel,
        )
        sp_valid = sp_valid.bool().to(device)
        _, _, h_full, w_full = labels.shape

        if h_full % h_low != 0 or w_full % w_low != 0:
            labels_f = labels.reshape(b * v_sel, 1, h_full, w_full).float()
            valid_f = valid.reshape(b * v_sel, 1, h_full, w_full).float()
            labels = F.interpolate(labels_f, size=(h_low, w_low), mode="nearest").reshape(b, v_sel, h_low, w_low).long()
            valid = F.interpolate(valid_f, size=(h_low, w_low), mode="nearest").reshape(b, v_sel, h_low, w_low).bool()
            patch_labels = labels.reshape(b * v_sel, h_low * w_low, 1)
            patch_valid = valid.reshape(b * v_sel, h_low * w_low, 1)
        else:
            sy = h_full // h_low
            sx = w_full // w_low
            labels_unfold = F.unfold(
                labels.reshape(b * v_sel, 1, h_full, w_full).float(),
                kernel_size=(sy, sx),
                stride=(sy, sx),
            )
            valid_unfold = F.unfold(
                valid.reshape(b * v_sel, 1, h_full, w_full).float(),
                kernel_size=(sy, sx),
                stride=(sy, sx),
            )
            patch_labels = labels_unfold.transpose(1, 2).long()
            patch_valid = valid_unfold.transpose(1, 2).bool()

        p = h_low * w_low
        if patch_labels.shape[1] != p:
            return zero
        attn_idx_flat = attn_idx.long().to(device).clamp(0, max(num_queries - 1, 0)).reshape(b * v_sel, p, k)
        attn_flat = attn.reshape(b * v_sel, p, k).float().clamp_min(1e-8)
        candidate_valid = torch.gather(
            sp_valid[:, None, :].expand(b, v_sel, num_queries).reshape(b * v_sel, num_queries),
            1,
            attn_idx_flat.reshape(b * v_sel, p * k),
        ).reshape(b * v_sel, p, k)

        label_ok = patch_valid & (patch_labels >= 0) & (patch_labels < num_queries)
        match = patch_labels[..., None] == attn_idx_flat[:, :, None, :]
        counts = (match & label_ok[..., None]).float().sum(dim=2)
        counts = counts * candidate_valid.float()
        denom = counts.sum(dim=-1, keepdim=True)
        valid_cell = denom.squeeze(-1) > 0
        if not valid_cell.any():
            return zero
        target = counts / denom.clamp_min(1.0)
        loss = -(target * attn_flat.log()).sum(dim=-1)
        return loss[valid_cell].mean().to(z_sp.dtype)

    def compute_losses(self, result_dict: dict, batch: dict) -> dict[str, torch.Tensor]:
        pred_depth_full = result_dict["rendered_depths"][..., 0]
        b, v, h, w = pred_depth_full.shape
        v_all = batch["image"].shape[1]

        gt_depth = batch["norm_depth"]
        if gt_depth.dim() == 3:
            gt_depth = _expand_to_views(gt_depth, v_all)
        valid = batch["valid_mask"].bool()
        if valid.dim() == 3:
            valid = _expand_to_views(valid, v_all)
        view_idx_depth = result_dict.get("selected_view_idx_depth", None)
        if view_idx_depth is not None:
            view_idx_depth = view_idx_depth.long()
            v_depth = view_idx_depth.shape[1]
            gather_depth = view_idx_depth[:, :, None, None].expand(b, v_depth, h, w)
            gt_depth = torch.gather(gt_depth, 1, gather_depth)
            valid = torch.gather(valid, 1, gather_depth)

        pred_depth_full = torch.nan_to_num(pred_depth_full, nan=0.0, posinf=1e3, neginf=-1e3)
        dep_diff = (pred_depth_full - gt_depth).abs()
        depth_l1 = dep_diff[valid].mean() if valid.any() else dep_diff.mean() * 0.0
        depth_grad = self._compute_dense_depth_grad_loss(pred_depth_full, gt_depth, valid)
        depth_loss = depth_l1 + self.depth_grad_weight * depth_grad

        view_idx_rgb = result_dict.get("selected_view_idx_rgb", None)
        if view_idx_rgb is not None:
            view_idx_rgb = view_idx_rgb.long()

        rendered_rgb = result_dict["rendered_rgb"]  # [B, V_rgb, H, W, 3], always populated
        b_, v_rgb_, h_, w_, c_ = rendered_rgb.shape
        pred_rgb_full = rendered_rgb.permute(0, 1, 4, 2, 3).reshape(b_ * v_rgb_, c_, h_, w_)

        gt_rgb_full = self._gather_selected_gt_rgb_full(batch["image"], view_idx_rgb, v_rgb_).to(rendered_rgb.dtype)
        rgb_full = _charbonnier(rendered_rgb - gt_rgb_full)
        gt_rgb_nchw = gt_rgb_full.permute(0, 1, 4, 2, 3).reshape(b_ * v_rgb_, c_, h_, w_)
        rgb_grad = _image_gradient_loss(pred_rgb_full, gt_rgb_nchw)

        vae_distill = rendered_rgb.new_zeros(())

        rgb_lpips = torch.zeros((), device=rendered_rgb.device, dtype=rendered_rgb.dtype)
        if self.lpips_loss_weight > 0 and self._lpips_model is not None:
            self._lpips_model.to(rendered_rgb.device)
            pred_full = pred_rgb_full * 2.0 - 1.0
            gt_full = gt_rgb_nchw * 2.0 - 1.0
            rgb_lpips = self._lpips_model(pred_full, gt_full.to(pred_full.dtype)).mean()

        rgb_loss = (
            self.rgb_fullres_weight * rgb_full
            + self.rgb_grad_weight * rgb_grad
            + self.lpips_loss_weight * rgb_lpips
        )

        latent_reg = result_dict["free_latent_sp"].pow(2).mean()
        pixel_cls_loss, pixel_cls_texture_loss, pixel_cls_geometry_loss = self._compute_pixel_cls_loss(
            result_dict, batch, rendered_rgb.device
        )
        coarse_pull_loss = self._compute_coarse_pull_loss(result_dict["free_latent_sp"], batch, rendered_rgb.device)
        (
            geometry_loss,
            geometry_radius_loss,
            geometry_eigvals_loss,
            geometry_normal_loss,
            geometry_scalar_loss,
            visible_query_ratio,
        ) = self._compute_geometry_loss(result_dict, batch, rendered_rgb.device)
        depth_da3_distill = self._compute_depth_da3_distill_loss(result_dict, batch)
        sp_vae_texture_loss, sp_vae_texture_cos, sp_vae_texture_mse, sp_vae_texture_valid_ratio = self._compute_sp_vae_texture_loss(
            result_dict, batch, rendered_rgb.device
        )
        anchor_offset_loss = rendered_rgb.new_zeros(())
        anchor_surface_loss = rendered_rgb.new_zeros(())
        anchor_offset_mean = rendered_rgb.new_zeros(())
        anchor_offset_p90 = rendered_rgb.new_zeros(())
        rgb_attn_sp = self._compute_decoder_attn_sp_loss(
            result_dict,
            batch,
            "rgb_attn",
            "rgb_attn_idx",
            "selected_view_idx_rgb",
            result_dict["free_latent_sp"].shape[1],
        )
        depth_attn_sp = self._compute_decoder_attn_sp_loss(
            result_dict,
            batch,
            "depth_attn",
            "depth_attn_idx",
            "selected_view_idx_depth",
            result_dict["free_latent_sp"].shape[1],
        )
        depth_smooth = rendered_rgb.new_zeros(())

        if self.enable_sp_feat_supervision and batch.get("sp_feat", None) is not None and batch["sp_feat"].shape[-1] > 0:
            gt_sp = batch["sp_feat"]
            gt_dino = gt_sp[..., : self.dino_dim]
            gt_da3 = gt_sp[..., self.dino_dim : self.dino_dim + self.da3_dim]
            gt_vae = gt_sp[..., self.dino_dim + self.da3_dim : self.dino_dim + self.da3_dim + self.vae_latent_channels]

            pred_dino = result_dict["pred_sp_dino"]
            pred_da3 = result_dict["pred_sp_da3"]
            pred_vae = result_dict["pred_sp_vae"]
            dino_norm_mse = _normalized_mse(pred_dino, gt_dino)
            da3_norm_mse = _normalized_mse(pred_da3, gt_da3)
            # VAE latents have a wide dynamic range; align direction via cosine instead of magnitude
            # so the loss is bounded in [0, 2] and doesn't swamp the other teacher terms.
            vae_teacher = 1.0 - F.cosine_similarity(pred_vae, gt_vae, dim=-1).mean()
            sp_feat_loss = (
                self.dino_cos_weight * dino_norm_mse
                + self.dino_mse_weight * dino_norm_mse
                + self.da3_teacher_weight * da3_norm_mse
                + self.vae_teacher_weight * vae_teacher
            )
        else:
            zero = rendered_rgb.new_zeros(())
            dino_norm_mse = zero
            da3_norm_mse = zero
            vae_teacher = zero
            sp_feat_loss = zero

        total = (
            self.rgb_loss_weight * rgb_loss
            + self.depth_loss_weight * depth_loss
            + self.latent_reg_weight * latent_reg
            + (self.sp_feat_loss_weight * sp_feat_loss if self.enable_sp_feat_supervision else 0.0)
            + self.pixel_cls_weight * pixel_cls_loss
            + self.coarse_pull_weight * coarse_pull_loss
            + (self.geometry_loss_weight * geometry_loss if self.geometry_loss_weight > 0 else 0.0)
            + self.depth_da3_distill_weight * depth_da3_distill
            + self.sp_vae_loss_weight * sp_vae_texture_loss
            + self.rgb_decoder_attn_sp_weight * rgb_attn_sp
            + self.depth_decoder_attn_sp_weight * depth_attn_sp
            + self.depth_smooth_weight * depth_smooth
        )
        losses = {
            "loss": total,
            "rgb_loss": rgb_loss.detach(),
            "vae_distill": vae_distill.detach(),
            "rgb_full": rgb_full.detach(),
            "rgb_grad": rgb_grad.detach(),
            "rgb_lpips": rgb_lpips.detach() if torch.is_tensor(rgb_lpips) else rgb_lpips,
            "depth_loss": depth_loss.detach(),
            "depth_l1": depth_l1.detach(),
            "depth_grad": depth_grad.detach(),
            "latent_reg": latent_reg.detach(),
            "sp_feat_loss": sp_feat_loss.detach(),
            "pixel_cls_loss": pixel_cls_loss.detach(),
            "pixel_cls_texture": pixel_cls_texture_loss.detach(),
            "pixel_cls_geometry": pixel_cls_geometry_loss.detach(),
            "pixel_cls_vae": pixel_cls_texture_loss.detach(),
            "pixel_cls_da3": pixel_cls_geometry_loss.detach(),
            "coarse_pull_loss": coarse_pull_loss.detach(),
            "geometry_loss": geometry_loss.detach(),
            "geometry_radius": geometry_radius_loss.detach(),
            "geometry_eigvals": geometry_eigvals_loss.detach(),
            "geometry_normal": geometry_normal_loss.detach(),
            "geometry_scalar": geometry_scalar_loss.detach(),
            "visible_query_ratio": visible_query_ratio.detach(),
            "depth_da3_distill": depth_da3_distill.detach(),
            "sp_vae_texture_loss": sp_vae_texture_loss.detach(),
            "sp_vae_texture_cos": sp_vae_texture_cos.detach(),
            "sp_vae_texture_mse": sp_vae_texture_mse.detach(),
            "sp_vae_texture_valid_ratio": sp_vae_texture_valid_ratio.detach(),
            "anchor_offset_loss": anchor_offset_loss.detach(),
            "anchor_surface_loss": anchor_surface_loss.detach(),
            "anchor_offset_mean": anchor_offset_mean.detach(),
            "anchor_offset_p90": anchor_offset_p90.detach(),
            "rgb_attn_sp": rgb_attn_sp.detach(),
            "depth_attn_sp": depth_attn_sp.detach(),
            "depth_smooth": depth_smooth.detach(),
            "dino_norm_mse": dino_norm_mse.detach(),
            "da3_norm_mse": da3_norm_mse.detach(),
            "vae_teacher": vae_teacher.detach(),
        }
        if self.debug_nan_checks:
            for key, value in losses.items():
                _debug_check_finite(f"losses.{key}", value)
        return losses

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
        )
        losses = self.compute_losses(result, batch)
        return result, losses
