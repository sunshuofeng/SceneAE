"""Superpoint Grid Decoder (v27).

Replaces C20/C20_v2's ``SuperpointDecoderPipeline`` with a
「稀疏 anchor → 逐视角稠密 token 网格 → 2D ViT 主干 → 卷积上采样」structure,
following SUPERPOINT_GRID_DECODER_DESIGN.md.  Key properties:

  * AnchorToGrid frontend: project global anchors into each view's camera
    frame, gather a local top-K neighbourhood per token, score them with a
    soft z-buffer prior + zero-init residual MLP, aggregate position-conditioned
    values, and provide a learned null-slot fallback for empty / occluded cells.
  * Shared 2D ViT trunk -> RGB branch + depth branch with a gradient gate
    (alpha) so a collapsing depth branch cannot steal RGB's learning rate.
  * Convolution-coupled two-stage PixelShuffle upsampling heads (no GAN).
  * Inverse-depth output parameterization with a per-view adaptive range
    derived from the anchor z distribution (available at generation time).

The decoder consumes the frozen encoder's latent ``z_sp`` (cached sidecar) and
anchor positions ``sp_xyz``; it never calls DA3 or the encoder.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _inverse_softplus(y: float) -> float:
    """softplus^{-1}(y) = log(e^y - 1); used to init softplus-constrained params."""
    y = max(float(y), 1e-6)
    # numerically stable: log(expm1(y)) for small y, y for large y
    return math.log(math.expm1(y)) if y < 20.0 else y


def build_2d_sincos(h: int, w: int, dim: int, temperature: float = 10000.0) -> torch.Tensor:
    """Standard MAE-style 2D sin-cos positional embedding. Returns [h*w, dim]."""
    if dim % 4 != 0:
        raise ValueError(f"2D sincos dim must be divisible by 4, got {dim}")
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    half = dim // 2
    omega = torch.arange(half // 2, dtype=torch.float32) / (half // 2)
    omega = 1.0 / (temperature ** omega)
    out_x = xx.flatten()[:, None] * omega[None, :]  # [h*w, half//2]
    out_y = yy.flatten()[:, None] * omega[None, :]
    pos = torch.cat(
        [out_x.sin(), out_x.cos(), out_y.sin(), out_y.cos()], dim=-1
    )  # [h*w, dim]
    return pos


class MLP(nn.Module):
    """Small MLP with GELU; optional zero-init of the final layer."""

    def __init__(self, in_dim: int, out_dim: int, hidden: Optional[int] = None,
                 n_layers: int = 2, zero_init_last: bool = False) -> None:
        super().__init__()
        hidden = int(hidden) if hidden is not None else max(in_dim, out_dim)
        layers = []
        d = in_dim
        for _ in range(max(n_layers - 1, 0)):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        last = nn.Linear(d, out_dim)
        if zero_init_last:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# AnchorToGrid frontend
# ---------------------------------------------------------------------------
class AnchorToGrid(nn.Module):
    """Project sparse anchors into each view and splat them onto a token grid.

    For every (batch, view) produces a dense token grid [BV, H*W, token_dim]
    plus a per-token depth prior and the local top-K context reused by the
    cross-attention layers.
    """

    def __init__(
        self,
        feat_dim: int = 64,
        token_dim: int = 792,
        patch: int = 14,
        k_init: int = 8,
        k_xattn: int = 64,
        n_groups: int = 4,
        desc_dim: int = 64,
        beta_init: float = 3.0,
        r_min: float = 0.5,
        null_bias_init: float = -2.2,
        pos_dim: int = 64,
    ) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.token_dim = int(token_dim)
        self.patch = int(patch)
        self.k_init = int(k_init)
        self.k_xattn = int(k_xattn)
        self.n_groups = int(n_groups)
        self.group_dim = self.token_dim // self.n_groups
        self.desc_dim = int(desc_dim)
        self.r_min = float(r_min)
        self.pos_dim = int(pos_dim)
        if self.token_dim % self.n_groups != 0:
            raise ValueError("token_dim must be divisible by n_groups")

        # geometric descriptor -> embedding
        self.desc_in = 9
        self.desc_mlp = MLP(self.desc_in, self.desc_dim, hidden=self.desc_dim)
        # feature projection used in scoring
        self.W_f = nn.Linear(self.feat_dim, self.desc_dim)
        # score residual (zero-init last) -> per-group logits
        self.score_mlp = MLP(2 * self.desc_dim, self.n_groups, hidden=self.desc_dim,
                             zero_init_last=True)
        # null-slot logit from absolute-scale stats -> per-group logit
        self.null_mlp = MLP(4, self.n_groups, hidden=32)
        nn.init.constant_(self.null_mlp.net[-1].bias, float(null_bias_init))
        # value projections (W_p zero-init -> position conditioning is "earned")
        self.W_v = nn.Linear(self.feat_dim, self.token_dim)
        self.W_p = MLP(self.desc_dim, self.token_dim, hidden=self.desc_dim,
                       zero_init_last=True)
        # learned "empty" token (small random, not zero, so it is distinguishable)
        self.empty_token = nn.Parameter(torch.randn(self.token_dim) * 0.02)
        # camera ray encoding
        self.ray_mlp = MLP(3, self.token_dim, hidden=self.token_dim)
        # soft z-buffer / locality priors (softplus-constrained, init enabled)
        self.beta_z_raw = nn.Parameter(torch.tensor(_inverse_softplus(beta_init)))
        self.beta_r_raw = nn.Parameter(torch.tensor(_inverse_softplus(beta_init)))

        self._pos_cache: Dict[tuple, torch.Tensor] = {}

    # -- fixed 2D sincos embedding (cached per grid size) --------------------
    def _pos_embed(self, h: int, w: int, device, dtype) -> torch.Tensor:
        key = (h, w)
        pos = self._pos_cache.get(key)
        if pos is None:
            pos = build_2d_sincos(h, w, self.token_dim)
            self._pos_cache[key] = pos
        return pos.to(device=device, dtype=dtype)

    def forward(
        self,
        sp_feat: torch.Tensor,       # [B, N, feat_dim]
        sp_xyz: torch.Tensor,        # [B, N, 3]
        camtoworlds: torch.Tensor,   # [B, V, 4, 4]
        Ks: torch.Tensor,            # [B, V, 3, 3]
        grid_hw: tuple,              # (H_lat, W_lat)
        sp_valid_mask: Optional[torch.Tensor] = None,  # [B, N]
    ) -> Dict[str, torch.Tensor]:
        B, N, _ = sp_feat.shape
        V = camtoworlds.shape[1]
        H_lat, W_lat = grid_hw
        T = H_lat * W_lat
        BV = B * V
        device = sp_feat.device
        f32 = torch.float32

        sp_feat_f = sp_feat.to(f32)
        sp_xyz_f = torch.nan_to_num(sp_xyz.to(f32), nan=0.0, posinf=0.0, neginf=0.0)
        # features are shared across views; expand to per-view for gathering
        sp_feat_bv = sp_feat_f[:, None].expand(B, V, N, self.feat_dim).reshape(BV, N, self.feat_dim)

        # ---- project anchors into each view's camera frame -----------------
        w2c = torch.linalg.inv(camtoworlds.to(f32)).reshape(BV, 4, 4)
        K = Ks.to(f32).reshape(BV, 3, 3)
        ones = torch.ones((B, N, 1), device=device, dtype=f32)
        sp_h = torch.cat([sp_xyz_f, ones], dim=-1)[:, None].expand(B, V, N, 4).reshape(BV, N, 4)
        cam_pts = torch.matmul(sp_h, w2c.transpose(1, 2))[..., :3]  # [BV,N,3]
        z = cam_pts[..., 2]
        zc = z.clamp_min(1e-4)
        u = K[:, None, 0, 0] * (cam_pts[..., 0] / zc) + K[:, None, 0, 2]
        v = K[:, None, 1, 1] * (cam_pts[..., 1] / zc) + K[:, None, 1, 2]
        # token-grid coords (token centre at integer)
        u_t = u / float(self.patch) - 0.5
        v_t = v / float(self.patch) - 0.5

        valid = z > 1e-4
        if sp_valid_mask is not None:
            valid = valid & sp_valid_mask.to(device=device).bool()[:, None].expand(B, V, N).reshape(BV, N)

        # ---- token centres --------------------------------------------------
        ii = torch.arange(H_lat, device=device, dtype=f32)
        jj = torch.arange(W_lat, device=device, dtype=f32)
        gi, gj = torch.meshgrid(ii, jj, indexing="ij")  # [H_lat, W_lat]
        tok_u = gj.reshape(-1)  # [T] token centre u (token coords)
        tok_v = gi.reshape(-1)  # [T]

        # ---- distance matrix + single topk(k_xattn) -------------------------
        du = u_t[:, None, :] - tok_u[None, :, None]  # [BV,T,N]
        dv = v_t[:, None, :] - tok_v[None, :, None]
        r = torch.sqrt(du * du + dv * dv + 1e-12)
        r = torch.where(valid[:, None, :], r, torch.full_like(r, 1e6))
        k = min(self.k_xattn, N)
        r_top, idx = torch.topk(r, k, dim=-1, largest=False)  # [BV,T,k] ascending

        def gather_nd(x: torch.Tensor) -> torch.Tensor:
            # x: [BV, N, *] -> gather idx [BV,T,k] along dim=1 -> [BV,T,k,*]
            # (gathers along N without materialising a [BV,T,N,*] tensor)
            tail = x.shape[2:]
            x_flat = x.reshape(BV, N, -1)                     # [BV,N,Cprod]
            idx_flat = idx.reshape(BV, T * k, 1).expand(BV, T * k, x_flat.shape[-1])
            g = torch.gather(x_flat, 1, idx_flat)
            return g.reshape(BV, T, k, *tail)

        z_k = gather_nd(z.unsqueeze(-1)).squeeze(-1)          # [BV,T,k]
        valid_k = gather_nd(valid.unsqueeze(-1)).squeeze(-1)  # [BV,T,k]
        feat_k = gather_nd(sp_feat_bv)                         # [BV,T,k,feat]
        campts_k = gather_nd(cam_pts)                          # [BV,T,k,3]
        u_k = gather_nd(u_t.unsqueeze(-1)).squeeze(-1)        # [BV,T,k]
        v_k = gather_nd(v_t.unsqueeze(-1)).squeeze(-1)
        du_k = u_k - tok_u[None, :, None]                     # offset from token centre
        dv_k = v_k - tok_v[None, :, None]
        # r_top already holds the distances for the gathered anchors.

        # ---- init neighbourhood = first k_init ----------------------------
        ki = min(self.k_init, k)
        r_i = r_top[..., :ki]
        z_i = z_k[..., :ki]
        valid_i = valid_k[..., :ki]
        feat_i = feat_k[..., :ki, :]
        du_i = du_k[..., :ki]
        dv_i = dv_k[..., :ki]
        eps = 1e-6

        # group statistics over the init neighbourhood
        rms_r = torch.sqrt((r_i * r_i).mean(dim=-1) + 1e-12)
        R_t = rms_r.clamp_min(self.r_min)                     # [BV,T]
        z_min = z_i.min(dim=-1).values
        z_max = z_i.max(dim=-1).values
        z_span = (z_max - z_min)                              # [BV,T]
        dz = (z_i - z_min.unsqueeze(-1)) / (z_span.unsqueeze(-1) + eps)  # [BV,T,ki]
        r_mean = r_i.mean(dim=-1)                             # [BV,T]
        K_valid = valid_i.float().sum(dim=-1)                 # [BV,T]
        kf = float(ki)

        # ---- geometric descriptor (per pair) --------------------------------
        desc = torch.stack(
            [
                du_i / R_t.unsqueeze(-1),
                dv_i / R_t.unsqueeze(-1),
                dz,
                torch.log(z_i.clamp_min(eps)),
                valid_i.float(),
                torch.log(r_mean.unsqueeze(-1) + eps).expand(BV, T, ki),
                torch.log(R_t.unsqueeze(-1)).expand(BV, T, ki),
                torch.log(z_span.unsqueeze(-1) + eps).expand(BV, T, ki),
                (K_valid / kf).unsqueeze(-1).expand(BV, T, ki),
            ],
            dim=-1,
        )  # [BV,T,ki,9]
        g_emb = self.desc_mlp(desc)                           # [BV,T,ki,desc_dim]

        # ---- scoring: soft z-buffer prior + zero-init residual -------------
        beta_z = F.softplus(self.beta_z_raw)
        beta_r = F.softplus(self.beta_r_raw)
        f_emb = self.W_f(feat_i)                              # [BV,T,ki,desc_dim]
        score_in = torch.cat([g_emb, f_emb], dim=-1)
        score_logit = self.score_mlp(score_in)                # [BV,T,ki,G]
        prior = -beta_z * dz - beta_r * (r_i / R_t.unsqueeze(-1))  # [BV,T,ki]
        logit = prior.unsqueeze(-1) + score_logit             # [BV,T,ki,G]
        logit = logit + (-1e9) * (1.0 - valid_i.float()).unsqueeze(-1)

        null_in = torch.stack(
            [
                torch.log(r_mean + eps),
                torch.log(R_t),
                K_valid / kf,
                torch.log(z_span + eps),
            ],
            dim=-1,
        )  # [BV,T,4]
        logit_null = self.null_mlp(null_in)                   # [BV,T,G]

        logits_all = torch.cat([logit, logit_null.unsqueeze(2)], dim=2)  # [BV,T,ki+1,G]
        w = F.softmax(logits_all, dim=2)                      # over slots
        w_anchor = w[:, :, :ki, :]                            # [BV,T,ki,G]
        w_null = w[:, :, ki, :]                               # [BV,T,G]

        # ---- value aggregation (grouped) ------------------------------------
        v = self.W_v(feat_i) + self.W_p(g_emb)                # [BV,T,ki,token_dim]
        v = v.view(BV, T, ki, self.n_groups, self.group_dim)
        empty = self.empty_token.view(self.n_groups, self.group_dim)
        tok_anchor = torch.einsum("btkg,btkgc->btgc", w_anchor, v)
        tok_null = w_null.unsqueeze(-1) * empty               # [BV,T,G,gc]
        tok = (tok_anchor + tok_null).reshape(BV, T, self.token_dim)

        # ---- depth prior (group-averaged weights) ---------------------------
        w_mean = w_anchor.mean(dim=-1)                        # [BV,T,ki]
        z_tok = (w_mean * z_i).sum(dim=-1)                    # [BV,T]

        # ---- position + camera-ray encoding ---------------------------------
        tok = tok + self._pos_embed(H_lat, W_lat, device, f32).unsqueeze(0)
        # token centre full-res pixel -> camera ray
        u_pix = (tok_u + 0.5) * float(self.patch)             # [T]
        v_pix = (tok_v + 0.5) * float(self.patch)
        fx = K[:, 0, 0].unsqueeze(1)                          # [BV,1]
        fy = K[:, 1, 1].unsqueeze(1)
        cx = K[:, 0, 2].unsqueeze(1)
        cy = K[:, 1, 2].unsqueeze(1)
        rx = (u_pix[None, :] - cx) / fx
        ry = (v_pix[None, :] - cy) / fy
        rays = F.normalize(torch.stack([rx, ry, torch.ones_like(rx)], dim=-1), dim=-1, eps=1e-6)
        tok = tok + self.ray_mlp(rays)                        # [BV,T,token_dim]

        stats = {
            "w_null_mean": w_null.mean().detach(),
            "w_entropy": (-(w.clamp_min(1e-9) * w.clamp_min(1e-9).log()).sum(dim=2).mean()).detach(),
            "k_valid_mean": (K_valid / kf).mean().detach(),
            "beta_z": beta_z.detach(),
            "beta_r": beta_r.detach(),
        }

        return {
            "tok": tok,                                       # [BV,T,token_dim] f32
            "z_tok": z_tok,                                   # [BV,T]
            # cross-attn context (local top-k_xattn)
            "ctx_feat": feat_k,                               # [BV,T,k,feat]
            "ctx_campts": campts_k,                           # [BV,T,k,3]
            "ctx_r": r_top,                                   # [BV,T,k]
            "ctx_valid": valid_k,                             # [BV,T,k]
            "stats": stats,
            # debug / monitoring (design §6.4, §8 step-1 unit tests)
            "w_anchor": w_anchor.detach(),                    # [BV,T,ki,G]
            "w_null": w_null.detach(),                        # [BV,T,G]
            "logit_null": logit_null.detach(),                # [BV,T,G]
            "topk_idx": idx,                                  # [BV,T,k]
            "topk_r": r_top,                                  # [BV,T,k]
        }


# ---------------------------------------------------------------------------
# Cross-attention from tokens to their local anchor neighbourhood
# ---------------------------------------------------------------------------
class CrossAttnLocal(nn.Module):
    """Per-token local cross-attention: each of the T tokens attends to its own
    top-k anchors (plus a learned null slot whose value is zero-init, so a null
    win means "no update").  An absolute-scale distance bias keeps init local.
    """

    def __init__(self, token_dim: int, n_heads: int, feat_dim: int,
                 pos_dim: int = 64, gamma_init: float = 1.0) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.n_heads = int(n_heads)
        self.head_dim = self.token_dim // self.n_heads
        self.pos_dim = int(pos_dim)
        self.norm = nn.LayerNorm(self.token_dim)
        self.q_proj = nn.Linear(self.token_dim, self.token_dim)
        self.pos_mlp = MLP(3, self.pos_dim, hidden=self.pos_dim)
        kv_in = feat_dim + self.pos_dim
        self.k_proj = nn.Linear(kv_in, self.token_dim)
        self.v_proj = nn.Linear(kv_in, self.token_dim)
        self.out_proj = nn.Linear(self.token_dim, self.token_dim)
        # null slot: key learned, value zero-init ("no update")
        self.null_k = nn.Parameter(torch.randn(self.n_heads, self.head_dim) * 0.02)
        self.null_v = nn.Parameter(torch.zeros(self.n_heads, self.head_dim))
        self.gamma_raw = nn.Parameter(torch.tensor(_inverse_softplus(gamma_init)))

    def forward(self, tok: torch.Tensor, ctx: Dict[str, torch.Tensor]) -> torch.Tensor:
        BV, T, _ = tok.shape
        k = ctx["ctx_feat"].shape[2]
        x = self.norm(tok)
        q = self.q_proj(x).view(BV, T, self.n_heads, self.head_dim)  # [BV,T,H,D]
        pos = self.pos_mlp(ctx["ctx_campts"])                        # [BV,T,k,pos_dim]
        kv_in = torch.cat([ctx["ctx_feat"], pos], dim=-1)            # [BV,T,k,kv_in]
        kk = self.k_proj(kv_in).view(BV, T, k, self.n_heads, self.head_dim)
        vv = self.v_proj(kv_in).view(BV, T, k, self.n_heads, self.head_dim)
        # append null slot
        null_k = self.null_k.view(1, 1, 1, self.n_heads, self.head_dim).expand(BV, T, 1, -1, -1)
        null_v = self.null_v.view(1, 1, 1, self.n_heads, self.head_dim).expand(BV, T, 1, -1, -1)
        kk = torch.cat([kk, null_k], dim=2)                          # [BV,T,k+1,H,D]
        vv = torch.cat([vv, null_v], dim=2)
        gamma = F.softplus(self.gamma_raw)
        # attention logits
        logit = torch.einsum("bthd,btkhd->bt hk".replace(" ", ""), q, kk) / math.sqrt(self.head_dim)
        # absolute-scale locality bias on real anchors; null gets 0
        r_bias = torch.cat(
            [ctx["ctx_r"], torch.zeros_like(ctx["ctx_r"][..., :1])], dim=-1
        )  # [BV,T,k+1]
        logit = logit - gamma * r_bias.unsqueeze(2)                  # [BV,T,H,k+1]
        valid = torch.cat(
            [ctx["ctx_valid"], torch.ones_like(ctx["ctx_valid"][..., :1])], dim=-1
        ).float()
        logit = logit + (-1e9) * (1.0 - valid).unsqueeze(2)
        attn = F.softmax(logit, dim=-1)                              # [BV,T,H,k+1]
        out = torch.einsum("bthk,btkhd->bthd", attn, vv)             # [BV,T,H,D]
        out = self.out_proj(out.reshape(BV, T, self.token_dim))
        return tok + out


# ---------------------------------------------------------------------------
# ViT block (pre-LN self-attn + FFN), optional cross-attn after
# ---------------------------------------------------------------------------
class GridBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0,
                 cross: Optional[CrossAttnLocal] = None) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )
        self.cross = cross

    def forward(self, x: torch.Tensor, ctx: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        if self.cross is not None and ctx is not None:
            x = self.cross(x, ctx)
        return x


# ---------------------------------------------------------------------------
# Output heads (convolution-coupled two-stage PixelShuffle)
# ---------------------------------------------------------------------------
class RGBHead(nn.Module):
    def __init__(self, token_dim: int, patch: int = 14, mid_ch: int = 96) -> None:
        super().__init__()
        self.patch = int(patch)
        self.mid_ch = int(mid_ch)
        r1 = self.patch // 2  # 7
        r2 = 2
        self.r1, self.r2 = r1, r2
        self.proj = nn.Linear(token_dim, r1 * r1 * mid_ch)
        self.conv1 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1), nn.GroupNorm(8, mid_ch), nn.GELU(),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1), nn.GroupNorm(8, mid_ch), nn.GELU(),
        )
        self.conv2 = nn.Conv2d(mid_ch, r2 * r2 * 3, 3, padding=1)
        self.ps1 = nn.PixelShuffle(r1)
        self.ps2 = nn.PixelShuffle(r2)

    def forward(self, tok: torch.Tensor, grid_hw: tuple) -> torch.Tensor:
        BV, T, _ = tok.shape
        H_lat, W_lat = grid_hw
        x = self.proj(tok)  # [BV,T, r1*r1*mid]
        x = x.view(BV, H_lat, W_lat, self.mid_ch, self.r1, self.r1)
        x = x.permute(0, 3, 4, 5, 1, 2).reshape(BV, self.mid_ch * self.r1 * self.r1, H_lat, W_lat)
        x = self.ps1(x)     # [BV, mid, H_lat*r1, W_lat*r1]
        x = self.conv1(x)
        x = self.conv2(x)   # [BV, r2*r2*3, ...]
        x = self.ps2(x)     # [BV, 3, H, W]
        return x


class DepthHead(nn.Module):
    def __init__(self, token_dim: int, patch: int = 14, mid_ch: int = 32) -> None:
        super().__init__()
        self.patch = int(patch)
        self.mid_ch = int(mid_ch)
        r1 = self.patch // 2
        r2 = 2
        self.r1, self.r2 = r1, r2
        # +1 input channel: normalized inverse-depth prior (u_prior)
        self.proj = nn.Linear(token_dim + 1, r1 * r1 * mid_ch)
        self.conv1 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1), nn.GroupNorm(8, mid_ch), nn.GELU(),
        )
        self.conv2 = nn.Conv2d(mid_ch, r2 * r2 * 1, 3, padding=1)
        self.ps1 = nn.PixelShuffle(r1)
        self.ps2 = nn.PixelShuffle(r2)

    def forward(self, tok: torch.Tensor, u_prior: torch.Tensor, grid_hw: tuple) -> torch.Tensor:
        BV, T, _ = tok.shape
        H_lat, W_lat = grid_hw
        x = torch.cat([tok, u_prior.unsqueeze(-1)], dim=-1)  # [BV,T,token_dim+1]
        x = self.proj(x)
        x = x.view(BV, H_lat, W_lat, self.mid_ch, self.r1, self.r1)
        x = x.permute(0, 3, 4, 5, 1, 2).reshape(BV, self.mid_ch * self.r1 * self.r1, H_lat, W_lat)
        x = self.ps1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.ps2(x)     # [BV, 1, H, W]
        return x


# ---------------------------------------------------------------------------
# Full decoder
# ---------------------------------------------------------------------------
class SuperpointGridDecoder(nn.Module):
    def __init__(
        self,
        superpoint_dim: int = 64,
        token_dim: int = 792,
        patch: int = 14,
        image_size: tuple = (378, 518),
        k_init: int = 8,
        k_xattn: int = 64,
        n_groups: int = 4,
        shared_layers: int = 4,
        rgb_layers: int = 8,
        depth_layers: int = 3,
        depth_prep_layers: int = 0,
        depth_dim: int = 384,
        n_heads: int = 12,
        depth_heads: int = 8,
        mlp_ratio: float = 4.0,
        rgb_num_views: int = 3,
        depth_num_views: int = 3,
        alpha_grad_gate: float = 0.5,
        beta_init: float = 3.0,
        gamma_init: float = 1.0,
        # cross-attn insertion points (block indices in the 12-layer rgb path)
        xattn_after_shared: tuple = (0, 2),
        xattn_after_rgb: tuple = (0, 2, 4, 6),  # relative to rgb branch
        xattn_after_depth_prep: tuple = (),
        depth_min_norm: float = 0.02,
        depth_max_norm: float = 50.0,
        depth_pct_lo: float = 1.0,
        depth_pct_hi: float = 99.0,
    ) -> None:
        super().__init__()
        self.superpoint_dim = int(superpoint_dim)
        self.token_dim = int(token_dim)
        self.patch = int(patch)
        self.image_size = tuple(image_size)
        self.rgb_num_views = int(rgb_num_views)
        self.depth_num_views = int(depth_num_views)
        self.alpha_grad_gate = float(alpha_grad_gate)
        self.depth_min_norm = float(depth_min_norm)
        self.depth_max_norm = float(depth_max_norm)
        self.depth_pct_lo = float(depth_pct_lo)
        self.depth_pct_hi = float(depth_pct_hi)
        H, W = self.image_size
        self.grid_hw = (H // self.patch, W // self.patch)

        self.frontend = AnchorToGrid(
            feat_dim=superpoint_dim, token_dim=token_dim, patch=patch,
            k_init=k_init, k_xattn=k_xattn, n_groups=n_groups,
            beta_init=beta_init,
        )

        def make_cross(dim, heads):
            return CrossAttnLocal(dim, heads, superpoint_dim, gamma_init=gamma_init)

        # shared trunk
        self.shared_blocks = nn.ModuleList([
            GridBlock(token_dim, n_heads, mlp_ratio,
                      cross=make_cross(token_dim, n_heads) if i in xattn_after_shared else None)
            for i in range(shared_layers)
        ])
        # rgb branch
        self.rgb_blocks = nn.ModuleList([
            GridBlock(token_dim, n_heads, mlp_ratio,
                      cross=make_cross(token_dim, n_heads) if i in xattn_after_rgb else None)
            for i in range(rgb_layers)
        ])
        # optional depth-only prep (token_dim).  Empty in v27/v29.
        self.depth_prep_blocks = nn.ModuleList([
            GridBlock(token_dim, n_heads, mlp_ratio,
                      cross=make_cross(token_dim, n_heads) if i in xattn_after_depth_prep else None)
            for i in range(int(depth_prep_layers))
        ])
        # depth branch
        self.depth_proj = nn.Linear(token_dim, depth_dim)
        self.depth_blocks = nn.ModuleList([
            GridBlock(depth_dim, depth_heads, mlp_ratio,
                      cross=make_cross(depth_dim, depth_heads) if i == 0 else None)
            for i in range(depth_layers)
        ])

        self.rgb_head = RGBHead(token_dim, patch=patch, mid_ch=96)
        self.depth_head = DepthHead(depth_dim, patch=patch, mid_ch=32)
        # Zero residual reproduces the upsampled anchor inverse-depth prior.
        nn.init.zeros_(self.depth_head.conv2.weight)
        if self.depth_head.conv2.bias is not None:
            nn.init.zeros_(self.depth_head.conv2.bias)

    # -- per-view adaptive inverse-depth range from anchor z -----------------
    def _depth_inv_range(self, z: torch.Tensor, valid: torch.Tensor):
        with torch.no_grad():
            zv = torch.where(valid, z.float(), torch.full_like(z, float("nan")))
            qs = torch.tensor([self.depth_pct_lo / 100.0, self.depth_pct_hi / 100.0],
                              device=z.device, dtype=torch.float32)
            per = torch.nanquantile(zv, qs, dim=-1)
            d_lo = torch.nan_to_num(per[0], nan=self.depth_min_norm).clamp(min=self.depth_min_norm)
            d_hi = torch.nan_to_num(per[1], nan=self.depth_max_norm).clamp(max=self.depth_max_norm)
            d_hi = torch.maximum(d_hi, d_lo + 1e-3)
        return d_lo, d_hi

    _LOGIT_EPS = 1e-4

    def _compose_inv_depth(
        self,
        raw: torch.Tensor,
        u_prior: torch.Tensor,
        inv_lo: torch.Tensor,
        inv_hi: torch.Tensor,
        grid_hw: tuple,
    ):
        """Residual inverse depth: zero residual keeps the anchor prior."""
        h_lat, w_lat = grid_hw
        bv, _, h, w = raw.shape
        u_map = u_prior.float().reshape(bv, 1, h_lat, w_lat)
        u_up = F.interpolate(u_map, size=(h, w), mode="nearest")
        logit_prior = torch.logit(u_up.clamp(self._LOGIT_EPS, 1.0 - self._LOGIT_EPS))
        residual = raw.float()
        u_out = torch.sigmoid(logit_prior + residual)
        scale = (inv_hi - inv_lo).view(-1, 1, 1, 1)
        inv_out = inv_lo.view(-1, 1, 1, 1) + u_out * scale
        extras = {
            "depth_residual_abs": residual.detach().abs().mean(),
            "depth_u_prior_mean": u_up.detach().mean(),
        }
        return 1.0 / inv_out.clamp_min(1e-6), extras

    def forward(
        self,
        sp_feat: torch.Tensor,       # [B,N,C] latent (cached z_sp)
        camtoworlds: torch.Tensor,   # [B,V,4,4]
        Ks: torch.Tensor,            # [B,V,3,3]
        image,                       # [B,V,H,W,3]
        sp_xyz: torch.Tensor,        # [B,N,3]
        valid_mask: Optional[torch.Tensor] = None,
        sp_valid_mask: Optional[torch.Tensor] = None,
        decode_rgb: bool = True,
        **_unused,  # tolerate geom_radius/eigvals/normal passed by the online path
    ) -> Dict[str, torch.Tensor]:
        B, V = camtoworlds.shape[:2]
        device = sp_feat.device
        H, W = self.image_size
        grid_hw = self.grid_hw

        # ---- view selection (shared trunk views) ---------------------------
        if self.training:
            V_render = max(1, min(max(self.rgb_num_views, self.depth_num_views), V))
            perm = torch.rand(B, V, device=device).argsort(dim=-1)[:, :V_render]
            render_idx, _ = perm.sort(dim=-1)
        else:
            V_render = V
            render_idx = torch.arange(V, device=device).unsqueeze(0).expand(B, V).contiguous()
        # rgb / depth view indices into the *global* view axis
        if self.training:
            rgb_idx = render_idx
            depth_idx = render_idx
        else:
            rgb_idx = torch.arange(min(self.rgb_num_views, V), device=device).unsqueeze(0).expand(B, -1).contiguous()
            depth_idx = torch.arange(V, device=device).unsqueeze(0).expand(B, V).contiguous()

        # gather the render views' camera params
        def gather_views(x):
            # x: [B,V,...] -> [B,V_render,...]
            idx = render_idx.reshape(B, V_render, *([1] * (x.dim() - 2))).expand(
                B, V_render, *x.shape[2:])
            return torch.gather(x, 1, idx)

        c2w_r = gather_views(camtoworlds)
        K_r = gather_views(Ks)

        # ---- frontend on the render views ----------------------------------
        fe = self.frontend(sp_feat, sp_xyz, c2w_r, K_r, grid_hw, sp_valid_mask)
        tok = fe["tok"]                       # [B*V_render, T, token_dim] f32
        z_tok = fe["z_tok"]                   # [B*V_render, T]
        # cross-attn context (autocast handles dtypes inside the blocks)
        ctx = {
            "ctx_feat": fe["ctx_feat"],
            "ctx_campts": fe["ctx_campts"],
            "ctx_r": fe["ctx_r"],
            "ctx_valid": fe["ctx_valid"],
        }

        # ---- shared trunk ---------------------------------------------------
        h = tok
        for blk in self.shared_blocks:
            h = blk(h, ctx)

        # ---- rgb branch -----------------------------------------------------
        trunk_rgb = h
        for blk in self.rgb_blocks:
            trunk_rgb = blk(trunk_rgb, ctx)

        # ---- depth branch ---------------------------------------------------
        if len(self.depth_prep_blocks) > 0:
            # Isolated depth stack: no shared ViT.  Detach so depth does not
            # write into frontend / RGB tokens.
            trunk_dep = tok.detach()
            for blk in self.depth_prep_blocks:
                trunk_dep = blk(trunk_dep, ctx)
        else:
            a = self.alpha_grad_gate
            trunk_dep = a * h + (1.0 - a) * h.detach()
        d = self.depth_proj(trunk_dep)
        for blk in self.depth_blocks:
            d = blk(d, ctx)

        # ---- heads -----------------------------------------------------------
        out: Dict[str, torch.Tensor] = {}
        # RGB
        if decode_rgb:
            rgb_full = self.rgb_head(trunk_rgb.float(), grid_hw)  # [BV,3,H,W]
            rgb_full = rgb_full.permute(0, 2, 3, 1).contiguous()  # [BV,H,W,3]
            rgb_full = rgb_full.view(B, V_render, H, W, 3)
            # map back to global rgb view order: rgb_idx selects from render views
            out["rendered_rgb"] = self._select_render_views(rgb_full, render_idx, rgb_idx)
            out["rendered_colors"] = out["rendered_rgb"]
        # Depth
        # adaptive inverse-depth range from anchor z (per render view)
        w2c_r = torch.linalg.inv(c2w_r.to(torch.float32))
        ones = torch.ones((B, sp_xyz.shape[1], 1), device=device, dtype=torch.float32)
        sp_h = torch.cat([torch.nan_to_num(sp_xyz.float(), nan=0.0), ones], dim=-1)
        sp_h = sp_h[:, None].expand(B, V_render, -1, 4).reshape(B * V_render, -1, 4)
        cam_pts = torch.matmul(sp_h, w2c_r.reshape(B * V_render, 4, 4).transpose(1, 2))[..., :3]
        z_all = cam_pts[..., 2]
        valid_all = z_all > 1e-4
        if sp_valid_mask is not None:
            valid_all = valid_all & sp_valid_mask.bool()[:, None].expand(B, V_render, -1).reshape(B * V_render, -1)
        d_lo, d_hi = self._depth_inv_range(z_all, valid_all)  # [B*V_render]
        inv_lo = 1.0 / d_hi
        inv_hi = 1.0 / d_lo
        # normalized inverse-depth prior channel from z_tok
        inv_prior = 1.0 / z_tok.clamp_min(1e-4)               # [BV,T]
        u_prior = ((inv_prior - inv_lo.unsqueeze(-1)) / (inv_hi - inv_lo).unsqueeze(-1)).clamp(0, 1)
        if len(self.depth_prep_blocks) > 0:
            # residual compose would otherwise send depth grads through z_tok.
            u_prior = u_prior.detach()
        raw = self.depth_head(d.float(), u_prior.float(), grid_hw)  # [BV,1,H,W]
        depth, depth_extra = self._compose_inv_depth(raw, u_prior, inv_lo, inv_hi, grid_hw)
        depth = depth.permute(0, 2, 3, 1).contiguous().view(B, V_render, H, W, 1)
        out["rendered_depths"] = self._select_render_views(depth, render_idx, depth_idx)
        out.update(depth_extra)

        out["selected_view_idx_rgb"] = rgb_idx
        out["selected_view_idx_depth"] = depth_idx
        out["grid_hw"] = torch.tensor(grid_hw, device=device)
        # monitoring stats
        for kk, vv in fe["stats"].items():
            out[f"frontend_{kk}"] = vv
        out["frontend_wp_over_wv"] = (
            self.frontend.W_p.net[-1].weight.norm() / self.frontend.W_v.weight.norm().clamp_min(1e-8)
        ).detach()
        gammas = [b.cross for b in list(self.shared_blocks) + list(self.rgb_blocks) if b.cross is not None]
        if gammas:
            out["frontend_gamma"] = torch.stack([F.softplus(c.gamma_raw) for c in gammas]).mean().detach()
        return out

    @staticmethod
    def _select_render_views(x: torch.Tensor, render_idx: torch.Tensor, want_idx: torch.Tensor) -> torch.Tensor:
        """x: [B, V_render, ...]; render_idx / want_idx: global view ids [B, Vr] / [B, Vw].
        Returns x gathered at the positions of want_idx within render_idx -> [B, Vw, ...].
        During training render_idx == want_idx so this is identity; at eval render
        covers all views and want_idx is a prefix."""
        B, Vr = render_idx.shape
        Vw = want_idx.shape[1]
        if Vr == Vw and bool((render_idx == want_idx).all()):
            return x
        # build position of each wanted global view inside render_idx
        # render_idx sorted ascending; want_idx is a subset
        pos = torch.zeros(B, Vw, dtype=torch.long, device=x.device)
        for b in range(B):
            # searchsorted needs sorted; render_idx is sorted
            pos[b] = torch.searchsorted(render_idx[b], want_idx[b])
        idx = pos.reshape(B, Vw, *([1] * (x.dim() - 2))).expand(B, Vw, *x.shape[2:])
        return torch.gather(x, 1, idx)
