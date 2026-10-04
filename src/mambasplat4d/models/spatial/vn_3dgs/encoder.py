"""VN3DGSSpatialEncoder: VN-3DGS as SpatialEncoder, forward(gaussian_data, mask) -> [B, embed_dim].

Dims from feature_mode (no zero-padding):
  C:       K=1, dim_inv=0, dim_coor=3
  C_O:     K=1, dim_inv=1, dim_coor=4
  C_SH:    K=1, dim_inv=3, dim_coor=6
  C_S_R:   K=4, dim_inv=0, dim_coor=3
  C_O_S_R: K=4, dim_inv=1, dim_coor=4
  X:       K=4, dim_inv=4, dim_coor=7
"""
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig

from ..base import SpatialEncoder
from ...registry import register_spatial

from .vn_linear import VNLinear, Rearrange
from .vn_transformer_block import VNTransformerEncoder
from .invariant_head import VNWeightedPool, VN3DGSClassifierHead
from .fusion import gaussian_lifting, count_lifting_vectors
from ....utils.feature_config import get_invariant_dim, has_rotation_axes, has_scale_weighting


class GaussianVNProjection(nn.Module):
    """K lifted vectors -> VN features."""

    def __init__(self, dim: int, num_vectors: int = 4, bias_epsilon: float = 0.):
        super().__init__()
        self.vn_linear = VNLinear(num_vectors, dim, bias_epsilon=bias_epsilon)

    def forward(self, x: Tensor) -> Tensor:
        return self.vn_linear(x)


class StandardVNProjection(nn.Module):
    """Single-vector VN projection."""

    def __init__(self, dim: int, bias_epsilon: float = 0.):
        super().__init__()
        self.proj = nn.Sequential(
            Rearrange('... c -> ... 1 c'),
            VNLinear(1, dim, bias_epsilon=bias_epsilon),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


@register_spatial("vn_3dgs")
class VN3DGSSpatialEncoder(SpatialEncoder):
    """Rotation-equivariant 3DGS encoder: lifting -> VN projection -> invariant
    injection (opacity, SH DC; skipped if dim_invariant=0) -> VN-Transformer
    -> norm invariants -> [B, embed_dim]."""

    def __init__(self, cfg: DictConfig):
        super().__init__()

        dim = cfg.dim
        depth = cfg.depth
        heads = cfg.heads
        dim_head = cfg.dim_head
        embed_dim = cfg.embed_dim
        bias_epsilon = cfg.bias_epsilon
        flash_attn = cfg.flash_attn
        l2_dist_attn = cfg.l2_dist_attn
        dropout = cfg.dropout

        feature_mode = cfg.get('feature_mode', 'X')

        self.use_centroid = True
        self.use_rotation_axes = has_rotation_axes(feature_mode)
        self.use_scale_weighting = has_scale_weighting(feature_mode)
        self.translation_invariant = cfg.translation_invariant

        self.dim_invariant = get_invariant_dim(feature_mode)
        self.zero_invariant = bool(cfg.get('zero_invariant', False))
        self.match_invariant_scale = bool(cfg.get('match_invariant_scale', False))

        self.expected_num_vectors = count_lifting_vectors(
            self.use_centroid, self.use_rotation_axes,
        )

        if self.expected_num_vectors > 1:
            self.input_proj = GaussianVNProjection(
                dim=dim, num_vectors=self.expected_num_vectors,
                bias_epsilon=bias_epsilon,
            )
        else:
            self.input_proj = StandardVNProjection(dim=dim, bias_epsilon=bias_epsilon)

        dim_coor_total = 3 + self.dim_invariant

        self.encoder = VNTransformerEncoder(
            dim=dim, depth=depth,
            dim_head=dim_head, heads=heads,
            dim_coor=dim_coor_total,
            bias_epsilon=bias_epsilon,
            flash_attn=flash_attn,
            l2_dist_attn=l2_dist_attn,
            final_norm=True,
        )

        self.classifier_head = VN3DGSClassifierHead(
            dim=dim, embed_dim=embed_dim,
            dim_coor=dim_coor_total,
            bias_epsilon=bias_epsilon,
            dropout=dropout,
        )

    def forward(
        self,
        gaussian_data: dict,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """gaussian_data: 'position' required; 'quaternion', 'scale', 'opacity', 'sh_dc' optional.
        Returns (B, embed_dim), or (V (B,C,3), X_inv (B,C,F)) if return_vectors."""
        return_vectors = getattr(self, 'return_vectors', False)
        position = gaussian_data['position']
        quaternion = gaussian_data.get('quaternion', None)
        scale = gaussian_data.get('scale', None)
        opacity = gaussian_data.get('opacity', None)
        sh_dc = gaussian_data.get('sh_dc', None)

        B, N, _ = position.shape

        if self.translation_invariant:
            if mask is not None:
                mask_f = mask.float().unsqueeze(-1)
                centroid = (position * mask_f).sum(dim=1, keepdim=True) / mask_f.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                centroid = position.mean(dim=1, keepdim=True)
            position = position - centroid

        vectors = gaussian_lifting(
            position, quaternion, scale,
            use_centroid=self.use_centroid,
            use_rotation_axes=self.use_rotation_axes,
            use_scale_weighting=self.use_scale_weighting,
        )

        if self.expected_num_vectors > 1:
            x = self.input_proj(vectors)
        else:
            x = self.input_proj(vectors[:, :, 0, :])

        # invariant scalars: opacity (1), sh_dc (3), scale (3) if not axis-weighted
        if self.dim_invariant > 0:
            inv_parts = []
            if opacity is not None:
                inv_parts.append(opacity)          # (B, N, 1)
            if sh_dc is not None:
                inv_parts.append(sh_dc)            # (B, N, 3)
            if scale is not None and not self.use_scale_weighting:
                inv_parts.append(scale)            # (B, N, 3)

            if inv_parts:
                inv_feats = torch.cat(inv_parts, dim=-1)[..., :self.dim_invariant]
            else:
                inv_feats = torch.zeros(B, N, self.dim_invariant, device=x.device, dtype=x.dtype)

            if self.zero_invariant:
                inv_feats = torch.zeros_like(inv_feats)
            elif self.match_invariant_scale:
                inv_feats = inv_feats - inv_feats.mean(dim=1, keepdim=True)
                inv_rms = inv_feats.pow(2).mean().sqrt().clamp_min(1e-6)
                inv_feats = inv_feats * (x.pow(2).mean().sqrt() / inv_rms)

            inv_expanded = inv_feats.unsqueeze(2).expand(B, N, x.shape[2], -1)
            x = torch.cat([x, inv_expanded], dim=-1)

        x = self.encoder(x, mask=mask)

        if return_vectors:
            x_global = self.classifier_head.global_pool(x, mask=mask)  # (B, C, 3+dim_inv)
            return x_global[..., :3], x_global[..., 3:]  # (V, X_inv)

        embeddings = self.classifier_head(x, mask=mask)

        return embeddings
