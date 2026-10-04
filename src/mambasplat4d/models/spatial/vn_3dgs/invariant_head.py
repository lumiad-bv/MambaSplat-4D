"""Pooling and invariant extraction (from VN-3DGS pooling.py + classifier.py)."""
import torch
from torch import nn, einsum

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from .vn_linear import VNLinear, exists, default
from .vn_nonlinear import VNReLU


class VNWeightedPool(nn.Module):
    """Equivariant learnable-weight pooling; num_pooled_tokens output tokens
    (squeezed when 1)."""

    def __init__(self, dim, dim_out=None, num_pooled_tokens=1, squeeze_out_pooled_dim=True):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.weight = nn.Parameter(torch.randn(num_pooled_tokens, dim, dim_out))
        self.squeeze_out_pooled_dim = num_pooled_tokens == 1 and squeeze_out_pooled_dim

    def forward(self, x, mask=None):
        if exists(mask):
            mask = rearrange(mask, 'b n -> b n 1 1')
            x = x.masked_fill(~mask, 0.)
            numer = reduce(x, 'b n d c -> b d c', 'sum')
            denom = mask.sum(dim=1)
            mean_pooled = numer / denom.clamp(min=1e-6)
        else:
            mean_pooled = reduce(x, 'b n d c -> b d c', 'mean')

        out = einsum('b d c, m d e -> b m e c', mean_pooled, self.weight)

        if not self.squeeze_out_pooled_dim:
            return out

        out = rearrange(out, 'b 1 d c -> b d c')
        return out

class VNInvariant(nn.Module):
    """Invariant extraction via inner products x . mlp(x): VNInvariant(x @ R) = VNInvariant(x).
    dim_coor = 3 or 3+F."""

    def __init__(
        self,
        dim,
        dim_coor=3,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            VNLinear(dim, dim_coor),
            VNReLU(dim_coor),
            Rearrange('... d e -> ... e d')
        )

    def forward(self, x):
        """(B, N, dim, C) -> (B, N, C, C)."""
        return einsum('b n d i, b n i o -> b n o', x, self.mlp(x))

class VN3DGSClassifierHead(nn.Module):
    """global pool -> vector norms -> MLP embedding. Final classifier lives in heads/classifier.py."""

    def __init__(
        self,
        dim: int,
        embed_dim: int = 128,
        dim_coor: int = 3,
        bias_epsilon: float = 0.,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dim = dim
        self.dim_coor = dim_coor

        self.global_pool = VNWeightedPool(
            dim=dim, dim_out=dim,
            num_pooled_tokens=1, squeeze_out_pooled_dim=True,
        )

        self.projection = nn.Sequential(
            nn.Linear(dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """(B, N, C, D), mask (B, N) -> (B, embed_dim)."""
        x_global = self.global_pool(x, mask=mask)  # (B, C, D)
        norms = x_global.norm(dim=-1)               # (B, C)
        embeddings = self.projection(norms)          # (B, embed_dim)
        return embeddings
