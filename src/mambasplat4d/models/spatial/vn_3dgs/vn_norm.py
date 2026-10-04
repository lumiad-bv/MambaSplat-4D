"""VN normalisation (from VN-3DGS/src/models/layers.py)."""
import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange


class LayerNorm(nn.Module):
    """LayerNorm, learnable gamma, fixed zero beta."""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class VNLayerNorm(nn.Module):
    """Equivariant LayerNorm: unit directions scaled by LayerNorm of norms."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.ln = LayerNorm(dim)

    def forward(self, x):
        norms = x.norm(dim=-1)
        x = x / rearrange(norms.clamp(min=self.eps), '... -> ... 1')
        ln_out = self.ln(norms)
        return x * rearrange(ln_out, '... -> ... 1')
