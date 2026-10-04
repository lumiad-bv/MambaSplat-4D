"""VN linear layer and helpers (from VN-3DGS/src/models/layers.py)."""
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def inner_dot_product(x, y, *, dim=-1, keepdim=True):
    return (x * y).sum(dim=dim, keepdim=keepdim)


class Rearrange(nn.Module):
    """einops rearrange as module."""

    def __init__(self, pattern):
        super().__init__()
        self.pattern = pattern

    def forward(self, x):
        return rearrange(x, self.pattern)


class VNLinear(nn.Module):
    """VNLinear(x @ R) = VNLinear(x) @ R. bias_epsilon > 0 adds quasi-equivariant bias."""

    def __init__(self, dim_in, dim_out, bias_epsilon=0.):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim_out, dim_in))

        self.bias = None
        self.bias_epsilon = bias_epsilon

        if bias_epsilon > 0.:
            self.bias = nn.Parameter(torch.randn(dim_out))

    def forward(self, x):
        out = einsum('... i c, o i -> ... o c', x, self.weight)

        if exists(self.bias):
            bias = F.normalize(self.bias, dim=-1) * self.bias_epsilon
            out = out + rearrange(bias, '... -> ... 1')

        return out
