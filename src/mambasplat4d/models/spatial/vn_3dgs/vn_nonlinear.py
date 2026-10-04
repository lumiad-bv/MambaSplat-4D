"""VNReLU, VNLeakyReLU, VNFeedForward (from VN-3DGS/src/models/layers.py)."""
from torch import nn, einsum
import torch

from .vn_linear import VNLinear

def inner_dot_product(x, y):
    return (x * y).sum(dim=-1, keepdim=True)


class VNReLU(nn.Module):
    """Equivariant ReLU: keep q if <q,k> >= 0, else project q onto plane orthogonal to k."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        
        self.W = nn.Parameter(torch.randn(dim, dim))
        self.U = nn.Parameter(torch.randn(dim, dim))

    def forward(self, x):
        q = einsum('... i c, o i -> ... o c', x, self.W)
        k = einsum('... i c, o i -> ... o c', x, self.U)

        qk = inner_dot_product(q, k)

        k_norm = k.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        k_hat = k / k_norm
        q_projected_on_k = q - inner_dot_product(q, k / k_norm) * k_hat

        out = torch.where(qk >= 0., q, q_projected_on_k)
        return out

class VNLeakyReLU(nn.Module):
    """q if <q,k> >= 0, else q - (1 - negative_slope) <q,k_hat> k_hat."""
    def __init__(self, dim, negative_slope=0.01, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.negative_slope = negative_slope
        self.W = nn.Parameter(torch.randn(dim, dim))
        self.U = nn.Parameter(torch.randn(dim, dim))

    def forward(self, x):
        q = einsum('... i c, o i -> ... o c', x, self.W)
        k = einsum('... i c, o i -> ... o c', x, self.U)

        qk = inner_dot_product(q, k)

        k_norm = k.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        k_hat = k / k_norm
        
        q_on_k = inner_dot_product(q, k_hat) * k_hat
        
        # slope 0 = VNReLU, slope 1 = identity
        out_neg = q - (1 - self.negative_slope) * q_on_k
        
        out = torch.where(qk >= 0., q, out_neg)
        return out


def VNFeedForward(dim, mult=4, bias_epsilon=0.):
    """VNLinear -> VNReLU -> VNLinear."""
    dim_inner = int(dim * mult)
    return nn.Sequential(
        VNLinear(dim, dim_inner, bias_epsilon=bias_epsilon),
        VNReLU(dim_inner),
        VNLinear(dim_inner, dim, bias_epsilon=bias_epsilon),
    )
