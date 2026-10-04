"""VN-Transformer attention (from VN-3DGS/src/models/attention.py)."""
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from einops import rearrange, reduce

from .vn_linear import VNLinear, exists


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner


print_once = once(print)


class Attend(nn.Module):
    """Standard, flash, or L2-distance attention."""

    def __init__(self, dropout=0., flash=False, l2_dist=False):
        super().__init__()
        assert not (flash and l2_dist)
        self.l2_dist = l2_dist
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.flash = flash

        self.cpu_backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
        self.cuda_backends = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))
        if device_properties.major >= 8:
            print_once(f'{device_properties.name} (sm_{device_properties.major}{device_properties.minor}), using flash attention')
            self.cuda_backends = [SDPBackend.FLASH_ATTENTION]
        else:
            print_once(f'{device_properties.name} (sm_{device_properties.major}{device_properties.minor}), using math/mem efficient attention')
            self.cuda_backends = [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]

    def flash_attn(self, q, k, v, mask=None):
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        if exists(mask):
            mask = mask.expand(-1, heads, q_len, -1)

        backends = self.cuda_backends if is_cuda else self.cpu_backends
        # flash kernel rejects attn_mask
        if exists(mask) and backends == [SDPBackend.FLASH_ATTENTION]:
            backends = [SDPBackend.MATH]

        with sdpa_kernel(backends):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.
            )
        return out

    def forward(self, q, k, v, mask=None):
        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device
        scale = q.shape[-1] ** -0.5

        if exists(mask) and mask.ndim != 4:
            mask = rearrange(mask, 'b j -> b 1 1 j')

        if self.flash:
            return self.flash_attn(q, k, v, mask=mask)

        sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale

        if self.l2_dist:
            q_squared = reduce(q ** 2, 'b h i d -> b h i 1', 'sum')
            k_squared = reduce(k ** 2, 'b h j d -> b h 1 j', 'sum')
            sim = sim * 2 - q_squared - k_squared

        if exists(mask):
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        return out


class VNAttention(nn.Module):
    """Equivariant multi-head attention; scores via Frobenius inner products."""

    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        dim_coor=3,
        bias_epsilon=0.,
        l2_dist_attn=False,
        flash=False,
        num_latents=None,
    ):
        super().__init__()
        assert not (l2_dist_attn and flash)

        self.scale = (dim_coor * dim_head) ** -0.5
        dim_inner = dim_head * heads
        self.heads = heads

        from .invariant_head import VNWeightedPool

        self.to_q_input = None
        if exists(num_latents):
            self.to_q_input = VNWeightedPool(
                dim, num_pooled_tokens=num_latents, squeeze_out_pooled_dim=False
            )

        self.to_q = VNLinear(dim, dim_inner, bias_epsilon=bias_epsilon)
        self.to_k = VNLinear(dim, dim_inner, bias_epsilon=bias_epsilon)
        self.to_v = VNLinear(dim, dim_inner, bias_epsilon=bias_epsilon)
        self.to_out = VNLinear(dim_inner, dim, bias_epsilon=bias_epsilon)

        if l2_dist_attn and not exists(num_latents):
            self.to_k = self.to_q

        self.attend = Attend(flash=flash, l2_dist=l2_dist_attn)

    def forward(self, x, mask=None):
        c = x.shape[-1]

        if exists(self.to_q_input):
            q_input = self.to_q_input(x, mask=mask)
        else:
            q_input = x

        q, k, v = self.to_q(q_input), self.to_k(x), self.to_v(x)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) c -> b h n (d c)', h=self.heads),
            (q, k, v)
        )

        out = self.attend(q, k, v, mask=mask)
        out = rearrange(out, 'b h n (d c) -> b n (h d) c', c=c)
        return self.to_out(out)
