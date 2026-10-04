"""VN-Transformer encoder (from VN-3DGS/src/models/encoder.py)."""
from torch import nn

from .vn_norm import VNLayerNorm
from .vn_nonlinear import VNFeedForward
from .vn_attention import VNAttention
from .vn_linear import exists


class VNTransformerEncoder(nn.Module):
    """depth x (VNAttention + VNFeedForward, post-LN residuals), equivariant throughout.
    dim_coor = 3 or 3+F with invariant feats."""

    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head=64,
        heads=8,
        dim_coor=3,
        ff_mult=4,
        final_norm=False,
        bias_epsilon=0.,
        l2_dist_attn=False,
        flash_attn=False,
    ):
        super().__init__()
        self.dim = dim
        self.dim_coor = dim_coor

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                VNAttention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    bias_epsilon=bias_epsilon,
                    l2_dist_attn=l2_dist_attn,
                    flash=flash_attn,
                ),
                VNLayerNorm(dim),
                VNFeedForward(dim=dim, mult=ff_mult, bias_epsilon=bias_epsilon),
                VNLayerNorm(dim),
            ]))

        self.norm = VNLayerNorm(dim) if final_norm else nn.Identity()

    def forward(self, x, mask=None):
        *_, d, c = x.shape
        assert x.ndim == 4 and d == self.dim and c == self.dim_coor, \
            f'expected (B, N, {self.dim}, {self.dim_coor}), got {x.shape}'

        for attn, attn_post_ln, ff, ff_post_ln in self.layers:
            x = attn_post_ln(attn(x, mask=mask)) + x
            x = ff_post_ln(ff(x)) + x

        return self.norm(x)
