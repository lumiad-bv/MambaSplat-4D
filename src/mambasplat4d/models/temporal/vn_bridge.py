"""VN-Mamba bridge: post-pool (V, X_inv) -> invariant d_mamba embedding. No mamba_ssm import."""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig

from ..spatial.vn_3dgs.vn_linear import VNLinear
from ..spatial.vn_3dgs.vn_nonlinear import VNLeakyReLU


class VNInBridge(nn.Module):
    """VN-In: V . T^T -> 3C invariants, plus C * dim_inv passthrough scalars."""

    def __init__(
        self,
        C: int,
        dim_invariant: int,
        d_mamba: int = 256,
        frame_hidden: Optional[int] = None,
    ):
        super().__init__()
        self.C = C
        self.dim_invariant = dim_invariant
        H = C if frame_hidden is None else int(frame_hidden)
        self.frame_hidden = H

        self.frame_mlp = nn.Sequential(
            VNLinear(2 * C, H),
            VNLeakyReLU(H, negative_slope=0.2),
            VNLinear(H, H),
            VNLeakyReLU(H, negative_slope=0.2),
            VNLinear(H, 3),
        )

        d_in = 3 * C + C * dim_invariant
        self.proj = nn.Sequential(
            nn.Linear(d_in, d_mamba),
            nn.GELU(),
            nn.LayerNorm(d_mamba),
        )

    def forward(
        self,
        V: Tensor,
        X_inv: Tensor,
        gaussian_data: Optional[dict] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        B = V.shape[0]

        V_mean = V.mean(dim=1, keepdim=True).expand_as(V)
        V_cat = torch.cat([V, V_mean], dim=1)
        T = self.frame_mlp(V_cat)

        S = torch.bmm(V, T.transpose(1, 2))
        s_vn_in = S.reshape(B, -1)

        parts = [s_vn_in]
        if self.dim_invariant > 0:
            s_inv = X_inv.reshape(B, -1)
            parts.append(s_inv)

        features = torch.cat(parts, dim=-1)
        return self.proj(features)


def build_bridge(
    cfg_bridge: Optional[DictConfig],
    C: int,
    dim_invariant: int,
    d_mamba: int,
) -> nn.Module:
    """cfg_bridge None or name="vn_in" -> VNInBridge (optional frame_hidden)."""
    if cfg_bridge is None:
        return VNInBridge(C=C, dim_invariant=dim_invariant, d_mamba=d_mamba)

    name = cfg_bridge.get("name", "vn_in")
    if name != "vn_in":
        raise ValueError(f"Unknown bridge '{name}'. Known: ['vn_in']")

    frame_hidden = cfg_bridge.get("frame_hidden", None)
    return VNInBridge(
        C=C,
        dim_invariant=dim_invariant,
        d_mamba=d_mamba,
        frame_hidden=None if frame_hidden is None else int(frame_hidden),
    )
