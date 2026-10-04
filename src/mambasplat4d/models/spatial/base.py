"""Spatial encoder base: per-frame Gaussians -> [B, embed_dim]."""
from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn, Tensor


class SpatialEncoder(ABC, nn.Module):
    """Spatial encoder interface."""

    @abstractmethod
    def forward(
        self,
        gaussian_data: dict,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """gaussian_data: position (B,N,3), quaternion (B,N,4), scale (B,N,3),
        opacity (B,N,1), optional sh_dc (B,N,3); mask (B,N) bool.
        Returns rotation-invariant (B, embed_dim)."""
        ...
