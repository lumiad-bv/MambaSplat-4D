"""Temporal encoder base: [B, T, d] -> [B, d]."""
from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn, Tensor


class TemporalEncoder(ABC, nn.Module):
    """Temporal encoder interface."""

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """x (B,T,d) per-frame embeddings, mask (B,T) bool -> (B,d)."""
        ...
