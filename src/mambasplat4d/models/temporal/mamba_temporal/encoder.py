"""MambaTemporalEncoder: headless Mamba, [B, T, d] -> [B, hidden_dim]."""
import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig

from ..base import TemporalEncoder
from ...registry import register_temporal

from .mamba_blocks import MixerModel
# re-export from mamba_ssm-free ..pos_encoding
from ..pos_encoding import SinusoidalPositionalEncoding, LearnablePositionalEncoding  # noqa: F401


@register_temporal("mamba")
class MambaTemporalEncoder(TemporalEncoder):
    """Mamba SSM over per-frame embeddings; head lives in pipeline. cfg: configs/temporal/mamba.yaml."""

    def __init__(self, cfg: DictConfig):
        super().__init__()

        input_dim = cfg.input_dim
        hidden_dim = cfg.hidden_dim
        n_layers = cfg.n_layers
        pooling = cfg.pooling
        dropout = cfg.dropout
        drop_path = cfg.drop_path
        rms_norm = cfg.rms_norm
        max_seq_len = cfg.max_seq_len
        pos_encoding = cfg.pos_encoding

        self.hidden_dim = hidden_dim
        self.pooling_mode = pooling

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if pos_encoding == "sinusoidal":
            self.pos_encoder = SinusoidalPositionalEncoding(
                hidden_dim, max_seq_len, dropout
            )
        elif pos_encoding == "learnable":
            self.pos_encoder = LearnablePositionalEncoding(
                hidden_dim, max_seq_len, dropout
            )
        else:
            self.pos_encoder = nn.Identity()

        self.mamba = MixerModel(
            d_model=hidden_dim,
            n_layer=n_layers,
            rms_norm=rms_norm,
            drop_out_in_block=dropout,
            drop_path=drop_path,
            fused_add_norm=False,
        )

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """x (B, T, input_dim), mask (B, T) bool -> (B, hidden_dim)."""
        B, T, _ = x.shape

        x = self.input_proj(x)

        x = self.pos_encoder(x)

        x = self.mamba(x)

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            if self.pooling_mode == "max":
                x = x.masked_fill(~mask_expanded, float('-inf'))
            elif self.pooling_mode == "mean":
                x = x * mask_expanded.float()

        if self.pooling_mode == "max":
            features = x.max(dim=1)[0]
        elif self.pooling_mode == "mean":
            if mask is not None:
                features = x.sum(dim=1) / mask.unsqueeze(-1).float().sum(dim=1).clamp(min=1)
            else:
                features = x.mean(dim=1)
        elif self.pooling_mode == "last":
            if mask is not None:
                lengths = mask.sum(dim=1).long() - 1
                features = x[torch.arange(B, device=x.device), lengths]
            else:
                features = x[:, -1]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling_mode}")

        return features
