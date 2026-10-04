"""VN-Mamba: VNInBridge (complete SO(3) invariants, 3C-3 vs C from norms) -> Mamba SSM."""

from typing import Optional, Tuple

import torch.nn as nn
from torch import Tensor

from .base import TemporalEncoder
from ..registry import register_temporal
from .mamba_temporal.mamba_blocks import MixerModel
from .mamba_temporal.encoder import (
    SinusoidalPositionalEncoding,
    LearnablePositionalEncoding,
)
from .vn_bridge import build_bridge
from omegaconf import DictConfig
from ...utils.feature_config import get_invariant_dim


@register_temporal("vn_mamba")
class VNMambaTemporalEncoder(TemporalEncoder):
    """VNInBridge -> pos-enc -> Mamba -> pool (mean|last).
    Returns (features (B, d_mamba), h_frames (B*T, d_mamba))."""

    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.d_mamba = cfg.hidden_dim
        n_layers = cfg.n_layers
        dropout = cfg.get("dropout", 0.1)
        drop_path = cfg.get("drop_path", 0.1)
        pos_encoding = cfg.get("pos_encoding", "sinusoidal")
        max_seq_len = cfg.get("max_seq_len", 512)
        # "mean" (default) or "last" (causal final state). Fixed-length clips: no padding handling.
        self.pooling = cfg.get("pooling", "mean")
        if self.pooling not in ("mean", "last"):
            raise ValueError(
                f"temporal.pooling must be 'mean' or 'last', got {self.pooling!r}"
            )

        C = cfg.get("input_dim", 64)
        feature_mode = cfg.get("feature_mode", "X")
        dim_invariant = get_invariant_dim(feature_mode)

        self.bridge = build_bridge(
            cfg.get("bridge", None),
            C=C,
            dim_invariant=dim_invariant,
            d_mamba=self.d_mamba,
        )

        if pos_encoding == "sinusoidal":
            self.pos_encoder = SinusoidalPositionalEncoding(
                self.d_mamba, max_seq_len, dropout
            )
        elif pos_encoding == "learnable":
            self.pos_encoder = LearnablePositionalEncoding(
                self.d_mamba, max_seq_len, dropout
            )
        else:
            self.pos_encoder = nn.Identity()

        self.mamba = MixerModel(
            d_model=self.d_mamba,
            n_layer=n_layers,
            drop_out_in_block=dropout,
            drop_path=drop_path,
            fused_add_norm=False,
        )

        self.output_dim = self.d_mamba

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        *,
        X_inv: Optional[Tensor] = None,
        gaussian_data: Optional[dict] = None,
        T: int = 1,
    ) -> Tuple[Tensor, Tensor]:
        """x = V (B*T, C, 3), X_inv (B*T, C, F), mask (B*T, N), T timesteps.
        Returns features (B, d_mamba) pooled per cfg.pooling, h_frames (B*T, d_mamba)."""
        BT = x.shape[0]
        B = BT // T

        h_frames = self.bridge(x, X_inv, gaussian_data, mask)  # (B*T, d_mamba)

        h_seq = h_frames.reshape(B, T, self.d_mamba)

        h_seq = self.pos_encoder(h_seq)

        h_seq = self.mamba(h_seq)  # (B, T, d_mamba)

        if self.pooling == "last":
            features = h_seq[:, -1]  # (B, d_mamba)
        else:
            features = h_seq.mean(dim=1)  # (B, d_mamba)

        return features, h_frames

    def bridge_parameters(self):
        """Bridge params (differential LR)."""
        return self.bridge.parameters()

    def mamba_parameters(self):
        """Mamba + pos-enc params (differential LR)."""
        yield from self.pos_encoder.parameters()
        yield from self.mamba.parameters()
