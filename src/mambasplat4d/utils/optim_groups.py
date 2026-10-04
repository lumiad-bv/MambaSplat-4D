"""AdamW [decay, no_decay] group split: biases, 1-D norm weights and
``_no_weight_decay`` params (mamba ``A_log``, ``D``) skip decay.
Empty groups kept so wandb log keys stay stable."""
from __future__ import annotations

from typing import Iterable, List, Dict, Any

import torch.nn as nn


def _is_no_decay(p: nn.Parameter) -> bool:
    """``_no_weight_decay`` flag (mamba ``A_log``, ``D``) or dim <= 1 (biases, norm weights)."""
    if getattr(p, '_no_weight_decay', False):
        return True
    return p.dim() <= 1


def split_no_decay(
    params: Iterable[nn.Parameter],
    lr: float,
    weight_decay: float,
) -> List[Dict[str, Any]]:
    """[decay, no_decay] groups; frozen params dropped."""
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    for p in params:
        if not p.requires_grad:
            continue
        (no_decay if _is_no_decay(p) else decay).append(p)
    return [
        {'params': decay,    'lr': lr, 'weight_decay': float(weight_decay)},
        {'params': no_decay, 'lr': lr, 'weight_decay': 0.0},
    ]
