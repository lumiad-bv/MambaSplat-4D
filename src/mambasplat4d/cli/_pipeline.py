"""Pure helpers shared by train.py and eval_rotation.py. Not a Hydra entrypoint."""
from __future__ import annotations

from typing import Any, Dict

import torch
from omegaconf import DictConfig


def get_device(cfg: DictConfig) -> torch.device:
    """`cfg.hardware.device`: "auto" = CUDA if available else CPU; else `torch.device(str)`."""
    device_str = cfg.hardware.device
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def normalize_checkpoint_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Strip torch.compile (``_orig_mod.``) and DDP (``module.``) prefixes. Identity if clean."""
    prefixes = ("_orig_mod.", "module.")
    normalized: Dict[str, Any] = {}
    for key, value in state_dict.items():
        nk = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
                    changed = True
        normalized[nk] = value
    return normalized
