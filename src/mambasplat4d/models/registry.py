"""Encoder registry: decorator registration, Hydra-config factories."""
import importlib
from typing import Dict, Type

import torch.nn as nn
from omegaconf import DictConfig


SPATIAL_REGISTRY: Dict[str, Type[nn.Module]] = {}
TEMPORAL_REGISTRY: Dict[str, Type[nn.Module]] = {}

# mamba_ssm encoders imported lazily: import compiles CUDA kernels, may segfault.
_TEMPORAL_LAZY: Dict[str, str] = {
    "vn_mamba": ".temporal.vn_mamba",
    "mamba": ".temporal.mamba_temporal",
}


def register_spatial(name: str):
    """Register spatial encoder class."""
    def decorator(cls):
        SPATIAL_REGISTRY[name] = cls
        return cls
    return decorator


def register_temporal(name: str):
    """Register temporal encoder class."""
    def decorator(cls):
        TEMPORAL_REGISTRY[name] = cls
        return cls
    return decorator


def build_spatial_encoder(cfg: DictConfig) -> nn.Module:
    """Build spatial encoder from cfg.spatial."""
    name = cfg.name
    if name not in SPATIAL_REGISTRY:
        raise ValueError(
            f"Unknown spatial encoder '{name}'. "
            f"Available: {list(SPATIAL_REGISTRY.keys())}"
        )
    return SPATIAL_REGISTRY[name](cfg)


def build_temporal_encoder(cfg: DictConfig) -> nn.Module:
    """Build temporal encoder from cfg.temporal; lazy-imports mamba encoders."""
    name = cfg.name
    if name not in TEMPORAL_REGISTRY and name in _TEMPORAL_LAZY:
        importlib.import_module(_TEMPORAL_LAZY[name], package=__package__)
    if name not in TEMPORAL_REGISTRY:
        raise ValueError(
            f"Unknown temporal encoder '{name}'. "
            f"Available: {list(TEMPORAL_REGISTRY.keys())}"
        )
    return TEMPORAL_REGISTRY[name](cfg)
