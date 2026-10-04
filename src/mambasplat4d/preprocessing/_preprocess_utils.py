"""Low-level helpers for preprocessing scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mambasplat4d.paths import REPO_ROOT, data_root
from mambasplat4d.utils.fps import farthest_point_sample_np


def splits_dir(*parts: str) -> Path:
    """`<REPO_ROOT>/assets/splits/<parts>`."""
    return REPO_ROOT.joinpath("assets", "splits", *parts)


def default_data_path(*parts: str) -> str:
    """`data_root()/<parts>`; AEROSPLAT_DATA_ROOT overrides."""
    return str(data_root().joinpath(*parts))


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    """1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(-x))


def fps_subsample_np(xyz: np.ndarray, npoint: int) -> np.ndarray:
    """(N, 3) float32 -> (npoint,) int64 FPS indices; identity if N <= npoint."""
    return farthest_point_sample_np(xyz, npoint)
