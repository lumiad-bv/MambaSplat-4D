from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = str(REPO_ROOT / "configs")


def _from_env(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value).expanduser() if value else default


def data_root() -> Path:
    return _from_env("AEROSPLAT_DATA_ROOT", REPO_ROOT / "dataset")


def results_root() -> Path:
    return _from_env("AEROSPLAT_RESULTS_ROOT", REPO_ROOT / "results")


def rotations_dir() -> Path:
    return _from_env("AEROSPLAT_ROTATIONS_DIR", REPO_ROOT / "dataset" / "rotation_cache")
