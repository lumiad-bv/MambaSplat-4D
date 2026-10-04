"""Config loading helpers."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict


def expand_env(obj: Any) -> Any:
    """Expand ${VAR} in all strings of YAML tree."""
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_env(v) for v in obj]
    return obj


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge; override wins at leaves."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_main_config(configs_dir: Path | str) -> Dict[str, Any]:
    """Load configs_dir/main.yaml; {} if missing."""
    import yaml

    configs_dir = Path(configs_dir)
    main_path = configs_dir / "main.yaml"
    if not main_path.exists():
        print(f"[CONFIG] WARNING: main.yaml not found at {main_path}, using empty base")
        return {}

    with open(main_path, "r") as f:
        cfg = expand_env(yaml.safe_load(f))
    return cfg if isinstance(cfg, dict) else {}


def load_merged_config(
    configs_dir: Path | str, mode_config_name: str
) -> Dict[str, Any] | None:
    """main.yaml deep-merged with mode config (e.g. "config_batch.yaml"). None if mode config missing."""
    import yaml

    configs_dir = Path(configs_dir)
    base = load_main_config(configs_dir)

    mode_path = configs_dir / mode_config_name
    if not mode_path.exists():
        print(f"[CONFIG] ERROR: {mode_config_name} not found at {mode_path}")
        return None

    with open(mode_path, "r") as f:
        mode_cfg = expand_env(yaml.safe_load(f))

    if not isinstance(mode_cfg, dict):
        mode_cfg = {}

    return deep_merge(base, mode_cfg)


def resolve_fps_from_config(cfg: Dict[str, Any] | None, default: float = 30.0) -> float:
    """cfg['fps'] if valid, else default."""
    if not isinstance(cfg, dict):
        return float(default)

    fps_value = cfg.get("fps")
    if fps_value is not None:
        try:
            fps_val = float(fps_value)
            if fps_val > 0.0:
                return fps_val
        except (TypeError, ValueError):
            pass

    return float(default)


def resolve_timeline_from_config(cfg: Dict[str, Any] | None, default_num_frames: float = 120.0) -> Dict[str, float]:
    """Timeline from render.num_frames: start=1, end=num_frames, middle=(1+end)/2."""
    if not isinstance(cfg, dict):
        return {"start_frame": 1.0, "middle_frame": 60.5, "end_frame": 120.0}

    num_frames = float(default_num_frames)
    render_cfg = cfg.get("render", {})
    if isinstance(render_cfg, dict):
        nf = render_cfg.get("num_frames")
        if nf is not None:
            try:
                nf_val = float(nf)
                if nf_val > 0:
                    num_frames = nf_val
            except (TypeError, ValueError):
                pass

    start_frame = 1.0
    end_frame = num_frames
    middle_frame = (start_frame + end_frame) / 2.0

    return {
        "start_frame": start_frame,
        "middle_frame": middle_frame,
        "end_frame": end_frame,
    }
