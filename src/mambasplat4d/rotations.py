"""Shared rotation-matrix cache: resolve, generate, verify before eval.

All models in a table score on identical rotations. Cache keyed by (split, seed,
seq_len, frame_interval, n_trials); miss is a hard error, no inline fallback.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from mambasplat4d import paths

GENERATOR_MODULE = "mambasplat4d.cli.generate_rotations"
DEFAULT_N_TRIALS = 3
REQUIRED_MODES = ("z", "so3", "so3_per_frame")


class RotationCacheError(RuntimeError):
    """Cache missing, unreadable, or mismatched with eval."""


@dataclass(frozen=True)
class RotationKey:
    split: str
    seed: int
    seq_len: int
    frame_interval: int
    n_trials: int

    @property
    def filename(self) -> str:
        return (
            f"rotation_matrices_{self.split}_seed{self.seed}"
            f"_T{self.seq_len}_dt{self.frame_interval}_n{self.n_trials}.pt"
        )

    def path(self, rotations_dir: str | Path) -> Path:
        return Path(rotations_dir) / self.filename


def _int(value, default=None):
    if value is None:
        return default
    return int(value)


def rotations_dir_from(cfg=None) -> Path:
    if cfg is not None:
        configured = (
            cfg.get("paths", {}).get("rotations_dir", None)
            if hasattr(cfg, "get")
            else None
        )
        if configured:
            return Path(str(configured))
    return paths.rotations_dir()


def data_root_from(cfg) -> Path:
    return Path(str(cfg.dataset.data_root))


def key_from_cfg(
    cfg, split: str, n_trials: int | None = None, seed: int | None = None
) -> RotationKey:
    return RotationKey(
        split=str(split),
        seed=_int(seed, _int(cfg.get("seed", 42))),
        seq_len=_int(cfg.dataset.get("seq_len", None), 12),
        frame_interval=_int(cfg.dataset.get("frame_interval", None), 1),
        n_trials=_int(n_trials, DEFAULT_N_TRIALS),
    )


def expected_clips(data_root: str | Path, key: RotationKey) -> int:
    from mambasplat4d.cli.generate_rotations import count_clips

    n_clips, _ = count_clips(str(data_root), key.split, key.seq_len, key.frame_interval)
    return int(n_clips)


def generate(
    key: RotationKey, data_root: str | Path, rotations_dir: str | Path, log=print
) -> Path:
    out = key.path(rotations_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        sys.executable,
        "-m",
        GENERATOR_MODULE,
        "--data-root",
        str(data_root),
        "--split",
        key.split,
        "--seed",
        str(key.seed),
        "--n-trials",
        str(key.n_trials),
        "--seq-len",
        str(key.seq_len),
        "--frame-interval",
        str(key.frame_interval),
        "--output",
        str(out),
    ]
    log(f"[rotations] generate {out.name}")
    completed = subprocess.run(argv, check=False)
    if completed.returncode != 0:
        raise RotationCacheError(
            f"rotation-cache generation failed (exit {completed.returncode}) for {out}\n"
            f"  argv: {' '.join(argv)}"
        )
    if not out.exists():
        raise RotationCacheError(
            f"rotation-cache generator reported success but {out} is absent"
        )
    return out


def verify(
    key: RotationKey,
    rotations_dir: str | Path,
    data_root: str | Path,
    n_clips: int | None = None,
    modes: Sequence[str] = REQUIRED_MODES,
    log=print,
) -> Mapping:
    """Return cache metadata or raise RotationCacheError. No fallback."""
    import torch

    path = key.path(rotations_dir)
    if not path.exists():
        raise RotationCacheError(
            f"rotation cache missing: {path}\n"
            f"  needed for split={key.split} seed={key.seed} T={key.seq_len} "
            f"dt={key.frame_interval} n_trials={key.n_trials}\n"
            f"  generate it with: {sys.executable} -m {GENERATOR_MODULE} "
            f"--data-root {data_root} --split {key.split} --seed {key.seed} "
            f"--n-trials {key.n_trials} --seq-len {key.seq_len} "
            f"--frame-interval {key.frame_interval} --output {path}"
        )
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RotationCacheError(f"rotation cache unreadable: {path}: {exc}") from exc

    meta = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    if not meta:
        raise RotationCacheError(f"rotation cache has no metadata block: {path}")

    problems = []
    if _int(meta.get("seq_len"), -1) != key.seq_len:
        problems.append(f"seq_len {meta.get('seq_len')} != {key.seq_len}")
    if _int(meta.get("frame_interval"), -1) != key.frame_interval:
        problems.append(
            f"frame_interval {meta.get('frame_interval')} != {key.frame_interval}"
        )
    if _int(meta.get("n_trials"), -1) < key.n_trials:
        problems.append(f"n_trials {meta.get('n_trials')} < {key.n_trials}")
    if _int(meta.get("seed"), -1) != key.seed:
        problems.append(f"seed {meta.get('seed')} != {key.seed}")

    expected = expected_clips(data_root, key) if n_clips is None else int(n_clips)
    cached_clips = _int(meta.get("n_clips"), -1)
    if cached_clips != expected:
        problems.append(f"n_clips {cached_clips} != dataset clips {expected}")

    for mode in modes:
        for trial in range(key.n_trials):
            tensor = payload.get(f"{mode}_{trial}")
            if tensor is None:
                problems.append(f"missing entry {mode}_{trial}")
                continue
            if int(tensor.shape[0]) != expected:
                problems.append(
                    f"{mode}_{trial} has {int(tensor.shape[0])} clips, dataset has {expected}"
                )

    if problems:
        raise RotationCacheError(
            f"rotation cache is not usable for this eval: {path}\n  "
            + "\n  ".join(problems)
            + "\n  Delete the file and re-run, or fix the (split, seed, T, dt, n_trials) tuple."
        )
    log(f"[rotations] verified {path.name} (n_clips={expected})")
    return meta


def ensure(
    cfg,
    split: str,
    n_trials: int | None = None,
    resume: bool = True,
    seed: int | None = None,
    rotations_dir: str | Path | None = None,
    data_root: str | Path | None = None,
    n_clips: int | None = None,
    dry_run: bool = False,
    log=print,
) -> Path:
    """Resolve cache for (split, seed, T, dt, n_trials); generate if absent."""
    key = key_from_cfg(cfg, split, n_trials=n_trials, seed=seed)
    rot_dir = (
        Path(rotations_dir) if rotations_dir is not None else rotations_dir_from(cfg)
    )
    root = Path(data_root) if data_root is not None else data_root_from(cfg)
    path = key.path(rot_dir)

    if not root.exists():
        raise RotationCacheError(f"dataset root does not exist: {root}")
    split_index = root / f"{key.split}.json"
    if not split_index.exists():
        raise RotationCacheError(
            f"{split_index} does not exist, so the clip count that keys the shared cache "
            f"cannot be derived. {GENERATOR_MODULE} enumerates clips from "
            f"<data_root>/<split>.json."
        )

    if dry_run:
        state = "present" if path.exists() else "MISSING (would be generated)"
        log(f"[rotations] {path} — {state}")
        return path

    if path.exists() and not resume:
        log(f"[rotations] regenerate (resume=False) {path.name}")
        generate(key, root, rot_dir, log=log)
    elif not path.exists():
        generate(key, root, rot_dir, log=log)

    verify(key, rot_dir, root, n_clips=n_clips, log=log)
    return path

