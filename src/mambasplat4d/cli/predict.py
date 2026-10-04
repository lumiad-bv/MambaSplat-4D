"""Classify one reconstructed sequence with a trained MambaSplat-4D checkpoint.

    python -m mambasplat4d.cli.predict experiment=aerosplat4d \\
        +checkpoint=/path/to/best_model.pt \\
        +input=/path/to/<sequence>/        # frame_XXXX.ply or .pt from batch_reconstruct

Preprocessed as training data (per-frame normalisation, rand(8192) -> FPS(1200) ->
rand(1024)), cut into non-overlapping dataset.seq_len clips, clip logits averaged.
"""
from __future__ import annotations

import re
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from mambasplat4d import paths
from mambasplat4d.categories import CATEGORY_NAMES
from mambasplat4d.cli._pipeline import get_device, normalize_checkpoint_keys
from mambasplat4d.datasets.isaacsim_pre import FEATURE_SLICES
from mambasplat4d.models.pipeline import ClassificationPipeline
from mambasplat4d.preprocessing.preprocess_aerosplat4d import (
    generate_version,
    normalize_gaussians,
    read_raw_file,
)
from mambasplat4d.utils.feature_config import FEATURE_MODES

_FRAME_RE = re.compile(r"frame_(\d+)")


def load_sequence(seq_dir: Path, seed: int) -> torch.Tensor:
    files = sorted(
        (p for p in seq_dir.iterdir() if p.suffix in (".ply", ".pt") and _FRAME_RE.search(p.stem)),
        key=lambda p: int(_FRAME_RE.search(p.stem).group(1)),
    )
    if not files:
        raise FileNotFoundError(f"no frame_XXXX.ply/.pt files in {seq_dir}")
    frames = []
    for i, f in enumerate(files):
        data = normalize_gaussians(read_raw_file(f))
        frames.append(generate_version(data, np.random.RandomState(seed + i)))
    return torch.from_numpy(np.stack(frames))  # (T_total, 1024, 14)


def to_clips(frames: torch.Tensor, seq_len: int, num_points: int, feature_keys, rng) -> dict:
    T_total = frames.shape[0]
    n_clips = max(1, T_total // seq_len)
    clips = [frames[i * seq_len:(i + 1) * seq_len] for i in range(n_clips)]
    if clips[-1].shape[0] < seq_len:
        clips[-1] = frames[-seq_len:] if T_total >= seq_len else frames
    T = clips[0].shape[0]
    x = torch.stack(clips)  # (B, T, N, 14)
    if x.shape[2] > num_points:
        sel = torch.from_numpy(rng.choice(x.shape[2], num_points, replace=False))
        x = x[:, :, sel]
    B, T, N, _ = x.shape
    data = {k: x[..., s:e].reshape(B * T, N, -1) for k, (s, e) in FEATURE_SLICES.items() if k in feature_keys}
    mask = torch.ones(B * T, N, dtype=torch.bool)
    return data, mask, T


@hydra.main(config_path=paths.CONFIG_DIR, config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    device = get_device(cfg)
    seq_dir = Path(str(cfg.input)).expanduser()
    feature_keys = FEATURE_MODES[cfg.dataset.get("feature_mode", "X")]

    pipeline = ClassificationPipeline(cfg).to(device).eval()
    state = torch.load(str(cfg.checkpoint), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    missing, unexpected = pipeline.load_state_dict(normalize_checkpoint_keys(state), strict=False)
    if missing or unexpected:
        print(f"[predict] missing={len(missing)} unexpected={len(unexpected)} keys")

    frames = load_sequence(seq_dir, int(cfg.seed))
    data, mask, T = to_clips(frames, int(cfg.dataset.seq_len), int(cfg.dataset.num_points),
                             feature_keys, np.random.RandomState(int(cfg.seed)))
    data = {k: v.to(device) for k, v in data.items()}
    with torch.no_grad():
        logits = pipeline(data, mask=mask.to(device), T=T)["logits"].float()
    probs = F.softmax(logits, dim=-1).mean(dim=0)

    cat_file = seq_dir.parent.parent / "category.txt"
    names = dict(CATEGORY_NAMES) if probs.numel() == len(CATEGORY_NAMES) else {}
    if cat_file.exists():
        for line in cat_file.read_text().splitlines():
            if "," in line:
                i, n = line.split(",", 1)
                names[int(i)] = n.strip()
    pred = int(probs.argmax())
    print(f"{seq_dir.name}: {names.get(pred, pred)}")
    for i, p in enumerate(probs.tolist()):
        print(f"  {names.get(i, i):<12} {p:.3f}")


if __name__ == "__main__":
    main()
