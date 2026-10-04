"""Preprocessed AeroSplat-4D dataset: deterministic version selection, clip assembly,
clip-level augmentation.

Each .pt holds N pre-subsampled versions of one frame (preprocess_aerosplat4d).
Frames grouped by (asset_id, sweep_id), then sliding-window clips.

Layout:
    data_root/
      config.json
      category.txt
      train.json / val.json / test.json
      train/{class_idx}/{name}_frame_{XXXX}_s{sweep}.pt
      val/...
      test/...
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms.rotation import rotate_quaternion
from mambasplat4d.utils.rotation_utils import random_rotation_matrix_np


def _item_seed(seed: int, epoch: int, idx: int) -> int:
    """32-bit per-item seed: SplitMix64-style mix of (seed, epoch, idx), PYTHONHASHSEED-independent."""
    MASK64 = 0xFFFFFFFFFFFFFFFF
    h = ((int(seed)  * 0x9E3779B97F4A7C15
        + int(epoch) * 0xBF58476D1CE4E5B9
        + int(idx)   * 0x94D049BB133111EB) & MASK64)
    return h & 0xFFFFFFFF

# (N, 14) layout, matches preprocess_aerosplat4d.py
FEATURE_SLICES = {
    "position": (0, 3),
    "quaternion": (3, 7),
    "scale": (7, 10),
    "opacity": (10, 11),
    "sh_dc": (11, 14),
}

ALL_GAUSSIAN_KEYS = ("position", "quaternion", "scale", "opacity", "sh_dc")

# {model_id}_frame_{XXXX}[_s{sweep}]
_NAME_RE = re.compile(r'^(.+)_frame_(\d+)(?:_s(\d+))?$')


class IsaacSimPreDataset(Dataset):
    """Preprocessed AeroSplat-4D clips.

    num_points: Gaussians per frame, must match preprocessing.
    seq_len: clip frames; None = full sequence.
    frame_interval: frame stride within clip.
    train_stride: train sliding-window stride (1 = max overlap); test stride = seq_len.
    train_rotation_mode: 'none' | 'z' | 'so3', train-only clip rotation.
    scale_range: (min, max) clip scale augmentation; scale_per_axis: 3 independent factors.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        num_points: Optional[int] = 1024,
        seq_len: Optional[int] = 24,
        feature_keys: Sequence[str] = ALL_GAUSSIAN_KEYS,
        frame_interval: int = 1,
        n_versions: int = 25,
        train_stride: int = 1,
        train_rotation_mode: str = "none",
        scale_range: tuple = (1.0, 1.0),
        scale_per_axis: bool = True,
        version_per_frame: bool = True,
        seed: int = 0,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.num_points = num_points
        self.seq_len = seq_len
        self.feature_keys = tuple(feature_keys)
        self.frame_interval = frame_interval
        self.n_versions = n_versions
        self.train_stride = max(1, train_stride)
        self.train_rotation_mode = train_rotation_mode
        self.scale_range = tuple(scale_range)
        self.scale_per_axis = scale_per_axis
        self.version_per_frame = version_per_frame
        self.train = (split == "train")
        self._epoch = 0
        self._seed = int(seed)

        self.class_name_map = self._read_categories()
        self.class_names = [self.class_name_map[k] for k in sorted(self.class_name_map)]
        self.num_classes = len(self.class_names)

        manifest_path = self.data_root / f"{self.split}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
        self.samples = json.loads(manifest_path.read_text())

        self.sequences = self._group_sequences(self.samples)

        self.index_map = self._build_index_map()

        total_frames = sum(len(s["paths"]) for s in self.sequences)
        print(f"[IsaacSimPreDataset] {self.split}: {len(self.sequences)} sequences, "
              f"{len(self.index_map)} clips from {total_frames} frames "
              f"(avg {total_frames / max(len(self.sequences), 1):.0f} frames/seq)")

    def _read_categories(self) -> Dict[int, str]:
        cat_file = self.data_root / "category.txt"
        if not cat_file.exists():
            return {}
        class_names: Dict[int, str] = {}
        for line in cat_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            idx_str, name = line.split(",", 1)
            class_names[int(idx_str)] = name
        return class_names

    def _group_sequences(self, samples: list) -> list:
        """Group frames into sequences by (class, asset, sweep)."""
        groups: Dict[tuple, list] = defaultdict(list)
        group_meta: Dict[tuple, dict] = {}

        for rel_path in samples:
            class_idx = int(rel_path.split("/")[1])  # split/{class_idx}/{name}.pt
            stem = Path(rel_path).stem
            m = _NAME_RE.match(stem)
            if m is None:
                print(f"[IsaacSimPreDataset] WARNING: cannot parse '{stem}', skipping")
                continue
            asset_id = m.group(1)
            frame_num = int(m.group(2))
            sweep_id = int(m.group(3)) if m.group(3) is not None else 0
            key = (class_idx, asset_id, sweep_id)
            groups[key].append((frame_num, rel_path))
            if key not in group_meta:
                class_name = self.class_name_map.get(class_idx, str(class_idx))
                group_meta[key] = {
                    "label": class_idx,
                    "class_name": class_name,
                    "asset_id": asset_id,
                    "sweep_id": sweep_id,
                }

        sequences = []
        for key in sorted(groups.keys()):
            frames = groups[key]
            frames.sort(key=lambda x: x[0])
            meta = group_meta[key]
            sequences.append({
                "paths": [self.data_root / rel for _, rel in frames],
                "label": meta["label"],
                "class_name": meta["class_name"],
                "model_id": f"{meta['asset_id']}_s{meta['sweep_id']}",
            })
        return sequences

    def _build_index_map(self) -> list:
        """(seq_idx, start_frame) clip index."""
        index_map = []
        if self.seq_len is None:
            for seq_idx in range(len(self.sequences)):
                index_map.append((seq_idx, 0))
            return index_map

        for seq_idx, seq in enumerate(self.sequences):
            nframes = len(seq["paths"])
            span = self.frame_interval * (self.seq_len - 1)
            if nframes <= span:
                continue
            if self.train:
                for t in range(0, nframes - span, self.train_stride):
                    index_map.append((seq_idx, t))
            else:
                for t in range(0, nframes - span, self.seq_len):
                    index_map.append((seq_idx, t))
        return index_map

    def set_epoch(self, epoch: int) -> None:
        """Epoch drives deterministic version selection."""
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Per-item RNG: same (seed, epoch, idx) gives same version and augmentation.
        item_rng = np.random.RandomState(_item_seed(self._seed, self._epoch, idx))

        seq_idx, start = self.index_map[idx]
        seq = self.sequences[seq_idx]
        label = seq["label"]
        class_name = seq["class_name"]
        model_id = seq["model_id"]
        all_paths = seq["paths"]

        if self.seq_len is not None:
            paths = [all_paths[start + i * self.frame_interval]
                     for i in range(self.seq_len)]
        else:
            paths = all_paths

        T = len(paths)
        target_n = self.num_points or 1024

        frames = {k: [] for k in self.feature_keys}
        frame_masks = []

        for frame_idx, pt_path in enumerate(paths):
            cached = torch.load(pt_path, map_location="cpu", weights_only=True)
            versions = cached["versions"]  # (n_versions, num_points, 14)

            n_avail = versions.shape[0]
            if self.version_per_frame:
                # version varies per frame (legacy)
                global_idx = idx * T + frame_idx
            else:
                # one version per clip: all T frames share preprocessing seed
                global_idx = idx
            version_idx = (self._epoch * len(self) * T + global_idx) % n_avail
            feat = versions[version_idx]  # (num_points, 14)

            data: Dict[str, np.ndarray] = {}
            for key, (s, e) in FEATURE_SLICES.items():
                data[key] = feat[:, s:e].numpy()

            N = data["position"].shape[0]

            # Pad / truncate to target_n; no-op if preprocessing matched
            if N > target_n:
                indices = item_rng.choice(N, target_n, replace=False)
                data = {k: v[indices] for k, v in data.items()}
                mask = np.ones(target_n, dtype=bool)
            elif N < target_n:
                pad_n = target_n - N
                mask = np.zeros(target_n, dtype=bool)
                mask[:N] = True
                for key in data:
                    shape = list(data[key].shape)
                    shape[0] = pad_n
                    data[key] = np.concatenate(
                        [data[key], np.zeros(shape, dtype=data[key].dtype)], axis=0
                    )
            else:
                mask = np.ones(target_n, dtype=bool)

            for k in self.feature_keys:
                frames[k].append(torch.from_numpy(data[k]))
            frame_masks.append(torch.from_numpy(mask))

        # (T, N, F) per key, (T, N) mask
        result = {k: torch.stack(frames[k]) for k in self.feature_keys}
        result["mask"] = torch.stack(frame_masks)
        result["label"] = label
        result["class_name"] = class_name
        result["subject"] = model_id
        result["num_frames"] = T
        result["video_idx"] = seq_idx

        if self.train:
            self._apply_clip_augmentation(result, item_rng)

        return result

    def _apply_clip_augmentation(self, result: Dict[str, Any],
                                 rng: np.random.RandomState) -> None:
        """In-place clip scale + rotation augmentation; all randomness from per-item `rng`."""
        lo, hi = self.scale_range
        if lo < hi:
            if self.scale_per_axis:
                scale = torch.from_numpy(
                    rng.uniform(lo, hi, size=3).astype(np.float32)
                )
            else:
                s = rng.uniform(lo, hi)
                scale = torch.tensor([s, s, s], dtype=torch.float32)
            result["position"] = result["position"] * scale
            if "scale" in result and isinstance(result["scale"], torch.Tensor):
                result["scale"] = result["scale"] * scale

        if self.train_rotation_mode == "none":
            return

        if self.train_rotation_mode == "z":
            angle = rng.uniform(0, 2 * np.pi)
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        elif self.train_rotation_mode == "so3":
            R = random_rotation_matrix_np(rng=rng)
        else:
            raise ValueError(f"Unknown train_rotation_mode: {self.train_rotation_mode!r}")

        # numpy matmul keeps bit-identity with eval_rotation cache
        pos_np = np.ascontiguousarray(result["position"].numpy()) @ R.T
        result["position"] = torch.from_numpy(pos_np)

        if "quaternion" in result and isinstance(result["quaternion"], torch.Tensor):
            q = result["quaternion"]
            orig_shape = q.shape
            q_flat = q.reshape(-1, 4).numpy()
            q_rot = rotate_quaternion(q_flat, R)
            result["quaternion"] = torch.from_numpy(q_rot).reshape(orig_shape)
