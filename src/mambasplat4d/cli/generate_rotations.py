#!/usr/bin/env python3
"""Pre-generate per-clip rotation matrices so every model is evaluated on identical rotations.

Output .pt:
    '{mode}_{trial}': Tensor of shape (n_clips, 3, 3) or (n_clips, T, 3, 3)
    'metadata': dict with generation parameters

Usage:
    python -m mambasplat4d.cli.generate_rotations \
        --data-root /path/to/aerosplat4d_pre \
        --split test --seed 42 --n-trials 3 \
        --seq-len 12 --frame-interval 1 \
        --output rotation_cache/rotation_matrices_test_seed42_T12_dt1_n3.pt
"""

import argparse
import json
import math
import os
import re
from collections import defaultdict

import torch

# Haar-uniform SO(3) sampler vendored from PyTorch3D (Shoemake 1992 quaternions).
# Old axis-angle sampler with angle ~ U[0, 2π] was not Haar (Miles 1965).
from mambasplat4d.utils.pytorch3d_transforms import random_rotations  # noqa: E402


def random_rotation_matrix_z_batch(n, dtype=torch.float32):
    """Random Z rotations, (n, 3, 3). Uniform angle on [0, 2π) is Haar on SO(2)."""
    angles = torch.rand(n, dtype=dtype) * 2 * math.pi
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    R = torch.eye(3, dtype=dtype).unsqueeze(0).repeat(n, 1, 1)
    R[:, 0, 0] = cos_a
    R[:, 0, 1] = -sin_a
    R[:, 1, 0] = sin_a
    R[:, 1, 1] = cos_a
    return R


def random_rotation_matrix_so3_batch(n, dtype=torch.float32):
    """Haar-uniform SO(3) rotations, (n, 3, 3). Wraps vendored `pytorch3d.random_rotations`."""
    return random_rotations(n, dtype=dtype)


# {name}_frame_{num}[_s{sweep}]: same stem grammar `IsaacSimPreDataset` groups by.
_NAME_RE = re.compile(r"^(.+)_frame_(\d+)(?:_s(\d+))?$")


def count_clips(data_root, split, seq_len, frame_interval):
    """Count clips as `IsaacSimPreDataset` indexes them.

    Reads `<data_root>/<split>.json`, groups by (class, model, sweep); one clip per
    non-overlapping `seq_len` window at stride `frame_interval`.
    Returns (n_clips, n_sequences); n_clips == `len(dataset)`, the cache index.
    """
    split_json = os.path.join(data_root, f"{split}.json")
    with open(split_json) as f:
        entries = json.load(f)

    sequences = defaultdict(list)
    for rel_path in entries:
        parts = rel_path.split("/")
        stem = os.path.splitext(parts[-1])[0]
        m = _NAME_RE.match(stem)
        if m is None:
            continue
        class_id = int(parts[1]) if len(parts) >= 3 else int(parts[0])
        model_id = m.group(1)
        frame_num = int(m.group(2))
        sweep_id = int(m.group(3)) if m.group(3) is not None else 0
        sequences[(class_id, model_id, sweep_id)].append(frame_num)

    if not sequences:
        raise ValueError(
            f"{split_json} holds no frame-indexed entries "
            f"({{name}}_frame_{{num}}[_s{{sweep}}]), so no clips can be counted."
        )

    n_clips = 0
    n_sequences = 0
    for key in sorted(sequences.keys()):
        nframes = len(sequences[key])
        span = frame_interval * (seq_len - 1)
        for t in range(0, nframes - span, seq_len):
            n_clips += 1
        n_sequences += 1

    return n_clips, n_sequences


def main():
    parser = argparse.ArgumentParser(
        description="Pre-generate rotation matrices for fair cross-model evaluation"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--split", default="test", help="Split to count clips from (test or val)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--frame-interval", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    n_clips, n_sequences = count_clips(
        args.data_root, args.split, args.seq_len, args.frame_interval
    )
    print(
        f"{args.split} set: {n_sequences} sequences, {n_clips} clips "
        f"(seq_len={args.seq_len}, frame_interval={args.frame_interval})"
    )

    modes = ["z", "so3", "so3_per_frame"]
    result = {
        "metadata": {
            "data_root": args.data_root,
            "seed": args.seed,
            "n_trials": args.n_trials,
            "n_clips": n_clips,
            "seq_len": args.seq_len,
            "frame_interval": args.frame_interval,
            "modes": modes,
            "sampler": "pytorch3d.random_rotations",
            "sampler_version": "haar_v1",
        }
    }

    for mode in modes:
        for trial in range(args.n_trials):
            trial_seed = args.seed + trial
            torch.manual_seed(trial_seed)
            key = f"{mode}_{trial}"

            if mode == "z":
                R = random_rotation_matrix_z_batch(n_clips)
            elif mode == "so3":
                R = random_rotation_matrix_so3_batch(n_clips)
            elif mode == "so3_per_frame":
                R = random_rotation_matrix_so3_batch(n_clips * args.seq_len)
                R = R.view(n_clips, args.seq_len, 3, 3)

            result[key] = R
            print(f"  {key}: shape={tuple(R.shape)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(result, args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
