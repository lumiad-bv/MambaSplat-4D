#!/usr/bin/env python3
"""Supplementary Tab. 1 cells from the eval_rotation JSONs of experiment=aerosplat4d_tab1.

Accuracy: mean +- std (n-1) over per-seed video accuracy, each seed averaged over its rotation
trials. Per-class at SO(3)/SO(3): mean over seeds x trials.

Usage:
    python -m mambasplat4d.cli.aggregate --results-root $AEROSPLAT_RESULTS_ROOT
"""

import argparse
import json
import statistics
from pathlib import Path

from mambasplat4d import paths
from mambasplat4d.categories import CATEGORY_NAMES

PROTOCOLS = [
    ("z/SO(3)pf", "z", "so3_per_frame"),
    ("z/z", "z", "z"),
    ("z/SO(3)", "z", "so3"),
    ("SO(3)/SO(3)", "so3", "so3"),
]


def load_mode(
    root: Path, run_name: str, train: str, seed: int, split: str, mode: str
) -> dict:
    path = (
        root
        / run_name
        / f"train_{train}"
        / f"seed_{seed}"
        / f"eval_rotation_{split}.json"
    )
    return json.loads(path.read_text())["results"][split][mode]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", type=Path, default=paths.results_root())
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    header = [p[0] for p in PROTOCOLS] + [
        CATEGORY_NAMES[i] for i in sorted(CATEGORY_NAMES)
    ]
    print(f"{'model':<6}" + "".join(f"{h:>14}" for h in header))
    for feature_mode in ("C", "X"):
        run_name = f"mambasplat_{feature_mode}"
        cells = []
        for _, train, mode in PROTOCOLS:
            seed_means = [
                100
                * load_mode(args.results_root, run_name, train, s, args.split, mode)[
                    "mean"
                ]
                for s in args.seeds
            ]
            cells.append(
                f"{statistics.mean(seed_means):.1f} ± {statistics.stdev(seed_means):.1f}"
            )
        class_trials = [
            trial
            for s in args.seeds
            for trial in load_mode(
                args.results_root, run_name, "so3", s, args.split, "so3"
            )["class_acc"]
        ]
        for c in sorted(CATEGORY_NAMES):
            cells.append(f"{100 * statistics.mean(t[c] for t in class_trials):.1f}")
        print(f"E({feature_mode}):".ljust(6) + "".join(f"{x:>14}" for x in cells))


if __name__ == "__main__":
    main()
