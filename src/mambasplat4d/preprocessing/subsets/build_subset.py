#!/usr/bin/env python3
"""Subset-builder core for the 50/25/25 four-class AeroSplat-4D roots.

Copies `.pt` frames of the 50/25/25 asset split at the 45az / 4cams / 3.0x sweep
from a full preprocessed cache into a new root:

    <dst>/{config.json, category.txt, train.json, val.json, test.json}
    <dst>/{split}/{class_idx}/<seq_name>_frame_{XXXX}.pt

`--duplications N`: each frame N times, k-th copy renumbered `frame_{n + k * FRAMES_PER_BIN}`
(60-frame variants).
`--dry-run`: plan manifests only. `--compare-to`: diff plan against an existing root.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from mambasplat4d.categories import CATEGORIES
from mambasplat4d.preprocessing.preprocess_aerosplat4d import _load_split_yaml
from mambasplat4d.preprocessing.subsets.assignments import (
    ASSIGNMENTS,
    DISTANCE_BINS,
    ELEVATIONS,
    FRAMES_PER_BIN,
    SWEEP_TAIL,
)

FRAME_RE = re.compile(r"^(?P<head>.+_frame_)(?P<n>\d{4})\.pt$")


def sweep_prefixes(distance_bins: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        f"blank_lgm_{d}_{e}{SWEEP_TAIL}" for d in distance_bins for e in ELEVATIONS
    )


def find_asset_files(
    src: Path, asset: str, class_idx: int, distance_bins: tuple[str, ...]
) -> list[Path]:
    """All source `.pt` for `asset` under the sweep prefixes; deduped by name across source splits."""
    by_name: dict[str, Path] = {}
    targets = tuple(f"{p}{asset}_frame_" for p in sweep_prefixes(distance_bins))
    for split_dir in ("train", "val", "test"):
        cls_dir = src / split_dir / str(class_idx)
        if not cls_dir.is_dir():
            continue
        for f in cls_dir.iterdir():
            if f.name.endswith(".pt") and f.name.startswith(targets):
                by_name.setdefault(f.name, f)
    return sorted(by_name.values(), key=lambda p: p.name)


def plan(
    src: Path,
    distance_bins: tuple[str, ...],
    duplications: int,
    tag: str = "build_subset",
    verbose: bool = True,
    assignments: dict[str, dict[int, list[str]]] = ASSIGNMENTS,
) -> tuple[dict[str, list[str]], dict[str, list[tuple[Path, str]]]]:
    """Manifests and copy list; writes nothing."""
    expected_source_frames = FRAMES_PER_BIN * len(distance_bins) * len(ELEVATIONS)
    expected_bins = {(d, e) for d in distance_bins for e in ELEVATIONS}

    manifests: dict[str, list[str]] = {}
    copies: dict[str, list[tuple[Path, str]]] = {}

    for split, by_class in assignments.items():
        rel_paths: list[str] = []
        split_copies: list[tuple[Path, str]] = []
        for class_idx, assets in by_class.items():
            for asset in assets:
                src_files = find_asset_files(src, asset, class_idx, distance_bins)
                bins_seen: set[tuple[str, str]] = set()
                for f in src_files:
                    for d in distance_bins:
                        for e in ELEVATIONS:
                            if f.name.startswith(f"blank_lgm_{d}_{e}{SWEEP_TAIL}"):
                                bins_seen.add((d, e))
                missing_bins = expected_bins - bins_seen
                if missing_bins:
                    raise RuntimeError(
                        f"asset {asset!r} (class {class_idx}) is missing "
                        f"entire (distance, elevation) bins: "
                        f"{sorted(missing_bins)}"
                    )
                if len(src_files) != expected_source_frames:
                    print(
                        f"[{tag}] WARN {split}/{class_idx} "
                        f"{asset}: {len(src_files)} src frames "
                        f"(expected {expected_source_frames}); "
                        f"some bins have dropout"
                    )
                for src_f in src_files:
                    m = FRAME_RE.match(src_f.name)
                    if m is None:
                        raise RuntimeError(f"unexpected filename shape: {src_f.name}")
                    head = m.group("head")
                    n = int(m.group("n"))
                    for k in range(duplications):
                        new_name = f"{head}{n + k * FRAMES_PER_BIN:04d}.pt"
                        rel = f"{split}/{class_idx}/{new_name}"
                        rel_paths.append(rel)
                        split_copies.append((src_f, rel))
                if verbose:
                    print(
                        f"[{tag}] {split}/{class_idx} {asset}: "
                        f"{len(src_files)} src -> "
                        f"{len(src_files) * duplications} out"
                    )
        rel_paths.sort()
        manifests[split] = rel_paths
        copies[split] = split_copies
    return manifests, copies


def write_root(
    src: Path,
    dst: Path,
    manifests: dict[str, list[str]],
    copies: dict[str, list[tuple[Path, str]]],
    distance_bins: tuple[str, ...],
    tag: str = "build_subset",
) -> None:
    if dst.exists():
        print(f"[{tag}] removing existing {dst}")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    for split, split_copies in copies.items():
        for src_f, rel in split_copies:
            dst_f = dst / rel
            dst_f.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_f, dst_f)
        (dst / f"{split}.json").write_text(json.dumps(manifests[split], indent=2))

    shutil.copy2(src / "category.txt", dst / "category.txt")

    src_cfg = json.loads((src / "config.json").read_text())
    src_cfg["output_dir"] = str(dst)
    src_cfg["splits"] = {k: len(v) for k, v in manifests.items()}
    if tuple(distance_bins) != DISTANCE_BINS:
        src_cfg["distance_bins"] = list(distance_bins)
    (dst / "config.json").write_text(json.dumps(src_cfg, indent=2))


def manifest_summary(manifest: list[str]) -> Counter:
    counts: Counter = Counter()
    for rel in manifest:
        split, class_idx, _name = rel.split("/")
        counts[(split, class_idx)] += 1
    return counts


def compare_to(
    manifests: dict[str, list[str]], existing: Path, tag: str = "build_subset"
) -> bool:
    """Diff planned manifests against existing root; True if identical."""
    ok = True
    print()
    print(f"[{tag}] manifest diff against {existing}")
    for split in ("train", "val", "test"):
        planned = manifests.get(split, [])
        ref_file = existing / f"{split}.json"
        if not ref_file.is_file():
            print(f"  {split}: MISSING {ref_file}")
            ok = False
            continue
        ref = json.loads(ref_file.read_text())
        ps, rs = set(planned), set(ref)
        pc, rc = manifest_summary(planned), manifest_summary(ref)
        classes = sorted({k[1] for k in pc} | {k[1] for k in rc})
        per_class = "  ".join(f"{c}:{pc[(split, c)]}/{rc[(split, c)]}" for c in classes)
        same = planned == ref
        print(
            f"  {split}: planned {len(planned)} vs on-disk {len(ref)} "
            f"[class planned/on-disk {per_class}] "
            f"{'IDENTICAL' if same else 'DIFFERS'}"
        )
        if not same:
            ok = False
            only_p, only_r = sorted(ps - rs), sorted(rs - ps)
            print(f"    only planned ({len(only_p)}): {only_p[:5]}")
            print(f"    only on-disk ({len(only_r)}): {only_r[:5]}")
            if not only_p and not only_r:
                print("    same set, different order")
    print(f"[{tag}] {'MANIFESTS MATCH' if ok else 'MANIFESTS DIFFER'}")
    return ok


def assignments_from_yaml(split_yaml: Path) -> dict[str, dict[int, list[str]]]:
    assignments: dict[str, dict[int, list[str]]] = {
        split: {idx: [] for idx in CATEGORIES.values()} for split in ("train", "val", "test")
    }
    for asset, (class_name, split) in _load_split_yaml(split_yaml).items():
        assignments[split][CATEGORIES[class_name]].append(asset)
    return assignments


def build_parser(description: str, duplications: int) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Full preprocessed cache to carve the subset out of "
        "(the root written by preprocess_aerosplat4d.py)",
    )
    ap.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="Output root; required unless --dry-run",
    )
    ap.add_argument(
        "--duplications",
        type=int,
        default=duplications,
        help="Copies of each source frame, renumbered by FRAMES_PER_BIN",
    )
    ap.add_argument(
        "--distance-bins",
        nargs="+",
        default=list(DISTANCE_BINS),
        metavar="BIN",
        help="Radius tokens to keep, e.g. 4r 8r 16r 32r 64r 100r",
    )
    ap.add_argument(
        "--split-yaml",
        type=Path,
        default=None,
        help="Split YAML to take the asset assignment from instead of ASSIGNMENTS",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan and print the manifests without copying tensors",
    )
    ap.add_argument(
        "--compare-to",
        type=Path,
        default=None,
        help="Existing root whose manifests the plan is diffed against",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the per-asset progress lines",
    )
    return ap


def run(args: argparse.Namespace, tag: str) -> int:
    distance_bins = tuple(args.distance_bins)
    assignments = (
        ASSIGNMENTS if args.split_yaml is None else assignments_from_yaml(args.split_yaml)
    )
    manifests, copies = plan(
        args.src,
        distance_bins,
        args.duplications,
        tag=tag,
        verbose=not args.quiet,
        assignments=assignments,
    )

    print()
    print(f"[{tag}] planned manifests")
    for split in ("train", "val", "test"):
        counts = manifest_summary(manifests[split])
        per_class = "  ".join(f"{c}:{n}" for (_s, c), n in sorted(counts.items()))
        print(f"  {split}.json: {len(manifests[split])} entries  [{per_class}]")

    ok = True
    if args.compare_to is not None:
        ok = compare_to(manifests, args.compare_to, tag=tag)

    if not args.dry_run:
        if args.dst is None:
            raise SystemExit("--dst is required unless --dry-run is given")
        write_root(args.src, args.dst, manifests, copies, distance_bins, tag=tag)
        print()
        print(f"[{tag}] DONE -> {args.dst}")

    return 0 if ok else 1


def main() -> int:
    ap = build_parser(__doc__, duplications=1)
    return run(ap.parse_args(), tag="build_subset")


if __name__ == "__main__":
    raise SystemExit(main())
