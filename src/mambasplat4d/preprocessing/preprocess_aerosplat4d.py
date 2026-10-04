#!/usr/bin/env python3
"""Preprocess AeroSplat-4D .pt/.ply into deterministic subsampled .pt files.

Input:  pipelines/lgm/batch_reconstruct.py output,
            data/<class_name>/<seq_name>/frame_XXXX[_sX].{ply,pt}
        <class_name> in {bird, drone, airplane, helicopter}.
Output: <output>/<split>/<class_id>/<sample_name>.pt (N versions per frame),
        train.json / val.json / test.json manifests, category.txt.

Split is asset-disjoint via assets/splits/train-test-val_split.yaml; `--split-yaml`
REQUIRED, no random fallback.

Per frame:
  1. Load raw (.pt dict, or .ply with inverted activations)
  2. Normalize (center, unit sphere, opacity, scale, sh)
  3. N versions of rand(8192, replace) -> FPS(1200) -> rand(1024)
  4. Save {versions: (N, 1024, 14), label, class_name, sample_name}
  5. Write manifests

Usage:
    python -m mambasplat4d.preprocessing.preprocess_aerosplat4d \
        --input <reconstruction_root>/data \
        --output <data_root>/aerosplat4d_pre \
        --split-yaml assets/splits/train-test-val_split.yaml \
        --n-versions 25 --seed 42
"""
import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm

from mambasplat4d.categories import CATEGORIES, CATEGORY_NAMES  # noqa: E402
from mambasplat4d.asset_tokens import extract_asset_token  # noqa: E402


def read_raw_pt(path: Path) -> Dict[str, np.ndarray]:
    """Raw .pt from batch_reconstruct.py."""
    data = torch.load(str(path), map_location="cpu", weights_only=True)
    out = {k: data[k].numpy() for k in ("position", "quaternion", "scale", "opacity", "sh_dc")}

    # canonical sign, w >= 0
    q = out["quaternion"]
    sign = np.sign(q[:, 0:1])
    sign[sign == 0] = 1.0
    out["quaternion"] = q * sign

    return out


from mambasplat4d.preprocessing._preprocess_utils import sigmoid_np as _sigmoid_np  # noqa: E402
from mambasplat4d.preprocessing._preprocess_utils import fps_subsample_np  # noqa: E402, F401


def read_raw_ply(path: Path) -> Dict[str, np.ndarray]:
    """Raw 3DGS PLY, activations inverted."""
    from plyfile import PlyData
    vertex = PlyData.read(str(path))["vertex"]

    position = np.stack(
        [vertex["x"], vertex["y"], vertex["z"]], axis=-1,
    ).astype(np.float32)

    # logit -> [0, 1]
    opacity = _sigmoid_np(
        np.array(vertex["opacity"], dtype=np.float32)
    ).reshape(-1, 1)

    # log -> linear
    scale_names = sorted(
        [p.name for p in vertex.properties if p.name.startswith("scale_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    scale = np.exp(np.stack(
        [np.asarray(vertex[n], dtype=np.float32) for n in scale_names],
        axis=-1,
    )).astype(np.float32)

    # normalize, canonical sign w >= 0
    rot_names = sorted(
        [p.name for p in vertex.properties if p.name.startswith("rot")],
        key=lambda x: int(x.split("_")[-1]),
    )
    quaternion = np.stack(
        [np.asarray(vertex[n], dtype=np.float32) for n in rot_names],
        axis=-1,
    )
    quaternion = quaternion / (np.linalg.norm(quaternion, axis=-1,
                                              keepdims=True) + 1e-9)
    sign = np.sign(quaternion[:, 0:1])
    sign[sign == 0] = 1.0
    quaternion = quaternion * sign

    sh_dc = np.stack(
        [vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=-1,
    ).astype(np.float32)

    return {
        "position": position,
        "quaternion": quaternion,
        "scale": scale,
        "opacity": opacity,
        "sh_dc": sh_dc,
    }


def read_raw_file(path: Path) -> Dict[str, np.ndarray]:
    """Load raw Gaussians; format by extension."""
    if path.suffix == ".ply":
        return read_raw_ply(path)
    return read_raw_pt(path)


def normalize_gaussians(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = {k: v.copy() for k, v in data.items()}

    # center, unit sphere
    xyz = out["position"]
    centroid = np.mean(xyz, axis=0, keepdims=True)
    xyz = xyz - centroid
    radius = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    if radius < 1e-8:
        radius = 1.0
    out["position"] = xyz / radius
    out["scale"] = out["scale"] / radius

    # [0,1] -> [-1,1]
    out["opacity"] = out["opacity"] * 2.0 - 1.0

    # unit sphere, no centering: keeps positivity
    s_mag = np.max(np.sqrt(np.sum(out["scale"] ** 2, axis=1)))
    if s_mag < 1e-8:
        s_mag = 1.0
    out["scale"] = out["scale"] / s_mag

    # SH DC -> normalized color
    sh = out["sh_dc"] * 0.28209479177387814
    sh = np.clip(sh, -0.5, 0.5)
    out["sh_dc"] = (2.0 * sh / math.sqrt(3.0)).astype(np.float32)

    return out


def generate_version(
    data: Dict[str, np.ndarray],
    rng: np.random.RandomState,
    sample_points_num: int = 8192,
    fps_point_all: int = 1200,
    num_points: int = 1024,
) -> np.ndarray:
    """rand(sample_points_num) -> FPS(fps_point_all) -> rand(num_points); returns (num_points, 14)."""
    N = data["position"].shape[0]

    pre_idx = rng.choice(N, sample_points_num, replace=True)
    buf = {k: v[pre_idx] for k, v in data.items()}

    fps_idx = fps_subsample_np(buf["position"], fps_point_all)
    point_all = len(fps_idx)

    replace = point_all < num_points
    choice = rng.choice(point_all, num_points, replace=replace)
    final_idx = fps_idx[choice]

    # pos(3) + quat(4) + scale(3) + opacity(1) + sh(3) = 14
    feats = np.concatenate([
        buf["position"][final_idx],
        buf["quaternion"][final_idx],
        buf["scale"][final_idx],
        buf["opacity"][final_idx],
        buf["sh_dc"][final_idx],
    ], axis=-1)  # (num_points, 14)

    return feats


# (N, 14) layout
FEATURE_SLICES = {
    "position": (0, 3),
    "quaternion": (3, 7),
    "scale": (7, 10),
    "opacity": (10, 11),
    "sh_dc": (11, 14),
}


def parse_rel_path(rel_path: str) -> Tuple[int, str, str, str]:
    """`<class_name>/<seq_name>/<file_stem>.{ply,pt}` -> (class_idx, class_name, seq_name, sample_name).

    sample_name = <seq_name>_<file_stem>, collision-free output basename.
    """
    parts = rel_path.split("/")
    if len(parts) != 3:
        raise ValueError(
            f"Expected rel_path `<class>/<seq>/<file>`, got: {rel_path!r}"
        )
    class_name, seq_name, file_name = parts
    if class_name not in CATEGORIES:
        raise ValueError(
            f"Unknown class name in rel_path: {class_name!r} "
            f"(expected one of {sorted(CATEGORIES)})"
        )
    class_idx = CATEGORIES[class_name]
    file_stem = Path(file_name).stem
    sample_name = f"{seq_name}_{file_stem}"
    return class_idx, class_name, seq_name, sample_name


def process_one(
    rel_path: str,
    data_dir: Path,
    n_versions: int,
    seed: int,
    sample_points_num: int,
    fps_point_all: int,
    num_points: int,
) -> Optional[Dict]:
    file_path = data_dir / rel_path
    class_idx, class_name, _seq, sample_name = parse_rel_path(rel_path)

    try:
        raw = read_raw_file(file_path)
    except Exception as e:
        print(f"  Error reading {file_path}: {e}")
        return None

    if raw["position"].shape[0] == 0:
        print(f"  Skipping empty file (0 Gaussians): {rel_path}")
        return None

    data = normalize_gaussians(raw)

    base_seed = seed + hash(rel_path) % (2**31)

    versions = []
    for v in range(n_versions):
        rng = np.random.RandomState(base_seed + v)
        feats = generate_version(data, rng, sample_points_num, fps_point_all, num_points)
        versions.append(feats)

    versions_tensor = torch.from_numpy(np.stack(versions, axis=0))  # (n_versions, num_points, 14)

    return {
        "versions": versions_tensor,
        "label": class_idx,
        "class_name": class_name,
        "sample_name": sample_name,
        "rel_path": rel_path,
    }


def preprocess_split(
    file_list: List[str],
    data_dir: Path,
    output_dir: Path,
    n_versions: int,
    seed: int,
    sample_points_num: int,
    fps_point_all: int,
    num_points: int,
    split: str,
    pbar: Optional[tqdm] = None,
) -> List[str]:
    """Process split, save per-sample .pt, return manifest."""
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    manifest: List[str] = []
    for rel_path in file_list:
        # resume
        class_idx_hint, _cn, _seq, sample_name_hint = parse_rel_path(rel_path)
        pt_rel_hint = f"{split}/{class_idx_hint}/{sample_name_hint}.pt"
        existing = output_dir / pt_rel_hint
        if existing.exists() and existing.stat().st_size > 0:
            manifest.append(pt_rel_hint)
            if pbar is not None:
                pbar.update(1)
            continue

        result = process_one(
            rel_path, data_dir, n_versions, seed,
            sample_points_num, fps_point_all, num_points,
        )
        if result is None:
            if pbar is not None:
                pbar.update(1)
            continue

        # numeric class id dir, as loader expects
        class_idx = result["label"]
        class_dir = split_dir / str(class_idx)
        class_dir.mkdir(parents=True, exist_ok=True)

        pt_name = f"{result['sample_name']}.pt"
        pt_rel = f"{split}/{class_idx}/{pt_name}"
        torch.save({
            "versions": result["versions"],
            "label": result["label"],
            "class_name": result["class_name"],
            "sample_name": result["sample_name"],
        }, output_dir / pt_rel)

        manifest.append(pt_rel)
        if pbar is not None:
            pbar.update(1)

    return manifest


def read_categories(data_dir: Path) -> Dict[int, str]:
    cat_file = data_dir / "category.txt"
    if not cat_file.exists():
        return {}
    class_names = {}
    for line in cat_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        idx_str, name = line.split(",", 1)
        class_names[int(idx_str)] = name
    return class_names


def _load_split_yaml(split_yaml: Path) -> Dict[str, Tuple[str, str]]:
    """Split YAML -> {asset_token: (class_name, split_name)}.

    Enforces: class keys == CATEGORIES, splits == {train, val, test}, each asset once.
    """
    if not split_yaml.is_file():
        raise FileNotFoundError(f"--split-yaml file does not exist: {split_yaml}")
    with open(split_yaml) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or not cfg:
        raise ValueError(f"Empty or malformed split YAML: {split_yaml}")

    expected_classes = set(CATEGORIES)
    got_classes = set(cfg.keys())
    if got_classes != expected_classes:
        raise ValueError(
            f"Split YAML class keys mismatch: got {sorted(got_classes)}, "
            f"expected {sorted(expected_classes)}"
        )

    asset_to_split: Dict[str, Tuple[str, str]] = {}
    for class_name, splits in cfg.items():
        if set(splits.keys()) != {"train", "val", "test"}:
            raise ValueError(
                f"Class {class_name!r} must contain exactly train/val/test "
                f"keys (got {sorted(splits.keys())})"
            )
        for split_name in ("train", "val", "test"):
            for asset in splits[split_name]:
                if asset in asset_to_split:
                    prev = asset_to_split[asset]
                    raise ValueError(
                        f"Asset {asset!r} appears in two splits: "
                        f"({prev[0]},{prev[1]}) and ({class_name},{split_name})"
                    )
                asset_to_split[asset] = (class_name, split_name)
    return asset_to_split


def load_asset_splits(
    split_yaml: Path, data_dir: Path
) -> Dict[str, List[str]]:
    """Bucket `<data_dir>/<class_name>/<seq_name>/frame_XXXX[_sX].*` by asset split.

    Asset token from seq folder name. Returns {split: [`<class_name>/<seq_name>/<filename>`]}.
    Raises on coverage, disjointness or class mismatch.
    """
    asset_to_split = _load_split_yaml(split_yaml)

    asset_files: Dict[str, List[str]] = defaultdict(list)
    asset_classes: Dict[str, set] = defaultdict(set)
    seen_any = False

    for class_name in sorted(CATEGORIES):
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            continue
        for seq_dir in sorted(class_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            try:
                asset = extract_asset_token(seq_dir.name)
            except ValueError as exc:
                raise ValueError(
                    f"Cannot extract asset token from {seq_dir}: {exc}"
                ) from exc
            asset_classes[asset].add(class_name)
            for f in sorted(seq_dir.iterdir()):
                if not f.is_file():
                    continue
                if f.suffix not in (".ply", ".pt"):
                    continue
                rel = f"{class_name}/{seq_dir.name}/{f.name}"
                asset_files[asset].append(rel)
                seen_any = True

    if not seen_any:
        raise FileNotFoundError(
            f"No reconstructed .ply/.pt files found under {data_dir}. "
            f"Expected layout: <class_name>/<seq>/frame_XXXX[_sX].{{ply,pt}}"
        )

    yaml_assets = set(asset_to_split)
    disk_assets = set(asset_files)

    missing_on_disk = yaml_assets - disk_assets
    orphan_on_disk = disk_assets - yaml_assets
    if missing_on_disk:
        raise FileNotFoundError(
            f"{len(missing_on_disk)} asset(s) in split YAML have no reconstructed "
            f"files under {data_dir}:\n  " + "\n  ".join(sorted(missing_on_disk))
        )
    if orphan_on_disk:
        raise ValueError(
            f"{len(orphan_on_disk)} asset(s) on disk are NOT in split YAML "
            f"{split_yaml}:\n  " + "\n  ".join(sorted(orphan_on_disk))
        )
    for asset, (yaml_class, _split) in asset_to_split.items():
        disk_class_set = asset_classes.get(asset, set())
        if disk_class_set and disk_class_set != {yaml_class}:
            raise ValueError(
                f"Class mismatch for asset {asset!r}: "
                f"YAML says {yaml_class!r}, disk says {sorted(disk_class_set)}"
            )

    buckets: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    asset_counts: Dict[str, Dict[str, int]] = {
        s: defaultdict(int) for s in buckets
    }
    for asset, files in asset_files.items():
        _, split = asset_to_split[asset]
        buckets[split].extend(sorted(files))
        class_name = asset_classes[asset].pop() if asset_classes[asset] else "?"
        asset_counts[split][class_name] += 1

    for split in buckets:
        buckets[split].sort()

    print("Asset-disjoint split summary (source: " + str(split_yaml) + "):")
    for split in ("train", "val", "test"):
        per_class = ", ".join(
            f"{cls}:{asset_counts[split][cls]}"
            for cls in sorted(CATEGORIES)
        )
        print(
            f"  {split:5s}: {len(buckets[split]):6d} files  "
            f"across {sum(asset_counts[split].values())} assets  "
            f"[{per_class}]"
        )
    return buckets


def main():
    parser = argparse.ArgumentParser(
        description="Pre-process AeroSplat-4D into deterministic subsampled .pt files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input directory (contains <class_name>/<seq>/frame_*.{ply,pt} + category.txt)")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output directory for pre-processed .pt files")
    parser.add_argument("--split-yaml", type=str, required=True,
                        help="Path to assets/train-test-val_split.yaml "
                             "(REQUIRED — asset-disjoint splitting has no fallback).")
    parser.add_argument("--n-versions", type=int, default=25,
                        help="Number of subsampled versions per frame")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-points-num", type=int, default=8192,
                        help="Random pre-sample buffer size")
    parser.add_argument("--fps-point-all", type=int, default=1200,
                        help="FPS target before final random subset")
    parser.add_argument("--num-points", type=int, default=1024,
                        help="Final points per version")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    split_yaml = Path(args.split_yaml)

    print("=" * 70)
    print("AeroSplat-4D Pre-Processing")
    print("=" * 70)
    print(f"  Input:           {input_dir}")
    print(f"  Output:          {output_dir}")
    print(f"  Split YAML:      {split_yaml}")
    print(f"  Versions/frame:  {args.n_versions}")
    print(f"  Pipeline:        rand({args.sample_points_num}) -> FPS({args.fps_point_all}) -> rand({args.num_points})")
    print(f"  Seed:            {args.seed}")
    print()

    start = time.time()

    splits = load_asset_splits(split_yaml, input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_files = sum(len(v) for v in splits.values())
    print(f"  Files: {' + '.join(f'{len(v)} {k}' for k, v in splits.items())} = {total_files} total")
    print()

    manifests = {}
    pbar = tqdm(total=total_files, unit="file",
                desc="Preprocessing", dynamic_ncols=True)
    for split_name, file_list in splits.items():
        pbar.set_postfix_str(split_name)
        manifests[split_name] = preprocess_split(
            file_list, input_dir, output_dir,
            args.n_versions, args.seed, args.sample_points_num,
            args.fps_point_all, args.num_points, split_name,
            pbar=pbar,
        )
    pbar.close()
    print()

    for split_name, manifest in manifests.items():
        with open(output_dir / f"{split_name}.json", "w") as f:
            json.dump(manifest, f, indent=2)

    # copy category.txt, or write from CATEGORIES
    cat_src = input_dir / "category.txt"
    cat_dst = output_dir / "category.txt"
    if cat_src.exists():
        cat_dst.write_text(cat_src.read_text())
    else:
        with open(cat_dst, "w") as f:
            for cat_id in sorted(CATEGORY_NAMES):
                f.write(f"{cat_id},{CATEGORY_NAMES[cat_id]}\n")

    # config with split YAML provenance
    config = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "split_yaml": str(split_yaml),
        "n_versions": args.n_versions,
        "seed": args.seed,
        "sample_points_num": args.sample_points_num,
        "fps_point_all": args.fps_point_all,
        "num_points": args.num_points,
        "feature_slices": FEATURE_SLICES,
        "splits": {k: len(v) for k, v in manifests.items()},
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    elapsed = time.time() - start
    print("=" * 70)
    print("PRE-PROCESSING COMPLETE")
    print("=" * 70)
    print(f"  Time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    for k, v in manifests.items():
        print(f"  {k:6s}: {len(v)} samples")
    print(f"  Output: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
