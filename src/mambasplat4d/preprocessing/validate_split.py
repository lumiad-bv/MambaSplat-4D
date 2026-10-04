#!/usr/bin/env python3
"""Validate assets/splits/train-test-val_split.yaml.

Checks:
  1. Class keys match CATEGORIES (bird / drone / airplane / helicopter).
  2. Each asset in exactly one split.
  With `--data`:
  3. Each YAML asset has >=1 file under <reconstruction_dir>/data/<class_name>/<seq>/.
  4. Each on-disk asset is in the YAML.
  5. YAML class matches on-disk class dir.

Syntax only:
    python -m mambasplat4d.preprocessing.validate_split \
        --yaml assets/splits/train-test-val_split.yaml

With disk coverage:
    python -m mambasplat4d.preprocessing.validate_split \
        --yaml assets/splits/train-test-val_split.yaml \
        --data <reconstruction_root>/data

Exit 0 on success, non-zero on violation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mambasplat4d.categories import CATEGORIES  # noqa: E402
from mambasplat4d.asset_tokens import extract_asset_token  # noqa: E402

from mambasplat4d.preprocessing.preprocess_aerosplat4d import _load_split_yaml, load_asset_splits  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yaml", required=True, type=Path,
                    help="Path to train-test-val_split.yaml")
    ap.add_argument("--data", type=Path, default=None,
                    help="Reconstruction data dir (e.g. .../_ply/data). "
                         "If provided, coverage + class consistency are checked "
                         "against files on disk.")
    args = ap.parse_args()

    print(f"Validating {args.yaml} ...")
    try:
        asset_to_split = _load_split_yaml(args.yaml)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    from collections import Counter
    counts: Counter = Counter()
    for asset, (cls, split) in asset_to_split.items():
        counts[(cls, split)] += 1

    print(f"  ✓ YAML structure OK  ({len(asset_to_split)} assets total)")
    print()
    header = f"  {'class':<12} {'train':>6} {'val':>6} {'test':>6} {'total':>6}"
    print(header); print("  " + "-" * (len(header) - 2))
    gtot = 0
    for cls in sorted(CATEGORIES):
        tr, va, te = counts[(cls, 'train')], counts[(cls, 'val')], counts[(cls, 'test')]
        tot = tr + va + te
        gtot += tot
        print(f"  {cls:<12} {tr:>6} {va:>6} {te:>6} {tot:>6}")
    print("  " + "-" * (len(header) - 2))
    print(f"  {'TOTAL':<12} {'':>6} {'':>6} {'':>6} {gtot:>6}")
    print()

    if args.data is not None:
        print(f"Checking against files on disk: {args.data}")
        try:
            splits = load_asset_splits(args.yaml, args.data)
        except Exception as exc:
            print(f"FAIL (disk check): {exc}", file=sys.stderr)
            return 3
        sizes = {k: len(v) for k, v in splits.items()}
        print(f"  ✓ Disk coverage OK  (files per split: {sizes})")

    print()
    print("OK: split YAML is valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
