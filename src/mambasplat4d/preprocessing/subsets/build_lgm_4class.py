#!/usr/bin/env python3
"""Build `preprocessed_50-25-25_4class`, the paper's training root.

Ten distance bins (2r ... 280r), both elevations (20el, -20el), 45az / 4cams / 3.0x
sweep, four classes, hand-picked ~50/25/25 asset split. One copy per source frame.

    python -m mambasplat4d.preprocessing.subsets.build_lgm_4class \\
        --src <data_root>/preprocessed \\
        --dst <data_root>/preprocessed_50-25-25_4class
"""

from mambasplat4d.preprocessing.subsets.build_subset import build_parser, run

TAG = "build_lgm_4class"


def main() -> int:
    ap = build_parser(__doc__, duplications=1)
    return run(ap.parse_args(), tag=TAG)


if __name__ == "__main__":
    raise SystemExit(main())
