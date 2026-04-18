#!/usr/bin/env python3
"""
Build the v2 unified ball-training artifacts (Wave 4 Phase 1).

Produces one numpy npz file per gender containing all formats jointly,
with 9-class extras-aware labels, recency-weighted sample weights, and
the era feature. Output goes to data/processed/ball_training_v2_{gender}.npz.

Run:
    # Both genders, both formats, default half-life 365 days
    python scripts/build_ball_training_v2.py

    # Just male, just T20 (smoke test)
    python scripts/build_ball_training_v2.py --gender male --formats T20 --limit 50000

    # Tighter recency: 6-month half-life
    python scripts/build_ball_training_v2.py --half-life-days 180
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.ball_training_data_v2 import BuildConfig, build_ball_training_v2  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build v2 unified ball-training artifacts.")
    parser.add_argument(
        "--gender",
        choices=["male", "female", "both"],
        default="both",
        help="Which gender(s) to build. Each gender gets its own npz file.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["T20", "ODI"],
        choices=["T20", "ODI"],
        help="Formats to include (joint output - format is a row column).",
    )
    parser.add_argument(
        "--half-life-days",
        type=float,
        default=365.0,
        help="Sample weight half-life in days (default 365).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap deliveries per gender for smoke testing.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Where to write ball_training_v2_{gender}.npz (default data/processed).",
    )
    args = parser.parse_args()

    genders = ["male", "female"] if args.gender == "both" else [args.gender]

    for g in genders:
        cfg = BuildConfig(
            formats=tuple(args.formats),
            gender=g,
            half_life_days=args.half_life_days,
            limit=args.limit,
            output_dir=Path(args.output_dir),
        )
        result = build_ball_training_v2(cfg)
        print()
        print(f"=== {g} build complete ===")
        print(f"  Output       : {result.output_path}")
        print(f"  Rows         : {result.n_rows:,}")
        print(f"  Elapsed      : {result.elapsed_seconds:.1f}s")
        print(f"  By format    : {result.by_format}")
        print(f"  By class     :")
        for cls, n in result.n_classes.items():
            pct = 100.0 * n / result.n_rows if result.n_rows else 0.0
            print(f"    {cls:>7}: {n:>10,}  ({pct:5.2f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
