#!/usr/bin/env python3
"""
Build the v3 unified ball-training artifacts (Wave 5.5 Phase A1).

Same shape as v2 but with three extra continuous features:
toss_won_by_batting_team, chose_to_bat, xi_overlap_recent_3.

Run:
    # Both genders, both formats (default)
    python scripts/build_ball_training_v3.py

    # Smoke test
    python scripts/build_ball_training_v3.py --gender male --formats T20 --limit 50000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.ball_training_data_v3 import BuildConfigV3, build_ball_training_v3  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build v3 unified ball-training artifacts.")
    parser.add_argument(
        "--gender", choices=["male", "female", "both"], default="both",
        help="Which gender(s) to build.",
    )
    parser.add_argument(
        "--formats", nargs="+", default=["T20", "ODI"], choices=["T20", "ODI"],
        help="Formats to include (joint output).",
    )
    parser.add_argument(
        "--half-life-days", type=float, default=365.0,
        help="Sample weight half-life in days (default 365).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap deliveries per gender for smoke testing.",
    )
    parser.add_argument(
        "--output-dir", default="data/processed",
        help="Where to write ball_training_v3_{gender}.npz.",
    )
    args = parser.parse_args()

    genders = ["male", "female"] if args.gender == "both" else [args.gender]
    for g in genders:
        cfg = BuildConfigV3(
            formats=tuple(args.formats),
            gender=g,
            half_life_days=args.half_life_days,
            limit=args.limit,
            output_dir=Path(args.output_dir),
        )
        result = build_ball_training_v3(cfg)
        print()
        print(f"=== {g} v3 build complete ===")
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
