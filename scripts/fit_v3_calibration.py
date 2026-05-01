#!/usr/bin/env python3
"""Wave 5.5 Phase A5: fit per-(format, gender) Platt calibration for V3.

Runs V3 backtests on a small recent holdout per format-gender route, then
fits Platt scalars on the combined backtest CSVs. Output goes to
`data/models/v3/calibration.json` (4 routes).

Usage:
    venv311/bin/python scripts/fit_v3_calibration.py
    venv311/bin/python scripts/fit_v3_calibration.py --limit-per-route 30 --n-sims 200
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# (format, gender, since_date, optional tournament pattern)
# Recent dates per route to give the calibration a recent prior on V3's
# raw probabilities. Uses match-rich tournaments where available.
ROUTES = [
    ("T20", "male",   "2025-01-01", None),
    ("T20", "female", "2024-01-01", None),
    ("ODI", "male",   "2024-01-01", None),
    ("ODI", "female", "2024-01-01", None),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit V3 Platt calibration per route")
    parser.add_argument("--limit-per-route", type=int, default=40,
                        help="Max matches per (format, gender) route. Smaller = faster.")
    parser.add_argument("--n-sims", type=int, default=200,
                        help="Monte Carlo iterations per match.")
    parser.add_argument("--output", default="data/models/v3/calibration.json")
    parser.add_argument("--label-prefix", default="v3_calib")
    parser.add_argument("--skip-backtests", action="store_true",
                        help="Skip the backtest stage; assume CSVs already exist.")
    args = parser.parse_args()

    backtest_csvs = []
    for fmt, gen, since, _pat in ROUTES:
        label = f"{args.label_prefix}_{fmt.lower()}_{gen}"
        csv_path = Path("data/backtest") / f"backtest_{label}.csv"

        if not args.skip_backtests:
            cmd = [
                "venv311/bin/python", "scripts/backtest_simulator.py",
                "--model-version", "v3",
                "--format", fmt,
                "--gender", gen,
                "--since-date", since,
                "--limit", str(args.limit_per_route),
                "--n-sims", str(args.n_sims),
                "--label", label,
            ]
            logger.info(f"Running backtest for {fmt}/{gen}")
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                logger.warning(f"Backtest for {fmt}/{gen} failed (rc={result.returncode}); skipping")
                continue
        if csv_path.exists():
            backtest_csvs.append(csv_path)
        else:
            logger.warning(f"Expected CSV not found: {csv_path}")

    if not backtest_csvs:
        logger.error("No backtest CSVs available; cannot fit calibration.")
        return 1

    # Use existing fit_calibration.py to do the actual Platt fitting
    cmd = [
        "venv311/bin/python", "scripts/fit_calibration.py",
        "--output", args.output,
        "--notes", f"V3 Platt calibration fit on {len(backtest_csvs)} routes "
                   f"({datetime.now().strftime('%Y-%m-%d %H:%M')})",
    ]
    for csv in backtest_csvs:
        cmd.extend(["--backtest-csv", str(csv)])
    logger.info(f"Fitting calibration on: {[str(p) for p in backtest_csvs]}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error(f"fit_calibration.py failed (rc={result.returncode})")
        return 1

    logger.info(f"V3 calibration written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
