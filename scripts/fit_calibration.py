#!/usr/bin/env python3
"""
Fit per-(format, gender) Platt-scaling calibration from a backtest CSV.

Pairs with `scripts/backtest_simulator.py`: run a backtest first, then fit
calibration on the resulting per-match CSV.

Run:
    # Fit on the latest IPL T20 baseline + write to data/models/v2_calibration.json
    python scripts/fit_calibration.py \
        --backtest-csv data/backtest/backtest_baseline_ipl_2025.csv \
        --output data/models/v2_calibration.json

    # Multi-source fit: union several CSVs into one bundle
    python scripts/fit_calibration.py \
        --backtest-csv data/backtest/backtest_baseline_ipl_2025.csv \
        --backtest-csv data/backtest/backtest_baseline_odi_phase0.csv \
        --output data/models/v2_calibration.json
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.calibration import (  # noqa: E402
    CalibrationBundle,
    fit_platt,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit per-(format, gender) Platt calibration from backtest CSVs.")
    parser.add_argument(
        "--backtest-csv",
        action="append",
        required=True,
        help="Path to a backtest_simulator.py CSV. Can be specified multiple times to union multiple backtests.",
    )
    parser.add_argument(
        "--output",
        default="data/models/v2_calibration.json",
        help="Where to write the calibration bundle JSON (default data/models/v2_calibration.json).",
    )
    parser.add_argument(
        "--notes",
        default=None,
        help="Optional notes string to embed in the bundle (e.g. 'fit on Wave 3.5 baseline').",
    )
    args = parser.parse_args()

    by_route: Dict[str, Tuple[List[float], List[int]]] = {}
    for csv_path in args.backtest_csv:
        if not Path(csv_path).exists():
            logger.error(f"Backtest CSV not found: {csv_path}")
            return 1
        with open(csv_path, "r") as fp:
            reader = csv.DictReader(fp)
            n_rows = 0
            for row in reader:
                if not row.get("team1_won"):
                    continue
                try:
                    fmt = row["match_type"].upper()
                    gen = row["gender"].lower()
                    p = float(row["sim_team1_win_prob"])
                    y = int(row["team1_won"])
                except (TypeError, ValueError, KeyError):
                    continue
                key = f"{fmt}_{gen}"
                probs, ys = by_route.setdefault(key, ([], []))
                probs.append(p)
                ys.append(y)
                n_rows += 1
        logger.info(f"loaded {n_rows} rows from {csv_path}")

    bundle = CalibrationBundle(notes=args.notes)
    for key, (probs, ys) in sorted(by_route.items()):
        params = fit_platt(np.asarray(probs), np.asarray(ys))
        bundle.params[key] = params
        logger.info(
            f"[{key}] n={params.n_train}  a={params.a:.3f}  b={params.b:.3f}  "
            f"nll {params.nll_before:.4f} -> {params.nll_after:.4f}"
            + ("  (identity fallback)" if params.nll_after == params.nll_before and params.a == 1.0 else "")
        )

    from datetime import datetime
    bundle.fit_at = datetime.utcnow().isoformat() + "Z"
    bundle.save(args.output)
    logger.info(f"Saved calibration bundle to {args.output}")

    print()
    print("=" * 60)
    print("Calibration summary")
    print("=" * 60)
    print(f"  routes fit : {len(bundle.params)}")
    for key, p in bundle.params.items():
        print(
            f"  {key:>15}  n={p.n_train:>4}  a={p.a:6.3f}  b={p.b:+6.3f}  "
            f"NLL {p.nll_before:.3f} -> {p.nll_after:.3f}"
        )
    print(f"  saved to   : {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
