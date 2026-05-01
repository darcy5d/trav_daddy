#!/usr/bin/env python3
"""Wave 5.6 Phase 3: refit V3 Platt calibration using the Wave 5.6 sweep's
calibration_master.csv (much larger n than the original Wave 5 calibration fit).

Outputs to data/models/v3/calibration.json (overwriting the Wave 5.5 calibration).

Note: the existing scripts/fit_calibration.py expects backtest CSVs with the
format produced by scripts/backtest_simulator.py. The Wave 5.6 calibration_master.csv
has a DIFFERENT format (per-(sim_version, toss_mode) rows). This script
rewrites the calibration rows into the per-route format fit_calibration.py expects.

Run AFTER scripts/run_wave_5_6_focused_sweep.py finishes:
    venv311/bin/python scripts/refit_v3_calibration_w56.py
"""

from __future__ import annotations

import csv
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CALIB_MASTER = Path("data/diagnostics/wave_5_6_modern_era_sweep/calibration_master.csv")
TMP_DIR = Path("data/backtest")
OUTPUT_CAL = "data/models/v3/calibration.json"


def _classify_route(tournament: str) -> tuple[str, str]:
    """Map tournament -> (format, gender)."""
    t = tournament.lower()
    if "wpl" in t:
        return "T20", "female"
    if "odi" in t:
        if "women" in t:
            return "ODI", "female"
        return "ODI", "male"
    if "t20i" in t:
        if "women" in t:
            return "T20", "female"
        return "T20", "male"
    # Default to T20 male (IPL/PSL/BBL/CPL)
    return "T20", "male"


def main() -> int:
    if not CALIB_MASTER.exists():
        logger.error(f"Wave 5.6 calibration master not found: {CALIB_MASTER}")
        logger.error("Run scripts/run_wave_5_6_focused_sweep.py first.")
        return 1

    # Filter to V3 rows only (calibration is for V3 model). Use toss=pinned
    # by default since that's the live-deployment mode at T-30min entry.
    # Going with pinned because at long lookbacks we'll just use the same
    # calibration (slight mis-fit there but acceptable; the alternative is
    # two separate calibrations which adds bookkeeping).
    rows_by_route: dict[tuple[str, str], list[dict]] = {}
    with CALIB_MASTER.open() as fp:
        for row in csv.DictReader(fp):
            if row.get("sim_version") != "v3":
                continue
            if row.get("toss_mode") != "pinned":
                continue
            tournament = row.get("tournament", "")
            fmt, gender = _classify_route(tournament)
            rows_by_route.setdefault((fmt, gender), []).append(row)

    if not rows_by_route:
        logger.error("No V3 pinned rows in calibration master.")
        return 1

    # Write per-route CSVs in the format fit_calibration.py expects
    # (matches scripts/backtest_simulator.py's BacktestRow output).
    csv_paths_args = []
    for (fmt, gender), rows in rows_by_route.items():
        tmp_path = TMP_DIR / f"backtest_w5_6_calib_{fmt.lower()}_{gender}.csv"
        with tmp_path.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=[
                "match_id", "date", "match_type", "gender", "event_name",
                "team1_id", "team2_id", "canonical_team1_id", "canonical_team2_id",
                "team1_elo_used", "team2_elo_used",
                "sim_team1_win_prob", "sim_avg_team1_score", "sim_avg_team2_score",
                "actual_winner_id", "actual_team1_total", "actual_team2_total",
                "team1_won", "margin_runs", "sim_margin_runs",
                "score_mae_team1", "score_mae_team2", "dist_quality_overall_pct",
            ])
            writer.writeheader()
            for r in rows:
                if r.get("actual_team1_won") in (None, "", "None"):
                    continue
                writer.writerow({
                    "match_id": r.get("match_id", ""),
                    "date": r.get("match_date", ""),
                    "match_type": fmt,
                    "gender": gender,
                    "event_name": r.get("event_name", ""),
                    "team1_id": "",
                    "team2_id": "",
                    "canonical_team1_id": "",
                    "canonical_team2_id": "",
                    "team1_elo_used": "",
                    "team2_elo_used": "",
                    "sim_team1_win_prob": r.get("sim_team1_win_prob", ""),
                    "sim_avg_team1_score": r.get("sim_avg_team1_score", ""),
                    "sim_avg_team2_score": r.get("sim_avg_team2_score", ""),
                    "actual_winner_id": "",
                    "actual_team1_total": r.get("actual_team1_total", ""),
                    "actual_team2_total": r.get("actual_team2_total", ""),
                    "team1_won": r.get("actual_team1_won", ""),
                    "margin_runs": "",
                    "sim_margin_runs": "",
                    "score_mae_team1": "",
                    "score_mae_team2": "",
                    "dist_quality_overall_pct": "",
                })
        logger.info(f"Wrote {len(rows)} rows to {tmp_path} for ({fmt}, {gender})")
        csv_paths_args.extend(["--backtest-csv", str(tmp_path)])

    # Call fit_calibration.py with all per-route CSVs
    cmd = [
        "venv311/bin/python", "scripts/fit_calibration.py",
        "--output", OUTPUT_CAL,
        "--notes", f"Wave 5.6 V3 Platt refit on bigger holdout (V3-pinned mode)",
    ] + csv_paths_args
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
