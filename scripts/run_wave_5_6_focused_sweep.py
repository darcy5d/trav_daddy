#!/usr/bin/env python3
"""Wave 5.6 Phase 1: focused EV sweep on the V3-helps tournaments.

Wave 5.5 found V3 calibration improves meaningfully over V2 only on
T20I men (Brier 0.217 vs 0.237), ODI men (0.226 vs 0.246), and WPL women
(0.271 vs 0.323). Other tournaments either tied or V2 was better.

This sweep re-runs JUST those three tournaments with much bigger
per-tournament holdouts to get statistical confidence on whether V3's
calibration lift translates into auto-eligible (n_bets >= 50, CI95 > 0)
ROI combos.

Outputs to `data/diagnostics/wave_5_6_focused_sweep/` so it doesn't
clobber Wave 5.5's data.

Default sizing (~3-3.5 hours overnight):
- T20I men: limit 200 matches
- ODI men: limit 95 matches (all available since 2024)
- WPL women: limit 50 matches (all available)

Run:
    venv311/bin/python scripts/run_wave_5_6_focused_sweep.py
    SIM_VERSIONS="v3" LIMIT_T20I=120 venv311/bin/python scripts/run_wave_5_6_focused_sweep.py
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from pathlib import Path

PYTHON = os.environ.get("PYTHON", "venv311/bin/python")
N_SIMS = int(os.environ.get("N_SIMS", "200"))
LOOKBACK_HOURS = os.environ.get("LOOKBACK_HOURS", "0.5,1,3,6,12,24,48,72")
EDGE_THRESHOLDS = os.environ.get("EDGE_THRESHOLDS", "3,5,10,20")
BET_SIZE = float(os.environ.get("BET_SIZE", "25"))
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "data/diagnostics/wave_5_6_focused_sweep"))
SIM_VERSIONS = os.environ.get("SIM_VERSIONS", "v2 v3").split()

# Per-tournament limits (override individually via env)
LIMIT_T20I_MEN = int(os.environ.get("LIMIT_T20I", "200"))
LIMIT_ODI_MEN = int(os.environ.get("LIMIT_ODI", "95"))
LIMIT_WPL = int(os.environ.get("LIMIT_WPL", "50"))


# (label_suffix, format, gender, since_date, tournament_pattern, limit)
RUN_DEFINITIONS = [
    ("t20i_men", "T20", "male",   "2024-01-01", "%T20I%",                  LIMIT_T20I_MEN),
    ("odi_men",  "ODI", "male",   "2024-01-01", "%",                       LIMIT_ODI_MEN),
    ("wpl",      "T20", "female", "2024-01-01", "%Women's Premier League%", LIMIT_WPL),
]


def run_one(sim: str, suffix: str, fmt: str, gender: str, since: str, pattern: str, limit: int) -> bool:
    label = f"w5_6_{sim}_{suffix}"
    out_dir = OUTPUT_ROOT / "per_tournament"
    out_dir.mkdir(parents=True, exist_ok=True)
    bet_csv = out_dir / f"{label}_bets.csv"
    calib_csv = out_dir / f"{label}_calibration.csv"
    md = out_dir / f"{label}_summary.md"

    print()
    print("=" * 70)
    print(f"  {label}  ({sim}, {fmt} {gender}, tournament={pattern}, since={since}, limit={limit})")
    print("=" * 70, flush=True)

    cmd = [
        PYTHON, "scripts/backtest_polymarket_ev.py",
        "--model-version", sim,
        "--format", fmt,
        "--gender", gender,
        "--tournament-pattern", pattern,
        "--since-date", since,
        "--limit", str(limit),
        "--n-sims", str(N_SIMS),
        "--edge-thresholds", EDGE_THRESHOLDS,
        "--lookback-hours", LOOKBACK_HOURS,
        "--bet-size", str(BET_SIZE),
        "--output-csv", str(bet_csv),
        "--output-md", str(md),
        "--output-calibration-csv", str(calib_csv),
    ]
    t0 = time.time()
    try:
        result = subprocess.run(cmd, check=False)
        elapsed = time.time() - t0
        ok = result.returncode == 0
        print(f">>> {label} done in {elapsed:.1f}s (rc={result.returncode})", flush=True)
        return ok
    except Exception as exc:
        print(f">>> {label} CRASHED: {exc}", flush=True)
        return False


def consolidate_csvs(suffix: str, out_csv: Path) -> int:
    """Merge per-tournament CSVs into a master CSV, tagging tournament + sim columns."""
    per_tournament_dir = OUTPUT_ROOT / "per_tournament"
    matched_files = sorted(per_tournament_dir.glob(f"w5_6_*_{suffix}.csv"))
    if not matched_files:
        print(f"WARN: no input CSVs found for suffix '{suffix}' in {per_tournament_dir}")
        return 0

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    header_written = False
    with out_csv.open("w", newline="") as out_fp:
        writer = None
        for path in matched_files:
            stem_parts = path.stem.split("_")
            try:
                if stem_parts[0] != "w5" or stem_parts[1] != "6":
                    continue
                sim_str = stem_parts[2]
                tournament_str = "_".join(stem_parts[3:-1])
            except IndexError:
                tournament_str = path.stem
                sim_str = "?"

            with path.open() as in_fp:
                reader = csv.DictReader(in_fp)
                for row in reader:
                    row["tournament"] = tournament_str
                    if "sim_version" not in row:
                        row["sim_version"] = sim_str
                    if not header_written:
                        fieldnames = list(row.keys())
                        writer = csv.DictWriter(out_fp, fieldnames=fieldnames)
                        writer.writeheader()
                        header_written = True
                    writer.writerow(row)
                    n_total += 1
    print(f">>> wrote {n_total:,} rows to {out_csv}")
    return n_total


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Wave 5.6 focused sweep (V3-helps tournaments only)")
    print(f"  SIM_VERSIONS={SIM_VERSIONS}  N_SIMS={N_SIMS}")
    print(f"  Per-tournament limits: T20I men={LIMIT_T20I_MEN}, ODI men={LIMIT_ODI_MEN}, WPL={LIMIT_WPL}")
    print(f"  LOOKBACK_HOURS={LOOKBACK_HOURS}  EDGE_THRESHOLDS={EDGE_THRESHOLDS}")
    print(f"  OUTPUT_ROOT={OUTPUT_ROOT}")
    print(f"  Total backtests: {len(SIM_VERSIONS) * len(RUN_DEFINITIONS)}")

    overall_t0 = time.time()
    n_ok = 0
    n_fail = 0
    for sim in SIM_VERSIONS:
        for suffix, fmt, gender, since, pattern, limit in RUN_DEFINITIONS:
            ok = run_one(sim, suffix, fmt, gender, since, pattern, limit)
            if ok:
                n_ok += 1
            else:
                n_fail += 1
    elapsed = time.time() - overall_t0
    print()
    print("=" * 70)
    print(f"Focused sweep done: {n_ok} OK / {n_fail} failed in {elapsed/60:.1f} min")
    print("=" * 70)

    print()
    print("Consolidating per-tournament CSVs into master files...")
    n_bets = consolidate_csvs("bets", OUTPUT_ROOT / "master_bets.csv")
    n_calib = consolidate_csvs("calibration", OUTPUT_ROOT / "calibration_master.csv")
    print(f"Master bets: {n_bets:,} rows")
    print(f"Master calibration: {n_calib:,} rows")
    print()
    print("Run scripts/analyse_wave_5_6_master.py next (with outlier filtering + Winsorized ROI).")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
