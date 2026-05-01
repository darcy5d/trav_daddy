#!/usr/bin/env python3
"""Wave 5.5 Phase B: wide multi-format Polymarket EV sweep.

Runs `backtest_polymarket_ev.py` across N tournament-format-gender combos
for V2 and V3 simulators. Each invocation produces:
- per-tournament bet rows CSV (1 row per match x market x outcome x lookback x toss_mode)
- per-tournament calibration CSV (1 row per match x toss_mode, regardless of Polymarket coverage)

After all runs complete, consolidates into:
- data/diagnostics/wave_5_5_wide_sweep/master_bets.csv
- data/diagnostics/wave_5_5_wide_sweep/calibration_master.csv

Each subprocess is fully isolated so TF/Metal GPU state never leaks
between runs (Wave 5 lesson learned).

Usage:
    venv311/bin/python scripts/run_wave_5_5_wide_sweep.py
    LIMIT=30 N_SIMS=200 venv311/bin/python scripts/run_wave_5_5_wide_sweep.py
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
LIMIT = int(os.environ.get("LIMIT", "50"))
LOOKBACK_HOURS = os.environ.get("LOOKBACK_HOURS", "0.5,1,3,6,12,24,48,72")
EDGE_THRESHOLDS = os.environ.get("EDGE_THRESHOLDS", "3,5,10,20")
BET_SIZE = float(os.environ.get("BET_SIZE", "25"))
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "data/diagnostics/wave_5_5_wide_sweep"))
SIM_VERSIONS = os.environ.get("SIM_VERSIONS", "v2 v3").split()


# (label_suffix, format, gender, since_date, tournament_pattern)
RUN_DEFINITIONS = [
    ("ipl",            "T20", "male",   "2024-03-01", "%Indian Premier League%"),
    ("psl",            "T20", "male",   "2024-01-01", "%Pakistan Super League%"),
    ("bbl",            "T20", "male",   "2024-12-01", "%Big Bash League%"),
    ("wpl",            "T20", "female", "2024-01-01", "%Women's Premier League%"),
    ("cpl",            "T20", "male",   "2024-08-01", "%Caribbean Premier League%"),
    ("t20i_men",       "T20", "male",   "2024-01-01", "%T20I%"),
    ("t20i_women",     "T20", "female", "2024-01-01", "%T20I%"),
    ("odi_men",        "ODI", "male",   "2024-01-01", "%"),
    ("odi_women",      "ODI", "female", "2024-01-01", "%"),
]


def run_one(sim: str, suffix: str, fmt: str, gender: str, since: str, pattern: str) -> bool:
    label = f"w5_5_{sim}_{suffix}"
    out_dir = OUTPUT_ROOT / "per_tournament"
    out_dir.mkdir(parents=True, exist_ok=True)
    bet_csv = out_dir / f"{label}_bets.csv"
    calib_csv = out_dir / f"{label}_calibration.csv"
    md = out_dir / f"{label}_summary.md"

    print()
    print("=" * 70)
    print(f"  {label}  ({sim}, {fmt} {gender}, tournament={pattern}, since={since})")
    print("=" * 70, flush=True)

    cmd = [
        PYTHON, "scripts/backtest_polymarket_ev.py",
        "--model-version", sim,
        "--format", fmt,
        "--gender", gender,
        "--tournament-pattern", pattern,
        "--since-date", since,
        "--limit", str(LIMIT),
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
    """Merge per-tournament CSVs into a single master CSV. Add tournament + sim columns
    if they're not already there (they should be tagged by the bet-row sim_version
    field; we add tournament since the per-row CSV doesn't carry it).
    """
    per_tournament_dir = OUTPUT_ROOT / "per_tournament"
    matched_files = sorted(per_tournament_dir.glob(f"w5_5_*_{suffix}.csv"))
    if not matched_files:
        print(f"WARN: no input CSVs found for suffix '{suffix}' in {per_tournament_dir}")
        return 0

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    header_written = False
    with out_csv.open("w", newline="") as out_fp:
        writer = None
        for path in matched_files:
            # Filename: w5_5_<sim>_<tournament>_(bets|calibration).csv
            stem_parts = path.stem.split("_")
            # stem is "w5_5_<sim>_<tournament>_<suffix>"
            # extract <tournament> via slice
            try:
                # Skip "w5", "5", "<sim>" -> rest is tournament + suffix; suffix is known
                if stem_parts[0] != "w5" or stem_parts[1] != "5":
                    continue
                sim_str = stem_parts[2]
                # Everything between sim and the final suffix word(s) is tournament
                # Suffix is a single word ("bets" or "calibration"), so tournament is parts[3:-1]
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
    print(f"Wave 5.5 wide sweep")
    print(f"  SIM_VERSIONS={SIM_VERSIONS}  LIMIT={LIMIT}  N_SIMS={N_SIMS}")
    print(f"  LOOKBACK_HOURS={LOOKBACK_HOURS}  EDGE_THRESHOLDS={EDGE_THRESHOLDS}")
    print(f"  OUTPUT_ROOT={OUTPUT_ROOT}")
    print(f"  Total backtests: {len(SIM_VERSIONS) * len(RUN_DEFINITIONS)}")

    overall_t0 = time.time()
    n_ok = 0
    n_fail = 0
    for sim in SIM_VERSIONS:
        for suffix, fmt, gender, since, pattern in RUN_DEFINITIONS:
            ok = run_one(sim, suffix, fmt, gender, since, pattern)
            if ok:
                n_ok += 1
            else:
                n_fail += 1
    elapsed = time.time() - overall_t0
    print()
    print("=" * 70)
    print(f"Sweep done: {n_ok} OK / {n_fail} failed in {elapsed/60:.1f} min")
    print("=" * 70)

    # Consolidate
    print()
    print("Consolidating per-tournament CSVs into master files...")
    n_bets = consolidate_csvs("bets", OUTPUT_ROOT / "master_bets.csv")
    n_calib = consolidate_csvs("calibration", OUTPUT_ROOT / "calibration_master.csv")
    print(f"Master bets: {n_bets:,} rows")
    print(f"Master calibration: {n_calib:,} rows")
    print()
    print(f"Run scripts/analyse_wave_5_5_master.py next.")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
