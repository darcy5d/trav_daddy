#!/usr/bin/env python3
"""Wave 5.6 (revised): modern-coverage-era EV sweep.

Polymarket cricket coverage was effectively absent before Sep 2025
(verified via Gamma `/events?tag_slug=cricket&closed=true` date histogram:
<60 events/month before Sep 2025; 700+ events/month from Dec 2025 onwards).

This sweep targets ONLY the modern-coverage era per tournament so the
backtest's effective sample size matches its nominal N (no more silent
skipping of pre-coverage matches).

Tournament-specific windows reflect when Polymarket actually started
listing each one regularly:
- IPL: since 2025-12-01 (IPL 2026 only; IPL 2025 had thin coverage)
- PSL: since 2025-12-01 (PSL 2026)
- BBL: since 2025-09-01 (BBL 2025-26 season)
- WPL: since 2024-01-01 (WPL 2024+2026; coverage was light early but worth including)
- T20I men: since 2025-09-01
- ODI men: since 2025-09-01

Outputs to `data/diagnostics/wave_5_6_modern_era_sweep/` so it doesn't
clobber the previous Wave 5.5/5.6 data.
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
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "data/diagnostics/wave_5_6_modern_era_sweep"))
SIM_VERSIONS = os.environ.get("SIM_VERSIONS", "v2 v3").split()


# (label_suffix, format, gender, since_date, tournament_pattern, limit)
RUN_DEFINITIONS = [
    # Top-coverage tournaments first (most data per minute of compute)
    ("ipl",       "T20", "male",   "2025-12-01", "%Indian Premier League%",   30),
    ("psl",       "T20", "male",   "2025-12-01", "%Pakistan Super League%",   30),
    ("bbl",       "T20", "male",   "2025-09-01", "%Big Bash League%",         50),
    ("wpl",       "T20", "female", "2024-01-01", "%Women's Premier League%",  70),
    ("t20i_men",  "T20", "male",   "2025-09-01", "%T20I%",                    50),
    ("odi_men",   "ODI", "male",   "2025-09-01", "%",                         80),
]


def run_one(sim: str, suffix: str, fmt: str, gender: str, since: str, pattern: str, limit: int) -> bool:
    label = f"w56m_{sim}_{suffix}"
    out_dir = OUTPUT_ROOT / "per_tournament"
    out_dir.mkdir(parents=True, exist_ok=True)
    bet_csv = out_dir / f"{label}_bets.csv"
    calib_csv = out_dir / f"{label}_calibration.csv"
    md = out_dir / f"{label}_summary.md"

    print()
    print("=" * 70)
    print(f"  {label}  ({sim}, {fmt} {gender}, since={since}, limit={limit})")
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
    per_tournament_dir = OUTPUT_ROOT / "per_tournament"
    matched_files = sorted(per_tournament_dir.glob(f"w56m_*_{suffix}.csv"))
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
                # filename: w56m_<sim>_<tournament>_<bets|calibration>
                if stem_parts[0] != "w56m":
                    continue
                sim_str = stem_parts[1]
                tournament_str = "_".join(stem_parts[2:-1])
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
    print(f"Wave 5.6 MODERN-ERA sweep (Polymarket coverage-era only)")
    print(f"  SIM_VERSIONS={SIM_VERSIONS}  N_SIMS={N_SIMS}")
    print(f"  6 tournaments x V2+V3 = {len(SIM_VERSIONS) * len(RUN_DEFINITIONS)} backtests")
    print(f"  OUTPUT_ROOT={OUTPUT_ROOT}")

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
    print(f"Modern-era sweep done: {n_ok} OK / {n_fail} failed in {elapsed/60:.1f} min")
    print("=" * 70)

    print()
    print("Consolidating per-tournament CSVs into master files...")
    n_bets = consolidate_csvs("bets", OUTPUT_ROOT / "master_bets.csv")
    n_calib = consolidate_csvs("calibration", OUTPUT_ROOT / "calibration_master.csv")
    print(f"Master bets: {n_bets:,} rows")
    print(f"Master calibration: {n_calib:,} rows")
    print()
    print("Next: scripts/analyse_wave_5_6_master.py "
          "--bets-csv data/diagnostics/wave_5_6_modern_era_sweep/master_bets.csv "
          "--calibration-csv data/diagnostics/wave_5_6_modern_era_sweep/calibration_master.csv "
          "--output data/diagnostics/wave_5_6_modern_era_summary.md")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
