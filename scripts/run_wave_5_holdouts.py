#!/usr/bin/env python3
"""Wave 5 Phase 3: Wider holdout backtests across simulators + tournaments.

Python orchestrator (replaces the bash version which was segfaulting on
Metal GPU re-init when bash respawned the Python process rapidly).

Each backtest runs as a fully-isolated subprocess so TF/Metal GPU state
never leaks between runs. Failures are logged but don't abort the sweep.

Defaults: V2 only (Wave 4.5 already established V2 > V1 on Brier/log-loss).
Set SIMS=v1 v2 (env var) to also run V1 for the side-by-side comparison.

Run:
    venv311/bin/python scripts/run_wave_5_holdouts.py
    SIMS="v1 v2" LIMIT=30 N_SIMS=500 venv311/bin/python scripts/run_wave_5_holdouts.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

PYTHON = os.environ.get("PYTHON", "venv311/bin/python")
N_SIMS = int(os.environ.get("N_SIMS", "300"))
LIMIT = int(os.environ.get("LIMIT", "20"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "data/backtest")
SIMS = os.environ.get("SIMS", "v2").split()


# (label_suffix, sim_version, format, gender, tournament_pattern, since_date)
RUN_DEFINITIONS = [
    ("ipl",       "T20", "male",   "%Indian Premier League%", "2025-01-01"),
    ("psl",       "T20", "male",   "%Pakistan Super League%", "2025-01-01"),
    ("bbl",       "T20", "male",   "%Big Bash League%",       "2024-12-01"),
    ("wpl",       "T20", "female", "%Women.s Premier League%","2024-01-01"),
    ("t20i_men",  "T20", "male",   "%T20I%",                  "2025-01-01"),
    ("odi_men",   "ODI", "male",   "%",                       "2025-01-01"),
]


def run_one(sim: str, suffix: str, fmt: str, gender: str, pattern: str, since: str) -> bool:
    label = f"w5_{sim}_{suffix}"
    print()
    print("=" * 60)
    print(f"  {label}  ({sim}, {fmt} {gender}, tournament={pattern})")
    print("=" * 60, flush=True)
    cmd = [
        PYTHON, "scripts/backtest_simulator.py",
        "--model-version", sim,
        "--format", fmt,
        "--gender", gender,
        "--tournament-pattern", pattern,
        "--since-date", since,
        "--limit", str(LIMIT),
        "--n-sims", str(N_SIMS),
        "--output-dir", OUTPUT_DIR,
        "--label", label,
    ]
    t0 = time.time()
    # Each subprocess gets a fully fresh TF/Metal GPU context. We DON'T
    # set stdin=DEVNULL or capture stdout because the per-match progress
    # logs are useful in real time.
    try:
        result = subprocess.run(cmd, check=False)
        ok = result.returncode == 0
        elapsed = time.time() - t0
        print(f">>> {label} done in {elapsed:.1f}s (rc={result.returncode})", flush=True)
        return ok
    except KeyboardInterrupt:
        print(">>> Aborted by user.", flush=True)
        raise
    except Exception as exc:
        print(f">>> {label} CRASHED: {exc}", flush=True)
        return False


def main() -> int:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Wave 5 holdout sweep")
    print(f"  SIMS={SIMS}  LIMIT={LIMIT}  N_SIMS={N_SIMS}  OUTPUT_DIR={OUTPUT_DIR}")
    print(f"  Total backtests: {len(SIMS) * len(RUN_DEFINITIONS)}")
    overall_t0 = time.time()
    n_ok = 0
    n_fail = 0
    for sim in SIMS:
        for suffix, fmt, gender, pattern, since in RUN_DEFINITIONS:
            ok = run_one(sim, suffix, fmt, gender, pattern, since)
            if ok:
                n_ok += 1
            else:
                n_fail += 1
    elapsed = time.time() - overall_t0
    print()
    print("=" * 60)
    print(f"Sweep complete: {n_ok} OK / {n_fail} failed in {elapsed/60:.1f} min")
    print("Per-match CSVs + summary JSONs in", OUTPUT_DIR, "(label prefix: w5_)")
    print("Run scripts/analyse_wave_5_holdouts.py next.")
    print("=" * 60)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
