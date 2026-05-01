#!/usr/bin/env bash
# Wave 5 Phase 3: Wider holdout backtests across V1+V2 simulators and several
# tournaments / formats / genders.
#
# Captures per-match Polymarket sub-market predictions (top batter, most
# sixes) alongside the moneyline, so Phase 4's analysis script can compute
# per-tournament per-market calibration matrices.
#
# Each run produces a CSV (per-match) and a summary JSON in data/backtest/.
# Results are tagged with a label of the form `w5_<sim>_<tournament>` so the
# Phase 4 analysis script can find them.
#
# Run this in the background; expect ~20-60 minutes depending on LIMIT/N_SIMS.
#
# Defaults are sized for V2-only (LIMIT=20, N_SIMS=300, 6 backtests, ~20 min).
# Set SIMS=v1 v2 to also run V1 for the side-by-side comparison.

set -euo pipefail

PYTHON="${PYTHON:-venv311/bin/python}"
N_SIMS="${N_SIMS:-300}"
LIMIT="${LIMIT:-20}"
OUTPUT_DIR="${OUTPUT_DIR:-data/backtest}"
SIMS="${SIMS:-v2}"  # Space-separated list, e.g. "v1 v2"

mkdir -p "$OUTPUT_DIR"

run_backtest() {
    local sim="$1"
    local fmt="$2"
    local gender="$3"
    local tournament_pattern="$4"
    local since_date="$5"
    local label="$6"

    echo ""
    echo "=========================================================="
    echo "  $label  ($sim, $fmt $gender, tournament=$tournament_pattern)"
    echo "=========================================================="
    "$PYTHON" scripts/backtest_simulator.py \
        --model-version "$sim" \
        --format "$fmt" \
        --gender "$gender" \
        --tournament-pattern "$tournament_pattern" \
        --since-date "$since_date" \
        --limit "$LIMIT" \
        --n-sims "$N_SIMS" \
        --output-dir "$OUTPUT_DIR" \
        --label "$label" \
        || echo "WARN: $label failed (continuing with the rest of the sweep)"
}

# 6 backtests by default (V2 across 6 tournament/format/gender combos).
# When SIMS="v1 v2" -> 12 backtests for the side-by-side comparison.
# Tournament patterns are SQL LIKE strings matched against matches.event_name.

# 1. IPL T20 men 2025
for sim in $SIMS; do
    run_backtest "$sim" T20 male '%Indian Premier League%' 2025-01-01 "w5_${sim}_ipl"
done

# 2. PSL T20 men 2025-26
for sim in $SIMS; do
    run_backtest "$sim" T20 male '%Pakistan Super League%' 2025-01-01 "w5_${sim}_psl"
done

# 3. BBL T20 men 2024-25
for sim in $SIMS; do
    run_backtest "$sim" T20 male '%Big Bash League%' 2024-12-01 "w5_${sim}_bbl"
done

# 4. WPL T20 women 2024-25
for sim in $SIMS; do
    run_backtest "$sim" T20 female '%Women.s Premier League%' 2024-01-01 "w5_${sim}_wpl"
done

# 5. T20I men 2025
for sim in $SIMS; do
    run_backtest "$sim" T20 male '%T20I%' 2025-01-01 "w5_${sim}_t20i_men"
done

# 6. ODI men 2025
for sim in $SIMS; do
    run_backtest "$sim" ODI male '%' 2025-01-01 "w5_${sim}_odi_men"
done

echo ""
echo "All Wave 5 holdout backtests complete (SIMS=$SIMS)."
echo "Per-match CSVs and summary JSONs in $OUTPUT_DIR (label prefix: w5_)."
echo "Run scripts/analyse_wave_5_holdouts.py next to produce the validation report."
