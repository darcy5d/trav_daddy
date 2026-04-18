#!/usr/bin/env bash
# Wave 4.5 Phase 3 - run the A/B sweep over three v2 training variations,
# back-test each on the same 50-match IPL holdout, and append a row to
# data/backtest/wave_4_5_ab.csv per variation.
#
# Variations:
#   A1: uniform class weights (no upweighting)
#   A2: per-over loss weight 0.1 (was 0.3)
#   A3: half-life 180 days (REBUILDS data; ~5 min per gender extra)
#
# Each variation: ~10 min train + ~10 min backtest = ~20 min.
# A3 adds ~10 min for data rebuild on top.
#
# Total wall-clock estimate: 60-75 minutes.

set -e

PYTHON=venv311/bin/python
HOLDOUT_FLAGS="--tournament-pattern '%Indian Premier League%' --since-date 2025-01-01 --limit 50 --n-sims 500"
CSV=data/backtest/wave_4_5_ab.csv

mkdir -p data/backtest data/diagnostics

if [ ! -f "$CSV" ]; then
    echo "label,description,brier,log_loss,accuracy,mae_total_runs,mae_margin_runs" > "$CSV"
fi

extract_metric() {
    # extract_metric <summary.json> <metric>
    $PYTHON -c "import json; d=json.load(open('$1'))['metrics']; print(d.get('$2', ''))"
}

run_variant() {
    local label="$1"
    local desc="$2"
    shift 2
    local extra_train_flags="$*"

    echo
    echo "============================================================"
    echo "VARIANT: $label - $desc"
    echo "  train flags: $extra_train_flags"
    echo "============================================================"

    # Train
    $PYTHON scripts/train_cricket_model_v2.py \
        --epochs 12 --batch-size 4096 --vocab-min-count 5 \
        --label "$label" $extra_train_flags 2>&1 | tee "/tmp/train_$label.log"

    # Backtest
    eval $PYTHON scripts/backtest_simulator.py --model-version v2 \
        $HOLDOUT_FLAGS --label "v2_${label}_ipl_2025" 2>&1 | tee "/tmp/backtest_${label}.log"

    # Append metrics row
    summary="data/backtest/backtest_v2_${label}_ipl_2025_summary.json"
    brier=$(extract_metric "$summary" brier_score)
    logloss=$(extract_metric "$summary" log_loss)
    acc=$(extract_metric "$summary" accuracy_top_pick)
    mae_runs=$(extract_metric "$summary" mae_score_runs)
    mae_margin=$(extract_metric "$summary" mae_margin_runs)
    echo "$label,$desc,$brier,$logloss,$acc,$mae_runs,$mae_margin" >> "$CSV"
    echo "Captured: $label brier=$brier log_loss=$logloss mae_runs=$mae_runs"
}

# Add baselines to CSV (one-off; idempotent on file)
if ! grep -q "^v1_baseline," "$CSV"; then
    summary="data/backtest/backtest_baseline_ipl_2025_summary.json"
    if [ -f "$summary" ]; then
        brier=$(extract_metric "$summary" brier_score)
        logloss=$(extract_metric "$summary" log_loss)
        acc=$(extract_metric "$summary" accuracy_top_pick)
        mae_runs=$(extract_metric "$summary" mae_score_runs)
        mae_margin=$(extract_metric "$summary" mae_margin_runs)
        echo "v1_baseline,V1 simulator (Wave 3.5 baseline),$brier,$logloss,$acc,$mae_runs,$mae_margin" >> "$CSV"
    fi
fi
if ! grep -q "^v2_first_pass," "$CSV"; then
    summary="data/backtest/backtest_v2_winfix_ipl_2025_summary.json"
    if [ -f "$summary" ]; then
        brier=$(extract_metric "$summary" brier_score)
        logloss=$(extract_metric "$summary" log_loss)
        acc=$(extract_metric "$summary" accuracy_top_pick)
        mae_runs=$(extract_metric "$summary" mae_score_runs)
        mae_margin=$(extract_metric "$summary" mae_margin_runs)
        echo "v2_first_pass,V2 first pass (mild class weights + win-cond fix),$brier,$logloss,$acc,$mae_runs,$mae_margin" >> "$CSV"
    fi
fi

echo
echo "================ Phase 3 A/B sweep starting ================"

# A1: uniform class weights
run_variant "ab_a1_uniform" "uniform class weights" --class-weight-mode uniform

# A2: per-over loss weight 0.1
run_variant "ab_a2_overw01" "per-over loss weight 0.1" --over-loss-weight 0.1

# A3: tighter half-life (180 days) - requires data rebuild
echo
echo "============================================================"
echo "VARIANT A3: rebuilding training data with half-life=180"
echo "============================================================"
$PYTHON scripts/build_ball_training_v2.py --gender male   --half-life-days 180 2>&1 | tee /tmp/build_male_180.log
$PYTHON scripts/build_ball_training_v2.py --gender female --half-life-days 180 2>&1 | tee /tmp/build_female_180.log
run_variant "ab_a3_halflife180" "half-life 180 days (data rebuilt)"

echo
echo "================ Phase 3 A/B sweep complete ================"
echo
echo "Results table:"
column -t -s, "$CSV"
