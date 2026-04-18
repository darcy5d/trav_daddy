#!/usr/bin/env python3
"""
Diff two backtest summary JSONs side-by-side and verdict against the v2
promotion gate (Wave 4 Phase 6).

Promotion gate per (format, gender):
  v2 must beat v1 on Brier AND log-loss AND not regress accuracy by more
  than 1pp on >= MIN_MATCHES matches.

Run:
    python scripts/compare_backtests.py \
        --baseline data/backtest/backtest_baseline_ipl_2025_summary.json \
        --candidate data/backtest/backtest_v2_ipl_2025_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

MIN_MATCHES_FOR_VERDICT = 50  # below this, treat as "not enough data" rather than a fail
ACCURACY_REGRESSION_TOLERANCE_PP = 1.0  # max acceptable accuracy drop in percentage points


def _fmt_delta(new: Optional[float], old: Optional[float], lower_is_better: bool = True) -> str:
    if new is None or old is None:
        return ""
    delta = new - old
    direction = "improved" if (lower_is_better and delta < 0) or (not lower_is_better and delta > 0) else "regressed"
    if abs(delta) < 1e-6:
        direction = "unchanged"
    sign = "+" if delta > 0 else ""
    return f" ({sign}{delta:.4f}, {direction})"


def _verdict(baseline_metrics: Dict[str, Any], candidate_metrics: Dict[str, Any]) -> Dict[str, Any]:
    n_b = baseline_metrics.get("n_matches", 0)
    n_c = candidate_metrics.get("n_matches", 0)
    if n_c < MIN_MATCHES_FOR_VERDICT:
        return {"status": "insufficient_data", "reason": f"candidate n={n_c} < {MIN_MATCHES_FOR_VERDICT}"}

    brier_b = baseline_metrics.get("brier_score")
    brier_c = candidate_metrics.get("brier_score")
    log_b = baseline_metrics.get("log_loss")
    log_c = candidate_metrics.get("log_loss")
    acc_b = baseline_metrics.get("accuracy_top_pick")
    acc_c = candidate_metrics.get("accuracy_top_pick")

    failures = []
    if brier_b is not None and brier_c is not None and brier_c >= brier_b:
        failures.append(f"Brier did not improve: {brier_c:.4f} >= {brier_b:.4f}")
    if log_b is not None and log_c is not None and log_c >= log_b:
        failures.append(f"Log loss did not improve: {log_c:.4f} >= {log_b:.4f}")
    if acc_b is not None and acc_c is not None:
        regression_pp = (acc_b - acc_c) * 100
        if regression_pp > ACCURACY_REGRESSION_TOLERANCE_PP:
            failures.append(
                f"Accuracy regressed by {regression_pp:.2f}pp (> {ACCURACY_REGRESSION_TOLERANCE_PP}pp)"
            )

    if not failures:
        return {"status": "pass", "reason": "all promotion-gate criteria satisfied"}
    return {"status": "fail", "reason": " ; ".join(failures)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two backtest summary JSONs and verdict")
    parser.add_argument("--baseline", required=True, help="Baseline backtest summary JSON (v1)")
    parser.add_argument("--candidate", required=True, help="Candidate backtest summary JSON (v2)")
    args = parser.parse_args()

    with open(args.baseline) as fp:
        b = json.load(fp)
    with open(args.candidate) as fp:
        c = json.load(fp)

    bm = b.get("metrics", {})
    cm = c.get("metrics", {})
    b_label = b.get("label", "baseline")
    c_label = c.get("label", "candidate")
    b_version = b.get("args", {}).get("model_version", "v1")
    c_version = c.get("args", {}).get("model_version", "?")

    print()
    print("=" * 84)
    print(f"Backtest comparison: {b_label} ({b_version})  vs  {c_label} ({c_version})")
    print("=" * 84)
    print(f"  {'metric':<22}  {'baseline':>12}  {'candidate':>12}  {'change':<28}")

    def line(metric, key, lower_is_better=True, fmt="{:.4f}"):
        bv = bm.get(key)
        cv = cm.get(key)
        bv_s = fmt.format(bv) if bv is not None else "(n/a)"
        cv_s = fmt.format(cv) if cv is not None else "(n/a)"
        delta = _fmt_delta(cv, bv, lower_is_better=lower_is_better)
        print(f"  {metric:<22}  {bv_s:>12}  {cv_s:>12}  {delta}")

    line("n_matches", "n_matches", fmt="{:,}")
    line("n_decisive", "n_decisive", fmt="{:,}")
    line("accuracy_top_pick", "accuracy_top_pick", lower_is_better=False)
    line("brier_score", "brier_score", lower_is_better=True)
    line("log_loss", "log_loss", lower_is_better=True)
    line("mae_score_runs", "mae_score_runs", lower_is_better=True, fmt="{:.2f}")
    line("mae_margin_runs", "mae_margin_runs", lower_is_better=True, fmt="{:.2f}")

    # Per-tournament breakdown (intersection of keys)
    by_event_b = b.get("by_event", {})
    by_event_c = c.get("by_event", {})
    common_events = sorted(set(by_event_b) & set(by_event_c))
    if common_events:
        print()
        print("  Per-tournament (intersection):")
        for ev in common_events:
            b_acc = by_event_b[ev].get("accuracy_top_pick")
            c_acc = by_event_c[ev].get("accuracy_top_pick")
            b_brier = by_event_b[ev].get("brier_score")
            c_brier = by_event_c[ev].get("brier_score")
            n = by_event_b[ev].get("n_matches", 0)
            print(
                f"    {ev[:55]:<55} n={n:>4}  "
                f"acc {b_acc:.3f}->{c_acc:.3f}  "
                f"brier {b_brier:.4f}->{c_brier:.4f}"
            )

    # Verdict
    v = _verdict(bm, cm)
    print()
    print("-" * 84)
    print(f"Promotion gate verdict: {v['status'].upper()}")
    print(f"  reason: {v['reason']}")
    print("=" * 84)

    return 0 if v["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
