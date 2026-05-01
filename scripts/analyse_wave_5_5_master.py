#!/usr/bin/env python3
"""Wave 5.5 Phase C: master analyser for the wide sweep.

Reads:
- data/diagnostics/wave_5_5_wide_sweep/master_bets.csv (one row per
  (match, market, outcome, lookback, sim_version, toss_mode) bet)
- data/diagnostics/wave_5_5_wide_sweep/calibration_master.csv (one row per
  (match, sim_version, toss_mode) - all matches incl. no-Polymarket)

Produces:
- data/diagnostics/wave_5_5_master_summary.md with:
    * Top combos by ROI * sqrt(n)
    * Per-tournament heatmap: ROI by (lookback, threshold) cells
    * V3-vs-V2 lift table at each (tournament, market, lookback)
    * Calibration improvement table per tournament
    * Recommended BETTING_AUTO_MARKETS line
    * Honest summary of where V3 didn't help

Bootstrap-CI on ROI uses 1000 resamples by default. A combo qualifies as
'auto-eligible' if n_bets >= 50 AND ci_lower_95 > 0.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _safe_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _safe_int(v) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def _bootstrap_ci(values: List[float], n_resamples: int = 1000, alpha: float = 0.05, rng_seed: int = 42) -> Tuple[float, float, float]:
    """Bootstrap CI on the MEAN of `values`. Returns (mean, lower, upper)."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(rng_seed)
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    boot_means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = arr[idx].mean()
    lower = float(np.quantile(boot_means, alpha / 2))
    upper = float(np.quantile(boot_means, 1 - alpha / 2))
    return float(arr.mean()), lower, upper


def _safe_log(p: float, eps: float = 1e-6) -> float:
    return math.log(min(max(p, eps), 1.0 - eps))


def _lookback_sort_key(label: str) -> float:
    s = label.replace("T-", "").strip()
    if s.endswith("min"):
        return float(s[:-3]) / 60.0
    if s.endswith("h"):
        return float(s[:-1])
    if s.endswith("d"):
        return float(s[:-1]) * 24.0
    return 0.0


def load_bets(path: Path) -> List[Dict[str, Any]]:
    """Load and lightly typecast the master bets CSV."""
    rows = []
    with path.open() as fp:
        for row in csv.DictReader(fp):
            row["edge_pp"] = _safe_float(row.get("edge_pp"))
            row["pnl_usd"] = _safe_float(row.get("pnl_usd"))
            row["bet_size_usd"] = _safe_float(row.get("bet_size_usd")) or 25.0
            row["model_prob"] = _safe_float(row.get("model_prob"))
            row["market_price_pre"] = _safe_float(row.get("market_price_pre"))
            row["settle_outcome"] = _safe_float(row.get("settle_outcome"))
            rows.append(row)
    return rows


def load_calibration(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open() as fp:
        for row in csv.DictReader(fp):
            row["sim_team1_win_prob"] = _safe_float(row.get("sim_team1_win_prob"))
            row["actual_team1_won"] = _safe_int(row.get("actual_team1_won"))
            row["sim_avg_team1_score"] = _safe_float(row.get("sim_avg_team1_score"))
            row["sim_avg_team2_score"] = _safe_float(row.get("sim_avg_team2_score"))
            row["actual_team1_total"] = _safe_int(row.get("actual_team1_total"))
            row["actual_team2_total"] = _safe_int(row.get("actual_team2_total"))
            rows.append(row)
    return rows


def aggregate_combo(bets: List[Dict[str, Any]], threshold_pp: float) -> Dict[str, Any]:
    """Per-combo aggregation: filter by edge_threshold, compute ROI + bootstrap CI."""
    qualifying = [b for b in bets if b["edge_pp"] is not None and b["edge_pp"] >= threshold_pp and b["pnl_usd"] is not None]
    n = len(qualifying)
    if n == 0:
        return {"n": 0, "win_rate": None, "roi_pct": None, "roi_ci_lower": None, "roi_ci_upper": None, "total_pnl": 0.0}

    pnls = [b["pnl_usd"] for b in qualifying]
    sizes = [b["bet_size_usd"] for b in qualifying]
    total_pnl = sum(pnls)
    total_staked = sum(sizes)
    n_wins = sum(1 for p in pnls if p > 0)

    # ROI per bet: pnl / bet_size, then bootstrap-mean.
    per_bet_roi = [p / s for p, s in zip(pnls, sizes)]
    mean_roi, ci_lower, ci_upper = _bootstrap_ci(per_bet_roi)

    return {
        "n": n,
        "win_rate": round(n_wins / n, 4),
        "roi_pct": round(mean_roi * 100, 2),
        "roi_ci_lower": round(ci_lower * 100, 2),
        "roi_ci_upper": round(ci_upper * 100, 2),
        "total_pnl": round(total_pnl, 2),
        "total_staked": round(total_staked, 2),
    }


def calibration_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Brier, log-loss, accuracy, score MAE on a calibration row set."""
    decisive = [r for r in rows if r["actual_team1_won"] is not None and r["sim_team1_win_prob"] is not None]
    n = len(decisive)
    if n == 0:
        return {"n": 0}
    brier = sum((r["sim_team1_win_prob"] - r["actual_team1_won"]) ** 2 for r in decisive) / n
    log_loss = -sum(
        r["actual_team1_won"] * _safe_log(r["sim_team1_win_prob"])
        + (1 - r["actual_team1_won"]) * _safe_log(1 - r["sim_team1_win_prob"])
        for r in decisive
    ) / n
    hit = sum(1 for r in decisive
              if (r["sim_team1_win_prob"] > 0.5 and r["actual_team1_won"] == 1)
              or (r["sim_team1_win_prob"] < 0.5 and r["actual_team1_won"] == 0))
    accuracy = hit / n

    score_maes = []
    margin_diffs = []
    for r in decisive:
        if r["actual_team1_total"] is not None and r["sim_avg_team1_score"] is not None:
            score_maes.append(abs(r["actual_team1_total"] - r["sim_avg_team1_score"]))
        if r["actual_team2_total"] is not None and r["sim_avg_team2_score"] is not None:
            score_maes.append(abs(r["actual_team2_total"] - r["sim_avg_team2_score"]))
        if (r["actual_team1_total"] is not None and r["actual_team2_total"] is not None
                and r["sim_avg_team1_score"] is not None and r["sim_avg_team2_score"] is not None):
            actual_margin = r["actual_team1_total"] - r["actual_team2_total"]
            sim_margin = r["sim_avg_team1_score"] - r["sim_avg_team2_score"]
            margin_diffs.append(abs(sim_margin - actual_margin))
    return {
        "n": n,
        "accuracy_top_pick": round(accuracy, 4),
        "brier_score": round(brier, 4),
        "log_loss": round(log_loss, 4),
        "mae_score_runs": round(sum(score_maes) / len(score_maes), 2) if score_maes else None,
        "mae_margin_runs": round(sum(margin_diffs) / len(margin_diffs), 2) if margin_diffs else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bets-csv", default="data/diagnostics/wave_5_5_wide_sweep/master_bets.csv")
    parser.add_argument("--calibration-csv", default="data/diagnostics/wave_5_5_wide_sweep/calibration_master.csv")
    parser.add_argument("--output", default="data/diagnostics/wave_5_5_master_summary.md")
    parser.add_argument("--edge-thresholds", default="3,5,10,20")
    parser.add_argument("--min-n-for-auto", type=int, default=50,
                        help="Min n_bets to qualify a combo for AUTO mode")
    args = parser.parse_args()

    bets_path = Path(args.bets_csv)
    calib_path = Path(args.calibration_csv)
    if not bets_path.exists():
        logger.error(f"Bets CSV not found: {bets_path}")
        return 1
    if not calib_path.exists():
        logger.warning(f"Calibration CSV not found: {calib_path}; calibration sections will be empty")

    bets = load_bets(bets_path)
    calib_rows = load_calibration(calib_path) if calib_path.exists() else []
    edge_thresholds = [float(x) for x in args.edge_thresholds.split(",")]
    logger.info(f"Loaded {len(bets):,} bet rows + {len(calib_rows):,} calibration rows")

    # ----- Index bets by (sim_version, toss_mode, tournament, market, lookback) -----
    grouped: Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for b in bets:
        key = (
            b.get("sim_version", "?"),
            b.get("toss_mode", "?"),
            b.get("tournament", "?"),
            b.get("market_type", "?"),
            b.get("lookback_label", "?"),
        )
        grouped[key].append(b)

    # ----- Aggregate per combo per threshold -----
    combo_rows = []
    for (sim_v, toss, tour, market, lookback), group in grouped.items():
        for thr in edge_thresholds:
            agg = aggregate_combo(group, thr)
            if agg["n"] == 0:
                continue
            combo_rows.append({
                "sim": sim_v, "toss": toss, "tournament": tour,
                "market": market, "lookback": lookback, "threshold_pp": thr,
                **agg,
            })

    # ----- Top combos by ROI * sqrt(n) -----
    def _score(r):
        if r["roi_pct"] is None or r["n"] is None:
            return -float("inf")
        return r["roi_pct"] * math.sqrt(r["n"])
    top_combos = sorted(combo_rows, key=_score, reverse=True)[:25]

    # Auto-eligible: n >= min AND ci_lower > 0
    auto_eligible = [
        r for r in combo_rows
        if (r["n"] or 0) >= args.min_n_for_auto and (r.get("roi_ci_lower") or -1) > 0
    ]
    auto_eligible.sort(key=_score, reverse=True)

    # ----- V3 vs V2 lift -----
    # For each (tournament, market, lookback, threshold), find both V2-uncertain
    # and V3-{pinned, marginalised} rows; lift = v3_roi - v2_roi.
    lift_rows = []
    by_axes: Dict[Tuple[str, str, str, float], Dict[Tuple[str, str], Dict]] = defaultdict(dict)
    for r in combo_rows:
        axes = (r["tournament"], r["market"], r["lookback"], r["threshold_pp"])
        by_axes[axes][(r["sim"], r["toss"])] = r
    for axes, sim_dict in by_axes.items():
        v2 = sim_dict.get(("v2", "uncertain"))
        if not v2:
            continue
        for (sim, toss), v3 in sim_dict.items():
            if sim != "v3":
                continue
            lift_rows.append({
                "tournament": axes[0],
                "market": axes[1],
                "lookback": axes[2],
                "threshold_pp": axes[3],
                "v3_toss": toss,
                "v2_n": v2["n"], "v2_roi": v2["roi_pct"],
                "v3_n": v3["n"], "v3_roi": v3["roi_pct"],
                "lift_pp": (v3["roi_pct"] or 0) - (v2["roi_pct"] or 0),
            })
    lift_rows.sort(key=lambda r: r["lift_pp"], reverse=True)

    # ----- Per-(sim_version, toss_mode, tournament) calibration -----
    calib_grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for c in calib_rows:
        # calibration CSV's tournament needs to be derived from event_name or
        # from the tournament column added by consolidate_csvs.
        tour = c.get("tournament") or c.get("event_name", "?")[:30]
        calib_grouped[(c.get("sim_version", "?"), c.get("toss_mode", "?"), tour)].append(c)

    calib_aggs = {}
    for key, rows in calib_grouped.items():
        calib_aggs[key] = calibration_metrics(rows)

    # ----- Render markdown -----
    lines = []
    lines.append("# Wave 5.5 Master Sweep Summary\n\n")
    lines.append(f"Generated by `scripts/analyse_wave_5_5_master.py`. "
                 f"{len(bets):,} bet rows; {len(calib_rows):,} calibration rows.\n\n")

    lines.append("## Top 25 combos by ROI * sqrt(n) (rewards both magnitude and sample size)\n\n")
    lines.append("| Rank | Sim | Toss | Tournament | Market | Lookback | Edge>= | n | Win rate | ROI | CI95 lower | CI95 upper |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for i, r in enumerate(top_combos, 1):
        lines.append(
            f"| {i} | {r['sim']} | {r['toss']} | {r['tournament']} | {r['market']} | "
            f"{r['lookback']} | {r['threshold_pp']}pp | {r['n']} | {r['win_rate']} | "
            f"{r['roi_pct']}% | {r['roi_ci_lower']}% | {r['roi_ci_upper']}% |\n"
        )
    lines.append("\n")

    lines.append(f"## Auto-eligible combos (n >= {args.min_n_for_auto} AND CI95-lower > 0)\n\n")
    if not auto_eligible:
        lines.append("**No combos passed the auto-eligibility gate.**\n\n")
        lines.append("Recommend `BETTING_MODE=MANUAL` for first week. Operator manually reviews\n")
        lines.append("the highest-EV surfaced edges in the Live Betting UI.\n\n")
    else:
        lines.append(f"{len(auto_eligible)} combos qualify. Recommended env vars:\n\n")
        markets = sorted({r["market"] for r in auto_eligible})
        thresholds = sorted({r["threshold_pp"] for r in auto_eligible})
        lines.append("```\n")
        lines.append(f"BETTING_AUTO_MARKETS={','.join(markets)}\n")
        lines.append(f"BETTING_AUTO_MIN_EDGE={min(thresholds)}\n")
        lines.append("```\n\n")
        lines.append("| Sim | Toss | Tournament | Market | Lookback | Edge>= | n | ROI | CI95 lower |\n")
        lines.append("|---|---|---|---|---|---|---|---|---|\n")
        for r in auto_eligible[:30]:
            lines.append(
                f"| {r['sim']} | {r['toss']} | {r['tournament']} | {r['market']} | "
                f"{r['lookback']} | {r['threshold_pp']}pp | {r['n']} | "
                f"{r['roi_pct']}% | {r['roi_ci_lower']}% |\n"
            )
        lines.append("\n")

    # V3 vs V2 lift
    lines.append("## V3 vs V2 ROI lift (top 30 combos by lift_pp)\n\n")
    lines.append("Positive lift_pp = V3 outperforms V2 on the same (tournament, market, lookback, threshold).\n\n")
    lines.append("| Tournament | Market | Lookback | Edge>= | V3 toss | V2 n / ROI | V3 n / ROI | Lift |\n")
    lines.append("|---|---|---|---|---|---|---|---|\n")
    for r in lift_rows[:30]:
        lines.append(
            f"| {r['tournament']} | {r['market']} | {r['lookback']} | {r['threshold_pp']}pp | "
            f"{r['v3_toss']} | {r['v2_n']} / {r['v2_roi']}% | {r['v3_n']} / {r['v3_roi']}% | "
            f"{r['lift_pp']:+.2f}pp |\n"
        )
    lines.append("\n")

    # Calibration table
    lines.append("## Calibration metrics per (sim_version, toss_mode, tournament)\n\n")
    lines.append("Brier 0.25 = always 50/50; lower is better. Log-loss 0.693 = always 50/50.\n\n")
    lines.append("| Sim | Toss | Tournament | n | Acc | Brier | LogLoss | MAE runs | MAE margin |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|\n")
    for (sim, toss, tour), agg in sorted(calib_aggs.items()):
        if agg.get("n", 0) < 5:
            continue
        lines.append(
            f"| {sim} | {toss} | {tour} | {agg['n']} | {agg.get('accuracy_top_pick','--')} | "
            f"{agg.get('brier_score','--')} | {agg.get('log_loss','--')} | "
            f"{agg.get('mae_score_runs','--')} | {agg.get('mae_margin_runs','--')} |\n"
        )
    lines.append("\n")

    # Honest summary of where V3 didn't help
    v3_worse = [r for r in lift_rows if (r.get("lift_pp") or 0) < -2]
    lines.append("## Where V3 underperformed V2 (>=2pp lift gap on the wrong side)\n\n")
    if not v3_worse:
        lines.append("None - V3 either matched or beat V2 on every meaningful combo.\n\n")
    else:
        lines.append(f"{len(v3_worse)} combos. Top 15 by negative lift:\n\n")
        lines.append("| Tournament | Market | Lookback | Edge>= | V3 toss | V2 ROI | V3 ROI | Lift |\n")
        lines.append("|---|---|---|---|---|---|---|---|\n")
        for r in sorted(v3_worse, key=lambda x: x["lift_pp"])[:15]:
            lines.append(
                f"| {r['tournament']} | {r['market']} | {r['lookback']} | {r['threshold_pp']}pp | "
                f"{r['v3_toss']} | {r['v2_roi']}% | {r['v3_roi']}% | {r['lift_pp']:+.2f}pp |\n"
            )
        lines.append("\n")

    # Final recommendation
    lines.append("## Recommendation\n\n")
    if auto_eligible:
        lines.append("- Update `.env` with the BETTING_AUTO_MARKETS and BETTING_AUTO_MIN_EDGE shown above.\n")
        lines.append("- Bootstrap a Polygon wallet with $200 USDC.\n")
        lines.append("- Switch BETTING_MODE=MANUAL via the Live Betting UI.\n")
        lines.append(f"- Place small ($5-10) test bets in the highest-ROI combos until 50 settled bets accumulate.\n")
        lines.append("- After 50 settled bets, use the Phase 7 dashboard scale-up gate to graduate to $500 envelope.\n")
    else:
        lines.append("- DO NOT bootstrap wallet yet. No combo has confident positive ROI.\n")
        lines.append("- Investigate calibration of V3 in tournaments where Brier > 0.30 (likely needs separate refit).\n")
        lines.append("- Consider Wave 5.6: per-tournament calibration models, or per-player XI attention (the deferred Wave 7 item).\n")
    lines.append("\n")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("".join(lines))
    logger.info(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
