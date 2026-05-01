#!/usr/bin/env python3
"""Wave 5.6 Phase 4: outlier-robust master analyser.

Improvements over `analyse_wave_5_5_master.py`:

1. **Drop lottery-ticket bets**: filters out bets where the entry market_price
   is < 0.10 or > 0.90. Below 10c (or above 90c, since shares cost
   1-price for the OTHER side) means the model is betting on heavy
   underdogs/favourites where 1 win pays 10x+ stake. These rows generated
   spurious 199,898% ROIs that drowned out real signal in Wave 5.5.

2. **Winsorized per-bet ROI** at the 5th/95th percentile for headline
   metrics. Caps the influence of any single outlier bet on the aggregate
   ROI. Untrimmed ROI is still shown alongside as 'roi_raw_pct' for
   transparency.

3. **Median per-bet ROI** as a robust complement to the (Winsorized) mean.

4. **Sharpe-like ratio** = mean_roi / std_roi (sample-size adjusted).
   Helps distinguish "small but consistent edge" combos from "high variance
   wins offset by losses" combos.

Usage:
    venv311/bin/python scripts/analyse_wave_5_6_master.py
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Filter: drop bets entered at very long-odds prices. Anything below 0.10
# (or above 0.90, since the OTHER side then trades below 0.10) generates
# >10x payouts where 1 win can dominate the bootstrap.
PRICE_MIN = 0.10
PRICE_MAX = 0.90

# Winsorize per-bet ROI at the 5th/95th percentile. Caps each bet's
# contribution to the aggregate at its empirical p95 / p5.
WINSORIZE_LOWER_PCT = 5.0
WINSORIZE_UPPER_PCT = 95.0


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


def _winsorize(values: np.ndarray, lower_pct: float, upper_pct: float) -> np.ndarray:
    """Cap values at the lower_pct / upper_pct percentiles (in-place safe)."""
    if values.size == 0:
        return values
    lo = np.percentile(values, lower_pct)
    hi = np.percentile(values, upper_pct)
    return np.clip(values, lo, hi)


def _bootstrap_ci(values: np.ndarray, n_resamples: int = 1000, alpha: float = 0.05, rng_seed: int = 42) -> Tuple[float, float, float]:
    if values.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(rng_seed)
    n = values.size
    boot_means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = values[idx].mean()
    lower = float(np.quantile(boot_means, alpha / 2))
    upper = float(np.quantile(boot_means, 1 - alpha / 2))
    return float(values.mean()), lower, upper


def _safe_log(p: float, eps: float = 1e-6) -> float:
    return math.log(min(max(p, eps), 1.0 - eps))


def load_bets(path: Path) -> List[Dict[str, Any]]:
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
    """Per-combo aggregation with outlier filtering + Winsorization."""
    qualifying = [
        b for b in bets
        if b["edge_pp"] is not None and b["edge_pp"] >= threshold_pp
        and b["pnl_usd"] is not None
        and b["market_price_pre"] is not None
        and PRICE_MIN <= b["market_price_pre"] <= PRICE_MAX
    ]
    n = len(qualifying)
    if n == 0:
        return {"n": 0, "win_rate": None, "roi_pct": None, "roi_raw_pct": None,
                "roi_ci_lower": None, "roi_ci_upper": None,
                "roi_median_pct": None, "sharpe": None, "total_pnl": 0.0}

    pnls = np.array([b["pnl_usd"] for b in qualifying], dtype=np.float64)
    sizes = np.array([b["bet_size_usd"] for b in qualifying], dtype=np.float64)
    per_bet_roi = pnls / sizes  # raw per-bet returns

    raw_mean = float(per_bet_roi.mean())
    median_roi = float(np.median(per_bet_roi))

    # Winsorize for headline ROI + bootstrap CI
    wins_roi = _winsorize(per_bet_roi.copy(), WINSORIZE_LOWER_PCT, WINSORIZE_UPPER_PCT)
    wins_mean, ci_lower, ci_upper = _bootstrap_ci(wins_roi)

    n_wins = int((per_bet_roi > 0).sum())
    sharpe = (
        wins_mean / per_bet_roi.std() * math.sqrt(n)
        if per_bet_roi.std() > 1e-9 else None
    )

    return {
        "n": n,
        "win_rate": round(n_wins / n, 4),
        "roi_pct": round(wins_mean * 100, 2),         # Winsorized headline
        "roi_raw_pct": round(raw_mean * 100, 2),       # raw for transparency
        "roi_median_pct": round(median_roi * 100, 2),
        "roi_ci_lower": round(ci_lower * 100, 2),
        "roi_ci_upper": round(ci_upper * 100, 2),
        "sharpe": round(sharpe, 3) if sharpe is not None else None,
        "total_pnl": round(float(pnls.sum()), 2),
        "total_staked": round(float(sizes.sum()), 2),
    }


def calibration_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
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
    for r in decisive:
        if r["actual_team1_total"] is not None and r["sim_avg_team1_score"] is not None:
            score_maes.append(abs(r["actual_team1_total"] - r["sim_avg_team1_score"]))
        if r["actual_team2_total"] is not None and r["sim_avg_team2_score"] is not None:
            score_maes.append(abs(r["actual_team2_total"] - r["sim_avg_team2_score"]))
    return {
        "n": n,
        "accuracy_top_pick": round(accuracy, 4),
        "brier_score": round(brier, 4),
        "log_loss": round(log_loss, 4),
        "mae_score_runs": round(sum(score_maes) / len(score_maes), 2) if score_maes else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bets-csv", default="data/diagnostics/wave_5_6_focused_sweep/master_bets.csv")
    parser.add_argument("--calibration-csv", default="data/diagnostics/wave_5_6_focused_sweep/calibration_master.csv")
    parser.add_argument("--output", default="data/diagnostics/wave_5_6_master_summary.md")
    parser.add_argument("--edge-thresholds", default="3,5,10,20")
    parser.add_argument("--min-n-for-auto", type=int, default=50)
    args = parser.parse_args()

    bets_path = Path(args.bets_csv)
    calib_path = Path(args.calibration_csv)
    if not bets_path.exists():
        logger.error(f"Bets CSV not found: {bets_path}")
        return 1

    bets_raw = load_bets(bets_path)
    bets_filtered = [
        b for b in bets_raw
        if b["market_price_pre"] is not None
        and PRICE_MIN <= b["market_price_pre"] <= PRICE_MAX
    ]
    n_dropped = len(bets_raw) - len(bets_filtered)
    calib_rows = load_calibration(calib_path) if calib_path.exists() else []
    edge_thresholds = [float(x) for x in args.edge_thresholds.split(",")]
    logger.info(f"Loaded {len(bets_raw):,} bet rows, filtered out {n_dropped} long-shot bets "
                f"(market_price < {PRICE_MIN} or > {PRICE_MAX}); {len(bets_filtered):,} remaining")
    logger.info(f"Loaded {len(calib_rows):,} calibration rows")

    # Group bets by combo
    grouped: Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for b in bets_filtered:
        key = (
            b.get("sim_version", "?"),
            b.get("toss_mode", "?"),
            b.get("tournament", "?"),
            b.get("market_type", "?"),
            b.get("lookback_label", "?"),
        )
        grouped[key].append(b)

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

    # Top combos by Winsorized ROI * sqrt(n)
    def _score(r):
        if r["roi_pct"] is None or r["n"] is None:
            return -float("inf")
        return r["roi_pct"] * math.sqrt(r["n"])
    top_combos = sorted(combo_rows, key=_score, reverse=True)[:25]

    auto_eligible = [
        r for r in combo_rows
        if (r["n"] or 0) >= args.min_n_for_auto and (r.get("roi_ci_lower") or -1) > 0
    ]
    auto_eligible.sort(key=_score, reverse=True)

    # V3 vs V2 lift
    by_axes: Dict[Tuple[str, str, str, float], Dict[Tuple[str, str], Dict]] = defaultdict(dict)
    for r in combo_rows:
        axes = (r["tournament"], r["market"], r["lookback"], r["threshold_pp"])
        by_axes[axes][(r["sim"], r["toss"])] = r
    lift_rows = []
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

    # Calibration aggregations per (sim, toss, tournament)
    calib_grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for c in calib_rows:
        tour = c.get("tournament") or c.get("event_name", "?")[:30]
        calib_grouped[(c.get("sim_version", "?"), c.get("toss_mode", "?"), tour)].append(c)
    calib_aggs = {key: calibration_metrics(rows) for key, rows in calib_grouped.items()}

    # ----- Render markdown -----
    lines = []
    lines.append("# Wave 5.6 Master Summary (focused, outlier-robust)\n\n")
    lines.append(f"Generated by `scripts/analyse_wave_5_6_master.py`. "
                 f"{len(bets_raw):,} raw bet rows; {n_dropped:,} dropped as long-shot lottery tickets "
                 f"(market_price outside [{PRICE_MIN}, {PRICE_MAX}]); {len(bets_filtered):,} retained. "
                 f"Headline ROI is Winsorized at the {WINSORIZE_LOWER_PCT}/{WINSORIZE_UPPER_PCT} "
                 f"percentile to neutralise any remaining lucky-bet dominance.\n\n")

    lines.append("## Top 25 combos by Winsorized ROI * sqrt(n)\n\n")
    lines.append("| Rank | Sim | Toss | Tournament | Market | Lookback | Edge>= | n | Win | ROI (W) | ROI (raw) | Median | CI95 lower | CI95 upper | Sharpe |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for i, r in enumerate(top_combos, 1):
        lines.append(
            f"| {i} | {r['sim']} | {r['toss']} | {r['tournament']} | {r['market']} | "
            f"{r['lookback']} | {r['threshold_pp']}pp | {r['n']} | {r['win_rate']} | "
            f"{r['roi_pct']}% | {r['roi_raw_pct']}% | {r['roi_median_pct']}% | "
            f"{r['roi_ci_lower']}% | {r['roi_ci_upper']}% | {r['sharpe']} |\n"
        )
    lines.append("\n")

    lines.append(f"## Auto-eligible combos (n >= {args.min_n_for_auto} AND CI95-lower > 0)\n\n")
    if not auto_eligible:
        lines.append("**No combos passed the auto-eligibility gate.**\n\n")
        lines.append("Recommended: stay BETTING_MODE=OFF. Investigate individual high-ROI low-n combos manually,\n")
        lines.append("but do not bootstrap a wallet for AUTO-mode betting on this evidence.\n\n")
    else:
        lines.append(f"{len(auto_eligible)} combos qualify. Recommended env vars:\n\n")
        markets = sorted({r["market"] for r in auto_eligible})
        thresholds = sorted({r["threshold_pp"] for r in auto_eligible})
        lines.append("```\n")
        lines.append(f"BETTING_AUTO_MARKETS={','.join(markets)}\n")
        lines.append(f"BETTING_AUTO_MIN_EDGE={min(thresholds)}\n")
        lines.append("```\n\n")
        lines.append("| Sim | Toss | Tournament | Market | Lookback | Edge>= | n | Win | ROI (W) | CI95 lower | Sharpe |\n")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in auto_eligible[:30]:
            lines.append(
                f"| {r['sim']} | {r['toss']} | {r['tournament']} | {r['market']} | "
                f"{r['lookback']} | {r['threshold_pp']}pp | {r['n']} | {r['win_rate']} | "
                f"{r['roi_pct']}% | {r['roi_ci_lower']}% | {r['sharpe']} |\n"
            )
        lines.append("\n")

    lines.append("## V3 vs V2 ROI lift (top 30 combos by lift_pp on Winsorized ROI)\n\n")
    lines.append("| Tournament | Market | Lookback | Edge>= | V3 toss | V2 n / ROI | V3 n / ROI | Lift |\n")
    lines.append("|---|---|---|---|---|---|---|---|\n")
    for r in lift_rows[:30]:
        lines.append(
            f"| {r['tournament']} | {r['market']} | {r['lookback']} | {r['threshold_pp']}pp | "
            f"{r['v3_toss']} | {r['v2_n']} / {r['v2_roi']}% | {r['v3_n']} / {r['v3_roi']}% | "
            f"{r['lift_pp']:+.2f}pp |\n"
        )
    lines.append("\n")

    lines.append("## Calibration metrics per (sim_version, toss_mode, tournament)\n\n")
    lines.append("Brier 0.25 = always 50/50; lower is better. Log-loss 0.693 = always 50/50.\n\n")
    lines.append("| Sim | Toss | Tournament | n | Acc | Brier | LogLoss | MAE runs |\n")
    lines.append("|---|---|---|---|---|---|---|---|\n")
    for (sim, toss, tour), agg in sorted(calib_aggs.items()):
        if agg.get("n", 0) < 5:
            continue
        lines.append(
            f"| {sim} | {toss} | {tour} | {agg['n']} | {agg.get('accuracy_top_pick','--')} | "
            f"{agg.get('brier_score','--')} | {agg.get('log_loss','--')} | "
            f"{agg.get('mae_score_runs','--')} |\n"
        )
    lines.append("\n")

    v3_worse = [r for r in lift_rows if (r.get("lift_pp") or 0) < -2]
    lines.append(f"## Where V3 underperformed V2 (>=2pp negative lift; {len(v3_worse)} combos)\n\n")
    if not v3_worse:
        lines.append("None.\n\n")
    else:
        lines.append("| Tournament | Market | Lookback | Edge>= | V3 toss | V2 ROI | V3 ROI | Lift |\n")
        lines.append("|---|---|---|---|---|---|---|---|\n")
        for r in sorted(v3_worse, key=lambda x: x["lift_pp"])[:15]:
            lines.append(
                f"| {r['tournament']} | {r['market']} | {r['lookback']} | {r['threshold_pp']}pp | "
                f"{r['v3_toss']} | {r['v2_roi']}% | {r['v3_roi']}% | {r['lift_pp']:+.2f}pp |\n"
            )
        lines.append("\n")

    lines.append("## Recommendation\n\n")
    if auto_eligible:
        lines.append("- Update `.env` with the BETTING_AUTO_MARKETS and BETTING_AUTO_MIN_EDGE shown above.\n")
        lines.append("- Bootstrap a Polygon wallet with $200 USDC.\n")
        lines.append("- Switch BETTING_MODE=MANUAL via the Live Betting UI.\n")
        lines.append("- Place small ($5-10) test bets in the highest-Sharpe / highest-ROI combos.\n")
        lines.append("- After 50 settled bets, use the Phase 7 dashboard scale-up gate.\n")
    else:
        lines.append("- DO NOT bootstrap wallet yet. No combo has confident positive ROI.\n")
        lines.append("- The Winsorized ROI is now robust to lucky long-shot bets, so this is a real verdict.\n")
        lines.append("- Next: per-tournament calibration models, or focus on EXPANSION of holdout via paper-trading on live fixtures.\n")
    lines.append("\n")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("".join(lines))
    logger.info(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
