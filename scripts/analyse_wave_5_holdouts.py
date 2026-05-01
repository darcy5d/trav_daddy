#!/usr/bin/env python3
"""Wave 5 Phase 4: Per-tournament per-market validation report.

Reads the 12 backtest summary JSONs + per-match CSVs produced by
`scripts/run_wave_5_holdouts.sh` and produces
`data/diagnostics/wave_5_polymarket_summary.md` with:

- Per-tournament moneyline calibration matrix.
- Per-tournament top-batter top-1 + top-3 accuracy (vs Andrew Kuo's 25% / 80%).
- Per-tournament most-sixes accuracy.
- Per-tournament prediction compression band on each market type.
- Reference: V2 vs always-50/50 vs always-pick-favourite for each market.
- "Polymarket exploit potential" section: rough EV estimate per market type
  assuming current Polymarket prices (moneyline 95-99%/1-5% spread,
  Top Batter 91/9/91 spread, Most Sixes similar, TMD typically 22/28/22/28).

Usage:

    python scripts/analyse_wave_5_holdouts.py \\
        --backtest-dir data/backtest \\
        --label-prefix w5_ \\
        --output data/diagnostics/wave_5_polymarket_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Andrew Kuo reference numbers (towardsdatascience 2021):
ANDREW_KUO_TOP_BATTER_ACC = 0.25  # ~25% top-1 accuracy on top scorer (T20)
ANDREW_KUO_TOP_BATTER_ACC_TOP3 = 0.80  # rough estimate for top-3

# Rough current Polymarket spread assumptions (Wave 5 plan introduction).
# Used to project "if our model is right at this accuracy, what's our EV?".
POLYMARKET_SPREADS = {
    "moneyline": {"yes_typical": 0.50, "spread_pp": 5},  # tight, near-50/50 IPL
    "top_batter": {"yes_typical": 0.30, "spread_pp": 35},  # 91/9/91 sums to 1.91
    "most_sixes": {"yes_typical": 0.30, "spread_pp": 30},
    "toss_match_double": {"yes_typical": 0.25, "spread_pp": 10},
}
TAKER_FEE_PCT = 0.02


def discover_runs(backtest_dir: Path, label_prefix: str) -> List[Dict[str, Any]]:
    """Find every (label, summary_json, csv) triple in backtest_dir matching prefix."""
    out = []
    for json_path in sorted(backtest_dir.glob(f"backtest_{label_prefix}*_summary.json")):
        label = json_path.name[len("backtest_"):-len("_summary.json")]
        csv_path = backtest_dir / f"backtest_{label}.csv"
        if not csv_path.exists():
            logger.warning(f"Missing CSV for {label}: {csv_path}")
            continue
        with json_path.open() as fp:
            summary = json.load(fp)
        out.append({"label": label, "summary": summary, "csv_path": csv_path})
    return out


def _safe_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _safe_int(value: Optional[str]) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def load_csv(csv_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with csv_path.open() as fp:
        reader = csv.DictReader(fp)
        rows.extend(dict(r) for r in reader)
    return rows


def _format_pct(value: Optional[float], digits: int = 1) -> str:
    if value is None:
        return "--"
    return f"{value * 100:.{digits}f}%"


def _format_score(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def baseline_compares(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Reference baselines for each market: always-50/50 + always-favourite."""
    out: Dict[str, Any] = {
        "always_50_50_brier": 0.25,
        "always_50_50_log_loss": 0.6931,
    }
    decisive = [r for r in rows if _safe_int(r.get("team1_won")) is not None]
    if not decisive:
        return out
    # Always pick favourite: predicts 1.0 for the side with elo edge
    fav_correct = 0
    for r in decisive:
        team1_elo = _safe_float(r.get("team1_elo_used"))
        team2_elo = _safe_float(r.get("team2_elo_used"))
        if team1_elo is None or team2_elo is None:
            continue
        favourite = 1 if team1_elo > team2_elo else 0
        if int(r["team1_won"]) == favourite:
            fav_correct += 1
    out["always_favourite_accuracy"] = round(fav_correct / max(1, len(decisive)), 3)
    return out


def top_batter_metrics_for_run(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-run top-1 + top-3 accuracy across both teams, plus mean prob distribution."""
    pairs = []
    for r in rows:
        for prefix in ("team1", "team2"):
            sim_id = _safe_int(r.get(f"sim_{prefix}_top_batter_id"))
            sim_prob = _safe_float(r.get(f"sim_{prefix}_top_batter_prob"))
            top3_str = r.get(f"sim_{prefix}_top_3_batter_ids", "")
            actual_id = _safe_int(r.get(f"actual_{prefix}_top_batter_id"))
            if sim_id is None or actual_id is None:
                continue
            top3_ids = set()
            if top3_str:
                top3_ids = {int(x) for x in top3_str.split(",") if x.strip().lstrip("-").isdigit()}
            pairs.append((sim_id, top3_ids, sim_prob, actual_id))
    if not pairs:
        return {"n_pairs": 0, "top_1_accuracy": None, "top_3_accuracy": None, "mean_top_1_prob": None}
    n = len(pairs)
    top1 = sum(1 for sim, _t3, _p, actual in pairs if sim == actual)
    top3 = sum(1 for _sim, t3, _p, actual in pairs if actual in t3)
    mean_prob = sum((p or 0.0) for _sim, _t3, p, _actual in pairs) / n
    return {
        "n_pairs": n,
        "top_1_accuracy": round(top1 / n, 4),
        "top_3_accuracy": round(top3 / n, 4),
        "mean_top_1_prob": round(mean_prob, 4),
    }


def most_sixes_metrics_for_run(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    pairs = []
    for r in rows:
        sim_t1 = _safe_float(r.get("sim_most_sixes_team1_prob"))
        sim_dr = _safe_float(r.get("sim_most_sixes_draw_prob"))
        sim_t2 = _safe_float(r.get("sim_most_sixes_team2_prob"))
        actual_t1 = _safe_int(r.get("actual_team1_sixes"))
        actual_t2 = _safe_int(r.get("actual_team2_sixes"))
        if sim_t1 is None or actual_t1 is None or actual_t2 is None:
            continue
        if actual_t1 > actual_t2:
            outcome = "team1"
        elif actual_t1 < actual_t2:
            outcome = "team2"
        else:
            outcome = "draw"
        probs = {"team1": sim_t1, "draw": sim_dr or 0.0, "team2": sim_t2 or 0.0}
        pick = max(probs, key=probs.get)
        pairs.append((pick, outcome, probs))
    if not pairs:
        return {"n_pairs": 0, "accuracy": None, "brier": None}
    n = len(pairs)
    accuracy = sum(1 for pick, actual, _ in pairs if pick == actual) / n
    brier_total = 0.0
    for _pick, actual, probs in pairs:
        for outcome, p in probs.items():
            truth = 1.0 if outcome == actual else 0.0
            brier_total += (p - truth) ** 2
    return {
        "n_pairs": n,
        "accuracy": round(accuracy, 4),
        "brier": round(brier_total / n, 4),
    }


def project_polymarket_ev(top1_acc: Optional[float], market_yes_typical: float, spread_pp: int, fee_pct: float = TAKER_FEE_PCT) -> Optional[float]:
    """Rough EV estimate: if model picks a specific outcome at top1_acc accuracy
    and the market quote for the FAVOURED outcome is `market_yes_typical`, then
    EV per unit stake is approximately:

        EV = (model_correct_prob / market_yes_typical * (1 - fee_pct)) - 1

    Subject to the assumption that we only bet WHEN the model edge is positive.
    """
    if top1_acc is None:
        return None
    payout = (1.0 - fee_pct) / max(market_yes_typical, 0.01)
    ev = top1_acc * payout - 1.0
    return round(ev, 4)


def render_report(runs: List[Dict[str, Any]], output_path: Path) -> None:
    """Compose the markdown report and write it out."""
    lines: List[str] = []
    lines.append("# Wave 5 Polymarket multi-market validation\n")
    lines.append("Auto-generated by `scripts/analyse_wave_5_holdouts.py`.\n\n")

    # Per-tournament moneyline + sub-market table
    lines.append("## Headline summary (V1 vs V2 across tournaments)\n\n")
    lines.append("| Run | Sim | Tournament | n_decisive | Acc | Brier | LogLoss | MAE_runs | MAE_margin | TopBatter top1 / top3 | MostSixes acc / brier |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|\n")

    runs_sorted = sorted(runs, key=lambda r: r["label"])
    for run in runs_sorted:
        label = run["label"]
        summary = run["summary"]
        metrics = summary.get("metrics", {})
        sim_version = "v2" if "v2" in label else "v1"
        tournament = label.replace(f"w5_{sim_version}_", "")
        n_dec = metrics.get("n_decisive", 0)
        if n_dec == 0:
            continue
        acc = _format_pct(metrics.get("accuracy_top_pick"))
        brier = _format_score(metrics.get("brier_score"))
        log_loss = _format_score(metrics.get("log_loss"))
        mae_runs = _format_score(metrics.get("mae_score_runs"), digits=2) if metrics.get("mae_score_runs") is not None else "--"
        mae_margin = _format_score(metrics.get("mae_margin_runs"), digits=2) if metrics.get("mae_margin_runs") is not None else "--"
        tb1 = _format_pct(metrics.get("top_batter_accuracy_top_1"))
        tb3 = _format_pct(metrics.get("top_batter_accuracy_top_3"))
        ms_acc = _format_pct(metrics.get("most_sixes_accuracy"))
        ms_brier = _format_score(metrics.get("most_sixes_brier"))
        lines.append(
            f"| {label} | {sim_version} | {tournament} | {n_dec} | {acc} | {brier} | {log_loss} | "
            f"{mae_runs} | {mae_margin} | {tb1} / {tb3} | {ms_acc} / {ms_brier} |\n"
        )
    lines.append("\n")

    # Andrew Kuo reference call-out
    lines.append("### Reference: Andrew Kuo (T20 ball-by-ball model, 2021)\n\n")
    lines.append(f"- Top-batter top-1 accuracy: **{ANDREW_KUO_TOP_BATTER_ACC*100:.0f}%** (our V2 must beat this to be useful for that market).\n")
    lines.append(f"- Top-batter top-3 accuracy: **{ANDREW_KUO_TOP_BATTER_ACC_TOP3*100:.0f}%** (rough; if we beat this we're meaningfully informed).\n\n")

    # Per-tournament prediction compression
    lines.append("## Per-tournament prediction compression bands\n\n")
    lines.append("How tightly the model's probabilities cluster (0.5 = 'always coin-flip', wider = 'opinionated').\n\n")
    lines.append("| Run | n | mean(team1_win_prob) | std(team1_win_prob) | min | max |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for run in runs_sorted:
        label = run["label"]
        rows = load_csv(run["csv_path"])
        ps = [_safe_float(r.get("sim_team1_win_prob")) for r in rows]
        ps = [p for p in ps if p is not None]
        if not ps:
            continue
        n = len(ps)
        mean = sum(ps) / n
        std = (sum((p - mean) ** 2 for p in ps) / n) ** 0.5
        lines.append(
            f"| {label} | {n} | {mean:.3f} | {std:.3f} | {min(ps):.3f} | {max(ps):.3f} |\n"
        )
    lines.append("\n")

    # Polymarket exploit potential
    lines.append("## Polymarket exploit potential (per-market projected EV)\n\n")
    lines.append(
        "Rough projection: if the model has the per-tournament accuracy "
        "shown above and the market quotes its YES side at the typical "
        "fraction noted below (with 2% Polymarket taker fee), expected "
        "return per $1 staked is shown.\n\n"
    )
    lines.append("Market spread assumptions (manually sampled from live Polymarket cricket markets, Apr 2026):\n\n")
    for m, info in POLYMARKET_SPREADS.items():
        lines.append(f"- {m}: typical YES price = {info['yes_typical']:.2f}, spread ~ {info['spread_pp']} pp\n")
    lines.append("\n")

    lines.append("| Run | Top batter EV | Most sixes EV | Notes |\n")
    lines.append("|---|---|---|---|\n")
    for run in runs_sorted:
        label = run["label"]
        metrics = run["summary"].get("metrics", {})
        tb_ev = project_polymarket_ev(
            metrics.get("top_batter_accuracy_top_1"),
            POLYMARKET_SPREADS["top_batter"]["yes_typical"],
            POLYMARKET_SPREADS["top_batter"]["spread_pp"],
        )
        # most-sixes accuracy is across 3 outcomes, so the "favourite YES"
        # picked-it-correct prob is the same as accuracy.
        ms_ev = project_polymarket_ev(
            metrics.get("most_sixes_accuracy"),
            POLYMARKET_SPREADS["most_sixes"]["yes_typical"],
            POLYMARKET_SPREADS["most_sixes"]["spread_pp"],
        )
        notes = ""
        if metrics.get("n_top_batter_pairs", 0) < 50:
            notes = "n<50, not enough sample"
        lines.append(f"| {label} | {tb_ev if tb_ev is not None else '--'} | {ms_ev if ms_ev is not None else '--'} | {notes} |\n")
    lines.append("\n")

    # Verdict / recommendation
    lines.append("## Verdict\n\n")
    v2_runs = [r for r in runs_sorted if "v2" in r["label"]]
    if v2_runs:
        # Average V2 top-1 accuracy
        accs = [r["summary"].get("metrics", {}).get("top_batter_accuracy_top_1") for r in v2_runs]
        accs = [a for a in accs if a is not None]
        avg_acc = sum(accs) / len(accs) if accs else None
        lines.append(f"- V2 mean top-batter top-1 accuracy across {len(v2_runs)} runs: **{(avg_acc * 100):.1f}%** (Andrew Kuo bar: {ANDREW_KUO_TOP_BATTER_ACC*100:.0f}%).\n")
        # Average V2 most-sixes accuracy
        ms_accs = [r["summary"].get("metrics", {}).get("most_sixes_accuracy") for r in v2_runs]
        ms_accs = [a for a in ms_accs if a is not None]
        avg_ms = sum(ms_accs) / len(ms_accs) if ms_accs else None
        if avg_ms is not None:
            lines.append(f"- V2 mean most-sixes accuracy across {len(v2_runs)} runs: **{(avg_ms * 100):.1f}%** (random baseline: 33%).\n")

    lines.append(
        "\nUse this report as input to **Wave 5 Phase 5** (historical EV "
        "backtest using `/prices-history`). Markets where V2 top-1 / "
        "most-sixes accuracy is materially above random get included in "
        "Phase 6's `auto_enabled_markets` list; markets that don't get "
        "limited to MANUAL-only mode.\n"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(lines))
    logger.info(f"Wrote {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest-dir", default="data/backtest")
    parser.add_argument("--label-prefix", default="w5_")
    parser.add_argument("--output", default="data/diagnostics/wave_5_polymarket_summary.md")
    args = parser.parse_args()

    backtest_dir = Path(args.backtest_dir)
    runs = discover_runs(backtest_dir, args.label_prefix)
    if not runs:
        logger.error(f"No matching runs in {backtest_dir} with prefix {args.label_prefix}")
        return 1
    logger.info(f"Found {len(runs)} runs")
    render_report(runs, Path(args.output))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
