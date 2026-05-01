#!/usr/bin/env python3
"""
Match-level backtest harness CLI (NEXT_OVERHAUL item 16).

For a chronological holdout of completed matches, runs the simulator with
as-of-date team and player ELOs and reports calibration / Brier / log-loss /
score MAE. Persists per-match rows to CSV + an overall JSON summary so
subsequent Wave 4 changes (margin-of-victory, recency decay, etc.) get an
attributable before/after.

Run:
    # Default: last 6 months of T20 men
    python scripts/backtest_simulator.py

    # Just IPL 2025-26
    python scripts/backtest_simulator.py --tournament-pattern '%Indian Premier League%' \
        --since-date 2025-09-01 --limit 200

    # Smoke test on 5 matches with 200 sims (~1 min)
    python scripts/backtest_simulator.py --limit 5 --n-sims 200
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_connection, init_franchise_tables  # noqa: E402
from src.models.backtest import (  # noqa: E402
    BacktestRow,
    compute_metrics,
    load_holdout_matches,
    rows_to_dicts,
    simulate_match,
    stratified_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Match-level backtest harness for the Monte Carlo simulator")
    parser.add_argument("--format", default="T20", choices=["T20", "ODI"], help="Match format (default: T20)")
    parser.add_argument("--gender", default="male", choices=["male", "female"], help="Gender (default: male)")
    parser.add_argument(
        "--since-date",
        default=None,
        help="ISO date YYYY-MM-DD; default: 6 months before today",
    )
    parser.add_argument("--until-date", default=None, help="ISO date YYYY-MM-DD; default: today")
    parser.add_argument(
        "--tournament-pattern",
        default=None,
        help="SQL LIKE pattern matched against matches.event_name (e.g. '%%Indian Premier League%%')",
    )
    parser.add_argument("--limit", type=int, default=200, help="Max matches to backtest (default: 200)")
    parser.add_argument("--n-sims", type=int, default=1000, help="Monte Carlo iterations per match (default: 1000)")
    parser.add_argument("--simulator", default="nn", choices=["nn"], help="Simulator backend (default: nn)")
    parser.add_argument(
        "--model-version",
        default="v1",
        choices=["v1", "v2", "v3"],
        help=(
            "Which model architecture to use. v1 = legacy ball_prediction_model_*.keras. "
            "v2 = cricket_model_v2 (multi-task joint, 9-class extras, per-over head). "
            "v3 = cricket_model_v3 (V2 + toss + lineup-stability features). "
            "Default v1 for backwards compatibility."
        ),
    )
    parser.add_argument(
        "--no-toss",
        action="store_true",
        help="Disable toss simulation; team1 always bats first.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/backtest",
        help="Where to write the per-match CSV and JSON summary.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional human-readable label for this run (defaults to a timestamp).",
    )
    args = parser.parse_args()

    # Default to a 6-month window ending today.
    since = args.since_date or (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
    until = args.until_date

    init_franchise_tables()

    with get_connection() as conn:
        matches = load_holdout_matches(
            conn,
            formats=(args.format,),
            genders=(args.gender,),
            since_date=since,
            until_date=until,
            tournament_pattern=args.tournament_pattern,
            limit=args.limit,
        )

    if not matches:
        logger.error("No matches found for the holdout filters; aborting.")
        return 1

    logger.info(
        f"Holdout: {len(matches)} {args.format} {args.gender} matches "
        f"({matches[-1].date} → {matches[0].date}); n_sims={args.n_sims}"
    )

    # Build the simulator once and reuse it across matches; the per-match
    # ELO override context manager keeps it stateless between invocations.
    if args.simulator == "nn":
        if args.model_version == "v1":
            from src.models.vectorized_nn_sim import VectorizedNNSimulator
            simulator = VectorizedNNSimulator(gender=args.gender, format_type=args.format)
        elif args.model_version == "v2":
            from src.models.vectorized_nn_sim_v2 import V2Simulator, V2SimulatorConfig
            simulator = V2Simulator(V2SimulatorConfig(format_type=args.format, gender=args.gender))
        else:  # v3
            from src.models.vectorized_nn_sim_v3 import V3Simulator, V3SimulatorConfig
            simulator = V3Simulator(V3SimulatorConfig(format_type=args.format, gender=args.gender))
    else:
        raise SystemExit(f"Unsupported simulator: {args.simulator}")

    rows: list[BacktestRow] = []
    skipped_incomplete = 0
    start = time.time()

    with get_connection() as conn:
        for i, m in enumerate(matches, 1):
            try:
                row = simulate_match(
                    conn,
                    simulator,
                    m,
                    n_sims=args.n_sims,
                    use_toss=not args.no_toss,
                )
            except Exception as exc:
                logger.warning(f"match {m.match_id} ({m.event_name}): {exc.__class__.__name__}: {exc}")
                continue
            if row is None:
                skipped_incomplete += 1
                continue
            rows.append(row)
            if i % 25 == 0 or i == len(matches):
                elapsed = time.time() - start
                rate = i / elapsed if elapsed else 0.0
                logger.info(f"  [{i}/{len(matches)}] {rate:.1f} matches/s; {len(rows)} rows captured")

    elapsed_total = time.time() - start
    logger.info(f"Backtest complete: {len(rows)} rows in {elapsed_total:.1f}s ({skipped_incomplete} skipped: incomplete lineup)")

    if not rows:
        logger.error("No rows produced; aborting metric calculation.")
        return 1

    metrics = compute_metrics(rows)
    by_event = stratified_metrics(rows, by="event_name")
    by_format = stratified_metrics(rows, by="match_type")

    # Output paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"backtest_{label}.csv"
    summary_path = output_dir / f"backtest_{label}_summary.json"

    # Per-match CSV
    fieldnames = list(rows_to_dicts(rows[:1])[0].keys())
    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for d in rows_to_dicts(rows):
            writer.writerow(d)

    # Summary JSON
    summary = {
        "label": label,
        "args": vars(args),
        "since_date": since,
        "until_date": until,
        "n_matches_in_holdout": len(matches),
        "n_rows": len(rows),
        "n_skipped_incomplete_lineup": skipped_incomplete,
        "elapsed_seconds": round(elapsed_total, 1),
        "metrics": metrics,
        "by_event": by_event,
        "by_format": by_format,
        "documented_caveats": [
            "Per-batter / per-bowler outcome distributions still aggregate across the player's "
            "full career including the holdout matches (~one match's contribution per player). "
            "Acceptable for v1; revisit in v2 with per-match distribution recompute.",
            "Toss is simulated independently per Monte Carlo iteration rather than fixed to the "
            "actual toss outcome (matches the live Bulk Predict path).",
            "Bowling lineup uses the 5 actual highest-overs bowlers; impact-sub selection is not modelled.",
        ],
    }
    with summary_path.open("w") as fp:
        json.dump(summary, fp, indent=2, default=str)

    # Console summary
    print()
    print("=" * 78)
    print(f"Backtest summary  ({label})")
    print("=" * 78)
    print(f"  Holdout window     : {since} -> {until or 'today'}")
    print(f"  Filter             : {args.format} {args.gender} {args.tournament_pattern or '(any tournament)'}")
    print(f"  Rows / decisive    : {metrics['n_matches']} / {metrics['n_decisive']} (no-result: {metrics['n_no_result']})")
    print(f"  Top-pick accuracy  : {metrics['accuracy_top_pick']:.3f}")
    print(f"  Brier score        : {metrics['brier_score']:.4f}   (lower is better; 0.25 = always 50/50; 0 = perfect)")
    print(f"  Log loss           : {metrics['log_loss']:.4f}   (lower is better; 0.693 = always 50/50)")
    print(f"  MAE total runs     : {metrics['mae_score_runs']}    (per innings)")
    print(f"  MAE margin runs    : {metrics['mae_margin_runs']}")
    print()
    print("  Calibration deciles:")
    print("    bucket          n    mean pred    actual win rate")
    for c in metrics["calibration_deciles"]:
        if c["n"] == 0:
            print(f"    [{c['lo']:.1f}, {c['hi']:.1f})    0    --           --")
            continue
        print(
            f"    [{c['lo']:.1f}, {c['hi']:.1f})  {c['n']:>4}    {c['mean_pred']:.3f}        {c['actual_win_rate']:.3f}"
        )
    print()
    if by_event:
        print("  Per-tournament (>=5 matches):")
        for ev, mm in sorted(by_event.items(), key=lambda kv: -kv[1].get("n_matches", 0)):
            print(
                f"    {ev[:60]:<60} n={mm['n_matches']:>4}  "
                f"acc={mm['accuracy_top_pick']:.3f}  brier={mm['brier_score']:.4f}  log_loss={mm['log_loss']:.4f}"
            )
    print()
    print(f"  Per-match CSV   : {csv_path}")
    print(f"  Summary JSON    : {summary_path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
