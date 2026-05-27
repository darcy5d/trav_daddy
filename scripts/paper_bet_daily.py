#!/usr/bin/env python3
"""Wave 5.7: daily paper-bet runner.

One-shot script that:
    1. Scans Polymarket for upcoming fixtures, places paper bets
    2. Reconciles any settled paper bets and updates P&L + bankroll
    3. Prints a summary table of per-strategy bankroll, win rate, and P&L
    4. Appends a JSON report to data/paper_trading/daily_reports/

Designed to be invoked by cron - hourly recommended for tighter entry-time
granularity, daily minimum to keep the bankroll ticking.

Recommended cron entry (hourly):

    0 * * * * cd /path/to/indias_dad && \
        venv311/bin/python scripts/paper_bet_daily.py >> logs/paper_daily.log 2>&1

Usage:
    venv311/bin/python scripts/paper_bet_daily.py [--hours-ahead 96]
                                                  [--strategies ALL|name1,name2]
                                                  [--no-scan]
                                                  [--no-reconcile]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_connection, get_db_connection
from src.integrations.polymarket.paper_strategies import (
    STRATEGIES,
    get_strategy_bankroll,
    get_enabled_strategies,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


REPORT_DIR = Path(__file__).resolve().parent.parent / "data" / "paper_trading" / "daily_reports"


def _strategy_summary(conn) -> Dict[str, Any]:
    """Build a per-strategy P&L + bankroll table."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT strategy_label,
               COUNT(*) AS n_bets,
               SUM(CASE WHEN status = 'settled' THEN 1 ELSE 0 END) AS n_settled,
               SUM(CASE WHEN status = 'settled' AND settle_outcome = 1 THEN 1 ELSE 0 END) AS n_won,
               SUM(CASE WHEN status = 'settled' THEN pnl_realised_usdc ELSE 0 END) AS realised_pnl,
               SUM(CASE WHEN status NOT IN ('settled', 'cancelled', 'errored') THEN size_usdc ELSE 0 END) AS open_size,
               MIN(proposed_at) AS first_bet_at,
               MAX(proposed_at) AS last_bet_at
        FROM bet_ledger
        WHERE bet_kind = 'paper'
        GROUP BY strategy_label
        ORDER BY strategy_label
        """
    )
    rows = [dict(r) for r in cur.fetchall()]
    by_strat: Dict[str, Any] = {}
    for r in rows:
        name = r["strategy_label"] or "(none)"
        starting = next((s.starting_bankroll_usdc for s in STRATEGIES if s.name == name), 1000.0)
        bankroll_now = starting + (r["realised_pnl"] or 0.0)
        win_rate = (r["n_won"] / r["n_settled"]) if r["n_settled"] else None
        roi = ((r["realised_pnl"] or 0.0) / starting) if starting > 0 else None
        by_strat[name] = {
            "n_bets":         int(r["n_bets"] or 0),
            "n_settled":      int(r["n_settled"] or 0),
            "n_won":          int(r["n_won"] or 0),
            "win_rate":       win_rate,
            "starting":       starting,
            "realised_pnl":   round(r["realised_pnl"] or 0.0, 2),
            "open_size":      round(r["open_size"] or 0.0, 2),
            "bankroll":       round(bankroll_now, 2),
            "roi_pct":        round(roi * 100, 2) if roi is not None else None,
            "first_bet_at":   r["first_bet_at"],
            "last_bet_at":    r["last_bet_at"],
        }
    # Add zero rows for enabled strategies that haven't placed any bets
    for s in get_enabled_strategies():
        if s.name not in by_strat:
            by_strat[s.name] = {
                "n_bets": 0, "n_settled": 0, "n_won": 0, "win_rate": None,
                "starting": s.starting_bankroll_usdc, "realised_pnl": 0.0,
                "open_size": 0.0, "bankroll": s.starting_bankroll_usdc,
                "roi_pct": 0.0, "first_bet_at": None, "last_bet_at": None,
            }
    return by_strat


def _print_summary_table(by_strat: Dict[str, Any]) -> None:
    print()
    print("=" * 100)
    print("PAPER TRADING - PER-STRATEGY SUMMARY")
    print("=" * 100)
    fmt = "  {:18s}  {:>5s}  {:>5s}  {:>5s}  {:>8s}  {:>10s}  {:>10s}  {:>9s}  {:>9s}"
    print(fmt.format("strategy", "bets", "settl", "won", "win%", "p&l_usdc", "bankroll", "roi%", "open_$"))
    print("  " + "-" * 96)
    for name, s in sorted(by_strat.items()):
        win_pct = f"{s['win_rate']*100:.1f}" if s["win_rate"] is not None else "n/a"
        roi_str = f"{s['roi_pct']:+.2f}" if s["roi_pct"] is not None else "n/a"
        print(fmt.format(
            name[:18],
            str(s["n_bets"]),
            str(s["n_settled"]),
            str(s["n_won"]),
            win_pct,
            f"{s['realised_pnl']:+.2f}",
            f"{s['bankroll']:.2f}",
            roi_str,
            f"{s['open_size']:.2f}",
        ))


def _save_daily_report(report: Dict[str, Any]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fname = REPORT_DIR / f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')}.json"
    with fname.open("w") as f:
        json.dump(report, f, indent=2, default=str)
    return fname


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily paper-bet runner")
    parser.add_argument("--hours-ahead", type=float, default=96.0)
    parser.add_argument("--strategies", default="ALL")
    parser.add_argument("--no-scan", action="store_true", help="Skip the scan step")
    parser.add_argument("--no-reconcile", action="store_true", help="Skip the reconcile step")
    parser.add_argument("--dry-run", action="store_true", help="Run sims but don't write")
    args = parser.parse_args()

    strategy_filter = None
    if args.strategies.upper() != "ALL":
        strategy_filter = [s.strip() for s in args.strategies.split(",") if s.strip()]

    started = datetime.now(timezone.utc)
    report: Dict[str, Any] = {
        "started_at": started.isoformat(),
        "hours_ahead": args.hours_ahead,
        "strategy_filter": strategy_filter,
        "scan_summary": None,
        "reconcile_summary": None,
        "strategies": {},
        "errors": [],
    }

    if not args.no_scan:
        try:
            from scripts.paper_bet_scan import scan_and_place_paper_bets
            scan_summary = scan_and_place_paper_bets(
                hours_ahead=args.hours_ahead,
                strategy_filter=strategy_filter,
                dry_run=args.dry_run,
            )
            report["scan_summary"] = scan_summary
        except Exception as exc:
            logger.error(f"Scan step failed: {exc}")
            report["errors"].append({"step": "scan", "error": str(exc), "trace": traceback.format_exc()})
    else:
        logger.info("Scan step skipped (--no-scan)")

    if not args.no_reconcile and not args.dry_run:
        # Step A: backfill any TWAP maker fills whose chunks/bets are not yet
        # reflected. Cheap (single CLOB paginated read) and closes the loop
        # for fills that landed on repriced/cancelled order IDs.
        try:
            from scripts.backfill_twap_maker_fills import backfill as twap_backfill
            twap_summary = twap_backfill(run_reconcile=False)
            report["twap_backfill_summary"] = twap_summary
            logger.info(
                f"TWAP backfill: chunks_updated={twap_summary.get('chunks_updated')} "
                f"plans_finalized={len(twap_summary.get('plans_finalized') or [])}"
            )
        except Exception as exc:
            logger.warning(f"TWAP backfill step failed (non-fatal): {exc}")
            report["errors"].append({"step": "twap_backfill", "error": str(exc), "trace": traceback.format_exc()})

        # Step B: settlement + stale-proposed hygiene
        try:
            from src.integrations.polymarket.reconcile import reconcile_pending_bets
            reconcile_summary = reconcile_pending_bets()
            # Cast errors to str so they're JSON-serializable
            reconcile_summary["errors"] = [
                {"bet_id": b, "msg": str(m)} for b, m in reconcile_summary.get("errors", [])
            ]
            report["reconcile_summary"] = reconcile_summary
        except Exception as exc:
            logger.error(f"Reconcile step failed: {exc}")
            report["errors"].append({"step": "reconcile", "error": str(exc), "trace": traceback.format_exc()})
    else:
        logger.info("Reconcile step skipped")

    # Strategy summary table
    with get_db_connection() as conn:
        by_strat = _strategy_summary(conn)
    report["strategies"] = by_strat
    _print_summary_table(by_strat)

    # Compute totals
    total_realised = sum(s["realised_pnl"] for s in by_strat.values())
    total_starting = sum(s["starting"] for s in by_strat.values())
    total_bankroll = sum(s["bankroll"] for s in by_strat.values())
    total_open = sum(s["open_size"] for s in by_strat.values())
    print("  " + "-" * 96)
    print(f"  TOTAL                                          ${total_realised:+,.2f}  "
          f"${total_bankroll:,.2f}  {(total_realised/total_starting*100) if total_starting else 0:+.2f}%  ${total_open:,.2f}")
    print()
    report["totals"] = {
        "starting": total_starting,
        "realised_pnl": round(total_realised, 2),
        "bankroll": round(total_bankroll, 2),
        "open_size": round(total_open, 2),
        "roi_pct": round((total_realised / total_starting * 100) if total_starting else 0, 2),
    }

    finished = datetime.now(timezone.utc)
    report["finished_at"] = finished.isoformat()
    report["elapsed_seconds"] = (finished - started).total_seconds()

    # Persist daily report
    if not args.dry_run:
        report_path = _save_daily_report(report)
        logger.info(f"Saved report -> {report_path}")

    return 0 if not report["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
