#!/usr/bin/env python3
"""Wave 5.10: In-game cashout scanner.

Polls all open (filled) positions and triggers a SELL order whenever
the return ratio (current_price / fill_price) meets the tiered threshold
for that entry price bucket:

    Heavy underdog  5–20¢  → cash out at 1.30x
    Underdog       20–35¢  → cash out at 1.20x
    Slight underdog35–50¢  → cash out at 1.25x
    Coin flip+     50–95¢  → hold to settlement (no cashout)

Run every 2–5 minutes while matches are live:

    # Dry-run — show what would cashout without placing any orders
    venv311/bin/python scripts/inplay_cashout_scan.py --dry-run

    # Live — executes real SELL orders for real bets, simulates for paper
    venv311/bin/python scripts/inplay_cashout_scan.py

Cron example (every 3 minutes during a match window):
    */3 * * * * cd /path/to/indias_dad && venv311/bin/python scripts/inplay_cashout_scan.py >> logs/cashout_scan.log 2>&1
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_connection, init_cashout_columns
from src.integrations.polymarket import PolymarketClient
from src.integrations.polymarket.cashout import scan_for_cashouts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan filled bets for cashout eligibility and execute sells"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simulate cashouts without placing real SELL orders",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Debug-level logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Ensure cashout columns exist (idempotent, fast)
    init_cashout_columns()

    conn = get_connection()
    poly_client = PolymarketClient()

    summary = scan_for_cashouts(
        conn=conn,
        poly_client=poly_client,
        dry_run=args.dry_run,
    )

    conn.close()

    mode_tag = "[DRY-RUN] " if args.dry_run else ""

    print()
    print("=" * 70)
    print(f"CASHOUT SCAN {mode_tag}— {run_ts}")
    print("=" * 70)
    print(f"  Filled positions checked: {summary['n_checked']}")
    print(f"  Trigger(s) met:           {summary['n_triggered']}")
    print(f"  Sells executed:           {summary['n_executed']}")
    print(f"    of which stop-losses:   {summary.get('n_stops', 0)}")

    if summary["cashouts"]:
        print()
        print(f"  {'bet_id':>8}  {'reason':>9}  {'return':>8}  {'PnL':>10}  {'simulated':>10}")
        print(f"  {'─'*8}  {'─'*9}  {'─'*8}  {'─'*10}  {'─'*10}")
        for c in summary["cashouts"]:
            if c.get("success"):
                sim_tag = "YES" if c.get("is_simulated") else "NO (live)"
                reason = "STOP-LOSS" if c.get("reason") == "stop" else "profit"
                print(
                    f"  {c['bet_id']:>8}  "
                    f"{reason:>9}  "
                    f"{c['return_ratio']:>7.2f}x  "
                    f"${c['cashout_pnl']:>+9.2f}  "
                    f"{sim_tag:>10}"
                )

    if summary["errors"]:
        print()
        print(f"  Errors ({len(summary['errors'])}):")
        for bet_id, msg in summary["errors"]:
            print(f"    bet_id={bet_id}: {msg}")

    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
