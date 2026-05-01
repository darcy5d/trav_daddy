#!/usr/bin/env python3
"""Wave 5.7: settle paper bets whose Polymarket markets have closed.

Calls the shared `reconcile_pending_bets()` which handles BOTH real and
paper bets uniformly (paper bets are stored with status='filled' so they
flow through the same path).

For paper bets specifically we also populate bankroll_after_settle so the
bankroll-over-time chart works without recomputing.

Usage:
    venv311/bin/python scripts/paper_bet_reconcile.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.integrations.polymarket.reconcile import reconcile_pending_bets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconcile paper (and real) bet ledger entries")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    summary = reconcile_pending_bets()

    print()
    print("=" * 70)
    print("RECONCILE SUMMARY")
    print("=" * 70)
    print(f"  Bets checked:        {summary['n_checked']}")
    print(f"  Settled this run:    {summary['n_settled']}")
    print(f"  Still pending:       {summary['n_still_pending']}")
    if summary.get("errors"):
        print(f"  Errors:              {len(summary['errors'])}")
        for bet_id, msg in summary["errors"]:
            print(f"    bet_id={bet_id}: {msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
