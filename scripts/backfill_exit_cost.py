#!/usr/bin/env python3
"""One-shot backfill for bet_ledger.exit_cost_usdc (Wave 6).

Context: partial exits (rebalance / cashout SELLs) decrement a BUY row's open
stake (``fill_size_usdc``). The TWAP fill reconcile in
``clob_fills.finalize_plan_from_chunks`` now computes the open-stake target as
``gross_filled - exit_cost_usdc`` so it never re-inflates a partially-exited
row back to the gross on-chain fill. For that to be a no-op on EXISTING rows
(which predate the column and were trued-down by
``reconcile_open_positions_to_chain.py``), we seed ``exit_cost_usdc`` with the
amount already removed:

    exit_cost_usdc = max(0, gross_filled_from_chunks - current_fill_size_usdc)

Only rows whose chunk-derived gross exceeds the current open stake by more than
the top-up threshold are seeded; everything else is left at 0. Idempotent: only
touches rows where exit_cost_usdc is currently 0/NULL.

Usage:
    venv311/bin/python scripts/backfill_exit_cost.py            # dry-run
    venv311/bin/python scripts/backfill_exit_cost.py --apply    # write
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATABASE_PATH

# Match clob_fills.PARTIAL_FILL_TOPUP_THRESHOLD: ignore sub-2c rounding noise.
TOPUP_THRESHOLD = 0.02


def _f(v: Any) -> float:
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="Write exit_cost_usdc (default: dry-run report only)")
    args = parser.parse_args()

    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT b.bet_id, b.side_label, b.strategy_label,
               b.fill_size_usdc AS cur_fill,
               COALESCE(b.exit_cost_usdc, 0.0) AS exit_cost,
               (SELECT COALESCE(SUM(c.fill_size_usdc), 0.0)
                  FROM order_chunks c
                  JOIN order_plans p ON p.plan_id = c.plan_id
                 WHERE p.bet_ledger_id = b.bet_id
                   AND c.fill_size_usdc IS NOT NULL) AS gross_chunks
        FROM bet_ledger b
        WHERE COALESCE(b.bet_kind, 'real') = 'real'
          AND b.side = 'BUY'
          AND b.status = 'filled'
          AND b.settled_at IS NULL
          AND b.fill_size_usdc IS NOT NULL AND b.fill_size_usdc > 0
        ORDER BY b.bet_id
        """
    )
    rows = cur.fetchall()

    seeds = []
    for r in rows:
        if _f(r["exit_cost"]) > 1e-9:
            continue  # already seeded; idempotent
        gap = round(_f(r["gross_chunks"]) - _f(r["cur_fill"]), 6)
        if gap > TOPUP_THRESHOLD:
            seeds.append((r, gap))

    print()
    print("=" * 92)
    print(f"EXIT-COST BACKFILL {'[APPLY]' if args.apply else '[DRY-RUN]'}")
    print("=" * 92)
    print(f"  {'bet':>5} {'side':<16} {'strategy':<16} {'cur_fill':>9} {'gross':>9} {'seed_exit':>10}")
    print(f"  {'-'*5} {'-'*16} {'-'*16} {'-'*9} {'-'*9} {'-'*10}")
    for r, gap in seeds:
        print(f"  {r['bet_id']:>5} {str(r['side_label'])[:16]:<16} "
              f"{str(r['strategy_label'])[:16]:<16} {_f(r['cur_fill']):>9.2f} "
              f"{_f(r['gross_chunks']):>9.2f} {gap:>10.4f}")
    print()
    print(f"  Rows examined: {len(rows)}    Rows to seed: {len(seeds)}")
    print()

    if not args.apply:
        print("  DRY-RUN — no changes written. Re-run with --apply to commit.")
        conn.close()
        return 0

    for r, gap in seeds:
        cur.execute(
            "UPDATE bet_ledger SET exit_cost_usdc = ? "
            "WHERE bet_id = ? AND COALESCE(exit_cost_usdc, 0.0) <= 1e-9",
            (gap, r["bet_id"]),
        )
    conn.commit()
    conn.close()
    print(f"  APPLIED exit_cost_usdc seed to {len(seeds)} rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
