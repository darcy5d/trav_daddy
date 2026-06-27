#!/usr/bin/env python3
"""Cancel and finalize the in-flight TWAP backlog.

Operational cleanup for when the `order_plans` table accumulates plans stuck in
'pending'/'executing' that can no longer execute — e.g. the wallet ran out of
collateral, or the fixtures were later excluded from live (county T20 Blast).
The post-toss daemon otherwise retries these every poll, spamming
"not enough balance / allowance" errors and hammering the CLOB.

Unlike reconcile._reconcile_stale_twap_plans (which only touches plans whose
kickoff is >2h past), this finalizes ALL non-terminal plans regardless of
kickoff. It is conservative about money:

  1. Sync any real MAKER fills from the CLOB first (never drop a real fill).
  2. Cancel every resting order on the book (cancel_all_orders).
  3. Mark still-resting chunks cancelled, then finalize each plan:
     filled>0 -> 'completed', else -> 'cancelled' (bet_ledger updated to match).

Usage:
    venv311/bin/python scripts/cleanup_twap_backlog.py [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_connection

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List the plans that would be cancelled without touching the CLOB or DB.",
    )
    args = parser.parse_args()

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT plan_id, bet_ledger_id, fixture_key, strategy_label,
               chunks_placed, chunks_filled, filled_size_usdc, status, kickoff_at
        FROM order_plans
        WHERE status IN ('pending', 'executing')
        ORDER BY kickoff_at
        """
    )
    plans = cur.fetchall()

    if not plans:
        logger.info("No pending/executing TWAP plans — nothing to clean up.")
        return 0

    logger.info("Found %d non-terminal TWAP plan(s):", len(plans))
    for p in plans:
        logger.info(
            "  plan #%s  %s  %s  placed=%s filled=%s  filled_usdc=%.2f  kickoff=%s",
            p["plan_id"], p["fixture_key"], p["strategy_label"],
            p["chunks_placed"], p["chunks_filled"],
            (p["filled_size_usdc"] or 0.0), p["kickoff_at"],
        )

    if args.dry_run:
        logger.info("[DRY-RUN] No CLOB or DB changes made.")
        return 0

    from src.integrations.polymarket import PolymarketClient
    from src.integrations.polymarket.clob_fills import (
        fetch_all_clob_trades,
        finalize_plan_from_chunks,
        index_fills_by_order_id,
        sync_placed_chunk_fills,
    )

    pm = PolymarketClient()

    # 1. Cancel every resting order on the book so nothing fills mid-cleanup.
    try:
        resp = pm.cancel_all_orders()
        logger.info("cancel_all_orders -> %s", resp)
    except Exception as exc:
        logger.warning("cancel_all_orders failed (continuing): %s", exc)

    # 2. Pull trade history once so we can capture any real maker fills.
    try:
        trades = fetch_all_clob_trades(pm)
        fills = index_fills_by_order_id(trades)
    except Exception as exc:
        logger.warning("CLOB trade fetch failed; finalizing from local chunks only: %s", exc)
        fills = {}

    now_iso = datetime.now(timezone.utc).isoformat()
    n_completed = 0
    n_cancelled = 0

    for p in plans:
        plan_id = p["plan_id"]
        try:
            if fills:
                sync_placed_chunk_fills(conn, cur, fills, plan_id=plan_id)
            # Resting chunks were cancelled on-chain above; reflect that locally
            # so finalize can transition the plan to a terminal state.
            cur.execute(
                "UPDATE order_chunks SET status = 'cancelled' "
                "WHERE plan_id = ? AND status IN ('pending', 'placed')",
                (plan_id,),
            )
            result = finalize_plan_from_chunks(conn, cur, plan_id)
            final_status = result.get("status") if isinstance(result, dict) else None
            if final_status == "completed":
                n_completed += 1
            else:
                n_cancelled += 1
            logger.info("plan #%s finalized -> %s", plan_id, final_status or result)
        except Exception as exc:
            logger.error("plan #%s finalize failed: %s", plan_id, exc)

    conn.commit()
    conn.close()

    logger.info(
        "Done. %d plan(s) completed (had real fills), %d plan(s) cancelled.",
        n_completed, n_cancelled,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
