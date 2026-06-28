#!/usr/bin/env python3
"""Cancel resting TWAP orders for specific league(s), leaving other leagues alone.

Targeted version of cleanup_twap_backlog.py. Where that script finalizes ALL
non-terminal plans (and uses cancel_all_orders), this only touches plans whose
fixture_key starts with one of the given league prefixes. Used to pull a league
out of live execution without disturbing legitimate working orders in other
leagues (e.g. cancel county Blast while MLC keeps quoting).

Money-safe ordering:
  1. Sync any real MAKER fills from the CLOB first (never drop a real fill).
  2. Cancel each resting chunk order on the book (per-order, not cancel_all).
  3. Mark still-resting chunks cancelled, then finalize each plan:
     filled>0 -> 'completed', else -> 'cancelled' (bet_ledger updated to match).

Note: a prefix like "crict20blast" also matches "crict20blastw" (startswith),
which is intended — it covers both the men's and women's Blast.

Usage:
    venv311/bin/python scripts/cancel_league_open_orders.py --dry-run
    venv311/bin/python scripts/cancel_league_open_orders.py --league-prefix crict20blast
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_connection

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--league-prefix", action="append", default=None,
        help="Fixture-key prefix to cancel (repeatable). Default: crict20blast "
             "(covers crict20blast + crict20blastw).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List the plans/orders that would be cancelled without touching the CLOB or DB.",
    )
    args = parser.parse_args()
    prefixes = tuple(args.league_prefix or ["crict20blast"])

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT plan_id, bet_ledger_id, fixture_key, strategy_label,
               chunks_placed, chunks_filled, filled_size_usdc, total_size_usdc, status, kickoff_at
        FROM order_plans
        WHERE status IN ('pending', 'executing')
        ORDER BY kickoff_at
        """
    )
    plans = [dict(r) for r in cur.fetchall()
             if (r["fixture_key"] or "").startswith(prefixes)]

    if not plans:
        logger.info("No pending/executing plans matching prefixes %s.", prefixes)
        return 0

    plan_ids = [p["plan_id"] for p in plans]
    qmarks = ",".join("?" for _ in plan_ids)
    cur.execute(
        f"""
        SELECT chunk_id, plan_id, status, size_usdc, limit_price, polymarket_order_id
        FROM order_chunks
        WHERE plan_id IN ({qmarks})
          AND status IN ('pending', 'placed')
          AND polymarket_order_id IS NOT NULL
        """,
        plan_ids,
    )
    resting = [dict(r) for r in cur.fetchall()]

    notional = sum(p["total_size_usdc"] or 0.0 for p in plans)
    logger.info(
        "Matched %d plan(s) across prefixes %s (~$%.2f notional), %d resting order(s):",
        len(plans), prefixes, notional, len(resting),
    )
    for p in plans:
        logger.info(
            "  plan #%s  %s  %s  placed=%s filled=%s  filled_usdc=%.2f  total=%.2f  kickoff=%s",
            p["plan_id"], p["fixture_key"], p["strategy_label"],
            p["chunks_placed"], p["chunks_filled"],
            (p["filled_size_usdc"] or 0.0), (p["total_size_usdc"] or 0.0), p["kickoff_at"],
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
    from src.integrations.polymarket.order_audit import record_order_cancelled

    pm = PolymarketClient()

    # 1. Capture any real maker fills before we cancel anything.
    try:
        trades = fetch_all_clob_trades(pm)
        fills = index_fills_by_order_id(trades)
    except Exception as exc:
        logger.warning("CLOB trade fetch failed; finalizing from local chunks only: %s", exc)
        fills = {}

    for plan_id in plan_ids:
        if fills:
            try:
                sync_placed_chunk_fills(conn, cur, fills, plan_id=plan_id)
            except Exception as exc:
                logger.warning("sync fills failed for plan #%s: %s", plan_id, exc)

    # 2. Cancel each resting order individually (leaves other leagues untouched).
    n_cancelled_orders = 0
    for ch in resting:
        oid = ch["polymarket_order_id"]
        try:
            pm.cancel_order(oid)
            record_order_cancelled(oid, reason="league_pull_blast", conn=conn)
            n_cancelled_orders += 1
            logger.info(
                "  cancelled order %s (plan #%s, $%.2f @ %.4f)",
                oid[:18], ch["plan_id"], ch["size_usdc"] or 0.0, ch["limit_price"] or 0.0,
            )
        except Exception as exc:
            logger.warning("  cancel_order failed for %s: %s", oid[:18], exc)

    # 3. Reflect cancellations locally, then finalize each plan.
    n_completed = 0
    n_cancelled = 0
    for plan_id in plan_ids:
        cur.execute(
            "UPDATE order_chunks SET status = 'cancelled' "
            "WHERE plan_id = ? AND status IN ('pending', 'placed')",
            (plan_id,),
        )
        try:
            result = finalize_plan_from_chunks(conn, cur, plan_id)
            final_status = result.get("plan_status") if isinstance(result, dict) else None
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
        "Done. orders_cancelled=%d  plans_completed=%d (had real fills)  plans_cancelled=%d",
        n_cancelled_orders, n_completed, n_cancelled,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
