#!/usr/bin/env python3
"""Backfill TWAP limit-order fills that landed as MAKER on Polymarket.

Usage:
    venv311/bin/python scripts/backfill_twap_maker_fills.py --dry-run
    venv311/bin/python scripts/backfill_twap_maker_fills.py
    venv311/bin/python scripts/backfill_twap_maker_fills.py --fixture cricipl-mum-raj-2026-05-24
    venv311/bin/python scripts/backfill_twap_maker_fills.py --reconcile
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def backfill(
    *,
    dry_run: bool = False,
    fixture_key: str | None = None,
    run_reconcile: bool = False,
) -> dict:
    from src.data.database import get_connection
    from src.integrations.polymarket import PolymarketClient
    from src.integrations.polymarket.clob_fills import (
        fetch_all_clob_trades,
        finalize_plan_from_chunks,
        index_fills_by_order_id,
        sync_placed_chunk_fills,
    )
    from src.integrations.polymarket.reconcile import reconcile_pending_bets

    pm = PolymarketClient()
    logger.info("Fetching CLOB trades...")
    trades = fetch_all_clob_trades(pm)

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT polymarket_order_id FROM order_chunks WHERE polymarket_order_id IS NOT NULL"
        )
        known_ids = {str(r[0]) for r in cur.fetchall()}
        cur.execute(
            "SELECT polymarket_order_id FROM bet_ledger WHERE polymarket_order_id IS NOT NULL"
        )
        known_ids.update(str(r[0]) for r in cur.fetchall())

    fills_by_order = index_fills_by_order_id(trades, known_order_ids=known_ids)
    logger.info("Indexed fills for %d order ids", len(fills_by_order))

    summary = {"chunks_updated": 0, "plans_finalized": [], "settle": None}

    with get_connection() as conn:
        cur = conn.cursor()
        if fixture_key:
            cur.execute(
                """
                SELECT DISTINCT plan_id FROM order_plans
                WHERE fixture_key = ?
                ORDER BY plan_id
                """,
                (fixture_key,),
            )
            plan_ids = [r["plan_id"] for r in cur.fetchall()]
        else:
            cur.execute(
                """
                SELECT DISTINCT op.plan_id
                FROM order_plans op
                JOIN order_chunks oc ON oc.plan_id = op.plan_id
                WHERE oc.polymarket_order_id IS NOT NULL
                  AND oc.status IN ('placed', 'cancelled', 'pending')
                ORDER BY op.plan_id
                """
            )
            plan_ids = [r["plan_id"] for r in cur.fetchall()]

        if dry_run:
            for pid in plan_ids:
                cur.execute(
                    """
                    SELECT chunk_id, polymarket_order_id, status, fill_size_usdc
                    FROM order_chunks WHERE plan_id = ?
                    """,
                    (pid,),
                )
                for ch in cur.fetchall():
                    oid = str(ch["polymarket_order_id"] or "")
                    on_chain = fills_by_order.get(oid)
                    if on_chain and float(on_chain.get("fill_usdc") or 0) > 0:
                        logger.info(
                            "  [DRY] plan=%s chunk=%s status=%s -> fill $%.2f",
                            pid, ch["chunk_id"], ch["status"], on_chain["fill_usdc"],
                        )
                        summary["chunks_updated"] += 1
            return summary

        n_chunks = sync_placed_chunk_fills(conn, cur, fills_by_order)
        summary["chunks_updated"] = n_chunks
        logger.info("Updated %d chunk(s) from on-chain maker fills", n_chunks)

        for pid in plan_ids:
            result = finalize_plan_from_chunks(conn, cur, pid)
            if result.get("bet_updated") or result.get("filled_usdc", 0) > 0:
                summary["plans_finalized"].append(result)
                logger.info(
                    "  plan #%s -> %s, bet_id=%s filled=$%.2f",
                    pid,
                    result["plan_status"],
                    result.get("bet_id"),
                    result.get("filled_usdc", 0),
                )

    if run_reconcile and not dry_run:
        logger.info("Running reconcile_pending_bets for settlement...")
        summary["settle"] = reconcile_pending_bets()
        logger.info("Settlement: %s", summary["settle"])

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill TWAP MAKER fills from CLOB trades")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fixture", dest="fixture_key", default=None)
    parser.add_argument("--reconcile", action="store_true", help="Settle filled bets after backfill")
    args = parser.parse_args()
    summary = backfill(
        dry_run=args.dry_run,
        fixture_key=args.fixture_key,
        run_reconcile=args.reconcile,
    )
    print(summary)


if __name__ == "__main__":
    main()
