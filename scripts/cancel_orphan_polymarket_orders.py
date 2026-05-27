#!/usr/bin/env python3
"""Cancel resting Polymarket orders whose bet/plan was wrongly marked cancelled.

Caused by a bug in finalize_plan_from_chunks() (fixed) that marked plans as
'cancelled' even when chunks were still resting on the book. This left the
on-chain orders live with no corresponding active bet in our DB.

This script:
  1. Fetches all live open orders on Polymarket CLOB
  2. Cross-references each to order_chunks
  3. For chunks whose bet status is 'cancelled' (orphan): cancels the order
     on Polymarket, marks the chunk 'cancelled', records in order_audit.
  4. Leaves orders intact when the parent bet is still active (filled, placed,
     proposed) — those are legitimate working orders.

Usage:
    venv311/bin/python scripts/cancel_orphan_polymarket_orders.py --dry-run
    venv311/bin/python scripts/cancel_orphan_polymarket_orders.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def cancel_orphans(dry_run: bool = False) -> Dict[str, Any]:
    from src.data.database import get_connection, get_db_connection
    from src.integrations.polymarket import PolymarketClient
    from src.integrations.polymarket.order_audit import record_order_cancelled

    pm = PolymarketClient()
    sdk = pm._get_clob_sdk_client()
    open_orders = sdk.get_open_orders() or []
    logger.info("Polymarket open orders: %d", len(open_orders))

    live_by_oid = {}
    for o in open_orders:
        oid = o.get("id") or o.get("orderID") or o.get("order_id")
        if oid:
            live_by_oid[str(oid)] = o

    summary: Dict[str, Any] = {
        "n_live": len(live_by_oid),
        "n_orphans_found": 0,
        "n_cancelled": 0,
        "n_kept": 0,
        "cancelled": [],
        "kept": [],
        "errors": [],
    }

    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT oc.chunk_id, oc.polymarket_order_id, oc.status as chunk_status,
                   oc.size_usdc, oc.limit_price,
                   op.plan_id, op.fixture_key, op.bet_ledger_id,
                   bl.status as bet_status, bl.strategy_label, bl.side_label
            FROM order_chunks oc
            JOIN order_plans op ON op.plan_id = oc.plan_id
            LEFT JOIN bet_ledger bl ON bl.bet_id = op.bet_ledger_id
            WHERE oc.polymarket_order_id IS NOT NULL
            """
        )
        rows_by_oid = {str(r["polymarket_order_id"]): dict(r) for r in cur.fetchall()}

        for oid, order in live_by_oid.items():
            row = rows_by_oid.get(oid)
            outcome = order.get("outcome") or "?"
            orig_size = float(order.get("original_size") or order.get("size") or 0)
            matched = float(order.get("size_matched") or 0)
            remaining = orig_size - matched
            price = float(order.get("price") or 0)
            stake = remaining * price if price > 0 else 0
            tag = f"{outcome:24s} $remaining={stake:6.2f} oid={oid[:18]}"
            if row is None:
                summary["errors"].append({"oid": oid, "reason": "not in DB"})
                logger.warning(f"  {tag} -> NOT IN DB (ghost on-chain order)")
                continue
            bet_status = (row["bet_status"] or "").lower()
            chunk_status = (row["chunk_status"] or "").lower()
            if bet_status == "cancelled":
                summary["n_orphans_found"] += 1
                logger.info(
                    f"  ORPHAN: {row['strategy_label']:16s} {row['side_label']:22s} "
                    f"{row['fixture_key']:40s} {tag} bet_status={bet_status}"
                )
                if dry_run:
                    summary["cancelled"].append({**row, "oid": oid, "stake": stake})
                    continue
                try:
                    pm.cancel_order(oid)
                except Exception as exc:
                    summary["errors"].append({"oid": oid, "reason": f"cancel_order failed: {exc}"})
                    logger.warning(f"    cancel_order failed: {exc}")
                    continue
                # Mark chunk cancelled in DB
                cur.execute(
                    "UPDATE order_chunks SET status = 'cancelled' WHERE chunk_id = ?",
                    (row["chunk_id"],),
                )
                record_order_cancelled(oid, reason="orphan_cleanup", conn=conn)
                summary["n_cancelled"] += 1
                summary["cancelled"].append({**row, "oid": oid, "stake": stake})
            else:
                summary["n_kept"] += 1
                summary["kept"].append({**row, "oid": oid, "stake": stake, "bet_status": bet_status})
                logger.info(
                    f"  KEEP:   {row['strategy_label']:16s} {row['side_label']:22s} "
                    f"{tag} bet_status={bet_status} chunk_status={chunk_status}"
                )

        if not dry_run:
            conn.commit()

    logger.info(
        "Done. orphans_found=%d cancelled=%d kept=%d errors=%d",
        summary["n_orphans_found"],
        summary["n_cancelled"],
        summary["n_kept"],
        len(summary["errors"]),
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cancel_orphans(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
