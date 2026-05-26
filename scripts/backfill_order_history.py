#!/usr/bin/env python3
"""Seed order_history from existing bet_ledger + order_chunks rows.

Idempotent. Run once after deploying the V10 schema; safe to re-run.

For each existing polymarket_order_id we have on record, this script writes
a row into order_history with the best-effort metadata we can reconstruct
(token_id, side, kind, prices, sizes, fill status) so reconcile can use
order_history as the source of truth for "did we place this order".

Usage:
    venv311/bin/python scripts/backfill_order_history.py
    venv311/bin/python scripts/backfill_order_history.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def backfill(dry_run: bool = False) -> dict:
    from src.data.database import get_connection, init_order_history

    init_order_history()
    summary = {"fok_rows": 0, "chunk_rows": 0, "cashout_rows": 0, "ghost_rows": 0}

    with get_connection() as conn:
        cur = conn.cursor()

        existing = {
            str(r[0]) for r in cur.execute(
                "SELECT polymarket_order_id FROM order_history WHERE polymarket_order_id IS NOT NULL"
            ).fetchall()
        }

        # --- 1. bet_ledger FOK orders ---
        # bet_ledger.polymarket_order_id is the FOK / placement-time order id.
        # For TWAP bets, this is COALESCEd from the first chunk later, so we
        # only record it as FOK when there is no plan attached.
        cur.execute(
            """
            SELECT bl.bet_id, bl.polymarket_order_id, bl.polymarket_token_id,
                   bl.side, bl.fill_size_usdc, bl.fill_price, bl.size_usdc,
                   bl.status, bl.placed_at, bl.filled_at, bl.cashout_order_id,
                   bl.strategy_label, op.plan_id
            FROM bet_ledger bl
            LEFT JOIN order_plans op ON op.bet_ledger_id = bl.bet_id
            WHERE bl.polymarket_order_id IS NOT NULL
              AND COALESCE(bl.bet_kind, 'real') = 'real'
            """
        )
        now = _utc_now_iso()
        for row in cur.fetchall():
            oid = str(row["polymarket_order_id"])
            if oid in existing:
                continue
            # If a plan exists for this bet AND the order_id matches a chunk,
            # the chunk pass below will own this. Skip to avoid mis-tagging
            # the kind on a row chunk-pass would tag as twap_chunk.
            chunk_match = cur.execute(
                "SELECT 1 FROM order_chunks WHERE polymarket_order_id = ?",
                (oid,),
            ).fetchone()
            if chunk_match:
                continue

            is_ghost = (row["strategy_label"] == "RECONCILE_GHOST")
            order_kind = "twap_chunk" if is_ghost else "fok"
            final_status = row["status"]
            # Map bet_ledger.status -> order_history.final_status
            if final_status == "settled":
                final_status = "filled"
            elif final_status == "errored":
                final_status = "error"
            elif final_status == "cancelled":
                final_status = "cancelled"
            elif final_status in ("placed", "proposed"):
                final_status = "placed"
            elif final_status == "filled":
                final_status = "filled"
            else:
                final_status = "placed"

            if not dry_run:
                cur.execute(
                    """
                    INSERT INTO order_history (
                        polymarket_order_id, bet_id,
                        token_id, side, order_kind,
                        size_usdc, posted_at,
                        final_status, final_reason,
                        fill_usdc, fill_price, filled_at,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        oid, row["bet_id"],
                        row["polymarket_token_id"] or "",
                        (row["side"] or "BUY").upper(),
                        order_kind,
                        row["size_usdc"],
                        row["placed_at"] or row["filled_at"] or now,
                        final_status,
                        "reconcile_ghost_backfill" if is_ghost else "backfill",
                        row["fill_size_usdc"] or 0,
                        row["fill_price"],
                        row["filled_at"],
                        now, now,
                    ),
                )
            existing.add(oid)
            if is_ghost:
                summary["ghost_rows"] += 1
            else:
                summary["fok_rows"] += 1

        # --- 2. order_chunks TWAP orders ---
        cur.execute(
            """
            SELECT oc.chunk_id, oc.plan_id, oc.polymarket_order_id,
                   oc.limit_price, oc.size_usdc, oc.size_shares,
                   oc.status, oc.placed_at, oc.filled_at,
                   oc.fill_size_usdc, oc.fill_price,
                   op.token_id, op.side, op.bet_ledger_id
            FROM order_chunks oc
            JOIN order_plans op ON op.plan_id = oc.plan_id
            WHERE oc.polymarket_order_id IS NOT NULL
            """
        )
        for row in cur.fetchall():
            oid = str(row["polymarket_order_id"])
            if oid in existing:
                continue
            ch_status = (row["status"] or "").lower()
            if ch_status == "filled":
                final_status = "filled"
            elif ch_status in ("cancelled", "canceled"):
                final_status = "cancelled"
            elif ch_status in ("placed", "pending"):
                final_status = "placed"
            elif ch_status == "partially_filled":
                final_status = "filled"  # treat as filled (we have a fill_usdc)
            else:
                final_status = "placed"

            if not dry_run:
                cur.execute(
                    """
                    INSERT INTO order_history (
                        polymarket_order_id, bet_id, chunk_id, plan_id,
                        token_id, side, order_kind,
                        limit_price, size_usdc, size_shares, posted_at,
                        final_status, final_reason,
                        fill_usdc, fill_price, filled_at,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        oid, row["bet_ledger_id"], row["chunk_id"], row["plan_id"],
                        row["token_id"] or "",
                        (row["side"] or "BUY").upper(),
                        "twap_chunk",
                        row["limit_price"], row["size_usdc"], row["size_shares"],
                        row["placed_at"] or row["filled_at"] or now,
                        final_status, "backfill",
                        row["fill_size_usdc"] or 0,
                        row["fill_price"], row["filled_at"],
                        now, now,
                    ),
                )
            existing.add(oid)
            summary["chunk_rows"] += 1

        # --- 3. cashout SELL orders ---
        cur.execute(
            """
            SELECT bet_id, cashout_order_id, polymarket_token_id,
                   cashout_price, cashout_pnl_usdc, cashout_triggered_at,
                   fill_size_usdc, fill_price
            FROM bet_ledger
            WHERE cashout_order_id IS NOT NULL
              AND COALESCE(bet_kind, 'real') = 'real'
            """
        )
        for row in cur.fetchall():
            oid = str(row["cashout_order_id"])
            if oid in existing:
                continue
            # Reconstruct rough shares from original buy fill / cashout price.
            shares = None
            if row["fill_price"] and row["fill_size_usdc"] and row["fill_price"] > 0:
                shares = float(row["fill_size_usdc"]) / float(row["fill_price"])
            cashout_usdc = None
            if row["cashout_price"] and shares is not None:
                cashout_usdc = float(row["cashout_price"]) * shares
            if not dry_run:
                cur.execute(
                    """
                    INSERT INTO order_history (
                        polymarket_order_id, bet_id,
                        token_id, side, order_kind,
                        limit_price, size_usdc, size_shares, posted_at,
                        final_status, final_reason,
                        fill_usdc, fill_price, filled_at,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        oid, row["bet_id"],
                        row["polymarket_token_id"] or "",
                        "SELL", "cashout",
                        row["cashout_price"], cashout_usdc, shares,
                        row["cashout_triggered_at"] or now,
                        "filled", "backfill",
                        cashout_usdc or 0, row["cashout_price"],
                        row["cashout_triggered_at"], now, now,
                    ),
                )
            existing.add(oid)
            summary["cashout_rows"] += 1

        # --- 4. RECONCILE_GHOST rows (treat polymarket_order_id as the actual on-chain id) ---
        cur.execute(
            """
            SELECT bet_id, polymarket_order_id, polymarket_token_id,
                   side, fill_size_usdc, fill_price, filled_at
            FROM bet_ledger
            WHERE strategy_label = 'RECONCILE_GHOST'
              AND polymarket_order_id IS NOT NULL
            """
        )
        for row in cur.fetchall():
            oid = str(row["polymarket_order_id"])
            if oid in existing:
                continue
            if not dry_run:
                cur.execute(
                    """
                    INSERT INTO order_history (
                        polymarket_order_id, bet_id,
                        token_id, side, order_kind,
                        size_usdc, posted_at,
                        final_status, final_reason,
                        fill_usdc, fill_price, filled_at,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        oid, row["bet_id"],
                        row["polymarket_token_id"] or "",
                        (row["side"] or "BUY").upper(),
                        "twap_chunk",
                        row["fill_size_usdc"],
                        row["filled_at"] or now,
                        "filled", "reconcile_ghost_backfill",
                        row["fill_size_usdc"] or 0, row["fill_price"], row["filled_at"],
                        now, now,
                    ),
                )
            existing.add(oid)
            summary["ghost_rows"] += 1

        if not dry_run:
            conn.commit()

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    summary = backfill(dry_run=args.dry_run)
    logger.info(
        "Backfill summary: fok=%(fok_rows)d twap_chunk=%(chunk_rows)d "
        "cashout=%(cashout_rows)d reconcile_ghost=%(ghost_rows)d",
        summary,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
