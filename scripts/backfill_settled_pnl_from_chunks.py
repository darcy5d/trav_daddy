#!/usr/bin/env python3
"""Recompute fill_size_usdc / fill_price / pnl_realised_usdc for settled bets
whose chunk fills have grown since settlement (partial-fill top-ups).

When a TWAP maker order partial-fills, our code marked the chunk 'filled' with
the partial amount and the bet got settled with that stake. If the same
resting order matched more liquidity later, the chunk's fill_size_usdc was
under-counted at settlement time, which under-counted both the realised pnl
and the on-chain cost basis. This script restates each affected settled bet.

Usage:
    venv311/bin/python scripts/backfill_settled_pnl_from_chunks.py --dry-run
    venv311/bin/python scripts/backfill_settled_pnl_from_chunks.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

THRESHOLD_USDC = 0.02


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _recompute(bet: Dict[str, Any], fee_pct: float) -> Dict[str, Any]:
    """Return the corrected fill / pnl fields for a single bet."""
    new_fill = float(bet["chunk_total"])
    if new_fill <= 0:
        return {}
    weighted = float(bet["weighted_price_sum"] or 0)
    new_price = (weighted / new_fill) if new_fill > 0 else None
    if not new_price or new_price <= 0:
        return {}
    new_shares = new_fill / new_price

    is_cashout = bet["cashout_triggered_at"] is not None
    if is_cashout:
        cashout_price = float(bet["cashout_price"] or 0)
        proceeds = new_shares * cashout_price
        fee = proceeds * fee_pct
        new_pnl = round(proceeds - new_fill - fee, 4)
        return {
            "fill_size_usdc": round(new_fill, 4),
            "fill_price": round(new_price, 6),
            "pnl_realised_usdc": new_pnl,
            "cashout_pnl_usdc": new_pnl,
            "settle_outcome": None,
        }

    outcome = bet["settle_outcome"]
    if outcome is None:
        return {}
    if int(outcome) == 1:
        gross_payout = new_shares * 1.0
        fee = gross_payout * fee_pct
        new_pnl = round(gross_payout - new_fill - fee, 4)
    else:
        new_pnl = round(-new_fill, 4)
    return {
        "fill_size_usdc": round(new_fill, 4),
        "fill_price": round(new_price, 6),
        "pnl_realised_usdc": new_pnl,
        "cashout_pnl_usdc": None,
        "settle_outcome": int(outcome),
    }


def backfill(dry_run: bool = False) -> Dict[str, Any]:
    from src.data.database import get_connection, get_db_connection
    from src.integrations.odds.polymarket_compare import POLYMARKET_TAKER_FEE

    summary: Dict[str, Any] = {"n_examined": 0, "n_updated": 0, "rows": []}
    now = _utc_now_iso()

    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT bl.bet_id, bl.strategy_label, bl.side_label, bl.fixture_key,
                   bl.fill_size_usdc, bl.fill_price,
                   bl.pnl_realised_usdc, bl.settle_outcome,
                   bl.cashout_triggered_at, bl.cashout_price, bl.cashout_pnl_usdc,
                   op.plan_id,
                   (SELECT COALESCE(SUM(fill_size_usdc),0) FROM order_chunks WHERE plan_id=op.plan_id AND status='filled') as chunk_total,
                   (SELECT COALESCE(SUM(fill_price*fill_size_usdc),0) FROM order_chunks WHERE plan_id=op.plan_id AND status='filled') as weighted_price_sum
            FROM bet_ledger bl
            JOIN order_plans op ON op.bet_ledger_id = bl.bet_id
            WHERE bl.status='settled'
              AND COALESCE(bl.bet_kind,'real')='real'
              AND bl.strategy_label != 'RECONCILE_GHOST'
            """
        ).fetchall()

        for row in rows:
            summary["n_examined"] += 1
            old_fill = float(row["fill_size_usdc"] or 0)
            chunk_total = float(row["chunk_total"] or 0)
            if chunk_total - old_fill <= THRESHOLD_USDC:
                continue
            corrected = _recompute(dict(row), POLYMARKET_TAKER_FEE)
            if not corrected:
                continue

            line = {
                "bet_id": row["bet_id"],
                "strategy_label": row["strategy_label"],
                "side_label": row["side_label"],
                "old_fill": old_fill,
                "new_fill": corrected["fill_size_usdc"],
                "old_price": row["fill_price"],
                "new_price": corrected["fill_price"],
                "old_pnl": row["pnl_realised_usdc"],
                "new_pnl": corrected["pnl_realised_usdc"],
                "delta_pnl": round(corrected["pnl_realised_usdc"] - (row["pnl_realised_usdc"] or 0), 4),
                "kind": "cashout" if row["cashout_triggered_at"] else (
                    "win" if row["settle_outcome"] == 1 else "loss"
                ),
            }
            summary["rows"].append(line)
            logger.info(
                f"  #{row['bet_id']} {line['kind']:7s} {row['side_label']:20s} "
                f"stake ${old_fill:6.2f}→${corrected['fill_size_usdc']:6.2f} "
                f"pnl ${row['pnl_realised_usdc']:+6.2f}→${corrected['pnl_realised_usdc']:+6.2f} "
                f"(Δ {line['delta_pnl']:+.2f})"
            )
            if dry_run:
                continue

            if corrected.get("cashout_pnl_usdc") is not None:
                conn.execute(
                    """
                    UPDATE bet_ledger
                    SET fill_size_usdc = ?, fill_price = ?,
                        pnl_realised_usdc = ?, cashout_pnl_usdc = ?,
                        reconciled_at = ?
                    WHERE bet_id = ?
                    """,
                    (
                        corrected["fill_size_usdc"], corrected["fill_price"],
                        corrected["pnl_realised_usdc"], corrected["cashout_pnl_usdc"],
                        now, row["bet_id"],
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE bet_ledger
                    SET fill_size_usdc = ?, fill_price = ?,
                        pnl_realised_usdc = ?, reconciled_at = ?
                    WHERE bet_id = ?
                    """,
                    (
                        corrected["fill_size_usdc"], corrected["fill_price"],
                        corrected["pnl_realised_usdc"], now, row["bet_id"],
                    ),
                )
            summary["n_updated"] += 1

        if not dry_run and summary["n_updated"]:
            conn.commit()

    total_delta = round(sum(r["delta_pnl"] for r in summary["rows"]), 4)
    summary["total_delta_pnl"] = total_delta
    logger.info(
        f"Examined {summary['n_examined']} settled bets, "
        f"updated {summary['n_updated']}, total P&L delta = ${total_delta:+.2f}"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    backfill(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
