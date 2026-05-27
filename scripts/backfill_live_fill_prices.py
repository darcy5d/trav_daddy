#!/usr/bin/env python3
"""One-shot backfill for bet_ledger rows missing fill_price / fill_size_usdc.

Before bet_placement.py was taught how to parse v2 CLOB responses, every
real (bet_kind='real') bet landed in the ledger with fill_price=NULL and
fill_size_usdc defaulted to the proposed stake. That in turn made reconcile
return None for pnl_realised_usdc on every settled bet, so cumulative P&L
read as $0.00 on the live-betting dashboard even though ~$76 of stakes had
actually cleared the CLOB.

This script walks every bet_kind='real' row with fill_price IS NULL AND
polymarket_order_id IS NOT NULL, pulls the corresponding trade(s) from the
v2 CLOB via `get_trades`, and updates the row with the exact fill_price
(USDC/share) + fill_size_usdc (USDC actually spent). Idempotent — already-
backfilled rows are skipped. After running this, re-run reconcile to
recompute pnl_realised_usdc on any previously-settled rows.

Usage:
    venv311/bin/python scripts/backfill_live_fill_prices.py
    venv311/bin/python scripts/backfill_live_fill_prices.py --dry-run
    venv311/bin/python scripts/backfill_live_fill_prices.py --reconcile
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _fetch_candidate_bets(conn) -> List[Dict[str, Any]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT bet_id, polymarket_order_id, size_usdc, side, side_label, fixture_key, status
        FROM bet_ledger
        WHERE COALESCE(bet_kind, 'real') = 'real'
          AND fill_price IS NULL
          AND polymarket_order_id IS NOT NULL
          AND polymarket_order_id != ''
        ORDER BY bet_id ASC
        """
    )
    return [dict(r) for r in cur.fetchall()]


def _fetch_all_recent_trades(sdk) -> Dict[str, Dict[str, Any]]:
    """Return {taker_order_id: trade_dict} indexed by orderID for fast lookup."""
    from py_clob_client_v2 import TradeParams

    trades = sdk.get_trades(TradeParams())
    if not isinstance(trades, list):
        return {}
    by_order: Dict[str, Dict[str, Any]] = {}
    for t in trades:
        if not isinstance(t, dict):
            continue
        tid = t.get("taker_order_id")
        if tid and tid not in by_order:
            by_order[tid] = t
    return by_order


def _compute_fill_from_trade(trade: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Extract (fill_price, fill_size_usdc) from a single CLOB trade dict."""
    try:
        price = float(trade.get("price"))
    except (TypeError, ValueError):
        return None
    try:
        size = float(trade.get("size"))
    except (TypeError, ValueError):
        return None
    if price <= 0 or size <= 0:
        return None
    side = (trade.get("side") or "").upper()
    # For BUY the trade `size` is the outcome-tokens field received (shares);
    # USDC spent = price * size. For SELL the reverse — we'd have sold `size`
    # shares and received `price * size` USDC. Both cases land the same
    # `fill_size_usdc` because our ledger tracks USDC regardless of side.
    fill_size_usdc = price * size if side == "BUY" else price * size
    return {"fill_price": price, "fill_size_usdc": fill_size_usdc}


def backfill(dry_run: bool = False, run_reconcile: bool = False) -> Dict[str, Any]:
    from src.data.database import get_connection, get_db_connection
    from src.integrations.polymarket import PolymarketClient

    pm = PolymarketClient()
    sdk = pm._get_clob_sdk_client()

    summary: Dict[str, Any] = {
        "n_candidates": 0,
        "n_updated": 0,
        "n_missing_trade": 0,
        "updates": [],
        "missing": [],
    }

    logger.info("Fetching recent trades from CLOB (v2)...")
    trades_by_order = _fetch_all_recent_trades(sdk)
    logger.info(f"  loaded {len(trades_by_order)} trade records")

    with get_db_connection() as conn:
        candidates = _fetch_candidate_bets(conn)
    summary["n_candidates"] = len(candidates)
    logger.info(f"Found {len(candidates)} bets with fill_price=NULL to backfill")

    for bet in candidates:
        order_id = bet["polymarket_order_id"]
        trade = trades_by_order.get(order_id)
        if trade is None:
            summary["n_missing_trade"] += 1
            summary["missing"].append({
                "bet_id": bet["bet_id"],
                "order_id": order_id,
                "side_label": bet["side_label"],
            })
            logger.warning(
                f"  bet_id={bet['bet_id']}  order={order_id[:12]}...  no matching trade found"
            )
            continue
        fill = _compute_fill_from_trade(trade)
        if fill is None:
            summary["n_missing_trade"] += 1
            summary["missing"].append({
                "bet_id": bet["bet_id"], "order_id": order_id, "reason": "malformed trade",
            })
            continue

        summary["updates"].append({
            "bet_id": bet["bet_id"],
            "side_label": bet["side_label"],
            "fill_price": round(fill["fill_price"], 4),
            "fill_size_usdc": round(fill["fill_size_usdc"], 4),
        })

        if dry_run:
            logger.info(
                f"  [DRY] bet_id={bet['bet_id']:>3d}  {bet['side_label']:25s}  "
                f"price={fill['fill_price']:.4f}  stake=${fill['fill_size_usdc']:.4f}"
            )
            continue

        with get_db_connection() as conn:
            conn.execute(
                """
                UPDATE bet_ledger
                SET fill_price = ?,
                    fill_size_usdc = ?
                WHERE bet_id = ?
                """,
                (fill["fill_price"], fill["fill_size_usdc"], bet["bet_id"]),
            )
            conn.commit()
        summary["n_updated"] += 1
        logger.info(
            f"  updated bet_id={bet['bet_id']:>3d}  {bet['side_label']:25s}  "
            f"price={fill['fill_price']:.4f}  stake=${fill['fill_size_usdc']:.4f}"
        )

    if run_reconcile and not dry_run:
        logger.info("Running reconcile to recompute pnl_realised_usdc on settled bets...")
        from src.integrations.polymarket.reconcile import reconcile_pending_bets
        # Reset any previously-settled rows back to 'filled' so reconcile
        # picks them up again and recomputes pnl with the fresh fill_price.
        # This is safe because settle_outcome + pnl are overwritten.
        with get_db_connection() as conn:
            conn.execute(
                """
                UPDATE bet_ledger
                SET status = 'filled',
                    settled_at = NULL,
                    pnl_realised_usdc = NULL
                WHERE COALESCE(bet_kind, 'real') = 'real'
                  AND status = 'settled'
                  AND pnl_realised_usdc IS NULL
                """
            )
            conn.commit()
        rec = reconcile_pending_bets()
        summary["reconcile"] = rec
        logger.info(f"reconcile summary: {rec}")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill fill_price/fill_size_usdc for real bets")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change; don't write to DB")
    parser.add_argument("--reconcile", action="store_true", help="Re-run reconcile after backfill to fix pnl_realised_usdc")
    args = parser.parse_args()

    summary = backfill(dry_run=args.dry_run, run_reconcile=args.reconcile)

    print()
    print("=" * 70)
    print("BACKFILL SUMMARY")
    print("=" * 70)
    print(f"  Candidates (fill_price=NULL, order_id present): {summary['n_candidates']}")
    print(f"  Rows updated:                                    {summary['n_updated']}")
    print(f"  No matching trade:                               {summary['n_missing_trade']}")
    if summary.get("reconcile"):
        r = summary["reconcile"]
        print(f"  Reconcile: checked={r['n_checked']}  settled={r['n_settled']}  still_pending={r['n_still_pending']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
