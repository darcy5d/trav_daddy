"""CLOB fill detection for TWAP limit orders (MAKER) and market orders (TAKER).

Polymarket's /data/trades payload distinguishes:
  - TAKER: our order crossed the book immediately (FOK/market-style).
  - MAKER: our resting limit order was hit; fills appear under
    trade['maker_orders'] with our order_id, not as taker_order_id.

TWAP limit orders almost always fill as MAKER. Without indexing maker_orders,
fills are invisible to bet_ledger and reconcile.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

END_CURSOR = "LTE="


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def trade_usdc(trade: Dict[str, Any]) -> float:
    try:
        price = float(trade.get("price") or 0)
        size = float(trade.get("size") or 0)
    except (TypeError, ValueError):
        return 0.0
    return price * size if price > 0 and size > 0 else 0.0


def maker_entry_usdc(entry: Dict[str, Any]) -> Tuple[float, float]:
    """Return (shares, usdc) for one maker_orders leg."""
    try:
        shares = float(entry.get("matched_amount") or 0)
        price = float(entry.get("price") or 0)
    except (TypeError, ValueError):
        return 0.0, 0.0
    if shares <= 0 or price <= 0:
        return 0.0, 0.0
    return shares, shares * price


def index_fills_by_order_id(
    trades: List[Dict[str, Any]],
    known_order_ids: Optional[set] = None,
) -> Dict[str, Dict[str, Any]]:
    """Aggregate all MAKER and TAKER fills keyed by our order id.

    On batched MAKER trades Polymarket lists every counterparty leg in
    maker_orders[], not only ours. When there are multiple legs we only
    index an order id that appears in known_order_ids (order_chunks /
    bet_ledger). Single-leg MAKER trades are treated as entirely ours.
    """
    known = known_order_ids or set()
    by_order: Dict[str, Dict[str, Any]] = {}

    def _accum(order_id: str, shares: float, usdc: float, price: float, role: str,
               asset_id: Optional[str] = None, outcome: Optional[str] = None) -> None:
        if not order_id or shares <= 0 or usdc <= 0:
            return
        rec = by_order.setdefault(
            order_id,
            {
                "order_id": order_id,
                "fill_shares": 0.0,
                "fill_usdc": 0.0,
                "roles": set(),
                "asset_id": None,
                "outcome": None,
            },
        )
        rec["fill_shares"] += shares
        rec["fill_usdc"] += usdc
        rec["roles"].add(role)
        rec["avg_fill_price"] = rec["fill_usdc"] / rec["fill_shares"] if rec["fill_shares"] > 0 else price
        if asset_id and not rec["asset_id"]:
            rec["asset_id"] = asset_id
        if outcome and not rec["outcome"]:
            rec["outcome"] = outcome

    for trade in trades:
        trader_side = (trade.get("trader_side") or "").upper()
        tid = trade.get("taker_order_id")
        if tid and trader_side == "TAKER":
            shares = float(trade.get("size") or 0)
            price = float(trade.get("price") or 0)
            _accum(
                str(tid), shares, trade_usdc(trade), price, "taker",
                asset_id=str(trade.get("asset_id") or "") or None,
                outcome=trade.get("outcome"),
            )
            # maker_orders on TAKER trades are counterparty resting orders — not ours.
            continue

        if trader_side != "MAKER":
            continue

        maker_orders = trade.get("maker_orders") or []
        multi_leg = len(maker_orders) > 1
        for mo in maker_orders:
            oid = mo.get("order_id")
            if not oid:
                continue
            oid = str(oid)
            if multi_leg and oid not in known:
                continue
            shares, usdc = maker_entry_usdc(mo)
            price = float(mo.get("price") or 0)
            _accum(
                oid, shares, usdc, price, "maker",
                asset_id=str(mo.get("asset_id") or "") or None,
                outcome=mo.get("outcome"),
            )

    # JSON-serialise roles for callers that log/report
    for rec in by_order.values():
        rec["roles"] = sorted(rec["roles"])
    return by_order


def fetch_all_clob_trades(pm_client: Any, since_days: Optional[int] = None) -> List[Dict[str, Any]]:
    """Paginate all wallet trades from the CLOB API."""
    from datetime import timedelta
    from py_clob_client_v2 import TradeParams

    sdk = pm_client._get_clob_sdk_client()
    params = TradeParams()
    if since_days is not None and since_days > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
        params.after = int(cutoff.timestamp())

    all_trades: List[Dict[str, Any]] = []
    cursor: Optional[str] = None
    while True:
        page = sdk.get_trades_paginated(params, next_cursor=cursor)
        batch = page.get("data") or page.get("trades") or []
        if isinstance(batch, list):
            all_trades.extend(t for t in batch if isinstance(t, dict))
        cursor = page.get("next_cursor")
        if not cursor or cursor == END_CURSOR:
            break
    return all_trades


PARTIAL_FILL_TOPUP_THRESHOLD = 0.02  # ignore <2c rounding noise


def apply_fill_to_chunk(
    conn: sqlite3.Connection,
    cur: sqlite3.Cursor,
    chunk_id: int,
    fill: Dict[str, Any],
) -> bool:
    """Mark a chunk filled with on-chain amounts. Returns True if updated.

    Handles two cases:
      1. Chunk in ('placed','cancelled','pending') and an on-chain fill exists
         -> upgrade to 'filled' with the on-chain values.
      2. Chunk already 'filled' but the on-chain fill_usdc exceeds the recorded
         fill_size_usdc by more than PARTIAL_FILL_TOPUP_THRESHOLD -> top up.
         Polymarket maker limit orders can keep resting after a partial fill and
         match additional liquidity later. Without this top-up the bet/plan
         under-counts what we actually own on-chain.
    """
    fill_usdc = float(fill.get("fill_usdc") or 0)
    if fill_usdc <= 0:
        return False
    avg_price = float(fill.get("avg_fill_price") or 0)
    now_iso = _utc_now_iso()

    cur.execute(
        "SELECT status, fill_size_usdc FROM order_chunks WHERE chunk_id = ?",
        (chunk_id,),
    )
    row = cur.fetchone()
    if row is None:
        return False
    status = (row["status"] or "").lower()
    current = float(row["fill_size_usdc"] or 0)

    if status in ("placed", "cancelled", "pending"):
        cur.execute(
            """
            UPDATE order_chunks
            SET status = 'filled', filled_at = ?, fill_price = ?, fill_size_usdc = ?
            WHERE chunk_id = ?
            """,
            (now_iso, avg_price, round(fill_usdc, 4), chunk_id),
        )
        if cur.rowcount:
            conn.commit()
            return True
        return False

    if status == "filled" and (fill_usdc - current) > PARTIAL_FILL_TOPUP_THRESHOLD:
        cur.execute(
            """
            UPDATE order_chunks
            SET fill_price = ?, fill_size_usdc = ?, filled_at = COALESCE(filled_at, ?)
            WHERE chunk_id = ?
            """,
            (avg_price, round(fill_usdc, 4), now_iso, chunk_id),
        )
        if cur.rowcount:
            conn.commit()
            logger.info(
                "Top-up partial fill on chunk %d: $%.2f -> $%.2f",
                chunk_id, current, fill_usdc,
            )
            return True
    return False


def sync_placed_chunk_fills(
    conn: sqlite3.Connection,
    cur: sqlite3.Cursor,
    fills_by_order: Dict[str, Dict[str, Any]],
    *,
    plan_id: Optional[int] = None,
) -> int:
    """Apply on-chain fills to order_chunks rows with polymarket_order_id.

    Sweeps both:
      - non-filled chunks (placed/cancelled/pending) to mark them filled.
      - already-filled chunks where the on-chain amount has grown via
        subsequent partial fills on the same maker order.
    """
    if plan_id is not None:
        cur.execute(
            """
            SELECT chunk_id, polymarket_order_id
            FROM order_chunks
            WHERE plan_id = ?
              AND polymarket_order_id IS NOT NULL
              AND status IN ('placed', 'cancelled', 'pending', 'filled')
            """,
            (plan_id,),
        )
    else:
        cur.execute(
            """
            SELECT chunk_id, polymarket_order_id
            FROM order_chunks
            WHERE polymarket_order_id IS NOT NULL
              AND status IN ('placed', 'cancelled', 'pending', 'filled')
            """
        )
    n_updated = 0
    for row in cur.fetchall():
        oid = str(row["polymarket_order_id"] or "")
        fill = fills_by_order.get(oid)
        if not fill:
            continue
        if apply_fill_to_chunk(conn, cur, row["chunk_id"], fill):
            n_updated += 1
    return n_updated


def finalize_plan_from_chunks(
    conn: sqlite3.Connection,
    cur: sqlite3.Cursor,
    plan_id: int,
) -> Dict[str, Any]:
    """Recompute plan + bet_ledger status from chunk fill state.

    Only transition the plan to a terminal state ('completed' or 'cancelled')
    when no chunks are still in flight ('placed' or 'pending'). If any chunks
    are still resting on the book the plan stays in 'executing' (or its
    existing in-progress state) so we don't spuriously mark live orders as
    cancelled.
    """
    cur.execute(
        """
        SELECT
          SUM(CASE WHEN status = 'filled' THEN 1 ELSE 0 END) as filled_cnt,
          SUM(CASE WHEN status IN ('placed', 'pending') THEN 1 ELSE 0 END) as inflight_cnt,
          COALESCE(SUM(CASE WHEN status = 'filled' THEN fill_size_usdc ELSE 0 END), 0) as total_usdc,
          COALESCE(SUM(CASE WHEN status = 'filled' THEN fill_price * fill_size_usdc ELSE 0 END), 0) as weighted_price
        FROM order_chunks
        WHERE plan_id = ?
        """,
        (plan_id,),
    )
    agg = cur.fetchone()
    filled_count = int(agg["filled_cnt"] or 0)
    inflight_count = int(agg["inflight_cnt"] or 0)
    total_filled_usdc = float(agg["total_usdc"] or 0)
    weighted_price_sum = float(agg["weighted_price"] or 0)
    avg_fill = (weighted_price_sum / total_filled_usdc) if total_filled_usdc > 0 else None

    # If chunks are still working, don't change plan/bet status; just refresh
    # the fill statistics so callers can see in-progress fill totals.
    if inflight_count > 0:
        cur.execute(
            """
            UPDATE order_plans
            SET chunks_filled = ?, filled_size_usdc = ?, avg_fill_price = ?, updated_at = ?
            WHERE plan_id = ?
            """,
            (filled_count, total_filled_usdc, avg_fill, _utc_now_iso(), plan_id),
        )
        conn.commit()
        return {
            "plan_id": plan_id,
            "bet_id": None,
            "plan_status": "in_progress",
            "filled_usdc": round(total_filled_usdc, 4),
            "bet_updated": False,
            "skipped": "inflight_chunks_remaining",
        }

    final_status = "completed" if filled_count > 0 else "cancelled"

    cur.execute(
        """
        UPDATE order_plans
        SET status = ?, chunks_filled = ?, filled_size_usdc = ?, avg_fill_price = ?, updated_at = ?
        WHERE plan_id = ?
        """,
        (final_status, filled_count, total_filled_usdc, avg_fill, _utc_now_iso(), plan_id),
    )

    cur.execute("SELECT bet_ledger_id FROM order_plans WHERE plan_id = ?", (plan_id,))
    plan_row = cur.fetchone()
    bet_id = plan_row["bet_ledger_id"] if plan_row else None
    bet_updated = False

    if bet_id:
        cur.execute(
            "SELECT polymarket_order_id FROM order_chunks WHERE plan_id = ? AND polymarket_order_id IS NOT NULL LIMIT 1",
            (plan_id,),
        )
        chunk_oid_row = cur.fetchone()
        primary_oid = chunk_oid_row["polymarket_order_id"] if chunk_oid_row else None

        now_iso = _utc_now_iso()
        if total_filled_usdc > 0:
            cur.execute(
                """
                UPDATE bet_ledger
                SET status = 'filled',
                    filled_at = COALESCE(filled_at, ?),
                    fill_price = ?,
                    fill_size_usdc = ?,
                    placed_at = COALESCE(placed_at, ?),
                    polymarket_order_id = COALESCE(polymarket_order_id, ?),
                    reconciled_at = ?
                WHERE bet_id = ?
                  AND status IN ('proposed', 'cancelled', 'placed')
                """,
                (now_iso, avg_fill, total_filled_usdc, now_iso, primary_oid, now_iso, bet_id),
            )
            bet_updated = cur.rowcount > 0

            # Top-up case: bet already in 'filled' (or even 'settled' pre-resolution)
            # status but new partial fills have pushed total_filled_usdc above the
            # recorded fill_size_usdc. Update fill_size_usdc and fill_price so the
            # bet reflects what we actually own on-chain. Do NOT touch settled rows
            # where pnl is already realized.
            #
            # Open stake target is gross-filled MINUS what we have already sold
            # off this row (exit_cost_usdc, accrued by rebalance/cashout partial
            # exits). Without this subtraction the top-up would re-inflate
            # fill_size_usdc back to the gross on-chain fill after every exit,
            # wiping the decrement and over-counting the open position.
            cur.execute(
                "SELECT status, fill_size_usdc, COALESCE(exit_cost_usdc, 0.0) AS exit_cost_usdc "
                "FROM bet_ledger WHERE bet_id = ?",
                (bet_id,),
            )
            br = cur.fetchone()
            if br is not None and (br["status"] or "").lower() == "filled":
                current_fill = float(br["fill_size_usdc"] or 0)
                exit_cost = float(br["exit_cost_usdc"] or 0)
                target_fill = max(0.0, total_filled_usdc - exit_cost)
                if target_fill - current_fill > PARTIAL_FILL_TOPUP_THRESHOLD:
                    cur.execute(
                        """
                        UPDATE bet_ledger
                        SET fill_size_usdc = ?,
                            fill_price = ?,
                            reconciled_at = ?
                        WHERE bet_id = ? AND status = 'filled'
                        """,
                        (round(target_fill, 6), avg_fill, now_iso, bet_id),
                    )
                    if cur.rowcount:
                        bet_updated = True
                        logger.info(
                            "Top-up bet_ledger #%s fill_size: $%.2f -> $%.2f "
                            "(gross $%.2f - exited $%.2f, plan %s)",
                            bet_id, current_fill, target_fill,
                            total_filled_usdc, exit_cost, plan_id,
                        )
        else:
            cur.execute(
                """
                UPDATE bet_ledger
                SET status = 'cancelled',
                    cancel_reason = COALESCE(cancel_reason, 'twap_no_fill'),
                    cancelled_at = COALESCE(cancelled_at, ?),
                    reconciled_at = ?
                WHERE bet_id = ? AND status = 'proposed'
                """,
                (now_iso, now_iso, bet_id),
            )
            bet_updated = cur.rowcount > 0

    conn.commit()
    return {
        "plan_id": plan_id,
        "bet_id": bet_id,
        "plan_status": final_status,
        "filled_usdc": round(total_filled_usdc, 4),
        "bet_updated": bet_updated,
    }
