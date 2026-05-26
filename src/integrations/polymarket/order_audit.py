"""Permanent audit log of every Polymarket order id we have ever posted.

Reprice / resize on TWAP chunks overwrites or NULLs the live
order_chunks.polymarket_order_id. When a CLOB fill later arrives on the old
id we have no way to attribute it back to the originating bet unless we
keep an append-only history of every id we ever placed.

The reconcile loop treats order_history as the canonical "did we ever
place this order" set so stray maker fills are matched to the right bet
instead of inserted as RECONCILE_GHOST rows.

Public API (all idempotent; all failures are logged + swallowed so audit
errors never break order placement / reprice):

    record_order_placed(oid, *, bet_id=None, chunk_id=None, plan_id=None,
                        token_id, side, order_kind, limit_price=None,
                        size_usdc=None, size_shares=None, posted_at=None)
    record_order_filled(oid, *, fill_usdc, fill_price=None, filled_at=None)
    record_order_cancelled(oid, *, reason)
    record_order_replaced_by_reprice(old_oid, new_oid, *, reason='reprice')
    record_order_error(oid_or_none, *, bet_id, error_category, error_message)
    lookup_bet_for_order(oid)      -> dict | None  (follows reprice chain)
    all_known_order_ids(conn=None) -> set
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe(fn):
    """Decorator: log + swallow any exception so audit can never break a caller."""

    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            logger.warning(f"order_audit.{fn.__name__} failed: {exc}")
            return None

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


def _get_conn(conn: Optional[sqlite3.Connection] = None):
    """Return (conn, own) — own=True means caller-allocated."""
    if conn is not None:
        return conn, False
    from src.data.database import get_connection
    return get_connection(), True


@_safe
def record_order_placed(
    polymarket_order_id: str,
    *,
    bet_id: Optional[int] = None,
    chunk_id: Optional[int] = None,
    plan_id: Optional[int] = None,
    token_id: str,
    side: str,
    order_kind: str,
    limit_price: Optional[float] = None,
    size_usdc: Optional[float] = None,
    size_shares: Optional[float] = None,
    posted_at: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    """Insert a row for a freshly posted order. Idempotent on polymarket_order_id."""
    if not polymarket_order_id:
        return
    now = _utc_now_iso()
    c, own = _get_conn(conn)
    try:
        c.execute(
            """
            INSERT OR IGNORE INTO order_history (
                polymarket_order_id, bet_id, chunk_id, plan_id,
                token_id, side, order_kind, limit_price, size_usdc, size_shares,
                posted_at, final_status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'placed', ?, ?)
            """,
            (
                str(polymarket_order_id), bet_id, chunk_id, plan_id,
                str(token_id), side, order_kind, limit_price, size_usdc, size_shares,
                posted_at or now, now, now,
            ),
        )
        # If row already exists (e.g. backfill ran first) refresh placement metadata.
        c.execute(
            """
            UPDATE order_history
            SET bet_id = COALESCE(bet_id, ?),
                chunk_id = COALESCE(chunk_id, ?),
                plan_id = COALESCE(plan_id, ?),
                token_id = COALESCE(token_id, ?),
                limit_price = COALESCE(limit_price, ?),
                size_usdc = COALESCE(size_usdc, ?),
                size_shares = COALESCE(size_shares, ?),
                posted_at = COALESCE(posted_at, ?),
                updated_at = ?
            WHERE polymarket_order_id = ?
            """,
            (
                bet_id, chunk_id, plan_id, str(token_id),
                limit_price, size_usdc, size_shares,
                posted_at or now, now, str(polymarket_order_id),
            ),
        )
        c.commit()
    finally:
        if own:
            c.close()


@_safe
def record_order_filled(
    polymarket_order_id: str,
    *,
    fill_usdc: float,
    fill_price: Optional[float] = None,
    filled_at: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    """Mark an order as filled in history. Adds the row if it doesn't exist yet."""
    if not polymarket_order_id:
        return
    now = _utc_now_iso()
    c, own = _get_conn(conn)
    try:
        c.execute(
            """
            UPDATE order_history
            SET final_status = 'filled',
                fill_usdc = ?,
                fill_price = COALESCE(?, fill_price),
                filled_at = COALESCE(?, filled_at, ?),
                last_seen_at = ?,
                updated_at = ?
            WHERE polymarket_order_id = ?
            """,
            (
                float(fill_usdc or 0), fill_price,
                filled_at, now, now, now,
                str(polymarket_order_id),
            ),
        )
        c.commit()
    finally:
        if own:
            c.close()


@_safe
def record_order_cancelled(
    polymarket_order_id: str,
    *,
    reason: str,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    """Mark an order as cancelled with a reason tag. No-op for already-filled rows."""
    if not polymarket_order_id:
        return
    now = _utc_now_iso()
    c, own = _get_conn(conn)
    try:
        c.execute(
            """
            UPDATE order_history
            SET final_status = 'cancelled',
                final_reason = COALESCE(final_reason, ?),
                updated_at = ?
            WHERE polymarket_order_id = ?
              AND final_status NOT IN ('filled', 'reprice_replaced')
            """,
            (reason, now, str(polymarket_order_id)),
        )
        c.commit()
    finally:
        if own:
            c.close()


@_safe
def record_order_replaced_by_reprice(
    old_order_id: str,
    new_order_id: Optional[str],
    *,
    reason: str = "reprice",
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    """Old order is being replaced by a new posting at a different price.

    Old row is marked `reprice_replaced` and pointed at the new id so reconcile
    can follow the chain. The new id row should be inserted separately via
    `record_order_placed` (callers do that next).
    """
    if not old_order_id:
        return
    now = _utc_now_iso()
    c, own = _get_conn(conn)
    try:
        c.execute(
            """
            UPDATE order_history
            SET final_status = 'reprice_replaced',
                final_reason = COALESCE(final_reason, ?),
                replaced_by_order_id = ?,
                updated_at = ?
            WHERE polymarket_order_id = ?
              AND final_status != 'filled'
            """,
            (reason, new_order_id, now, str(old_order_id)),
        )
        c.commit()
    finally:
        if own:
            c.close()


@_safe
def record_order_error(
    polymarket_order_id: Optional[str],
    *,
    bet_id: Optional[int] = None,
    chunk_id: Optional[int] = None,
    plan_id: Optional[int] = None,
    token_id: Optional[str] = None,
    side: Optional[str] = None,
    order_kind: str = "fok",
    error_category: str = "unknown",
    error_message: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    """Record an order-placement error. Creates a history row even when no oid was returned."""
    now = _utc_now_iso()
    oid = str(polymarket_order_id or f"error:{bet_id or '?'}:{now}")
    c, own = _get_conn(conn)
    try:
        c.execute(
            """
            INSERT OR IGNORE INTO order_history (
                polymarket_order_id, bet_id, chunk_id, plan_id,
                token_id, side, order_kind,
                posted_at, final_status, final_reason, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'error', ?, ?, ?)
            """,
            (
                oid, bet_id, chunk_id, plan_id,
                token_id or "", side or "", order_kind,
                now, error_category, now, now,
            ),
        )
        if error_message:
            c.execute(
                "UPDATE order_history SET final_reason = ?, updated_at = ? WHERE polymarket_order_id = ?",
                (f"{error_category}: {str(error_message)[:240]}", now, oid),
            )
        c.commit()
    finally:
        if own:
            c.close()


def lookup_bet_for_order(
    polymarket_order_id: str,
    *,
    conn: Optional[sqlite3.Connection] = None,
    max_depth: int = 8,
) -> Optional[Dict[str, Any]]:
    """Return {'bet_id','chunk_id','plan_id','final_status', ...} for an order id.

    Follows the reprice chain via `replaced_by_order_id` up to `max_depth` hops
    so a fill that landed on a long-ago-repriced order id still resolves to the
    same originating bet/chunk.
    """
    if not polymarket_order_id:
        return None
    c, own = _get_conn(conn)
    try:
        oid = str(polymarket_order_id)
        seen: set = set()
        for _ in range(max_depth):
            if oid in seen:
                return None
            seen.add(oid)
            row = c.execute(
                """
                SELECT polymarket_order_id, bet_id, chunk_id, plan_id,
                       final_status, final_reason, replaced_by_order_id
                FROM order_history
                WHERE polymarket_order_id = ?
                """,
                (oid,),
            ).fetchone()
            if row is None:
                return None
            d = dict(row)
            replaced = d.get("replaced_by_order_id")
            if d.get("final_status") == "reprice_replaced" and replaced:
                oid = str(replaced)
                continue
            return d
        return None
    finally:
        if own:
            c.close()


def all_known_order_ids(conn: Optional[sqlite3.Connection] = None) -> set:
    """Every order id we have ever placed (history + live + ledger)."""
    c, own = _get_conn(conn)
    try:
        ids: set = set()
        try:
            ids.update(
                str(r[0]) for r in c.execute(
                    "SELECT polymarket_order_id FROM order_history WHERE polymarket_order_id IS NOT NULL"
                ).fetchall()
            )
        except sqlite3.OperationalError:
            pass
        ids.update(
            str(r[0]) for r in c.execute(
                "SELECT polymarket_order_id FROM order_chunks WHERE polymarket_order_id IS NOT NULL"
            ).fetchall()
        )
        ids.update(
            str(r[0]) for r in c.execute(
                "SELECT polymarket_order_id FROM bet_ledger WHERE polymarket_order_id IS NOT NULL"
            ).fetchall()
        )
        return ids
    finally:
        if own:
            c.close()
