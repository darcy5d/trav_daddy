"""Marketable SELL execution with real fill read-back.

Both the in-play cashout (`cashout.execute_cashout`) and the XI-aware
rebalancer (`bet_placement.reduce_position`) used to exit a position by
posting a GTC limit SELL *at the midpoint*. A sell at the midpoint sits above
the best bid, so it almost never crosses the book — only a tiny sliver fills
while the bulk rests unfilled (and is cancelled at resolution). The callers
then assumed a full fill and booked phantom proceeds.

`marketable_sell` fixes that root cause:

  1. Read the live order book and price the SELL to cross — a marketable
     limit at `reference_price - max_slippage_cents`, which fills against every
     bid at/above that floor (sweeping the book down to an acceptable price).
  2. Read back the ACTUAL matched size from the exchange (`get_order`), never
     assume the requested size filled.
  3. Cancel any resting remainder (confirmed) so we never leave an orphan
     order on the book.

The helper is pure execution: it places/reads/cancels orders and returns what
actually happened. All ledger/PnL booking stays in the callers, which book
only the `filled_shares` this returns.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Polymarket enforces a >= 5 share minimum per order.
POLYMARKET_MIN_SHARES = 5.0
DEFAULT_TICK = 0.01
_EPS = 1e-6


def sell_execution_config() -> Dict[str, Any]:
    """Read SELL-execution settings from BETTING_CONFIG (lazy import so .env
    changes are picked up per process and the module stays importable)."""
    try:
        from config import BETTING_CONFIG
    except Exception:  # pragma: no cover - config always present in practice
        return {
            "max_slippage_cents": 0.03,
            "cancel_remainder": True,
            "min_fill_shares": POLYMARKET_MIN_SHARES,
        }
    return {
        "max_slippage_cents": float(
            BETTING_CONFIG.get("cashout_sell_max_slippage_cents", 0.03)
        ),
        "cancel_remainder": bool(BETTING_CONFIG.get("cashout_sell_cancel_remainder", True)),
        "min_fill_shares": float(
            BETTING_CONFIG.get("cashout_sell_min_fill_shares", POLYMARKET_MIN_SHARES)
        ),
    }


def _f(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_down_tick(price: float, tick: float) -> float:
    if tick <= 0:
        tick = DEFAULT_TICK
    return (int(price / tick + _EPS)) * tick


def _round_down_size(shares: float, decimals: int = 2) -> float:
    factor = 10 ** decimals
    return int(shares * factor + _EPS) / factor


def _parse_book_bids(book: Any) -> List[Tuple[float, float]]:
    """Return bids as [(price, size), ...] sorted by price descending.

    Accepts the REST `/book` response shape (dict with `bids`) or an SDK
    OrderBookSummary-like object. Each level may be a dict {price,size} or an
    object with .price/.size.
    """
    raw = None
    if isinstance(book, dict):
        raw = book.get("bids")
    else:
        raw = getattr(book, "bids", None)
    if not raw:
        return []

    out: List[Tuple[float, float]] = []
    for level in raw:
        if isinstance(level, dict):
            p = _f(level.get("price"))
            s = _f(level.get("size"))
        else:
            p = _f(getattr(level, "price", None))
            s = _f(getattr(level, "size", None))
        if p is not None and s is not None and s > 0:
            out.append((p, s))
    out.sort(key=lambda x: x[0], reverse=True)
    return out


def _book_tick(book: Any) -> float:
    tick = None
    if isinstance(book, dict):
        tick = _f(book.get("tick_size") or book.get("tickSize"))
    else:
        tick = _f(getattr(book, "tick_size", None))
    return tick if tick and tick > 0 else DEFAULT_TICK


def _vwap_for_shares(
    bids_desc: List[Tuple[float, float]],
    shares: float,
    floor_price: float,
) -> Tuple[float, float]:
    """Volume-weighted average price obtained by selling `shares` into the
    book, consuming bids best-first down to `floor_price`.

    Returns (shares_priced, vwap). Any shares beyond the visible depth at/above
    the floor are priced at `floor_price` (a conservative lower bound, since
    every real fill executed at >= floor_price)."""
    remaining = shares
    cost = 0.0
    priced = 0.0
    for price, size in bids_desc:
        if remaining <= _EPS:
            break
        if price < floor_price - _EPS:
            break
        take = min(remaining, size)
        cost += take * price
        priced += take
        remaining -= take
    if remaining > _EPS:
        # Beyond visible depth — price the rest at the floor (lower bound).
        cost += remaining * floor_price
        priced += remaining
    if priced <= _EPS:
        return 0.0, floor_price
    return priced, cost / priced


def _extract_order_id(resp: Any) -> Optional[str]:
    if not isinstance(resp, dict):
        return None
    oid = resp.get("orderID") or resp.get("orderId") or resp.get("order_id") or resp.get("id")
    return str(oid) if oid else None


def _read_fill(
    poly_client: Any,
    order_id: str,
    requested_shares: float,
    attempts: int,
    sleep_s: float,
) -> Tuple[Optional[float], Optional[str], Any]:
    """Poll get_order until the matched size stabilises or a terminal status.

    Returns (size_matched, status, raw_order). size_matched is None when the
    exchange has no record we could read (caller falls back to the post
    response)."""
    last: Tuple[Optional[float], Optional[str], Any] = (None, None, None)
    for i in range(max(1, attempts)):
        order = None
        try:
            order = poly_client.get_order(order_id)
        except Exception as exc:  # API flakiness — keep trying
            logger.debug(f"get_order({order_id[:12]}) raised: {exc}")
            order = None
        if order:
            sm = _f(order.get("size_matched") if isinstance(order, dict) else None)
            if sm is None and isinstance(order, dict):
                sm = _f(order.get("sizeMatched"))
            status = ""
            if isinstance(order, dict):
                status = str(order.get("status") or "").lower()
            last = (sm, status, order)
            terminal = status in ("matched", "filled", "canceled", "cancelled")
            if terminal or (sm is not None and sm >= requested_shares - _EPS):
                return last
        if i < attempts - 1:
            time.sleep(sleep_s)
    return last


def _confirm_cancel(poly_client: Any, order_id: str) -> bool:
    """Cancel an order and confirm it is gone (mirrors the TWAP daemon's
    cancel-confirm guard). Returns True only when the exchange acknowledges."""
    try:
        resp = poly_client.cancel_order(order_id)
    except Exception as exc:
        logger.warning(f"cancel_order({order_id[:12]}) raised: {exc}")
        resp = None

    if isinstance(resp, dict):
        canceled = resp.get("canceled") or resp.get("cancelled") or []
        if order_id in canceled:
            return True
        not_canceled = resp.get("not_canceled") or resp.get("notCanceled") or {}
        if isinstance(not_canceled, dict) and order_id in not_canceled:
            return False

    # Fall back to a read-back: gone or status cancelled == confirmed.
    try:
        order = poly_client.get_order(order_id)
    except Exception:
        order = None
    if order is None:
        return True
    if isinstance(order, dict):
        return str(order.get("status") or "").lower() in ("canceled", "cancelled")
    return False


def marketable_sell(
    poly_client: Any,
    token_id: str,
    shares: float,
    *,
    reference_price: float,
    max_slippage_cents: Optional[float] = None,
    cancel_remainder: Optional[bool] = None,
    min_fill_shares: Optional[float] = None,
    liquidate: bool = False,
    min_exit_price: Optional[float] = None,
    poll_attempts: int = 3,
    poll_sleep_s: float = 1.0,
) -> Dict[str, Any]:
    """Sell `shares` of `token_id` with a marketable limit and read back the
    real fill.

    Args:
        reference_price: Current midpoint (or other reference); the SELL floor
            is `reference_price - max_slippage_cents`.
        max_slippage_cents: Max cents below the reference we will accept on the
            exit. Defaults to the configured value.
        cancel_remainder: Cancel any resting unfilled remainder. Default config.
        min_fill_shares: Below this, treat as no-fill (dust / sub-minimum).
        liquidate: Progressive forced exit (stop-loss). Ignores the tight
            slippage floor and the "best-bid-below-floor" hold guard: prices the
            SELL at `min_exit_price` (a hard ruin floor) so it sweeps EVERY
            resting bid at/above that floor in one marketable order, taking
            whatever liquidity exists rather than waiting for a price that
            never returns on a collapsing book. Partial fills leave the
            remainder open for the next scan tick to walk down further.
        min_exit_price: Hard floor for `liquidate` mode (absolute lowest price
            we will accept). Defaults to one tick.

    Returns dict:
        success (bool — True iff filled_shares > 0),
        filled_shares, avg_fill_price, proceeds_usdc,
        requested_shares, limit_price, order_id, status,
        remainder_cancelled (bool), reason (on failure), order_response.
    """
    cfg = sell_execution_config()
    if max_slippage_cents is None:
        max_slippage_cents = cfg["max_slippage_cents"]
    if cancel_remainder is None:
        cancel_remainder = cfg["cancel_remainder"]
    if min_fill_shares is None:
        min_fill_shares = cfg["min_fill_shares"]

    base = {
        "success": False,
        "filled_shares": 0.0,
        "avg_fill_price": 0.0,
        "proceeds_usdc": 0.0,
        "requested_shares": shares,
        "limit_price": None,
        "order_id": None,
        "status": None,
        "remainder_cancelled": False,
        "order_response": None,
    }

    shares = _round_down_size(float(shares))
    if shares < min_fill_shares or shares < POLYMARKET_MIN_SHARES:
        return {**base, "reason": "below-min-shares"}
    if reference_price <= 0 or reference_price >= 1.0:
        return {**base, "reason": "invalid-reference-price"}

    # --- Read the book to price a marketable SELL ---
    try:
        book = poly_client.get_clob_order_book(token_id)
    except Exception as exc:
        logger.warning(f"marketable_sell: book fetch failed for {token_id[:12]}: {exc}")
        return {**base, "reason": f"book-fetch-failed: {exc}"}

    bids_desc = _parse_book_bids(book)
    if not bids_desc:
        return {**base, "reason": "no-bids"}

    tick = _book_tick(book)
    best_bid = bids_desc[0][0]
    if liquidate:
        # Forced exit: drop the limit to a hard ruin floor so the marketable
        # order sweeps every resting bid at/above it. We accept whatever the
        # book offers down to this price rather than holding for a tight slip.
        hard_floor = min_exit_price if min_exit_price is not None else tick
        floor_price = max(tick, _round_down_tick(float(hard_floor), tick))
    else:
        floor_price = max(tick, _round_down_tick(reference_price - float(max_slippage_cents), tick))
    floor_price = min(floor_price, _round_down_tick(1.0 - tick, tick))

    # If even the best bid is below our acceptable floor, the spread is too wide
    # to exit without eating more than max_slippage — hold and retry next tick.
    # In liquidate mode the floor is the ruin floor itself, so any visible bid
    # is acceptable and we never hold (the book sweep takes what's there).
    if not liquidate and best_bid < floor_price - _EPS:
        return {
            **base,
            "limit_price": floor_price,
            "reason": "best-bid-below-floor",
        }

    # --- Place the marketable SELL (limit at the floor; sweeps bids >= floor) ---
    try:
        resp = poly_client.place_limit_order(
            token_id=token_id,
            side="SELL",
            price=round(floor_price, 4),
            size_shares=shares,
        )
    except Exception as exc:
        logger.error(f"marketable_sell: SELL placement failed for {token_id[:12]}: {exc}")
        return {**base, "limit_price": floor_price, "reason": f"sell-failed: {exc}"}

    order_id = _extract_order_id(resp)
    resp_status = (resp.get("status") if isinstance(resp, dict) else "") or ""
    resp_status = str(resp_status).lower()

    if not order_id:
        # No id back: if the exchange reported a full match anyway, treat as
        # filled at our floor; otherwise treat as no-fill (don't book phantom).
        if resp_status == "matched":
            _, vwap = _vwap_for_shares(bids_desc, shares, floor_price)
            return {
                **base,
                "success": True,
                "filled_shares": shares,
                "avg_fill_price": round(vwap, 6),
                "proceeds_usdc": round(shares * vwap, 6),
                "limit_price": floor_price,
                "status": resp_status,
                "order_response": resp,
            }
        return {**base, "limit_price": floor_price, "status": resp_status,
                "order_response": resp, "reason": "no-order-id"}

    # --- Read back the ACTUAL matched size ---
    size_matched, status, _raw = _read_fill(
        poly_client, order_id, shares, poll_attempts, poll_sleep_s
    )
    if status:
        resp_status = status

    if size_matched is None:
        # Couldn't read the order back. Fall back to the post-response status:
        # 'matched' => assume full; anything else => assume unfilled (safe).
        filled = shares if resp_status == "matched" else 0.0
    else:
        filled = max(0.0, min(size_matched, shares))

    remainder = shares - filled
    remainder_cancelled = False
    if remainder > _EPS and cancel_remainder and resp_status not in ("canceled", "cancelled"):
        remainder_cancelled = _confirm_cancel(poly_client, order_id)

    if filled < min_fill_shares:
        return {
            **base,
            "limit_price": floor_price,
            "order_id": order_id,
            "status": resp_status,
            "remainder_cancelled": remainder_cancelled,
            "order_response": resp,
            "reason": "no-meaningful-fill",
        }

    _, vwap = _vwap_for_shares(bids_desc, filled, floor_price)
    proceeds = filled * vwap

    return {
        "success": True,
        "filled_shares": round(filled, 6),
        "avg_fill_price": round(vwap, 6),
        "proceeds_usdc": round(proceeds, 6),
        "requested_shares": shares,
        "limit_price": floor_price,
        "order_id": order_id,
        "status": resp_status,
        "remainder_cancelled": remainder_cancelled,
        "order_response": resp,
    }
