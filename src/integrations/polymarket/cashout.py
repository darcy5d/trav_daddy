"""Wave 5.10: In-game cashout — evaluate and execute mid-match SELL orders.

A "cashout" is triggered when the current Polymarket mid-price has risen
enough relative to our fill price that locking in the profit is preferable
to holding the position to binary settlement.

Trigger rule:
    return_ratio = current_price / fill_price
    cashout if: return_ratio >= threshold_for_fill_price(fill_price)

Thresholds are tiered by entry price (optimised on 14-day backtest + Wave 5.11 update):
    Heavy underdog  (5–20¢):  1.30x  — spikes are brief; snap at 1.3x
    Underdog       (20–35¢):  1.20x  — peak PnL; falls off sharply above
    Slight underdog(35–50¢):  1.25x  — sweet spot between capture and reversal
    Coin flip      (50–65¢):  1.30x  — loose safety net; fires on genuine mid-match spikes
    Slight favourite(65–80¢): hold   — rarely moves enough to benefit
    Heavy favourite(80–95¢):  hold   — never moves; ceiling too close

Once the threshold is met, a SELL limit order is placed at the current CLOB
midpoint.  For paper bets the CLOB call is skipped and the result is
simulated in-DB only.

P&L accounting (mirrors reconcile._compute_pnl_for_settled_bet):
    shares      = fill_size_usdc / fill_price
    gross       = shares * cashout_price
    fee         = gross * POLYMARKET_TAKER_FEE   (2% taker on the SELL)
    cashout_pnl = gross - fill_size_usdc - fee

The cashed-out bet_ledger row is immediately marked status='settled' so the
reconciler skips it.  It can be distinguished from a naturally-settled row by:
    WHERE cashout_triggered_at IS NOT NULL

Wave 5.11 — guarded stop-loss (loss mitigation):
    A separate SELL trigger fires when a losing position's price has *fallen*
    to a floor, but only after a 2nd-innings time-gate (minutes from kickoff).
    Ships OFF; only this gated, deep-floor config was net-positive in the
    35-day backtest (no re-entry). Layered on top of the profit-take so the
    scanner evaluates both per bet.

Public API:
    tiered_cashout_threshold(fill_price) -> Optional[float]
    evaluate_cashout(bet_row, current_price, threshold) -> bool
    stop_loss_config() -> dict
    evaluate_stop_loss(bet_row, current_price, cfg, now) -> bool
    execute_cashout(bet_row, cashout_price, conn, poly_client, dry_run, reason) -> dict
    scan_for_cashouts(conn, poly_client, dry_run) -> dict
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.integrations.polymarket import PolymarketClient
from src.integrations.polymarket.sell_execution import marketable_sell
from src.integrations.odds.polymarket_compare import POLYMARKET_TAKER_FEE

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso_ts(iso: Optional[str]) -> Optional[datetime]:
    """Parse an ISO8601 string to an aware UTC datetime; None on failure."""
    if not iso:
        return None
    try:
        d = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Tiered threshold lookup  (optimised on 14-day, 388-bet backtest)
# ---------------------------------------------------------------------------

# Each entry: (upper_bound_exclusive, threshold_or_None)
# Evaluated top-to-bottom; first matching bucket wins.
# None means "hold to settlement — no cashout for this bucket".
_TIERED_THRESHOLDS: list = [
    (0.20, 1.30),   # Heavy underdog  5–20¢ : snap at 1.30x
    (0.35, 1.20),   # Underdog       20–35¢ : peak PnL at 1.20x
    (0.50, 1.25),   # Slight underdog35–50¢ : sweet spot 1.25x
    (0.65, 1.30),   # Coin flip      50–65¢ : loose safety net; fires on genuine mid-match spike
    (0.80, None),   # Slight fav     65–80¢ : hold — rarely moves enough
    (0.95, None),   # Heavy fav      80–95¢ : hold — ceiling too close
]


def tiered_cashout_threshold(fill_price: float) -> Optional[float]:
    """Return the cashout return-ratio threshold for a given entry price.

    Returns None for buckets where holding is preferable to any cashout
    (coin flips and favourites).

    Examples:
        tiered_cashout_threshold(0.15) -> 1.30   # heavy underdog
        tiered_cashout_threshold(0.28) -> 1.20   # underdog
        tiered_cashout_threshold(0.42) -> 1.25   # slight underdog
        tiered_cashout_threshold(0.55) -> 1.30  # coin flip — loose safety net
        tiered_cashout_threshold(0.70) -> None   # slight favourite — hold
    """
    for upper, threshold in _TIERED_THRESHOLDS:
        if fill_price < upper:
            return threshold
    return None  # above 0.95 — hold


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_cashout(
    bet_row: sqlite3.Row,
    current_price: float,
    threshold: float,
) -> bool:
    """Return True if the position should be cashed out at current_price.

    Args:
        bet_row:       A bet_ledger row (sqlite3.Row or dict-like).
        current_price: Current CLOB midpoint for the outcome token [0, 1].
        threshold:     Minimum return_ratio (current_price / fill_price) to
                       trigger a cashout.
    """
    fill_price = bet_row["fill_price"]
    if not fill_price or fill_price <= 0:
        return False
    if current_price <= 0 or current_price >= 1.0:
        return False
    return (current_price / fill_price) >= threshold


# ---------------------------------------------------------------------------
# Guarded stop-loss (Wave 5.11)
# ---------------------------------------------------------------------------

def stop_loss_config() -> Dict[str, Any]:
    """Read the live stop-loss settings from BETTING_CONFIG.

    Returns a dict with: enabled (bool), floor (float), gate_min (float).
    Imported lazily so this module stays importable without config side
    effects and so .env changes are picked up per process.
    """
    try:
        from config import BETTING_CONFIG
    except Exception:  # pragma: no cover - config always present in practice
        return {"enabled": False, "floor": 0.20, "gate_min": 105.0, "min_exit_price": 0.01}
    return {
        "enabled": bool(BETTING_CONFIG.get("stop_loss_enabled", False)),
        "floor": float(BETTING_CONFIG.get("stop_loss_floor", 0.20)),
        "gate_min": float(BETTING_CONFIG.get("stop_loss_gate_min", 105.0)),
        "min_exit_price": float(BETTING_CONFIG.get("stop_loss_min_exit_price", 0.01)),
    }


def evaluate_stop_loss(
    bet_row: sqlite3.Row,
    current_price: float,
    cfg: Dict[str, Any],
    now: Optional[datetime] = None,
) -> bool:
    """Return True if the position should be stopped out at current_price.

    Guarded: fires only when the stop-loss is enabled, the price has fallen
    to/through the floor, and at least gate_min minutes have elapsed since
    kickoff (2nd-innings proxy). A missing/unparseable kickoff_at means we
    cannot place the time-gate, so we hold — matching the backtest fallback.

    Args:
        bet_row:       A bet_ledger row (sqlite3.Row or dict-like).
        current_price: Current CLOB midpoint for the outcome token [0, 1].
        cfg:           Output of stop_loss_config().
        now:           Override for the current time (tests); defaults to UTC now.
    """
    if not cfg.get("enabled"):
        return False
    if current_price <= 0 or current_price >= 1.0:
        return False
    if current_price > cfg["floor"]:
        return False

    kickoff = _parse_iso_ts(bet_row["kickoff_at"] if "kickoff_at" in bet_row.keys() else None)
    if kickoff is None:
        return False  # no anchor for the time-gate -> hold
    now = now or datetime.now(timezone.utc)
    minutes_since_kickoff = (now - kickoff).total_seconds() / 60.0
    return minutes_since_kickoff >= cfg["gate_min"]


def _compute_cashout_pnl(
    fill_price: float,
    fill_size_usdc: float,
    cashout_price: float,
) -> float:
    """Net USD P&L from selling shares at cashout_price (signed)."""
    shares = fill_size_usdc / fill_price
    gross = shares * cashout_price
    fee = gross * POLYMARKET_TAKER_FEE
    return round(gross - fill_size_usdc - fee, 4)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _record_sell_audit(
    order_id: Optional[str],
    *,
    bet_id: Optional[int],
    token_id: str,
    reason: str,
    avg_price: float,
    proceeds: float,
    shares: float,
    conn: sqlite3.Connection,
) -> None:
    """Append the cashout SELL to the order-history audit log (best-effort)."""
    if not order_id:
        return
    try:
        from src.integrations.polymarket.order_audit import (
            record_order_filled, record_order_placed,
        )
        record_order_placed(
            order_id,
            bet_id=bet_id,
            token_id=token_id,
            side="SELL",
            order_kind="cashout" if reason == "profit" else "stop_loss",
            limit_price=avg_price,
            size_usdc=proceeds,
            size_shares=shares,
            conn=conn,
        )
        record_order_filled(
            order_id,
            fill_usdc=proceeds,
            fill_price=avg_price,
            conn=conn,
        )
    except Exception as exc:
        logger.warning(f"Audit record_order failed for cashout {order_id}: {exc}")


def _insert_partial_cashout_sell_row(
    bet_row: sqlite3.Row,
    *,
    avg_price: float,
    entry_cost_sold: float,
    proceeds: float,
    fee: float,
    cashout_pnl: float,
    return_ratio: float,
    cashout_oid: Optional[str],
    reason: str,
    now: str,
    conn: sqlite3.Connection,
) -> int:
    """Insert a settled SELL adjustment row for a PARTIAL cashout fill.

    The realized PnL of the filled tranche lives on this row; the originating
    BUY row stays `filled` with its open stake decremented so the next scan
    tick retries the remainder. Cashout SELL rows are identified by
    side='SELL' AND cashout_triggered_at IS NOT NULL (rebalance sells leave
    cashout_triggered_at NULL).
    """
    def _g(key: str, default: Any = None) -> Any:
        return bet_row[key] if key in bet_row.keys() else default

    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO bet_ledger (
            proposed_at, placed_at, filled_at, settled_at,
            fixture_key, market_type,
            polymarket_market_id, polymarket_token_id,
            side_label, model_prob, market_price_at_proposal, edge_pp,
            side, size_usdc, fees_estimated_usdc,
            fill_price, fill_size_usdc,
            status, mode, bet_kind, strategy_label, phase,
            pnl_realised_usdc,
            cashout_triggered_at, cashout_price, cashout_pnl_usdc,
            cashout_threshold_used, cashout_order_id, cashout_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'SELL', ?, ?, ?, ?,
                  'settled', 'auto', 'real', ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            now, now, now, now,
            _g("fixture_key"), _g("market_type", "moneyline"),
            _g("polymarket_market_id", ""), _g("polymarket_token_id"),
            _g("side_label"), avg_price, avg_price, 0.0,
            round(entry_cost_sold, 4), round(fee, 4),
            avg_price, round(entry_cost_sold, 6),
            _g("strategy_label"), _g("phase"),
            cashout_pnl,
            now, avg_price, cashout_pnl,
            return_ratio, cashout_oid, reason,
        ),
    )
    return int(cur.lastrowid)


def execute_cashout(
    bet_row: sqlite3.Row,
    cashout_price: float,
    conn: sqlite3.Connection,
    poly_client: Optional[PolymarketClient] = None,
    dry_run: bool = False,
    reason: str = "profit",
) -> Dict[str, Any]:
    """Sell the position and update bet_ledger, booking ONLY what actually fills.

    Real bets are exited with a marketable SELL (`marketable_sell`) priced to
    cross the book; the matched size is read back from the exchange. We never
    assume the requested size filled:
      * Zero / dust fill  -> nothing booked, row stays 'filled' (retry next tick).
      * Partial fill       -> a settled SELL adjustment row carries the realized
                              PnL; the BUY row's open stake is decremented and it
                              stays 'filled' so the remainder is retried.
      * Full fill          -> the BUY row is marked 'settled' with the actual
                              avg fill price and PnL.

    For paper bets (bet_kind='paper') or dry_run=True the CLOB is skipped and a
    full fill at `cashout_price` is simulated in-DB only.

    Args:
        cashout_price: Current CLOB midpoint — the reference for the marketable
                       SELL floor (real) or the simulated fill price (paper).
        reason: 'profit' for a tiered profit-take, 'stop' for a guarded
                stop-loss. Recorded in bet_ledger.cashout_reason.

    Returns a result dict with keys:
        success, bet_id, cashout_price, cashout_pnl, return_ratio,
        is_simulated, reason, filled_shares, partial, order_response
    """
    bet_id = bet_row["bet_id"]
    fill_price = bet_row["fill_price"]
    fill_size_usdc = bet_row["fill_size_usdc"]
    token_id = bet_row["polymarket_token_id"]
    bet_kind = (bet_row["bet_kind"] if "bet_kind" in bet_row.keys() else None) or "real"
    strategy_label = (bet_row["strategy_label"] if "strategy_label" in bet_row.keys() else None)

    if not fill_price or not fill_size_usdc:
        return {
            "success": False,
            "bet_id": bet_id,
            "error": "fill_price or fill_size_usdc is NULL — cannot cashout",
        }

    total_shares = fill_size_usdc / fill_price
    now = _utc_now_iso()
    is_simulated = dry_run or (bet_kind == "paper")

    # --- Determine the ACTUAL fill ---
    order_response: Optional[Dict[str, Any]] = None
    cashout_oid: Optional[str] = None
    if is_simulated:
        # Paper / dry-run: simulate a full fill at the current midpoint.
        filled_shares = total_shares
        avg_price = float(cashout_price)
        proceeds = total_shares * avg_price
        logger.info(
            f"Cashout SIMULATED: bet_id={bet_id} side={bet_kind} "
            f"return_ratio={avg_price / fill_price:.3f}x"
        )
    else:
        if poly_client is None:
            poly_client = PolymarketClient()
        # A stop-loss is a forced exit: liquidate progressively into whatever
        # bids exist down to the ruin floor instead of holding for the tight
        # profit-take slippage (which rotted collapsing positions to ~0).
        is_stop = reason == "stop"
        stop_min_exit = stop_loss_config().get("min_exit_price", 0.01) if is_stop else None
        sell = marketable_sell(
            poly_client,
            token_id,
            total_shares,
            reference_price=float(cashout_price),
            liquidate=is_stop,
            min_exit_price=stop_min_exit,
        )
        if not sell.get("success"):
            logger.info(
                f"Cashout SELL did not fill for bet_id={bet_id}: "
                f"{sell.get('reason')} (holding, will retry)"
            )
            return {
                "success": False,
                "bet_id": bet_id,
                "error": f"sell-not-filled: {sell.get('reason')}",
                "is_simulated": False,
                "filled_shares": 0.0,
            }
        filled_shares = float(sell["filled_shares"])
        avg_price = float(sell["avg_fill_price"])
        proceeds = float(sell["proceeds_usdc"])
        order_response = sell.get("order_response")
        cashout_oid = sell.get("order_id")
        logger.info(
            f"Cashout SELL filled: bet_id={bet_id} shares={filled_shares:.4f}/"
            f"{total_shares:.4f} @ {avg_price:.4f} (remainder_cancelled="
            f"{sell.get('remainder_cancelled')})"
        )

    # --- Book only the filled tranche ---
    filled_fraction = min(1.0, filled_shares / total_shares) if total_shares > 0 else 0.0
    entry_cost_sold = fill_size_usdc * filled_fraction
    fee = proceeds * POLYMARKET_TAKER_FEE
    cashout_pnl = round(proceeds - entry_cost_sold - fee, 4)
    return_ratio = avg_price / fill_price if fill_price else 0.0
    is_full = filled_fraction >= 1.0 - 1e-6

    if is_full:
        # --- Full exit: settle the BUY row with the real fill ---
        bankroll_after = None
        if bet_kind == "paper" and strategy_label:
            try:
                from src.integrations.polymarket.paper_strategies import get_strategy
                cur2 = conn.cursor()
                cur2.execute(
                    """
                    SELECT COALESCE(SUM(pnl_realised_usdc), 0.0)
                    FROM bet_ledger
                    WHERE bet_kind = 'paper'
                      AND strategy_label = ?
                      AND status = 'settled'
                      AND bet_id != ?
                    """,
                    (strategy_label, bet_id),
                )
                prior_pnl = float(cur2.fetchone()[0] or 0.0)
                strat = get_strategy(strategy_label)
                starting = strat.starting_bankroll_usdc if strat else 1000.0
                bankroll_after = starting + prior_pnl + float(cashout_pnl)
            except Exception as exc:
                logger.warning(f"Could not compute bankroll_after for cashout: {exc}")

        _record_sell_audit(
            cashout_oid, bet_id=bet_id, token_id=token_id, reason=reason,
            avg_price=avg_price, proceeds=proceeds, shares=filled_shares, conn=conn,
        )

        conn.execute(
            """
            UPDATE bet_ledger
            SET status                 = 'settled',
                settled_at             = ?,
                settle_outcome         = NULL,
                fill_size_usdc         = 0,
                pnl_realised_usdc      = ?,
                bankroll_after_settle  = COALESCE(?, bankroll_after_settle),
                cashout_triggered_at   = ?,
                cashout_price          = ?,
                cashout_pnl_usdc       = ?,
                cashout_threshold_used = ?,
                cashout_order_id       = ?,
                cashout_reason         = ?
            WHERE bet_id = ?
            """,
            (
                now, cashout_pnl, bankroll_after,
                now, avg_price, cashout_pnl, return_ratio,
                cashout_oid, reason,
                bet_id,
            ),
        )
        conn.commit()
        partial = False
    else:
        # --- Partial exit (real only): book the tranche on a SELL row, keep
        #     the BUY row open with its stake decremented for the next retry ---
        _record_sell_audit(
            cashout_oid, bet_id=bet_id, token_id=token_id, reason=reason,
            avg_price=avg_price, proceeds=proceeds, shares=filled_shares, conn=conn,
        )
        _insert_partial_cashout_sell_row(
            bet_row,
            avg_price=avg_price, entry_cost_sold=entry_cost_sold,
            proceeds=proceeds, fee=fee, cashout_pnl=cashout_pnl,
            return_ratio=return_ratio, cashout_oid=cashout_oid,
            reason=reason, now=now, conn=conn,
        )
        new_fill = round(float(fill_size_usdc) - entry_cost_sold, 6)
        conn.execute(
            "UPDATE bet_ledger SET fill_size_usdc = ? WHERE bet_id = ?",
            (max(0.0, new_fill), bet_id),
        )
        conn.commit()
        partial = True

    return {
        "success": True,
        "bet_id": bet_id,
        "cashout_price": avg_price,
        "cashout_pnl": cashout_pnl,
        "return_ratio": return_ratio,
        "is_simulated": is_simulated,
        "reason": reason,
        "filled_shares": round(filled_shares, 6),
        "partial": partial,
        "order_response": order_response,
    }


# ---------------------------------------------------------------------------
# Scanner — iterate all open positions
# ---------------------------------------------------------------------------

def scan_for_cashouts(
    conn: Optional[sqlite3.Connection] = None,
    poly_client: Optional[PolymarketClient] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Check all filled bets for cashout / stop-loss eligibility and execute.

    For each bet_ledger row with status='filled':
      1. Profit-take: threshold via tiered_cashout_threshold(fill_price).
         Buckets where holding is optimal (coin flip / favourites) return None.
      2. Fetch the current CLOB midpoint for the outcome token (needed for
         either trigger). Skipped only when profit-take holds AND stop-loss
         is disabled.
      3. Profit-take fires if evaluate_cashout() is True (reason='profit').
         Otherwise, if the guarded stop-loss is enabled and evaluate_stop_loss()
         is True, the position is stopped out (reason='stop'). The stop-loss is
         evaluated across ALL price buckets, including favourites that collapse.

    Args:
        conn:        Open SQLite connection (created if None).
        poly_client: PolymarketClient (created if None).
        dry_run:     If True, simulate cashouts without placing SELL orders.

    Returns:
        {
            "n_checked":   int,
            "n_triggered": int,
            "n_executed":  int,
            "n_stops":     int,   # subset of n_executed that were stop-losses
            "cashouts":    [result_dict, ...],
            "errors":      [(bet_id, message), ...],
        }
    """
    own_conn = False
    if conn is None:
        from src.data.database import get_connection
        conn = get_connection()
        own_conn = True

    if poly_client is None:
        poly_client = PolymarketClient()

    cur = conn.cursor()
    cur.execute(
        """
        SELECT *
        FROM bet_ledger
        WHERE status = 'filled'
          AND fill_price IS NOT NULL
          AND fill_size_usdc IS NOT NULL
          AND polymarket_token_id IS NOT NULL
        ORDER BY filled_at ASC
        """
    )
    rows = cur.fetchall()

    n_checked = len(rows)
    n_triggered = 0
    n_executed = 0
    n_stops = 0
    cashouts: list = []
    errors: list = []

    stop_cfg = stop_loss_config()
    stop_enabled = bool(stop_cfg.get("enabled"))
    now = datetime.now(timezone.utc)

    for bet in rows:
        bet_id = bet["bet_id"]
        token_id = bet["polymarket_token_id"]
        fill_price = float(bet["fill_price"])

        # Tiered profit-take threshold — None for coin-flip / favourite buckets.
        threshold = tiered_cashout_threshold(fill_price)

        # Nothing to do for this bet if profit-take holds AND stop-loss is off.
        if threshold is None and not stop_enabled:
            continue

        # Fetch current price (needed for either trigger).
        try:
            mid_resp = poly_client.get_clob_midpoint(token_id)
            current_price_raw = (
                mid_resp.get("mid") or mid_resp.get("midpoint")
                if isinstance(mid_resp, dict) else None
            )
            if current_price_raw is None:
                logger.debug(f"  bet_id={bet_id}: no midpoint in response {mid_resp!r}")
                continue
            current_price = float(current_price_raw)
        except Exception as exc:
            errors.append((bet_id, f"midpoint fetch failed: {exc}"))
            continue

        # Decide which trigger (if any) fires. Profit-take takes precedence;
        # the two are mutually exclusive by price (a ratio >= ~1.2 vs a price
        # at/under the floor) so ordering is safe either way.
        reason: Optional[str] = None
        if threshold is not None and evaluate_cashout(bet, current_price, threshold):
            reason = "profit"
        elif stop_enabled and evaluate_stop_loss(bet, current_price, stop_cfg, now):
            reason = "stop"

        if reason is None:
            ratio = current_price / fill_price if fill_price else 0
            logger.debug(
                f"  bet_id={bet_id}: ratio={ratio:.3f}x — hold "
                f"(profit thr={threshold}, stop floor={stop_cfg['floor'] if stop_enabled else 'off'})"
            )
            continue

        n_triggered += 1

        # Execute
        result = execute_cashout(
            bet_row=bet,
            cashout_price=current_price,
            conn=conn,
            poly_client=poly_client,
            dry_run=dry_run,
            reason=reason,
        )
        cashouts.append(result)
        if result.get("success"):
            n_executed += 1
            if reason == "stop":
                n_stops += 1
            bet_kind_val = bet["bet_kind"] if "bet_kind" in bet.keys() else "real"
            tag = "[DRY-RUN]" if dry_run else ("[PAPER]" if bet_kind_val == "paper" else "[REAL]")
            side_label = bet["side_label"] if "side_label" in bet.keys() else "?"
            label = "STOP-LOSS" if reason == "stop" else "CASHOUT"
            logger.info(
                f"{label} {tag} bet_id={bet_id} "
                f"side={side_label} "
                f"return={result['return_ratio']:.2f}x "
                f"pnl=${result['cashout_pnl']:+.2f}"
            )
        else:
            errors.append((bet_id, result.get("error", "unknown error")))

    if own_conn:
        conn.close()

    return {
        "n_checked":   n_checked,
        "n_triggered": n_triggered,
        "n_executed":  n_executed,
        "n_stops":     n_stops,
        "cashouts":    cashouts,
        "errors":      errors,
    }
