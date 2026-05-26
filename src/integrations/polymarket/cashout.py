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

Public API:
    tiered_cashout_threshold(fill_price) -> Optional[float]
    evaluate_cashout(bet_row, current_price, threshold) -> bool
    execute_cashout(bet_row, cashout_price, conn, poly_client, dry_run) -> dict
    scan_for_cashouts(conn, poly_client, dry_run) -> dict
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.integrations.polymarket import PolymarketClient
from src.integrations.odds.polymarket_compare import POLYMARKET_TAKER_FEE

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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

def execute_cashout(
    bet_row: sqlite3.Row,
    cashout_price: float,
    conn: sqlite3.Connection,
    poly_client: Optional[PolymarketClient] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Sell the position at cashout_price and update bet_ledger.

    For paper bets (bet_kind='paper') or dry_run=True the CLOB call is
    skipped and only the DB is updated.

    Returns a result dict with keys:
        success, bet_id, cashout_price, cashout_pnl, return_ratio,
        is_simulated, order_response (None if simulated)
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

    shares_to_sell = fill_size_usdc / fill_price
    cashout_pnl = _compute_cashout_pnl(fill_price, fill_size_usdc, cashout_price)
    return_ratio = cashout_price / fill_price
    now = _utc_now_iso()
    is_simulated = dry_run or (bet_kind == "paper")

    # --- Place SELL order (real bets only) ---
    order_response: Optional[Dict[str, Any]] = None
    if not is_simulated:
        if poly_client is None:
            poly_client = PolymarketClient()
        try:
            order_response = poly_client.place_limit_order(
                token_id=token_id,
                side="SELL",
                price=cashout_price,
                size_shares=shares_to_sell,
            )
            logger.info(
                f"Cashout SELL placed: bet_id={bet_id} token={token_id} "
                f"shares={shares_to_sell:.4f} @ {cashout_price:.4f} "
                f"pnl=${cashout_pnl:+.2f}"
            )
        except Exception as exc:
            logger.error(f"Cashout SELL order failed for bet_id={bet_id}: {exc}")
            return {
                "success": False,
                "bet_id": bet_id,
                "error": str(exc),
            }
    else:
        logger.info(
            f"Cashout SIMULATED: bet_id={bet_id} side={bet_kind} "
            f"return_ratio={return_ratio:.3f}x pnl=${cashout_pnl:+.2f}"
        )

    # --- Bankroll update for paper bets ---
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

    cashout_oid = None
    if order_response:
        cashout_oid = order_response.get("orderID") or order_response.get("orderId")
        if cashout_oid:
            try:
                from src.integrations.polymarket.order_audit import (
                    record_order_filled, record_order_placed,
                )
                record_order_placed(
                    cashout_oid,
                    bet_id=bet_id,
                    token_id=token_id,
                    side="SELL",
                    order_kind="cashout",
                    limit_price=cashout_price,
                    size_usdc=cashout_price * shares_to_sell,
                    size_shares=shares_to_sell,
                    conn=conn,
                )
                # SELL crossed the book — record as filled at the cashout price.
                record_order_filled(
                    cashout_oid,
                    fill_usdc=cashout_price * shares_to_sell,
                    fill_price=cashout_price,
                    conn=conn,
                )
            except Exception as exc:
                logger.warning(f"Audit record_order failed for cashout {cashout_oid}: {exc}")

    # --- Update bet_ledger ---
    conn.execute(
        """
        UPDATE bet_ledger
        SET status                 = 'settled',
            settled_at             = ?,
            settle_outcome         = NULL,
            pnl_realised_usdc      = ?,
            bankroll_after_settle  = COALESCE(?, bankroll_after_settle),
            cashout_triggered_at   = ?,
            cashout_price          = ?,
            cashout_pnl_usdc       = ?,
            cashout_threshold_used = ?,
            cashout_order_id       = ?
        WHERE bet_id = ?
        """,
        (
            now, cashout_pnl, bankroll_after,
            now, cashout_price, cashout_pnl, return_ratio,
            cashout_oid,
            bet_id,
        ),
    )
    conn.commit()

    return {
        "success": True,
        "bet_id": bet_id,
        "cashout_price": cashout_price,
        "cashout_pnl": cashout_pnl,
        "return_ratio": return_ratio,
        "is_simulated": is_simulated,
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
    """Check all filled bets for cashout eligibility and execute if met.

    For each bet_ledger row with status='filled':
      1. Determine the cashout threshold via tiered_cashout_threshold(fill_price).
         Buckets where holding is optimal (coin flip / favourites) return None
         and are skipped automatically.
      2. Fetch the current CLOB midpoint for the outcome token.
      3. If evaluate_cashout() is True: call execute_cashout().

    Args:
        conn:        Open SQLite connection (created if None).
        poly_client: PolymarketClient (created if None).
        dry_run:     If True, simulate cashouts without placing SELL orders.

    Returns:
        {
            "n_checked":   int,
            "n_triggered": int,
            "n_executed":  int,
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
    cashouts: list = []
    errors: list = []

    for bet in rows:
        bet_id = bet["bet_id"]
        token_id = bet["polymarket_token_id"]
        fill_price = float(bet["fill_price"])

        # Tiered threshold — returns None for coin-flip / favourite buckets
        threshold = tiered_cashout_threshold(fill_price)
        if threshold is None:
            continue

        # Fetch current price
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

        # Evaluate
        if not evaluate_cashout(bet, current_price, threshold):
            fill_price = bet["fill_price"]
            ratio = current_price / fill_price if fill_price else 0
            logger.debug(
                f"  bet_id={bet_id}: ratio={ratio:.3f}x < threshold={threshold}x — hold"
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
        )
        cashouts.append(result)
        if result.get("success"):
            n_executed += 1
            bet_kind_val = bet["bet_kind"] if "bet_kind" in bet.keys() else "real"
            tag = "[DRY-RUN]" if dry_run else ("[PAPER]" if bet_kind_val == "paper" else "[REAL]")
            side_label = bet["side_label"] if "side_label" in bet.keys() else "?"
            logger.info(
                f"CASHOUT {tag} bet_id={bet_id} "
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
        "cashouts":    cashouts,
        "errors":      errors,
    }
