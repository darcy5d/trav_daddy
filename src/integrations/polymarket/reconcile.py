"""Wave 5 Phase 6b + Wave 5.7: Bet ledger reconciliation.

For each bet in `placed`/`filled` status whose match has completed, fetch
the resolved Polymarket market and compute realised P&L. Update
`settled_at`, `settle_outcome`, `pnl_realised_usdc`, `status='settled'`.

Resolution detection (in priority order):
  1. Gamma /markets/{id} returns `closed: true` + `outcomePrices`.
     This is the canonical resolution signal - clean, deterministic,
     sets outcomePrices to exact ["0","1"] or ["1","0"].
  2. CLOB /prices-history shows last_price <= 0.02 or >= 0.98 AND the
     proposal was at least 6h ago. Fallback for markets where Gamma is
     lagging or the market_id wasn't recorded.

Idempotent - already-settled rows are skipped.

Usage:

    from src.integrations.polymarket.reconcile import reconcile_pending_bets
    summary = reconcile_pending_bets()
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.integrations.polymarket import PolymarketClient
from src.integrations.odds.polymarket_compare import (
    POLYMARKET_TAKER_FEE, PolymarketComparisonService,
)

logger = logging.getLogger(__name__)

GAMMA_MARKETS_BASE = "https://gamma-api.polymarket.com/markets"
SETTLED_PRICE_HIGH = 0.98
SETTLED_PRICE_LOW = 0.02
MIN_HOURS_SINCE_PROPOSAL_FOR_PRICE_FALLBACK = 6.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fetch_unsettled_bets(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT *
        FROM bet_ledger
        WHERE status IN ('placed', 'filled')
        ORDER BY placed_at ASC
        """
    )
    return cur.fetchall()


def _last_price_from_history(history: List[Dict[str, Any]]) -> Optional[float]:
    """Return the very last price in a /prices-history series."""
    if not history:
        return None
    last = history[-1]
    if isinstance(last, dict) and "p" in last:
        return float(last["p"])
    return None


def _hours_since(iso_ts: Optional[str]) -> Optional[float]:
    if not iso_ts:
        return None
    try:
        d = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    now = datetime.now(timezone.utc)
    return (now - d).total_seconds() / 3600.0


def _resolve_via_gamma(
    market_id: str,
    side_label: str,
) -> Optional[Tuple[int, float]]:
    """Query Gamma /markets/<id> and return (settle_outcome, last_traded_price).

    settle_outcome: 1 if OUR side won, 0 if it lost.
    last_traded_price: outcomePrices entry for OUR side (used as the
        effective resolution price for P&L verification).

    Returns None if:
      - the market isn't `closed: true`
      - we can't match `side_label` to any outcome
      - the response is malformed
    """
    if not market_id or not side_label:
        return None
    try:
        r = requests.get(f"{GAMMA_MARKETS_BASE}/{market_id}", timeout=10)
        if not r.ok:
            return None
        m = r.json()
    except Exception as exc:
        logger.debug(f"  Gamma /markets/{market_id} fetch failed: {exc}")
        return None
    if not isinstance(m, dict):
        return None
    if not m.get("closed"):
        return None

    outcomes_raw = m.get("outcomes")
    prices_raw = m.get("outcomePrices")

    def _parse(value: Any) -> List[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                p = json.loads(value)
                if isinstance(p, list):
                    return p
            except json.JSONDecodeError:
                pass
        return []

    outcomes = _parse(outcomes_raw)
    prices = _parse(prices_raw)
    if not outcomes or not prices or len(outcomes) != len(prices):
        return None

    # Find which outcome matches our side_label
    matched_idx: Optional[int] = None
    for idx, out_label in enumerate(outcomes):
        if PolymarketComparisonService.label_matches_team(str(out_label), side_label):
            matched_idx = idx
            break
    if matched_idx is None:
        # Substring fallback (handles 'Rawalpindiz' vs 'Rawalpindi Pindiz')
        side_norm = side_label.lower()
        for idx, out_label in enumerate(outcomes):
            if str(out_label).lower() == side_norm:
                matched_idx = idx
                break
    if matched_idx is None:
        return None

    try:
        resolved_price = float(prices[matched_idx])
    except (TypeError, ValueError):
        return None
    settle_outcome = 1 if resolved_price >= 0.5 else 0
    return settle_outcome, resolved_price


def _compute_pnl_for_settled_bet(
    fill_price: Optional[float],
    fill_size_usdc: Optional[float],
    settle_outcome: float,
    fee_pct: float = POLYMARKET_TAKER_FEE,
) -> Optional[float]:
    """Net USD P&L (signed) for a filled bet whose outcome is now known."""
    if fill_price is None or fill_size_usdc is None:
        return None
    if fill_price <= 0:
        return None
    shares = fill_size_usdc / fill_price
    gross_payout = shares * float(settle_outcome)
    fee = fill_size_usdc * fee_pct
    return round(gross_payout - fill_size_usdc - fee, 4)


def reconcile_pending_bets(
    conn: Optional[sqlite3.Connection] = None,
    poly_client: Optional[PolymarketClient] = None,
) -> Dict[str, Any]:
    """Walk all unsettled bets and try to mark them settled.

    Returns:
        {
            "n_checked": int,
            "n_settled": int,
            "n_still_pending": int,
            "errors": [(bet_id, message), ...]
        }
    """
    own_conn = False
    if conn is None:
        from src.data.database import get_connection
        conn = get_connection()
        own_conn = True

    if poly_client is None:
        poly_client = PolymarketClient()

    bets = _fetch_unsettled_bets(conn)
    n_checked = len(bets)
    n_settled = 0
    errors: List[Any] = []

    for bet in bets:
        market_id = bet["polymarket_market_id"]
        token_id = bet["polymarket_token_id"]
        side_label = bet["side_label"] or ""

        settle_outcome: Optional[int] = None
        resolution_method: Optional[str] = None

        # Path 1: Gamma /markets/{id} - canonical resolution signal
        if market_id:
            gamma_result = _resolve_via_gamma(str(market_id), side_label)
            if gamma_result is not None:
                settle_outcome, _resolved_price = gamma_result
                resolution_method = "gamma"

        # Path 2: prices-history fallback (requires a generous time buffer
        # since we don't have the canonical resolution signal)
        if settle_outcome is None and token_id:
            try:
                resp = poly_client.get_prices_history(
                    token_id=token_id, interval="all", fidelity=60,
                )
                history = resp.get("history") if isinstance(resp, dict) else resp
            except Exception as exc:
                errors.append((bet["bet_id"], f"prices-history failed: {exc}"))
                continue
            last_price = _last_price_from_history(history or [])
            if last_price is None:
                continue
            hours_since = _hours_since(bet["proposed_at"])
            buffer_ok = hours_since is None or hours_since >= MIN_HOURS_SINCE_PROPOSAL_FOR_PRICE_FALLBACK
            if not buffer_ok:
                continue
            if last_price >= SETTLED_PRICE_HIGH:
                settle_outcome = 1
                resolution_method = "price-history-high"
            elif last_price <= SETTLED_PRICE_LOW:
                settle_outcome = 0
                resolution_method = "price-history-low"

        if settle_outcome is None:
            continue

        pnl = _compute_pnl_for_settled_bet(
            fill_price=bet["fill_price"],
            fill_size_usdc=bet["fill_size_usdc"],
            settle_outcome=float(settle_outcome),
        )

        try:
            # For paper bets, snapshot the LIVE strategy bankroll AT
            # SETTLEMENT TIME (starting + cumsum of all pnl already
            # settled for this strategy + this bet's pnl). We can't just
            # use bankroll_at_proposal because if multiple bets are
            # placed when the bankroll is $1000 and they settle in
            # arbitrary order, each one would have stale bankroll_at_proposal.
            bet_kind = bet["bet_kind"] if "bet_kind" in bet.keys() else "real"
            bankroll_after = None
            if bet_kind == "paper" and pnl is not None:
                strategy_label = bet["strategy_label"]
                if strategy_label:
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
                        (strategy_label, bet["bet_id"]),
                    )
                    prior_pnl = float(cur2.fetchone()[0] or 0.0)
                    from src.integrations.polymarket.paper_strategies import get_strategy
                    strat = get_strategy(strategy_label)
                    starting = strat.starting_bankroll_usdc if strat else 1000.0
                    bankroll_after = starting + prior_pnl + float(pnl)

            conn.execute(
                """
                UPDATE bet_ledger
                SET status = 'settled',
                    settled_at = ?,
                    settle_outcome = ?,
                    pnl_realised_usdc = ?,
                    bankroll_after_settle = COALESCE(?, bankroll_after_settle)
                WHERE bet_id = ?
                """,
                (_utc_now_iso(), int(settle_outcome), pnl, bankroll_after, bet["bet_id"]),
            )
            conn.commit()
            n_settled += 1
            tag = "[PAPER]" if bet_kind == "paper" else ""
            logger.info(
                f"Reconciled bet_id={bet['bet_id']} {tag} via={resolution_method} "
                f"market={bet['market_type']} side={bet['side_label']} "
                f"settle={settle_outcome} pnl=${pnl}"
            )
        except Exception as exc:
            errors.append((bet["bet_id"], f"DB update failed: {exc}"))

    if own_conn:
        conn.close()

    return {
        "n_checked": n_checked,
        "n_settled": n_settled,
        "n_still_pending": n_checked - n_settled,
        "errors": errors,
    }
