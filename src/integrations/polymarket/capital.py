"""Wave 6 follow-up: capital-flow accounting for true live ROI.

The bankroll module knows the *current* portfolio value but not how much
external capital was put in. This module owns the capital_flows ledger and
derives the only ROI numbers that stay correct across deposits/withdrawals:

    net_contributed = SUM(deposits) - SUM(withdrawals)
    net_pnl         = current_portfolio_value - net_contributed
    roi_on_capital  = net_pnl / net_contributed        (cash-on-cash return)

Depositing increases both portfolio_value and net_contributed by the same
amount, so net_pnl (and ROI) are invariant to funding events - which is the
whole point.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

VALID_FLOW_TYPES = ("deposit", "withdrawal")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _wallet_cash_usdc() -> Optional[float]:
    """Best-effort current wallet USDC cash; None if wallet unavailable."""
    try:
        from config import POLYMARKET_CONFIG
        if not POLYMARKET_CONFIG.get("private_key"):
            return None
        from src.integrations.polymarket import PolymarketClient
        pm = PolymarketClient()
        bal = pm.get_usdc_balance() or {}
        val = bal.get("balance_usdc")
        return float(val) if val is not None else None
    except Exception as exc:  # pragma: no cover - network/credential dependent
        logger.debug(f"wallet cash read failed: {exc}")
        return None


def _portfolio_value(conn: sqlite3.Connection) -> Optional[float]:
    try:
        from src.integrations.polymarket.live_bankroll import get_portfolio_value
        return float(get_portfolio_value(conn))
    except Exception as exc:  # pragma: no cover
        logger.debug(f"portfolio value read failed: {exc}")
        return None


def record_flow(
    conn: sqlite3.Connection,
    *,
    flow_type: str,
    amount_usdc: float,
    tx_hash: Optional[str] = None,
    source: str = "manual",
    note: Optional[str] = None,
    capture_wallet: bool = True,
    ts: Optional[str] = None,
) -> int:
    """Insert a deposit/withdrawal row; snapshot wallet/portfolio for audit.

    Returns the new flow_id. `amount_usdc` is always a positive magnitude;
    direction is carried by `flow_type`.
    """
    flow_type = flow_type.lower().strip()
    if flow_type not in VALID_FLOW_TYPES:
        raise ValueError(f"flow_type must be one of {VALID_FLOW_TYPES}, got {flow_type!r}")
    if amount_usdc is None or float(amount_usdc) <= 0:
        raise ValueError("amount_usdc must be a positive number")

    wallet_cash = _wallet_cash_usdc() if capture_wallet else None
    portfolio = _portfolio_value(conn) if capture_wallet else None

    # For a deposit, wallet_cash (read after funding) is the "after"; we can't
    # know "before" reliably, so store what we can. The amount is the source of
    # truth for accounting; the snapshot is audit context only.
    wallet_cash_after = wallet_cash
    wallet_cash_before = None
    if wallet_cash is not None:
        if flow_type == "deposit":
            wallet_cash_before = wallet_cash - float(amount_usdc)
        else:
            wallet_cash_before = wallet_cash + float(amount_usdc)

    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO capital_flows (
            ts, flow_type, amount_usdc, wallet_cash_before, wallet_cash_after,
            portfolio_value_at_flow, tx_hash, source, note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ts or _now_iso(), flow_type, float(amount_usdc),
            wallet_cash_before, wallet_cash_after, portfolio,
            tx_hash, source, note,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def get_flows(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    cur = conn.cursor()
    cur.execute("SELECT * FROM capital_flows ORDER BY ts ASC, flow_id ASC")
    return [dict(r) for r in cur.fetchall()]


def flow_exists_by_tx(conn: sqlite3.Connection, tx_hash: str) -> bool:
    """True if a capital flow with this on-chain tx hash is already recorded.

    Used by the chain reconciler to stay idempotent across re-runs.
    """
    if not tx_hash:
        return False
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM capital_flows WHERE LOWER(tx_hash) = LOWER(?) LIMIT 1",
        (tx_hash,),
    )
    return cur.fetchone() is not None


def _flow_totals(conn: sqlite3.Connection) -> Dict[str, float]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            COALESCE(SUM(CASE WHEN flow_type='deposit'    THEN amount_usdc END), 0) AS deposits,
            COALESCE(SUM(CASE WHEN flow_type='withdrawal' THEN amount_usdc END), 0) AS withdrawals,
            SUM(CASE WHEN flow_type='deposit' THEN 1 ELSE 0 END) AS n_deposits,
            SUM(CASE WHEN flow_type='withdrawal' THEN 1 ELSE 0 END) AS n_withdrawals
        FROM capital_flows
        """
    )
    row = cur.fetchone()
    return {
        "deposits": float(row[0] or 0.0),
        "withdrawals": float(row[1] or 0.0),
        "n_deposits": int(row[2] or 0),
        "n_withdrawals": int(row[3] or 0),
    }


def _realized_pnl(conn: sqlite3.Connection) -> float:
    """Booked P&L from settled real bets (cross-check against portfolio math)."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COALESCE(SUM(pnl_realised_usdc), 0)
        FROM bet_ledger
        WHERE COALESCE(bet_kind, 'real') = 'real'
          AND status = 'settled'
          AND pnl_realised_usdc IS NOT NULL
        """
    )
    return float(cur.fetchone()[0] or 0.0)


def get_capital_summary(
    conn: sqlite3.Connection,
    pm: Optional[Any] = None,
) -> Dict[str, Any]:
    """True cash-on-cash ROI on funded capital.

    Returns deposits/withdrawals totals, net contributed capital, current
    portfolio value, net P&L and ROI on capital (the headline number), plus a
    deployment ratio and a realized-P&L cross-check.
    """
    totals = _flow_totals(conn)
    net_contributed = totals["deposits"] - totals["withdrawals"]

    try:
        from src.integrations.polymarket.live_bankroll import get_portfolio_breakdown
        breakdown = get_portfolio_breakdown(conn, pm=pm)
    except Exception as exc:  # pragma: no cover
        logger.warning(f"portfolio breakdown failed in capital summary: {exc}")
        breakdown = {}

    portfolio = float(breakdown.get("portfolio_value_usdc") or 0.0)
    open_mtm = float(breakdown.get("open_positions_market_value_usdc") or 0.0)

    funded = net_contributed > 0
    net_pnl = (portfolio - net_contributed) if funded else 0.0
    roi_on_capital_pct = (100.0 * net_pnl / net_contributed) if funded else None
    roi_on_total_deposits_pct = (
        100.0 * net_pnl / totals["deposits"] if totals["deposits"] > 0 else None
    )
    deployment_pct = (100.0 * open_mtm / portfolio) if portfolio > 0 else None

    return {
        "total_deposits_usdc": round(totals["deposits"], 2),
        "total_withdrawals_usdc": round(totals["withdrawals"], 2),
        "n_deposits": totals["n_deposits"],
        "n_withdrawals": totals["n_withdrawals"],
        "net_contributed_usdc": round(net_contributed, 2),
        "current_portfolio_value_usdc": round(portfolio, 2),
        "net_pnl_usdc": round(net_pnl, 2) if funded else None,
        "roi_on_capital_pct": round(roi_on_capital_pct, 2) if roi_on_capital_pct is not None else None,
        "roi_on_total_deposits_pct": round(roi_on_total_deposits_pct, 2) if roi_on_total_deposits_pct is not None else None,
        "capital_deployed_pct": round(deployment_pct, 2) if deployment_pct is not None else None,
        "realized_pnl_usdc": round(_realized_pnl(conn), 2),
        "wallet_cash_usdc": breakdown.get("wallet_cash_usdc"),
        "wallet_driven": breakdown.get("wallet_driven"),
        "funded": funded,
    }
