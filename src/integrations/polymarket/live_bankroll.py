"""Wallet-aware live bankroll — single source of truth for Kelly sizing and risk caps.

When the Polymarket wallet is configured, portfolio value (USDC cash + open
positions marked to market + unredeemed winning tokens) drives all limits.
Top-ups increase portfolio value automatically; losses shrink it.

Per-strategy bankroll = portfolio_value × allocation_weight[strategy].
Weights come from BETTING_MAX_DEPOSIT_<STRATEGY> env vars (relative shares)
or equal split among BETTING_LIVE_STRATEGIES.

When the wallet is unavailable (tests, no private key), falls back to
BETTING_MAX_DEPOSIT as the portfolio proxy.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# A resolved-winner token still held on-chain is only counted as "pending
# redemption" when BOTH hold: (1) we have a settled-won real BUY ledger row for
# it (our reconcile confirmed the match actually resolved), and (2) its on-chain
# mid has converged to ~$1. The ledger gate is the real protection against the
# "team comes back from 1%" edge case (a live position is never settled-won);
# the price gate is a secondary sanity check against data inconsistencies.
PENDING_REDEMPTION_MIN_PRICE = 0.99


def _betting_config() -> Dict[str, Any]:
    from config import BETTING_CONFIG
    return BETTING_CONFIG


def live_strategies() -> List[str]:
    return list(_betting_config().get("live_strategies") or [])


def strategy_allocation_weight(strategy_label: str) -> float:
    """Relative allocation share among live strategies (sums to 1.0)."""
    strategies = live_strategies()
    if not strategies:
        return 1.0
    if strategy_label not in strategies:
        # Historical / retired strategy — treat as solo slice for display.
        return 1.0

    cfg = _betting_config()
    default_slice = float(cfg.get("max_deposit_per_strategy_usdc") or 100.0)
    raw: Dict[str, float] = {}
    for name in strategies:
        env_key = f"BETTING_MAX_DEPOSIT_{name.upper().replace('-', '_')}"
        override = os.getenv(env_key)
        raw[name] = float(override) if override is not None else default_slice
    total = sum(raw.values())
    if total <= 0:
        return 1.0 / len(strategies)
    return raw[strategy_label] / total


def strategy_allocation_weights() -> Dict[str, float]:
    strategies = live_strategies()
    return {s: strategy_allocation_weight(s) for s in strategies}


def _open_positions_from_db(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT bet_id, polymarket_token_id AS token_id,
               fill_size_usdc, fill_price, side_label, strategy_label
        FROM bet_ledger
        WHERE COALESCE(bet_kind, 'real') = 'real'
          AND status = 'filled'
          AND settled_at IS NULL
          AND fill_price IS NOT NULL
          AND fill_price > 0
          AND polymarket_token_id IS NOT NULL
          AND polymarket_token_id != ''
        """
    )
    return [dict(r) for r in cur.fetchall()]


def _settled_won_token_ids(conn: sqlite3.Connection) -> set:
    """Token ids of real BUY bets we settled as winners (settle_outcome=1).

    Settlement is driven by reconcile, which only marks a bet settled once the
    match has actually resolved (Gamma closed / price gates past kickoff). So a
    token in this set is a genuine winner — used to recognise winning tokens we
    still hold on-chain before Polymarket flips their `redeemable` flag to True.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT DISTINCT polymarket_token_id
        FROM bet_ledger
        WHERE COALESCE(bet_kind, 'real') = 'real'
          AND side = 'BUY'
          AND status = 'settled'
          AND settle_outcome = 1
          AND polymarket_token_id IS NOT NULL
          AND polymarket_token_id != ''
        """
    )
    return {str(r[0]) for r in cur.fetchall()}


def _mark_open_positions(
    positions: List[Dict[str, Any]],
    pm: Any,
) -> float:
    """Sum MTM value of open positions; cost basis fallback when no mid."""
    _, market_value, _ = _mark_open_positions_detail(positions, pm)
    return market_value


def _mark_open_positions_detail(
    positions: List[Dict[str, Any]],
    pm: Any,
) -> tuple[float, float, List[Dict[str, Any]]]:
    """Return (cost_basis, market_value, per-bet rows) for open positions."""
    if not positions:
        return 0.0, 0.0, []

    unique_tokens = sorted({str(p["token_id"]) for p in positions})
    midpoints: Dict[str, float] = {}
    try:
        midpoints = pm.get_token_midpoints(unique_tokens) or {}
    except Exception as exc:
        logger.debug(f"Midpoint fetch failed, using cost basis: {exc}")

    total_cost = 0.0
    total_market = 0.0
    rows: List[Dict[str, Any]] = []
    for p in positions:
        cost = float(p["fill_size_usdc"] or 0.0)
        fill_price = float(p["fill_price"] or 0.0)
        if fill_price <= 0:
            continue
        shares = cost / fill_price
        mid = midpoints.get(str(p["token_id"]))
        if mid is not None and mid > 0:
            market_value = shares * float(mid)
            marked = True
        else:
            market_value = cost
            marked = False
        total_cost += cost
        total_market += market_value
        rows.append({
            "bet_id": p.get("bet_id"),
            "strategy_label": p.get("strategy_label"),
            "side_label": p.get("side_label"),
            "token_id": str(p["token_id"]),
            "shares": round(shares, 4),
            "fill_price": round(fill_price, 4),
            "cost_basis_usdc": round(cost, 4),
            "mid_price": round(mid, 4) if mid is not None else None,
            "market_value_usdc": round(market_value, 4),
            "unrealised_pnl_usdc": round(market_value - cost, 4),
            "marked_to_market": marked,
        })
    return total_cost, total_market, rows


def _redeemable_positions_from_pm(
    pm: Any,
    settled_won_token_ids: Optional[set] = None,
    exclude_token_ids: Optional[set] = None,
) -> tuple[float, float, List[Dict[str, Any]]]:
    """Value winning tokens still held on-chain, split by claim status.

    Returns ``(redeemable_now_usdc, pending_redemption_usdc, rows)``:

      * redeemable_now    -> Polymarket flagged ``redeemable=True``; claimable now.
      * pending_redemption-> we settled this token as a winner in the ledger and
                             its on-chain mid has converged to ~$1, but the
                             ``redeemable`` flag has not flipped yet (on-chain
                             resolution lags). Real winnings; not yet claimable.

    Open-ledger tokens are excluded to avoid double-counting against open MTM.
    """
    exclude = exclude_token_ids or set()
    won_tokens = settled_won_token_ids or set()
    try:
        raw_positions = pm.get_data_api_positions()
    except Exception as exc:
        logger.debug(f"Redeemable position fetch failed: {exc}")
        return 0.0, 0.0, []

    redeemable_now = 0.0
    pending = 0.0
    rows: List[Dict[str, Any]] = []
    for p in raw_positions:
        token_id = str(p.get("asset") or p.get("asset_id") or "")
        if not token_id or token_id in exclude:
            continue
        value = float(p.get("currentValue") or 0.0)
        if value <= 0:
            continue
        cur_price = float(p.get("curPrice") or 0.0)
        is_redeemable = bool(p.get("redeemable"))
        if is_redeemable:
            claim_status = "redeemable"
            redeemable_now += value
        elif token_id in won_tokens and cur_price >= PENDING_REDEMPTION_MIN_PRICE:
            claim_status = "pending_redemption"
            pending += value
        else:
            continue
        rows.append({
            "token_id": token_id,
            "title": p.get("title") or p.get("question") or "",
            "outcome": p.get("outcome") or "",
            "shares": round(float(p.get("size") or 0.0), 4),
            "cur_price": round(cur_price, 4),
            "current_value_usdc": round(value, 4),
            "initial_value_usdc": round(float(p.get("initialValue") or 0.0), 4),
            "redeemable": is_redeemable,
            "claim_status": claim_status,
        })
    return redeemable_now, pending, rows


def get_portfolio_breakdown(
    conn: sqlite3.Connection,
    pm: Optional[Any] = None,
) -> Dict[str, Any]:
    """Portfolio components: cash + open MTM + unredeemed winning tokens."""
    cfg = _betting_config()
    positions = _open_positions_from_db(conn)
    open_token_ids = {str(p["token_id"]) for p in positions}
    settled_won_token_ids = _settled_won_token_ids(conn)

    from config import POLYMARKET_CONFIG
    if POLYMARKET_CONFIG.get("private_key"):
        try:
            if pm is None:
                from src.integrations.polymarket import PolymarketClient
                pm = PolymarketClient()
            cash = float(pm.get_usdc_balance().get("balance_usdc") or 0.0)
            cost_basis, open_mtm, position_rows = _mark_open_positions_detail(
                positions, pm
            )
            redeemable_usdc, pending_usdc, redeemable_rows = _redeemable_positions_from_pm(
                pm,
                settled_won_token_ids=settled_won_token_ids,
                exclude_token_ids=open_token_ids,
            )
            redeemable_count = sum(
                1 for r in redeemable_rows if r["claim_status"] == "redeemable"
            )
            pending_count = sum(
                1 for r in redeemable_rows if r["claim_status"] == "pending_redemption"
            )
            portfolio = max(0.0, cash + open_mtm + redeemable_usdc + pending_usdc)
            return {
                "wallet_cash_usdc": round(cash, 2),
                "open_positions_count": len(positions),
                "open_positions_cost_basis_usdc": round(cost_basis, 2),
                "open_positions_market_value_usdc": round(open_mtm, 2),
                "unrealised_pnl_usdc": round(open_mtm - cost_basis, 2),
                "redeemable_usdc": round(redeemable_usdc, 2),
                "redeemable_count": redeemable_count,
                "pending_redemption_usdc": round(pending_usdc, 2),
                "pending_redemption_count": pending_count,
                "portfolio_value_usdc": round(portfolio, 2),
                "positions": position_rows,
                "redeemable_positions": redeemable_rows,
                "wallet_driven": True,
            }
        except Exception as exc:
            logger.warning(f"Wallet portfolio read failed, using config fallback: {exc}")

    open_cost = sum(float(p["fill_size_usdc"] or 0.0) for p in positions)
    fallback = float(cfg.get("max_deposit_usdc") or 200.0)
    portfolio = max(fallback, open_cost)
    return {
        "wallet_cash_usdc": None,
        "open_positions_count": len(positions),
        "open_positions_cost_basis_usdc": round(open_cost, 2),
        "open_positions_market_value_usdc": round(open_cost, 2),
        "unrealised_pnl_usdc": 0.0,
        "redeemable_usdc": 0.0,
        "redeemable_count": 0,
        "pending_redemption_usdc": 0.0,
        "pending_redemption_count": 0,
        "portfolio_value_usdc": round(portfolio, 2),
        "positions": [],
        "redeemable_positions": [],
        "wallet_driven": False,
    }


def get_portfolio_value(
    conn: sqlite3.Connection,
    pm: Optional[Any] = None,
) -> float:
    """Total live capital: wallet USDC + MTM open positions + redeemable tokens.

    Falls back to BETTING_MAX_DEPOSIT when wallet is not configured.
    """
    return float(get_portfolio_breakdown(conn, pm=pm)["portfolio_value_usdc"])


def get_strategy_bankroll(
    strategy_label: str,
    conn: sqlite3.Connection,
    pm: Optional[Any] = None,
) -> float:
    """Kelly / cap bankroll for one strategy = portfolio × allocation weight."""
    portfolio = get_portfolio_value(conn, pm=pm)
    weight = strategy_allocation_weight(strategy_label)
    return max(0.0, portfolio * weight)


def get_total_live_bankroll(
    conn: sqlite3.Connection,
    pm: Optional[Any] = None,
) -> float:
    """Sum of live-strategy bankrolls (= portfolio when weights cover live set)."""
    return get_portfolio_value(conn, pm=pm)


def _fraction_cap(fraction_key: str, dollar_key: str, portfolio: float) -> float:
    """Resolve a cap: prefer fraction × portfolio; fall back to dollar env."""
    cfg = _betting_config()
    frac = float(cfg.get(fraction_key) or 0.0)
    if frac > 0 and portfolio > 0:
        return frac * portfolio
    return float(cfg.get(dollar_key) or 0.0)


def get_max_deploy_usdc(conn: sqlite3.Connection, pm: Optional[Any] = None) -> float:
    """Max total open exposure across all strategies."""
    portfolio = get_portfolio_value(conn, pm=pm)
    return _fraction_cap("max_deploy_fraction", "max_deposit_usdc", portfolio)


def kickoff_utc_date(
    kickoff_at: Optional[str],
    fixture_key: Optional[str] = None,
) -> Optional[date]:
    """UTC calendar date for a match kickoff (from kickoff_at or fixture_key)."""
    iso = kickoff_at
    if not iso and fixture_key:
        from src.integrations.polymarket.reconcile import _derive_kickoff_iso_from_fixture_key
        iso = _derive_kickoff_iso_from_fixture_key(fixture_key)
    if not iso:
        return None
    try:
        normalized = str(iso).replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.date()
    except (ValueError, TypeError):
        m = re.search(r"(\d{4}-\d{2}-\d{2})", str(iso))
        if not m:
            return None
        return date.fromisoformat(m.group(1))


def uses_kickoff_day_open_cap() -> bool:
    return float(_betting_config().get("max_open_fraction_per_kickoff_day") or 0.0) > 0


def get_strategy_open_cap_per_kickoff_day_usdc(
    strategy_label: str,
    conn: sqlite3.Connection,
    pm: Optional[Any] = None,
) -> float:
    """Max open exposure for one strategy on one UTC kickoff date."""
    frac = float(_betting_config().get("max_open_fraction_per_kickoff_day") or 0.0)
    if frac <= 0:
        return 0.0
    bankroll = get_strategy_bankroll(strategy_label, conn, pm=pm)
    return frac * bankroll if bankroll > 0 else 0.0


def get_strategy_open_by_kickoff_day_utc(
    strategy_label: str,
    conn: sqlite3.Connection,
) -> Dict[str, float]:
    """Open stake grouped by UTC kickoff date for one strategy."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT kickoff_at, fixture_key, fill_size_usdc
        FROM bet_ledger
        WHERE COALESCE(bet_kind, 'real') = 'real'
          AND strategy_label = ?
          AND filled_at IS NOT NULL
          AND settled_at IS NULL
        """,
        (strategy_label,),
    )
    by_day: Dict[str, float] = {}
    for row in cur.fetchall():
        kickoff_date = kickoff_utc_date(row["kickoff_at"], row["fixture_key"])
        if kickoff_date is None:
            continue
        key = kickoff_date.isoformat()
        by_day[key] = by_day.get(key, 0.0) + float(row["fill_size_usdc"] or 0.0)
    return {k: round(v, 2) for k, v in sorted(by_day.items())}


def get_strategy_open_exposure_on_kickoff_day(
    strategy_label: str,
    conn: sqlite3.Connection,
    utc_date: Union[date, str],
) -> float:
    """Open stake for one strategy on a single UTC kickoff date."""
    key = utc_date.isoformat() if isinstance(utc_date, date) else str(utc_date)
    return float(get_strategy_open_by_kickoff_day_utc(strategy_label, conn).get(key, 0.0))


def get_strategy_flat_open_cap_usdc(
    strategy_label: str,
    conn: sqlite3.Connection,
    pm: Optional[Any] = None,
) -> float:
    """Flat per-strategy open cap across all kickoff dates (legacy mode)."""
    cfg = _betting_config()
    bankroll = get_strategy_bankroll(strategy_label, conn, pm=pm)
    open_frac = float(cfg.get("max_open_fraction_per_strategy") or 0.0)
    if open_frac > 0 and bankroll > 0:
        return open_frac * bankroll
    return float(cfg.get("max_deposit_per_strategy_usdc") or 0.0)


def get_strategy_open_cap_usdc(
    strategy_label: str,
    conn: sqlite3.Connection,
    pm: Optional[Any] = None,
) -> float:
    """Max open exposure for one strategy (flat cap, or per-day cap when configured)."""
    if uses_kickoff_day_open_cap():
        return get_strategy_open_cap_per_kickoff_day_usdc(strategy_label, conn, pm=pm)
    return get_strategy_flat_open_cap_usdc(strategy_label, conn, pm=pm)


def get_max_per_day_usdc(conn: sqlite3.Connection, pm: Optional[Any] = None) -> float:
    portfolio = get_portfolio_value(conn, pm=pm)
    return _fraction_cap("max_per_day_fraction", "max_per_day_usdc", portfolio)


def get_max_loss_per_day_usdc(conn: sqlite3.Connection, pm: Optional[Any] = None) -> float:
    portfolio = get_portfolio_value(conn, pm=pm)
    return _fraction_cap("max_loss_per_day_fraction", "max_loss_per_day_usdc", portfolio)


def get_max_per_bet_usdc(
    strategy_label: Optional[str],
    conn: sqlite3.Connection,
    pm: Optional[Any] = None,
) -> float:
    """Per-bet cap: fraction × strategy bankroll, or legacy dollar cap."""
    cfg = _betting_config()
    frac = float(cfg.get("max_per_bet_fraction") or 0.0)
    if frac > 0 and strategy_label:
        bankroll = get_strategy_bankroll(strategy_label, conn, pm=pm)
        if bankroll > 0:
            return frac * bankroll
    return float(cfg.get("max_per_bet_usdc") or 0.0)


def bankroll_snapshot(conn: sqlite3.Connection, pm: Optional[Any] = None) -> Dict[str, Any]:
    """Debug / API payload describing current bankroll math."""
    breakdown = get_portfolio_breakdown(conn, pm=pm)
    portfolio = float(breakdown["portfolio_value_usdc"])
    weights = strategy_allocation_weights()
    kickoff_day_mode = uses_kickoff_day_open_cap()
    strategies = {
        label: {
            "weight": round(w, 4),
            "bankroll_usdc": round(get_strategy_bankroll(label, conn, pm=pm), 2),
            "open_cap_usdc": round(get_strategy_open_cap_usdc(label, conn, pm=pm), 2),
            "kickoff_day_cap_usdc": round(
                get_strategy_open_cap_per_kickoff_day_usdc(label, conn, pm=pm), 2
            ) if kickoff_day_mode else None,
            "open_by_kickoff_day_utc": get_strategy_open_by_kickoff_day_utc(label, conn)
            if kickoff_day_mode else {},
            "max_per_bet_usdc": round(get_max_per_bet_usdc(label, conn, pm=pm), 2),
        }
        for label, w in weights.items()
    }
    # Capital-flow-aware ROI (deposits/withdrawals tracked separately). Lazy
    # import avoids a circular dependency (capital.py reads this module).
    capital_summary: Optional[Dict[str, Any]] = None
    try:
        from src.integrations.polymarket.capital import get_capital_summary
        capital_summary = get_capital_summary(conn, pm=pm)
    except Exception as exc:  # pragma: no cover - telemetry only
        logger.debug(f"capital summary unavailable: {exc}")

    return {
        "portfolio_value_usdc": round(portfolio, 2),
        "wallet_cash_usdc": breakdown.get("wallet_cash_usdc"),
        "open_positions_market_value_usdc": breakdown.get("open_positions_market_value_usdc"),
        "redeemable_usdc": breakdown.get("redeemable_usdc"),
        "redeemable_count": breakdown.get("redeemable_count"),
        "pending_redemption_usdc": breakdown.get("pending_redemption_usdc"),
        "pending_redemption_count": breakdown.get("pending_redemption_count"),
        "wallet_driven": breakdown.get("wallet_driven"),
        "kickoff_day_open_cap_mode": kickoff_day_mode,
        "max_open_fraction_per_kickoff_day": float(
            _betting_config().get("max_open_fraction_per_kickoff_day") or 0.0
        ),
        "max_deploy_usdc": round(get_max_deploy_usdc(conn, pm=pm), 2),
        "max_per_day_usdc": round(get_max_per_day_usdc(conn, pm=pm), 2),
        "max_loss_per_day_usdc": round(get_max_loss_per_day_usdc(conn, pm=pm), 2),
        "strategy_weights": weights,
        "strategies": strategies,
        "capital": capital_summary,
    }
