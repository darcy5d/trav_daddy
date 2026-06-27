"""Wave 5 Phase 6c: Server-side risk gate for live Polymarket betting.

Single source of truth for the "can this bet be placed?" decision.
Always called BEFORE any write to Polymarket. UI can re-check for UX
but cannot bypass.

Key invariants enforced:
- BETTING_MODE=OFF blocks everything.
- BETTING_KILL_SWITCH=1 blocks everything until manually flipped off.
- Sum of today's filled stakes + proposed_size <= max_per_day_usdc.
- Today's realised loss + worst-case proposed loss <= max_loss_per_day_usdc.
- Total open exposure (filled, not yet settled) + proposed_size <= max_deposit_usdc.
- Per-bet stake <= max_per_bet_usdc.
- AUTO mode: market_type must be in BETTING_AUTO_MARKETS list AND edge_pp >= BETTING_AUTO_MIN_EDGE.
- Phase 7 scale-up gate: envelope can ONLY graduate after `scale_up_min_settled_bets`
  AND live brier_on_bets within `scale_up_max_brier_drift` of backtest.

The kill switch persists across Flask restarts by writing
`BETTING_KILL_SWITCH=1` to .env when toggled via the UI endpoint.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RiskGateDecision:
    allowed: bool
    reason: str  # human-readable; safe to surface in API response
    cap_remaining_today: float
    cap_remaining_deposit: float


def _today_start_utc_iso() -> str:
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()


def _todays_filled_stakes(conn: sqlite3.Connection) -> float:
    """Sum of fill_size_usdc for REAL bets filled today (UTC).

    Wave 5.8: excludes bet_kind='paper' so paper-trading history doesn't
    consume live daily-stake capacity. Legacy rows with NULL bet_kind are
    treated as 'real' (pre-dates the column).
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COALESCE(SUM(fill_size_usdc), 0.0) AS total
        FROM bet_ledger
        WHERE COALESCE(bet_kind, 'real') = 'real'
          AND filled_at IS NOT NULL
          AND filled_at >= ?
        """,
        (_today_start_utc_iso(),),
    )
    row = cur.fetchone()
    return float(row["total"] or 0.0)


def _todays_realised_loss(conn: sqlite3.Connection) -> float:
    """Negative pnl_realised_usdc summed across REAL bets settled today."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COALESCE(SUM(CASE WHEN pnl_realised_usdc < 0 THEN -pnl_realised_usdc ELSE 0 END), 0.0) AS loss
        FROM bet_ledger
        WHERE COALESCE(bet_kind, 'real') = 'real'
          AND settled_at IS NOT NULL
          AND settled_at >= ?
        """,
        (_today_start_utc_iso(),),
    )
    row = cur.fetchone()
    return float(row["loss"] or 0.0)


def _open_exposure_usdc(conn: sqlite3.Connection) -> float:
    """Sum of fill_size_usdc for REAL bets filled but not yet settled."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COALESCE(SUM(fill_size_usdc), 0.0) AS exposure
        FROM bet_ledger
        WHERE COALESCE(bet_kind, 'real') = 'real'
          AND filled_at IS NOT NULL
          AND settled_at IS NULL
        """
    )
    row = cur.fetchone()
    return float(row["exposure"] or 0.0)


def _parse_heartbeat_ts(iso: Optional[str]) -> Optional[datetime]:
    """Parse an ISO8601 heartbeat timestamp to an aware UTC datetime; None on failure."""
    if not iso:
        return None
    try:
        d = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d
    except (ValueError, AttributeError):
        return None


def _exit_health_block(conn: sqlite3.Connection) -> Optional[str]:
    """Return a block reason if new LIVE risk should be refused because the
    in-play exit scanner is unhealthy, else None. ("Don't open risk you can't
    close.")

    Post-outage breaker (Jun 1-6): the cashout/stop scanner could not run
    (disk/iCloud hang), so open positions rode to settlement while
    live_bet_scan kept opening new exposure and the realised-loss cap stayed
    at ~$0. We refuse new live exposure when there are open positions AND the
    cashout scanner is stale (not running) or reports the CLOB unreachable.
    """
    from config import BETTING_CONFIG

    max_stale_min = float(BETTING_CONFIG.get("exit_health_max_stale_min", 0) or 0)
    if max_stale_min <= 0:
        return None  # breaker disabled

    # Only relevant when we actually hold open positions that need exits.
    if _open_exposure_usdc(conn) <= 0:
        return None

    from src.integrations.polymarket.cashout import read_cashout_heartbeat

    hb = read_cashout_heartbeat()
    if not hb:
        return (
            "Exit-health breaker: no cashout-scanner heartbeat found while live "
            "positions are open — not opening new risk that can't be closed."
        )
    if hb.get("clob_reachable") is False:
        return (
            "Exit-health breaker: cashout scanner reports the CLOB unreachable "
            "— not opening new risk while exits cannot execute."
        )
    last_ok = _parse_heartbeat_ts(hb.get("last_success_utc") or hb.get("last_run_utc"))
    if last_ok is None:
        return (
            "Exit-health breaker: cashout heartbeat has no usable timestamp "
            "— not opening new risk until exits are confirmed running."
        )
    age_min = (datetime.now(timezone.utc) - last_ok).total_seconds() / 60.0
    if age_min > max_stale_min:
        return (
            f"Exit-health breaker: cashout scanner last succeeded {age_min:.0f}m "
            f"ago (> {max_stale_min:.0f}m threshold) while live positions are "
            f"open — not opening new risk while exits may be stalled."
        )
    return None


def _strategy_open_exposure_usdc(conn: sqlite3.Connection, strategy_label: str) -> float:
    """Sum of fill_size_usdc for real (not paper) bets under this strategy
    that are filled but not yet settled.

    Wave 5.8: per-strategy cap enforcement. Paper bets are excluded so a
    historical paper run doesn't consume live capacity.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COALESCE(SUM(fill_size_usdc), 0.0) AS exposure
        FROM bet_ledger
        WHERE COALESCE(bet_kind, 'real') = 'real'
          AND strategy_label = ?
          AND filled_at IS NOT NULL
          AND settled_at IS NULL
        """,
        (strategy_label,),
    )
    row = cur.fetchone()
    return float(row["exposure"] or 0.0)


def _todays_bet_count(conn: sqlite3.Connection) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COUNT(*) AS n FROM bet_ledger
        WHERE COALESCE(bet_kind, 'real') = 'real'
          AND proposed_at >= ?
        """,
        (_today_start_utc_iso(),),
    )
    row = cur.fetchone()
    return int(row["n"] or 0)


def _settled_bet_count(conn: sqlite3.Connection) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COUNT(*) AS n FROM bet_ledger
        WHERE COALESCE(bet_kind, 'real') = 'real'
          AND settled_at IS NOT NULL
        """
    )
    row = cur.fetchone()
    return int(row["n"] or 0)


def can_place_bet(
    proposed_size_usdc: float,
    market_type: str,
    edge_pp: float,
    requested_mode: str = "manual",
    conn: Optional[sqlite3.Connection] = None,
    strategy_label: Optional[str] = None,
    kickoff_at: Optional[str] = None,
    fixture_key: Optional[str] = None,
) -> RiskGateDecision:
    """Return RiskGateDecision indicating whether the bet may proceed.

    Args:
        proposed_size_usdc: Stake size in USD.
        market_type: One of moneyline / top_batter / most_sixes / toss_match_double.
        edge_pp: Model edge over market in percentage points (positive = favourable).
        requested_mode: "manual" or "auto" (matches the bet's `mode` column).
        conn: Optional sqlite connection; opens its own if None.
        strategy_label: Optional strategy name; when provided the per-strategy
            cap and live_strategies whitelist are enforced in addition to global caps.
        kickoff_at: Match kickoff ISO (for per-kickoff-day open cap).
        fixture_key: Fixture slug fallback when kickoff_at is missing.

    All caps come from BETTING_CONFIG which is env-driven (BETTING_*).
    """
    from config import BETTING_CONFIG

    own_conn = False
    if conn is None:
        from src.data.database import get_connection
        conn = get_connection()
        own_conn = True

    try:
        # 1. Kill switch
        if BETTING_CONFIG.get("kill_switch"):
            return RiskGateDecision(
                allowed=False,
                reason="Kill switch is ENGAGED. Flip BETTING_KILL_SWITCH=0 to resume.",
                cap_remaining_today=0.0,
                cap_remaining_deposit=0.0,
            )

        # 2. Mode gate
        mode = BETTING_CONFIG.get("mode", "OFF").upper()
        if mode == "OFF":
            return RiskGateDecision(
                allowed=False,
                reason="BETTING_MODE=OFF. Switch to MANUAL or AUTO via /api/betting/mode.",
                cap_remaining_today=0.0,
                cap_remaining_deposit=0.0,
            )
        if requested_mode == "auto" and mode != "AUTO":
            return RiskGateDecision(
                allowed=False,
                reason=f"Requested mode=auto but BETTING_MODE={mode}; AUTO bets disabled.",
                cap_remaining_today=0.0,
                cap_remaining_deposit=0.0,
            )

        # 2b. Per-strategy whitelist gate (Wave 5.8).
        # When a strategy_label is provided, it must appear in BETTING_LIVE_STRATEGIES
        # (empty list = no strategies are live; blocks all strategy-tagged bets).
        # Untagged bets (strategy_label=None) bypass this check and only face the
        # global caps — used for ad-hoc manual bets from the /live-betting UI.
        if strategy_label is not None:
            live_strategies = BETTING_CONFIG.get("live_strategies", []) or []
            if strategy_label not in live_strategies:
                return RiskGateDecision(
                    allowed=False,
                    reason=(
                        f"Strategy '{strategy_label}' is not in BETTING_LIVE_STRATEGIES="
                        f"{live_strategies}. Add the label to .env to enable."
                    ),
                    cap_remaining_today=0.0,
                    cap_remaining_deposit=0.0,
                )

        # 2c. Exit-health breaker (post-outage). Refuse to open new live risk
        # when positions are open but the in-play exit scanner is stale or the
        # CLOB is unreachable. This is the primary fix for the Jun 1-6 failure
        # mode, where unrealised losses never tripped the realised-loss cap.
        exit_block_reason = _exit_health_block(conn)
        if exit_block_reason is not None:
            return RiskGateDecision(
                allowed=False,
                reason=exit_block_reason,
                cap_remaining_today=0.0,
                cap_remaining_deposit=0.0,
            )

        # 3. Per-bet cap — fraction × strategy bankroll (wallet-driven).
        from src.integrations.polymarket.live_bankroll import get_max_per_bet_usdc
        max_per_bet_fraction = float(BETTING_CONFIG.get("max_per_bet_fraction", 0))
        max_per_bet = get_max_per_bet_usdc(strategy_label, conn)
        if max_per_bet_fraction > 0 and strategy_label is not None:
            live_bankroll = _live_bankroll_for_strategy(conn, strategy_label)
            cap_label = f"{max_per_bet_fraction*100:.1f}% of ${live_bankroll:.2f} strategy bankroll"
        elif max_per_bet > 0:
            cap_label = "per-bet cap"
        else:
            cap_label = "hard cap"
        if max_per_bet > 0 and proposed_size_usdc > max_per_bet:
            return RiskGateDecision(
                allowed=False,
                reason=(
                    f"Proposed ${proposed_size_usdc:.2f} exceeds per-bet cap "
                    f"${max_per_bet:.2f} ({cap_label})."
                ),
                cap_remaining_today=0.0,
                cap_remaining_deposit=0.0,
            )

        # 4. Daily stake cap (portfolio-proportional when fraction configured)
        from src.integrations.polymarket.live_bankroll import (
            get_max_deploy_usdc,
            get_max_loss_per_day_usdc,
            get_max_per_day_usdc,
            get_strategy_flat_open_cap_usdc,
            get_strategy_open_cap_per_kickoff_day_usdc,
            get_strategy_open_exposure_on_kickoff_day,
            kickoff_utc_date,
            uses_kickoff_day_open_cap,
        )
        max_per_day = get_max_per_day_usdc(conn)
        todays_stakes = _todays_filled_stakes(conn)
        cap_remaining_today = max(0.0, max_per_day - todays_stakes)
        if max_per_day > 0 and proposed_size_usdc > cap_remaining_today:
            return RiskGateDecision(
                allowed=False,
                reason=(
                    f"Proposed ${proposed_size_usdc} would exceed daily stake cap "
                    f"(${todays_stakes:.2f} already today, max ${max_per_day})."
                ),
                cap_remaining_today=cap_remaining_today,
                cap_remaining_deposit=0.0,
            )

        # 5. Daily loss cap
        max_loss_per_day = get_max_loss_per_day_usdc(conn)
        todays_loss = _todays_realised_loss(conn)
        # Worst case for THIS bet: lose the entire stake (price~1, settle=0)
        worst_case_loss = proposed_size_usdc
        if max_loss_per_day > 0 and todays_loss + worst_case_loss > max_loss_per_day:
            return RiskGateDecision(
                allowed=False,
                reason=(
                    f"Worst-case projected loss (${todays_loss:.2f} actual + "
                    f"${worst_case_loss:.2f} new) would exceed daily loss cap "
                    f"(${max_loss_per_day})."
                ),
                cap_remaining_today=cap_remaining_today,
                cap_remaining_deposit=0.0,
            )

        # 6. Total open exposure (portfolio-proportional)
        max_deposit = get_max_deploy_usdc(conn)
        open_exposure = _open_exposure_usdc(conn)
        cap_remaining_deposit = max(0.0, max_deposit - open_exposure)
        if max_deposit > 0 and proposed_size_usdc > cap_remaining_deposit:
            return RiskGateDecision(
                allowed=False,
                reason=(
                    f"Proposed ${proposed_size_usdc} would exceed total open exposure cap "
                    f"(${open_exposure:.2f} already open, max ${max_deposit})."
                ),
                cap_remaining_today=cap_remaining_today,
                cap_remaining_deposit=cap_remaining_deposit,
            )

        # 6b. Per-strategy open exposure — kickoff-day or flat cap.
        if strategy_label is not None:
            kickoff_day_frac = float(
                BETTING_CONFIG.get("max_open_fraction_per_kickoff_day") or 0.0
            )
            if kickoff_day_frac > 0 and uses_kickoff_day_open_cap():
                utc_date = kickoff_utc_date(kickoff_at, fixture_key)
                if utc_date is None:
                    strategy_cap = get_strategy_flat_open_cap_usdc(strategy_label, conn)
                    if strategy_cap > 0:
                        strategy_open = _strategy_open_exposure_usdc(conn, strategy_label)
                        cap_remaining_strategy = max(0.0, strategy_cap - strategy_open)
                        if proposed_size_usdc > cap_remaining_strategy:
                            return RiskGateDecision(
                                allowed=False,
                                reason=(
                                    f"Strategy '{strategy_label}' open ${strategy_open:.2f} + "
                                    f"${proposed_size_usdc} would exceed per-strategy cap "
                                    f"${strategy_cap:.2f} (remaining ${cap_remaining_strategy:.2f}). "
                                    f"(No kickoff date for kickoff-day cap; flat cap fallback.)"
                                ),
                                cap_remaining_today=cap_remaining_today,
                                cap_remaining_deposit=cap_remaining_deposit,
                            )
                else:
                    day_cap = get_strategy_open_cap_per_kickoff_day_usdc(
                        strategy_label, conn
                    )
                    if day_cap > 0:
                        day_open = get_strategy_open_exposure_on_kickoff_day(
                            strategy_label, conn, utc_date
                        )
                        cap_remaining_day = max(0.0, day_cap - day_open)
                        if proposed_size_usdc > cap_remaining_day:
                            return RiskGateDecision(
                                allowed=False,
                                reason=(
                                    f"Strategy '{strategy_label}' open ${day_open:.2f} on "
                                    f"{utc_date.isoformat()} UTC + ${proposed_size_usdc} would "
                                    f"exceed kickoff-day cap ${day_cap:.2f} "
                                    f"(remaining ${cap_remaining_day:.2f})."
                                ),
                                cap_remaining_today=cap_remaining_today,
                                cap_remaining_deposit=cap_remaining_deposit,
                            )
            else:
                strategy_cap = get_strategy_flat_open_cap_usdc(strategy_label, conn)
                if strategy_cap > 0:
                    strategy_open = _strategy_open_exposure_usdc(conn, strategy_label)
                    cap_remaining_strategy = max(0.0, strategy_cap - strategy_open)
                    if proposed_size_usdc > cap_remaining_strategy:
                        return RiskGateDecision(
                            allowed=False,
                            reason=(
                                f"Strategy '{strategy_label}' open ${strategy_open:.2f} + "
                                f"${proposed_size_usdc} would exceed per-strategy cap "
                                f"${strategy_cap:.2f} (remaining ${cap_remaining_strategy:.2f})."
                            ),
                            cap_remaining_today=cap_remaining_today,
                            cap_remaining_deposit=cap_remaining_deposit,
                        )

        # 7. AUTO-mode market gate
        if requested_mode == "auto":
            auto_markets = BETTING_CONFIG.get("auto_enabled_markets", [])
            if market_type not in auto_markets:
                return RiskGateDecision(
                    allowed=False,
                    reason=(
                        f"Market type '{market_type}' not in BETTING_AUTO_MARKETS="
                        f"{auto_markets}. AUTO bets blocked for this market."
                    ),
                    cap_remaining_today=cap_remaining_today,
                    cap_remaining_deposit=cap_remaining_deposit,
                )
            min_edge = float(BETTING_CONFIG.get("auto_min_edge_pp", 0))
            if edge_pp < min_edge:
                return RiskGateDecision(
                    allowed=False,
                    reason=(
                        f"Edge {edge_pp:.1f}pp below auto threshold {min_edge}pp."
                    ),
                    cap_remaining_today=cap_remaining_today,
                    cap_remaining_deposit=cap_remaining_deposit,
                )

        return RiskGateDecision(
            allowed=True,
            reason="OK",
            cap_remaining_today=cap_remaining_today,
            cap_remaining_deposit=cap_remaining_deposit,
        )
    finally:
        if own_conn:
            conn.close()


def _exit_health_snapshot(open_exposure: float) -> Dict[str, Any]:
    """Read-only view of the exit-scanner heartbeat for the UI/verification."""
    from config import BETTING_CONFIG
    from src.integrations.polymarket.cashout import read_cashout_heartbeat

    max_stale_min = float(BETTING_CONFIG.get("exit_health_max_stale_min", 0) or 0)
    hb = read_cashout_heartbeat()
    age_min = None
    if hb:
        last_ok = _parse_heartbeat_ts(hb.get("last_success_utc") or hb.get("last_run_utc"))
        if last_ok is not None:
            age_min = round((datetime.now(timezone.utc) - last_ok).total_seconds() / 60.0, 1)
    return {
        "enabled": max_stale_min > 0,
        "max_stale_min": max_stale_min,
        "heartbeat_present": hb is not None,
        "last_run_utc": (hb or {}).get("last_run_utc"),
        "last_success_utc": (hb or {}).get("last_success_utc"),
        "clob_reachable": (hb or {}).get("clob_reachable"),
        "age_min": age_min,
        # True when the breaker would currently block a new live bet.
        "would_block_new_live": bool(
            max_stale_min > 0
            and open_exposure > 0
            and (
                hb is None
                or (hb.get("clob_reachable") is False)
                or (age_min is not None and age_min > max_stale_min)
            )
        ),
    }


def get_risk_status() -> Dict[str, Any]:
    """Return a snapshot of current risk-gate state for the UI."""
    from config import BETTING_CONFIG
    from src.data.database import get_connection
    from src.integrations.polymarket.live_bankroll import (
        bankroll_snapshot,
        get_max_deploy_usdc,
        get_max_loss_per_day_usdc,
        get_max_per_day_usdc,
        get_portfolio_value,
    )

    conn = get_connection()
    try:
        todays_stakes = _todays_filled_stakes(conn)
        todays_loss = _todays_realised_loss(conn)
        open_exposure = _open_exposure_usdc(conn)
        todays_bets = _todays_bet_count(conn)
        settled_bets_total = _settled_bet_count(conn)
        portfolio = get_portfolio_value(conn)
        max_deposit = get_max_deploy_usdc(conn)
        max_per_day = get_max_per_day_usdc(conn)
        max_loss_per_day = get_max_loss_per_day_usdc(conn)
        snap = bankroll_snapshot(conn)
    finally:
        conn.close()

    return {
        "mode": BETTING_CONFIG.get("mode", "OFF").upper(),
        "kill_switch": bool(BETTING_CONFIG.get("kill_switch", False)),
        "portfolio_value_usdc": round(portfolio, 2),
        "bankroll_snapshot": snap,
        "max_deposit_usdc": max_deposit,
        "max_per_bet_usdc": float(BETTING_CONFIG.get("max_per_bet_usdc", 0)),
        "max_per_bet_fraction": float(BETTING_CONFIG.get("max_per_bet_fraction", 0)),
        "max_deploy_fraction": float(BETTING_CONFIG.get("max_deploy_fraction", 0)),
        "max_open_fraction_per_strategy": float(BETTING_CONFIG.get("max_open_fraction_per_strategy", 0)),
        "max_per_day_usdc": max_per_day,
        "max_loss_per_day_usdc": max_loss_per_day,
        "auto_min_edge_pp": float(BETTING_CONFIG.get("auto_min_edge_pp", 0)),
        "auto_enabled_markets": list(BETTING_CONFIG.get("auto_enabled_markets", [])),
        "today_utc_start": _today_start_utc_iso(),
        "today_filled_stakes_usdc": round(todays_stakes, 2),
        "today_realised_loss_usdc": round(todays_loss, 2),
        "today_bet_count": todays_bets,
        "today_remaining_cap_usdc": max(0.0, max_per_day - todays_stakes) if max_per_day > 0 else None,
        "open_exposure_usdc": round(open_exposure, 2),
        "deposit_remaining_cap_usdc": max(0.0, max_deposit - open_exposure) if max_deposit > 0 else None,
        "settled_bets_total": settled_bets_total,
        "scale_up_eligible": _is_scale_up_eligible(settled_bets_total),
        # Operational controls (additive; surfaced for the dashboard + ops checks).
        "risk_controls": {
            "stop_loss_enabled": bool(BETTING_CONFIG.get("stop_loss_enabled", False)),
            "associate_throttle_enabled": bool(
                BETTING_CONFIG.get("associate_throttle_enabled", False)
            ),
            "associate_kelly_mult": float(BETTING_CONFIG.get("associate_kelly_mult", 0)),
            "model_prob_cap": float(BETTING_CONFIG.get("model_prob_cap", 0)),
            "live_exclude_prefixes": list(
                BETTING_CONFIG.get("live_exclude_prefixes", []) or []
            ),
            "rebalance_no_average_down": bool(
                BETTING_CONFIG.get("rebalance_no_average_down", False)
            ),
        },
        "exit_health": _exit_health_snapshot(open_exposure),
    }


def _live_bankroll_for_strategy(conn: sqlite3.Connection, strategy_label: str) -> float:
    """Strategy bankroll for Kelly sizing and per-bet caps (wallet-driven)."""
    from src.integrations.polymarket.live_bankroll import get_strategy_bankroll
    return get_strategy_bankroll(strategy_label, conn)


def _is_scale_up_eligible(settled_count: int) -> bool:
    """Phase 7 scale-up gate: minimum settled bets OR live calibration check."""
    from config import BETTING_CONFIG
    min_required = int(BETTING_CONFIG.get("scale_up_min_settled_bets", 50))
    return settled_count >= min_required


def get_strategy_breakdown() -> Dict[str, Any]:
    """Wave 5.8: per-strategy roll-up for /live-betting UI.

    Bankroll is wallet-driven: portfolio_value × allocation_weight per strategy.
    """
    from config import BETTING_CONFIG
    from src.data.database import get_connection
    from src.integrations.polymarket.live_bankroll import (
        get_portfolio_value,
        get_strategy_bankroll,
        get_strategy_open_by_kickoff_day_utc,
        get_strategy_open_cap_per_kickoff_day_usdc,
        get_strategy_open_cap_usdc,
        strategy_allocation_weight,
        uses_kickoff_day_open_cap,
    )

    live_strategies = list(BETTING_CONFIG.get("live_strategies", []) or [])
    default_cap = float(BETTING_CONFIG.get("max_deposit_per_strategy_usdc", 0))
    kickoff_day_mode = uses_kickoff_day_open_cap()

    conn = get_connection()
    try:
        portfolio = get_portfolio_value(conn)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT DISTINCT strategy_label
            FROM bet_ledger
            WHERE COALESCE(bet_kind, 'real') = 'real'
              AND strategy_label IS NOT NULL
            """
        )
        historical = [r["strategy_label"] for r in cur.fetchall()]
        all_labels = sorted(set(live_strategies) | set(historical))

        rows = []
        for label in all_labels:
            enabled = label in live_strategies
            weight = strategy_allocation_weight(label) if enabled else 0.0
            if enabled:
                current_bankroll = get_strategy_bankroll(label, conn)
                open_cap = get_strategy_open_cap_usdc(label, conn)
            else:
                # Retired strategies hold no wallet slice; keep historical P&L only.
                current_bankroll = 0.0
                open_cap = 0.0

            cur.execute(
                """
                SELECT
                    COUNT(*) AS n_total,
                    COALESCE(SUM(CASE WHEN status='settled' THEN 1 ELSE 0 END), 0) AS n_settled,
                    COALESCE(SUM(CASE WHEN status='filled' AND settled_at IS NULL THEN 1 ELSE 0 END), 0) AS n_open,
                    COALESCE(SUM(CASE WHEN status='errored' THEN 1 ELSE 0 END), 0) AS n_errored,
                    COALESCE(SUM(CASE WHEN filled_at IS NOT NULL AND settled_at IS NULL THEN fill_size_usdc ELSE 0 END), 0.0) AS open_exposure,
                    COALESCE(SUM(CASE WHEN status='settled' THEN pnl_realised_usdc ELSE 0 END), 0.0) AS realised_pnl,
                    COALESCE(SUM(CASE WHEN status='settled' AND pnl_realised_usdc > 0 THEN 1 ELSE 0 END), 0) AS n_wins
                FROM bet_ledger
                WHERE COALESCE(bet_kind, 'real') = 'real'
                  AND strategy_label = ?
                """,
                (label,),
            )
            row = cur.fetchone()
            n_settled = int(row["n_settled"] or 0)
            n_wins = int(row["n_wins"] or 0)
            open_exposure = float(row["open_exposure"] or 0.0)
            realised_pnl = float(row["realised_pnl"] or 0.0)
            open_by_kickoff_day: Dict[str, float] = {}
            kickoff_day_cap = 0.0
            if enabled and kickoff_day_mode:
                open_by_kickoff_day = get_strategy_open_by_kickoff_day_utc(label, conn)
                kickoff_day_cap = get_strategy_open_cap_per_kickoff_day_usdc(label, conn)
                open_cap = kickoff_day_cap
                if open_by_kickoff_day:
                    cap_remaining = min(
                        max(0.0, kickoff_day_cap - day_open)
                        for day_open in open_by_kickoff_day.values()
                    )
                else:
                    cap_remaining = kickoff_day_cap
            else:
                cap_remaining = max(0.0, open_cap - open_exposure) if open_cap > 0 else 0.0
            basis = max(0.0, current_bankroll - realised_pnl) if enabled else 0.0
            roi_pct = round(100.0 * realised_pnl / basis, 2) if enabled and basis > 0 else None
            rows.append({
                "strategy_label": label,
                "enabled": enabled,
                "retired": not enabled,
                "allocation_weight": round(weight, 4) if enabled else None,
                "portfolio_value_usdc": round(portfolio, 2),
                "starting_bankroll_usdc": round(basis, 2),
                "current_bankroll_usdc": round(current_bankroll, 2),
                "open_exposure_usdc": round(open_exposure, 2),
                "open_cap_usdc": round(open_cap, 2),
                "kickoff_day_cap_usdc": round(kickoff_day_cap, 2) if kickoff_day_mode and enabled else None,
                "open_by_kickoff_day_utc": open_by_kickoff_day if kickoff_day_mode and enabled else {},
                "cap_remaining_usdc": round(cap_remaining, 2),
                "realised_pnl_usdc": round(realised_pnl, 2),
                "roi_pct": roi_pct,
                "n_total": int(row["n_total"] or 0),
                "n_settled": n_settled,
                "n_open": int(row["n_open"] or 0),
                "n_errored": int(row["n_errored"] or 0),
                "win_rate": round(n_wins / n_settled, 3) if n_settled > 0 else None,
            })
    finally:
        conn.close()

    return {
        "strategies": rows,
        "portfolio_value_usdc": round(portfolio, 2),
        "max_deposit_per_strategy_usdc": default_cap,
        "live_strategies": live_strategies,
        "kickoff_day_open_cap_mode": kickoff_day_mode,
    }


def write_env_var(key: str, value: str, env_path: Optional[Path] = None) -> bool:
    """Persist a single env var to .env so it survives Flask restart.

    Used by kill-switch endpoint and mode-toggle endpoint. Idempotent;
    overwrites existing key if present.
    """
    if env_path is None:
        env_path = Path(".env")
    if not env_path.exists():
        logger.warning(f".env does not exist; not writing {key}={value}")
        return False
    lines = env_path.read_text().splitlines()
    found = False
    new_lines = []
    for line in lines:
        if line.startswith(f"{key}="):
            new_lines.append(f"{key}={value}")
            found = True
        else:
            new_lines.append(line)
    if not found:
        new_lines.append(f"{key}={value}")
    env_path.write_text("\n".join(new_lines) + "\n")
    # Live-update the in-process value too
    os.environ[key] = value
    return True


def engage_kill_switch() -> Dict[str, Any]:
    """Flip kill switch ON in both env and live config."""
    from config import BETTING_CONFIG
    BETTING_CONFIG["kill_switch"] = True
    persisted = write_env_var("BETTING_KILL_SWITCH", "1")
    logger.warning("KILL SWITCH ENGAGED — all betting blocked")
    return {"kill_switch": True, "persisted_to_env": persisted}


def disengage_kill_switch() -> Dict[str, Any]:
    from config import BETTING_CONFIG
    BETTING_CONFIG["kill_switch"] = False
    persisted = write_env_var("BETTING_KILL_SWITCH", "0")
    logger.warning("Kill switch released — betting will resume per BETTING_MODE")
    return {"kill_switch": False, "persisted_to_env": persisted}


def set_mode(new_mode: str) -> Dict[str, Any]:
    """Toggle between OFF / MANUAL / AUTO."""
    from config import BETTING_CONFIG
    new_mode_upper = new_mode.upper()
    if new_mode_upper not in ("OFF", "MANUAL", "AUTO"):
        return {"success": False, "error": f"Invalid mode '{new_mode}'. Use OFF / MANUAL / AUTO."}
    BETTING_CONFIG["mode"] = new_mode_upper
    persisted = write_env_var("BETTING_MODE", new_mode_upper)
    logger.info(f"Betting mode set to {new_mode_upper}")
    return {"success": True, "mode": new_mode_upper, "persisted_to_env": persisted}
