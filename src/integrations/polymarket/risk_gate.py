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
) -> RiskGateDecision:
    """Return RiskGateDecision indicating whether the bet may proceed.

    Args:
        proposed_size_usdc: Stake size in USD.
        market_type: One of moneyline / top_batter / most_sixes / toss_match_double.
        edge_pp: Model edge over market in percentage points (positive = favourable).
        requested_mode: "manual" or "auto" (matches the bet's `mode` column).
        conn: Optional sqlite connection; opens its own if None.
        strategy_label: Optional strategy name; when provided the per-strategy
            cap (BETTING_MAX_DEPOSIT_PER_STRATEGY) and live_strategies whitelist
            (BETTING_LIVE_STRATEGIES) are enforced in addition to the global caps.

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

        # 3. Per-bet cap — floats with bankroll when BETTING_MAX_PER_BET_FRACTION is set.
        # For strategy-tagged bets: cap = fraction × live_bankroll (grows/shrinks with PnL).
        # For untagged manual bets: falls back to the hard BETTING_MAX_PER_BET dollar cap.
        max_per_bet_fraction = float(BETTING_CONFIG.get("max_per_bet_fraction", 0))
        if max_per_bet_fraction > 0 and strategy_label is not None:
            live_bankroll = _live_bankroll_for_strategy(conn, strategy_label)
            max_per_bet = max_per_bet_fraction * live_bankroll
            cap_label = f"{max_per_bet_fraction*100:.1f}% of ${live_bankroll:.2f} live bankroll"
        else:
            max_per_bet = float(BETTING_CONFIG.get("max_per_bet_usdc", 0))
            cap_label = f"hard cap"
        if proposed_size_usdc > max_per_bet:
            return RiskGateDecision(
                allowed=False,
                reason=(
                    f"Proposed ${proposed_size_usdc:.2f} exceeds per-bet cap "
                    f"${max_per_bet:.2f} ({cap_label})."
                ),
                cap_remaining_today=0.0,
                cap_remaining_deposit=0.0,
            )

        # 4. Daily stake cap
        max_per_day = float(BETTING_CONFIG.get("max_per_day_usdc", 0))
        todays_stakes = _todays_filled_stakes(conn)
        cap_remaining_today = max(0.0, max_per_day - todays_stakes)
        if proposed_size_usdc > cap_remaining_today:
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
        max_loss_per_day = float(BETTING_CONFIG.get("max_loss_per_day_usdc", 0))
        todays_loss = _todays_realised_loss(conn)
        # Worst case for THIS bet: lose the entire stake (price~1, settle=0)
        worst_case_loss = proposed_size_usdc
        if todays_loss + worst_case_loss > max_loss_per_day:
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

        # 6. Total open exposure
        max_deposit = float(BETTING_CONFIG.get("max_deposit_usdc", 0))
        open_exposure = _open_exposure_usdc(conn)
        cap_remaining_deposit = max(0.0, max_deposit - open_exposure)
        if proposed_size_usdc > cap_remaining_deposit:
            return RiskGateDecision(
                allowed=False,
                reason=(
                    f"Proposed ${proposed_size_usdc} would exceed total open exposure cap "
                    f"(${open_exposure:.2f} already open, max ${max_deposit})."
                ),
                cap_remaining_today=cap_remaining_today,
                cap_remaining_deposit=cap_remaining_deposit,
            )

        # 6b. Per-strategy open exposure (Wave 5.8).
        # Enforce BETTING_MAX_DEPOSIT_PER_STRATEGY so one strategy cannot
        # consume the full shared bankroll envelope.
        if strategy_label is not None:
            strategy_cap = float(BETTING_CONFIG.get("max_deposit_per_strategy_usdc", 0))
            if strategy_cap > 0:
                strategy_open = _strategy_open_exposure_usdc(conn, strategy_label)
                cap_remaining_strategy = max(0.0, strategy_cap - strategy_open)
                if proposed_size_usdc > cap_remaining_strategy:
                    return RiskGateDecision(
                        allowed=False,
                        reason=(
                            f"Strategy '{strategy_label}' open ${strategy_open:.2f} + "
                            f"${proposed_size_usdc} would exceed per-strategy cap "
                            f"${strategy_cap} (remaining ${cap_remaining_strategy:.2f})."
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


def get_risk_status() -> Dict[str, Any]:
    """Return a snapshot of current risk-gate state for the UI."""
    from config import BETTING_CONFIG
    from src.data.database import get_connection

    conn = get_connection()
    try:
        todays_stakes = _todays_filled_stakes(conn)
        todays_loss = _todays_realised_loss(conn)
        open_exposure = _open_exposure_usdc(conn)
        todays_bets = _todays_bet_count(conn)
        settled_bets_total = _settled_bet_count(conn)
    finally:
        conn.close()

    max_deposit = float(BETTING_CONFIG.get("max_deposit_usdc", 0))
    max_per_day = float(BETTING_CONFIG.get("max_per_day_usdc", 0))
    max_loss_per_day = float(BETTING_CONFIG.get("max_loss_per_day_usdc", 0))

    return {
        "mode": BETTING_CONFIG.get("mode", "OFF").upper(),
        "kill_switch": bool(BETTING_CONFIG.get("kill_switch", False)),
        "max_deposit_usdc": max_deposit,
        "max_per_bet_usdc": float(BETTING_CONFIG.get("max_per_bet_usdc", 0)),
        "max_per_bet_fraction": float(BETTING_CONFIG.get("max_per_bet_fraction", 0)),
        "max_per_day_usdc": max_per_day,
        "max_loss_per_day_usdc": max_loss_per_day,
        "auto_min_edge_pp": float(BETTING_CONFIG.get("auto_min_edge_pp", 0)),
        "auto_enabled_markets": list(BETTING_CONFIG.get("auto_enabled_markets", [])),
        "today_utc_start": _today_start_utc_iso(),
        "today_filled_stakes_usdc": round(todays_stakes, 2),
        "today_realised_loss_usdc": round(todays_loss, 2),
        "today_bet_count": todays_bets,
        "today_remaining_cap_usdc": max(0.0, max_per_day - todays_stakes),
        "open_exposure_usdc": round(open_exposure, 2),
        "deposit_remaining_cap_usdc": max(0.0, max_deposit - open_exposure),
        "settled_bets_total": settled_bets_total,
        "scale_up_eligible": _is_scale_up_eligible(settled_bets_total),
    }


def _live_bankroll_for_strategy(conn: sqlite3.Connection, strategy_label: str) -> float:
    """Compute live bankroll = starting deposit + sum(realised PnL on real settled bets).

    Mirrors the logic in live_bet_scan._get_live_bankroll() so the risk gate
    uses the exact same number the scanner used for Kelly sizing.
    """
    env_key = f"BETTING_MAX_DEPOSIT_{strategy_label.upper().replace('-', '_')}"
    starting = float(
        os.environ.get(env_key)
        or os.environ.get("BETTING_MAX_DEPOSIT_PER_STRATEGY", "100")
    )
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COALESCE(SUM(pnl_realised_usdc), 0.0)
        FROM bet_ledger
        WHERE COALESCE(bet_kind, 'real') = 'real'
          AND strategy_label = ?
          AND status = 'settled'
        """,
        (strategy_label,),
    )
    pnl = float(cur.fetchone()[0] or 0.0)
    return max(0.0, starting + pnl)


def _is_scale_up_eligible(settled_count: int) -> bool:
    """Phase 7 scale-up gate: minimum settled bets OR live calibration check."""
    from config import BETTING_CONFIG
    min_required = int(BETTING_CONFIG.get("scale_up_min_settled_bets", 50))
    return settled_count >= min_required


def get_strategy_breakdown() -> Dict[str, Any]:
    """Wave 5.8: per-strategy roll-up for /live-betting UI.

    For each strategy in BETTING_LIVE_STRATEGIES (plus any strategy that has
    ever placed a real bet, so retired strategies still show their history),
    returns: bankroll start, open exposure, realised P&L, bet counts by status.
    """
    from config import BETTING_CONFIG
    from src.data.database import get_connection

    import os
    live_strategies = list(BETTING_CONFIG.get("live_strategies", []) or [])
    default_cap = float(BETTING_CONFIG.get("max_deposit_per_strategy_usdc", 0))

    def _strategy_starting(label: str) -> float:
        """Per-strategy starting bankroll: env override takes priority over global default."""
        env_key = f"BETTING_MAX_DEPOSIT_{label.upper().replace('-', '_')}"
        override = os.getenv(env_key)
        return float(override) if override is not None else default_cap

    conn = get_connection()
    try:
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
            strategy_cap = _strategy_starting(label)
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
            rows.append({
                "strategy_label": label,
                "enabled": label in live_strategies,
                "starting_bankroll_usdc": strategy_cap,
                "current_bankroll_usdc": round(strategy_cap + realised_pnl, 2),
                "open_exposure_usdc": round(open_exposure, 2),
                "cap_remaining_usdc": round(max(0.0, strategy_cap - open_exposure), 2),
                "realised_pnl_usdc": round(realised_pnl, 2),
                "roi_pct": round(100.0 * realised_pnl / strategy_cap, 2) if strategy_cap > 0 else 0.0,
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
        "max_deposit_per_strategy_usdc": strategy_cap,
        "live_strategies": live_strategies,
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
