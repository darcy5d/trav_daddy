"""XI-aware rebalancing: decide how to move live exposure toward the fresh
Kelly target as updated CREX lineups change the model.

This module is intentionally side-effect free for the decision path: it reads
the ledger / order plans to measure current exposure and returns a structured
`RebalanceAction`. The caller (the live scanner) executes the action using its
existing TWAP/FOK placement code (for adds) and `bet_placement.reduce_position`
(for reduces/exits), so order routing stays in one place.

Exposure is measured in ENTRY-COST USDC (the originally-staked dollars still on
the table), which is exactly what `reduce_position` decrements and what the risk
gate sums, so the numbers compose cleanly.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.integrations.polymarket.paper_strategies import PaperStrategy
from src.integrations.polymarket.sizing import live_scaled_kelly_stake

logger = logging.getLogger(__name__)


def _rebalance_config() -> Dict[str, Any]:
    from config import BETTING_CONFIG
    return BETTING_CONFIG


def rebalance_enabled() -> bool:
    return bool(_rebalance_config().get("rebalance_enabled", False))


# ---------------------------------------------------------------------------
# Exposure measurement
# ---------------------------------------------------------------------------

def current_filled_exposure(
    conn: sqlite3.Connection,
    strategy_label: str,
    fixture_key: str,
    side_label: str,
    phase: str = "pre_toss",
) -> float:
    """Entry-cost USDC currently held (filled, unsettled) on this side.

    `reduce_position` decrements `fill_size_usdc` on the BUY rows as it sells,
    so this value is already net of any rebalance exits.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COALESCE(SUM(fill_size_usdc), 0.0) AS total
        FROM bet_ledger
        WHERE COALESCE(bet_kind, 'real') = 'real'
          AND side = 'BUY'
          AND status = 'filled'
          AND strategy_label = ?
          AND fixture_key = ?
          AND side_label = ?
          AND COALESCE(phase, 'pre_toss') = ?
          AND fill_size_usdc IS NOT NULL
        """,
        (strategy_label, fixture_key, side_label, phase),
    )
    return float(cur.fetchone()["total"] or 0.0)


def pending_twap_exposure(
    conn: sqlite3.Connection,
    strategy_label: str,
    fixture_key: str,
    token_id: Optional[str] = None,
) -> float:
    """Queued-but-unfilled USDC from active TWAP plans for this fixture.

    Counts pending/placed chunks of pending/executing plans so a top-up
    doesn't double-stack on exposure already on its way to the book. Scoped
    by token_id when provided (one outcome side), else by fixture/strategy.
    """
    cur = conn.cursor()
    params: List[Any] = [strategy_label, fixture_key]
    token_clause = ""
    if token_id:
        token_clause = " AND op.token_id = ?"
        params.append(token_id)
    try:
        cur.execute(
            f"""
            SELECT COALESCE(SUM(oc.size_usdc), 0.0) AS total
            FROM order_chunks oc
            JOIN order_plans op ON op.plan_id = oc.plan_id
            WHERE op.strategy_label = ?
              AND op.fixture_key = ?
              AND op.side = 'BUY'
              AND op.status IN ('pending', 'executing')
              AND oc.status IN ('pending', 'placed')
              {token_clause}
            """,
            params,
        )
        return float(cur.fetchone()["total"] or 0.0)
    except Exception:
        return 0.0


def current_exposure(
    conn: sqlite3.Connection,
    strategy_label: str,
    fixture_key: str,
    side_label: str,
    phase: str = "pre_toss",
    token_id: Optional[str] = None,
) -> float:
    """Total live exposure on a side: filled holdings + queued TWAP chunks."""
    return (
        current_filled_exposure(conn, strategy_label, fixture_key, side_label, phase)
        + pending_twap_exposure(conn, strategy_label, fixture_key, token_id)
    )


def held_sides(
    conn: sqlite3.Connection,
    strategy_label: str,
    fixture_key: str,
    phase: str = "pre_toss",
) -> List[Tuple[str, float]]:
    """All sides this strategy currently holds (filled) exposure on."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT side_label, COALESCE(SUM(fill_size_usdc), 0.0) AS total
        FROM bet_ledger
        WHERE COALESCE(bet_kind, 'real') = 'real'
          AND side = 'BUY'
          AND status = 'filled'
          AND strategy_label = ?
          AND fixture_key = ?
          AND COALESCE(phase, 'pre_toss') = ?
          AND fill_size_usdc IS NOT NULL
        GROUP BY side_label
        HAVING total > 0
        """,
        (strategy_label, fixture_key, phase),
    )
    return [(r["side_label"], float(r["total"])) for r in cur.fetchall()]


def bet_time_edge_pp(
    conn: sqlite3.Connection,
    strategy_label: str,
    fixture_key: str,
    side_label: str,
    phase: str = "pre_toss",
) -> Optional[float]:
    """Edge (pp) recorded on the earliest non-errored bet we hold on this side."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT edge_pp
        FROM bet_ledger
        WHERE COALESCE(bet_kind, 'real') = 'real'
          AND side = 'BUY'
          AND status != 'errored'
          AND strategy_label = ?
          AND fixture_key = ?
          AND side_label = ?
          AND COALESCE(phase, 'pre_toss') = ?
        ORDER BY proposed_at ASC
        LIMIT 1
        """,
        (strategy_label, fixture_key, side_label, phase),
    )
    row = cur.fetchone()
    if row is None or row["edge_pp"] is None:
        return None
    return float(row["edge_pp"])


def count_adjustments(
    conn: sqlite3.Connection,
    strategy_label: str,
    fixture_key: str,
    phase: str = "pre_toss",
) -> int:
    """How many rebalance actions have we already taken on this fixture.

    Counts rebalance SELL rows plus any BUY rows beyond the first (extra
    top-ups), as a churn guard against runaway re-sizing.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            COALESCE(SUM(CASE WHEN side = 'SELL' AND cancel_reason = 'rebalance' THEN 1 ELSE 0 END), 0) AS sells,
            COALESCE(SUM(CASE WHEN side = 'BUY' AND status != 'errored' THEN 1 ELSE 0 END), 0) AS buys
        FROM bet_ledger
        WHERE COALESCE(bet_kind, 'real') = 'real'
          AND strategy_label = ?
          AND fixture_key = ?
          AND COALESCE(phase, 'pre_toss') = ?
        """,
        (strategy_label, fixture_key, phase),
    )
    row = cur.fetchone()
    sells = int(row["sells"] or 0)
    buys = int(row["buys"] or 0)
    return sells + max(0, buys - 1)


# ---------------------------------------------------------------------------
# Target sizing
# ---------------------------------------------------------------------------

def target_exposure(
    model_prob: float,
    market_price: float,
    bankroll: float,
    strat: PaperStrategy,
    edge_pp: Optional[float] = None,
) -> float:
    """Fresh Kelly target for this side; 0 when the edge no longer qualifies."""
    if edge_pp is None:
        edge_pp = (model_prob - market_price) * 100.0
    if edge_pp < strat.min_edge_pp:
        return 0.0
    if not (strat.min_market_price <= market_price <= strat.max_market_price):
        return 0.0
    if strat.min_model_prob is not None and model_prob < strat.min_model_prob:
        return 0.0
    if strat.max_model_prob is not None and model_prob > strat.max_model_prob:
        return 0.0
    return live_scaled_kelly_stake(model_prob, market_price, bankroll, strat)


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

@dataclass
class RebalanceAction:
    action: str  # 'hold' | 'add' | 'reduce' | 'exit_flip'
    reason: str
    size_usdc: float = 0.0           # add: USDC to buy; reduce: entry-cost to sell
    current_exposure: float = 0.0
    target_exposure: float = 0.0
    # For exit_flip: sides to fully sell before adding to the chosen side.
    exits: List[Tuple[str, float]] = field(default_factory=list)


def decide_rebalance(
    conn: sqlite3.Connection,
    *,
    strat: PaperStrategy,
    fixture_key: str,
    chosen_side_label: str,
    model_prob: float,
    market_price: float,
    edge_pp: float,
    bankroll: float,
    minutes_to_kickoff: Optional[float],
    phase: str = "pre_toss",
) -> RebalanceAction:
    """Decide the rebalance action for one (strategy, fixture, chosen side).

    Pure read-only decision: measures current exposure, computes the fresh
    Kelly target, applies edge-delta / dollar-delta / churn / freeze guards,
    and returns a structured action for the caller to execute.
    """
    cfg = _rebalance_config()
    edge_delta_pp = float(cfg.get("rebalance_edge_delta_pp", 1.5))
    min_delta_frac = float(cfg.get("rebalance_min_delta_frac", 0.20))
    max_per_fixture = int(cfg.get("rebalance_max_per_fixture", 6))
    freeze_min = float(cfg.get("rebalance_freeze_min_before_toss", 20))

    frozen = minutes_to_kickoff is not None and minutes_to_kickoff <= freeze_min

    current = current_exposure(conn, strat.name, fixture_key, chosen_side_label, phase)
    target = target_exposure(model_prob, market_price, bankroll, strat, edge_pp)

    # Churn guard.
    if count_adjustments(conn, strat.name, fixture_key, phase) >= max_per_fixture:
        return RebalanceAction(
            action="hold", reason="max-rebalances-per-fixture-reached",
            current_exposure=current, target_exposure=target,
        )

    # Edge-flip handling: sell any side we hold that is NOT the chosen side.
    wrong_sides = [
        (sl, exp) for sl, exp in held_sides(conn, strat.name, fixture_key, phase)
        if sl != chosen_side_label and exp > 0
    ]
    if wrong_sides:
        if frozen:
            return RebalanceAction(
                action="hold",
                reason=f"edge-flip-but-frozen ({minutes_to_kickoff:.0f}m to toss)",
                current_exposure=current, target_exposure=target,
            )
        return RebalanceAction(
            action="exit_flip",
            reason="model flipped side; exit stale side(s) then re-enter",
            size_usdc=target,  # desired exposure on the new chosen side
            current_exposure=current, target_exposure=target,
            exits=wrong_sides,
        )

    # No position yet: defer to the normal first-entry path (returns 'add'
    # only if there's a meaningful target; the scanner's dedup/risk gate
    # still governs the very first bet).
    if current <= 0:
        if target <= 0:
            return RebalanceAction(
                action="hold", reason="no-position-no-target",
                current_exposure=current, target_exposure=target,
            )
        return RebalanceAction(
            action="add", reason="open-initial-position", size_usdc=target,
            current_exposure=current, target_exposure=target,
        )

    # Edge-delta guard: only churn an existing position when the model edge
    # has actually moved enough since we bet.
    bet_edge = bet_time_edge_pp(conn, strat.name, fixture_key, chosen_side_label, phase)
    if bet_edge is not None and abs(edge_pp - bet_edge) < edge_delta_pp:
        return RebalanceAction(
            action="hold",
            reason=f"edge moved <{edge_delta_pp}pp since bet ({bet_edge:.1f}->{edge_pp:.1f})",
            current_exposure=current, target_exposure=target,
        )

    delta = target - current
    ref = max(target, current, 1e-9)
    if abs(delta) / ref < min_delta_frac:
        return RebalanceAction(
            action="hold",
            reason=f"|delta| {abs(delta):.2f} < {min_delta_frac:.0%} of {ref:.2f}",
            current_exposure=current, target_exposure=target,
        )

    if delta > 0:
        return RebalanceAction(
            action="add",
            reason=f"edge grew; top up ${delta:.2f} (cur ${current:.2f} -> tgt ${target:.2f})",
            size_usdc=round(delta, 2),
            current_exposure=current, target_exposure=target,
        )

    # delta < 0 -> reduce. Selling is frozen near the toss.
    if frozen:
        return RebalanceAction(
            action="hold",
            reason=f"reduce wanted but frozen ({minutes_to_kickoff:.0f}m to toss)",
            current_exposure=current, target_exposure=target,
        )
    return RebalanceAction(
        action="reduce",
        reason=f"edge shrank; sell ${abs(delta):.2f} (cur ${current:.2f} -> tgt ${target:.2f})",
        size_usdc=round(abs(delta), 2),
        current_exposure=current, target_exposure=target,
    )
