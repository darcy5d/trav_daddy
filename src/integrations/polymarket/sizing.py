"""Live Kelly stake sizing — shared by live_bet_scan and the bets API."""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional

from src.integrations.polymarket.paper_strategies import PaperStrategy, get_strategy

POLYMARKET_MIN_ORDER_USDC = 1.0
LIVE_MIN_STAKE_FRACTION = 0.005
LIVE_MAX_STAKE_FRACTION = 0.25

# Polymarket fixture-key prefix for international fixtures. Associate-nation
# internationals (where either side is not a Tier-1 Full Member) are the
# low-data matchups we down-size. Franchise / domestic leagues use other
# prefixes (cricipl, crict20blast, ...) and are unaffected.
INTERNATIONAL_PREFIX = "crint"
# teams.tier value for ICC Full Member nations.
FULL_MEMBER_TIER = 1


def _model_prob_cap() -> float:
    """Sizing ceiling on model_prob (lazy import so .env changes are picked up)."""
    try:
        from config import BETTING_CONFIG
        return float(BETTING_CONFIG.get("model_prob_cap", 0.95))
    except Exception:  # pragma: no cover - config always present in practice
        return 0.95


def _associate_kelly_mult() -> float:
    """Fractional Kelly applied to associate-nation internationals."""
    try:
        from config import BETTING_CONFIG
        return float(BETTING_CONFIG.get("associate_kelly_mult", 0.05))
    except Exception:  # pragma: no cover
        return 0.05


def _associate_throttle_enabled() -> bool:
    """Master switch for the associate-league Kelly throttle.

    Defaults OFF: the 60-day ledger review found associate-nation internationals
    were the most profitable live segment in the clean window, so they are sized
    at the strategy's normal kelly_mult unless this flag is explicitly armed.
    """
    try:
        from config import BETTING_CONFIG
        return bool(BETTING_CONFIG.get("associate_throttle_enabled", False))
    except Exception:  # pragma: no cover
        return False


def get_team_tier(conn: sqlite3.Connection, team_id: Optional[int]) -> Optional[int]:
    """Read teams.tier for a team_id; None if unknown/unmatched."""
    if team_id is None:
        return None
    try:
        cur = conn.cursor()
        cur.execute("SELECT tier FROM teams WHERE team_id = ?", (team_id,))
        row = cur.fetchone()
    except Exception:
        return None
    if row is None:
        return None
    tier = row[0] if not isinstance(row, sqlite3.Row) else row["tier"]
    try:
        return int(tier) if tier is not None else None
    except (TypeError, ValueError):
        return None


def is_low_data_fixture(
    tournament_prefix: Optional[str],
    team1_tier: Optional[int],
    team2_tier: Optional[int],
) -> bool:
    """Associate-nation international detection.

    True iff the fixture is an international (`crint`) AND it is NOT a
    Tier-1-vs-Tier-1 (Full Member) matchup. An unknown/NULL tier on a `crint`
    fixture is treated as non-Tier-1 (conservative: covers development sides
    like "New Zealand A" that don't tier-match). Non-international fixtures
    (county, franchise leagues) are never low-data here.
    """
    if (tournament_prefix or "").lower() != INTERNATIONAL_PREFIX:
        return False
    both_full_members = (team1_tier == FULL_MEMBER_TIER) and (team2_tier == FULL_MEMBER_TIER)
    return not both_full_members


def effective_kelly_mult(
    base_mult: float,
    tournament_prefix: Optional[str],
    team1_tier: Optional[int],
    team2_tier: Optional[int],
    assoc_mult: Optional[float] = None,
) -> float:
    """Return the Kelly multiplier to use for a fixture.

    When the associate throttle is armed, associate-nation internationals get
    the (smaller) associate multiplier; everything else (and everything when the
    throttle is disabled) keeps the strategy's base ``kelly_mult``.
    """
    if not _associate_throttle_enabled():
        return base_mult
    if is_low_data_fixture(tournament_prefix, team1_tier, team2_tier):
        return _associate_kelly_mult() if assoc_mult is None else assoc_mult
    return base_mult


def _kelly_fraction(
    model_prob: float,
    market_price: float,
    strategy: PaperStrategy,
    kelly_mult_override: Optional[float] = None,
) -> float:
    if market_price <= 0 or market_price >= 1:
        return 0.0
    if model_prob <= 0 or model_prob >= 1:
        return 0.0
    # Never size off a simulator "certainty"; clamp before computing f*.
    model_prob = min(model_prob, _model_prob_cap())
    f_star = (model_prob - market_price) / (1.0 - market_price)
    f_star = max(0.0, min(f_star, 1.0))
    mult = kelly_mult_override if kelly_mult_override is not None else strategy.kelly_mult
    return min(f_star * mult, strategy.kelly_fraction_cap)


def live_scaled_kelly_stake(
    model_prob: float,
    market_price: float,
    live_bankroll_usdc: float,
    strategy: PaperStrategy,
    kelly_mult_override: Optional[float] = None,
) -> float:
    """Fractional-Kelly stake sized relative to the live bankroll.

    The Kelly fraction is scaled by ``strategy.kelly_mult`` (e.g. 0.5 = half
    Kelly for the live strategies), or by ``kelly_mult_override`` when provided
    (e.g. the 5% associate-league throttle), and capped by
    ``strategy.kelly_fraction_cap``, then the resulting dollar stake is clamped
    to LIVE_MAX_STAKE_FRACTION (25%) of bankroll. Stakes below
    max(LIVE_MIN_STAKE_FRACTION, min-order) return 0.
    """
    f_capped = _kelly_fraction(model_prob, market_price, strategy, kelly_mult_override)
    raw_stake = f_capped * live_bankroll_usdc
    scaled_min = max(
        LIVE_MIN_STAKE_FRACTION * live_bankroll_usdc,
        POLYMARKET_MIN_ORDER_USDC,
    )
    scaled_max = LIVE_MAX_STAKE_FRACTION * live_bankroll_usdc
    if raw_stake < scaled_min:
        return 0.0
    return min(raw_stake, scaled_max)


def compute_sizing_context(
    bet: Dict[str, Any],
    risk_gate_max_per_bet: Optional[float] = None,
    kelly_mult_override: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Derive Kelly / cap / fill sizing notes for display on a bet row.

    ``kelly_mult_override`` lets the caller reflect a per-fixture Kelly
    multiplier (e.g. the associate-league throttle) in the displayed Kelly
    figures; when None the strategy's normal ``kelly_mult`` is used.
    """
    bankroll = bet.get("bankroll_at_proposal")
    model_prob = bet.get("model_prob")
    market_price = bet.get("market_price_at_proposal")
    strategy_label = bet.get("strategy_label")
    if bankroll is None or model_prob is None or market_price is None:
        return None

    strat = get_strategy(strategy_label) if strategy_label else None
    if strat is None:
        return None

    try:
        bankroll_f = float(bankroll)
        model_f = float(model_prob)
        market_f = float(market_price)
    except (TypeError, ValueError):
        return None

    if bankroll_f <= 0:
        return None

    eff_mult = kelly_mult_override if kelly_mult_override is not None else strat.kelly_mult
    f_capped = _kelly_fraction(model_f, market_f, strat, kelly_mult_override)
    kelly_stake = f_capped * bankroll_f
    kelly_capped_stake = live_scaled_kelly_stake(
        model_f, market_f, bankroll_f, strat, kelly_mult_override
    )

    # Uncapped Kelly (before kelly_fraction_cap, still after the effective mult)
    f_star = max(0.0, min((model_f - market_f) / (1.0 - market_f), 1.0))
    uncapped_fraction = f_star * eff_mult
    uncapped_stake = uncapped_fraction * bankroll_f

    proposed = float(bet.get("size_usdc") or 0)
    filled = bet.get("fill_size_usdc")
    filled_f = float(filled) if filled is not None else None

    notes: List[str] = []
    cap_pct = int(strat.kelly_fraction_cap * 100)

    if uncapped_stake > kelly_stake + 0.01:
        notes.append(
            f"Would have bet ${uncapped_stake:.2f} without {cap_pct}% Kelly cap"
        )

    if kelly_stake > kelly_capped_stake + 0.01:
        notes.append(
            f"Kelly ${kelly_stake:.2f} → capped at {cap_pct}% bankroll "
            f"(${kelly_capped_stake:.2f})"
        )

    binding_cap = kelly_capped_stake
    if risk_gate_max_per_bet is not None and risk_gate_max_per_bet > 0:
        if proposed <= risk_gate_max_per_bet + 0.01 and kelly_capped_stake > risk_gate_max_per_bet + 0.01:
            notes.append(f"Risk gate per-bet cap ${risk_gate_max_per_bet:.2f}")
            binding_cap = min(binding_cap, risk_gate_max_per_bet)

    if proposed > 0 and abs(proposed - binding_cap) > 0.05 and not notes:
        notes.append(f"Proposed ${proposed:.2f} (Kelly ${kelly_capped_stake:.2f})")

    if filled_f is not None and proposed > 0:
        ratio = filled_f / proposed
        if ratio > 1.05:
            notes.append(f"Filled above proposal (TWAP liquidity +{((ratio - 1) * 100):.0f}%)")
        elif ratio < 0.95:
            notes.append(f"Filled below proposal ({(ratio * 100):.0f}% of proposed)")

    return {
        "kelly_raw_stake": round(kelly_stake, 2),
        "kelly_uncapped_stake": round(uncapped_stake, 2),
        "kelly_capped_stake": round(kelly_capped_stake, 2),
        "sizing_notes": notes,
    }
