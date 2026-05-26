"""Live Kelly stake sizing — shared by live_bet_scan and the bets API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.integrations.polymarket.paper_strategies import PaperStrategy, get_strategy

POLYMARKET_MIN_ORDER_USDC = 1.0
LIVE_MIN_STAKE_FRACTION = 0.005
LIVE_MAX_STAKE_FRACTION = 0.25


def _kelly_fraction(model_prob: float, market_price: float, strategy: PaperStrategy) -> float:
    if market_price <= 0 or market_price >= 1:
        return 0.0
    if model_prob <= 0 or model_prob >= 1:
        return 0.0
    f_star = (model_prob - market_price) / (1.0 - market_price)
    f_star = max(0.0, min(f_star, 1.0))
    return min(f_star * strategy.kelly_mult, strategy.kelly_fraction_cap)


def live_scaled_kelly_stake(
    model_prob: float,
    market_price: float,
    live_bankroll_usdc: float,
    strategy: PaperStrategy,
) -> float:
    """Half-Kelly stake sized relative to the live bankroll."""
    f_capped = _kelly_fraction(model_prob, market_price, strategy)
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
) -> Optional[Dict[str, Any]]:
    """Derive Kelly / cap / fill sizing notes for display on a bet row."""
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

    f_capped = _kelly_fraction(model_f, market_f, strat)
    kelly_stake = f_capped * bankroll_f
    kelly_capped_stake = live_scaled_kelly_stake(model_f, market_f, bankroll_f, strat)

    # Uncapped Kelly (before kelly_fraction_cap, still after half-Kelly mult)
    f_star = max(0.0, min((model_f - market_f) / (1.0 - market_f), 1.0))
    uncapped_fraction = f_star * strat.kelly_mult
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
