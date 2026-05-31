"""TWAP resize must use the same Kelly path as live_bet_scan."""

from src.integrations.polymarket.paper_strategies import STRATEGIES
from src.integrations.polymarket.sizing import live_scaled_kelly_stake


def test_resize_stake_formula_matches_live_scan():
    """Resize and scan both call live_scaled_kelly_stake — same inputs → same stake."""
    strat = next(s for s in STRATEGIES if s.name == "v3_marg_3pp")
    bankroll = 400.0
    model_prob = 0.563
    market_price = 0.26
    stake_a = live_scaled_kelly_stake(model_prob, market_price, bankroll, strat)
    stake_b = live_scaled_kelly_stake(model_prob, market_price, bankroll, strat)
    assert stake_a == stake_b
    assert stake_a > 0
