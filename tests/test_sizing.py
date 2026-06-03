"""Unit tests for live Kelly sizing (src/integrations/polymarket/sizing.py):
the model_prob overconfidence clamp and the associate-league Kelly throttle.
"""

from __future__ import annotations

import pytest

from src.integrations.polymarket.paper_strategies import STRATEGIES
from src.integrations.polymarket import sizing


# v2_diag_2pp: kelly_mult=0.25, kelly_fraction_cap=0.25 — low enough mult that
# the fraction cap doesn't mask the model_prob clamp at the prices used below.
DIAG = next(s for s in STRATEGIES if s.name == "v2_diag_2pp")


# --------------------------------------------------------------------------
# model_prob overconfidence clamp
# --------------------------------------------------------------------------

def test_model_prob_clamped_to_cap():
    """A 0.99 'near-certainty' is sized as if it were the cap (0.95), not 0.99."""
    cap = sizing._model_prob_cap()
    assert cap == pytest.approx(0.95)
    # Market high enough that f_star * mult stays under the fraction cap, so the
    # binding constraint is the model_prob clamp (not kelly_fraction_cap).
    f_overconfident = sizing._kelly_fraction(0.99, 0.90, DIAG)
    f_at_cap = sizing._kelly_fraction(0.95, 0.90, DIAG)
    assert f_overconfident == pytest.approx(f_at_cap)
    # And it is strictly smaller than the unclamped 0.99 fraction would be.
    unclamped = min(((0.99 - 0.90) / (1.0 - 0.90)) * DIAG.kelly_mult, DIAG.kelly_fraction_cap)
    assert f_overconfident < unclamped


def test_clamp_flows_through_to_stake():
    s_overconfident = sizing.live_scaled_kelly_stake(0.99, 0.90, 1000.0, DIAG)
    s_at_cap = sizing.live_scaled_kelly_stake(0.95, 0.90, 1000.0, DIAG)
    assert s_overconfident == pytest.approx(s_at_cap)


def test_prob_one_is_no_bet():
    # An exact 1.0 still trips the degenerate guard (returns 0).
    assert sizing._kelly_fraction(1.0, 0.50, DIAG) == 0.0


# --------------------------------------------------------------------------
# associate-league detection + effective Kelly multiplier
# --------------------------------------------------------------------------

def test_is_low_data_fixture_associate_internationals():
    # crint with a non-Tier-1 side -> low data.
    assert sizing.is_low_data_fixture("crint", 5, 2) is True   # Bahrain v Hong Kong
    assert sizing.is_low_data_fixture("crint", 1, 2) is True   # England v Netherlands
    assert sizing.is_low_data_fixture("crint", None, None) is True  # unknown -> conservative


def test_is_low_data_fixture_full_member_internationals():
    # crint Tier-1 v Tier-1 -> NOT low data.
    assert sizing.is_low_data_fixture("crint", 1, 1) is False  # England v India


def test_is_low_data_fixture_non_internationals():
    # County / franchise leagues are never low-data here regardless of tier.
    assert sizing.is_low_data_fixture("crict20blast", 4, 4) is False  # Sussex v Hampshire
    assert sizing.is_low_data_fixture("cricipl", 5, 5) is False
    assert sizing.is_low_data_fixture(None, 5, 5) is False


def test_effective_kelly_mult_throttles_associates_only():
    assoc = sizing._associate_kelly_mult()
    assert assoc == pytest.approx(0.05)
    # Associate internationals -> throttled to the associate mult.
    assert sizing.effective_kelly_mult(0.5, "crint", 5, 2) == pytest.approx(assoc)
    assert sizing.effective_kelly_mult(0.5, "crint", None, None) == pytest.approx(assoc)
    # Full-member internationals and domestic leagues -> keep the base mult.
    assert sizing.effective_kelly_mult(0.5, "crint", 1, 1) == pytest.approx(0.5)
    assert sizing.effective_kelly_mult(0.5, "crict20blast", 4, 4) == pytest.approx(0.5)
    assert sizing.effective_kelly_mult(0.5, "cricipl", 5, 5) == pytest.approx(0.5)


def test_associate_stake_is_one_tenth_of_full():
    """Sanity: the 5% throttle yields ~1/10th the half-Kelly stake at a price
    where neither the fraction cap nor the 25% bankroll cap binds."""
    half_kelly = next(s for s in STRATEGIES if s.name == "v3_marg_3pp")  # kelly_mult=0.5
    full = sizing.live_scaled_kelly_stake(0.60, 0.50, 1000.0, half_kelly)
    assoc = sizing.live_scaled_kelly_stake(
        0.60, 0.50, 1000.0, half_kelly,
        sizing.effective_kelly_mult(half_kelly.kelly_mult, "crint", 5, 2),
    )
    assert full > assoc
    assert assoc == pytest.approx(full * (0.05 / 0.5), rel=1e-6)
