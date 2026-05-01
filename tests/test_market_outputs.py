"""Unit tests for src/models/market_outputs.py (Wave 5 Phase 1)."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.market_outputs import (
    derive_polymarket_market_probs,
    market_summary_for_ui,
)


def _make_sim_result(
    win_prob: float = 0.6,
    t1_player_runs: np.ndarray = None,
    t2_player_runs: np.ndarray = None,
    t1_sixes: np.ndarray = None,
    t2_sixes: np.ndarray = None,
    t1_batter_ids=None,
    t2_batter_ids=None,
):
    return {
        "team1_win_prob": win_prob,
        "team1_player_runs": t1_player_runs if t1_player_runs is not None else np.zeros((0, 0), dtype=np.int32),
        "team2_player_runs": t2_player_runs if t2_player_runs is not None else np.zeros((0, 0), dtype=np.int32),
        "team1_sixes": t1_sixes if t1_sixes is not None else np.zeros((0,), dtype=np.int32),
        "team2_sixes": t2_sixes if t2_sixes is not None else np.zeros((0,), dtype=np.int32),
        "team1_batter_ids": t1_batter_ids if t1_batter_ids is not None else [],
        "team2_batter_ids": t2_batter_ids if t2_batter_ids is not None else [],
    }


def test_moneyline_passthrough():
    sim = _make_sim_result(win_prob=0.62)
    out = derive_polymarket_market_probs(sim)
    assert out["moneyline"]["team1"] == pytest.approx(0.62)
    assert out["moneyline"]["team2"] == pytest.approx(0.38)


def test_top_batter_three_way_obvious():
    """Team 1 batter 0 always tops; team2 batter 0 never tops."""
    t1 = np.array(
        [[80, 10, 20], [60, 10, 5], [70, 12, 0]], dtype=np.int32
    )
    # Team 2: match 0 batter 0 tops (40>30,20); match 1 batter 1 tops (55);
    # match 2 batter 2 tops (45 vs 20,30) -- mix across all three slots.
    t2 = np.array(
        [[40, 30, 20], [35, 55, 40], [20, 30, 45]], dtype=np.int32
    )
    sim = _make_sim_result(
        t1_player_runs=t1, t2_player_runs=t2,
        t1_batter_ids=[100, 101, 102], t2_batter_ids=[200, 201, 202],
    )
    out = derive_polymarket_market_probs(sim)
    tb = out["top_batter"]
    # Team 1 max is always >= 60, team 2 max is always <= 55. Team 1 always wins.
    assert tb["team1_higher"] == pytest.approx(1.0)
    assert tb["draw"] == pytest.approx(0.0)
    assert tb["team2_higher"] == pytest.approx(0.0)
    # Per-batter top distribution: team1 batter 100 always tops -> 1.0.
    assert tb["team1_top_batter_distribution"][100] == pytest.approx(1.0)
    assert tb["team1_top_batter_distribution"][101] == pytest.approx(0.0)
    # Team 2 top is each batter once (3 matches, each different) -> 1/3 each.
    assert tb["team2_top_batter_distribution"][200] == pytest.approx(1 / 3)
    assert tb["team2_top_batter_distribution"][201] == pytest.approx(1 / 3)
    assert tb["team2_top_batter_distribution"][202] == pytest.approx(1 / 3)


def test_top_batter_tie_split():
    """Two batters tied for top -> split contribution."""
    t1 = np.array([[50, 50, 10]], dtype=np.int32)
    sim = _make_sim_result(
        t1_player_runs=t1,
        t2_player_runs=np.array([[30, 20, 10]], dtype=np.int32),
        t1_batter_ids=[10, 11, 12],
        t2_batter_ids=[20, 21, 22],
    )
    out = derive_polymarket_market_probs(sim)
    dist = out["top_batter"]["team1_top_batter_distribution"]
    # Both batters 10 and 11 tied at 50 -> 0.5 each.
    assert dist[10] == pytest.approx(0.5)
    assert dist[11] == pytest.approx(0.5)
    assert dist[12] == pytest.approx(0.0)


def test_most_sixes_three_way():
    """Team1 hits more sixes than team2 in 60% of matches, ties in 20%."""
    t1 = np.array([5, 6, 4, 7, 3, 4, 5, 6, 7, 5], dtype=np.int32)
    t2 = np.array([3, 4, 4, 5, 3, 6, 7, 8, 9, 4], dtype=np.int32)
    # match-by-match comparison:
    #   t1>t2: 0,1,3,9 -> 4
    #   t1==t2: 2,4 -> 2
    #   t2>t1: 5,6,7,8 -> 4
    sim = _make_sim_result(t1_sixes=t1, t2_sixes=t2)
    out = derive_polymarket_market_probs(sim)
    ms = out["most_sixes"]
    assert ms["team1"] == pytest.approx(0.4)
    assert ms["draw"] == pytest.approx(0.2)
    assert ms["team2"] == pytest.approx(0.4)
    assert ms["team1_avg_sixes"] == pytest.approx(t1.mean())
    assert ms["team2_avg_sixes"] == pytest.approx(t2.mean())


def test_toss_match_double_cross_product():
    sim = _make_sim_result(win_prob=0.7)
    out = derive_polymarket_market_probs(sim)
    tmd = out["toss_match_double"]
    # Toss is 50/50; match is 0.7/0.3
    assert tmd["toss_team1_match_team1"] == pytest.approx(0.35)
    assert tmd["toss_team1_match_team2"] == pytest.approx(0.15)
    assert tmd["toss_team2_match_team1"] == pytest.approx(0.35)
    assert tmd["toss_team2_match_team2"] == pytest.approx(0.15)
    # Sums to 1.0
    assert sum(tmd.values()) == pytest.approx(1.0)


def test_market_summary_for_ui_strips_distribution():
    sim = _make_sim_result(
        win_prob=0.55,
        t1_player_runs=np.array([[40, 30, 20]], dtype=np.int32),
        t2_player_runs=np.array([[35, 25, 15]], dtype=np.int32),
        t1_sixes=np.array([5], dtype=np.int32),
        t2_sixes=np.array([4], dtype=np.int32),
        t1_batter_ids=[1, 2, 3], t2_batter_ids=[4, 5, 6],
    )
    full = derive_polymarket_market_probs(sim)
    compact = market_summary_for_ui(full)
    assert "team1_top_batter_distribution" not in compact["top_batter"]
    assert "team2_top_batter_distribution" not in compact["top_batter"]
    # Headline three-way still present.
    assert "team1_higher" in compact["top_batter"]
    assert "draw" in compact["top_batter"]
    assert "team2_higher" in compact["top_batter"]
    # Most sixes summary keeps the team1/draw/team2 keys but drops avg.
    assert set(compact["most_sixes"].keys()) == {"team1", "draw", "team2"}


def test_empty_sim_safe():
    """A simulation that returns no per-batter data shouldn't crash."""
    sim = _make_sim_result(win_prob=0.5)
    out = derive_polymarket_market_probs(sim)
    assert out["moneyline"]["team1"] == pytest.approx(0.5)
    # With no player runs at all, top-batter market collapses to "draw".
    assert out["top_batter"]["draw"] == pytest.approx(1.0)
