"""Tests for V2's toss channel.

V2's 22-feature vector has no toss column, so `use_toss`/`toss_field_prob` can
never change the network's inputs. What a known toss *can* change is the
innings order: V2 always has the team in the team1 slot bat first. The new
`toss_bat_first_team_id` kwarg lets a caller with a known toss put the real
bat-first side in the team1 slot and get the answer back in the original
orientation.

These tests pin the contract without loading a model: they stub the innings
loop and the calibration so only the plumbing under test runs.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

# `vectorized_nn_sim_v2` raises SystemExit when tensorflow/keras are absent, and
# other test modules in this suite install a partial `tensorflow` stub into
# sys.modules which makes that import fail. Running this file on its own works;
# under full-suite collection it must skip rather than blow up collection.
try:
    from src.models.vectorized_nn_sim_v2 import V2Simulator
except (ImportError, SystemExit) as _exc:  # pragma: no cover
    pytest.skip(
        f"vectorized_nn_sim_v2 not importable in this process ({_exc})",
        allow_module_level=True,
    )


class _FakeCalibration:
    def get(self, fmt, gender):
        class P:
            a = 1.0
            b = 0.0
        return P()


def _bare_simulator():
    """A V2Simulator with every heavy dependency stubbed out."""
    sim = V2Simulator.__new__(V2Simulator)
    sim.format_type = "T20"
    sim.gender = "male"
    sim.calibration = _FakeCalibration()
    sim.venue_stats = None
    sim.vocabs = {"batter": {}, "bowler": {}, "venue": {}, "team": {}}
    sim._ensure_model_loaded = lambda: None
    sim._team_elo = lambda team_id: 1500.0
    sim._player_batting = lambda p: 1500.0
    sim._player_bowling = lambda p: 1500.0
    sim._vocab_id = lambda kind, ident: 1
    return sim


def _innings_stub(first_runs_by_slot, second_runs_by_slot):
    """Record which lineup got innings 1, then return fixed score arrays."""
    calls = []

    def _simulate_innings(self, *, innings, batter_vocab, batter_elos, **kw):
        calls.append((innings, tuple(batter_vocab)))
        n = 100
        if innings == 1:
            runs = np.full(n, first_runs_by_slot, dtype=np.int32)
        else:
            runs = np.full(n, second_runs_by_slot, dtype=np.int32)
        player_runs = np.zeros((n, len(batter_vocab)), dtype=np.int32)
        sixes = np.zeros(n, dtype=np.int32)
        return runs, None, player_runs, sixes

    return _simulate_innings, calls


def test_no_toss_kwarg_leaves_team1_batting_first():
    sim = _bare_simulator()
    stub, calls = _innings_stub(160, 140)
    # Distinct ids so we can see the order in the recorded vocab calls.
    sim._vocab_id = lambda kind, ident: ident if kind == "batter" else 1
    with patch.object(V2Simulator, "_simulate_innings", stub):
        out = sim.simulate_matches(
            100, [11, 12], [21, 22], [31, 32], [41, 42],
            team1_id=1, team2_id=2, seed=1,
        )
    assert calls[0] == (1, (11, 12)), "team1's batters must take innings 1"
    assert calls[1] == (2, (31, 32)), "team2's batters must chase"
    assert out["toss_bat_first_applied"] is False


def test_toss_naming_team1_keeps_the_original_order():
    sim = _bare_simulator()
    stub, calls = _innings_stub(160, 140)
    sim._vocab_id = lambda kind, ident: ident if kind == "batter" else 1
    with patch.object(V2Simulator, "_simulate_innings", stub):
        out = sim.simulate_matches(
            100, [11, 12], [21, 22], [31, 32], [41, 42],
            team1_id=1, team2_id=2, seed=1, toss_bat_first_team_id=1,
        )
    assert calls[0] == (1, (11, 12))
    assert out["toss_bat_first_applied"] is False


def test_toss_naming_team2_swaps_the_innings_order():
    sim = _bare_simulator()
    stub, calls = _innings_stub(160, 140)
    sim._vocab_id = lambda kind, ident: ident if kind == "batter" else 1
    with patch.object(V2Simulator, "_simulate_innings", stub):
        out = sim.simulate_matches(
            100, [11, 12], [21, 22], [31, 32], [41, 42],
            team1_id=1, team2_id=2, seed=1, toss_bat_first_team_id=2,
        )
    assert calls[0] == (1, (31, 32)), "team2's batters must take innings 1 after the toss"
    assert out["toss_bat_first_applied"] is True


def test_toss_swap_reports_the_answer_for_the_original_team1():
    """The swapped run must not leak the internal orientation to the caller."""
    sim = _bare_simulator()
    # In the swapped run, "team1" is really team2 and scores 160 vs 140, so the
    # internal team1 (= real team2) wins. The caller asked about real team1, so
    # it must get the losing probability.
    stub, _ = _innings_stub(160, 140)
    sim._vocab_id = lambda kind, ident: 1
    with patch.object(V2Simulator, "_simulate_innings", stub):
        out = sim.simulate_matches(
            100, [11], [21], [31], [41],
            team1_id=1, team2_id=2, seed=1, toss_bat_first_team_id=2,
        )
    assert out["team1_win_prob"] == pytest.approx(0.0, abs=1e-9)
    assert out["team2_win_prob"] == pytest.approx(1.0, abs=1e-9)
    assert out["team1_win_prob"] + out["team2_win_prob"] == pytest.approx(1.0)


def test_toss_probs_are_complementary_when_team2_is_put_in_first():
    sim = _bare_simulator()
    stub, _ = _innings_stub(140, 160)  # chasing side reaches the target
    sim._vocab_id = lambda kind, ident: 1
    with patch.object(V2Simulator, "_simulate_innings", stub):
        out = sim.simulate_matches(
            100, [11], [21], [31], [41],
            team1_id=1, team2_id=2, seed=1, toss_bat_first_team_id=2,
        )
    assert out["team1_win_prob"] == pytest.approx(1.0, abs=1e-9)


def test_toss_swap_keeps_the_elo_and_player_arrays_aligned():
    sim = _bare_simulator()
    stub, _ = _innings_stub(160, 140)
    sim._vocab_id = lambda kind, ident: ident if kind == "batter" else 1
    with patch.object(V2Simulator, "_simulate_innings", stub):
        out = sim.simulate_matches(
            100, [11, 12], [21, 22], [31, 32], [41, 42],
            team1_id=1, team2_id=2, seed=1, toss_bat_first_team_id=2,
        )
    # Reported batting arrays must describe the caller's team1, not the sim's.
    assert list(out["team1_batter_ids"]) == [11, 12]
    assert list(out["team2_batter_ids"]) == [31, 32]
    assert out["team1_player_runs"].shape[1] == len(out["team1_batter_ids"])


def test_unknown_toss_team_id_is_ignored():
    sim = _bare_simulator()
    stub, calls = _innings_stub(160, 140)
    sim._vocab_id = lambda kind, ident: ident if kind == "batter" else 1
    with patch.object(V2Simulator, "_simulate_innings", stub):
        out = sim.simulate_matches(
            100, [11, 12], [21, 22], [31, 32], [41, 42],
            team1_id=1, team2_id=2, seed=1, toss_bat_first_team_id=99,
        )
    assert calls[0] == (1, (11, 12))
    assert out["toss_bat_first_applied"] is False


def test_use_toss_flag_alone_does_not_reorder_innings():
    """`use_toss=True` is what the scanner already passes; it changes nothing."""
    sim = _bare_simulator()
    stub, calls = _innings_stub(160, 140)
    sim._vocab_id = lambda kind, ident: ident if kind == "batter" else 1
    with patch.object(V2Simulator, "_simulate_innings", stub):
        sim.simulate_matches(
            100, [11, 12], [21, 22], [31, 32], [41, 42],
            team1_id=1, team2_id=2, seed=1, use_toss=True, toss_field_prob=0.65,
        )
    assert calls[0] == (1, (11, 12)), (
        "use_toss is accepted but must not silently reorder the innings"
    )


def test_kwarg_defaults_to_none_so_existing_callers_are_unaffected():
    """V3 calls super().simulate_matches without this kwarg; it must stay off."""
    import inspect

    sig = inspect.signature(V2Simulator.simulate_matches)
    assert "toss_bat_first_team_id" in sig.parameters
    assert sig.parameters["toss_bat_first_team_id"].default is None
