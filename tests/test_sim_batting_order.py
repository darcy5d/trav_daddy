"""Batting order in the V2/V3 simulators.

The first-innings side and the chasing side win different shares of
simulated games (about 44% vs 56% for identical T20 teams on the current V3
weights), so always putting team1 in first skews pre-toss P(team1). With a
pinned toss, V3 must also bat the side the toss sends in first, otherwise the
toss features describe a combination that never occurs in training.
"""

import unittest

import numpy as np

from src.models.calibration import CalibrationBundle
from src.models.vectorized_nn_sim_v2 import V2Simulator, swap_result_sides
from src.models.vectorized_nn_sim_v3 import V3Simulator

FIRST_INNINGS_WIN = 0.4


def _fake_result(n, t1_bat, t2_bat, t1_elo, t2_elo):
    p = FIRST_INNINGS_WIN
    return {
        "n_matches": n,
        "team1_win_prob": p, "team2_win_prob": 1 - p,
        "team1_win_prob_raw": p, "team2_win_prob_raw": 1 - p,
        "avg_team1_score": 160.0, "avg_team2_score": 150.0,
        "std_team1_score": 1.0, "std_team2_score": 2.0,
        "team1_score_range": (150.0, 170.0), "team2_score_range": (140.0, 160.0),
        "team1_scores": np.full(n, 160.0), "team2_scores": np.full(n, 150.0),
        "dist_quality": {"team1": {"tag": t1_bat[0]}, "team2": {"tag": t2_bat[0]}},
        "team1_elo_used": t1_elo, "team2_elo_used": t2_elo,
        "team1_player_runs": np.zeros((n, len(t1_bat))), "team2_player_runs": np.ones((n, len(t2_bat))),
        "team1_sixes": np.zeros(n), "team2_sixes": np.ones(n),
        "team1_batter_ids": list(t1_bat), "team2_batter_ids": list(t2_bat),
    }


def _stub(cls):
    sim = object.__new__(cls)
    sim.format_type, sim.gender = "T20", "male"
    sim.calibration = CalibrationBundle()
    sim.calls = []

    def ordered(n, t1_bat, t1_bowl, t2_bat, t2_bowl, team1_id=None, team2_id=None, seed=None,
                team1_default_elo=None, team2_default_elo=None, **kw):
        sim.calls.append({"first": t1_bat[0], "toss_idx": getattr(sim, "_toss_winner_idx", None),
                          "xi1": getattr(sim, "_xi_overlap_team1", None)})
        return _fake_result(n, t1_bat, t2_bat, team1_default_elo, team2_default_elo)

    sim._simulate_matches_ordered = ordered
    return sim


A_BAT, A_BOWL, B_BAT, B_BOWL = [101, 102], [103], [201, 202], [203]


class V2BattingOrderTest(unittest.TestCase):
    def test_default_keeps_team1_first(self):
        sim = _stub(V2Simulator)
        r = sim.simulate_matches(100, A_BAT, A_BOWL, B_BAT, B_BOWL)
        self.assertEqual(sim.calls[0]["first"], 101)
        self.assertAlmostEqual(r["team1_win_prob"], FIRST_INNINGS_WIN)

    def test_team2_first_swaps_inputs_and_relabels_output(self):
        sim = _stub(V2Simulator)
        r = sim.simulate_matches(100, A_BAT, A_BOWL, B_BAT, B_BOWL, team1_default_elo=1600,
                                 team2_default_elo=1400, batting_order="team2_first")
        self.assertEqual(sim.calls[0]["first"], 201)
        self.assertAlmostEqual(r["team1_win_prob"], 1 - FIRST_INNINGS_WIN)
        self.assertEqual(r["team1_elo_used"], 1600)
        self.assertEqual(r["team1_batter_ids"], A_BAT)
        self.assertEqual(r["dist_quality"]["team1"]["tag"], 101)
        self.assertTrue(np.all(r["team1_player_runs"] == 1))  # team1 filled the chasing slot

    def test_symmetric_cancels_first_innings_bias(self):
        sim = _stub(V2Simulator)
        r = sim.simulate_matches(300, A_BAT, A_BOWL, B_BAT, B_BOWL, seed=7, batting_order="symmetric")
        self.assertEqual([c["first"] for c in sim.calls], [101, 201])
        self.assertAlmostEqual(r["team1_win_prob_raw"], 0.5)
        self.assertEqual(r["n_matches"], 300)
        self.assertEqual(len(r["team1_scores"]), 300)
        self.assertEqual(r["team1_player_runs"].shape, (300, 2))

    def test_rejects_unknown_order(self):
        with self.assertRaises(ValueError):
            _stub(V2Simulator).simulate_matches(10, A_BAT, A_BOWL, B_BAT, B_BOWL, batting_order="x")

    def test_swap_is_an_involution(self):
        r = _fake_result(5, A_BAT, B_BAT, 1500, 1400)
        back = swap_result_sides(swap_result_sides(r))
        self.assertEqual(back["team1_elo_used"], 1500)
        self.assertEqual(back["avg_team1_score"], 160.0)


class V3PinnedTossTest(unittest.TestCase):
    def _v3(self):
        sim = _stub(V3Simulator)
        sim._toss_winner_idx = None
        sim._chose_to_bat_arr = None
        sim._xi_overlap_team1 = sim._xi_overlap_team2 = 1.0
        sim._xi_overlap_for_team = lambda team_id, match_date_str, this_xi=None: 0.25 if team_id == 1 else 0.75
        return sim

    def _run(self, winner, chose_field, **kw):
        sim = self._v3()
        r = sim.simulate_matches(100, A_BAT, A_BOWL, B_BAT, B_BOWL, team1_id=1, team2_id=2,
                                 toss_pinned=True, toss_winner_team_id=winner,
                                 toss_chose_field=chose_field, match_date_for_xi="2026-01-01", **kw)
        return sim, r

    def test_team2_wins_toss_and_bats_puts_team2_in_first(self):
        sim, r = self._run(winner=2, chose_field=False)
        call = sim.calls[0]
        self.assertEqual(call["first"], 201)
        self.assertEqual(int(call["toss_idx"]), 0)      # the side batting first won the toss
        self.assertEqual(call["xi1"], 0.75)            # team2's overlap rides with team2
        self.assertAlmostEqual(r["team1_win_prob"], 1 - FIRST_INNINGS_WIN)
        self.assertEqual(int(sim._toss_winner_idx), 1)  # state restored after the swap

    def test_team1_wins_toss_and_fields_puts_team2_in_first(self):
        sim, _ = self._run(winner=1, chose_field=True)
        self.assertEqual(sim.calls[0]["first"], 201)
        self.assertEqual(int(sim.calls[0]["toss_idx"]), 1)

    def test_team2_wins_toss_and_fields_keeps_team1_first(self):
        sim, _ = self._run(winner=2, chose_field=True)
        self.assertEqual(sim.calls[0]["first"], 101)

    def test_explicit_order_overrides_toss(self):
        sim, _ = self._run(winner=2, chose_field=False, batting_order="team1_first")
        self.assertEqual(sim.calls[0]["first"], 101)


if __name__ == "__main__":
    unittest.main()
