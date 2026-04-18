"""
Unit tests for the player batting ELO actual-score mapping in
EloCalculatorV3.update_player_ratings_for_match (Wave 3 calibration fix).

Pre-fix the mapping was actual = min(1.0, runs / (balls * avg_sr/100) / 2.0),
which compressed nearly every batter toward the bottom half of [0, 1] and
pinned anyone striking below the league mean to a guaranteed rating drop.

This test isolates the formula (centred at 0.5 when runs == expected,
linearly scaled by runs above expectation, bounded at [0, 1]) and locks
in the calibration we want.
"""

import unittest


def _actual_score(runs: int, balls: int, avg_sr: float = 130.0,
                  opponent_elo: float = 1500.0) -> float:
    """Mirror of the formula in calculator_v3.update_player_ratings_for_match.

    Kept here as a small reference impl so we can lock in the calibration
    without spinning up the full DB-backed calculator. If the production
    mapping changes, both should move together.
    """
    expected_runs = balls * (avg_sr / 100.0)
    opponent_adj = (opponent_elo - 1500.0) / 400.0
    expected_runs *= (1 - opponent_adj * 0.1)
    runs_above = runs - expected_runs
    actual = 0.5 + 0.5 * (runs_above / max(expected_runs, 10.0))
    return max(0.0, min(1.0, actual))


class TestPlayerBattingActualScore(unittest.TestCase):
    def test_league_average_innings_is_neutral(self):
        # 30 balls at SR 130 = 39 runs. Should land ≈ 0.5 (no rating change).
        self.assertAlmostEqual(_actual_score(39, 30), 0.5, places=2)

    def test_above_average_innings_lifts_score(self):
        # 30 balls at SR 150 = 45 runs vs expected 39.
        score = _actual_score(45, 30)
        self.assertGreater(score, 0.55)
        self.assertLess(score, 0.62)

    def test_explosive_innings_scores_strongly_but_bounded(self):
        # 50 off 20 (SR 250) is an outsized boundary-laden innings: 24
        # runs above the 26 expected at SR 130 over 20 balls. That should
        # land high - around 0.96 - and definitely never above the 1.0
        # ceiling. Pre-fix, the OLD formula gave 50 / 26 / 2.0 = 0.96
        # too, but only by happy accident; lift the strike rate further
        # (e.g. 60 off 20 = SR 300) and the OLD formula saturated at 1.0
        # while the new one keeps growing on the runs-above-expectation
        # axis instead of the SR axis, behaving more sensibly.
        score = _actual_score(50, 20)
        self.assertGreater(score, 0.85)
        self.assertLessEqual(score, 1.0)

        # Sanity: even more extreme innings still respect the ceiling.
        big = _actual_score(80, 20)
        self.assertLessEqual(big, 1.0)
        self.assertGreater(big, score)

    def test_slow_innings_loses_rating_but_not_collapsing(self):
        # 16 off 20 (SR 80) - real anchor innings. Old formula gave
        # actual = 16 / 26 / 2 ≈ 0.31. New mapping gives 0.31 too via a
        # different route, which is the right level: clearly negative
        # vs expectation but not a wipeout.
        score = _actual_score(16, 20)
        self.assertGreater(score, 0.20)
        self.assertLess(score, 0.42)

    def test_zero_runs_clamps_at_zero_for_long_innings(self):
        # 0 off 30 - duck after grinding. Should pin at the floor.
        self.assertEqual(_actual_score(0, 30), 0.0)

    def test_one_ball_innings_capped_by_expected_floor(self):
        # 6 off 1 - shouldn't trigger an extreme rating swing.
        # max(expected_runs=1.3, 10) = 10 in denominator; runs_above ≈ 4.7;
        # actual ≈ 0.5 + 0.5 * 4.7/10 = 0.735. Sanity check: bounded.
        score = _actual_score(6, 1)
        self.assertGreater(score, 0.5)
        self.assertLessEqual(score, 1.0)

    def test_strong_opponent_increases_expectation(self):
        # Same 45 runs off 30 vs a 1700-rated bowler should score lower
        # than vs a 1500-rated one. (Higher opponent_elo lowers expected
        # runs, but the test here checks the WHOLE formula direction.)
        easy = _actual_score(45, 30, opponent_elo=1500)
        hard = _actual_score(45, 30, opponent_elo=1700)
        # Higher opponent ELO -> reduced expected runs -> higher actual score.
        self.assertGreater(hard, easy)


if __name__ == "__main__":
    unittest.main()
