"""
Unit tests for the v2 simulator's extras + free-hit logic and budget-bias.
These tests exercise the pure-numpy helpers in V2Simulator that don't need
a trained model loaded.
"""

import numpy as np
import unittest

from src.features.ball_training_data_v2 import (
    LABEL_DOT, LABEL_WICKET, LABEL_WIDE, LABEL_NOBALL, NUM_CLASSES_V2,
)
from src.models.vectorized_nn_sim_v2 import (
    OUTCOME_RUNS_V2, V2Simulator,
)


class TestOutcomeRuns(unittest.TestCase):
    """The runs-added table is the single source of truth for outcome scoring."""

    def test_class_runs_table_matches_specification(self):
        # dot, 1, 2, 3, 4, 6, wicket, wide, noball
        self.assertEqual(OUTCOME_RUNS_V2.tolist(), [0, 1, 2, 3, 4, 6, 0, 1, 1])

    def test_runs_table_has_correct_size(self):
        self.assertEqual(len(OUTCOME_RUNS_V2), NUM_CLASSES_V2)


class TestBudgetBias(unittest.TestCase):
    """The budget-bias function nudges per-ball probs by over budget."""

    def _uniform_prob(self, n_matches=4):
        return np.full((n_matches, NUM_CLASSES_V2), 1.0 / NUM_CLASSES_V2, dtype=np.float32)

    def test_alpha_zero_is_identity(self):
        proba = self._uniform_prob()
        biased = V2Simulator._bias_per_ball_logits(
            proba.copy(),
            runs_remaining_per_ball=np.array([3.0, 0.0, 5.0, 1.0]),
            wkts_remaining=np.array([0.0, 0.0, 0.0, 0.0]),
            alpha=0.0,
        )
        np.testing.assert_array_almost_equal(biased, proba)

    def test_high_runs_remaining_boosts_boundary_classes(self):
        proba = self._uniform_prob(n_matches=1)
        # Need 5 runs per ball -> way above baseline 1.5; should heavily
        # boost classes 4 and 5 (4-run, 6-run) and suppress dot/single.
        biased = V2Simulator._bias_per_ball_logits(
            proba,
            runs_remaining_per_ball=np.array([5.0]),
            wkts_remaining=np.array([0.0]),
            alpha=0.5,
        )
        # boundary classes should now exceed uniform
        self.assertGreater(biased[0, 4], 1.0 / NUM_CLASSES_V2)  # 4 runs
        self.assertGreater(biased[0, 5], 1.0 / NUM_CLASSES_V2)  # 6 runs
        # dot/single should be below uniform
        self.assertLess(biased[0, 0], 1.0 / NUM_CLASSES_V2)
        self.assertLess(biased[0, 1], 1.0 / NUM_CLASSES_V2)
        # 6-run class should be boosted more than 4-run (1.5x weight)
        self.assertGreater(biased[0, 5], biased[0, 4])
        # Probabilities still sum to 1
        self.assertAlmostEqual(biased.sum(), 1.0, places=4)

    def test_negative_runs_remaining_boosts_dot_single(self):
        proba = self._uniform_prob(n_matches=1)
        # Massive surplus (no runs needed); should boost dot/single
        biased = V2Simulator._bias_per_ball_logits(
            proba,
            runs_remaining_per_ball=np.array([0.0]),  # need 0; baseline 1.5 -> negative signal
            wkts_remaining=np.array([0.0]),
            alpha=0.5,
        )
        self.assertGreater(biased[0, 0], 1.0 / NUM_CLASSES_V2)  # dot
        self.assertGreater(biased[0, 1], 1.0 / NUM_CLASSES_V2)  # single
        self.assertLess(biased[0, 4], 1.0 / NUM_CLASSES_V2)
        self.assertLess(biased[0, 5], 1.0 / NUM_CLASSES_V2)
        self.assertAlmostEqual(biased.sum(), 1.0, places=4)

    def test_wkts_remaining_boosts_wicket_class(self):
        proba = self._uniform_prob(n_matches=1)
        biased = V2Simulator._bias_per_ball_logits(
            proba,
            runs_remaining_per_ball=np.array([1.5]),  # neutral on runs
            wkts_remaining=np.array([3.0]),           # large wicket budget
            alpha=0.5,
        )
        # wicket class should rise above baseline
        self.assertGreater(biased[0, LABEL_WICKET], 1.0 / NUM_CLASSES_V2)


class TestSampleOverTargets(unittest.TestCase):
    """Sample over budgets from the per-over head's predicted Gaussian/Poisson."""

    def test_mean_runs_recovered_within_noise(self):
        rng = np.random.default_rng(0)
        # mu=8 runs/over, sigma=exp(-1)=0.37, lambda=exp(-1)=0.37 wkts/over
        over_params = np.tile(
            np.array([8.0, -1.0, -1.0, 0.0], dtype=np.float32), (5000, 1)
        )
        runs, wkts = V2Simulator._sample_over_targets(over_params, rng)
        self.assertAlmostEqual(runs.mean(), 8.0, delta=0.1)
        self.assertAlmostEqual(wkts.mean(), np.exp(-1.0), delta=0.05)

    def test_runs_target_clamped_at_zero(self):
        rng = np.random.default_rng(0)
        # Negative mu should clamp output at 0 (no negative runs)
        over_params = np.tile(
            np.array([-5.0, 0.0, -1.0, 0.0], dtype=np.float32), (1000, 1)
        )
        runs, _ = V2Simulator._sample_over_targets(over_params, rng)
        self.assertTrue((runs >= 0.0).all())


class TestSampleOutcomes(unittest.TestCase):
    """Vectorised categorical sampling from per-ball softmax."""

    def test_deterministic_distribution_recovers_class(self):
        rng = np.random.default_rng(0)
        # All probability mass on class 4
        proba = np.zeros((100, NUM_CLASSES_V2), dtype=np.float32)
        proba[:, 4] = 1.0
        outcomes = V2Simulator._sample_outcomes(proba, rng)
        self.assertTrue((outcomes == 4).all())

    def test_uniform_distribution_recovers_uniform_marginal(self):
        rng = np.random.default_rng(0)
        proba = np.full((20000, NUM_CLASSES_V2), 1.0 / NUM_CLASSES_V2, dtype=np.float32)
        outcomes = V2Simulator._sample_outcomes(proba, rng)
        counts = np.bincount(outcomes, minlength=NUM_CLASSES_V2)
        # Each class should be sampled 20000/9 ~ 2222 times. Allow generous slack.
        for c in range(NUM_CLASSES_V2):
            self.assertGreater(counts[c], 1900)
            self.assertLess(counts[c], 2600)


if __name__ == "__main__":
    unittest.main()
