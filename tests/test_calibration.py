"""
Unit tests for the post-hoc calibration layer (Wave 4 Phase 3).

Focus: verify that Platt scaling on synthetic over-confident probabilities
genuinely softens them, that perfectly-calibrated probabilities are left
alone (a=1, b=0), and that the apply / fit round-trip works.
"""

import math
import unittest

import numpy as np

from src.models.calibration import (
    CalibrationBundle,
    CalibrationParams,
    calibrate_probabilities,
    calibrate_probability,
    fit_platt,
)


class TestCalibrationApply(unittest.TestCase):
    def test_identity_calibration_is_noop(self):
        params = CalibrationParams(a=1.0, b=0.0)
        for p in [0.01, 0.25, 0.5, 0.75, 0.99]:
            self.assertAlmostEqual(calibrate_probability(p, params), p, places=6)

    def test_identity_vectorised_is_noop(self):
        params = CalibrationParams(a=1.0, b=0.0)
        probs = np.array([0.01, 0.25, 0.5, 0.75, 0.99])
        out = calibrate_probabilities(probs, params)
        np.testing.assert_array_almost_equal(out, probs)

    def test_softening_calibration_pulls_toward_05(self):
        # a < 1 should shrink probabilities toward 0.5
        params = CalibrationParams(a=0.5, b=0.0)
        # 0.95 -> Platt softens to ~0.78
        self.assertLess(calibrate_probability(0.95, params), 0.95)
        self.assertGreater(calibrate_probability(0.95, params), 0.5)
        # 0.05 -> ~0.22 (symmetrically)
        self.assertGreater(calibrate_probability(0.05, params), 0.05)
        self.assertLess(calibrate_probability(0.05, params), 0.5)

    def test_bias_shifts_off_centre(self):
        # b != 0 with a = 1 should shift the prediction. Positive b lifts
        # all predictions upward.
        params = CalibrationParams(a=1.0, b=0.5)
        # logit(0.5) = 0; output sigmoid(0 + 0.5) ~ 0.622
        self.assertAlmostEqual(calibrate_probability(0.5, params), 0.6225, places=3)


class TestCalibrationFit(unittest.TestCase):
    def test_fit_on_perfectly_calibrated_data_returns_near_identity(self):
        # If raw probs are already calibrated, the optimal a,b should sit
        # near (1, 0). Sample 1000 raw probs uniformly and draw
        # outcomes from them.
        rng = np.random.default_rng(42)
        probs = rng.uniform(0.05, 0.95, size=1000)
        outcomes = (rng.uniform(size=1000) < probs).astype(int)
        params = fit_platt(probs, outcomes)
        # Allow some slack because of finite sample noise.
        self.assertGreater(params.a, 0.5)
        self.assertLess(params.a, 1.5)
        self.assertGreater(params.b, -0.5)
        self.assertLess(params.b, 0.5)

    def test_fit_softens_overconfident_probs(self):
        # Synthetic: model says 95% confident, but reality is 60%.
        # Expected: fit should pull a < 1 (softening).
        rng = np.random.default_rng(7)
        n = 2000
        # Half the matches: model says 0.95, reality 0.6
        probs1 = np.full(n // 2, 0.95)
        outcomes1 = (rng.uniform(size=n // 2) < 0.6).astype(int)
        # Other half: model says 0.05, reality 0.4 (symmetric overconfidence)
        probs2 = np.full(n // 2, 0.05)
        outcomes2 = (rng.uniform(size=n // 2) < 0.4).astype(int)
        probs = np.concatenate([probs1, probs2])
        outcomes = np.concatenate([outcomes1, outcomes2])
        params = fit_platt(probs, outcomes)
        # Fit should softening: a < 1
        self.assertLess(params.a, 1.0)
        # And the softened 0.95 should be much closer to 0.5
        softened = calibrate_probability(0.95, params)
        self.assertLess(softened, 0.85)
        self.assertGreater(softened, 0.5)

    def test_fit_handles_tiny_sample(self):
        # < 5 rows should return identity rather than crash or overfit
        params = fit_platt(np.array([0.7, 0.3]), np.array([1, 0]))
        self.assertEqual(params.a, 1.0)
        self.assertEqual(params.b, 0.0)


class TestCalibrationBundle(unittest.TestCase):
    def test_route_key_and_get(self):
        b = CalibrationBundle()
        # Unknown route -> identity
        params = b.get("T20", "male")
        self.assertEqual(params.a, 1.0)
        self.assertEqual(params.b, 0.0)
        # Inserted route -> returns it
        b.params["T20_male"] = CalibrationParams(a=0.7, b=-0.1, n_train=100)
        params = b.get("T20", "male")
        self.assertEqual(params.a, 0.7)

    def test_round_trip_save_load(self, tmp_path=None):
        # tmp dir
        import tempfile, os
        b = CalibrationBundle(notes="round trip test")
        b.params["T20_male"] = CalibrationParams(a=0.62, b=0.04, n_train=44, nll_before=1.03, nll_after=0.71)
        b.params["ODI_male"] = CalibrationParams(a=0.85, b=-0.02, n_train=25)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "cal.json")
            b.save(path)
            loaded = CalibrationBundle.load(path)
        self.assertEqual(loaded.notes, "round trip test")
        self.assertAlmostEqual(loaded.params["T20_male"].a, 0.62, places=4)
        self.assertAlmostEqual(loaded.params["T20_male"].nll_after, 0.71, places=4)
        self.assertAlmostEqual(loaded.params["ODI_male"].a, 0.85, places=4)


if __name__ == "__main__":
    unittest.main()
