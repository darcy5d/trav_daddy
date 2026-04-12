"""API tests for Polymarket compare endpoints."""

import sys
import types
import unittest
from unittest.mock import patch

# app.main imports TensorFlow at module load. Provide a tiny stub for test envs
# that do not install tensorflow.
if "tensorflow" not in sys.modules:
    tf_stub = types.SimpleNamespace(
        config=types.SimpleNamespace(
            threading=types.SimpleNamespace(
                set_inter_op_parallelism_threads=lambda *args, **kwargs: None,
                set_intra_op_parallelism_threads=lambda *args, **kwargs: None,
            ),
            optimizer=types.SimpleNamespace(
                set_jit=lambda *args, **kwargs: None,
            ),
        )
    )
    sys.modules["tensorflow"] = tf_stub
if "flask_cors" not in sys.modules:
    sys.modules["flask_cors"] = types.SimpleNamespace(CORS=lambda *args, **kwargs: None)

from app.main import app


class _FakeCompareService:
    def compare_fixture(self, **kwargs):
        model_team1 = kwargs.get("model_team1_win_pct")
        edge = None
        if model_team1 is not None:
            edge = round(float(model_team1) - 55.0, 2)
        return {
            "success": True,
            "status": "ok",
            "fixture_key": "fixture1",
            "linked_market": {"market_id": "m1", "question": "q", "slug": "s", "link_confidence": 0.9},
            "selection": {"selected_side_label": kwargs.get("team1"), "token_id": "t1"},
            "probabilities": {
                "model_win_pct": model_team1,
                "market_implied_pct": 55.0,
                "edge_pct_points": edge,
            },
            "quote": {"source": "polymarket", "quote_age_sec": 10, "observed_at_utc": "2026-04-10T00:00:00Z"},
            "warnings": [],
        }

    def compare_batch(self, fixtures):
        return {"success": True, "count": len(fixtures), "results": []}


class TestPolymarketCompareAPI(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    @patch("app.main.get_polymarket_compare_service")
    def test_compare_requires_teams(self, _mock_get_service):
        resp = self.client.get("/api/integrations/polymarket/compare")
        self.assertEqual(resp.status_code, 400)
        payload = resp.get_json()
        self.assertFalse(payload["success"])

    @patch("app.main.get_polymarket_compare_service")
    def test_compare_success(self, mock_get_service):
        mock_get_service.return_value = _FakeCompareService()
        resp = self.client.get(
            "/api/integrations/polymarket/compare?team1=Australia%20Women&team2=India%20Women&model_team1_win_pct=61.2&model_team2_win_pct=38.8"
        )
        self.assertEqual(resp.status_code, 200)
        payload = resp.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["probabilities"]["model_win_pct"], 61.2)

    @patch("app.main.get_polymarket_compare_service")
    def test_compare_success_without_model_probs(self, mock_get_service):
        mock_get_service.return_value = _FakeCompareService()
        resp = self.client.get(
            "/api/integrations/polymarket/compare?team1=Karachi%20Kings&team2=Peshawar%20Zalmi"
        )
        self.assertEqual(resp.status_code, 200)
        payload = resp.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["status"], "ok")
        self.assertIsNone(payload["probabilities"]["model_win_pct"])
        self.assertIsNone(payload["probabilities"]["edge_pct_points"])
        self.assertEqual(payload["probabilities"]["market_implied_pct"], 55.0)

    @patch("app.main.get_polymarket_compare_service")
    def test_compare_batch_requires_list(self, mock_get_service):
        mock_get_service.return_value = _FakeCompareService()
        resp = self.client.post(
            "/api/integrations/polymarket/compare/batch",
            json={"fixtures": {"not": "a-list"}},
        )
        self.assertEqual(resp.status_code, 400)
        payload = resp.get_json()
        self.assertFalse(payload["success"])

    @patch("app.main.get_polymarket_compare_service")
    def test_compare_batch_success(self, mock_get_service):
        mock_get_service.return_value = _FakeCompareService()
        resp = self.client.post(
            "/api/integrations/polymarket/compare/batch",
            json={"fixtures": [{"row_key": "row1", "team1": "A", "team2": "B"}]},
        )
        self.assertEqual(resp.status_code, 200)
        payload = resp.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["count"], 1)


if __name__ == "__main__":
    unittest.main()

