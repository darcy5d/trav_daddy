"""Unit tests for Polymarket comparison service and odds helpers."""

import unittest
from datetime import datetime, timezone

from src.integrations.odds.polymarket_compare import (
    PolymarketComparisonService,
    build_fixture_key,
    compute_edge_pct_points,
    infer_selected_side,
)


class _FakePolymarketClient:
    def __init__(self, markets, orderbooks, slug_markets=None):
        self._markets = markets
        self._orderbooks = orderbooks
        self._slug_markets = slug_markets or {}

    def get_markets(self, limit=200, active=True, closed=False):
        return self._markets

    def get_markets_by_slug(self, slug):
        return self._slug_markets.get(slug, [])

    def get_clob_order_book(self, token_id):
        if token_id not in self._orderbooks:
            raise ValueError("token not found")
        return self._orderbooks[token_id]


class TestPolymarketCompareHelpers(unittest.TestCase):
    def test_build_fixture_key(self):
        key = build_fixture_key("Australia Women", "India Women", "2026-04-10T09:00:00Z")
        self.assertIn("australia_women", key)
        self.assertIn("india_women", key)

    def test_infer_selected_side(self):
        side = infer_selected_side("A", "B", 60.0, 40.0)
        self.assertEqual(side, "A")
        side2 = infer_selected_side("A", "B", 49.5, 50.5)
        self.assertEqual(side2, "B")

    def test_compute_edge(self):
        self.assertEqual(compute_edge_pct_points(61.2, 54.0), 7.2)
        self.assertIsNone(compute_edge_pct_points(None, 54.0))


class TestPolymarketComparisonService(unittest.TestCase):
    def test_compare_fixture_ok(self):
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        market = {
            "id": "m1",
            "question": "Will Australia Women beat India Women?",
            "slug": "aus-w-v-ind-w",
            "clobTokenIds": ["yes-token", "no-token"],
            "startDate": "2026-04-10T09:00:00Z",
        }
        orderbooks = {
            "yes-token": {
                "bids": [{"price": "0.53", "size": "10"}],
                "asks": [{"price": "0.55", "size": "8"}],
                "timestamp": now_ms,
            }
        }
        service = PolymarketComparisonService(
            client=_FakePolymarketClient([market], orderbooks),
            market_cache_ttl_sec=999,
        )
        payload = service.compare_fixture(
            team1="Australia Women",
            team2="India Women",
            start_utc="2026-04-10T09:00:00Z",
            model_team1_win_pct=61.2,
            model_team2_win_pct=38.8,
        )
        self.assertTrue(payload["success"])
        self.assertEqual(payload["status"], "ok")
        self.assertAlmostEqual(payload["probabilities"]["market_implied_pct"], 54.0)
        self.assertAlmostEqual(payload["probabilities"]["edge_pct_points"], 7.2)

    def test_compare_fixture_no_match(self):
        unrelated_market = {
            "id": "m2",
            "question": "Will Team X beat Team Y?",
            "slug": "x-v-y",
            "clobTokenIds": ["x-yes", "x-no"],
        }
        service = PolymarketComparisonService(
            client=_FakePolymarketClient([unrelated_market], {}),
            market_cache_ttl_sec=999,
        )
        payload = service.compare_fixture("Australia Women", "India Women")
        self.assertEqual(payload["status"], "no_match")
        self.assertIsNone(payload["linked_market"])

    def test_compare_fixture_quote_unavailable(self):
        market = {
            "id": "m3",
            "question": "Will Australia Women beat India Women?",
            "slug": "aus-w-v-ind-w",
            "clobTokenIds": ["yes-token", "no-token"],
        }
        # Empty ladders -> quote unavailable.
        service = PolymarketComparisonService(
            client=_FakePolymarketClient([market], {"yes-token": {"bids": [], "asks": []}}),
            market_cache_ttl_sec=999,
        )
        payload = service.compare_fixture("Australia Women", "India Women")
        self.assertEqual(payload["status"], "quote_unavailable")

    def test_compare_fixture_uses_best_bid_ask_not_first_row(self):
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        market = {
            "id": "m7",
            "question": "Will Australia Women beat India Women?",
            "slug": "aus-w-v-ind-w",
            "clobTokenIds": ["yes-token", "no-token"],
        }
        # Orderbook levels intentionally unsorted (like live CLOB payloads).
        # Best bid=0.59, best ask=0.61 -> midpoint=0.60 (not 0.50).
        orderbooks = {
            "yes-token": {
                "bids": [{"price": "0.01"}, {"price": "0.59"}],
                "asks": [{"price": "0.99"}, {"price": "0.61"}],
                "timestamp": str(now_ms),  # numeric string timestamp should parse correctly
            }
        }
        service = PolymarketComparisonService(
            client=_FakePolymarketClient([market], orderbooks),
            market_cache_ttl_sec=999,
        )
        payload = service.compare_fixture(
            team1="Australia Women",
            team2="India Women",
            model_team1_win_pct=61.2,
            model_team2_win_pct=38.8,
        )
        self.assertEqual(payload["status"], "ok")
        self.assertAlmostEqual(payload["probabilities"]["market_implied_pct"], 60.0)

    def test_compare_batch(self):
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        market = {
            "id": "m4",
            "question": "Will Australia Women beat India Women?",
            "slug": "aus-w-v-ind-w",
            "clobTokenIds": ["yes-token", "no-token"],
        }
        orderbooks = {"yes-token": {"bids": [{"price": "0.50"}], "asks": [{"price": "0.52"}], "timestamp": now_ms}}
        service = PolymarketComparisonService(client=_FakePolymarketClient([market], orderbooks))
        batch = service.compare_batch(
            [
                {
                    "row_key": "r1",
                    "team1": "Australia Women",
                    "team2": "India Women",
                    "model_team1_win_pct": 55.0,
                    "model_team2_win_pct": 45.0,
                }
            ]
        )
        self.assertTrue(batch["success"])
        self.assertEqual(batch["count"], 1)
        self.assertEqual(batch["results"][0]["row_key"], "r1")

    def test_slug_fallback_lookup(self):
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        slug_market = {
            "id": "m5",
            "question": "Indian Premier League: Kolkata Knight Riders vs Lucknow Super Giants",
            "slug": "cricipl-kol-luc-2026-04-09",
            "clobTokenIds": ["yes-token", "no-token"],
        }
        orderbooks = {"yes-token": {"bids": [{"price": "0.49"}], "asks": [{"price": "0.51"}], "timestamp": now_ms}}
        fake = _FakePolymarketClient(
            markets=[],  # Force fallback path.
            orderbooks=orderbooks,
            slug_markets={"cricipl-kol-luc-2026-04-09": [slug_market]},
        )
        service = PolymarketComparisonService(client=fake, market_cache_ttl_sec=999)
        payload = service.compare_fixture(
            team1="Kolkata Knight Riders",
            team2="Lucknow Super Giants",
            series="IPL 2026",
            start_utc="2026-04-09T14:00:00Z",
            model_team1_win_pct=60.0,
            model_team2_win_pct=40.0,
        )
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["linked_market"]["slug"], "cricipl-kol-luc-2026-04-09")
        self.assertIn("Matched via slug fallback", payload["warnings"])

    def test_slug_fallback_lookup_without_series(self):
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        slug_market = {
            "id": "m6",
            "question": "Pakistan Super League: Karachi Kings vs Peshawar Zalmi",
            "slug": "cricpsl-kar-pes-2026-04-09",
            "clobTokenIds": ["yes-token", "no-token"],
        }
        orderbooks = {"yes-token": {"bids": [{"price": "0.49"}], "asks": [{"price": "0.51"}], "timestamp": now_ms}}
        fake = _FakePolymarketClient(
            markets=[],
            orderbooks=orderbooks,
            slug_markets={"cricpsl-kar-pes-2026-04-09": [slug_market]},
        )
        service = PolymarketComparisonService(client=fake, market_cache_ttl_sec=999)
        payload = service.compare_fixture(
            team1="Karachi Kings",
            team2="Peshawar Zalmi",
            series=None,
            start_utc="2026-04-09T19:30:00Z",
            model_team1_win_pct=60.0,
            model_team2_win_pct=40.0,
        )
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["linked_market"]["slug"], "cricpsl-kar-pes-2026-04-09")


if __name__ == "__main__":
    unittest.main()

