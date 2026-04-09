"""Unit tests for Polymarket client helpers."""

import unittest
from unittest.mock import patch, Mock

from src.integrations.polymarket.client import PolymarketClient


class TestPolymarketClient(unittest.TestCase):
    @patch("src.integrations.polymarket.client.requests.get")
    def test_get_markets_uses_public_gamma_endpoint(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = [{"id": "market-1"}]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        client = PolymarketClient(api_base_url="https://gamma-api.polymarket.com")
        data = client.get_markets(limit=10, active=True, closed=False)

        self.assertEqual(len(data), 1)
        called_url = mock_get.call_args[0][0]
        called_params = mock_get.call_args[1]["params"]
        self.assertEqual(called_url, "https://gamma-api.polymarket.com/markets")
        self.assertEqual(called_params["limit"], 10)
        self.assertEqual(called_params["active"], "true")
        self.assertEqual(called_params["closed"], "false")

    @patch("src.integrations.polymarket.client.requests.get")
    def test_health_check_calls_clob_ok(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = {"status": "ok"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        client = PolymarketClient(clob_base_url="https://clob.polymarket.com")
        payload = client.health_check()

        self.assertTrue(payload["success"])
        self.assertEqual(mock_get.call_args[0][0], "https://clob.polymarket.com/ok")


if __name__ == "__main__":
    unittest.main()

