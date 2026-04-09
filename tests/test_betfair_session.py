"""Unit tests for Betfair session bootstrap manager."""

import unittest
from unittest.mock import Mock, patch

from src.integrations.betfair import session as session_module
from src.integrations.betfair.session import BetfairSessionManager


class TestBetfairSessionManager(unittest.TestCase):
    def setUp(self):
        self.original = dict(session_module.BETFAIR_CONFIG)
        session_module.BETFAIR_CONFIG.update(
            {
                "enabled": True,
                "app_key": "app-key-123",
                "username": "user123",
                "password": "pass123",
                "session_token": "",
                "cert_file": "",
                "key_file": "",
                "sso_base_url": "https://identitysso.betfair.com",
                "login_path": "/api/login",
                "cert_login_path": "/api/certlogin",
                "keep_alive_path": "/api/keepAlive",
                "betting_api_base_url": "https://api.betfair.com/exchange/betting",
            }
        )

    def tearDown(self):
        session_module.BETFAIR_CONFIG.clear()
        session_module.BETFAIR_CONFIG.update(self.original)

    @patch("src.integrations.betfair.session.requests.post")
    def test_bootstrap_interactive_login(self, mock_post):
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"status": "SUCCESS", "token": "token-abc-1234"}
        mock_post.return_value = mock_response

        manager = BetfairSessionManager()
        result = manager.bootstrap(force_refresh=True)

        self.assertTrue(result["success"])
        self.assertEqual(result["login_method"], "interactive_api_login")
        self.assertIn("session_token_masked", result)

    @patch("src.integrations.betfair.session.requests.post")
    def test_keep_alive_uses_existing_token(self, mock_post):
        keep_alive_response = Mock()
        keep_alive_response.raise_for_status.return_value = None
        keep_alive_response.json.return_value = {"status": "SUCCESS", "token": "token-next-9876"}
        mock_post.return_value = keep_alive_response

        manager = BetfairSessionManager()
        manager._session_token = "token-old-1111"
        payload = manager.keep_alive()

        self.assertTrue(payload["success"])
        self.assertIn("last_refreshed_at", payload)
        called_url = mock_post.call_args[0][0]
        self.assertEqual(called_url, "https://identitysso.betfair.com/api/keepAlive")


if __name__ == "__main__":
    unittest.main()

