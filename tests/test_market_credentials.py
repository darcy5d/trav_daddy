"""Unit tests for market credential readiness helpers."""

import unittest

from src.integrations import credentials


class TestMarketCredentials(unittest.TestCase):
    def setUp(self):
        self.poly_original = dict(credentials.POLYMARKET_CONFIG)
        self.betfair_original = dict(credentials.BETFAIR_CONFIG)

    def tearDown(self):
        credentials.POLYMARKET_CONFIG.clear()
        credentials.POLYMARKET_CONFIG.update(self.poly_original)
        credentials.BETFAIR_CONFIG.clear()
        credentials.BETFAIR_CONFIG.update(self.betfair_original)

    def test_polymarket_public_read_ready_without_keys(self):
        credentials.POLYMARKET_CONFIG.clear()
        credentials.POLYMARKET_CONFIG.update(
            {
                "enabled": True,
                "api_base_url": "https://gamma-api.polymarket.com",
                "clob_base_url": "https://clob.polymarket.com",
                "chain_id": 137,
                "api_key": "",
                "api_secret": "",
                "passphrase": "",
                "private_key": "",
            }
        )

        status = credentials.get_polymarket_credential_status()
        self.assertTrue(status["public_read_ready"])
        self.assertFalse(status["authenticated_ready"])
        self.assertEqual(status["mode"], "public_read_only")

    def test_betfair_requires_app_key_and_auth_path_for_read(self):
        credentials.BETFAIR_CONFIG.clear()
        credentials.BETFAIR_CONFIG.update(
            {
                "enabled": True,
                "app_key": "",
                "username": "",
                "password": "",
                "session_token": "",
                "cert_file": "",
                "key_file": "",
                "sso_base_url": "https://identitysso.betfair.com",
                "betting_api_base_url": "https://api.betfair.com/exchange/betting",
            }
        )
        status_missing = credentials.get_betfair_credential_status()
        self.assertFalse(status_missing["read_ready"])
        self.assertIn("app_key", status_missing["missing_required_for_read"])

        credentials.BETFAIR_CONFIG["app_key"] = "abc123xyz"
        credentials.BETFAIR_CONFIG["session_token"] = "session-token-1234"
        status_ready = credentials.get_betfair_credential_status()
        self.assertTrue(status_ready["read_ready"])
        self.assertEqual(status_ready["auth_path"], "session_token")

    def test_consolidated_status_reflects_provider_readiness(self):
        credentials.POLYMARKET_CONFIG["enabled"] = True
        credentials.POLYMARKET_CONFIG["api_key"] = ""
        credentials.POLYMARKET_CONFIG["api_secret"] = ""
        credentials.POLYMARKET_CONFIG["passphrase"] = ""
        credentials.POLYMARKET_CONFIG["private_key"] = ""

        credentials.BETFAIR_CONFIG["enabled"] = True
        credentials.BETFAIR_CONFIG["app_key"] = "betfair-app"
        credentials.BETFAIR_CONFIG["session_token"] = "session-1234"
        credentials.BETFAIR_CONFIG["username"] = ""
        credentials.BETFAIR_CONFIG["password"] = ""
        credentials.BETFAIR_CONFIG["cert_file"] = ""
        credentials.BETFAIR_CONFIG["key_file"] = ""

        status = credentials.get_market_credentials_status()
        self.assertTrue(status["success"])
        self.assertTrue(status["ready_for_wave2_read"])
        self.assertTrue(status["ready_for_wave2_read_polymarket_only"])
        self.assertTrue(status["ready_for_wave2_read_with_betfair"])
        self.assertIn("polymarket", status["providers"])
        self.assertIn("betfair", status["providers"])

    def test_consolidated_status_allows_polymarket_only(self):
        credentials.POLYMARKET_CONFIG["enabled"] = True
        credentials.POLYMARKET_CONFIG["api_key"] = ""
        credentials.POLYMARKET_CONFIG["api_secret"] = ""
        credentials.POLYMARKET_CONFIG["passphrase"] = ""
        credentials.POLYMARKET_CONFIG["private_key"] = ""

        credentials.BETFAIR_CONFIG["enabled"] = False
        credentials.BETFAIR_CONFIG["app_key"] = ""
        credentials.BETFAIR_CONFIG["session_token"] = ""
        credentials.BETFAIR_CONFIG["username"] = ""
        credentials.BETFAIR_CONFIG["password"] = ""
        credentials.BETFAIR_CONFIG["cert_file"] = ""
        credentials.BETFAIR_CONFIG["key_file"] = ""

        status = credentials.get_market_credentials_status()
        self.assertTrue(status["ready_for_wave2_read"])
        self.assertTrue(status["ready_for_wave2_read_polymarket_only"])
        self.assertFalse(status["ready_for_wave2_read_with_betfair"])
        self.assertTrue(status["betfair_optional_for_current_wave"])


if __name__ == "__main__":
    unittest.main()

