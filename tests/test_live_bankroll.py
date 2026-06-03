"""Unit tests for wallet-driven portfolio / bankroll math."""

import sqlite3
import unittest
from unittest.mock import MagicMock, patch

from src.integrations.polymarket import live_bankroll


class TestLiveBankrollPortfolio(unittest.TestCase):
    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            """
            CREATE TABLE bet_ledger (
                bet_id INTEGER PRIMARY KEY,
                bet_kind TEXT,
                status TEXT,
                settled_at TEXT,
                fill_size_usdc REAL,
                fill_price REAL,
                polymarket_token_id TEXT,
                side_label TEXT,
                side TEXT,
                settle_outcome INTEGER,
                strategy_label TEXT
            )
            """
        )
        return conn

    def test_redeemable_positions_sum_current_value(self):
        pm = MagicMock()
        pm.get_data_api_positions.return_value = [
            {
                "asset": "token-win",
                "redeemable": True,
                "currentValue": 134.54,
                "size": 134.54,
                "title": "T20 Blast: Sussex vs Middlesex",
                "outcome": "Middlesex",
                "initialValue": 50.84,
            },
            {
                "asset": "token-open",
                "redeemable": False,
                "currentValue": 21.03,
            },
            {
                "asset": "token-dust",
                "redeemable": True,
                "currentValue": 0,
            },
        ]
        redeemable_now, pending, rows = live_bankroll._redeemable_positions_from_pm(pm)
        self.assertAlmostEqual(redeemable_now, 134.54)
        self.assertAlmostEqual(pending, 0.0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["token_id"], "token-win")
        self.assertEqual(rows[0]["claim_status"], "redeemable")

    def test_redeemable_excludes_open_ledger_tokens(self):
        pm = MagicMock()
        pm.get_data_api_positions.return_value = [
            {"asset": "token-open", "redeemable": True, "currentValue": 50.0},
        ]
        redeemable_now, pending, rows = live_bankroll._redeemable_positions_from_pm(
            pm, exclude_token_ids={"token-open"}
        )
        self.assertEqual(redeemable_now, 0.0)
        self.assertEqual(pending, 0.0)
        self.assertEqual(rows, [])

    @patch("config.POLYMARKET_CONFIG", {"private_key": "0xabc"})
    def test_portfolio_breakdown_includes_redeemable(self):
        conn = self._conn()
        conn.execute(
            """
            INSERT INTO bet_ledger (
                bet_id, bet_kind, status, settled_at, fill_size_usdc, fill_price,
                polymarket_token_id, side_label, strategy_label
            ) VALUES (1, 'real', 'filled', NULL, 25.0, 0.5, 'token-open', 'Yes', 's1')
            """
        )
        conn.commit()

        pm = MagicMock()
        pm.get_usdc_balance.return_value = {"balance_usdc": 799.93}
        pm.get_token_midpoints.return_value = {"token-open": 0.4}
        pm.get_data_api_positions.return_value = [
            {
                "asset": "token-win",
                "redeemable": True,
                "currentValue": 134.54,
                "size": 134.54,
                "title": "Winner",
                "outcome": "Yes",
                "initialValue": 50.0,
            },
        ]

        breakdown = live_bankroll.get_portfolio_breakdown(conn, pm=pm)
        self.assertAlmostEqual(breakdown["wallet_cash_usdc"], 799.93)
        self.assertAlmostEqual(breakdown["open_positions_market_value_usdc"], 20.0)
        self.assertAlmostEqual(breakdown["redeemable_usdc"], 134.54)
        self.assertAlmostEqual(breakdown["portfolio_value_usdc"], 954.47)


if __name__ == "__main__":
    unittest.main()
