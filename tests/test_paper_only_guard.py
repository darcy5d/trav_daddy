"""Pin the paper-only guarantee for real-order submission.

Background: on 2026-11-08 the operator switched the system to paper-only by
setting BETTING_MODE=OFF, BETTING_KILL_SWITCH=1 and BETTING_LIVE_STRATEGIES=""
(the only supported modes are OFF | MANUAL | AUTO — there is no "PAPER").

These tests fail if a future change silently re-opens the real-order path. They
exercise the two layers that must BOTH refuse:

  1. src.integrations.polymarket.risk_gate.can_place_bet  (the server-side gate)
  2. src.integrations.polymarket.client.assert_live_ordering_enabled
     (the last-line hard stop, called before the CLOB SDK is even constructed)

No network is touched: the client under test raises if its SDK factory is ever
reached, and every risk-gate call supplies an empty in-memory ledger connection
so the live database is never opened.
"""

from __future__ import annotations

import sqlite3
import sys
import types
import unittest
from unittest.mock import patch

# config/app pull in heavy optional deps at import time; stub the ones that are
# irrelevant to this gate (same approach as test_betting_place_gate).
if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = types.SimpleNamespace(
        config=types.SimpleNamespace(
            threading=types.SimpleNamespace(
                set_inter_op_parallelism_threads=lambda *a, **k: None,
                set_intra_op_parallelism_threads=lambda *a, **k: None,
            ),
            optimizer=types.SimpleNamespace(set_jit=lambda *a, **k: None),
        )
    )
if "flask_cors" not in sys.modules:
    sys.modules["flask_cors"] = types.SimpleNamespace(CORS=lambda *a, **k: None)

from src.integrations.polymarket.client import (
    LiveOrderingDisabled,
    PolymarketClient,
    assert_live_ordering_enabled,
)
from src.integrations.polymarket.risk_gate import can_place_bet, set_mode


def _cfg(**overrides):
    """BETTING_CONFIG with generous caps so ONLY the disarm gates can decide."""
    cfg = {
        "mode": "OFF",
        "kill_switch": True,
        "live_strategies": [],
        "max_per_bet_usdc": 1000.0,
        "max_per_bet_fraction": 0.0,
        "max_deploy_fraction": 0.0,
        "max_open_fraction_per_strategy": 0.0,
        "max_open_fraction_per_kickoff_day": 0.0,
        "max_per_day_usdc": 1000.0,
        "max_loss_per_day_usdc": 1000.0,
        "max_deposit_per_strategy_usdc": 1000.0,
        "auto_min_edge_pp": 0.0,
        "auto_enabled_markets": ["moneyline"],
        "exit_health_max_stale_min": 0,
        "scale_up_min_settled_bets": 0,
        "live_exclude_prefixes": [],
        "league_policy": {},
        "stop_loss_enabled": False,
        "associate_throttle_enabled": False,
        "associate_kelly_mult": 0.0,
        "model_prob_cap": 0.95,
        "rebalance_no_average_down": False,
    }
    cfg.update(overrides)
    return cfg


def _empty_ledger():
    """Minimal bet_ledger so the gate's aggregates read 0 without a live DB."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE bet_ledger (
            bet_id INTEGER PRIMARY KEY AUTOINCREMENT,
            proposed_at TEXT, filled_at TEXT, settled_at TEXT,
            fixture_key TEXT, market_type TEXT, strategy_label TEXT,
            side TEXT, size_usdc REAL, fill_size_usdc REAL,
            pnl_realised_usdc REAL, status TEXT, mode TEXT, bet_kind TEXT,
            kickoff_at TEXT
        );
        """
    )
    return conn


class _NoSdkClient(PolymarketClient):
    """PolymarketClient whose SDK factory explodes if it is ever reached."""

    def _get_clob_sdk_client(self):  # pragma: no cover - must never run
        raise AssertionError("CLOB SDK constructed — the guard let an order through")

    def get_usdc_balance(self):  # pragma: no cover - must never run
        raise AssertionError("network reached — the guard let an order through")


# A size/edge combo that comfortably passes every numeric cap, so a refusal can
# only come from the mode / kill-switch / whitelist gates.
PASSING_BET = dict(
    proposed_size_usdc=10.0,
    market_type="moneyline",
    edge_pp=8.0,
    requested_mode="auto",
    strategy_label="v3_marg_3pp",
)


class PaperOnlyRiskGateTests(unittest.TestCase):
    def setUp(self):
        self.conn = _empty_ledger()

    def tearDown(self):
        self.conn.close()

    def _gate(self, cfg):
        with patch("config.BETTING_CONFIG", cfg):
            return can_place_bet(**PASSING_BET, conn=self.conn)

    def test_paper_only_config_refuses_and_names_the_kill_switch(self):
        d = self._gate(_cfg())
        self.assertFalse(d.allowed)
        self.assertIn("Kill switch", d.reason)

    def test_mode_off_alone_refuses(self):
        d = self._gate(_cfg(kill_switch=False, mode="OFF"))
        self.assertFalse(d.allowed)
        self.assertIn("BETTING_MODE=OFF", d.reason)

    def test_empty_whitelist_alone_refuses(self):
        d = self._gate(_cfg(kill_switch=False, mode="AUTO", live_strategies=[]))
        self.assertFalse(d.allowed)
        self.assertIn("BETTING_LIVE_STRATEGIES", d.reason)

    def test_untagged_bet_still_refused_on_paper_only_config(self):
        """Omitting a strategy label must not bypass the paper-only posture."""
        with patch("config.BETTING_CONFIG", _cfg()):
            d = can_place_bet(
                **dict(PASSING_BET, strategy_label=None), conn=self.conn
            )
        self.assertFalse(d.allowed)
        self.assertNotEqual(d.reason, "OK")


class PaperOnlyClientGuardTests(unittest.TestCase):
    def test_paper_only_config_blocks_market_order(self):
        c = _NoSdkClient()
        with patch("config.BETTING_CONFIG", _cfg()):
            with self.assertRaises(LiveOrderingDisabled):
                c.place_market_order(token_id="t", side="BUY", amount_usdc=10.0)

    def test_paper_only_config_blocks_limit_order(self):
        c = _NoSdkClient()
        with patch("config.BETTING_CONFIG", _cfg()):
            with self.assertRaises(LiveOrderingDisabled):
                c.place_limit_order(token_id="t", side="SELL", price=0.5, size_shares=10)

    def test_mode_off_alone_blocks_at_client(self):
        with patch("config.BETTING_CONFIG", _cfg(kill_switch=False, mode="OFF")):
            with self.assertRaises(LiveOrderingDisabled):
                assert_live_ordering_enabled()

    def test_manual_mode_blocks_at_client(self):
        with patch("config.BETTING_CONFIG",
                   _cfg(kill_switch=False, mode="MANUAL",
                        live_strategies=["v3_marg_3pp"])):
            with self.assertRaises(LiveOrderingDisabled):
                assert_live_ordering_enabled("v3_marg_3pp")

    def test_empty_whitelist_blocks_at_client(self):
        with patch("config.BETTING_CONFIG",
                   _cfg(kill_switch=False, mode="AUTO", live_strategies=[])):
            with self.assertRaises(LiveOrderingDisabled):
                assert_live_ordering_enabled("v3_marg_3pp")

    def test_strategy_not_whitelisted_blocks_at_client(self):
        with patch("config.BETTING_CONFIG",
                   _cfg(kill_switch=False, mode="AUTO",
                        live_strategies=["consensus_5pp"])):
            with self.assertRaises(LiveOrderingDisabled):
                assert_live_ordering_enabled("v3_marg_3pp")

    def test_guard_passes_only_when_fully_armed(self):
        """Sanity: the guard is liftable, not an unconditional wall."""
        with patch("config.BETTING_CONFIG",
                   _cfg(kill_switch=False, mode="AUTO",
                        live_strategies=["v3_marg_3pp"])):
            assert_live_ordering_enabled("v3_marg_3pp")  # must not raise


class PaperOnlyCashoutSellTests(unittest.TestCase):
    """The cashout/stop-loss scanner is cron-driven and has NO mode gate of its
    own, so the client hard-stop is the only thing standing between an armed
    sell trigger and a real SELL. Pin that it holds.
    """

    def test_marketable_sell_cannot_reach_exchange_while_disarmed(self):
        from src.integrations.polymarket.sell_execution import marketable_sell

        client = _NoSdkClient()
        with patch("config.BETTING_CONFIG", _cfg()):
            with patch.object(
                type(client), "get_clob_order_book",
                lambda self, token_id: {"bids": [{"price": "0.60", "size": "100000"}]},
            ):
                res = marketable_sell(client, "TOKEN", 10.0, reference_price=0.60)

        self.assertFalse(res["success"])
        self.assertIn("refusing to submit a real order", res["reason"])


class SupportedModesTests(unittest.TestCase):
    def test_no_paper_mode_is_invented(self):
        """Only OFF | MANUAL | AUTO are supported; 'PAPER' must be rejected."""
        with patch("config.BETTING_CONFIG", _cfg()), \
                patch("src.integrations.polymarket.risk_gate.write_env_var",
                      return_value=True):
            res = set_mode("PAPER")
        self.assertFalse(res.get("success"))
        self.assertIn("OFF", res.get("error", ""))


if __name__ == "__main__":
    unittest.main()
