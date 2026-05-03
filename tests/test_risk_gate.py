"""Unit tests for src/integrations/polymarket/risk_gate.py (Wave 5 Phase 6c)."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Generator
from unittest.mock import patch

import pytest

from src.integrations.polymarket.risk_gate import (
    can_place_bet,
    get_risk_status,
)


@pytest.fixture
def in_memory_ledger() -> Generator[sqlite3.Connection, None, None]:
    """SQLite in-memory connection with the bet_ledger schema applied.

    Schema mirrors V5 (schema_v5_betting.sql) + V6 paper-trading extensions
    (schema_v6_paper_betting.sql) + phase/xi/toss tracking columns used by
    paper_bet_scan.py and bet_placement.py. All columns default to NULL so
    pre-existing tests that don't set them still pass.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    schema_sql = (
        "CREATE TABLE bet_ledger ("
        "  bet_id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  proposed_at TEXT NOT NULL,"
        "  placed_at TEXT,"
        "  filled_at TEXT,"
        "  cancelled_at TEXT,"
        "  settled_at TEXT,"
        "  match_id INTEGER,"
        "  fixture_key TEXT NOT NULL,"
        "  market_type TEXT NOT NULL,"
        "  polymarket_market_id TEXT,"
        "  polymarket_token_id TEXT,"
        "  polymarket_order_id TEXT,"
        "  side_label TEXT,"
        "  model_prob REAL NOT NULL,"
        "  market_price_at_proposal REAL NOT NULL,"
        "  edge_pp REAL NOT NULL,"
        "  side TEXT NOT NULL,"
        "  size_usdc REAL NOT NULL,"
        "  fees_estimated_usdc REAL,"
        "  fill_price REAL,"
        "  fill_size_usdc REAL,"
        "  settle_outcome INTEGER,"
        "  pnl_realised_usdc REAL,"
        "  status TEXT NOT NULL,"
        "  mode TEXT NOT NULL,"
        "  error_message TEXT,"
        "  bet_kind TEXT,"
        "  strategy_label TEXT,"
        "  bankroll_at_proposal REAL,"
        "  bankroll_after_settle REAL,"
        "  phase TEXT,"
        "  xi_signature TEXT,"
        "  toss_winner_team_id INTEGER,"
        "  toss_chose_to TEXT"
        ")"
    )
    conn.execute(schema_sql)
    yield conn
    conn.close()


def _insert_bet(conn, **kwargs):
    defaults = {
        "proposed_at": datetime.now(timezone.utc).isoformat(),
        "fixture_key": "team1_team2_2026-04-19",
        "market_type": "moneyline",
        "model_prob": 0.55,
        "market_price_at_proposal": 0.50,
        "edge_pp": 5.0,
        "side": "BUY",
        "size_usdc": 25.0,
        "status": "proposed",
        "mode": "manual",
    }
    defaults.update(kwargs)
    cols = ",".join(defaults.keys())
    placeholders = ",".join("?" * len(defaults))
    conn.execute(f"INSERT INTO bet_ledger ({cols}) VALUES ({placeholders})", list(defaults.values()))
    conn.commit()


def _make_betting_config(**overrides):
    defaults = {
        "mode": "MANUAL",
        "kill_switch": False,
        "max_deposit_usdc": 200.0,
        "max_per_bet_usdc": 25.0,
        "max_per_day_usdc": 50.0,
        "max_loss_per_day_usdc": 30.0,
        "auto_min_edge_pp": 5.0,
        "auto_enabled_markets": ["moneyline"],
        "scale_up_min_settled_bets": 50,
        "scale_up_max_brier_drift": 0.02,
        # Wave 5.8: per-strategy envelope. Empty live_strategies list means any
        # bet with a strategy_label will be rejected — matches production default.
        "max_deposit_per_strategy_usdc": 100.0,
        "live_strategies": [],
    }
    defaults.update(overrides)
    return defaults


def test_kill_switch_blocks_everything(in_memory_ledger):
    with patch("config.BETTING_CONFIG", _make_betting_config(kill_switch=True)):
        decision = can_place_bet(25.0, "moneyline", 5.0, "manual", conn=in_memory_ledger)
    assert decision.allowed is False
    assert "kill switch" in decision.reason.lower()


def test_mode_off_blocks(in_memory_ledger):
    with patch("config.BETTING_CONFIG", _make_betting_config(mode="OFF")):
        decision = can_place_bet(25.0, "moneyline", 5.0, "manual", conn=in_memory_ledger)
    assert decision.allowed is False
    assert "off" in decision.reason.lower()


def test_per_bet_cap(in_memory_ledger):
    with patch("config.BETTING_CONFIG", _make_betting_config(max_per_bet_usdc=25.0)):
        decision = can_place_bet(50.0, "moneyline", 5.0, "manual", conn=in_memory_ledger)
    assert decision.allowed is False
    assert "per-bet cap" in decision.reason


def test_daily_stake_cap(in_memory_ledger):
    now = datetime.now(timezone.utc).isoformat()
    _insert_bet(in_memory_ledger, status="filled",
                placed_at=now, filled_at=now,
                fill_price=0.5, fill_size_usdc=40.0)
    with patch("config.BETTING_CONFIG", _make_betting_config(max_per_day_usdc=50.0)):
        decision = can_place_bet(20.0, "moneyline", 5.0, "manual", conn=in_memory_ledger)
    assert decision.allowed is False
    assert "daily stake cap" in decision.reason


def test_daily_loss_cap(in_memory_ledger):
    now = datetime.now(timezone.utc).isoformat()
    # A settled bet that lost $25 today
    _insert_bet(in_memory_ledger, status="settled",
                placed_at=now, filled_at=now, settled_at=now,
                fill_price=0.5, fill_size_usdc=25.0,
                settle_outcome=0, pnl_realised_usdc=-25.0)
    # Worst case for next $20 bet is -$20 -> total worst case -$45 > $30 cap
    with patch("config.BETTING_CONFIG", _make_betting_config(max_loss_per_day_usdc=30.0)):
        decision = can_place_bet(20.0, "moneyline", 5.0, "manual", conn=in_memory_ledger)
    assert decision.allowed is False
    assert "loss cap" in decision.reason


def test_open_exposure_cap(in_memory_ledger):
    now = datetime.now(timezone.utc).isoformat()
    # An open bet of $180 -> deposit cap $200, so only $20 left
    _insert_bet(in_memory_ledger, status="filled",
                placed_at=now, filled_at=now,
                fill_price=0.5, fill_size_usdc=180.0)
    # Bump caps so we trip the open-exposure check, not the daily-stake or
    # daily-loss caps which would otherwise fire first.
    with patch("config.BETTING_CONFIG", _make_betting_config(
        max_deposit_usdc=200.0,
        max_per_bet_usdc=50.0,
        max_per_day_usdc=500.0,
        max_loss_per_day_usdc=500.0,
    )):
        decision = can_place_bet(50.0, "moneyline", 5.0, "manual", conn=in_memory_ledger)
    assert decision.allowed is False
    assert "open exposure" in decision.reason


def test_auto_mode_market_not_enabled(in_memory_ledger):
    with patch("config.BETTING_CONFIG", _make_betting_config(mode="AUTO", auto_enabled_markets=["moneyline"])):
        decision = can_place_bet(25.0, "top_batter", 10.0, "auto", conn=in_memory_ledger)
    assert decision.allowed is False
    assert "BETTING_AUTO_MARKETS" in decision.reason


def test_auto_mode_min_edge(in_memory_ledger):
    with patch("config.BETTING_CONFIG", _make_betting_config(mode="AUTO", auto_min_edge_pp=5.0)):
        decision = can_place_bet(25.0, "moneyline", 3.0, "auto", conn=in_memory_ledger)
    assert decision.allowed is False
    assert "below auto threshold" in decision.reason


def test_auto_mode_when_global_mode_is_manual(in_memory_ledger):
    with patch("config.BETTING_CONFIG", _make_betting_config(mode="MANUAL")):
        decision = can_place_bet(25.0, "moneyline", 10.0, "auto", conn=in_memory_ledger)
    assert decision.allowed is False
    assert "auto" in decision.reason.lower()


def test_clean_pass_manual(in_memory_ledger):
    with patch("config.BETTING_CONFIG", _make_betting_config()):
        decision = can_place_bet(25.0, "moneyline", 5.0, "manual", conn=in_memory_ledger)
    assert decision.allowed is True
    assert decision.reason == "OK"


def test_clean_pass_auto_with_qualifying_edge(in_memory_ledger):
    with patch("config.BETTING_CONFIG", _make_betting_config(mode="AUTO", auto_enabled_markets=["moneyline"], auto_min_edge_pp=5.0)):
        decision = can_place_bet(25.0, "moneyline", 7.5, "auto", conn=in_memory_ledger)
    assert decision.allowed is True


# ---- Wave 5.8: per-strategy whitelist + cap tests ----

def test_strategy_not_in_whitelist_rejected(in_memory_ledger):
    with patch(
        "config.BETTING_CONFIG",
        _make_betting_config(live_strategies=["v2_odi_3pp"]),
    ):
        decision = can_place_bet(
            25.0, "moneyline", 5.0, "manual",
            conn=in_memory_ledger, strategy_label="not_a_real_strategy",
        )
    assert decision.allowed is False
    assert "not in BETTING_LIVE_STRATEGIES" in decision.reason


def test_strategy_in_whitelist_passes(in_memory_ledger):
    with patch(
        "config.BETTING_CONFIG",
        _make_betting_config(live_strategies=["v2_odi_3pp"]),
    ):
        decision = can_place_bet(
            25.0, "moneyline", 5.0, "manual",
            conn=in_memory_ledger, strategy_label="v2_odi_3pp",
        )
    assert decision.allowed is True
    assert decision.reason == "OK"


def test_strategy_open_exposure_cap(in_memory_ledger):
    """A strategy with $90 open cannot accept a $20 bet when cap is $100."""
    now = datetime.now(timezone.utc).isoformat()
    # Open real bet of $90 under this strategy
    _insert_bet(
        in_memory_ledger, status="filled",
        placed_at=now, filled_at=now,
        fill_price=0.5, fill_size_usdc=90.0,
        bet_kind="real", strategy_label="v2_odi_3pp",
    )
    with patch(
        "config.BETTING_CONFIG",
        _make_betting_config(
            live_strategies=["v2_odi_3pp"],
            max_deposit_per_strategy_usdc=100.0,
            # Bump global caps so the per-strategy cap is what fires
            max_deposit_usdc=500.0,
            max_per_day_usdc=500.0,
            max_loss_per_day_usdc=500.0,
        ),
    ):
        decision = can_place_bet(
            20.0, "moneyline", 5.0, "manual",
            conn=in_memory_ledger, strategy_label="v2_odi_3pp",
        )
    assert decision.allowed is False
    assert "per-strategy cap" in decision.reason


def test_paper_bets_excluded_from_live_caps(in_memory_ledger):
    """Paper bets in bet_ledger must NOT consume live daily-stake or deposit capacity."""
    now = datetime.now(timezone.utc).isoformat()
    # A bunch of paper bets that, if counted, would instantly trip caps
    for _ in range(10):
        _insert_bet(
            in_memory_ledger, status="filled",
            placed_at=now, filled_at=now,
            fill_price=0.5, fill_size_usdc=50.0,
            bet_kind="paper", strategy_label="v2_odi_3pp",
        )
    # Despite $500 of paper "filled today", a fresh $25 real bet should pass
    with patch(
        "config.BETTING_CONFIG",
        _make_betting_config(
            mode="AUTO",
            live_strategies=["v2_odi_3pp"],
            auto_enabled_markets=["moneyline"],
            auto_min_edge_pp=5.0,
        ),
    ):
        decision = can_place_bet(
            25.0, "moneyline", 5.0, "auto",
            conn=in_memory_ledger, strategy_label="v2_odi_3pp",
        )
    assert decision.allowed is True, f"expected pass; got rejected: {decision.reason}"


def test_untagged_bet_bypasses_strategy_whitelist(in_memory_ledger):
    """When strategy_label is None (ad-hoc manual bet from /live-betting UI),
    the whitelist check is skipped so a user can still place one-off bets."""
    with patch(
        "config.BETTING_CONFIG",
        _make_betting_config(live_strategies=[]),  # empty whitelist
    ):
        decision = can_place_bet(
            25.0, "moneyline", 5.0, "manual",
            conn=in_memory_ledger, strategy_label=None,
        )
    assert decision.allowed is True
    assert decision.reason == "OK"
