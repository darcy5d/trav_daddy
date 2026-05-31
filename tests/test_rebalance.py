"""Unit tests for XI-aware rebalancing (src/integrations/polymarket/rebalance.py)
and the SELL/reduce path (src/integrations/polymarket/bet_placement.reduce_position),
plus the shared xi_signature helper.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Generator

import pytest

from src.integrations.polymarket.paper_strategies import STRATEGIES
from src.integrations.polymarket.paper_inputs import compute_xi_signature
from src.integrations.polymarket import rebalance
from src.integrations.polymarket.bet_placement import reduce_position
from src.integrations.odds.polymarket_compare import POLYMARKET_TAKER_FEE


STRAT = next(s for s in STRATEGIES if s.name == "v2_diag_2pp")  # min_edge_pp=2.0, no prob bounds
FK = "crint-aaa-bbb-2026-06-01"


class _FakeClient:
    """Stand-in PolymarketClient: records SELLs, never hits the network."""

    def __init__(self):
        self.sells = []

    def place_limit_order(self, *, token_id, side, price, size_shares):
        self.sells.append((token_id, side, price, size_shares))
        return {"orderID": "fake-sell-1"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@pytest.fixture
def ledger() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE bet_ledger (
            bet_id INTEGER PRIMARY KEY AUTOINCREMENT,
            proposed_at TEXT NOT NULL,
            placed_at TEXT,
            filled_at TEXT,
            cancelled_at TEXT,
            settled_at TEXT,
            match_id INTEGER,
            fixture_key TEXT NOT NULL,
            market_type TEXT NOT NULL,
            polymarket_market_id TEXT,
            polymarket_token_id TEXT,
            polymarket_order_id TEXT,
            side_label TEXT,
            model_prob REAL NOT NULL,
            market_price_at_proposal REAL NOT NULL,
            edge_pp REAL NOT NULL,
            side TEXT NOT NULL,
            size_usdc REAL NOT NULL,
            fees_estimated_usdc REAL,
            fill_price REAL,
            fill_size_usdc REAL,
            settle_outcome INTEGER,
            pnl_realised_usdc REAL,
            status TEXT NOT NULL,
            mode TEXT NOT NULL,
            error_message TEXT,
            bet_kind TEXT,
            strategy_label TEXT,
            bankroll_at_proposal REAL,
            bankroll_after_settle REAL,
            phase TEXT,
            xi_signature TEXT,
            toss_winner_team_id INTEGER,
            toss_chose_to TEXT,
            kickoff_at TEXT,
            cancel_reason TEXT
        )
        """
    )
    yield conn
    conn.close()


def _insert_filled_buy(
    conn, *, side_label, fill_size_usdc, edge_pp, fill_price=0.50, strategy=STRAT.name
):
    conn.execute(
        """
        INSERT INTO bet_ledger (
            proposed_at, placed_at, filled_at, fixture_key, market_type,
            polymarket_token_id, side_label, model_prob, market_price_at_proposal,
            edge_pp, side, size_usdc, fill_price, fill_size_usdc, status, mode,
            bet_kind, strategy_label, phase
        ) VALUES (?, ?, ?, ?, 'moneyline', 'TOK', ?, ?, ?, ?, 'BUY', ?, ?, ?, 'filled', 'auto', 'real', ?, 'pre_toss')
        """,
        (
            _now(), _now(), _now(), FK, side_label, fill_price + edge_pp / 100.0,
            fill_price, edge_pp, fill_size_usdc, fill_price, fill_size_usdc, strategy,
        ),
    )
    conn.commit()


def _insert_rebalance_sell(conn, *, side_label="TeamB", strategy=STRAT.name):
    conn.execute(
        """
        INSERT INTO bet_ledger (
            proposed_at, fixture_key, market_type, side_label, model_prob,
            market_price_at_proposal, edge_pp, side, size_usdc, status, mode,
            bet_kind, strategy_label, phase, cancel_reason
        ) VALUES (?, ?, 'moneyline', ?, 0.5, 0.5, 0.0, 'SELL', 10.0, 'settled', 'auto', 'real', ?, 'pre_toss', 'rebalance')
        """,
        (_now(), FK, side_label, strategy),
    )
    conn.commit()


# --------------------------------------------------------------------------
# xi_signature
# --------------------------------------------------------------------------

def test_xi_signature_is_stable_and_order_sensitive():
    a = compute_xi_signature([1, 2, 3], [4, 5], [6, 7, 8], [9, 10])
    b = compute_xi_signature([1, 2, 3], [4, 5], [6, 7, 8], [9, 10])
    assert a == b and len(a) == 12
    # A change in any lineup array changes the signature (detects XI swaps).
    assert compute_xi_signature([1, 2, 99], [4, 5], [6, 7, 8], [9, 10]) != a


# --------------------------------------------------------------------------
# target_exposure
# --------------------------------------------------------------------------

def test_target_zero_when_edge_below_threshold():
    assert rebalance.target_exposure(0.51, 0.50, 1000.0, STRAT, edge_pp=0.5) == 0.0


def test_target_positive_when_edge_qualifies():
    tgt = rebalance.target_exposure(0.60, 0.50, 1000.0, STRAT, edge_pp=10.0)
    assert tgt > 0


# --------------------------------------------------------------------------
# decide_rebalance
# --------------------------------------------------------------------------

def test_decide_add_when_no_position(ledger):
    action = rebalance.decide_rebalance(
        ledger, strat=STRAT, fixture_key=FK, chosen_side_label="TeamB",
        model_prob=0.60, market_price=0.50, edge_pp=10.0, bankroll=1000.0,
        minutes_to_kickoff=300.0,
    )
    assert action.action == "add"
    assert action.size_usdc > 0
    assert action.current_exposure == 0.0


def test_decide_hold_within_dollar_tolerance(ledger):
    # Big bet-time edge so the edge-delta guard passes; exposure within 20%.
    _insert_filled_buy(ledger, side_label="TeamB", fill_size_usdc=52.0, edge_pp=30.0)
    action = rebalance.decide_rebalance(
        ledger, strat=STRAT, fixture_key=FK, chosen_side_label="TeamB",
        model_prob=0.60, market_price=0.50, edge_pp=10.0, bankroll=1000.0,
        minutes_to_kickoff=300.0,
    )
    assert action.action == "hold"


def test_decide_hold_when_edge_barely_moved(ledger):
    _insert_filled_buy(ledger, side_label="TeamB", fill_size_usdc=120.0, edge_pp=10.5)
    action = rebalance.decide_rebalance(
        ledger, strat=STRAT, fixture_key=FK, chosen_side_label="TeamB",
        model_prob=0.60, market_price=0.50, edge_pp=10.0, bankroll=1000.0,
        minutes_to_kickoff=300.0,
    )
    assert action.action == "hold"
    assert "edge moved" in action.reason


def test_decide_reduce_when_edge_shrinks(ledger):
    _insert_filled_buy(ledger, side_label="TeamB", fill_size_usdc=120.0, edge_pp=20.0)
    action = rebalance.decide_rebalance(
        ledger, strat=STRAT, fixture_key=FK, chosen_side_label="TeamB",
        model_prob=0.60, market_price=0.50, edge_pp=10.0, bankroll=1000.0,
        minutes_to_kickoff=300.0,
    )
    assert action.action == "reduce"
    assert action.size_usdc > 0
    assert action.target_exposure < action.current_exposure


def test_decide_add_topup_when_edge_grows(ledger):
    _insert_filled_buy(ledger, side_label="TeamB", fill_size_usdc=20.0, edge_pp=30.0)
    action = rebalance.decide_rebalance(
        ledger, strat=STRAT, fixture_key=FK, chosen_side_label="TeamB",
        model_prob=0.60, market_price=0.50, edge_pp=10.0, bankroll=1000.0,
        minutes_to_kickoff=300.0,
    )
    assert action.action == "add"
    assert action.size_usdc > 0


def test_decide_exit_flip_when_model_changes_side(ledger):
    _insert_filled_buy(ledger, side_label="TeamA", fill_size_usdc=80.0, edge_pp=10.0)
    action = rebalance.decide_rebalance(
        ledger, strat=STRAT, fixture_key=FK, chosen_side_label="TeamB",
        model_prob=0.60, market_price=0.50, edge_pp=10.0, bankroll=1000.0,
        minutes_to_kickoff=300.0,
    )
    assert action.action == "exit_flip"
    assert action.exits and action.exits[0][0] == "TeamA"


def test_freeze_blocks_reduce(ledger):
    _insert_filled_buy(ledger, side_label="TeamB", fill_size_usdc=120.0, edge_pp=20.0)
    action = rebalance.decide_rebalance(
        ledger, strat=STRAT, fixture_key=FK, chosen_side_label="TeamB",
        model_prob=0.60, market_price=0.50, edge_pp=10.0, bankroll=1000.0,
        minutes_to_kickoff=5.0,  # inside the default 20-min freeze
    )
    assert action.action == "hold"
    assert "frozen" in action.reason


def test_freeze_blocks_exit_flip(ledger):
    _insert_filled_buy(ledger, side_label="TeamA", fill_size_usdc=80.0, edge_pp=10.0)
    action = rebalance.decide_rebalance(
        ledger, strat=STRAT, fixture_key=FK, chosen_side_label="TeamB",
        model_prob=0.60, market_price=0.50, edge_pp=10.0, bankroll=1000.0,
        minutes_to_kickoff=5.0,
    )
    assert action.action == "hold"
    assert "frozen" in action.reason


def test_max_per_fixture_churn_guard(ledger):
    _insert_filled_buy(ledger, side_label="TeamB", fill_size_usdc=120.0, edge_pp=20.0)
    for _ in range(6):
        _insert_rebalance_sell(ledger, side_label="TeamB")
    action = rebalance.decide_rebalance(
        ledger, strat=STRAT, fixture_key=FK, chosen_side_label="TeamB",
        model_prob=0.60, market_price=0.50, edge_pp=10.0, bankroll=1000.0,
        minutes_to_kickoff=300.0,
    )
    assert action.action == "hold"
    assert "max-rebalances" in action.reason


# --------------------------------------------------------------------------
# reduce_position (SELL path)
# --------------------------------------------------------------------------

def test_reduce_position_sells_and_decrements(ledger):
    _insert_filled_buy(ledger, side_label="TeamB", fill_size_usdc=100.0, edge_pp=10.0, fill_price=0.50)
    fake = _FakeClient()
    res = reduce_position(
        strategy_label=STRAT.name,
        fixture_key=FK,
        side_label="TeamB",
        polymarket_token_id="TOK",
        reduce_usdc=50.0,
        current_price=0.60,
        conn=ledger,
        poly_client=fake,
        dry_run=False,
    )
    assert res["success"] is True
    assert len(fake.sells) == 1 and fake.sells[0][1] == "SELL"
    # 100 USDC @ 0.50 = 200 shares; selling half = 100 shares.
    assert res["shares_sold"] == pytest.approx(100.0, rel=1e-3)
    expected_proceeds = 100.0 * 0.60
    expected_pnl = round(expected_proceeds - 50.0 - expected_proceeds * POLYMARKET_TAKER_FEE, 4)
    assert res["realized_pnl_usdc"] == pytest.approx(expected_pnl, rel=1e-3)

    # The BUY row's open stake was decremented by the entry-cost removed.
    cur = ledger.cursor()
    cur.execute("SELECT fill_size_usdc FROM bet_ledger WHERE side='BUY'")
    assert cur.fetchone()["fill_size_usdc"] == pytest.approx(50.0, rel=1e-3)
    # A SELL adjustment row carries the realized pnl.
    cur.execute("SELECT pnl_realised_usdc, cancel_reason FROM bet_ledger WHERE side='SELL'")
    sell = cur.fetchone()
    assert sell["cancel_reason"] == "rebalance"
    assert sell["pnl_realised_usdc"] == pytest.approx(expected_pnl, rel=1e-3)


def test_reduce_position_noop_when_no_position(ledger):
    res = reduce_position(
        strategy_label=STRAT.name,
        fixture_key=FK,
        side_label="TeamB",
        polymarket_token_id="TOK",
        reduce_usdc=50.0,
        current_price=0.60,
        conn=ledger,
        dry_run=True,
    )
    assert res["success"] is False
    assert res["reason"] == "no-open-position-to-reduce"


# --------------------------------------------------------------------------
# Wiring regression guards (behaviour otherwise only exercised by a full scan)
# --------------------------------------------------------------------------

def _scan_source() -> str:
    import scripts.live_bet_scan as m
    with open(m.__file__, "r") as fh:
        return fh.read()


def test_skip_resim_shortcircuit_removed():
    """The 'all eligible have bets -> skip sim' optimisation must be gone so
    updated CREX lineups are re-simulated every scan."""
    src = _scan_source()
    assert "all_eligible_have_bets" not in src


def test_pre_toss_xi_signature_is_wired():
    """Pre-toss live placements must stamp the lineup signature."""
    src = _scan_source()
    assert src.count('xi_signature=fix.get("_xi_signature")') >= 2


def test_full_exit_settles_buy_row(ledger):
    _insert_filled_buy(ledger, side_label="TeamB", fill_size_usdc=100.0, edge_pp=10.0, fill_price=0.50)
    res = reduce_position(
        strategy_label=STRAT.name,
        fixture_key=FK,
        side_label="TeamB",
        polymarket_token_id="TOK",
        reduce_usdc=100.0,  # sell the whole position
        current_price=0.60,
        conn=ledger,
        poly_client=_FakeClient(),
        dry_run=False,
    )
    assert res["success"] is True
    cur = ledger.cursor()
    cur.execute("SELECT status, fill_size_usdc FROM bet_ledger WHERE side='BUY'")
    row = cur.fetchone()
    assert row["status"] == "settled"
    assert row["fill_size_usdc"] == pytest.approx(0.0, abs=1e-6)


def test_dry_run_is_preview_only_no_db_or_order(ledger):
    """A dry-run reduce must NOT place an order or mutate the ledger."""
    _insert_filled_buy(ledger, side_label="TeamB", fill_size_usdc=100.0, edge_pp=10.0, fill_price=0.50)
    fake = _FakeClient()
    res = reduce_position(
        strategy_label=STRAT.name,
        fixture_key=FK,
        side_label="TeamB",
        polymarket_token_id="TOK",
        reduce_usdc=50.0,
        current_price=0.60,
        conn=ledger,
        poly_client=fake,
        dry_run=True,
    )
    assert res["success"] is True and res["dry_run"] is True
    assert res["shares_sold"] == pytest.approx(100.0, rel=1e-3)
    # No CLOB order placed.
    assert fake.sells == []
    cur = ledger.cursor()
    # BUY row untouched; no SELL row written.
    cur.execute("SELECT fill_size_usdc, status FROM bet_ledger WHERE side='BUY'")
    row = cur.fetchone()
    assert row["fill_size_usdc"] == pytest.approx(100.0, rel=1e-9)
    assert row["status"] == "filled"
    cur.execute("SELECT COUNT(*) n FROM bet_ledger WHERE side='SELL'")
    assert cur.fetchone()["n"] == 0
