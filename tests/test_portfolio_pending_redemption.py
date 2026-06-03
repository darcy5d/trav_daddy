"""Portfolio value must count resolved winners we still hold on-chain.

When a market resolves, Polymarket's `redeemable` flag flips to True only after
on-chain resolution finalizes — which lags the price converging to $1. In that
window a winning token is worth ~$1/share but is neither an open ledger position
(it's settled) nor flagged redeemable, so the old calc dropped it entirely.

This is exactly what happened with the Kent v Surrey win: ~$544 of winning
tokens vanished from portfolio value (showed ~$700 instead of ~$1,250).

A token counts as "pending redemption" only when BOTH:
  * we settled a real BUY for it as a winner (settle_outcome=1) — our reconcile
    only settles once the match actually resolved, so a live favourite at 99%
    is never eligible, and
  * its on-chain mid has converged to >= PENDING_REDEMPTION_MIN_PRICE.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Generator, List, Dict, Any

import pytest

from src.integrations.polymarket import live_bankroll as lb


SCHEMA_SQL = """
CREATE TABLE bet_ledger (
    bet_id INTEGER PRIMARY KEY AUTOINCREMENT,
    proposed_at TEXT, placed_at TEXT, filled_at TEXT, settled_at TEXT,
    fixture_key TEXT, market_type TEXT,
    polymarket_market_id TEXT, polymarket_token_id TEXT,
    side_label TEXT, side TEXT,
    size_usdc REAL, fill_price REAL, fill_size_usdc REAL,
    settle_outcome INTEGER, pnl_realised_usdc REAL,
    status TEXT NOT NULL, mode TEXT, bet_kind TEXT, strategy_label TEXT
);
"""

# Real Kent v Surrey moneyline token (the actual on-chain winner).
KENT = "63481202800640501291080291426432074042406150825592145646442778681524904627806"
BAHRAIN = "503252026930"   # genuinely open position (mid-priced)
ENGLAND = "207430017965"   # live favourite, NOT settled in our ledger
GLOUCS = "861270224341"    # already flagged redeemable on-chain


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class FakePM:
    def __init__(self, positions: List[Dict[str, Any]], cash: float = 390.30):
        self._positions = positions
        self._cash = cash

    def get_usdc_balance(self):
        return {"balance_usdc": self._cash}

    def get_token_midpoints(self, tokens):
        # No mids -> open positions fall back to cost basis (keeps the test
        # focused on the redeemable/pending split, not MTM drift).
        return {}

    def get_data_api_positions(self, *args, **kwargs):
        return self._positions


@pytest.fixture
def db() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    yield conn
    conn.close()


def _insert(db, **kw):
    cols = ", ".join(kw.keys())
    ph = ", ".join("?" for _ in kw)
    db.execute(f"INSERT INTO bet_ledger ({cols}) VALUES ({ph})", tuple(kw.values()))
    db.commit()


def _seed_kent_win(db):
    # Real Kent BUY settled as a winner — the canonical pending-redemption case.
    _insert(db, polymarket_token_id=KENT, side_label="Kent", side="BUY",
            size_usdc=73.5375, fill_price=0.37, fill_size_usdc=73.5375,
            settle_outcome=1, status="settled", settled_at=_now(),
            bet_kind="real", strategy_label="v3", mode="auto", proposed_at=_now())


# --------------------------------------------------------------------------- #
# settled-won token gate
# --------------------------------------------------------------------------- #

def test_settled_won_token_ids_only_real_winning_buys(db):
    _seed_kent_win(db)
    # paper winner -> excluded
    _insert(db, polymarket_token_id="PAPER", side_label="X", side="BUY",
            settle_outcome=1, status="settled", bet_kind="paper", mode="auto",
            proposed_at=_now())
    # real loser -> excluded
    _insert(db, polymarket_token_id="LOSER", side_label="Y", side="BUY",
            settle_outcome=0, status="settled", bet_kind="real", mode="auto",
            proposed_at=_now())
    # real winner not yet settled -> excluded
    _insert(db, polymarket_token_id="OPEN", side_label="Z", side="BUY",
            status="filled", bet_kind="real", mode="auto", proposed_at=_now())

    assert lb._settled_won_token_ids(db) == {KENT}


# --------------------------------------------------------------------------- #
# redeemable / pending split
# --------------------------------------------------------------------------- #

def test_pending_redemption_counts_settled_winner_before_flag_flips(db):
    _seed_kent_win(db)
    positions = [
        # Kent: won, mid at $1, but redeemable flag not flipped yet -> pending.
        {"asset": KENT, "outcome": "Kent", "size": 198.75, "curPrice": 1.0,
         "currentValue": 198.5, "redeemable": False, "title": "Surrey vs Kent"},
        # Gloucs: flagged redeemable -> claimable now.
        {"asset": GLOUCS, "outcome": "Gloucestershire", "size": 51.0,
         "curPrice": 1.0, "currentValue": 51.05, "redeemable": True,
         "title": "Yorkshire vs Gloucs"},
        # England: live favourite at 99% but NOT settled-won -> excluded
        # (guards the "comeback from 1%" edge case).
        {"asset": ENGLAND, "outcome": "England", "size": 60.0, "curPrice": 0.99,
         "currentValue": 59.4, "redeemable": False, "title": "England vs India"},
    ]
    redeemable_now, pending, rows = lb._redeemable_positions_from_pm(
        FakePM(positions), settled_won_token_ids=lb._settled_won_token_ids(db),
        exclude_token_ids=set(),
    )

    assert redeemable_now == pytest.approx(51.05)
    assert pending == pytest.approx(198.5)
    assert {r["token_id"]: r["claim_status"] for r in rows} == {
        GLOUCS: "redeemable",
        KENT: "pending_redemption",
    }
    assert ENGLAND not in {r["token_id"] for r in rows}


def test_settled_winner_below_price_floor_is_not_pending(db):
    """Ledger says won but on-chain mid hasn't converged -> data inconsistency,
    excluded by the secondary price gate."""
    _seed_kent_win(db)
    positions = [
        {"asset": KENT, "outcome": "Kent", "size": 100.0, "curPrice": 0.80,
         "currentValue": 80.0, "redeemable": False, "title": "Surrey vs Kent"},
    ]
    redeemable_now, pending, rows = lb._redeemable_positions_from_pm(
        FakePM(positions), settled_won_token_ids=lb._settled_won_token_ids(db),
        exclude_token_ids=set(),
    )
    assert (redeemable_now, pending, rows) == (0.0, 0.0, [])


def test_open_ledger_token_excluded_from_redeemable(db):
    _seed_kent_win(db)
    positions = [
        {"asset": KENT, "outcome": "Kent", "size": 100.0, "curPrice": 1.0,
         "currentValue": 100.0, "redeemable": False, "title": "Surrey vs Kent"},
    ]
    # Pretend Kent is also an open position -> must not double count.
    redeemable_now, pending, rows = lb._redeemable_positions_from_pm(
        FakePM(positions), settled_won_token_ids=lb._settled_won_token_ids(db),
        exclude_token_ids={KENT},
    )
    assert (redeemable_now, pending, rows) == (0.0, 0.0, [])


# --------------------------------------------------------------------------- #
# end-to-end portfolio breakdown
# --------------------------------------------------------------------------- #

def test_portfolio_value_includes_pending_redemption(db, monkeypatch):
    import config
    monkeypatch.setitem(config.POLYMARKET_CONFIG, "private_key", "0xtestkey")

    _seed_kent_win(db)
    # An open Bahrain position (filled, unsettled) marked to cost basis.
    _insert(db, polymarket_token_id=BAHRAIN, side_label="Bahrain", side="BUY",
            size_usdc=100.0, fill_price=0.50, fill_size_usdc=100.0,
            status="filled", bet_kind="real", strategy_label="v3", mode="auto",
            proposed_at=_now())

    positions = [
        {"asset": KENT, "outcome": "Kent", "size": 198.75, "curPrice": 1.0,
         "currentValue": 198.5, "redeemable": False, "title": "Surrey vs Kent"},
        {"asset": BAHRAIN, "outcome": "Bahrain", "size": 200.0, "curPrice": 0.50,
         "currentValue": 100.0, "redeemable": False, "title": "Bahrain"},
    ]
    pm = FakePM(positions, cash=390.30)

    bd = lb.get_portfolio_breakdown(db, pm=pm)

    assert bd["wallet_driven"] is True
    assert bd["wallet_cash_usdc"] == pytest.approx(390.30)
    assert bd["open_positions_market_value_usdc"] == pytest.approx(100.0)  # cost basis
    assert bd["redeemable_usdc"] == pytest.approx(0.0)
    assert bd["pending_redemption_usdc"] == pytest.approx(198.5)
    assert bd["pending_redemption_count"] == 1
    # cash + open + pending = 390.30 + 100 + 198.5
    assert bd["portfolio_value_usdc"] == pytest.approx(688.8)
