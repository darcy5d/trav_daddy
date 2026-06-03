"""execute_cashout real-bet fill accounting (the core of the cashout-fill fix).

A real cashout now books ONLY what actually fills on the exchange:
  * full fill   -> the BUY row is settled with the real avg price + PnL,
  * partial fill-> a SELL adjustment row carries the realized PnL and the BUY
                   row stays 'filled' with its stake decremented (retry next tick),
  * zero fill   -> nothing is written; the position is held.

Uses an in-process fake client that serves a book and fills a configurable
fraction (no network).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Generator

import pytest

from src.integrations.polymarket import cashout
from src.integrations.odds.polymarket_compare import POLYMARKET_TAKER_FEE


SCHEMA_SQL = """
CREATE TABLE bet_ledger (
    bet_id INTEGER PRIMARY KEY AUTOINCREMENT,
    proposed_at TEXT, placed_at TEXT, filled_at TEXT, settled_at TEXT,
    fixture_key TEXT,
    market_type TEXT,
    polymarket_market_id TEXT,
    polymarket_token_id TEXT,
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
    bankroll_after_settle REAL,
    status TEXT NOT NULL,
    mode TEXT NOT NULL,
    bet_kind TEXT,
    strategy_label TEXT,
    phase TEXT,
    kickoff_at TEXT,
    cashout_triggered_at TEXT,
    cashout_price REAL,
    cashout_pnl_usdc REAL,
    cashout_threshold_used REAL,
    cashout_order_id TEXT,
    cashout_reason TEXT
);
CREATE TABLE order_history (
    polymarket_order_id TEXT PRIMARY KEY, bet_id INTEGER, chunk_id INTEGER,
    plan_id INTEGER, token_id TEXT, side TEXT, order_kind TEXT,
    limit_price REAL, size_usdc REAL, size_shares REAL, posted_at TEXT,
    final_status TEXT, final_reason TEXT, fill_usdc REAL, fill_price REAL,
    filled_at TEXT, last_seen_at TEXT, replaced_by_order_id TEXT,
    created_at TEXT, updated_at TEXT
);
"""


class FakeClient:
    def __init__(self, *, bid_price=0.50, depth=100000.0, fill_fraction=1.0):
        self.bid_price = bid_price
        self.depth = depth
        self.fill_fraction = fill_fraction
        self.placed = []
        self.cancels = []
        self._last_size = 0.0

    def get_clob_order_book(self, token_id):
        return {
            "bids": [{"price": str(self.bid_price), "size": str(self.depth)}],
            "asks": [{"price": str(self.bid_price + 0.02), "size": str(self.depth)}],
            "tick_size": "0.01",
        }

    def place_limit_order(self, *, token_id, side, price, size_shares):
        self.placed.append((token_id, side, price, size_shares))
        self._last_size = size_shares
        return {"orderID": f"oid-{len(self.placed)}",
                "status": "matched" if self.fill_fraction >= 1.0 else "live"}

    def get_order(self, order_id):
        return {
            "size_matched": self._last_size * self.fill_fraction,
            "original_size": self._last_size,
            "status": "matched" if self.fill_fraction >= 1.0 else "live",
        }

    def cancel_order(self, order_id):
        self.cancels.append(order_id)
        return {"canceled": [order_id]}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@pytest.fixture
def db() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    yield conn
    conn.close()


def _insert_real_buy(db, *, fill_price=0.37, fill_size_usdc=73.5375):
    cur = db.execute(
        """
        INSERT INTO bet_ledger (
            proposed_at, placed_at, filled_at, fixture_key, market_type,
            polymarket_market_id, polymarket_token_id, side_label,
            model_prob, market_price_at_proposal, edge_pp, side, size_usdc,
            fill_price, fill_size_usdc, status, mode, bet_kind, strategy_label, phase
        ) VALUES (?, ?, ?, 'crict20blast-ken-sur-2026-05-31', 'moneyline',
                  '2383564', 'TOK-KENT', 'Kent', 0.65, 0.355, 29.85, 'BUY', ?,
                  ?, ?, 'filled', 'auto', 'real', 'v3_marg_3pp', 'pre_toss')
        """,
        (_now(), _now(), _now(), fill_size_usdc, fill_price, fill_size_usdc),
    )
    db.commit()
    return db.execute("SELECT * FROM bet_ledger WHERE bet_id = ?", (cur.lastrowid,)).fetchone()


def test_full_real_cashout_settles_with_actual_fill(db):
    bet = _insert_real_buy(db)
    fake = FakeClient(bid_price=0.50, fill_fraction=1.0)
    res = cashout.execute_cashout(bet_row=bet, cashout_price=0.50, conn=db,
                                  poly_client=fake, reason="profit")
    assert res["success"] is True
    assert res["partial"] is False
    assert res["is_simulated"] is False
    # 73.5375 / 0.37 = 198.75 shares sold @ 0.50.
    shares = 73.5375 / 0.37
    proceeds = shares * 0.50
    expected_pnl = round(proceeds - 73.5375 - proceeds * POLYMARKET_TAKER_FEE, 4)
    assert res["cashout_pnl"] == pytest.approx(expected_pnl, rel=1e-3)
    assert fake.placed[0][1] == "SELL" and fake.placed[0][2] == pytest.approx(0.47)

    row = db.execute("SELECT * FROM bet_ledger WHERE bet_id=?", (bet["bet_id"],)).fetchone()
    assert row["status"] == "settled"
    assert row["fill_size_usdc"] == pytest.approx(0.0, abs=1e-6)
    assert row["cashout_order_id"] == "oid-1"
    assert row["cashout_reason"] == "profit"
    assert row["cashout_price"] == pytest.approx(0.50, rel=1e-3)
    assert row["pnl_realised_usdc"] == pytest.approx(expected_pnl, rel=1e-3)
    # No separate SELL row for a clean full exit.
    n_sell = db.execute("SELECT COUNT(*) n FROM bet_ledger WHERE side='SELL'").fetchone()["n"]
    assert n_sell == 0


def test_partial_real_cashout_books_sell_row_and_keeps_open(db):
    bet = _insert_real_buy(db)
    fake = FakeClient(bid_price=0.50, fill_fraction=0.5)
    res = cashout.execute_cashout(bet_row=bet, cashout_price=0.50, conn=db,
                                  poly_client=fake, reason="profit")
    assert res["success"] is True
    assert res["partial"] is True
    # Half filled -> entry cost sold = ~36.77; BUY row stays open with the rest.
    buy = db.execute("SELECT * FROM bet_ledger WHERE bet_id=?", (bet["bet_id"],)).fetchone()
    assert buy["status"] == "filled"
    assert buy["fill_size_usdc"] == pytest.approx(73.5375 / 2, rel=1e-2)
    # Resting remainder cancelled (no orphan).
    assert len(fake.cancels) == 1
    # A settled SELL adjustment row carries the realized PnL and is tagged as cashout.
    sell = db.execute(
        "SELECT * FROM bet_ledger WHERE side='SELL'"
    ).fetchone()
    assert sell is not None
    assert sell["status"] == "settled"
    assert sell["cashout_triggered_at"] is not None
    assert sell["cashout_reason"] == "profit"
    assert sell["pnl_realised_usdc"] == pytest.approx(res["cashout_pnl"], rel=1e-3)


def test_zero_fill_real_cashout_writes_nothing(db):
    bet = _insert_real_buy(db)
    fake = FakeClient(bid_price=0.40)  # best bid below floor (0.50-0.03=0.47)
    res = cashout.execute_cashout(bet_row=bet, cashout_price=0.50, conn=db,
                                  poly_client=fake, reason="profit")
    assert res["success"] is False
    assert "sell-not-filled" in res["error"]
    assert fake.placed == []
    row = db.execute("SELECT * FROM bet_ledger WHERE bet_id=?", (bet["bet_id"],)).fetchone()
    assert row["status"] == "filled"
    assert row["fill_size_usdc"] == pytest.approx(73.5375, rel=1e-9)
    assert row["cashout_triggered_at"] is None
    n = db.execute("SELECT COUNT(*) n FROM bet_ledger").fetchone()["n"]
    assert n == 1  # no SELL row added
