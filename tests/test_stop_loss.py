"""Unit tests for the Wave 5.11 guarded stop-loss (loss mitigation).

Covers:
  - evaluate_stop_loss() truth table: enabled flag, floor, time-gate,
    missing kickoff, untradeable price.
  - execute_cashout(reason="stop") writes cashout_reason='stop', a
    negative-but-mitigated PnL, and marks the row settled (paper bet, no CLOB).

Uses in-memory SQLite so tests are hermetic. The schema mirrors the minimum
bet_ledger columns the stop-loss path touches, including the Wave 5.10/5.11
cashout columns.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Generator

import pytest

from src.integrations.polymarket import cashout


SCHEMA_SQL = """
CREATE TABLE bet_ledger (
    bet_id INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_key TEXT,
    side_label TEXT,
    polymarket_token_id TEXT,
    fill_price REAL,
    fill_size_usdc REAL,
    pnl_realised_usdc REAL,
    settle_outcome INTEGER,
    bankroll_after_settle REAL,
    status TEXT NOT NULL,
    bet_kind TEXT DEFAULT 'real',
    strategy_label TEXT,
    settled_at TEXT,
    kickoff_at TEXT,
    cashout_triggered_at TEXT,
    cashout_price REAL,
    cashout_pnl_usdc REAL,
    cashout_threshold_used REAL,
    cashout_order_id TEXT,
    cashout_reason TEXT
);
"""


def _iso(dt: datetime) -> str:
    return dt.isoformat()


@pytest.fixture
def db() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    yield conn
    conn.close()


def _insert_bet(
    db: sqlite3.Connection,
    *,
    fill_price: float = 0.70,
    fill_size_usdc: float = 10.0,
    kickoff_at: str | None = None,
    bet_kind: str = "paper",
    strategy_label: str | None = None,
) -> sqlite3.Row:
    cur = db.execute(
        """
        INSERT INTO bet_ledger
            (fixture_key, side_label, polymarket_token_id, fill_price,
             fill_size_usdc, status, bet_kind, strategy_label, kickoff_at)
        VALUES (?, ?, ?, ?, ?, 'filled', ?, ?, ?)
        """,
        ("crint-aaa-bbb-2026-05-28", "Team A", "tok-1", fill_price,
         fill_size_usdc, bet_kind, strategy_label, kickoff_at),
    )
    db.commit()
    return db.execute(
        "SELECT * FROM bet_ledger WHERE bet_id = ?", (cur.lastrowid,)
    ).fetchone()


# ---------------------------------------------------------------------------
# stop_loss_config defaults
# ---------------------------------------------------------------------------

def test_stop_loss_config_shape():
    cfg = cashout.stop_loss_config()
    assert set(cfg.keys()) == {"enabled", "floor", "gate_min", "min_exit_price"}
    assert isinstance(cfg["enabled"], bool)
    assert isinstance(cfg["floor"], float)
    assert isinstance(cfg["gate_min"], float)
    assert isinstance(cfg["min_exit_price"], float)


# ---------------------------------------------------------------------------
# evaluate_stop_loss truth table
# ---------------------------------------------------------------------------

CFG = {"enabled": True, "floor": 0.20, "gate_min": 105.0}
NOW = datetime(2026, 5, 28, 18, 0, 0, tzinfo=timezone.utc)


def _bet_with_kickoff(db, minutes_ago: float, fill_price: float = 0.70):
    kickoff = NOW - timedelta(minutes=minutes_ago)
    return _insert_bet(db, fill_price=fill_price, kickoff_at=_iso(kickoff))


def test_fires_below_floor_after_gate(db):
    bet = _bet_with_kickoff(db, minutes_ago=120)  # past the 105m gate
    assert cashout.evaluate_stop_loss(bet, current_price=0.15, cfg=CFG, now=NOW) is True


def test_fires_exactly_at_floor(db):
    bet = _bet_with_kickoff(db, minutes_ago=120)
    assert cashout.evaluate_stop_loss(bet, current_price=0.20, cfg=CFG, now=NOW) is True


def test_holds_above_floor(db):
    bet = _bet_with_kickoff(db, minutes_ago=120)
    assert cashout.evaluate_stop_loss(bet, current_price=0.25, cfg=CFG, now=NOW) is False


def test_holds_before_gate(db):
    bet = _bet_with_kickoff(db, minutes_ago=60)  # before the 105m gate
    assert cashout.evaluate_stop_loss(bet, current_price=0.10, cfg=CFG, now=NOW) is False


def test_fires_exactly_at_gate_boundary(db):
    bet = _bet_with_kickoff(db, minutes_ago=105)
    assert cashout.evaluate_stop_loss(bet, current_price=0.10, cfg=CFG, now=NOW) is True


def test_holds_when_disabled(db):
    bet = _bet_with_kickoff(db, minutes_ago=120)
    disabled = {**CFG, "enabled": False}
    assert cashout.evaluate_stop_loss(bet, current_price=0.05, cfg=disabled, now=NOW) is False


def test_holds_when_kickoff_missing(db):
    bet = _insert_bet(db, fill_price=0.70, kickoff_at=None)
    assert cashout.evaluate_stop_loss(bet, current_price=0.05, cfg=CFG, now=NOW) is False


def test_holds_on_untradeable_price(db):
    bet = _bet_with_kickoff(db, minutes_ago=120)
    assert cashout.evaluate_stop_loss(bet, current_price=0.0, cfg=CFG, now=NOW) is False
    assert cashout.evaluate_stop_loss(bet, current_price=1.0, cfg=CFG, now=NOW) is False


def test_fires_for_favourite_that_collapsed(db):
    # A 0.70 favourite that has cratered to 0.12 is exactly what stop-loss
    # protects — independent of the profit-take tier (which would hold it).
    bet = _bet_with_kickoff(db, minutes_ago=130, fill_price=0.70)
    assert cashout.evaluate_stop_loss(bet, current_price=0.12, cfg=CFG, now=NOW) is True


# ---------------------------------------------------------------------------
# execute_cashout(reason="stop") DB writes
# ---------------------------------------------------------------------------

def test_execute_stop_writes_reason_and_mitigated_pnl(db):
    bet = _bet_with_kickoff(db, minutes_ago=120, fill_price=0.70)
    result = cashout.execute_cashout(
        bet_row=bet,
        cashout_price=0.15,
        conn=db,
        dry_run=False,   # paper bet -> simulated, no CLOB call
        reason="stop",
    )
    assert result["success"] is True
    assert result["reason"] == "stop"
    assert result["is_simulated"] is True  # bet_kind='paper'
    # return_ratio < 1 for a stop.
    assert result["return_ratio"] == pytest.approx(0.15 / 0.70)
    # Mitigated loss: better than -fill_size (full loss) but still negative.
    assert -10.0 < result["cashout_pnl"] < 0.0

    row = db.execute("SELECT * FROM bet_ledger WHERE bet_id = ?", (bet["bet_id"],)).fetchone()
    assert row["status"] == "settled"
    assert row["cashout_reason"] == "stop"
    assert row["settle_outcome"] is None
    assert row["cashout_triggered_at"] is not None
    assert row["pnl_realised_usdc"] == pytest.approx(result["cashout_pnl"])


def test_execute_profit_defaults_reason(db):
    bet = _bet_with_kickoff(db, minutes_ago=120, fill_price=0.20)
    result = cashout.execute_cashout(
        bet_row=bet,
        cashout_price=0.26,
        conn=db,
    )
    assert result["reason"] == "profit"
    row = db.execute("SELECT * FROM bet_ledger WHERE bet_id = ?", (bet["bet_id"],)).fetchone()
    assert row["cashout_reason"] == "profit"
    assert result["cashout_pnl"] > 0.0
