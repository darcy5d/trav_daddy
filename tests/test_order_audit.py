"""Unit tests for order_audit and reconcile finalisation paths.

Uses in-memory SQLite so tests are hermetic and fast. The schema mirrors the
minimum needed columns from bet_ledger / order_plans / order_chunks /
order_history.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator

import pytest

from src.integrations.polymarket import order_audit


SCHEMA_SQL = """
CREATE TABLE bet_ledger (
    bet_id INTEGER PRIMARY KEY AUTOINCREMENT,
    proposed_at TEXT NOT NULL,
    placed_at TEXT,
    filled_at TEXT,
    cancelled_at TEXT,
    settled_at TEXT,
    fixture_key TEXT,
    market_type TEXT,
    polymarket_market_id TEXT,
    polymarket_token_id TEXT,
    polymarket_order_id TEXT,
    side_label TEXT,
    side TEXT,
    size_usdc REAL,
    fill_price REAL,
    fill_size_usdc REAL,
    pnl_realised_usdc REAL,
    settle_outcome INTEGER,
    status TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'auto',
    bet_kind TEXT DEFAULT 'real',
    strategy_label TEXT,
    error_message TEXT,
    cancel_reason TEXT,
    error_category TEXT,
    reconciled_at TEXT,
    kickoff_at TEXT
);

CREATE TABLE order_plans (
    plan_id INTEGER PRIMARY KEY AUTOINCREMENT,
    bet_ledger_id INTEGER,
    fixture_key TEXT,
    strategy_label TEXT,
    token_id TEXT,
    side TEXT,
    total_size_usdc REAL,
    chunk_size_usdc REAL,
    chunks_total INTEGER,
    chunks_placed INTEGER DEFAULT 0,
    chunks_filled INTEGER DEFAULT 0,
    filled_size_usdc REAL DEFAULT 0,
    avg_fill_price REAL,
    max_acceptable_price REAL,
    base_price REAL,
    price_step_pp REAL,
    kickoff_at TEXT,
    status TEXT DEFAULT 'pending',
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE order_chunks (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    limit_price REAL,
    size_usdc REAL,
    size_shares REAL,
    polymarket_order_id TEXT,
    status TEXT DEFAULT 'pending',
    placed_at TEXT,
    filled_at TEXT,
    fill_price REAL,
    fill_size_usdc REAL
);

CREATE TABLE order_history (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    polymarket_order_id TEXT NOT NULL UNIQUE,
    bet_id INTEGER,
    chunk_id INTEGER,
    plan_id INTEGER,
    token_id TEXT NOT NULL,
    side TEXT NOT NULL,
    order_kind TEXT NOT NULL,
    limit_price REAL,
    size_usdc REAL,
    size_shares REAL,
    posted_at TEXT NOT NULL,
    final_status TEXT NOT NULL,
    final_reason TEXT,
    replaced_by_order_id TEXT,
    fill_usdc REAL DEFAULT 0,
    fill_price REAL,
    filled_at TEXT,
    last_seen_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT
);
"""


def _utc(delta_minutes: int = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=delta_minutes)).isoformat()


@pytest.fixture
def db() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    yield conn
    conn.close()


def test_record_order_placed_inserts_row(db):
    order_audit.record_order_placed(
        "0xa1",
        bet_id=1, chunk_id=10, plan_id=100,
        token_id="tk1", side="BUY", order_kind="twap_chunk",
        limit_price=0.45, size_usdc=10.0, size_shares=22.2,
        conn=db,
    )
    row = db.execute(
        "SELECT * FROM order_history WHERE polymarket_order_id='0xa1'"
    ).fetchone()
    assert row is not None
    assert row["final_status"] == "placed"
    assert row["bet_id"] == 1
    assert row["chunk_id"] == 10
    assert row["token_id"] == "tk1"


def test_record_order_placed_is_idempotent(db):
    order_audit.record_order_placed(
        "0xa1", token_id="tk1", side="BUY", order_kind="fok",
        bet_id=1, size_usdc=10, conn=db,
    )
    order_audit.record_order_placed(
        "0xa1", token_id="tk1", side="BUY", order_kind="fok",
        bet_id=1, size_usdc=10, conn=db,
    )
    n = db.execute("SELECT COUNT(*) FROM order_history WHERE polymarket_order_id='0xa1'").fetchone()[0]
    assert n == 1


def test_record_order_filled_updates(db):
    order_audit.record_order_placed(
        "0xa1", token_id="tk1", side="BUY", order_kind="fok",
        bet_id=1, size_usdc=10, conn=db,
    )
    order_audit.record_order_filled("0xa1", fill_usdc=9.5, fill_price=0.5, conn=db)
    row = db.execute("SELECT * FROM order_history WHERE polymarket_order_id='0xa1'").fetchone()
    assert row["final_status"] == "filled"
    assert row["fill_usdc"] == 9.5
    assert row["fill_price"] == 0.5


def test_reprice_chain_follows_to_new_order(db):
    order_audit.record_order_placed(
        "0xold", token_id="tk1", side="BUY", order_kind="twap_chunk",
        bet_id=5, chunk_id=42, conn=db,
    )
    order_audit.record_order_replaced_by_reprice("0xold", "0xnew", reason="reprice", conn=db)
    order_audit.record_order_placed(
        "0xnew", token_id="tk1", side="BUY", order_kind="twap_chunk",
        bet_id=5, chunk_id=42, conn=db,
    )
    order_audit.record_order_filled("0xnew", fill_usdc=12.0, fill_price=0.6, conn=db)

    # Lookup of the old id should resolve to the new (filled) row.
    info = order_audit.lookup_bet_for_order("0xold", conn=db)
    assert info is not None
    assert info["polymarket_order_id"] == "0xnew"
    assert info["bet_id"] == 5
    assert info["chunk_id"] == 42
    assert info["final_status"] == "filled"


def test_record_order_cancelled_does_not_override_filled(db):
    order_audit.record_order_placed(
        "0xa1", token_id="tk1", side="BUY", order_kind="fok",
        bet_id=1, conn=db,
    )
    order_audit.record_order_filled("0xa1", fill_usdc=10, conn=db)
    order_audit.record_order_cancelled("0xa1", reason="kickoff_cancel", conn=db)
    row = db.execute("SELECT * FROM order_history WHERE polymarket_order_id='0xa1'").fetchone()
    assert row["final_status"] == "filled"  # not overwritten


def test_record_order_error_creates_row_without_oid(db):
    order_audit.record_order_error(
        None,
        bet_id=7, token_id="tk1", side="BUY", order_kind="fok",
        error_category="order_version_mismatch",
        error_message="API said no",
        conn=db,
    )
    rows = db.execute("SELECT * FROM order_history WHERE bet_id=7").fetchall()
    assert len(rows) == 1
    assert rows[0]["final_status"] == "error"
    assert "order_version_mismatch" in (rows[0]["final_reason"] or "")


def test_all_known_order_ids_unions_sources(db):
    db.execute("INSERT INTO bet_ledger (proposed_at, fixture_key, market_type, status, mode, polymarket_order_id) VALUES (?, ?, ?, ?, ?, ?)",
               (_utc(), "fk", "moneyline", "filled", "auto", "0xfrom_bet"))
    db.execute("INSERT INTO order_plans (created_at, fixture_key, strategy_label, token_id, side, total_size_usdc, chunk_size_usdc, chunks_total, max_acceptable_price, base_price) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
               (_utc(), "fk", "s", "tk", "BUY", 10.0, 10.0, 1, 0.5, 0.4))
    db.execute("INSERT INTO order_chunks (plan_id, chunk_index, limit_price, size_usdc, polymarket_order_id, status) VALUES (?, ?, ?, ?, ?, ?)",
               (1, 0, 0.5, 10.0, "0xfrom_chunk", "placed"))
    order_audit.record_order_placed(
        "0xfrom_history", token_id="tk", side="BUY", order_kind="fok",
        conn=db,
    )
    db.commit()
    ids = order_audit.all_known_order_ids(conn=db)
    assert {"0xfrom_bet", "0xfrom_chunk", "0xfrom_history"}.issubset(ids)


# --- _finalize_stale_proposed_bets ---


def _insert_bet(db, *, status, proposed_at, kickoff_at=None, polymarket_order_id=None):
    db.execute(
        """INSERT INTO bet_ledger (
            proposed_at, fixture_key, market_type, status, mode,
            kickoff_at, polymarket_order_id, side_label, side, size_usdc
        ) VALUES (?, ?, ?, ?, 'auto', ?, ?, ?, 'BUY', 10.0)""",
        (proposed_at, "crint-test-2026-01-01", "moneyline", status, kickoff_at, polymarket_order_id, "Team A"),
    )
    return db.execute("SELECT last_insert_rowid()").fetchone()[0]


def _insert_plan(db, bet_id, status="pending"):
    db.execute(
        """INSERT INTO order_plans (
            bet_ledger_id, fixture_key, strategy_label, token_id, side,
            total_size_usdc, chunk_size_usdc, chunks_total,
            max_acceptable_price, base_price, status, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (bet_id, "crint-test-2026-01-01", "s", "tk", "BUY", 10.0, 10.0, 1, 0.5, 0.4, status, _utc()),
    )
    return db.execute("SELECT last_insert_rowid()").fetchone()[0]


def test_finalize_stale_proposed_marks_never_posted(db):
    from src.integrations.polymarket.reconcile import _finalize_stale_proposed_bets

    bet_old = _insert_bet(db, status="proposed", proposed_at=_utc(-60 * 8), kickoff_at=_utc(-30))
    bet_future = _insert_bet(db, status="proposed", proposed_at=_utc(-60 * 8), kickoff_at=_utc(60 * 8))
    bet_with_plan = _insert_bet(db, status="proposed", proposed_at=_utc(-60 * 8), kickoff_at=_utc(-30))
    _insert_plan(db, bet_with_plan, status="pending")

    n = _finalize_stale_proposed_bets(db)
    assert n == 1  # only the no-plan, kickoff-passed bet

    row_old = db.execute("SELECT * FROM bet_ledger WHERE bet_id=?", (bet_old,)).fetchone()
    row_future = db.execute("SELECT * FROM bet_ledger WHERE bet_id=?", (bet_future,)).fetchone()
    row_with_plan = db.execute("SELECT * FROM bet_ledger WHERE bet_id=?", (bet_with_plan,)).fetchone()
    assert row_old["status"] == "errored"
    assert row_old["error_category"] == "never_posted"
    assert row_old["cancelled_at"] is not None
    assert row_future["status"] == "proposed"
    assert row_with_plan["status"] == "proposed"


def test_finalize_stale_proposed_propagates_plan_outcome(db):
    """A proposed bet whose plan is already cancelled with zero fill should be
    cancelled with reason='twap_no_fill'."""
    from src.integrations.polymarket.reconcile import _finalize_stale_proposed_bets

    bet_id = _insert_bet(db, status="proposed", proposed_at=_utc(-60), kickoff_at=_utc(-10))
    plan_id = _insert_plan(db, bet_id, status="cancelled")
    db.execute(
        """INSERT INTO order_chunks (plan_id, chunk_index, limit_price, size_usdc, status)
           VALUES (?, ?, ?, ?, 'cancelled')""",
        (plan_id, 0, 0.5, 10.0),
    )
    db.commit()

    n = _finalize_stale_proposed_bets(db)
    assert n >= 1
    row = db.execute("SELECT * FROM bet_ledger WHERE bet_id=?", (bet_id,)).fetchone()
    assert row["status"] == "cancelled"
    assert row["cancel_reason"] == "twap_no_fill"
    assert row["cancelled_at"] is not None
    assert row["reconciled_at"] is not None


# --- finalize_plan_from_chunks reprice-chain promotion ---


def test_finalize_plan_from_chunks_promotes_cancelled_bet(db):
    """When a fill arrives late on a chunk, the bet should be promoted from
    cancelled -> filled and stamped with reconciled_at."""
    from src.integrations.polymarket.clob_fills import finalize_plan_from_chunks

    bet_id = _insert_bet(db, status="cancelled", proposed_at=_utc(-120), polymarket_order_id=None)
    plan_id = _insert_plan(db, bet_id, status="executing")
    db.execute(
        """INSERT INTO order_chunks (plan_id, chunk_index, limit_price, size_usdc, polymarket_order_id, status, fill_size_usdc, fill_price)
           VALUES (?, ?, ?, ?, ?, 'filled', ?, ?)""",
        (plan_id, 0, 0.5, 10.0, "0xchunk_oid", 9.6, 0.48),
    )
    db.commit()

    cur = db.cursor()
    result = finalize_plan_from_chunks(db, cur, plan_id)
    assert result["bet_updated"] is True
    row = db.execute("SELECT * FROM bet_ledger WHERE bet_id=?", (bet_id,)).fetchone()
    assert row["status"] == "filled"
    assert row["fill_size_usdc"] == 9.6
    assert row["reconciled_at"] is not None


def test_apply_fill_to_chunk_tops_up_partial_fill(db):
    """A resting limit order that partial-filled and was marked 'filled' should
    accept additional on-chain fills as a top-up."""
    from src.integrations.polymarket.clob_fills import apply_fill_to_chunk

    plan_id = db.execute("""
        INSERT INTO order_plans (
            bet_ledger_id, fixture_key, strategy_label, token_id, side,
            total_size_usdc, chunk_size_usdc, chunks_total,
            max_acceptable_price, base_price, created_at
        ) VALUES (NULL, 'fk', 's', 'tk', 'BUY', 10, 10, 1, 0.5, 0.4, ?)
    """, (_utc(),)).lastrowid
    db.execute(
        """INSERT INTO order_chunks (plan_id, chunk_index, limit_price, size_usdc, polymarket_order_id, status, fill_size_usdc, fill_price)
           VALUES (?, 0, 0.46, 10.0, '0xres', 'filled', 0.59, 0.46)""",
        (plan_id,),
    )
    db.commit()
    chunk_id = db.execute("SELECT chunk_id FROM order_chunks WHERE polymarket_order_id='0xres'").fetchone()[0]

    fill = {"fill_usdc": 10.91, "avg_fill_price": 0.46}
    cur = db.cursor()
    changed = apply_fill_to_chunk(db, cur, chunk_id, fill)
    assert changed is True
    row = db.execute("SELECT fill_size_usdc, status FROM order_chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
    assert row["status"] == "filled"
    assert abs(row["fill_size_usdc"] - 10.91) < 1e-6


def test_apply_fill_to_chunk_ignores_below_threshold(db):
    """Tiny rounding-noise differences (<2c) should not trigger an update."""
    from src.integrations.polymarket.clob_fills import apply_fill_to_chunk

    plan_id = db.execute("""
        INSERT INTO order_plans (
            bet_ledger_id, fixture_key, strategy_label, token_id, side,
            total_size_usdc, chunk_size_usdc, chunks_total,
            max_acceptable_price, base_price, created_at
        ) VALUES (NULL, 'fk', 's', 'tk', 'BUY', 10, 10, 1, 0.5, 0.4, ?)
    """, (_utc(),)).lastrowid
    db.execute(
        """INSERT INTO order_chunks (plan_id, chunk_index, limit_price, size_usdc, polymarket_order_id, status, fill_size_usdc, fill_price)
           VALUES (?, 0, 0.46, 10.0, '0xres2', 'filled', 11.40, 0.46)""",
        (plan_id,),
    )
    db.commit()
    chunk_id = db.execute("SELECT chunk_id FROM order_chunks WHERE polymarket_order_id='0xres2'").fetchone()[0]

    fill = {"fill_usdc": 11.41, "avg_fill_price": 0.46}  # +1c, below threshold
    cur = db.cursor()
    changed = apply_fill_to_chunk(db, cur, chunk_id, fill)
    assert changed is False  # too small a change to trigger update


def test_finalize_plan_tops_up_already_filled_bet(db):
    """A bet whose chunk grew via partial-fill top-up should get fill_size_usdc updated."""
    from src.integrations.polymarket.clob_fills import finalize_plan_from_chunks

    bet_id = _insert_bet(db, status="filled", proposed_at=_utc(-120))
    db.execute(
        """UPDATE bet_ledger SET fill_size_usdc=11.45, fill_price=0.54, filled_at=? WHERE bet_id=?""",
        (_utc(-60), bet_id),
    )
    plan_id = _insert_plan(db, bet_id, status="executing")
    db.execute(
        """INSERT INTO order_chunks (plan_id, chunk_index, limit_price, size_usdc, polymarket_order_id, status, fill_size_usdc, fill_price)
           VALUES (?, 0, 0.54, 11.45, '0xc1', 'filled', 10.86, 0.5447)""",
        (plan_id,),
    )
    db.execute(
        """INSERT INTO order_chunks (plan_id, chunk_index, limit_price, size_usdc, polymarket_order_id, status, fill_size_usdc, fill_price)
           VALUES (?, 1, 0.46, 0.59, '0xc2', 'filled', 10.91, 0.46)""",
        (plan_id,),
    )
    db.commit()

    cur = db.cursor()
    result = finalize_plan_from_chunks(db, cur, plan_id)
    assert result["bet_updated"] is True
    row = db.execute("SELECT fill_size_usdc, status FROM bet_ledger WHERE bet_id=?", (bet_id,)).fetchone()
    assert row["status"] == "filled"
    assert abs(row["fill_size_usdc"] - (10.86 + 10.91)) < 1e-6
