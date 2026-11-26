"""Pin the paper-cohort split (V15).

The legacy paper book is the calibration holdout. These tests assert:

  * the cohort filter defaults to the CURRENT cohort (legacy excluded),
  * legacy is still reachable and complete,
  * 'all' spans both cohorts *including* untagged rows (a partial backfill can
    never hide history),
  * a PARTIAL cashout SELL row inherits its parent's cohort.
"""

from __future__ import annotations

import sqlite3

import pytest

from src.integrations.polymarket.cashout import _insert_partial_cashout_sell_row
from src.integrations.polymarket.paper_cohorts import (
    ALL_COHORTS,
    CURRENT_PAPER_COHORT,
    LEGACY_PAPER_COHORT,
    cohort_filter_sql,
    resolve_cohort_arg,
)


def _db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE bet_ledger (
            bet_id INTEGER PRIMARY KEY AUTOINCREMENT,
            proposed_at TEXT, placed_at TEXT, filled_at TEXT, settled_at TEXT,
            fixture_key TEXT, market_type TEXT,
            polymarket_market_id TEXT, polymarket_token_id TEXT,
            side_label TEXT, model_prob REAL, market_price_at_proposal REAL,
            edge_pp REAL, side TEXT, size_usdc REAL, fees_estimated_usdc REAL,
            fill_price REAL, fill_size_usdc REAL, status TEXT, mode TEXT,
            bet_kind TEXT, strategy_label TEXT, phase TEXT,
            pnl_realised_usdc REAL, cashout_triggered_at TEXT, cashout_price REAL,
            cashout_pnl_usdc REAL, cashout_threshold_used REAL,
            cashout_order_id TEXT, cashout_reason TEXT,
            paper_cohort TEXT
        );
        """
    )
    return conn


def _row(conn, *, bet_kind, cohort, pnl=1.0, strategy="s1"):
    conn.execute(
        """
        INSERT INTO bet_ledger (proposed_at, fixture_key, market_type, side_label,
            model_prob, market_price_at_proposal, edge_pp, side, size_usdc,
            fill_size_usdc, status, mode, bet_kind, strategy_label, phase,
            pnl_realised_usdc, paper_cohort)
        VALUES ('2026-11-01T00:00:00+00:00','f','moneyline','L', 0.6, 0.5, 10.0,
                'BUY', 10.0, 10.0, 'settled', 'auto', ?, ?, 'pre_toss', ?, ?)
        """,
        (bet_kind, strategy, pnl, cohort),
    )


class TestCohortFilter:
    def test_default_is_current_cohort_not_legacy(self):
        assert resolve_cohort_arg(None) == CURRENT_PAPER_COHORT
        assert resolve_cohort_arg("") == CURRENT_PAPER_COHORT
        assert resolve_cohort_arg(CURRENT_PAPER_COHORT) == CURRENT_PAPER_COHORT

    def test_legacy_is_explicitly_selectable(self):
        assert resolve_cohort_arg(LEGACY_PAPER_COHORT) == LEGACY_PAPER_COHORT

    def test_all_is_case_insensitive(self):
        assert resolve_cohort_arg("ALL") == ALL_COHORTS

    def test_unknown_cohort_is_an_error_not_a_silent_fallback(self):
        with pytest.raises(ValueError):
            resolve_cohort_arg("legacy")

    def test_legacy_query_excludes_current_and_real(self):
        conn = _db()
        _row(conn, bet_kind="paper", cohort=LEGACY_PAPER_COHORT, pnl=2.0)
        _row(conn, bet_kind="paper", cohort=CURRENT_PAPER_COHORT, pnl=5.0)
        _row(conn, bet_kind="real", cohort=None, pnl=100.0)

        sql, params = cohort_filter_sql(LEGACY_PAPER_COHORT)
        got = conn.execute(
            f"SELECT COALESCE(SUM(pnl_realised_usdc),0) FROM bet_ledger "
            f"WHERE bet_kind='paper' AND {sql}", params
        ).fetchone()[0]
        assert got == pytest.approx(2.0)

    def test_current_query_excludes_legacy(self):
        conn = _db()
        _row(conn, bet_kind="paper", cohort=LEGACY_PAPER_COHORT, pnl=2.0)
        _row(conn, bet_kind="paper", cohort=CURRENT_PAPER_COHORT, pnl=5.0)

        sql, params = cohort_filter_sql(CURRENT_PAPER_COHORT)
        got = conn.execute(
            f"SELECT COALESCE(SUM(pnl_realised_usdc),0) FROM bet_ledger "
            f"WHERE bet_kind='paper' AND {sql}", params
        ).fetchone()[0]
        assert got == pytest.approx(5.0)

    def test_all_includes_untagged_rows(self):
        """A partially backfilled table must not lose rows from the 'all' view."""
        conn = _db()
        _row(conn, bet_kind="paper", cohort=LEGACY_PAPER_COHORT, pnl=2.0)
        _row(conn, bet_kind="paper", cohort=CURRENT_PAPER_COHORT, pnl=5.0)
        _row(conn, bet_kind="paper", cohort=None, pnl=0.5)

        sql, params = cohort_filter_sql(ALL_COHORTS)
        got = conn.execute(
            f"SELECT COUNT(*), ROUND(SUM(pnl_realised_usdc),6) FROM bet_ledger "
            f"WHERE bet_kind='paper' AND {sql}", params
        ).fetchone()
        assert got[0] == 3
        assert got[1] == pytest.approx(7.5)

    def test_untagged_current_rows_are_treated_as_legacy(self):
        """A pre-backfill paper row must not leak into the default view."""
        conn = _db()
        _row(conn, bet_kind="paper", cohort=None, pnl=3.0)

        sql, params = cohort_filter_sql(CURRENT_PAPER_COHORT)
        got = conn.execute(
            f"SELECT COUNT(*) FROM bet_ledger WHERE bet_kind='paper' AND {sql}",
            params,
        ).fetchone()[0]
        assert got == 0


class TestPartialCashoutCohort:
    def test_partial_sell_row_inherits_parent_cohort(self):
        conn = _db()
        _row(conn, bet_kind="paper", cohort=LEGACY_PAPER_COHORT)
        parent = conn.execute("SELECT * FROM bet_ledger").fetchone()

        sell_id = _insert_partial_cashout_sell_row(
            parent,
            avg_price=0.55,
            entry_cost_sold=5.0,
            proceeds=5.5,
            fee=0.11,
            cashout_pnl=0.39,
            return_ratio=1.1,
            cashout_oid="oid",
            reason="profit",
            now="2026-11-08T00:00:00+00:00",
            conn=conn,
        )
        row = conn.execute(
            "SELECT bet_kind, paper_cohort FROM bet_ledger WHERE bet_id=?", (sell_id,)
        ).fetchone()
        assert row["bet_kind"] == "paper"
        assert row["paper_cohort"] == LEGACY_PAPER_COHORT
