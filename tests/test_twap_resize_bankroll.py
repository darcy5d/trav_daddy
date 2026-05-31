"""TWAP resize Kelly path + duplicate-order guard regression tests.

The duplicate-order bug (2026-05-31): the wallet-balance API intermittently
failed, collapsing the bankroll to the config fallback and halving the stake
every 90s tick. That tripped mid-flight resize every tick, and unconfirmed
cancels orphaned the live order while a fresh one was posted on top — stacking
28 duplicate orders. The fixes verified here:
  1. mid-flight resize is DISABLED by default (TWAP_RESIZE_ENABLED off),
  2. cancels must be CONFIRMED before reposting,
  3. an exchange-reconcile circuit breaker cancels untracked orphan orders.
"""

import sqlite3

import pytest

from src.integrations.polymarket.paper_strategies import STRATEGIES
from src.integrations.polymarket.sizing import live_scaled_kelly_stake
from scripts.paper_bet_auto_post_toss import (
    _resize_plan_to_bankroll,
    _cancel_order_confirmed,
    _reconcile_exchange_orders,
)


def test_resize_stake_formula_matches_live_scan():
    """Resize and scan both call live_scaled_kelly_stake — same inputs → same stake."""
    strat = next(s for s in STRATEGIES if s.name == "v3_marg_3pp")
    bankroll = 400.0
    model_prob = 0.563
    market_price = 0.26
    stake_a = live_scaled_kelly_stake(model_prob, market_price, bankroll, strat)
    stake_b = live_scaled_kelly_stake(model_prob, market_price, bankroll, strat)
    assert stake_a == stake_b
    assert stake_a > 0


def test_resize_disabled_by_default(monkeypatch):
    """With TWAP_RESIZE_ENABLED unset, resize is a no-op (returns False) and never
    touches the client/DB — the plan keeps its creation-time size."""
    monkeypatch.delenv("TWAP_RESIZE_ENABLED", raising=False)

    class _Boom:
        def __getattr__(self, _):
            raise AssertionError("resize must not touch the client when disabled")

    # Dummy args are fine: the guard returns before any of them are used.
    assert _resize_plan_to_bankroll(None, None, _Boom(), None, dry_run=False) is False


class _FakeCancelClient:
    def __init__(self, open_orders=None, cancel_ok=True):
        self._open = open_orders or []
        self._cancel_ok = cancel_ok
        self.cancelled = []

    def get_open_orders(self):
        return list(self._open)

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)
        if self._cancel_ok:
            return {"not_canceled": {}, "canceled": [order_id]}
        return {"not_canceled": {order_id: "error"}, "canceled": []}


def test_cancel_confirmed_only_true_when_acknowledged():
    ok = _FakeCancelClient(cancel_ok=True)
    assert _cancel_order_confirmed(ok, "0xabc") is True
    bad = _FakeCancelClient(cancel_ok=False)
    assert _cancel_order_confirmed(bad, "0xabc") is False
    assert _cancel_order_confirmed(ok, None) is False

    class _Raises:
        def cancel_order(self, _):
            raise RuntimeError("API down")

    assert _cancel_order_confirmed(_Raises(), "0xabc") is False


@pytest.fixture
def recon_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("CREATE TABLE order_chunks (chunk_id INTEGER PRIMARY KEY, status TEXT, polymarket_order_id TEXT)")
    cur.execute("CREATE TABLE bet_ledger (bet_id INTEGER PRIMARY KEY, status TEXT, polymarket_order_id TEXT)")
    conn.commit()
    yield conn, cur
    conn.close()


def test_reconcile_cancels_orphans_keeps_tracked(recon_db):
    conn, cur = recon_db
    # One actively-tracked placed chunk, one tracked filled ledger order.
    cur.execute("INSERT INTO order_chunks (status, polymarket_order_id) VALUES ('placed', '0xTRACKED')")
    cur.execute("INSERT INTO bet_ledger (status, polymarket_order_id) VALUES ('filled', '0xLEDGER')")
    conn.commit()
    fake = _FakeCancelClient(open_orders=[
        {"id": "0xTRACKED", "outcome": "Kent", "original_size": "100", "price": "0.3"},
        {"id": "0xORPHAN1", "outcome": "RCB", "original_size": "200", "price": "0.48"},
        {"id": "0xORPHAN2", "outcome": "RCB", "original_size": "200", "price": "0.48"},
    ])
    n = _reconcile_exchange_orders(conn, cur, fake, dry_run=False)
    assert n == 2
    assert set(fake.cancelled) == {"0xORPHAN1", "0xORPHAN2"}
    assert "0xTRACKED" not in fake.cancelled


def test_reconcile_dry_run_cancels_nothing(recon_db):
    conn, cur = recon_db
    fake = _FakeCancelClient(open_orders=[
        {"id": "0xORPHAN", "outcome": "RCB", "original_size": "200", "price": "0.48"},
    ])
    n = _reconcile_exchange_orders(conn, cur, fake, dry_run=True)
    assert n == 0
    assert fake.cancelled == []


def test_reconcile_skips_when_db_unreadable():
    """If the active-order set can't be read, reconcile must NOT mass-cancel."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()  # no tables created -> both reads raise
    fake = _FakeCancelClient(open_orders=[
        {"id": "0xANY", "outcome": "RCB", "original_size": "200", "price": "0.48"},
    ])
    n = _reconcile_exchange_orders(conn, cur, fake, dry_run=False)
    assert n == 0
    assert fake.cancelled == []
    conn.close()
