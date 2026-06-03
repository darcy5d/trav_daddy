"""Unit tests for the marketable SELL helper
(src/integrations/polymarket/sell_execution.marketable_sell).

This is the core fix for the cashout/rebalance "assumed the midpoint limit
filled" bug: the SELL is now priced to cross the book, the ACTUAL matched size
is read back, and any resting remainder is cancelled. These tests pin that
behaviour with an in-process fake client (no network).
"""

from __future__ import annotations

import pytest

from src.integrations.polymarket import sell_execution
from src.integrations.polymarket.sell_execution import (
    marketable_sell, _vwap_for_shares, _round_down_tick, _parse_book_bids,
)


class FakeClient:
    def __init__(
        self,
        *,
        bids=None,
        tick="0.01",
        fill_fraction=1.0,
        order_id="oid-1",
        place_status=None,
        raise_book=False,
        get_order_resp="auto",
    ):
        self.bids = [{"price": "0.60", "size": "100000"}] if bids is None else bids
        self.tick = tick
        self.fill_fraction = fill_fraction
        self.order_id = order_id
        self.place_status = place_status
        self.raise_book = raise_book
        self.get_order_resp = get_order_resp
        self.placed = []
        self.cancels = []
        self._last_size = 0.0

    def get_clob_order_book(self, token_id):
        if self.raise_book:
            raise RuntimeError("book down")
        return {"bids": self.bids, "asks": [], "tick_size": self.tick}

    def place_limit_order(self, *, token_id, side, price, size_shares):
        self.placed.append((token_id, side, price, size_shares))
        self._last_size = size_shares
        status = self.place_status or ("matched" if self.fill_fraction >= 1.0 else "live")
        resp = {"status": status}
        if self.order_id:
            resp["orderID"] = self.order_id
        return resp

    def get_order(self, order_id):
        if self.get_order_resp == "auto":
            return {
                "size_matched": self._last_size * self.fill_fraction,
                "original_size": self._last_size,
                "status": "matched" if self.fill_fraction >= 1.0 else "live",
            }
        return self.get_order_resp

    def cancel_order(self, order_id):
        self.cancels.append(order_id)
        return {"canceled": [order_id]}


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_round_down_tick():
    assert _round_down_tick(0.4734, 0.01) == pytest.approx(0.47)
    assert _round_down_tick(0.50, 0.01) == pytest.approx(0.50)


def test_parse_book_bids_sorts_desc():
    bids = _parse_book_bids({"bids": [{"price": "0.45", "size": "5"},
                                      {"price": "0.50", "size": "3"}]})
    assert bids == [(0.50, 3.0), (0.45, 5.0)]


def test_vwap_walks_best_first_to_floor():
    bids = [(0.60, 10), (0.58, 100), (0.40, 100)]
    priced, vwap = _vwap_for_shares(bids, 50, floor_price=0.57)
    # 10@0.60 + 40@0.58 = 29.2 / 50 = 0.584; the 0.40 level is below the floor.
    assert priced == pytest.approx(50.0)
    assert vwap == pytest.approx(0.584)


# ---------------------------------------------------------------------------
# marketable_sell
# ---------------------------------------------------------------------------

def test_full_fill_prices_at_floor_and_reads_back():
    fake = FakeClient(bids=[{"price": "0.60", "size": "100000"}])
    res = marketable_sell(fake, "TOK", 100.0, reference_price=0.60,
                          max_slippage_cents=0.03)
    assert res["success"] is True
    assert res["filled_shares"] == pytest.approx(100.0)
    # Floor = 0.60 - 0.03 = 0.57; the SELL is posted there to cross the book.
    assert res["limit_price"] == pytest.approx(0.57)
    assert fake.placed[0][1] == "SELL"
    assert fake.placed[0][2] == pytest.approx(0.57)
    # VWAP fills against the 0.60 bid.
    assert res["avg_fill_price"] == pytest.approx(0.60)
    assert res["proceeds_usdc"] == pytest.approx(60.0)
    assert res["remainder_cancelled"] is False  # nothing left to cancel


def test_partial_fill_cancels_remainder():
    fake = FakeClient(fill_fraction=0.4)
    res = marketable_sell(fake, "TOK", 100.0, reference_price=0.60,
                          max_slippage_cents=0.03)
    assert res["success"] is True
    assert res["filled_shares"] == pytest.approx(40.0)
    assert res["remainder_cancelled"] is True
    assert len(fake.cancels) == 1


def test_partial_fill_keeps_remainder_when_cancel_disabled():
    fake = FakeClient(fill_fraction=0.4)
    res = marketable_sell(fake, "TOK", 100.0, reference_price=0.60,
                          max_slippage_cents=0.03, cancel_remainder=False)
    assert res["success"] is True
    assert res["remainder_cancelled"] is False
    assert fake.cancels == []


def test_no_bids_is_no_fill():
    fake = FakeClient(bids=[])
    res = marketable_sell(fake, "TOK", 100.0, reference_price=0.60)
    assert res["success"] is False
    assert res["reason"] == "no-bids"
    assert fake.placed == []


def test_best_bid_below_floor_is_no_fill():
    fake = FakeClient(bids=[{"price": "0.50", "size": "100000"}])
    res = marketable_sell(fake, "TOK", 100.0, reference_price=0.60,
                          max_slippage_cents=0.03)
    assert res["success"] is False
    assert res["reason"] == "best-bid-below-floor"
    assert fake.placed == []  # never placed below acceptable price


def test_liquidate_sells_through_below_floor_bids():
    # Best bid (0.10) is far below the tight profit floor (ref 0.20 - 0.03);
    # a normal exit would hold ("best-bid-below-floor") and rot to settlement.
    # Liquidate mode prices at the ruin floor and sweeps the bid instead.
    fake = FakeClient(bids=[{"price": "0.10", "size": "100000"}])
    res = marketable_sell(fake, "TOK", 100.0, reference_price=0.20,
                          max_slippage_cents=0.03, liquidate=True,
                          min_exit_price=0.01)
    assert res["success"] is True
    assert res["filled_shares"] == pytest.approx(100.0)
    # SELL posted at the ruin floor so it crosses every resting bid >= 0.01.
    assert res["limit_price"] == pytest.approx(0.01)
    assert fake.placed[0][1] == "SELL"
    assert fake.placed[0][2] == pytest.approx(0.01)
    # VWAP reflects the actual bid it swept (0.10), not the ruin floor.
    assert res["avg_fill_price"] == pytest.approx(0.10)


def test_liquidate_still_no_fill_on_empty_book():
    # An empty bid side cannot be swept — nothing to sell into.
    fake = FakeClient(bids=[])
    res = marketable_sell(fake, "TOK", 100.0, reference_price=0.20,
                          liquidate=True, min_exit_price=0.01)
    assert res["success"] is False
    assert res["reason"] == "no-bids"
    assert fake.placed == []


def test_below_min_shares_is_no_fill():
    fake = FakeClient()
    res = marketable_sell(fake, "TOK", 3.0, reference_price=0.60)
    assert res["success"] is False
    assert res["reason"] == "below-min-shares"


def test_book_fetch_failure_is_no_fill():
    fake = FakeClient(raise_book=True)
    res = marketable_sell(fake, "TOK", 100.0, reference_price=0.60)
    assert res["success"] is False
    assert "book-fetch-failed" in res["reason"]
    assert fake.placed == []


def test_no_order_id_unmatched_is_no_fill():
    # No orderID and status not 'matched' -> treat as unfilled (never book phantom).
    fake = FakeClient(order_id=None, place_status="live")
    res = marketable_sell(fake, "TOK", 100.0, reference_price=0.60)
    assert res["success"] is False
    assert res["reason"] == "no-order-id"


def test_no_order_id_but_matched_books_full():
    fake = FakeClient(order_id=None, place_status="matched")
    res = marketable_sell(fake, "TOK", 100.0, reference_price=0.60,
                          max_slippage_cents=0.03)
    assert res["success"] is True
    assert res["filled_shares"] == pytest.approx(100.0)
    assert res["avg_fill_price"] == pytest.approx(0.60)


def test_unreadable_order_falls_back_to_post_status_live():
    # get_order returns None (exchange has no record we can read) and the post
    # status was 'live' -> conservative: treat as unfilled.
    fake = FakeClient(place_status="live", get_order_resp=None)
    res = marketable_sell(fake, "TOK", 100.0, reference_price=0.60)
    assert res["success"] is False


def test_config_defaults_used_when_not_overridden(monkeypatch):
    cfg = sell_execution.sell_execution_config()
    assert set(cfg) == {"max_slippage_cents", "cancel_remainder", "min_fill_shares"}
    assert cfg["min_fill_shares"] >= 5.0
