"""Tests for CLOB maker/taker fill indexing."""

from src.integrations.polymarket.clob_fills import index_fills_by_order_id, maker_entry_usdc


def test_maker_entry_usdc():
    shares, usdc = maker_entry_usdc({"matched_amount": "10", "price": "0.48"})
    assert shares == 10.0
    assert abs(usdc - 4.8) < 1e-6


def test_index_aggregates_maker_legs():
    trades = [
        {
            "taker_order_id": "0xtaker1",
            "trader_side": "TAKER",
            "side": "BUY",
            "size": "5",
            "price": "0.50",
            "asset_id": "token-a",
            "outcome": "Team A",
            "maker_orders": [
                {
                    "order_id": "0xcounterparty",
                    "matched_amount": "9.66",
                    "price": "0.48",
                    "asset_id": "token-b",
                    "outcome": "Team B",
                    "side": "BUY",
                },
            ],
        },
        {
            "taker_order_id": "0xother_taker",
            "trader_side": "MAKER",
            "side": "BUY",
            "size": "11.78",
            "price": "0.48",
            "asset_id": "token-b",
            "outcome": "Team B",
            "maker_orders": [
                {
                    "order_id": "0xmaker1",
                    "matched_amount": "9.66",
                    "price": "0.48",
                    "asset_id": "token-b",
                    "outcome": "Team B",
                    "side": "BUY",
                },
                {
                    "order_id": "0xmaker1",
                    "matched_amount": "2.12",
                    "price": "0.48",
                    "asset_id": "token-b",
                    "outcome": "Team B",
                    "side": "BUY",
                },
            ],
        },
    ]
    fills = index_fills_by_order_id(trades, known_order_ids={"0xmaker1"})
    assert "0xtaker1" in fills
    assert fills["0xtaker1"]["fill_usdc"] == 2.5
    assert "0xcounterparty" not in fills
    assert "0xmaker1" in fills
    assert abs(fills["0xmaker1"]["fill_usdc"] - (11.78 * 0.48)) < 1e-4
    assert fills["0xmaker1"]["asset_id"] == "token-b"
    assert "maker" in fills["0xmaker1"]["roles"]


def test_multi_leg_maker_trade_only_indexes_known_order():
    trades = [
        {
            "taker_order_id": "0xtaker_batch",
            "trader_side": "MAKER",
            "side": "BUY",
            "size": "35",
            "price": "0.38",
            "asset_id": "token-c",
            "outcome": "Team C",
            "maker_orders": [
                {
                    "order_id": "0xours",
                    "matched_amount": "19.1",
                    "price": "0.62",
                    "asset_id": "token-d",
                    "outcome": "Team D",
                },
                {
                    "order_id": "0xtheirs",
                    "matched_amount": "2.6",
                    "price": "0.62",
                    "asset_id": "token-d",
                    "outcome": "Team D",
                },
            ],
        },
    ]
    fills = index_fills_by_order_id(trades, known_order_ids={"0xours"})
    assert "0xours" in fills
    assert abs(fills["0xours"]["fill_usdc"] - 11.842) < 1e-3
    assert "0xtheirs" not in fills


def test_replaced_order_id_remains_known_via_history():
    """When a chunk has been repriced, the old order id is no longer on the
    live order_chunks row, but is in order_history. The index_fills helper
    must still recognise it as ours when known_order_ids includes it."""
    trades = [
        {
            "taker_order_id": "0xbatch",
            "trader_side": "MAKER",
            "side": "BUY",
            "size": "10",
            "price": "0.50",
            "asset_id": "token-e",
            "outcome": "Team E",
            "maker_orders": [
                {
                    "order_id": "0xold_repriced",
                    "matched_amount": "8",
                    "price": "0.50",
                    "asset_id": "token-e",
                    "outcome": "Team E",
                },
                {
                    "order_id": "0xcounterparty",
                    "matched_amount": "2",
                    "price": "0.50",
                    "asset_id": "token-e",
                    "outcome": "Team E",
                },
            ],
        },
    ]
    # Caller passes order_history-derived ids as the master known set; the
    # repriced (no longer live on order_chunks) id is still in there.
    fills = index_fills_by_order_id(trades, known_order_ids={"0xold_repriced"})
    assert "0xold_repriced" in fills
    assert "0xcounterparty" not in fills
    assert abs(fills["0xold_repriced"]["fill_usdc"] - 4.0) < 1e-6
