"""Order-book reading must not depend on the order the CLOB returns levels in.

The live /book payload arrives with bids ascending and asks descending, so
element 0 is the worst price on each side. Reading it put the bid near 0.001
and the ask near 0.999 on every market, which reported a ~96c spread and made
the live scanner's `best_ask <= max_acceptable_price` check unreachable, so
every order went passive TWAP instead of taking a real ask.
"""

import unittest
from unittest import mock

from src.integrations.polymarket.client import PolymarketClient

# Shape as returned by https://clob.polymarket.com/book (verified live).
UNSORTED_BOOK = {
    "bids": [
        {"price": "0.001", "size": "500"},
        {"price": "0.05", "size": "120"},
        {"price": "0.10", "size": "80"},
        {"price": "0.29", "size": "40"},
    ],
    "asks": [
        {"price": "0.999", "size": "900"},
        {"price": "0.60", "size": "50"},
        {"price": "0.31", "size": "30"},
        {"price": "0.30", "size": "25"},
    ],
}


class TestBookLevelOrder(unittest.TestCase):
    def _client(self, book):
        client = PolymarketClient.__new__(PolymarketClient)
        client.get_clob_order_book = mock.Mock(return_value=book)
        return client

    def test_spread_uses_best_prices_not_first_elements(self):
        info = self._client(UNSORTED_BOOK).get_book_spread("tok")
        self.assertAlmostEqual(info["bid"], 0.29)
        self.assertAlmostEqual(info["ask"], 0.30)
        self.assertAlmostEqual(info["spread_pp"], 1.0, places=6)
        self.assertAlmostEqual(info["midpoint"], 0.295, places=6)
        self.assertAlmostEqual(info["best_bid_size"], 40)
        self.assertAlmostEqual(info["best_ask_size"], 25)

    def test_handles_payload_already_sorted_either_way(self):
        ascending = {
            "bids": list(reversed(UNSORTED_BOOK["bids"])),
            "asks": list(reversed(UNSORTED_BOOK["asks"])),
        }
        for book in (UNSORTED_BOOK, ascending):
            info = self._client(book).get_book_spread("tok")
            self.assertAlmostEqual(info["ask"], 0.30)

    def test_vwap_walks_cheapest_asks_first(self):
        book = {
            "bids": [{"price": "0.001", "size": "500"}, {"price": "0.29", "size": "40"}],
            "asks": [
                {"price": "0.999", "size": "900"},
                {"price": "0.31", "size": "1000"},
                {"price": "0.30", "size": "100"},
            ],
        }
        info = self._client(book).get_effective_fill_price("tok", 30.0)
        # 100 shares at 0.30 = $30 exactly, so only the best level is touched.
        self.assertAlmostEqual(info["vwap"], 0.30, places=5)
        self.assertEqual(info["levels_consumed"], 1)

        deeper = self._client(book).get_effective_fill_price("tok", 60.0)
        # $30 at 0.30, then $30 of the 0.31 level.
        expected_vwap = 60.0 / (100 + (30.0 / 0.31))
        self.assertAlmostEqual(deeper["vwap"], expected_vwap, places=5)
        self.assertEqual(deeper["levels_consumed"], 2)
        self.assertGreater(deeper["slippage_pp"], 0)
        self.assertEqual(deeper["asks_below_price"][0]["price"], 0.30)

    def test_asks_below_price_is_cheapest_first_and_filtered(self):
        asks = self._client(UNSORTED_BOOK).get_asks_below_price("tok", 0.35)
        self.assertEqual([a["price"] for a in asks], [0.30, 0.31])

    def test_two_sided_book_yields_real_spread_not_a_hundred_points(self):
        info = self._client(UNSORTED_BOOK).get_book_spread("tok")
        self.assertLess(info["spread_pp"], 5.0)

    def test_malformed_levels_are_skipped(self):
        book = {
            "bids": [{"price": "abc", "size": "1"}, {"price": "0.2", "size": "5"}],
            "asks": [{"size": "5"}, {"price": "0.25", "size": "5"}],
        }
        info = self._client(book).get_book_spread("tok")
        self.assertAlmostEqual(info["bid"], 0.2)
        self.assertAlmostEqual(info["ask"], 0.25)


if __name__ == "__main__":
    unittest.main()
