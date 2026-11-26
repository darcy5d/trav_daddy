"""Manual /api/betting/place must respect the live league allowlist."""

import sys
import types
import unittest
from unittest.mock import patch

if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = types.SimpleNamespace(
        config=types.SimpleNamespace(
            threading=types.SimpleNamespace(
                set_inter_op_parallelism_threads=lambda *a, **k: None,
                set_intra_op_parallelism_threads=lambda *a, **k: None,
            ),
            optimizer=types.SimpleNamespace(set_jit=lambda *a, **k: None),
        )
    )
if "flask_cors" not in sys.modules:
    sys.modules["flask_cors"] = types.SimpleNamespace(CORS=lambda *a, **k: None)

from app.main import app

PLACE = "src.integrations.polymarket.bet_placement.place_bet"


def _body(fixture_key):
    return {
        "fixture_key": fixture_key, "market_type": "moneyline",
        "polymarket_market_id": "m", "polymarket_token_id": "t", "side_label": "X",
        "model_prob": 0.6, "market_price_at_proposal": 0.5, "side": "BUY", "size_usdc": 1,
    }


class BettingPlaceGateTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_unlisted_and_paper_only_prefixes_are_refused(self):
        for key in ("cricadt10-uae-yas-2026-11-07", "cricwbbl-sys-syt-2026-11-07",
                    "cricnewleague-a-b-2026-11-07", ""):
            with patch(PLACE) as place:
                r = self.client.post("/api/betting/place", json=_body(key))
            self.assertEqual(r.status_code, 403, key)
            place.assert_not_called()

    def test_allowlisted_prefix_reaches_place_bet(self):
        with patch(PLACE, return_value={"success": True}) as place:
            r = self.client.post("/api/betting/place", json=_body("crint-ind-nzl-2026-11-09"))
        self.assertEqual(r.status_code, 200)
        place.assert_called_once()
        self.assertEqual(place.call_args.kwargs["fixture_key"], "crint-ind-nzl-2026-11-09")


if __name__ == "__main__":
    unittest.main()
