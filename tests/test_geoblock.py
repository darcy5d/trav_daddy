"""Tests for Polymarket geoblock probe."""

import json
from unittest.mock import MagicMock, patch

from src.integrations.polymarket import geoblock as gb


def test_check_geoblock_blocked():
    gb._cache = None
    payload = json.dumps({"blocked": True, "ip": "1.2.3.4", "country": "AU", "region": "VIC"}).encode()

    with patch("urllib.request.urlopen") as mock_open:
        mock_resp = MagicMock()
        mock_resp.read.return_value = payload
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_open.return_value = mock_resp

        out = gb.check_geoblock(force=True)

    assert out["success"] is True
    assert out["blocked"] is True
    assert out["trading_ok"] is False
    assert "BLOCKED" in out["message"]
    assert out["country"] == "AU"


def test_check_geoblock_ok():
    gb._cache = None
    payload = json.dumps({"blocked": False, "ip": "9.9.9.9", "country": "NL"}).encode()

    with patch("urllib.request.urlopen") as mock_open:
        mock_resp = MagicMock()
        mock_resp.read.return_value = payload
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_open.return_value = mock_resp

        out = gb.check_geoblock(force=True)

    assert out["trading_ok"] is True
    assert out["blocked"] is False
    assert "available" in out["message"].lower()


def test_check_geoblock_uses_cache():
    gb._cache = {"success": True, "blocked": False, "trading_ok": True, "message": "cached"}
    gb._cache_at = __import__("time").time()

    with patch("urllib.request.urlopen") as mock_open:
        out = gb.check_geoblock(force=False)
        mock_open.assert_not_called()

    assert out["message"] == "cached"
