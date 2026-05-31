"""Polymarket CLOB geoblock probe (public API, no auth).

Used by the Flask UI to warn when this machine's egress IP cannot place orders
(e.g. VPN down while physically in AU).
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

GEOBLOCK_URL = "https://polymarket.com/api/geoblock"
_CACHE_TTL_SEC = 30.0

_cache: Optional[Dict[str, Any]] = None
_cache_at: float = 0.0


def check_geoblock(*, force: bool = False) -> Dict[str, Any]:
    """Return geoblock status for the server's current egress IP.

    Keys:
        success: API call succeeded
        blocked: True if CLOB trading is blocked (None if check failed)
        trading_ok: True only when success and not blocked
        ip, country, region: from Polymarket when available
        message: human-readable summary for UI
        check_failed: True when the HTTP probe failed
    """
    global _cache, _cache_at

    now = time.time()
    if not force and _cache is not None and (now - _cache_at) < _CACHE_TTL_SEC:
        return dict(_cache)

    try:
        req = urllib.request.Request(
            GEOBLOCK_URL,
            headers={"User-Agent": "indias-dad-geoblock/1.0", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError) as exc:
        logger.warning("Polymarket geoblock check failed: %s", exc)
        result: Dict[str, Any] = {
            "success": False,
            "blocked": None,
            "trading_ok": False,
            "check_failed": True,
            "error": str(exc),
            "message": f"Could not verify trading region: {exc}",
            "checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        }
        _cache = result
        _cache_at = now
        return dict(result)

    blocked = bool(raw.get("blocked"))
    country = raw.get("country")
    region = raw.get("region")
    ip = raw.get("ip")

    if blocked:
        loc = country or "your region"
        if region:
            loc = f"{loc} ({region})"
        message = (
            f"Trading BLOCKED from {loc} — reconnect VPN or use an allowed region. "
            "No live orders will reach Polymarket until this clears."
        )
    else:
        loc_bits = [b for b in (country, region) if b]
        loc = " / ".join(loc_bits) if loc_bits else "this IP"
        message = f"CLOB trading available ({loc})."

    result = {
        "success": True,
        "blocked": blocked,
        "trading_ok": not blocked,
        "check_failed": False,
        "ip": ip,
        "country": country,
        "region": region,
        "message": message,
        "checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
    }
    _cache = result
    _cache_at = now
    return dict(result)
