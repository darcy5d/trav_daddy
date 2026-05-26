"""Cached Gamma lookups for live-betting bet display enrichment.

Settled-bet metadata (team names, match winner) is stable once a market
closes, so we cache Gamma responses in-process for several hours and
prefetch unique market_ids in parallel per API request.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

GAMMA_MARKETS_BASE = "https://gamma-api.polymarket.com/markets"
_GAMMA_FETCH_TIMEOUT_SEC = 4.0

_GAMMA_CACHE: Dict[str, Tuple[float, Optional[Dict[str, Any]]]] = {}
_FIXTURE_META_CACHE: Dict[str, Tuple[float, Dict[str, str]]] = {}
_CACHE_TTL_SEC = 6 * 3600  # settled display data is immutable
_lock = threading.Lock()


def _parse_outcomes(raw: Any) -> List[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except json.JSONDecodeError:
            pass
    return []


def teams_from_gamma_market(market: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    """Extract two team names from a Gamma market dict."""
    if not market:
        return None, None

    from src.integrations.polymarket.upcoming import _parse_team_names_from_title

    for title_key in ("eventTitle", "title", "question"):
        title = market.get(title_key) or ""
        t1, t2 = _parse_team_names_from_title(str(title))
        if t1 and t2:
            return t1, t2

    outcomes = _parse_outcomes(market.get("outcomes"))
    if len(outcomes) >= 2:
        o0, o1 = outcomes[0], outcomes[1]
        if o0.lower() not in ("yes", "no") and o1.lower() not in ("yes", "no"):
            return o0, o1
    return None, None


def _gamma_cache_get(market_id: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Return (cache_hit, market). cache_hit=False if missing, expired, or failed fetch."""
    now = time.time()
    with _lock:
        entry = _GAMMA_CACHE.get(market_id)
        # Do not treat cached fetch failures as hits — they poison cashout counterfactuals.
        if entry and entry[1] is not None and (now - entry[0]) < _CACHE_TTL_SEC:
            return True, entry[1]
    return False, None


def _gamma_cache_set(market_id: str, market: Optional[Dict[str, Any]]) -> None:
    if not market:
        return
    with _lock:
        _GAMMA_CACHE[market_id] = (time.time(), market)


def _fetch_gamma_market_display(market_id: str) -> Optional[Dict[str, Any]]:
    """Fetch Gamma market for UI display (short timeout)."""
    if not market_id:
        return None
    try:
        import requests
        r = requests.get(
            f"{GAMMA_MARKETS_BASE}/{market_id}",
            timeout=_GAMMA_FETCH_TIMEOUT_SEC,
        )
        if not r.ok:
            return None
        m = r.json()
        return m if isinstance(m, dict) else None
    except Exception as exc:
        logger.debug(f"Gamma display fetch failed for {market_id}: {exc}")
        return None


def get_gamma_market_cached(market_id: str) -> Optional[Dict[str, Any]]:
    """Return Gamma market dict, using process cache when fresh."""
    if not market_id:
        return None
    hit, cached = _gamma_cache_get(market_id)
    if hit:
        return cached
    market = _fetch_gamma_market_display(market_id)
    _gamma_cache_set(market_id, market)
    return market


def prefetch_gamma_markets(market_ids: List[str], max_workers: int = 8) -> None:
    """Warm cache for many market_ids in parallel (no-op if already cached)."""
    unique = []
    seen = set()
    for mid in market_ids:
        mid = str(mid or "").strip()
        if not mid or mid in seen:
            continue
        seen.add(mid)
        hit, _ = _gamma_cache_get(mid)
        if not hit:
            unique.append(mid)

    if not unique:
        return

    def _fetch_one(mid: str) -> None:
        get_gamma_market_cached(mid)

    workers = min(max_workers, max(1, len(unique)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_fetch_one, mid) for mid in unique]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:
                logger.debug(f"Gamma prefetch worker failed: {exc}")


def fixture_meta_cache_get(fixture_key: str) -> Optional[Dict[str, str]]:
    now = time.time()
    with _lock:
        entry = _FIXTURE_META_CACHE.get(fixture_key)
        if entry and (now - entry[0]) < _CACHE_TTL_SEC:
            return entry[1]
    return None


def fixture_meta_cache_set(fixture_key: str, meta: Dict[str, str]) -> None:
    with _lock:
        _FIXTURE_META_CACHE[fixture_key] = (time.time(), meta)
