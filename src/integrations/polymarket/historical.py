"""Wave 5 Phase 5: Polymarket historical price helpers.

For a completed cricket match in our DB:

1. `lookup_settled_markets_for_match()` — find Polymarket markets that
   settled for it. Uses Gamma API with `closed=true` plus the existing
   fuzzy fixture matcher in `polymarket_compare.py`.

2. `fetch_pre_match_prices()` — pull `/prices-history` at multiple
   pre-match timestamps (T-1h / T-30min / T-15min before scheduled match
   start) so we know how prices moved as the match approached.

3. `compute_realised_settle_price()` — read the very-last price in the
   history series; for a settled market this is 0.0 or 1.0.

These helpers feed `scripts/backtest_polymarket_ev.py`, which simulates
"if we'd bet at edge >= N% threshold, what's our P&L?" over a holdout
of completed matches with Polymarket coverage.

Caveats:

- Polymarket cricket coverage really only ramped up mid-2024. Older
  matches will return empty results.
- Some Polymarket events have multiple closed markets per fixture
  (e.g. the moneyline + side markets); we return the full list rather
  than picking one.
- `/prices-history` returns sample-fidelity pricing; for thin markets
  with few trades the price may be stale or even constant for hours.
- Polymarket's labels can be slightly inconsistent across leagues
  (e.g. team abbreviations vs full names). The `polymarket_compare`
  fuzzy matcher already handles this.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.integrations.polymarket import PolymarketClient
from src.integrations.odds.polymarket_compare import (
    PolymarketComparisonService,
    select_best_market,
    _coerce_list,
    _coerce_float,
    normalize_team_name,
)

logger = logging.getLogger(__name__)


def _to_unix_ts(dt: datetime) -> int:
    """Return Unix timestamp in seconds (UTC)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


_CRICKET_EVENTS_CACHE: List[Dict[str, Any]] = []
_CRICKET_EVENTS_CACHE_AT: Optional[datetime] = None


def _load_all_cricket_events(
    client: PolymarketClient,
    closed: bool = True,
    max_offset: int = 5000,
    page_size: int = 100,
) -> List[Dict[str, Any]]:
    """Fetch all cricket-tagged events from Gamma with pagination.

    Cached in-process for 1 hour so repeated lookups inside one EV
    backtest don't re-hit the API. Polymarket's `/events?tag_slug=cricket`
    returns events with `closed=true` filter; each event carries a
    `markets` list with per-market token ids.
    """
    global _CRICKET_EVENTS_CACHE, _CRICKET_EVENTS_CACHE_AT
    now = datetime.now(timezone.utc)
    if (
        _CRICKET_EVENTS_CACHE
        and _CRICKET_EVENTS_CACHE_AT is not None
        and (now - _CRICKET_EVENTS_CACHE_AT).total_seconds() < 3600
    ):
        return _CRICKET_EVENTS_CACHE
    out: List[Dict[str, Any]] = []
    for offset in range(0, max_offset, page_size):
        try:
            rows = client.get_events(
                limit=page_size,
                offset=offset,
                tag_slug="cricket",
                closed=closed,
            )
        except Exception as exc:
            logger.warning(f"Polymarket Gamma /events page {offset} failed: {exc}")
            break
        if not isinstance(rows, list) or not rows:
            break
        out.extend(rows)
        if len(rows) < page_size:
            break
    _CRICKET_EVENTS_CACHE = out
    _CRICKET_EVENTS_CACHE_AT = now
    logger.info(f"Loaded {len(out)} cricket events from Polymarket Gamma")
    return out


def _event_to_pseudo_markets(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Polymarket Gamma /events returns events with embedded markets list.
    Flatten into the same shape as /markets so existing comparator code
    keeps working: each market gets the parent event's metadata stamped on.
    """
    parent_meta = {
        "eventId": event.get("id"),
        "eventSlug": event.get("slug"),
        "eventTitle": event.get("title"),
        "startDate": event.get("startDate") or event.get("startTime"),
        "endDate": event.get("endDate") or event.get("closedTime"),
    }
    out = []
    for market in event.get("markets", []) or []:
        if not isinstance(market, dict):
            continue
        merged = dict(market)
        merged.update({k: v for k, v in parent_meta.items() if v is not None})
        # Inject event title into the market's question for fuzzy match
        if not merged.get("question") and parent_meta.get("eventTitle"):
            merged["question"] = parent_meta["eventTitle"]
        out.append(merged)
    return out


def lookup_settled_markets_for_match(
    client: PolymarketClient,
    team1: str,
    team2: str,
    match_date_iso: str,
    series: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return settled Polymarket markets matching the fixture, if any.

    Wave 5 Phase 5 implementation: pages through the Gamma `/events`
    endpoint with `tag_slug=cricket&closed=true`, runs our existing fuzzy
    fixture matcher across the flattened market list, and returns all
    sibling markets within the matched event.
    """
    events = _load_all_cricket_events(client, closed=True)
    if not events:
        return []
    # Flatten events->markets but keep event linkage
    all_markets: List[Dict[str, Any]] = []
    event_to_markets: Dict[str, List[Dict[str, Any]]] = {}
    for event in events:
        markets = _event_to_pseudo_markets(event)
        if not markets:
            continue
        event_id = str(event.get("id") or event.get("slug") or "")
        event_to_markets[event_id] = markets
        all_markets.extend(markets)
    if not all_markets:
        return []

    anchor, conf = select_best_market(
        markets=all_markets,
        team1=team1,
        team2=team2,
        start_utc=match_date_iso,
        series_name=series,
    )
    if anchor is None:
        return []

    event_id = str(anchor.get("eventId") or "")
    if event_id and event_id in event_to_markets:
        return event_to_markets[event_id]
    return [anchor]


def _label_for_hours(hours_before: float) -> str:
    """Format a lookback into a human-friendly label (T-30min, T-6h, T-3d)."""
    if hours_before < 1.0:
        return f"T-{int(hours_before * 60)}min"
    if hours_before < 24.0:
        return f"T-{int(hours_before)}h"
    days = hours_before / 24.0
    return f"T-{days:g}d"


def fetch_pre_match_prices(
    client: PolymarketClient,
    token_id: str,
    match_start_utc: datetime,
    hours_before_list: List[float] = (0.5, 1.0, 6.0, 12.0, 24.0, 48.0, 72.0),
    fidelity: int = 60,
    prefetch_days_before: float = 7.0,
) -> Dict[str, Optional[float]]:
    """Pull a price history window before match start; return the last
    observed price at each lookback.

    The prefetch window defaults to 7 days before match start (was 4
    hours pre-Wave-5.1, which silently returned nothing for any lookback
    > 4h). Polymarket cricket markets typically open 1-5 days before a
    fixture; a longer prefetch covers the full listing period.

    Returns:
        {"T-30min": 0.42, "T-1h": 0.41, "T-6h": 0.40, "T-24h": 0.55, ..., "settle": 1.0}
        Lookbacks with no sample available come back as None.
    """
    out: Dict[str, Optional[float]] = {_label_for_hours(h): None for h in hours_before_list}
    out["settle"] = None
    if not token_id:
        return out
    if match_start_utc.tzinfo is None:
        match_start_utc = match_start_utc.replace(tzinfo=timezone.utc)

    start_ts = _to_unix_ts(match_start_utc) - int(prefetch_days_before * 86400)
    end_ts = _to_unix_ts(match_start_utc) + 24 * 3600

    try:
        resp = client.get_prices_history(
            token_id=token_id,
            start_ts=start_ts,
            end_ts=end_ts,
            interval="1h",
            fidelity=fidelity,
        )
    except Exception as exc:
        logger.debug(f"prices-history fetch failed for {token_id}: {exc}")
        return out

    history = resp.get("history") if isinstance(resp, dict) else resp
    if not isinstance(history, list) or not history:
        return out

    samples = [
        (int(s.get("t", 0)), float(s.get("p", 0.0)))
        for s in history
        if isinstance(s, dict) and s.get("t") is not None and s.get("p") is not None
    ]
    samples.sort()
    if not samples:
        return out

    match_ts = _to_unix_ts(match_start_utc)
    for hours_before in hours_before_list:
        target_ts = match_ts - int(hours_before * 3600)
        # Find the most recent sample at or before target_ts. If the
        # market hadn't opened yet at target_ts, return None for that
        # lookback (rather than the post-open price, which would be
        # forward-looking from the model's perspective).
        if target_ts < samples[0][0]:
            out[_label_for_hours(hours_before)] = None
            continue
        best_p: Optional[float] = None
        for t, p in samples:
            if t > target_ts:
                break
            best_p = p
        out[_label_for_hours(hours_before)] = best_p

    out["settle"] = samples[-1][1]
    return out


def compute_realised_settle_price(history: List[Dict[str, Any]]) -> Optional[float]:
    """Return the last price in a history series; None if empty."""
    if not history:
        return None
    last = history[-1]
    if isinstance(last, dict) and "p" in last:
        return float(last["p"])
    return None


def market_outcome_label_to_token_id(
    market: Dict[str, Any], desired_label_normalized: str
) -> Optional[str]:
    """Find the clob_token_id corresponding to a desired outcome label."""
    labels = _coerce_list(market.get("outcomes"))
    token_ids = _coerce_list(market.get("clobTokenIds") or market.get("clobTokenIDs"))
    if not labels or not token_ids:
        return None
    norm_desired = normalize_team_name(desired_label_normalized)
    for idx, label in enumerate(labels):
        if normalize_team_name(str(label)) == norm_desired or norm_desired in normalize_team_name(str(label)):
            if idx < len(token_ids):
                return str(token_ids[idx])
    return None
