"""Join Polymarket T-1h moneylines onto match-level rows.

Reads Gamma and the CLOB prices-history endpoint. Does not open bet_ledger
and does not place an order. A missing T-1h price drops the fixture. No other
lookback is substituted. `crict20wbbl` is not a prefix. `crictnpl` is not NPL.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

_WOMEN_RE = re.compile(r"\bwom[ae]n(?:'s)?\b|\bwomens\b", re.I)
_WOMENS_PREFIXES = {"cricwbbl", "crwpl20", "cricodcw", "cricodcl2w", "crict20blastw", "crichundredw"}

import pandas as pd

from src.integrations.odds.polymarket_compare import (
    PolymarketComparisonService,
    _coerce_list,
    select_best_market,
)
from src.integrations.polymarket.historical import fetch_pre_match_prices
from src.integrations.polymarket.upcoming import (
    _classify_market_within_event,
    _classify_slug,
    _parse_iso,
)
from src.models.match_level.model import DECISION_HOURS_BEFORE, _UNUSED_PREFIX

_BBL_PREFIXES = {"cricbbl", "cricwbbl"}


def team1_outcome_token(outcomes: List[dict], team1: str, team2: str) -> Optional[str]:
    """Token for team1's moneyline outcome, or None when the mapping is not unique."""
    hits = []
    for outcome in outcomes or []:
        label = str(outcome.get("label") or "")
        token = outcome.get("token_id")
        match_1 = PolymarketComparisonService.label_matches_team(label, team1)
        match_2 = PolymarketComparisonService.label_matches_team(label, team2)
        if match_1 and not match_2 and token:
            hits.append(str(token))
    if len(hits) != 1:
        return None
    return hits[0]


def _outcomes(market: dict) -> List[dict]:
    labels = _coerce_list(market.get("outcomes"))
    tokens = _coerce_list(market.get("clobTokenIds") or market.get("clobTokenIDs"))
    out = []
    for idx, label in enumerate(labels):
        out.append({
            "label": str(label),
            "token_id": str(tokens[idx]) if idx < len(tokens) and tokens[idx] else None,
        })
    return out


def _kickoff(prefix: str, slug_date: str, market: dict) -> datetime:
    parsed = _parse_iso(market.get("gameStartTime") or market.get("game_start_time"))
    if parsed is not None:
        return parsed
    hour = "09:00:00" if prefix in _BBL_PREFIXES else "14:00:00"
    return datetime.fromisoformat(f"{slug_date}T{hour}+00:00")


# Gamma rejects offset past 2000, so a single closed=true crawl stops short of the
# holdout. Walk listing-date windows instead, and split a window that fills the cap.
_EVENT_TAGS = ("cricket", "t20-blast")
_OFFSET_CAP = 2000
_PAGE = 100


def _page_events(client, tag: str, start: datetime, end: datetime, closed: bool) -> Optional[List[dict]]:
    rows: List[dict] = []
    offset = 0
    while offset <= _OFFSET_CAP:
        params = {
            "limit": _PAGE,
            "offset": offset,
            "tag_slug": tag,
            "closed": str(bool(closed)).lower(),
            "start_date_min": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "start_date_max": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        try:
            page = client._get_json(f"{client.api_base_url}/events", params=params)
        except Exception as exc:
            print(f"gamma {tag} {start.date()} {end.date()} offset {offset} closed={closed}: {exc}", flush=True)
            return rows
        if not isinstance(page, list) or not page:
            return rows
        rows.extend(page)
        if len(page) < _PAGE:
            return rows
        offset += _PAGE
    return None


def _window_events(client, tag: str, start: datetime, end: datetime, closed: bool) -> List[dict]:
    got = _page_events(client, tag, start, end, closed)
    if got is not None:
        return got
    if (end - start) <= timedelta(days=1):
        print(f"gamma window still full at 1 day: {tag} {start.date()} closed={closed}", flush=True)
        return []
    mid = start + (end - start) / 2
    return _window_events(client, tag, start, mid, closed) + _window_events(client, tag, mid, end, closed)


def fetch_cricket_events(client, start: datetime, end: datetime) -> List[dict]:
    """Closed cricket events whose Gamma listing date falls in [start, end).

    Also pulls still-open events from the last 60 days so a just-finished
    match is not dropped. Deduped by slug. Read-only.
    """
    seen = set()
    out = []
    recent = max(start, end - timedelta(days=60))
    for tag in _EVENT_TAGS:
        cursor = start
        while cursor < end:
            nxt = min(end, cursor + timedelta(days=14))
            batch = _window_events(client, tag, cursor, nxt, closed=True)
            if nxt > recent:
                batch = batch + _window_events(client, tag, max(cursor, recent), nxt, closed=False)
            for event in batch:
                slug = event.get("slug") or str(event.get("id") or "")
                if not slug or slug in seen:
                    continue
                seen.add(slug)
                out.append(event)
            print(f"  events {tag} through {nxt.date()}: {len(out):,}", flush=True)
            cursor = nxt
    return out


def moneyline_markets(events: List[dict]) -> List[dict]:
    """One moneyline market per bare-slug event, with the scanner prefix attached."""
    found = []
    for event in events or []:
        if not isinstance(event, dict):
            continue
        slug = event.get("slug") or ""
        info = _classify_slug(slug)
        if not info or info.get("kind") != "moneyline" or "suffix" in info:
            continue
        prefix = info.get("prefix")
        if not prefix or prefix == _UNUSED_PREFIX:
            continue
        chosen = None
        for market in event.get("markets") or []:
            if not isinstance(market, dict):
                continue
            outcomes = _coerce_list(market.get("outcomes"))
            kind = _classify_market_within_event(
                question=market.get("question") or "",
                outcomes=outcomes,
                event_slug_kind="moneyline",
                sports_market_type=market.get("sportsMarketType"),
            )
            if kind != "moneyline":
                continue
            chosen = market
            break
        if chosen is None:
            continue
        slug_date = info["date"]
        start = _kickoff(prefix, slug_date, chosen)
        slim = {
            "question": chosen.get("question") or event.get("title") or "",
            "slug": slug,
            "eventTitle": event.get("title") or "",
            "eventSlug": slug,
            "gameStartTime": start.isoformat(),
            "prefix": prefix,
            "slug_date": slug_date,
            "outcomes": _outcomes(chosen),
        }
        found.append(slim)
    return found


def _market_is_womens(market: dict) -> bool:
    text = f"{market.get('eventTitle') or ''} {market.get('question') or ''}"
    if _WOMEN_RE.search(text):
        return True
    return market.get("prefix") in _WOMENS_PREFIXES


def _nearby_dates(day: str) -> List[str]:
    base = datetime.strptime(day, "%Y-%m-%d")
    return [(base + timedelta(days=k)).strftime("%Y-%m-%d") for k in (-1, 0, 1)]


def join_moneylines(
    matches: pd.DataFrame,
    markets: List[dict],
    team_names: Dict[int, str],
) -> List[dict]:
    """Greedy one-to-one join. Each item has match_id, prefix, token_id, start."""
    by_date: Dict[str, List[dict]] = {}
    for market in markets:
        by_date.setdefault(market["slug_date"], []).append(market)

    proposals = []
    for row in matches.itertuples(index=False):
        day = pd.Timestamp(row.date).strftime("%Y-%m-%d")
        pool: List[dict] = []
        for key in _nearby_dates(day):
            pool.extend(by_date.get(key, ()))
        gender = getattr(row, "gender", None)
        if gender == "male":
            pool = [m for m in pool if not _market_is_womens(m)]
        elif gender == "female":
            pool = [m for m in pool if _market_is_womens(m)]
        if not pool:
            continue
        name1 = team_names.get(int(row.team1_id))
        name2 = team_names.get(int(row.team2_id))
        if not name1 or not name2:
            continue
        start = f"{day}T12:00:00+00:00"
        series = getattr(row, "event_name", None)
        market, score = select_best_market(
            pool, name1, name2, start_utc=start,
            series_name=None if series is None or (isinstance(series, float) and pd.isna(series)) else str(series),
        )
        if market is None:
            continue
        proposals.append((float(score), int(row.match_id), name1, name2, market))

    proposals.sort(key=lambda item: -item[0])
    used_matches = set()
    used_events = set()
    joins = []
    for score, match_id, name1, name2, market in proposals:
        event_slug = market["eventSlug"]
        if match_id in used_matches or event_slug in used_events:
            continue
        used_matches.add(match_id)
        used_events.add(event_slug)
        joins.append({
            "match_id": match_id,
            "prefix": market["prefix"],
            "token_id": team1_outcome_token(market["outcomes"], name1, name2),
            "start": market["gameStartTime"],
            "score": round(score, 6),
        })
    return joins


def load_price_cache(path) -> Dict[str, Optional[float]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    raw = payload.get("by_match_id") or {}
    out = {}
    for key, value in raw.items():
        price = value.get("price") if isinstance(value, dict) else None
        out[str(key)] = None if price is None else float(price)
    return out


def save_price_cache(path, prices: Dict[str, Optional[float]]) -> None:
    payload = {
        "hours_before": DECISION_HOURS_BEFORE,
        "label": "T-1h",
        "by_match_id": {
            str(match_id): {"price": price}
            for match_id, price in prices.items()
        },
    }
    path.write_text(json.dumps(payload))


def fetch_t1h_prices(client, joins: List[dict], cache_path, pause_s: float = 0.05) -> Dict[int, Optional[float]]:
    """Fetch the T-1h price only. Cache under cache_path. Missing stays missing."""
    cached = load_price_cache(cache_path)
    prices: Dict[int, Optional[float]] = {}
    pending = []
    for join in joins:
        match_id = int(join["match_id"])
        key = str(match_id)
        if not join.get("token_id"):
            prices[match_id] = None
            continue
        if key in cached:
            prices[match_id] = cached[key]
            continue
        pending.append(join)

    for i, join in enumerate(pending, start=1):
        match_id = int(join["match_id"])
        start = _parse_iso(join["start"])
        price = None
        if start is not None:
            got = fetch_pre_match_prices(
                client,
                str(join["token_id"]),
                start,
                hours_before_list=(DECISION_HOURS_BEFORE,),
            )
            raw = got.get("T-1h")
            if raw is not None:
                price = float(raw)
                if not (0.0 <= price <= 1.0):
                    price = None
        prices[match_id] = price
        cached[str(match_id)] = price
        if i % 25 == 0:
            save_price_cache(cache_path, cached)
            print(f"  T-1h prices {i}/{len(pending)}", flush=True)
        if pause_s:
            time.sleep(pause_s)
    save_price_cache(cache_path, cached)
    return prices
