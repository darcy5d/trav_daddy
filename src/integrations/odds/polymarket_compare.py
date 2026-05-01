"""Polymarket fixture linking and comparison helpers.

Wave 5 extension: alongside the existing moneyline compare service,
this module now classifies each linked Polymarket market into one of
4 modellable cricket market types (moneyline, top_batter, most_sixes,
toss_match_double) and exposes a `compare_fixture_multi` method that
returns probabilities and edges per market.

Polymarket lists ~6 cricket market types per fixture event, but only
4 carry meaningful cricket-skill signal we can edge:

- Moneyline (2-way): which team wins
- Team Top Batter (3-way: team1 / draw / team2)
- Most Sixes (3-way: team1 / draw / team2)
- Toss Match Double (4-way: toss x match cross product)

Toss Winner is genuinely 50/50; Completed Match is a weather event.
Both are dropped on the model side but recognised so they don't pollute
the matching logic.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from src.integrations.polymarket import PolymarketClient

# Polymarket taker fee. Updated 2026-05-01 (Wave 5.7 follow-up):
# Polymarket migrated to their own token / sports_fees_v2 schedule in
# late April 2026. New rate is 3% taker-only with a 25% maker rebate.
# (Confirmed via Gamma /markets response: feeType='sports_fees_v2',
# feeSchedule={'rate': 0.03, 'takerOnly': True, 'rebateRate': 0.25}.)
# Historical bets settled before this date were charged 2%; those
# entries in bet_ledger keep their original computed fees and pnl.
POLYMARKET_TAKER_FEE = 0.03

# Wave 5: cricket market-type classifier. Patterns checked in order.
# Each pattern matches against the lowercased market title + question +
# slug. The 4 modellable types map to constants used by market_outputs.
MARKET_TYPE_MONEYLINE = "moneyline"
MARKET_TYPE_TOP_BATTER = "top_batter"
MARKET_TYPE_MOST_SIXES = "most_sixes"
MARKET_TYPE_TOSS_MATCH_DOUBLE = "toss_match_double"
MARKET_TYPE_TOSS_WINNER = "toss_winner"
MARKET_TYPE_COMPLETED = "completed_match"

# Order matters: more specific patterns FIRST. e.g. "toss match double"
# would also match the moneyline pattern (because it contains "match"),
# so it must be tested first.
MARKET_TYPE_PATTERNS: List[Tuple[str, re.Pattern]] = [
    (MARKET_TYPE_TOSS_MATCH_DOUBLE, re.compile(r"toss[\s\-]+match[\s\-]+double", re.I)),
    (MARKET_TYPE_TOP_BATTER, re.compile(r"top[\s\-]+(batter|batsman|run[\s\-]*scorer)", re.I)),
    (MARKET_TYPE_MOST_SIXES, re.compile(r"most[\s\-]+sixes", re.I)),
    (MARKET_TYPE_TOSS_WINNER, re.compile(r"toss[\s\-]+winner|wins?[\s\-]+the[\s\-]+toss", re.I)),
    (MARKET_TYPE_COMPLETED, re.compile(r"complete[d]?[\s\-]+match|match[\s\-]+complete[d]?", re.I)),
    # Moneyline last since "will X beat Y" is the broadest cricket pattern
    (MARKET_TYPE_MONEYLINE, re.compile(r"\b(beat|defeat|win\s+vs|moneyline|to[\s\-]+win)\b", re.I)),
]

GENERIC_TEAM_WORDS = {
    "men",
    "women",
    "w",
    "a",
    "xi",
    "u19",
    "u23",
    "knight",
    "knights",
    "rider",
    "riders",
    "super",
    "giants",
    "kings",
    "warriors",
    "united",
    "city",
    "club",
}

SERIES_PREFIX_ALIASES = {
    "ipl": "cricipl",
    "indian premier league": "cricipl",
    "psl": "cricpsl",
    "pakistan super league": "cricpsl",
    "international": "cricint",
    "t20 international": "cricint",
}


def normalize_team_name(value: str) -> str:
    """Normalize names for fuzzy matching."""
    text = (value or "").strip().lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(value: str) -> List[str]:
    text = normalize_team_name(value)
    return [tok for tok in text.split(" ") if tok]


def _coerce_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        if "," in stripped:
            return [x.strip() for x in stripped.split(",") if x.strip()]
        return [stripped]
    return []


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def build_fixture_key(team1: str, team2: str, start_utc: Optional[str]) -> str:
    t1 = normalize_team_name(team1).replace(" ", "_") or "team1"
    t2 = normalize_team_name(team2).replace(" ", "_") or "team2"
    s = (start_utc or "unknown_start").replace(" ", "_")
    return f"{t1}_{t2}_{s}"


def infer_selected_side(team1: str, team2: str, model_team1_win_pct: Optional[float], model_team2_win_pct: Optional[float]) -> str:
    """
    Pick the side shown for edge comparison.

    If model probabilities are available, choose the model-favored side.
    Otherwise default to team 1.
    """
    t1 = team1 or "Team 1"
    t2 = team2 or "Team 2"
    if model_team1_win_pct is None or model_team2_win_pct is None:
        return t1
    return t1 if model_team1_win_pct >= model_team2_win_pct else t2


def compute_edge_pct_points(model_win_pct: Optional[float], market_implied_pct: Optional[float]) -> Optional[float]:
    """Compute model minus market in percentage points."""
    if model_win_pct is None or market_implied_pct is None:
        return None
    return round(model_win_pct - market_implied_pct, 2)


def _extract_market_text(market: Dict[str, Any]) -> str:
    parts = [
        str(market.get("question") or ""),
        str(market.get("slug") or ""),
        str(market.get("title") or ""),
        str(market.get("description") or ""),
        str(market.get("eventTitle") or ""),
    ]
    return normalize_team_name(" ".join([p for p in parts if p]))


def _token_overlap_score(team_name: str, text: str) -> float:
    team_tokens = set(_tokenize(team_name))
    text_tokens = set(_tokenize(text))
    if not team_tokens or not text_tokens:
        return 0.0
    overlap = len(team_tokens.intersection(text_tokens))
    return overlap / max(1, len(team_tokens))


def _fuzzy_score(team_name: str, text: str) -> float:
    if not team_name or not text:
        return 0.0
    return SequenceMatcher(a=normalize_team_name(team_name), b=normalize_team_name(text)).ratio()


def _extract_market_start(market: Dict[str, Any]) -> Optional[datetime]:
    candidates = [
        market.get("startDate"),
        market.get("gameStartTime"),
        market.get("startTime"),
        market.get("endDate"),
    ]
    for value in candidates:
        parsed = _parse_iso_datetime(value)
        if parsed is not None:
            return parsed
    return None


def _team_code(team_name: str) -> str:
    tokens = _tokenize(team_name)
    if not tokens:
        return ""
    meaningful = [tok for tok in tokens if tok not in GENERIC_TEAM_WORDS]
    source = meaningful[0] if meaningful else tokens[0]
    return source[:3]


def _series_slug_prefix(series_name: Optional[str]) -> Optional[str]:
    if not series_name:
        return None
    s = normalize_team_name(series_name)
    for key, value in SERIES_PREFIX_ALIASES.items():
        if key in s:
            return value
    return None


def _date_part(start_utc: Optional[str]) -> Optional[str]:
    parsed = _parse_iso_datetime(start_utc)
    if parsed is None:
        return None
    return parsed.strftime("%Y-%m-%d")


def _date_score(fixture_start: Optional[datetime], market_start: Optional[datetime]) -> float:
    if fixture_start is None or market_start is None:
        return 0.0
    delta_hours = abs((market_start - fixture_start).total_seconds()) / 3600.0
    if delta_hours <= 3:
        return 1.0
    if delta_hours <= 12:
        return 0.75
    if delta_hours <= 24:
        return 0.5
    if delta_hours <= 48:
        return 0.2
    return 0.0


def _series_score(series_name: Optional[str], market_text: str) -> float:
    if not series_name:
        return 0.0
    score = _token_overlap_score(series_name, market_text)
    normalized_series = normalize_team_name(series_name)
    # Bridge short league names (e.g., "IPL 2026") to slug prefixes ("cricipl").
    for key, slug_prefix in SERIES_PREFIX_ALIASES.items():
        if key in normalized_series and (slug_prefix in market_text or key in market_text):
            score = max(score, 1.0)
    return score


def select_best_market(
    markets: List[Dict[str, Any]],
    team1: str,
    team2: str,
    start_utc: Optional[str] = None,
    series_name: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], float]:
    """Pick best market candidate for fixture."""
    fixture_start = _parse_iso_datetime(start_utc)
    best_market: Optional[Dict[str, Any]] = None
    best_score = 0.0

    for market in markets:
        text = _extract_market_text(market)
        if not text:
            continue

        t1_overlap = _token_overlap_score(team1, text)
        t2_overlap = _token_overlap_score(team2, text)
        if t1_overlap < 0.25 or t2_overlap < 0.25:
            continue

        t1_fuzzy = _fuzzy_score(team1, text)
        t2_fuzzy = _fuzzy_score(team2, text)
        team_score = (0.5 * t1_overlap + 0.5 * t1_fuzzy + 0.5 * t2_overlap + 0.5 * t2_fuzzy) / 2.0

        m_start = _extract_market_start(market)
        d_score = _date_score(fixture_start, m_start)
        s_score = _series_score(series_name, text)

        total = 0.7 * team_score + 0.2 * d_score + 0.1 * s_score
        if total > best_score:
            best_market = market
            best_score = total

    # Avoid low-confidence accidental links.
    # 0.40 keeps safety while avoiding floating-point edge misses around 0.45.
    if best_score < 0.40:
        return None, best_score
    return best_market, best_score


def _infer_yes_side(question: str, team1: str, team2: str) -> Optional[str]:
    q = normalize_team_name(question)
    t1 = normalize_team_name(team1)
    t2 = normalize_team_name(team2)
    if not q or not t1 or not t2:
        return None

    i1 = q.find(t1)
    i2 = q.find(t2)
    if i1 == -1 and i2 == -1:
        return None
    if i1 != -1 and i2 == -1:
        return team1
    if i2 != -1 and i1 == -1:
        return team2

    # In "will X beat Y" style questions, earlier team is YES side.
    return team1 if i1 <= i2 else team2


def _choose_token_for_side(market: Dict[str, Any], selected_side_label: str, team1: str, team2: str) -> Optional[str]:
    token_ids = _coerce_list(market.get("clobTokenIds") or market.get("clobTokenIDs"))
    if len(token_ids) < 2:
        return None

    outcomes = _coerce_list(market.get("outcomes"))
    if outcomes:
        norm_selected = normalize_team_name(selected_side_label)
        for idx, outcome in enumerate(outcomes):
            if norm_selected and norm_selected in normalize_team_name(str(outcome)):
                if idx < len(token_ids):
                    return str(token_ids[idx])

    yes_side = _infer_yes_side(str(market.get("question") or ""), team1, team2)
    if yes_side is None:
        # Default fallback: yes token is index 0.
        return str(token_ids[0])

    if normalize_team_name(selected_side_label) == normalize_team_name(yes_side):
        return str(token_ids[0])
    return str(token_ids[1])


def _extract_implied_probability_from_book(orderbook: Dict[str, Any]) -> Optional[float]:
    bids = orderbook.get("bids") or []
    asks = orderbook.get("asks") or []

    def _best_price(levels: List[Any], side: str) -> Optional[float]:
        prices: List[float] = []
        for level in levels:
            if not isinstance(level, dict):
                continue
            p = _coerce_float(level.get("price"))
            if p is None:
                continue
            prices.append(p)
        if not prices:
            return None
        # CLOB payloads are not always sorted, so compute best explicitly.
        return max(prices) if side == "bid" else min(prices)

    best_bid = _best_price(bids, "bid")
    best_ask = _best_price(asks, "ask")

    midpoint = None
    if best_bid is not None and best_ask is not None:
        midpoint = (best_bid + best_ask) / 2.0
    elif best_bid is not None:
        midpoint = best_bid
    elif best_ask is not None:
        midpoint = best_ask

    if midpoint is None:
        return None
    midpoint = max(0.0, min(1.0, midpoint))
    return round(midpoint * 100.0, 2)


def _book_observed_time(orderbook: Dict[str, Any]) -> datetime:
    raw = orderbook.get("timestamp")
    if isinstance(raw, (int, float)):
        # Many APIs return milliseconds. Treat > 1e11 as ms.
        ts = raw / 1000.0 if raw > 1e11 else float(raw)
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.isdigit():
            numeric = float(stripped)
            ts = numeric / 1000.0 if numeric > 1e11 else numeric
            return datetime.fromtimestamp(ts, tz=timezone.utc)
    parsed = _parse_iso_datetime(raw if isinstance(raw, str) else None)
    return parsed or datetime.now(timezone.utc)


class PolymarketComparisonService:
    """Compare app fixture model probabilities vs Polymarket implied probabilities."""

    def __init__(self, client: Optional[PolymarketClient] = None, market_cache_ttl_sec: int = 45) -> None:
        self.client = client or PolymarketClient()
        self.market_cache_ttl_sec = market_cache_ttl_sec
        self._cached_markets: List[Dict[str, Any]] = []
        self._cached_at: Optional[datetime] = None

    def _get_markets_cached(self) -> List[Dict[str, Any]]:
        now = datetime.now(timezone.utc)
        if self._cached_at is not None:
            age = (now - self._cached_at).total_seconds()
            if age <= self.market_cache_ttl_sec and self._cached_markets:
                return self._cached_markets

        markets = self.client.get_markets(limit=500, active=True, closed=False)
        if not isinstance(markets, list):
            markets = []
        self._cached_markets = markets
        self._cached_at = now
        return markets

    def _get_markets_by_slug_candidates(self, team1: str, team2: str, start_utc: Optional[str], series: Optional[str]) -> List[Dict[str, Any]]:
        """Try deterministic slug candidate lookups for sports pages."""
        date_part = _date_part(start_utc)
        code1 = _team_code(team1)
        code2 = _team_code(team2)
        prefix = _series_slug_prefix(series)
        if not date_part or not code1 or not code2:
            return []

        candidates = []
        if prefix:
            candidates.extend(
                [
                    f"{prefix}-{code1}-{code2}-{date_part}",
                    f"{prefix}-{code2}-{code1}-{date_part}",
                ]
            )
        else:
            # CREX live routes sometimes omit series labels; probe all known cricket prefixes.
            for known_prefix in sorted(set(SERIES_PREFIX_ALIASES.values())):
                candidates.extend(
                    [
                        f"{known_prefix}-{code1}-{code2}-{date_part}",
                        f"{known_prefix}-{code2}-{code1}-{date_part}",
                    ]
                )
        # Safety fallback without known prefix.
        candidates.extend([f"{code1}-{code2}-{date_part}", f"{code2}-{code1}-{date_part}"])

        seen_ids = set()
        found: List[Dict[str, Any]] = []
        for slug in candidates:
            try:
                rows = self.client.get_markets_by_slug(slug)
            except Exception:
                continue
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                market_id = str(row.get("id") or row.get("conditionId") or slug)
                if market_id in seen_ids:
                    continue
                seen_ids.add(market_id)
                found.append(row)
        return found

    def compare_fixture(
        self,
        team1: str,
        team2: str,
        start_utc: Optional[str] = None,
        series: Optional[str] = None,
        model_team1_win_pct: Optional[float] = None,
        model_team2_win_pct: Optional[float] = None,
        stale_after_sec: int = 180,
    ) -> Dict[str, Any]:
        selected_side = infer_selected_side(team1, team2, model_team1_win_pct, model_team2_win_pct)
        fixture_key = build_fixture_key(team1, team2, start_utc)
        model_side_win_pct = None
        if model_team1_win_pct is not None and model_team2_win_pct is not None:
            model_side_win_pct = model_team1_win_pct if normalize_team_name(selected_side) == normalize_team_name(team1) else model_team2_win_pct

        markets = self._get_markets_cached()
        linked_market, confidence = select_best_market(
            markets=markets,
            team1=team1,
            team2=team2,
            start_utc=start_utc,
            series_name=series,
        )
        fallback_used = False
        if linked_market is None:
            slug_markets = self._get_markets_by_slug_candidates(team1, team2, start_utc, series)
            if slug_markets:
                linked_market, confidence = select_best_market(
                    markets=slug_markets,
                    team1=team1,
                    team2=team2,
                    start_utc=start_utc,
                    series_name=series,
                )
                if linked_market is None:
                    # Deterministic slug lookups can still miss threshold when series/date
                    # metadata is sparse; accept strong two-team token matches.
                    best_slug_market: Optional[Dict[str, Any]] = None
                    best_slug_score = 0.0
                    for candidate in slug_markets:
                        text = _extract_market_text(candidate)
                        t1_overlap = _token_overlap_score(team1, text)
                        t2_overlap = _token_overlap_score(team2, text)
                        score = min(t1_overlap, t2_overlap)
                        if score >= 0.5 and score > best_slug_score:
                            best_slug_market = candidate
                            best_slug_score = score
                    if best_slug_market is not None:
                        linked_market = best_slug_market
                        confidence = max(confidence, 0.401)
                fallback_used = linked_market is not None

        if linked_market is None:
            return {
                "success": True,
                "status": "no_match",
                "fixture_key": fixture_key,
                "linked_market": None,
                "selection": {"selected_side_label": selected_side, "token_id": None},
                "probabilities": {
                    "model_win_pct": model_side_win_pct,
                    "market_implied_pct": None,
                    "edge_pct_points": None,
                },
                "quote": None,
                "warnings": ["No reliable Polymarket market match found"],
            }

        token_id = _choose_token_for_side(
            market=linked_market,
            selected_side_label=selected_side,
            team1=team1,
            team2=team2,
        )
        if token_id is None:
            return {
                "success": True,
                "status": "quote_unavailable",
                "fixture_key": fixture_key,
                "linked_market": {
                    "market_id": str(linked_market.get("id") or linked_market.get("conditionId") or ""),
                    "question": linked_market.get("question"),
                    "slug": linked_market.get("slug"),
                    "link_confidence": round(confidence, 3),
                },
                "selection": {"selected_side_label": selected_side, "token_id": None},
                "probabilities": {
                    "model_win_pct": model_side_win_pct,
                    "market_implied_pct": None,
                    "edge_pct_points": None,
                },
                "quote": None,
                "warnings": ["Market linked, but token selection failed"],
            }

        try:
            orderbook = self.client.get_clob_order_book(token_id)
            implied_pct = _extract_implied_probability_from_book(orderbook)
            if implied_pct is None:
                raise ValueError("No usable price levels in orderbook")
            observed_at = _book_observed_time(orderbook)
        except Exception as exc:
            return {
                "success": True,
                "status": "quote_unavailable",
                "fixture_key": fixture_key,
                "linked_market": {
                    "market_id": str(linked_market.get("id") or linked_market.get("conditionId") or ""),
                    "question": linked_market.get("question"),
                    "slug": linked_market.get("slug"),
                    "link_confidence": round(confidence, 3),
                },
                "selection": {"selected_side_label": selected_side, "token_id": token_id},
                "probabilities": {
                    "model_win_pct": model_side_win_pct,
                    "market_implied_pct": None,
                    "edge_pct_points": None,
                },
                "quote": None,
                "warnings": [f"Orderbook unavailable: {exc}"],
            }

        now = datetime.now(timezone.utc)
        quote_age_sec = max(0, int((now - observed_at).total_seconds()))
        status = "stale" if quote_age_sec > stale_after_sec else "ok"
        return {
            "success": True,
            "status": status,
            "fixture_key": fixture_key,
            "linked_market": {
                "market_id": str(linked_market.get("id") or linked_market.get("conditionId") or ""),
                "question": linked_market.get("question"),
                "slug": linked_market.get("slug"),
                "link_confidence": round(confidence, 3),
            },
            "selection": {"selected_side_label": selected_side, "token_id": token_id},
            "probabilities": {
                "model_win_pct": model_side_win_pct,
                "market_implied_pct": implied_pct,
                "edge_pct_points": compute_edge_pct_points(model_side_win_pct, implied_pct),
            },
            "quote": {
                "source": "polymarket",
                "quote_age_sec": quote_age_sec,
                "observed_at_utc": observed_at.isoformat(),
            },
            "warnings": (
                (["Quote is stale"] if status != "ok" else [])
                + (["Matched via slug fallback"] if fallback_used else [])
            ),
        }

    def compare_batch(self, fixtures: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare a batch of fixtures for bulk row rendering."""
        results = []
        for fixture in fixtures:
            result = self.compare_fixture(
                team1=str(fixture.get("team1") or ""),
                team2=str(fixture.get("team2") or ""),
                start_utc=fixture.get("start_utc"),
                series=fixture.get("series"),
                model_team1_win_pct=_coerce_float(fixture.get("model_team1_win_pct")),
                model_team2_win_pct=_coerce_float(fixture.get("model_team2_win_pct")),
            )
            result["row_key"] = fixture.get("row_key")
            results.append(result)
        return {"success": True, "count": len(results), "results": results}

    # ------------------------------------------------------------------
    # Wave 5: Multi-market comparison
    # ------------------------------------------------------------------

    def find_event_markets(
        self,
        team1: str,
        team2: str,
        start_utc: Optional[str] = None,
        series: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return ALL Polymarket markets linked to this fixture.

        Polymarket groups multiple markets (moneyline, top batter, most
        sixes, toss, etc.) under a single 'event'. We first locate the
        best moneyline-style match using the existing fuzzy matching,
        then expand to sibling markets sharing the same `eventId` /
        `eventSlug`.
        """
        markets = self._get_markets_cached()
        anchor, _conf = select_best_market(
            markets=markets,
            team1=team1,
            team2=team2,
            start_utc=start_utc,
            series_name=series,
        )
        if anchor is None:
            slug_markets = self._get_markets_by_slug_candidates(team1, team2, start_utc, series)
            if slug_markets:
                anchor, _conf = select_best_market(
                    markets=slug_markets,
                    team1=team1,
                    team2=team2,
                    start_utc=start_utc,
                    series_name=series,
                )
                if anchor is None and slug_markets:
                    anchor = slug_markets[0]
        if anchor is None:
            return []
        # Expand to siblings sharing the event id/slug
        event_id = str(anchor.get("eventId") or anchor.get("event_id") or "")
        event_slug = str(anchor.get("eventSlug") or anchor.get("event_slug") or "")
        siblings: List[Dict[str, Any]] = []
        for market in markets:
            if not isinstance(market, dict):
                continue
            mid = str(market.get("eventId") or market.get("event_id") or "")
            mslug = str(market.get("eventSlug") or market.get("event_slug") or "")
            if (event_id and mid == event_id) or (event_slug and mslug == event_slug):
                siblings.append(market)
        if not siblings:
            siblings = [anchor]
        # Belt-and-braces: dedupe by market id
        seen = set()
        unique: List[Dict[str, Any]] = []
        for m in siblings:
            key = str(m.get("id") or m.get("conditionId") or m.get("question") or len(unique))
            if key in seen:
                continue
            seen.add(key)
            unique.append(m)
        return unique

    def _classify_market_type(self, market: Dict[str, Any]) -> Optional[str]:
        """Classify a Polymarket market into one of the cricket types.

        Order: try regex patterns (covers explicit "Top Batter" / "Most Sixes"
        markets); if those don't match, fall back to outcome-shape inference:
        a 2-outcome market with non-Yes/No labels is treated as a moneyline
        (this catches the "Team A vs. Team B" Polymarket events that only
        list team names as outcomes without a verb in the title).
        """
        text_parts = [
            str(market.get("question") or ""),
            str(market.get("title") or ""),
            str(market.get("slug") or ""),
            str(market.get("groupItemTitle") or ""),
            str(market.get("eventTitle") or ""),
        ]
        text = " ".join([p for p in text_parts if p])
        if text:
            for market_type, pattern in MARKET_TYPE_PATTERNS:
                if pattern.search(text):
                    return market_type
        # Fallback: 2-outcome market with non-Yes/No labels -> moneyline
        outcomes = _coerce_list(market.get("outcomes"))
        if len(outcomes) == 2:
            normalized = [normalize_team_name(str(o)) for o in outcomes]
            if not any(n in ("yes", "no") for n in normalized):
                return MARKET_TYPE_MONEYLINE
        return None

    @staticmethod
    def _market_outcomes_with_prices(market: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return [{label, token_id, last_price}, ...] for a Polymarket market.

        Polymarket markets carry `outcomes` (label list) parallel with
        `clobTokenIds` (token list) and optionally `outcomePrices` (last
        traded prices in a stringified JSON array). Best-effort parser
        for that schema.
        """
        labels = _coerce_list(market.get("outcomes"))
        token_ids = _coerce_list(market.get("clobTokenIds") or market.get("clobTokenIDs"))
        last_prices_raw = market.get("outcomePrices")
        last_prices: List[Optional[float]] = []
        if isinstance(last_prices_raw, str):
            try:
                parsed = json.loads(last_prices_raw)
                last_prices = [_coerce_float(v) for v in parsed if isinstance(parsed, list)]
            except json.JSONDecodeError:
                last_prices = []
        elif isinstance(last_prices_raw, list):
            last_prices = [_coerce_float(v) for v in last_prices_raw]
        rows: List[Dict[str, Any]] = []
        for idx, label in enumerate(labels):
            row = {
                "label": str(label),
                "token_id": str(token_ids[idx]) if idx < len(token_ids) else None,
                "last_price": last_prices[idx] if idx < len(last_prices) else None,
            }
            rows.append(row)
        return rows

    def _live_market_implied_pct(self, token_id: Optional[str]) -> Optional[float]:
        """Try to fetch a fresh CLOB midpoint for a token; fall back to None."""
        if not token_id:
            return None
        try:
            book = self.client.get_clob_order_book(token_id)
            implied = _extract_implied_probability_from_book(book)
            return implied
        except Exception:
            return None

    @staticmethod
    def label_matches_team(label: str, team_name: str) -> bool:
        """Wave 5 Phase 5: token-overlap label-to-team matcher.

        Substring matching breaks on franchise rebrands like
        'Royal Challengers Bengaluru' vs 'Royal Challengers Bangalore'.
        Token-overlap with generic-word filtering handles both.
        """
        if not label or not team_name:
            return False
        label_tokens = {tok for tok in _tokenize(label) if tok not in GENERIC_TEAM_WORDS}
        team_tokens = {tok for tok in _tokenize(team_name) if tok not in GENERIC_TEAM_WORDS}
        if not label_tokens or not team_tokens:
            return False
        # Match if any meaningful token overlaps. e.g. "royal challengers
        # bangalore" and "royal challengers bengaluru" share "royal" and
        # "challengers" -> match.
        return bool(label_tokens & team_tokens)

    @staticmethod
    def _model_outcome_prob_for_label(
        market_type: str,
        label: str,
        team1: str,
        team2: str,
        market_probs: Dict[str, Any],
    ) -> Optional[float]:
        """Map an outcome label string to a per-outcome model probability.

        Uses token-overlap matching (handles franchise rebrands).
        Returns None if we can't confidently map.
        """
        norm_label = normalize_team_name(label)
        is_draw = any(tok in norm_label for tok in ("tie", "draw"))
        match_t1 = PolymarketComparisonService.label_matches_team(label, team1)
        match_t2 = PolymarketComparisonService.label_matches_team(label, team2)
        if market_type == MARKET_TYPE_MONEYLINE:
            ml = market_probs.get("moneyline", {})
            if match_t1 and not match_t2:
                return float(ml.get("team1") or 0.0) * 100.0
            if match_t2 and not match_t1:
                return float(ml.get("team2") or 0.0) * 100.0
            return None
        if market_type == MARKET_TYPE_TOP_BATTER:
            tb = market_probs.get("top_batter", {})
            if is_draw:
                return float(tb.get("draw") or 0.0) * 100.0
            if match_t1 and not match_t2:
                return float(tb.get("team1_higher") or 0.0) * 100.0
            if match_t2 and not match_t1:
                return float(tb.get("team2_higher") or 0.0) * 100.0
            return None
        if market_type == MARKET_TYPE_MOST_SIXES:
            ms = market_probs.get("most_sixes", {})
            if is_draw:
                return float(ms.get("draw") or 0.0) * 100.0
            if match_t1 and not match_t2:
                return float(ms.get("team1") or 0.0) * 100.0
            if match_t2 and not match_t1:
                return float(ms.get("team2") or 0.0) * 100.0
            return None
        if market_type == MARKET_TYPE_TOSS_MATCH_DOUBLE:
            tmd = market_probs.get("toss_match_double", {})
            # Polymarket labels are typically of the form
            # "Win Toss & Win Match" / "Win Toss & Lose Match" / "Lose Toss & Win Match"
            # But the specific team association varies. As a fallback,
            # we don't auto-map TMD outcomes; the UI shows the four
            # composite probs and the user manually picks.
            # Heuristic: substring match for team name + "& win/lose".
            if norm_t1 and norm_t1 in norm_label:
                if "win match" in norm_label or "match win" in norm_label:
                    if "win toss" in norm_label or "toss win" in norm_label:
                        return float(tmd.get("toss_team1_match_team1") or 0.0) * 100.0
                    return float(tmd.get("toss_team2_match_team1") or 0.0) * 100.0
                if "lose match" in norm_label or "match lose" in norm_label:
                    # implied: team2 wins match
                    if "win toss" in norm_label or "toss win" in norm_label:
                        return float(tmd.get("toss_team1_match_team2") or 0.0) * 100.0
                    return float(tmd.get("toss_team2_match_team2") or 0.0) * 100.0
            return None
        return None

    def compare_fixture_multi(
        self,
        team1: str,
        team2: str,
        market_probs: Dict[str, Any],
        start_utc: Optional[str] = None,
        series: Optional[str] = None,
        stale_after_sec: int = 180,
    ) -> Dict[str, Any]:
        """Multi-market Polymarket comparison.

        Args:
            team1, team2: fixture team names.
            market_probs: output dict from
                `src.models.market_outputs.derive_polymarket_market_probs`.
                Required because moneyline-only callers can use the older
                `compare_fixture` -- this method's whole point is the 4
                markets together.

        Returns:
            {
                "success": True,
                "fixture_key": "...",
                "markets": {
                    "moneyline": {market_meta, outcomes: [{label, model_prob_pct, market_implied_pct, edge_pp, token_id, ...}]},
                    "top_batter": {...},
                    "most_sixes": {...},
                    "toss_match_double": {...},
                },
                "warnings": [...],
            }

        Markets that aren't found on Polymarket appear with `status = "no_match"`.
        Markets that are found but have no usable price appear with `status = "quote_unavailable"`.
        """
        fixture_key = build_fixture_key(team1, team2, start_utc)
        all_markets = self.find_event_markets(team1, team2, start_utc, series)
        warnings: List[str] = []
        if not all_markets:
            return {
                "success": True,
                "status": "no_match",
                "fixture_key": fixture_key,
                "markets": {
                    MARKET_TYPE_MONEYLINE: {"status": "no_match"},
                    MARKET_TYPE_TOP_BATTER: {"status": "no_match"},
                    MARKET_TYPE_MOST_SIXES: {"status": "no_match"},
                    MARKET_TYPE_TOSS_MATCH_DOUBLE: {"status": "no_match"},
                },
                "warnings": ["No reliable Polymarket markets matched"],
            }

        # Group markets by classified type. Take the first match per type
        # (Polymarket typically lists each market type once per event).
        markets_by_type: Dict[str, Dict[str, Any]] = {}
        for market in all_markets:
            market_type = self._classify_market_type(market)
            if not market_type:
                continue
            if market_type not in markets_by_type:
                markets_by_type[market_type] = market

        out: Dict[str, Dict[str, Any]] = {}
        now = datetime.now(timezone.utc)
        for market_type in (
            MARKET_TYPE_MONEYLINE,
            MARKET_TYPE_TOP_BATTER,
            MARKET_TYPE_MOST_SIXES,
            MARKET_TYPE_TOSS_MATCH_DOUBLE,
        ):
            market = markets_by_type.get(market_type)
            if market is None:
                out[market_type] = {"status": "no_match"}
                continue

            outcomes = self._market_outcomes_with_prices(market)
            outcome_rows: List[Dict[str, Any]] = []
            quote_observed_at = now
            quote_age_sec = 0
            had_live_quote = False
            for outcome in outcomes:
                token_id = outcome.get("token_id")
                live_implied_pct = self._live_market_implied_pct(token_id)
                if live_implied_pct is not None:
                    market_implied_pct = live_implied_pct
                    had_live_quote = True
                elif outcome.get("last_price") is not None:
                    market_implied_pct = round(float(outcome["last_price"]) * 100.0, 2)
                else:
                    market_implied_pct = None
                model_pct = self._model_outcome_prob_for_label(
                    market_type=market_type,
                    label=outcome.get("label", ""),
                    team1=team1,
                    team2=team2,
                    market_probs=market_probs,
                )
                edge_pp = (
                    compute_edge_pct_points(model_pct, market_implied_pct)
                    if (model_pct is not None and market_implied_pct is not None)
                    else None
                )
                outcome_rows.append({
                    "label": outcome.get("label"),
                    "token_id": token_id,
                    "model_pct": model_pct,
                    "market_implied_pct": market_implied_pct,
                    "edge_pp": edge_pp,
                })

            status = "ok"
            if not had_live_quote and not any(o["market_implied_pct"] is not None for o in outcome_rows):
                status = "quote_unavailable"
            out[market_type] = {
                "status": status,
                "linked_market": {
                    "market_id": str(market.get("id") or market.get("conditionId") or ""),
                    "question": market.get("question"),
                    "slug": market.get("slug"),
                    "event_id": str(market.get("eventId") or market.get("event_id") or ""),
                    "event_slug": str(market.get("eventSlug") or market.get("event_slug") or ""),
                },
                "outcomes": outcome_rows,
                "quote": {
                    "source": "polymarket",
                    "quote_age_sec": quote_age_sec,
                    "observed_at_utc": quote_observed_at.isoformat(),
                },
                "fee_applied_taker": POLYMARKET_TAKER_FEE,
            }

        return {
            "success": True,
            "status": "ok",
            "fixture_key": fixture_key,
            "markets": out,
            "warnings": warnings,
        }

    def compare_batch_multi(
        self,
        fixtures: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Multi-market batch wrapper.

        Each input fixture should include `market_probs` (from
        `derive_polymarket_market_probs`). Fixtures missing market_probs
        fall back to a moneyline-only response (single team1_win_prob).
        """
        results = []
        for fixture in fixtures:
            market_probs = fixture.get("market_probs") or {}
            if not market_probs:
                t1_pct = _coerce_float(fixture.get("model_team1_win_pct"))
                t2_pct = _coerce_float(fixture.get("model_team2_win_pct"))
                if t1_pct is not None and t2_pct is not None:
                    market_probs = {
                        "moneyline": {"team1": t1_pct / 100.0, "team2": t2_pct / 100.0},
                        "top_batter": {"team1_higher": 0.0, "draw": 1.0, "team2_higher": 0.0},
                        "most_sixes": {"team1": 0.0, "draw": 1.0, "team2": 0.0},
                        "toss_match_double": {
                            "toss_team1_match_team1": (t1_pct / 100.0) * 0.5,
                            "toss_team1_match_team2": (t2_pct / 100.0) * 0.5,
                            "toss_team2_match_team1": (t1_pct / 100.0) * 0.5,
                            "toss_team2_match_team2": (t2_pct / 100.0) * 0.5,
                        },
                    }

            result = self.compare_fixture_multi(
                team1=str(fixture.get("team1") or ""),
                team2=str(fixture.get("team2") or ""),
                market_probs=market_probs,
                start_utc=fixture.get("start_utc"),
                series=fixture.get("series"),
            )
            result["row_key"] = fixture.get("row_key")
            results.append(result)
        return {"success": True, "count": len(results), "results": results}

