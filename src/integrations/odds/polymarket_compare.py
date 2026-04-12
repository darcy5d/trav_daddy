"""Polymarket fixture linking and comparison helpers."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from src.integrations.polymarket import PolymarketClient

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

