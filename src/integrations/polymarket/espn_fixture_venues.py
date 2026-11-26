"""Ground for a Polymarket cricket fixture, taken from ESPN's public scorepanel.

ESPN (which owns ESPNcricinfo) lists every scheduled fixture for a date with
its ground, city, country, UTC start, both teams and the match class. Cricsheet
venue names are largely Cricinfo's, so ESPN grounds map back onto
`venues.venue_id` well. Endpoint:

    https://site.api.espn.com/apis/site/v2/sports/cricket/scorepanel?dates=YYYYMMDD

It is unauthenticated and undocumented. Keep it to one request per date per
scan, cache the result, and treat a failed fetch as "no opinion".

`match_fixture` pairs a Polymarket fixture (two team labels, gender, kickoff or
slug date) with one ESPN event. `map_ground_to_venue` turns the ESPN ground into
a `venue_id`. `resolve` chains the two and returns a `VenueMatch` carrying a
confidence grade so callers can refuse anything below "high".
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

SCOREPANEL_URLS = (
    "https://site.api.espn.com/apis/site/v2/sports/cricket/scorepanel?dates={d}",
    "https://site.web.api.espn.com/apis/site/v2/sports/cricket/scorepanel?dates={d}",
)

TEAM_ALIASES = {
    "uae": "united arab emirates",
    "usa": "united states of america",
    "united states": "united states of america",
    "png": "papua new guinea",
    "rcb": "royal challengers bengaluru",
    "royal challengers bangalore": "royal challengers bengaluru",
    "kings xi punjab": "punjab kings",
    "delhi daredevils": "delhi capitals",
}
TEAM_STOP = {"cricket", "club", "team", "the", "ccc", "cc", "and", "of"}

COUNTRY_ALIASES = {
    "usa": "united states of america",
    "united states": "united states of america",
    "uae": "united arab emirates",
    "wales": "england",
    "united kingdom": "england",
}
# cricket.db and ESPN disagree on whether a Caribbean ground is in "West Indies"
# or its island, so the country guard is skipped for these.
NO_COUNTRY_CHECK = {
    "west indies", "jamaica", "barbados", "trinidad", "trinidad and tobago", "guyana",
    "antigua", "antigua and barbuda", "st lucia", "saint lucia", "st kitts", "st kitts and nevis",
    "grenada", "dominica", "st vincent", "st vincent and the grenadines", "cayman islands",
}

T20_CLASSES = ("twenty20", "t20")
ODI_CLASSES = ("odi", "list a")


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def _compact(league: dict, e: dict) -> dict:
    c = e["competitions"][0]
    v = c.get("venue") or {}
    a = v.get("address") or {}
    cl = c.get("class") or {}
    return {
        "espn_id": e.get("id"), "league_id": league.get("id"), "league": league.get("name"),
        "start": e.get("date"), "name": e.get("name"),
        "teams": [x["team"].get("displayName") for x in c.get("competitors", [])],
        "team_abbr": [x["team"].get("abbreviation") for x in c.get("competitors", [])],
        "ground_id": v.get("id"), "ground": v.get("fullName"), "city": a.get("city"),
        "country": a.get("country"), "neutral": c.get("neutralSite"),
        "class": cl.get("generalClassCard"), "event_type": cl.get("eventType"),
    }


_MEMO: Dict[str, Tuple[float, List[dict]]] = {}
FAILURE_BACKOFF_S = 600.0


def fetch_scorepanel(day: date, *, session=None, cache_dir: Optional[Path] = None,
                     max_age_s: float = 6 * 3600, retries: int = 3) -> List[dict]:
    """Compact ESPN events listed for `day`. Returns [] on any failure.

    Results are memoised in-process for `max_age_s`, and on disk too when
    `cache_dir` is given, so a scan loop asks ESPN about each date at most
    once per window.
    """
    q = day.strftime("%Y%m%d")
    hit = _MEMO.get(q)
    if hit and time.time() - hit[0] < max_age_s:
        return hit[1]
    cache = Path(cache_dir) / f"scorepanel_{q}.json" if cache_dir else None
    if cache and cache.exists() and time.time() - cache.stat().st_mtime < max_age_s:
        try:
            out = json.loads(cache.read_text())
            _MEMO[q] = (time.time(), out)
            return out
        except ValueError:
            pass
    import requests

    s = session or requests.Session()
    payload = None
    for attempt in range(retries):
        try:
            r = s.get(SCOREPANEL_URLS[attempt % 2].format(d=q), timeout=20,
                      headers={"User-Agent": "Mozilla/5.0 (fixture venue lookup)"})
            if r.status_code == 200:
                payload = r.json()
                break
        except Exception as exc:
            logger.debug(f"scorepanel {q} attempt {attempt}: {exc}")
        time.sleep(1.5 * 2 ** attempt)
    if payload is None:
        _MEMO[q] = (time.time() - max_age_s + FAILURE_BACKOFF_S, [])
        return []
    out = []
    for block in payload.get("scores", []):
        league = (block.get("leagues") or [{}])[0]
        for e in block.get("events", []):
            try:
                out.append(_compact(league, e))
            except (KeyError, IndexError, TypeError):
                continue
    _MEMO[q] = (time.time(), out)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out))
    return out


# ---------------------------------------------------------------------------
# Event classification
# ---------------------------------------------------------------------------

def parse_utc(value) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        s = str(value).replace("Z", "+00:00")
        d = datetime.fromisoformat(s)
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def event_gender(ev: dict) -> str:
    blob = " ".join([ev.get("class") or "", ev.get("name") or ""] + list(ev.get("teams") or []))
    return "female" if re.search(r"\bwomen", blob.lower()) else "male"


def event_is_youth(ev: dict) -> bool:
    blob = " ".join([ev.get("class") or "", ev.get("league") or ""]).lower()
    teams = " ".join(ev.get("teams") or []).lower()
    return bool(re.search(r"youth|under-1\d|u1\d\b|\bu-1\d", blob)
                or re.search(r"under-1\d|\bu1\d\b|\bu-1\d", teams))


def event_format(ev: dict) -> str:
    """'T20', 'ODI' or 'other' (first-class, Test, 10-over, unknown)."""
    et = (ev.get("event_type") or "").upper()
    cl = (ev.get("class") or "").lower()
    if et == "T20" or any(k in cl for k in T20_CLASSES):
        return "T20"
    if et == "ODI" or any(k in cl for k in ODI_CLASSES):
        return "ODI"
    return "other"


# ---------------------------------------------------------------------------
# Team matching
# ---------------------------------------------------------------------------

def norm_team(s: str) -> str:
    s = str(s or "").lower()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"\bwomen'?s?\b", " ", s)
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return TEAM_ALIASES.get(s, s)


def _tokens_align(small: set, big: set) -> bool:
    """Every token of `small` has a near-identical token in `big`, at least one exact."""
    exact = small & big
    if not exact:
        return False
    for t in small - exact:
        if not any(len(t) >= 5 and SequenceMatcher(None, t, u).ratio() >= 0.8 for u in big - exact):
            return False
    return True


def team_score(label: str, espn_team: str) -> float:
    """1.0 exact; 0.9 one name's tokens sit inside the other's (allowing a
    transliteration such as Gurkhas/Gorkhas); 0.85 whole-string fuzzy match."""
    a, b = norm_team(label), norm_team(espn_team)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ta, tb = set(a.split()) - TEAM_STOP, set(b.split()) - TEAM_STOP
    if ta and tb and (ta <= tb or tb <= ta):
        return 0.9
    if ta and tb:
        small, big = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
        if _tokens_align(small, big):
            return 0.9
    return 0.85 if SequenceMatcher(None, a, b).ratio() >= 0.9 else 0.0


def _pair_score(l1: str, l2: str, ev: dict) -> float:
    teams = ev.get("teams") or []
    if len(teams) != 2:
        return 0.0
    fwd = min(team_score(l1, teams[0]), team_score(l2, teams[1]))
    rev = min(team_score(l1, teams[1]), team_score(l2, teams[0]))
    return max(fwd, rev)


@dataclass
class EventMatch:
    event: Optional[dict]
    team_score: float = 0.0
    hours_off: Optional[float] = None
    n_candidates: int = 0
    runner_up: float = 0.0
    reason: str = ""


def match_fixture(team1_label: str, team2_label: str, gender: str,
                  events: Iterable[dict], *, kickoff: Optional[datetime] = None,
                  slug_date: Optional[date] = None, fmt: Optional[str] = None,
                  max_hours: float = 30.0) -> EventMatch:
    """Best ESPN event for a fixture; `event` is None when nothing qualifies.

    Needs `kickoff` (UTC) or `slug_date` (Polymarket names slugs in US Pacific
    time, so the UTC start can fall on the following day).
    """
    if kickoff is None and slug_date is None:
        return EventMatch(None, reason="no_time")
    if kickoff is None:
        anchor = datetime(slug_date.year, slug_date.month, slug_date.day, 19, tzinfo=timezone.utc)
        max_hours = max(max_hours, 30.0)
    else:
        anchor = kickoff
    scored = []
    seen = set()
    for ev in events:
        if ev.get("espn_id") in seen:
            continue
        seen.add(ev.get("espn_id"))
        if event_gender(ev) != gender or event_is_youth(ev):
            continue
        ef = event_format(ev)
        if ef == "other" or (fmt and ef != fmt):
            continue
        st = parse_utc(ev.get("start"))
        if st is None:
            continue
        dh = abs((st - anchor).total_seconds()) / 3600.0
        if dh > max_hours:
            continue
        ts = _pair_score(team1_label, team2_label, ev)
        if ts < 0.85:
            continue
        scored.append((ts, -dh, ev))
    if not scored:
        return EventMatch(None, reason="no_candidate")
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    ts, ndh, ev = scored[0]
    runner = scored[1][0] if len(scored) > 1 else 0.0
    return EventMatch(ev, ts, -ndh, len(scored), runner, "ok")


# ---------------------------------------------------------------------------
# Ground -> venue_id
# ---------------------------------------------------------------------------

def _norm_country(c: Optional[str]) -> str:
    c = (c or "").strip().lower()
    if c in NO_COUNTRY_CHECK:
        return ""
    return COUNTRY_ALIASES.get(c, c)


def _norm_ground(name: str) -> str:
    from src.data.venue_normalizer import normalize_venue_name
    return normalize_venue_name(re.sub(r"^\s*the\s+", "", name or "", flags=re.I))


def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


@dataclass
class VenueRow:
    venue_id: int
    name: str
    city: str
    country: str
    n_matches: int = 0


def map_ground_to_venue(ground: Optional[str], city: Optional[str], country: Optional[str],
                        venues: Sequence[VenueRow], *, ground_id: Optional[str] = None,
                        ground_map: Optional[Dict[str, int]] = None
                        ) -> Tuple[Optional[int], float, str]:
    """(venue_id, score, method). Score 1.0 for a learned ground-id hit."""
    if ground_map and ground_id and str(ground_id) in ground_map:
        return int(ground_map[str(ground_id)]), 1.0, "ground_id_map"
    if not ground:
        return None, 0.0, "no_ground"
    g_full = re.sub(r"^the\s+", "", (ground or "").lower().strip())
    g = _norm_ground(ground)
    gc = (city or "").strip().lower()
    gco = _norm_country(country)
    best: Tuple[float, int, Optional[VenueRow]] = (0.0, -1, None)
    for v in venues:
        vco = _norm_country(v.country)
        if gco and vco and gco != vco and not (gco in vco or vco in gco):
            continue
        s = max(_ratio(g, _norm_ground(v.name)),
                _ratio(g_full, re.sub(r"^the\s+", "", (v.name or "").lower().strip())))
        vcity = (v.city or "").lower()
        if gc and (gc == vcity or gc in (v.name or "").lower()):
            s += 0.08
        elif gc and vcity and gc != vcity:
            s -= 0.15
        key = (round(s, 4), v.n_matches)
        if key > (round(best[0], 4), best[1]):
            best = (s, v.n_matches, v)
    s, _, v = best
    if v is None or s < 0.80:
        return None, max(0.0, s), "no_name_match"
    return v.venue_id, min(1.0, s), "name"


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

@dataclass
class VenueMatch:
    venue_id: Optional[int]
    confidence: str            # 'high' / 'medium' / 'low' / 'none'
    score: float
    espn_id: Optional[str] = None
    ground: Optional[str] = None
    method: str = ""
    hours_off: Optional[float] = None
    reason: str = ""


def grade(em: EventMatch, venue_score: float, venue_method: str, kickoff_known: bool) -> Tuple[str, float]:
    if em.event is None:
        return "none", 0.0
    t = em.team_score
    dt = em.hours_off if em.hours_off is not None else 99.0
    time_ok = dt <= 4.0 if kickoff_known else dt <= 30.0
    ambiguous = em.runner_up >= em.team_score and em.n_candidates > 1
    vs = venue_score if venue_method != "no_ground" else 0.0
    score = t * (1.0 if time_ok else 0.8) * (vs if vs else 0.5) * (0.6 if ambiguous else 1.0)
    if t >= 0.9 and time_ok and not ambiguous and vs >= 0.9:
        return "high", score
    if t >= 0.85 and vs >= 0.8 and not ambiguous:
        return "medium", score
    return "low", score


def resolve(team1_label: str, team2_label: str, gender: str, events: Iterable[dict],
            venues: Sequence[VenueRow], *, kickoff: Optional[datetime] = None,
            slug_date: Optional[date] = None, fmt: Optional[str] = None,
            ground_map: Optional[Dict[str, int]] = None) -> VenueMatch:
    em = match_fixture(team1_label, team2_label, gender, events, kickoff=kickoff,
                       slug_date=slug_date, fmt=fmt)
    if em.event is None:
        return VenueMatch(None, "none", 0.0, reason=em.reason)
    ev = em.event
    vid, vs, method = map_ground_to_venue(ev.get("ground"), ev.get("city"), ev.get("country"),
                                          venues, ground_id=ev.get("ground_id"),
                                          ground_map=ground_map)
    if vid is None:
        # The fixture is identified but its ground has no cricket.db history.
        # Callers should treat it as a neutral, history-free venue rather
        # than substitute a team's home ground.
        conf, score = grade(em, 1.0, "new_ground", kickoff is not None)
        return VenueMatch(None, conf, round(score, 4), ev.get("espn_id"), ev.get("ground"),
                          "new_ground", em.hours_off, "new_ground")
    conf, score = grade(em, vs, method, kickoff is not None)
    return VenueMatch(vid, conf, round(score, 4), ev.get("espn_id"), ev.get("ground"), method,
                      em.hours_off, em.reason)


def events_around(day: date, fetch=fetch_scorepanel, **kw) -> List[dict]:
    """ESPN events for day-1..day+1, which covers Pacific-dated slugs and late UTC starts."""
    out: List[dict] = []
    for off in (-1, 0, 1):
        out.extend(fetch(day + timedelta(days=off), **kw))
    return out
