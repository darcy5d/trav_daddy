"""Resolve a Polymarket fixture to the ground it is actually played at.

STATUS: ESPN PATH MEASURED, NOT WIRED INTO THE SCANNER.
--------------------------------------------------------
`venue_from_espn` matches the fixture to ESPN's public fixture list by team
names, gender, format and kickoff. On the 633 evaluated Polymarket fixtures it
resolved 96.7% at high confidence with 99.2% exact `venue_id` accuracy (99.8%
same ground; the rest are duplicate `venues` rows). The scanner's fallback,
team1's most-played ground, is right for 31% of them. In the walk-forward
match-level model the resolved ground is worth 0.0024-0.0042 Brier over the
fallback on 1,640 2026 matches, all CIs excluding zero, and is within 0.0003
of the true ground. See docs/polymarket-venue-matching.md in the project store.

Earlier V2-only measurement, kept for context
----------------------------------------------
Substituting the true ground for the scanner's
`paper_inputs.get_default_venue_for_team` fallback was tested on 388 paired
2026 matches (identical XI, as-of-date ELOs, seed and venue only differing).
Held-out (>= 2026-07-01, n=304): Brier went from 0.22701 to 0.22896, a delta of
+0.00195 - i.e. slightly WORSE, against a pass bar of <= -0.005. 142/304
matches improved, a coin flip. It failed, so nothing calls this module. Do not
wire it into `paper_bet_scan.py` on the strength of the idea alone; the
measurement says the venue input does not carry 0.005 Brier.

Why the effect is small: the four venue features (expected total, boundary
rate, wicket rate, embedding) move P(team1 wins) by a median of 1.6pp, and the
network appears to have learned little from them.

What this module is good for
----------------------------
It answers "what ground is this?" for analysis and for any future model that
re-trains with the ground as a first-class input. `resolve_fixture_venue`
returns `(venue_id, source)` and never raises.

Resolution order, first hit wins. `espn` runs only when the caller passes the
fixture's teams; `event_text` and `competition` are enabled by default; both
were 100% accurate when they fired on 2026 data.
`same_day_series` is available but OFF by default because it returned the wrong
ground for 26% of the matches it touched - a silent-wrong-input risk this
codebase has already been bitten by once.

  0. `espn` - ESPN fixture list (`espn_fixture_venues`). High confidence only
     by default. A confidently matched fixture at a ground with no
     cricket.db history returns `(None, "espn_new_ground")`: the caller should
     use a neutral venue, not a team's home ground.
  1. `event_text` - ground named in Polymarket event/market copy. Measured: 0
     of 51 live cricket fixtures mention a ground at all, so in practice this
     fires only for the handful of competitions that name their stadium.
  2. `competition` - the competition has only ever used one ground in the DB
     and has >= 3 matches there. Covers 13.7% of 2026 matches at 100% accuracy.
     Refuses known multi-venue containers (BBL, IPL, PSL, SA20, ...).
  3. `same_day_series` - ground of a sibling DB match from the same series on
     the same date. 86% coverage but 74% accuracy; opt in via
     `allow_same_day_series=True` only where a wrong ground is tolerable.

Alias table
-----------
`VENUE_ALIASES` maps normalized competition/venue text to a fragment of
`venues.name`, mirroring the `team_external_ids` pattern: the key is what
Polymarket says, the value is a substring of the DB name. An unknown key simply
falls through, so extending the table can only add resolutions.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time as _time
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from src.integrations.polymarket import espn_fixture_venues as efv
from src.integrations.polymarket.espn_ground_map import GROUND_TO_VENUE

logger = logging.getLogger(__name__)

CONFIDENCE_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}
_VENUE_ROWS: Dict[int, Tuple[sqlite3.Connection, float, List[efv.VenueRow]]] = {}

# Competition/venue text -> a fragment of venues.name.
VENUE_ALIASES: Dict[str, str] = {
    # Competitions pinned to one ground.
    "abu dhabi t10": "Sheikh Zayed",
    "abu dhabi league": "Sheikh Zayed",
    "t10 league": "Sheikh Zayed",
    # Grounds that appear by name in competition titles.
    "narendra modi stadium": "Narendra Modi",
    "wankhede": "Wankhede",
    "eden gardens": "Eden Gardens",
    "chinnaswamy": "Chinnaswamy",
    "chepauk": "Chepauk",
    "gaddafi stadium": "Gaddafi",
    "dubai international": "Dubai International",
    "sharjah": "Sharjah",
    "sheikh zayed": "Sheikh Zayed",
    "lord's": "Lord's",
    "lords": "Lord's",
    "the oval": "The Oval",
    "old trafford": "Old Trafford",
    "headingley": "Headingley",
    "edgbaston": "Edgbaston",
    "trent bridge": "Trent Bridge",
    "melbourne cricket ground": "Melbourne Cricket Ground",
    "sydney cricket ground": "Sydney Cricket Ground",
    "adelaide oval": "Adelaide Oval",
    "optus stadium": "Optus Stadium",
    "eden park": "Eden Park",
    "basin reserve": "Basin Reserve",
    "newlands": "Newlands",
    "supersport park": "SuperSport Park",
    "kingsmead": "Kingsmead",
    "r. premadasa": "R. Premadasa",
    "pallekele": "Pallekele",
    "rangiri dambulla": "Rangiri Dambulla",
    "shere bangla": "Shere Bangla",
    "zahur ahmed chowdhury": "Zahur Ahmed Chowdhury",
    "ekana": "Ekana",
    "sawai mansingh": "Sawai Mansingh",
    "punjab cricket association": "Punjab Cricket Association",
    "arun jaitley": "Arun Jaitley",
    "rajiv gandhi": "Rajiv Gandhi",
    "green park": "Green Park",
    "barsapara": "Barsapara",
    "vidarbha": "Vidarbha",
    "maharaja yadavindra singh": "Maharaja Yadavindra Singh",
    "niranjan shah": "Niranjan Shah",
    "maharashtra cricket association": "Maharashtra Cricket Association",
}

# Competitions that deliberately move between grounds. `competition_venue`
# refuses to pin these even if the DB happens to show one ground so far.
MULTI_VENUE_COMPETITIONS = (
    "big bash", "indian premier league", "ipl", "pakistan super league", "psl",
    "caribbean premier league", "cpl", "sa20", "the hundred", "county",
    "t20 blast", "vitality blast", "super smash", "big smash", "mlc",
    "major league cricket", "lanka premier league", "lpl", "ilt20",
    "international", "world cup", "test", "odi", "t20i",
)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).strip()


def _venue_name_map(conn: sqlite3.Connection) -> List[Tuple[int, str]]:
    cache = getattr(conn, "_venue_name_cache", None)
    if cache is not None:
        return cache
    cur = conn.cursor()
    cur.execute("SELECT venue_id, COALESCE(name, '') AS name FROM venues")
    out = [(int(r[0]), (r[1] or "").lower()) for r in cur.fetchall()]
    try:
        setattr(conn, "_venue_name_cache", out)
    except Exception:
        pass
    return out


def _find_venue_by_fragment(conn: sqlite3.Connection, fragment: str) -> Optional[int]:
    """Whole-name or unique-substring match of `fragment` against venues.name."""
    frag = (fragment or "").lower().strip()
    if not frag:
        return None
    names = _venue_name_map(conn)
    exact = [vid for vid, nm in names if nm.strip() == frag]
    if len(exact) == 1:
        return exact[0]
    subs = [(vid, nm) for vid, nm in names if frag in nm]
    if len(subs) == 1:
        return subs[0][0]
    if len(subs) > 1:
        # Ambiguous fragment is an alias-table bug: refuse rather than guess.
        logger.debug(f"venue fragment {fragment!r} matched {len(subs)} venues; skipping")
    return None


def _normalized_aliases() -> List[Tuple[str, str]]:
    """Alias keys run through the same normalization as the text being matched.

    Without this an apostrophe in a key ("lord's") can never match, because
    `_norm` turns the incoming text's apostrophe into a space.
    """
    cache = getattr(_normalized_aliases, "_cache", None)
    if cache is not None:
        return cache
    pairs = [(_norm(k), v) for k, v in VENUE_ALIASES.items()]
    pairs = [(k, v) for k, v in pairs if k]
    pairs.sort(key=lambda kv: len(kv[0]), reverse=True)
    setattr(_normalized_aliases, "_cache", pairs)
    return pairs


def venue_from_event_text(conn: sqlite3.Connection, blob: str) -> Optional[int]:
    """Pull a ground out of free text (Polymarket event/market copy)."""
    text = _norm(blob)
    if not text:
        return None
    for key, fragment in _normalized_aliases():
        if key in text:
            vid = _find_venue_by_fragment(conn, fragment)
            if vid is not None:
                return vid
    return None


def competition_venue(
    conn: sqlite3.Connection, tournament_name: Optional[str], *, min_matches: int = 3
) -> Optional[int]:
    """The single ground a competition has always used, if it has one.

    Fires only when the competition is not in `MULTI_VENUE_COMPETITIONS` and
    every historical match under that `event_name` was at the same ground, with
    at least `min_matches` behind it. Measured 100% accurate on 2026 data.
    """
    name = (tournament_name or "").strip()
    if not name:
        return None
    if any(token in _norm(name) for token in MULTI_VENUE_COMPETITIONS):
        return None
    cur = conn.cursor()
    cur.execute(
        """
        SELECT venue_id, COUNT(*) AS n
        FROM matches
        WHERE COALESCE(event_name, '') = ? AND venue_id IS NOT NULL
        GROUP BY venue_id
        ORDER BY n DESC
        """,
        (name,),
    )
    rows = cur.fetchall()
    if len(rows) != 1:
        return None
    venue_id, n = int(rows[0][0]), int(rows[0][1])
    return venue_id if n >= min_matches else None


def venue_from_same_day_series(
    conn: sqlite3.Connection,
    event_name: Optional[str],
    match_date: Optional[str],
) -> Optional[int]:
    """Ground used by another DB match in the same series on the same date.

    WARNING: 74% accurate on 2026 data. Off by default; see module docstring.
    """
    if not event_name or not match_date:
        return None
    cur = conn.cursor()
    cur.execute(
        """
        SELECT venue_id
        FROM matches
        WHERE COALESCE(event_name, '') = ? AND date = ? AND venue_id IS NOT NULL
        GROUP BY venue_id
        ORDER BY COUNT(*) DESC
        LIMIT 1
        """,
        (event_name, match_date),
    )
    row = cur.fetchone()
    return int(row[0]) if row else None


def _espn_venue_rows(conn: sqlite3.Connection) -> List[efv.VenueRow]:
    """Venues with their match count over the last three years.

    The count breaks ties between duplicate `venues` rows for one ground in
    favour of the row current matches are filed under.
    """
    hit = _VENUE_ROWS.get(id(conn))
    if hit and hit[0] is conn and _time.time() - hit[1] < 3600:
        return hit[2]
    cur = conn.cursor()
    cur.execute(
        """
        SELECT v.venue_id, COALESCE(v.name, ''), COALESCE(v.city, ''),
               COALESCE(v.country, ''), COUNT(m.match_id)
        FROM venues v
        LEFT JOIN matches m
          ON m.venue_id = v.venue_id AND m.date >= date('now', '-1095 day')
        GROUP BY v.venue_id
        """
    )
    out = [efv.VenueRow(int(r[0]), r[1], r[2], r[3], int(r[4])) for r in cur.fetchall()]
    _VENUE_ROWS[id(conn)] = (conn, _time.time(), out)
    return out


def venue_from_espn(
    conn: sqlite3.Connection,
    *,
    team1_label: str,
    team2_label: str,
    gender: str,
    kickoff: Optional[datetime] = None,
    match_date: Optional[str] = None,
    fmt: Optional[str] = None,
    min_confidence: str = "high",
    events: Optional[Sequence[dict]] = None,
    fetch: Optional[Callable[..., List[dict]]] = None,
    ground_map: Optional[Mapping[str, int]] = None,
) -> Tuple[Optional[int], str]:
    """Ground from ESPN's fixture list.

    Returns `(venue_id, "espn")`, `(None, "espn_new_ground")` when the fixture
    is identified but its ground has no cricket.db history, or
    `(None, "unresolved")`. `events` skips the network (tests, replays).
    """
    day = None
    if match_date:
        day = date.fromisoformat(str(match_date)[:10])
    elif kickoff is not None:
        day = efv.parse_utc(kickoff).date()
    if day is None or not team1_label or not team2_label:
        return None, "unresolved"
    if events is None:
        events = efv.events_around(day, fetch=fetch or efv.fetch_scorepanel)
    vm = efv.resolve(team1_label, team2_label, gender, events, _espn_venue_rows(conn),
                     kickoff=efv.parse_utc(kickoff) if kickoff is not None else None,
                     slug_date=day, fmt=fmt,
                     ground_map=GROUND_TO_VENUE if ground_map is None else ground_map)
    if CONFIDENCE_RANK.get(vm.confidence, 0) < CONFIDENCE_RANK[min_confidence]:
        return None, "unresolved"
    if vm.method == "new_ground":
        return None, "espn_new_ground"
    return vm.venue_id, "espn"


def resolve_fixture_venue(
    conn: sqlite3.Connection,
    *,
    event_text: Optional[str] = None,
    tournament_name: Optional[str] = None,
    match_date: Optional[str] = None,
    allow_same_day_series: bool = False,
    espn: Optional[Mapping[str, Any]] = None,
) -> Tuple[Optional[int], str]:
    """Best-effort ground for a Polymarket fixture.

    Returns `(venue_id, source)` with source one of 'espn' / 'espn_new_ground'
    / 'event_text' / 'competition' / 'same_day_series' / 'unresolved'.
    `None` with 'unresolved' means "no opinion"; the caller keeps its existing
    fallback. `None` with 'espn_new_ground' means the ground is known to have
    no history, so a neutral venue beats any team's home ground.

    `espn` takes the keyword arguments of `venue_from_espn` (team labels,
    gender, kickoff, ...); omit it to skip the network path.
    `allow_same_day_series` is off by default because that path is only 74%
    accurate.
    """
    try:
        if espn:
            vid, source = venue_from_espn(conn, **dict(espn))
            if source != "unresolved":
                return vid, source
    except Exception as exc:  # never let venue resolution break a scan
        logger.debug(f"espn venue resolve failed: {exc}")
    try:
        if event_text:
            vid = venue_from_event_text(conn, event_text)
            if vid is not None:
                return vid, "event_text"
        vid = competition_venue(conn, tournament_name)
        if vid is not None:
            return vid, "competition"
        if allow_same_day_series:
            vid = venue_from_same_day_series(conn, tournament_name, match_date)
            if vid is not None:
                return vid, "same_day_series"
    except Exception as exc:  # never let venue resolution break a scan
        logger.debug(f"venue resolve failed: {exc}")
    return None, "unresolved"
