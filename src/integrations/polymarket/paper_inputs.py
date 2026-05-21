"""Wave 5.7 / 5.11: build simulator inputs (XI + venue) for an upcoming fixture.

Extracted from scripts/probe_live_fixtures.py so both the live-probe CLI and
the paper-bet scanner can share the same lineup-derivation logic.

Wave 5.11 adds:
  - resolve_xi_names_to_ids(): fuzzy-match CREX player name strings to DB ids.
  - get_cached_xi(): read crex_xi_cache and return (batters, bowlers) if fresh.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Minimum fraction of XI that must resolve to DB player_ids before we trust
# the CREX lineup over historical fallback.
CREX_XI_MIN_MATCH = 7  # out of 11

# Maximum age of a crex_xi_cache row before we consider it stale.
CREX_XI_MAX_AGE_HOURS = 3.0


def get_recent_xi(
    conn,
    team_id: int,
    fmt: str,
    gender: str,
    n_recent_matches: int = 3,
) -> Tuple[List[int], List[int]]:
    """Return (batters, bowlers) for the team's last N matches.

    batters: 11 player_ids most-frequently selected, in batting position.
    bowlers: 5 player_ids by overs bowled.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT pms.match_id, pms.player_id, pms.batting_position,
               pms.balls_faced, pms.overs_bowled
        FROM player_match_stats pms
        JOIN matches m ON m.match_id = pms.match_id
        WHERE pms.team_id = ? AND m.match_type = ? AND m.gender = ?
          AND m.winner_id IS NOT NULL
        ORDER BY m.date DESC
        LIMIT 200
        """,
        (team_id, fmt, gender),
    )
    rows = [dict(r) for r in cur.fetchall()]
    if not rows:
        return [], []

    last_matches: List[int] = []
    seen_matches = set()
    for r in rows:
        if r["match_id"] not in seen_matches:
            seen_matches.add(r["match_id"])
            last_matches.append(r["match_id"])
        if len(last_matches) >= n_recent_matches:
            break

    cur.execute(
        f"""
        SELECT pms.match_id, pms.player_id, pms.batting_position,
               pms.balls_faced, pms.overs_bowled, m.date
        FROM player_match_stats pms
        JOIN matches m ON m.match_id = pms.match_id
        WHERE pms.team_id = ? AND pms.match_id IN ({",".join("?" * len(last_matches))})
        """,
        [team_id] + last_matches,
    )
    rows = [dict(r) for r in cur.fetchall()]
    if not rows:
        return [], []

    appearances: defaultdict = defaultdict(int)
    for r in rows:
        appearances[r["player_id"]] += 1

    cur.execute(
        """
        SELECT pms.player_id, pms.batting_position, pms.balls_faced, pms.overs_bowled
        FROM player_match_stats pms
        WHERE pms.team_id = ? AND pms.match_id = ?
        """,
        (team_id, last_matches[0]),
    )
    most_recent = [dict(r) for r in cur.fetchall()]
    most_recent.sort(key=lambda r: (
        r["batting_position"] if r["batting_position"] is not None else 99,
        -(r["balls_faced"] or 0),
    ))
    batters = [r["player_id"] for r in most_recent][:11]
    if len(batters) < 11:
        seen = set(batters)
        for p in sorted(appearances.keys(), key=lambda p: -appearances[p]):
            if p not in seen:
                batters.append(p)
                seen.add(p)
            if len(batters) >= 11:
                break

    most_recent.sort(key=lambda r: -(r["overs_bowled"] or 0))
    bowlers = [r["player_id"] for r in most_recent if (r["overs_bowled"] or 0) > 0][:5]
    if len(bowlers) < 5:
        seen = set(bowlers)
        for r in most_recent:
            if r["player_id"] not in seen:
                bowlers.append(r["player_id"])
                seen.add(r["player_id"])
            if len(bowlers) >= 5:
                break
    return batters[:11], bowlers[:5]


def get_default_venue_for_team(
    conn,
    team_id: int,
    fmt: str,
    gender: str,
) -> Optional[int]:
    """Pick the team's most-played venue as a venue fallback.

    For paper trading we don't always know which ground the upcoming match
    is at; the team's home venue is a reasonable proxy for venue features.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT m.venue_id, COUNT(*) AS n
        FROM matches m
        WHERE (m.team1_id = ? OR m.team2_id = ?)
          AND m.match_type = ? AND m.gender = ?
          AND m.venue_id IS NOT NULL
        GROUP BY m.venue_id
        ORDER BY n DESC
        LIMIT 1
        """,
        (team_id, team_id, fmt, gender),
    )
    row = cur.fetchone()
    return row["venue_id"] if row else None


# ---------------------------------------------------------------------------
# Wave 5.11: CREX XI cache helpers
# ---------------------------------------------------------------------------

def resolve_xi_names_to_ids(
    conn,
    xi_names: List[str],
    team_id: int,
    fmt: str,
    gender: str,
    threshold: float = 0.55,
) -> Tuple[List[int], int, int]:
    """Fuzzy-match a list of CREX player name strings to DB player_ids.

    Returns (player_ids, n_matched, n_input).

    Candidate pool: players who've appeared for the team since 2023. This
    keeps the pool small and team-scoped, improving match precision.
    Uses the same `match_abbreviated_name` utility already used by the
    post-toss daemon.
    """
    from src.utils.name_matcher import match_abbreviated_name

    if not xi_names:
        return [], 0, 0

    cur = conn.cursor()
    cur.execute(
        """
        SELECT DISTINCT pms.player_id, p.name
        FROM player_match_stats pms
        JOIN players p ON p.player_id = pms.player_id
        JOIN matches m ON m.match_id = pms.match_id
        WHERE pms.team_id = ?
          AND m.match_type = ? AND m.gender = ?
          AND m.date >= '2023-01-01'
        """,
        (team_id, fmt, gender),
    )
    candidates: List[Tuple[int, str]] = [
        (r["player_id"], r["name"]) for r in cur.fetchall()
    ]
    if not candidates:
        return [], 0, len(xi_names)

    ids: List[int] = []
    matched = 0
    for raw in xi_names:
        raw = (raw or "").strip()
        if not raw:
            continue
        m = match_abbreviated_name(raw, candidates, threshold=threshold)
        if m:
            ids.append(m[0])
            matched += 1

    return ids, matched, len(xi_names)


def get_cached_xi(
    conn,
    fixture_key: str,
    team_id: int,
    max_age_hours: float = CREX_XI_MAX_AGE_HOURS,
) -> Optional[Tuple[List[int], List[int]]]:
    """Return (batters, bowlers) from crex_xi_cache if the entry is fresh and
    has enough resolved players, otherwise return None.

    The cache stores a flat JSON array of player_ids in batting order (CREX
    playingxi-button lists XI in batting order). We mirror the same split
    used by get_recent_xi(): first 11 as batters, first 5 as bowlers.

    Returns None when:
      - crex_xi_cache table doesn't exist yet (first run before migration)
      - no cache row for this (fixture_key, team_id)
      - row is older than max_age_hours
      - fewer than CREX_XI_MIN_MATCH player_ids were resolved at scrape time
    """
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT players_json, n_matched, n_input, fetched_at
            FROM crex_xi_cache
            WHERE fixture_key = ? AND team_id = ?
            """,
            (fixture_key, team_id),
        )
        row = cur.fetchone()
    except Exception:
        # Table doesn't exist yet — fall through to historical.
        return None

    if row is None:
        return None

    # Staleness check.
    try:
        fetched = datetime.fromisoformat(row["fetched_at"].replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - fetched).total_seconds() / 3600
        if age > max_age_hours:
            logger.debug(
                f"crex_xi_cache stale for {fixture_key} team {team_id} "
                f"(age {age:.1f}h > {max_age_hours}h)"
            )
            return None
    except (ValueError, TypeError):
        return None

    # Quality gate.
    n_matched = row["n_matched"] or 0
    if n_matched < CREX_XI_MIN_MATCH:
        logger.debug(
            f"crex_xi_cache below threshold for {fixture_key} team {team_id} "
            f"({n_matched}/{row['n_input']} matched < {CREX_XI_MIN_MATCH})"
        )
        return None

    try:
        player_ids: List[int] = json.loads(row["players_json"])
    except (json.JSONDecodeError, TypeError):
        return None

    if not player_ids:
        return None

    batters = player_ids[:11]
    # Use all available as potential bowlers — the sim selects bowling ELOs
    # from this list; first 5 with non-zero overs are preferred but we pass
    # the full resolved list so the sim can pick.
    bowlers = player_ids[:5]
    return batters, bowlers
