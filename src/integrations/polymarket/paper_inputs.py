"""Wave 5.7 / 5.11: build simulator inputs (XI + venue) for an upcoming fixture.

Extracted from scripts/probe_live_fixtures.py so both the live-probe CLI and
the paper-bet scanner can share the same lineup-derivation logic.

Wave 5.11 adds:
  - resolve_xi_names_to_ids(): fuzzy-match CREX player name strings to DB ids.
  - get_cached_xi(): read crex_xi_cache and return (batters, bowlers) if fresh.
"""

from __future__ import annotations

import hashlib
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


def compute_xi_signature(
    t1_bat: List[int],
    t1_bowl: List[int],
    t2_bat: List[int],
    t2_bowl: List[int],
) -> str:
    """Stable 12-char hash of the four lineup arrays.

    Shared by the pre-toss live/paper scanners and the post-toss scan
    scripts so the signature comparison is apples-to-apples across phases.
    A change in this value between scans means the lineup the model used
    has changed (e.g. CREX published a confirmed XI, or a late swap).
    """
    payload = json.dumps(
        {
            "t1_bat": list(t1_bat),
            "t1_bowl": list(t1_bowl),
            "t2_bat": list(t2_bat),
            "t2_bowl": list(t2_bowl),
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


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

    Two-pass matching:
      Pass 1 — team-scoped: players who've ever appeared for this team in
               this format (no date cap so rotation/loan players are included).
      Pass 2 — global T20 fallback: any player in the full T20 pool. Used
               for import/overseas players who haven't played for this
               franchise yet but exist in the DB from other competitions.

    The global pass uses a stricter threshold (0.65) to reduce false positives
    when searching across thousands of players.
    """
    from src.utils.name_matcher import match_abbreviated_name

    if not xi_names:
        return [], 0, 0

    cur = conn.cursor()

    # Pass 1: team-scoped (all historical matches, no date cap)
    cur.execute(
        """
        SELECT DISTINCT pms.player_id, p.name
        FROM player_match_stats pms
        JOIN players p ON p.player_id = pms.player_id
        JOIN matches m ON m.match_id = pms.match_id
        WHERE pms.team_id = ?
          AND m.match_type = ? AND m.gender = ?
        """,
        (team_id, fmt, gender),
    )
    team_candidates: List[Tuple[int, str]] = [
        (r["player_id"], r["name"]) for r in cur.fetchall()
    ]

    # Pass 2: global T20 pool (loaded lazily — only if needed)
    _global_candidates: Optional[List[Tuple[int, str]]] = None

    def _get_global() -> List[Tuple[int, str]]:
        nonlocal _global_candidates
        if _global_candidates is None:
            cur.execute(
                """
                SELECT DISTINCT pms.player_id, p.name
                FROM player_match_stats pms
                JOIN players p ON p.player_id = pms.player_id
                JOIN matches m ON m.match_id = pms.match_id
                WHERE m.match_type = ? AND m.gender = ?
                """,
                (fmt, gender),
            )
            _global_candidates = [(r["player_id"], r["name"]) for r in cur.fetchall()]
        return _global_candidates

    ids: List[int] = []
    matched = 0
    matched_ids_set: set = set()

    for raw in xi_names:
        raw = (raw or "").strip()
        if not raw:
            continue

        pid = None

        # Pass 1: team-scoped
        if team_candidates:
            m = match_abbreviated_name(raw, team_candidates, threshold=threshold)
            if m and m[0] not in matched_ids_set:
                pid = m[0]

        # Pass 2: global fallback (stricter threshold to avoid cross-team noise)
        if pid is None:
            m = match_abbreviated_name(raw, _get_global(), threshold=max(threshold, 0.65))
            if m and m[0] not in matched_ids_set:
                pid = m[0]
                logger.debug(f"  resolve_xi: '{raw}' matched globally → {m[1]} (id={m[0]})")

        if pid is not None:
            ids.append(pid)
            matched_ids_set.add(pid)
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

    # The sim hard-requires exactly 11 batters and 5 bowlers. If CREX only
    # resolved < 11 players (partial XI), fall back to get_recent_xi() which
    # always fills to 11 from historical matches.
    if len(player_ids) < 11:
        logger.debug(
            f"crex_xi_cache has only {len(player_ids)} ids for {fixture_key} "
            f"team {team_id} — falling back to historical"
        )
        return None

    batters = player_ids[:11]
    bowlers = player_ids[:5]
    return batters, bowlers
