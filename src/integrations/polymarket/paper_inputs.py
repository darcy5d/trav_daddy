"""Wave 5.7: build simulator inputs (XI + venue) for an upcoming fixture.

Extracted from scripts/probe_live_fixtures.py so both the live-probe CLI and
the paper-bet scanner can share the same lineup-derivation logic.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Optional, Tuple


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
