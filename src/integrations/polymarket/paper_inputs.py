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
import os
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Minimum fraction of XI that must resolve to DB player_ids before we trust
# the CREX lineup over historical fallback.
CREX_XI_MIN_MATCH = 7  # out of 11

# Maximum age of a crex_xi_cache row before we consider it stale.
CREX_XI_MAX_AGE_HOURS = 3.0

# A CREX XI that shares fewer than this many players with the team's
# last-match XI is more likely a scraped squad list than the playing XI.
MIN_OVERLAP_WITH_RECENT_XI = 6

XI_HISTORICAL_PROXY = "historical_proxy"
XI_CREX_CONFIRMED = "crex_confirmed"
XI_HISTORICAL_FALLBACK = "historical_fallback"
# Unconfirmed XIs from src/models/xi_selector (pre-toss, and post-toss when CREX
# is not confirmed).
XI_PREDICTED = "predicted_xi"
XI_PREDICTED_FALLBACK = "predicted_fallback"
# Post-toss match_full only: both ESPN rosters were exactly 11 and no ball
# had been bowled. Not a CREX XI and not a predicted lineup.
XI_ESPN_CONFIRMED = "espn_confirmed"
# Below this the selector's lineup is a guess: on the 2026 holdout, sides it
# rated under 0.5 recalled 0.28-0.45 of the declared XI. Such fixtures are skipped.
MIN_EXPECTED_RECALL = 0.5


def _db_file(conn) -> Optional[str]:
    for row in conn.execute("PRAGMA database_list"):
        if row[1] == "main" and row[2]:
            return row[2]
    return None


def _squads_disabled() -> bool:
    return os.environ.get("PAPER_XI_SQUADS", "auto").strip().lower() == "off"


def _saved_squad(conn, hist, team_id: int, gender: str, fmt: str, day) -> Optional[List[int]]:
    """Snapshotted ESPN squad for this side. Never fetches; None on any problem."""
    try:
        from src.integrations.polymarket import espn_squads as es

        db = _db_file(conn)
        known = hist.known_players(team_id, gender, day.normalize())
        return es.squad_for_team(known, gender, fmt, day.to_pydatetime(), es.cricinfo_player_map(db))
    except Exception as exc:
        logger.warning(f"  [XI] ESPN squad lookup failed for team {team_id}: {exc}")
        return None


def _espn_squad(conn, hist, team_id: int, gender: str, fmt: str, day) -> Optional[List[int]]:
    """Snapshotted ESPN squad for this side, only where the squad gate passed
    and PAPER_XI_SQUADS is not 'off'. Never fetches; None on any problem."""
    from src.models import xi_selector as xs

    if _squads_disabled():
        return None
    if not xs.squad_gate().get(f"{fmt}_{gender}", False):
        return None
    return _saved_squad(conn, hist, team_id, gender, fmt, day)


def match_full_lineup(
    crex_ok: bool,
    crex_t1: Optional[List[int]],
    crex_t2: Optional[List[int]],
    espn_t1: Optional[List[int]],
    espn_t2: Optional[List[int]],
) -> Optional[Tuple[List[int], List[int], str]]:
    """XI pair match_full may bet on, and the xi_status to store.

    A confirmed CREX XI wins and stays crex_confirmed. Otherwise both sides
    need an ESPN roster of exactly 11 seen before a ball, tagged
    espn_confirmed. A squad ranking is not a lineup. None means skip.
    """
    if crex_ok and crex_t1 and crex_t2 and len(crex_t1) >= 11 and len(crex_t2) >= 11:
        return list(crex_t1[:11]), list(crex_t2[:11]), XI_CREX_CONFIRMED
    if espn_t1 and espn_t2 and len(espn_t1) >= 11 and len(espn_t2) >= 11:
        return list(espn_t1[:11]), list(espn_t2[:11]), XI_ESPN_CONFIRMED
    return None


def match_full_espn_ok(model_version: str, lineup) -> bool:
    """True when match_full may bet an ESPN-confirmed XI.

    The caller has already required that the scan-wide XI is not crex_confirmed.
    """
    return (
        model_version == "match_full"
        and lineup is not None
        and lineup[2] == XI_ESPN_CONFIRMED
    )


def espn_confirmed_sides(
    conn,
    team1_label: str,
    team2_label: str,
    gender: str,
    fmt: str,
    kickoff,
    root=None,
) -> Optional[Tuple[List[int], List[int]]]:
    """(team1, team2) player ids from an ESPN summary, or None.

    The event has to be one we already snapshotted. That file identifies the
    match; it does not supply the XI (a saved squad is not an XI). One summary
    read then applies the roster rule: both lists length 11, and no over
    bowled. No snapshot, a failed read, a squad-length roster, a ball already
    bowled, or fewer than 11 mapped ids skips the match. Does not rank a
    squad and does not call the XI selector.
    """
    from src.integrations.polymarket import espn_squads as es
    from src.integrations.polymarket.espn_fixture_venues import match_fixture, parse_utc, team_score

    try:
        events = [ev for ev in es.iter_events(es.SNAPSHOT_DIR if root is None else root)
                  if ev.get("espn_id") and ev.get("league_id")]
        hit = match_fixture(team1_label, team2_label, gender, events, kickoff=parse_utc(kickoff), fmt=fmt)
        if hit.event is None:
            logger.info("  [XI] no saved ESPN event; match_full will not fetch a summary")
            return None
        summary = es.fetch_summary(hit.event["league_id"], hit.event["espn_id"], retries=1)
        if not summary:
            logger.info("  [XI] ESPN summary unavailable; match_full skipped")
            return None
        rosters = es.confirmed_rosters(summary)
        if not rosters:
            logger.info("  [XI] ESPN rosters are not 11 and 11 before a ball; match_full skipped")
            return None
        names = list(rosters)
        fwd = min(team_score(team1_label, names[0]), team_score(team2_label, names[1]))
        rev = min(team_score(team1_label, names[1]), team_score(team2_label, names[0]))
        if fwd >= 0.85 and fwd >= rev:
            raw = (rosters[names[0]], rosters[names[1]])
        elif rev >= 0.85:
            raw = (rosters[names[1]], rosters[names[0]])
        else:
            logger.info("  [XI] ESPN roster names did not match the fixture; match_full skipped")
            return None
        id_map = es.cricinfo_player_map(_db_file(conn))
        mapped = []
        for ids in raw:
            side = [id_map[i] for i in ids if i in id_map]
            if len(side) != 11 or len(set(side)) != 11:
                logger.info("  [XI] ESPN XI did not map to 11 players; match_full skipped")
                return None
            mapped.append(side)
        logger.info("  [XI] ESPN confirmed both XIs before a ball")
        return mapped[0], mapped[1]
    except Exception as exc:
        logger.warning(f"  [XI] ESPN confirmed-XI lookup failed: {exc}")
        return None


def get_pretoss_xi(
    conn,
    team_id: int,
    fmt: str,
    gender: str,
    match_date=None,
) -> Tuple[List[int], List[int], bool]:
    """Unconfirmed XI for a fixture: (batters, bowlers, predicted).

    Uses the XI selector where its holdout gate passed for this format/gender
    and PAPER_PRETOSS_XI is not 'last_match'; otherwise, or if it cannot
    produce 11 players, get_recent_xi. `predicted` says which one was used.
    Where the squad gate also passed, the selector ranks within the snapshotted
    ESPN squad first. Returns ([], [], True) when the selector's expected recall is below
    MIN_EXPECTED_RECALL, so callers skip the fixture.
    """
    from src.models import xi_selector as xs

    if xs.pretoss_mode() != "last_match":
        try:
            sel = xs.load_selector()
            if sel is not None and sel.allowed(fmt, gender):
                hist = xs.load_runtime_history(_db_file(conn))
                if hist is not None:
                    import pandas as pd
                    day = pd.Timestamp(match_date) if match_date is not None else pd.Timestamp.now(tz="UTC")
                    if day.tzinfo is not None:
                        day = day.tz_convert("UTC").tz_localize(None)
                    xi, info = [], {}
                    squad = _espn_squad(conn, hist, team_id, gender, fmt, day)
                    if squad:
                        xi, info = sel.predict_with_squad(hist, team_id, gender, fmt, day.normalize(), squad)
                        if xi:
                            logger.info(f"  [XI] team {team_id}: ranked within ESPN squad of {len(squad)}")
                    if not xi:
                        xi, info = sel.predict(hist, team_id, gender, fmt, day.normalize())
                    min_exp = float(os.environ.get("PAPER_XI_MIN_EXPECTED_RECALL", MIN_EXPECTED_RECALL))
                    if len(xi) == 11 and info.get("expected_recall", 1.0) < min_exp:
                        logger.warning(f"  [XI] team {team_id}: lineup unknown (stale {info.get('stale_days')}d, "
                                       f"expected recall {info['expected_recall']:.2f} < {min_exp}); skipping")
                        return [], [], True
                    if len(xi) == 11:
                        logger.info(f"  [XI] team {team_id} predicted (stale {info.get('stale_days')}d, "
                                    f"expected recall {info.get('expected_recall', 0):.2f})")
                        return xi, select_bowlers_from_xi(conn, xi, fmt, gender), True
        except Exception as exc:
            logger.warning(f"  [XI] selector failed for team {team_id}: {exc}; using last-match XI")
    bat, bowl = get_recent_xi(conn, team_id, fmt, gender)
    return bat, bowl, False


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


def select_bowlers_from_xi(
    conn,
    xi: List[int],
    fmt: str,
    gender: str,
    n_bowlers: int = 5,
) -> List[int]:
    """Pick the XI's main bowlers: most overs per match in this format over
    the last two years, then the lowest batting positions to fill up."""
    if not xi:
        return []
    cur = conn.cursor()
    placeholders = ",".join("?" * len(xi))
    cur.execute(
        f"""
        SELECT pms.player_id, AVG(COALESCE(pms.overs_bowled, 0)) AS avg_overs
        FROM player_match_stats pms
        JOIN matches m ON m.match_id = pms.match_id
        WHERE pms.player_id IN ({placeholders})
          AND m.match_type = ? AND m.gender = ?
          AND m.date >= date('now', '-730 days')
        GROUP BY pms.player_id
        """,
        list(xi) + [fmt, gender],
    )
    avg_overs = {r["player_id"]: r["avg_overs"] or 0.0 for r in cur.fetchall()}
    ranked = sorted(
        (p for p in xi if avg_overs.get(p, 0.0) > 0.5),
        key=lambda p: -avg_overs[p],
    )
    bowlers = ranked[:n_bowlers]
    for p in reversed(xi):
        if len(bowlers) >= n_bowlers:
            break
        if p not in bowlers:
            bowlers.append(p)
    return bowlers[:n_bowlers]


def check_confirmed_xi(
    conn,
    xi: Optional[List[int]],
    team_id: int,
    fmt: str,
    gender: str,
) -> Tuple[bool, str]:
    """Is this post-toss XI safe to use? Returns (ok, reason).

    Requires 11 ids that all have match history in the DB (no stub players)
    and at least MIN_OVERLAP_WITH_RECENT_XI players in common with the team's
    last-match XI.
    """
    if not xi or len(xi) < 11:
        return False, f"unmapped ({len(xi or [])}/11 ids)"
    xi = list(xi)[:11]
    cur = conn.cursor()
    placeholders = ",".join("?" * len(xi))
    cur.execute(
        f"SELECT DISTINCT player_id FROM player_match_stats WHERE player_id IN ({placeholders})",
        xi,
    )
    known = {r["player_id"] for r in cur.fetchall()}
    if len(known) < 11:
        return False, f"unmapped ({11 - len(known)} ids without match history)"
    recent_bat, _ = get_recent_xi(conn, team_id, fmt, gender)
    overlap = len(set(recent_bat) & set(xi))
    if recent_bat and overlap < MIN_OVERLAP_WITH_RECENT_XI:
        return False, f"suspect (only {overlap}/11 in common with last XI)"
    return True, "ok"


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
    fmt: Optional[str] = None,
    gender: Optional[str] = None,
) -> Optional[Tuple[List[int], List[int]]]:
    """Return (batters, bowlers) from crex_xi_cache if the entry is fresh and
    has enough resolved players, otherwise return None.

    The cache stores a flat JSON array of player_ids in batting order (CREX
    playingxi-button lists XI in batting order). The first 11 are the
    batters; bowlers are picked by recent overs when fmt/gender are given,
    otherwise batting positions 7-11.

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
    if fmt and gender:
        bowlers = select_bowlers_from_xi(conn, batters, fmt, gender)
    else:
        bowlers = batters[6:11]
    return batters, bowlers
