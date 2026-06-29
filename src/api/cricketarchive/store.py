"""Isolated CricketArchive datastore (ca_archive.db).

DELIBERATELY separate from cricket.db. This lets us harvest, audit and run
experiments (aggregate features, synthetic-delivery validation, source-quality
comparison) on CricketArchive data without ever touching the trusted production
database. The delivery table mirrors cricket.db's `deliveries` shape so the two
sources can be compared and features built with minimal glue.

Nothing here imports or writes cricket.db.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

from .models import CAScorecard, CADelivery

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS ca_meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS ca_players (
    ca_id TEXT PRIMARY KEY,
    name TEXT,
    full_name TEXT,
    born_date TEXT,
    born_place TEXT,
    batting_style TEXT,
    bowling_styles TEXT,
    playing_role TEXT,
    profile_fetched INTEGER DEFAULT 0,
    first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ca_teams (ca_team_id TEXT PRIMARY KEY, name TEXT);
CREATE TABLE IF NOT EXISTS ca_grounds (ca_ground_id TEXT PRIMARY KEY, name TEXT);

CREATE TABLE IF NOT EXISTS ca_matches (
    scorecard_id TEXT PRIMARY KEY,
    alt_ids TEXT,
    title TEXT,
    competition TEXT,
    competition_url TEXT,
    match_label TEXT,
    match_date TEXT,
    overs_per_innings INTEGER,
    balls_per_over INTEGER,
    ground_ca_id TEXT,
    ground_name TEXT,
    team1_ca_id TEXT, team1_name TEXT,
    team2_ca_id TEXT, team2_name TEXT,
    toss TEXT,
    result TEXT,
    player_of_match_ca_id TEXT,
    has_ball_by_ball INTEGER DEFAULT 0,
    source_url TEXT,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ca_innings (
    scorecard_id TEXT, innings_number INTEGER, batting_team TEXT,
    total_runs INTEGER, total_wickets INTEGER, extras_text TEXT, fall_of_wickets TEXT,
    PRIMARY KEY (scorecard_id, innings_number)
);

CREATE TABLE IF NOT EXISTS ca_batting (
    scorecard_id TEXT, innings_number INTEGER, position INTEGER,
    batter_ca_id TEXT, name TEXT, dismissal TEXT,
    bowler_ca_id TEXT, fielder_ca_id TEXT,
    runs INTEGER, balls INTEGER, mins INTEGER,
    fours INTEGER, sixes INTEGER, dots INTEGER, strike_rate REAL,
    PRIMARY KEY (scorecard_id, innings_number, position)
);

CREATE TABLE IF NOT EXISTS ca_bowling (
    scorecard_id TEXT, innings_number INTEGER, position INTEGER,
    bowler_ca_id TEXT, name TEXT, overs TEXT, maidens INTEGER, runs INTEGER,
    wickets INTEGER, wides INTEGER, noballs INTEGER, dots INTEGER,
    fours INTEGER, sixes INTEGER, econ REAL,
    PRIMARY KEY (scorecard_id, innings_number, position)
);

CREATE TABLE IF NOT EXISTS ca_deliveries (
    scorecard_id TEXT, innings_number INTEGER, seq INTEGER,
    over_number INTEGER, ball_number INTEGER,
    batter TEXT, bowler TEXT, non_striker TEXT,
    batter_ca_id TEXT, bowler_ca_id TEXT,
    runs_batter INTEGER, runs_extras INTEGER, runs_total INTEGER,
    extras_wides INTEGER, extras_noballs INTEGER, extras_byes INTEGER, extras_legbyes INTEGER,
    is_wicket INTEGER, wicket_type TEXT, dismissed_player TEXT, fielder1 TEXT,
    is_boundary_four INTEGER, is_boundary_six INTEGER, commentary_raw TEXT,
    PRIMARY KEY (scorecard_id, innings_number, seq)
);

CREATE TABLE IF NOT EXISTS ca_match_officials (
    scorecard_id TEXT, role TEXT, ca_id TEXT, name TEXT
);

CREATE INDEX IF NOT EXISTS idx_ca_deliv_batter ON ca_deliveries(batter_ca_id);
CREATE INDEX IF NOT EXISTS idx_ca_deliv_bowler ON ca_deliveries(bowler_ca_id);
CREATE INDEX IF NOT EXISTS idx_ca_batting_player ON ca_batting(batter_ca_id);
CREATE INDEX IF NOT EXISTS idx_ca_bowling_player ON ca_bowling(bowler_ca_id);
CREATE INDEX IF NOT EXISTS idx_ca_matches_date ON ca_matches(match_date);
"""


@contextmanager
def get_connection(db_path: Optional[Path] = None):
    if db_path is None:
        from config import CRICKETARCHIVE_CONFIG
        db_path = CRICKETARCHIVE_CONFIG["archive_db_path"]
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


# ---------------------------------------------------------------------------
# delivery <-> scorecard player resolution
# ---------------------------------------------------------------------------
def _name_to_caid(entries) -> Dict[str, str]:
    """surname/name -> ca_id, dropping ambiguous surnames (e.g. two Tectors)."""
    by_full, by_sur = {}, {}
    counts: Dict[str, int] = {}
    for e in entries:
        if not e.ca_id:
            continue
        by_full[e.name] = e.ca_id
        sur = e.name.split()[-1]
        counts[sur] = counts.get(sur, 0) + 1
        by_sur[sur] = e.ca_id
    # remove ambiguous surnames
    for sur, c in counts.items():
        if c > 1:
            by_sur.pop(sur, None)
    by_full.update(by_sur)
    return by_full


def _resolve(name: Optional[str], name_map: Dict[str, str]) -> Optional[str]:
    if not name:
        return None
    if name in name_map:
        return name_map[name]
    return name_map.get(name.split()[-1])


# ---------------------------------------------------------------------------
# upserts
# ---------------------------------------------------------------------------
def upsert_players_seen(conn, players_seen: Dict[str, str]) -> None:
    for ca_id, name in players_seen.items():
        conn.execute(
            "INSERT INTO ca_players (ca_id, name) VALUES (?, ?) "
            "ON CONFLICT(ca_id) DO UPDATE SET name=COALESCE(ca_players.name, excluded.name)",
            (ca_id, name),
        )


def update_player_bio(conn, player) -> None:
    conn.execute(
        """
        INSERT INTO ca_players (ca_id, name, full_name, born_date, born_place,
                                batting_style, bowling_styles, playing_role, profile_fetched)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
        ON CONFLICT(ca_id) DO UPDATE SET
            full_name=excluded.full_name, born_date=excluded.born_date,
            born_place=excluded.born_place, batting_style=excluded.batting_style,
            bowling_styles=excluded.bowling_styles, playing_role=excluded.playing_role,
            profile_fetched=1
        """,
        (player.ca_id, player.display_name or player.full_name, player.full_name,
         player.born_date, player.born_place, player.batting_style,
         json.dumps(player.bowling_styles) if player.bowling_styles else None,
         player.playing_role),
    )


def _delete_match(conn, sid: str) -> None:
    for tbl in ("ca_matches", "ca_innings", "ca_batting", "ca_bowling",
                "ca_deliveries", "ca_match_officials"):
        conn.execute(f"DELETE FROM {tbl} WHERE scorecard_id = ?", (sid,))


def write_scorecard(conn, sc: CAScorecard,
                    deliveries_by_innings: Optional[Dict[int, List[CADelivery]]] = None) -> None:
    """Idempotently write a parsed scorecard (+ optional ball-by-ball) to the store."""
    deliveries_by_innings = deliveries_by_innings or {}
    sid = sc.ca_id
    _delete_match(conn, sid)

    upsert_players_seen(conn, sc.players_seen)
    for tid, tname in sc.teams:
        conn.execute("INSERT OR REPLACE INTO ca_teams (ca_team_id, name) VALUES (?, ?)", (tid, tname))
    if sc.ground_ca_id:
        conn.execute("INSERT OR REPLACE INTO ca_grounds (ca_ground_id, name) VALUES (?, ?)",
                     (sc.ground_ca_id, sc.ground))

    t1 = sc.teams[0] if len(sc.teams) > 0 else (None, None)
    t2 = sc.teams[1] if len(sc.teams) > 1 else (None, None)
    conn.execute(
        """
        INSERT INTO ca_matches (scorecard_id, alt_ids, title, competition, competition_url,
            match_label, match_date, overs_per_innings, balls_per_over, ground_ca_id, ground_name,
            team1_ca_id, team1_name, team2_ca_id, team2_name, toss, result,
            player_of_match_ca_id, has_ball_by_ball, source_url)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (sid, json.dumps(sc.alt_ids), sc.title, sc.competition, sc.competition_url,
         sc.match_label, sc.match_date, sc.overs_per_innings, sc.balls_per_over,
         sc.ground_ca_id, sc.ground, t1[1], t1[0], t2[1], t2[0], sc.toss, sc.result,
         sc.player_of_match_ca_id, 1 if deliveries_by_innings else 0, sc.url),
    )

    for role, people in sc.officials.items():
        for nm, pid in people:
            conn.execute(
                "INSERT INTO ca_match_officials (scorecard_id, role, ca_id, name) VALUES (?,?,?,?)",
                (sid, role, pid, nm))

    for i, inn in enumerate(sc.innings, start=1):
        conn.execute(
            """INSERT INTO ca_innings (scorecard_id, innings_number, batting_team,
               total_runs, total_wickets, extras_text, fall_of_wickets) VALUES (?,?,?,?,?,?,?)""",
            (sid, i, inn.batting_team, inn.total_runs, inn.total_wickets,
             inn.extras_text, inn.fall_of_wickets))
        for pos, b in enumerate(inn.batting, start=1):
            conn.execute(
                """INSERT INTO ca_batting (scorecard_id, innings_number, position, batter_ca_id,
                   name, dismissal, bowler_ca_id, fielder_ca_id, runs, balls, mins, fours, sixes,
                   dots, strike_rate) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (sid, i, pos, b.ca_id, b.name, b.dismissal, b.bowler_ca_id, b.fielder_ca_id,
                 b.runs, b.balls, b.mins, b.fours, b.sixes, b.dots, b.strike_rate))
        for pos, bw in enumerate(inn.bowling, start=1):
            conn.execute(
                """INSERT INTO ca_bowling (scorecard_id, innings_number, position, bowler_ca_id,
                   name, overs, maidens, runs, wickets, wides, noballs, dots, fours, sixes, econ)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (sid, i, pos, bw.ca_id, bw.name, bw.overs, bw.maidens, bw.runs, bw.wickets,
                 bw.wides, bw.noballs, bw.dots, bw.fours, bw.sixes, bw.econ))

        # deliveries: resolve striker/bowler names -> ca_ids using this match's cards
        deliveries = deliveries_by_innings.get(i, [])
        bat_map = _name_to_caid(inn.batting)
        bowl_map = _name_to_caid(inn.bowling)
        for seq, d in enumerate(deliveries, start=1):
            conn.execute(
                """INSERT INTO ca_deliveries (scorecard_id, innings_number, seq, over_number,
                   ball_number, batter, bowler, non_striker, batter_ca_id, bowler_ca_id,
                   runs_batter, runs_extras, runs_total, extras_wides, extras_noballs, extras_byes,
                   extras_legbyes, is_wicket, wicket_type, dismissed_player, fielder1,
                   is_boundary_four, is_boundary_six, commentary_raw)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (sid, i, seq, d.over_number, d.ball_number, d.batter, d.bowler, d.non_striker,
                 _resolve(d.batter, bat_map), _resolve(d.bowler, bowl_map),
                 d.runs_batter, d.runs_extras, d.runs_total, d.extras_wides, d.extras_noballs,
                 d.extras_byes, d.extras_legbyes, int(d.is_wicket), d.wicket_type,
                 d.dismissed_player, d.fielder1, int(d.is_boundary_four), int(d.is_boundary_six),
                 d.commentary_raw))
