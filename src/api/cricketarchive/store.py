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
    -- Match classification (see store.classify_competition / store.infer_gender).
    -- match_category: 'premium_franchise' | 't20_league' | 'international' | 'domestic'
    -- match_gender:   'men' | 'women'
    -- Both are set at ingest and can be bulk-updated via store.reclassify_all().
    match_category TEXT,
    match_gender TEXT,
    source_url TEXT,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ca_innings (
    scorecard_id TEXT, innings_number INTEGER, batting_team TEXT,
    total_runs INTEGER, total_wickets INTEGER, extras_text TEXT, fall_of_wickets TEXT,
    identity_verified INTEGER, identity_report TEXT,
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
    batter TEXT, batter_ca_id TEXT, batter_initials TEXT,
    non_striker TEXT, non_striker_ca_id TEXT, non_striker_initials TEXT,
    bowler TEXT, bowler_ca_id TEXT, bowler_initials TEXT,
    runs_batter INTEGER, runs_extras INTEGER, runs_total INTEGER,
    extras_wides INTEGER, extras_noballs INTEGER, extras_byes INTEGER, extras_legbyes INTEGER,
    is_wicket INTEGER, wicket_type TEXT,
    dismissed_player TEXT, dismissed_player_ca_id TEXT, dismissed_player_initials TEXT,
    fielder1 TEXT, is_boundary_four INTEGER, is_boundary_six INTEGER,
    resolution_status TEXT, commentary_raw TEXT,
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


# ---------------------------------------------------------------------------
# Match classification
# ---------------------------------------------------------------------------
# Four categories designed around betting-market reality:
#
#   premium_franchise  — city-based franchise leagues with player auctions/drafts,
#                        international stars, and high betting liquidity
#                        (IPL, BBL, PSL, CPL, SA20, Hundred, ILT20, MLC, …)
#
#   t20_league         — commercially branded domestic T20 competitions where
#                        traditional county/state teams play in T20 format
#                        (T20 Blast, Super Smash, Charlotte Edwards Cup, Kia SL, …)
#                        Betting markets exist but team identities are stable domestic ones.
#
#   international      — national representative cricket (bilateral series, World Cups,
#                        ICC events, multi-nation tournaments)
#
#   domestic           — non-commercial provincial/state/club competitions
#                        (women's county cups, CSA provincial, National T20 Cup, …)
#
# Order matters: check most-specific first.

_PREMIUM_FRANCHISE = [
    "Indian Premier League",
    "Big Bash League", "KFC Twenty20 Big Bash", "Women's Big Bash League",
    "Weber Women's Big Bash", "Rebel Women's Big Bash",
    "Pakistan Super League", "HBL Pakistan Super League",
    "Caribbean Premier League", "Women's Caribbean Premier League",
    "SA20",
    "International League T20", "ILT20",
    "Major League Cricket",
    "Global T20 Canada", "GT20 Canada",
    "Lanka Premier League", "Sri Lanka Premier League",
    "Afghanistan Premier League",
    "Bangladesh Premier League",
    "The Hundred", "Men's Hundred", "Women's Hundred",
    "T20 Global League",
    "Mzansi Super League",
    "Women's Premier League",
    "T20 Mumbai",
]

_T20_LEAGUE = [
    # English T20 Blast (all naming eras: 2003–present)
    "Friends Provident T20", "Friends Life t20",
    "NatWest T20 Blast", "Natwest t20 blast",
    "Vitality Blast", "The Vitality Blast", "Men's Vitality Blast",
    "The Blast",
    # NZ domestic T20 (HRV Cup era → Super Smash)
    "Super Smash", "HRV Cup", "HRV Twenty20",
    # Women's domestic T20 leagues (county/state-based but commercial T20 product)
    "Kia Super League",
    "Charlotte Edwards Cup",
    "Vitality Women's Twenty20 Cup", "NatWest Women's Twenty20 Cup",
    "Vitality Women's County T20",
    "Women's Vitality Blast",
    # SA domestic T20 leagues
    "Ram Slam T20", "CSA T20 Challenge", "Hollywoodbets Pro 20",
    # Other domestic-franchise hybrids
    "Syed Mushtaq Ali",
    "Vijay Hazare",  # ODI but just in case
    "Deodhar Trophy",
]

_INTERNATIONAL = [
    "World Twenty20", "T20 World Cup", "ICC T20", "ICC Men's T20", "ICC Women's T20",
    "World Cup",
    "Asia Cup", "Nidahas Trophy",
    " in ",          # "Australia in India", "England in New Zealand" etc.
    " tour of ",     # "Australia tour of India"
    "tri-series", "Tri-Nation", "Tri-Series",
    "African Games", "Asian Games",
]


def classify_competition(competition: str) -> str:
    """Return one of: 'premium_franchise' | 't20_league' | 'international' | 'domestic'."""
    if not competition:
        return "domestic"
    c = competition.lower()
    for pat in _PREMIUM_FRANCHISE:
        if pat.lower() in c:
            return "premium_franchise"
    for pat in _T20_LEAGUE:
        if pat.lower() in c:
            return "t20_league"
    for pat in _INTERNATIONAL:
        if pat.lower() in c:
            return "international"
    return "domestic"


def infer_gender(competition: str) -> str:
    """Return 'women' if competition name clearly indicates a women's event, else 'men'."""
    if not competition:
        return "men"
    c = competition.lower()
    return "women" if "women" in c or "female" in c else "men"


def reclassify_all(conn: sqlite3.Connection) -> dict:
    """Re-classify every row (used after updating the classifier logic). Returns counts."""
    rows = conn.execute(
        "SELECT scorecard_id, competition FROM ca_matches"
    ).fetchall()
    updates = [
        (classify_competition(r["competition"] or ""),
         infer_gender(r["competition"] or ""),
         r["scorecard_id"])
        for r in rows
    ]
    conn.executemany(
        "UPDATE ca_matches SET match_category=?, match_gender=? WHERE scorecard_id=?",
        updates,
    )
    conn.commit()
    counts: dict = {}
    for row in conn.execute(
        "SELECT match_category, match_gender, COUNT(*) as n FROM ca_matches "
        "GROUP BY match_category, match_gender ORDER BY n DESC"
    ).fetchall():
        counts[f"{row[0]}/{row[1]}"] = row[2]
    logger.info("reclassify_all: updated %d rows", len(rows))
    return counts


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
                    deliveries_by_innings: Optional[Dict[int, List[CADelivery]]] = None,
                    reports_by_innings: Optional[Dict[int, dict]] = None) -> None:
    """Idempotently write a parsed scorecard (+ optional resolved ball-by-ball).

    Deliveries are expected to be ALREADY identity-resolved (see
    identity.resolve_innings) so each carries batter/non_striker/bowler ca_id +
    initials + resolution_status. `reports_by_innings` carries the per-innings
    verification report.
    """
    deliveries_by_innings = deliveries_by_innings or {}
    reports_by_innings = reports_by_innings or {}
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
            player_of_match_ca_id, has_ball_by_ball, match_category, match_gender, source_url)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (sid, json.dumps(sc.alt_ids), sc.title, sc.competition, sc.competition_url,
         sc.match_label, sc.match_date, sc.overs_per_innings, sc.balls_per_over,
         sc.ground_ca_id, sc.ground, t1[1], t1[0], t2[1], t2[0], sc.toss, sc.result,
         sc.player_of_match_ca_id, 1 if deliveries_by_innings else 0,
         classify_competition(sc.competition or ""),
         infer_gender(sc.competition or ""), sc.url),
    )

    for role, people in sc.officials.items():
        for nm, pid in people:
            conn.execute(
                "INSERT INTO ca_match_officials (scorecard_id, role, ca_id, name) VALUES (?,?,?,?)",
                (sid, role, pid, nm))

    for i, inn in enumerate(sc.innings, start=1):
        rep = reports_by_innings.get(i)
        conn.execute(
            """INSERT INTO ca_innings (scorecard_id, innings_number, batting_team,
               total_runs, total_wickets, extras_text, fall_of_wickets,
               identity_verified, identity_report) VALUES (?,?,?,?,?,?,?,?,?)""",
            (sid, i, inn.batting_team, inn.total_runs, inn.total_wickets,
             inn.extras_text, inn.fall_of_wickets,
             (1 if rep and rep.get("identity_verified") else 0) if rep else None,
             json.dumps(rep) if rep else None))
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

        # deliveries are already identity-resolved by identity.resolve_innings
        deliveries = deliveries_by_innings.get(i, [])
        for seq, d in enumerate(deliveries, start=1):
            conn.execute(
                """INSERT INTO ca_deliveries (scorecard_id, innings_number, seq, over_number,
                   ball_number, batter, batter_ca_id, batter_initials,
                   non_striker, non_striker_ca_id, non_striker_initials,
                   bowler, bowler_ca_id, bowler_initials,
                   runs_batter, runs_extras, runs_total, extras_wides, extras_noballs, extras_byes,
                   extras_legbyes, is_wicket, wicket_type,
                   dismissed_player, dismissed_player_ca_id, dismissed_player_initials,
                   fielder1, is_boundary_four, is_boundary_six, resolution_status, commentary_raw)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (sid, i, seq, d.over_number, d.ball_number,
                 d.batter, d.batter_ca_id, d.batter_initials,
                 d.non_striker, d.non_striker_ca_id, d.non_striker_initials,
                 d.bowler, d.bowler_ca_id, d.bowler_initials,
                 d.runs_batter, d.runs_extras, d.runs_total, d.extras_wides, d.extras_noballs,
                 d.extras_byes, d.extras_legbyes, int(d.is_wicket), d.wicket_type,
                 d.dismissed_player, d.dismissed_player_ca_id, d.dismissed_player_initials,
                 d.fielder1, int(d.is_boundary_four), int(d.is_boundary_six),
                 d.resolution_status, d.commentary_raw))
