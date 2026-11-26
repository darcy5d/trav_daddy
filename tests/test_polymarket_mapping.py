"""Polymarket fixture -> DB mapping: gender, age group, market kind, team match."""

from __future__ import annotations

import sqlite3

import pytest

from src.integrations.polymarket.upcoming import (
    TOURNAMENT_PREFIX_MAP,
    _classify_market_within_event,
    _infer_format_gender_from_title,
    infer_age_group,
    map_team_label_to_db,
)


@pytest.mark.parametrize("title,gender", [
    ("ACC Men's U19 Asia Cup: India Under-19s vs Pakistan Under-19s", "male"),
    ("Australia U19 Tour of India Youth ODIs: India Under-19s vs Australia Under-19s", "male"),
    ("ICC Women's Emerging Nations Trophy T20s: Scotland Women vs Sri Lanka Women", "female"),
    ("T20 Hong Kong Tri-Series, Women: Hong Kong, China vs Malaysia", "female"),
    ("Test Match: Pakistan vs Sri Lanka", "male"),
])
def test_crint_gender_from_title(title, gender):
    assert _infer_format_gender_from_title(title, "crint")[1] == gender


def test_prefix_map_has_no_stale_taxonomy_and_knows_wpl():
    for stale in ("cricwpl", "cricint", "cricodi", "cricwt", "cricwt20", "cricwodi",
                  "cricbifa", "crict20wblast"):
        assert stale not in TOURNAMENT_PREFIX_MAP
    assert TOURNAMENT_PREFIX_MAP["crwpl20"][1] == "female"
    assert TOURNAMENT_PREFIX_MAP["cricranji"][0] == "TEST"


@pytest.mark.parametrize("title,labels,age", [
    ("ACC Men's U19 Asia Cup: India Under-19s vs Nepal Under-19s", ("India Under-19s", "Nepal Under-19s"), "u19"),
    ("A Team Test: New Zealand A vs India A", ("New Zealand A", "India A"), "a_team"),
    ("England Lions Tour of West Indies Unofficial ODIs: West Indies A vs England Lions", ("West Indies A", "England Lions"), "a_team"),
    ("Top End T20 Series: Nepal vs Victoria Academy", ("Nepal", "Victoria Academy"), "a_team"),
    ("T20 Series South Africa, Women: Dolphins vs Lions", ("Dolphins", "Lions"), "senior"),
    ("Nepal Premier League: Kathmandu Gurkhas vs Lumbini Lions", ("Kathmandu Gurkhas", "Lumbini Lions"), "senior"),
])
def test_age_group(title, labels, age):
    assert infer_age_group(title, *labels) == age


@pytest.mark.parametrize("smt,outcomes,kind", [
    ("moneyline", ["UAE Bulls", "Yas Lions"], "moneyline"),
    ("moneyline", ["Yes", "No"], "moneyline_3way_leg"),
    ("cricket_toss_winner", ["UAE Bulls", "Yas Lions"], "toss_winner"),
    ("cricket_completed_match", ["Yes", "No"], "completed_match"),
    ("cricket_match_to_go_till", ["Yes", "No"], "match_to_go_till"),
])
def test_sports_market_type_wins_over_question_text(smt, outcomes, kind):
    q = "Abu Dhabi T10 League: UAE Bulls vs Yas Lions - Who wins the toss?"
    assert _classify_market_within_event(q, outcomes, "moneyline", smt) == kind


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.executescript(
        """
        CREATE TABLE teams (team_id INTEGER PRIMARY KEY, name TEXT, team_type TEXT,
                            franchise_id INTEGER, canonical_team_id INTEGER);
        CREATE TABLE matches (match_id INTEGER PRIMARY KEY, team1_id INT, team2_id INT,
                              gender TEXT, match_type TEXT, date TEXT);
        CREATE TABLE team_external_ids (group_id INT, source TEXT, external_id TEXT,
                                        external_name TEXT, confidence REAL, last_validated TEXT);
        INSERT INTO teams VALUES (109,'Hong Kong','domestic',109,109),(275,'China','domestic',275,275),
                                 (361,'Barisal Bulls','domestic',361,361),(23,'Jammu & Kashmir','domestic',23,23),
                                 (58,'Dolphins','domestic',58,58),(8,'India','international',8,8);
        INSERT INTO matches VALUES (1,109,275,'male','T20','2026-01-01'),(2,109,275,'female','T20','2026-01-01'),
                                   (3,361,8,'male','T20','2026-01-01'),(4,23,8,'male','T20','2026-01-01'),
                                   (5,58,8,'male','T20','2026-01-01'),(6,8,275,'female','T20','2026-01-01');
        INSERT INTO team_external_ids VALUES (109,'polymarket','hong kong china','Hong Kong, China',NULL,NULL);
        """
    )
    return c


def _tid(conn, label, gender="male"):
    m = map_team_label_to_db(label, "T20", gender, conn)
    return m and m["team_id"]


def test_alias_beats_longest_token(conn):
    assert _tid(conn, "Hong Kong, China") == 109
    assert _tid(conn, "Hong Kong, China Women", "female") == 109


def test_wrong_fuzzy_matches_fail_closed(conn):
    assert _tid(conn, "UAE Bulls") is None


def test_punctuation_variant_matches(conn):
    assert _tid(conn, "Jammu And Kashmir") == 23


def test_gender_without_history_fails_closed(conn):
    assert _tid(conn, "Dolphins", "male") == 58
    assert _tid(conn, "Dolphins", "female") is None
    assert _tid(conn, "India Women", "female") == 8
