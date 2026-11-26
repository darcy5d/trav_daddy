"""Tests for the Polymarket venue resolver.

The scanner calls the resolver only with PAPER_VENUE_SOURCE=espn. These tests
pin the behaviour that must hold when it does, and in particular pin that the
known-risky same-day-series path stays opt-in and that a network failure never
breaks a scan.
"""

from __future__ import annotations

import sqlite3

import pytest

from src.integrations.polymarket.venue_resolver import (
    MULTI_VENUE_COMPETITIONS,
    competition_venue,
    resolve_fixture_venue,
    venue_from_event_text,
    venue_from_same_day_series,
)


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute("CREATE TABLE venues (venue_id INTEGER PRIMARY KEY, name TEXT)")
    c.execute(
        "CREATE TABLE matches (match_id INTEGER PRIMARY KEY, date TEXT, "
        "event_name TEXT, venue_id INTEGER)"
    )
    c.executemany(
        "INSERT INTO venues (venue_id, name) VALUES (?, ?)",
        [
            (1, "Sheikh Zayed Stadium"),
            (2, "Lord's"),
            (3, "Wankhede Stadium"),
            (4, "Adelaide Oval"),
            (5, "Sydney Cricket Ground"),
            (6, "Melbourne Cricket Ground"),
        ],
    )
    c.executemany(
        "INSERT INTO matches (date, event_name, venue_id) VALUES (?, ?, ?)",
        [
            # single-ground competition: should resolve
            ("2026-11-01", "Abu Dhabi T10 League", 1),
            ("2026-11-02", "Abu Dhabi T10 League", 1),
            ("2026-11-03", "Abu Dhabi T10 League", 1),
            # multi-ground competition: must never be pinned
            ("2026-01-01", "Big Bash League", 4),
            ("2026-01-04", "Big Bash League", 5),
            ("2026-01-05", "Big Bash League", 6),
            # too few matches to pin
            ("2026-06-01", "Minor Trophy", 3),
        ],
    )
    return c


def test_alias_table_keeps_known_multi_venue_competitions_out():
    """Pinning a travelling league to one ground is the failure mode we refuse."""
    for token in ("big bash", "indian premier league", "sa20", "the hundred"):
        assert token in MULTI_VENUE_COMPETITIONS


def test_event_text_resolves_a_named_ground(conn):
    assert venue_from_event_text(conn, "Match at Wankhede Stadium, Mumbai") == 3


def test_event_text_is_case_and_punctuation_insensitive(conn):
    assert venue_from_event_text(conn, "played at LORD'S") == 2
    assert venue_from_event_text(conn, "played at lords") == 2


def test_event_text_returns_none_without_a_ground(conn):
    assert venue_from_event_text(conn, "USA vs India, Cricket T20 World Cup") is None
    assert venue_from_event_text(conn, "") is None


def test_competition_resolves_a_single_ground_league(conn):
    assert competition_venue(conn, "Abu Dhabi T10 League") == 1


def test_competition_refuses_a_multi_ground_league(conn):
    assert competition_venue(conn, "Big Bash League") is None


def test_competition_requires_a_minimum_history(conn):
    assert competition_venue(conn, "Minor Trophy") is None
    assert competition_venue(conn, "Minor Trophy", min_matches=1) == 3


def test_competition_ignores_unknown_titles(conn):
    assert competition_venue(conn, "") is None
    assert competition_venue(conn, None) is None
    assert competition_venue(conn, "Never Played Cup") is None


def test_same_day_series_path_is_available_but_opt_in(conn):
    """74% accurate on 2026 data, so it must not fire unless asked for."""
    assert venue_from_same_day_series(conn, "Big Bash League", "2026-01-01") == 4

    vid, src = resolve_fixture_venue(
        conn, tournament_name="Big Bash League", match_date="2026-01-01"
    )
    assert (vid, src) == (None, "unresolved")

    vid, src = resolve_fixture_venue(
        conn,
        tournament_name="Big Bash League",
        match_date="2026-01-01",
        allow_same_day_series=True,
    )
    assert (vid, src) == (4, "same_day_series")


def test_resolution_order_prefers_event_text(conn):
    vid, src = resolve_fixture_venue(
        conn, event_text="at Sheikh Zayed Stadium", tournament_name="Minor Trophy"
    )
    assert (vid, src) == (1, "event_text")


def test_resolution_order_falls_back_to_competition(conn):
    vid, src = resolve_fixture_venue(conn, tournament_name="Abu Dhabi T10 League")
    assert (vid, src) == (1, "competition")


def test_unresolved_returns_none_and_a_reason(conn):
    vid, src = resolve_fixture_venue(conn, event_text="nothing useful here")
    assert vid is None
    assert src == "unresolved"


def test_resolver_never_raises_on_a_broken_connection():
    """A malformed DB must degrade to 'no opinion', not break a scan."""
    broken = sqlite3.connect(":memory:")  # no tables at all
    vid, src = resolve_fixture_venue(
        broken, event_text="at Lord's", tournament_name="Abu Dhabi T10 League"
    )
    assert vid is None
    assert src == "unresolved"


# ---------------------------------------------------------------------------
# ESPN fixture-list path
# ---------------------------------------------------------------------------

from datetime import datetime, timezone

from src.integrations.polymarket.venue_resolver import venue_from_espn

KICKOFF = datetime(2026, 7, 15, 17, 30, tzinfo=timezone.utc)


def _espn_event(ground, city, gid, t1="Sussex", t2="Somerset"):
    return {"espn_id": "x", "league": "Vitality Blast", "start": "2026-07-15T17:30Z",
            "name": f"{t1} v {t2}", "teams": [t1, t2], "ground_id": gid, "ground": ground,
            "city": city, "country": "England", "class": "Twenty20", "event_type": "T20"}


@pytest.fixture()
def espn_conn():
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE venues (venue_id INTEGER PRIMARY KEY, name TEXT, city TEXT, country TEXT)")
    c.execute("CREATE TABLE matches (match_id INTEGER PRIMARY KEY, date TEXT, venue_id INTEGER)")
    c.executemany("INSERT INTO venues VALUES (?, ?, ?, ?)", [
        (5, "County Ground, Taunton", "Taunton", "England"),
        (6, "County Ground, Hove", "Hove", "England"),
    ])
    return c


def _espn(**kw):
    base = dict(team1_label="Sussex Sharks", team2_label="Somerset", gender="male",
                kickoff=KICKOFF, match_date="2026-07-15", fmt="T20", ground_map={})
    base.update(kw)
    return base


def test_espn_resolves_the_fixture_ground(espn_conn):
    events = [_espn_event("County Ground, Hove", "Hove", "20")]
    assert venue_from_espn(espn_conn, events=events, **_espn()) == (6, "espn")


def test_espn_flags_a_ground_without_history(espn_conn):
    events = [_espn_event("Arundel Castle", "Arundel", "99")]
    assert venue_from_espn(espn_conn, events=events, **_espn()) == (None, "espn_new_ground")


def test_espn_below_min_confidence_is_no_opinion(espn_conn):
    events = [_espn_event("County Ground, Hove", "Hove", "20", t2="Surrey")]
    assert venue_from_espn(espn_conn, events=events, **_espn()) == (None, "unresolved")


def test_espn_runs_before_text_and_competition(espn_conn):
    events = [_espn_event("County Ground, Hove", "Hove", "20")]
    vid, src = resolve_fixture_venue(
        espn_conn, event_text="at County Ground, Taunton", espn=_espn(events=events))
    assert (vid, src) == (6, "espn")


def test_espn_network_failure_degrades_to_other_paths(conn):
    def down(*a, **k):
        raise OSError("down")

    vid, src = resolve_fixture_venue(
        conn, tournament_name="Abu Dhabi T10 League", espn=_espn(fetch=down))
    assert (vid, src) == (1, "competition")
