"""The paper scanner keeps team1's home ground unless PAPER_VENUE_SOURCE=espn."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def scan(monkeypatch):
    spec = importlib.util.spec_from_file_location("paper_bet_scan", ROOT / "scripts" / "paper_bet_scan.py")
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "paper_bet_scan", mod)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "get_default_venue_for_team", lambda conn, t, f, g: 77)
    return mod


FIXTURE = {"team1_id": 1, "team2_id": 2, "format": "T20", "gender": "male",
           "team1_label": "Sussex", "team2_label": "Somerset", "match_date": "2026-07-15",
           "tournament_name": "Vitality Blast"}


def test_default_is_the_fallback_and_skips_the_resolver(scan, monkeypatch):
    monkeypatch.delenv("PAPER_VENUE_SOURCE", raising=False)
    monkeypatch.setattr(scan, "resolve_fixture_venue", lambda *a, **k: pytest.fail("called"))
    assert scan._fixture_venue(None, dict(FIXTURE)) == 77


@pytest.mark.parametrize("resolved, expected", [
    ((6, "espn"), 6),
    ((None, "espn_new_ground"), None),
    ((None, "unresolved"), 77),
    ((3, "competition"), 3),
])
def test_espn_mode(scan, monkeypatch, resolved, expected):
    monkeypatch.setenv("PAPER_VENUE_SOURCE", "espn")
    monkeypatch.setattr(scan, "resolve_fixture_venue", lambda *a, **k: resolved)
    fx = dict(FIXTURE)
    assert scan._fixture_venue(None, fx) == expected
    assert fx["_venue_source"] == resolved[1]
