"""ESPN squad snapshots and the squad-aware pre-toss XI."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.integrations.polymarket import espn_squads as es
from src.integrations.polymarket import paper_inputs
from src.models import xi_selector as xs


def _summary_upcoming():
    return {"squads": [
        {"team": {"displayName": "Hobart"}, "athletes": [{"athlete": {"id": str(i)}} for i in range(1, 16)]},
        {"team": {"displayName": "Adelaide"}, "athletes": []},
    ], "rosters": [{"team": {"displayName": "Hobart"}, "roster": []}]}


def test_parse_squads_keeps_only_announced_sides():
    assert es.parse_squads(_summary_upcoming()) == {"Hobart": [str(i) for i in range(1, 16)]}


def test_parse_played_xi_needs_exactly_eleven_starters():
    ro = [{"athlete": {"id": str(i)}, "starter": True} for i in range(11)]
    s = {"rosters": [{"team": {"displayName": "A"}, "roster": ro},
                     {"team": {"displayName": "B"}, "roster": ro + [{"athlete": {"id": "x"}, "starter": True}]}]}
    assert list(es.parse_played_xi(s)) == ["A"]


def test_pre_start_squads_ignores_snapshots_after_the_start():
    rec = {"start": "2026-11-10T08:00:00Z", "snapshots": [
        {"fetched_at": "2026-11-09T08:00:00Z", "squads": {"A": ["1"]}},
        {"fetched_at": "2026-11-10T05:00:00Z", "squads": {"A": ["1", "2"]}},
        {"fetched_at": "2026-11-10T09:00:00Z", "squads": {"A": ["9"]}},
    ]}
    assert es.pre_start_squads(rec) == {"A": ["1", "2"]}


def test_squad_for_team_matches_by_player_overlap(tmp_path):
    idm = {str(i): 100 + i for i in range(1, 41)}
    es.save_event({"espn_id": 1, "gender": "female", "fmt": "T20", "start": "2026-11-10T08:00:00Z",
                   "snapshots": [{"fetched_at": "2026-11-09T00:00:00Z", "squads": {
                       "A": [str(i) for i in range(1, 15)], "B": [str(i) for i in range(21, 35)]}}]}, tmp_path)
    day = datetime(2026, 11, 10, 8, tzinfo=timezone.utc)
    known_b = set(range(121, 130))
    assert es.squad_for_team(known_b, "female", "T20", day, idm, tmp_path) == list(range(121, 135))
    assert es.squad_for_team(known_b, "male", "T20", day, idm, tmp_path) is None
    assert es.squad_for_team({101, 102}, "female", "T20", day, idm, tmp_path) is None


class _Model:
    def predict_proba(self, X):
        p = np.clip(X["n10"].to_numpy() / 10 + 0.01, 0, 1)
        return np.c_[1 - p, p]


def _hist():
    rows = [{"match_id": k, "team_id": 7, "date": f"2026-06-{k:02d}", "created_at": f"2026-06-{k:02d}",
             "fmt": "T20", "gender": "male", "xi": list(range(1, 12))} for k in range(1, 6)]
    s = pd.DataFrame(rows)
    s["d"] = pd.to_datetime(s.date)
    s["created"] = s.d
    return xs.History(s)


def test_predict_with_squad_drops_absentees_and_adds_newcomers():
    sel = xs.XISelector(_Model(), {"T20_male": True}, "2026-06-05")
    squad = list(range(2, 12)) + [50, 51]
    xi, info = sel.predict_with_squad(_hist(), 7, "male", "T20", pd.Timestamp("2026-06-10"), squad)
    assert 1 not in xi and len(xi) == 11 and set(range(2, 12)) <= set(xi)
    assert info["squad_scored"] == 12
    xi, _ = sel.predict_with_squad(_hist(), 7, "male", "T20", pd.Timestamp("2026-06-10"), list(range(2, 8)))
    assert xi == []


def _stub_runtime(monkeypatch, gate):
    class Sel:
        def allowed(self, f, g):
            return True

        def predict(self, *a):
            return list(range(101, 112)), {"stale_days": 3, "expected_recall": 0.8}

        def predict_with_squad(self, *a):
            return list(range(201, 212)), {"stale_days": 3, "expected_recall": 0.8}

    class Hist:
        def known_players(self, *a):
            return set()

    monkeypatch.setattr(xs, "load_selector", lambda: Sel())
    monkeypatch.setattr(xs, "load_runtime_history", lambda db: Hist())
    monkeypatch.setattr(xs, "squad_gate", lambda: gate)
    monkeypatch.setattr(paper_inputs, "_db_file", lambda conn: "x.db")
    monkeypatch.setattr(paper_inputs, "select_bowlers_from_xi", lambda conn, xi, f, g: xi[-5:])
    monkeypatch.setattr(es, "cricinfo_player_map", lambda db: {})
    monkeypatch.setattr(es, "squad_for_team", lambda *a, **k: list(range(201, 216)))
    monkeypatch.delenv("PAPER_PRETOSS_XI", raising=False)
    monkeypatch.delenv("PAPER_XI_SQUADS", raising=False)


def test_squads_used_only_where_squad_gate_passed(monkeypatch):
    _stub_runtime(monkeypatch, {})
    assert paper_inputs.get_pretoss_xi(None, 7, "T20", "male", "2026-11-10")[0] == list(range(101, 112))
    _stub_runtime(monkeypatch, {"T20_male": True})
    bat, _, predicted = paper_inputs.get_pretoss_xi(None, 7, "T20", "male", "2026-11-10")
    assert predicted and bat == list(range(201, 212))
    monkeypatch.setenv("PAPER_XI_SQUADS", "off")
    assert paper_inputs.get_pretoss_xi(None, 7, "T20", "male", "2026-11-10")[0] == list(range(101, 112))


def test_squad_gate_missing_file_means_off(tmp_path):
    assert xs.squad_gate(tmp_path) == {}
    (tmp_path / "squad_gate.json").write_text('{"gate": {"T20_male": true, "ODI_male": false}}')
    assert xs.squad_gate(tmp_path) == {"T20_male": True, "ODI_male": False}


def _xi(ids, starter=True):
    return [{"athlete": {"id": str(i)}, "starter": starter} for i in ids]


def _summary(sides, overs=None, squads=None, live=False, note=None):
    competitors = []
    for name in sides:
        lines = [] if overs is None else [{"overs": overs, "period": 1, "runs": 1, "wickets": 0}]
        competitors.append({"team": {"displayName": name}, "linescores": lines})
    summary = {
        "header": {"competitions": [{
            "status": {"type": {"state": "in" if live else "pre", "description": "Live" if live else "Pre"}},
            "competitors": competitors,
        }]},
        "rosters": [{"team": {"displayName": name}, "roster": players} for name, players in sides.items()],
        "squads": squads or [],
    }
    if note:
        summary["notes"] = [{"text": note}]
    return summary


def test_confirmed_rosters_require_eleven_and_no_ball():
    a, b = _xi(range(1, 12)), _xi(range(21, 32))
    sides = {"Adelaide Strikers Women": a, "Perth Scorchers Women": b}
    got = es.confirmed_rosters(_summary(sides, live=True))
    assert got["Adelaide Strikers Women"] == [str(i) for i in range(1, 12)]
    assert got["Perth Scorchers Women"] == [str(i) for i in range(21, 32)]
    # A non-starter still counts: the list length is the XI, not the flag.
    mixed = _xi(range(1, 11)) + [{"athlete": {"id": "11"}, "starter": False}]
    got = es.confirmed_rosters(_summary({"A": mixed, "B": b}))
    assert got["A"][-1] == "11"
    squad = [{"team": {"displayName": "A"}, "athletes": _xi(range(1, 16))}]
    assert es.confirmed_rosters(_summary(
        {"A": _xi(range(1, 16)), "B": _xi(range(1, 16))},
        squads=squad, note="A won the toss and elected to bat",
    )) is None
    assert es.confirmed_rosters(_summary({"A": _xi(range(1, 19)), "B": _xi(range(21, 39))})) is None
    assert es.confirmed_rosters(_summary({"A": a, "B": _xi(range(1, 16))})) is None
    assert es.confirmed_rosters(_summary({"A": [], "B": []})) is None
    assert es.confirmed_rosters(_summary(sides, overs=0.3)) is None
    assert es.confirmed_rosters(_summary(sides, overs=2.4, live=True)) is None


def test_match_full_lineup_prefers_crex_and_tags_espn_separately():
    crex_t1, crex_t2 = list(range(11)), list(range(20, 31))
    espn_t1, espn_t2 = list(range(201, 212)), list(range(301, 312))
    got = paper_inputs.match_full_lineup(True, crex_t1, crex_t2, espn_t1, espn_t2)
    assert got == (crex_t1, crex_t2, "crex_confirmed")
    got = paper_inputs.match_full_lineup(False, crex_t1, crex_t2, espn_t1, espn_t2)
    assert got == (espn_t1, espn_t2, "espn_confirmed")
    assert got[2] not in ("crex_confirmed", "predicted_xi", "predicted_fallback")
    assert paper_inputs.match_full_lineup(False, None, None, espn_t1, None) is None
    assert paper_inputs.match_full_espn_ok("match_full", got)
    assert not paper_inputs.match_full_espn_ok("match_full", None)
    assert not paper_inputs.match_full_espn_ok("v3", got)
    crex = paper_inputs.match_full_lineup(True, crex_t1, crex_t2, None, None)
    assert not paper_inputs.match_full_espn_ok("match_full", crex)


def test_espn_confirmed_sides_reads_one_summary_for_a_saved_event(monkeypatch, tmp_path):
    es.save_event({
        "espn_id": "1543965", "league_id": "21284", "league": "Women's Big Bash League",
        "teams": ["Perth Scorchers Women", "Adelaide Strikers Women"],
        "class": "Women's T20", "fmt": "T20", "gender": "female",
        "start": "2026-11-14T08:10:00Z", "snapshots": [{
            "fetched_at": "2026-11-14T06:00:00Z",
            "squads": {"Perth Scorchers Women": [str(i) for i in range(1, 16)],
                       "Adelaide Strikers Women": [str(i) for i in range(21, 36)]},
        }],
    }, tmp_path)
    calls = []

    def fetch(league_id, espn_id, session=None, retries=2):
        calls.append((league_id, espn_id, retries))
        return _summary({
            "Perth Scorchers Women": _xi(range(1, 12)),
            "Adelaide Strikers Women": _xi(range(21, 32)),
        }, live=True, squads=[{"team": {"displayName": "Perth Scorchers Women"},
                               "athletes": _xi(range(1, 16))}])

    monkeypatch.setattr(es, "fetch_summary", fetch)
    monkeypatch.setattr(es, "refresh_register", lambda *a, **k: (_ for _ in ()).throw(AssertionError("register fetch")))
    monkeypatch.setattr(xs, "load_selector", lambda *a, **k: (_ for _ in ()).throw(AssertionError("selector")))
    monkeypatch.setattr(paper_inputs, "_db_file", lambda conn: "x.db")
    monkeypatch.setattr(es, "cricinfo_player_map", lambda db: {str(i): 1000 + i for i in range(1, 40)})
    monkeypatch.setenv("PAPER_PRETOSS_XI", "last_match")
    kickoff = datetime(2026, 11, 14, 8, 10, tzinfo=timezone.utc)
    t1, t2 = paper_inputs.espn_confirmed_sides(
        None, "Perth Scorchers Women", "Adelaide Strikers Women", "female", "T20", kickoff, tmp_path,
    )
    assert calls == [("21284", "1543965", 1)]
    assert t1 == list(range(1001, 1012)) and t2 == list(range(1021, 1032))

    calls.clear()
    monkeypatch.setattr(es, "fetch_summary", lambda *a, **k: _summary({
        "Perth Scorchers Women": _xi(range(1, 12)),
        "Adelaide Strikers Women": _xi(range(21, 32)),
    }, overs=0.3))
    assert paper_inputs.espn_confirmed_sides(
        None, "Perth Scorchers Women", "Adelaide Strikers Women", "female", "T20", kickoff, tmp_path,
    ) is None


def test_espn_confirmed_sides_does_not_fetch_without_a_saved_event(monkeypatch, tmp_path):
    def boom(*_a, **_k):
        raise AssertionError("live fetch")

    monkeypatch.setattr(es, "fetch_summary", boom)
    monkeypatch.setattr(es, "refresh_register", boom)
    assert paper_inputs.espn_confirmed_sides(
        None, "India", "Australia", "male", "T20",
        datetime(2026, 11, 14, 8, tzinfo=timezone.utc), tmp_path,
    ) is None
