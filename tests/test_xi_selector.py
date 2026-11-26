"""Pre-toss XI selector: as-of history, pick order, and the scanner fallback."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.integrations.polymarket import paper_inputs
from src.models import xi_selector as xs


def _sides():
    rows = []
    for k, (d, created) in enumerate([("2026-06-01", "2026-06-02"), ("2026-06-05", "2026-06-06"),
                                      ("2026-06-09", "2026-06-20")]):
        xi = list(range(1, 12)) if k < 2 else list(range(2, 13))
        rows.append({"match_id": k + 1, "team_id": 7, "date": d, "created_at": created, "fmt": "T20",
                     "gender": "male", "xi": xi})
    s = pd.DataFrame(rows)
    s["d"] = pd.to_datetime(s.date)
    s["created"] = pd.to_datetime(s.created_at)
    return s


def test_candidates_use_only_matches_known_by_the_scan_day():
    s = _sides()
    hist = xs.History(s, vis=s.created)
    cands, info = hist.candidates(7, "male", "T20", pd.Timestamp("2026-06-12"))
    assert info == {"n_hist": 2, "stale_days": 7}
    assert 12 not in set(cands.player_id)
    row = cands.set_index("player_id").loc[1]
    assert (row.in1, row.in2, row.n10) == (1, 1, 2)

    cands, info = hist.candidates(7, "male", "T20", pd.Timestamp("2026-06-21"))
    assert info["n_hist"] == 3
    assert cands.set_index("player_id").loc[1].in1 == 0


def test_no_history_gives_no_candidates():
    s = _sides()
    cands, info = xs.History(s).candidates(7, "female", "T20", pd.Timestamp("2026-06-12"))
    assert cands.empty and info["n_hist"] == 0


def test_pick_xi_takes_top_probabilities_in_batting_order():
    c = pd.DataFrame({"player_id": range(1, 14), "mean_pos": [13 - i for i in range(13)]})
    probs = np.array([0.9] * 11 + [0.1, 0.2])
    xi = xs.pick_xi(c, probs)
    assert set(xi) == set(range(1, 12))
    assert xi == sorted(xi, reverse=True)


class _Sel:
    def __init__(self, allowed):
        self._allowed = allowed

    def allowed(self, fmt, gender):
        return self._allowed

    def predict(self, hist, team_id, gender, fmt, d):
        return list(range(101, 112)), {"stale_days": 3, "expected_recall": 0.8}


@pytest.fixture()
def stubs(monkeypatch):
    monkeypatch.setattr(paper_inputs, "get_recent_xi", lambda conn, t, f, g: (list(range(1, 12)), [1, 2, 3, 4, 5]))
    monkeypatch.setattr(paper_inputs, "select_bowlers_from_xi", lambda conn, xi, f, g: xi[-5:])
    monkeypatch.setattr(paper_inputs, "_db_file", lambda conn: "x.db")
    monkeypatch.setattr(xs, "load_runtime_history", lambda db: object())
    monkeypatch.setattr(xs, "squad_gate", lambda: {})
    return monkeypatch


def test_pretoss_xi_uses_selector_where_gate_passed(stubs):
    stubs.setattr(xs, "load_selector", lambda: _Sel(True))
    stubs.delenv("PAPER_PRETOSS_XI", raising=False)
    bat, bowl, predicted = paper_inputs.get_pretoss_xi(None, 7, "T20", "male", "2026-06-12")
    assert predicted and bat == list(range(101, 112)) and bowl == list(range(107, 112))


@pytest.mark.parametrize("allowed, mode", [(False, "auto"), (True, "last_match")])
def test_pretoss_xi_falls_back_to_last_match(stubs, allowed, mode):
    stubs.setattr(xs, "load_selector", lambda: _Sel(allowed))
    stubs.setenv("PAPER_PRETOSS_XI", mode)
    bat, _, predicted = paper_inputs.get_pretoss_xi(None, 7, "T20", "male", "2026-06-12")
    assert not predicted and bat == list(range(1, 12))


def test_pretoss_xi_skips_an_unknown_lineup(stubs):
    class Unsure(_Sel):
        def predict(self, *a):
            return list(range(101, 112)), {"stale_days": 700, "expected_recall": 0.3}
    stubs.setattr(xs, "load_selector", lambda: Unsure(True))
    stubs.delenv("PAPER_PRETOSS_XI", raising=False)
    stubs.delenv("PAPER_XI_MIN_EXPECTED_RECALL", raising=False)
    assert paper_inputs.get_pretoss_xi(None, 7, "T20", "male") == ([], [], True)
    stubs.setenv("PAPER_XI_MIN_EXPECTED_RECALL", "0.2")
    assert paper_inputs.get_pretoss_xi(None, 7, "T20", "male")[0] == list(range(101, 112))


def test_pretoss_xi_falls_back_when_selector_missing_or_fails(stubs):
    stubs.delenv("PAPER_PRETOSS_XI", raising=False)
    stubs.setattr(xs, "load_selector", lambda: None)
    assert paper_inputs.get_pretoss_xi(None, 7, "T20", "male")[2] is False

    def boom():
        raise RuntimeError("bad pickle")
    stubs.setattr(xs, "load_selector", boom)
    assert paper_inputs.get_pretoss_xi(None, 7, "T20", "male")[2] is False
