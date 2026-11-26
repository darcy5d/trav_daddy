"""Match-level model: pre-match feature state and orientation symmetry."""
import pickle

import numpy as np
import pandas as pd

from src.models.match_level.decision_prices import moneyline_markets, team1_outcome_token
from src.models.match_level.features import MatchFeatureBuilder
from src.models.match_level.model import (
    MatchLevelPredictor, fit, fit_block_ref, has_both_declared_xi, predict,
    score_league_gate, split_trainable, time_split_bounds, time_split_output_allowed,
    trainable,
)
from src.models.calibration import fit_platt


def _tables(n_matches=240, seed=0):
    rng = np.random.default_rng(seed)
    teams = pd.DataFrame({
        "team_id": [1, 2, 3, 4], "name": ["A", "B", "C", "D"],
        "team_type": ["domestic"] * 4, "tier": [1, 1, 2, 2], "canonical_team_id": [None] * 4,
    })
    venues = pd.DataFrame({"venue_id": [10, 11], "country": ["X", None]})
    strength = {1: 300.0, 2: 100.0, 3: -100.0, 4: -300.0}
    rows, telo, innings, elo = [], [], [], {1: 1500.0, 2: 1500.0, 3: 1500.0, 4: 1500.0}
    for i in range(n_matches):
        a, b = rng.choice([1, 2, 3, 4], size=2, replace=False)
        p = 1 / (1 + 10 ** ((strength[b] - strength[a]) / 400))
        w = a if rng.random() < p else b
        mid = i + 1
        rows.append({
            "match_id": mid, "match_type": "T20", "date": str(pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)),
            "venue_id": 10 + i % 2, "team1_id": a, "team2_id": b, "toss_winner_id": a, "toss_decision": "bat",
            "winner_id": w, "event_name": "L", "gender": "male", "match_category": "domestic", "cricsheet_id": str(mid),
        })
        for t in (a, b):
            change = 10.0 if t == w else -10.0
            elo[t] += change
            telo.append({"team_id": t, "match_id": mid, "elo": elo[t], "elo_change": change})
        innings.append({"match_id": mid, "innings_number": 1, "batting_team_id": a, "total_runs": 160})
    empty_pe = pd.DataFrame(columns=["player_id", "match_id", "batting_elo", "bowling_elo",
                                     "batting_elo_change", "bowling_elo_change"])
    empty_pms = pd.DataFrame(columns=["match_id", "player_id", "team_id", "batting_innings", "runs_scored",
                                      "balls_faced", "overs_bowled", "runs_conceded", "wickets_taken"])
    return {"matches": pd.DataFrame(rows), "teams": teams, "venues": venues,
            "innings": pd.DataFrame(innings), "team_elo": pd.DataFrame(telo),
            "player_elo": empty_pe, "pms": empty_pms}


def test_upcoming_features_do_not_mutate_state_and_survive_pickle():
    b = MatchFeatureBuilder(_tables(), {})
    b.run_history()
    b.compact()
    before = pickle.dumps(b)
    f1 = b.features_for_upcoming(1, 4, "T20", "male", 10, pd.Timestamp("2025-01-01"))
    assert pickle.dumps(b) == before
    b2 = pickle.loads(before)
    f2 = b2.features_for_upcoming(1, 4, "T20", "male", 10, pd.Timestamp("2025-01-01"))
    assert f1["tiered_elo_pre_1"] == f2["tiered_elo_pre_1"] > f1["tiered_elo_pre_2"]


def test_prediction_is_orientation_symmetric_and_ordered():
    t = _tables()
    b = MatchFeatureBuilder(t, {})
    df = b.run_history()
    clf = fit(trainable(df), "team_venue", df.date.max(), {"n_estimators": 50})
    pred = MatchLevelPredictor(models={"team_venue_xgb": clf}, builder=b.compact(),
                               gate={"team_venue_xgb": {"x": True}})
    kw = dict(fmt="T20", gender="male", venue_id=10, date="2025-01-01")
    p14 = pred.predict_fixture("team_venue_xgb", team1_id=1, team2_id=4, **kw)
    p41 = pred.predict_fixture("team_venue_xgb", team1_id=4, team2_id=1, **kw)
    assert abs(p14 + p41 - 1.0) < 1e-6
    assert p14 > 0.5
    assert pred.allowed("team_venue_xgb", "x") and not pred.allowed("team_venue_xgb", "y")
    assert not pred.allowed("full_xgb", "x")
    aware = pd.Timestamp("2025-01-01 09:30", tz="UTC")
    p_aware = pred.predict_fixture("team_venue_xgb", team1_id=1, team2_id=4, fmt="T20", gender="male",
                                   venue_id=10, date=aware)
    assert abs(p_aware - p14) < 1e-9
    row = b.features_for_upcoming(1, 4, "T20", "male", 10, pd.Timestamp("2025-01-01"))
    raw = float(predict(clf, pd.DataFrame([row]), "team_venue")[0])
    assert abs(p14 - raw) < 1e-9


def _dated_rows():
    rows = []
    for i, day in enumerate(["2025-12-01", "2026-01-25", "2026-01-26", "2026-04-25", "2026-04-26", "2026-05-01"]):
        rows.append({
            "match_id": i + 1, "date": pd.Timestamp(day), "y": i % 2,
            "tiered_elo_pre_1": 1500.0, "tiered_elo_pre_2": 1480.0,
            "xi_n_1": 11 if i != 4 else 0, "xi_n_2": 11,
        })
    return pd.DataFrame(rows)


def test_time_split_dates_and_blocks():
    cal_start, holdout = time_split_bounds()
    assert str(cal_start.date()) == "2026-01-26"
    assert str(holdout.date()) == "2026-04-26"
    fit_df, cal_df, gate_df = split_trainable(_dated_rows(), holdout, cal_start)
    assert set(fit_df.match_id) == {1, 2}
    assert set(cal_df.match_id) == {3, 4}
    assert set(gate_df.match_id) == {5, 6}
    assert fit_block_ref(fit_df) == pd.Timestamp("2026-01-26")
    # The holdout row is in neither the tree matrix nor the Platt input.
    assert 5 not in set(fit_df.match_id) and 5 not in set(cal_df.match_id)
    assert 6 not in set(fit_df.match_id) and 6 not in set(cal_df.match_id)
    # The calibration-window row is absent from the trees and present for Platt.
    assert 3 not in set(fit_df.match_id) and 3 in set(cal_df.match_id)
    held = fit(fit_df.assign(**_min_design_columns(fit_df)), "team_venue", fit_block_ref(fit_df),
               {"n_estimators": 4, "max_depth": 2})
    cal_probs = predict(held, cal_df.assign(**_min_design_columns(cal_df)), "team_venue")
    params = fit_platt(np.asarray(cal_probs, dtype=float), cal_df.y.to_numpy(dtype=float))
    assert params.n_train == len(cal_df)
    assert set(gate_df.match_id).isdisjoint(set(fit_df.match_id) | set(cal_df.match_id))


def _min_design_columns(df):
    """Columns design() reads, filled so a tiny tree can fit the dated rows."""
    extra = {}
    for side in ("1", "2"):
        extra[f"simple_elo_{side}"] = 1500.0
        extra[f"fast_elo_{side}"] = 1500.0
        extra[f"elo_mom90_{side}"] = 0.0
        extra[f"form_ewm_{side}"] = 0.5
        extra[f"form5_{side}"] = 0.5
        extra[f"n_prior_{side}"] = 10.0
        extra[f"rest_days_{side}"] = 5.0
        extra[f"games_30d_{side}"] = 2.0
        extra[f"tier_{side}"] = 1.0
        extra[f"country_share_{side}"] = 0.5
        extra[f"is_home_{side}"] = 0.0
        extra[f"venue_games_{side}"] = 3.0
    extra.update({
        "h2h_rate_1": 0.5, "h2h_n": 4.0, "venue_n": 10.0,
        "venue_bat1_rate": 0.5, "venue_par_rel": 1.0,
        "is_t20": 1.0, "is_female": 0.0, "intl": 1.0, "franchise": 0.0,
    })
    return extra


def test_declared_xi_flag_and_live_dir_refusal(tmp_path):
    df = _dated_rows()
    mask = has_both_declared_xi(df)
    assert list(df.loc[mask, "match_id"]) == [1, 2, 3, 4, 6]
    assert not bool(mask.loc[df.match_id == 5].iloc[0])
    live = tmp_path / "match_level"
    live.mkdir()
    side = tmp_path / "match_level_timesplit"
    assert time_split_output_allowed(side, live)
    assert not time_split_output_allowed(live, live)
    assert not time_split_output_allowed(live / "", live)


def _holdout(prefix, n, model_prob, price, y=1.0, model="team_venue_xgb"):
    return [
        {"model": model, "prefix": prefix, "kind": "holdout",
         "y": y, "model_prob": model_prob, "price": price}
        for _ in range(n)
    ]


def _history(prefix, n, model="team_venue_xgb"):
    return [{"model": model, "prefix": prefix, "kind": "history"} for _ in range(n)]


def test_league_gate_rule_is_the_same_for_every_prefix():
    # Eight holdout fixtures fail the minimum, even when the model is perfect.
    thin = score_league_gate(_history("cricnpl", 200) + _holdout("cricnpl", 8, 1.0, 0.5))
    assert thin["allowed"]["team_venue_xgb"]["cricnpl"] is False
    assert thin["prefixes"]["team_venue_xgb"]["cricnpl"]["n_holdout"] == 8

    short_hist = score_league_gate(_history("cricwbbl", 50) + _holdout("cricwbbl", 30, 1.0, 0.5))
    assert short_hist["allowed"]["team_venue_xgb"]["cricwbbl"] is False
    assert short_hist["prefixes"]["team_venue_xgb"]["cricwbbl"]["n_history"] == 50

    # Model worse than the price: bootstrap share is under 0.90.
    worse = score_league_gate(
        _history("cricipl", 120) + _holdout("cricipl", 24, 0.1, 0.9, y=1.0)
    )
    ipl = worse["prefixes"]["team_venue_xgb"]["cricipl"]
    assert ipl["bootstrap_share"] < 0.90
    assert ipl["log_loss_model"] is not None
    assert worse["allowed"]["team_venue_xgb"]["cricipl"] is False

    # A missing T-1h price is not a holdout row.
    dropped = score_league_gate(
        _history("crint", 120) + [
            {"model": "team_venue_xgb", "prefix": "crint", "kind": "holdout",
             "y": 1, "model_prob": 0.8, "price": None},
        ]
    )
    assert dropped["prefixes"]["team_venue_xgb"]["crint"]["n_holdout"] == 0

    # The same numbers pass, including for cricnpl. Nothing hard-bans it.
    passed = score_league_gate(_history("cricnpl", 100) + _holdout("cricnpl", 20, 1.0, 0.4))
    assert passed["allowed"]["team_venue_xgb"]["cricnpl"] is True
    assert passed["prefixes"]["team_venue_xgb"]["cricnpl"]["bootstrap_share"] == 1.0

    # Tamil Nadu is not Nepal. The unused WBBL string is not a key.
    other = score_league_gate(
        _history("crictnpl", 150) + _holdout("crictnpl", 25, 1.0, 0.4)
        + _history("crict20wbbl", 400) + _holdout("crict20wbbl", 40, 1.0, 0.4)
    )
    assert other["allowed"]["team_venue_xgb"]["crictnpl"] is True
    assert other["allowed"]["team_venue_xgb"]["cricnpl"] is False
    assert "crict20wbbl" not in other["allowed"]["team_venue_xgb"]
    assert "cricwbbl" in other["allowed"]["team_venue_xgb"]
    assert other["allowed"]["team_venue_xgb"]["cricwbbl"] is False


def test_moneyline_token_is_team1_and_wrong_slug_is_dropped():
    outcomes = [
        {"label": "Sydney Sixers", "token_id": "sixers"},
        {"label": "Melbourne Stars", "token_id": "stars"},
    ]
    assert team1_outcome_token(outcomes, "Sydney Sixers", "Melbourne Stars") == "sixers"
    assert team1_outcome_token(outcomes, "Perth Scorchers", "Melbourne Stars") is None
    events = [
        {
            "slug": "cricwbbl-syd-mel-2026-11-01",
            "title": "Women's Big Bash League: Sydney Sixers vs Melbourne Stars",
            "markets": [{
                "question": "Sydney Sixers vs Melbourne Stars",
                "outcomes": '["Sydney Sixers", "Melbourne Stars"]',
                "clobTokenIds": '["a", "b"]',
                "sportsMarketType": "moneyline",
                "gameStartTime": "2026-11-01T09:00:00Z",
            }],
        },
        {
            "slug": "crict20wbbl-syd-mel-2026-11-01",
            "title": "not a scanner prefix",
            "markets": [{
                "question": "Sydney Sixers vs Melbourne Stars",
                "outcomes": ["Sydney Sixers", "Melbourne Stars"],
                "clobTokenIds": ["c", "d"],
                "sportsMarketType": "moneyline",
            }],
        },
        {
            "slug": "cricnpl-ktm-bir-2026-11-02-toss-winner",
            "title": "Nepal Premier League: Kathmandu vs Biratnagar",
            "markets": [],
        },
    ]
    got = moneyline_markets(events)
    assert [m["prefix"] for m in got] == ["cricwbbl"]
    assert got[0]["outcomes"][0]["token_id"] == "a"
