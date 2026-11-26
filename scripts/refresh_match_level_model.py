#!/usr/bin/env python3
"""Rebuild the match-level model artifacts in data/models/match_level/.

Reads cricket.db read-only and the Cricsheet JSON (declared XIs), walks every
match to build pre-match features and the rolling state, fits team_venue_xgb
and full_xgb on all labelled matches, and writes predictor.pkl. The per-league
gate (which Polymarket prefixes each model may bet) is a fixed decision from
the walk-forward holdout and is written alongside so scanners can read it.

`--time-split` is the opt-in experiment. It fits only on matches before the
calibration block, fits one Platt map per model on that block, and scores the
league gate on the later holdout. It refuses to write the live artifact
directory. The default, with no flag, is unchanged.

Usage:
    nice -n 10 python scripts/refresh_match_level_model.py [--check data/model_opt/match_features.pkl]
    python scripts/refresh_match_level_model.py --time-split --out data/models/match_level_timesplit
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.models.match_level.features import MatchFeatureBuilder, load_declared_xi, load_tables  # noqa: E402
from src.models.match_level.model import (  # noqa: E402
    ARTIFACT_DIR, MODEL_FEATURES, MatchLevelPredictor, fit, fit_block_ref,
    has_both_declared_xi, predict, score_league_gate, split_trainable,
    time_split_bounds, time_split_output_allowed, trainable,
)

RAW_DIRS = [REPO / "data" / "raw" / d for d in ("all_male", "all_female", "t20i", "odi", "t20s_female")]

# Walk-forward holdout 2026-04-26 onwards against the Polymarket decision price.
# Pass rule: >= 20 holdout fixtures, >= 100 history matches, model Brier below
# price with bootstrap confidence >= 0.90. WBBL (crict20wbbl) had 8 fixtures.
LEAGUE_GATE = {
    "rule": {"min_holdout": 20, "min_history": 100, "min_conf": 0.90, "holdout_start": "2026-04-26"},
    "allowed": {
        "team_venue_xgb": {"crint": True, "cricodc": True, "cricipl": True},
        "full_xgb": {"crint": True},
    },
}


def _build_frame(args, out: Path):
    """Read-only database load and feature walk. Writes the XI cache only under `out`."""
    t0 = time.time()
    tables = load_tables(args.db)
    print(f"tables loaded {time.time() - t0:.0f}s", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    xi_cache_path = out / "declared_xi.pkl"
    seed = Path(args.declared_xi_seed) if args.declared_xi_seed else xi_cache_path
    cache = {}
    if seed.exists():
        cdf = pd.read_pickle(seed)
        cache = {(int(r.match_id), int(r.team_id)): list(r.players) for r in cdf.itertuples()}
    dxi = load_declared_xi(tables, args.db, RAW_DIRS, cache=cache)
    pd.DataFrame(
        [{"match_id": m, "team_id": t, "players": p} for (m, t), p in dxi.items()]
    ).to_pickle(xi_cache_path)
    print(f"declared XIs {len(dxi):,} ({len(dxi) - len(cache):,} new; {time.time() - t0:.0f}s)", flush=True)
    builder = MatchFeatureBuilder(tables, dxi)
    df = builder.run_history()
    print(f"features {df.shape} ({time.time() - t0:.0f}s)", flush=True)
    return tables, df, t0


def _check_frame(df: pd.DataFrame, check_path: str) -> None:
    ref = pd.read_pickle(check_path)
    common = ref.merge(df, on="match_id", suffixes=("_ref", ""))
    bad = []
    for col in ref.columns:
        if col == "match_id" or col not in df.columns:
            continue
        a, b = common[f"{col}_ref"], common[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            same = np.isclose(a.astype(float), b.astype(float), equal_nan=True, atol=1e-9)
        else:
            same = (a == b) | (a.isna() & b.isna())
        if not same.all():
            bad.append((col, int((~same).sum())))
    print(f"check vs {check_path}: {len(common):,} common matches, mismatching cols {bad}", flush=True)


def run_time_split(args, out: Path) -> None:
    """Fit the predeclared time split and write only under `out`.

    Does not open bet_ledger. Does not write data/models/match_level.
    """
    from datetime import datetime, timezone

    from src.integrations.polymarket import PolymarketClient
    from src.models.calibration import calibrate_probabilities, fit_platt
    from src.models.match_level.decision_prices import (
        fetch_cricket_events, fetch_t1h_prices, join_moneylines, moneyline_markets,
    )

    if not args.declared_xi_seed:
        live_xi = ARTIFACT_DIR / "declared_xi.pkl"
        if live_xi.exists():
            args.declared_xi_seed = str(live_xi)
    tables, df, t0 = _build_frame(args, out)
    if args.check:
        _check_frame(df, args.check)

    cal_start, holdout = time_split_bounds()
    fit_df, cal_df, gate_df = split_trainable(df, holdout, cal_start)
    if fit_df.empty:
        raise SystemExit("time split fit block is empty")
    ref = fit_block_ref(fit_df)
    print(
        f"split fit={len(fit_df):,} cal={len(cal_df):,} gate={len(gate_df):,} "
        f"ref={ref.date()} trained_through={fit_df.date.max().date()}",
        flush=True,
    )

    models = {}
    platt = {}
    calibrated = {}
    for name, features in MODEL_FEATURES.items():
        clf = fit(fit_df, features, ref)
        models[name] = clf
        if len(cal_df):
            raw_cal = predict(clf, cal_df, features)
            params = fit_platt(np.asarray(raw_cal, dtype=float), cal_df.y.to_numpy(dtype=float))
        else:
            params = fit_platt(np.array([]), np.array([]))
        platt[name] = {"a": params.a, "b": params.b, "n": int(params.n_train)}
        if len(gate_df):
            gate_raw = predict(clf, gate_df, features)
            calibrated[name] = calibrate_probabilities(np.asarray(gate_raw, dtype=float), params)
        else:
            calibrated[name] = np.array([])
        print(
            f"fit {name} on {len(fit_df):,}  platt a={params.a:.4f} b={params.b:.4f} n={params.n_train} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

    with open(out / "classifiers.pkl", "wb") as fh:
        pickle.dump(models, fh, protocol=pickle.HIGHEST_PROTOCOL)
    (out / "platt.json").write_text(json.dumps(platt, indent=1))

    names = {}
    team_table = tables["teams"]
    for tid, raw_name in zip(team_table["team_id"], team_table["name"]):
        if isinstance(raw_name, str) and raw_name:
            names[int(tid)] = raw_name
    market_path = out / "moneyline_markets.json"
    markets = json.loads(market_path.read_text()) if market_path.exists() else []
    holdout_day = str(pd.Timestamp(holdout).date())
    covers = bool(markets) and max(m["slug_date"] for m in markets) >= holdout_day
    if not covers:
        client = PolymarketClient()
        events = fetch_cricket_events(
            client,
            datetime(2024, 6, 1, tzinfo=timezone.utc),
            datetime(2026, 12, 15, tzinfo=timezone.utc),
        )
        if not events:
            raise SystemExit("no cricket events from Gamma; gate not written")
        markets = moneyline_markets(events)
        market_path.write_text(json.dumps(markets))
        print(
            f"moneyline markets {len(markets):,} from {len(events):,} events "
            f"({min(m['slug_date'] for m in markets)} .. {max(m['slug_date'] for m in markets)})",
            flush=True,
        )
        del events
    else:
        print(f"moneyline markets from cache {len(markets):,}", flush=True)
    price_client = PolymarketClient()

    scored = pd.concat([fit_df, cal_df, gate_df], ignore_index=True)
    joins = join_moneylines(scored, markets, names)
    (out / "joins.json").write_text(json.dumps(joins))
    print(f"joined {len(joins):,} matches to a moneyline", flush=True)

    gate_ids = set(int(m) for m in gate_df.match_id)
    price_joins = [j for j in joins if int(j["match_id"]) in gate_ids and j.get("token_id")]
    prices = fetch_t1h_prices(price_client, price_joins, out / "prices.json")
    n_priced = sum(1 for v in prices.values() if v is not None)
    print(f"T-1h prices {n_priced:,} of {len(price_joins):,} token-mapped gate matches", flush=True)

    by_id = {int(j["match_id"]): j for j in joins}
    gate_index = {int(m): i for i, m in enumerate(gate_df.match_id)}
    both_xi = set(int(m) for m in gate_df.loc[has_both_declared_xi(gate_df), "match_id"])
    holdout_ts = pd.Timestamp(holdout)
    observations = []
    for row in scored.itertuples(index=False):
        mid = int(row.match_id)
        join = by_id.get(mid)
        if join is None:
            continue
        prefix = join["prefix"]
        if pd.Timestamp(row.date) < holdout_ts:
            for model in MODEL_FEATURES:
                observations.append({"model": model, "prefix": prefix, "kind": "history"})
            continue
        price = prices.get(mid)
        if price is None:
            continue
        idx = gate_index.get(mid)
        if idx is None:
            continue
        y = int(row.y)
        observations.append({
            "model": "team_venue_xgb", "prefix": prefix, "kind": "holdout",
            "y": y, "model_prob": float(calibrated["team_venue_xgb"][idx]), "price": float(price),
        })
        if mid in both_xi:
            observations.append({
                "model": "full_xgb", "prefix": prefix, "kind": "holdout",
                "y": y, "model_prob": float(calibrated["full_xgb"][idx]), "price": float(price),
            })

    scored_gate = score_league_gate(observations)
    (out / "league_gate.json").write_text(json.dumps(
        {"rule": scored_gate["rule"], "allowed": scored_gate["allowed"]}, indent=1,
    ))
    (out / "prefix_table.json").write_text(json.dumps(scored_gate["prefixes"], indent=1))
    meta = {
        "n_fit": int(len(fit_df)),
        "n_cal": int(len(cal_df)),
        "n_gate": int(len(gate_df)),
        "trained_through": str(fit_df.date.max().date()),
        "calibration_start": str(pd.Timestamp(cal_start).date()),
        "holdout_start": str(pd.Timestamp(holdout).date()),
        "built_at": pd.Timestamp.now("UTC").isoformat(),
        "secs": round(time.time() - t0, 1),
        "n_joined": int(len(joins)),
        "n_priced_holdout": int(n_priced),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out} {meta}", flush=True)
    for model, flags in scored_gate["allowed"].items():
        passed = sorted(p for p, ok in flags.items() if ok)
        failed = sorted(p for p, ok in flags.items() if not ok)
        print(f"  {model} pass={passed or '-'} fail={failed or '-'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "cricket.db"))
    ap.add_argument("--out", default=str(ARTIFACT_DIR))
    ap.add_argument("--check", help="feature table to regression-check against (pickle)")
    ap.add_argument("--declared-xi-seed", help="declared-XI pickle to start from instead of <out>/declared_xi.pkl")
    ap.add_argument(
        "--time-split", action="store_true",
        help="Fit the predeclared time split. Refuses to write the live artifact directory.",
    )
    args = ap.parse_args()

    if args.time_split:
        out = Path(args.out).resolve()
        if not time_split_output_allowed(out):
            raise SystemExit(
                f"refusing --time-split --out {out}: that is the live artifact directory "
                f"{ARTIFACT_DIR.resolve()}"
            )
        run_time_split(args, out)
        return

    t0 = time.time()
    tables = load_tables(args.db)
    print(f"tables loaded {time.time() - t0:.0f}s", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    xi_cache_path = out / "declared_xi.pkl"
    seed = Path(args.declared_xi_seed) if args.declared_xi_seed else xi_cache_path
    cache = {}
    if seed.exists():
        cdf = pd.read_pickle(seed)
        cache = {(int(r.match_id), int(r.team_id)): list(r.players) for r in cdf.itertuples()}
    dxi = load_declared_xi(tables, args.db, RAW_DIRS, cache=cache)
    pd.DataFrame([{"match_id": m, "team_id": t, "players": p} for (m, t), p in dxi.items()]).to_pickle(xi_cache_path)
    print(f"declared XIs {len(dxi):,} ({len(dxi) - len(cache):,} new; {time.time() - t0:.0f}s)", flush=True)
    builder = MatchFeatureBuilder(tables, dxi)
    df = builder.run_history()
    print(f"features {df.shape} ({time.time() - t0:.0f}s)", flush=True)

    if args.check:
        ref = pd.read_pickle(args.check)
        common = ref.merge(df, on="match_id", suffixes=("_ref", ""))
        bad = []
        for col in ref.columns:
            if col == "match_id" or col not in df.columns:
                continue
            a, b = common[f"{col}_ref"], common[col]
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                same = np.isclose(a.astype(float), b.astype(float), equal_nan=True, atol=1e-9)
            else:
                same = (a == b) | (a.isna() & b.isna())
            if not same.all():
                bad.append((col, int((~same).sum())))
        print(f"check vs {args.check}: {len(common):,} common matches, mismatching cols {bad}", flush=True)

    train = trainable(df)
    ref_date = df.date.max() + pd.Timedelta(days=1)
    models = {}
    for name, features in MODEL_FEATURES.items():
        models[name] = fit(train, features, ref_date)
        print(f"fit {name} on {len(train):,} matches ({time.time() - t0:.0f}s)", flush=True)

    pred = MatchLevelPredictor(models=models, builder=builder.compact(),
                               gate=LEAGUE_GATE["allowed"], trained_through=str(df.date.max().date()))
    tmp = out / "predictor.pkl.tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(pred, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, out / "predictor.pkl")
    (out / "league_gate.json").write_text(json.dumps(LEAGUE_GATE, indent=1))
    meta = {"trained_through": pred.trained_through, "n_train": int(len(train)),
            "built_at": pd.Timestamp.now("UTC").isoformat(), "secs": round(time.time() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out} {meta}", flush=True)


if __name__ == "__main__":
    main()
