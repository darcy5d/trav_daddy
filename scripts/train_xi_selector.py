#!/usr/bin/env python3
"""Fit the pre-toss XI selector and decide, per format/gender, whether paper
scans may use it.

Holdout: every side from --split onwards, scored against its declared XI. A
match counts as known on day D only if it was ingested (`matches.created_at`)
by D; bulk-loaded history (created_at more than 30 days after the match, or
before it) gets a lag drawn from the 2026 ingest lags instead. The baseline is
`paper_inputs.get_recent_xi` as it runs today. A segment passes when the
selector's mean recall beats the baseline with the 95% bootstrap interval of
the paired difference above zero on at least 100 holdout sides. The final model
is then refit on every side and written with that gate.

Reads cricket.db read-only.

    nice -n 10 python scripts/train_xi_selector.py
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.models.xi_selector import (  # noqa: E402
    ARTIFACT_DIR, DECLARED_XI_PATH, FEATURES, History, XISelector, load_sides, pick_xi,
)

MIN_HOLDOUT_SIDES = 100


def ingest_visibility(sides: pd.DataFrame, seed: int = 0) -> pd.Series:
    lag = (sides.created - sides.d).dt.total_seconds() / 86400
    real = (lag > 0) & (lag <= 30)
    emp = lag[real & (sides.d >= "2026-03-01")].to_numpy()
    rng = np.random.default_rng(seed)
    sim = sides.d + pd.to_timedelta(rng.choice(emp, len(sides)), unit="D")
    return pd.Series(np.where(real, sides.created, sim), index=sides.index)


def baseline_xi(pms_by_side, hist_keys):
    """paper_inputs.get_recent_xi on player_match_stats rows."""
    if not hist_keys:
        return []
    last = sorted(pms_by_side.get(hist_keys[0], []),
                  key=lambda r: (r[1] if r[1] is not None and r[1] == r[1] else 99, -(r[2] or 0)))
    out = [p for p, _, _ in last][:11]
    if len(out) < 11:
        cnt = defaultdict(int)
        for k in hist_keys[:3]:
            for p, _, _ in pms_by_side.get(k, []):
                cnt[p] += 1
        out += [p for p in sorted(cnt, key=lambda p: -cnt[p]) if p not in out]
    return out[:11]


def recall(pred, truth):
    return len(set(pred) & set(truth)) / max(1, len(truth))


def boot_ci(diff, n=2000, seed=1):
    rng = np.random.default_rng(seed)
    d = np.asarray(diff, dtype=float)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return [round(float(d.mean()), 4), round(float(np.percentile(bs, 2.5)), 4),
            round(float(np.percentile(bs, 97.5)), 4)]


def new_model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "cricket.db"))
    ap.add_argument("--declared-xi", default=str(DECLARED_XI_PATH))
    ap.add_argument("--train-from", default="2024-07-01")
    ap.add_argument("--split", default="2026-05-01")
    ap.add_argument("--out", default=str(ARTIFACT_DIR))
    args = ap.parse_args()
    t0 = time.time()

    sides = load_sides(args.db, pd.read_pickle(args.declared_xi))
    hist = History(sides, ingest_visibility(sides))
    c = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    pms = pd.read_sql(
        "SELECT pms.match_id, pms.team_id, pms.player_id, pms.batting_position, pms.balls_faced "
        "FROM player_match_stats pms JOIN matches m USING(match_id) WHERE m.date >= '2022-01-01'", c)
    c.close()
    pms_by_side = defaultdict(list)
    for r in pms.itertuples():
        pms_by_side[(r.match_id, r.team_id)].append((r.player_id, r.batting_position, r.balls_faced))
    print(f"loaded {len(sides):,} sides ({time.time() - t0:.0f}s)", flush=True)

    targets, frames = [], []
    for s in sides[sides.d >= args.train_from].itertuples():
        cands, info = hist.candidates(s.team_id, s.gender, s.fmt, s.d)
        if info["n_hist"] == 0:
            continue
        g = hist.groups[(s.team_id, s.gender)]
        known = g[(g.d < s.d) & (g.vis <= s.d) & (g.fmt == s.fmt)]
        base = baseline_xi(pms_by_side, [(int(m), s.team_id) for m in known.match_id.head(3)])
        i = len(targets)
        targets.append({"i": i, "match_id": s.match_id, "team_id": s.team_id, "d": s.d,
                        "seg": f"{s.fmt}_{s.gender}", "truth": set(s.xi), "base": base,
                        "stale": info["stale_days"]})
        cands = cands.assign(i=i, y=cands.player_id.isin(set(s.xi)).astype(int))
        frames.append(cands)
    T = pd.DataFrame(targets)
    C = pd.concat(frames, ignore_index=True)
    print(f"{len(T):,} target sides, {len(C):,} candidates ({time.time() - t0:.0f}s)", flush=True)

    tr = T[T.d < args.split].i
    te = T[T.d >= args.split]
    ctr, cte = C[C.i.isin(tr)], C[C.i.isin(te.i)]
    clf = new_model().fit(ctr[FEATURES].astype(float), ctr.y)
    cte = cte.assign(prob=clf.predict_proba(cte[FEATURES].astype(float))[:, 1])
    picks = {i: pick_xi(g, g.prob.to_numpy()) for i, g in cte.groupby("i")}
    te = te.assign(r_sel=[recall(picks.get(i, []), t) for i, t in zip(te.i, te.truth)],
                   r_base=[recall(b, t) for b, t in zip(te.base, te.truth)])

    holdout, gate = {}, {}
    for seg, sub in list(te.groupby("seg")) + [("all", te)]:
        ci = boot_ci(sub.r_sel - sub.r_base)
        holdout[seg] = {"n": int(len(sub)), "recall_last_match": round(float(sub.r_base.mean()), 4),
                        "recall_selector": round(float(sub.r_sel.mean()), 4), "diff_ci": ci}
        if seg != "all":
            gate[seg] = bool(len(sub) >= MIN_HOLDOUT_SIDES and ci[1] > 0)
    sb = pd.cut(te.stale, [-1, 7, 30, 100_000], labels=["<=7d", "8-30d", ">30d"])
    by_stale = {str(k): {"n": int(len(s)), "recall_last_match": round(float(s.r_base.mean()), 4),
                         "recall_selector": round(float(s.r_sel.mean()), 4)}
                for k, s in te.groupby(sb, observed=True)}
    print(json.dumps({"holdout": holdout, "by_staleness": by_stale, "gate": gate}, indent=1), flush=True)

    final = new_model().fit(C[FEATURES].astype(float), C.y)
    meta = {"split": args.split, "train_from": args.train_from, "holdout": holdout,
            "by_staleness": by_stale, "n_sides": int(len(T)), "built_at": pd.Timestamp.now("UTC").isoformat(),
            "secs": round(time.time() - t0, 1)}
    sel = XISelector(model=final, gate=gate, trained_through=str(sides.d.max().date()), meta=meta)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "selector.pkl.tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(sel, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, out / "selector.pkl")
    (out / "gate.json").write_text(json.dumps({"gate": gate, **meta}, indent=1))
    print(f"wrote {out} gate={gate} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
