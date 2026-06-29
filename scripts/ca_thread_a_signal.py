#!/usr/bin/env python3
"""Thread A: does CA scorecard data carry predictive signal for match outcomes?

Builds a LEAK-FREE pre-match feature set from ca_archive.db (each match's
features use only matches that occurred strictly earlier), then trains simple
models (logistic regression, gradient boosting) to predict the winner and
compares them to naive baselines. Two feature tiers:

  * team form  - rolling team win-rate / scoring (the weakest use of scorecards)
  * player XI  - strength of the selected XI from each player's prior per-match
                 batting/bowling aggregates (CA's real value: player IDs)

The XI is known at the toss, so using the selected XI's PRIOR form is legitimate
for outcome prediction (career aggregates are shrunk toward the running league
mean so debutants do not distort).

READ-ONLY on ca_archive.db. Prints metrics + writes a JSON report.

Usage:
    venv311/bin/python scripts/ca_thread_a_signal.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROCESSED_DATA_DIR
from src.api.cricketarchive import store

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _winner(result: str, t1: str, t2: str):
    if not result or " won by " not in result:
        return None
    who = _norm(result.split(" won by ", 1)[0])
    n1, n2 = _norm(t1), _norm(t2)
    if who and (who == n1 or who in n1 or n1 in who):
        return 1
    if who and (who == n2 or who in n2 or n2 in who):
        return 0
    return None


def _overs_to_balls(o):
    if o is None:
        return 0
    o = str(o).strip()
    if "." in o:
        a, _, b = o.partition(".")
        return (int(a) * 6 + int(b)) if a.isdigit() and b.isdigit() else 0
    return int(o) * 6 if o.isdigit() else 0


def load_matches() -> list:
    with store.get_connection() as c:
        rows = c.execute("""
            SELECT scorecard_id, match_date, competition_url, team1_name, team2_name, toss, result
            FROM ca_matches
            WHERE match_date IS NOT NULL AND team1_name IS NOT NULL AND team2_name IS NOT NULL
            ORDER BY match_date, scorecard_id
        """).fetchall()
        matches = []
        for r in rows:
            inn_team = {}  # innings_number -> normalised batting team
            runs = {}
            for i in c.execute("""SELECT innings_number, batting_team, total_runs FROM ca_innings
                                  WHERE scorecard_id=? AND innings_number IN (1,2)""", (r["scorecard_id"],)):
                if i["batting_team"]:
                    inn_team[i["innings_number"]] = _norm(i["batting_team"])
                    if i["total_runs"] is not None:
                        runs[_norm(i["batting_team"])] = i["total_runs"]
            # batters belong to the innings' batting team; bowlers to the other team
            bat = defaultdict(list); bowl = defaultdict(list)
            for b in c.execute("""SELECT innings_number, batter_ca_id, runs, balls, dismissal
                                  FROM ca_batting WHERE scorecard_id=?""", (r["scorecard_id"],)):
                t = inn_team.get(b["innings_number"])
                if t and b["batter_ca_id"]:
                    out = (b["dismissal"] or "").strip().lower() not in ("", "not out", "did not bat")
                    bat[t].append((b["batter_ca_id"], b["runs"] or 0, b["balls"] or 0, out))
            for bw in c.execute("""SELECT innings_number, bowler_ca_id, runs, wickets, overs
                                   FROM ca_bowling WHERE scorecard_id=?""", (r["scorecard_id"],)):
                bteam = inn_team.get(bw["innings_number"])
                others = [t for t in (_norm(r["team1_name"]), _norm(r["team2_name"])) if t != bteam]
                t = others[0] if others else None
                if t and bw["bowler_ca_id"]:
                    bowl[t].append((bw["bowler_ca_id"], bw["runs"] or 0, bw["wickets"] or 0,
                                    _overs_to_balls(bw["overs"])))
            matches.append({"id": r["scorecard_id"], "date": r["match_date"], "toss": r["toss"] or "",
                            "result": r["result"] or "", "t1": r["team1_name"], "t2": r["team2_name"],
                            "runs": runs, "bat": bat, "bowl": bowl})
    return matches


def _toss_features(toss: str, t1: str, t2: str):
    if " won the toss" not in toss:
        return 0.5, 0.5
    who = _norm(toss.split(" won the toss", 1)[0]); n1 = _norm(t1)
    won_t1 = 1.0 if (who and (who in n1 or n1 in who)) else 0.0
    bat = any(p in toss for p in ("decided to bat", "elected to bat", "chose to bat"))
    t1_bats_first = 1.0 if (won_t1 == bat) else 0.0  # toss winner bats first iff chose bat
    return won_t1, t1_bats_first


FEATURES = ["d_winrate", "d_roll_winrate", "d_net_form",
            "d_xi_bat_sr", "d_xi_bat_avg", "d_xi_bowl_econ", "d_xi_bowl_wpm",
            "toss_won_t1", "t1_bats_first"]


def build_dataset(matches: list, roll: int = 5):
    team = defaultdict(lambda: {"g": 0, "w": 0, "sc": [], "co": [],
                                "rw": deque(maxlen=roll)})
    # player career aggregates (leak-free)
    pbat = defaultdict(lambda: {"runs": 0, "balls": 0, "inns": 0, "outs": 0})
    pbowl = defaultdict(lambda: {"runs": 0, "balls": 0, "wkts": 0, "inns": 0})
    g_bat = {"runs": 0, "balls": 0, "outs": 0, "inns": 0}
    g_bowl = {"runs": 0, "balls": 0, "wkts": 0, "inns": 0}

    def league():
        sr = 100 * g_bat["runs"] / max(g_bat["balls"], 1)
        avg = g_bat["runs"] / max(g_bat["outs"], 1)
        econ = 6 * g_bowl["runs"] / max(g_bowl["balls"], 1)
        wpm = g_bowl["wkts"] / max(g_bowl["inns"], 1)
        return sr or 120.0, avg or 22.0, econ or 7.8, wpm or 1.0

    def xi_strength(players_bat, players_bowl):
        lsr, lavg, lecon, lwpm = league()
        srs, avgs = [], []
        for pid, *_ in players_bat:
            h = pbat[pid]
            # shrinkage toward league mean
            sr = (100 * h["runs"] + 30 * lsr) / (h["balls"] + 30)
            avg = (h["runs"] + 2 * lavg) / (h["outs"] + 2)
            srs.append(sr); avgs.append(avg)
        econs, wpms = [], []
        for pid, *_ in players_bowl:
            h = pbowl[pid]
            econ = (6 * h["runs"] + 12 * lecon) / (h["balls"] + 12)
            wpm = (h["wkts"] + 1 * lwpm) / (h["inns"] + 1)
            econs.append(econ); wpms.append(wpm)
        return (np.mean(srs) if srs else lsr, np.mean(avgs) if avgs else lavg,
                np.mean(econs) if econs else lecon, np.mean(wpms) if wpms else lwpm)

    X, y, meta = [], [], []
    for m in matches:
        w = _winner(m["result"], m["t1"], m["t2"])
        k1, k2 = _norm(m["t1"]), _norm(m["t2"])
        h1, h2 = team[k1], team[k2]

        if w is not None and h1["g"] >= roll and h2["g"] >= roll:
            wr1 = h1["w"] / h1["g"]; wr2 = h2["w"] / h2["g"]
            rwr1 = sum(h1["rw"]) / len(h1["rw"]) if h1["rw"] else 0.5
            rwr2 = sum(h2["rw"]) / len(h2["rw"]) if h2["rw"] else 0.5
            nf1 = (np.mean(h1["sc"]) - np.mean(h1["co"])) if h1["sc"] else 0.0
            nf2 = (np.mean(h2["sc"]) - np.mean(h2["co"])) if h2["sc"] else 0.0
            s1 = xi_strength(m["bat"].get(k1, []), m["bowl"].get(k1, []))
            s2 = xi_strength(m["bat"].get(k2, []), m["bowl"].get(k2, []))
            won_t1, t1bf = _toss_features(m["toss"], m["t1"], m["t2"])
            X.append([wr1 - wr2, rwr1 - rwr2, nf1 - nf2,
                      s1[0] - s2[0], s1[1] - s2[1], s1[2] - s2[2], s1[3] - s2[3],
                      won_t1 - 0.5, t1bf - 0.5])
            y.append(w); meta.append({"date": m["date"], "id": m["id"]})

        # --- update history AFTER use (no leakage) ---
        r1 = m["runs"].get(k1); r2 = m["runs"].get(k2)
        if w is not None and r1 is not None and r2 is not None:
            h1["g"] += 1; h2["g"] += 1; h1["w"] += w; h2["w"] += (1 - w)
            h1["sc"].append(r1); h1["co"].append(r2); h2["sc"].append(r2); h2["co"].append(r1)
            h1["rw"].append(w); h2["rw"].append(1 - w)
        for k in (k1, k2):
            for pid, runs, balls, out in m["bat"].get(k, []):
                pbat[pid]["runs"] += runs; pbat[pid]["balls"] += balls
                pbat[pid]["inns"] += 1; pbat[pid]["outs"] += int(out)
                g_bat["runs"] += runs; g_bat["balls"] += balls
                g_bat["outs"] += int(out); g_bat["inns"] += 1
            for pid, runs, wkts, balls in m["bowl"].get(k, []):
                pbowl[pid]["runs"] += runs; pbowl[pid]["balls"] += balls
                pbowl[pid]["wkts"] += wkts; pbowl[pid]["inns"] += 1
                g_bowl["runs"] += runs; g_bowl["balls"] += balls
                g_bowl["wkts"] += wkts; g_bowl["inns"] += 1

    return np.array(X, dtype=float), np.array(y, dtype=int), meta


def _metrics(y_true, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"accuracy": round(accuracy_score(y_true, (p >= 0.5).astype(int)), 4),
            "log_loss": round(log_loss(y_true, p), 4),
            "brier": round(brier_score_loss(y_true, p), 4),
            "auc": round(roc_auc_score(y_true, p), 4) if len(set(y_true)) > 1 else None}


def main() -> int:
    matches = load_matches()
    X, y, meta = build_dataset(matches)
    n = len(y)
    if n < 100:
        print(f"Not enough samples ({n}).")
        return 1
    split = int(n * 0.75)
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr_s, Xte_s = (Xtr - mu) / sd, (Xte - mu) / sd

    report = {"n_samples": n, "n_train": len(ytr), "n_test": len(yte),
              "test_base_rate_t1_win": round(float(yte.mean()), 4),
              "features": FEATURES, "models": {}, "baselines": {}}

    report["baselines"]["majority_class"] = _metrics(yte, np.full(len(yte), ytr.mean()))
    ti = FEATURES.index("toss_won_t1")
    report["baselines"]["toss_winner_wins"] = _metrics(yte, np.where(Xte[:, ti] > 0, 0.55, 0.45))
    fi = FEATURES.index("d_roll_winrate")
    report["baselines"]["better_recent_form"] = _metrics(yte, np.where(Xte[:, fi] > 0, 0.6, 0.4))

    lr = LogisticRegression(max_iter=2000).fit(Xtr_s, ytr)
    report["models"]["logreg"] = _metrics(yte, lr.predict_proba(Xte_s)[:, 1])
    report["models"]["logreg"]["coefs"] = {f: round(float(c), 3) for f, c in zip(FEATURES, lr.coef_[0])}
    gb = GradientBoostingClassifier(random_state=0, n_estimators=120, max_depth=2,
                                    learning_rate=0.05).fit(Xtr, ytr)
    report["models"]["gbm"] = _metrics(yte, gb.predict_proba(Xte)[:, 1])
    report["models"]["gbm"]["importances"] = {f: round(float(c), 3) for f, c in zip(FEATURES, gb.feature_importances_)}

    # logreg on player-XI features ONLY (isolate CA player-data signal)
    xi_idx = [FEATURES.index(f) for f in ("d_xi_bat_sr", "d_xi_bat_avg", "d_xi_bowl_econ", "d_xi_bowl_wpm")]
    lr_xi = LogisticRegression(max_iter=2000).fit(Xtr_s[:, xi_idx], ytr)
    report["models"]["logreg_xi_only"] = _metrics(yte, lr_xi.predict_proba(Xte_s[:, xi_idx])[:, 1])

    out = Path(PROCESSED_DATA_DIR) / "cricketarchive"; out.mkdir(parents=True, exist_ok=True)
    (out / "thread_a_report.json").write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 72)
    print("THREAD A - SCORECARD SIGNAL CHECK (predict match winner)")
    print("=" * 72)
    print(f"  samples: {n}  (train {len(ytr)} / test {len(yte)})   "
          f"test base-rate(t1 win): {report['test_base_rate_t1_win']}")
    print(f"\n  {'model/baseline':<22} {'acc':>7} {'logloss':>9} {'brier':>7} {'auc':>7}")
    print("  " + "-" * 56)
    for name, mtr in {**report["baselines"], **report["models"]}.items():
        print(f"  {name:<22} {mtr['accuracy']:>7} {mtr['log_loss']:>9} {mtr['brier']:>7} "
              f"{(mtr['auc'] if mtr['auc'] is not None else 0):>7}")
    print("\n  logreg coefficients (standardised, |.|-sorted):")
    for f, cval in sorted(report["models"]["logreg"]["coefs"].items(), key=lambda kv: -abs(kv[1])):
        print(f"    {f:<18} {cval:+.3f}")
    print("\n  report -> data/processed/cricketarchive/thread_a_report.json")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
