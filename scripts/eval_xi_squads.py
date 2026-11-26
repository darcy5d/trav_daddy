#!/usr/bin/env python3
"""Does ranking within the ESPN squad beat the plain XI selector?

ESPN keeps no squads for past matches, so this is a prospective test on the
events scripts/snapshot_espn_squads.py captured: every side with a squad
fetched before the start and a played XI recorded after it. Truth is ESPN's
played XI mapped to player_ids (sides with fewer than 10 mappable players are
dropped). Our team is the one whose recent declared XIs share most players
with the truth. Both methods use the installed selector (fitted only on
matches before these events) and only matches ingested by the snapshot time.

A segment passes, as for the selector itself, when the squad-aware recall
beats the plain selector's with the 95% bootstrap interval of the paired
difference above zero on at least 100 sides. --write-gate backs up
data/models/xi_selector/ to the Elements drive and then writes
squad_gate.json there; paper scans use squads only for passing segments.

    nice -n 10 python scripts/eval_xi_squads.py [--write-gate]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.train_xi_selector import MIN_HOLDOUT_SIDES, boot_ci, recall  # noqa: E402
from src.integrations.polymarket import espn_squads as es  # noqa: E402
from src.models.xi_selector import (  # noqa: E402
    ARTIFACT_DIR, DECLARED_XI_PATH, History, load_selector, load_sides,
)

BACKUP_ROOT = Path("/Volumes/Elements/indias_dad_backups")


def identify_team(hist: History, truth: set, gender: str, d: pd.Timestamp):
    """Team most of the truth XI played for in the last year (at least 5 of them)."""
    count: dict = {}
    for p in truth:
        teams = {a[2] for a in hist.apps.get(p, [])
                 if a[3] == gender and a[0] < d and a[1] <= d and (d - a[0]).days <= 365}
        for t in teams:
            count[t] = count.get(t, 0) + 1
    best = max(count.items(), key=lambda kv: kv[1], default=(None, 0))
    return best[0] if best[1] >= 5 else None


def evaluate(db: str, decl: str, root: Path) -> pd.DataFrame:
    sel = load_selector()
    sides = load_sides(db, pd.read_pickle(decl), since="2022-01-01")
    id_map = es.cricinfo_player_map(db, es.refresh_register())
    rows, hists = [], {}
    for rec in es.iter_events(root):
        squads, played = es.pre_start_squads(rec), rec.get("played_xi") or {}
        if not squads or not played:
            continue
        # Start-of-day of the first snapshot: only matches ingested by then count as known.
        snap_day = pd.Timestamp(min(s["fetched_at"] for s in rec["snapshots"])).tz_localize(None).normalize()
        if snap_day not in hists:
            known = sides[sides.created <= snap_day]
            hists[snap_day] = History(known, known.created)
        hist = hists[snap_day]
        d = pd.Timestamp(rec["start"]).tz_localize(None).normalize()
        for team, xi_ids in played.items():
            if team not in squads:
                continue
            truth = set(es.map_ids(xi_ids, id_map))
            squad = es.map_ids(squads[team], id_map)
            if len(truth) < 10 or len(squad) < es.MIN_SQUAD:
                continue
            team_id = identify_team(hist, truth, rec["gender"], d)
            if team_id is None:
                continue
            plain, info = sel.predict(hist, team_id, rec["gender"], rec["fmt"], d)
            if not plain:
                continue
            with_sq, _ = sel.predict_with_squad(hist, team_id, rec["gender"], rec["fmt"], d, squad)
            g = hist.groups[(team_id, rec["gender"])]
            last = g[(g.d < d) & (g.fmt == rec["fmt"])].xi.head(1).tolist()
            rows.append({
                "espn_id": rec["espn_id"], "team": team, "team_id": team_id, "d": d,
                "seg": f"{rec['fmt']}_{rec['gender']}", "stale": info["stale_days"],
                "squad_ceiling": len(truth & set(squad)) / len(truth),
                "r_last": recall(last[0] if last else [], truth),
                "r_sel": recall(plain, truth),
                "r_squad": recall(with_sq or plain, truth),
            })
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame):
    out, gate = {}, {}
    for seg, sub in list(df.groupby("seg")) + [("all", df)]:
        ci = boot_ci(sub.r_squad - sub.r_sel) if len(sub) > 1 else None
        out[seg] = {"n": int(len(sub)), "recall_last_match": round(float(sub.r_last.mean()), 4),
                    "recall_selector": round(float(sub.r_sel.mean()), 4),
                    "recall_selector_squad": round(float(sub.r_squad.mean()), 4),
                    "squad_ceiling": round(float(sub.squad_ceiling.mean()), 4), "diff_ci": ci}
        if seg != "all":
            gate[seg] = bool(len(sub) >= MIN_HOLDOUT_SIDES and ci is not None and ci[1] > 0)
    return out, gate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "cricket.db"))
    ap.add_argument("--declared-xi", default=str(DECLARED_XI_PATH))
    ap.add_argument("--root", default=str(es.SNAPSHOT_DIR))
    ap.add_argument("--write-gate", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    df = evaluate(args.db, args.declared_xi, Path(args.root))
    if df.empty:
        print("no evaluable sides yet (need pre-start squads plus played XIs)")
        return
    res, gate = summarise(df)
    report = {"gate": gate, "results": res, "n_events": int(df.espn_id.nunique()),
              "first": str(df.d.min().date()), "last": str(df.d.max().date()),
              "built_at": pd.Timestamp.now("UTC").isoformat(), "secs": round(time.time() - t0, 1)}
    print(json.dumps(report, indent=1))
    if not args.write_gate:
        return
    if not BACKUP_ROOT.is_dir():
        sys.exit(f"{BACKUP_ROOT} not mounted; refusing to write the gate without a backup")
    dst = BACKUP_ROOT / f"xi_selector_{pd.Timestamp.now('UTC').strftime('%Y%m%dT%H%M%SZ')}"
    shutil.copytree(ARTIFACT_DIR, dst)
    tmp = ARTIFACT_DIR / "squad_gate.json.tmp"
    tmp.write_text(json.dumps(report, indent=1))
    os.replace(tmp, ARTIFACT_DIR / "squad_gate.json")
    print(f"backed up to {dst}; wrote {ARTIFACT_DIR / 'squad_gate.json'}")


if __name__ == "__main__":
    main()
