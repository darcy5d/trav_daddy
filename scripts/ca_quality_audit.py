#!/usr/bin/env python3
"""Thread C: CricketArchive vs Cricsheet (cricket.db) overlap data-quality audit.

READ-ONLY on both databases. Answers "does CA agree with our trusted Cricsheet
data, and can it be trusted / eventually replace it?" by matching overlapping
matches and diffing innings totals (runs + wickets).

Join key: match date + best team-set similarity (token Jaccard), so it is robust
to franchise renames (Bangalore/Bengaluru, Kings XI/Punjab Kings, Sunrisers/...).

Outputs a summary to stdout and a JSON report to
data/processed/cricketarchive/audit_report.json. Never writes either DB.

Usage:
    venv311/bin/python scripts/ca_quality_audit.py
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATABASE_PATH, PROCESSED_DATA_DIR, CRICKETARCHIVE_CONFIG

_STOP = {"the"}


def _norm_tokens(name: str) -> set:
    name = re.sub(r"[^a-z0-9 ]", " ", (name or "").lower())
    return {t for t in name.split() if t and t not in _STOP}


def _team_set_similarity(ca_teams: list, db_teams: list) -> float:
    """Best bipartite token-Jaccard between two 2-team line-ups (0..1)."""
    def jac(a, b):
        A, B = _norm_tokens(a), _norm_tokens(b)
        return len(A & B) / len(A | B) if (A | B) else 0.0
    if len(ca_teams) < 2 or len(db_teams) < 2:
        return 0.0
    a1, a2 = ca_teams[0], ca_teams[1]
    b1, b2 = db_teams[0], db_teams[1]
    straight = jac(a1, b1) + jac(a2, b2)
    swapped = jac(a1, b2) + jac(a2, b1)
    return max(straight, swapped) / 2.0


def load_ca_matches(ca_db: Path) -> list:
    c = sqlite3.connect(str(ca_db)); c.row_factory = sqlite3.Row
    rows = c.execute("""
        SELECT scorecard_id, match_date, competition, team1_name, team2_name,
               has_ball_by_ball
        FROM ca_matches WHERE match_date IS NOT NULL
    """).fetchall()
    out = []
    for r in rows:
        inns = c.execute("""
            SELECT innings_number, total_runs, total_wickets
            FROM ca_innings WHERE scorecard_id=? ORDER BY innings_number
        """, (r["scorecard_id"],)).fetchall()
        out.append({
            "scorecard_id": r["scorecard_id"], "date": r["match_date"],
            "competition": r["competition"], "has_bbb": r["has_ball_by_ball"],
            "teams": [r["team1_name"], r["team2_name"]],
            "innings": {i["innings_number"]: (i["total_runs"], i["total_wickets"]) for i in inns},
        })
    c.close()
    return out


def load_db_matches_by_date(start="2018-09-01") -> dict:
    c = sqlite3.connect(str(DATABASE_PATH)); c.row_factory = sqlite3.Row
    rows = c.execute("""
        SELECT m.match_id, m.date, t1.name AS team1, t2.name AS team2
        FROM matches m
        JOIN teams t1 ON t1.team_id=m.team1_id
        JOIN teams t2 ON t2.team_id=m.team2_id
        WHERE m.match_type='T20' AND m.date>=?
    """, (start,)).fetchall()
    by_date = defaultdict(list)
    for r in rows:
        inns = c.execute("""
            SELECT innings_number, total_runs, total_wickets
            FROM innings WHERE match_id=? ORDER BY innings_number
        """, (r["match_id"],)).fetchall()
        by_date[r["date"]].append({
            "match_id": r["match_id"], "teams": [r["team1"], r["team2"]],
            "innings": {i["innings_number"]: (i["total_runs"], i["total_wickets"]) for i in inns},
        })
    c.close()
    return by_date


def main() -> int:
    ca_db = Path(CRICKETARCHIVE_CONFIG["archive_db_path"])
    ca_matches = load_ca_matches(ca_db)
    db_by_date = load_db_matches_by_date()

    stats = {
        "ca_matches": len(ca_matches), "matched": 0, "unmatched": 0,
        "innings_compared": 0, "runs_agree": 0, "wkts_agree": 0,
        "bbb_matches": sum(1 for m in ca_matches if m["has_bbb"]),
    }
    mismatches = []
    unmatched = []

    for m in ca_matches:
        candidates = db_by_date.get(m["date"], [])
        best, best_sim = None, 0.0
        for d in candidates:
            sim = _team_set_similarity(m["teams"], d["teams"])
            if sim > best_sim:
                best, best_sim = d, sim
        if not best or best_sim < 0.5:
            stats["unmatched"] += 1
            unmatched.append({"scorecard_id": m["scorecard_id"], "date": m["date"],
                              "teams": m["teams"], "best_sim": round(best_sim, 2)})
            continue
        stats["matched"] += 1
        for inn_no, (ca_r, ca_w) in m["innings"].items():
            if inn_no not in best["innings"]:
                continue
            db_r, db_w = best["innings"][inn_no]
            stats["innings_compared"] += 1
            r_ok = (ca_r == db_r)
            w_ok = (ca_w == db_w)
            stats["runs_agree"] += int(r_ok)
            stats["wkts_agree"] += int(w_ok)
            if not (r_ok and w_ok):
                mismatches.append({
                    "scorecard_id": m["scorecard_id"], "date": m["date"],
                    "teams": m["teams"], "innings": inn_no,
                    "ca": [ca_r, ca_w], "db": [db_r, db_w],
                })

    ic = stats["innings_compared"] or 1
    report = {
        "stats": stats,
        "runs_agree_pct": round(100 * stats["runs_agree"] / ic, 2),
        "wkts_agree_pct": round(100 * stats["wkts_agree"] / ic, 2),
        "match_rate_pct": round(100 * stats["matched"] / (len(ca_matches) or 1), 2),
        "mismatch_samples": mismatches[:40],
        "unmatched_samples": unmatched[:40],
    }
    out_dir = Path(PROCESSED_DATA_DIR) / "cricketarchive"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit_report.json").write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 72)
    print("CRICKETARCHIVE vs CRICSHEET (cricket.db) - OVERLAP AUDIT")
    print("=" * 72)
    print(f"  CA matches:           {stats['ca_matches']}")
    print(f"  with ball-by-ball:    {stats['bbb_matches']}")
    print(f"  matched to cricket.db:{stats['matched']}  ({report['match_rate_pct']}%)")
    print(f"  unmatched:            {stats['unmatched']}")
    print(f"  innings compared:     {stats['innings_compared']}")
    print(f"  runs agreement:       {stats['runs_agree']}/{stats['innings_compared']}  ({report['runs_agree_pct']}%)")
    print(f"  wkts agreement:       {stats['wkts_agree']}/{stats['innings_compared']}  ({report['wkts_agree_pct']}%)")
    print(f"  innings mismatches:   {len(mismatches)}")
    if mismatches:
        print("\n  sample mismatches:")
        for mm in mismatches[:8]:
            print(f"    {mm['date']} {mm['teams'][0]} v {mm['teams'][1]} inn{mm['innings']}: "
                  f"CA={mm['ca']} DB={mm['db']}")
    print("\n  report -> data/processed/cricketarchive/audit_report.json")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
