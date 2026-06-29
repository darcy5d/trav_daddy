#!/usr/bin/env python3
"""Thread B: validate the synthetic ball-by-ball generator against real data.

For every innings that has REAL CA ball-by-ball, generate a synthetic sequence
from the same innings' scorecard and compare. This quantifies how faithfully an
aggregate-consistent generator reproduces what the scorecard does NOT encode:
intra-innings dynamics (powerplay/middle/death scoring, wicket timing).

Marginals (overall outcome distribution) should match closely by construction;
the phase-rate gap is the honest measure of synthetic's limits.

READ-ONLY on ca_archive.db. Prints metrics + writes a JSON report.

Usage:
    venv311/bin/python scripts/ca_thread_b_validate.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROCESSED_DATA_DIR
from src.api.cricketarchive import store
from src.api.cricketarchive.synthetic import generate_innings


def _overs_to_balls(o):
    if o is None:
        return 0
    o = str(o).strip()
    if "." in o:
        a, _, b = o.partition(".")
        return (int(a) * 6 + int(b)) if a.isdigit() and b.isdigit() else 0
    return int(o) * 6 if o.isdigit() else 0


def _phase(over: int) -> str:
    if over <= 5:
        return "powerplay"
    if over <= 14:
        return "middle"
    return "death"


OUTCOMES = [0, 1, 2, 3, 4, 6]


def _summary(deliveries):
    """deliveries: list of (runs_batter, over, is_wicket) -> distribution stats."""
    n = len(deliveries)
    if not n:
        return None
    oc = Counter()
    phase_runs = Counter(); phase_balls = Counter(); wk = 0
    for runs, over, isw in deliveries:
        oc[runs if runs in OUTCOMES else (6 if runs >= 6 else runs)] += 1
        ph = _phase(over)
        phase_runs[ph] += runs; phase_balls[ph] += 1
        wk += int(isw)
    dist = {str(v): oc.get(v, 0) / n for v in OUTCOMES}
    phase_rate = {ph: (phase_runs[ph] / phase_balls[ph] if phase_balls[ph] else 0.0)
                  for ph in ("powerplay", "middle", "death")}
    return {"n": n, "dist": dist, "wicket_rate": wk / n, "phase_rate": phase_rate,
            "runs_per_ball": sum(r for r, _, _ in deliveries) / n}


def main() -> int:
    real_pool, synth_pool = [], []
    innings_used = 0

    with store.get_connection() as c:
        innings = c.execute("""
            SELECT i.scorecard_id, i.innings_number, i.fall_of_wickets
            FROM ca_innings i JOIN ca_matches m ON m.scorecard_id=i.scorecard_id
            WHERE m.has_ball_by_ball=1 AND i.innings_number IN (1,2)
        """).fetchall()

        for inn in innings:
            sid, no = inn["scorecard_id"], inn["innings_number"]
            # legal balls only (exclude wides/no-balls) for a fair comparison vs
            # the generator, which models legal deliveries faced by batters.
            real = [(d["runs_batter"] or 0, d["over_number"] or 0, bool(d["is_wicket"]))
                    for d in c.execute("""SELECT runs_batter, over_number, is_wicket
                                          FROM ca_deliveries WHERE scorecard_id=? AND innings_number=?
                                          AND COALESCE(extras_wides,0)=0 AND COALESCE(extras_noballs,0)=0
                                          ORDER BY seq""", (sid, no))]
            if len(real) < 30:
                continue
            batting = [{"ca_id": b["batter_ca_id"], "runs": b["runs"], "balls": b["balls"],
                        "fours": b["fours"], "sixes": b["sixes"],
                        "dismissed": (b["dismissal"] or "").strip().lower()
                        not in ("", "not out", "did not bat")}
                       for b in c.execute("""SELECT batter_ca_id, runs, balls, fours, sixes, dismissal
                                             FROM ca_batting WHERE scorecard_id=? AND innings_number=?
                                             ORDER BY position""", (sid, no))]
            bowling = [{"ca_id": bw["bowler_ca_id"], "balls": _overs_to_balls(bw["overs"])}
                       for bw in c.execute("""SELECT bowler_ca_id, overs FROM ca_bowling
                                              WHERE scorecard_id=? AND innings_number=?""", (sid, no))]
            synth = generate_innings(batting, bowling, inn["fall_of_wickets"] or "", seed=hash(sid) & 0xffff)
            if len(synth) < 30:
                continue
            real_pool.extend(real)
            synth_pool.extend((d.runs_batter, d.over_number, d.is_wicket) for d in synth)
            innings_used += 1

    rs, ss = _summary(real_pool), _summary(synth_pool)
    tv = 0.5 * sum(abs(rs["dist"][k] - ss["dist"][k]) for k in rs["dist"])
    phase_mae = np.mean([abs(rs["phase_rate"][p] - ss["phase_rate"][p])
                         for p in rs["phase_rate"]])
    report = {"innings_used": innings_used, "real_balls": rs["n"], "synth_balls": ss["n"],
              "outcome_tv_distance": round(tv, 4),
              "phase_rate_mae": round(float(phase_mae), 4),
              "real": rs, "synth": ss}
    out = Path(PROCESSED_DATA_DIR) / "cricketarchive"; out.mkdir(parents=True, exist_ok=True)
    (out / "thread_b_report.json").write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 72)
    print("THREAD B - SYNTHETIC vs REAL BALL-BY-BALL FIDELITY")
    print("=" * 72)
    print(f"  paired innings: {innings_used}   real balls: {rs['n']:,}   synth balls: {ss['n']:,}")
    print(f"\n  outcome distribution (P per ball):")
    print(f"    {'outcome':<8} {'real':>8} {'synth':>8}")
    for k in ("0", "1", "2", "3", "4", "6"):
        print(f"    {k:<8} {rs['dist'][k]:>8.3f} {ss['dist'][k]:>8.3f}")
    print(f"    {'wicket':<8} {rs['wicket_rate']:>8.3f} {ss['wicket_rate']:>8.3f}")
    print(f"    outcome total-variation distance: {tv:.4f}  (0=identical)")
    print(f"\n  phase run-rate (runs/ball):")
    print(f"    {'phase':<10} {'real':>8} {'synth':>8}")
    for p in ("powerplay", "middle", "death"):
        print(f"    {p:<10} {rs['phase_rate'][p]:>8.3f} {ss['phase_rate'][p]:>8.3f}")
    print(f"    phase-rate MAE: {phase_mae:.4f}")
    print(f"\n  interpretation: low TV distance => marginals reproduced; phase-rate")
    print(f"  gap => dynamics the scorecard does not encode (synthetic's true limit).")
    print(f"\n  report -> data/processed/cricketarchive/thread_b_report.json")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
