#!/usr/bin/env python3
"""Ledger-only analyses for the simplification proposal (Tests 2, 3, 4).

Authoritative actuals from bet_ledger.pnl_realised_usdc. No network.
Temporary diagnostic; safe to delete.
"""
from __future__ import annotations
import sqlite3
from collections import defaultdict

DB = "cricket.db"
OUTAGE = ("2026-06-01", "2026-06-06")   # inclusive blow-up week
STABLE = ("2026-06-07", "2026-06-27")   # post-stabilisation


def conn():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


def fmt(x):
    return f"{x:,.1f}"


def seg_label(c, fixture_key, match_id, tier_cache):
    prefix = (fixture_key or "").split("-")[0].lower()
    if prefix == "crint":
        t1 = t2 = None
        row = c.execute(
            "SELECT team1_id, team2_id FROM matches WHERE match_id=?", (match_id,)
        ).fetchone()
        if row:
            for tid in (row["team1_id"], row["team2_id"]):
                if tid not in tier_cache:
                    tr = c.execute("SELECT tier FROM teams WHERE team_id=?", (tid,)).fetchone()
                    tier_cache[tid] = tr["tier"] if tr else None
            t1 = tier_cache.get(row["team1_id"])
            t2 = tier_cache.get(row["team2_id"])
        both_full = (t1 == 1 and t2 == 1)
        return "crint_FULLvFULL" if both_full else "crint_ASSOCIATE(low-data)"
    if prefix in ("cricipl", "cricmlc", "cricpsl"):
        return "franchise(ipl/mlc/psl)"
    if prefix.startswith("crict20blast"):
        return "county_T20_blast"
    return f"other:{prefix}"


def window(d, lo, hi):
    return lo <= d <= hi


def main():
    c = conn()
    rows = c.execute(
        """
        SELECT fixture_key, match_id, side_label, fill_price, fill_size_usdc,
               cashout_price, cashout_reason, cashout_triggered_at,
               pnl_realised_usdc AS pnl, settle_outcome, match_winner,
               date(settled_at) AS d
        FROM bet_ledger
        WHERE COALESCE(bet_kind,'real')='real' AND status='settled'
          AND fill_price IS NOT NULL AND fill_size_usdc IS NOT NULL
          AND pnl_realised_usdc IS NOT NULL
          AND proposed_at >= datetime('now','-60 days')
        """
    ).fetchall()

    tier_cache: dict = {}

    # ---- Test 2: segment ROI by window ----
    # agg[seg][window] = [n, staked, pnl]
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0.0, 0.0]))
    for r in rows:
        seg = seg_label(c, r["fixture_key"], r["match_id"], tier_cache)
        staked = r["fill_size_usdc"] or 0.0
        pnl = r["pnl"] or 0.0
        for wname, (lo, hi) in (("full60d", ("2026-01-01", "2026-12-31")),
                                ("outage", OUTAGE), ("stable", STABLE)):
            if window(r["d"], lo, hi):
                a = agg[seg][wname]
                a[0] += 1; a[1] += staked; a[2] += pnl

    print("=" * 86)
    print("TEST 2 — Segment ROI (PnL / dollars staked). ROI<0 => any Kelly multiple loses.")
    print("=" * 86)
    hdr = f"{'segment':<28}{'window':<9}{'n':>5}{'staked$':>11}{'pnl$':>10}{'ROI%':>9}"
    print(hdr)
    for seg in sorted(agg):
        for wname in ("full60d", "outage", "stable"):
            n, st, pnl = agg[seg][wname]
            if n == 0:
                continue
            roi = (pnl / st * 100) if st else 0.0
            print(f"{seg:<28}{wname:<9}{n:>5}{fmt(st):>11}{fmt(pnl):>10}{roi:>8.1f}%")
        print("-" * 86)

    print("\nKelly-sweep implication for crint_ASSOCIATE in STABLE window")
    a = agg["crint_ASSOCIATE(low-data)"]["stable"]
    if a[1]:
        roi = a[2] / a[1]
        cur_mult = 0.5  # historical half-Kelly baseline these were placed under
        print(f"  realised ROI = {roi*100:.1f}% on ${fmt(a[1])} staked ({a[0]} bets)")
        for mult in (0.05, 0.10, 0.25, 0.50):
            scale = mult / cur_mult
            print(f"   {int(mult*100):>3}% Kelly -> staked ~${fmt(a[1]*scale)} | est PnL ~${fmt(a[2]*scale)}")

    # ---- Test 3: liquidation-fill quality (stop rows) ----
    print("\n" + "=" * 86)
    print("TEST 3 — Stop-loss fill quality: exit price vs 0.20 floor vs settlement")
    print("=" * 86)
    stops = [r for r in rows if (r["cashout_reason"] or "").startswith("stop")
             and r["cashout_price"] is not None]
    below = at = above = won_after = 0
    cps = []
    for r in stops:
        cp = r["cashout_price"]
        cps.append(cp)
        if cp < 0.195:
            below += 1
        elif cp <= 0.205:
            at += 1
        else:
            above += 1
        # did the stopped side go on to win?
        won = 1 if (r["match_winner"] and r["side_label"] and
                    r["match_winner"].strip() == r["side_label"].strip()) else 0
        won_after += won
    n = len(stops)
    if n:
        cps.sort()
        med = cps[n // 2]
        print(f"  stop events: {n}")
        print(f"  exit price:  min={min(cps):.3f}  median={med:.3f}  max={max(cps):.3f}")
        print(f"  vs 0.20 floor:  below={below}  ~at={at}  above={above}")
        print(f"  stopped side WON anyway (false stop): {won_after}/{n} = {won_after/n*100:.0f}%")
        print("  (false-stop = we sold a position that would have paid $1)")

    # ---- Test 4: exclude-outage rebaseline ----
    print("\n" + "=" * 86)
    print("TEST 4 — Re-baseline with outage (Jun1-6) masked out")
    print("=" * 86)
    def totals(pred):
        n = 0; pnl = 0.0
        for r in rows:
            if pred(r):
                n += 1; pnl += r["pnl"] or 0.0
        return n, pnl
    n_all, p_all = totals(lambda r: True)
    n_out, p_out = totals(lambda r: window(r["d"], *OUTAGE))
    n_ex, p_ex = totals(lambda r: not window(r["d"], *OUTAGE))
    print(f"  full 60d           : {n_all:>4} bets   ${fmt(p_all)}")
    print(f"  outage Jun1-6      : {n_out:>4} bets   ${fmt(p_out)}")
    print(f"  ex-outage (rest)   : {n_ex:>4} bets   ${fmt(p_ex)}")
    # by exit reason ex-outage
    by_reason = defaultdict(lambda: [0, 0.0])
    for r in rows:
        if window(r["d"], *OUTAGE):
            continue
        reason = r["cashout_reason"] or ("held_to_settle" if not r["cashout_triggered_at"] else "other")
        by_reason[reason][0] += 1
        by_reason[reason][1] += r["pnl"] or 0.0
    print("  ex-outage by exit reason:")
    for reason, (n, pnl) in sorted(by_reason.items(), key=lambda kv: kv[1][1]):
        print(f"    {reason:<18}{n:>4} bets   ${fmt(pnl)}")

    c.close()


if __name__ == "__main__":
    main()
