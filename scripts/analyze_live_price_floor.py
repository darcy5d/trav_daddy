#!/usr/bin/env python3
"""Wave 6 follow-up: what would a min_market_price floor do to LIVE money?

Uses the REAL bet_ledger with ACTUAL realised P&L (pnl_realised_usdc - which
already bakes in entry slippage, partial fills and cashout reality), not a
hold-to-settle counterfactual. Answers: "if live betting had refused to back
sides priced below X, how much money would we have kept/made?"

This is the real-money decision input for adding a min_market_price floor (and
a deep-longshot exclusion) to the live strategy whitelist.

Read-only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_db_connection


def _fetch(since: str | None, strategies: list[str] | None):
    where = [
        "COALESCE(bet_kind,'real') = 'real'",
        "status = 'settled'",
        "pnl_realised_usdc IS NOT NULL",
        "size_usdc IS NOT NULL",
        "market_price_at_proposal IS NOT NULL",
    ]
    params: list = []
    if since:
        where.append("proposed_at >= ?")
        params.append(since)
    if strategies:
        where.append("strategy_label IN (%s)" % ",".join("?" for _ in strategies))
        params.extend(strategies)
    sql = (
        "SELECT market_price_at_proposal, size_usdc, pnl_realised_usdc, "
        "settle_outcome, match_settle_outcome "
        "FROM bet_ledger WHERE " + " AND ".join(where)
    )
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(sql, params)
        return [
            (float(r[0]), float(r[1]), float(r[2]),
             r[3], r[4])
            for r in cur.fetchall()
        ]


def _agg(rows):
    n = len(rows)
    staked = sum(r[1] for r in rows)
    pnl = sum(r[2] for r in rows)
    roi = 100.0 * pnl / staked if staked else 0.0
    return n, staked, pnl, roi


def run(since: str | None, strategies: list[str] | None) -> None:
    rows = _fetch(since, strategies)
    print("\n" + "=" * 82)
    print("LIVE min_market_price FLOOR ANALYSIS (actual realised P&L)")
    if strategies:
        print(f"strategies: {', '.join(strategies)}")
    print("=" * 82)
    if not rows:
        print("\nNo settled real bets found.\n")
        return

    n, staked, pnl, roi = _agg(rows)
    print(f"\n  Full book: n={n}  staked=${staked:,.2f}  pnl=${pnl:,.2f}  ROI={roi:.2f}%")

    # ROI by entry-price bucket (where does the money actually go?)
    print("\n  Actual realised ROI by entry-price bucket:")
    print(f"    {'bucket':12} {'n':>4} {'staked$':>10} {'pnl$':>10} {'ROI%':>8}")
    print("    " + "-" * 48)
    buckets = [
        ("<0.20", 0.0, 0.20), ("0.20-0.35", 0.20, 0.35), ("0.35-0.50", 0.35, 0.50),
        ("0.50-0.65", 0.50, 0.65), ("0.65-0.80", 0.65, 0.80), (">=0.80", 0.80, 1.01),
    ]
    for name, lo, hi in buckets:
        br = [r for r in rows if lo <= r[0] < hi]
        if br:
            bn, bs, bp, broi = _agg(br)
            print(f"    {name:12} {bn:>4} {bs:>10.2f} {bp:>10.2f} {broi:>8.2f}")

    # Cumulative effect of applying a floor (keep only price >= floor)
    print("\n  Cumulative effect of a min_market_price floor (keep price >= floor):")
    print(f"    {'floor':>7} {'n_kept':>7} {'staked$':>11} {'pnl$':>11} {'ROI%':>8} "
          f"{'pnl_delta$':>12}")
    print("    " + "-" * 60)
    for floor in (0.0, 0.20, 0.35, 0.50, 0.65, 0.80):
        kept = [r for r in rows if r[0] >= floor]
        kn, ks, kp, kroi = _agg(kept)
        delta = kp - pnl
        print(f"    {floor:>7.2f} {kn:>7} {ks:>11.2f} {kp:>11.2f} {kroi:>8.2f} "
              f"{delta:>+12.2f}")

    print(
        "\n  Read: 'pnl_delta$' is how much MORE (or less) we'd have ended with by\n"
        "  refusing every bet priced below the floor. A large positive delta with\n"
        "  a small n_kept drop = a cheap, high-impact live guardrail."
    )
    print("=" * 82 + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since", default=None, help="ISO timestamp lower bound")
    ap.add_argument(
        "--strategies", default=None,
        help="comma list of strategy_label to restrict to (default: all live)",
    )
    args = ap.parse_args()
    strategies = (
        [s.strip() for s in args.strategies.split(",") if s.strip()]
        if args.strategies else None
    )
    run(args.since, strategies)
    return 0


if __name__ == "__main__":
    sys.exit(main())
