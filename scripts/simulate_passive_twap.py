#!/usr/bin/env python3
"""Wave 6 pre-work (W5, Phase A): simulate a post-only / never-chase TWAP.

Our TWAP currently escalates limit prices from a discount up toward
`max_acceptable_price`, i.e. it chases the book up. The W1 analysis showed even
maker fills still pay +3.1c of slippage and the maker book is -8.9%. This asks:

    If we had been strictly passive - never bidding above the screen mid
    (`market_price_at_plan`) - what would ROI and filled volume have been?

Method (chunk-level, hold-to-settle):
  * Each FILLED order_chunk is "passive" if its limit_price <= the plan's
    market_price_at_plan (we never crossed above the mid we anchored on).
  * Per-chunk hold P&L from chunk.fill_price + the plan's bet outcome
    (match_settle_outcome), fee on proceeds.
  * Compare ALL filled chunks vs the PASSIVE-only subset: ROI, win%, and the
    notional we'd forgo by refusing to chase.

If the passive subset has materially better ROI without forgoing too much
volume, capping the chase is worth flipping live (behind BETTING_TWAP_MAX_CHASE_PP).

Read-only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_db_connection


def _hold_pnl(entry_price: float, stake: float, won: int, fee: float) -> float:
    if entry_price <= 0 or entry_price >= 1:
        return 0.0
    shares = stake / entry_price
    if won:
        gross = shares * 1.0
        return gross - stake - gross * fee
    return -stake


def run(fee: float, chase_tol_pp: float) -> None:
    tol = chase_tol_pp / 100.0
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT c.limit_price, c.fill_price, c.fill_size_usdc,
                   p.market_price_at_plan, b.match_settle_outcome
            FROM order_chunks c
            JOIN order_plans p ON p.plan_id = c.plan_id
            JOIN bet_ledger b ON b.bet_id = p.bet_ledger_id
            WHERE c.status = 'filled'
              AND c.fill_price IS NOT NULL
              AND c.fill_size_usdc IS NOT NULL
              AND p.market_price_at_plan IS NOT NULL
              AND b.match_settle_outcome IS NOT NULL
            """
        )
        rows = cur.fetchall()

    print("\n" + "=" * 80)
    print(f"PASSIVE TWAP SIMULATION  (fee={fee:.0%}, chase_tol={chase_tol_pp:.1f}pp, "
          f"hold-to-settle)")
    print("=" * 80)
    if not rows:
        print("\nNo filled TWAP chunks with plan mid + outcome to analyze.\n")
        return

    all_stats = {"n": 0, "notional": 0.0, "pnl": 0.0, "wins": 0}
    passive_stats = {"n": 0, "notional": 0.0, "pnl": 0.0, "wins": 0}
    chased_stats = {"n": 0, "notional": 0.0, "pnl": 0.0, "wins": 0}

    for limit_price, fill_price, fill_usdc, mid, won in rows:
        limit_price = float(limit_price)
        fill_price = float(fill_price)
        fill_usdc = float(fill_usdc)
        mid = float(mid)
        won = int(won)
        pnl = _hold_pnl(fill_price, fill_usdc, won, fee)

        all_stats["n"] += 1
        all_stats["notional"] += fill_usdc
        all_stats["pnl"] += pnl
        all_stats["wins"] += won

        is_passive = limit_price <= mid + tol
        target = passive_stats if is_passive else chased_stats
        target["n"] += 1
        target["notional"] += fill_usdc
        target["pnl"] += pnl
        target["wins"] += won

    def _line(name, s):
        if s["n"] == 0:
            print(f"  {name:28} n=   0")
            return
        roi = 100.0 * s["pnl"] / s["notional"] if s["notional"] else 0.0
        wr = 100.0 * s["wins"] / s["n"]
        print(f"  {name:28} n={s['n']:>4}  win%={wr:>5.1f}  "
              f"notional=${s['notional']:>9.2f}  pnl=${s['pnl']:>9.2f}  ROI={roi:>7.2f}%")

    print(f"\n  Chunk-level hold-to-settle economics:\n")
    _line("ALL filled chunks", all_stats)
    _line("PASSIVE (limit<=mid)", passive_stats)
    _line("CHASED (limit>mid)", chased_stats)

    # Verdict
    print("\n" + "-" * 80)
    if all_stats["notional"] > 0 and passive_stats["notional"] > 0:
        all_roi = 100.0 * all_stats["pnl"] / all_stats["notional"]
        pas_roi = 100.0 * passive_stats["pnl"] / passive_stats["notional"]
        vol_kept = 100.0 * passive_stats["notional"] / all_stats["notional"]
        print(
            f"  Passive-only ROI {pas_roi:+.2f}% vs all-chunks ROI {all_roi:+.2f}% "
            f"({pas_roi - all_roi:+.2f}pp),\n"
            f"  while keeping {vol_kept:.0f}% of filled notional."
        )
        if pas_roi > all_roi + 1.0:
            print(
                "  -> Capping the chase improves ROI. Worth flipping live behind\n"
                "     BETTING_TWAP_MAX_CHASE_PP (default off) and forward-testing."
            )
        else:
            print(
                "  -> Capping the chase does NOT clearly help here; keep the flag\n"
                "     off and revisit with more data."
            )
    print("=" * 80 + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fee", type=float, default=0.02)
    ap.add_argument(
        "--chase-tol-pp", type=float, default=0.0,
        help="treat limit prices within this many pp above mid as still passive",
    )
    args = ap.parse_args()
    run(args.fee, args.chase_tol_pp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
