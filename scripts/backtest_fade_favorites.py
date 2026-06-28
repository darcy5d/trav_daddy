#!/usr/bin/env python3
"""Wave 6 pre-work (W3): backtest favourites-only and fade-the-underdog.

Validates the two new paper strategies against our own logged paper bet
history, BEFORE forward-testing them live in paper. Everything is computed
hold-to-settlement from `match_settle_outcome` (the side's eventual result),
so the cashout-simulation fantasy that inflated headline paper P&L is excluded
and we see the underlying signal.

Two experiments (mirroring src/integrations/polymarket/paper_strategies.py):

  favourites-only  -- take the source strategy's picks but only when the backed
                      side is a market favourite (entry price >= fav_floor).
                      Sized like the original (uses logged stake).

  fade-underdog    -- when the source strategy backed an UNDERDOG (entry price
                      < fade_max), back the OPPOSITE (favourite) side instead.
                      The faded bet wins iff the original underdog pick lost.
                      Flat-sized (ROI is stake-independent), favourite entry
                      price approximated as (1 - underdog_price) for the 2-way
                      moneyline (documented approximation).

Read-only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_db_connection


def _hold_pnl(entry_price: float, stake: float, won: int, fee: float) -> float:
    """Hold-to-settlement P&L for a BUY at entry_price settling 1.0/0.0.

    Mirrors reconcile._compute_pnl_for_settled_bet: fee charged on gross
    proceeds, not on stake.
    """
    if entry_price <= 0 or entry_price >= 1:
        return 0.0
    shares = stake / entry_price
    if won:
        gross = shares * 1.0
        return gross - stake - gross * fee
    return -stake


def _summary(rows, label, fee, flat_stake=None, fade=False):
    """rows: list of (entry_price, stake, won, edge_pp). Returns metrics dict."""
    n = 0
    wins = 0
    pnl = 0.0
    staked = 0.0
    for entry_price, stake, won, _edge in rows:
        if fade:
            # Back the favourite (opposite side). Faded bet wins iff the
            # original underdog pick LOST. Favourite price approx 1 - p.
            fav_price = 1.0 - entry_price
            s = flat_stake if flat_stake is not None else stake
            fade_won = 0 if won else 1
            p = _hold_pnl(fav_price, s, fade_won, fee)
            pnl += p
            staked += s
            wins += fade_won
        else:
            s = stake
            p = _hold_pnl(entry_price, s, won, fee)
            pnl += p
            staked += s
            wins += int(won)
        n += 1
    roi = 100.0 * pnl / staked if staked else 0.0
    wr = 100.0 * wins / n if n else 0.0
    return {"label": label, "n": n, "wins": wins, "win_pct": wr,
            "pnl": pnl, "staked": staked, "roi": roi}


def _print_row(m):
    print(
        f"  {m['label']:30} n={m['n']:>4}  win%={m['win_pct']:>5.1f}  "
        f"pnl=${m['pnl']:>9.2f}  staked=${m['staked']:>9.2f}  ROI={m['roi']:>7.2f}%"
    )


def run(source_strategy: str, fav_floor: float, fade_max: float,
        fee: float, flat_stake: float) -> None:
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT market_price_at_proposal, size_usdc, match_settle_outcome, edge_pp
            FROM bet_ledger
            WHERE bet_kind = 'paper'
              AND strategy_label = ?
              AND match_settle_outcome IS NOT NULL
              AND market_price_at_proposal IS NOT NULL
              AND size_usdc IS NOT NULL
            """,
            (source_strategy,),
        )
        rows = [
            (float(r[0]), float(r[1]), int(r[2]), float(r[3]) if r[3] is not None else 0.0)
            for r in cur.fetchall()
        ]

    if not rows:
        print(f"\nNo settled paper bets found for source strategy '{source_strategy}'.\n")
        return

    print("\n" + "=" * 84)
    print(f"FADE / FAVOURITES BACKTEST  (source={source_strategy}, fee={fee:.0%}, "
          f"hold-to-settle)")
    print("=" * 84)
    print(f"  Universe: {len(rows)} settled paper picks from {source_strategy}\n")

    # --- Baseline: the source strategy held to settlement (the real signal) ---
    print("Baseline - source strategy, hold-to-settle (no cashout fantasy):")
    _print_row(_summary(rows, f"{source_strategy} (all)", fee))

    # --- Experiment 1: favourites-only ---
    print("\nExperiment 1 - FAVOURITES-ONLY (back model pick only when price >= "
          f"{fav_floor:.2f}):")
    fav_rows = [r for r in rows if r[0] >= fav_floor]
    dog_rows = [r for r in rows if r[0] < fav_floor]
    _print_row(_summary(fav_rows, f"favourites (price>={fav_floor:.2f})", fee))
    _print_row(_summary(dog_rows, f"underdogs (price<{fav_floor:.2f}) [excluded]", fee))

    # --- Experiment 2: fade-the-underdog ---
    print("\nExperiment 2 - FADE-THE-UNDERDOG (when model picks price < "
          f"{fade_max:.2f}, back the favourite instead, flat ${flat_stake:.0f}):")
    fade_universe = [r for r in rows if r[0] < fade_max]
    _print_row(
        _summary(fade_universe, "model underdog picks (as-is)", fee)
    )
    _print_row(
        _summary(fade_universe, "FADED (back favourite)", fee,
                 flat_stake=flat_stake, fade=True)
    )

    # --- Price-bucket detail on the baseline (where the edge actually is) ---
    print("\nHold-to-settle ROI by entry-price bucket (baseline universe):")
    buckets = [
        ("<0.20", 0.0, 0.20), ("0.20-0.35", 0.20, 0.35), ("0.35-0.50", 0.35, 0.50),
        ("0.50-0.65", 0.50, 0.65), ("0.65-0.80", 0.65, 0.80), (">=0.80", 0.80, 1.01),
    ]
    for name, lo, hi in buckets:
        br = [r for r in rows if lo <= r[0] < hi]
        if br:
            _print_row(_summary(br, f"bucket {name}", fee))

    print("\n" + "=" * 84 + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-strategy", default="v3_marg_3pp",
                    help="paper strategy whose logged picks form the universe")
    ap.add_argument("--fav-floor", type=float, default=0.65,
                    help="favourites-only: minimum backed entry price")
    ap.add_argument("--fade-max", type=float, default=0.50,
                    help="fade: only fade picks below this entry price")
    ap.add_argument("--fee", type=float, default=0.02,
                    help="fee rate on gross proceeds (paper convention 0.02)")
    ap.add_argument("--flat-stake", type=float, default=50.0,
                    help="flat stake per faded bet")
    args = ap.parse_args()
    run(args.source_strategy, args.fav_floor, args.fade_max, args.fee,
        args.flat_stake)
    return 0


if __name__ == "__main__":
    sys.exit(main())
