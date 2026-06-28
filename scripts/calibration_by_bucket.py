#!/usr/bin/env python3
"""Wave 6 pre-work (W4): model calibration by price/probability bucket.

Companion to scripts/calibration_diagnostic.py (which works off the external
results CSV). This one works off our own bet_ledger, comparing the model's
stated probability for the backed side (`model_prob`) against the realized
hold-to-settle outcome (`match_settle_outcome`). It answers, with our real
proposal history:

  * Is the model well-calibrated, or systematically over/under-confident?
  * Where does mis-calibration concentrate (deep longshots? favourites?)
  * Is the model's probability a better forecast than the market price?

This is the "is the favourites-vs-underdog asymmetry a calibration bug?" check.
Recommendation only - no model changes here.

Read-only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_db_connection


def _fetch(bet_kind: str, since: str | None):
    where = ["match_settle_outcome IS NOT NULL", "model_prob IS NOT NULL",
             "market_price_at_proposal IS NOT NULL"]
    params: list = []
    if bet_kind != "all":
        where.append("bet_kind = ?")
        params.append(bet_kind)
    if since:
        where.append("proposed_at >= ?")
        params.append(since)
    sql = (
        "SELECT model_prob, market_price_at_proposal, match_settle_outcome "
        "FROM bet_ledger WHERE " + " AND ".join(where)
    )
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(sql, params)
        return [(float(r[0]), float(r[1]), int(r[2])) for r in cur.fetchall()]


def _brier(rows, use_market=False) -> float:
    if not rows:
        return float("nan")
    total = 0.0
    for mp, mkt, won in rows:
        p = mkt if use_market else mp
        total += (p - won) ** 2
    return total / len(rows)


def _reliability_table(rows, key_idx: int, title: str, edges):
    """Bin rows by rows[key_idx] into edges; print predicted vs actual."""
    print(f"\n{title}")
    print(f"  {'bucket':14} {'n':>5} {'avg_model_p':>12} {'avg_mkt':>9} "
          f"{'actual_win%':>12} {'model_gap':>10} {'mkt_gap':>9}")
    print("  " + "-" * 76)
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = [r for r in rows if lo <= r[key_idx] < hi]
        if not b:
            continue
        n = len(b)
        avg_mp = sum(r[0] for r in b) / n
        avg_mkt = sum(r[1] for r in b) / n
        actual = sum(r[2] for r in b) / n
        model_gap = (actual - avg_mp) * 100
        mkt_gap = (actual - avg_mkt) * 100
        print(f"  {f'{lo:.2f}-{hi:.2f}':14} {n:>5} {avg_mp:>12.3f} "
              f"{avg_mkt:>9.3f} {actual * 100:>11.1f}% {model_gap:>+9.1f} "
              f"{mkt_gap:>+8.1f}")


def run(bet_kind: str, since: str | None) -> None:
    rows = _fetch(bet_kind, since)
    print("\n" + "=" * 84)
    print(f"MODEL CALIBRATION BY BUCKET  (bet_kind={bet_kind}, n={len(rows)})")
    print("=" * 84)
    if not rows:
        print("\nNo settled rows with model_prob + outcome. Nothing to analyze.\n")
        return

    overall_win = sum(r[2] for r in rows) / len(rows)
    avg_model = sum(r[0] for r in rows) / len(rows)
    print(f"\n  Backed-side win rate (actual):  {overall_win * 100:.1f}%")
    print(f"  Mean model_prob (predicted):    {avg_model * 100:.1f}%")
    print(f"  Overall calibration gap:        {(overall_win - avg_model) * 100:+.1f}pp "
          f"({'over' if avg_model > overall_win else 'under'}-confident)")

    brier_model = _brier(rows, use_market=False)
    brier_market = _brier(rows, use_market=True)
    print(f"\n  Brier score - model_prob:  {brier_model:.4f}")
    print(f"  Brier score - market_px:   {brier_market:.4f}  "
          f"(lower is better; the model {'BEATS' if brier_model < brier_market else 'LOSES to'} "
          f"the market as a forecaster)")

    edges = [0.0, 0.20, 0.35, 0.50, 0.65, 0.80, 1.001]
    _reliability_table(
        rows, 0, "Reliability binned by MODEL probability:", edges
    )
    _reliability_table(
        rows, 1, "Reliability binned by MARKET price (entry):", edges
    )

    # Favourite vs underdog summary (by backed-side market price)
    fav = [r for r in rows if r[1] >= 0.50]
    dog = [r for r in rows if r[1] < 0.50]
    print("\nFavourite vs underdog (by backed-side market price):")
    for name, grp in (("favourite (>=0.50)", fav), ("underdog (<0.50)", dog)):
        if not grp:
            continue
        n = len(grp)
        avg_mp = sum(r[0] for r in grp) / n
        actual = sum(r[2] for r in grp) / n
        print(f"  {name:22} n={n:>4} avg_model_p={avg_mp:.3f} "
              f"actual_win={actual * 100:.1f}% gap={(actual - avg_mp) * 100:+.1f}pp")

    # ---- Recommendation ----
    print("\n" + "-" * 84)
    print("RECOMMENDATION (diagnostic only):")
    notes = []
    # Deep longshot check
    deep = [r for r in rows if r[1] < 0.20]
    if deep:
        d_actual = sum(r[2] for r in deep) / len(deep)
        d_model = sum(r[0] for r in deep) / len(deep)
        if d_model - d_actual > 0.05:
            notes.append(
                f"- Deep longshots (market<0.20, n={len(deep)}): model says "
                f"{d_model * 100:.0f}% but only {d_actual * 100:.0f}% win -> "
                f"overconfident; exclude or down-weight (a min_market_price floor)."
            )
    if brier_model < brier_market:
        notes.append(
            "- Model Brier beats market Brier: model_prob carries genuine "
            "forecast information overall; calibration (e.g. Platt per bucket) "
            "is worth pursuing rather than discarding the signal."
        )
    else:
        notes.append(
            "- Model Brier does NOT beat the market: at these prices the market "
            "is the better forecaster; lean on market-mid, not model fair value."
        )
    if fav and dog:
        fav_gap = (sum(r[2] for r in fav) / len(fav)) - (sum(r[0] for r in fav) / len(fav))
        dog_gap = (sum(r[2] for r in dog) / len(dog)) - (sum(r[0] for r in dog) / len(dog))
        if abs(fav_gap - dog_gap) > 0.05:
            notes.append(
                f"- Asymmetric calibration: favourite gap {fav_gap * 100:+.0f}pp vs "
                f"underdog gap {dog_gap * 100:+.0f}pp -> a per-bucket calibration "
                f"correction (not a structural fade) is the right fix."
            )
    if not notes:
        notes.append("- No strong mis-calibration signal in this sample.")
    for nt in notes:
        print(nt)
    print("=" * 84 + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bet-kind", default="paper",
                    choices=["paper", "real", "all"])
    ap.add_argument("--since", default=None, help="ISO timestamp lower bound")
    args = ap.parse_args()
    run(args.bet_kind, args.since)
    return 0


if __name__ == "__main__":
    sys.exit(main())
