#!/usr/bin/env python3
"""Wave 6 pre-work (W1): empirical maker economics from our own TWAP fills.

We already post resting limit orders via the TWAP execution path, so our
`order_plans` / `order_chunks` / `bet_ledger` history is a real-money proxy for
"what happens when we quote on cricket". This script decomposes that history to
answer the question Wave 6 market-making lives or dies on:

    Does spread capture beat adverse-selection cost, and how big a maker
    rebate / liquidity reward would we need to break even?

Definitions used here
---------------------
* MAKER bet   = a settled real bet that had a TWAP `order_plan` (we posted
                resting limit chunks). FOK/market bets are TAKERs.
* spread captured (filled plan) = market_price_at_plan - avg_fill_price.
                Positive = we bought below the screen price we anchored on.
* adverse selection = win% of plans that FILLED vs plans that were CANCELLED
                (never filled). If unfilled > filled, the market pulled the
                winners away and handed us the losers.
* break-even reward rate = -realised_maker_pnl / filled_notional. The maker
                rebate / liquidity reward (as a % of filled notional) that
                would have been required to drag the maker book to zero.

Read-only. No writes, no live calls.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_db_connection


def _first_live_bet_ts(cur) -> str | None:
    cur.execute(
        """
        SELECT MIN(proposed_at)
        FROM bet_ledger
        WHERE COALESCE(bet_kind, 'real') = 'real'
        """
    )
    row = cur.fetchone()
    return row[0] if row and row[0] else None


def _maker_bet_ids(cur) -> set:
    """bet_ledger ids that went through the TWAP (maker) path."""
    cur.execute(
        "SELECT bet_ledger_id FROM order_plans WHERE bet_ledger_id IS NOT NULL"
    )
    return {r[0] for r in cur.fetchall()}


def _winning_side_map(cur) -> dict:
    """(fixture_key, side_label) -> 1 if that side eventually won else 0.

    Reconstructed from any row that carries an outcome: match_settle_outcome
    (set even on cashed-out rows) or settle_outcome. Lets us score plans that
    never filled (and so never got an outcome stamped on their own row).
    """
    cur.execute(
        """
        SELECT fixture_key, side_label, match_settle_outcome, settle_outcome
        FROM bet_ledger
        WHERE (match_settle_outcome IS NOT NULL OR settle_outcome IS NOT NULL)
        """
    )
    win: dict = {}
    for fk, side, mo, so in cur.fetchall():
        outcome = mo if mo is not None else so
        if outcome is None or side is None or fk is None:
            continue
        win[(fk, side)] = int(outcome)
    return win


def _print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def analyze(start_ts: str | None = None) -> None:
    with get_db_connection() as conn:
        cur = conn.cursor()

        if start_ts is None:
            start_ts = _first_live_bet_ts(cur)
        if not start_ts:
            print("\nNo live bets found; nothing to analyze.\n")
            return

        maker_ids = _maker_bet_ids(cur)
        win_map = _winning_side_map(cur)

        print(f"\nComparison period: settled real bets proposed since {start_ts}")
        print(f"TWAP (maker) plans on record: {len(maker_ids)} bet_ledger ids\n")

        # ---------------------------------------------------------------
        # 1. Maker vs taker headline economics (settled real bets)
        # ---------------------------------------------------------------
        cur.execute(
            """
            SELECT bet_id, pnl_realised_usdc, size_usdc, settle_outcome,
                   fill_price, market_price_at_proposal
            FROM bet_ledger
            WHERE COALESCE(bet_kind, 'real') = 'real'
              AND status = 'settled'
              AND proposed_at >= ?
            """,
            (start_ts,),
        )
        groups = {"TWAP_maker": [], "FOK_taker": []}
        for row in cur.fetchall():
            grp = "TWAP_maker" if row[0] in maker_ids else "FOK_taker"
            groups[grp].append(row)

        _print_header("1. MAKER (TWAP) vs TAKER (FOK) - settled real bets")
        print(
            f"  {'group':12} {'n':>5} {'pnl$':>10} {'staked$':>10} "
            f"{'ROI%':>8} {'win%':>7} {'entry_slip¢':>12}"
        )
        print("  " + "-" * 70)
        maker_pnl = maker_notional = 0.0
        for grp, rows in groups.items():
            if not rows:
                continue
            n = len(rows)
            pnl = sum(float(r[1] or 0) for r in rows)
            staked = sum(float(r[2] or 0) for r in rows)
            wins = sum(1 for r in rows if r[3] == 1)
            slips = [
                (float(r[4]) - float(r[5])) * 100.0
                for r in rows
                if r[4] is not None and r[5] is not None
            ]
            roi = 100.0 * pnl / staked if staked else 0.0
            wr = 100.0 * wins / n if n else 0.0
            avg_slip = sum(slips) / len(slips) if slips else 0.0
            print(
                f"  {grp:12} {n:>5} {pnl:>10.2f} {staked:>10.2f} "
                f"{roi:>8.2f} {wr:>6.1f}% {avg_slip:>+11.2f}"
            )
            if grp == "TWAP_maker":
                maker_pnl = pnl
                maker_notional = staked

        # ---------------------------------------------------------------
        # 2. Fill efficiency: how often do posted maker plans actually fill?
        # ---------------------------------------------------------------
        _print_header("2. TWAP plan fill efficiency")
        cur.execute(
            """
            SELECT status, COUNT(*),
                   ROUND(AVG(1.0 * chunks_filled / NULLIF(chunks_total, 0)), 3)
            FROM order_plans
            GROUP BY status
            ORDER BY COUNT(*) DESC
            """
        )
        n_filled_plans = n_cancelled_plans = 0
        for st, n, frac in cur.fetchall():
            print(f"  plan status={str(st):12} n={n:>4}  avg_chunk_fill_frac={frac}")
            if st == "completed":
                n_filled_plans = n
            elif st == "cancelled":
                n_cancelled_plans = n
        total_plans = n_filled_plans + n_cancelled_plans
        if total_plans:
            print(
                f"\n  Fill rate (completed / [completed+cancelled]): "
                f"{100.0 * n_filled_plans / total_plans:.1f}%"
            )

        # ---------------------------------------------------------------
        # 3. Adverse selection: filled vs unfilled plan win rates
        # ---------------------------------------------------------------
        _print_header("3. ADVERSE SELECTION - did the orders that DIDN'T fill win?")
        cur.execute(
            """
            SELECT p.status, b.fixture_key, b.side_label
            FROM order_plans p
            JOIN bet_ledger b ON b.bet_id = p.bet_ledger_id
            WHERE p.status IN ('cancelled', 'completed')
            """
        )
        agg = defaultdict(lambda: [0, 0, 0])  # status -> [won, lost, unknown]
        for st, fk, side in cur.fetchall():
            o = win_map.get((fk, side))
            if o == 1:
                agg[st][0] += 1
            elif o == 0:
                agg[st][1] += 1
            else:
                agg[st][2] += 1

        filled_wr = unfilled_wr = None
        for st in ("completed", "cancelled"):
            w, l, u = agg[st]
            known = w + l
            wr = 100.0 * w / known if known else 0.0
            tag = "FILLED  " if st == "completed" else "UNFILLED"
            print(
                f"  {tag} (plan={st:9}) our_side won={w:>4} lost={l:>4} "
                f"unknown={u:>4}  win%(known)={wr:.1f}%"
            )
            if st == "completed":
                filled_wr = wr
            else:
                unfilled_wr = wr
        if filled_wr is not None and unfilled_wr is not None:
            gap = unfilled_wr - filled_wr
            verdict = (
                "ADVERSE SELECTION (we miss winners, fill losers)"
                if gap > 0
                else "no adverse selection signal"
            )
            print(f"\n  Unfilled - filled win% gap = {gap:+.1f}pp  ->  {verdict}")

        # ---------------------------------------------------------------
        # 4. Maker ROI by market-price bucket
        # ---------------------------------------------------------------
        _print_header("4. MAKER ROI by entry market-price bucket")
        maker_ids_list = list(maker_ids) or [-1]
        placeholders = ",".join("?" for _ in maker_ids_list)
        cur.execute(
            f"""
            SELECT
                CASE
                  WHEN market_price_at_proposal < 0.20 THEN '1. <0.20'
                  WHEN market_price_at_proposal < 0.35 THEN '2. 0.20-0.35'
                  WHEN market_price_at_proposal < 0.50 THEN '3. 0.35-0.50'
                  WHEN market_price_at_proposal < 0.65 THEN '4. 0.50-0.65'
                  WHEN market_price_at_proposal < 0.80 THEN '5. 0.65-0.80'
                  ELSE '6. >=0.80'
                END AS bucket,
                COUNT(*) AS n,
                ROUND(SUM(pnl_realised_usdc), 2) AS pnl,
                ROUND(SUM(size_usdc), 2) AS staked
            FROM bet_ledger
            WHERE bet_id IN ({placeholders})
              AND status = 'settled'
              AND proposed_at >= ?
            GROUP BY bucket
            ORDER BY bucket
            """,
            (*maker_ids_list, start_ts),
        )
        print(f"  {'bucket':14} {'n':>5} {'pnl$':>10} {'staked$':>10} {'ROI%':>8}")
        print("  " + "-" * 50)
        for bucket, n, pnl, staked in cur.fetchall():
            roi = 100.0 * (pnl or 0) / staked if staked else 0.0
            print(f"  {bucket:14} {n:>5} {pnl or 0:>10.2f} {staked or 0:>10.2f} {roi:>8.1f}")

        # ---------------------------------------------------------------
        # 5. Maker ROI by hours-to-kickoff (when does adverse selection bite?)
        # ---------------------------------------------------------------
        _print_header("5. MAKER ROI by hours-to-kickoff at proposal")
        cur.execute(
            f"""
            SELECT
                CASE
                  WHEN (julianday(kickoff_at) - julianday(proposed_at)) * 24 < 1 THEN '1. <1h'
                  WHEN (julianday(kickoff_at) - julianday(proposed_at)) * 24 < 6 THEN '2. 1-6h'
                  WHEN (julianday(kickoff_at) - julianday(proposed_at)) * 24 < 24 THEN '3. 6-24h'
                  WHEN (julianday(kickoff_at) - julianday(proposed_at)) * 24 < 48 THEN '4. 24-48h'
                  ELSE '5. >48h'
                END AS bucket,
                COUNT(*) AS n,
                ROUND(SUM(pnl_realised_usdc), 2) AS pnl,
                ROUND(SUM(size_usdc), 2) AS staked
            FROM bet_ledger
            WHERE bet_id IN ({placeholders})
              AND status = 'settled'
              AND kickoff_at IS NOT NULL
              AND proposed_at >= ?
            GROUP BY bucket
            ORDER BY bucket
            """,
            (*maker_ids_list, start_ts),
        )
        rows = cur.fetchall()
        if rows:
            print(f"  {'bucket':10} {'n':>5} {'pnl$':>10} {'staked$':>10} {'ROI%':>8}")
            print("  " + "-" * 46)
            for bucket, n, pnl, staked in rows:
                roi = 100.0 * (pnl or 0) / staked if staked else 0.0
                print(f"  {bucket:10} {n:>5} {pnl or 0:>10.2f} {staked or 0:>10.2f} {roi:>8.1f}")
        else:
            print("  (no kickoff_at timestamps on maker bets in window)")

        # ---------------------------------------------------------------
        # 6. The Wave 6 number: break-even reward rate
        # ---------------------------------------------------------------
        _print_header("6. BREAK-EVEN REWARD RATE (the Wave 6 go/no-go number)")
        if maker_notional > 0:
            be_reward_pct = -maker_pnl / maker_notional * 100.0
            print(
                f"  Realised maker P&L:        ${maker_pnl:,.2f}\n"
                f"  Maker filled notional:     ${maker_notional:,.2f}\n"
                f"  Realised maker ROI:        {100.0 * maker_pnl / maker_notional:.2f}%\n"
            )
            if be_reward_pct <= 0:
                print(
                    "  Maker book is already profitable on spread alone; any reward\n"
                    "  income is pure upside."
                )
            else:
                print(
                    f"  >> Maker rebate + liquidity reward would need to clear "
                    f"{be_reward_pct:.2f}% of\n"
                    f"     filled notional just to BREAK EVEN, before any directional\n"
                    f"     edge. Compare this against the reward rate the W2 recon\n"
                    f"     scanner finds for cricket. If cricket rewards < "
                    f"{be_reward_pct:.2f}%, model-\n"
                    f"     centered market making on these books is structurally -EV."
                )
        else:
            print("  No filled maker notional in window; cannot compute.")

        print("\n" + "=" * 78 + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--since",
        default=None,
        help="ISO timestamp lower bound (default: first live bet)",
    )
    args = ap.parse_args()
    analyze(start_ts=args.since)
    return 0


if __name__ == "__main__":
    sys.exit(main())
