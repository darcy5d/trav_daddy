#!/usr/bin/env python3
"""One-time backfill for the Live Risk Management dashboard.

Populates the persisted risk-analytics columns added by
`init_risk_columns()` so the dashboard reads everything from the DB with no
live Polymarket calls at render time:

  Pass A - kelly_uncapped_stake (offline): replay compute_sizing_context()
           over every row missing it. All inputs are already stored.
  Pass B - match_settle_outcome for HELD bets (offline): a bet held to
           settlement already has settle_outcome == the match result, so copy
           it across; set match_winner = side_label when our side won.
  Pass C - match_settle_outcome / match_winner for CASHED-OUT bets (Gamma):
           settle_outcome is NULL on these, so resolve the eventual result via
           the cached Gamma market. Rows whose market is not yet closed are
           left NULL and will be filled by the next reconcile pass.

Usage:
    python scripts/backfill_risk_fields.py                 # real cashouts only
    python scripts/backfill_risk_fields.py --kind both     # also paper
    python scripts/backfill_risk_fields.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_connection, init_risk_columns
from src.integrations.polymarket.sizing import compute_sizing_context

RATE_LIMIT_SLEEP = 0.3


def _backfill_kelly(conn, dry_run: bool) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT bet_id, bankroll_at_proposal, model_prob,
               market_price_at_proposal, strategy_label
        FROM bet_ledger
        WHERE kelly_uncapped_stake IS NULL
          AND bankroll_at_proposal IS NOT NULL
          AND model_prob IS NOT NULL
          AND market_price_at_proposal IS NOT NULL
          AND strategy_label IS NOT NULL
        """
    )
    rows = cur.fetchall()
    n = 0
    for r in rows:
        ctx = compute_sizing_context({
            "bankroll_at_proposal": r["bankroll_at_proposal"],
            "model_prob": r["model_prob"],
            "market_price_at_proposal": r["market_price_at_proposal"],
            "strategy_label": r["strategy_label"],
        })
        if not ctx or ctx.get("kelly_uncapped_stake") is None:
            continue
        if not dry_run:
            conn.execute(
                "UPDATE bet_ledger SET kelly_uncapped_stake = ? WHERE bet_id = ?",
                (ctx["kelly_uncapped_stake"], r["bet_id"]),
            )
        n += 1
    if not dry_run:
        conn.commit()
    print(f"Pass A (kelly_uncapped_stake): {n} row(s) {'would be ' if dry_run else ''}updated")
    return n


def _backfill_held(conn, dry_run: bool) -> int:
    cur = conn.cursor()
    # match_settle_outcome <- settle_outcome for held (non-cashed) settled bets.
    cur.execute(
        """
        SELECT COUNT(*) FROM bet_ledger
        WHERE settle_outcome IS NOT NULL
          AND match_settle_outcome IS NULL
          AND cashout_triggered_at IS NULL
        """
    )
    n = cur.fetchone()[0]
    if not dry_run and n:
        conn.execute(
            """
            UPDATE bet_ledger
            SET match_settle_outcome = settle_outcome,
                match_winner = COALESCE(
                    match_winner,
                    CASE WHEN settle_outcome = 1 THEN side_label ELSE match_winner END
                )
            WHERE settle_outcome IS NOT NULL
              AND match_settle_outcome IS NULL
              AND cashout_triggered_at IS NULL
            """
        )
        conn.commit()
    print(f"Pass B (held match_settle_outcome): {n} row(s) {'would be ' if dry_run else ''}updated")
    return n


def _backfill_cashed(conn, kinds, dry_run: bool) -> int:
    from src.integrations.polymarket.reconcile import (
        _resolve_via_gamma,
        _winner_label_from_gamma,
    )
    from src.integrations.polymarket.bet_display_enrichment import get_gamma_market_cached

    placeholders = ",".join("?" for _ in kinds)
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT bet_id, polymarket_market_id, side_label
        FROM bet_ledger
        WHERE cashout_triggered_at IS NOT NULL
          AND match_settle_outcome IS NULL
          AND polymarket_market_id IS NOT NULL
          AND side_label IS NOT NULL
          AND COALESCE(bet_kind, 'real') IN ({placeholders})
        ORDER BY cashout_triggered_at DESC
        """,
        list(kinds),
    )
    rows = cur.fetchall()
    print(f"Pass C (cashed-out via Gamma): {len(rows)} candidate row(s)...")
    n = pending = 0
    for r in rows:
        market = get_gamma_market_cached(str(r["polymarket_market_id"]))
        resolved = _resolve_via_gamma(market, r["side_label"] or "") if market else None
        if resolved is None:
            pending += 1
            continue
        outcome, _price = resolved
        winner = _winner_label_from_gamma(market)
        if not dry_run:
            conn.execute(
                "UPDATE bet_ledger SET match_settle_outcome = ?, match_winner = ? WHERE bet_id = ?",
                (int(outcome), winner, r["bet_id"]),
            )
        n += 1
        time.sleep(RATE_LIMIT_SLEEP)
    if not dry_run:
        conn.commit()
    print(f"Pass C: {n} resolved, {pending} still pending (market not closed) "
          f"{'(dry-run)' if dry_run else ''}")
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["real", "paper", "both"], default="real",
                    help="Which cashed-out bets to resolve via Gamma (default: real)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    init_risk_columns()
    kinds = ["real", "paper"] if args.kind == "both" else [args.kind]

    conn = get_connection()
    try:
        _backfill_kelly(conn, args.dry_run)
        _backfill_held(conn, args.dry_run)
        _backfill_cashed(conn, kinds, args.dry_run)
    finally:
        conn.close()
    print("Backfill complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
