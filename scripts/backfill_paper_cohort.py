#!/usr/bin/env python3
"""V15 one-time backfill: tag the pre-existing paper book as the legacy cohort.

Every ``bet_kind='paper'`` row that currently has a NULL ``paper_cohort`` is
stamped ``legacy_pre_edge_20261108``. Nothing else is written: no P&L, no
status, timestamp or size column is touched, and ``bet_kind='real'`` rows are
excluded by the WHERE clause. The legacy book is the calibration holdout, so
the only permitted change to it is this label.

Idempotent: re-running tags nothing because the predicate is ``IS NULL``.

Usage:
    venv311/bin/python scripts/backfill_paper_cohort.py --dry-run
    venv311/bin/python scripts/backfill_paper_cohort.py --apply

After applying, verify with the count/fingerprint printout it emits: the
per-bet_kind row counts must be unchanged and the paper P&L / stake sums must
be identical to the pre-backfill values.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_connection, init_paper_cohort
from src.integrations.polymarket.paper_cohorts import LEGACY_PAPER_COHORT


def _snapshot(conn) -> dict:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COALESCE(bet_kind, 'real') AS kind,
               COUNT(*) AS n,
               ROUND(COALESCE(SUM(pnl_realised_usdc), 0.0), 6) AS pnl,
               ROUND(COALESCE(SUM(size_usdc), 0.0), 6) AS staked
        FROM bet_ledger
        GROUP BY 1
        ORDER BY 1
        """
    )
    return {r["kind"]: dict(r) for r in cur.fetchall()}


def _cohort_counts(conn) -> list:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COALESCE(bet_kind, 'real') AS kind,
               COALESCE(paper_cohort, '(untagged)') AS cohort,
               COUNT(*) AS n
        FROM bet_ledger
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    )
    return [dict(r) for r in cur.fetchall()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="Write the tags. Without this the run is a dry run.")
    args = ap.parse_args()

    if not init_paper_cohort():
        print("FAILED: could not ensure bet_ledger.paper_cohort exists", file=sys.stderr)
        return 1

    conn = get_connection()
    try:
        before = _snapshot(conn)
        print("=== BEFORE (by bet_kind) ===")
        for kind, row in before.items():
            print(f"  {kind:6s} n={row['n']:<6d} pnl={row['pnl']:<14} staked={row['staked']}")

        cur = conn.cursor()
        cur.execute(
            """
            SELECT COUNT(*) AS n
            FROM bet_ledger
            WHERE bet_kind = 'paper' AND paper_cohort IS NULL
            """
        )
        n_to_tag = int(cur.fetchone()["n"])
        cur.execute(
            """
            SELECT COUNT(*) AS n FROM bet_ledger
            WHERE COALESCE(bet_kind, 'real') = 'real' AND paper_cohort IS NOT NULL
            """
        )
        n_real_tagged = int(cur.fetchone()["n"])
        print(f"\npaper rows pending legacy tag : {n_to_tag}")
        print(f"real rows already tagged      : {n_real_tagged} (must stay 0)")

        if not args.apply:
            print("\nDRY-RUN — no writes. Re-run with --apply to tag.")
            return 0

        cur.execute(
            """
            UPDATE bet_ledger
            SET paper_cohort = ?
            WHERE bet_kind = 'paper' AND paper_cohort IS NULL
            """,
            (LEGACY_PAPER_COHORT,),
        )
        n_written = cur.rowcount
        conn.commit()
        print(f"\ntagged {n_written} paper rows as '{LEGACY_PAPER_COHORT}'")

        after = _snapshot(conn)
        print("\n=== AFTER (by bet_kind) — counts/P&L must equal BEFORE ===")
        for kind, row in after.items():
            print(f"  {kind:6s} n={row['n']:<6d} pnl={row['pnl']:<14} staked={row['staked']}")

        ok = before == after
        print(f"\nidentical to before: {ok}")
        if not ok:
            print("WARNING: aggregate totals changed — investigate before relying on this.",
                  file=sys.stderr)

        print("\n=== AFTER (by kind x cohort) ===")
        for row in _cohort_counts(conn):
            print(f"  {row['kind']:6s} {row['cohort']:26s} n={row['n']}")
        return 0 if ok else 2
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
