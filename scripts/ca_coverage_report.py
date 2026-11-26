#!/usr/bin/env python3
"""Coverage report for the CricketArchive archive store (ca_archive.db).

READ-ONLY. Summarises what we have harvested, per season (grouped by the stable
competition_url): match count, ball-by-ball coverage, delivery count, and the
share of innings whose ball-by-ball identity reconciled (identity_verified).

Usage:
    venv311/bin/python scripts/ca_coverage_report.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.cricketarchive import store


def _season(url: str) -> str:
    if not url:
        return "(unknown)"
    m = re.search(r"/Events/\d+/(.+?)\.html", url)
    return m.group(1).replace("_", " ") if m else url


def main() -> int:
    with store.get_connection() as c:
        totals = {
            "matches": c.execute("SELECT COUNT(*) FROM ca_matches").fetchone()[0],
            "with_bbb": c.execute("SELECT COUNT(*) FROM ca_matches WHERE has_ball_by_ball=1").fetchone()[0],
            "deliveries": c.execute("SELECT COUNT(*) FROM ca_deliveries").fetchone()[0],
            "players": c.execute("SELECT COUNT(*) FROM ca_players").fetchone()[0],
        }
        rows = c.execute("""
            SELECT competition_url AS u,
                   COUNT(*) AS matches,
                   SUM(has_ball_by_ball) AS bbb
            FROM ca_matches GROUP BY competition_url
        """).fetchall()
        # identity-verified innings share (only meaningful for bbb innings)
        idv = c.execute("""
            SELECT SUM(CASE WHEN identity_verified=1 THEN 1 ELSE 0 END) AS ok,
                   SUM(CASE WHEN identity_verified IS NOT NULL THEN 1 ELSE 0 END) AS tot
            FROM ca_innings
        """).fetchone()
        # delivery resolution
        dres = c.execute("""
            SELECT COUNT(*) AS tot,
                   SUM(CASE WHEN batter_ca_id IS NOT NULL THEN 1 ELSE 0 END) AS bat,
                   SUM(CASE WHEN bowler_ca_id IS NOT NULL THEN 1 ELSE 0 END) AS bowl
            FROM ca_deliveries
        """).fetchone()

    seasons = sorted(((_season(r["u"]), r["matches"], r["bbb"] or 0) for r in rows),
                     key=lambda x: x[0])

    print("\n" + "=" * 78)
    print("CRICKETARCHIVE ARCHIVE COVERAGE (ca_archive.db)")
    print("=" * 78)
    print(f"{'SEASON':<42} {'MATCHES':>8} {'BBB':>6} {'BBB%':>6}")
    print("-" * 78)
    for name, n, bbb in seasons:
        pct = (100 * bbb / n) if n else 0
        print(f"{name:<42} {n:>8} {bbb:>6} {pct:>5.0f}%")
    print("-" * 78)
    print(f"  Total matches:        {totals['matches']}")
    print(f"  With ball-by-ball:    {totals['with_bbb']}")
    print(f"  Deliveries:           {totals['deliveries']:,}")
    print(f"  Distinct CA players:  {totals['players']:,}")
    if dres and dres["tot"]:
        print(f"  Delivery resolution:  batter {dres['bat']}/{dres['tot']}  bowler {dres['bowl']}/{dres['tot']}")
    if idv and idv["tot"]:
        print(f"  Innings identity-verified: {idv['ok']}/{idv['tot']}  ({100*idv['ok']/idv['tot']:.1f}%)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
