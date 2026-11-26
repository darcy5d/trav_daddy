#!/usr/bin/env python3
"""Seed team_external_ids (source='polymarket') with verified label aliases.

Polymarket team labels that don't exactly equal a `teams.name` and that the
fail-closed matcher in `src/integrations/polymarket/upcoming.py` would
otherwise leave unmatched (or that the old fuzzy matcher mapped to the wrong
team, e.g. "Hong Kong, China" -> China). Keys are `polymarket_alias_key(label)`
(normalised, Women/W suffix stripped), so one row covers both genders.

Labels with no DB team (Abu Dhabi T10, JCL, academy, legends, U19, A sides...)
are deliberately NOT seeded: the matcher leaves them unmatched.

Usage:
    venv311/bin/python scripts/seed_polymarket_team_aliases.py           # dry-run
    venv311/bin/python scripts/seed_polymarket_team_aliases.py --apply
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATABASE_PATH
from src.integrations.polymarket.upcoming import polymarket_alias_key

# Polymarket label -> exact teams.name
POLYMARKET_TEAM_ALIASES = {
    "Hong Kong, China": "Hong Kong",
    "Korea Republic": "South Korea",
    "USA": "United States of America",
    "United States": "United States of America",
    "Turks And Caicos": "Turks and Caicos Island",
    "Jammu And Kashmir": "Jammu & Kashmir",
    "New South Wales Blues": "New South Wales",
    "Queensland Bulls": "Queensland",
    "Tasmania Tigers": "Tasmania",
    "Sudurpaschim Royals": "Sudur Paschim Royals",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="write rows (default: dry-run)")
    ap.add_argument("--db", default=str(DATABASE_PATH))
    args = ap.parse_args()

    conn = sqlite3.connect(args.db, timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for label, team_name in POLYMARKET_TEAM_ALIASES.items():
        hit = conn.execute(
            "SELECT team_id, franchise_id FROM teams WHERE name = ?", (team_name,)
        ).fetchone()
        if not hit or hit[1] is None:
            print(f"SKIP  {label!r}: team {team_name!r} not found / has no group")
            continue
        key = polymarket_alias_key(label)
        rows.append((hit[1], "polymarket", key, label, None, now))
        print(f"{'APPLY' if args.apply else 'DRY  '} {key!r:32} -> {team_name} (team {hit[0]}, group {hit[1]})")

    if args.apply and rows:
        with conn:
            conn.executemany(
                """
                INSERT INTO team_external_ids
                    (group_id, source, external_id, external_name, confidence, last_validated)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(group_id, source, external_id) DO UPDATE SET
                    external_name = excluded.external_name,
                    last_validated = excluded.last_validated
                """,
                rows,
            )
        n = conn.execute(
            "SELECT COUNT(*) FROM team_external_ids WHERE source = 'polymarket'"
        ).fetchone()[0]
        print(f"polymarket aliases now in DB: {n}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
