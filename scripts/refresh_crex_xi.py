#!/usr/bin/env python3
"""Wave 5.11: refresh CREX Playing XI cache for upcoming fixtures.

Fetches the current predicted/confirmed Playing XI from CREX for every
upcoming Polymarket cricket fixture within the lookahead window, resolves
player names to DB player_ids via fuzzy-matching, and writes the result to
the crex_xi_cache table.

Designed to run as a cron job every 2 hours:
    0 */2 * * * cd /path/to/indias_dad && \\
        venv311/bin/python scripts/refresh_crex_xi.py >> logs/crex_xi_refresh.log 2>&1

Cache rows are keyed by (fixture_key, team_id) and include the fetch
timestamp so consumers can enforce their own staleness threshold.

Minimum match gate: a cache row is only written if at least
CREX_XI_MIN_MATCH of 11 players resolve to known DB player_ids.
Below that the XI is too uncertain and consumers fall back to get_recent_xi().

Usage:
    venv311/bin/python scripts/refresh_crex_xi.py [--hours-ahead 48]
                                                   [--fixture-key KEY]
                                                   [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_connection, init_crex_xi_cache
from src.integrations.polymarket import PolymarketClient
from src.integrations.polymarket.upcoming import (
    find_upcoming_cricket_events,
    attach_db_team_ids,
)
from src.integrations.polymarket.crex_xi import fetch_xi_from_crex
from src.integrations.polymarket.paper_inputs import (
    resolve_xi_names_to_ids,
    CREX_XI_MIN_MATCH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _upsert_cache_row(
    conn,
    fixture_key: str,
    team_id: int,
    player_ids: List[int],
    n_matched: int,
    n_input: int,
    dry_run: bool = False,
) -> None:
    """Write or overwrite a crex_xi_cache row."""
    fetched_at = datetime.now(timezone.utc).isoformat()
    players_json = json.dumps(player_ids)
    if dry_run:
        logger.info(
            f"  [dry-run] would upsert cache: {fixture_key} team {team_id} "
            f"n_matched={n_matched}/{n_input} ids={player_ids[:5]}..."
        )
        return
    conn.execute(
        """
        INSERT INTO crex_xi_cache
            (fixture_key, team_id, players_json, n_matched, n_input, source, fetched_at)
        VALUES (?, ?, ?, ?, ?, 'crex', ?)
        ON CONFLICT(fixture_key, team_id) DO UPDATE SET
            players_json = excluded.players_json,
            n_matched    = excluded.n_matched,
            n_input      = excluded.n_input,
            source       = excluded.source,
            fetched_at   = excluded.fetched_at
        """,
        (fixture_key, team_id, players_json, n_matched, n_input, fetched_at),
    )


def refresh_fixture(
    fixture: Dict[str, Any],
    conn,
    dry_run: bool = False,
) -> bool:
    """Fetch CREX XI for one fixture and populate the cache.

    Returns True if at least one team's XI was written to the cache.
    """
    fkey = fixture.get("fixture_key") or ""
    t1_name = fixture.get("team1_db_name") or fixture.get("team1_label") or "?"
    t2_name = fixture.get("team2_db_name") or fixture.get("team2_label") or "?"
    t1_id = fixture.get("team1_id")
    t2_id = fixture.get("team2_id")
    fmt = fixture.get("format", "T20")
    gender = fixture.get("gender", "male")

    logger.info(f"Fetching CREX XI for {t1_name} vs {t2_name} ({fkey})")

    xi_data = fetch_xi_from_crex(fixture)
    if xi_data is None:
        logger.warning(f"  No CREX match found or scrape failed for {fkey}")
        return False

    crex_status = xi_data.get("crex_status", "unknown")
    logger.info(f"  CREX status: {crex_status}")

    any_written = False

    for side, team_id, xi_names in [
        ("team1", t1_id, xi_data.get("team1_xi", [])),
        ("team2", t2_id, xi_data.get("team2_xi", [])),
    ]:
        if team_id is None:
            logger.warning(f"  [{side}] no team_id in fixture, skipping")
            continue
        if not xi_names:
            logger.warning(f"  [{side}] CREX returned no player names, skipping")
            continue

        player_ids, n_matched, n_input = resolve_xi_names_to_ids(
            conn, xi_names, team_id, fmt, gender
        )

        logger.info(
            f"  [{side}] resolved {n_matched}/{n_input} players "
            f"(need >= {CREX_XI_MIN_MATCH})"
        )

        if n_matched < CREX_XI_MIN_MATCH:
            logger.warning(
                f"  [{side}] below threshold — skipping cache write "
                f"(unmatched names: "
                f"{[n for n in xi_names if n not in [str(p) for p in player_ids]][:5]})"
            )
            continue

        _upsert_cache_row(conn, fkey, team_id, player_ids, n_matched, n_input, dry_run)
        logger.info(f"  [{side}] cache updated for {fkey} team {team_id}")
        any_written = True

    return any_written


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh CREX XI cache for upcoming fixtures")
    parser.add_argument(
        "--hours-ahead", type=float, default=48.0,
        help="Look ahead window in hours (default: 48)",
    )
    parser.add_argument(
        "--fixture-key", type=str, default=None,
        help="Only refresh a specific fixture_key (for debugging)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch and resolve but do not write to the database",
    )
    args = parser.parse_args()

    # Ensure table exists.
    init_crex_xi_cache()

    with get_connection() as conn:
        # Discover upcoming fixtures.
        logger.info(f"Discovering upcoming fixtures ({args.hours_ahead}h ahead)...")
        try:
            pm_client = PolymarketClient()
            events = find_upcoming_cricket_events(pm_client, hours_ahead=args.hours_ahead)
        except Exception as exc:
            logger.error(f"Failed to fetch upcoming events: {exc}")
            return 1

        if not events:
            logger.info("No upcoming fixtures found.")
            return 0

        fixtures = attach_db_team_ids(events)
        logger.info(f"Found {len(fixtures)} upcoming fixture(s)")

        # Filter to specific fixture if requested.
        if args.fixture_key:
            fixtures = [f for f in fixtures if f.get("fixture_key") == args.fixture_key]
            if not fixtures:
                logger.error(f"No fixture found with key: {args.fixture_key}")
                return 1

        written = 0
        skipped = 0
        for fixture in fixtures:
            # Skip fixtures where we couldn't resolve team_ids.
            if not fixture.get("team1_id") or not fixture.get("team2_id"):
                logger.warning(
                    f"Skipping {fixture.get('fixture_key')} — missing team_ids "
                    f"(team resolution failed)"
                )
                skipped += 1
                continue

            ok = refresh_fixture(fixture, conn, dry_run=args.dry_run)
            if ok:
                written += 1
            else:
                skipped += 1

            # Commit after each fixture so partial progress is preserved.
            if not args.dry_run:
                conn.commit()

        logger.info(
            f"Done. {written} fixture(s) cached, {skipped} skipped."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
