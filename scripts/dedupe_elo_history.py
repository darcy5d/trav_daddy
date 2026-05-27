#!/usr/bin/env python3
"""
Dedupe ELO history and rebuild ratings cleanly.

Why: the V3 calculator previously inserted into team_elo_history /
player_elo_history without any uniqueness guard. Re-running the recalc with
matches already in the DB cascaded the K-factor on the same loss multiple
times, pinning teams like KKR (current ELO 1350.0 = tier-3 floor) to the
floor and burying historically dominant sides like MI (1351.7) at the same
point. The diagnosis showed ~32,260 (team, match) pairs with duplicates and
~527K extra rows total.

What this script does:
  1. Take a timestamped backup of cricket.db.
  2. Run calculate_all_elos_v3(force_recalculate=True). The franchise-aware
     calculator wipes non-snapshot history, recomputes from scratch, writes
     ratings under the canonical_team_id of each franchise, and uses
     INSERT OR IGNORE so a stray re-run can never double-apply.
  3. Add a partial UNIQUE INDEX on team_elo_history and player_elo_history
     so any future regression is structurally impossible.
  4. Print a validation table comparing IPL teams' ELOs before vs after.

Run:
    python scripts/dedupe_elo_history.py --dry-run   # validation report only
    python scripts/dedupe_elo_history.py             # backup + recalc + index
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATABASE_PATH  # noqa: E402
from src.data.database import get_connection, get_db_connection, init_franchise_tables  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


IPL_TEAM_IDS = [111, 112, 113, 114, 115, 116, 117, 118, 196, 233, 234, 283, 362, 363, 364, 370]
IPL_PLAYER_NAMES = [
    "V Kohli", "RM Patidar", "PD Salt", "GJ Maxwell", "VR Iyer",
    "SH Johnson", "HV Patel", "CV Varun", "SP Narine", "AD Russell",
    "Q de Kock", "MA Starc", "RG Sharma", "MS Dhoni", "RD Gaikwad",
    "HH Pandya", "SA Yadav", "JM Sharma", "RA Tripathi",
]


def snapshot_team_elos(conn) -> Dict[int, Tuple[str, float, float, int]]:
    """Capture per-team rating snapshots for the IPL teams of interest.

    Returns: {team_id: (name, own_row_elo_or_NaN, canonical_row_elo, canonical_id)}.
    `own_row_elo` is the rating sitting in team_current_elo against the team's
    own id (the legacy-pre-unification value). `canonical_row_elo` is the
    rating against the franchise's canonical owner. After the recalc the own
    row may be gone and only the canonical row survives.
    """
    cur = conn.cursor()
    placeholders = ",".join("?" * len(IPL_TEAM_IDS))
    cur.execute(
        f"""
        SELECT
            t.team_id,
            t.name,
            COALESCE(t.canonical_team_id, t.team_id) AS canonical_id,
            own.elo_t20_male AS own_elo,
            canon.elo_t20_male AS canon_elo
        FROM teams t
        LEFT JOIN team_current_elo own
            ON own.team_id = t.team_id
        LEFT JOIN team_current_elo canon
            ON canon.team_id = COALESCE(t.canonical_team_id, t.team_id)
        WHERE t.team_id IN ({placeholders})
        """,
        IPL_TEAM_IDS,
    )
    out: Dict[int, Tuple[str, float, float, int]] = {}
    for row in cur.fetchall():
        own = row["own_elo"] if row["own_elo"] is not None else float("nan")
        canon = row["canon_elo"] if row["canon_elo"] is not None else float("nan")
        out[row["team_id"]] = (row["name"], own, canon, row["canonical_id"])
    return out


def snapshot_player_elos(conn) -> Dict[str, Tuple[float, float]]:
    """Capture {player_name: (batting_elo, bowling_elo)} for spotlight IPL players."""
    cur = conn.cursor()
    placeholders = ",".join("?" * len(IPL_PLAYER_NAMES))
    cur.execute(
        f"""
        SELECT p.name, pe.batting_elo_t20_male AS bat, pe.bowling_elo_t20_male AS bowl
        FROM players p
        JOIN player_current_elo pe ON pe.player_id = p.player_id
        WHERE p.name IN ({placeholders})
        """,
        IPL_PLAYER_NAMES,
    )
    return {row["name"]: (row["bat"] or 1500.0, row["bowl"] or 1500.0) for row in cur.fetchall()}


def count_duplicate_history_rows(conn) -> Tuple[int, int]:
    """Return (team_dupe_keys, player_dupe_keys) - number of (team, match) pairs
    that have more than one non-snapshot history row."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT 1
            FROM team_elo_history
            WHERE NOT is_monthly_snapshot AND match_id IS NOT NULL
            GROUP BY team_id, match_id, format, gender
            HAVING COUNT(*) > 1
        )
        """
    )
    team_dupes = cur.fetchone()[0]
    cur.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT 1
            FROM player_elo_history
            WHERE NOT is_monthly_snapshot AND match_id IS NOT NULL
            GROUP BY player_id, match_id, format, gender
            HAVING COUNT(*) > 1
        )
        """
    )
    player_dupes = cur.fetchone()[0]
    return team_dupes, player_dupes


def add_unique_indexes(conn) -> None:
    """Create partial UNIQUE indexes that make the duplicate-insert bug
    structurally impossible going forward.

    Partial indexes (WHERE clause) so that monthly snapshots (which legitimately
    share team_id+match_id=NULL) are exempt.
    """
    cur = conn.cursor()
    cur.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_team_elo_history_match
        ON team_elo_history(team_id, match_id, format, gender)
        WHERE is_monthly_snapshot = 0 AND match_id IS NOT NULL
        """
    )
    cur.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_player_elo_history_match
        ON player_elo_history(player_id, match_id, format, gender)
        WHERE is_monthly_snapshot = 0 AND match_id IS NOT NULL
        """
    )
    logger.info("Created partial UNIQUE indexes on elo history tables")


def _fmt(v: float, width: int = 9) -> str:
    """Right-justified ELO number, or '(none)' for NaN."""
    return f"{v:>{width}.1f}" if v == v else f"{'(none)':>{width}}"


def print_team_diff(before: Dict[int, Tuple[str, float, float, int]],
                    after: Dict[int, Tuple[str, float, float, int]]) -> None:
    logger.info("\nIPL team ELO change (T20 male):")
    logger.info(
        f"  {'team_id':<8} {'name':<30} {'canon':<6} "
        f"{'own pre':>9} {'own post':>9} {'canon pre':>10} {'canon post':>11} {'delta':>9}"
    )

    def _canon_after(tid: int) -> float:
        v = after.get(tid, (None, 0, float("nan"), 0))[2]
        return v if v == v else 0.0

    for tid in sorted(before, key=lambda t: -_canon_after(t)):
        name, own_b, can_b, can_id = before[tid]
        _, own_a, can_a, _ = after.get(tid, (name, float("nan"), float("nan"), can_id))
        delta = can_a - can_b if (can_a == can_a and can_b == can_b) else float("nan")
        delta_str = f"{delta:+9.1f}" if delta == delta else f"{'(n/a)':>9}"
        logger.info(
            f"  {tid:<8} {name:<30} {can_id:<6} "
            f"{_fmt(own_b)} {_fmt(own_a)} {_fmt(can_b, 10)} {_fmt(can_a, 11)} {delta_str}"
        )


def print_player_diff(before: Dict[str, Tuple[float, float]],
                      after: Dict[str, Tuple[float, float]]) -> None:
    logger.info("\nSpotlight IPL player ELO change (T20 male) - batting only:")
    logger.info(f"  {'player':<20} {'bat before':>10} {'bat after':>10} {'delta':>9}")
    for name in sorted(before, key=lambda n: -(after.get(n, (0, 0))[0] or 0)):
        b = before[name][0]
        a = after.get(name, (None, None))[0]
        if a is None:
            logger.info(f"  {name:<20} {b:>10.1f} {'(missing)':>10}")
        else:
            logger.info(f"  {name:<20} {b:>10.1f} {a:>10.1f} {a - b:>+9.1f}")


def backup_db() -> Path:
    src = Path(DATABASE_PATH)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = src.with_suffix(f".db.bak.{ts}")
    logger.info(f"Backing up {src} -> {dst} (this can take a minute on large DBs)")
    t0 = time.time()
    shutil.copy2(src, dst)
    # Also copy the WAL/SHM if present so the backup is consistent.
    for suffix in ("-wal", "-shm"):
        sidecar = src.with_name(src.name + suffix)
        if sidecar.exists():
            shutil.copy2(sidecar, dst.with_name(dst.name + suffix))
    logger.info(f"Backup complete in {time.time() - t0:.1f}s")
    return dst


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the duplicate-row counts and a current rating snapshot. No writes.",
    )
    parser.add_argument(
        "--skip-backup",
        action="store_true",
        help="Skip the timestamped DB backup. Use only when you already have one.",
    )
    args = parser.parse_args()

    init_franchise_tables()

    with get_db_connection() as conn:
        team_dupes, player_dupes = count_duplicate_history_rows(conn)
        logger.info(
            f"Pre-state: {team_dupes} (team_id, match_id) duplicate keys in team_elo_history, "
            f"{player_dupes} in player_elo_history"
        )
        before_teams = snapshot_team_elos(conn)
        before_players = snapshot_player_elos(conn)

    if args.dry_run:
        logger.info("Dry run: skipping backup + recalc")
        with get_db_connection() as conn:
            print_team_diff(before_teams, before_teams)
            print_player_diff(before_players, before_players)
        return 0

    if not args.skip_backup:
        backup_db()

    # Recalc - this is the heavy step. The calculator's force_recalculate=True
    # branch wipes non-snapshot history and current_elo before iterating.
    from src.elo.calculator_v3 import calculate_all_elos_v3  # noqa: E402

    logger.info("Starting full ELO recalc (force_recalculate=True). This can take 30+ minutes on large DBs.")
    t0 = time.time()
    calculate_all_elos_v3(force_recalculate=True)
    logger.info(f"Recalc complete in {time.time() - t0:.1f}s")

    # Now that history is clean, install the structural guard.
    with get_db_connection() as conn:
        add_unique_indexes(conn)
        team_dupes_after, player_dupes_after = count_duplicate_history_rows(conn)
        logger.info(
            f"Post-state: {team_dupes_after} team duplicates, "
            f"{player_dupes_after} player duplicates (both should be 0)"
        )
        after_teams = snapshot_team_elos(conn)
        after_players = snapshot_player_elos(conn)

    print_team_diff(before_teams, after_teams)
    print_player_diff(before_players, after_players)

    if team_dupes_after or player_dupes_after:
        logger.error("Duplicates remain after recalc - investigate!")
        return 1
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
