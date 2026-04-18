#!/usr/bin/env python3
"""
Backfill explicit franchise unifications on top of the V4 schema.

`init_franchise_tables()` already self-groups every existing team (one
team_groups row per teams row). This script merges known multi-id franchises
under a single team_groups row and elects one team_id as the canonical owner
of the rating series.

Designed to be:
  * Idempotent. Re-running is a no-op once everything is in place.
  * Auditable. --dry-run prints every change without touching the DB.
  * Conservative. Only the explicitly listed unifications are applied; anything
    ambiguous is left alone for the Team Explorer UI to handle later.

Run:
    python scripts/backfill_franchises.py --dry-run
    python scripts/backfill_franchises.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_connection, init_franchise_tables  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class Unification:
    """One franchise unification spec."""

    canonical_name: str          # The team_groups.canonical_name we want to end up with.
    group_type: str              # franchise / national / etc.
    country_code: Optional[str]  # ISO-style code if known; left NULL otherwise.
    canonical_team_id: int       # Which teams.team_id should own the rating series.
    member_team_ids: List[int]   # All teams.team_id rows belonging to this group.
    notes: str = ""
    flag_only: bool = False      # If True, do nothing destructive; just print + record a note.


# Explicit IPL/T20 franchise unifications. Order doesn't matter; each is applied
# independently. Keep this list short and high-confidence; the Team Explorer UI
# is the right place for everything else.
UNIFICATIONS: List[Unification] = [
    Unification(
        canonical_name="Royal Challengers Bengaluru",
        group_type="franchise",
        country_code="IND",
        canonical_team_id=283,         # Bengaluru (modern)
        member_team_ids=[111, 283],     # Bangalore + Bengaluru
        notes="Rebranded mid-2024 from Royal Challengers Bangalore.",
    ),
    Unification(
        canonical_name="Delhi Capitals",
        group_type="franchise",
        country_code="IND",
        canonical_team_id=115,         # Capitals (modern)
        member_team_ids=[364, 115],     # Daredevils + Capitals
        notes="Rebranded from Delhi Daredevils ahead of IPL 2019.",
    ),
    Unification(
        canonical_name="Punjab Kings",
        group_type="franchise",
        country_code="IND",
        canonical_team_id=196,         # Punjab Kings (modern)
        member_team_ids=[117, 196],     # Kings XI Punjab + Punjab Kings
        notes="Rebranded from Kings XI Punjab ahead of IPL 2021.",
    ),
    Unification(
        canonical_name="Rising Pune Supergiant",
        group_type="franchise",
        country_code="IND",
        canonical_team_id=362,         # 2017 spelling
        member_team_ids=[362, 370],     # Supergiant + Supergiants (s dropped in 2017)
        notes="Same franchise across IPL 2016 (Supergiants) and IPL 2017 (Supergiant).",
    ),
]

# Soft flags. We don't merge these automatically; the Team Explorer UI will
# surface them as candidates so a human can confirm.
FLAGS_FOR_REVIEW: List[dict] = [
    {
        "candidate_team_ids": [113],
        "rationale": (
            "Sunrisers Hyderabad: Deccan Chargers predecessor (2008-2012) is not "
            "currently in the database; if Deccan rows are added later, propose merging "
            "them into the Sunrisers Hyderabad group."
        ),
    },
]


def get_or_create_group(
    conn,
    canonical_name: str,
    group_type: str,
    country_code: Optional[str],
    notes: str,
    dry_run: bool,
) -> Optional[int]:
    """Return the group_id for `canonical_name`, creating or upgrading metadata.

    When the row already exists from the V4 self-grouping pass, its group_type
    and country_code may be the generic auto-assigned defaults. Promote them to
    whatever this unification specifies so the explorer / API always sees the
    richer metadata.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT group_id, group_type, country_code, notes
        FROM team_groups WHERE canonical_name = ?
        """,
        (canonical_name,),
    )
    row = cur.fetchone()
    if row:
        needs_update = (
            row["group_type"] != group_type
            or (country_code and row["country_code"] != country_code)
        )
        if needs_update:
            if dry_run:
                logger.info(
                    f"[DRY RUN] Would UPDATE team_groups #{row['group_id']} "
                    f"'{canonical_name}': group_type {row['group_type']} -> {group_type}, "
                    f"country_code {row['country_code']} -> {country_code}"
                )
            else:
                # Append the new note instead of clobbering the existing one.
                merged_notes = (
                    f"{row['notes']}\n{notes}".strip()
                    if row["notes"] and notes and notes not in (row["notes"] or "")
                    else (notes or row["notes"])
                )
                cur.execute(
                    """
                    UPDATE team_groups
                    SET group_type = ?, country_code = ?, notes = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE group_id = ?
                    """,
                    (group_type, country_code, merged_notes, row["group_id"]),
                )
        return row["group_id"]

    if dry_run:
        logger.info(f"[DRY RUN] Would CREATE team_groups row '{canonical_name}' ({group_type})")
        return None

    cur.execute(
        """
        INSERT INTO team_groups (canonical_name, group_type, country_code, notes)
        VALUES (?, ?, ?, ?)
        """,
        (canonical_name, group_type, country_code, notes),
    )
    return cur.lastrowid


def cleanup_orphan_self_groups(conn, dry_run: bool) -> int:
    """Remove team_groups rows that no team currently points at.

    Self-grouped rows from the V4 migration become orphans once a team is
    re-pointed to a unified franchise group. We don't drop the FK chain (so
    other tables that already reference these groups stay valid); we just
    remove rows that are demonstrably abandoned.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT g.group_id, g.canonical_name
        FROM team_groups g
        LEFT JOIN teams t ON t.franchise_id = g.group_id
        LEFT JOIN team_external_ids ext ON ext.group_id = g.group_id
        LEFT JOIN team_merge_proposals p ON p.target_group_id = g.group_id
        WHERE t.team_id IS NULL
          AND ext.group_id IS NULL
          AND p.proposal_id IS NULL
        """
    )
    orphans = cur.fetchall()
    if not orphans:
        return 0

    if dry_run:
        for o in orphans:
            logger.info(f"[DRY RUN] Would DELETE orphan team_groups row #{o['group_id']} '{o['canonical_name']}'")
        return len(orphans)

    cur.executemany(
        "DELETE FROM team_groups WHERE group_id = ?",
        [(o["group_id"],) for o in orphans],
    )
    return len(orphans)


def apply_unification(conn, u: Unification, dry_run: bool) -> bool:
    """Apply one unification. Returns True if any row was changed."""
    cur = conn.cursor()

    # Sanity: every member team must exist.
    cur.execute(
        f"SELECT team_id, name, franchise_id, canonical_team_id FROM teams "
        f"WHERE team_id IN ({','.join('?' * len(u.member_team_ids))})",
        u.member_team_ids,
    )
    rows = {r["team_id"]: r for r in cur.fetchall()}
    missing = [tid for tid in u.member_team_ids if tid not in rows]
    if missing:
        logger.warning(f"[{u.canonical_name}] missing team_ids in DB: {missing} - skipping unification")
        return False
    if u.canonical_team_id not in rows:
        logger.warning(f"[{u.canonical_name}] canonical_team_id {u.canonical_team_id} not in member_team_ids - skipping")
        return False

    group_id = get_or_create_group(
        conn,
        canonical_name=u.canonical_name,
        group_type=u.group_type,
        country_code=u.country_code,
        notes=u.notes,
        dry_run=dry_run,
    )
    if group_id is None and not dry_run:
        logger.error(f"[{u.canonical_name}] failed to obtain group_id")
        return False

    changed = False
    for tid in u.member_team_ids:
        row = rows[tid]
        already_grouped = row["franchise_id"] == group_id
        already_canonical = row["canonical_team_id"] == u.canonical_team_id
        if already_grouped and already_canonical:
            continue

        changed = True
        if dry_run:
            logger.info(
                f"[DRY RUN] Would UPDATE team #{tid} ({row['name']}): "
                f"franchise_id {row['franchise_id']} -> {group_id}, "
                f"canonical_team_id {row['canonical_team_id']} -> {u.canonical_team_id}"
            )
        else:
            cur.execute(
                """
                UPDATE teams
                SET franchise_id = ?, canonical_team_id = ?
                WHERE team_id = ?
                """,
                (group_id, u.canonical_team_id, tid),
            )

    if changed:
        logger.info(
            f"[{u.canonical_name}] unified {len(u.member_team_ids)} team_ids "
            f"under group_id {group_id} (canonical team_id={u.canonical_team_id})"
        )
    else:
        logger.info(f"[{u.canonical_name}] already unified - no change")
    return changed


def report_flags(conn) -> None:
    if not FLAGS_FOR_REVIEW:
        return
    logger.info("Soft flags for review (not auto-applied):")
    for f in FLAGS_FOR_REVIEW:
        logger.info(f"  - team_ids {f['candidate_team_ids']}: {f['rationale']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print intended changes without writing to the database.",
    )
    args = parser.parse_args()

    init_franchise_tables()

    with get_connection() as conn:
        any_changed = False
        for u in UNIFICATIONS:
            if apply_unification(conn, u, dry_run=args.dry_run):
                any_changed = True

        orphans_removed = cleanup_orphan_self_groups(conn, dry_run=args.dry_run)
        if orphans_removed:
            logger.info(f"Cleaned up {orphans_removed} orphan team_groups rows")

        if not args.dry_run:
            conn.commit()

    report_flags(conn)
    if args.dry_run:
        logger.info("Dry run complete - no changes written.")
    elif any_changed or orphans_removed:
        logger.info("Backfill complete.")
    else:
        logger.info("Backfill complete - no changes needed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
