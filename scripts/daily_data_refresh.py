#!/usr/bin/env python3
"""Wave 5.7 follow-up: daily Cricsheet data refresh + ELO recalc.

Runs four stages sequentially:
    1. Download Cricsheet ZIPs (force-redownload if local is older than 12h)
    2. Ingest new match JSONs into SQLite (idempotent via match_exists)
    3. Snapshot DB to a rolling backup (`cricket.db.bak.daily_refresh`)
       and recalculate ELOs from scratch via calculator_v3
    4. Write a status JSON + log a one-line summary

Designed to be invoked by cron once a day at 11:00 UTC. Safe to re-run
manually; PID-file lock prevents concurrent runs.

USAGE
    venv311/bin/python scripts/daily_data_refresh.py
    venv311/bin/python scripts/daily_data_refresh.py --dry-run
    venv311/bin/python scripts/daily_data_refresh.py --skip-download
    venv311/bin/python scripts/daily_data_refresh.py --skip-elo

OUT OF SCOPE (for now)
    - Vocab rebuild + V2/V3 model retrain (monthly cadence, separate workflow)
    - Smart incremental ELO recalc (would skip already-processed matches; the
      cross-pool tier-promotion logic in calculator_v3 needs to see the full
      chronology, so safest path is full recalc)
    - CrEx/CrickAPI metadata backfill for newly ingested matches
    - Notification on failure (email/Slack)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import DATABASE_PATH, RAW_DATA_DIR  # noqa: E402
from src.data.database import get_db_connection  # noqa: E402
from src.data.downloader import download_cricsheet_data  # noqa: E402
from src.data.ingest import ingest_matches  # noqa: E402
from src.elo.calculator_v3 import calculate_all_elos_v3  # noqa: E402


LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
SCRIPT_LOG = LOG_DIR / "daily_data_refresh.log"
PID_FILE = LOG_DIR / "daily_data_refresh.pid"
STATUS_FILE = REPO_ROOT / "data" / "paper_trading" / "daily_data_refresh_status.json"
BACKUP_PATH = DATABASE_PATH.parent / "cricket.db.bak.daily_refresh"

DEFAULT_FORMATS = ["all_male", "all_female"]
DEFAULT_AGE_THRESHOLD_HOURS = 12.0


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(SCRIPT_LOG),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("daily_data_refresh")


_SHUTDOWN = False


def _handle_signal(signum, _frame):
    global _SHUTDOWN
    logger.warning(f"Received signal {signum}, finishing current stage then exiting")
    _SHUTDOWN = True


def _acquire_pid_lock() -> None:
    """Single-instance lock. Exits the process if another run is in flight."""
    if PID_FILE.exists():
        try:
            other_pid = int(PID_FILE.read_text().strip())
            try:
                os.kill(other_pid, 0)
                logger.error(
                    f"Another refresh is already running (pid {other_pid}). "
                    f"To force-restart, kill that process or `rm {PID_FILE}` if stale."
                )
                sys.exit(1)
            except OSError:
                logger.warning(f"Stale PID file (pid {other_pid} not running), reclaiming")
        except (ValueError, OSError):
            pass
    PID_FILE.write_text(str(os.getpid()))


def _release_pid_lock() -> None:
    if not PID_FILE.exists():
        return
    try:
        if PID_FILE.read_text().strip() == str(os.getpid()):
            PID_FILE.unlink()
    except Exception:
        pass


def _count_matches() -> int:
    """Total rows in the matches table."""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM matches")
        return int(cur.fetchone()[0])


def _max_match_date() -> Optional[str]:
    """ISO date string of the latest match in DB (for logging only)."""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT MAX(date) FROM matches")
        row = cur.fetchone()
        return row[0] if row else None


def _save_rolling_backup() -> Optional[Path]:
    """Copy the live cricket.db to a single rolling backup file.

    Overwrites the previous daily backup so disk usage is capped at 2x DB size.
    Skips if the DB doesn't exist yet (edge case for fresh installs).
    """
    if not DATABASE_PATH.exists():
        logger.warning(f"DB at {DATABASE_PATH} does not exist; skipping backup")
        return None
    logger.info(f"Snapshotting DB to {BACKUP_PATH} (rolling)")
    shutil.copy2(DATABASE_PATH, BACKUP_PATH)
    size_mb = BACKUP_PATH.stat().st_size / (1024 * 1024)
    logger.info(f"  Backup size: {size_mb:.1f} MB")
    return BACKUP_PATH


def _write_status(state: Dict[str, Any]) -> None:
    state["written_at_utc"] = datetime.now(timezone.utc).isoformat()
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with STATUS_FILE.open("w") as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as exc:
        logger.warning(f"Failed to write status file: {exc}")


def run_refresh(
    formats: Optional[List[str]] = None,
    age_threshold_hours: float = DEFAULT_AGE_THRESHOLD_HOURS,
    skip_download: bool = False,
    skip_elo: bool = False,
    skip_backup: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Main pipeline. Returns a status dict suitable for JSON dump."""
    if formats is None:
        formats = DEFAULT_FORMATS

    started = datetime.now(timezone.utc)
    state: Dict[str, Any] = {
        "started_at_utc": started.isoformat(),
        "formats": formats,
        "age_threshold_hours": age_threshold_hours,
        "skip_download": skip_download,
        "skip_elo": skip_elo,
        "skip_backup": skip_backup,
        "dry_run": dry_run,
        "stages": {},
        "errors": [],
    }

    pre_count = _count_matches()
    pre_max_date = _max_match_date()
    state["pre_total_matches"] = pre_count
    state["pre_max_match_date"] = pre_max_date
    logger.info(f"=== daily_data_refresh starting ===")
    logger.info(f"  pre-state: {pre_count:,} matches, max date={pre_max_date}")
    if dry_run:
        logger.info("  [DRY-RUN] no downloads, no DB writes, no recalcs")

    # ----- Stage 1: download -----
    if skip_download:
        logger.info("Stage 1 skipped (--skip-download)")
        state["stages"]["download"] = {"skipped": True}
    elif dry_run:
        logger.info("Stage 1 [DRY-RUN]: would download cricsheet zips for "
                    f"formats={formats} (age_threshold={age_threshold_hours}h)")
        state["stages"]["download"] = {"dry_run": True, "would_run": True}
    else:
        logger.info(f"Stage 1: download cricsheet (formats={formats}, "
                    f"age_threshold={age_threshold_hours}h)")
        t0 = time.time()
        try:
            results = download_cricsheet_data(
                formats=formats,
                age_threshold_hours=age_threshold_hours,
            )
            state["stages"]["download"] = {
                "elapsed_s": round(time.time() - t0, 1),
                "formats_returned": list(results.keys()),
                "format_paths": {k: str(v) for k, v in results.items()},
            }
            logger.info(f"  download complete in {state['stages']['download']['elapsed_s']}s")
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error(f"Stage 1 download failed: {exc}\n{tb}")
            state["errors"].append({"stage": "download", "error": str(exc), "traceback": tb})

    # ----- Stage 2: ingest -----
    if dry_run:
        logger.info("Stage 2 [DRY-RUN]: would ingest matches")
        state["stages"]["ingest"] = {"dry_run": True, "would_run": True}
    else:
        logger.info(f"Stage 2: ingest matches (formats={formats})")
        t0 = time.time()
        try:
            ingest_stats = ingest_matches(formats=formats)
            state["stages"]["ingest"] = {
                "elapsed_s": round(time.time() - t0, 1),
                **ingest_stats,
            }
            logger.info(f"  ingest complete in {state['stages']['ingest']['elapsed_s']}s: "
                        f"processed={ingest_stats.get('matches_processed', 0)} "
                        f"skipped={ingest_stats.get('matches_skipped', 0)} "
                        f"failed={ingest_stats.get('matches_failed', 0)}")
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error(f"Stage 2 ingest failed: {exc}\n{tb}")
            state["errors"].append({"stage": "ingest", "error": str(exc), "traceback": tb})

    # ----- New-match accounting -----
    post_count = _count_matches() if not dry_run else pre_count
    post_max_date = _max_match_date() if not dry_run else pre_max_date
    new_matches = post_count - pre_count
    state["post_total_matches"] = post_count
    state["post_max_match_date"] = post_max_date
    state["new_matches"] = new_matches
    logger.info(f"  new matches ingested: {new_matches} (DB now {post_count:,}, max date={post_max_date})")

    # ----- Stage 3: backup + ELO recalc (only if there's something new) -----
    if dry_run:
        logger.info("Stage 3 [DRY-RUN]: would back up DB and recalc ELOs if new_matches > 0")
        state["stages"]["backup"] = {"dry_run": True}
        state["stages"]["elo_recalc"] = {"dry_run": True, "would_run": new_matches > 0}
    elif skip_elo:
        logger.info("Stage 3 ELO recalc skipped (--skip-elo)")
        state["stages"]["elo_recalc"] = {"skipped": True}
    elif new_matches <= 0:
        logger.info("Stage 3 ELO recalc skipped: no new matches in DB")
        state["stages"]["elo_recalc"] = {"skipped": True, "reason": "no-new-matches"}
    else:
        # Backup first
        if skip_backup:
            logger.info("Stage 3a backup skipped (--skip-backup)")
            state["stages"]["backup"] = {"skipped": True}
        else:
            t0 = time.time()
            try:
                bp = _save_rolling_backup()
                state["stages"]["backup"] = {
                    "elapsed_s": round(time.time() - t0, 1),
                    "path": str(bp) if bp else None,
                    "size_mb": round(bp.stat().st_size / (1024 * 1024), 1) if bp else None,
                }
            except Exception as exc:
                tb = traceback.format_exc()
                logger.error(f"Stage 3a backup failed: {exc}\n{tb}")
                state["errors"].append({"stage": "backup", "error": str(exc), "traceback": tb})

        # ELO recalc
        logger.info(f"Stage 3b: full ELO recalc via calculator_v3 (force_recalculate=True)")
        t0 = time.time()
        try:
            calculate_all_elos_v3(force_recalculate=True)
            state["stages"]["elo_recalc"] = {
                "elapsed_s": round(time.time() - t0, 1),
                "force_recalculate": True,
            }
            logger.info(f"  ELO recalc complete in {state['stages']['elo_recalc']['elapsed_s']}s")
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error(f"Stage 3b ELO recalc failed: {exc}\n{tb}")
            state["errors"].append({"stage": "elo_recalc", "error": str(exc), "traceback": tb})

    # ----- Stage 4: status + summary -----
    finished = datetime.now(timezone.utc)
    state["finished_at_utc"] = finished.isoformat()
    state["elapsed_s"] = round((finished - started).total_seconds(), 1)
    state["success"] = len(state["errors"]) == 0

    if not dry_run:
        _write_status(state)

    print()
    print("=" * 70)
    print("DAILY DATA REFRESH SUMMARY")
    print("=" * 70)
    print(f"  Started:            {state['started_at_utc']}")
    print(f"  Elapsed:            {state['elapsed_s']}s")
    print(f"  Pre-state:          {pre_count:,} matches (max date {pre_max_date})")
    print(f"  Post-state:         {post_count:,} matches (max date {post_max_date})")
    print(f"  New matches:        {new_matches}")
    for stage_name, stage_data in state["stages"].items():
        if isinstance(stage_data, dict):
            elapsed = stage_data.get("elapsed_s", "-")
            skipped = stage_data.get("skipped", False)
            tag = "[skipped]" if skipped else f"{elapsed}s"
            print(f"  Stage {stage_name:<14} {tag}")
    if state["errors"]:
        print(f"  Errors:             {len(state['errors'])}")
        for e in state["errors"]:
            print(f"    [{e['stage']}] {e['error']}")
    print(f"  Status file:        {STATUS_FILE if not dry_run else '(dry-run, not written)'}")
    print()

    return state


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily Cricsheet refresh + ELO recalc")
    parser.add_argument("--formats", nargs="+", default=DEFAULT_FORMATS,
                        help="Cricsheet format keys (default: all_male all_female)")
    parser.add_argument("--age-threshold-hours", type=float, default=DEFAULT_AGE_THRESHOLD_HOURS,
                        help="Re-download zips if local is older than this (default 12)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Use existing zips on disk; skip download")
    parser.add_argument("--skip-elo", action="store_true",
                        help="Skip ELO recalc even if new matches were ingested")
    parser.add_argument("--skip-backup", action="store_true",
                        help="Skip the rolling DB backup before ELO recalc (saves ~3-8GB I/O)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what WOULD happen without making any changes")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    if not args.dry_run:
        _acquire_pid_lock()

    try:
        state = run_refresh(
            formats=args.formats,
            age_threshold_hours=args.age_threshold_hours,
            skip_download=args.skip_download,
            skip_elo=args.skip_elo,
            skip_backup=args.skip_backup,
            dry_run=args.dry_run,
        )
        return 0 if state["success"] else 1
    finally:
        if not args.dry_run:
            _release_pid_lock()


if __name__ == "__main__":
    sys.exit(main())
