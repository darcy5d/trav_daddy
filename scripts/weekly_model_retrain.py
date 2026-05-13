#!/usr/bin/env python3
"""Wave 5.9.1: weekly neural-network model retrain.

Runs the full `full_retrain.run_full_pipeline()` workflow (ingest → ELO →
feature build → train V2+V3 → validate → save model_versions row) once a
week, or when models are older than MAX_AGE_DAYS days.

Designed to be invoked by cron every Sunday at 02:00 UTC — after Saturday
cricket concludes and before Monday markets open.

CRON ENTRY (add to crontab -e):
    0 2 * * 0  /path/to/venv311/bin/python /path/to/scripts/weekly_model_retrain.py >> data/logs/weekly_retrain.log 2>&1

USAGE
    venv311/bin/python scripts/weekly_model_retrain.py
    venv311/bin/python scripts/weekly_model_retrain.py --force          # bypass age check
    venv311/bin/python scripts/weekly_model_retrain.py --skip-ingest    # skip Cricsheet download
    venv311/bin/python scripts/weekly_model_retrain.py --skip-elo       # skip ELO recalc
    venv311/bin/python scripts/weekly_model_retrain.py --dry-run        # log what WOULD happen
    venv311/bin/python scripts/weekly_model_retrain.py --max-age-days 14
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import DATABASE_PATH  # noqa: E402
from src.data.database import get_db_connection, get_model_versions  # noqa: E402


LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
SCRIPT_LOG = LOG_DIR / "weekly_model_retrain.log"
PID_FILE = LOG_DIR / "weekly_model_retrain.pid"
STATUS_FILE = REPO_ROOT / "data" / "paper_trading" / "weekly_retrain_status.json"

DEFAULT_MAX_AGE_DAYS = 7


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(SCRIPT_LOG),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("weekly_model_retrain")


_SHUTDOWN = False


def _handle_signal(signum, _frame):
    global _SHUTDOWN
    logger.warning(f"Received signal {signum} — finishing current stage then exiting")
    _SHUTDOWN = True


def _acquire_pid_lock() -> None:
    """Single-instance lock. Exits the process if another retrain is running."""
    if PID_FILE.exists():
        try:
            other_pid = int(PID_FILE.read_text().strip())
            try:
                os.kill(other_pid, 0)
                logger.error(
                    f"Another retrain is already running (pid {other_pid}). "
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


def _should_retrain(max_age_days: int = DEFAULT_MAX_AGE_DAYS) -> tuple[bool, Optional[str]]:
    """Return (should_retrain, reason_string).

    True when any of the following hold:
    - No active model versions exist in the DB.
    - The oldest active model version was created > max_age_days ago.
    """
    try:
        versions = get_model_versions(active_only=True)
    except Exception as exc:
        logger.warning(f"Could not query model_versions: {exc} — assuming retrain needed")
        return True, f"model_versions query failed: {exc}"

    if not versions:
        return True, "no active model versions found in model_versions table"

    oldest_created_at = min(v["created_at"] for v in versions if v.get("created_at"))
    if not oldest_created_at:
        return True, "active model(s) have no created_at timestamp"

    # Handle both datetime objects and ISO strings
    if isinstance(oldest_created_at, str):
        oldest_dt = datetime.fromisoformat(oldest_created_at.replace("Z", "+00:00"))
    else:
        oldest_dt = oldest_created_at
    if oldest_dt.tzinfo is None:
        oldest_dt = oldest_dt.replace(tzinfo=timezone.utc)

    now_utc = datetime.now(timezone.utc)
    age_days = (now_utc - oldest_dt).total_seconds() / 86400.0
    if age_days > max_age_days:
        return True, f"oldest active model is {age_days:.1f}d old (threshold={max_age_days}d)"

    return False, f"models are {age_days:.1f}d old — within {max_age_days}d threshold"


def _get_active_model_summary() -> list[Dict[str, Any]]:
    """Return a compact list of active model version dicts for logging."""
    try:
        return [
            {
                "model_name": v.get("model_name"),
                "gender": v.get("gender"),
                "format_type": v.get("format_type"),
                "created_at": str(v.get("created_at", "")),
            }
            for v in get_model_versions(active_only=True)
        ]
    except Exception:
        return []


def _write_status(state: Dict[str, Any]) -> None:
    state["written_at_utc"] = datetime.now(timezone.utc).isoformat()
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with STATUS_FILE.open("w") as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as exc:
        logger.warning(f"Failed to write status file: {exc}")


def run_retrain(
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    force: bool = False,
    skip_ingest: bool = False,
    skip_elo: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Main retrain pipeline. Returns a status dict suitable for JSON dump."""
    started = datetime.now(timezone.utc)
    state: Dict[str, Any] = {
        "started_at_utc": started.isoformat(),
        "max_age_days": max_age_days,
        "force": force,
        "skip_ingest": skip_ingest,
        "skip_elo": skip_elo,
        "dry_run": dry_run,
        "stages": {},
        "errors": [],
    }

    logger.info("=== weekly_model_retrain starting ===")

    # ----- Age check -----
    should_train, reason = _should_retrain(max_age_days)
    state["age_check_reason"] = reason
    state["pre_active_models"] = _get_active_model_summary()

    if not force and not should_train:
        logger.info(f"Skipping retrain: {reason}")
        state["skipped"] = True
        state["success"] = True
        finished = datetime.now(timezone.utc)
        state["finished_at_utc"] = finished.isoformat()
        state["elapsed_s"] = round((finished - started).total_seconds(), 1)
        if not dry_run:
            _write_status(state)
        return state

    if force:
        logger.info(f"--force flag set; bypassing age check ({reason})")
    else:
        logger.info(f"Retrain triggered: {reason}")

    if dry_run:
        logger.info("[DRY-RUN] would run full_retrain.run_full_pipeline()")
        state["skipped"] = False
        state["dry_run_would_train"] = True
        state["success"] = True
        finished = datetime.now(timezone.utc)
        state["finished_at_utc"] = finished.isoformat()
        state["elapsed_s"] = round((finished - started).total_seconds(), 1)
        return state

    # ----- Run full retrain pipeline -----
    try:
        from scripts.full_retrain import run_full_pipeline  # noqa: E402
    except ImportError:
        try:
            # Fallback: import relative to REPO_ROOT already on sys.path
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "full_retrain", REPO_ROOT / "scripts" / "full_retrain.py"
            )
            full_retrain_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(full_retrain_mod)
            run_full_pipeline = full_retrain_mod.run_full_pipeline
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error(f"Could not import full_retrain.run_full_pipeline: {exc}\n{tb}")
            state["errors"].append({"stage": "import", "error": str(exc), "traceback": tb})
            state["success"] = False
            state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            state["elapsed_s"] = round(
                (datetime.now(timezone.utc) - started).total_seconds(), 1
            )
            _write_status(state)
            return state

    logger.info(
        f"Running run_full_pipeline(skip_ingest={skip_ingest}, skip_elo={skip_elo})"
    )
    t0 = time.time()
    try:
        pipeline_results = run_full_pipeline(
            skip_ingest=skip_ingest,
            skip_elo=skip_elo,
        )
        elapsed_pipeline = round(time.time() - t0, 1)
        state["stages"]["full_retrain"] = {
            "elapsed_s": elapsed_pipeline,
            "pipeline_results": pipeline_results,
        }
        logger.info(f"full_retrain pipeline completed in {elapsed_pipeline}s")
    except Exception as exc:
        tb = traceback.format_exc()
        elapsed_pipeline = round(time.time() - t0, 1)
        logger.error(f"full_retrain pipeline failed after {elapsed_pipeline}s: {exc}\n{tb}")
        state["errors"].append({"stage": "full_retrain", "error": str(exc), "traceback": tb})

    # ----- Post-retrain model snapshot -----
    state["post_active_models"] = _get_active_model_summary()
    if state["post_active_models"]:
        names = [m["model_name"] for m in state["post_active_models"]]
        logger.info(f"Active models after retrain: {names}")

    # ----- Summary -----
    finished = datetime.now(timezone.utc)
    state["finished_at_utc"] = finished.isoformat()
    state["elapsed_s"] = round((finished - started).total_seconds(), 1)
    state["success"] = len(state["errors"]) == 0
    state["skipped"] = False

    _write_status(state)

    print()
    print("=" * 70)
    print("WEEKLY MODEL RETRAIN SUMMARY")
    print("=" * 70)
    print(f"  Started:         {state['started_at_utc']}")
    print(f"  Elapsed:         {state['elapsed_s']}s")
    print(f"  Age check:       {reason}")
    print(f"  Force:           {force}")
    for stage_name, stage_data in state.get("stages", {}).items():
        if isinstance(stage_data, dict):
            elapsed = stage_data.get("elapsed_s", "-")
            print(f"  Stage {stage_name:<16} {elapsed}s")
    if state["errors"]:
        print(f"  Errors:          {len(state['errors'])}")
        for e in state["errors"]:
            print(f"    [{e['stage']}] {e['error']}")
    print(f"  Pre-models:      {[m['model_name'] for m in state.get('pre_active_models', [])]}")
    print(f"  Post-models:     {[m['model_name'] for m in state.get('post_active_models', [])]}")
    print(f"  Status file:     {STATUS_FILE}")
    print()

    return state


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Weekly NN model retrain (ingest → ELO → train → validate)"
    )
    parser.add_argument(
        "--max-age-days", type=int, default=DEFAULT_MAX_AGE_DAYS,
        help=f"Retrain if oldest active model is older than this many days (default {DEFAULT_MAX_AGE_DAYS})"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Bypass age check and retrain unconditionally"
    )
    parser.add_argument(
        "--skip-ingest", action="store_true",
        help="Skip Cricsheet data download/ingestion step"
    )
    parser.add_argument(
        "--skip-elo", action="store_true",
        help="Skip ELO recalculation step"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Log what would happen without running any training"
    )
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    if not args.dry_run:
        _acquire_pid_lock()

    try:
        state = run_retrain(
            max_age_days=args.max_age_days,
            force=args.force,
            skip_ingest=args.skip_ingest,
            skip_elo=args.skip_elo,
            dry_run=args.dry_run,
        )
        return 0 if state["success"] else 1
    finally:
        if not args.dry_run:
            _release_pid_lock()


if __name__ == "__main__":
    sys.exit(main())
