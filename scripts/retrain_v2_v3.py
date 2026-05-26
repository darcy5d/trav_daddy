#!/usr/bin/env python3
"""Wave 5.11: Automated V2/V3 model retrain orchestrator.

Retrains the two multi-task cricket simulators that live/paper betting actually
uses (V2 at data/models/v2/ and V3 at data/models/v3/).

Why this script exists
----------------------
scripts/full_retrain.py and scripts/weekly_model_retrain.py only retrain the
legacy V1 ball-prediction model (data/processed/ball_prediction_model_*.keras),
which is NOT used by live_bet_scan.py or paper_bet_scan.py.  Those scanners
load V2Simulator / V3Simulator, whose default model paths point at this repo's
data/models/v2/ and data/models/v3/ directories.

Pipeline
--------
1.  build_ball_training_v2  →  data/processed/ball_training_v2_{gender}.npz
2.  train_cricket_model_v2  →  data/models/v2/cricket_model_v2.keras + vocabs
3.  build_ball_training_v3  →  data/processed/ball_training_v3_{gender}.npz
4.  train_cricket_model_v3  →  data/models/v3/cricket_model_v3.keras + vocabs

Calibration (data/models/v2/calibration.json and v3/calibration.json) is kept
from the previous run unless --refit-calibration is given.  Re-fitting requires
pre-existing backtest CSVs / calibration_master.csv; skip it in automated cron
and refit manually after a dedicated backtest run.

Usage
-----
    # Full retrain (Tue/Fri cron)
    venv311/bin/python scripts/retrain_v2_v3.py

    # Skip NPZ rebuild if training data is fresh (saves ~15 min)
    venv311/bin/python scripts/retrain_v2_v3.py --skip-build

    # Dry-run: log what would happen, no files written
    venv311/bin/python scripts/retrain_v2_v3.py --dry-run

    # Force even if data hasn't changed
    venv311/bin/python scripts/retrain_v2_v3.py --force

    # Also refit V3 calibration (requires calibration_master.csv)
    venv311/bin/python scripts/retrain_v2_v3.py --refit-calibration

CRON ENTRIES (added by setup — do not edit here)
    0 2 * * 2,5  cd REPO && venv311/bin/python scripts/retrain_v2_v3.py >> logs/v2v3_retrain.log 2>&1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

STATUS_FILE = REPO_ROOT / "data" / "paper_trading" / "v2v3_retrain_status.json"
PID_FILE = LOG_DIR / "retrain_v2_v3.pid"

PYTHON = str(REPO_ROOT / "venv311" / "bin" / "python")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("retrain_v2_v3")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _run(cmd: list[str], step_name: str, dry_run: bool = False) -> Dict[str, Any]:
    """Run a subprocess step and return a result dict."""
    logger.info(f"[{step_name}] Running: {' '.join(cmd)}")
    if dry_run:
        logger.info(f"[{step_name}] DRY-RUN — skipping")
        return {"step": step_name, "skipped": True, "elapsed_s": 0}

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=False,   # let output stream to log
            text=True,
        )
        elapsed = round(time.time() - t0, 1)
        if result.returncode != 0:
            raise RuntimeError(
                f"[{step_name}] exited with code {result.returncode}"
            )
        logger.info(f"[{step_name}] Done in {elapsed}s")
        return {"step": step_name, "elapsed_s": elapsed, "returncode": 0}
    except Exception as exc:
        elapsed = round(time.time() - t0, 1)
        logger.error(f"[{step_name}] FAILED after {elapsed}s: {exc}")
        raise


def _model_info(keras_path: Path) -> Dict[str, Any]:
    """Return size and mtime for a .keras file, or empty dict if missing."""
    if not keras_path.exists():
        return {}
    stat = keras_path.stat()
    return {
        "path": str(keras_path),
        "size_mb": round(stat.st_size / 1_048_576, 2),
        "mtime_utc": datetime.utcfromtimestamp(stat.st_mtime).isoformat(),
    }


def _db_max_date() -> Optional[str]:
    """Query cricket.db for the latest match date (quick sanity check)."""
    try:
        import sqlite3
        from config import DATABASE_PATH
        conn = sqlite3.connect(str(DATABASE_PATH))
        row = conn.execute("SELECT MAX(date) FROM matches").fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None


def _write_status(payload: Dict[str, Any]) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(payload, indent=2))
    logger.info(f"Status written → {STATUS_FILE}")


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_v2_v3_pipeline(
    dry_run: bool = False,
    skip_build: bool = False,
    refit_calibration: bool = False,
) -> Dict[str, Any]:
    """Execute the full V2/V3 retrain pipeline.  Returns a status dict."""

    started = datetime.now(timezone.utc).isoformat()
    t_global = time.time()
    results: list[Dict[str, Any]] = []
    errors: list[str] = []

    db_max_date = _db_max_date()
    logger.info(f"DB max match date: {db_max_date}")

    def step(cmd: list[str], name: str) -> bool:
        try:
            r = _run(cmd, name, dry_run=dry_run)
            results.append(r)
            return True
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            return False

    # ── Stage 1: Build V2 training NPZ ──
    if not skip_build:
        ok = step(
            [PYTHON, "scripts/build_ball_training_v2.py", "--gender", "both"],
            "build_v2",
        )
        if not ok:
            return _finish(started, t_global, results, errors, dry_run)
    else:
        logger.info("[build_v2] Skipped (--skip-build)")
        results.append({"step": "build_v2", "skipped": True})

    # ── Stage 2: Train V2 model ──
    ok = step(
        [PYTHON, "scripts/train_cricket_model_v2.py"],
        "train_v2",
    )
    if not ok:
        return _finish(started, t_global, results, errors, dry_run)

    # ── Stage 3: Build V3 training NPZ ──
    if not skip_build:
        ok = step(
            [PYTHON, "scripts/build_ball_training_v3.py", "--gender", "both"],
            "build_v3",
        )
        if not ok:
            return _finish(started, t_global, results, errors, dry_run)
    else:
        logger.info("[build_v3] Skipped (--skip-build)")
        results.append({"step": "build_v3", "skipped": True})

    # ── Stage 4: Train V3 model ──
    ok = step(
        [PYTHON, "scripts/train_cricket_model_v3.py"],
        "train_v3",
    )
    if not ok:
        return _finish(started, t_global, results, errors, dry_run)

    # ── Stage 5 (optional): Refit V3 calibration ──
    if refit_calibration:
        calib_master = REPO_ROOT / "data/diagnostics/wave_5_6_modern_era_sweep/calibration_master.csv"
        if calib_master.exists():
            step(
                [PYTHON, "scripts/refit_v3_calibration_w56.py"],
                "refit_v3_calibration",
            )
        else:
            msg = (
                f"[refit_calibration] Skipped — calibration_master.csv not found "
                f"at {calib_master}. Run a backtest sweep first."
            )
            logger.warning(msg)
            results.append({"step": "refit_v3_calibration", "skipped": True, "reason": str(msg)})
    else:
        logger.info(
            "[calibration] Keeping existing calibration.json files "
            "(pass --refit-calibration to update)"
        )

    return _finish(started, t_global, results, errors, dry_run)


def _finish(
    started: str,
    t_global: float,
    results: list,
    errors: list,
    dry_run: bool,
) -> Dict[str, Any]:
    elapsed = round(time.time() - t_global, 1)
    success = len(errors) == 0

    v2_info = _model_info(REPO_ROOT / "data/models/v2/cricket_model_v2.keras")
    v3_info = _model_info(REPO_ROOT / "data/models/v3/cricket_model_v3.keras")

    status = {
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": elapsed,
        "dry_run": dry_run,
        "success": success,
        "db_max_date": _db_max_date(),
        "stages": results,
        "errors": errors,
        "v2_model": v2_info,
        "v3_model": v3_info,
    }

    _write_status(status)

    if success:
        logger.info(
            f"V2/V3 retrain {'(DRY-RUN) ' if dry_run else ''}completed in {elapsed}s"
        )
        if v2_info:
            logger.info(f"  V2: {v2_info['size_mb']} MB  mtime={v2_info['mtime_utc']}")
        if v3_info:
            logger.info(f"  V3: {v3_info['size_mb']} MB  mtime={v3_info['mtime_utc']}")
    else:
        logger.error(f"V2/V3 retrain FAILED after {elapsed}s")
        for e in errors:
            logger.error(f"  {e}")

    return status


# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

def _handle_signal(sig, frame):
    logger.warning(f"Caught signal {sig} — cleaning up and exiting")
    PID_FILE.unlink(missing_ok=True)
    sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retrain V2 and V3 cricket betting models."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Log what would happen without running any training steps.",
    )
    parser.add_argument(
        "--skip-build", action="store_true",
        help="Skip NPZ feature rebuild (build_ball_training_v2/v3). "
             "Use when data/processed/ball_training_v*_*.npz are already up to date.",
    )
    parser.add_argument(
        "--refit-calibration", action="store_true",
        help="Also refit V3 Platt calibration via refit_v3_calibration_w56.py. "
             "Requires data/diagnostics/wave_5_6_modern_era_sweep/calibration_master.csv.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Alias for clarity; same as default behaviour (always retrains).",
    )
    args = parser.parse_args()

    # PID guard (prevent two simultaneous retrains)
    if PID_FILE.exists():
        old_pid = PID_FILE.read_text().strip()
        try:
            os.kill(int(old_pid), 0)
            logger.error(
                f"Another retrain process appears to be running (PID {old_pid}). "
                "Remove {PID_FILE} if stale."
            )
            return 1
        except (ProcessLookupError, ValueError):
            PID_FILE.unlink(missing_ok=True)

    PID_FILE.write_text(str(os.getpid()))
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    try:
        logger.info(
            f"=== retrain_v2_v3 started "
            f"(dry_run={args.dry_run}, skip_build={args.skip_build}, "
            f"refit_calibration={args.refit_calibration}) ==="
        )
        status = run_v2_v3_pipeline(
            dry_run=args.dry_run,
            skip_build=args.skip_build,
            refit_calibration=args.refit_calibration,
        )
        return 0 if status["success"] else 1
    except Exception as exc:
        logger.error(f"Unexpected error: {exc}\n{traceback.format_exc()}")
        return 1
    finally:
        PID_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
