#!/usr/bin/env python3
"""Catch-up trigger for the weekly CricketArchive top-up.

Why this exists: StartCalendarInterval does not fire a job whose time passed
while the machine was asleep, and does not fire at all for that week if the Mac
was off, logged out, or the agent was only loaded after 03:00. A LaunchAgent
runs this every 30 minutes; it starts the same top-up the weekly job runs, but
only when the last successful run is stale.

Why Python rather than the shell script: see scripts/ca_topup_catchup.sh. Under
launchd on this machine /bin/bash is denied TCC access to ~/Desktop (which is
where this repo lives), so anything bash has to *read or exec* under the repo
fails with EPERM. The venv python binary is not denied, so the logic lives here;
the .sh wrapper uses `exec` on this file so bash never opens it.

Guarantees:
  * Idempotent  — a fresh stamp turns every tick into a cheap no-op.
  * No overlap  — refuses while any ca_topup.py / ca_harvest_parallel.py runs
                  and holds an atomic mkdir lock against concurrent ticks.
                  scripts/ca_topup.py has its own guard as a third layer.
  * Success only — the stamp is only touched when the run exits 0.

Usage:
    scripts/ca_topup_catchup.py            # catch-up: run only if stale
    scripts/ca_topup_catchup.py --force    # weekly: run unless it just ran
    scripts/ca_topup_catchup.py --dry-run  # print the decision, run nothing
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

MAX_AGE_SECS = 7 * 24 * 60 * 60
# Floor that even --force cannot override. Without it a missed-wake replay of
# the Sunday job could start a second full run minutes after a catch-up
# already finished one.
MIN_AGE_SECS = 6 * 60 * 60
# A lock older than this is from a killed run, not a live one.
LOCK_STALE_SECS = 6 * 60 * 60

OVERLAP_PATTERNS = ("ca_topup.py", "ca_harvest_parallel.py")


def _ts() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


class Runner:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.log = root / "logs" / "ca_topup.log"
        self.trigger_log = root / "logs" / "ca_topup_catchup.log"
        self.stamp = root / "data" / "raw" / "cricketarchive" / ".last_topup_success"
        self.lock = root / "logs" / ".ca_topup_catchup.lock"
        self.python = root / "venv311" / "bin" / "python"
        self.topup = root / "scripts" / "ca_topup.py"

    def emit(self, msg: str) -> None:
        line = f"{_ts()} catchup[{os.getpid()}]: {msg}"
        print(line, flush=True)
        try:
            with self.trigger_log.open("a") as fh:
                fh.write(line + "\n")
        except OSError:
            pass

    def skip(self, msg: str) -> int:
        self.emit(msg)
        return 0

    def running_pids(self) -> list[str]:
        """PIDs of other top-up/harvest processes (never our own)."""
        found: list[str] = []
        for pattern in OVERLAP_PATTERNS:
            try:
                proc = subprocess.run(
                    ["pgrep", "-f", pattern],
                    capture_output=True, text=True, timeout=10,
                )
            except (OSError, subprocess.SubprocessError):
                continue
            if proc.returncode != 0:
                continue
            for pid in proc.stdout.split():
                if pid.isdigit() and int(pid) != os.getpid():
                    found.append(f"{pattern}:{pid}")
        return found

    def stamp_age(self) -> int | None:
        try:
            return int(time.time() - self.stamp.stat().st_mtime)
        except OSError:
            return None

    def take_lock(self) -> bool:
        try:
            self.lock.mkdir()
            return True
        except FileExistsError:
            try:
                lock_age = time.time() - self.lock.stat().st_mtime
            except OSError:
                lock_age = 0
            if lock_age <= LOCK_STALE_SECS:
                self.emit(f"skip: lock held by another run ({int(lock_age)}s)")
                return False
            self.emit(f"removing stale lock ({int(lock_age)}s old)")
            try:
                self.lock.rmdir()
                self.lock.mkdir()
                return True
            except OSError:
                self.emit("skip: could not take lock")
                return False

    def release_lock(self) -> None:
        try:
            self.lock.rmdir()
        except OSError:
            pass

    def topup_cmd(self) -> str:
        """The same string the weekly plist runs, quoted for the log."""
        return (
            f"cd {self.root} && AUTO_INSTALL_PLAYWRIGHT_CHROMIUM=false "
            f"{self.python} {self.topup} --harvest >> {self.log} 2>&1"
        )

    def run(self, mode: str) -> int:
        busy = self.running_pids()
        if busy:
            return self.skip(f"skip: already running ({', '.join(busy)})")

        age = self.stamp_age()
        what = "no stamp yet" if age is None else f"stamp is {age}s old"

        if age is not None and age < MIN_AGE_SECS:
            return self.skip(f"skip: {what} (< {MIN_AGE_SECS}s floor) — ran recently")

        due = age is None or age >= MAX_AGE_SECS
        if mode == "dry-run":
            self.emit(f"dry-run: due={due} ({what}), mode={mode}")
            return 0
        if not due and mode != "force":
            return self.skip(f"skip: {what} (< {MAX_AGE_SECS}s) — not due")

        if not self.take_lock():
            return 0

        try:
            self.emit(f"starting top-up ({what}, mode={mode})")
            try:
                with self.log.open("a") as log_fh:
                    log_fh.write(f"\n===== {mode} run {_ts()} ({what}) =====\n")
                    log_fh.flush()
                    proc = subprocess.run(
                        [str(self.python), str(self.topup), "--harvest"],
                        cwd=str(self.root),
                        stdout=log_fh,
                        stderr=subprocess.STDOUT,
                        env={**os.environ,
                             "AUTO_INSTALL_PLAYWRIGHT_CHROMIUM": "false"},
                    )
                    rc = proc.returncode
            except OSError as exc:
                self.emit(f"could not start top-up: {exc}")
                return 1

            if rc == 0:
                try:
                    self.stamp.parent.mkdir(parents=True, exist_ok=True)
                    self.stamp.touch()
                except OSError as exc:
                    self.emit(f"top-up ok but stamp failed: {exc}")
                    return 1
                self.emit("top-up ok (rc=0); stamp refreshed")
            else:
                self.emit(f"top-up failed (rc={rc}); stamp untouched so the next tick retries")
            return rc
        finally:
            self.release_lock()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--force", action="store_true",
                   help="run unless a run happened in the last 6h (weekly mode)")
    g.add_argument("--dry-run", action="store_true",
                   help="print the decision and exit without running")
    args = ap.parse_args()

    # Resolve the symlink so the real .nosync path is used everywhere.
    root = Path(__file__).resolve().parent.parent
    mode = "force" if args.force else "dry-run" if args.dry_run else "auto"

    runner = Runner(root)
    runner.log.parent.mkdir(parents=True, exist_ok=True)
    runner.stamp.parent.mkdir(parents=True, exist_ok=True)
    return runner.run(mode)


if __name__ == "__main__":
    sys.exit(main())
