#!/usr/bin/env python3
"""Read-only health check: is this indias_dad instance worth driving?

Usage:
    venv311/bin/python .cursor/skills/verify-indias-dad/scripts/doctor.py [PORT ...]

Defaults to the live port 5001 plus any scratch instance recorded in
tmp/verify-indias-dad/. Never starts, stops, or writes to anything except
stdout. Exit 0 if every checked port is listening and /healthz says db_ok.
"""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRATCH_DIR = REPO_ROOT / "tmp" / "verify-indias-dad"
LIVE_PORT = 5001


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def listener_pid(port: int) -> int | None:
    out = sh(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"])
    return int(out.splitlines()[0]) if out else None


def healthz(port: int) -> tuple[int | None, str]:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=5) as r:
            return r.status, r.read().decode()[:200]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:200]
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def newest_code_change() -> tuple[float, str]:
    newest, name = 0.0, ""
    for root in ("app", "src"):
        for p in (REPO_ROOT / root).rglob("*.py"):
            m = p.stat().st_mtime
            if m > newest:
                newest, name = m, str(p.relative_to(REPO_ROOT))
    cfg = REPO_ROOT / "config.py"
    if cfg.stat().st_mtime > newest:
        newest, name = cfg.stat().st_mtime, "config.py"
    return newest, name


def check(port: int) -> bool:
    print(f"--- port {port} ({'LIVE' if port == LIVE_PORT else 'scratch'})")
    pid = listener_pid(port)
    if not pid:
        print("  listener: NONE")
        return False
    cmd = sh(["ps", "-o", "command=", "-p", str(pid)])
    lstart = sh(["ps", "-o", "lstart=", "-p", str(pid)])
    print(f"  listener: pid {pid} started {lstart}")
    print(f"  command:  {cmd}")
    if str(REPO_ROOT) not in cmd and "flask" not in cmd:
        print("  WARN: listener does not look like this repo's Flask")
    status, body = healthz(port)
    print(f"  /healthz: {status} {body}")
    ok = status == 200 and '"db_ok":true' in body.replace(" ", "")
    try:
        started = datetime.strptime(" ".join(lstart.split()), "%a %b %d %H:%M:%S %Y").timestamp()
        newest, name = newest_code_change()
        if newest > started:
            print(f"  STALE: {name} changed after this process started; its Python code is NOT what is on disk "
                  "(templates auto-reload, Python does not). Verify Python changes on a scratch instance.")
        else:
            print("  code: process is newer than every app/src/config .py change")
    except ValueError:
        print("  code: could not parse process start time")
    return ok


def main() -> int:
    ports = [int(a) for a in sys.argv[1:]] or [LIVE_PORT]
    if not sys.argv[1:] and SCRATCH_DIR.exists():
        ports += [int(p.stem.split("-")[1]) for p in SCRATCH_DIR.glob("scratch-*.pid")]
    print(f"repo: {REPO_ROOT}  branch: {sh(['git', '-C', str(REPO_ROOT), 'rev-parse', '--abbrev-ref', 'HEAD'])}"
          f"  head: {sh(['git', '-C', str(REPO_ROOT), 'rev-parse', '--short', 'HEAD'])}")
    try:
        import playwright  # noqa: F401
        print("playwright: importable")
    except ImportError:
        print("playwright: MISSING in this interpreter (use venv311/bin/python)")
    results = [check(p) for p in ports]
    print("DOCTOR " + ("OK" if all(results) else "PROBLEMS"))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
