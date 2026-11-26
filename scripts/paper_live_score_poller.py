#!/usr/bin/env python3
"""Paper-only CREX score poller for in-play fair-value consumers."""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.api.crex_scraper import CREXScraper
from src.integrations.polymarket import PolymarketClient
from src.integrations.polymarket.crex_xi import find_crex_match_for_fixture
from src.integrations.polymarket.upcoming import find_upcoming_cricket_events

ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = ROOT / "data/paper_trading/live_match_states.json"
PID_FILE = ROOT / "logs/paper_live_score_poller.pid"
STATUS_FILE = ROOT / "logs/paper_live_score_poller_status.json"
RUNNING = True


def _stop(_signum, _frame):
    global RUNNING
    RUNNING = False


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    if PID_FILE.exists() and not args.once:
        try:
            os.kill(int(PID_FILE.read_text()), 0)
            raise RuntimeError("poller already running")
        except ProcessLookupError:
            PID_FILE.unlink(missing_ok=True)
    if not args.once:
        PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        PID_FILE.write_text(str(os.getpid()))
        signal.signal(signal.SIGTERM, _stop)
        signal.signal(signal.SIGINT, _stop)
    client, scraper, iteration = PolymarketClient(), CREXScraper(), 0
    try:
        while RUNNING:
            iteration += 1
            states = {}
            try:
                for fixture in find_upcoming_cricket_events(client, hours_ahead=6, include_started=True):
                    crex = find_crex_match_for_fixture(fixture)
                    if not crex:
                        continue
                    live = scraper.get_live_match(crex.match_url)
                    # Keep the raw CREX payload as well as normalized fields:
                    # score layouts vary by competition and parsers evolve.
                    if live:
                        states[fixture["fixture_key"]] = {
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                            "fixture_key": fixture["fixture_key"],
                            "crex_url": crex.match_url,
                            "raw": live,
                        }
                _write_json(STATE_FILE, states)
                _write_json(STATUS_FILE, {
                    "written_at_utc": datetime.now(timezone.utc).isoformat(),
                    "db_ok": True, "iter_count": iteration, "fixtures": len(states),
                })
            except Exception as exc:
                _write_json(STATUS_FILE, {
                    "written_at_utc": datetime.now(timezone.utc).isoformat(),
                    "db_ok": False, "iter_count": iteration, "error": str(exc),
                })
            if args.once:
                break
            time.sleep(max(30, args.interval))
    finally:
        if not args.once:
            PID_FILE.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
