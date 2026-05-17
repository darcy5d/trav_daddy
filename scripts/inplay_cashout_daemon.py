#!/usr/bin/env python3
"""Wave 5.10: In-game cashout daemon.

Start once before a match session and leave it running. It polls all
open (filled) positions every INTERVAL seconds and automatically sells
if the return threshold is met.

Usage:
    # Start the daemon (runs until Ctrl+C)
    venv311/bin/python scripts/inplay_cashout_daemon.py

    # Dry-run — logs what would trigger without placing orders
    venv311/bin/python scripts/inplay_cashout_daemon.py --dry-run

    # Custom interval and threshold override
    venv311/bin/python scripts/inplay_cashout_daemon.py --interval 120 --threshold 1.3

    # Log to file as well as stdout
    venv311/bin/python scripts/inplay_cashout_daemon.py 2>&1 | tee logs/cashout_daemon.log

Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_connection, init_cashout_columns
from src.integrations.polymarket import PolymarketClient
from src.integrations.polymarket.cashout import scan_for_cashouts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_INTERVAL = 180  # seconds (3 minutes)

_running = True


def _handle_sigterm(signum, frame):
    global _running
    logger.info("SIGTERM received — shutting down after current scan")
    _running = False


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Continuously monitor open positions for cashout opportunities"
    )
    parser.add_argument(
        "--interval", type=int, default=DEFAULT_INTERVAL,
        help=f"Poll interval in seconds (default: {DEFAULT_INTERVAL})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simulate cashouts without placing real SELL orders",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help=(
            "Override return-ratio threshold for ALL bets (e.g. 1.3). "
            "If omitted, uses each strategy's cashout_return_threshold."
        ),
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Debug-level logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    # Ensure cashout columns exist (idempotent, fast)
    init_cashout_columns()

    mode = "DRY-RUN" if args.dry_run else "LIVE"
    threshold_str = f"{args.threshold}x" if args.threshold else "per-strategy"
    print()
    print("=" * 70)
    print(f"  CASHOUT DAEMON — {mode}")
    print(f"  Poll interval : {args.interval}s ({args.interval / 60:.1f} min)")
    print(f"  Threshold     : {threshold_str}")
    print(f"  Started       : {_ts()}")
    print("  Press Ctrl+C to stop.")
    print("=" * 70)

    scan_count = 0
    total_executed = 0
    total_pnl_locked = 0.0

    while _running:
        scan_count += 1
        scan_start = time.monotonic()

        try:
            conn = get_connection()
            poly_client = PolymarketClient()

            summary = scan_for_cashouts(
                conn=conn,
                poly_client=poly_client,
                dry_run=args.dry_run,
                default_threshold=args.threshold,
            )
            conn.close()

            elapsed = time.monotonic() - scan_start
            ts = _ts()

            if summary["n_triggered"] > 0:
                for c in summary["cashouts"]:
                    if c.get("success"):
                        total_executed += 1
                        total_pnl_locked += c.get("cashout_pnl", 0.0)

                print(f"\n[{ts}] Scan #{scan_count} ({elapsed:.1f}s) — "
                      f"{summary['n_checked']} open, "
                      f"{summary['n_triggered']} triggered")
                for c in summary["cashouts"]:
                    if c.get("success"):
                        sim = " [simulated]" if c.get("is_simulated") else ""
                        print(f"  CASHOUT{sim} bet_id={c['bet_id']}  "
                              f"{c['return_ratio']:.2f}x → ${c['cashout_pnl']:+.2f}")
            else:
                print(f"[{ts}] #{scan_count} — "
                      f"{summary['n_checked']} open positions, "
                      f"none at threshold ({elapsed:.1f}s)")

            if summary["errors"]:
                for bet_id, msg in summary["errors"]:
                    logger.warning(f"  error bet_id={bet_id}: {msg}")

        except KeyboardInterrupt:
            break
        except Exception as exc:
            logger.error(f"Scan #{scan_count} failed: {exc}", exc_info=args.verbose)

        if not _running:
            break

        # Sleep in short increments so Ctrl+C is responsive
        deadline = time.monotonic() + args.interval
        try:
            while time.monotonic() < deadline and _running:
                time.sleep(1)
        except KeyboardInterrupt:
            break

    print()
    print("=" * 70)
    print(f"  DAEMON STOPPED — {_ts()}")
    print(f"  Scans run     : {scan_count}")
    print(f"  Cashouts fired: {total_executed}")
    print(f"  PnL locked in : ${total_pnl_locked:+.2f}")
    print("=" * 70)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
