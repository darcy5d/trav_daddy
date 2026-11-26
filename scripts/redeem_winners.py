#!/usr/bin/env python3
"""Redeem confirmed Polymarket winners through the gasless proxy relayer.

Default is dry-run. Live submission requires both --apply and
BETTING_AUTO_REDEEM_ENABLED=1. This script never submits a trade order.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_db_connection, init_redemption_attempts
from src.integrations.polymarket.redemption import redeem_winners

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Submit gasless redemptions")
    args = parser.parse_args()

    init_redemption_attempts()
    with get_db_connection() as conn:
        summary = redeem_winners(conn, dry_run=not args.apply)

    print(json.dumps(summary, indent=2, sort_keys=True))
    failures = [r for r in summary["results"] if r.get("status") == "failed"]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
