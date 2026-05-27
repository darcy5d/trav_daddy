#!/usr/bin/env python3
"""Wave 5 Phase 7: Daily betting rollup.

Once per day (typically run via cron or manual trigger), compute the
day's P&L, bet counts, win rate, calibration drift on settled bets, and
append a summary JSON to `data/betting/rollup/YYYY-MM-DD.json`.

Useful for weekly review and as the data source for the per-day
Live Betting dashboard chart.

Usage:
    python scripts/daily_betting_rollup.py [--date YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_connection, get_db_connection  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _day_window_iso(target_date_iso: str) -> tuple:
    base = datetime.strptime(target_date_iso, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = base + timedelta(days=1)
    return base.isoformat(), end.isoformat()


def _safe_log(p: float, eps: float = 1e-6) -> float:
    return math.log(min(max(p, eps), 1.0 - eps))


def compute_rollup(target_date_iso: str) -> Dict[str, Any]:
    start, end = _day_window_iso(target_date_iso)
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT *
            FROM bet_ledger
            WHERE proposed_at >= ? AND proposed_at < ?
            """,
            (start, end),
        )
        bets = [dict(r) for r in cur.fetchall()]

    n_proposed = len(bets)
    placed = [b for b in bets if b.get("placed_at")]
    filled = [b for b in bets if b.get("filled_at")]
    settled_today = [
        b for b in bets
        if b.get("settled_at") and start <= b["settled_at"] < end
    ]

    total_pnl = sum((b.get("pnl_realised_usdc") or 0.0) for b in settled_today)
    n_wins = sum(1 for b in settled_today if (b.get("pnl_realised_usdc") or 0.0) > 0)
    n_losses = sum(1 for b in settled_today if (b.get("pnl_realised_usdc") or 0.0) < 0)
    win_rate = (n_wins / len(settled_today)) if settled_today else None

    # Brier on settled bets: model said P(outcome=1) was model_prob,
    # actual outcome was settle_outcome (0 or 1).
    brier_terms = []
    log_loss_terms = []
    for b in settled_today:
        if b.get("settle_outcome") is None or b.get("model_prob") is None:
            continue
        actual = float(b["settle_outcome"])
        p = float(b["model_prob"])
        brier_terms.append((p - actual) ** 2)
        log_loss_terms.append(-(actual * _safe_log(p) + (1 - actual) * _safe_log(1 - p)))
    live_brier = (sum(brier_terms) / len(brier_terms)) if brier_terms else None
    live_log_loss = (sum(log_loss_terms) / len(log_loss_terms)) if log_loss_terms else None

    rollup = {
        "date": target_date_iso,
        "n_bets_proposed": n_proposed,
        "n_bets_placed": len(placed),
        "n_bets_filled": len(filled),
        "n_bets_settled_today": len(settled_today),
        "win_rate": round(win_rate, 4) if win_rate is not None else None,
        "n_wins": n_wins,
        "n_losses": n_losses,
        "total_pnl_usdc": round(total_pnl, 4),
        "live_brier_on_settled": round(live_brier, 4) if live_brier is not None else None,
        "live_log_loss_on_settled": round(live_log_loss, 4) if live_log_loss is not None else None,
    }
    return rollup


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily betting rollup (Wave 5 Phase 7)")
    parser.add_argument("--date", default=None, help="Target date (UTC) YYYY-MM-DD; defaults to today")
    parser.add_argument("--output-dir", default="data/betting/rollup")
    args = parser.parse_args()

    target_iso = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rollup = compute_rollup(target_iso)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{target_iso}.json"
    out_path.write_text(json.dumps(rollup, indent=2))
    logger.info(f"Wrote rollup for {target_iso} to {out_path}")
    print(json.dumps(rollup, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
