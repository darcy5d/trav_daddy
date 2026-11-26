#!/usr/bin/env python3
"""Write paper-only per-competition model-quality gates from held bets."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_connection

OUT = Path(__file__).resolve().parent.parent / "data/paper_trading/competition_gate.json"


def _prefix(fixture_key: str) -> str:
    # Fixture keys end with date segments; the stable competition slug precedes them.
    return "-".join((fixture_key or "").split("-")[:-3]) or (fixture_key or "unknown")


def main() -> int:
    since = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
    conn = get_connection()
    rows = conn.execute(
        """SELECT fixture_key, model_prob, market_price_at_proposal, settle_outcome
           FROM bet_ledger
           WHERE bet_kind='paper' AND status='settled'
             AND cashout_triggered_at IS NULL AND proposed_at >= ?
             AND model_prob IS NOT NULL AND market_price_at_proposal IS NOT NULL
             AND settle_outcome IN (0, 1)""",
        (since,),
    ).fetchall()
    conn.close()
    buckets = {}
    for row in rows:
        buckets.setdefault(_prefix(row["fixture_key"]), []).append(row)
    gates = {}
    for prefix, values in buckets.items():
        n = len(values)
        model_brier = sum((float(r["model_prob"]) - int(r["settle_outcome"])) ** 2 for r in values) / n
        market_brier = sum((float(r["market_price_at_proposal"]) - int(r["settle_outcome"])) ** 2 for r in values) / n
        gates[prefix] = {
            "go": n >= 30 and model_brier <= market_brier + 0.01,
            "n": n, "model_brier": round(model_brier, 6), "market_brier": round(market_brier, 6),
        }
    gates.setdefault("ipl", {"go": True, "seeded": True})
    gates.setdefault("t20blast", {"go": False, "seeded": True})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(), "gates": gates}, indent=2))
    print(f"Wrote {OUT} ({len(gates)} competition gates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
