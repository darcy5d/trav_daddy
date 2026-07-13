#!/usr/bin/env python3
"""Fit a bounded V3/market blend weight per competition from held paper bets."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.database import get_connection

OUT = Path(__file__).resolve().parent.parent / "data/models/paper_blend_weights.json"


def _prefix(key: str) -> str:
    return "-".join((key or "").split("-")[:-3]) or (key or "unknown")


def main() -> int:
    conn = get_connection()
    rows = conn.execute(
        """SELECT fixture_key, model_prob, market_price_at_proposal, settle_outcome
           FROM bet_ledger
           WHERE bet_kind='paper' AND status='settled' AND cashout_triggered_at IS NULL
             AND model_prob IS NOT NULL AND market_price_at_proposal IS NOT NULL
             AND settle_outcome IN (0, 1)"""
    ).fetchall()
    conn.close()
    grouped = {}
    for row in rows:
        grouped.setdefault(_prefix(row["fixture_key"]), []).append(row)
    result = {}
    for prefix, values in grouped.items():
        # Grid search avoids a scipy dependency and is robust for sparse cohorts.
        best_w, best_loss = 0.25, float("inf")
        for i in range(21):
            w = i / 20
            loss = sum(
                (w * float(r["model_prob"]) + (1 - w) * float(r["market_price_at_proposal"])
                 - int(r["settle_outcome"])) ** 2 for r in values
            ) / len(values)
            if loss < best_loss:
                best_w, best_loss = w, loss
        result[prefix] = {"weight": best_w, "n": len(values), "brier": round(best_loss, 6)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(), "weights": result}, indent=2))
    print(f"Wrote {OUT} ({len(result)} competition weights)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
