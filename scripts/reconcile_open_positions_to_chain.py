#!/usr/bin/env python3
"""Books-only true-up: shrink OPEN ledger positions to match on-chain reality.

Background: a partial cashout is supposed to decrement the originating BUY
row's open stake (``fill_size_usdc``) by the entry cost actually sold. When the
cashout scanner ran stale in-memory code (long-running daemon started before the
decrement fix), it booked the settled SELL adjustment rows but left the BUY rows
at their gross fill. Result: the ledger believes we still hold shares we already
sold, overstating open exposure and inflating portfolio mark-to-market (the sale
proceeds are already counted in wallet cash).

This script reconciles OPEN real positions against on-chain held shares
(Polymarket data-api) and corrects the LEDGER ONLY -- it never trades or
redeems. For each token where the ledger's open shares exceed what we actually
hold on-chain (beyond a tolerance), it scales every open BUY row's
``fill_size_usdc`` down pro-rata so ledger shares == on-chain shares. Realized
PnL on the historical SELL rows is left untouched.

It only ever reduces (never inflates) open exposure. Tokens where on-chain >=
ledger (within tolerance) are left alone. A token with zero on-chain holding but
open rows is flagged and, on --apply, those rows are zeroed and settled.

Usage:
    venv311/bin/python scripts/reconcile_open_positions_to_chain.py            # dry-run
    venv311/bin/python scripts/reconcile_open_positions_to_chain.py --apply    # write
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqlite3

from config import DATABASE_PATH
from src.integrations.polymarket import PolymarketClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Only correct material overstatements: ledger shares must exceed on-chain by
# both a relative and an absolute margin to avoid clobbering positions on
# transient data-api lag or just-placed (not-yet-indexed) fills.
REL_TOLERANCE = 0.02   # 2%
ABS_TOLERANCE_SHARES = 2.0
DUST_SHARES = 1.0      # on-chain holding below this => treat as fully exited


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _f(v: Any) -> float:
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _open_rows_by_token(conn: sqlite3.Connection) -> Dict[str, List[sqlite3.Row]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT bet_id, polymarket_token_id, side_label, strategy_label,
               fill_size_usdc, fill_price
        FROM bet_ledger
        WHERE COALESCE(bet_kind, 'real') = 'real'
          AND side = 'BUY'
          AND status = 'filled'
          AND settled_at IS NULL
          AND fill_price IS NOT NULL AND fill_price > 0
          AND fill_size_usdc IS NOT NULL AND fill_size_usdc > 0
          AND polymarket_token_id IS NOT NULL AND polymarket_token_id != ''
        ORDER BY bet_id
        """
    )
    out: Dict[str, List[sqlite3.Row]] = defaultdict(list)
    for r in cur.fetchall():
        out[str(r["polymarket_token_id"])].append(r)
    return out


def _onchain_shares_by_token(client: PolymarketClient) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        rows = client.get_data_api_positions(size_threshold=0.0, limit=500)
    except Exception as exc:
        logger.error(f"Failed to fetch on-chain positions: {exc}")
        return out
    for p in rows:
        tok = str(p.get("asset") or p.get("token_id") or p.get("tokenId") or "")
        if tok:
            out[tok] = _f(p.get("size"))
    return out


def _plan_token(token: str, rows: List[sqlite3.Row], onchain_shares: float) -> Dict[str, Any]:
    ledger_shares = sum(_f(r["fill_size_usdc"]) / _f(r["fill_price"]) for r in rows)
    ledger_cost = sum(_f(r["fill_size_usdc"]) for r in rows)
    side_label = rows[0]["side_label"]
    strategies = sorted({str(r["strategy_label"]) for r in rows})

    base = {
        "token": token,
        "side_label": side_label,
        "strategies": strategies,
        "n_rows": len(rows),
        "ledger_shares": ledger_shares,
        "onchain_shares": onchain_shares,
        "ledger_cost": ledger_cost,
        "rows": rows,
    }

    excess = ledger_shares - onchain_shares
    if excess <= max(ABS_TOLERANCE_SHARES, REL_TOLERANCE * ledger_shares):
        return {**base, "action": "leave", "scale": 1.0,
                "new_cost": ledger_cost, "removed_cost": 0.0}

    if onchain_shares < DUST_SHARES:
        return {**base, "action": "zero", "scale": 0.0,
                "new_cost": 0.0, "removed_cost": ledger_cost}

    scale = onchain_shares / ledger_shares if ledger_shares > 0 else 0.0
    new_cost = ledger_cost * scale
    return {**base, "action": "scale", "scale": scale,
            "new_cost": new_cost, "removed_cost": ledger_cost - new_cost}


def _apply(conn: sqlite3.Connection, plan: Dict[str, Any]) -> None:
    now = _now()
    if plan["action"] == "scale":
        for r in plan["rows"]:
            new_fill = round(_f(r["fill_size_usdc"]) * plan["scale"], 6)
            if new_fill <= 1e-6:
                conn.execute(
                    "UPDATE bet_ledger SET fill_size_usdc = 0, status = 'settled', "
                    "settled_at = ?, pnl_realised_usdc = COALESCE(pnl_realised_usdc, 0.0), "
                    "error_message = 'open-position-truedown: scaled to on-chain (=0)' "
                    "WHERE bet_id = ?",
                    (now, r["bet_id"]),
                )
            else:
                conn.execute(
                    "UPDATE bet_ledger SET fill_size_usdc = ?, "
                    "error_message = 'open-position-truedown: scaled to on-chain shares' "
                    "WHERE bet_id = ?",
                    (new_fill, r["bet_id"]),
                )
    elif plan["action"] == "zero":
        for r in plan["rows"]:
            conn.execute(
                "UPDATE bet_ledger SET fill_size_usdc = 0, status = 'settled', "
                "settled_at = ?, pnl_realised_usdc = COALESCE(pnl_realised_usdc, 0.0), "
                "error_message = 'open-position-truedown: no on-chain holding; closed' "
                "WHERE bet_id = ?",
                (now, r["bet_id"]),
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="Write the corrections (default: dry-run report only)")
    args = parser.parse_args()

    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.row_factory = sqlite3.Row

    open_by_token = _open_rows_by_token(conn)
    if not open_by_token:
        print("No open real BUY positions to reconcile.")
        conn.close()
        return 0

    onchain = _onchain_shares_by_token(PolymarketClient())

    plans = [
        _plan_token(tok, rows, onchain.get(tok, 0.0))
        for tok, rows in open_by_token.items()
    ]

    print()
    print("=" * 104)
    print(f"OPEN POSITION TRUE-UP {'[APPLY]' if args.apply else '[DRY-RUN]'} — {_now()}")
    print("=" * 104)
    print(f"  {'side':<16} {'action':<7} {'rows':>4} {'ledgerSh':>10} {'chainSh':>10} "
          f"{'ledger$':>10} {'new$':>10} {'removed$':>10}")
    print(f"  {'-'*16} {'-'*7} {'-'*4} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    total_removed = 0.0
    n_changes = 0
    for p in sorted(plans, key=lambda x: -x["removed_cost"]):
        if p["action"] in ("scale", "zero"):
            n_changes += 1
            total_removed += p["removed_cost"]
        print(f"  {str(p['side_label'])[:16]:<16} {p['action']:<7} {p['n_rows']:>4} "
              f"{p['ledger_shares']:>10.2f} {p['onchain_shares']:>10.2f} "
              f"{p['ledger_cost']:>10.2f} {p['new_cost']:>10.2f} {p['removed_cost']:>10.2f}")

    print()
    print(f"  Positions examined:        {len(plans)}")
    print(f"  Corrections proposed:      {n_changes}")
    print(f"  Total open cost removed:   ${total_removed:.2f}")
    print()

    if not args.apply:
        print("  DRY-RUN — no changes written. Re-run with --apply to commit.")
        conn.close()
        return 0

    for p in plans:
        _apply(conn, p)
    conn.commit()
    conn.close()
    print(f"  APPLIED {n_changes} corrections to bet_ledger.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
