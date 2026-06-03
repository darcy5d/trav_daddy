#!/usr/bin/env python3
"""Books-only true-up for historically mis-booked cashouts.

Before the cashout-fill fix, `execute_cashout` posted a SELL at the midpoint
(which barely crossed the book) but then UNCONDITIONALLY marked the bet
`settled` with a full cashout PnL. The result: rows that show a tidy cashout
profit while we actually still hold the position on-chain.

This script reconciles those rows against on-chain reality (Polymarket
data-api positions) and corrects the LEDGER ONLY — it never places SELL orders
and never redeems. For positions we still hold:

  * resolved WON  -> book the true settlement PnL (shares*$1 - cost),
                     settle_outcome=1, and flag the winner as redeemable.
  * resolved LOST -> book the true loss (-cost), settle_outcome=0.
  * still LIVE     -> reopen the bet (status='filled') and clear the phantom
                     cashout_* booking so the fixed scanner/settlement handles it.

Rows whose token no longer shows a held position are treated as genuinely
exited/redeemed and left untouched (idempotent — safe to re-run, and harmless
against cashouts booked by the new, fixed code path).

Approximation: for a row that partially sold a sliver before the bug stranded
the rest, the recompute treats the whole fill as held-to-resolution. The sliver
is small (single-digit %) and was sold near the eventual outcome, so the books
end up far closer to truth than the phantom full-cashout they replace.

Usage:
    # Dry-run (default): print the proposed corrections, write nothing.
    venv311/bin/python scripts/reconcile_cashout_fills.py

    # Apply the corrections to the ledger.
    venv311/bin/python scripts/reconcile_cashout_fills.py --apply
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATABASE_PATH
import sqlite3

from src.integrations.polymarket import PolymarketClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WON_PRICE = 0.95   # curPrice at/above this => resolved winner
LOST_PRICE = 0.05  # curPrice at/below this => resolved loser
DUST_SHARES = 1.0  # token position below this => treat as exited/redeemed


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _f(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _positions_by_token(client: PolymarketClient) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    try:
        rows = client.get_data_api_positions(size_threshold=0.0, limit=500)
    except Exception as exc:
        logger.error(f"Failed to fetch on-chain positions: {exc}")
        return out
    for p in rows:
        tok = str(p.get("asset") or p.get("token_id") or p.get("tokenId") or "")
        if tok:
            out[tok] = p
    return out


def _classify(bet: sqlite3.Row, pos: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Decide the correction for one mis-booked cashout BUY row."""
    fill_price = _f(bet["fill_price"]) or 0.0
    # The original entry cost: cashout zeroed fill_size on full exit, so prefer
    # size_usdc when fill_size is 0 (it carries the fill notional pre-cashout).
    fill_size = _f(bet["fill_size_usdc"]) or 0.0
    entry_cost = fill_size if fill_size > 0 else (_f(bet["size_usdc"]) or 0.0)
    shares = (entry_cost / fill_price) if fill_price > 0 else 0.0

    booked_pnl = _f(bet["pnl_realised_usdc"]) or 0.0
    held = pos is not None and (_f(pos.get("size")) or 0.0) > DUST_SHARES
    cur_price = _f(pos.get("curPrice")) if pos else None
    redeemable = bool(pos.get("redeemable")) if pos else False

    base = {
        "bet_id": bet["bet_id"],
        "side_label": bet["side_label"],
        "fixture_key": bet["fixture_key"],
        "entry_cost": entry_cost,
        "shares": shares,
        "booked_pnl": booked_pnl,
        "held_shares": (_f(pos.get("size")) or 0.0) if pos else 0.0,
        "cur_price": cur_price,
        "redeemable": redeemable,
    }

    if not held:
        return {**base, "action": "leave", "classification": "exited-or-redeemed",
                "true_pnl": booked_pnl}

    if cur_price is None:
        return {**base, "action": "review", "classification": "held-unknown-price",
                "true_pnl": booked_pnl}

    if cur_price >= WON_PRICE:
        true_pnl = round(shares * 1.0 - entry_cost, 4)
        return {**base, "action": "settle_win", "classification": "held-winner",
                "true_pnl": true_pnl}
    if cur_price <= LOST_PRICE:
        true_pnl = round(-entry_cost, 4)
        return {**base, "action": "settle_loss", "classification": "held-loser",
                "true_pnl": true_pnl}
    return {**base, "action": "reopen", "classification": "held-live",
            "true_pnl": None}


def _apply(conn: sqlite3.Connection, plan: Dict[str, Any]) -> None:
    bet_id = plan["bet_id"]
    action = plan["action"]
    now = _now()
    if action in ("settle_win", "settle_loss"):
        outcome = 1 if action == "settle_win" else 0
        conn.execute(
            """
            UPDATE bet_ledger
            SET status            = 'settled',
                settled_at        = COALESCE(settled_at, ?),
                settle_outcome    = ?,
                pnl_realised_usdc = ?,
                fill_size_usdc    = ?,
                cashout_reason    = COALESCE(cashout_reason || '+recon', cashout_reason),
                error_message     = 'cashout-fill-recon: held to resolution; cashout SELL did not fill'
            WHERE bet_id = ?
            """,
            (now, outcome, plan["true_pnl"], plan["entry_cost"], bet_id),
        )
    elif action == "reopen":
        conn.execute(
            """
            UPDATE bet_ledger
            SET status                 = 'filled',
                settled_at             = NULL,
                settle_outcome         = NULL,
                pnl_realised_usdc      = NULL,
                fill_size_usdc         = ?,
                cashout_triggered_at   = NULL,
                cashout_price          = NULL,
                cashout_pnl_usdc       = NULL,
                cashout_threshold_used = NULL,
                cashout_order_id       = NULL,
                cashout_reason         = NULL,
                error_message          = 'cashout-fill-recon: reopened; cashout SELL did not fill, position still live'
            WHERE bet_id = ?
            """,
            (plan["entry_cost"], bet_id),
        )
    # 'leave' / 'review' -> no write.


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="Write the corrections (default: dry-run report only)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Cap the number of rows examined (0 = all)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT *
        FROM bet_ledger
        WHERE COALESCE(bet_kind, 'real') = 'real'
          AND side = 'BUY'
          AND cashout_triggered_at IS NOT NULL
          AND status = 'settled'
        ORDER BY cashout_triggered_at DESC
        """
    )
    rows = cur.fetchall()
    if args.limit > 0:
        rows = rows[: args.limit]

    if not rows:
        print("No real cashed-out BUY rows to reconcile.")
        conn.close()
        return 0

    positions = _positions_by_token(PolymarketClient())

    plans: List[Dict[str, Any]] = []
    for bet in rows:
        pos = positions.get(str(bet["polymarket_token_id"] or ""))
        plans.append(_classify(bet, pos))

    # --- Report ---
    counts: Dict[str, int] = {}
    booked_total = 0.0
    corrected_total = 0.0
    print()
    print("=" * 100)
    print(f"CASHOUT FILL RECONCILE {'[APPLY]' if args.apply else '[DRY-RUN]'} — {_now()}")
    print("=" * 100)
    print(f"  {'bet':>6}  {'side':<16} {'class':<18} {'action':<12} "
          f"{'held':>9} {'px':>6} {'booked$':>10} {'true$':>10} {'Δ$':>9}")
    print(f"  {'-'*6}  {'-'*16} {'-'*18} {'-'*12} {'-'*9} {'-'*6} {'-'*10} {'-'*10} {'-'*9}")
    for p in plans:
        counts[p["classification"]] = counts.get(p["classification"], 0) + 1
        booked = p["booked_pnl"] or 0.0
        true_pnl = p["true_pnl"]
        booked_total += booked
        delta_str = ""
        if true_pnl is not None and p["action"] in ("settle_win", "settle_loss"):
            corrected_total += true_pnl
            delta_str = f"{true_pnl - booked:+.2f}"
            true_str = f"{true_pnl:+.2f}"
        else:
            corrected_total += booked
            true_str = "reopen" if p["action"] == "reopen" else f"{booked:+.2f}"
        px = f"{p['cur_price']:.3f}" if p["cur_price"] is not None else "-"
        flag = " *REDEEM" if (p["action"] == "settle_win" and p["redeemable"]) else ""
        print(f"  {p['bet_id']:>6}  {str(p['side_label'])[:16]:<16} "
              f"{p['classification']:<18} {p['action']:<12} "
              f"{p['held_shares']:>9.2f} {px:>6} {booked:>10.2f} {true_str:>10} {delta_str:>9}{flag}")

    print()
    print("  Summary by classification:")
    for k, v in sorted(counts.items()):
        print(f"    {k:<22} {v}")
    n_changes = sum(1 for p in plans if p["action"] in ("settle_win", "settle_loss", "reopen"))
    n_review = sum(1 for p in plans if p["action"] == "review")
    print()
    print(f"  Rows examined:        {len(plans)}")
    print(f"  Corrections proposed: {n_changes}")
    print(f"  Need manual review:   {n_review}")
    print(f"  Booked PnL (these):   ${booked_total:+.2f}")
    print(f"  Corrected PnL:        ${corrected_total:+.2f}  (Δ ${corrected_total - booked_total:+.2f})")
    print()

    if not args.apply:
        print("  DRY-RUN — no changes written. Re-run with --apply to commit.")
        conn.close()
        return 0

    applied = 0
    for p in plans:
        if p["action"] in ("settle_win", "settle_loss", "reopen"):
            _apply(conn, p)
            applied += 1
    conn.commit()
    conn.close()
    print(f"  APPLIED {applied} corrections to bet_ledger.")
    if any(p["action"] == "settle_win" and p["redeemable"] for p in plans):
        print("  NOTE: winners flagged *REDEEM are still held on-chain — claim them "
              "manually on Polymarket to realise the USDC.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
