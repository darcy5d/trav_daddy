#!/usr/bin/env python3
"""Wave 6 follow-up: record wallet deposits/withdrawals and show true ROI.

Run this whenever you fund (or withdraw from) the live Polymarket wallet so the
system can compute cash-on-cash ROI on the capital you actually put in, instead
of mistaking a deposit for profit.

Examples
--------
  # You just deposited $500 USDC into the wallet:
  python scripts/record_capital_flow.py --deposit 500 --note "initial funding"

  # You withdrew $200:
  python scripts/record_capital_flow.py --withdraw 200

  # Record a deposit with its on-chain tx hash:
  python scripts/record_capital_flow.py --deposit 1000 --tx 0xabc...

  # Just show the current capital / ROI summary (no new flow):
  python scripts/record_capital_flow.py --summary

Tip: record the deposit AFTER the USDC has landed in the wallet so the wallet
snapshot reconciles.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_db_connection, init_capital_flows
from src.integrations.polymarket.capital import get_capital_summary, record_flow


def _print_summary(summary: dict) -> None:
    print("\n" + "=" * 64)
    print("LIVE CAPITAL / ROI SUMMARY")
    print("=" * 64)
    print(f"  Total deposited:        ${summary['total_deposits_usdc']:,.2f}"
          f"  ({summary['n_deposits']} deposits)")
    print(f"  Total withdrawn:        ${summary['total_withdrawals_usdc']:,.2f}"
          f"  ({summary['n_withdrawals']} withdrawals)")
    print(f"  Net contributed:        ${summary['net_contributed_usdc']:,.2f}")
    print(f"  Current portfolio:      ${summary['current_portfolio_value_usdc']:,.2f}", end="")
    if summary.get("wallet_cash_usdc") is not None:
        print(f"   (wallet cash ${summary['wallet_cash_usdc']:,.2f})")
    else:
        print()
    print("  " + "-" * 60)
    if summary["funded"]:
        npnl = summary["net_pnl_usdc"]
        roi = summary["roi_on_capital_pct"]
        print(f"  Net P&L (cash-on-cash): ${npnl:,.2f}")
        print(f"  ROI on capital:         {roi:+.2f}%   <-- true return on funded capital")
        if summary.get("roi_on_total_deposits_pct") is not None:
            print(f"  ROI on gross deposits:  {summary['roi_on_total_deposits_pct']:+.2f}%")
    else:
        print("  No deposits recorded yet - ROI on capital unavailable.")
    if summary.get("capital_deployed_pct") is not None:
        print(f"  Capital deployed:       {summary['capital_deployed_pct']:.1f}% "
              f"(open positions / portfolio)")
    print(f"  Realized P&L (booked):  ${summary['realized_pnl_usdc']:,.2f}  (settled-bets cross-check)")
    if not summary.get("wallet_driven"):
        print("\n  NOTE: wallet not configured/readable; portfolio is a config fallback.")
    print("=" * 64 + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--deposit", type=float, metavar="USDC", help="record a deposit")
    g.add_argument("--withdraw", type=float, metavar="USDC", help="record a withdrawal")
    g.add_argument("--summary", action="store_true", help="show summary only")
    ap.add_argument("--tx", default=None, help="on-chain tx hash (optional)")
    ap.add_argument("--note", default=None, help="free-text note")
    ap.add_argument("--source", default="manual", help="manual / onchain (default manual)")
    ap.add_argument("--no-wallet-snapshot", action="store_true",
                    help="skip the wallet/portfolio snapshot (no network call)")
    args = ap.parse_args()

    init_capital_flows()

    with get_db_connection() as conn:
        if args.deposit is not None or args.withdraw is not None:
            flow_type = "deposit" if args.deposit is not None else "withdrawal"
            amount = args.deposit if args.deposit is not None else args.withdraw
            try:
                flow_id = record_flow(
                    conn,
                    flow_type=flow_type,
                    amount_usdc=amount,
                    tx_hash=args.tx,
                    source=args.source,
                    note=args.note,
                    capture_wallet=not args.no_wallet_snapshot,
                )
            except ValueError as exc:
                print(f"Error: {exc}")
                return 2
            print(f"\nRecorded {flow_type} of ${amount:,.2f} (flow_id={flow_id}).")

        summary = get_capital_summary(conn)
        _print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
