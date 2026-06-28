#!/usr/bin/env python3
"""Wave 6 follow-up: reconcile the capital_flows ledger from on-chain truth.

Reads the real USDC deposits / withdrawals that touched the live Polymarket
proxy wallet, filters out trade/redeem settlement movements (via the Polymarket
data-api activity feed + known protocol contracts), and shows the most-recent
capital flows so you can verify them against your own records.

Once verified, re-run with --commit to write the last N deposits into the
capital_flows ledger (idempotent: a tx hash already recorded is skipped). After
that, true cash-on-cash ROI (scripts/record_capital_flow.py --summary) is
grounded in what you actually funded.

Examples
--------
  # Dry run: show the last 2 deposits (and recent withdrawals) for verification.
  python scripts/reconcile_capital_from_chain.py

  # Show the last 5 capital flows.
  python scripts/reconcile_capital_from_chain.py -n 5

  # After verifying, commit the last 2 deposits into the ledger.
  python scripts/reconcile_capital_from_chain.py --commit --record-last 2

  # Use a Polygonscan/Etherscan API key for an instant, complete history.
  python scripts/reconcile_capital_from_chain.py --etherscan-key YOURKEY

  # Also reconcile withdrawals into the ledger.
  python scripts/reconcile_capital_from_chain.py --commit --record-last 2 --with-withdrawals
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stderr
)

from config import POLYMARKET_CONFIG
from src.data.database import get_db_connection, init_capital_flows
from src.integrations.polymarket.capital import (
    flow_exists_by_tx,
    get_capital_summary,
    record_flow,
)
from src.integrations.polymarket.chain_reconcile import (
    PUSD_ADDRESS,
    USDC_E_ADDRESS,
    USDC_NATIVE_ADDRESS,
    CapitalFlow,
    reconcile_capital_flows,
)

_TOKEN_LABELS = {
    PUSD_ADDRESS.lower(): "pUSD",
    USDC_E_ADDRESS.lower(): "USDC.e",
    USDC_NATIVE_ADDRESS.lower(): "USDC",
}


def _fmt_flow(f: CapitalFlow) -> str:
    short_cp = f.counterparty[:10] + "…" + f.counterparty[-6:]
    short_tx = f.tx_hash[:10] + "…" + f.tx_hash[-6:]
    token = _TOKEN_LABELS.get(f.token.lower(), f.token[:8])
    return (
        f"  {f.ts_iso[:19]}Z  ${f.amount_usdc:>10,.2f}  {token:<7}  "
        f"blk {f.block:>10}  src {short_cp}  tx {short_tx}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--wallet", default=None,
                    help="proxy wallet address (default: POLYGON_FUNDER_ADDR)")
    ap.add_argument("-n", "--show", type=int, default=2,
                    help="how many recent deposits to surface (default 2)")
    ap.add_argument("--lookback-days", type=int, default=120,
                    help="how far back the keyless RPC scan looks (default 120)")
    ap.add_argument("--include-native", action="store_true",
                    help="also scan native USDC (in case the wrong token was sent)")
    ap.add_argument("--rpc-url", default=None,
                    help="override the Polygon RPC URL for the keyless scan")
    ap.add_argument("--etherscan-key", default=None,
                    help="Polygonscan/Etherscan V2 API key (else env, else keyless RPC)")
    ap.add_argument("--commit", action="store_true",
                    help="write the reconciled deposits into capital_flows")
    ap.add_argument("--record-last", type=int, default=None,
                    help="how many of the most-recent deposits to commit (default = --show)")
    ap.add_argument("--with-withdrawals", action="store_true",
                    help="also display/commit withdrawals")
    args = ap.parse_args()

    wallet = args.wallet or POLYMARKET_CONFIG.get("funder_address")
    if not wallet:
        print("ERROR: no wallet address (set POLYGON_FUNDER_ADDR or pass --wallet).")
        return 2

    print("\n" + "=" * 74)
    print("CHAIN -> CAPITAL-FLOW RECONCILER")
    print("=" * 74)
    print(f"  Wallet: {wallet}")
    print("  Reading pUSD + USDC.e transfers, filtering out trade/redeem settlements…")

    result = reconcile_capital_flows(
        wallet,
        min_deposits=max(args.show, args.record_last or 0),
        lookback_days=args.lookback_days,
        include_native=args.include_native,
        etherscan_key=args.etherscan_key,
        rpc_url=args.rpc_url,
    )

    print(f"  Source: {result.source}  |  transfers scanned: "
          f"{result.n_transfers_scanned}  |  protocol txs excluded: "
          f"{result.n_protocol_txs}")

    deposits = result.deposits[: args.show]
    print("\n  DEPOSITS (most recent first):")
    if deposits:
        for f in deposits:
            print(_fmt_flow(f))
    else:
        print("    (none found — try --lookback-days or --etherscan-key)")

    if args.with_withdrawals:
        withdrawals = result.withdrawals[: args.show]
        print("\n  WITHDRAWALS (most recent first):")
        if withdrawals:
            for f in withdrawals:
                print(_fmt_flow(f))
        else:
            print("    (none found)")

    if not args.commit:
        print("\n  Dry run — nothing written. Verify the deposits above against your")
        print("  records, then re-run with:  --commit --record-last N")
        print("=" * 74 + "\n")
        return 0

    n_commit = args.record_last if args.record_last is not None else args.show
    to_record = list(result.deposits[:n_commit])
    if args.with_withdrawals:
        to_record += list(result.withdrawals[:n_commit])

    init_capital_flows()
    recorded, skipped = 0, 0
    with get_db_connection() as conn:
        for f in to_record:
            if flow_exists_by_tx(conn, f.tx_hash):
                skipped += 1
                print(f"  skip (already recorded): {f.flow_type} ${f.amount_usdc:,.2f} tx {f.tx_hash[:12]}…")
                continue
            flow_id = record_flow(
                conn,
                flow_type=f.flow_type,
                amount_usdc=f.amount_usdc,
                tx_hash=f.tx_hash,
                source="onchain",
                note=f"reconciled from chain; counterparty {f.counterparty}",
                capture_wallet=False,
                ts=f.ts_iso,
            )
            recorded += 1
            print(f"  recorded flow_id={flow_id}: {f.flow_type} ${f.amount_usdc:,.2f} "
                  f"@ {f.ts_iso[:19]} tx {f.tx_hash[:12]}…")

        print(f"\n  Done: {recorded} recorded, {skipped} skipped (idempotent).")
        summary = get_capital_summary(conn)

    print("\n  --- updated capital summary ---")
    print(f"  Total deposited:   ${summary['total_deposits_usdc']:,.2f} "
          f"({summary['n_deposits']} deposits)")
    print(f"  Net contributed:   ${summary['net_contributed_usdc']:,.2f}")
    print(f"  Current portfolio: ${summary['current_portfolio_value_usdc']:,.2f}")
    if summary["funded"]:
        print(f"  Net P&L:           ${summary['net_pnl_usdc']:,.2f}")
        print(f"  ROI on capital:    {summary['roi_on_capital_pct']:+.2f}%")
    print("=" * 74 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
