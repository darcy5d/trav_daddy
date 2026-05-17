#!/usr/bin/env python3
"""Wave 5.10: Backtest in-game cashout strategy on historical bets.

For each settled bet in bet_ledger, fetches 1-minute Polymarket price
history and simulates "what if we had sold at a target return ratio?".
Compares simulated cashout PnL vs. the actual settlement outcome across
a grid of thresholds.

Key question: at threshold X would we have saved last night's losses
(England, Sweden) without sacrificing too much on the winners (Jersey)?

Usage:
    python scripts/backtest_inplay_cashout.py
    python scripts/backtest_inplay_cashout.py --days 30
    python scripts/backtest_inplay_cashout.py --bet-kind paper --days 7
    python scripts/backtest_inplay_cashout.py --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_connection
from src.integrations.polymarket import PolymarketClient
from src.integrations.odds.polymarket_compare import POLYMARKET_TAKER_FEE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Cashout return-ratio thresholds to simulate
THRESHOLDS = [1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5]

# Polymarket CLOB rate-limit: leave a small gap between price-history calls
RATE_LIMIT_SLEEP = 0.4  # seconds


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BetRecord:
    bet_id: int
    fixture_key: str
    side_label: str
    bet_kind: str
    strategy_label: Optional[str]
    fill_price: float
    fill_size_usdc: float
    filled_at: Optional[str]
    kickoff_at: Optional[str]
    actual_pnl: float
    settle_outcome: Optional[int]
    token_id: str


@dataclass
class ThresholdResult:
    threshold: float
    triggered: bool
    trigger_ts: Optional[int]       # unix timestamp when first met
    trigger_price: Optional[float]
    cashout_pnl: Optional[float]
    delta_vs_actual: Optional[float]  # cashout_pnl - actual_pnl (positive = we did better)
    minutes_into_match: Optional[float]


@dataclass
class BetAnalysis:
    bet: BetRecord
    results: List[ThresholdResult]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_ts(iso: Optional[str]) -> Optional[int]:
    if not iso:
        return None
    try:
        d = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return int(d.timestamp())
    except (ValueError, AttributeError):
        return None


def _cashout_pnl(fill_price: float, fill_size_usdc: float, sell_price: float) -> float:
    shares = fill_size_usdc / fill_price
    gross = shares * sell_price
    fee = gross * POLYMARKET_TAKER_FEE
    return round(gross - fill_size_usdc - fee, 4)


def _fetch_settled_bets(
    conn,
    days: int,
    bet_kind: Optional[str],
    min_fill_usdc: float,
) -> List[BetRecord]:
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    params: list = [since]

    kind_filter = ""
    if bet_kind:
        kind_filter = " AND (bet_kind = ? OR (bet_kind IS NULL AND ? = 'real'))"
        params.extend([bet_kind, bet_kind])

    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT bet_id, fixture_key, side_label,
               COALESCE(bet_kind, 'real') AS bet_kind,
               strategy_label,
               fill_price, fill_size_usdc, filled_at, kickoff_at,
               pnl_realised_usdc, settle_outcome,
               polymarket_token_id
        FROM bet_ledger
        WHERE status = 'settled'
          AND proposed_at >= ?
          AND fill_price IS NOT NULL
          AND fill_size_usdc IS NOT NULL
          AND pnl_realised_usdc IS NOT NULL
          AND polymarket_token_id IS NOT NULL
          AND fill_size_usdc >= ?
          AND cashout_triggered_at IS NULL  -- exclude already cashed-out rows
          {kind_filter}
        ORDER BY proposed_at DESC
        """,
        [since, min_fill_usdc] + (params[1:] if bet_kind else []),
    )
    rows = cur.fetchall()
    records = []
    for r in rows:
        records.append(BetRecord(
            bet_id=r["bet_id"],
            fixture_key=r["fixture_key"],
            side_label=r["side_label"] or "?",
            bet_kind=r["bet_kind"],
            strategy_label=r["strategy_label"],
            fill_price=float(r["fill_price"]),
            fill_size_usdc=float(r["fill_size_usdc"]),
            filled_at=r["filled_at"],
            kickoff_at=r["kickoff_at"],
            actual_pnl=float(r["pnl_realised_usdc"]),
            settle_outcome=r["settle_outcome"],
            token_id=r["polymarket_token_id"],
        ))
    return records


def _simulate_thresholds(
    bet: BetRecord,
    history: List[Dict[str, Any]],
    thresholds: List[float],
) -> List[ThresholdResult]:
    """Walk 1-min price history and find first time each threshold is met.

    We only look at prices AFTER the bet was filled (filled_at timestamp).
    """
    if not history:
        return [
            ThresholdResult(threshold=t, triggered=False, trigger_ts=None,
                            trigger_price=None, cashout_pnl=None,
                            delta_vs_actual=None, minutes_into_match=None)
            for t in thresholds
        ]

    fill_ts = _parse_ts(bet.filled_at) or 0
    kickoff_ts = _parse_ts(bet.kickoff_at) or fill_ts

    # Filter to post-fill entries
    post_fill = [p for p in history if isinstance(p, dict) and p.get("t", 0) >= fill_ts]

    results = []
    for threshold in thresholds:
        triggered = False
        trigger_ts = None
        trigger_price = None
        cashout_pnl = None
        delta = None
        minutes_in = None

        for point in post_fill:
            price = float(point.get("p", 0))
            ts = int(point.get("t", 0))
            if price <= 0 or price >= 1.0:
                continue
            ratio = price / bet.fill_price
            if ratio >= threshold:
                triggered = True
                trigger_ts = ts
                trigger_price = price
                cashout_pnl = _cashout_pnl(bet.fill_price, bet.fill_size_usdc, price)
                delta = cashout_pnl - bet.actual_pnl
                if ts >= kickoff_ts:
                    minutes_in = (ts - kickoff_ts) / 60.0
                else:
                    minutes_in = (ts - kickoff_ts) / 60.0  # negative = pre-kickoff
                break

        results.append(ThresholdResult(
            threshold=threshold,
            triggered=triggered,
            trigger_ts=trigger_ts,
            trigger_price=trigger_price,
            cashout_pnl=cashout_pnl,
            delta_vs_actual=delta,
            minutes_into_match=minutes_in,
        ))

    return results


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _outcome_tag(bet: BetRecord) -> str:
    if bet.settle_outcome == 1:
        return "WIN "
    if bet.settle_outcome == 0:
        return "LOSS"
    return "?   "


def _format_row(label: str, value: str) -> str:
    return f"  {label:<28} {value}"


def _print_bet_section(analysis: BetAnalysis) -> None:
    bet = analysis.bet
    tag = _outcome_tag(bet)
    print(f"\n{'─' * 72}")
    print(f"  bet_id={bet.bet_id}  [{tag}]  {bet.side_label}")
    print(f"  fixture:  {bet.fixture_key}")
    print(f"  strategy: {bet.strategy_label or 'manual'} ({bet.bet_kind})")
    print(f"  entry:    ${bet.fill_size_usdc:.2f} @ {bet.fill_price:.3f}  "
          f"(max possible: ${bet.fill_size_usdc / bet.fill_price - bet.fill_size_usdc:.2f})")
    print(f"  actual PnL: ${bet.actual_pnl:+.2f}")
    print()
    print(f"  {'Threshold':>10}  {'Triggered':>10}  {'@Price':>8}  "
          f"{'T+min':>7}  {'Cashout $':>10}  {'vs Actual':>11}  {'Verdict':>8}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*8}  "
          f"{'-'*7}  {'-'*10}  {'-'*11}  {'-'*8}")

    for r in analysis.results:
        if r.triggered:
            trig = "YES"
            price_str = f"{r.trigger_price:.3f}"
            mins_str = f"{r.minutes_into_match:+.0f}m" if r.minutes_into_match is not None else "  ?"
            cashout_str = f"${r.cashout_pnl:+.2f}"
            delta_str = f"${r.delta_vs_actual:+.2f}"
            # Verdict: positive delta = cashout was better
            if r.delta_vs_actual is not None and r.delta_vs_actual > 0:
                verdict = "SAVED" if bet.actual_pnl < 0 else "LESS +"
            else:
                verdict = "WORSE"
        else:
            trig = "no"
            price_str = mins_str = cashout_str = delta_str = verdict = "–"

        print(f"  {r.threshold:>10.1f}x  {trig:>10}  {price_str:>8}  "
              f"{mins_str:>7}  {cashout_str:>10}  {delta_str:>11}  {verdict:>8}")


def _print_aggregate(analyses: List[BetAnalysis]) -> None:
    print(f"\n{'═' * 72}")
    print("  AGGREGATE SUMMARY")
    print(f"  Total bets: {len(analyses)}")
    print(f"  {'Threshold':>10}  "
          f"{'Triggered':>10}  {'Total Cashout $':>16}  "
          f"{'Actual PnL $':>13}  {'Net Improvement $':>18}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*16}  {'-'*13}  {'-'*18}")

    total_actual = sum(a.bet.actual_pnl for a in analyses)

    for i, threshold in enumerate(THRESHOLDS):
        n_triggered = 0
        total_cashout = 0.0
        total_improvement = 0.0

        for analysis in analyses:
            r = analysis.results[i]
            if r.triggered and r.cashout_pnl is not None:
                n_triggered += 1
                total_cashout += r.cashout_pnl
                # For non-triggered bets, actual PnL counts as-is
            else:
                total_cashout += analysis.bet.actual_pnl
            total_improvement = total_cashout - total_actual

        print(f"  {threshold:>10.1f}x  {n_triggered:>10}  "
              f"${total_cashout:>+14.2f}  "
              f"${total_actual:>+11.2f}  "
              f"${total_improvement:>+16.2f}")

    print(f"  {'–'*10}  {'–'*10}  {'–'*16}  {'–'*13}  {'–'*18}")
    print(f"  {'Hold all':>10}   {'–':>9}  {'–':>16}  "
          f"${total_actual:>+11.2f}  {'$0.00':>18}")

    # Highlight the best threshold
    best_threshold = None
    best_improvement = float("-inf")
    for i, threshold in enumerate(THRESHOLDS):
        total_cashout = 0.0
        for analysis in analyses:
            r = analysis.results[i]
            if r.triggered and r.cashout_pnl is not None:
                total_cashout += r.cashout_pnl
            else:
                total_cashout += analysis.bet.actual_pnl
        improvement = total_cashout - total_actual
        if improvement > best_improvement:
            best_improvement = improvement
            best_threshold = threshold

    if best_threshold and best_improvement > 0:
        print(f"\n  >>> Best threshold: {best_threshold}x  "
              f"(+${best_improvement:.2f} over hold-all) <<<")
    else:
        print("\n  >>> No threshold improved on hold-all for this dataset <<<")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backtest in-game cashout thresholds on historical bet_ledger data"
    )
    parser.add_argument(
        "--days", type=int, default=14,
        help="Lookback window in days (default: 14)",
    )
    parser.add_argument(
        "--bet-kind", choices=["paper", "real", "both"], default="both",
        help="Filter by bet kind (default: both)",
    )
    parser.add_argument(
        "--min-stake", type=float, default=1.0,
        help="Minimum fill_size_usdc to include (default: 1.0)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Debug-level logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    from src.data.database import init_cashout_columns
    init_cashout_columns()

    conn = get_connection()
    poly_client = PolymarketClient()

    kind_filter = None if args.bet_kind == "both" else args.bet_kind
    bets = _fetch_settled_bets(conn, args.days, kind_filter, args.min_stake)
    conn.close()

    if not bets:
        print(f"No settled bets found in the last {args.days} days.")
        return 0

    print(f"\nFetching price history for {len(bets)} settled bet(s)...")
    print("(1-minute fidelity via Polymarket CLOB /prices-history)")

    analyses: List[BetAnalysis] = []

    for idx, bet in enumerate(bets, 1):
        print(f"  [{idx}/{len(bets)}] bet_id={bet.bet_id} {bet.side_label[:40]:<40}", end=" ", flush=True)

        try:
            resp = poly_client.get_prices_history(
                token_id=bet.token_id,
                interval="all",
                fidelity=1,
            )
            history = resp.get("history") if isinstance(resp, dict) else resp
            history = history or []
            print(f"({len(history)} price points)")
        except Exception as exc:
            print(f"ERROR: {exc}")
            logger.warning(f"  Price history failed for bet_id={bet.bet_id}: {exc}")
            history = []

        results = _simulate_thresholds(bet, history, THRESHOLDS)
        analyses.append(BetAnalysis(bet=bet, results=results))

        time.sleep(RATE_LIMIT_SLEEP)

    # ---- Per-bet detail ----
    print(f"\n{'═' * 72}")
    print("  PER-BET CASHOUT SIMULATION")
    print(f"  Lookback: last {args.days} days | bet_kind: {args.bet_kind}")
    for analysis in analyses:
        _print_bet_section(analysis)

    # ---- Aggregate ----
    _print_aggregate(analyses)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
