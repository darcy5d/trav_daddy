#!/usr/bin/env python3
"""Replay Wave 5.6 modern-era bets chronologically with a starting bankroll.

Compares several stake-sizing strategies side-by-side:
  - Flat $25 (matches the captured bet_size_usd)
  - Flat 2.5% of starting bankroll ($25 on $1000)
  - Flat 5% of starting bankroll ($50 on $1000) - more aggressive
  - Half-Kelly (per-bet stake = 0.5 * (p - q) / (1 - q) * current_bankroll)
  - Quarter-Kelly (safety margin variant)

For each strategy and each (tournament, market, lookback, edge_threshold)
filter, prints final bankroll, total return %, max drawdown, # bets,
win rate, and the realised Sharpe per-bet.

Bets with the same (match_id, market_type, outcome_label) are deduped:
within the same match we only place ONE bet per market - the one at the
specified entry lookback. Without dedupe we'd double-count by adding the
same bet at every lookback in our captured set.

Settlement is recomputed for the chosen stake from (market_price_pre,
settle_outcome) using:
  shares = stake / market_price
  payout = shares * settle_outcome
  pnl = payout - stake - (stake * fee_pct)

Polymarket taker fee = 2%.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

POLYMARKET_TAKER_FEE = 0.02


def _safe_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def load_bets(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open() as fp:
        for row in csv.DictReader(fp):
            row["edge_pp"] = _safe_float(row.get("edge_pp"))
            row["model_prob"] = _safe_float(row.get("model_prob"))
            row["market_price_pre"] = _safe_float(row.get("market_price_pre"))
            row["settle_outcome"] = _safe_float(row.get("settle_outcome"))
            rows.append(row)
    return rows


def filter_bets(
    bets: List[Dict[str, Any]],
    sim_version: Optional[str] = None,
    toss_mode: Optional[str] = None,
    tournament: Optional[str] = None,
    market_type: Optional[str] = None,
    lookback: Optional[str] = None,
    min_edge_pp: float = 3.0,
    price_range: Tuple[float, float] = (0.10, 0.90),
) -> List[Dict[str, Any]]:
    out = []
    for b in bets:
        if sim_version and b.get("sim_version") != sim_version:
            continue
        if toss_mode and b.get("toss_mode") != toss_mode:
            continue
        if tournament and b.get("tournament") != tournament:
            continue
        if market_type and b.get("market_type") != market_type:
            continue
        if lookback and b.get("lookback_label") != lookback:
            continue
        if b["edge_pp"] is None or b["edge_pp"] < min_edge_pp:
            continue
        if (b["market_price_pre"] is None
                or b["market_price_pre"] < price_range[0]
                or b["market_price_pre"] > price_range[1]):
            continue
        if b["settle_outcome"] is None:
            continue
        out.append(b)
    # Dedupe: one bet per (match_id, market_type, outcome_label, sim_version, toss_mode)
    # If multiple lookbacks exist, keep the FIRST in chronological order (already
    # filtered by lookback above if user specified one).
    seen = set()
    deduped = []
    for b in out:
        key = (b["match_id"], b["market_type"], b["outcome_label"],
               b["sim_version"], b["toss_mode"], b.get("lookback_label", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(b)
    return deduped


def _pnl_for_stake(stake: float, market_price: float, settle_outcome: float, fee_pct: float = POLYMARKET_TAKER_FEE) -> float:
    if market_price <= 0:
        return -stake
    shares = stake / market_price
    payout = shares * settle_outcome
    fee = stake * fee_pct
    return payout - stake - fee


def _kelly_fraction(model_prob: float, market_price: float, kelly_mult: float = 0.5) -> float:
    """Kelly fraction for buying YES at price q with true prob p.
    f* = (p - q) / (1 - q). Returns 0 if non-positive. Capped at 25% by default
    (so even full-Kelly is at most 25% of bankroll on a single bet).
    """
    if market_price >= 1.0 or market_price <= 0:
        return 0.0
    f_star = (model_prob - market_price) / (1.0 - market_price)
    f_star = max(0.0, min(f_star, 1.0))
    return min(f_star * kelly_mult, 0.25)  # hard cap at 25% of bankroll


def replay_strategy(
    bets: List[Dict[str, Any]],
    starting_bankroll: float,
    strategy: str,
    flat_amount: Optional[float] = None,
    flat_pct_of_initial: Optional[float] = None,
    kelly_mult: float = 0.5,
) -> Dict[str, Any]:
    """Replay bets chronologically; return summary stats.

    strategy in {"flat_dollars", "flat_pct_initial", "kelly"}
    """
    sorted_bets = sorted(bets, key=lambda b: b.get("match_date", ""))
    bankroll = starting_bankroll
    peak = starting_bankroll
    max_dd = 0.0
    bet_log = []
    n_wins = 0

    for b in sorted_bets:
        mp = b["market_price_pre"]
        settle = b["settle_outcome"]
        p = b["model_prob"]

        if strategy == "flat_dollars":
            stake = flat_amount or 25.0
        elif strategy == "flat_pct_initial":
            stake = (flat_pct_of_initial or 0.025) * starting_bankroll
        elif strategy == "kelly":
            f = _kelly_fraction(p, mp, kelly_mult=kelly_mult)
            stake = f * bankroll
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Cap stake at the available bankroll (can't bet more than you have)
        stake = max(0.0, min(stake, bankroll))
        if stake <= 0:
            continue

        pnl = _pnl_for_stake(stake, mp, settle)
        bankroll += pnl
        if pnl > 0:
            n_wins += 1
        if bankroll > peak:
            peak = bankroll
        dd = (peak - bankroll) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
        bet_log.append({
            "date": b.get("match_date", ""),
            "stake": round(stake, 2),
            "pnl": round(pnl, 2),
            "bankroll_after": round(bankroll, 2),
        })
        if bankroll <= 0.01:  # Wipeout
            break

    n = len(bet_log)
    if n == 0:
        return {"n_bets": 0, "final_bankroll": starting_bankroll}
    final = bankroll
    total_return = (final - starting_bankroll) / starting_bankroll
    return {
        "n_bets": n,
        "n_wins": n_wins,
        "win_rate": round(n_wins / n, 4) if n else None,
        "final_bankroll": round(final, 2),
        "total_return_pct": round(total_return * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "first_bet": bet_log[0] if bet_log else None,
        "last_bet": bet_log[-1] if bet_log else None,
    }


def print_strategy_table(label: str, bets: List[Dict[str, Any]], bankroll: float = 1000.0):
    if not bets:
        print(f"\n{'=' * 78}\n  {label}: NO BETS QUALIFIED\n{'=' * 78}")
        return
    print(f"\n{'=' * 78}")
    print(f"  {label}")
    print(f"  n_bets={len(bets)}  starting_bankroll=${bankroll:.0f}")
    print('=' * 78)

    # Date range
    dates = sorted(b.get("match_date", "") for b in bets if b.get("match_date"))
    if dates:
        print(f"  Date range: {dates[0]} -> {dates[-1]}")

    print(f"\n  {'Strategy':<28} {'n':>4} {'Win':>6} {'Final $':>10} {'Return':>10} {'Max DD':>8}")
    print(f"  {'-'*28} {'-'*4} {'-'*6} {'-'*10} {'-'*10} {'-'*8}")
    strategies = [
        ("Flat $25 per bet", "flat_dollars", {"flat_amount": 25.0}),
        ("Flat $50 per bet (5% init)", "flat_dollars", {"flat_amount": 50.0}),
        ("Flat $100 per bet (10% init)", "flat_dollars", {"flat_amount": 100.0}),
        ("Half-Kelly (cap 25%)", "kelly", {"kelly_mult": 0.5}),
        ("Quarter-Kelly (cap 25%)", "kelly", {"kelly_mult": 0.25}),
        ("Eighth-Kelly", "kelly", {"kelly_mult": 0.125}),
    ]
    for name, strat, kwargs in strategies:
        result = replay_strategy(bets, bankroll, strat, **kwargs)
        if result["n_bets"] == 0:
            print(f"  {name:<28} {'--':>4} {'--':>6} {'--':>10} {'--':>10} {'--':>8}")
            continue
        print(f"  {name:<28} {result['n_bets']:>4} {result['win_rate']:>6.2f} "
              f"${result['final_bankroll']:>9.2f} {result['total_return_pct']:>9.2f}% "
              f"{result['max_drawdown_pct']:>7.2f}%")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bets-csv", default="data/diagnostics/wave_5_6_modern_era_sweep/master_bets.csv")
    parser.add_argument("--bankroll", type=float, default=1000.0)
    args = parser.parse_args()

    bets = load_bets(Path(args.bets_csv))
    print(f"Loaded {len(bets):,} raw bet rows from {args.bets_csv}")

    # Strategy 1: ODI men T-3h V3 pinned, edge >= 3pp (the strongest signal cell)
    odi_t3h_v3 = filter_bets(bets, sim_version="v3", toss_mode="pinned",
                              tournament="odi_men", lookback="T-3h", min_edge_pp=3.0)
    print_strategy_table("STRATEGY 1: ODI men T-3h, V3 pinned, edge >= 3pp", odi_t3h_v3, args.bankroll)

    # Strategy 2: ODI men T-3h V2 (V2 actually had higher win rate in the report)
    odi_t3h_v2 = filter_bets(bets, sim_version="v2", toss_mode="uncertain",
                              tournament="odi_men", lookback="T-3h", min_edge_pp=3.0)
    print_strategy_table("STRATEGY 2: ODI men T-3h, V2, edge >= 3pp", odi_t3h_v2, args.bankroll)

    # Strategy 3: Broader - V3 pinned, ANY tournament, T-3h, edge >= 5pp
    broad_t3h_v3 = filter_bets(bets, sim_version="v3", toss_mode="pinned",
                                lookback="T-3h", min_edge_pp=5.0)
    print_strategy_table("STRATEGY 3: V3 pinned, T-3h, all tournaments, edge >= 5pp", broad_t3h_v3, args.bankroll)

    # Strategy 4: Even broader - V3 pinned, all lookbacks <= 6h, edge >= 5pp
    # Pick best lookback per match -> we need to dedupe by (match, market, outcome, sim, toss)
    # taking the EARLIEST lookback that qualifies (so we'd actually bet earlier)
    # Implementation: filter to short-window lookbacks, then dedupe gives one per match
    short_lookback_v3 = [b for b in bets
                         if b.get("sim_version") == "v3"
                         and b.get("toss_mode") == "pinned"
                         and b.get("lookback_label") in {"T-30min", "T-1h", "T-3h", "T-6h"}
                         and b.get("edge_pp") is not None and b["edge_pp"] >= 5.0
                         and b.get("market_price_pre") is not None
                         and 0.10 <= b["market_price_pre"] <= 0.90
                         and b.get("settle_outcome") is not None]
    # Dedupe by (match, market, outcome) keeping the EARLIEST lookback
    # i.e. T-6h > T-3h > T-1h > T-30min in priority (we bet earlier)
    lookback_priority = {"T-6h": 4, "T-3h": 3, "T-1h": 2, "T-30min": 1}
    short_lookback_v3.sort(key=lambda b: -lookback_priority.get(b.get("lookback_label", ""), 0))
    seen = set()
    short_dedupe = []
    for b in short_lookback_v3:
        key = (b["match_id"], b["market_type"], b["outcome_label"])
        if key in seen:
            continue
        seen.add(key)
        short_dedupe.append(b)
    print_strategy_table("STRATEGY 4: V3 pinned, earliest qualifying lookback in [T-6h, T-30min], all tournaments, edge >= 5pp",
                         short_dedupe, args.bankroll)

    # Strategy 5: ALL V3 pinned bets, edge >= 5pp, picking earliest entry
    all_lookback_v3 = [b for b in bets
                       if b.get("sim_version") == "v3"
                       and b.get("toss_mode") == "pinned"
                       and b.get("edge_pp") is not None and b["edge_pp"] >= 5.0
                       and b.get("market_price_pre") is not None
                       and 0.10 <= b["market_price_pre"] <= 0.90
                       and b.get("settle_outcome") is not None]
    all_lookback_priority = {"T-3d": 8, "T-2d": 7, "T-1d": 6, "T-12h": 5, "T-6h": 4, "T-3h": 3, "T-1h": 2, "T-30min": 1}
    all_lookback_v3.sort(key=lambda b: -all_lookback_priority.get(b.get("lookback_label", ""), 0))
    seen2 = set()
    all_dedupe = []
    for b in all_lookback_v3:
        key = (b["match_id"], b["market_type"], b["outcome_label"])
        if key in seen2:
            continue
        seen2.add(key)
        all_dedupe.append(b)
    print_strategy_table("STRATEGY 5: V3 pinned, EARLIEST entry across all lookbacks, edge >= 5pp",
                         all_dedupe, args.bankroll)

    print(f"\n{'=' * 78}")
    print("Notes:")
    print("  - Each strategy replays its filtered bet list chronologically.")
    print("  - Stakes are capped at current bankroll (no leverage / borrowing).")
    print("  - Half-Kelly = recommended safety margin; full Kelly is too aggressive.")
    print("  - Compounding effect = bankroll grows so per-bet stakes grow too (Kelly modes).")
    print("  - Slippage NOT modelled: actual fills at $25-100 stake on thin markets")
    print("    (especially side markets) would face material price impact. Haircut")
    print("    realised returns by 5-15% before trusting these as 'live performance'.")
    print('=' * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
