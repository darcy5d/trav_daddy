#!/usr/bin/env python3
"""Stop-loss / loss-mitigation analysis (analyze-only).

Mirror image of scripts/backtest_inplay_cashout.py. That harness simulates
profit-taking (selling when price RISES). This one simulates loss mitigation:
selling when price FALLS below a floor, optionally after a time-gate, with an
optional re-entry leg ("sold at 0.20, re-buy at 0.40, chase 1.4-2.0x").

Key constraint: the model probability is pre-match / at-toss only. Every
in-play decision here (stop-loss, the live profit-take, re-entry) is a pure
price-action decision with no fresh model edge. Re-entry in particular is a
momentum bet and is judged purely on backtested returns.

The whole point is to measure BOTH sides of the trade:
  - recovered-loss $: residual we'd claw back on bets that went to zero, and
  - sacrificed-win $: the false-stop cost on bets that dipped, got stopped
    out, then recovered to win.

This script makes NO changes to the live system. It only reads bet_ledger
and the public Polymarket /prices-history endpoint.

Usage:
    python scripts/backtest_stop_loss.py
    python scripts/backtest_stop_loss.py --days 30
    python scripts/backtest_stop_loss.py --days 12 --min-stake 1.0
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the existing harness plumbing (fetch + ts parsing + BetRecord).
_bic = importlib.import_module("backtest_inplay_cashout")
BetRecord = _bic.BetRecord
_fetch_settled_bets = _bic._fetch_settled_bets
_parse_ts = _bic._parse_ts

from src.data.database import get_connection
from src.integrations.polymarket import PolymarketClient
from src.integrations.odds.polymarket_compare import POLYMARKET_TAKER_FEE
from src.integrations.polymarket.cashout import tiered_cashout_threshold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RATE_LIMIT_SLEEP = 0.4  # seconds between price-history calls

# --- Grids (from the plan) ---
FLOORS = [0.10, 0.15, 0.20, 0.25, 0.30]
# Time-gates in minutes into the match. 0 = no gate; 105 ~= 2nd-innings proxy
# for a T20 (each innings ~90min + breaks => 2nd innings starts ~T+105m).
GATES = [0, 30, 45, 60, 105]
GATE_LABELS = {0: "none", 30: "T+30", 45: "T+45", 60: "T+60", 105: "2ndInn"}
# Re-entry variants: name -> ("off"|"abs"|"ratio", level)
REENTRY_VARIANTS: Dict[str, Tuple[str, float]] = {
    "off": ("off", 0.0),
    "reentry@0.40": ("abs", 0.40),
    "reentry@1.5x": ("ratio", 1.5),
}

# Fixtures the user called out for per-bet inspection.
NAMED_FIXTURE_SUBSTRINGS = ["nld-sco", "irl-wst", "ess-ham", "ken-sle", "cmr-ivo"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_outcome(bet: BetRecord) -> int:
    """1 if our side won (token settles to $1), else 0."""
    if bet.settle_outcome in (0, 1):
        return int(bet.settle_outcome)
    # Fallback: a full/partial loss means the side lost; profit means it won.
    return 1 if bet.actual_pnl > 0 else 0


def _post_fill_ticks(
    bet: BetRecord, history: List[Dict[str, Any]]
) -> Tuple[List[Tuple[int, float, float]], Optional[int]]:
    """Return [(ts, price, minutes_into_match), ...] for ticks after fill.

    minutes_into_match is relative to kickoff_at; None-kickoff bets get
    minutes measured from fill (so the "no gate" case still works, but
    time-gated cells are flagged separately as un-evaluable).
    """
    fill_ts = _parse_ts(bet.filled_at) or 0
    kickoff_ts = _parse_ts(bet.kickoff_at)
    anchor = kickoff_ts if kickoff_ts is not None else fill_ts
    out: List[Tuple[int, float, float]] = []
    for p in history:
        if not isinstance(p, dict):
            continue
        ts = int(p.get("t", 0))
        if ts < fill_ts:
            continue
        price = float(p.get("p", 0.0))
        mins = (ts - anchor) / 60.0
        out.append((ts, price, mins))
    out.sort(key=lambda x: x[0])
    return out, kickoff_ts


@dataclass
class PolicyResult:
    pnl: float
    stop_fired: bool
    stop_price: Optional[float]
    stop_minute: Optional[float]
    profit_sold: bool
    reentered: bool
    final_action: str  # held | stopped | profit | stopped_reentered_*


def _simulate(
    bet: BetRecord,
    ticks: List[Tuple[int, float, float]],
    outcome: int,
    *,
    stop_floor: Optional[float],
    gate_min: float,
    reentry: Tuple[str, float],
    use_profit_take: bool,
    fee: float = POLYMARKET_TAKER_FEE,
) -> PolicyResult:
    """Walk the 1-min path as a state machine and return realised PnL.

    Cashflow convention (absolute, anchored to the real outcome):
        pnl = -stake
              + sell proceeds net of taker fee
              - re-buy cost incl taker fee
              + remaining_shares * outcome   (gross redemption at settlement)

    A policy that never trades reproduces the hold-to-settlement baseline
    exactly, so deltas isolate only the actively-traded portion.

    Assumptions (documented):
      - At most ONE re-entry; after re-buying we do not stop-loss again
        (avoids pathological churn), only profit-take or settle.
      - Re-buy restores the same share count at the recovery price, which
        becomes the new effective fill for the profit-take ratio.
      - Taker fee applied on both sell and re-buy notionals (conservative).
    """
    shares = bet.fill_size_usdc / bet.fill_price
    effective_fill = bet.fill_price
    pnl = -bet.fill_size_usdc

    state = "HOLDING"  # HOLDING | WATCHING | FLAT
    stop_fired = False
    stop_price: Optional[float] = None
    stop_minute: Optional[float] = None
    profit_sold = False
    reentered = False
    final_action = "held"

    re_mode, re_level = reentry

    for _ts, price, mins in ticks:
        if price <= 0.0 or price >= 1.0:
            # Untradeable tick (settled extreme); skip for decisions.
            continue

        if state == "HOLDING":
            # 1. Profit-take (live tiered config) on current effective fill.
            if use_profit_take:
                thr = tiered_cashout_threshold(effective_fill)
                if thr is not None and (price / effective_fill) >= thr:
                    pnl += shares * price * (1.0 - fee)
                    shares = 0.0
                    profit_sold = True
                    final_action = "profit" if not stop_fired else "stopped_reentered_profit"
                    state = "FLAT"
                    break
            # 2. Stop-loss (only before any re-entry, and past the time-gate).
            if stop_floor is not None and not stop_fired and mins >= gate_min and price <= stop_floor:
                pnl += shares * price * (1.0 - fee)
                shares = 0.0
                stop_fired = True
                stop_price = price
                stop_minute = mins
                if re_mode != "off":
                    state = "WATCHING"
                    final_action = "stopped_watching"
                else:
                    state = "FLAT"
                    final_action = "stopped"
                continue

        elif state == "WATCHING":
            # Re-entry: re-buy the same share count at the recovery price.
            trigger = False
            if re_mode == "abs" and price >= re_level:
                trigger = True
            elif re_mode == "ratio" and stop_price and price >= re_level * stop_price:
                trigger = True
            if trigger:
                rebuy_shares = bet.fill_size_usdc / bet.fill_price
                cost = rebuy_shares * price
                pnl -= cost * (1.0 + fee)
                shares = rebuy_shares
                effective_fill = price
                reentered = True
                state = "HOLDING"
                final_action = "stopped_reentered_held"
                # Do not stop-loss again; continue to profit-take/settle.
                continue

    # Settlement of anything still held.
    if shares > 0:
        pnl += shares * outcome

    return PolicyResult(
        pnl=round(pnl, 4),
        stop_fired=stop_fired,
        stop_price=stop_price,
        stop_minute=stop_minute,
        profit_sold=profit_sold,
        reentered=reentered,
        final_action=final_action,
    )


def _baseline_pnl(bet: BetRecord, outcome: int) -> float:
    """Engine hold-to-settlement PnL (consistent anchor for deltas)."""
    shares = bet.fill_size_usdc / bet.fill_price
    return round(-bet.fill_size_usdc + shares * outcome, 4)


# ---------------------------------------------------------------------------
# Per-bet record assembled once, then re-used across all policy cells.
# ---------------------------------------------------------------------------

@dataclass
class BetPaths:
    bet: BetRecord
    ticks: List[Tuple[int, float, float]]
    outcome: int
    kickoff_ts: Optional[int]
    n_points: int
    min_price: Optional[float]
    max_price: Optional[float]
    last_price: Optional[float]
    baseline_pnl: float

    @property
    def has_kickoff(self) -> bool:
        return self.kickoff_ts is not None


def _build_paths(bet: BetRecord, history: List[Dict[str, Any]]) -> BetPaths:
    ticks, kickoff_ts = _post_fill_ticks(bet, history)
    tradeable = [p for _t, p, _m in ticks if 0.0 < p < 1.0]
    outcome = _resolve_outcome(bet)
    return BetPaths(
        bet=bet,
        ticks=ticks,
        outcome=outcome,
        kickoff_ts=kickoff_ts,
        n_points=len(ticks),
        min_price=min(tradeable) if tradeable else None,
        max_price=max(tradeable) if tradeable else None,
        last_price=tradeable[-1] if tradeable else None,
        baseline_pnl=_baseline_pnl(bet, outcome),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _aggregate_pnl(
    paths: List[BetPaths],
    *,
    stop_floor: Optional[float],
    gate_min: float,
    reentry: Tuple[str, float],
    use_profit_take: bool,
) -> Tuple[float, int, int]:
    """Return (total_pnl, n_stops, n_unevaluable_gate) for a policy cell."""
    total = 0.0
    n_stops = 0
    n_uneval = 0
    for bp in paths:
        # Time-gated stop on a bet with no kickoff anchor is un-evaluable;
        # fall back to hold (no stop) and count it.
        if stop_floor is not None and gate_min > 0 and not bp.has_kickoff:
            total += bp.baseline_pnl if not use_profit_take else _simulate(
                bp.bet, bp.ticks, bp.outcome, stop_floor=None, gate_min=0,
                reentry=("off", 0.0), use_profit_take=use_profit_take,
            ).pnl
            n_uneval += 1
            continue
        res = _simulate(
            bp.bet, bp.ticks, bp.outcome,
            stop_floor=stop_floor, gate_min=gate_min,
            reentry=reentry, use_profit_take=use_profit_take,
        )
        total += res.pnl
        if res.stop_fired:
            n_stops += 1
    return round(total, 2), n_stops, n_uneval


def _print_grid_floor_x_gate(paths: List[BetPaths], baseline_total: float) -> None:
    print(f"\n{'=' * 78}")
    print("  GRID A: stop-loss + live profit-take, NO re-entry")
    print("  Cell value = net improvement vs hold-all ($).  (+ = better than holding)")
    print(f"{'=' * 78}")
    header = f"  {'floor':>7} |" + "".join(f"{GATE_LABELS[g]:>10}" for g in GATES)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for f in FLOORS:
        cells = []
        for g in GATES:
            total, n_stops, _ = _aggregate_pnl(
                paths, stop_floor=f, gate_min=g, reentry=("off", 0.0), use_profit_take=True
            )
            improvement = total - baseline_total
            cells.append(f"{improvement:>+8.2f}({n_stops})")
        print(f"  {f:>7.2f} |" + "".join(f"{c:>10}" for c in cells))
    print("  (number in parentheses = bets stopped out in that cell)")


def _print_grid_reentry(paths: List[BetPaths], baseline_total: float, gate_min: int) -> None:
    print(f"\n{'=' * 78}")
    print(f"  GRID B: re-entry effect at gate={GATE_LABELS[gate_min]} (stop-loss + profit-take)")
    print("  Net improvement vs hold-all ($) for each re-entry rule.")
    print(f"{'=' * 78}")
    header = f"  {'floor':>7} |" + "".join(f"{name:>16}" for name in REENTRY_VARIANTS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for f in FLOORS:
        cells = []
        for name, variant in REENTRY_VARIANTS.items():
            total, _, _ = _aggregate_pnl(
                paths, stop_floor=f, gate_min=gate_min, reentry=variant, use_profit_take=True
            )
            cells.append(f"{total - baseline_total:>+14.2f}")
        print(f"  {f:>7.2f} |" + "".join(f"{c:>16}" for c in cells))


def _print_winners_losers_split(
    paths: List[BetPaths], baseline_total: float, headline_cells: List[Tuple[float, int]]
) -> None:
    print(f"\n{'=' * 78}")
    print("  WINNERS vs LOSERS SPLIT  (stop-loss + profit-take, no re-entry)")
    print("  recovered = extra $ on bets that lost;  sacrificed = $ given up on bets that won")
    print(f"{'=' * 78}")
    print(f"  {'floor':>6} {'gate':>7} | {'recovered($)':>13} {'sacrificed($)':>14} "
          f"{'net($)':>10} {'#stop W':>8} {'#stop L':>8}")
    print("  " + "-" * 74)
    for f, g in headline_cells:
        recovered = 0.0
        sacrificed = 0.0
        n_stop_w = 0
        n_stop_l = 0
        for bp in paths:
            if g > 0 and not bp.has_kickoff:
                continue
            res = _simulate(
                bp.bet, bp.ticks, bp.outcome,
                stop_floor=f, gate_min=g, reentry=("off", 0.0), use_profit_take=True,
            )
            # Compare to profit-take-only (so we isolate the STOP effect, not
            # the pre-existing profit-take).
            pt_only = _simulate(
                bp.bet, bp.ticks, bp.outcome,
                stop_floor=None, gate_min=0, reentry=("off", 0.0), use_profit_take=True,
            )
            delta = res.pnl - pt_only.pnl
            if not res.stop_fired:
                continue
            if bp.outcome == 1:
                sacrificed += delta  # negative when we stopped a winner
                n_stop_w += 1
            else:
                recovered += delta  # positive when we clawed back a loser
                n_stop_l += 1
        net = recovered + sacrificed
        print(f"  {f:>6.2f} {GATE_LABELS[g]:>7} | {recovered:>+13.2f} {sacrificed:>+14.2f} "
              f"{net:>+10.2f} {n_stop_w:>8} {n_stop_l:>8}")


def _print_named_fixtures(paths: List[BetPaths]) -> None:
    print(f"\n{'=' * 78}")
    print("  PER-BET DETAIL: user-named fixtures")
    print("  Shows the price path so you can see if a window to exit existed.")
    print(f"{'=' * 78}")
    named = [bp for bp in paths
             if any(s in bp.bet.fixture_key for s in NAMED_FIXTURE_SUBSTRINGS)]
    if not named:
        print("  (none of the named fixtures were in the fetched set)")
        return
    for bp in named:
        b = bp.bet
        tag = "WIN" if bp.outcome == 1 else "LOSS"
        print(f"\n  {b.fixture_key}  [{tag}]  {b.side_label}  ({b.strategy_label or 'manual'})")
        print(f"    entry ${b.fill_size_usdc:.2f} @ {b.fill_price:.3f}   actual PnL ${b.actual_pnl:+.2f}")
        if bp.n_points == 0:
            print("    NO PRICE HISTORY (thin market / no ticks) — cannot evaluate")
            continue
        print(f"    path: {bp.n_points} pts   min={bp.min_price:.3f}  "
              f"max={bp.max_price:.3f}  last={bp.last_price:.3f}")
        # What each floor would have recovered (no gate, no re-entry).
        for f in FLOORS:
            res = _simulate(
                bp.bet, bp.ticks, bp.outcome,
                stop_floor=f, gate_min=0, reentry=("off", 0.0), use_profit_take=True,
            )
            if res.stop_fired:
                verdict = "recovered" if (bp.outcome == 0 and res.pnl > bp.baseline_pnl) else (
                    "FALSE-STOP" if bp.outcome == 1 else "worse")
                print(f"      floor {f:.2f}: stop @ {res.stop_price:.3f} "
                      f"(T{res.stop_minute:+.0f}m)  pnl ${res.pnl:+.2f}  "
                      f"vs hold ${bp.baseline_pnl:+.2f}  [{verdict}]")
            else:
                print(f"      floor {f:.2f}: never triggered (min price "
                      f"{bp.min_price:.3f} > floor)")


def _print_data_quality(paths: List[BetPaths]) -> None:
    print(f"\n{'=' * 78}")
    print("  DATA QUALITY")
    print(f"{'=' * 78}")
    no_hist = [bp for bp in paths if bp.n_points == 0]
    sparse = [bp for bp in paths if 0 < bp.n_points < 20]
    no_kick = [bp for bp in paths if not bp.has_kickoff]
    print(f"  bets analysed:               {len(paths)}")
    print(f"  no price history at all:     {len(no_hist)}  "
          f"(excluded from stop simulation; hold-to-settle)")
    print(f"  sparse (<20 pts):            {len(sparse)}  (treat paths with caution)")
    print(f"  missing kickoff_at:          {len(no_kick)}  "
          f"(time-gated cells fall back to hold for these)")
    if no_hist:
        print("    no-history fixtures: "
              + ", ".join(sorted({bp.bet.fixture_key for bp in no_hist}))[:200])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest stop-loss / loss-mitigation policies")
    parser.add_argument("--days", type=int, default=12, help="Lookback window in days (default 12)")
    parser.add_argument("--min-stake", type=float, default=1.0, help="Min fill_size_usdc (default 1.0)")
    parser.add_argument("--bet-kind", choices=["paper", "real", "both"], default="real")
    parser.add_argument("--verbose", action="store_true")
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

    print(f"\nStop-loss analysis | lookback {args.days}d | bet_kind={args.bet_kind} "
          f"| {len(bets)} bets | taker fee {POLYMARKET_TAKER_FEE * 100:.0f}%")
    print("Fetching 1-min Polymarket price history per bet...")

    paths: List[BetPaths] = []
    for idx, bet in enumerate(bets, 1):
        print(f"  [{idx}/{len(bets)}] bet_id={bet.bet_id} {bet.side_label[:34]:<34}", end=" ", flush=True)
        try:
            resp = poly_client.get_prices_history(token_id=bet.token_id, interval="all", fidelity=1)
            history = resp.get("history") if isinstance(resp, dict) else resp
            history = history or []
            print(f"({len(history)} pts)")
        except Exception as exc:
            print(f"ERROR: {exc}")
            history = []
        paths.append(_build_paths(bet, history))
        time.sleep(RATE_LIMIT_SLEEP)

    baseline_total = round(sum(bp.baseline_pnl for bp in paths), 2)
    actual_total = round(sum(bp.bet.actual_pnl for bp in paths), 2)

    print(f"\n{'=' * 78}")
    print(f"  HOLD-TO-SETTLEMENT BASELINE (engine): ${baseline_total:+.2f}")
    print(f"  (DB actual PnL total for reference:   ${actual_total:+.2f}; "
          f"small gap = entry-fee treatment)")
    print(f"{'=' * 78}")

    _print_grid_floor_x_gate(paths, baseline_total)
    _print_grid_reentry(paths, baseline_total, gate_min=0)

    # Headline cells for the winners/losers split: a spread of floors at no-gate
    # plus the 2nd-innings-gated 0.20 floor.
    headline = [(0.15, 0), (0.20, 0), (0.25, 0), (0.30, 0), (0.20, 105)]
    _print_winners_losers_split(paths, baseline_total, headline)

    _print_named_fixtures(paths)
    _print_data_quality(paths)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
