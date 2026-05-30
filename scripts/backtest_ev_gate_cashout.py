#!/usr/bin/env python3
"""Backtest an EV-aware profit-take gate against the current tiered rule.

Thesis
------
The live profit-take fires on a blind return-ratio (price/entry >= 1.2-1.3x),
ignoring model conviction, so it sells value bets the model still expects to
win. An EV-aware gate instead sells only once the market has caught up to (or
overshot) our model's fair value:

    SELL when  price * (1 - fee) >= model_prob          (fee-aware EV gate)

i.e. taking the cash now beats holding to settlement *in expectation* given
our own probability estimate. Below that, the model still sees edge -> HOLD.

Policies compared (all on the SAME held-to-settlement sample; we exclude rows
that were actually cashed out so `pnl_realised_usdc` is a clean hold baseline):

    hold        : keep every position to settlement (baseline = actual pnl)
    tiered      : current live rule (tiered_cashout_threshold, ratio-based)
    ev_gate     : sell at first price where price*(1-fee) >= model_prob
    ev_raw      : sell at first price >= model_prob (no fee buffer)
    ev_and_tier : require BOTH the tiered ratio AND price*(1-fee) >= model_prob

Fee note
--------
Cashout pnl always pays the taker fee; the held (paper) baseline is
frictionless. That asymmetry penalises *every* cashout policy equally, so the
head-to-head (ev_gate - tiered) is the fee-robust signal. We report both the
improvement-over-hold and the head-to-head, and keep paper/real separate.

Usage
-----
    python scripts/backtest_ev_gate_cashout.py --days 14 --bet-kind both
    python scripts/backtest_ev_gate_cashout.py --days 45 --bet-kind real
    python scripts/backtest_ev_gate_cashout.py --days 28 --no-cache
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_connection
from src.integrations.polymarket import PolymarketClient
from src.integrations.odds.polymarket_compare import POLYMARKET_TAKER_FEE
from src.integrations.polymarket.cashout import tiered_cashout_threshold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RATE_LIMIT_SLEEP = 0.4
_CACHE_DIR = Path("/tmp/_evgate_price_cache")


@dataclass
class Bet:
    bet_id: int
    fixture_key: str
    side_label: str
    bet_kind: str
    strategy_label: Optional[str]
    fill_price: float
    fill_size_usdc: float
    filled_at: Optional[str]
    kickoff_at: Optional[str]
    actual_pnl: float            # hold-to-settlement pnl (baseline)
    settle_outcome: Optional[int]
    model_prob: Optional[float]
    token_id: str
    history: List[Tuple[int, float]] = field(default_factory=list)


def _parse_ts(iso: Optional[str]) -> Optional[int]:
    if not iso:
        return None
    try:
        return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())
    except (ValueError, AttributeError):
        return None


def _cashout_pnl(fill_price: float, fill_size_usdc: float, sell_price: float) -> float:
    shares = fill_size_usdc / fill_price
    gross = shares * sell_price
    fee = gross * POLYMARKET_TAKER_FEE
    return round(gross - fill_size_usdc - fee, 4)


def _fetch_bets(conn, days: int, bet_kind: Optional[str], min_stake: float) -> List[Bet]:
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    sql = """
        SELECT bet_id, fixture_key, side_label, COALESCE(bet_kind,'real') AS bet_kind,
               strategy_label, fill_price, fill_size_usdc, filled_at, kickoff_at,
               pnl_realised_usdc, settle_outcome, model_prob, polymarket_token_id
        FROM bet_ledger
        WHERE status='settled' AND proposed_at >= ?
          AND fill_price IS NOT NULL AND fill_size_usdc IS NOT NULL
          AND pnl_realised_usdc IS NOT NULL AND polymarket_token_id IS NOT NULL
          AND model_prob IS NOT NULL
          AND fill_size_usdc >= ?
          AND cashout_triggered_at IS NULL
    """
    params: List[Any] = [since, min_stake]
    if bet_kind:
        sql += " AND (COALESCE(bet_kind,'real') = ?)"
        params.append(bet_kind)
    sql += " ORDER BY proposed_at DESC"
    cur = conn.cursor()
    cur.execute(sql, params)
    out = []
    for r in cur.fetchall():
        out.append(Bet(
            bet_id=r["bet_id"], fixture_key=r["fixture_key"], side_label=r["side_label"] or "?",
            bet_kind=r["bet_kind"], strategy_label=r["strategy_label"],
            fill_price=float(r["fill_price"]), fill_size_usdc=float(r["fill_size_usdc"]),
            filled_at=r["filled_at"], kickoff_at=r["kickoff_at"],
            actual_pnl=float(r["pnl_realised_usdc"]), settle_outcome=r["settle_outcome"],
            model_prob=float(r["model_prob"]) if r["model_prob"] is not None else None,
            token_id=r["polymarket_token_id"],
        ))
    return out


def _load_history(bet: Bet, client: PolymarketClient, use_cache: bool) -> List[Tuple[int, float]]:
    cache_f = _CACHE_DIR / f"{bet.token_id}.json"
    if use_cache and cache_f.exists():
        try:
            raw = json.loads(cache_f.read_text())
            return [(int(t), float(p)) for t, p in raw]
        except Exception:
            pass
    try:
        resp = client.get_prices_history(token_id=bet.token_id, interval="all", fidelity=1)
        hist = resp.get("history", []) if isinstance(resp, dict) else (resp or [])
        pts = [(int(p["t"]), float(p["p"])) for p in hist
               if isinstance(p, dict) and p.get("p") is not None]
        pts.sort()
    except Exception as exc:
        logger.warning(f"history fetch failed bet={bet.bet_id}: {exc}")
        pts = []
    if use_cache:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_f.write_text(json.dumps(pts))
    time.sleep(RATE_LIMIT_SLEEP)
    return pts


def _post_fill(bet: Bet) -> List[Tuple[int, float]]:
    fill_ts = _parse_ts(bet.filled_at) or 0
    return [(t, p) for t, p in bet.history if t >= fill_ts and 0.0 < p < 1.0]


def _first_where(series, predicate):
    for t, p in series:
        if predicate(p):
            return t, p
    return None


def _policy_pnl(bet: Bet, policy: str) -> Tuple[float, bool]:
    """Return (pnl, triggered). pnl falls back to hold (actual_pnl) when no trigger."""
    series = _post_fill(bet)
    if not series:
        return bet.actual_pnl, False
    fp, mp = bet.fill_price, (bet.model_prob or 1.0)
    fee = POLYMARKET_TAKER_FEE
    tier = tiered_cashout_threshold(fp)

    hit = None
    if policy == "hold":
        return bet.actual_pnl, False
    elif policy == "tiered":
        if tier is None:
            return bet.actual_pnl, False
        hit = _first_where(series, lambda p: (p / fp) >= tier)
    elif policy == "ev_gate":
        # take profit once selling beats holding in EV (net of fee); only at a profit
        thr = max(mp / (1 - fee), fp)
        hit = _first_where(series, lambda p: p >= thr)
    elif policy == "ev_raw":
        thr = max(mp, fp)
        hit = _first_where(series, lambda p: p >= thr)
    elif policy == "ev_and_tier":
        if tier is None:
            return bet.actual_pnl, False
        thr = max(mp / (1 - fee), fp * tier)
        hit = _first_where(series, lambda p: (p / fp) >= tier and p >= mp / (1 - fee))
    else:
        raise ValueError(policy)

    if hit is None:
        return bet.actual_pnl, False
    return _cashout_pnl(fp, bet.fill_size_usdc, hit[1]), True


POLICIES = ["hold", "tiered", "ev_gate", "ev_raw", "ev_and_tier"]


def _summarize(bets: List[Bet], label: str) -> None:
    base = sum(b.actual_pnl for b in bets)
    print(f"\n{'='*78}\n  {label}  (n={len(bets)} held bets)")
    print(f"  hold-to-settlement baseline P&L: ${base:+.2f}")
    print(f"\n  {'policy':>12} {'n_fire':>7} {'totalP&L':>11} {'vs hold':>10} "
          f"{'left':>9} {'saved':>9} {'vs tiered':>10}")
    print(f"  {'-'*12} {'-'*7} {'-'*11} {'-'*10} {'-'*9} {'-'*9} {'-'*10}")
    tiered_total = None
    rows = {}
    for pol in POLICIES:
        total = nfire = left = saved = 0.0
        nfire = 0
        for b in bets:
            pnl, fired = _policy_pnl(b, pol)
            total += pnl
            if fired:
                nfire += 1
            d = pnl - b.actual_pnl  # +saved / -left
            if d > 0:
                saved += d
            else:
                left += -d
        rows[pol] = total
        if pol == "tiered":
            tiered_total = total
        vs_hold = total - base
        vs_tiered = (total - tiered_total) if tiered_total is not None and pol != "hold" else 0.0
        vt = f"${vs_tiered:+.2f}" if pol not in ("hold", "tiered") else ("—" if pol == "hold" else "—")
        print(f"  {pol:>12} {nfire:>7} ${total:>+9.2f} ${vs_hold:>+8.2f} "
              f"${left:>+7.2f} ${saved:>+7.2f} {vt:>10}")


def _bucket(fp: float) -> str:
    if fp < 0.20: return "5-20c"
    if fp < 0.35: return "20-35c"
    if fp < 0.50: return "35-50c"
    if fp < 0.65: return "50-65c"
    if fp < 0.80: return "65-80c"
    return "80-95c"


def _bucket_breakdown(bets: List[Bet], label: str) -> None:
    print(f"\n  --- {label}: ev_gate vs tiered by entry bucket ---")
    print(f"  {'bucket':>8} {'n':>4} {'tiered_vs_hold':>15} {'evgate_vs_hold':>15} {'edge':>9}")
    by: Dict[str, List[Bet]] = {}
    for b in bets:
        by.setdefault(_bucket(b.fill_price), []).append(b)
    order = ["5-20c", "20-35c", "35-50c", "50-65c", "65-80c", "80-95c"]
    for bk in order:
        grp = by.get(bk)
        if not grp:
            continue
        base = sum(b.actual_pnl for b in grp)
        t = sum(_policy_pnl(b, "tiered")[0] for b in grp) - base
        e = sum(_policy_pnl(b, "ev_gate")[0] for b in grp) - base
        print(f"  {bk:>8} {len(grp):>4} ${t:>+13.2f} ${e:>+13.2f} ${e-t:>+7.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--bet-kind", choices=["paper", "real", "both"], default="both")
    ap.add_argument("--min-stake", type=float, default=1.0)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    conn = get_connection()
    kind = None if args.bet_kind == "both" else args.bet_kind
    bets = _fetch_bets(conn, args.days, kind, args.min_stake)
    conn.close()
    if not bets:
        print("No held settled bets in window.")
        return 0

    client = PolymarketClient()
    print(f"Loading price history for {len(bets)} bets (cache={'off' if args.no_cache else 'on'})...")
    for i, b in enumerate(bets, 1):
        b.history = _load_history(b, client, use_cache=not args.no_cache)
        if i % 20 == 0:
            print(f"  ...{i}/{len(bets)}")

    print(f"\nWindow: last {args.days} days | fee={POLYMARKET_TAKER_FEE}")
    for kind_label in (["real", "paper"] if args.bet_kind == "both" else [args.bet_kind]):
        grp = [b for b in bets if b.bet_kind == kind_label]
        if not grp:
            continue
        _summarize(grp, f"{kind_label.upper()} bets")
        _bucket_breakdown(grp, kind_label.upper())
    return 0


if __name__ == "__main__":
    sys.exit(main())
