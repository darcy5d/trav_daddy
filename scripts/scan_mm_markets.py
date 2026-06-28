#!/usr/bin/env python3
"""Wave 6 pre-work (W2): order-book spread / depth / reward recon scanner.

Polls live Polymarket cricket markets (moneyline + side markets) across the
pre-match -> toss -> in-play window and appends one `mm_market_snapshots` row
per (outcome token, poll). This builds the spread / depth / reward time-series
we currently have ZERO of, and answers empirically:

  * How wide are cricket spreads, per market type, and how do they evolve?
  * Where (if anywhere) is there non-toxic flow / book depth?
  * Is cricket in Polymarket's liquidity-reward set, and at what band?

Examples
--------
  # single sweep of everything in the next 96h
  python scripts/scan_mm_markets.py

  # only side markets, single sweep
  python scripts/scan_mm_markets.py --markets top_batter,most_sixes

  # poll every 10 min for 6 hours (build a real time series across a match)
  python scripts/scan_mm_markets.py --loop --interval-min 10 --duration-min 360

  # fast offline-ish pass (no CLOB reward probe network calls)
  python scripts/scan_mm_markets.py --no-rewards-probe

Read-mostly: only writes to the append-only mm_market_snapshots table.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_db_connection, init_mm_snapshots
from src.integrations.polymarket import PolymarketClient
from src.integrations.polymarket.rewards import summarize_market_rewards
from src.integrations.polymarket.upcoming import find_upcoming_cricket_events

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("scan_mm_markets")

DEFAULT_MARKETS = ["moneyline", "top_batter", "most_sixes", "toss_match_double"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _volume_num(raw_market: Dict[str, Any]) -> Optional[float]:
    if not isinstance(raw_market, dict):
        return None
    for key in ("volumeNum", "volume", "volume24hr", "volumeClob"):
        v = raw_market.get(key)
        if v in (None, ""):
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return None


def _iter_markets_for_fixture(
    fixture: Dict[str, Any], wanted: List[str]
) -> List[Dict[str, Any]]:
    """Flatten a fixture's moneyline + side markets into a scan list."""
    out: List[Dict[str, Any]] = []
    if "moneyline" in wanted and fixture.get("moneyline"):
        out.append(fixture["moneyline"])
    side_markets = fixture.get("side_markets") or {}
    for kind, market_list in side_markets.items():
        if kind not in wanted:
            continue
        for market_record in market_list or []:
            out.append(market_record)
    return out


def _snapshot_token(
    client: PolymarketClient,
    fixture: Dict[str, Any],
    market_record: Dict[str, Any],
    outcome: Dict[str, Any],
    rewards: Dict[str, Any],
    ts: str,
    hours_to_kickoff: Optional[float],
    kickoff_at_iso: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Build one snapshot row dict for an outcome token (book read here)."""
    token_id = outcome.get("token_id")
    if not token_id:
        return None
    try:
        book = client.get_book_spread(token_id)
    except Exception as exc:
        logger.debug(f"book read failed for {token_id}: {exc}")
        book = {
            "bid": None, "ask": None, "spread_pp": None,
            "best_bid_size": 0.0, "best_ask_size": 0.0, "midpoint": None,
        }
    return {
        "ts": ts,
        "fixture_key": fixture.get("fixture_key"),
        "tournament_prefix": fixture.get("tournament_prefix"),
        "format": fixture.get("format"),
        "gender": fixture.get("gender"),
        "market_type": market_record.get("kind"),
        "market_id": market_record.get("market_id"),
        "token_id": token_id,
        "outcome_label": outcome.get("label"),
        "best_bid": book.get("bid"),
        "best_ask": book.get("ask"),
        "spread_pp": book.get("spread_pp"),
        "best_bid_size": book.get("best_bid_size"),
        "best_ask_size": book.get("best_ask_size"),
        "midpoint": book.get("midpoint"),
        "last_price": outcome.get("last_price"),
        "volume_num": _volume_num(market_record.get("raw_market") or {}),
        "kickoff_at": kickoff_at_iso,
        "hours_to_kickoff": hours_to_kickoff,
        "in_reward_set": rewards.get("in_reward_set"),
        "reward_min_size": rewards.get("reward_min_size"),
        "reward_max_spread": rewards.get("reward_max_spread"),
        "reward_json": rewards.get("reward_json"),
        "fee_schedule_json": rewards.get("fee_schedule_json"),
    }


def _persist(rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0
    cols = [
        "ts", "fixture_key", "tournament_prefix", "format", "gender",
        "market_type", "market_id", "token_id", "outcome_label",
        "best_bid", "best_ask", "spread_pp", "best_bid_size", "best_ask_size",
        "midpoint", "last_price", "volume_num", "kickoff_at", "hours_to_kickoff",
        "in_reward_set", "reward_min_size", "reward_max_spread", "reward_json",
        "fee_schedule_json",
    ]
    placeholders = ",".join("?" for _ in cols)
    sql = f"INSERT INTO mm_market_snapshots ({','.join(cols)}) VALUES ({placeholders})"
    with get_db_connection() as conn:
        conn.executemany(sql, [[r.get(c) for c in cols] for r in rows])
    return len(rows)


def scan_once(
    client: PolymarketClient,
    wanted_markets: List[str],
    hours_ahead: float,
    probe_rewards: bool,
    limit_fixtures: Optional[int],
    dry_run: bool,
    max_stale_hours: float = 12.0,
) -> Dict[str, Any]:
    ts = _now_iso()
    now = datetime.now(timezone.utc)
    fixtures = find_upcoming_cricket_events(
        client, hours_ahead=hours_ahead, include_started=True
    )

    # Drop ancient fixtures that are still flagged "active" on Gamma but
    # resolved long ago (they quote bid~0.001/ask~0.999, i.e. fake ~100pp
    # spreads). Keep pre-match and in-play (kickoff up to max_stale_hours ago).
    kept = []
    for fx in fixtures:
        start_est = fx.get("scheduled_start_estimate")
        if isinstance(start_est, datetime):
            htk = (start_est - now).total_seconds() / 3600.0
            if htk < -abs(max_stale_hours):
                continue
        kept.append(fx)
    fixtures = kept

    if limit_fixtures:
        fixtures = fixtures[:limit_fixtures]

    rows: List[Dict[str, Any]] = []
    reward_hits = 0
    spreads: List[float] = []

    for fixture in fixtures:
        start_est = fixture.get("scheduled_start_estimate")
        hours_to_kickoff = None
        kickoff_at_iso = None
        if isinstance(start_est, datetime):
            kickoff_at_iso = start_est.isoformat()
            hours_to_kickoff = (start_est - now).total_seconds() / 3600.0

        for market_record in _iter_markets_for_fixture(fixture, wanted_markets):
            raw_market = market_record.get("raw_market") or {}
            if probe_rewards:
                try:
                    rewards = summarize_market_rewards(
                        raw_market,
                        condition_id=raw_market.get("conditionId"),
                        probe_clob=True,
                    )
                except Exception as exc:
                    logger.debug(f"reward summary failed: {exc}")
                    rewards = {
                        "in_reward_set": None, "reward_min_size": None,
                        "reward_max_spread": None, "reward_json": None,
                        "fee_schedule_json": None,
                    }
            else:
                rewards = {
                    "in_reward_set": None, "reward_min_size": None,
                    "reward_max_spread": None, "reward_json": None,
                    "fee_schedule_json": None,
                }
            if rewards.get("in_reward_set") == 1:
                reward_hits += 1

            for outcome in market_record.get("outcomes") or []:
                row = _snapshot_token(
                    client, fixture, market_record, outcome, rewards,
                    ts, hours_to_kickoff, kickoff_at_iso,
                )
                if row is None:
                    continue
                rows.append(row)
                if row["spread_pp"] is not None:
                    spreads.append(row["spread_pp"])

    written = 0
    if dry_run:
        for r in rows[:25]:
            logger.info(
                f"  [{r['market_type']:18}] {str(r['fixture_key'])[:34]:34} "
                f"{str(r['outcome_label'])[:14]:14} bid={r['best_bid']} ask={r['best_ask']} "
                f"spread_pp={r['spread_pp']} reward={r['in_reward_set']}"
            )
        if len(rows) > 25:
            logger.info(f"  ... and {len(rows) - 25} more (dry-run, not written)")
    else:
        written = _persist(rows)

    avg_spread = sum(spreads) / len(spreads) if spreads else None
    return {
        "ts": ts,
        "fixtures": len(fixtures),
        "rows": len(rows),
        "written": written,
        "reward_market_hits": reward_hits,
        "avg_spread_pp": avg_spread,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours-ahead", type=float, default=96.0)
    ap.add_argument(
        "--markets",
        default=",".join(DEFAULT_MARKETS),
        help="comma list: moneyline,top_batter,most_sixes,toss_match_double",
    )
    ap.add_argument("--loop", action="store_true", help="poll repeatedly")
    ap.add_argument("--interval-min", type=float, default=15.0)
    ap.add_argument(
        "--duration-min", type=float, default=0.0,
        help="with --loop: stop after this many minutes (0 = run until killed)",
    )
    ap.add_argument(
        "--no-rewards-probe", action="store_true",
        help="skip the CLOB reward network probe (Gamma fields only / faster)",
    )
    ap.add_argument("--limit-fixtures", type=int, default=None)
    ap.add_argument(
        "--max-stale-hours", type=float, default=12.0,
        help="drop fixtures whose kickoff is more than this many hours in the "
             "past (filters resolved-but-still-active junk markets)",
    )
    ap.add_argument("--dry-run", action="store_true", help="don't write to DB")
    args = ap.parse_args()

    wanted = [m.strip() for m in args.markets.split(",") if m.strip()]
    if not args.dry_run:
        init_mm_snapshots()

    client = PolymarketClient()
    probe_rewards = not args.no_rewards_probe

    def _one() -> None:
        res = scan_once(
            client, wanted, args.hours_ahead, probe_rewards,
            args.limit_fixtures, args.dry_run,
            max_stale_hours=args.max_stale_hours,
        )
        spread_str = (
            f"{res['avg_spread_pp']:.1f}pp" if res["avg_spread_pp"] is not None else "n/a"
        )
        logger.info(
            f"sweep {res['ts']}: fixtures={res['fixtures']} tokens={res['rows']} "
            f"written={res['written']} reward_markets={res['reward_market_hits']} "
            f"avg_spread={spread_str}"
        )

    if not args.loop:
        _one()
        return 0

    deadline = (
        time.time() + args.duration_min * 60 if args.duration_min > 0 else None
    )
    interval = max(30.0, args.interval_min * 60)
    while True:
        try:
            _one()
        except Exception as exc:
            logger.warning(f"sweep failed (continuing): {exc}")
        if deadline is not None and time.time() >= deadline:
            break
        time.sleep(interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
