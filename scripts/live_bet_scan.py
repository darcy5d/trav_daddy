#!/usr/bin/env python3
"""Wave 5.8: scan upcoming Polymarket cricket fixtures and place REAL bets.

This script is a deliberate clone of `scripts/paper_bet_scan.py` with the
minimum surface-area changes needed to cross from paper to live. Same
strategies, same filters, same 48h lookback, same Kelly sizing. Diffs:

    * Dedup check uses bet_kind='real' (paper bets don't block live bets).
    * Only strategies listed in BETTING_LIVE_STRATEGIES in .env are live-enabled.
    * Placement routes through `src.integrations.polymarket.bet_placement.place_bet`
      which runs the server-side risk gate (per-strategy cap + per-day stake
      cap + per-bet cap + mode gate) and calls the real Polymarket CLOB.
    * requested_mode='auto' — AUTO bets fire automatically when the risk gate
      passes; no human in the loop per bet.
    * Bankroll for Kelly sizing = BETTING_MAX_DEPOSIT_PER_STRATEGY + realised_pnl
      on this strategy's live settled bets (compounding starts at the cap).

Idempotent: re-running the same scan does NOT double-bet. The dedup key is
(strategy_label, fixture_key, market_id, side_label, phase) filtered to
bet_kind='real'. Paper bets on the same fixture stay in parallel, untouched.

Usage:
    venv311/bin/python scripts/live_bet_scan.py [--hours-ahead 96] [--dry-run]
                                                [--strategies ALL|name1,name2]

Cron (hourly):
    0 * * * * cd <repo> && venv311/bin/python scripts/live_bet_scan.py \
        >> logs/live_daily.log 2>&1
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_connection, get_active_model_snapshot
from src.integrations.polymarket import PolymarketClient
from src.integrations.polymarket.bet_placement import place_bet, place_bet_twap
from src.integrations.polymarket.upcoming import (
    find_upcoming_cricket_events,
    attach_db_team_ids,
)
from src.integrations.polymarket.paper_inputs import (
    get_recent_xi,
    get_cached_xi,
    get_default_venue_for_team,
)
from src.integrations.polymarket.paper_strategies import (
    PaperStrategy,
    get_enabled_strategies,
)

# Polymarket CLOB minimum order is $1 USDC; orders below this get rejected.
POLYMARKET_MIN_ORDER_USDC = 1.0

# Sizing fractions applied to the live bankroll.
# Min: 0.5% — below this Kelly is too thin to bother placing.
# Max: 25% — matches the kelly_fraction_cap in PaperStrategy so high-edge bets
#   get proportionally larger stakes rather than all hitting a flat 10% cap.
#   (Previously 10%, which made every bet with >10pp edge identical in size.)
LIVE_MIN_STAKE_FRACTION = 0.005
LIVE_MAX_STAKE_FRACTION = 0.25


def live_scaled_kelly_stake(
    model_prob: float,
    market_price: float,
    live_bankroll_usdc: float,
    strategy: PaperStrategy,
) -> float:
    """Half-Kelly stake sized RELATIVE to the live bankroll.

    Paper's `kelly_stake_usdc` uses ABSOLUTE dollar min/max ($5 / $100)
    tuned for the $1000 paper bankroll. When live runs at a $100 bankroll
    those translate to 5% / 25% of bankroll (via bankroll * 0.25), which
    distorts the skip pattern: any bet paper sizes under $50 falls below
    live's $5 min and gets silently dropped. That makes paper-vs-live
    non-comparable.

    This variant keeps kelly_mult and kelly_fraction_cap identical to
    paper but replaces absolute min/max with bankroll-relative 0.5% / 10%,
    producing exactly 1/10 the paper stake for the same model/market inputs
    at the same relative bankroll. Every skip paper makes, live makes too.
    Polymarket's $1 minimum order size is enforced as a floor on top.
    """
    if market_price <= 0 or market_price >= 1:
        return 0.0
    if model_prob <= 0 or model_prob >= 1:
        return 0.0
    f_star = (model_prob - market_price) / (1.0 - market_price)
    f_star = max(0.0, min(f_star, 1.0))
    f_capped = min(f_star * strategy.kelly_mult, strategy.kelly_fraction_cap)
    raw_stake = f_capped * live_bankroll_usdc
    scaled_min = max(
        LIVE_MIN_STAKE_FRACTION * live_bankroll_usdc,
        POLYMARKET_MIN_ORDER_USDC,
    )
    scaled_max = LIVE_MAX_STAKE_FRACTION * live_bankroll_usdc
    if raw_stake < scaled_min:
        return 0.0
    return min(raw_stake, scaled_max)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------- Filter & window helpers (identical to paper scanner) ----------

def _strategy_in_window(strategy: PaperStrategy, hours_to_kickoff: float) -> bool:
    return strategy.lookback_hours_min <= hours_to_kickoff <= strategy.lookback_hours


def _strategy_filters_match_fixture(strategy: PaperStrategy, fixture: Dict[str, Any]) -> bool:
    if strategy.enabled_tournament_prefixes is not None:
        if fixture.get("tournament_prefix") not in strategy.enabled_tournament_prefixes:
            return False
    if strategy.enabled_formats is not None:
        if fixture.get("format") not in strategy.enabled_formats:
            return False
    if strategy.enabled_genders is not None:
        if fixture.get("gender") not in strategy.enabled_genders:
            return False
    return True


def _moneyline_outcome_for_team(moneyline_market: Dict[str, Any], team_db_name: str) -> Optional[Dict[str, Any]]:
    if not moneyline_market or not team_db_name:
        return None
    from src.integrations.odds.polymarket_compare import PolymarketComparisonService

    outcomes = moneyline_market.get("outcomes", [])

    for outcome in outcomes:
        if PolymarketComparisonService.label_matches_team(outcome.get("label", ""), team_db_name):
            return outcome

    GENERIC = {"the", "and", "of", "fc", "cc", "kings", "super", "knight", "riders",
               "royal", "challengers", "indians", "pindiz", "warriors", "titans",
               "club", "team", "xi", "men", "women", "u19", "a"}
    def _meaningful_tokens(text: str):
        return [t for t in text.lower().replace(",", " ").replace(".", " ").replace("-", " ").split()
                if len(t) >= 5 and t not in GENERIC]

    db_tokens = _meaningful_tokens(team_db_name)
    if not db_tokens:
        return None
    for outcome in outcomes:
        lbl_tokens = _meaningful_tokens(outcome.get("label", ""))
        for db_t in db_tokens:
            for lbl_t in lbl_tokens:
                if (db_t.startswith(lbl_t[:5]) and lbl_t.startswith(db_t[:5])):
                    return outcome
    return None


# ---------- Idempotency helpers (filter on bet_kind='real') ----------

def _already_bet_live(conn, strategy_label: str, fixture_key: str, market_id: str,
                      side_label: str, phase: str = "pre_toss") -> bool:
    """Dedup check for live bets.

    EXCLUDES status='errored' rows — a bet that failed to place (risk-gate
    reject, CLOB error, etc.) should be retryable on the next cron tick
    since nothing actually hit the exchange. Filled / placed / settled /
    cancelled bets DO block retry (otherwise we'd double-bet).

    Also checks order_plans for pending/executing TWAP plans on this fixture.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT bet_id FROM bet_ledger
        WHERE bet_kind = 'real'
          AND status != 'errored'
          AND strategy_label = ?
          AND fixture_key = ?
          AND polymarket_market_id = ?
          AND side_label = ?
          AND COALESCE(phase, 'pre_toss') = ?
        LIMIT 1
        """,
        (strategy_label, fixture_key, market_id, side_label, phase),
    )
    if cur.fetchone() is not None:
        return True

    # Check for active TWAP plans on this fixture/strategy
    try:
        cur.execute(
            """
            SELECT plan_id FROM order_plans
            WHERE strategy_label = ?
              AND fixture_key = ?
              AND status IN ('pending', 'executing')
            LIMIT 1
            """,
            (strategy_label, fixture_key),
        )
        if cur.fetchone() is not None:
            return True
    except Exception:
        pass
    return False


def _strategy_has_any_live_bet_on_fixture(conn, strategy_label: str,
                                          fixture_key: str,
                                          phase: str = "pre_toss") -> bool:
    """Has this strategy placed a NON-errored live bet on the fixture?

    Excludes errored so the re-sim skip-optimisation doesn't suppress
    retries of previously-failed attempts.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT 1 FROM bet_ledger
        WHERE bet_kind = 'real'
          AND status != 'errored'
          AND strategy_label = ?
          AND fixture_key = ?
          AND COALESCE(phase, 'pre_toss') = ?
        LIMIT 1
        """,
        (strategy_label, fixture_key, phase),
    )
    return cur.fetchone() is not None


# ---------- Live-bankroll helper ----------

def get_live_strategy_bankroll(strategy_name: str, conn) -> float:
    """Compute the current live bankroll for a strategy.

    Starting bankroll: checks BETTING_MAX_DEPOSIT_<STRATEGY_NAME_UPPER> first
    (per-strategy override), then falls back to BETTING_MAX_DEPOSIT_PER_STRATEGY.
    Realised P&L on live (bet_kind='real') settled bets compounds on top.
    Open positions are NOT subtracted (they don't reduce the Kelly basis;
    the risk gate's per-strategy open-exposure cap enforces concentration).
    """
    import os
    from config import BETTING_CONFIG

    # Per-strategy override: e.g. BETTING_MAX_DEPOSIT_V3_MARG_3PP=146
    env_key = f"BETTING_MAX_DEPOSIT_{strategy_name.upper().replace('-', '_')}"
    override = os.getenv(env_key)
    if override is not None:
        starting = float(override)
    else:
        starting = float(BETTING_CONFIG.get("max_deposit_per_strategy_usdc", 100))
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COALESCE(SUM(pnl_realised_usdc), 0.0) AS realised
        FROM bet_ledger
        WHERE bet_kind = 'real'
          AND strategy_label = ?
          AND status = 'settled'
        """,
        (strategy_name,),
    )
    row = cur.fetchone()
    realised = float(row[0] if not isinstance(row, dict) else row.get("realised") or 0.0) if row else 0.0
    return starting + realised


def _sim_team1_win_prob(simulator, fixture: Dict[str, Any], conn) -> Optional[float]:
    """Run a sim and return P(team1 wins). Caches XI lookups in fixture dict."""
    if "_xi_cache" not in fixture:
        fixture["_xi_cache"] = {}
    cache = fixture["_xi_cache"]
    fmt = fixture["format"]
    gender = fixture["gender"]
    t1 = fixture["team1_id"]
    t2 = fixture["team2_id"]

    if "t1" not in cache:
        fkey = fixture.get("fixture_key", "")
        crex_t1 = get_cached_xi(conn, fkey, t1)
        crex_t2 = get_cached_xi(conn, fkey, t2)
        if crex_t1 is not None:
            cache["t1"] = crex_t1
            logger.info(f"  [XI] team1 from crex_xi_cache ({fkey})")
        else:
            cache["t1"] = get_recent_xi(conn, t1, fmt, gender)
            logger.info(f"  [XI] team1 from historical (crex cache miss/stale)")
        if crex_t2 is not None:
            cache["t2"] = crex_t2
            logger.info(f"  [XI] team2 from crex_xi_cache ({fkey})")
        else:
            cache["t2"] = get_recent_xi(conn, t2, fmt, gender)
            logger.info(f"  [XI] team2 from historical (crex cache miss/stale)")
        cache["venue"] = get_default_venue_for_team(conn, t1, fmt, gender)
    t1_bat, t1_bowl = cache["t1"]
    t2_bat, t2_bowl = cache["t2"]
    venue_id = cache["venue"]
    if len(t1_bat) < 11 or len(t1_bowl) < 5 or len(t2_bat) < 11 or len(t2_bowl) < 5:
        logger.warning(
            f"  [skip] insufficient lineup data for {fixture['team1_db_name']} vs {fixture['team2_db_name']}: "
            f"t1 bat={len(t1_bat)} bowl={len(t1_bowl)}, t2 bat={len(t2_bat)} bowl={len(t2_bowl)}"
        )
        return None

    result = simulator.simulate_matches(
        300, t1_bat, t1_bowl, t2_bat, t2_bowl,
        venue_id=venue_id, team1_id=t1, team2_id=t2,
        use_toss=True, toss_field_prob=0.65, seed=42,
    )
    return float(result["team1_win_prob"])


def scan_and_place_live_bets(
    hours_ahead: float = 96.0,
    strategy_filter: Optional[List[str]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Main scan logic. Returns a summary dict for cron/log consumption."""
    from config import BETTING_CONFIG

    summary: Dict[str, Any] = {
        "scan_started_at": datetime.now(timezone.utc).isoformat(),
        "hours_ahead": hours_ahead,
        "fixtures_seen": 0,
        "fixtures_in_window": 0,
        "fixtures_skipped_no_lineup": 0,
        "bets_placed": [],
        "bets_rejected": [],
        "bets_skipped": [],
    }

    # Enforce mode gate up-front so cron logs make it obvious when betting is off
    mode = str(BETTING_CONFIG.get("mode", "OFF")).upper()
    kill_switch = bool(BETTING_CONFIG.get("kill_switch", False))
    if kill_switch:
        logger.warning("BETTING_KILL_SWITCH=1 — kill switch engaged; exiting without scanning")
        summary["exit_reason"] = "kill_switch_engaged"
        return summary
    if mode == "OFF":
        logger.warning("BETTING_MODE=OFF — nothing to scan for live. Flip via /api/betting/mode or .env")
        summary["exit_reason"] = "betting_mode_off"
        return summary
    if mode != "AUTO":
        logger.warning(f"BETTING_MODE={mode} — live scanner only acts in AUTO mode; exiting")
        summary["exit_reason"] = f"betting_mode_{mode.lower()}"
        return summary

    live_strategy_names = set(BETTING_CONFIG.get("live_strategies", []) or [])
    if not live_strategy_names:
        logger.warning("BETTING_LIVE_STRATEGIES is empty — no strategies are live. Nothing to do.")
        summary["exit_reason"] = "no_live_strategies"
        return summary

    enabled_strats = [s for s in get_enabled_strategies() if s.name in live_strategy_names]
    if strategy_filter:
        enabled_strats = [s for s in enabled_strats if s.name in strategy_filter]
    if not enabled_strats:
        logger.warning("No live-enabled strategies match filter; nothing to do")
        summary["exit_reason"] = "no_matching_strategies"
        return summary

    logger.info(
        f"LIVE scanner active. Strategies: {[s.name for s in enabled_strats]}  "
        f"(mode={mode}, kill_switch={kill_switch})"
    )

    # Capture active model snapshot once per scan run for bet_ledger stamping.
    model_snapshot = get_active_model_snapshot()

    needs_v2 = any(s.model_version in ("v2", "consensus") for s in enabled_strats)
    needs_v3 = any(s.model_version in ("v3", "consensus") for s in enabled_strats)

    logger.info(f"Scanning Polymarket for upcoming cricket fixtures (next {hours_ahead}h)...")
    pm_client = PolymarketClient()
    raw_events = find_upcoming_cricket_events(pm_client, hours_ahead=hours_ahead)
    fixtures = attach_db_team_ids(raw_events)
    summary["fixtures_seen"] = len(fixtures)

    if not fixtures:
        logger.info("No upcoming cricket fixtures found")
        return summary

    logger.info(f"  Found {len(fixtures)} upcoming fixtures")

    sim_v2: Dict[str, Any] = {}
    sim_v3: Dict[str, Any] = {}

    now = datetime.now(timezone.utc)
    for fix in fixtures:
        if not (fix.get("team1_id") and fix.get("team2_id")):
            continue
        if not fix.get("moneyline"):
            continue
        kickoff = fix["scheduled_start_estimate"]
        hours_to_kickoff = (kickoff - now).total_seconds() / 3600.0

        fmt = fix["format"]
        gender = fix["gender"]
        sg_key = f"{fmt}_{gender}"

        v2_t1_prob: Optional[float] = None
        v3_t1_prob: Optional[float] = None
        eligible = [s for s in enabled_strats
                    if _strategy_filters_match_fixture(s, fix)
                    and _strategy_in_window(s, hours_to_kickoff)]
        if not eligible:
            continue
        summary["fixtures_in_window"] += 1

        # Same re-sim skip optimisation as paper scanner
        with get_connection() as conn:
            all_eligible_have_bets = all(
                _strategy_has_any_live_bet_on_fixture(conn, s.name, fix["fixture_key"])
                for s in eligible
            )
        if all_eligible_have_bets:
            logger.info(
                f"  [skip-resim] {fix['team1_db_name']} vs {fix['team2_db_name']} "
                f"(T-{hours_to_kickoff:.1f}h) - all {len(eligible)} eligible strategies "
                f"already placed live bets, no need to re-simulate"
            )
            summary["fixtures_in_window"] -= 1
            summary.setdefault("fixtures_skipped_resim", 0)
            summary["fixtures_skipped_resim"] += 1
            continue

        with get_connection() as conn:
            need_v2_now = needs_v2 or any(s.model_version == "consensus" for s in eligible)
            need_v3_now = needs_v3 or any(s.model_version == "consensus" for s in eligible)
            if need_v2_now:
                if sg_key not in sim_v2:
                    from src.models.vectorized_nn_sim_v2 import V2Simulator, V2SimulatorConfig
                    try:
                        sim_v2[sg_key] = V2Simulator(V2SimulatorConfig(format_type=fmt, gender=gender))
                    except Exception as exc:
                        logger.warning(f"  V2 init failed for {sg_key}: {exc}")
                        sim_v2[sg_key] = None
                if sim_v2.get(sg_key) is not None:
                    try:
                        v2_t1_prob = _sim_team1_win_prob(sim_v2[sg_key], fix, conn)
                    except Exception as exc:
                        logger.warning(f"  V2 sim failed for {fix['fixture_key']}: {exc}")

            if need_v3_now:
                if sg_key not in sim_v3:
                    from src.models.vectorized_nn_sim_v3 import V3Simulator, V3SimulatorConfig
                    try:
                        sim_v3[sg_key] = V3Simulator(V3SimulatorConfig(format_type=fmt, gender=gender))
                    except Exception as exc:
                        logger.warning(f"  V3 init failed for {sg_key}: {exc}")
                        sim_v3[sg_key] = None
                if sim_v3.get(sg_key) is not None:
                    try:
                        v3_t1_prob = _sim_team1_win_prob(sim_v3[sg_key], fix, conn)
                    except Exception as exc:
                        logger.warning(f"  V3 sim failed for {fix['fixture_key']}: {exc}")

        if v2_t1_prob is None and v3_t1_prob is None:
            summary["fixtures_skipped_no_lineup"] += 1
            continue

        logger.info(
            f"\n  {fix['team1_db_name']} vs {fix['team2_db_name']} "
            f"(T-{hours_to_kickoff:.1f}h, {fix['format']} {fix['gender']})"
        )
        if v2_t1_prob is not None:
            logger.info(f"    V2 P(team1)={v2_t1_prob*100:.1f}%")
        if v3_t1_prob is not None:
            logger.info(f"    V3 P(team1)={v3_t1_prob*100:.1f}%")
        ml = fix["moneyline"]
        logger.info(f"    Market: " + "  ".join(
            f"{o['label']}={o['last_price']:.3f}" for o in ml["outcomes"] if o["last_price"] is not None
        ))

        t1_outcome = _moneyline_outcome_for_team(ml, fix["team1_db_name"])
        t2_outcome = _moneyline_outcome_for_team(ml, fix["team2_db_name"])
        if t1_outcome is None or t2_outcome is None:
            logger.warning(f"  [skip] couldn't map moneyline outcomes to teams")
            continue
        market_price_t1 = float(t1_outcome.get("last_price") or 0)
        market_price_t2 = float(t2_outcome.get("last_price") or 0)
        if market_price_t1 <= 0 or market_price_t2 <= 0:
            logger.warning(f"  [skip] missing market price")
            continue

        for strat in eligible:
            if strat.model_version == "v2":
                if v2_t1_prob is None:
                    continue
                t1_pred = v2_t1_prob
            elif strat.model_version == "v3":
                if v3_t1_prob is None:
                    continue
                t1_pred = v3_t1_prob
            elif strat.model_version == "consensus":
                if v2_t1_prob is None or v3_t1_prob is None:
                    continue
                v2_back_t1 = (v2_t1_prob - market_price_t1) > 0
                v3_back_t1 = (v3_t1_prob - market_price_t1) > 0
                if v2_back_t1 != v3_back_t1:
                    logger.info(f"    [{strat.name}] V2/V3 disagree on side; skip")
                    continue
                t1_pred = (v2_t1_prob + v3_t1_prob) / 2.0
            else:
                continue
            t2_pred = 1.0 - t1_pred

            edge_t1_pp = (t1_pred - market_price_t1) * 100
            edge_t2_pp = (t2_pred - market_price_t2) * 100

            if edge_t1_pp >= edge_t2_pp:
                back_side, side_label, side_token_id, model_prob, market_price, edge_pp = (
                    "t1", t1_outcome.get("label"), t1_outcome.get("token_id"),
                    t1_pred, market_price_t1, edge_t1_pp,
                )
            else:
                back_side, side_label, side_token_id, model_prob, market_price, edge_pp = (
                    "t2", t2_outcome.get("label"), t2_outcome.get("token_id"),
                    t2_pred, market_price_t2, edge_t2_pp,
                )

            if edge_pp < strat.min_edge_pp:
                summary["bets_skipped"].append({
                    "strategy": strat.name, "fixture": fix["fixture_key"],
                    "reason": f"edge {edge_pp:.1f}pp < {strat.min_edge_pp:.1f}pp",
                    "side_label": side_label, "model_prob": model_prob,
                    "market_price": market_price,
                })
                continue
            if not (strat.min_market_price <= market_price <= strat.max_market_price):
                summary["bets_skipped"].append({
                    "strategy": strat.name, "fixture": fix["fixture_key"],
                    "reason": f"market_price {market_price:.3f} outside [{strat.min_market_price:.2f}, {strat.max_market_price:.2f}]",
                    "side_label": side_label,
                })
                continue
            # Model probability bounds: exclude coin-flip zone and overconfident bets.
            if strat.min_model_prob is not None and model_prob < strat.min_model_prob:
                summary["bets_skipped"].append({
                    "strategy": strat.name, "fixture": fix["fixture_key"],
                    "reason": f"model_prob {model_prob:.3f} < min {strat.min_model_prob}",
                    "side_label": side_label,
                })
                continue
            if strat.max_model_prob is not None and model_prob > strat.max_model_prob:
                summary["bets_skipped"].append({
                    "strategy": strat.name, "fixture": fix["fixture_key"],
                    "reason": f"model_prob {model_prob:.3f} > max {strat.max_model_prob}",
                    "side_label": side_label,
                })
                continue

            with get_connection() as conn:
                if _already_bet_live(conn, strat.name, fix["fixture_key"],
                                     ml.get("market_id") or "", side_label or ""):
                    summary["bets_skipped"].append({
                        "strategy": strat.name, "fixture": fix["fixture_key"],
                        "reason": "already-placed",
                    })
                    continue

                bankroll_now = get_live_strategy_bankroll(strat.name, conn)
                if bankroll_now <= 0:
                    summary["bets_skipped"].append({
                        "strategy": strat.name, "fixture": fix["fixture_key"],
                        "reason": "bankroll-exhausted",
                    })
                    continue
                stake = live_scaled_kelly_stake(model_prob, market_price, bankroll_now, strat)
                if stake <= 0:
                    summary["bets_skipped"].append({
                        "strategy": strat.name, "fixture": fix["fixture_key"],
                        "reason": "kelly-stake-zero",
                    })
                    continue

            # place_bet() opens its own DB connection; don't nest.
            if dry_run:
                logger.info(
                    f"    [{strat.name}] DRY-RUN: BACK {side_label} @ {market_price:.3f} "
                    f"size=${stake:.2f}  edge=+{edge_pp:.1f}pp  bank=${bankroll_now:.2f}"
                )
                summary["bets_placed"].append({
                    "bet_id": -1, "strategy": strat.name, "fixture": fix["fixture_key"],
                    "team1": fix["team1_db_name"], "team2": fix["team2_db_name"],
                    "side_label": side_label, "model_prob": model_prob,
                    "market_price": market_price, "edge_pp": edge_pp,
                    "stake_usdc": stake, "bankroll_at_proposal": bankroll_now,
                    "kickoff_utc": kickoff.isoformat(), "dry_run": True,
                })
                continue

            kickoff_iso = kickoff.isoformat() if kickoff else None

            # --- TWAP routing: check order book spread ---
            import os
            twap_fok_threshold_pp = float(os.getenv("TWAP_FOK_THRESHOLD_PP", "5"))
            max_acceptable_price = model_prob - (strat.min_edge_pp / 100.0)

            use_twap = False
            book_info = None
            try:
                book_info = pm_client.get_book_spread(side_token_id)
            except Exception as exc:
                logger.warning(f"    [{strat.name}] book spread fetch failed: {exc} — defaulting to FOK")

            if book_info and book_info.get("spread_pp") is not None:
                spread_pp = book_info["spread_pp"]
                best_ask = book_info.get("ask")
                best_bid = book_info.get("bid")
                best_ask_size = book_info.get("best_ask_size", 0)

                if spread_pp <= twap_fok_threshold_pp and best_ask is not None and best_ask <= max_acceptable_price:
                    use_twap = False
                    logger.info(
                        f"    [{strat.name}] Spread={spread_pp:.1f}pp <= {twap_fok_threshold_pp}pp, "
                        f"ask={best_ask:.4f} <= max={max_acceptable_price:.4f} -> FOK"
                    )
                elif best_ask is not None and best_ask <= max_acceptable_price:
                    use_twap = True
                    logger.info(
                        f"    [{strat.name}] Spread={spread_pp:.1f}pp > {twap_fok_threshold_pp}pp, "
                        f"asks exist below max={max_acceptable_price:.4f} -> TWAP"
                    )
                elif best_bid is not None:
                    use_twap = True
                    logger.info(
                        f"    [{strat.name}] No asks below max={max_acceptable_price:.4f}, "
                        f"but posting limit orders at ceiling -> TWAP (passive)"
                    )
                else:
                    logger.info(
                        f"    [{strat.name}] Empty book — no bids or asks -> FOK fallback"
                    )

            try:
                if use_twap and book_info:
                    # Anchor TWAP start to at most 10pp below mid. The raw
                    # book bid can be near 0 on thin/lopsided markets, which
                    # would result in chunks placed at 3c on a 90c market.
                    _raw_bid = book_info.get("bid") or 0.0
                    _twap_discount = float(os.getenv("TWAP_START_DISCOUNT_PP", "10")) / 100.0
                    base_price_for_plan = max(_raw_bid, market_price - _twap_discount)
                    result = place_bet_twap(
                        fixture_key=fix["fixture_key"],
                        match_id=None,
                        market_type="moneyline",
                        polymarket_market_id=ml.get("market_id") or "",
                        polymarket_token_id=side_token_id or "",
                        side_label=side_label or "",
                        model_prob=model_prob,
                        market_price_at_proposal=market_price,
                        side="BUY",
                        size_usdc=stake,
                        max_acceptable_price=max_acceptable_price,
                        base_price=base_price_for_plan,
                        best_ask_size=book_info.get("best_ask_size", 10.0),
                        requested_mode="auto",
                        strategy_label=strat.name,
                        bankroll_at_proposal=bankroll_now,
                        phase="pre_toss",
                        kickoff_at=kickoff_iso,
                        model_snapshot=model_snapshot,
                    )
                else:
                    result = place_bet(
                        fixture_key=fix["fixture_key"],
                        match_id=None,
                        market_type="moneyline",
                        polymarket_market_id=ml.get("market_id") or "",
                        polymarket_token_id=side_token_id or "",
                        side_label=side_label or "",
                        model_prob=model_prob,
                        market_price_at_proposal=market_price,
                        side="BUY",
                        size_usdc=stake,
                        requested_mode="auto",
                        strategy_label=strat.name,
                        bankroll_at_proposal=bankroll_now,
                        phase="pre_toss",
                        kickoff_at=kickoff_iso,
                        model_snapshot=model_snapshot,
                    )
            except Exception as exc:
                logger.error(f"    [{strat.name}] place_bet raised: {exc}")
                summary["bets_rejected"].append({
                    "strategy": strat.name, "fixture": fix["fixture_key"],
                    "reason": f"exception: {exc}",
                })
                continue

            if result.get("success"):
                route_tag = "TWAP" if use_twap else "FOK"
                status_str = result.get("status", "placed")
                extra = ""
                if use_twap:
                    extra = f"  plan_id={result.get('plan_id')} chunks={result.get('chunks_total')}"
                logger.info(
                    f"    [{strat.name}] LIVE BET #{result['bet_id']} ({route_tag}): BACK {side_label} @ {market_price:.3f} "
                    f"size=${stake:.2f}  edge=+{edge_pp:.1f}pp  bank=${bankroll_now:.2f}  status={status_str}{extra}"
                )
                summary["bets_placed"].append({
                    "bet_id": result["bet_id"], "strategy": strat.name,
                    "fixture": fix["fixture_key"],
                    "team1": fix["team1_db_name"], "team2": fix["team2_db_name"],
                    "side_label": side_label, "model_prob": model_prob,
                    "market_price": market_price, "edge_pp": edge_pp,
                    "stake_usdc": stake, "bankroll_at_proposal": bankroll_now,
                    "kickoff_utc": kickoff.isoformat(),
                    "status": status_str,
                    "route": route_tag,
                })
            else:
                logger.warning(
                    f"    [{strat.name}] REJECTED: {result.get('reason')} "
                    f"(size=${stake:.2f}, edge=+{edge_pp:.1f}pp)"
                )
                summary["bets_rejected"].append({
                    "strategy": strat.name, "fixture": fix["fixture_key"],
                    "reason": result.get("reason"),
                    "status": result.get("status"),
                    "bet_id": result.get("bet_id"),
                })

    summary["scan_finished_at"] = datetime.now(timezone.utc).isoformat()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan upcoming cricket fixtures for LIVE bets (Wave 5.8)")
    parser.add_argument("--hours-ahead", type=float, default=96.0, help="Look ahead window (default: 96h)")
    parser.add_argument("--strategies", default="ALL", help="Comma-separated strategy names or ALL")
    parser.add_argument("--dry-run", action="store_true", help="Run sims + risk checks but don't call CLOB or write to DB")
    args = parser.parse_args()

    strategy_filter = None
    if args.strategies.upper() != "ALL":
        strategy_filter = [s.strip() for s in args.strategies.split(",") if s.strip()]

    summary = scan_and_place_live_bets(
        hours_ahead=args.hours_ahead,
        strategy_filter=strategy_filter,
        dry_run=args.dry_run,
    )

    print()
    print("=" * 70)
    print("LIVE BET SCAN SUMMARY")
    print("=" * 70)
    if "exit_reason" in summary:
        print(f"  Exited early: {summary['exit_reason']}")
        return 0
    print(f"  Fixtures seen:          {summary['fixtures_seen']}")
    print(f"  In any strategy window: {summary['fixtures_in_window']}")
    print(f"  Skipped (no lineup):    {summary['fixtures_skipped_no_lineup']}")
    print(f"  Bets placed:            {len(summary['bets_placed'])}")
    print(f"  Bets rejected:          {len(summary['bets_rejected'])}")
    print(f"  Bets filter-skipped:    {len(summary['bets_skipped'])}")
    if summary["bets_placed"]:
        print()
        print("  Bets placed this scan:")
        for b in summary["bets_placed"]:
            tag = "DRY" if b.get("dry_run") else b.get("status", "placed")
            print(f"    [{b['strategy']:18s}] {b['team1'][:20]:20s} vs {b['team2'][:20]:20s}  BACK {b['side_label']:20s} @ {b['market_price']:.3f}  size=${b['stake_usdc']:.2f}  edge=+{b['edge_pp']:.1f}pp  ({tag})")
    if summary["bets_rejected"]:
        print()
        print("  Bets rejected (risk gate or CLOB error):")
        for b in summary["bets_rejected"]:
            print(f"    [{b['strategy']:18s}] {b.get('fixture', '')}  reason={b.get('reason')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
