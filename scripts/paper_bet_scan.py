#!/usr/bin/env python3
"""Wave 5.7: scan upcoming Polymarket cricket fixtures and place paper bets.

For each open cricket fixture in the next N hours:
  1. Map team labels to DB team_ids
  2. Build recent-XI inputs for each team
  3. Run V2 (and V3 if any strategy needs it) Monte Carlo sims
  4. For each enabled strategy, check the moneyline:
        - tournament + format + gender filters pass
        - market_price within [min_market_price, max_market_price]
        - now() is within [lookback_hours_min, lookback_hours] before kickoff
        - model_prob - market_price >= min_edge_pp / 100
  5. If qualifying: insert a 'paper' row in bet_ledger with status='filled'
     (paper bets fill instantly at the observed market price).

Idempotent: running the same scan twice does NOT double-bet. Dedupe key is
(strategy_label, fixture_key, market_id, side_label).

Usage:
    venv311/bin/python scripts/paper_bet_scan.py [--hours-ahead 96] [--dry-run]
                                                 [--strategies ALL|name1,name2]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_connection, get_active_model_snapshot
from src.integrations.polymarket import PolymarketClient
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
    STRATEGIES,
    PaperStrategy,
    get_enabled_strategies,
    kelly_stake_usdc,
    get_strategy_bankroll,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _strategy_in_window(strategy: PaperStrategy, hours_to_kickoff: float) -> bool:
    """Is the current time within this strategy's entry window?"""
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
    """Find the outcome dict in the moneyline market that corresponds to `team_db_name`.

    Layered matching:
      1. Token-overlap (existing label_matches_team) - handles franchise rebrands.
      2. Bidirectional substring on word-prefixes >=5 chars - handles cases like
         'Rawalpindiz' (DB) vs 'Rawalpindi Pindiz' (Polymarket).
    """
    if not moneyline_market or not team_db_name:
        return None
    from src.integrations.odds.polymarket_compare import PolymarketComparisonService

    outcomes = moneyline_market.get("outcomes", [])

    # Pass 1: token overlap
    for outcome in outcomes:
        if PolymarketComparisonService.label_matches_team(outcome.get("label", ""), team_db_name):
            return outcome

    # Pass 2: substring fallback. Look for ANY 5+ char common prefix between
    # the DB team's distinctive tokens and the label's distinctive tokens.
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
                # Either is a 5+ char prefix of the other? Catches
                # 'rawalpindi' <-> 'rawalpindiz', 'bengaluru' <-> 'bengalu...'
                if (db_t.startswith(lbl_t[:5]) and lbl_t.startswith(db_t[:5])):
                    return outcome
    return None


def _already_bet(conn, strategy_label: str, fixture_key: str, market_id: str,
                 side_label: str, phase: str = "pre_toss") -> bool:
    """Idempotency check. A strategy can place at most ONE bet per
    (fixture, market, side, phase). The phase axis lets pre-toss and
    post-toss runs each place a bet on the same side at different prices.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT bet_id FROM bet_ledger
        WHERE bet_kind = 'paper'
          AND strategy_label = ?
          AND fixture_key = ?
          AND polymarket_market_id = ?
          AND side_label = ?
          AND COALESCE(phase, 'pre_toss') = ?
        LIMIT 1
        """,
        (strategy_label, fixture_key, market_id, side_label, phase),
    )
    return cur.fetchone() is not None


def _strategy_has_any_bet_on_fixture(conn, strategy_label: str,
                                     fixture_key: str,
                                     phase: str = "pre_toss") -> bool:
    """Has THIS strategy placed any bet on this fixture in this phase?
    Used for the skip-already-fully-bet optimization. Phase-scoped so
    pre-toss bets don't suppress post-toss re-scans."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT 1 FROM bet_ledger
        WHERE bet_kind = 'paper'
          AND strategy_label = ?
          AND fixture_key = ?
          AND COALESCE(phase, 'pre_toss') = ?
        LIMIT 1
        """,
        (strategy_label, fixture_key, phase),
    )
    return cur.fetchone() is not None


def _insert_paper_bet(
    conn,
    strategy: PaperStrategy,
    fixture: Dict[str, Any],
    market: Dict[str, Any],
    side_label: str,
    side_token_id: Optional[str],
    model_prob: float,
    market_price: float,
    edge_pp: float,
    stake_usdc: float,
    bankroll_at_proposal: float,
    phase: str = "pre_toss",
    xi_signature: Optional[str] = None,
    toss_winner_team_id: Optional[int] = None,
    toss_chose_to: Optional[str] = None,
    model_snapshot: Optional[str] = None,
) -> int:
    """Insert a paper bet at status='filled' (paper fills instantly)."""
    now_iso = datetime.now(timezone.utc).isoformat()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO bet_ledger (
            proposed_at, placed_at, filled_at,
            match_id, fixture_key, market_type,
            polymarket_market_id, polymarket_token_id,
            side_label, model_prob, market_price_at_proposal, edge_pp,
            side, size_usdc, fees_estimated_usdc,
            fill_price, fill_size_usdc,
            status, mode, bet_kind, strategy_label,
            bankroll_at_proposal,
            phase, xi_signature, toss_winner_team_id, toss_chose_to,
            model_snapshot
        ) VALUES (
            ?, ?, ?,
            ?, ?, ?,
            ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?,
            ?, ?, ?, ?,
            ?,
            ?, ?, ?, ?,
            ?
        )
        """,
        (
            now_iso, now_iso, now_iso,
            None, fixture["fixture_key"], "moneyline",
            market.get("market_id"), side_token_id,
            side_label, model_prob, market_price, edge_pp,
            "BUY", stake_usdc, stake_usdc * 0.02,
            market_price, stake_usdc,
            "filled", "manual", "paper", strategy.name,
            bankroll_at_proposal,
            phase, xi_signature, toss_winner_team_id, toss_chose_to,
            model_snapshot,
        ),
    )
    return cur.lastrowid


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


def scan_and_place_paper_bets(
    hours_ahead: float = 96.0,
    strategy_filter: Optional[List[str]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Main scan logic. Returns a summary dict for the daily runner."""
    summary: Dict[str, Any] = {
        "scan_started_at": datetime.now(timezone.utc).isoformat(),
        "hours_ahead": hours_ahead,
        "fixtures_seen": 0,
        "fixtures_in_window": 0,
        "fixtures_skipped_no_lineup": 0,
        "bets_placed": [],
        "bets_skipped": [],
    }

    enabled_strats = get_enabled_strategies()
    if strategy_filter:
        enabled_strats = [s for s in enabled_strats if s.name in strategy_filter]
    if not enabled_strats:
        logger.warning("No enabled strategies match filter; nothing to do")
        return summary

    # Capture active model snapshot once per scan run so every bet in this
    # batch has the same snapshot string for post-hoc version grouping.
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

    # Lazy-load simulators (we cache one per format/gender combo)
    sim_v2: Dict[str, Any] = {}
    sim_v3: Dict[str, Any] = {}
    if needs_v2:
        logger.info("  V2 simulator will be lazy-loaded per format/gender")
    if needs_v3:
        logger.info("  V3 simulator will be lazy-loaded per format/gender")

    now = datetime.now(timezone.utc)
    for fix in fixtures:
        if not (fix.get("team1_id") and fix.get("team2_id")):
            continue
        if not fix.get("moneyline"):
            continue
        kickoff = fix["scheduled_start_estimate"]
        hours_to_kickoff = (kickoff - now).total_seconds() / 3600.0

        # Get/init simulators for this fixture's format/gender combo
        fmt = fix["format"]
        gender = fix["gender"]
        sg_key = f"{fmt}_{gender}"

        v2_t1_prob: Optional[float] = None
        v3_t1_prob: Optional[float] = None
        # Only run sims if at least one strategy *might* match this fixture
        eligible = [s for s in enabled_strats
                    if _strategy_filters_match_fixture(s, fix)
                    and _strategy_in_window(s, hours_to_kickoff)]
        if not eligible:
            continue
        summary["fixtures_in_window"] += 1

        # OPTIMIZATION: skip the (expensive) sim if every eligible strategy
        # has already placed a bet on this fixture. Each strategy can place
        # exactly one bet per fixture (idempotent), and the simulator is
        # deterministic given the same XI/venue inputs - so a re-sim would
        # just produce the same probs and the same bets we already have.
        # Trade-off: we miss the rare case where a strategy that previously
        # had edge < threshold now has edge >= threshold due to market price
        # drift. That's a small price for a 5x-10x speedup on re-scans.
        with get_connection() as conn:
            all_eligible_have_bets = all(
                _strategy_has_any_bet_on_fixture(conn, s.name, fix["fixture_key"])
                for s in eligible
            )
        if all_eligible_have_bets:
            logger.info(
                f"  [skip-resim] {fix['team1_db_name']} vs {fix['team2_db_name']} "
                f"(T-{hours_to_kickoff:.1f}h) - all {len(eligible)} eligible strategies "
                f"already placed bets, no need to re-simulate"
            )
            summary["fixtures_in_window"] -= 1  # didn't actually exercise sim path
            summary.setdefault("fixtures_skipped_resim", 0)
            summary["fixtures_skipped_resim"] += 1
            continue

        # Run sims - regardless of dry-run, we want to log what the model said.
        # Only the DB-write at the end is gated by dry_run.
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

        # Identify each side's outcome+price+team_name
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

        # For each eligible strategy, decide if and which side to back
        for strat in eligible:
            # Get the model probabilities this strategy uses
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
                # Both must agree on which side to back
                v2_back_t1 = (v2_t1_prob - market_price_t1) > 0
                v3_back_t1 = (v3_t1_prob - market_price_t1) > 0
                if v2_back_t1 != v3_back_t1:
                    logger.info(f"    [{strat.name}] V2/V3 disagree on side; skip")
                    continue
                t1_pred = (v2_t1_prob + v3_t1_prob) / 2.0
            else:
                continue
            t2_pred = 1.0 - t1_pred

            # Compute edge per side
            edge_t1_pp = (t1_pred - market_price_t1) * 100
            edge_t2_pp = (t2_pred - market_price_t2) * 100

            # Pick the side with bigger positive edge
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

            # Apply edge / price gate
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

            # Idempotent dedup
            with get_connection() as conn:
                if _already_bet(conn, strat.name, fix["fixture_key"],
                                ml.get("market_id") or "", side_label or ""):
                    summary["bets_skipped"].append({
                        "strategy": strat.name, "fixture": fix["fixture_key"],
                        "reason": "already-placed",
                    })
                    continue

                # Stake using current bankroll
                bankroll_now = get_strategy_bankroll(strat.name, conn)
                if bankroll_now <= 0:
                    summary["bets_skipped"].append({
                        "strategy": strat.name, "fixture": fix["fixture_key"],
                        "reason": "bankroll-exhausted",
                    })
                    continue
                stake = kelly_stake_usdc(model_prob, market_price, bankroll_now, strat)
                if stake <= 0:
                    summary["bets_skipped"].append({
                        "strategy": strat.name, "fixture": fix["fixture_key"],
                        "reason": "kelly-stake-zero",
                    })
                    continue

                # Place
                if dry_run:
                    bet_id = -1
                    logger.info(
                        f"    [{strat.name}] DRY-RUN: BACK {side_label} @ {market_price:.3f} "
                        f"size=${stake:.2f}  edge=+{edge_pp:.1f}pp  bank=${bankroll_now:.2f}"
                    )
                else:
                    bet_id = _insert_paper_bet(
                        conn, strat, fix, ml, side_label, side_token_id,
                        model_prob, market_price, edge_pp, stake, bankroll_now,
                        model_snapshot=model_snapshot,
                    )
                    conn.commit()
                    logger.info(
                        f"    [{strat.name}] PAPER BET #{bet_id}: BACK {side_label} @ {market_price:.3f} "
                        f"size=${stake:.2f}  edge=+{edge_pp:.1f}pp  bank=${bankroll_now:.2f}"
                    )

                summary["bets_placed"].append({
                    "bet_id": bet_id,
                    "strategy": strat.name,
                    "fixture": fix["fixture_key"],
                    "team1": fix["team1_db_name"],
                    "team2": fix["team2_db_name"],
                    "side_label": side_label,
                    "model_prob": model_prob,
                    "market_price": market_price,
                    "edge_pp": edge_pp,
                    "stake_usdc": stake,
                    "bankroll_at_proposal": bankroll_now,
                    "kickoff_utc": kickoff.isoformat(),
                })

    summary["scan_finished_at"] = datetime.now(timezone.utc).isoformat()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan upcoming cricket fixtures for paper bets")
    parser.add_argument("--hours-ahead", type=float, default=96.0, help="Look ahead window (default: 96h)")
    parser.add_argument("--strategies", default="ALL", help="Comma-separated strategy names or ALL")
    parser.add_argument("--dry-run", action="store_true", help="Run sims but don't write to DB")
    args = parser.parse_args()

    strategy_filter = None
    if args.strategies.upper() != "ALL":
        strategy_filter = [s.strip() for s in args.strategies.split(",") if s.strip()]

    summary = scan_and_place_paper_bets(
        hours_ahead=args.hours_ahead,
        strategy_filter=strategy_filter,
        dry_run=args.dry_run,
    )

    print()
    print("=" * 70)
    print(f"PAPER BET SCAN SUMMARY")
    print("=" * 70)
    print(f"  Fixtures seen:          {summary['fixtures_seen']}")
    print(f"  In any strategy window: {summary['fixtures_in_window']}")
    print(f"  Skipped (no lineup):    {summary['fixtures_skipped_no_lineup']}")
    print(f"  Bets placed:            {len(summary['bets_placed'])}")
    print(f"  Bets skipped:           {len(summary['bets_skipped'])}")
    if summary["bets_placed"]:
        print()
        print("  Bets placed this scan:")
        for b in summary["bets_placed"]:
            print(f"    [{b['strategy']:18s}] {b['team1'][:20]:20s} vs {b['team2'][:20]:20s}  BACK {b['side_label']:20s} @ {b['market_price']:.3f}  size=${b['stake_usdc']:.2f}  edge=+{b['edge_pp']:.1f}pp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
