#!/usr/bin/env python3
"""Wave 5.7b: post-toss re-scan for ONE fixture.

Once the toss is decided (and optionally a confirmed XI is published),
re-run the simulators for that fixture with:
    - V3 in PINNED toss mode (the actual toss winner + their decision)
    - V2 with no toss change (V2 doesn't model toss)
    - Optional XI overrides for one or both teams

For each strategy, place a NEW paper bet tagged phase='post_toss' if the
post-toss edge crosses the strategy's threshold. Pre-toss bets stay
exactly as-is - the post-toss bet is an additional one tagged for
analytic comparison.

Usage:
    venv311/bin/python scripts/paper_bet_post_toss_scan.py \\
        --fixture-key cricipl-del-roy-2026-04-27 \\
        --toss-winner team1 \\
        --chose-to bat \\
        [--team1-xi 12,34,56,...] [--team2-xi 78,90,...] \\
        [--dry-run]

Where:
    --toss-winner: 'team1' or 'team2' (referring to the slug-derived ordering)
    --chose-to:    'bat' or 'field' (what the toss winner chose)
    --team1-xi / --team2-xi: optional comma-separated player_ids for the
        confirmed XI (overrides the recent-XI proxy). Pass first 11 ids
        in batting order, then up to 5 main bowler ids.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_connection
from src.integrations.polymarket import PolymarketClient
from src.integrations.polymarket.upcoming import (
    find_upcoming_cricket_events, attach_db_team_ids,
)
from src.integrations.polymarket.paper_inputs import (
    get_recent_xi, get_default_venue_for_team,
)
from src.integrations.polymarket.paper_strategies import (
    STRATEGIES, kelly_stake_usdc, get_strategy_bankroll,
)
from src.integrations.odds.polymarket_compare import PolymarketComparisonService

from scripts.paper_bet_scan import (
    _already_bet, _insert_paper_bet, _moneyline_outcome_for_team,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _parse_xi_arg(arg: Optional[str]) -> Optional[List[int]]:
    if not arg:
        return None
    out = []
    for tok in arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            raise SystemExit(f"--xi value must be comma-separated integers, got: {tok!r}")
    return out


def _xi_signature(t1_bat: List[int], t1_bowl: List[int],
                  t2_bat: List[int], t2_bowl: List[int]) -> str:
    """Stable hash of the four lineup arrays. Stored alongside post-toss
    bets so we can later detect XI changes between scans."""
    payload = json.dumps({
        "t1_bat": t1_bat, "t1_bowl": t1_bowl,
        "t2_bat": t2_bat, "t2_bowl": t2_bowl,
    }, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def _load_fixture(fixture_key: str) -> Optional[Dict[str, Any]]:
    """Find the upcoming Polymarket fixture matching this key."""
    c = PolymarketClient()
    events = find_upcoming_cricket_events(c, hours_ahead=168, include_started=True)
    mapped = attach_db_team_ids(events)
    for ev in mapped:
        if ev["fixture_key"] == fixture_key:
            return ev
    return None


def post_toss_scan(
    fixture_key: str,
    toss_winner_side: str,        # 'team1' or 'team2'
    chose_to: str,                 # 'bat' or 'field'
    team1_xi_override: Optional[List[int]] = None,
    team2_xi_override: Optional[List[int]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run a post-toss re-scan for one fixture. Returns a summary dict."""
    summary: Dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "fixture_key": fixture_key,
        "toss_winner_side": toss_winner_side,
        "chose_to": chose_to,
        "fixture_loaded": False,
        "bets_placed": [],
        "bets_skipped": [],
    }

    fix = _load_fixture(fixture_key)
    if fix is None:
        msg = f"Could not find fixture {fixture_key} on Polymarket"
        logger.error(msg)
        summary["error"] = msg
        return summary
    summary["fixture_loaded"] = True
    summary["team1"] = fix.get("team1_db_name")
    summary["team2"] = fix.get("team2_db_name")
    if not (fix.get("team1_id") and fix.get("team2_id")):
        msg = "Fixture teams not mapped to DB - cannot run sim"
        logger.error(msg)
        summary["error"] = msg
        return summary
    if not fix.get("moneyline"):
        msg = "Fixture has no moneyline market on Polymarket"
        logger.error(msg)
        summary["error"] = msg
        return summary

    # Resolve toss inputs
    if toss_winner_side not in ("team1", "team2"):
        raise ValueError(f"toss_winner_side must be 'team1' or 'team2', got {toss_winner_side!r}")
    if chose_to not in ("bat", "field"):
        raise ValueError(f"chose_to must be 'bat' or 'field', got {chose_to!r}")
    toss_winner_team_id = fix["team1_id"] if toss_winner_side == "team1" else fix["team2_id"]
    toss_chose_field = (chose_to == "field")
    summary["toss_winner_team_id"] = toss_winner_team_id
    summary["toss_winner_db_name"] = fix["team1_db_name"] if toss_winner_side == "team1" else fix["team2_db_name"]
    summary["toss_chose_to"] = chose_to

    # Build XI inputs - prefer overrides, fall back to recent-XI proxy
    fmt = fix["format"]
    gender = fix["gender"]
    t1, t2 = fix["team1_id"], fix["team2_id"]

    def _derive_bowlers(xi: List[int]) -> List[int]:
        """Heuristic: if XI override has >= 5 entries, take the last 5 as
        the bowlers (positions 7-11 in an XI typically include all-rounders
        and specialist bowlers). Caller already validated XI has 11+ batters."""
        if len(xi) >= 11:
            return xi[6:11]   # batting positions 7-11 = mostly bowlers
        if len(xi) >= 5:
            return xi[-5:]    # last 5 of whatever we have
        return []

    with get_connection() as conn:
        # Always pull recent-XI as a fallback (cheap, avoids the "10/11 names
        # matched -> insufficient lineup" trap when CREX returned 11 but our
        # name resolver only resolved 10 of them).
        recent_t1_bat, recent_t1_bowl = get_recent_xi(conn, t1, fmt, gender)
        recent_t2_bat, recent_t2_bowl = get_recent_xi(conn, t2, fmt, gender)
        venue_id = get_default_venue_for_team(conn, t1, fmt, gender)

        # Apply XI override only if we have FULL 11; otherwise log and fall
        # through to the recent-XI proxy. Better to use a slightly stale
        # lineup than to crash.
        if team1_xi_override and len(team1_xi_override) >= 11:
            t1_bat = team1_xi_override[:11]
            t1_bowl = _derive_bowlers(team1_xi_override)
            if len(t1_bowl) < 5:
                # Pad with recent bowlers not already in the XI batters
                seen = set(t1_bat)
                for p in recent_t1_bowl:
                    if p not in seen and len(t1_bowl) < 5:
                        t1_bowl.append(p)
                        seen.add(p)
            logger.info(f"  Using team1 XI override: {len(t1_bat)} batters, {len(t1_bowl)} bowlers")
        else:
            if team1_xi_override:
                logger.warning(
                    f"  team1 XI override only has {len(team1_xi_override)} names "
                    f"(<11); falling back to recent-XI proxy"
                )
            t1_bat, t1_bowl = recent_t1_bat, recent_t1_bowl

        if team2_xi_override and len(team2_xi_override) >= 11:
            t2_bat = team2_xi_override[:11]
            t2_bowl = _derive_bowlers(team2_xi_override)
            if len(t2_bowl) < 5:
                seen = set(t2_bat)
                for p in recent_t2_bowl:
                    if p not in seen and len(t2_bowl) < 5:
                        t2_bowl.append(p)
                        seen.add(p)
            logger.info(f"  Using team2 XI override: {len(t2_bat)} batters, {len(t2_bowl)} bowlers")
        else:
            if team2_xi_override:
                logger.warning(
                    f"  team2 XI override only has {len(team2_xi_override)} names "
                    f"(<11); falling back to recent-XI proxy"
                )
            t2_bat, t2_bowl = recent_t2_bat, recent_t2_bowl

    if len(t1_bat) < 11 or len(t1_bowl) < 5 or len(t2_bat) < 11 or len(t2_bowl) < 5:
        msg = (f"Insufficient lineup data even after fallback: t1 bat={len(t1_bat)} bowl={len(t1_bowl)}, "
               f"t2 bat={len(t2_bat)} bowl={len(t2_bowl)}")
        logger.error(msg)
        summary["error"] = msg
        return summary

    xi_sig = _xi_signature(t1_bat, t1_bowl, t2_bat, t2_bowl)
    summary["xi_signature"] = xi_sig

    # ---- Run V2 (no toss model) and V3 (PINNED toss) ----
    logger.info(f"  Running post-toss sim: {fix['team1_db_name']} vs {fix['team2_db_name']}")
    logger.info(f"    Toss: {summary['toss_winner_db_name']} won, chose to {chose_to}")
    logger.info(f"    XI signature: {xi_sig}")
    logger.info(f"    Loading V2 simulator...")
    from src.models.vectorized_nn_sim_v2 import V2Simulator, V2SimulatorConfig
    sim_v2 = V2Simulator(V2SimulatorConfig(format_type=fmt, gender=gender))
    v2_result = sim_v2.simulate_matches(
        300, t1_bat, t1_bowl, t2_bat, t2_bowl,
        venue_id=venue_id, team1_id=t1, team2_id=t2,
        use_toss=True, toss_field_prob=(0.65 if not toss_chose_field else 0.35),
        seed=42,
    )
    v2_t1_prob = float(v2_result["team1_win_prob"])

    logger.info(f"    Loading V3 simulator (PINNED toss mode)...")
    from src.models.vectorized_nn_sim_v3 import V3Simulator, V3SimulatorConfig
    sim_v3 = V3Simulator(V3SimulatorConfig(format_type=fmt, gender=gender))
    v3_result = sim_v3.simulate_matches(
        300, t1_bat, t1_bowl, t2_bat, t2_bowl,
        venue_id=venue_id, team1_id=t1, team2_id=t2,
        toss_pinned=True,
        toss_winner_team_id=toss_winner_team_id,
        toss_chose_field=toss_chose_field,
        seed=42,
    )
    v3_t1_prob = float(v3_result["team1_win_prob"])

    summary["v2_team1_win_prob"] = round(v2_t1_prob, 4)
    summary["v3_team1_win_prob"] = round(v3_t1_prob, 4)

    # ---- Compute edges and place bets ----
    ml = fix["moneyline"]
    t1_outcome = _moneyline_outcome_for_team(ml, fix["team1_db_name"])
    t2_outcome = _moneyline_outcome_for_team(ml, fix["team2_db_name"])
    if t1_outcome is None or t2_outcome is None:
        msg = "Could not map moneyline outcomes to team names"
        logger.error(msg)
        summary["error"] = msg
        return summary
    market_price_t1 = float(t1_outcome.get("last_price") or 0)
    market_price_t2 = float(t2_outcome.get("last_price") or 0)
    if market_price_t1 <= 0 or market_price_t2 <= 0:
        msg = "Missing market price"
        logger.error(msg)
        summary["error"] = msg
        return summary

    summary["market_price_t1"] = market_price_t1
    summary["market_price_t2"] = market_price_t2
    logger.info(f"    V2 P(team1)={v2_t1_prob*100:.1f}%   V3 P(team1)={v3_t1_prob*100:.1f}%")
    logger.info(f"    Market: {fix['team1_db_name']}={market_price_t1:.3f}  "
                f"{fix['team2_db_name']}={market_price_t2:.3f}")

    enabled_strats = [s for s in STRATEGIES if s.enabled]
    for strat in enabled_strats:
        # Filter checks (format, gender, tournament)
        if strat.enabled_formats and fmt not in strat.enabled_formats:
            summary["bets_skipped"].append({"strategy": strat.name, "reason": f"format {fmt} not enabled"})
            continue
        if strat.enabled_genders and gender not in strat.enabled_genders:
            summary["bets_skipped"].append({"strategy": strat.name, "reason": f"gender {gender} not enabled"})
            continue
        if strat.enabled_tournament_prefixes and fix.get("tournament_prefix") not in strat.enabled_tournament_prefixes:
            summary["bets_skipped"].append({"strategy": strat.name, "reason": "tournament not enabled"})
            continue

        # Pick which model probabilities to use
        if strat.model_version == "v2":
            t1_pred = v2_t1_prob
        elif strat.model_version == "v3":
            t1_pred = v3_t1_prob
        elif strat.model_version == "consensus":
            v2_back_t1 = (v2_t1_prob - market_price_t1) > 0
            v3_back_t1 = (v3_t1_prob - market_price_t1) > 0
            if v2_back_t1 != v3_back_t1:
                summary["bets_skipped"].append({"strategy": strat.name, "reason": "V2/V3 disagree"})
                continue
            t1_pred = (v2_t1_prob + v3_t1_prob) / 2.0
        else:
            continue
        t2_pred = 1.0 - t1_pred

        # Decide which side to back
        edge_t1 = (t1_pred - market_price_t1) * 100
        edge_t2 = (t2_pred - market_price_t2) * 100
        if edge_t1 >= edge_t2:
            side_label = t1_outcome.get("label")
            side_token_id = t1_outcome.get("token_id")
            model_prob = t1_pred
            market_price = market_price_t1
            edge_pp = edge_t1
        else:
            side_label = t2_outcome.get("label")
            side_token_id = t2_outcome.get("token_id")
            model_prob = t2_pred
            market_price = market_price_t2
            edge_pp = edge_t2

        if edge_pp < strat.min_edge_pp:
            summary["bets_skipped"].append({
                "strategy": strat.name,
                "reason": f"edge {edge_pp:.1f}pp < {strat.min_edge_pp:.1f}pp threshold",
                "side_label": side_label, "model_prob": round(model_prob, 4),
                "market_price": market_price,
            })
            continue
        if not (strat.min_market_price <= market_price <= strat.max_market_price):
            summary["bets_skipped"].append({
                "strategy": strat.name,
                "reason": f"market_price {market_price:.3f} outside lottery filter",
            })
            continue

        # Place
        with get_connection() as conn:
            if _already_bet(conn, strat.name, fixture_key, ml.get("market_id") or "",
                            side_label or "", phase="post_toss"):
                summary["bets_skipped"].append({"strategy": strat.name, "reason": "already-placed-this-phase"})
                continue
            bankroll_now = get_strategy_bankroll(strat.name, conn)
            stake = kelly_stake_usdc(model_prob, market_price, bankroll_now, strat)
            if stake <= 0:
                summary["bets_skipped"].append({"strategy": strat.name, "reason": "kelly-stake-zero"})
                continue

            if dry_run:
                bet_id = -1
                logger.info(
                    f"    [{strat.name}] DRY-RUN POST-TOSS: BACK {side_label} @ {market_price:.3f} "
                    f"size=${stake:.2f}  edge=+{edge_pp:.1f}pp"
                )
            else:
                bet_id = _insert_paper_bet(
                    conn, strat, fix, ml, side_label, side_token_id,
                    model_prob, market_price, edge_pp, stake, bankroll_now,
                    phase="post_toss",
                    xi_signature=xi_sig,
                    toss_winner_team_id=toss_winner_team_id,
                    toss_chose_to=chose_to,
                )
                conn.commit()
                logger.info(
                    f"    [{strat.name}] POST-TOSS BET #{bet_id}: BACK {side_label} @ {market_price:.3f} "
                    f"size=${stake:.2f}  edge=+{edge_pp:.1f}pp  bank=${bankroll_now:.2f}"
                )
            summary["bets_placed"].append({
                "bet_id": bet_id, "strategy": strat.name, "side_label": side_label,
                "model_prob": round(model_prob, 4), "market_price": market_price,
                "edge_pp": round(edge_pp, 2), "stake_usdc": round(stake, 2),
                "bankroll_at_proposal": round(bankroll_now, 2),
            })

    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-toss re-scan for one fixture")
    parser.add_argument("--fixture-key", required=True)
    parser.add_argument("--toss-winner", required=True, choices=["team1", "team2"])
    parser.add_argument("--chose-to", required=True, choices=["bat", "field"])
    parser.add_argument("--team1-xi", help="Optional comma-separated player_ids (11 batters or 16+ to include bowlers)")
    parser.add_argument("--team2-xi", help="Optional comma-separated player_ids")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    summary = post_toss_scan(
        fixture_key=args.fixture_key,
        toss_winner_side=args.toss_winner,
        chose_to=args.chose_to,
        team1_xi_override=_parse_xi_arg(args.team1_xi),
        team2_xi_override=_parse_xi_arg(args.team2_xi),
        dry_run=args.dry_run,
    )

    print()
    print("=" * 70)
    print("POST-TOSS SCAN SUMMARY")
    print("=" * 70)
    if "error" in summary:
        print(f"  ERROR: {summary['error']}")
        return 1
    print(f"  Fixture: {summary['team1']} vs {summary['team2']}")
    print(f"  Toss: {summary['toss_winner_db_name']} chose to {summary['toss_chose_to']}")
    print(f"  V2 P(team1) = {summary['v2_team1_win_prob']*100:.1f}%")
    print(f"  V3 P(team1) = {summary['v3_team1_win_prob']*100:.1f}%  (with toss pinned)")
    print(f"  Market price t1={summary['market_price_t1']:.3f}  t2={summary['market_price_t2']:.3f}")
    print(f"  XI signature: {summary['xi_signature']}")
    print(f"  Bets placed:  {len(summary['bets_placed'])}")
    for b in summary["bets_placed"]:
        print(f"    [{b['strategy']:18s}] BACK {b['side_label']:25s} @ {b['market_price']:.3f}  "
              f"size=${b['stake_usdc']:.2f}  edge=+{b['edge_pp']:.1f}pp")
    print(f"  Bets skipped: {len(summary['bets_skipped'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
