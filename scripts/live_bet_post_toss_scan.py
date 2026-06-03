#!/usr/bin/env python3
"""Wave 5.8: post-toss re-scan for ONE fixture, placing LIVE bets.

The live counterpart of `paper_bet_post_toss_scan.py`. Once the toss is
decided (and optionally a confirmed XI is published), re-run the
simulators for that fixture with:
    - V3 in PINNED toss mode (the actual toss winner + their decision)
    - V2 with no toss change (V2 doesn't model toss)
    - Optional XI overrides for one or both teams

For each LIVE-enabled strategy (BETTING_LIVE_STRATEGIES in .env), check
the post-toss edge against the strategy threshold and route through
`place_bet()` if it qualifies. Bets are tagged phase='post_toss' for
analytic separation from the pre-toss bet (if any). The standard
risk-gate enforces per-strategy / per-bet / per-day caps before any
order hits CLOB.

Idempotent: re-running the same fixture does NOT double-bet. Dedup key
mirrors live_bet_scan's `_already_bet_live` (bet_kind='real',
strategy_label, fixture_key, market_id, side_label, phase='post_toss').

Mode/kill-switch:
    - Exits early if BETTING_MODE != AUTO or BETTING_KILL_SWITCH=1
    - This script is intended to be invoked by the auto-toss daemon
      (paper_bet_auto_post_toss.py --also-live) or by the live-betting
      UI's "Toss + Re-scan" button. Manual MANUAL-mode usage isn't
      wired up yet because there's no per-bet approval UI in the
      live betting page (yet); the surfaced edges flow already covers
      that case for pre-toss bets.

Usage:
    venv311/bin/python scripts/live_bet_post_toss_scan.py \\
        --fixture-key crint-pak-zwe-2026-05-06 \\
        --toss-winner team1 \\
        --chose-to bat \\
        [--team1-xi 12,34,56,...] [--team2-xi 78,90,...] \\
        [--dry-run]
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

from src.data.database import get_connection, get_db_connection, get_active_model_snapshot
from src.integrations.polymarket import PolymarketClient
from src.integrations.polymarket.bet_placement import place_bet
from src.integrations.polymarket.paper_inputs import (
    get_recent_xi, get_default_venue_for_team,
)
from src.integrations.polymarket.paper_strategies import get_enabled_strategies
from src.integrations.polymarket.upcoming import (
    find_upcoming_cricket_events, attach_db_team_ids,
)

# Reuse the live-bet-scan helpers so sizing / dedup behaviour stays
# identical between the pre-toss scan and the post-toss scan.
from scripts.live_bet_scan import (
    _moneyline_outcome_for_team,
    get_live_strategy_bankroll,
    live_scaled_kelly_stake,
)
from src.integrations.polymarket.sizing import effective_kelly_mult, get_team_tier

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
    bets so we can later detect XI changes between scans. Same algorithm
    as the paper post-toss script so the signature comparison is
    apples-to-apples between paper and live."""
    payload = json.dumps({
        "t1_bat": t1_bat, "t1_bowl": t1_bowl,
        "t2_bat": t2_bat, "t2_bowl": t2_bowl,
    }, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def _load_fixture(fixture_key: str) -> Optional[Dict[str, Any]]:
    """Find the upcoming Polymarket fixture matching this key.
    Uses include_started=True so a fixture whose toss has already
    happened (and is therefore minutes past kickoff) is still found."""
    c = PolymarketClient()
    events = find_upcoming_cricket_events(c, hours_ahead=168, include_started=True)
    mapped = attach_db_team_ids(events)
    for ev in mapped:
        if ev["fixture_key"] == fixture_key:
            return ev
    return None


def _already_live_bet_for_phase(
    conn,
    strategy_label: str,
    fixture_key: str,
    market_id: str,
    side_label: str,
    phase: str = "post_toss",
) -> bool:
    """Has this strategy already placed a non-errored LIVE bet for this
    fixture / market / side / phase? Mirrors _already_bet_live in
    live_bet_scan.py but parameterises the phase so the caller can
    explicitly check 'post_toss'."""
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
    return cur.fetchone() is not None


def post_toss_live_scan(
    fixture_key: str,
    toss_winner_side: str,        # 'team1' or 'team2'
    chose_to: str,                 # 'bat' or 'field'
    team1_xi_override: Optional[List[int]] = None,
    team2_xi_override: Optional[List[int]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run a post-toss re-scan for one fixture and place LIVE bets via
    `place_bet()`. Returns a summary dict with bets placed / skipped /
    rejected.
    """
    from config import BETTING_CONFIG

    summary: Dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "fixture_key": fixture_key,
        "toss_winner_side": toss_winner_side,
        "chose_to": chose_to,
        "fixture_loaded": False,
        "bets_placed": [],
        "bets_skipped": [],
        "bets_rejected": [],
    }

    # Capture active model snapshot once per scan for bet_ledger stamping.
    model_snapshot = get_active_model_snapshot()

    # Mode gate up-front so cron-style logs make it obvious when betting
    # is disabled. Same enforcement as live_bet_scan.scan_and_place_live_bets.
    mode = str(BETTING_CONFIG.get("mode", "OFF")).upper()
    kill_switch = bool(BETTING_CONFIG.get("kill_switch", False))
    if kill_switch:
        msg = "BETTING_KILL_SWITCH=1 — kill switch engaged; nothing placed"
        logger.warning(msg)
        summary["exit_reason"] = "kill_switch_engaged"
        return summary
    if mode == "OFF":
        msg = "BETTING_MODE=OFF — nothing placed"
        logger.warning(msg)
        summary["exit_reason"] = "betting_mode_off"
        return summary
    if mode != "AUTO":
        # Post-toss scans are spawned as background subprocesses with no
        # per-bet UI; only AUTO can fire them. MANUAL mode users should
        # run a manual /api/bulk-predict + surfaced-edges flow instead.
        msg = f"BETTING_MODE={mode} — post-toss live scan only acts in AUTO mode"
        logger.warning(msg)
        summary["exit_reason"] = f"betting_mode_{mode.lower()}"
        return summary

    live_strategy_names = set(BETTING_CONFIG.get("live_strategies", []) or [])
    if not live_strategy_names:
        logger.warning("BETTING_LIVE_STRATEGIES empty — no strategies are live; nothing placed")
        summary["exit_reason"] = "no_live_strategies"
        return summary

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

    if toss_winner_side not in ("team1", "team2"):
        raise ValueError(f"toss_winner_side must be 'team1' or 'team2', got {toss_winner_side!r}")
    if chose_to not in ("bat", "field"):
        raise ValueError(f"chose_to must be 'bat' or 'field', got {chose_to!r}")
    toss_winner_team_id = fix["team1_id"] if toss_winner_side == "team1" else fix["team2_id"]
    toss_chose_field = (chose_to == "field")
    summary["toss_winner_team_id"] = toss_winner_team_id
    summary["toss_winner_db_name"] = fix["team1_db_name"] if toss_winner_side == "team1" else fix["team2_db_name"]
    summary["toss_chose_to"] = chose_to

    # Build XI inputs - prefer overrides, fall back to recent-XI proxy.
    # This logic is intentionally identical to paper_bet_post_toss_scan
    # so live and paper sims see the same lineup state.
    fmt = fix["format"]
    gender = fix["gender"]
    t1, t2 = fix["team1_id"], fix["team2_id"]

    def _derive_bowlers(xi: List[int]) -> List[int]:
        if len(xi) >= 11:
            return xi[6:11]
        if len(xi) >= 5:
            return xi[-5:]
        return []

    with get_db_connection() as conn:
        recent_t1_bat, recent_t1_bowl = get_recent_xi(conn, t1, fmt, gender)
        recent_t2_bat, recent_t2_bowl = get_recent_xi(conn, t2, fmt, gender)
        venue_id = get_default_venue_for_team(conn, t1, fmt, gender)

        if team1_xi_override and len(team1_xi_override) >= 11:
            t1_bat = team1_xi_override[:11]
            t1_bowl = _derive_bowlers(team1_xi_override)
            if len(t1_bowl) < 5:
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

    enabled_strats = [s for s in get_enabled_strategies() if s.name in live_strategy_names]
    if not enabled_strats:
        msg = "No live-enabled strategies match BETTING_LIVE_STRATEGIES whitelist; nothing to do"
        logger.warning(msg)
        summary["exit_reason"] = "no_matching_live_strategies"
        return summary

    needs_v2 = any(s.model_version in ("v2", "consensus") for s in enabled_strats)
    needs_v3 = any(s.model_version in ("v3", "consensus") for s in enabled_strats)

    logger.info(
        f"  LIVE post-toss sim: {fix['team1_db_name']} vs {fix['team2_db_name']}  "
        f"strategies: {[s.name for s in enabled_strats]}"
    )
    logger.info(f"    Toss: {summary['toss_winner_db_name']} won, chose to {chose_to}")
    logger.info(f"    XI signature: {xi_sig}")

    v2_t1_prob: Optional[float] = None
    v3_t1_prob: Optional[float] = None
    if needs_v2:
        logger.info(f"    Loading V2 simulator (format={fmt}, gender={gender})...")
        from src.models.vectorized_nn_sim_v2 import V2Simulator, V2SimulatorConfig
        sim_v2 = V2Simulator(V2SimulatorConfig(format_type=fmt, gender=gender))
        # V2 doesn't pin toss; we adjust toss_field_prob to bias the
        # toss decision toward what actually happened. This is the same
        # convention paper_bet_post_toss_scan uses so the V2 number is
        # comparable across paper and live.
        v2_result = sim_v2.simulate_matches(
            300, t1_bat, t1_bowl, t2_bat, t2_bowl,
            venue_id=venue_id, team1_id=t1, team2_id=t2,
            use_toss=True, toss_field_prob=(0.65 if not toss_chose_field else 0.35),
            seed=42,
        )
        v2_t1_prob = float(v2_result["team1_win_prob"])

    if needs_v3:
        logger.info(f"    Loading V3 simulator (PINNED toss, format={fmt}, gender={gender})...")
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

    summary["v2_team1_win_prob"] = round(v2_t1_prob, 4) if v2_t1_prob is not None else None
    summary["v3_team1_win_prob"] = round(v3_t1_prob, 4) if v3_t1_prob is not None else None

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
    if v2_t1_prob is not None:
        logger.info(f"    V2 P(team1) = {v2_t1_prob*100:.1f}%")
    if v3_t1_prob is not None:
        logger.info(f"    V3 P(team1) = {v3_t1_prob*100:.1f}%  (toss pinned)")
    logger.info(f"    Market: {fix['team1_db_name']}={market_price_t1:.3f}  "
                f"{fix['team2_db_name']}={market_price_t2:.3f}")

    kickoff = fix.get("scheduled_start_estimate")
    kickoff_iso = kickoff.isoformat() if kickoff else None

    for strat in enabled_strats:
        # Post-toss eligibility gate: V2-based strategies cannot condition on
        # the toss (V2 sim ignores toss kwargs), so their "edge" post-toss is
        # illusory — the market has already priced in the toss outcome. Only
        # strategies explicitly marked post_toss_eligible=True may place real
        # money here. Paper bets are handled by paper_bet_post_toss_scan.py
        # which has its own (unrestricted) eligibility logic.
        if not strat.post_toss_eligible:
            summary["bets_skipped"].append({
                "strategy": strat.name,
                "reason": "post_toss_eligible=False (strategy uses V2 which is toss-blind)",
            })
            continue

        # Filter checks (format, gender, tournament). Same precedence
        # as live_bet_scan / paper_bet_post_toss_scan.
        if strat.enabled_formats and fmt not in strat.enabled_formats:
            summary["bets_skipped"].append({"strategy": strat.name, "reason": f"format {fmt} not enabled"})
            continue
        if strat.enabled_genders and gender not in strat.enabled_genders:
            summary["bets_skipped"].append({"strategy": strat.name, "reason": f"gender {gender} not enabled"})
            continue
        if strat.enabled_tournament_prefixes and fix.get("tournament_prefix") not in strat.enabled_tournament_prefixes:
            summary["bets_skipped"].append({"strategy": strat.name, "reason": "tournament not enabled"})
            continue

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
                summary["bets_skipped"].append({"strategy": strat.name, "reason": "V2/V3 disagree"})
                continue
            t1_pred = (v2_t1_prob + v3_t1_prob) / 2.0
        else:
            continue
        t2_pred = 1.0 - t1_pred

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
        # Model probability bounds: exclude coin-flip zone.
        if strat.min_model_prob is not None and model_prob < strat.min_model_prob:
            summary["bets_skipped"].append({
                "strategy": strat.name,
                "reason": f"model_prob {model_prob:.3f} < min {strat.min_model_prob}",
                "side_label": side_label,
            })
            continue
        if strat.max_model_prob is not None and model_prob > strat.max_model_prob:
            summary["bets_skipped"].append({
                "strategy": strat.name,
                "reason": f"model_prob {model_prob:.3f} > max {strat.max_model_prob}",
                "side_label": side_label,
            })
            continue
        # Fill-gap guard: if model_prob vs market_price gap is too large the market
        # has moved on information our model doesn't have (toss impact not fully
        # captured, late XI news, pitch reports). Block to avoid chasing ghost edge.
        if strat.max_model_minus_fill_pp is not None:
            gap_pp = (model_prob - market_price) * 100
            if gap_pp > strat.max_model_minus_fill_pp:
                summary["bets_skipped"].append({
                    "strategy": strat.name,
                    "reason": (
                        f"fill-gap {gap_pp:.1f}pp > max {strat.max_model_minus_fill_pp}pp "
                        f"(market moved on info model doesn't have)"
                    ),
                    "side_label": side_label, "model_prob": round(model_prob, 4),
                    "market_price": market_price,
                })
                continue

        with get_db_connection() as conn:
            if _already_live_bet_for_phase(
                conn, strat.name, fixture_key, ml.get("market_id") or "",
                side_label or "", phase="post_toss",
            ):
                summary["bets_skipped"].append({"strategy": strat.name, "reason": "already-placed-this-phase"})
                continue
            bankroll_now = get_live_strategy_bankroll(strat.name, conn)
            if bankroll_now <= 0:
                summary["bets_skipped"].append({"strategy": strat.name, "reason": "bankroll-exhausted"})
                continue
            # Low-data-league throttle: associate-nation internationals are
            # sized at the (smaller) associate Kelly multiplier.
            kelly_override = effective_kelly_mult(
                strat.kelly_mult,
                fix.get("tournament_prefix"),
                get_team_tier(conn, fix.get("team1_id")),
                get_team_tier(conn, fix.get("team2_id")),
            )
            stake = live_scaled_kelly_stake(
                model_prob, market_price, bankroll_now, strat, kelly_override
            )
            if stake <= 0:
                summary["bets_skipped"].append({"strategy": strat.name, "reason": "kelly-stake-zero"})
                continue

        if dry_run:
            logger.info(
                f"    [{strat.name}] DRY-RUN POST-TOSS LIVE: BACK {side_label} @ {market_price:.3f} "
                f"size=${stake:.2f}  edge=+{edge_pp:.1f}pp  bank=${bankroll_now:.2f}"
            )
            summary["bets_placed"].append({
                "bet_id": -1, "strategy": strat.name, "side_label": side_label,
                "model_prob": round(model_prob, 4), "market_price": market_price,
                "edge_pp": round(edge_pp, 2), "stake_usdc": round(stake, 2),
                "bankroll_at_proposal": round(bankroll_now, 2), "dry_run": True,
            })
            continue

        try:
            result = place_bet(
                fixture_key=fixture_key,
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
                phase="post_toss",
                kickoff_at=kickoff_iso,
                xi_signature=xi_sig,
                toss_winner_team_id=toss_winner_team_id,
                toss_chose_to=chose_to,
                model_snapshot=model_snapshot,
            )
        except Exception as exc:
            logger.error(f"    [{strat.name}] place_bet raised: {exc}")
            summary["bets_rejected"].append({
                "strategy": strat.name, "reason": f"exception: {exc}",
            })
            continue

        if result.get("success"):
            logger.info(
                f"    [{strat.name}] LIVE POST-TOSS BET #{result['bet_id']}: BACK {side_label} @ {market_price:.3f} "
                f"size=${stake:.2f}  edge=+{edge_pp:.1f}pp  bank=${bankroll_now:.2f}  status={result['status']}"
            )
            summary["bets_placed"].append({
                "bet_id": result["bet_id"], "strategy": strat.name, "side_label": side_label,
                "model_prob": round(model_prob, 4), "market_price": market_price,
                "edge_pp": round(edge_pp, 2), "stake_usdc": round(stake, 2),
                "bankroll_at_proposal": round(bankroll_now, 2), "status": result["status"],
            })
        else:
            logger.warning(
                f"    [{strat.name}] REJECTED: {result.get('reason')} "
                f"(size=${stake:.2f}, edge=+{edge_pp:.1f}pp)"
            )
            summary["bets_rejected"].append({
                "strategy": strat.name, "reason": result.get("reason"),
                "status": result.get("status"), "bet_id": result.get("bet_id"),
            })

    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="LIVE post-toss re-scan for one fixture")
    parser.add_argument("--fixture-key", required=True)
    parser.add_argument("--toss-winner", required=True, choices=["team1", "team2"])
    parser.add_argument("--chose-to", required=True, choices=["bat", "field"])
    parser.add_argument("--team1-xi", help="Optional comma-separated player_ids (11 batters or 16+ to include bowlers)")
    parser.add_argument("--team2-xi", help="Optional comma-separated player_ids")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    summary = post_toss_live_scan(
        fixture_key=args.fixture_key,
        toss_winner_side=args.toss_winner,
        chose_to=args.chose_to,
        team1_xi_override=_parse_xi_arg(args.team1_xi),
        team2_xi_override=_parse_xi_arg(args.team2_xi),
        dry_run=args.dry_run,
    )

    print()
    print("=" * 70)
    print("LIVE POST-TOSS SCAN SUMMARY")
    print("=" * 70)
    if "exit_reason" in summary:
        print(f"  Exited early: {summary['exit_reason']}")
        return 0
    if "error" in summary:
        print(f"  ERROR: {summary['error']}")
        return 1
    print(f"  Fixture: {summary['team1']} vs {summary['team2']}")
    print(f"  Toss: {summary['toss_winner_db_name']} chose to {summary['toss_chose_to']}")
    if summary.get("v2_team1_win_prob") is not None:
        print(f"  V2 P(team1) = {summary['v2_team1_win_prob']*100:.1f}%")
    if summary.get("v3_team1_win_prob") is not None:
        print(f"  V3 P(team1) = {summary['v3_team1_win_prob']*100:.1f}%  (with toss pinned)")
    print(f"  Market price t1={summary['market_price_t1']:.3f}  t2={summary['market_price_t2']:.3f}")
    print(f"  XI signature: {summary['xi_signature']}")
    print(f"  Bets placed:    {len(summary['bets_placed'])}")
    for b in summary["bets_placed"]:
        tag = "DRY" if b.get("dry_run") else b.get("status", "placed")
        print(f"    [{b['strategy']:18s}] BACK {b['side_label']:25s} @ {b['market_price']:.3f}  "
              f"size=${b['stake_usdc']:.2f}  edge=+{b['edge_pp']:.1f}pp  ({tag})")
    print(f"  Bets skipped:   {len(summary['bets_skipped'])}")
    print(f"  Bets rejected:  {len(summary['bets_rejected'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
