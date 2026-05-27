#!/usr/bin/env python3
"""Wave 5 Phase 5: Polymarket historical EV backtest.

For each completed cricket match in our holdout (last 6-12 months of
IPL/PSL/T20I/etc. where Polymarket has cricket coverage):

1. Run the V2 simulator at as-of-match-time ELOs.
2. Find the corresponding settled Polymarket markets.
3. For the moneyline (team1 favourite outcome), compute model edge =
   model_prob - market_price at T-30 min before match start.
4. If edge >= threshold, simulate "bet $X with 2% taker fee" and settle
   to the known historical outcome.
5. Aggregate per-market realised P&L, ROI %, max drawdown, # bets placed.

Output: `data/diagnostics/wave_5_polymarket_ev.md` (summary) and
`data/diagnostics/wave_5_polymarket_ev.csv` (per-bet rows). The summary
report's "auto-bet enabled markets" list drives Phase 6's
`BETTING_AUTO_MARKETS` env var.

Usage:

    python scripts/backtest_polymarket_ev.py \\
        --tournament-pattern '%Indian Premier League%' \\
        --since-date 2024-06-01 \\
        --edge-thresholds 3,5,10 \\
        --bet-size 25 \\
        --output-md data/diagnostics/wave_5_polymarket_ev.md
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _lookback_sort_key(label: str) -> float:
    """Sort lookbacks ascending by hours-before-match.

    Labels are 'T-30min', 'T-6h', 'T-3d' etc.
    """
    s = label.replace("T-", "").strip()
    if s.endswith("min"):
        return float(s[:-3]) / 60.0
    if s.endswith("h"):
        return float(s[:-1])
    if s.endswith("d"):
        return float(s[:-1]) * 24.0
    return 0.0

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_connection, get_db_connection, init_franchise_tables  # noqa: E402
from src.integrations.polymarket import PolymarketClient  # noqa: E402
from src.integrations.polymarket.historical import (  # noqa: E402
    fetch_pre_match_prices,
    lookup_settled_markets_for_match,
    market_outcome_label_to_token_id,
)
from src.integrations.odds.polymarket_compare import (  # noqa: E402
    PolymarketComparisonService,
    MARKET_TYPE_MONEYLINE,
    MARKET_TYPE_TOP_BATTER,
    MARKET_TYPE_MOST_SIXES,
    MARKET_TYPE_TOSS_MATCH_DOUBLE,
    POLYMARKET_TAKER_FEE,
    _coerce_list,
)
from src.models.backtest import (  # noqa: E402
    _historical_team_elo,
    _historical_player_elos,
    _team_lineup_from_stats,
    _override_simulator_elos,
    load_holdout_matches,
)
from src.models.market_outputs import derive_polymarket_market_probs  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EVBetRow:
    match_id: int
    match_date: str
    event_name: str
    team1_id: int
    team2_id: int
    market_type: str
    outcome_label: str
    token_id: Optional[str]
    model_prob: float
    market_price_pre: Optional[float]
    edge_pp: Optional[float]
    bet_size_usd: float
    fee_usd: float
    settle_outcome: Optional[float]  # 0.0 or 1.0
    pnl_usd: Optional[float]
    edge_threshold_pp: float
    lookback_label: str  # e.g. "T-30min", "T-24h"
    sim_version: str = "v2"      # v2 or v3
    toss_mode: str = "uncertain"  # uncertain (V2), pinned (V3), marginalised (V3)


@dataclass
class CalibrationRow:
    """Per-match calibration metrics, emitted regardless of Polymarket coverage.

    Used by the wide-sweep analyser to compute Brier/log-loss per (sim_version,
    tournament) so we can compare V3 calibration vs V2 even when no Polymarket
    market exists for the match.
    """
    match_id: int
    match_date: str
    event_name: str
    sim_version: str
    toss_mode: str
    sim_team1_win_prob: float
    actual_team1_won: Optional[int]   # 0 or 1, None if no result
    sim_avg_team1_score: float
    sim_avg_team2_score: float
    actual_team1_total: Optional[int]
    actual_team2_total: Optional[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polymarket historical EV backtest (Wave 5 Phase 5)")
    parser.add_argument("--tournament-pattern", default="%Indian Premier League%")
    parser.add_argument("--format", default="T20", choices=["T20", "ODI"])
    parser.add_argument("--gender", default="male", choices=["male", "female"])
    parser.add_argument("--since-date", default="2024-06-01")
    parser.add_argument("--until-date", default=None)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--n-sims", type=int, default=300)
    parser.add_argument("--edge-thresholds", default="3,5,10",
                        help="Comma-separated edge thresholds in percentage points")
    parser.add_argument("--bet-size", type=float, default=25.0,
                        help="USD bet size per qualifying edge")
    parser.add_argument("--lookback-hours", type=str, default="0.5,1,6,24,48,72",
                        help="Comma-separated hours-before-match values to evaluate as entry prices (default '0.5,1,6,24,48,72')")
    parser.add_argument("--model-version", default="v2", choices=["v2", "v3"],
                        help="Simulator version. v3 enables toss + lineup-stability features.")
    parser.add_argument("--toss-pinned-cutoff-hours", type=float, default=6.0,
                        help=("V3 only: lookbacks <= this many hours pin toss outcome "
                              "(market knows toss); lookbacks > this marginalise over toss "
                              "(toss not yet decided). Ignored for V2."))
    parser.add_argument("--output-calibration-csv", default=None,
                        help=("Optional path for per-match calibration rows (one row per match "
                              "regardless of Polymarket coverage). If unset, derived from --output-csv."))
    parser.add_argument("--output-csv", default="data/diagnostics/wave_5_polymarket_ev.csv")
    parser.add_argument("--output-md", default="data/diagnostics/wave_5_polymarket_ev.md")
    parser.add_argument("--rate-limit-sleep", type=float, default=0.2,
                        help="Seconds to sleep between Polymarket /prices-history calls")
    return parser.parse_args()


def _parse_iso_date(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        d = datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        try:
            d = datetime.fromisoformat(value)
        except ValueError:
            return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d


def _classify_market(market: Dict[str, Any]) -> Optional[str]:
    svc = PolymarketComparisonService.__new__(PolymarketComparisonService)
    return svc._classify_market_type(market)


def _ev_for_outcome(
    model_prob: float,
    market_price: Optional[float],
    settle_outcome: Optional[float],
    bet_size: float,
    fee_pct: float,
) -> Optional[float]:
    """Realised P&L for a $bet_size bet at price `market_price`.

    settle_outcome ∈ {0.0, 1.0}: did this outcome happen?
    Returns net USD (negative for loss, positive for win) AFTER fee.
    """
    if market_price is None or settle_outcome is None:
        return None
    # Pay `market_price` per share to buy 1.0 shares; sell at settle (0 or 1).
    shares = bet_size / max(market_price, 1e-6)
    gross_payout = shares * settle_outcome
    fee = bet_size * fee_pct
    return gross_payout - bet_size - fee


def _summarise_bets(bets: List[EVBetRow], threshold: float) -> Dict[str, Any]:
    """Aggregate metrics for bets at a given edge threshold."""
    qualifying = [
        b for b in bets
        if b.edge_pp is not None and b.edge_pp >= threshold
        and b.pnl_usd is not None
    ]
    n = len(qualifying)
    if n == 0:
        return {
            "edge_threshold_pp": threshold,
            "n_bets": 0,
            "total_pnl_usd": 0.0,
            "roi_pct": 0.0,
            "win_rate": None,
            "max_drawdown_usd": 0.0,
            "n_wins": 0,
            "n_losses": 0,
        }
    total_staked = n * (qualifying[0].bet_size_usd if qualifying else 0.0)
    total_pnl = sum(b.pnl_usd for b in qualifying)
    n_wins = sum(1 for b in qualifying if (b.pnl_usd or 0.0) > 0)
    # Cumulative drawdown (worst peak-to-trough)
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for b in qualifying:
        cum += b.pnl_usd or 0.0
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd
    return {
        "edge_threshold_pp": threshold,
        "n_bets": n,
        "total_pnl_usd": round(total_pnl, 2),
        "total_staked_usd": round(total_staked, 2),
        "roi_pct": round(total_pnl / total_staked * 100.0, 2) if total_staked > 0 else 0.0,
        "win_rate": round(n_wins / n, 3),
        "max_drawdown_usd": round(max_dd, 2),
        "n_wins": n_wins,
        "n_losses": n - n_wins,
    }


def _resolve_settle_outcome_for_label(
    label: str, team1: str, team2: str, market_type: str,
    actual_winner_id: Optional[int], team1_id: int, team2_id: int,
    actual_t1_sixes: Optional[int], actual_t2_sixes: Optional[int],
    actual_t1_top_runs: Optional[int], actual_t2_top_runs: Optional[int],
) -> Optional[float]:
    """Map an outcome label to its actual settled outcome (0.0 or 1.0).

    Uses the token-overlap matcher from PolymarketComparisonService to
    handle franchise-rebrand naming differences (e.g. "Royal Challengers
    Bangalore" vs "Royal Challengers Bengaluru").
    """
    match_t1 = PolymarketComparisonService.label_matches_team(label, team1)
    match_t2 = PolymarketComparisonService.label_matches_team(label, team2)
    norm_label = label.lower().strip()
    is_draw = "tie" in norm_label or "draw" in norm_label

    if market_type == MARKET_TYPE_MONEYLINE:
        if actual_winner_id is None:
            return None
        if match_t1 and not match_t2:
            return 1.0 if actual_winner_id == team1_id else 0.0
        if match_t2 and not match_t1:
            return 1.0 if actual_winner_id == team2_id else 0.0
        return None

    if market_type == MARKET_TYPE_MOST_SIXES:
        if actual_t1_sixes is None or actual_t2_sixes is None:
            return None
        if is_draw:
            return 1.0 if actual_t1_sixes == actual_t2_sixes else 0.0
        if match_t1 and not match_t2:
            return 1.0 if actual_t1_sixes > actual_t2_sixes else 0.0
        if match_t2 and not match_t1:
            return 1.0 if actual_t2_sixes > actual_t1_sixes else 0.0
        return None

    return None  # Top batter + TMD too noisy to settle from a label alone


def main() -> int:
    args = parse_args()

    init_franchise_tables()
    edge_thresholds = [float(x.strip()) for x in args.edge_thresholds.split(",") if x.strip()]
    lookback_hours_list = [float(x.strip()) for x in args.lookback_hours.split(",") if x.strip()]
    poly = PolymarketClient()

    # Lazy import simulator (TF startup cost)
    if args.model_version == "v3":
        from src.models.vectorized_nn_sim_v3 import V3Simulator, V3SimulatorConfig
        simulator = V3Simulator(V3SimulatorConfig(format_type=args.format, gender=args.gender))
    else:
        from src.models.vectorized_nn_sim_v2 import V2Simulator, V2SimulatorConfig
        simulator = V2Simulator(V2SimulatorConfig(format_type=args.format, gender=args.gender))

    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    with get_db_connection() as conn:
        matches = load_holdout_matches(
            conn,
            formats=(args.format,),
            genders=(args.gender,),
            since_date=args.since_date,
            until_date=args.until_date,
            tournament_pattern=args.tournament_pattern,
            limit=args.limit,
        )
    if not matches:
        logger.error("No matches in holdout; aborting.")
        return 1
    logger.info(f"Holdout: {len(matches)} matches; will sim each + price-lookup")

    bets: List[EVBetRow] = []
    calibration_rows: List[CalibrationRow] = []
    skipped_no_market = 0
    skipped_incomplete_lineup = 0
    bets_by_market = {
        MARKET_TYPE_MONEYLINE: [],
        MARKET_TYPE_TOP_BATTER: [],
        MARKET_TYPE_MOST_SIXES: [],
        MARKET_TYPE_TOSS_MATCH_DOUBLE: [],
    }

    # For V3 we run TWO sims per match (toss_pinned + marginalised) so we can
    # measure both:
    #   * pinned: the live entry advantage when toss is publicly known (T-30min)
    #   * marginalised: the honest measurement at early entry (T-1d+)
    # The toss_pinned_cutoff_hours arg controls which sim's market_probs feed
    # which lookback's bet rows in the EV calculation (see inner loop).
    if args.model_version == "v3":
        v3_toss_modes = ["pinned", "marginalised"]
    else:
        v3_toss_modes = ["uncertain"]  # V2 doesn't take toss as a feature

    with get_db_connection() as conn:
        for i, m in enumerate(matches, 1):
            t1 = _team_lineup_from_stats(conn, m.match_id, m.team1_id)
            t2 = _team_lineup_from_stats(conn, m.match_id, m.team2_id)
            if len(t1.batters) < 11 or len(t1.bowlers) < 5 or len(t2.batters) < 11 or len(t2.bowlers) < 5:
                skipped_incomplete_lineup += 1
                continue

            from src.data.franchise_resolver import get_resolver
            resolver = get_resolver()
            canonical1 = resolver.canonical(m.team1_id) or m.team1_id
            canonical2 = resolver.canonical(m.team2_id) or m.team2_id

            team1_elo = _historical_team_elo(conn, canonical1, m.match_type, m.gender, m.date)
            team2_elo = _historical_team_elo(conn, canonical2, m.match_type, m.gender, m.date)
            bat_overrides, bowl_overrides = _historical_player_elos(
                conn,
                set(t1.batters) | set(t1.bowlers) | set(t2.batters) | set(t2.bowlers),
                m.match_type, m.gender, m.date,
            )
            team_overrides = {}
            if team1_elo is not None:
                team_overrides[canonical1] = team1_elo
            if team2_elo is not None:
                team_overrides[canonical2] = team2_elo

            # Run sim once per V3 toss mode (or once total for V2).
            sim_results_by_mode: Dict[str, Dict] = {}
            with _override_simulator_elos(
                simulator,
                team_overrides=team_overrides,
                player_batting_overrides=bat_overrides,
                player_bowling_overrides=bowl_overrides,
            ):
                for toss_mode in v3_toss_modes:
                    sim_kwargs = dict(
                        n_matches=args.n_sims,
                        team1_batter_ids=t1.batters,
                        team1_bowler_ids=t1.bowlers,
                        team2_batter_ids=t2.batters,
                        team2_bowler_ids=t2.bowlers,
                        venue_id=m.venue_id,
                        team1_id=m.team1_id,
                        team2_id=m.team2_id,
                    )
                    if args.model_version == "v3":
                        sim_kwargs["match_date_for_xi"] = m.date.isoformat()
                        if toss_mode == "pinned" and m.winner_id is not None:
                            # Read actual toss from DB
                            cur_t = conn.cursor()
                            cur_t.execute(
                                "SELECT toss_winner_id, toss_decision FROM matches WHERE match_id = ?",
                                (m.match_id,),
                            )
                            tr = cur_t.fetchone()
                            if tr and tr["toss_winner_id"] is not None and tr["toss_decision"]:
                                sim_kwargs["toss_pinned"] = True
                                sim_kwargs["toss_winner_team_id"] = int(tr["toss_winner_id"])
                                sim_kwargs["toss_chose_field"] = (str(tr["toss_decision"]).lower() == "field")
                            else:
                                # Fall back to marginalised if toss data missing
                                sim_kwargs["use_toss"] = True
                        else:
                            # marginalised
                            sim_kwargs["use_toss"] = True
                    try:
                        sim_results_by_mode[toss_mode] = simulator.simulate_matches(**sim_kwargs)
                    except Exception as exc:
                        logger.warning(f"match {m.match_id} ({toss_mode}): simulator error: {exc}")

            if not sim_results_by_mode:
                continue

            # Compute market_probs per toss mode.
            market_probs_by_mode: Dict[str, Dict] = {}
            for toss_mode, sr in sim_results_by_mode.items():
                try:
                    market_probs_by_mode[toss_mode] = derive_polymarket_market_probs(sr)
                except Exception:
                    pass
            if not market_probs_by_mode:
                continue

            # Emit calibration rows per toss mode (regardless of Polymarket coverage).
            actual_t1_won: Optional[int] = None
            if m.winner_id is not None:
                if int(m.winner_id) == int(m.team1_id):
                    actual_t1_won = 1
                elif int(m.winner_id) == int(m.team2_id):
                    actual_t1_won = 0
            for toss_mode, sr in sim_results_by_mode.items():
                calibration_rows.append(CalibrationRow(
                    match_id=m.match_id,
                    match_date=m.date.isoformat(),
                    event_name=m.event_name,
                    sim_version=args.model_version,
                    toss_mode=toss_mode,
                    sim_team1_win_prob=float(sr.get("team1_win_prob", 0.5)),
                    actual_team1_won=actual_t1_won,
                    sim_avg_team1_score=float(sr.get("avg_team1_score", 0.0)),
                    sim_avg_team2_score=float(sr.get("avg_team2_score", 0.0)),
                    actual_team1_total=m.actual_team1_total,
                    actual_team2_total=m.actual_team2_total,
                ))

            # Find settled markets on Polymarket
            cur = conn.cursor()
            cur.execute("SELECT name FROM teams WHERE team_id = ?", (m.team1_id,))
            team1_name_row = cur.fetchone()
            team1_name = team1_name_row["name"] if team1_name_row else f"Team{m.team1_id}"
            cur.execute("SELECT name FROM teams WHERE team_id = ?", (m.team2_id,))
            team2_name_row = cur.fetchone()
            team2_name = team2_name_row["name"] if team2_name_row else f"Team{m.team2_id}"

            try:
                markets = lookup_settled_markets_for_match(
                    poly, team1_name, team2_name, m.date.isoformat(), series=m.event_name
                )
            except Exception as exc:
                logger.debug(f"match {m.match_id}: market lookup failed: {exc}")
                markets = []

            if not markets:
                skipped_no_market += 1
                if i % 10 == 0:
                    logger.info(f"  [{i}/{len(matches)}] no Polymarket coverage so far: skipped={skipped_no_market}")
                continue

            # Look up actual sixes / top runs for settlement
            cur.execute(
                """
                SELECT COUNT(*) AS s FROM deliveries d JOIN innings i ON i.innings_id = d.innings_id
                WHERE i.match_id = ? AND i.batting_team_id = ? AND d.runs_batter = 6
                """,
                (m.match_id, m.team1_id),
            )
            actual_t1_sixes = (cur.fetchone() or {"s": 0})["s"]
            cur.execute(
                """
                SELECT COUNT(*) AS s FROM deliveries d JOIN innings i ON i.innings_id = d.innings_id
                WHERE i.match_id = ? AND i.batting_team_id = ? AND d.runs_batter = 6
                """,
                (m.match_id, m.team2_id),
            )
            actual_t2_sixes = (cur.fetchone() or {"s": 0})["s"]

            # Construct match start datetime (use 18:00 UTC if no exact time)
            match_start = datetime.combine(m.date, datetime.min.time()).replace(tzinfo=timezone.utc) + timedelta(hours=18)

            for market in markets:
                market_type = _classify_market(market)
                if market_type not in (MARKET_TYPE_MONEYLINE, MARKET_TYPE_MOST_SIXES):
                    # Skip top_batter / TMD for now (label settlement is too noisy)
                    continue
                outcomes = _coerce_list(market.get("outcomes")) or []
                token_ids = _coerce_list(market.get("clobTokenIds") or market.get("clobTokenIDs")) or []
                if not outcomes or not token_ids:
                    continue
                for idx, label in enumerate(outcomes):
                    if idx >= len(token_ids):
                        continue
                    token_id = str(token_ids[idx])
                    settle_outcome = _resolve_settle_outcome_for_label(
                        str(label), team1_name, team2_name, market_type,
                        m.winner_id, m.team1_id, m.team2_id,
                        actual_t1_sixes, actual_t2_sixes,
                        None, None,
                    )
                    if settle_outcome is None:
                        continue

                    # Fetch ALL lookback prices in one API call (shared across toss modes)
                    try:
                        prices = fetch_pre_match_prices(
                            poly, token_id, match_start,
                            hours_before_list=tuple(lookback_hours_list),
                        )
                    except Exception:
                        prices = {}
                    if args.rate_limit_sleep > 0:
                        time.sleep(args.rate_limit_sleep)

                    # For each toss mode, compute model_prob for THIS outcome and
                    # emit bet rows per lookback. (V2 has only "uncertain" mode;
                    # V3 emits both "pinned" and "marginalised".)
                    for toss_mode, mp in market_probs_by_mode.items():
                        svc = PolymarketComparisonService.__new__(PolymarketComparisonService)
                        model_pct = svc._model_outcome_prob_for_label(
                            market_type, str(label), team1_name, team2_name, mp
                        )
                        if model_pct is None:
                            continue
                        model_prob = model_pct / 100.0

                        for hours_before in lookback_hours_list:
                            from src.integrations.polymarket.historical import _label_for_hours
                            lookback_label = _label_for_hours(hours_before)
                            market_price = prices.get(lookback_label)
                            if market_price is None:
                                continue

                            edge_pp = (model_prob - market_price) * 100.0
                            pnl = _ev_for_outcome(
                                model_prob, market_price, settle_outcome,
                                bet_size=args.bet_size, fee_pct=POLYMARKET_TAKER_FEE,
                            )

                            bet = EVBetRow(
                                match_id=m.match_id,
                                match_date=m.date.isoformat(),
                                event_name=m.event_name,
                                team1_id=m.team1_id,
                                team2_id=m.team2_id,
                                market_type=market_type,
                                outcome_label=str(label),
                                token_id=token_id,
                                model_prob=round(model_prob, 4),
                                market_price_pre=round(market_price, 4),
                                edge_pp=round(edge_pp, 2),
                                bet_size_usd=args.bet_size,
                                fee_usd=round(args.bet_size * POLYMARKET_TAKER_FEE, 4),
                                settle_outcome=float(settle_outcome),
                                pnl_usd=round(pnl, 2) if pnl is not None else None,
                                edge_threshold_pp=min(edge_thresholds),
                                lookback_label=lookback_label,
                                sim_version=args.model_version,
                                toss_mode=toss_mode,
                            )
                            bets.append(bet)
                            bets_by_market[market_type].append(bet)

            if i % 10 == 0:
                logger.info(f"  [{i}/{len(matches)}] bets so far={len(bets)} ({len(bets_by_market[MARKET_TYPE_MONEYLINE])} ML, {len(bets_by_market[MARKET_TYPE_MOST_SIXES])} sixes)")

    # Persist bet rows
    with output_csv.open("w", newline="") as fp:
        # Use the dataclass field order even when no rows; keeps schema stable.
        sample_row = EVBetRow(
            match_id=0, match_date="", event_name="", team1_id=0, team2_id=0,
            market_type="", outcome_label="", token_id=None, model_prob=0.0,
            market_price_pre=None, edge_pp=None, bet_size_usd=0.0, fee_usd=0.0,
            settle_outcome=None, pnl_usd=None, edge_threshold_pp=0.0,
            lookback_label="",
        )
        writer = csv.DictWriter(fp, fieldnames=list(asdict(sample_row).keys()))
        writer.writeheader()
        for b in bets:
            writer.writerow(asdict(b))
    logger.info(f"Wrote {len(bets)} bet rows to {output_csv}")

    # Persist calibration rows (one per match per toss_mode, regardless of Polymarket coverage)
    calib_csv_path = (
        Path(args.output_calibration_csv)
        if args.output_calibration_csv
        else output_csv.parent / (output_csv.stem + "_calibration.csv")
    )
    calib_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with calib_csv_path.open("w", newline="") as fp:
        sample_calib = CalibrationRow(
            match_id=0, match_date="", event_name="",
            sim_version="", toss_mode="",
            sim_team1_win_prob=0.0, actual_team1_won=None,
            sim_avg_team1_score=0.0, sim_avg_team2_score=0.0,
            actual_team1_total=None, actual_team2_total=None,
        )
        writer = csv.DictWriter(fp, fieldnames=list(asdict(sample_calib).keys()))
        writer.writeheader()
        for c in calibration_rows:
            writer.writerow(asdict(c))
    logger.info(f"Wrote {len(calibration_rows)} calibration rows to {calib_csv_path}")

    # Build summary report
    lines = [
        "# Wave 5 Polymarket historical EV backtest\n\n",
        f"Auto-generated by `scripts/backtest_polymarket_ev.py` on {datetime.now(timezone.utc).isoformat()}\n\n",
        "## Inputs\n\n",
        f"- Tournament pattern: `{args.tournament_pattern}`\n",
        f"- Format / gender: {args.format} / {args.gender}\n",
        f"- Date window: {args.since_date} -> {args.until_date or 'today'}\n",
        f"- Matches considered: {len(matches)}\n",
        f"- Skipped (incomplete lineup): {skipped_incomplete_lineup}\n",
        f"- Skipped (no Polymarket market match): {skipped_no_market}\n",
        f"- Bet size: ${args.bet_size:.0f}; taker fee: {POLYMARKET_TAKER_FEE * 100:.1f}%\n",
        f"- Entry timings (lookback hours): {args.lookback_hours}\n",
        f"- Total bet rows captured (ALL edges): {len(bets)}\n\n",
    ]

    auto_eligible: List[Tuple[str, str]] = []
    for market_type in (MARKET_TYPE_MONEYLINE, MARKET_TYPE_MOST_SIXES):
        market_bets = bets_by_market[market_type]
        lines.append(f"## {market_type}\n\n")
        if not market_bets:
            lines.append(f"_No bet rows for {market_type}._\n\n")
            continue
        # Per-lookback table (this is the new headline view).
        lookback_labels = sorted(
            {b.lookback_label for b in market_bets},
            key=lambda lab: _lookback_sort_key(lab),
        )
        lines.append("### ROI by entry timing (lookback) x edge threshold\n\n")
        lines.append("| Lookback | Threshold | n_bets | Win rate | Total P&L | ROI | Max DD |\n")
        lines.append("|---|---|---|---|---|---|---|\n")
        best_for_market = None
        for lookback in lookback_labels:
            lb_bets = [b for b in market_bets if b.lookback_label == lookback]
            for thr in edge_thresholds:
                summary = _summarise_bets(lb_bets, thr)
                if summary["n_bets"] == 0:
                    continue
                lines.append(
                    f"| {lookback} | ≥{thr}pp | {summary['n_bets']} | "
                    f"{summary['win_rate']} | ${summary['total_pnl_usd']} | "
                    f"{summary['roi_pct']}% | ${summary['max_drawdown_usd']} |\n"
                )
                # Track best AUTO-eligible (n>=50 AND roi>0) combination
                if (
                    summary["n_bets"] >= 50 and summary["roi_pct"] > 0
                    and (best_for_market is None or summary["roi_pct"] > best_for_market[1]["roi_pct"])
                ):
                    best_for_market = ((lookback, thr), summary)
        if best_for_market is not None:
            (lookback, thr), summary = best_for_market
            auto_eligible.append((market_type, f"{lookback}@{thr}pp"))
            lines.append(
                f"\n**Eligible for AUTO mode**: {market_type} at "
                f"lookback={lookback} edge>={thr}pp -> "
                f"{summary['roi_pct']}% ROI over {summary['n_bets']} bets.\n\n"
            )
        else:
            lines.append(
                f"\n**MANUAL-only**: no (lookback, threshold) combination "
                f"meets the AUTO gate (n>=50 AND ROI>0).\n\n"
            )

    lines.append("## Phase 6 envelope recommendations\n\n")
    if auto_eligible:
        markets = sorted({m for m, _ in auto_eligible})
        lines.append(f"`BETTING_AUTO_MARKETS={','.join(markets)}`\n\n")
        lines.append("Best (market, lookback@threshold) combinations:\n")
        for market, combo in auto_eligible:
            lines.append(f"- {market}: {combo}\n")
        lines.append("\n")
    else:
        lines.append(
            "No (market, lookback, threshold) combination meets the AUTO gate (n>=50 AND ROI>0). "
            "Recommend `BETTING_MODE=MANUAL` for first week of operation.\n\n"
        )

    lines.append("\n## Caveats\n\n")
    lines.append(
        "- Polymarket cricket coverage really only ramped up mid-2024; expect "
        "many older matches to skip with 'no market match'.\n"
        "- Top batter / Toss Match Double markets are skipped in this iteration "
        "since outcome-label settlement requires per-batter / per-toss data we "
        "don't aggregate from labels alone. Add explicit per-outcome settlement "
        "in a future iteration once we have stable label-to-player resolution.\n"
        "- This backtest assumes we ALWAYS get the bet at the price observed "
        "T-N min before match start. In practice, slippage on a $25 fill in a "
        "$80-volume market can be material; haircut realised ROI by 2-5pp before "
        "trusting it for live betting envelope sizing.\n"
        "- Settlement is computed from our DB's known-outcome data, not from "
        "Polymarket's settled price. If the market resolution disagrees with our "
        "DB (e.g. on rain-affected matches with DLS), the row will be wrong.\n"
    )

    output_md.write_text("".join(lines))
    logger.info(f"Wrote {output_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
