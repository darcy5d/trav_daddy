"""
Match-level backtest harness for the Monte Carlo simulator (NEXT_OVERHAUL item 16).

For a held-out set of completed matches, this module:
  1. Loads each match's actual XI from player_match_stats.
  2. Looks up team & player ELO ratings AS OF THE MATCH DATE
     (calculator_v3.get_team_rating / get_player_rating with as_of_date) so
     the simulator doesn't peek at ratings derived from this or later matches.
  3. Temporarily overrides the simulator's in-memory ELO caches with those
     historical values, runs N simulations, then restores the live caches.
  4. Compares simulated win probability and innings totals against the actual
     match outcome and computes Brier score, log loss, calibration deciles,
     and MAE of total runs / margin.

Documented v1 limitations (revisit in v2):
  - Per-batter / per-bowler outcome distributions
    (`src/features/player_distributions.py`) are aggregated across the
    player's full career, including the holdout matches themselves. The leak
    is small per individual match (one match's contribution to a career-long
    histogram) but real. A strict v2 would recompute distributions excluding
    each holdout match's ball-by-ball rows; the cost is roughly N times the
    work and is held back until v1 has shipped.
  - Bowling line-up uses the 5 actual highest-overs bowlers; impact-sub
    selection is not modelled.
  - Toss is simulated independently per Monte Carlo iteration rather than
    fixed to the actual toss outcome (matches the live Bulk Predict path).
"""

from __future__ import annotations

import logging
import math
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class MatchSpec:
    """The minimum we need to set up a simulator call from a historical match."""
    match_id: int
    date: date
    match_type: str
    gender: str
    team1_id: int
    team2_id: int
    venue_id: Optional[int]
    event_name: str
    winner_id: Optional[int]
    win_type: Optional[str]
    win_margin: Optional[int]
    actual_team1_total: Optional[int]
    actual_team2_total: Optional[int]


@dataclass
class TeamLineup:
    batters: List[int]   # 11 player_ids (best-effort: see _team_lineup_from_stats)
    bowlers: List[int]   # 5 player_ids


@dataclass
class BacktestRow:
    match_id: int
    date: str
    match_type: str
    gender: str
    event_name: str
    team1_id: int
    team2_id: int
    canonical_team1_id: int
    canonical_team2_id: int
    team1_elo_used: float
    team2_elo_used: float
    sim_team1_win_prob: float
    sim_avg_team1_score: float
    sim_avg_team2_score: float
    actual_winner_id: Optional[int]
    actual_team1_total: Optional[int]
    actual_team2_total: Optional[int]
    # Derived quick-look fields:
    team1_won: Optional[int]   # 1 / 0 / None (no-result)
    margin_runs: Optional[float]   # signed: positive when team1 outscored team2
    sim_margin_runs: float
    score_mae_team1: Optional[float]
    score_mae_team2: Optional[float]
    dist_quality_overall_pct: Optional[float]


# ============================================================================
# Holdout selection
# ============================================================================


def load_holdout_matches(
    conn: sqlite3.Connection,
    *,
    formats: Iterable[str] = ("T20",),
    genders: Iterable[str] = ("male",),
    since_date: Optional[str] = None,
    until_date: Optional[str] = None,
    tournament_pattern: Optional[str] = None,
    limit: int = 200,
    require_winner: bool = True,
) -> List[MatchSpec]:
    """Pull a chronological holdout of completed matches.

    Defaults to the most recent T20 men's matches. `tournament_pattern` is a
    SQL LIKE pattern matched against `matches.event_name`.
    """
    fmts = tuple(formats)
    gens = tuple(genders)
    where = ["m.match_type IN ({})".format(",".join(["?"] * len(fmts)))]
    args: List = list(fmts)
    where.append("m.gender IN ({})".format(",".join(["?"] * len(gens))))
    args.extend(gens)
    if since_date:
        where.append("m.date >= ?")
        args.append(since_date)
    if until_date:
        where.append("m.date <= ?")
        args.append(until_date)
    if tournament_pattern:
        where.append("m.event_name LIKE ?")
        args.append(tournament_pattern)
    if require_winner:
        where.append("m.winner_id IS NOT NULL")

    sql = f"""
        SELECT
            m.match_id, m.date, m.match_type, m.gender,
            m.team1_id, m.team2_id, m.venue_id, COALESCE(m.event_name, '') AS event_name,
            m.winner_id, m.win_type, m.win_margin,
            (SELECT total_runs FROM innings i1 WHERE i1.match_id = m.match_id AND i1.batting_team_id = m.team1_id ORDER BY i1.innings_number LIMIT 1) AS team1_total,
            (SELECT total_runs FROM innings i2 WHERE i2.match_id = m.match_id AND i2.batting_team_id = m.team2_id ORDER BY i2.innings_number LIMIT 1) AS team2_total
        FROM matches m
        WHERE {' AND '.join(where)}
        ORDER BY m.date DESC, m.match_id DESC
        LIMIT ?
    """
    args.append(limit)
    cur = conn.cursor()
    cur.execute(sql, args)
    rows = cur.fetchall()

    out: List[MatchSpec] = []
    for r in rows:
        d = r["date"]
        if isinstance(d, str):
            d = datetime.strptime(d, "%Y-%m-%d").date()
        out.append(
            MatchSpec(
                match_id=r["match_id"],
                date=d,
                match_type=r["match_type"],
                gender=r["gender"],
                team1_id=r["team1_id"],
                team2_id=r["team2_id"],
                venue_id=r["venue_id"],
                event_name=r["event_name"],
                winner_id=r["winner_id"],
                win_type=r["win_type"],
                win_margin=r["win_margin"],
                actual_team1_total=r["team1_total"],
                actual_team2_total=r["team2_total"],
            )
        )
    return out


# ============================================================================
# XI extraction
# ============================================================================


def _team_lineup_from_stats(
    conn: sqlite3.Connection, match_id: int, team_id: int
) -> TeamLineup:
    """Best-effort XI reconstruction from player_match_stats.

    Batters: 11 picked by `batting_position ASC NULLS LAST` then
    `balls_faced DESC` (the "actually batted" set first, then anyone who
    didn't bat). Bowlers: 5 picked by `overs_bowled DESC`.

    IPL 2023+ matches have 12 stats rows per side (11 + impact sub); we keep
    the same 11/5 shape the live simulator expects.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT player_id, batting_position, balls_faced, runs_scored, overs_bowled, wickets_taken
        FROM player_match_stats
        WHERE match_id = ? AND team_id = ?
        """,
        (match_id, team_id),
    )
    rows = [dict(r) for r in cur.fetchall()]
    if not rows:
        return TeamLineup(batters=[], bowlers=[])

    def _bat_key(r):
        # NULL batting_position sorts last; among NULLs, more balls_faced sorts higher.
        pos = r["batting_position"] if r["batting_position"] is not None else 99
        return (pos, -(r["balls_faced"] or 0))

    batters = [r["player_id"] for r in sorted(rows, key=_bat_key)][:11]
    # Pad if fewer than 11 unique stats rows (rare but possible for incomplete records).
    if len(batters) < 11 and rows:
        seen = set(batters)
        for r in rows:
            if r["player_id"] not in seen:
                batters.append(r["player_id"])
                seen.add(r["player_id"])
            if len(batters) >= 11:
                break

    bowlers_sorted = sorted(rows, key=lambda r: -(r["overs_bowled"] or 0))
    bowlers = [r["player_id"] for r in bowlers_sorted if (r["overs_bowled"] or 0) > 0][:5]
    if len(bowlers) < 5:
        # Fall back to filling with non-batter positions
        seen = set(bowlers)
        for r in rows:
            if r["player_id"] not in seen:
                bowlers.append(r["player_id"])
                seen.add(r["player_id"])
            if len(bowlers) >= 5:
                break

    return TeamLineup(batters=batters[:11], bowlers=bowlers[:5])


# ============================================================================
# As-of-date ELO override
# ============================================================================


def _historical_team_elo(
    conn: sqlite3.Connection,
    canonical_team_id: int,
    match_format: str,
    gender: str,
    as_of: date,
) -> Optional[float]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT elo FROM team_elo_history
        WHERE team_id = ? AND format = ? AND gender = ? AND date < ?
          AND NOT is_monthly_snapshot
        ORDER BY date DESC, elo_id DESC
        LIMIT 1
        """,
        (canonical_team_id, match_format, gender, as_of),
    )
    row = cur.fetchone()
    return row["elo"] if row else None


def _historical_player_elos(
    conn: sqlite3.Connection,
    player_ids: Iterable[int],
    match_format: str,
    gender: str,
    as_of: date,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Return ({player_id: batting_elo}, {player_id: bowling_elo}) as of `as_of`."""
    bat: Dict[int, float] = {}
    bowl: Dict[int, float] = {}
    cur = conn.cursor()
    for pid in player_ids:
        # One query per player keeps the SQL portable across sqlite versions.
        cur.execute(
            """
            SELECT batting_elo, bowling_elo
            FROM player_elo_history
            WHERE player_id = ? AND format = ? AND gender = ? AND date < ?
              AND NOT is_monthly_snapshot
            ORDER BY date DESC, elo_id DESC
            LIMIT 1
            """,
            (pid, match_format, gender, as_of),
        )
        row = cur.fetchone()
        if not row:
            continue
        if row["batting_elo"] is not None:
            bat[int(pid)] = row["batting_elo"]
        if row["bowling_elo"] is not None:
            bowl[int(pid)] = row["bowling_elo"]
    return bat, bowl


@contextmanager
def _override_simulator_elos(
    simulator,
    *,
    team_overrides: Dict[int, float],
    player_batting_overrides: Dict[int, float],
    player_bowling_overrides: Dict[int, float],
):
    """Temporarily replace the simulator's in-memory ELO caches.

    The VectorizedNNSimulator caches `team_current_elo`,
    `player_batting_elo`, and `player_bowling_elo` in process at construction
    time. For backtest we want as-of-date ratings instead. We swap in a
    layered dict (overrides first, fall back to live) for the duration of one
    match and restore on exit so the simulator stays usable across matches
    and after the backtest finishes.
    """
    orig_team = simulator.team_current_elo
    orig_bat = simulator.player_batting_elo
    orig_bowl = simulator.player_bowling_elo

    layered_team = dict(orig_team)
    layered_team.update(team_overrides)
    layered_bat = dict(orig_bat)
    layered_bat.update(player_batting_overrides)
    layered_bowl = dict(orig_bowl)
    layered_bowl.update(player_bowling_overrides)

    simulator.team_current_elo = layered_team
    simulator.player_batting_elo = layered_bat
    simulator.player_bowling_elo = layered_bowl
    try:
        yield
    finally:
        simulator.team_current_elo = orig_team
        simulator.player_batting_elo = orig_bat
        simulator.player_bowling_elo = orig_bowl


# ============================================================================
# Per-match simulation
# ============================================================================


def simulate_match(
    conn: sqlite3.Connection,
    simulator,
    match: MatchSpec,
    *,
    n_sims: int = 1000,
    use_toss: bool = True,
    toss_field_prob: float = 0.65,
) -> Optional[BacktestRow]:
    """Run a single backtest simulation for one historical match."""
    from src.data.franchise_resolver import get_resolver
    resolver = get_resolver()

    # XI from actual stats. If either side is missing players, the match is
    # not safely backtestable - log + skip.
    t1 = _team_lineup_from_stats(conn, match.match_id, match.team1_id)
    t2 = _team_lineup_from_stats(conn, match.match_id, match.team2_id)
    if len(t1.batters) < 11 or len(t1.bowlers) < 5 or len(t2.batters) < 11 or len(t2.bowlers) < 5:
        logger.debug(
            f"match {match.match_id}: incomplete lineup "
            f"(t1 bat={len(t1.batters)} bowl={len(t1.bowlers)}, t2 bat={len(t2.batters)} bowl={len(t2.bowlers)}) - skipping"
        )
        return None

    canonical1 = resolver.canonical(match.team1_id) or match.team1_id
    canonical2 = resolver.canonical(match.team2_id) or match.team2_id

    # As-of-date ELOs.
    team1_elo = _historical_team_elo(conn, canonical1, match.match_type, match.gender, match.date)
    team2_elo = _historical_team_elo(conn, canonical2, match.match_type, match.gender, match.date)

    bat_overrides, bowl_overrides = _historical_player_elos(
        conn,
        set(t1.batters) | set(t1.bowlers) | set(t2.batters) | set(t2.bowlers),
        match.match_type,
        match.gender,
        match.date,
    )

    team_overrides: Dict[int, float] = {}
    if team1_elo is not None:
        team_overrides[canonical1] = team1_elo
    if team2_elo is not None:
        team_overrides[canonical2] = team2_elo

    with _override_simulator_elos(
        simulator,
        team_overrides=team_overrides,
        player_batting_overrides=bat_overrides,
        player_bowling_overrides=bowl_overrides,
    ):
        results = simulator.simulate_matches(
            n_sims,
            t1.batters,
            t1.bowlers,
            t2.batters,
            t2.bowlers,
            venue_id=match.venue_id,
            use_toss=use_toss,
            toss_field_prob=toss_field_prob,
            team1_id=match.team1_id,
            team2_id=match.team2_id,
        )

    sim_t1_prob = float(results["team1_win_prob"])
    sim_avg_t1 = float(results["avg_team1_score"])
    sim_avg_t2 = float(results["avg_team2_score"])
    dist_quality = results.get("dist_quality") or {}

    actual_t1_won: Optional[int]
    if match.winner_id is None:
        actual_t1_won = None
    elif match.winner_id == match.team1_id:
        actual_t1_won = 1
    elif match.winner_id == match.team2_id:
        actual_t1_won = 0
    else:
        actual_t1_won = None

    margin_runs: Optional[float]
    if match.actual_team1_total is not None and match.actual_team2_total is not None:
        margin_runs = float(match.actual_team1_total - match.actual_team2_total)
    else:
        margin_runs = None
    sim_margin = sim_avg_t1 - sim_avg_t2

    score_mae_t1 = (
        abs(match.actual_team1_total - sim_avg_t1)
        if match.actual_team1_total is not None
        else None
    )
    score_mae_t2 = (
        abs(match.actual_team2_total - sim_avg_t2)
        if match.actual_team2_total is not None
        else None
    )

    return BacktestRow(
        match_id=match.match_id,
        date=match.date.isoformat(),
        match_type=match.match_type,
        gender=match.gender,
        event_name=match.event_name,
        team1_id=match.team1_id,
        team2_id=match.team2_id,
        canonical_team1_id=canonical1,
        canonical_team2_id=canonical2,
        team1_elo_used=float(results.get("team1_elo_used", float("nan"))),
        team2_elo_used=float(results.get("team2_elo_used", float("nan"))),
        sim_team1_win_prob=sim_t1_prob,
        sim_avg_team1_score=sim_avg_t1,
        sim_avg_team2_score=sim_avg_t2,
        actual_winner_id=match.winner_id,
        actual_team1_total=match.actual_team1_total,
        actual_team2_total=match.actual_team2_total,
        team1_won=actual_t1_won,
        margin_runs=margin_runs,
        sim_margin_runs=sim_margin,
        score_mae_team1=score_mae_t1,
        score_mae_team2=score_mae_t2,
        dist_quality_overall_pct=dist_quality.get("overall_pct"),
    )


# ============================================================================
# Aggregate metrics
# ============================================================================


def _safe_log(p: float, eps: float = 1e-6) -> float:
    return math.log(min(max(p, eps), 1.0 - eps))


def compute_metrics(rows: List[BacktestRow]) -> Dict:
    """Aggregate Brier, log-loss, MAE, calibration deciles."""
    decisive = [r for r in rows if r.team1_won is not None]
    n = len(decisive)
    if n == 0:
        return {"n_matches": 0, "n_decisive": 0}

    brier = sum((r.sim_team1_win_prob - r.team1_won) ** 2 for r in decisive) / n
    log_loss = -sum(
        r.team1_won * _safe_log(r.sim_team1_win_prob)
        + (1 - r.team1_won) * _safe_log(1 - r.sim_team1_win_prob)
        for r in decisive
    ) / n

    # Reliability: hit rate on top pick (sim says > 50%, did that team win?).
    hit = sum(
        1
        for r in decisive
        if (r.sim_team1_win_prob > 0.5 and r.team1_won == 1)
        or (r.sim_team1_win_prob < 0.5 and r.team1_won == 0)
    )
    accuracy = hit / n

    # Calibration deciles: predicted-prob bucket vs realised win rate.
    buckets = [(i / 10.0, (i + 1) / 10.0) for i in range(10)]
    calibration = []
    for lo, hi in buckets:
        in_bucket = [
            r for r in decisive if lo <= r.sim_team1_win_prob < hi
        ] if hi < 1.0 else [r for r in decisive if lo <= r.sim_team1_win_prob <= hi]
        if not in_bucket:
            calibration.append(
                {"lo": lo, "hi": hi, "n": 0, "mean_pred": None, "actual_win_rate": None}
            )
            continue
        mean_pred = sum(r.sim_team1_win_prob for r in in_bucket) / len(in_bucket)
        actual_rate = sum(r.team1_won for r in in_bucket) / len(in_bucket)
        calibration.append(
            {
                "lo": lo,
                "hi": hi,
                "n": len(in_bucket),
                "mean_pred": round(mean_pred, 4),
                "actual_win_rate": round(actual_rate, 4),
            }
        )

    # Score MAE
    score_maes = []
    for r in decisive:
        if r.score_mae_team1 is not None:
            score_maes.append(r.score_mae_team1)
        if r.score_mae_team2 is not None:
            score_maes.append(r.score_mae_team2)
    mae_score = sum(score_maes) / len(score_maes) if score_maes else None

    margin_diffs = [
        abs(r.sim_margin_runs - r.margin_runs)
        for r in decisive
        if r.margin_runs is not None
    ]
    mae_margin = sum(margin_diffs) / len(margin_diffs) if margin_diffs else None

    return {
        "n_matches": len(rows),
        "n_decisive": n,
        "n_no_result": len(rows) - n,
        "accuracy_top_pick": round(accuracy, 4),
        "brier_score": round(brier, 4),
        "log_loss": round(log_loss, 4),
        "mae_score_runs": round(mae_score, 2) if mae_score is not None else None,
        "mae_margin_runs": round(mae_margin, 2) if mae_margin is not None else None,
        "calibration_deciles": calibration,
    }


def stratified_metrics(rows: List[BacktestRow], by: str) -> Dict[str, Dict]:
    """Group rows by a label (event_name | match_type | gender) and compute metrics per group."""
    groups: Dict[str, List[BacktestRow]] = {}
    for r in rows:
        key = getattr(r, by) or "(none)"
        groups.setdefault(key, []).append(r)
    return {k: compute_metrics(v) for k, v in groups.items() if len(v) >= 5}


def rows_to_dicts(rows: List[BacktestRow]) -> List[Dict]:
    return [asdict(r) for r in rows]
