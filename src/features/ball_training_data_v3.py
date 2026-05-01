"""
Ball Training Data Generator v3 (Wave 5.5).

Inherits the v2 structure (9-class extras-aware labels, recency-weighted
sample weights, era feature, multi-format joint per gender, embedding-friendly
id columns) and adds three new continuous features that the Wave 5 EV
backtest identified as the missing context behind V2's ~50/50 priors:

    toss_won_by_batting_team   (binary)  - did the batting team win the toss?
    chose_to_bat               (binary)  - did the toss winner choose to bat?
    xi_overlap_recent_3        (0..1)    - fraction of this match's batting
                                           XI that played in the team's 3
                                           most-recent matches BEFORE this one
                                           (proxy for lineup stability)

The two toss features close the T-30min money-pit: by then the market has
already crystallised on the favourite because toss + decision are public,
but V2 didn't have these inputs so it was effectively betting blind.

The XI-overlap proxy gives the model a way to down-weight predictions for
matches with unstable XIs (impact subs, injury replacements, debutants).

CONTINUOUS_COLUMNS goes from 22 -> 25; ID_COLUMNS stay the same.
Output: data/processed/ball_training_v3_{gender}.npz with the same shape as
v2's npz plus the three extra columns at the END of the X matrix.

All other v2 helpers (classify_outcome, compute_sample_weight, ELO lookup,
venue stats, label constants) are reused unchanged.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.database import get_connection  # noqa: E402
from src.data.franchise_resolver import get_resolver  # noqa: E402
from src.features.ball_training_data import (  # noqa: E402
    get_team_elo_at_date,
    get_player_elo_at_date,
    normalize_elo,
)
from src.features.ball_training_data_v2 import (  # noqa: E402
    LABEL_NAMES,
    LABEL_WICKET,
    LABEL_WIDE,
    LABEL_NOBALL,
    NUM_CLASSES_V2,
    FORMAT_ID,
    GENDER_ID,
    ERA_REFERENCE_YEAR,
    ERA_NORM_DIVISOR,
    DEFAULT_HALF_LIFE_DAYS,
    OVER_TARGET_COLUMNS,
    classify_outcome,
    compute_sample_weight,
    _load_elo_lookups,
)
from src.features.venue_stats import VenueStatsBuilder  # noqa: E402
from src.utils.format_constants import balls_for_format, phase_for_over  # noqa: E402

logger = logging.getLogger(__name__)


# ============================================================================
# V3 column schema (22 v2 columns + 3 new at the end)
# ============================================================================

# IMPORTANT: The three new columns MUST be appended at the END so V2 sims
# loading a V3 npz can still index by the v2 CONTINUOUS_COLUMNS prefix.
# The V3 model + simulator know to look at the full 25-element vector.
CONTINUOUS_COLUMNS = [
    # --- v2 prefix (22 columns, identical order) ---
    "format_id_f",
    "innings",
    "over_idx",
    "ball_in_over",
    "legal_balls_in_innings_so_far",
    "runs_so_far",
    "wkts_so_far",
    "target_runs",
    "required_rate",
    "era_year_norm",
    "is_powerplay",
    "is_middle",
    "is_death",
    "venue_scoring_factor",
    "venue_boundary_rate",
    "venue_wicket_rate",
    "venue_reliable",
    "batting_team_elo_n",
    "bowling_team_elo_n",
    "team_elo_diff_n",
    "batter_elo_n",
    "bowler_elo_n",
    # --- v3 additions (3 columns) ---
    "toss_won_by_batting_team",   # binary 0/1
    "chose_to_bat",               # binary 0/1 (1 if toss winner chose to bat)
    "xi_overlap_recent_3",        # 0..1 stability proxy
]
N_CONTINUOUS = len(CONTINUOUS_COLUMNS)  # 25

# ID columns identical to V2
ID_COLUMNS = [
    "batter_id",
    "bowler_id",
    "venue_id",
    "batting_team_id",
    "bowling_team_id",
    "format_id",
    "gender_id",
]


@dataclass
class BuildConfigV3:
    """Knobs the operator can tune at the CLI."""
    formats: Tuple[str, ...] = ("T20", "ODI")
    gender: str = "male"
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS
    reference_date: Optional[date] = None
    limit: Optional[int] = None
    output_dir: Path = Path("data/processed")


@dataclass
class BuildResultV3:
    n_rows: int
    n_classes: Dict[str, int] = field(default_factory=dict)
    by_format: Dict[str, int] = field(default_factory=dict)
    output_path: Optional[Path] = None
    elapsed_seconds: float = 0.0


# ============================================================================
# XI lookup precomputation (the only non-trivial new piece)
# ============================================================================


def _precompute_xi_lookups(
    conn,
    formats: Tuple[str, ...],
    gender: str,
) -> Tuple[Dict[Tuple[int, int], Set[int]], Dict[int, List[Tuple[str, int]]]]:
    """Pre-compute two lookup tables for the XI-overlap feature.

    Returns:
        xi_by_match_team: {(match_id, team_id): set(player_ids in XI)}
        team_match_dates: {team_id: sorted list of (date_str, match_id) for that team}

    The team_match_dates list is sorted ascending by date so a binary search
    can find "the 3 most recent matches BEFORE date D" in O(log n).
    """
    cur = conn.cursor()
    fmt_placeholders = ",".join("?" * len(formats))

    # XI per (match, team) - one row per player who appeared in that match
    cur.execute(
        f"""
        SELECT pms.match_id, pms.team_id, pms.player_id
        FROM player_match_stats pms
        JOIN matches m ON m.match_id = pms.match_id
        WHERE m.match_type IN ({fmt_placeholders}) AND m.gender = ?
        """,
        list(formats) + [gender],
    )
    xi_by_match_team: Dict[Tuple[int, int], Set[int]] = {}
    for row in cur.fetchall():
        key = (row["match_id"], row["team_id"])
        xi_by_match_team.setdefault(key, set()).add(row["player_id"])

    # Per team, ordered list of (date, match_id) the team has played
    cur.execute(
        f"""
        SELECT m.match_id, m.date, m.team1_id, m.team2_id
        FROM matches m
        WHERE m.match_type IN ({fmt_placeholders}) AND m.gender = ?
        ORDER BY m.date ASC, m.match_id ASC
        """,
        list(formats) + [gender],
    )
    team_match_dates: Dict[int, List[Tuple[str, int]]] = {}
    for row in cur.fetchall():
        date_str = str(row["date"])[:10]
        for tid in (row["team1_id"], row["team2_id"]):
            if tid is None:
                continue
            team_match_dates.setdefault(tid, []).append((date_str, row["match_id"]))

    logger.info(
        f"[xi-lookup] {len(xi_by_match_team):,} match-team XIs, "
        f"{len(team_match_dates):,} teams indexed"
    )
    return xi_by_match_team, team_match_dates


def _xi_overlap_recent_3(
    team_id: int,
    match_id: int,
    match_date_str: str,
    this_xi: Set[int],
    xi_by_match_team: Dict[Tuple[int, int], Set[int]],
    team_match_dates: Dict[int, List[Tuple[str, int]]],
) -> float:
    """Fraction of this_xi that appeared in any of the team's 3 most-recent
    matches BEFORE match_date_str. Returns 0.0 if no prior matches exist
    (debut team / first match in window).
    """
    if not this_xi:
        return 0.0
    history = team_match_dates.get(team_id)
    if not history:
        return 0.0

    # Find prior matches: walk history backwards from the match_id position.
    # History is sorted ascending; binary search for insertion point at this date.
    # Simpler: linear walk since teams typically have <500 matches in our window.
    prior_xi: Set[int] = set()
    found = 0
    for date_str, mid in reversed(history):
        if date_str >= match_date_str and mid != match_id:
            # Future or same-day match (other than this one) - skip
            continue
        if date_str >= match_date_str:
            # Same-day match including this one - skip
            continue
        prior = xi_by_match_team.get((mid, team_id))
        if prior:
            prior_xi |= prior
            found += 1
            if found >= 3:
                break
    if found == 0:
        return 0.0
    overlap = len(this_xi & prior_xi)
    return overlap / max(1, len(this_xi))


# ============================================================================
# Builder
# ============================================================================


def build_ball_training_v3(config: BuildConfigV3) -> BuildResultV3:
    """Build the v3 unified per-ball training artifact for one gender.

    Same shape and semantics as v2; adds 3 new continuous features at the
    end of the X matrix. Output filename `ball_training_v3_{gender}.npz`.
    """
    t0 = time.time()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if config.reference_date is None:
        config.reference_date = datetime.now().date()

    resolver = get_resolver()

    conn = get_connection()
    try:
        cur = conn.cursor()

        team_lookup, player_lookup = _load_elo_lookups(cur, config.formats, config.gender)

        venue_builders: Dict[str, VenueStatsBuilder] = {}
        for fmt in config.formats:
            vb_path = Path(f"data/processed/venue_stats_{fmt.lower()}_{config.gender}.pkl")
            if vb_path.exists():
                logger.info(f"[{fmt} {config.gender}] loading cached venue stats from {vb_path}")
                venue_builders[fmt] = VenueStatsBuilder.load(str(vb_path))
            else:
                logger.warning(f"[{fmt} {config.gender}] venue stats not cached; building")
                vb = VenueStatsBuilder(fmt, config.gender)
                vb.build_from_database()
                vb_path.parent.mkdir(parents=True, exist_ok=True)
                vb.save(str(vb_path))
                venue_builders[fmt] = vb

        # Pre-compute XI lookups (the new bit). One pass over player_match_stats.
        xi_by_match_team, team_match_dates = _precompute_xi_lookups(
            conn, config.formats, config.gender
        )

        # Master query: same as v2 but JOIN toss columns from matches
        format_placeholders = ",".join("?" * len(config.formats))
        query = f"""
            SELECT
                d.delivery_id,
                d.innings_id,
                d.over_number,
                d.ball_number,
                d.batter_id,
                d.bowler_id,
                d.runs_batter,
                d.runs_extras,
                d.extras_wides,
                d.extras_noballs,
                d.is_wicket,
                i.innings_number,
                i.batting_team_id,
                i.bowling_team_id,
                m.match_id,
                m.match_type,
                m.date,
                m.venue_id,
                m.toss_winner_id,
                m.toss_decision
            FROM deliveries d
            JOIN innings i ON d.innings_id = i.innings_id
            JOIN matches m ON i.match_id = m.match_id
            WHERE m.match_type IN ({format_placeholders}) AND m.gender = ?
            ORDER BY m.date, d.delivery_id
        """
        params: List = list(config.formats) + [config.gender]
        if config.limit:
            query += f" LIMIT {int(config.limit)}"
        cur.execute(query, params)
        rows = cur.fetchall()
        logger.info(f"[{config.gender} v3] processing {len(rows):,} deliveries")

        n = len(rows)
        X = np.zeros((n, N_CONTINUOUS), dtype=np.float32)
        ids = np.zeros((n, len(ID_COLUMNS)), dtype=np.int32)
        y = np.zeros(n, dtype=np.int8)
        sw = np.zeros(n, dtype=np.float32)
        over_targets = np.zeros((n, len(OVER_TARGET_COLUMNS)), dtype=np.int16)

        match_ids_arr = np.zeros(n, dtype=np.int64)
        innings_ids_arr = np.zeros(n, dtype=np.int64)
        over_indices_arr = np.zeros(n, dtype=np.int16)

        # Cumulative innings state (same as v2)
        current_innings_id = None
        innings_runs = 0
        innings_wickets = 0
        innings_legal_balls = 0
        first_innings_score: Optional[int] = None
        first_innings_totals_cache: Dict[int, Optional[int]] = {}

        # Per-(match, batting_team) cache for xi_overlap so we don't recompute
        # 120-300 times per innings.
        xi_overlap_cache: Dict[Tuple[int, int], float] = {}

        for idx, row in enumerate(tqdm(rows, desc=f"{config.gender} v3")):
            innings_id = row["innings_id"]
            match_id = row["match_id"]
            fmt = row["match_type"]
            key = (fmt, config.gender)

            if innings_id != current_innings_id:
                if row["innings_number"] == 2:
                    if match_id in first_innings_totals_cache:
                        first_innings_score = first_innings_totals_cache[match_id]
                    else:
                        cur2 = conn.cursor()
                        cur2.execute(
                            "SELECT total_runs FROM innings WHERE match_id = ? AND innings_number = 1",
                            (match_id,),
                        )
                        r1 = cur2.fetchone()
                        first_innings_score = r1["total_runs"] if r1 else None
                        first_innings_totals_cache[match_id] = first_innings_score
                else:
                    first_innings_score = None
                current_innings_id = innings_id
                innings_runs = 0
                innings_wickets = 0
                innings_legal_balls = 0

            over = row["over_number"]
            ball = row["ball_number"]
            batter_id = row["batter_id"]
            bowler_id = row["bowler_id"]
            batting_team_id = row["batting_team_id"]
            bowling_team_id = row["bowling_team_id"]
            venue_id = row["venue_id"] or 0
            wides = row["extras_wides"] or 0
            noballs = row["extras_noballs"] or 0
            is_wicket = bool(row["is_wicket"])
            runs_batter = row["runs_batter"] or 0
            runs_extras = row["runs_extras"] or 0
            toss_winner_id = row["toss_winner_id"]
            toss_decision = row["toss_decision"]

            target = 0
            required_rate = 0.0
            if row["innings_number"] == 2 and first_innings_score is not None:
                target = first_innings_score + 1
                balls_remaining = balls_for_format(fmt) - innings_legal_balls
                if balls_remaining > 0:
                    runs_needed = target - innings_runs
                    if runs_needed > 0:
                        required_rate = runs_needed * 6.0 / balls_remaining

            is_pp, is_mid, is_death = phase_for_over(fmt, over)
            venue_features = venue_builders[fmt].get_venue_features(venue_id or None)

            canonical_bat = resolver.canonical(batting_team_id) or batting_team_id
            canonical_bowl = resolver.canonical(bowling_team_id) or bowling_team_id
            match_date_str = str(row["date"])[:10]
            batting_team_elo = get_team_elo_at_date(team_lookup[key], canonical_bat, match_date_str)
            bowling_team_elo = get_team_elo_at_date(team_lookup[key], canonical_bowl, match_date_str)
            batter_elo = get_player_elo_at_date(player_lookup[key], batter_id, match_date_str, "batting")
            bowler_elo = get_player_elo_at_date(player_lookup[key], bowler_id, match_date_str, "bowling")

            try:
                year = int(match_date_str[:4])
            except ValueError:
                year = ERA_REFERENCE_YEAR
            era_norm = (year - ERA_REFERENCE_YEAR) / ERA_NORM_DIVISOR

            try:
                d_match = datetime.strptime(match_date_str, "%Y-%m-%d").date()
            except ValueError:
                d_match = config.reference_date
            weight = compute_sample_weight(d_match, config.reference_date, config.half_life_days)

            # ---- v3 features ----
            # toss_won_by_batting_team: 1 if the team batting in THIS innings
            # won the toss; else 0. Note: applies to BOTH innings - in the 2nd
            # innings the bowling-side-of-1st-innings is now batting.
            toss_won_by_batting = 0
            if toss_winner_id is not None and batting_team_id is not None:
                toss_won_by_batting = 1 if int(toss_winner_id) == int(batting_team_id) else 0

            # chose_to_bat: 1 if toss winner chose to bat (regardless of which
            # team is batting in THIS innings - it's a property of the match).
            chose_to_bat = 0
            if toss_decision is not None:
                chose_to_bat = 1 if str(toss_decision).lower() == "bat" else 0

            # xi_overlap_recent_3: cache per (match_id, batting_team_id) since
            # it's constant within an innings.
            xi_key = (match_id, batting_team_id)
            if xi_key in xi_overlap_cache:
                xi_overlap = xi_overlap_cache[xi_key]
            else:
                this_xi = xi_by_match_team.get((match_id, batting_team_id), set())
                xi_overlap = _xi_overlap_recent_3(
                    team_id=batting_team_id,
                    match_id=match_id,
                    match_date_str=match_date_str,
                    this_xi=this_xi,
                    xi_by_match_team=xi_by_match_team,
                    team_match_dates=team_match_dates,
                )
                xi_overlap_cache[xi_key] = xi_overlap

            X[idx] = (
                # v2 prefix (22 cols)
                float(FORMAT_ID.get(fmt, 0)),
                float(row["innings_number"]),
                float(over),
                float(ball),
                float(innings_legal_balls),
                float(innings_runs),
                float(innings_wickets),
                float(target),
                float(required_rate),
                float(era_norm),
                float(is_pp),
                float(is_mid),
                float(is_death),
                float(venue_features[0]),
                float(venue_features[1]),
                float(venue_features[2]),
                float(venue_features[3]),
                float(normalize_elo(batting_team_elo)),
                float(normalize_elo(bowling_team_elo)),
                float((batting_team_elo - bowling_team_elo) / 200.0),
                float(normalize_elo(batter_elo)),
                float(normalize_elo(bowler_elo)),
                # v3 additions (3 cols)
                float(toss_won_by_batting),
                float(chose_to_bat),
                float(xi_overlap),
            )

            ids[idx] = (
                batter_id or 0,
                bowler_id or 0,
                venue_id,
                canonical_bat,
                canonical_bowl,
                FORMAT_ID.get(fmt, 0),
                GENDER_ID.get(config.gender, 0),
            )

            y[idx] = classify_outcome(
                runs_batter=runs_batter,
                runs_extras=runs_extras,
                extras_wides=wides,
                extras_noballs=noballs,
                is_wicket=is_wicket,
            )
            sw[idx] = weight

            match_ids_arr[idx] = match_id
            innings_ids_arr[idx] = innings_id
            over_indices_arr[idx] = over

            innings_runs += int(runs_batter) + int(runs_extras)
            if is_wicket:
                innings_wickets += 1
            if wides == 0 and noballs == 0:
                innings_legal_balls += 1

        # Per-over aggregation (same as v2)
        logger.info(f"[{config.gender} v3] aggregating per-over targets across {n:,} rows")
        over_run_lookup: Dict[Tuple[int, int], int] = {}
        over_wkt_lookup: Dict[Tuple[int, int], int] = {}
        for i in range(n):
            agg_key = (int(innings_ids_arr[i]), int(over_indices_arr[i]))
            cls = int(y[i])
            if cls in (1, 2, 3, 4):
                over_run_lookup[agg_key] = over_run_lookup.get(agg_key, 0) + cls
            elif cls == 5:
                over_run_lookup[agg_key] = over_run_lookup.get(agg_key, 0) + 6
            elif cls in (LABEL_WIDE, LABEL_NOBALL):
                over_run_lookup[agg_key] = over_run_lookup.get(agg_key, 0) + 1
            else:
                over_run_lookup.setdefault(agg_key, 0)
            if cls == LABEL_WICKET:
                over_wkt_lookup[agg_key] = over_wkt_lookup.get(agg_key, 0) + 1
            else:
                over_wkt_lookup.setdefault(agg_key, 0)

        for i in range(n):
            agg_key = (int(innings_ids_arr[i]), int(over_indices_arr[i]))
            over_targets[i, 0] = over_run_lookup.get(agg_key, 0)
            over_targets[i, 1] = over_wkt_lookup.get(agg_key, 0)

        out_path = config.output_dir / f"ball_training_v3_{config.gender}.npz"
        col_index = {
            "continuous_columns": np.array(CONTINUOUS_COLUMNS, dtype=object),
            "id_columns": np.array(ID_COLUMNS, dtype=object),
            "label_names": np.array(LABEL_NAMES, dtype=object),
            "over_target_columns": np.array(OVER_TARGET_COLUMNS, dtype=object),
        }
        np.savez_compressed(
            out_path,
            X=X,
            ids=ids,
            y=y,
            sample_weight=sw,
            over_targets=over_targets,
            **col_index,
        )

        n_per_class = {LABEL_NAMES[c]: int((y == c).sum()) for c in range(NUM_CLASSES_V2)}
        n_per_format = {fmt: int((ids[:, ID_COLUMNS.index("format_id")] == FORMAT_ID[fmt]).sum())
                        for fmt in config.formats if fmt in FORMAT_ID}

        elapsed = time.time() - t0
        result = BuildResultV3(
            n_rows=n,
            n_classes=n_per_class,
            by_format=n_per_format,
            output_path=out_path,
            elapsed_seconds=elapsed,
        )

        # Sanity-check the new features for non-degeneracy.
        toss_col = CONTINUOUS_COLUMNS.index("toss_won_by_batting_team")
        chose_col = CONTINUOUS_COLUMNS.index("chose_to_bat")
        xi_col = CONTINUOUS_COLUMNS.index("xi_overlap_recent_3")
        logger.info(f"[{config.gender} v3] wrote {n:,} rows to {out_path} in {elapsed:.1f}s")
        logger.info(f"  toss_won_by_batting_team mean={X[:, toss_col].mean():.3f} (~0.5 expected)")
        logger.info(f"  chose_to_bat            mean={X[:, chose_col].mean():.3f} (~0.4-0.5 expected)")
        logger.info(f"  xi_overlap_recent_3     mean={X[:, xi_col].mean():.3f} std={X[:, xi_col].std():.3f}")
        for cls_name, cnt in n_per_class.items():
            pct = 100.0 * cnt / n if n else 0.0
            logger.info(f"  class {cls_name:>7}: {cnt:>10,}  ({pct:5.2f}%)")
        for fmt, cnt in n_per_format.items():
            logger.info(f"  format {fmt}: {cnt:,} rows")
        return result
    finally:
        conn.close()
