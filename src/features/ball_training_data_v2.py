"""
Ball Training Data Generator v2 (Wave 4).

Builds a unified per-ball training artifact across all four (format, gender)
combinations for the v2 multi-task model. Key differences vs v1
(`ball_training_data.py`):

- 9-class outcome label: {0=dot, 1, 2, 3, 4, 6, wicket, wide, noball}.
  Extras are first-class. Wides and noballs no longer dropped or quietly
  miscoded; the model learns their per-ball probability and the v2
  simulator (Phase 4) consumes that to produce realistic over lengths.
  Bye/legbye runs are folded into the matching run class because the ball
  itself counts as legal.
- Recency-weighted sample weights: exp(-days_since_match / half_life_days).
  Half-life defaults to 365 days; tuned in the backtest harness later.
- Era feature: year-of-match (normalised) so the model can learn that par
  scores are different in 2018 vs 2026 without forgetting older patterns.
- Joint output: ONE numpy npz per gender (training is multi-format), with
  format included as a categorical feature. Simpler than the v1 4-way
  separate artifacts.
- Native ID columns (batter_id, bowler_id, venue_id, team1_id, team2_id)
  written through to the model; v2 model uses learned embeddings, not the
  pre-aggregated histograms v1 used.
- ELO features kept in the row (NOT replaced by embeddings; the v2 plan
  explicitly keeps temporal ELO as input alongside learned reps).

Output schema (one row per ball):

  Categorical / id columns (int32):
    format_id        # 0=T20, 1=ODI
    gender_id        # 0=male, 1=female (joint files are per gender, but kept
                     # as a column for sanity checks)
    venue_id
    batter_id
    bowler_id
    batting_team_id  # canonical (after franchise resolver)
    bowling_team_id  # canonical
    label_class      # 9-class target

  Continuous state columns (float32):
    innings, over_idx, ball_in_over, legal_balls_in_over_so_far,
    legal_balls_in_innings_so_far, runs_so_far, wkts_so_far, target_runs,
    required_rate, era_year_norm, sample_weight,
    is_powerplay, is_middle, is_death,                  # phase one-hot
    venue_scoring_factor, venue_boundary_rate, venue_wicket_rate, venue_reliable,
    batting_team_elo_n, bowling_team_elo_n, team_elo_diff_n,
    batter_elo_n, bowler_elo_n

Persisted as data/processed/ball_training_v2_{gender}.npz with one big
float32 X matrix, one int32 ids matrix, one int8 y vector, one float32
sample_weight vector, and a column_index dict in the same archive.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import DATABASE_PATH  # noqa: E402
from src.data.database import get_connection  # noqa: E402
from src.data.franchise_resolver import get_resolver  # noqa: E402
from src.features.ball_training_data import (  # noqa: E402
    get_team_elo_at_date,
    get_player_elo_at_date,
    normalize_elo,
)
from src.features.venue_stats import VenueStatsBuilder  # noqa: E402
from src.utils.format_constants import balls_for_format, phase_for_over  # noqa: E402

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# 9-class label space for v2. Indices match the simulator's outcome table.
LABEL_DOT     = 0
LABEL_SINGLE  = 1
LABEL_TWO     = 2
LABEL_THREE   = 3
LABEL_FOUR    = 4
LABEL_SIX     = 5
LABEL_WICKET  = 6
LABEL_WIDE    = 7
LABEL_NOBALL  = 8
NUM_CLASSES_V2 = 9
LABEL_NAMES = ["dot", "1", "2", "3", "4", "6", "wicket", "wide", "noball"]

FORMAT_ID = {"T20": 0, "ODI": 1}
GENDER_ID = {"male": 0, "female": 1}

DEFAULT_HALF_LIFE_DAYS = 365.0
ERA_REFERENCE_YEAR = 2026
ERA_NORM_DIVISOR = 10.0  # so era_year_norm sits roughly in [-1.5, 0.5]


@dataclass
class BuildConfig:
    """Knobs the operator can tune at the CLI."""
    formats: Tuple[str, ...] = ("T20", "ODI")
    gender: str = "male"
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS
    reference_date: Optional[date] = None  # for sample-weight calc; defaults to today
    limit: Optional[int] = None             # cap deliveries (smoke testing)
    output_dir: Path = Path("data/processed")


@dataclass
class BuildResult:
    """Returned for caller / CLI reporting."""
    n_rows: int
    n_classes: Dict[str, int] = field(default_factory=dict)
    by_format: Dict[str, int] = field(default_factory=dict)
    output_path: Optional[Path] = None
    elapsed_seconds: float = 0.0


# ============================================================================
# Label classification
# ============================================================================


def classify_outcome(
    runs_batter: int,
    runs_extras: int,
    extras_wides: int,
    extras_noballs: int,
    is_wicket: bool,
) -> int:
    """Map a delivery to one of NUM_CLASSES_V2 outcome classes.

    Priority order:
      1. wicket (overrides everything; if a wicket is taken on a wide/noball,
         the model still treats this as a wicket because the dismissal is
         the dominant signal for downstream simulation).
      2. wide  (extras_wides > 0 and not a wicket)
      3. noball (extras_noballs > 0 and not a wicket)
      4. otherwise classify by runs_batter (0..6); legitimate 5s are mapped
         to class 4 (the existing 7-class system has no "5 runs" slot, and
         5s are rare enough to fold without losing meaningful signal).

    Bye / legbye runs are NOT extras for label purposes - the ball still
    counts as legal and the runs are not attributed to a category we
    simulate. They get folded into the matching run class.
    """
    if is_wicket:
        return LABEL_WICKET
    if extras_wides > 0:
        return LABEL_WIDE
    if extras_noballs > 0:
        return LABEL_NOBALL
    r = runs_batter
    if r == 0:
        return LABEL_DOT
    if r == 1:
        return LABEL_SINGLE
    if r == 2:
        return LABEL_TWO
    if r == 3:
        return LABEL_THREE
    if r == 4:
        return LABEL_FOUR
    if r == 6:
        return LABEL_SIX
    # Rare 5s and 7s -> closest reasonable bucket.
    if r == 5:
        return LABEL_FOUR
    return LABEL_SIX  # 7+ extremely rare; treat as boundary tier


# ============================================================================
# Sample weighting
# ============================================================================


def compute_sample_weight(
    match_date: date,
    reference_date: date,
    half_life_days: float,
) -> float:
    """Exponential half-life decay. weight(t) = 0.5 ** (age / half_life)."""
    age_days = (reference_date - match_date).days
    if age_days < 0:
        return 1.0  # future-dated rows shouldn't exist but be safe
    return 0.5 ** (age_days / max(half_life_days, 1.0))


# ============================================================================
# ELO lookup loaders (per format/gender)
# ============================================================================


def _load_elo_lookups(cursor, formats: Tuple[str, ...], gender: str):
    """Load team and player ELO history into in-memory dicts for fast as-of-date
    lookup, keyed by (format, gender). Returns nested dicts:

        team_lookup[(format, gender)][team_id]   -> sorted list of (date, elo)
        player_lookup[(format, gender)][pid]      -> sorted list of (date, batting, bowling)
    """
    team_lookup: Dict[Tuple[str, str], Dict[int, list]] = {}
    player_lookup: Dict[Tuple[str, str], Dict[int, list]] = {}
    for fmt in formats:
        key = (fmt, gender)
        cursor.execute(
            """
            SELECT team_id, date, elo
            FROM team_elo_history
            WHERE format = ? AND gender = ? AND NOT is_monthly_snapshot
            ORDER BY team_id, date
            """,
            (fmt, gender),
        )
        team_dict: Dict[int, list] = {}
        for row in cursor.fetchall():
            team_dict.setdefault(row["team_id"], []).append((row["date"], row["elo"]))
        team_lookup[key] = team_dict
        logger.info(f"[{fmt} {gender}] loaded ELO history for {len(team_dict)} teams")

        cursor.execute(
            """
            SELECT player_id, date, batting_elo, bowling_elo
            FROM player_elo_history
            WHERE format = ? AND gender = ? AND NOT is_monthly_snapshot
            ORDER BY player_id, date
            """,
            (fmt, gender),
        )
        player_dict: Dict[int, list] = {}
        for row in cursor.fetchall():
            player_dict.setdefault(row["player_id"], []).append(
                (row["date"], row["batting_elo"], row["bowling_elo"])
            )
        player_lookup[key] = player_dict
        logger.info(f"[{fmt} {gender}] loaded ELO history for {len(player_dict)} players")
    return team_lookup, player_lookup


# ============================================================================
# Builder
# ============================================================================


# Continuous (float32) feature columns in fixed order. The training script
# loads this constant and uses it to slice / normalise the X matrix. Adding
# a column is a backwards-incompatible change; bump the file format version.
CONTINUOUS_COLUMNS = [
    "format_id_f",                          # 0/1 as float for the model
    "innings",
    "over_idx",
    "ball_in_over",
    "legal_balls_in_innings_so_far",
    "runs_so_far",
    "wkts_so_far",
    "target_runs",                          # 0 if first innings
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
]
N_CONTINUOUS = len(CONTINUOUS_COLUMNS)

# Categorical id columns (int32). venue_id can be NULL in the DB; we map NULL
# to 0 and reserve 0 in the embedding vocab as <UNK>.
ID_COLUMNS = [
    "batter_id",
    "bowler_id",
    "venue_id",
    "batting_team_id",   # canonical (post franchise resolver)
    "bowling_team_id",   # canonical
    "format_id",         # 0/1
    "gender_id",         # 0/1
]

# Per-row over-level targets used by the v2 model's auxiliary per-over head.
# These get pre-computed from the per-ball labels in the same builder pass
# so each per-ball row carries the (runs, wkts) for the OVER it belongs to.
# That lets the train loop apply the per-over loss without doing dynamic
# aggregation during batching.
OVER_TARGET_COLUMNS = [
    "over_runs",   # total runs scored in the over this ball belongs to
    "over_wkts",   # total wickets in the over this ball belongs to
]


def build_ball_training_v2(config: BuildConfig) -> BuildResult:
    """Build the v2 unified per-ball training artifact for one gender.

    All formats listed in `config.formats` are concatenated into one numpy
    archive. Each row carries a `format_id` so the multi-task v2 model can
    route to the right per-(format, gender) head.
    """
    import time

    t0 = time.time()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if config.reference_date is None:
        config.reference_date = datetime.now().date()

    resolver = get_resolver()

    conn = get_connection()
    try:
        cur = conn.cursor()

        # Pre-load ELO history into memory once.
        team_lookup, player_lookup = _load_elo_lookups(cur, config.formats, config.gender)

        # Pre-load + cache venue features (the existing builder already
        # caches; we just keep one VenueStatsBuilder per format).
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

        # Master query joining all the format-rows we need in one streaming pass.
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
                m.venue_id
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
        logger.info(f"[{config.gender}] processing {len(rows):,} deliveries")

        # Pre-allocate numpy buffers for speed (we know the final size).
        n = len(rows)
        X = np.zeros((n, N_CONTINUOUS), dtype=np.float32)
        ids = np.zeros((n, len(ID_COLUMNS)), dtype=np.int32)
        y = np.zeros(n, dtype=np.int8)
        sw = np.zeros(n, dtype=np.float32)
        # Per-over targets, stamped from the per-ball labels in a second pass
        # below. shape (n, 2) - [over_runs, over_wkts].
        over_targets = np.zeros((n, len(OVER_TARGET_COLUMNS)), dtype=np.int16)
        # Sidecar arrays needed for the per-over aggregation pass; not
        # persisted to the final npz.
        match_ids_arr = np.zeros(n, dtype=np.int64)
        innings_ids_arr = np.zeros(n, dtype=np.int64)
        over_indices_arr = np.zeros(n, dtype=np.int16)

        # Cumulative innings state needs reset on each new innings.
        current_innings_id = None
        innings_runs = 0
        innings_wickets = 0
        innings_legal_balls = 0
        first_innings_score: Optional[int] = None

        # First-innings totals cache so the second-innings target is O(1).
        first_innings_totals_cache: Dict[int, Optional[int]] = {}

        for idx, row in enumerate(tqdm(rows, desc=f"{config.gender} v2")):
            innings_id = row["innings_id"]
            match_id = row["match_id"]
            fmt = row["match_type"]
            key = (fmt, config.gender)

            # New-innings reset
            if innings_id != current_innings_id:
                # If we're entering a 2nd innings, look up the 1st innings total.
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

            # Required rate (format-aware)
            target = 0
            required_rate = 0.0
            if row["innings_number"] == 2 and first_innings_score is not None:
                target = first_innings_score + 1
                balls_remaining = balls_for_format(fmt) - innings_legal_balls
                if balls_remaining > 0:
                    runs_needed = target - innings_runs
                    if runs_needed > 0:
                        required_rate = runs_needed * 6.0 / balls_remaining

            # Phase (format-aware)
            is_pp, is_mid, is_death = phase_for_over(fmt, over)

            # Venue features
            # get_venue_features() handles None / missing venue_ids internally
            venue_features = venue_builders[fmt].get_venue_features(venue_id or None)

            # ELO lookups - canonical team ids so unified franchises share series
            canonical_bat = resolver.canonical(batting_team_id) or batting_team_id
            canonical_bowl = resolver.canonical(bowling_team_id) or bowling_team_id
            match_date_str = str(row["date"])[:10]
            batting_team_elo = get_team_elo_at_date(team_lookup[key], canonical_bat, match_date_str)
            bowling_team_elo = get_team_elo_at_date(team_lookup[key], canonical_bowl, match_date_str)
            batter_elo = get_player_elo_at_date(player_lookup[key], batter_id, match_date_str, "batting")
            bowler_elo = get_player_elo_at_date(player_lookup[key], bowler_id, match_date_str, "bowling")

            # Era feature (so the model can lean toward modern par scores
            # without forgetting older patterns). Reference 2026, divisor 10.
            try:
                year = int(match_date_str[:4])
            except ValueError:
                year = ERA_REFERENCE_YEAR
            era_norm = (year - ERA_REFERENCE_YEAR) / ERA_NORM_DIVISOR

            # Sample weight: exponential half-life in days
            try:
                d_match = datetime.strptime(match_date_str, "%Y-%m-%d").date()
            except ValueError:
                d_match = config.reference_date
            weight = compute_sample_weight(d_match, config.reference_date, config.half_life_days)

            # Continuous feature row (must match CONTINUOUS_COLUMNS order)
            X[idx] = (
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

            # Sidecar columns needed for the per-over aggregation pass
            match_ids_arr[idx] = match_id
            innings_ids_arr[idx] = innings_id
            over_indices_arr[idx] = over

            # Update innings state for the NEXT row.
            # Wides/noballs do not advance the legal-ball counter and
            # do not increment the (batter-only) runs scored attribution
            # but the team total grows by runs_total. For the running
            # totals carried into next ball's features we use innings
            # totals (i.runs etc would also work but is per-innings only,
            # so we maintain locally).
            innings_runs += int(runs_batter) + int(runs_extras)
            if is_wicket:
                innings_wickets += 1
            if wides == 0 and noballs == 0:
                innings_legal_balls += 1

        # ----------------------------------------------------------------
        # Per-over aggregation pass: stamp each ball's over_runs and
        # over_wkts so the train loop can apply the auxiliary per-over loss
        # without dynamic batch-time aggregation. Group by
        # (innings_id, over_idx) since innings_id is unique per match-innings.
        # ----------------------------------------------------------------
        logger.info(f"[{config.gender}] aggregating per-over targets across {n:,} rows")
        over_run_lookup: Dict[Tuple[int, int], int] = {}
        over_wkt_lookup: Dict[Tuple[int, int], int] = {}
        for i in range(n):
            key = (int(innings_ids_arr[i]), int(over_indices_arr[i]))
            cls = int(y[i])
            # OUTCOME_RUNS table mirroring vectorized_nn_sim_v2's contract
            if cls in (1, 2, 3, 4):
                over_run_lookup[key] = over_run_lookup.get(key, 0) + cls
            elif cls == 5:
                over_run_lookup[key] = over_run_lookup.get(key, 0) + 6
            elif cls in (LABEL_WIDE, LABEL_NOBALL):
                over_run_lookup[key] = over_run_lookup.get(key, 0) + 1
            else:
                over_run_lookup.setdefault(key, 0)
            if cls == LABEL_WICKET:
                over_wkt_lookup[key] = over_wkt_lookup.get(key, 0) + 1
            else:
                over_wkt_lookup.setdefault(key, 0)

        for i in range(n):
            key = (int(innings_ids_arr[i]), int(over_indices_arr[i]))
            over_targets[i, 0] = over_run_lookup.get(key, 0)
            over_targets[i, 1] = over_wkt_lookup.get(key, 0)

        # Persist
        out_path = config.output_dir / f"ball_training_v2_{config.gender}.npz"
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

        # Class distribution for sanity
        n_per_class = {LABEL_NAMES[c]: int((y == c).sum()) for c in range(NUM_CLASSES_V2)}
        n_per_format = {fmt: int((ids[:, ID_COLUMNS.index("format_id")] == FORMAT_ID[fmt]).sum())
                        for fmt in config.formats if fmt in FORMAT_ID}

        elapsed = time.time() - t0
        result = BuildResult(
            n_rows=n,
            n_classes=n_per_class,
            by_format=n_per_format,
            output_path=out_path,
            elapsed_seconds=elapsed,
        )
        logger.info(
            f"[{config.gender}] wrote {n:,} rows to {out_path} in {elapsed:.1f}s"
        )
        for cls_name, cnt in n_per_class.items():
            pct = 100.0 * cnt / n if n else 0.0
            logger.info(f"  class {cls_name:>7}: {cnt:>10,}  ({pct:5.2f}%)")
        for fmt, cnt in n_per_format.items():
            logger.info(f"  format {fmt}: {cnt:,} rows")
        return result
    finally:
        conn.close()
