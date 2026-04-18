"""
Vectorized Monte Carlo simulator v2 (Wave 4 Phase 4).

Differences from v1 (`vectorized_nn_sim.py`):

1. **9-class outcomes with extras as first-class citizens.**
   Outcome classes 0..8 are: dot, 1, 2, 3, 4, 6, wicket, wide, noball.
   - On `wide` or `noball`: add 1 run to team total, do NOT increment the
     legal-ball counter, do NOT advance the batter (stays on strike).
     The over only ends when 6 LEGAL balls have been bowled.
   - On `noball`: set a `free_hit` flag for the next legal ball. While the
     flag is set, sampled `wicket` outcomes are remapped to `dot` (free
     hit cannot be out except run-out, which we don't model separately).
   - Pre-allocate `max_balls = max_overs * 9` for headroom in extras-heavy
     overs; loop terminates on `legal_balls >= max_overs * 6` instead of
     ball index.

2. **Hierarchical inference via budget-bias.** For each over of each match
   in the batch, sample `(over_runs_target, over_wkts_target)` from the
   per-over head once, then walk the legal balls of that over with the
   per-ball softmax biased by the running over budget so high-budget
   overs lean toward 4/6 and low-budget overs lean toward 0/1.

3. **Format-aware over count + termination.** Pulled from the model's
   format_type via the format_constants helper. T20 = 20 overs (120
   legal), ODI = 50 overs (300 legal).

4. **Calibration applied at match-aggregation step.** A
   CalibrationBundle is loaded at construction; after the Monte Carlo
   batch completes, the team1 win probability is run through Platt
   scaling per (format, gender) before being returned.

5. **Multi-task model routing.** v2 model has 8 outputs (4 ball heads +
   4 over heads). The simulator picks the right pair via the model's
   format_type + gender at construction time.

6. **Learned embeddings instead of histograms.** Inputs to the model are
   raw vocab-mapped player/venue/team ids (int32) plus the 22-element
   state vector. The model's embedding layers do the rest.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

try:
    import tensorflow as tf
    import keras
except ImportError:  # pragma: no cover
    raise SystemExit(
        "vectorized_nn_sim_v2 requires tensorflow + keras (already a v1 dependency)"
    )

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.franchise_resolver import get_resolver  # noqa: E402
from src.features.ball_training_data_v2 import (  # noqa: E402
    FORMAT_ID,
    GENDER_ID,
    LABEL_DOT,
    LABEL_WICKET,
    LABEL_WIDE,
    LABEL_NOBALL,
    NUM_CLASSES_V2,
    ERA_REFERENCE_YEAR,
    ERA_NORM_DIVISOR,
    N_CONTINUOUS,
)
from src.features.ball_training_data import normalize_elo  # noqa: E402
from src.features.venue_stats import VenueStatsBuilder  # noqa: E402
from src.models.calibration import (  # noqa: E402
    CalibrationBundle,
    calibrate_probabilities,
)
from src.utils.format_constants import (  # noqa: E402
    overs_for_format,
    phase_arrays,
    balls_for_format,
)

logger = logging.getLogger(__name__)


# Mapping from class index to runs added to the team total. Wides and
# noballs add 1 run as the standard penalty; wide_runs > 1 (e.g. 4 wides
# from a runaway throw) are not modelled in v1 and fold into byes.
OUTCOME_RUNS_V2 = np.array(
    # dot, 1, 2, 3, 4, 6, wicket, wide, noball
    [0,    1, 2, 3, 4, 6, 0,      1,    1],
    dtype=np.int32,
)


@dataclass
class V2SimulatorConfig:
    """Construction parameters for the v2 simulator."""
    format_type: str
    gender: str
    model_path: Optional[str] = None
    vocab_path: Optional[str] = None
    venue_stats_path: Optional[str] = None
    calibration_path: Optional[str] = None
    # Hierarchical sampling: alpha controls how strictly the per-ball
    # outputs follow the per-over head's budget. 0 = ignore over head
    # entirely; high values = strict budget enforcement. Tuned via the
    # backtest harness once v2 is trained (Phase 5/6).
    over_budget_alpha: float = 0.4


class V2Simulator:
    """v2 vectorized Monte Carlo simulator.

    Construction loads:
    - The v2 model (multi-task, multi-output Keras model from Phase 2).
    - Vocab tables for batter / bowler / venue / team ids.
    - Venue stats (re-used from v1).
    - Calibration bundle (optional; identity if missing).
    """

    def __init__(self, config: V2SimulatorConfig) -> None:
        self.config = config
        self.format_type = config.format_type.upper()
        self.gender = config.gender.lower()
        self.format_id = FORMAT_ID.get(self.format_type, 0)
        self.gender_id = GENDER_ID.get(self.gender, 0)

        # Auto-derive paths if not specified
        model_path = config.model_path or "data/models/v2/cricket_model_v2.keras"
        vocab_path = config.vocab_path or "data/models/v2/vocabs.json"
        venue_stats_path = (
            config.venue_stats_path
            or f"data/processed/venue_stats_{self.format_type.lower()}_{self.gender}.pkl"
        )
        calibration_path = config.calibration_path or "data/models/v2/calibration.json"

        logger.info(
            f"[v2-sim] init format={self.format_type} gender={self.gender} model={model_path}"
        )

        # Lazy-friendly: tolerate missing artifacts so this module can be
        # imported even before training has produced the v2 model. We only
        # error out at the first call to simulate_matches if we need them.
        self._model: Optional[keras.Model] = None
        self._predict_compiled = None
        self._model_path = model_path

        self.vocabs: Dict[str, Dict[int, int]] = {
            "batter": {},
            "bowler": {},
            "venue": {},
            "team": {},
        }
        if Path(vocab_path).exists():
            with open(vocab_path) as fp:
                raw = json.load(fp)
            # JSON keys come back as strings; coerce.
            self.vocabs = {
                k: {int(rid): vid for rid, vid in v.items()} for k, v in raw.items()
            }
            logger.info(
                f"[v2-sim] loaded vocabs: batter={len(self.vocabs['batter'])} "
                f"bowler={len(self.vocabs['bowler'])} venue={len(self.vocabs['venue'])} "
                f"team={len(self.vocabs['team'])}"
            )
        else:
            logger.warning(f"[v2-sim] vocab file not found at {vocab_path}; will use UNK for everything")

        try:
            self.venue_stats = VenueStatsBuilder.load(venue_stats_path)
            logger.info(f"[v2-sim] loaded venue stats for {len(self.venue_stats.venue_stats)} venues")
        except Exception as exc:
            logger.warning(f"[v2-sim] venue stats unavailable ({exc}); using defaults")
            self.venue_stats = None

        self.calibration = CalibrationBundle()
        if Path(calibration_path).exists():
            self.calibration = CalibrationBundle.load(calibration_path)
            logger.info(f"[v2-sim] loaded calibration: {len(self.calibration.params)} routes")
        else:
            logger.warning(f"[v2-sim] no calibration at {calibration_path}; using identity (raw probs)")

        # ELO caches for as-of-date lookups at simulation time. Loaded
        # lazily because most callers will want CURRENT ratings (live
        # bulk-predict) rather than historical (backtest).
        self._team_current_elo: Optional[Dict[int, float]] = None
        self._player_batting_elo: Optional[Dict[int, float]] = None
        self._player_bowling_elo: Optional[Dict[int, float]] = None

    # ------------------------------------------------------------------
    # Model loading (lazy)
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return
        if not Path(self._model_path).exists():
            raise FileNotFoundError(
                f"v2 model not found at {self._model_path}; run "
                f"scripts/train_cricket_model_v2.py first"
            )
        logger.info(f"[v2-sim] loading model from {self._model_path}")
        self._model = keras.models.load_model(self._model_path)
        self._predict_compiled = tf.function(
            self._model, jit_compile=True, reduce_retracing=True
        )

    @property
    def ball_head_name(self) -> str:
        # Matches the output dict key set in cricket_model_v2.build_cricket_model_v2
        # (see line: outputs[f"ball_{key[0].lower()}_{key[1]}"] = Activation(softmax, ...)).
        return f"ball_{self.format_type.lower()}_{self.gender}"

    @property
    def over_head_name(self) -> str:
        # Matches the output dict key set in cricket_model_v2.build_cricket_model_v2
        # (see line: outputs[f"over_{key[0].lower()}_{key[1]}"] = ...).
        return f"over_{self.format_type.lower()}_{self.gender}"

    # ------------------------------------------------------------------
    # ELO loading (mirrors v1)
    # ------------------------------------------------------------------

    def _load_current_elos(self) -> None:
        if self._team_current_elo is not None:
            return
        from src.data.database import get_connection
        conn = get_connection()
        try:
            cur = conn.cursor()
            elo_col = f"elo_{self.format_type.lower()}_{self.gender}"
            cur.execute(
                f"SELECT team_id, {elo_col} FROM team_current_elo WHERE {elo_col} IS NOT NULL"
            )
            self._team_current_elo = {row["team_id"]: row[elo_col] for row in cur.fetchall()}
            bat_col = f"batting_elo_{self.format_type.lower()}_{self.gender}"
            bowl_col = f"bowling_elo_{self.format_type.lower()}_{self.gender}"
            cur.execute(
                f"SELECT player_id, {bat_col}, {bowl_col} FROM player_current_elo "
                f"WHERE {bat_col} IS NOT NULL OR {bowl_col} IS NOT NULL"
            )
            bat: Dict[int, float] = {}
            bowl: Dict[int, float] = {}
            for row in cur.fetchall():
                if row[bat_col] is not None:
                    bat[row["player_id"]] = row[bat_col]
                if row[bowl_col] is not None:
                    bowl[row["player_id"]] = row[bowl_col]
            self._player_batting_elo = bat
            self._player_bowling_elo = bowl
            logger.info(
                f"[v2-sim] loaded current ELOs: {len(self._team_current_elo)} teams, "
                f"{len(self._player_batting_elo)} batting, {len(self._player_bowling_elo)} bowling"
            )
        finally:
            conn.close()

    def _team_elo(self, team_id: Optional[int]) -> float:
        if team_id is None:
            return 1500.0
        self._load_current_elos()
        canonical = get_resolver().canonical(team_id) or team_id
        return self._team_current_elo.get(canonical, 1500.0)

    def _player_batting(self, pid: Optional[int]) -> float:
        if pid is None:
            return 1500.0
        self._load_current_elos()
        return self._player_batting_elo.get(int(pid), 1500.0)

    def _player_bowling(self, pid: Optional[int]) -> float:
        if pid is None:
            return 1500.0
        self._load_current_elos()
        return self._player_bowling_elo.get(int(pid), 1500.0)

    # ------------------------------------------------------------------
    # Vocab application (raw_id -> compact embedding index, UNK = 0)
    # ------------------------------------------------------------------

    def _vocab_id(self, kind: str, raw_id: Optional[int]) -> int:
        if raw_id is None:
            return 0
        return self.vocabs.get(kind, {}).get(int(raw_id), 0)

    # ------------------------------------------------------------------
    # Hierarchical sampling helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_over_targets(
        over_params: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample `(runs_target, wkts_target)` from the per-over head.

        over_params shape: (n_matches, 4) - [mu_runs, log_sigma_runs, log_lambda_wkts, _]
        Returns: (runs_target, wkts_target) each shape (n_matches,)
        """
        mu = over_params[:, 0]
        sigma = np.exp(np.clip(over_params[:, 1], -3.0, 3.0))
        lam = np.exp(np.clip(over_params[:, 2], -2.0, 3.0))
        runs_target = np.maximum(0.0, rng.normal(mu, sigma))
        wkts_target = rng.poisson(lam).astype(np.float32)
        return runs_target.astype(np.float32), wkts_target

    @staticmethod
    def _bias_per_ball_logits(
        proba: np.ndarray,
        runs_remaining_per_ball: np.ndarray,
        wkts_remaining: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """Apply budget-bias to per-ball softmax outputs.

        Boosts the relative weight on run-bearing classes (4, 6) when
        runs_remaining_per_ball is high (and corresponding suppression of
        dot/single); boosts wicket class when wkts_remaining is high.

        We work in log-prob space then re-softmax to keep a valid distribution.
        """
        if alpha <= 0.0:
            return proba
        # Normalise the budget signal: ~0 means at-pace; positive means
        # we still need runs faster than baseline; negative means we have
        # surplus.
        baseline_rpb = 1.5  # rough average runs per ball across formats
        budget_signal = (runs_remaining_per_ball - baseline_rpb)  # (n_matches,)

        # Per-class shifts: classes 4 and 5 (4 and 6 runs) get +1 unit per
        # signal unit; classes 0 and 1 (dot/single) get -1; others 0.
        # Symmetric so when signal is negative (surplus), dots/singles get
        # boosted and 4/6 suppressed.
        logits_shift = np.zeros((proba.shape[0], NUM_CLASSES_V2), dtype=np.float32)
        logits_shift[:, 4] += alpha * budget_signal      # class 4 (4 runs)
        logits_shift[:, 5] += 1.5 * alpha * budget_signal  # class 5 (6 runs)
        logits_shift[:, 0] -= alpha * budget_signal      # class 0 (dot)
        logits_shift[:, 1] -= 0.5 * alpha * budget_signal  # class 1 (single)

        # Wicket bias: small positive nudge per wkts_remaining unit.
        logits_shift[:, 6] += 0.3 * alpha * wkts_remaining

        # Apply shift in log-prob space, then softmax-normalise
        eps = 1e-8
        log_p = np.log(np.clip(proba, eps, 1.0)) + logits_shift
        log_p -= log_p.max(axis=1, keepdims=True)
        biased = np.exp(log_p)
        biased /= biased.sum(axis=1, keepdims=True)
        return biased.astype(np.float32)

    @staticmethod
    def _sample_outcomes(proba: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Vectorised categorical sampling. Returns int8 outcomes (0..8)."""
        cumprob = np.cumsum(proba, axis=1)
        r = rng.random(proba.shape[0])[:, None]
        return np.argmax(cumprob >= r, axis=1).astype(np.int8)

    # ------------------------------------------------------------------
    # Per-ball feature builder (matches CONTINUOUS_COLUMNS exactly)
    # ------------------------------------------------------------------

    def _build_state_features(
        self,
        n_matches: int,
        innings: int,
        over_idx: np.ndarray,
        ball_in_over: np.ndarray,
        legal_balls: np.ndarray,
        runs: np.ndarray,
        wkts: np.ndarray,
        target: int,
        venue_features: np.ndarray,
        bat_team_elo_n: float,
        bowl_team_elo_n: float,
        team_elo_diff_n: float,
        batter_elo_n: np.ndarray,
        bowler_elo_n: float,
        era_norm: float,
    ) -> np.ndarray:
        """Build the state matrix (n_matches, N_CONTINUOUS) for the model."""
        _pp, _mid, _death, mid_t, death_t = phase_arrays(self.format_type)
        # over_idx shape (n_matches,). Vectorised phase one-hot:
        is_pp = (over_idx < mid_t).astype(np.float32)
        is_mid = ((over_idx >= mid_t) & (over_idx < death_t)).astype(np.float32)
        is_death = (over_idx >= death_t).astype(np.float32)

        # Required rate (for innings 2; 0 otherwise)
        if innings == 2 and target > 0:
            balls_remaining = balls_for_format(self.format_type) - legal_balls
            balls_remaining = np.clip(balls_remaining, 1, None)
            runs_needed = np.maximum(0, target - runs)
            required_rate = runs_needed * 6.0 / balls_remaining
        else:
            required_rate = np.zeros(n_matches, dtype=np.float32)

        X = np.zeros((n_matches, N_CONTINUOUS), dtype=np.float32)
        # Column order MUST match CONTINUOUS_COLUMNS in ball_training_data_v2.
        X[:, 0] = float(self.format_id)
        X[:, 1] = float(innings)
        X[:, 2] = over_idx
        X[:, 3] = ball_in_over
        X[:, 4] = legal_balls
        X[:, 5] = runs
        X[:, 6] = wkts
        X[:, 7] = float(target if innings == 2 else 0)
        X[:, 8] = required_rate
        X[:, 9] = era_norm
        X[:, 10] = is_pp
        X[:, 11] = is_mid
        X[:, 12] = is_death
        X[:, 13:17] = venue_features  # broadcast (4,) to (n_matches, 4)
        X[:, 17] = bat_team_elo_n
        X[:, 18] = bowl_team_elo_n
        X[:, 19] = team_elo_diff_n
        X[:, 20] = batter_elo_n
        X[:, 21] = bowler_elo_n
        return X

    # ------------------------------------------------------------------
    # Public API: simulate_matches (mirrors v1's signature)
    # ------------------------------------------------------------------

    def simulate_matches(
        self,
        n_matches: int,
        team1_batter_ids: List[int],
        team1_bowler_ids: List[int],
        team2_batter_ids: List[int],
        team2_bowler_ids: List[int],
        max_overs: Optional[int] = None,
        venue_id: Optional[int] = None,
        team1_id: Optional[int] = None,
        team2_id: Optional[int] = None,
        seed: Optional[int] = None,
        # Toss-related kwargs accepted for v1-compat but currently not
        # used by v2 (toss-aware sim is a v2.1 follow-up).
        use_toss: bool = False,
        toss_field_prob: float = 0.65,
        team1_default_elo: Optional[float] = None,
        team2_default_elo: Optional[float] = None,
    ) -> Dict:
        """Run N hierarchical Monte Carlo simulations of the match.

        Returns a v1-compatible result dict so the backtest harness and
        the live web path can pick v1 vs v2 behind a single flag in
        Phase 6.
        """
        self._ensure_model_loaded()
        if max_overs is None:
            max_overs = overs_for_format(self.format_type)

        rng = np.random.default_rng(seed)

        # Resolve team ELOs (canonical via FranchiseResolver)
        team1_elo = self._team_elo(team1_id) if team1_id else (team1_default_elo or 1500.0)
        team2_elo = self._team_elo(team2_id) if team2_id else (team2_default_elo or 1500.0)

        # Era feature: simulator runs at "today" (prediction time)
        from datetime import datetime
        era_norm = (datetime.now().year - ERA_REFERENCE_YEAR) / ERA_NORM_DIVISOR

        # Pre-compute per-player ELO arrays + vocab ids
        t1_bat_elos = np.array([self._player_batting(p) for p in team1_batter_ids], dtype=np.float32)
        t1_bowl_elos = np.array([self._player_bowling(p) for p in team1_bowler_ids], dtype=np.float32)
        t2_bat_elos = np.array([self._player_batting(p) for p in team2_batter_ids], dtype=np.float32)
        t2_bowl_elos = np.array([self._player_bowling(p) for p in team2_bowler_ids], dtype=np.float32)
        t1_bat_vocab = np.array([self._vocab_id("batter", p) for p in team1_batter_ids], dtype=np.int32)
        t1_bowl_vocab = np.array([self._vocab_id("bowler", p) for p in team1_bowler_ids], dtype=np.int32)
        t2_bat_vocab = np.array([self._vocab_id("batter", p) for p in team2_batter_ids], dtype=np.int32)
        t2_bowl_vocab = np.array([self._vocab_id("bowler", p) for p in team2_bowler_ids], dtype=np.int32)
        venue_vocab = self._vocab_id("venue", venue_id)
        t1_team_vocab = self._vocab_id("team", get_resolver().canonical(team1_id) if team1_id else None)
        t2_team_vocab = self._vocab_id("team", get_resolver().canonical(team2_id) if team2_id else None)

        # Distribution quality (mirrors v1 reporting; here it's vocab-coverage)
        total_players = len(team1_batter_ids) + len(team1_bowler_ids) + len(team2_batter_ids) + len(team2_bowler_ids)
        in_vocab = (
            int(np.sum(t1_bat_vocab > 0))
            + int(np.sum(t1_bowl_vocab > 0))
            + int(np.sum(t2_bat_vocab > 0))
            + int(np.sum(t2_bowl_vocab > 0))
        )
        dist_quality = {
            "team1": {
                "batters_found": int(np.sum(t1_bat_vocab > 0)),
                "batters_total": len(team1_batter_ids),
                "bowlers_found": int(np.sum(t1_bowl_vocab > 0)),
                "bowlers_total": len(team1_bowler_ids),
            },
            "team2": {
                "batters_found": int(np.sum(t2_bat_vocab > 0)),
                "batters_total": len(team2_batter_ids),
                "bowlers_found": int(np.sum(t2_bowl_vocab > 0)),
                "bowlers_total": len(team2_bowler_ids),
            },
            "overall_found": in_vocab,
            "overall_total": total_players,
            "overall_pct": (in_vocab / total_players) if total_players else 0.0,
        }

        venue_features = (
            self.venue_stats.get_venue_features(venue_id)
            if (self.venue_stats is not None and venue_id is not None)
            else np.array([1.0, 0.55, 0.05, 0.0], dtype=np.float32)
        )

        # First innings: team1 bats, team2 bowls
        first_runs, _ = self._simulate_innings(
            n_matches=n_matches,
            innings=1,
            target=0,
            max_overs=max_overs,
            batter_vocab=t1_bat_vocab,
            bowler_vocab=t2_bowl_vocab,
            batter_elos=t1_bat_elos,
            bowler_elos=t2_bowl_elos,
            batting_team_vocab=t1_team_vocab,
            bowling_team_vocab=t2_team_vocab,
            batting_team_elo=team1_elo,
            bowling_team_elo=team2_elo,
            venue_vocab=venue_vocab,
            venue_features=venue_features,
            era_norm=era_norm,
            rng=rng,
        )

        # Second innings: team2 chases first_runs + 1
        targets = (first_runs + 1).astype(np.int32)
        second_runs, _ = self._simulate_innings(
            n_matches=n_matches,
            innings=2,
            target=targets,
            max_overs=max_overs,
            batter_vocab=t2_bat_vocab,
            bowler_vocab=t1_bowl_vocab,
            batter_elos=t2_bat_elos,
            bowler_elos=t1_bowl_elos,
            batting_team_vocab=t2_team_vocab,
            bowling_team_vocab=t1_team_vocab,
            batting_team_elo=team2_elo,
            bowling_team_elo=team1_elo,
            venue_vocab=venue_vocab,
            venue_features=venue_features,
            era_norm=era_norm,
            rng=rng,
        )

        team1_won = first_runs >= targets  # team1 wins if team2 fails to chase
        team1_win_prob_raw = float(team1_won.mean())

        # Apply calibration
        params = self.calibration.get(self.format_type, self.gender)
        team1_win_prob = float(calibrate_probabilities(np.array([team1_win_prob_raw]), params)[0])

        return {
            "n_matches": n_matches,
            "team1_win_prob": team1_win_prob,
            "team2_win_prob": 1.0 - team1_win_prob,
            "team1_win_prob_raw": team1_win_prob_raw,
            "team2_win_prob_raw": 1.0 - team1_win_prob_raw,
            "calibration_used": (
                {"a": params.a, "b": params.b}
                if (params.a != 1.0 or params.b != 0.0)
                else None
            ),
            "avg_team1_score": float(first_runs.mean()),
            "avg_team2_score": float(second_runs.mean()),
            "std_team1_score": float(first_runs.std()),
            "std_team2_score": float(second_runs.std()),
            "team1_score_range": (float(np.percentile(first_runs, 5)), float(np.percentile(first_runs, 95))),
            "team2_score_range": (float(np.percentile(second_runs, 5)), float(np.percentile(second_runs, 95))),
            "team1_scores": first_runs,
            "team2_scores": second_runs,
            "dist_quality": dist_quality,
            "team1_elo_used": float(team1_elo),
            "team2_elo_used": float(team2_elo),
        }

    # ------------------------------------------------------------------
    # Inner: hierarchical innings simulation
    # ------------------------------------------------------------------

    def _simulate_innings(
        self,
        n_matches: int,
        innings: int,
        target,                       # int 0 or np.ndarray (n_matches,)
        max_overs: int,
        batter_vocab: np.ndarray,     # (n_batters,)
        bowler_vocab: np.ndarray,     # (n_bowlers,)
        batter_elos: np.ndarray,
        bowler_elos: np.ndarray,
        batting_team_vocab: int,
        bowling_team_vocab: int,
        batting_team_elo: float,
        bowling_team_elo: float,
        venue_vocab: int,
        venue_features: np.ndarray,
        era_norm: float,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Hierarchical innings simulator. Returns (runs, wkts) per match."""
        max_legal_balls = max_overs * 6
        # Headroom for extras-heavy overs
        max_total_balls = max_overs * 9

        # Per-match state
        runs = np.zeros(n_matches, dtype=np.int32)
        wkts = np.zeros(n_matches, dtype=np.int32)
        legal_balls = np.zeros(n_matches, dtype=np.int32)
        current_batter = np.zeros(n_matches, dtype=np.int32)
        free_hit = np.zeros(n_matches, dtype=bool)
        active = np.ones(n_matches, dtype=bool)

        # Per-over budgets refreshed at the start of each over
        over_runs_so_far = np.zeros(n_matches, dtype=np.int32)
        over_wkts_so_far = np.zeros(n_matches, dtype=np.int32)
        legal_balls_in_over = np.zeros(n_matches, dtype=np.int32)
        # Track current over index per match (changes as overs complete)
        over_idx = np.zeros(n_matches, dtype=np.int32)

        target_arr = (
            target.astype(np.int32) if isinstance(target, np.ndarray)
            else np.full(n_matches, int(target or 0), dtype=np.int32)
        )

        # Normalise constant ELO inputs
        bat_team_elo_n = float(normalize_elo(batting_team_elo))
        bowl_team_elo_n = float(normalize_elo(bowling_team_elo))
        team_elo_diff_n = float((batting_team_elo - bowling_team_elo) / 200.0)
        bowler_elo_norms = (bowler_elos - 1500.0) / 200.0
        batter_elo_norms = (batter_elos - 1500.0) / 200.0
        max_batter_idx = len(batter_vocab) - 1
        n_bowlers = len(bowler_vocab)

        # First-over budget sample for everyone
        over_runs_target, over_wkts_target = self._refresh_over_budgets(
            n_matches=n_matches,
            innings=innings,
            target_arr=target_arr,
            max_overs=max_overs,
            over_idx=over_idx,
            legal_balls=legal_balls,
            runs=runs,
            wkts=wkts,
            current_batter=current_batter,
            batter_vocab=batter_vocab,
            bowler_vocab=bowler_vocab,
            batter_elo_norms=batter_elo_norms,
            bowler_elo_norms=bowler_elo_norms,
            batting_team_vocab=batting_team_vocab,
            bowling_team_vocab=bowling_team_vocab,
            bat_team_elo_n=bat_team_elo_n,
            bowl_team_elo_n=bowl_team_elo_n,
            team_elo_diff_n=team_elo_diff_n,
            venue_vocab=venue_vocab,
            venue_features=venue_features,
            era_norm=era_norm,
            rng=rng,
        )

        for ball_idx in range(max_total_balls):
            if not active.any():
                break

            bowler_idx_now = (over_idx % n_bowlers)
            current_bowler_vocab = bowler_vocab[bowler_idx_now]
            current_bowler_elo_n = bowler_elo_norms[bowler_idx_now]

            # Build state matrix for the model
            np.clip(current_batter, 0, max_batter_idx, out=current_batter)
            current_batter_vocab = batter_vocab[current_batter]
            current_batter_elo_n = batter_elo_norms[current_batter]
            ball_in_over = (legal_balls_in_over + 1).astype(np.float32)

            X = self._build_state_features(
                n_matches=n_matches,
                innings=innings,
                over_idx=over_idx.astype(np.float32),
                ball_in_over=ball_in_over,
                legal_balls=legal_balls.astype(np.float32),
                runs=runs.astype(np.float32),
                wkts=wkts.astype(np.float32),
                target=int(target_arr.max() if isinstance(target_arr, np.ndarray) else 0),
                venue_features=venue_features,
                bat_team_elo_n=bat_team_elo_n,
                bowl_team_elo_n=bowl_team_elo_n,
                team_elo_diff_n=team_elo_diff_n,
                batter_elo_n=current_batter_elo_n,
                bowler_elo_n=float(current_bowler_elo_n) if np.isscalar(current_bowler_elo_n) else current_bowler_elo_n,
                era_norm=era_norm,
            )

            inputs = {
                "state": tf.constant(X),
                "batter_id": tf.constant(current_batter_vocab.astype(np.int32)),
                "bowler_id": tf.constant(np.full(n_matches, current_bowler_vocab, dtype=np.int32)),
                "venue_id": tf.constant(np.full(n_matches, venue_vocab, dtype=np.int32)),
                "batting_team_id": tf.constant(np.full(n_matches, batting_team_vocab, dtype=np.int32)),
                "bowling_team_id": tf.constant(np.full(n_matches, bowling_team_vocab, dtype=np.int32)),
            }
            try:
                outputs = self._predict_compiled(inputs, training=False)
            except Exception:
                outputs = self._model(inputs, training=False)
            ball_proba = outputs[self.ball_head_name].numpy()  # (n_matches, 9)

            # Hierarchical bias from over budgets
            legal_remaining = np.maximum(1, 6 - legal_balls_in_over)
            runs_remaining = np.maximum(0.0, over_runs_target - over_runs_so_far)
            wkts_remaining = np.maximum(0.0, over_wkts_target - over_wkts_so_far)
            rrpb = runs_remaining / legal_remaining
            ball_proba = self._bias_per_ball_logits(
                ball_proba, rrpb, wkts_remaining, alpha=self.config.over_budget_alpha
            )

            # Free-hit: zero out wicket prob for matches with free_hit=True
            # then renormalise.
            if free_hit.any():
                fh_mask = free_hit & active
                if fh_mask.any():
                    fh_idx = np.where(fh_mask)[0]
                    ball_proba[fh_idx, LABEL_WICKET] = 0.0
                    row_sums = ball_proba[fh_idx].sum(axis=1, keepdims=True)
                    row_sums = np.maximum(row_sums, 1e-8)
                    ball_proba[fh_idx] = ball_proba[fh_idx] / row_sums

            # Sample outcomes
            outcome = self._sample_outcomes(ball_proba, rng)
            outcome[~active] = LABEL_DOT  # don't update inactive matches

            # Resolve outcome -> state updates
            is_wide = outcome == LABEL_WIDE
            is_noball = outcome == LABEL_NOBALL
            is_extra = is_wide | is_noball
            is_legal = active & ~is_extra
            is_wicket = active & (outcome == LABEL_WICKET)

            # Runs
            runs_added = OUTCOME_RUNS_V2[outcome]
            runs += np.where(active, runs_added, 0)
            over_runs_so_far += np.where(active, runs_added, 0)

            # Wickets (only on legal balls; wicket outcome != wide/noball labels)
            wkts += np.where(is_wicket, 1, 0)
            over_wkts_so_far += np.where(is_wicket, 1, 0)

            # Free-hit transitions:
            # - Set free_hit on the next ball if THIS was a noball
            # - Clear free_hit on the next legal ball (it consumes the flag)
            new_free_hit = free_hit.copy()
            new_free_hit[is_noball] = True
            new_free_hit[is_legal] = False
            free_hit = new_free_hit

            # Move to next batter on wicket
            wicket_active = is_wicket & (wkts < 10)
            current_batter[wicket_active] = np.minimum(
                wkts[wicket_active] + 1, max_batter_idx
            )

            # Legal-ball counters (innings + over)
            legal_balls += is_legal.astype(np.int32)
            legal_balls_in_over += is_legal.astype(np.int32)

            # Over completion: 6 legal balls -> reset over state, advance over_idx,
            # resample over budget for matches that just finished an over
            over_complete = is_legal & (legal_balls_in_over >= 6)
            if over_complete.any():
                over_idx_next = over_idx.copy()
                over_idx_next[over_complete] += 1
                over_idx = over_idx_next
                legal_balls_in_over[over_complete] = 0
                over_runs_so_far[over_complete] = 0
                over_wkts_so_far[over_complete] = 0
                completed_idx = np.where(over_complete)[0]
                if len(completed_idx) > 0:
                    new_runs, new_wkts = self._refresh_over_budgets(
                        n_matches=len(completed_idx),
                        innings=innings,
                        target_arr=(
                            target_arr[completed_idx]
                            if isinstance(target_arr, np.ndarray)
                            else np.full(len(completed_idx), 0, dtype=np.int32)
                        ),
                        max_overs=max_overs,
                        over_idx=over_idx[completed_idx],
                        legal_balls=legal_balls[completed_idx],
                        runs=runs[completed_idx],
                        wkts=wkts[completed_idx],
                        current_batter=current_batter[completed_idx],
                        batter_vocab=batter_vocab,
                        bowler_vocab=bowler_vocab,
                        batter_elo_norms=batter_elo_norms,
                        bowler_elo_norms=bowler_elo_norms,
                        batting_team_vocab=batting_team_vocab,
                        bowling_team_vocab=bowling_team_vocab,
                        bat_team_elo_n=bat_team_elo_n,
                        bowl_team_elo_n=bowl_team_elo_n,
                        team_elo_diff_n=team_elo_diff_n,
                        venue_vocab=venue_vocab,
                        venue_features=venue_features,
                        era_norm=era_norm,
                        rng=rng,
                    )
                    over_runs_target[completed_idx] = new_runs
                    over_wkts_target[completed_idx] = new_wkts

            # Termination conditions
            all_out = wkts >= 10
            innings_done_balls = legal_balls >= max_legal_balls
            target_reached = (
                runs >= target_arr if innings == 2 else np.zeros(n_matches, dtype=bool)
            )
            active = active & ~all_out & ~innings_done_balls & ~target_reached

        return runs, wkts

    def _refresh_over_budgets(
        self,
        n_matches: int,
        innings: int,
        target_arr,
        max_overs: int,
        over_idx: np.ndarray,
        legal_balls: np.ndarray,
        runs: np.ndarray,
        wkts: np.ndarray,
        current_batter: np.ndarray,
        batter_vocab: np.ndarray,
        bowler_vocab: np.ndarray,
        batter_elo_norms: np.ndarray,
        bowler_elo_norms: np.ndarray,
        batting_team_vocab: int,
        bowling_team_vocab: int,
        bat_team_elo_n: float,
        bowl_team_elo_n: float,
        team_elo_diff_n: float,
        venue_vocab: int,
        venue_features: np.ndarray,
        era_norm: float,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample (runs_target, wkts_target) for the upcoming over from the
        per-over head. Returns shape (n_matches,) each."""
        max_batter_idx = len(batter_vocab) - 1
        n_bowlers = len(bowler_vocab)
        bowler_idx_now = (over_idx % n_bowlers)
        np.clip(current_batter, 0, max_batter_idx, out=current_batter)
        current_batter_vocab = batter_vocab[current_batter]
        current_bowler_vocab = bowler_vocab[bowler_idx_now]
        current_batter_elo_n = batter_elo_norms[current_batter]
        current_bowler_elo_n = bowler_elo_norms[bowler_idx_now]
        target_for_state = (
            int(target_arr.max() if isinstance(target_arr, np.ndarray) else 0)
            if innings == 2 else 0
        )
        X = self._build_state_features(
            n_matches=n_matches,
            innings=innings,
            over_idx=over_idx.astype(np.float32),
            ball_in_over=np.ones(n_matches, dtype=np.float32),
            legal_balls=legal_balls.astype(np.float32),
            runs=runs.astype(np.float32),
            wkts=wkts.astype(np.float32),
            target=target_for_state,
            venue_features=venue_features,
            bat_team_elo_n=bat_team_elo_n,
            bowl_team_elo_n=bowl_team_elo_n,
            team_elo_diff_n=team_elo_diff_n,
            batter_elo_n=current_batter_elo_n,
            bowler_elo_n=float(current_bowler_elo_n) if np.isscalar(current_bowler_elo_n) else current_bowler_elo_n,
            era_norm=era_norm,
        )
        inputs = {
            "state": tf.constant(X),
            "batter_id": tf.constant(current_batter_vocab.astype(np.int32)),
            "bowler_id": tf.constant(np.full(n_matches, current_bowler_vocab, dtype=np.int32)),
            "venue_id": tf.constant(np.full(n_matches, venue_vocab, dtype=np.int32)),
            "batting_team_id": tf.constant(np.full(n_matches, batting_team_vocab, dtype=np.int32)),
            "bowling_team_id": tf.constant(np.full(n_matches, bowling_team_vocab, dtype=np.int32)),
        }
        try:
            outputs = self._predict_compiled(inputs, training=False)
        except Exception:
            outputs = self._model(inputs, training=False)
        over_params = outputs[self.over_head_name].numpy()  # (n_matches, 4)
        return self._sample_over_targets(over_params, rng)
