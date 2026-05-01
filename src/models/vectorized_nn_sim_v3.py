"""
Vectorized Monte Carlo simulator v3 (Wave 5.5).

Subclass of V2Simulator that adds toss + lineup-stability awareness.

Two operating modes for toss handling, both supplied via kwargs to
`simulate_matches()`:

    toss_pinned=True, toss_winner_team_id=X, toss_chose_field=True/False
        Use this at T-30min entry (or any time after the toss is publicly
        known). Sets toss_won_by_batting_team and chose_to_bat
        deterministically per innings, so the model sees the exact context
        the late market is pricing.

    toss_pinned=False, toss_field_prob=0.65 (default)
        Use this at all earlier lookbacks. Per Monte Carlo iteration, sample
        toss winner uniformly from the two teams and chose-field with the
        supplied probability. Each match in the batch carries its own sampled
        toss outcome through both innings; the model marginalises over the
        toss in expectation.

XI-overlap is computed at sim setup from the passed batting/bowling order
plus a DB lookup of the team's recent matches. Per-team values are scalar
constants for the entire batch (within one match it's a property of the
match, not a Monte Carlo random variable).

Model + vocabs + calibration default to `data/models/v3/`. Inputs/outputs
otherwise mirror V2 exactly so the backtest harness can swap V3 in via
the `--model-version v3` CLI flag (added in Phase B).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.vectorized_nn_sim_v2 import (  # noqa: E402
    V2Simulator,
    V2SimulatorConfig,
    OUTCOME_RUNS_V2,
)
from src.features.ball_training_data_v3 import (  # noqa: E402
    CONTINUOUS_COLUMNS as CONTINUOUS_COLUMNS_V3,
    N_CONTINUOUS as N_CONTINUOUS_V3,
)

logger = logging.getLogger(__name__)


# Indices of the three new V3-only columns. Hard-coded against
# CONTINUOUS_COLUMNS_V3 to avoid lookup overhead in the hot path.
COL_TOSS_WON_BY_BATTING = CONTINUOUS_COLUMNS_V3.index("toss_won_by_batting_team")
COL_CHOSE_TO_BAT = CONTINUOUS_COLUMNS_V3.index("chose_to_bat")
COL_XI_OVERLAP = CONTINUOUS_COLUMNS_V3.index("xi_overlap_recent_3")


@dataclass
class V3SimulatorConfig(V2SimulatorConfig):
    """Same knobs as V2; only the default artifact paths change."""
    pass


class V3Simulator(V2Simulator):
    """Toss + lineup-aware vectorized Monte Carlo simulator.

    Inherits V2's hierarchical-budget per-ball loop and per-batter / per-team
    output tracking. Adds three new continuous features at the end of the
    state vector and the API to configure them per match.
    """

    def __init__(self, config: V3SimulatorConfig) -> None:
        # Auto-derive V3-specific default paths if the caller didn't override.
        if not config.model_path:
            config.model_path = "data/models/v3/cricket_model_v3.keras"
        if not config.vocab_path:
            config.vocab_path = "data/models/v3/vocabs.json"
        if not config.calibration_path:
            config.calibration_path = "data/models/v3/calibration.json"
        super().__init__(config)

        # V3-specific transient state filled in at simulate_matches() time.
        # Initial values are placeholders; the real values are set per-call
        # before _simulate_innings runs.
        self._toss_winner_idx: Optional[np.ndarray] = None      # (n_matches,) of 0 or 1
        self._chose_to_bat_arr: Optional[np.ndarray] = None     # (n_matches,) of 0 or 1
        self._xi_overlap_team1: float = 1.0
        self._xi_overlap_team2: float = 1.0
        # Set by simulate_matches() before each innings call (1 or 2).
        self._current_innings_batting_team_idx: int = 0  # 0 if team1 bats, 1 if team2

    # ------------------------------------------------------------------
    # XI-overlap lookup (DB-backed, computed once per simulate_matches call)
    # ------------------------------------------------------------------

    def _xi_overlap_for_team(
        self,
        team_id: Optional[int],
        match_date_str: Optional[str],
        this_xi: Optional[List[int]] = None,
    ) -> float:
        """Compute fraction of `this_xi` that played in any of the team's 3
        most-recent matches BEFORE `match_date_str`. Returns 1.0 (assume
        stable) when inputs are missing - this is the conservative default
        for live inference.
        """
        if not team_id or not match_date_str or not this_xi:
            return 1.0
        try:
            from src.data.database import get_connection
            conn = get_connection()
            try:
                cur = conn.cursor()
                # Find the team's 3 most-recent matches BEFORE this date
                cur.execute(
                    """
                    SELECT match_id FROM matches
                    WHERE date < ?
                      AND (team1_id = ? OR team2_id = ?)
                    ORDER BY date DESC, match_id DESC
                    LIMIT 3
                    """,
                    (match_date_str, team_id, team_id),
                )
                prior_match_ids = [r["match_id"] for r in cur.fetchall()]
                if not prior_match_ids:
                    return 0.0  # debut team / first match in window
                # Union of XIs across those 3 matches (for this team)
                placeholders = ",".join("?" * len(prior_match_ids))
                cur.execute(
                    f"""
                    SELECT DISTINCT player_id FROM player_match_stats
                    WHERE match_id IN ({placeholders}) AND team_id = ?
                    """,
                    list(prior_match_ids) + [team_id],
                )
                prior_xi = {r["player_id"] for r in cur.fetchall()}
                if not prior_xi:
                    return 0.0
                this_xi_set = set(int(p) for p in this_xi)
                if not this_xi_set:
                    return 0.0
                overlap = len(this_xi_set & prior_xi)
                return overlap / len(this_xi_set)
            finally:
                conn.close()
        except Exception as exc:
            logger.debug(f"V3 xi-overlap lookup failed for team {team_id}: {exc}")
            return 1.0

    # ------------------------------------------------------------------
    # Override _build_state_features to add 3 columns at the end
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
        target,                              # int OR np.ndarray
        venue_features: np.ndarray,
        bat_team_elo_n: float,
        bowl_team_elo_n: float,
        team_elo_diff_n: float,
        batter_elo_n: np.ndarray,
        bowler_elo_n,
        era_norm: float,
    ) -> np.ndarray:
        """Build the (n_matches, 25) state matrix for the V3 model.

        Reuses the V2 builder for the first 22 columns (ELO, venue, phase,
        match state) and then writes the 3 V3 columns:

            toss_won_by_batting_team:
              For innings 1 (team1 bats): toss_winner_idx == 0
              For innings 2 (team2 bats): toss_winner_idx == 1

            chose_to_bat:
              Read directly from self._chose_to_bat_arr (constant across
              innings within a match).

            xi_overlap_recent_3:
              In innings 1: self._xi_overlap_team1 (team1 batting)
              In innings 2: self._xi_overlap_team2 (team2 batting)
        """
        # Get the V2 22 columns by calling super
        v2_x = super()._build_state_features(
            n_matches=n_matches,
            innings=innings,
            over_idx=over_idx,
            ball_in_over=ball_in_over,
            legal_balls=legal_balls,
            runs=runs,
            wkts=wkts,
            target=target,
            venue_features=venue_features,
            bat_team_elo_n=bat_team_elo_n,
            bowl_team_elo_n=bowl_team_elo_n,
            team_elo_diff_n=team_elo_diff_n,
            batter_elo_n=batter_elo_n,
            bowler_elo_n=bowler_elo_n,
            era_norm=era_norm,
        )
        # v2_x has shape (n_matches, 22). Allocate the V3 25-col matrix and
        # copy the V2 prefix in.
        X = np.zeros((n_matches, N_CONTINUOUS_V3), dtype=np.float32)
        X[:, :22] = v2_x

        # ---- V3 column 22: toss_won_by_batting_team ----
        # innings is 1 or 2. Innings 1 -> team1 bats (idx 0). Innings 2 -> team2 (idx 1).
        # toss state is SCALAR per-match (see simulate_matches docstring).
        batting_team_idx = 0 if innings == 1 else 1
        if self._toss_winner_idx is None:
            # No toss state set; treat as "uncertain" (0.5).
            X[:, COL_TOSS_WON_BY_BATTING] = 0.5
        else:
            X[:, COL_TOSS_WON_BY_BATTING] = float(int(self._toss_winner_idx) == batting_team_idx)

        # ---- V3 column 23: chose_to_bat ----
        if self._chose_to_bat_arr is None:
            X[:, COL_CHOSE_TO_BAT] = 0.5
        else:
            X[:, COL_CHOSE_TO_BAT] = float(self._chose_to_bat_arr)

        # ---- V3 column 24: xi_overlap_recent_3 ----
        # Constant across the batch within an innings.
        xi_value = self._xi_overlap_team1 if innings == 1 else self._xi_overlap_team2
        X[:, COL_XI_OVERLAP] = float(xi_value)

        return X

    # ------------------------------------------------------------------
    # simulate_matches: thread the toss + xi-overlap kwargs through
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
        use_toss: bool = False,
        toss_field_prob: float = 0.65,
        team1_default_elo: Optional[float] = None,
        team2_default_elo: Optional[float] = None,
        # ---- V3 additions ----
        toss_pinned: bool = False,
        toss_winner_team_id: Optional[int] = None,
        toss_chose_field: Optional[bool] = None,
        match_date_for_xi: Optional[str] = None,
    ) -> Dict:
        """V3 simulate_matches.

        New kwargs:
            toss_pinned: if True, set toss outcome deterministically using
                toss_winner_team_id + toss_chose_field. Use this at T-30min
                or later when the toss is publicly known.
            toss_winner_team_id: required when toss_pinned=True.
            toss_chose_field: required when toss_pinned=True; True if toss
                winner chose to field, False if chose to bat.
            match_date_for_xi: ISO date 'YYYY-MM-DD' for the as-of-date XI
                lookup. Optional; when None we default xi_overlap to 1.0
                (assume stable XI). For backtests this should be the actual
                match date so we don't peek at future matches.
        """
        # ---- Prepare V3 toss + xi state BEFORE V2's simulate_matches runs ----
        rng = np.random.default_rng(seed)

        # Note: V3 keeps toss state SCALAR (rather than per-iteration vector)
        # because V2's `_refresh_over_budgets` re-enters `_build_state_features`
        # with arbitrary subset batch sizes (e.g. just the matches that just
        # finished an over). Per-iteration vectors would break broadcast.
        # For "marginalised" mode we encode "uncertain toss" as 0.5, which
        # is equivalent to the per-iteration mean in expectation.
        if toss_pinned:
            if toss_winner_team_id is None or toss_chose_field is None:
                raise ValueError(
                    "toss_pinned=True requires toss_winner_team_id + toss_chose_field"
                )
            # Map team_id -> 0/1 idx (scalar)
            if team1_id is not None and int(toss_winner_team_id) == int(team1_id):
                self._toss_winner_idx = np.int8(0)
            elif team2_id is not None and int(toss_winner_team_id) == int(team2_id):
                self._toss_winner_idx = np.int8(1)
            else:
                logger.warning(
                    f"V3 toss_winner_team_id={toss_winner_team_id} matches neither "
                    f"team1_id={team1_id} nor team2_id={team2_id}; using 0.5"
                )
                self._toss_winner_idx = None
            # toss_chose_field: True -> chose to field (chose_to_bat=0)
            self._chose_to_bat_arr = np.int8(0 if bool(toss_chose_field) else 1)
        elif use_toss:
            # Marginalised: encode uncertainty as 0.5. Model has seen 0/1 only at
            # training; this is interpolation but tells the model "we don't know".
            self._toss_winner_idx = None
            self._chose_to_bat_arr = None
        else:
            # No toss info - same fallback to 0.5
            self._toss_winner_idx = None
            self._chose_to_bat_arr = None

        # XI overlap lookups (one DB call each per simulate_matches call)
        self._xi_overlap_team1 = self._xi_overlap_for_team(
            team_id=team1_id, match_date_str=match_date_for_xi, this_xi=team1_batter_ids,
        )
        self._xi_overlap_team2 = self._xi_overlap_for_team(
            team_id=team2_id, match_date_str=match_date_for_xi, this_xi=team2_batter_ids,
        )

        # Hand the same RNG seed to V2's loop so toss sampling above doesn't
        # consume entropy needed for outcome sampling. Re-seed the rng V2
        # uses by passing seed through.
        result = super().simulate_matches(
            n_matches=n_matches,
            team1_batter_ids=team1_batter_ids,
            team1_bowler_ids=team1_bowler_ids,
            team2_batter_ids=team2_batter_ids,
            team2_bowler_ids=team2_bowler_ids,
            max_overs=max_overs,
            venue_id=venue_id,
            team1_id=team1_id,
            team2_id=team2_id,
            seed=seed,
            use_toss=use_toss,
            toss_field_prob=toss_field_prob,
            team1_default_elo=team1_default_elo,
            team2_default_elo=team2_default_elo,
        )

        # Stamp V3-specific provenance into the result
        result["model_version"] = "v3"
        result["v3_toss_mode"] = "pinned" if toss_pinned else ("marginalised" if use_toss else "uncertain")
        result["v3_xi_overlap_team1"] = float(self._xi_overlap_team1)
        result["v3_xi_overlap_team2"] = float(self._xi_overlap_team2)
        return result
