"""
Cricket Model v3 (Wave 5.5).

Architecturally identical to V2 except the state vector is N_CONTINUOUS=25
instead of 22. The 3 extra continuous features come from
`ball_training_data_v3`:

    toss_won_by_batting_team  (binary)
    chose_to_bat              (binary)
    xi_overlap_recent_3       (0..1)

Everything else - shared backbone, embeddings, per-(format,gender) heads,
loss helpers, vocab utilities - is reused verbatim from cricket_model_v2.
This file exists so callers can `from src.models.cricket_model_v3 import ...`
and get a v3-tagged surface, while we don't fork the model code.

The output dict keys match V2 exactly (`ball_t20_male`, `over_t20_male`,
etc.) so the V3 simulator can be a near-drop-in for the V2 simulator.

Save trained weights to `data/models/v3/cricket_model_v3.keras`.
"""

from __future__ import annotations

# Re-export every V2 helper unchanged so v3 callers have one import surface
from src.models.cricket_model_v2 import (  # noqa: F401
    NUM_CLASSES_V2,
    ROUTE_KEYS,
    DEFAULT_DIM_BATTER,
    DEFAULT_DIM_BOWLER,
    DEFAULT_DIM_VENUE,
    DEFAULT_DIM_TEAM,
    CricketModelV2Config,
    build_cricket_model_v2,
    routed_cce_loss_fn,
    over_nll_loss_fn,
    build_vocab,
    apply_vocab,
    vocabs_from_npz,
)


# V3 default for state vector length. Must match the length of
# `src.features.ball_training_data_v3.CONTINUOUS_COLUMNS`. If you bump
# v3 features, bump this too.
N_CONTINUOUS_V3 = 25


def make_v3_config(
    n_batters: int,
    n_bowlers: int,
    n_venues: int,
    n_teams: int,
    **kwargs,
) -> CricketModelV2Config:
    """Build a CricketModelV2Config with V3's default n_continuous=25.

    Any other V2Config field (hidden_units, dropout, dim_batter, etc.)
    can be overridden via kwargs. We keep this as a factory rather than a
    subclass to sidestep dataclass field-order rules.
    """
    kwargs.setdefault("n_continuous", N_CONTINUOUS_V3)
    return CricketModelV2Config(
        n_batters=n_batters,
        n_bowlers=n_bowlers,
        n_venues=n_venues,
        n_teams=n_teams,
        **kwargs,
    )


# Alias for symmetry with V2's `CricketModelV2Config` import surface.
# Callers can `from src.models.cricket_model_v3 import CricketModelV3Config`
# and use it like a constructor.
CricketModelV3Config = make_v3_config


def build_cricket_model_v3(config: CricketModelV2Config):
    """Construct the V3 multi-task model.

    Calls build_cricket_model_v2 with the supplied config. The architecture
    is identical to V2; the only difference is the input state vector
    length (25 vs 22), which the config already carries. Set the model
    name to 'cricket_model_v3' for provenance.
    """
    model = build_cricket_model_v2(config)
    model._name = "cricket_model_v3"
    return model
