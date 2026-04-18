"""
Cricket Model v2 (Wave 4 Phase 2).

Multi-task neural network shared across all four (format, gender) combos.

Architecture (sketch):

  inputs:
    state_continuous     - shape (N_CONTINUOUS,) - the float32 row from
                           ball_training_data_v2 (incl. format_id, phase
                           one-hot, ELO features, era, venue stats, etc.)
    batter_id            - int32
    bowler_id            - int32
    venue_id             - int32
    batting_team_id      - int32  (already canonical)
    bowling_team_id      - int32  (already canonical)

  embeddings:
    batter_emb (24)  bowler_emb (24)  venue_emb (16)
    bat_team_emb (8)  bowl_team_emb (8)

  shared backbone:
    concat -> [Dense(256) BN ReLU Dropout] x 3

  per-(format, gender) heads (one of each per route_key):
    ball_logits     - Dense(9)             - softmax for the 9-class outcome
    over_params     - Dense(4)             - (mu_runs, log_sigma_runs,
                                              log_lambda_wkts, dispersion_unused)
                                              for Gaussian + Poisson NLL

  routing:
    Loss is computed only on the head matching the row's (format, gender).
    `route_key = format_id * 2 + gender_id`. Per-call masking via Keras
    `add_loss` lets us do this without 4 separate forward passes.

This file ONLY defines the architecture + losses. Training (chronological
splits, sample weights, class-weighted loss, early stopping on backtest
Brier) lives in `scripts/train_cricket_model_v2.py` (Phase 5).

Dependencies:
- Keras 3.x (matches the existing v1 stack so we share one TF runtime).
- Per-ball CCE uses `sample_weight` for the recency decay carried in the
  v2 npz; no need to re-implement.
- The per-over auxiliary loss aggregates a batch's per-ball labels into
  per-over (runs, wkts) pairs at training time using a small custom
  reduction (over-aggregator function); kept out of the model graph
  itself for simplicity.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Quiet TF startup
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

try:
    import keras
    from keras import layers, regularizers
except ImportError:  # pragma: no cover
    from tensorflow import keras  # type: ignore
    from tensorflow.keras import layers, regularizers  # type: ignore

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

NUM_CLASSES_V2 = 9   # mirrors ball_training_data_v2
ROUTE_KEYS = [
    ("T20", "male"),
    ("T20", "female"),
    ("ODI", "male"),
    ("ODI", "female"),
]

# Default embedding dims. Tunable from train script later.
DEFAULT_DIM_BATTER = 24
DEFAULT_DIM_BOWLER = 24
DEFAULT_DIM_VENUE = 16
DEFAULT_DIM_TEAM = 8

# Vocab convention: index 0 is reserved for <UNK> (unseen ids at inference
# time). Vocab size passed at construction must include +1 for the UNK slot.


@dataclass
class CricketModelV2Config:
    """Hyperparameters and shapes wired into the model graph."""
    n_continuous: int                       # state vector length (22 for v2)
    n_batters: int                          # vocab size INCL the UNK slot at index 0
    n_bowlers: int
    n_venues: int
    n_teams: int

    dim_batter: int = DEFAULT_DIM_BATTER
    dim_bowler: int = DEFAULT_DIM_BOWLER
    dim_venue: int = DEFAULT_DIM_VENUE
    dim_team: int = DEFAULT_DIM_TEAM

    hidden_units: Tuple[int, ...] = (256, 256, 256)
    dropout_rate: float = 0.30
    l2_reg: float = 1e-4
    learning_rate: float = 1e-3

    # Per-class weights for the 9-class CCE loss. None -> uniform.
    # Set in the train script via inverse-frequency clipping
    # (see Phase 2 plan).
    class_weights: Optional[Dict[int, float]] = None

    # Loss weighting (per-ball softmax vs per-over auxiliary)
    weight_ball_loss: float = 1.0
    weight_over_loss: float = 0.3


# ============================================================================
# Model construction
# ============================================================================


def _routed_outputs(
    name_prefix: str,
    n_units: int,
    backbone_output: keras.KerasTensor,
) -> Dict[Tuple[str, str], keras.KerasTensor]:
    """Build one Dense head per (format, gender) route from the backbone.

    Each head is a separate Dense; routing happens at the loss layer.
    """
    return {
        key: layers.Dense(n_units, name=f"{name_prefix}_{key[0].lower()}_{key[1]}")(
            backbone_output
        )
        for key in ROUTE_KEYS
    }


def build_cricket_model_v2(config: CricketModelV2Config) -> keras.Model:
    """Construct the v2 multi-task model.

    The model has multiple outputs (one ball-logits and one over-params
    head per route). Loss masking + routing happens during training in
    `scripts/train_cricket_model_v2.py` so this builder stays focused on
    the architecture.
    """
    state_in = keras.Input(shape=(config.n_continuous,), name="state", dtype="float32")
    batter_in = keras.Input(shape=(), name="batter_id", dtype="int32")
    bowler_in = keras.Input(shape=(), name="bowler_id", dtype="int32")
    venue_in = keras.Input(shape=(), name="venue_id", dtype="int32")
    bat_team_in = keras.Input(shape=(), name="batting_team_id", dtype="int32")
    bowl_team_in = keras.Input(shape=(), name="bowling_team_id", dtype="int32")

    batter_emb = layers.Embedding(
        config.n_batters, config.dim_batter, name="batter_embedding",
        mask_zero=False,
    )(batter_in)
    bowler_emb = layers.Embedding(
        config.n_bowlers, config.dim_bowler, name="bowler_embedding",
        mask_zero=False,
    )(bowler_in)
    venue_emb = layers.Embedding(
        config.n_venues, config.dim_venue, name="venue_embedding",
        mask_zero=False,
    )(venue_in)
    bat_team_emb = layers.Embedding(
        config.n_teams, config.dim_team, name="batting_team_embedding",
        mask_zero=False,
    )(bat_team_in)
    bowl_team_emb = layers.Embedding(
        config.n_teams, config.dim_team, name="bowling_team_embedding",
        mask_zero=False,
    )(bowl_team_in)

    # Embeddings are shape (batch, dim) - already flattened because input
    # shape is scalar. Concatenate with state vector.
    x = layers.Concatenate(name="all_features")([
        state_in,
        batter_emb,
        bowler_emb,
        venue_emb,
        bat_team_emb,
        bowl_team_emb,
    ])

    # Shared backbone
    for i, units in enumerate(config.hidden_units):
        x = layers.Dense(
            units,
            kernel_regularizer=regularizers.l2(config.l2_reg),
            name=f"backbone_dense_{i}",
        )(x)
        x = layers.BatchNormalization(name=f"backbone_bn_{i}")(x)
        x = layers.ReLU(name=f"backbone_relu_{i}")(x)
        x = layers.Dropout(config.dropout_rate, name=f"backbone_dropout_{i}")(x)

    # Per-(format, gender) heads.
    # Ball logits: shape (batch, 9). Softmax applied at loss time so we
    # can use sparse_categorical_crossentropy with from_logits=False after
    # an explicit softmax in the routing layer (see masked loss in train
    # script).
    ball_logits = _routed_outputs("ball_logits", NUM_CLASSES_V2, x)

    # Over-params: shape (batch, 4) per route.
    # Channels:
    #   [0] mu_runs       - mean runs per over (predicted)
    #   [1] log_sigma_runs - log std-dev of runs per over
    #   [2] log_lambda_wkts - log Poisson rate for wickets per over
    #   [3] reserved      - left for v2 extensions (e.g. dispersion)
    over_params = _routed_outputs("over_params", 4, x)

    # Bundle outputs as a flat dict so `model.predict()` returns one entry
    # per (head, route).
    outputs = {}
    for key in ROUTE_KEYS:
        outputs[f"ball_{key[0].lower()}_{key[1]}"] = layers.Activation(
            "softmax", name=f"ball_softmax_{key[0].lower()}_{key[1]}"
        )(ball_logits[key])
        outputs[f"over_{key[0].lower()}_{key[1]}"] = over_params[key]

    model = keras.Model(
        inputs={
            "state": state_in,
            "batter_id": batter_in,
            "bowler_id": bowler_in,
            "venue_id": venue_in,
            "batting_team_id": bat_team_in,
            "bowling_team_id": bowl_team_in,
        },
        outputs=outputs,
        name="cricket_model_v2",
    )
    return model


# ============================================================================
# Loss helpers (used by the Phase 5 training script)
# ============================================================================


def routed_cce_loss_fn(
    class_weights: Optional[Dict[int, float]] = None,
):
    """Return a callable suitable for Keras `loss=` per-output assignment.

    The callable takes (y_true, y_pred) where y_true is the 9-class label
    (int8 in our npz) and y_pred is the softmax output of one route head.
    Sample weights are passed via the `sample_weight` arg of `model.fit`.

    When class_weights is provided, applies per-sample weight = class_weights[y_true]
    multiplicatively. Combined with sample_weight (recency decay) in the
    train loop.

    Notes:
    - The training loop must mask non-routed rows (rows whose actual
      (format, gender) doesn't match this head's route) to zero so they
      don't contribute. The simplest pattern: build per-route sample
      weights that are 0 on non-route rows and sample_weight * class_weight
      on route rows.
    """
    import tensorflow as tf

    if class_weights is None:
        weight_lookup = None
    else:
        # Index 0..8 -> weight; missing -> 1.0
        wt = np.ones(NUM_CLASSES_V2, dtype=np.float32)
        for cls, w in class_weights.items():
            if 0 <= cls < NUM_CLASSES_V2:
                wt[cls] = float(w)
        weight_lookup = tf.constant(wt, dtype=tf.float32)

    def loss(y_true, y_pred):
        # Sparse categorical crossentropy per sample.
        y_true_int = tf.cast(y_true, tf.int32)
        per_sample = tf.keras.losses.sparse_categorical_crossentropy(
            y_true_int, y_pred, from_logits=False
        )
        if weight_lookup is not None:
            cls_w = tf.gather(weight_lookup, y_true_int)
            per_sample = per_sample * cls_w
        return per_sample

    return loss


def over_nll_loss_fn():
    """Per-over auxiliary loss: Gaussian NLL over runs + Poisson NLL over wkts.

    The training loop is responsible for AGGREGATING per-ball rows into
    (runs_in_over, wkts_in_over) labels and feeding the corresponding
    over-params head. Here we just compute the loss given matched
    (y_true, y_pred) tensors of shape (batch, 2) and (batch, 4) respectively.

    y_true columns: [actual_runs_in_over, actual_wkts_in_over]
    y_pred columns: [mu_runs, log_sigma_runs, log_lambda_wkts, _reserved]
    """
    import tensorflow as tf

    def loss(y_true, y_pred):
        # y_true shape: (batch, 2) - [runs, wkts]
        # y_pred shape: (batch, 4) - [mu, log_sigma, log_lambda, _]
        runs_true = tf.cast(y_true[:, 0], tf.float32)
        wkts_true = tf.cast(y_true[:, 1], tf.float32)
        mu = y_pred[:, 0]
        log_sigma = tf.clip_by_value(y_pred[:, 1], -3.0, 3.0)
        log_lambda = tf.clip_by_value(y_pred[:, 2], -2.0, 3.0)

        # Gaussian NLL: 0.5 * ((y - mu) / sigma)^2 + log_sigma + 0.5*log(2pi)
        sigma = tf.exp(log_sigma)
        gaussian_nll = 0.5 * tf.square((runs_true - mu) / sigma) + log_sigma + 0.5 * np.log(2 * np.pi)

        # Poisson NLL: lambda - y*log(lambda) + log(y!)
        # We drop log(y!) since it's a constant wrt parameters.
        lam = tf.exp(log_lambda)
        poisson_nll = lam - wkts_true * log_lambda

        return gaussian_nll + poisson_nll

    return loss


# ============================================================================
# Vocab helpers
# ============================================================================


def build_vocab(values: np.ndarray, min_count: int = 1) -> Dict[int, int]:
    """Map raw ids -> compact vocab indices, reserving 0 for <UNK>.

    Returns a dict: raw_id -> vocab_index (in range [1, len(unique_ids)+1)).
    Anything not in the dict at inference time gets mapped to 0 (UNK).

    `min_count` lets us drop ultra-rare ids (rare players who have a single
    delivery in 10 years would otherwise add an embedding row with 24 noise
    parameters); they collapse into the UNK token instead. Default 1 keeps
    everyone.
    """
    unique, counts = np.unique(values, return_counts=True)
    vocab: Dict[int, int] = {}
    next_idx = 1  # 0 reserved for UNK
    for raw_id, count in zip(unique, counts):
        if count >= min_count and raw_id != 0:
            vocab[int(raw_id)] = next_idx
            next_idx += 1
    return vocab


def apply_vocab(values: np.ndarray, vocab: Dict[int, int]) -> np.ndarray:
    """Apply a vocab mapping to a column. Unknown ids -> 0 (UNK)."""
    out = np.zeros_like(values, dtype=np.int32)
    for i, v in enumerate(values):
        out[i] = vocab.get(int(v), 0)
    return out


def vocabs_from_npz(npz_path: str, min_count: int = 1) -> Dict[str, Dict[int, int]]:
    """Build the four embedding vocabs from a v2 training npz.

    Returns a dict with keys: batter, bowler, venue, team.
    """
    data = np.load(npz_path, allow_pickle=True)
    ids = data["ids"]
    id_columns = list(data["id_columns"])

    def _col(name: str) -> np.ndarray:
        return ids[:, id_columns.index(name)]

    teams = np.concatenate([_col("batting_team_id"), _col("bowling_team_id")])
    return {
        "batter": build_vocab(_col("batter_id"), min_count=min_count),
        "bowler": build_vocab(_col("bowler_id"), min_count=min_count),
        "venue": build_vocab(_col("venue_id"), min_count=min_count),
        "team": build_vocab(teams, min_count=min_count),
    }
