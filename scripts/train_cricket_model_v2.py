#!/usr/bin/env python3
"""
Train the v2 multi-task cricket model (Wave 4 Phase 5).

Loads ball_training_v2_{male,female}.npz, builds vocabs (with UNK + min_count),
constructs cricket_model_v2 with the right vocab sizes, trains all four
(format, gender) heads jointly with class-weighted CCE for the per-ball
output and Gaussian/Poisson NLL for the per-over auxiliary.

Multi-task routing: each example only contributes loss to its (format, gender)
head. Implemented via per-output sample_weight masks (0 on non-route rows;
sample_weight * class_weight on route rows).

Run:
    # Build training data first if not done yet
    python scripts/build_ball_training_v2.py --gender both

    # Train
    python scripts/train_cricket_model_v2.py

    # Smoke / quick iteration
    python scripts/train_cricket_model_v2.py --epochs 3 --limit-rows 50000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import keras  # noqa: F401
    import tensorflow as tf
except ImportError:  # pragma: no cover
    raise SystemExit("Training v2 requires tensorflow + keras")

from src.features.ball_training_data_v2 import (  # noqa: E402
    CONTINUOUS_COLUMNS,
    FORMAT_ID,
    GENDER_ID,
    ID_COLUMNS,
    NUM_CLASSES_V2,
    LABEL_NAMES,
)
from src.models.cricket_model_v2 import (  # noqa: E402
    CricketModelV2Config,
    ROUTE_KEYS,
    build_cricket_model_v2,
    build_vocab,
    apply_vocab,
    over_nll_loss_fn,
    routed_cce_loss_fn,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data loading & vocab build
# ============================================================================


def load_npz(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Training data not found at {path}; run "
            f"scripts/build_ball_training_v2.py first"
        )
    data = np.load(path, allow_pickle=True)
    expected = {"X", "ids", "y", "sample_weight", "over_targets"}
    missing = expected - set(data.files)
    if missing:
        raise ValueError(f"{path} missing fields: {missing}")
    return data


def concat_genders(*archives) -> Dict[str, np.ndarray]:
    """Concatenate v2 npzs (one per gender) into a single in-memory dict.

    All files share schema (CONTINUOUS_COLUMNS / ID_COLUMNS / LABEL_NAMES)
    so we only need to vstack the X / ids / y / sample_weight / over_targets
    arrays. Date-ordered concatenation: each input is already chronological,
    and concatenating preserves a consistent global ordering only IF you
    later sort by (era, date) - we sort by sample_weight as a proxy because
    the npzs already came out sorted by date.
    """
    out: Dict[str, np.ndarray] = {}
    for k in ("X", "ids", "y", "sample_weight", "over_targets"):
        out[k] = np.concatenate([a[k] for a in archives])
    return out


def chronological_split(
    arrays: Dict[str, np.ndarray],
    val_fraction: float = 0.10,
    test_fraction: float = 0.05,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Split arrays into train / val / test by sample-weight ordering.

    sample_weight is a strict function of match age (recency decay), so
    sorting by sample_weight ascending puts old matches first and recent
    matches last. We hold out the most-recent `val_fraction + test_fraction`
    of rows for val + test - then split the held-out tail in half.
    """
    n = len(arrays["y"])
    order = np.argsort(arrays["sample_weight"], kind="stable")
    train_end = int(n * (1.0 - val_fraction - test_fraction))
    val_end = int(n * (1.0 - test_fraction))
    train_idx = order[:train_end]
    val_idx = order[train_end:val_end]
    test_idx = order[val_end:]

    def _slice(idx):
        return {k: v[idx] for k, v in arrays.items()}

    return _slice(train_idx), _slice(val_idx), _slice(test_idx)


# ============================================================================
# Multi-task routing
# ============================================================================


def route_index_for_row(format_id: int, gender_id: int) -> int:
    """Map (format, gender) ids to a route index in [0, 4)."""
    # Order: T20_male, T20_female, ODI_male, ODI_female. Matches ROUTE_KEYS.
    return format_id * 2 + gender_id


def build_per_route_sample_weights(
    arrays: Dict[str, np.ndarray],
    class_weights: Dict[int, float],  # noqa: ARG001 - kept for API compat
) -> Dict[str, np.ndarray]:
    """Build per-route sample weight arrays for the multi-task loss.

    For each output head, returns an array of shape (n,) with:
      - 0.0 on rows whose (format, gender) doesn't match the head's route.
      - sample_weight[i] (the recency-decay weight) on rows that do.

    Class weights are NOT multiplied in here. They're applied INSIDE the
    routed CCE loss function (cricket_model_v2.routed_cce_loss_fn) so
    multiplying them in here too would square the effective per-class
    weighting (cls_w * cls_w), wildly over-emphasising rare classes - in
    the v1 training run this caused wicket prob to be ~6x what it should
    have been, collapsing 2nd innings in the simulator. Bug discovered
    when the V2 trained model produced team2 averages of 50 runs vs
    team1 187, with team1_win_prob = 0.000 across 44 backtest matches.

    For per-over heads we use sample_weight directly (no class weighting -
    the over-level loss is regression, not classification).
    """
    n = len(arrays["y"])
    fmt_col = arrays["ids"][:, ID_COLUMNS.index("format_id")]
    gen_col = arrays["ids"][:, ID_COLUMNS.index("gender_id")]
    base_sw = arrays["sample_weight"]

    out: Dict[str, np.ndarray] = {}
    for fmt, gen in ROUTE_KEYS:
        fmt_id = FORMAT_ID[fmt]
        gen_id = GENDER_ID[gen]
        mask = (fmt_col == fmt_id) & (gen_col == gen_id)
        sw = np.where(mask, base_sw, 0.0).astype(np.float32)
        out[f"ball_{fmt.lower()}_{gen}"] = sw
        out[f"over_{fmt.lower()}_{gen}"] = sw
    return out


def build_targets_dict(arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Build the y_true dict for model.fit. Per-route ball heads share the
    same int label vector; per-route over heads share the (over_runs,
    over_wkts) target. The masking above ensures only the right route
    contributes loss for each row.
    """
    y_ball = arrays["y"].astype(np.int32)
    y_over = arrays["over_targets"].astype(np.float32)
    out: Dict[str, np.ndarray] = {}
    for fmt, gen in ROUTE_KEYS:
        out[f"ball_{fmt.lower()}_{gen}"] = y_ball
        out[f"over_{fmt.lower()}_{gen}"] = y_over
    return out


def build_inputs_dict(arrays: Dict[str, np.ndarray], vocabs: Dict[str, Dict[int, int]]) -> Dict[str, np.ndarray]:
    """Build the model's input dict. Categorical IDs are vocab-mapped here;
    everything outside the vocab collapses to UNK (index 0)."""
    ids = arrays["ids"]
    raw_batter = ids[:, ID_COLUMNS.index("batter_id")]
    raw_bowler = ids[:, ID_COLUMNS.index("bowler_id")]
    raw_venue = ids[:, ID_COLUMNS.index("venue_id")]
    raw_bat_team = ids[:, ID_COLUMNS.index("batting_team_id")]
    raw_bowl_team = ids[:, ID_COLUMNS.index("bowling_team_id")]

    return {
        "state": arrays["X"].astype(np.float32),
        "batter_id": apply_vocab(raw_batter, vocabs["batter"]),
        "bowler_id": apply_vocab(raw_bowler, vocabs["bowler"]),
        "venue_id": apply_vocab(raw_venue, vocabs["venue"]),
        "batting_team_id": apply_vocab(raw_bat_team, vocabs["team"]),
        "bowling_team_id": apply_vocab(raw_bowl_team, vocabs["team"]),
    }


def compute_class_weights(y: np.ndarray, clip_min: float = 0.7, clip_max: float = 1.5) -> Dict[int, float]:
    """Inverse-frequency class weights, clipped so the rarest classes don't
    blow up the loss.

    The first v2 training run used clip_max=5.0 which over-emphasised rare
    classes (wickets, noballs) so much that the per-ball softmax produced
    ~12% wicket prob (vs ~5% in reality), causing 2nd-innings batting
    collapse in the simulator. clip range tightened to [0.7, 1.5] for the
    next pass; v1 didn't use class weights at all and produced more
    realistic ball distributions, so we err on the side of trusting
    natural class frequencies more.
    """
    counts = np.bincount(y, minlength=NUM_CLASSES_V2).astype(np.float64)
    total = counts.sum()
    weights: Dict[int, float] = {}
    for c in range(NUM_CLASSES_V2):
        if counts[c] == 0:
            weights[c] = 1.0
        else:
            inv_freq = total / (NUM_CLASSES_V2 * counts[c])
            weights[c] = float(np.clip(inv_freq, clip_min, clip_max))
    return weights


# ============================================================================
# Main training loop
# ============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="Train cricket model v2 multi-task")
    parser.add_argument("--data-dir", default="data/processed", help="Where ball_training_v2_{gender}.npz lives")
    parser.add_argument("--output-dir", default="data/models/v2", help="Where to write the model + vocabs + summary")
    parser.add_argument("--genders", nargs="+", default=["male", "female"], choices=["male", "female"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--vocab-min-count", type=int, default=5,
                        help="Drop ids with < min_count occurrences (collapse to UNK).")
    parser.add_argument("--limit-rows", type=int, default=None,
                        help="Cap total rows for smoke testing.")
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--label", default=None,
                        help="Optional model label (defaults to a timestamp).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or time.strftime("%Y%m%d_%H%M%S")

    # ----- Load + concat genders -----
    archives = []
    for g in args.genders:
        path = Path(args.data_dir) / f"ball_training_v2_{g}.npz"
        logger.info(f"loading {path}")
        archives.append(load_npz(path))
    arrays = concat_genders(*archives)

    if args.limit_rows is not None and args.limit_rows < len(arrays["y"]):
        # Take the most-recent limit_rows by sample weight (largest first)
        order = np.argsort(arrays["sample_weight"], kind="stable")[::-1][:args.limit_rows]
        arrays = {k: v[order] for k, v in arrays.items()}
        logger.info(f"limited to {len(arrays['y']):,} rows (most-recent by sample_weight)")
    n_total = len(arrays["y"])
    logger.info(f"total rows: {n_total:,}")

    # ----- Vocab build (training set only ideally; using full set is fine
    #       because vocab itself doesn't leak label info) -----
    ids = arrays["ids"]
    vocab_batter = build_vocab(ids[:, ID_COLUMNS.index("batter_id")], min_count=args.vocab_min_count)
    vocab_bowler = build_vocab(ids[:, ID_COLUMNS.index("bowler_id")], min_count=args.vocab_min_count)
    vocab_venue = build_vocab(ids[:, ID_COLUMNS.index("venue_id")], min_count=args.vocab_min_count)
    teams = np.concatenate([
        ids[:, ID_COLUMNS.index("batting_team_id")],
        ids[:, ID_COLUMNS.index("bowling_team_id")],
    ])
    vocab_team = build_vocab(teams, min_count=args.vocab_min_count)
    vocabs = {
        "batter": vocab_batter,
        "bowler": vocab_bowler,
        "venue": vocab_venue,
        "team": vocab_team,
    }
    logger.info(
        f"vocabs: batter={len(vocab_batter)}  bowler={len(vocab_bowler)}  "
        f"venue={len(vocab_venue)}  team={len(vocab_team)}"
    )

    # Persist vocabs early so partial training can still produce a usable
    # bundle if interrupted.
    with open(output_dir / "vocabs.json", "w") as fp:
        json.dump({k: {str(rid): vid for rid, vid in v.items()} for k, v in vocabs.items()}, fp)

    # ----- Chronological split -----
    train, val, test = chronological_split(
        arrays, val_fraction=args.val_fraction, test_fraction=args.test_fraction
    )
    logger.info(
        f"split: train={len(train['y']):,}  val={len(val['y']):,}  test={len(test['y']):,}"
    )

    # ----- Class weights from train labels -----
    class_weights = compute_class_weights(train["y"])
    logger.info("class weights (clipped inverse frequency):")
    for c in range(NUM_CLASSES_V2):
        logger.info(f"  {LABEL_NAMES[c]:>7}: {class_weights[c]:.3f}")

    # ----- Build model -----
    cfg = CricketModelV2Config(
        n_continuous=len(CONTINUOUS_COLUMNS),
        # +1 for the UNK slot at index 0
        n_batters=len(vocab_batter) + 1,
        n_bowlers=len(vocab_bowler) + 1,
        n_venues=len(vocab_venue) + 1,
        n_teams=len(vocab_team) + 1,
        learning_rate=args.learning_rate,
        class_weights=class_weights,
    )
    model = build_cricket_model_v2(cfg)
    logger.info(f"model params: {model.count_params():,}")

    # ----- Compile -----
    losses = {}
    loss_weights = {}
    cce = routed_cce_loss_fn(class_weights)
    nll = over_nll_loss_fn()
    for fmt, gen in ROUTE_KEYS:
        ball_name = f"ball_{fmt.lower()}_{gen}"
        over_name = f"over_{fmt.lower()}_{gen}"
        losses[ball_name] = cce
        losses[over_name] = nll
        loss_weights[ball_name] = cfg.weight_ball_loss
        loss_weights[over_name] = cfg.weight_over_loss

    optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        # No accuracy metric here; per-route accuracy would require dynamic
        # masking. We rely on the backtest harness for the real eval signal.
    )

    # ----- Build inputs / targets / per-route sample weights -----
    train_x = build_inputs_dict(train, vocabs)
    val_x = build_inputs_dict(val, vocabs)
    train_y = build_targets_dict(train)
    val_y = build_targets_dict(val)
    train_sw = build_per_route_sample_weights(train, class_weights)
    val_sw = build_per_route_sample_weights(val, class_weights)

    # ----- Train -----
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.early_stopping_patience, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "cricket_model_v2.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    t0 = time.time()
    history = model.fit(
        train_x,
        train_y,
        sample_weight=train_sw,
        validation_data=(val_x, val_y, val_sw),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    elapsed = time.time() - t0
    logger.info(f"training complete in {elapsed:.1f}s")

    # ----- Save normalizer for the state vector (mean/std fit on train) -----
    train_X = train["X"].astype(np.float64)
    norm = {
        "mean": train_X.mean(axis=0).astype(np.float32),
        "std":  np.maximum(train_X.std(axis=0), 1e-6).astype(np.float32),
        "columns": list(CONTINUOUS_COLUMNS),
    }
    np.savez(output_dir / "cricket_model_v2_normalizer.npz", **norm)

    # ----- Write summary JSON -----
    summary = {
        "label": label,
        "args": vars(args),
        "n_train": len(train["y"]),
        "n_val": len(val["y"]),
        "n_test": len(test["y"]),
        "elapsed_seconds": round(elapsed, 1),
        "vocab_sizes": {k: len(v) for k, v in vocabs.items()},
        "class_weights": {LABEL_NAMES[c]: w for c, w in class_weights.items()},
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
        "model_params": int(model.count_params()),
    }
    with open(output_dir / f"cricket_model_v2_summary_{label}.json", "w") as fp:
        json.dump(summary, fp, indent=2, default=str)

    print()
    print("=" * 60)
    print(f"Train summary  ({label})")
    print("=" * 60)
    print(f"  rows train/val/test : {len(train['y']):,} / {len(val['y']):,} / {len(test['y']):,}")
    print(f"  vocabs              : {summary['vocab_sizes']}")
    print(f"  model params        : {summary['model_params']:,}")
    print(f"  final val loss      : {summary['history'].get('val_loss', [None])[-1]}")
    print(f"  best val loss       : {min(summary['history'].get('val_loss', [float('inf')]))}")
    print(f"  saved to            : {output_dir / 'cricket_model_v2.keras'}")
    print(f"  vocabs              : {output_dir / 'vocabs.json'}")
    print(f"  normalizer          : {output_dir / 'cricket_model_v2_normalizer.npz'}")
    print(f"  summary             : {output_dir / f'cricket_model_v2_summary_{label}.json'}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
