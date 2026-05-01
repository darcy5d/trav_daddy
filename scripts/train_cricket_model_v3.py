#!/usr/bin/env python3
"""
Train the v3 multi-task cricket model (Wave 5.5 Phase A3).

Same training loop as V2 but:
- Loads ball_training_v3_{gender}.npz (25-column state vector incl. toss + XI overlap)
- Writes to data/models/v3/cricket_model_v3.keras + vocabs.json + normalizer + summary
- Defaults bake in the Wave 4.5 winning recipe (hidden=512, uniform class weights,
  over_loss_weight=0.1, embeddings 32/24/12)

Run:
    # Build training data first
    python scripts/build_ball_training_v3.py --gender both

    # Train (1-2 hours)
    python scripts/train_cricket_model_v3.py

    # Smoke / quick iteration
    python scripts/train_cricket_model_v3.py --epochs 3 --limit-rows 50000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import keras
    import tensorflow as tf  # noqa: F401
except ImportError:  # pragma: no cover
    raise SystemExit("Training v3 requires tensorflow + keras")

from src.features.ball_training_data_v3 import (  # noqa: E402
    CONTINUOUS_COLUMNS,
    ID_COLUMNS,
)
from src.features.ball_training_data_v2 import (  # noqa: E402
    FORMAT_ID,
    GENDER_ID,
    NUM_CLASSES_V2,
    LABEL_NAMES,
)
from src.models.cricket_model_v3 import (  # noqa: E402
    CricketModelV3Config,
    ROUTE_KEYS,
    build_cricket_model_v3,
    build_vocab,
    apply_vocab,
    over_nll_loss_fn,
    routed_cce_loss_fn,
)

# Re-use V2's training utilities directly (load_npz, splits, masks, etc.)
# Identical logic; only paths and import sources differ.
from scripts.train_cricket_model_v2 import (  # noqa: E402
    load_npz,
    concat_genders,
    chronological_split,
    build_per_route_sample_weights,
    build_targets_dict,
    build_inputs_dict,
    compute_class_weights,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train cricket model v3 multi-task (toss + lineup-aware)")
    parser.add_argument("--data-dir", default="data/processed", help="Where ball_training_v3_{gender}.npz lives")
    parser.add_argument("--output-dir", default="data/models/v3", help="Where to write the V3 model bundle")
    parser.add_argument("--genders", nargs="+", default=["male", "female"], choices=["male", "female"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--vocab-min-count", type=int, default=5)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--label", default=None)
    # Wave 4.5 winning recipe is the default for V3
    parser.add_argument(
        "--class-weight-mode",
        choices=["inverse-freq", "uniform", "boundary-only"],
        default="uniform",
        help="Default 'uniform' = Wave 4.5 winner.",
    )
    parser.add_argument("--over-loss-weight", type=float, default=0.1,
                        help="Default 0.1 = Wave 4.5 winner (was 0.3).")
    parser.add_argument("--hidden-units", type=int, default=512,
                        help="Default 512 = Wave 4.5 winner.")
    parser.add_argument("--n-hidden-layers", type=int, default=3)
    parser.add_argument("--embedding-dim-batter", type=int, default=32,
                        help="Default 32 = Wave 4.5 winner.")
    parser.add_argument("--embedding-dim-venue", type=int, default=24,
                        help="Default 24 = Wave 4.5 winner.")
    parser.add_argument("--embedding-dim-team", type=int, default=12,
                        help="Default 12 = Wave 4.5 winner.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or time.strftime("%Y%m%d_%H%M%S")

    # ----- Load + concat genders (V3 npz) -----
    archives = []
    for g in args.genders:
        path = Path(args.data_dir) / f"ball_training_v3_{g}.npz"
        if not path.exists():
            raise SystemExit(
                f"V3 training data not found at {path}. Run "
                f"scripts/build_ball_training_v3.py first."
            )
        logger.info(f"loading {path}")
        archives.append(load_npz(path))
    arrays = concat_genders(*archives)

    # Sanity: V3 npz should have 25-col state vector
    n_cols = arrays["X"].shape[1]
    if n_cols != len(CONTINUOUS_COLUMNS):
        raise SystemExit(
            f"V3 state vector has {n_cols} columns; expected "
            f"{len(CONTINUOUS_COLUMNS)} ({CONTINUOUS_COLUMNS}). "
            f"Did you build with build_ball_training_v3.py?"
        )

    if args.limit_rows is not None and args.limit_rows < len(arrays["y"]):
        order = np.argsort(arrays["sample_weight"], kind="stable")[::-1][:args.limit_rows]
        arrays = {k: v[order] for k, v in arrays.items()}
        logger.info(f"limited to {len(arrays['y']):,} rows (most-recent by sample_weight)")
    n_total = len(arrays["y"])
    logger.info(f"total rows: {n_total:,}  (state cols: {n_cols})")

    # ----- Vocab build -----
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

    with open(output_dir / "vocabs.json", "w") as fp:
        json.dump({k: {str(rid): vid for rid, vid in v.items()} for k, v in vocabs.items()}, fp)

    # ----- Chronological split -----
    train, val, test = chronological_split(
        arrays, val_fraction=args.val_fraction, test_fraction=args.test_fraction
    )
    logger.info(
        f"split: train={len(train['y']):,}  val={len(val['y']):,}  test={len(test['y']):,}"
    )

    # ----- Class weights -----
    if args.class_weight_mode == "uniform":
        class_weights = {c: 1.0 for c in range(NUM_CLASSES_V2)}
        logger.info("class weights: UNIFORM (Wave 4.5 winner)")
    elif args.class_weight_mode == "boundary-only":
        class_weights = {c: 1.0 for c in range(NUM_CLASSES_V2)}
        class_weights[4] = 1.5
        class_weights[5] = 1.5
        logger.info("class weights: BOUNDARY-ONLY")
    else:
        class_weights = compute_class_weights(train["y"])
        logger.info("class weights: INVERSE-FREQ (clipped)")
    for c in range(NUM_CLASSES_V2):
        logger.info(f"  {LABEL_NAMES[c]:>7}: {class_weights[c]:.3f}")

    # ----- Build V3 model -----
    cfg = CricketModelV3Config(
        n_batters=len(vocab_batter) + 1,
        n_bowlers=len(vocab_bowler) + 1,
        n_venues=len(vocab_venue) + 1,
        n_teams=len(vocab_team) + 1,
        learning_rate=args.learning_rate,
        class_weights=class_weights,
        weight_over_loss=args.over_loss_weight,
        hidden_units=tuple([args.hidden_units] * args.n_hidden_layers),
        dim_batter=args.embedding_dim_batter,
        dim_bowler=args.embedding_dim_batter,
        dim_venue=args.embedding_dim_venue,
        dim_team=args.embedding_dim_team,
    )
    model = build_cricket_model_v3(cfg)
    logger.info(
        f"V3 model params: {model.count_params():,}  "
        f"(state cols={cfg.n_continuous}, hidden={args.hidden_units}x{args.n_hidden_layers}, "
        f"emb_batter={cfg.dim_batter}, over_loss_w={cfg.weight_over_loss})"
    )

    # ----- Compile -----
    losses: Dict[str, object] = {}
    loss_weights: Dict[str, float] = {}
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
    model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)

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
            filepath=str(output_dir / "cricket_model_v3.keras"),
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
    logger.info(f"V3 training complete in {elapsed:.1f}s")

    # ----- Save normalizer (V3) -----
    train_X = train["X"].astype(np.float64)
    norm = {
        "mean": train_X.mean(axis=0).astype(np.float32),
        "std":  np.maximum(train_X.std(axis=0), 1e-6).astype(np.float32),
        "columns": list(CONTINUOUS_COLUMNS),
    }
    np.savez(output_dir / "cricket_model_v3_normalizer.npz", **norm)

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
        "n_continuous": len(CONTINUOUS_COLUMNS),
        "continuous_columns": list(CONTINUOUS_COLUMNS),
    }
    with open(output_dir / f"cricket_model_v3_summary_{label}.json", "w") as fp:
        json.dump(summary, fp, indent=2, default=str)

    print()
    print("=" * 60)
    print(f"V3 train summary  ({label})")
    print("=" * 60)
    print(f"  rows train/val/test : {len(train['y']):,} / {len(val['y']):,} / {len(test['y']):,}")
    print(f"  vocabs              : {summary['vocab_sizes']}")
    print(f"  model params        : {summary['model_params']:,}")
    print(f"  final val loss      : {summary['history'].get('val_loss', [None])[-1]}")
    print(f"  best val loss       : {min(summary['history'].get('val_loss', [float('inf')]))}")
    print(f"  saved to            : {output_dir / 'cricket_model_v3.keras'}")
    print(f"  vocabs              : {output_dir / 'vocabs.json'}")
    print(f"  normalizer          : {output_dir / 'cricket_model_v3_normalizer.npz'}")
    print(f"  summary             : {output_dir / f'cricket_model_v3_summary_{label}.json'}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
