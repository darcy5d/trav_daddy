#!/usr/bin/env python3
"""
Hyperband Hyperparameter Tuner for Ball Prediction Neural Network.

Uses Keras Tuner's Hyperband algorithm (successive halving) to efficiently
search the hyperparameter space across all 4 format/gender combinations.

Hyperband calculates its own trial count from max_epochs and factor — there
is no max_trials parameter.  With max_epochs=30 and factor=3 the schedule is:

  Bracket s=3: 34 initial configs × 2 ep  →  9 promoted × 4 ep  → ...
  Bracket s=2: 12 configs × 3 ep  → ...
  Bracket s=1:  6 configs × 9 ep  → ...
  Bracket s=0:  4 configs × 30 ep

  Total ≈ 460 training epochs  (upper-bound; EarlyStopping may cut some short)

Optimised for Apple Silicon M2 Pro (Metal GPU, 12 GB RAM):
  - XLA disabled to avoid Metal backend crashes
  - Memory growth enabled for Metal
  - Larger batch sizes (256–2048) to saturate the GPU

Usage:
    # Tune a single combination
    python scripts/tune_ball_predictor.py --format T20 --gender male

    # Overwrite an existing tuner run (starts from scratch)
    python scripts/tune_ball_predictor.py --format ODI --gender female --overwrite

    # Tune all 4 combinations sequentially
    python scripts/tune_ball_predictor.py --all

    # Resume a partial run (default — no --overwrite flag)
    python scripts/tune_ball_predictor.py --format T20 --gender female

Results are saved to:
    data/processed/best_hparams_{format}_{gender}.json
    data/processed/tuner_{format}_{gender}/   (Keras Tuner checkpoint directory)
"""

import sys
import os
import time
import math
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Disable XLA for Apple Silicon compatibility before any TF import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR = project_root / 'data' / 'processed'


# ──────────────────────────────────────────────────────────────────────────────
# Metal / TensorFlow setup
# ──────────────────────────────────────────────────────────────────────────────

def _configure_tensorflow():
    """Enable Metal GPU memory growth; suppress verbose TF logs."""
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    if gpus:
        logger.info(f"Metal GPU detected: {len(gpus)} device(s)")
    else:
        logger.info("No GPU detected – running on CPU")


# ──────────────────────────────────────────────────────────────────────────────
# Keras imports (TF 2.16+ uses standalone keras 3.x)
# ──────────────────────────────────────────────────────────────────────────────

def _import_keras():
    try:
        import keras
        from keras import layers, regularizers
        from keras.callbacks import EarlyStopping
        return keras, layers, regularizers, EarlyStopping
    except ImportError:
        from tensorflow import keras
        from tensorflow.keras import layers, regularizers
        from tensorflow.keras.callbacks import EarlyStopping
        return keras, layers, regularizers, EarlyStopping


# ──────────────────────────────────────────────────────────────────────────────
# Timing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _fmt_duration(secs: float) -> str:
    """Format a duration in seconds as a human-readable string."""
    secs = max(0, int(secs))
    if secs >= 3600:
        return f"{secs // 3600}h {(secs % 3600) // 60:02d}m"
    if secs >= 60:
        return f"{secs // 60}m {secs % 60:02d}s"
    return f"{secs}s"


def _hyperband_total_epochs(max_epochs: int, factor: int, hyperband_iterations: int = 1) -> int:
    """
    Calculate the total number of training epochs Hyperband will run.

    This is an UPPER BOUND — EarlyStopping may cut individual trials short.
    Uses the Hyperband bracket schedule from Li et al. (2018).

    For max_epochs=30, factor=3, iterations=1 → 460 total epochs.
    """
    s_max = math.floor(math.log(max_epochs) / math.log(factor))
    total = 0
    for s in range(s_max, -1, -1):
        n = math.ceil((s_max + 1) / (s + 1) * factor ** s)
        configs_remaining = n
        for i in range(s + 1):
            budget = math.floor(max_epochs / factor ** (s - i))
            total += configs_remaining * budget
            configs_remaining = math.floor(configs_remaining / factor)
    return total * hyperband_iterations


# ──────────────────────────────────────────────────────────────────────────────
# Stateful progress tracker callback
# ──────────────────────────────────────────────────────────────────────────────

def _make_progress_tracker(total_expected_epochs: int, format_type: str, gender: str):
    """
    Return a Keras Callback that tracks progress across ALL Hyperband trials.

    One instance is passed to tuner.search() and Keras Tuner reuses it for
    every trial's model.fit().  Because it is a single object its state
    persists across trials — this is the key fix for the ETA bug where
    epochs_done was being reset to 0 at the start of each new trial.

    Output format (printed to logger after every epoch, after ≥3 samples):

      epoch   3 | val_loss=1.2341 val_acc=49.3% ← best | took 38s
      avg 37s/ep | elapsed 4m 12s | ~5h 06m remaining (+1σ ~5h 24m) [9/460 epochs done]
    """
    keras, _, _, _ = _import_keras()
    label = f"{format_type.upper()}/{gender.upper()}"

    class _Tracker(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            # ── persistent across ALL trials (never reset) ──
            self.epochs_done: int = 0
            self.trial_count: int = 0
            self.epoch_times: list = []
            self.search_start: float = time.time()
            # ── per-trial state (reset on_train_begin) ──
            self._epoch_start: float = 0.0
            self._trial_best_val_loss: float = float('inf')
            self._trial_epoch_count: int = 0

        def __deepcopy__(self, memo):
            # Keras Tuner deep-copies callbacks for every trial.
            # Returning self keeps the shared epoch counter intact —
            # this is the fix for the ETA resetting to 0 each trial.
            memo[id(self)] = self
            return self

        # ── Trial boundary ────────────────────────────────────────────────────

        def on_train_begin(self, logs=None):
            self.trial_count += 1
            self._trial_best_val_loss = float('inf')
            self._trial_epoch_count = 0
            # print() goes to sys.stdout which LogCapture redirects → GUI log panel
            print(
                f"── Trial {self.trial_count} [{label}]  "
                f"[{self.epochs_done}/{total_expected_epochs} epochs done] "
                f"{'─' * 20}"
            )

        def on_train_end(self, logs=None):
            print(
                f"   Trial {self.trial_count} done | "
                f"{self._trial_epoch_count} epoch(s) | "
                f"best val_loss: {self._trial_best_val_loss:.4f}"
            )

        # ── Epoch boundary ────────────────────────────────────────────────────

        def on_epoch_begin(self, epoch, logs=None):
            self._epoch_start = time.time()

        def on_epoch_end(self, epoch, logs=None):
            duration = time.time() - self._epoch_start
            self.epoch_times.append(duration)
            self.epochs_done += 1          # ← CUMULATIVE — never resets
            self._trial_epoch_count += 1

            logs = logs or {}
            val_loss = logs.get('val_loss', float('inf'))
            val_acc  = logs.get('val_accuracy', 0.0)

            best_flag = ""
            if val_loss < self._trial_best_val_loss:
                self._trial_best_val_loss = val_loss
                best_flag = " ← best"

            print(
                f"  epoch {epoch + 1:3d} | "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.2%}"
                f"{best_flag} | took {_fmt_duration(duration)}"
            )

            # ETA — show after ≥3 total epochs (across all trials)
            if len(self.epoch_times) >= 3:
                mean_ep   = float(np.mean(self.epoch_times))
                std_ep    = float(np.std(self.epoch_times))
                remaining = max(0, total_expected_epochs - self.epochs_done)
                eta_med   = mean_ep * remaining
                eta_high  = (mean_ep + std_ep) * remaining
                elapsed   = time.time() - self.search_start
                print(
                    f"  avg {_fmt_duration(mean_ep)}/epoch | "
                    f"elapsed {_fmt_duration(elapsed)} | "
                    f"~{_fmt_duration(eta_med)} remaining "
                    f"(+1σ ~{_fmt_duration(eta_high)}) "
                    f"[{self.epochs_done}/{total_expected_epochs} epochs done]"
                )

    return _Tracker()


# ──────────────────────────────────────────────────────────────────────────────
# Model builder for Keras Tuner
# ──────────────────────────────────────────────────────────────────────────────

def _build_model(hp, input_dim: int, num_classes: int = 7):
    """Build a model from Keras Tuner HyperParameters object."""
    keras, layers, regularizers, _ = _import_keras()

    n_layers = hp.Int('n_layers', min_value=1, max_value=4, default=1)
    units    = hp.Choice('units', values=[64, 128, 256], default=256)
    dropout  = hp.Float('dropout', min_value=0.1, max_value=0.4, step=0.1, default=0.3)
    l2_reg   = hp.Choice('l2_reg', values=[1e-4, 1e-3, 1e-2], default=1e-4)
    lr       = hp.Choice('learning_rate', values=[1e-4, 5e-4, 1e-3, 5e-3], default=1e-3)

    inputs = keras.Input(shape=(input_dim,))
    x = inputs

    for _ in range(n_layers):
        x = layers.Dense(units, kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        jit_compile=False,
    )
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Data loading + chronological split + normalisation
# ──────────────────────────────────────────────────────────────────────────────

def _load_and_split(format_type: str, gender: str, val_fraction: float = 0.2):
    """Load training data and return normalised train/val arrays."""
    import pandas as pd

    fmt = format_type.lower()
    X    = np.load(DATA_DIR / f'ball_X_{fmt}_{gender}.npy')
    y    = np.load(DATA_DIR / f'ball_y_{fmt}_{gender}.npy')
    meta = pd.read_csv(DATA_DIR / f'ball_meta_{fmt}_{gender}.csv')

    logger.info(f"Loaded {len(X):,} samples | {X.shape[1]} features")

    # Chronological split
    meta = meta.copy()
    meta['_idx'] = range(len(meta))
    split     = int(len(meta.sort_values('date')) * (1 - val_fraction))
    train_idx = meta.sort_values('date')['_idx'].iloc[:split].values
    val_idx   = meta.sort_values('date')['_idx'].iloc[split:].values

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Normalise using training stats only
    mean      = X_train.mean(axis=0)
    std       = X_train.std(axis=0) + 1e-8
    X_train_n = (X_train - mean) / std
    X_val_n   = (X_val   - mean) / std

    logger.info(f"Train: {len(X_train):,}  Val: {len(X_val):,}")
    return X_train_n, y_train, X_val_n, y_val, X.shape[1], mean, std


# ──────────────────────────────────────────────────────────────────────────────
# Main tuning function
# ──────────────────────────────────────────────────────────────────────────────

def run_tuner(
    format_type: str,
    gender: str,
    overwrite: bool = False,
    max_epochs: int = 30,
    factor: int = 3,
    hyperband_iterations: int = 1,
    batch_size: int = 512,
    val_fraction: float = 0.2,
    progress_callback=None,
):
    """
    Run Hyperband search for one (format, gender) combination.

    Hyperband automatically determines the number of trials from max_epochs
    and factor via successive halving — there is no max_trials parameter.

    If a previous partial run exists (overwrite=False) Keras Tuner will resume
    from where it left off, skipping completed trials.

    Args:
        format_type:          'T20' or 'ODI'
        gender:               'male' or 'female'
        overwrite:            Delete previous tuner run and start fresh
        max_epochs:           Max epochs per Hyperband bracket (default 30)
        factor:               Successive halving reduction factor (default 3)
        hyperband_iterations: How many full Hyperband brackets to run (default 1)
        batch_size:           Batch size during search (tuned for M2 Metal)
        val_fraction:         Fraction of data reserved for validation
        progress_callback:    Optional callable(pct: int, msg: str)

    Returns:
        dict of best hyperparameters
    """
    import keras_tuner as kt
    keras, _, _, EarlyStopping = _import_keras()

    _configure_tensorflow()

    tag          = f"{format_type.lower()}_{gender}"
    tuner_dir    = DATA_DIR / f'tuner_{tag}'
    hparams_path = DATA_DIR / f'best_hparams_{tag}.json'

    if overwrite and tuner_dir.exists():
        import shutil
        shutil.rmtree(tuner_dir)
        logger.info(f"Removed previous tuner dir: {tuner_dir}")

    # Pre-compute total epoch budget for the progress tracker
    total_expected = _hyperband_total_epochs(max_epochs, factor, hyperband_iterations)

    logger.info("=" * 60)
    logger.info(f"HYPERBAND TUNING: {format_type.upper()} / {gender.upper()}")
    logger.info(f"  max_epochs={max_epochs}, factor={factor}, iterations={hyperband_iterations}")
    logger.info(f"  Total epoch budget (upper bound): {total_expected}")
    logger.info("=" * 60)

    if progress_callback:
        progress_callback(5, f"Loading {format_type}/{gender} training data…")

    X_train, y_train, X_val, y_val, input_dim, mean, std = _load_and_split(
        format_type, gender, val_fraction
    )

    # Build tuner — max_trials is NOT a valid Hyperband parameter
    tuner = kt.Hyperband(
        hypermodel=lambda hp: _build_model(hp, input_dim),
        objective='val_loss',
        max_epochs=max_epochs,
        factor=factor,
        hyperband_iterations=hyperband_iterations,
        directory=str(DATA_DIR),
        project_name=f'tuner_{tag}',
        overwrite=overwrite,
    )

    tuner.search_space_summary()

    # Callbacks passed to every trial's model.fit() call.
    # The tracker is a SINGLE instance so its epoch counter persists across trials.
    tracker    = _make_progress_tracker(total_expected, format_type, gender)
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
    )

    if progress_callback:
        progress_callback(10, f"Starting Hyperband search (≈{total_expected} epochs total)…")

    logger.info(f"Starting search … (tuner dir: {tuner_dir})")
    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=[tracker, early_stop],
        verbose=0,   # suppress Keras bar; tracker handles clean output
    )

    if progress_callback:
        progress_callback(90, "Search complete – extracting best hyperparameters…")

    # Extract best hyperparameters
    best_hp    = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.get_best_models(num_models=1)[0]

    best = {
        'format_type':   format_type.upper(),
        'gender':        gender,
        'n_layers':      best_hp.get('n_layers'),
        'units':         best_hp.get('units'),
        'dropout':       best_hp.get('dropout'),
        'l2_reg':        best_hp.get('l2_reg'),
        'learning_rate': best_hp.get('learning_rate'),
        'tuned_at':      datetime.now().isoformat(),
    }

    # Quick final validation evaluation
    results     = best_model.evaluate(X_val, y_val, verbose=0)
    val_loss_idx = best_model.metrics_names.index('loss')
    val_acc_idx  = best_model.metrics_names.index('accuracy')
    best['val_loss']     = float(results[val_loss_idx])
    best['val_accuracy'] = float(results[val_acc_idx])

    # Save
    hparams_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hparams_path, 'w') as f:
        json.dump(best, f, indent=2)

    if progress_callback:
        progress_callback(100, "Best hyperparameters saved.")

    logger.info("=" * 60)
    logger.info(f"BEST HYPERPARAMETERS ({format_type.upper()}/{gender.upper()})")
    logger.info(
        f"  layers={best['n_layers']}, units={best['units']}, "
        f"dropout={best['dropout']}, l2={best['l2_reg']}, "
        f"lr={best['learning_rate']}"
    )
    logger.info(f"  val_loss={best['val_loss']:.4f}, val_acc={best['val_accuracy']:.2%}")
    logger.info(f"  Saved → {hparams_path}")
    logger.info("=" * 60)

    return best


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Hyperband hyperparameter search for ball prediction NN'
    )
    parser.add_argument('--format', default='T20', choices=['T20', 'ODI'],
                        help='Match format (default: T20)')
    parser.add_argument('--gender', default='male', choices=['male', 'female'],
                        help='Gender (default: male)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Delete previous tuner run and start fresh')
    parser.add_argument('--all', action='store_true',
                        help='Run tuner for all 4 format/gender combinations sequentially')
    parser.add_argument('--max-epochs', type=int, default=30,
                        help='Max epochs per Hyperband bracket (default: 30)')
    parser.add_argument('--factor', type=int, default=3,
                        help='Successive halving reduction factor (default: 3)')
    parser.add_argument('--hyperband-iterations', type=int, default=1,
                        help='Number of full Hyperband brackets to run (default: 1)')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size during search (default: 512, M2 optimised)')

    args = parser.parse_args()

    combinations = (
        [('T20', 'male'), ('T20', 'female'), ('ODI', 'male'), ('ODI', 'female')]
        if args.all
        else [(args.format, args.gender)]
    )

    for fmt, gender in combinations:
        run_tuner(
            format_type=fmt,
            gender=gender,
            overwrite=args.overwrite,
            max_epochs=args.max_epochs,
            factor=args.factor,
            hyperband_iterations=args.hyperband_iterations,
            batch_size=args.batch_size,
        )


if __name__ == '__main__':
    main()
