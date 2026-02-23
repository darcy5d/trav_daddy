#!/usr/bin/env python3
"""
Evaluate all ball prediction models on a chronological holdout (last 20% by date).

Reports accuracy, log loss, and per-outcome F1 for each model so we can assess
how well they perform at the target task: ball-by-ball outcome prediction
(7-way classification) used for match simulation.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_XLA_FLAGS', '--tf_xla_auto_jit=0')

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def main():
    from src.models.ball_prediction_nn import (
        load_training_data,
        evaluate_model,
        BallPredictionModel,
        OUTCOME_NAMES,
    )

    configs = [
        ('T20', 'male'),
        ('T20', 'female'),
        ('ODI', 'male'),
        ('ODI', 'female'),
    ]
    validation_split = 0.2
    all_results = {}

    for format_type, gender in configs:
        fmt, g = format_type.lower(), gender.lower()
        model_path = str(ROOT / 'data' / 'processed' / f'ball_prediction_model_{fmt}_{g}.keras')
        if not Path(model_path).exists():
            print(f"[SKIP] No model: {format_type} {gender}")
            continue

        print(f"\nLoading data and model: {format_type} / {gender} ...")
        X, y, meta = load_training_data(format_type, gender)
        meta = meta.copy()
        meta['idx'] = np.arange(len(meta))
        meta_sorted = meta.sort_values('date')
        split_idx = int(len(meta_sorted) * (1 - validation_split))
        val_indices = meta_sorted['idx'].iloc[split_idx:].values
        X_test = X[val_indices]
        y_test = y[val_indices]

        predictor = BallPredictionModel(model_path)
        predictor.load()
        mean = predictor.normalizer['mean']
        std = predictor.normalizer['std']
        X_test_norm = (X_test - mean) / (std + 1e-8)

        results = evaluate_model(predictor.model, X_test_norm, y_test)
        key = f"{format_type}_{gender}"
        all_results[key] = {
            'accuracy': float(results['accuracy']),
            'log_loss': float(results['log_loss']),
            'n_test': len(y_test),
            'per_class': {},
        }
        for name in OUTCOME_NAMES:
            if name in results['classification_report']:
                m = results['classification_report'][name]
                all_results[key]['per_class'][name] = {
                    'precision': m['precision'],
                    'recall': m['recall'],
                    'f1': m['f1-score'],
                    'support': int(m['support']),
                }

        print(f"  Accuracy: {results['accuracy']:.4f}  Log loss: {results['log_loss']:.4f}  n_test: {len(y_test)}")

    # Summary report
    print("\n" + "=" * 80)
    print("BALL PREDICTION MODELS — EVALUATION ON CHRONOLOGICAL HOLDOUT (LAST 20%)")
    print("=" * 80)
    print("\nTarget task: 7-way ball outcome (Dot, Single, Two, Three, Four, Six, Wicket)")
    print("These models drive Monte Carlo match simulation; ball accuracy affects win-prob quality.\n")

    print("-" * 80)
    print(f"{'Model':<20} {'Accuracy':>10} {'Log Loss':>10} {'Test N':>10}")
    print("-" * 80)
    for key in ['T20_male', 'T20_female', 'ODI_male', 'ODI_female']:
        if key not in all_results:
            continue
        r = all_results[key]
        print(f"{key:<20} {r['accuracy']:>10.2%} {r['log_loss']:>10.4f} {r['n_test']:>10,}")
    print("-" * 80)

    print("\nPer-outcome F1 (macro avg indicates balance across outcomes):")
    print("-" * 80)
    for key in ['T20_male', 'T20_female', 'ODI_male', 'ODI_female']:
        if key not in all_results:
            continue
        r = all_results[key]
        f1s = [r['per_class'][n]['f1'] for n in OUTCOME_NAMES if n in r['per_class']]
        macro_f1 = np.mean(f1s) if f1s else 0
        print(f"  {key}: macro F1 = {macro_f1:.3f}")
        for name in OUTCOME_NAMES:
            if name in r['per_class']:
                pc = r['per_class'][name]
                print(f"    {name:<8}  F1={pc['f1']:.3f}  prec={pc['precision']:.3f}  rec={pc['recall']:.3f}  support={pc['support']}")
    print("-" * 80)

    # Save machine-readable summary
    out_path = ROOT / 'data' / 'processed' / 'model_evaluation_summary.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary written to {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
