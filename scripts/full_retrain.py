#!/usr/bin/env python3
"""
Full Model Retraining Script.

Orchestrates the complete retraining pipeline across all format/gender combinations:
  1. Schema validation
  2. Re-ingest data from Cricsheet          (optional)
  3. Calculate ELO ratings                  (optional)
  4. Build player distributions             } per format × gender
  5. Build venue statistics                 }
  6. Generate training features             }
  7. Train ball prediction neural network   }
  8. Validate model                         }
  9. Save model version metadata            }

Supported combinations:
  T20/male, T20/female, ODI/male, ODI/female

Usage:
    # Train all 4 combinations (default)
    python scripts/full_retrain.py

    # Train ODI models only (skip data download + ELO)
    python scripts/full_retrain.py --skip-ingest --skip-elo --formats ODI

    # Train T20 male only, full pipeline
    python scripts/full_retrain.py --formats T20 --male-only

    # Quick retrain (no ingest/ELO) for a single gender
    python scripts/full_retrain.py --skip-ingest --skip-elo --female-only

Notes:
    - ELO calculation uses the tiered system (V3) by default.
    - Use --skip-elo when no new matches have been added.
    - ELO recalculation takes ~30 s for 11 K matches.
"""

import sys
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Individual pipeline steps
# ──────────────────────────────────────────────────────────────────────────────

def step_validate_schema():
    """Step 0: Validate database schema before training."""
    logger.info("=" * 60)
    logger.info("STEP 0: SCHEMA VALIDATION")
    logger.info("=" * 60)

    from scripts.validate_schema import validate_schema, print_fix_suggestions

    is_valid, issues = validate_schema()

    if not is_valid:
        logger.error("=" * 60)
        logger.error("SCHEMA VALIDATION FAILED")
        logger.error("=" * 60)
        logger.error("The database schema is missing required tables or columns.")
        logger.error("Training cannot proceed until schema issues are resolved.")
        print_fix_suggestions(issues)
        raise RuntimeError(
            f"Schema validation failed with {len(issues)} issue(s). See logs for details."
        )

    logger.info("Schema validation passed – proceeding with training")
    return True


def step_ingest_data():
    """Step 1: Ingest data from Cricsheet JSON files."""
    logger.info("=" * 60)
    logger.info("STEP 1: DATA INGESTION")
    logger.info("=" * 60)

    from src.data.ingest import ingest_matches
    from src.data.database import print_database_summary

    stats = ingest_matches(formats=['all_male', 'all_female'])
    print_database_summary()
    return stats


def step_calculate_elo(use_tiered_system: bool = True):
    """Step 2: Calculate ELO ratings for teams and players.

    Args:
        use_tiered_system: If True, use calculator_v3 (tiered system).
                          If False, use calculator_v2 (legacy system).

    Note: Tiered ELO calculation is expensive (~27s for 11K matches).
          Should only be run when:
          - New matches are added to the database
          - Team tiers are manually adjusted
          - Schema/logic changes require recalculation
    """
    logger.info("=" * 60)
    logger.info("STEP 2: ELO CALCULATIONS")
    logger.info("=" * 60)

    if use_tiered_system:
        from src.elo.calculator_v3 import calculate_all_elos_v3
        logger.info("Using tiered ELO system (V3)…")
        stats = calculate_all_elos_v3(force_recalculate=False)
        logger.info("Tiered ELO calculation complete")
    else:
        from src.elo.calculator_v2 import calculate_all_elos_v2
        logger.info("Using legacy ELO system (V2)…")
        stats = calculate_all_elos_v2(force_recalculate=True)
        logger.info("Legacy ELO calculation complete")

    return stats


def step_build_player_distributions(gender: str, format_type: str = 'T20'):
    """Step 3: Build player outcome distributions."""
    logger.info("=" * 60)
    logger.info(f"STEP 3: PLAYER DISTRIBUTIONS ({format_type.upper()} / {gender.upper()})")
    logger.info("=" * 60)

    from src.features.player_distributions import build_and_save_distributions

    stats = build_and_save_distributions(
        format_type=format_type, gender=gender, min_balls=10
    )
    logger.info(f"Built player distributions for {format_type}/{gender}")
    return stats


def step_build_venue_stats(gender: str, format_type: str = 'T20'):
    """Step 4: Build venue statistics."""
    logger.info("=" * 60)
    logger.info(f"STEP 4: VENUE STATISTICS ({format_type.upper()} / {gender.upper()})")
    logger.info("=" * 60)

    from src.features.venue_stats import VenueStatsBuilder

    builder = VenueStatsBuilder(format_type=format_type, gender=gender)
    builder.build_from_database()

    output_path = str(
        project_root / 'data' / 'processed' / f'venue_stats_{format_type.lower()}_{gender}.pkl'
    )
    builder.save(output_path)

    logger.info(f"Built venue stats for {format_type}/{gender}")
    return True


def step_generate_training_data(gender: str, format_type: str = 'T20'):
    """Step 5: Generate training features and labels."""
    logger.info("=" * 60)
    logger.info(f"STEP 5: TRAINING DATA GENERATION ({format_type.upper()} / {gender.upper()})")
    logger.info("=" * 60)

    from src.features.ball_training_data import main as generate_main

    generate_main(format_type=format_type, gender=gender)
    logger.info(f"Generated training data for {format_type}/{gender}")
    return True


def step_train_model(gender: str, format_type: str = 'T20'):
    """Step 6: Train the ball prediction neural network."""
    logger.info("=" * 60)
    logger.info(f"STEP 6: MODEL TRAINING ({format_type.upper()} / {gender.upper()})")
    logger.info("=" * 60)

    from src.models.ball_prediction_nn import main as train_main

    train_main(format_type=format_type, gender=gender)
    return True


def step_validate_model(gender: str, format_type: str = 'T20'):
    """Step 7: Run validation checks on the trained model."""
    logger.info("=" * 60)
    logger.info(f"STEP 7: MODEL VALIDATION ({format_type.upper()} / {gender.upper()})")
    logger.info("=" * 60)

    import numpy as np

    data_dir = project_root / 'data' / 'processed'
    fmt = format_type.lower()
    model_path = data_dir / f'ball_prediction_model_{fmt}_{gender}.keras'

    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        logger.info(f"Model saved: {model_path.name} ({size_mb:.2f} MB)")

        X_path = data_dir / f'ball_X_{fmt}_{gender}.npy'
        if X_path.exists():
            X = np.load(X_path)
            logger.info(f"Training data: {len(X):,} samples, {X.shape[1]} features")

        return True
    else:
        logger.warning(f"Model file not found: {model_path}")
        return False


def save_model_version_metadata(
    gender: str,
    format_type: str,
    start_time: datetime,
    end_time: datetime,
) -> bool:
    """Save model version metadata to the database and JSON file.

    Args:
        gender:      'male' or 'female'
        format_type: 'T20' or 'ODI'
        start_time:  Training start time
        end_time:    Training end time

    Returns:
        True if successful
    """
    try:
        import json
        import numpy as np
        from src.data.database import save_model_version, init_model_versions_table, get_connection

        logger.info("=" * 60)
        logger.info(f"SAVING MODEL VERSION METADATA ({format_type.upper()} / {gender.upper()})")
        logger.info("=" * 60)

        init_model_versions_table()

        fmt = format_type.lower()
        timestamp = end_time.strftime("%Y%m%d_%H%M%S")
        model_name = f"{gender}_{fmt}_{timestamp}"

        data_dir = project_root / 'data' / 'processed'
        model_path = str(data_dir / f'ball_prediction_model_{fmt}_{gender}.keras')
        normalizer_path = str(data_dir / f'ball_prediction_model_{fmt}_{gender}_normalizer.pkl')

        model_size_mb = (
            Path(model_path).stat().st_size / (1024 * 1024)
            if Path(model_path).exists() else None
        )

        X_path = data_dir / f'ball_X_{fmt}_{gender}.npy'
        training_samples = len(np.load(X_path)) if X_path.exists() else None

        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT MIN(date) as earliest, MAX(date) as latest
            FROM matches
            WHERE match_type = ? AND gender = ?
            """,
            (format_type.upper(), gender),
        )
        row = cursor.fetchone()
        data_earliest_date = row['earliest'] if row else None
        data_latest_date   = row['latest']   if row else None
        conn.close()

        training_duration_seconds = int((end_time - start_time).total_seconds())

        # Load accuracy metrics from meta CSV if available
        import pandas as pd
        accuracy_metrics = None
        meta_path = data_dir / f'ball_meta_{fmt}_{gender}.csv'
        if meta_path.exists():
            try:
                meta_df = pd.read_csv(meta_path)
                if not meta_df.empty and 'f1-score' in meta_df.columns:
                    accuracy_metrics = json.dumps({
                        'avg_f1_score': float(meta_df['f1-score'].mean()),
                        'per_outcome': meta_df.to_dict('records'),
                    })
            except Exception as e:
                logger.warning(f"Could not load accuracy metrics: {e}")

        model_id = save_model_version(
            model_name=model_name,
            gender=gender,
            format_type=format_type.upper(),
            model_path=model_path,
            normalizer_path=normalizer_path,
            data_earliest_date=data_earliest_date,
            data_latest_date=data_latest_date,
            training_samples=training_samples,
            training_duration_seconds=training_duration_seconds,
            model_size_mb=model_size_mb,
            accuracy_metrics=accuracy_metrics,
            is_active=True,
            notes=f"Trained via full_retrain.py on {end_time.strftime('%Y-%m-%d %H:%M:%S')}",
        )

        logger.info(f"Saved model version to database: {model_name} (ID: {model_id})")

        # JSON backup
        json_file = project_root / 'data' / 'model_versions.json'
        versions = []
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    versions = json.load(f)
            except Exception:
                versions = []

        versions.append({
            'id': model_id,
            'model_name': model_name,
            'gender': gender,
            'format_type': format_type.upper(),
            'created_at': end_time.isoformat(),
            'model_path': model_path,
            'normalizer_path': normalizer_path,
            'data_earliest_date': data_earliest_date,
            'data_latest_date': data_latest_date,
            'training_samples': training_samples,
            'training_duration_seconds': training_duration_seconds,
            'model_size_mb': model_size_mb,
            'accuracy_metrics': accuracy_metrics,
            'is_active': True,
        })

        with open(json_file, 'w') as f:
            json.dump(versions, f, indent=2)

        logger.info(f"Saved model version to JSON: {json_file}")
        return True

    except Exception as e:
        logger.error(f"Failed to save model version metadata: {e}")
        import traceback
        traceback.print_exc()
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Full pipeline orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(
    skip_ingest: bool = False,
    skip_elo: bool = False,
    male_only: bool = False,
    female_only: bool = False,
    formats: list = None,
    progress_callback=None,
):
    """Run the full retraining pipeline for all requested format/gender combos.

    Args:
        skip_ingest:       Skip Cricsheet data download/ingestion
        skip_elo:          Skip ELO recalculation
        male_only:         Only process male gender
        female_only:       Only process female gender
        formats:           List of formats to train, e.g. ['T20', 'ODI']. Defaults to both.
        progress_callback: Optional callable(pct: int, msg: str)

    Returns:
        dict with per-step results
    """
    start_time = time.time()
    if formats is None:
        formats = ['T20', 'ODI']

    genders = []
    if not female_only:
        genders.append('male')
    if not male_only:
        genders.append('female')

    n_combos = len(formats) * len(genders)

    logger.info("=" * 60)
    logger.info("FULL MODEL RETRAINING PIPELINE")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Formats: {formats}")
    logger.info(f"Genders: {genders}")
    logger.info(f"Combinations: {n_combos}")
    logger.info("=" * 60)

    # Step 0: Schema validation
    try:
        step_validate_schema()
    except RuntimeError as e:
        logger.error(f"Pipeline aborted: {e}")
        return {'error': str(e), 'stage': 'schema_validation'}

    results = {}
    current_pct = 0.0

    ingest_pct = 15.0 if not skip_ingest else 0.0
    elo_pct    = 10.0 if not skip_elo    else 0.0
    per_combo_pct = (100.0 - ingest_pct - elo_pct) / max(n_combos, 1)

    def report(step_name: str):
        if progress_callback:
            progress_callback(int(current_pct), step_name)

    # Step 1: Ingest
    if not skip_ingest:
        report("Step 1: Ingesting data from Cricsheet…")
        results['ingest'] = step_ingest_data()
        current_pct += ingest_pct
        report("Data ingestion complete")
    else:
        logger.info("Skipping data ingestion (--skip-ingest)")

    # Step 2: ELO
    if not skip_elo:
        report("Step 2: Calculating ELO ratings…")
        results['elo'] = step_calculate_elo()
        current_pct += elo_pct
        report("ELO calculation complete")
    else:
        logger.info("Skipping ELO calculation (--skip-elo)")

    # Steps 3–9: per format × gender
    sub_steps = 6  # distributions, venue, training-data, train, validate, metadata
    step_pct = per_combo_pct / sub_steps

    for fmt in formats:
        for gender in genders:
            combo_label = f"{fmt.upper()}/{gender.upper()}"
            logger.info(f"\n{'*' * 60}")
            logger.info(f"*** PROCESSING {combo_label} ***")
            logger.info(f"{'*' * 60}\n")

            combo_start = datetime.now()

            report(f"[{combo_label}] Building player distributions…")
            results[f'player_dist_{fmt}_{gender}'] = step_build_player_distributions(gender, fmt)
            current_pct += step_pct

            report(f"[{combo_label}] Building venue statistics…")
            results[f'venue_stats_{fmt}_{gender}'] = step_build_venue_stats(gender, fmt)
            current_pct += step_pct

            report(f"[{combo_label}] Generating training data…")
            results[f'training_data_{fmt}_{gender}'] = step_generate_training_data(gender, fmt)
            current_pct += step_pct

            report(f"[{combo_label}] Training neural network…")
            results[f'train_{fmt}_{gender}'] = step_train_model(gender, fmt)
            current_pct += step_pct

            report(f"[{combo_label}] Validating model…")
            results[f'validate_{fmt}_{gender}'] = step_validate_model(gender, fmt)
            current_pct += step_pct

            report(f"[{combo_label}] Saving model version metadata…")
            combo_end = datetime.now()
            results[f'version_{fmt}_{gender}'] = save_model_version_metadata(
                gender, fmt, combo_start, combo_end
            )
            current_pct += step_pct

    elapsed = time.time() - start_time

    if progress_callback:
        progress_callback(100, "Pipeline complete!")

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Total time: {elapsed / 60:.1f} minutes")
    logger.info("=" * 60)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Full model retraining pipeline')
    parser.add_argument('--skip-ingest', action='store_true',
                        help='Skip data ingestion step')
    parser.add_argument('--skip-elo', action='store_true',
                        help='Skip ELO calculation step')
    parser.add_argument('--male-only', action='store_true',
                        help='Only train male models')
    parser.add_argument('--female-only', action='store_true',
                        help='Only train female models')
    parser.add_argument('--formats', nargs='+', choices=['T20', 'ODI'],
                        default=['T20', 'ODI'],
                        help='Formats to train (default: T20 ODI)')

    args = parser.parse_args()

    results = run_full_pipeline(
        skip_ingest=args.skip_ingest,
        skip_elo=args.skip_elo,
        male_only=args.male_only,
        female_only=args.female_only,
        formats=args.formats,
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
