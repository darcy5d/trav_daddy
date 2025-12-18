#!/usr/bin/env python3
"""
Full Model Retraining Script.

Orchestrates the complete retraining pipeline:
1. Re-ingest data from Cricsheet
2. Calculate ELO ratings
3. Build player distributions
4. Build venue statistics
5. Generate training features
6. Train ball prediction neural network
7. Validate model

Usage:
    python scripts/full_retrain.py [--skip-ingest] [--skip-elo] [--male-only] [--female-only]
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


def step_calculate_elo():
    """Step 2: Calculate ELO ratings for teams and players."""
    logger.info("=" * 60)
    logger.info("STEP 2: ELO CALCULATIONS")
    logger.info("=" * 60)
    
    from src.elo.calculator_v2 import calculate_all_elos_v2
    
    # Calculate ELOs for all matches (handles both genders internally)
    stats = calculate_all_elos_v2(force_recalculate=True)
    
    logger.info(f"ELO calculation complete")
    return stats


def step_build_player_distributions(gender: str = 'male'):
    """Step 3: Build player outcome distributions."""
    logger.info("=" * 60)
    logger.info(f"STEP 3: PLAYER DISTRIBUTIONS ({gender.upper()})")
    logger.info("=" * 60)
    
    from src.features.player_distributions import build_and_save_distributions
    
    stats = build_and_save_distributions(format_type='T20', gender=gender, min_balls=10)
    
    logger.info(f"Built player distributions for {gender}")
    return stats


def step_build_venue_stats(gender: str = 'male'):
    """Step 4: Build venue statistics."""
    logger.info("=" * 60)
    logger.info(f"STEP 4: VENUE STATISTICS ({gender.upper()})")
    logger.info("=" * 60)
    
    from src.features.venue_stats import VenueStatsBuilder
    
    builder = VenueStatsBuilder(format_type='T20', gender=gender)
    builder.build_from_database()
    
    # Save to file
    output_path = str(project_root / 'data' / 'processed' / f'venue_stats_t20_{gender}.pkl')
    builder.save(output_path)
    
    logger.info(f"Built venue stats for {gender}")
    return True


def step_generate_training_data(gender: str = 'male'):
    """Step 5: Generate training features and labels."""
    logger.info("=" * 60)
    logger.info(f"STEP 5: TRAINING DATA GENERATION ({gender.upper()})")
    logger.info("=" * 60)
    
    from src.features.ball_training_data import main as generate_main
    
    # This generates and saves the training data
    generate_main(format_type='T20', gender=gender)
    
    logger.info(f"Generated training data for {gender}")
    return True


def step_train_model(gender: str = 'male'):
    """Step 6: Train the ball prediction neural network."""
    logger.info("=" * 60)
    logger.info(f"STEP 6: MODEL TRAINING ({gender.upper()})")
    logger.info("=" * 60)
    
    from src.models.ball_prediction_nn import main as train_main
    
    train_main(format_type='T20', gender=gender)
    return True


def step_validate_model(gender: str = 'male'):
    """Step 7: Run validation checks on the trained model."""
    logger.info("=" * 60)
    logger.info(f"STEP 7: MODEL VALIDATION ({gender.upper()})")
    logger.info("=" * 60)
    
    import numpy as np
    from pathlib import Path
    
    # Check if model file exists
    data_dir = project_root / 'data' / 'processed'
    model_path = data_dir / f'ball_prediction_model_t20_{gender}.keras'
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        logger.info(f"Model saved: {model_path.name} ({size_mb:.2f} MB)")
        
        # Load test data to check dimensions
        X = np.load(data_dir / f'ball_X_t20_{gender}.npy')
        logger.info(f"Training data: {len(X)} samples, {X.shape[1]} features")
        
        return True
    else:
        logger.warning(f"Model file not found: {model_path}")
        return False


def save_model_version_metadata(gender: str, start_time: datetime, end_time: datetime) -> bool:
    """
    Save model version metadata to database and JSON file.
    
    Args:
        gender: 'male' or 'female'
        start_time: Training start time
        end_time: Training end time
        
    Returns:
        True if successful
    """
    try:
        import json
        import numpy as np
        from src.data.database import save_model_version, init_model_versions_table, get_connection
        
        logger.info("=" * 60)
        logger.info(f"SAVING MODEL VERSION METADATA ({gender.upper()})")
        logger.info("=" * 60)
        
        # Ensure model_versions table exists
        init_model_versions_table()
        
        # Generate model name
        timestamp = end_time.strftime("%Y%m%d_%H%M%S")
        model_name = f"{gender}_t20_{timestamp}"
        
        # Get file paths
        data_dir = project_root / 'data' / 'processed'
        model_path = str(data_dir / f'ball_prediction_model_t20_{gender}.keras')
        normalizer_path = str(data_dir / f'ball_prediction_model_t20_{gender}_normalizer.pkl')
        
        # Get model file size
        model_size_mb = Path(model_path).stat().st_size / (1024 * 1024) if Path(model_path).exists() else None
        
        # Get training samples count
        X_path = data_dir / f'ball_X_t20_{gender}.npy'
        training_samples = len(np.load(X_path)) if X_path.exists() else None
        
        # Get data date range from database
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT MIN(date) as earliest, MAX(date) as latest
            FROM matches
            WHERE match_type = 'T20' AND gender = ?
        """, (gender,))
        row = cursor.fetchone()
        data_earliest_date = row['earliest'] if row else None
        data_latest_date = row['latest'] if row else None
        conn.close()
        
        # Calculate training duration
        training_duration_seconds = int((end_time - start_time).total_seconds())
        
        # Load accuracy metrics if available
        meta_path = data_dir / f'ball_meta_t20_{gender}.csv'
        accuracy_metrics = None
        if meta_path.exists():
            import pandas as pd
            try:
                meta_df = pd.read_csv(meta_path)
                if not meta_df.empty and 'f1-score' in meta_df.columns:
                    # Calculate weighted average F1 score
                    avg_f1 = meta_df['f1-score'].mean()
                    accuracy_metrics = json.dumps({
                        'avg_f1_score': float(avg_f1),
                        'per_outcome': meta_df.to_dict('records')
                    })
            except Exception as e:
                logger.warning(f"Could not load accuracy metrics: {e}")
        
        # Save to database
        model_id = save_model_version(
            model_name=model_name,
            gender=gender,
            format_type='T20',
            model_path=model_path,
            normalizer_path=normalizer_path,
            data_earliest_date=data_earliest_date,
            data_latest_date=data_latest_date,
            training_samples=training_samples,
            training_duration_seconds=training_duration_seconds,
            model_size_mb=model_size_mb,
            accuracy_metrics=accuracy_metrics,
            is_active=True,
            notes=f"Trained via full_retrain.py on {end_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        logger.info(f"Saved model version to database: {model_name} (ID: {model_id})")
        
        # Also save to JSON file for backup
        json_file = project_root / 'data' / 'model_versions.json'
        
        # Load existing versions
        versions = []
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    versions = json.load(f)
            except:
                versions = []
        
        # Add new version
        versions.append({
            'id': model_id,
            'model_name': model_name,
            'gender': gender,
            'format_type': 'T20',
            'created_at': end_time.isoformat(),
            'model_path': model_path,
            'normalizer_path': normalizer_path,
            'data_earliest_date': data_earliest_date,
            'data_latest_date': data_latest_date,
            'training_samples': training_samples,
            'training_duration_seconds': training_duration_seconds,
            'model_size_mb': model_size_mb,
            'accuracy_metrics': accuracy_metrics,
            'is_active': True
        })
        
        # Save JSON
        with open(json_file, 'w') as f:
            json.dump(versions, f, indent=2)
        
        logger.info(f"Saved model version to JSON: {json_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to save model version metadata: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_full_pipeline(
    skip_ingest: bool = False,
    skip_elo: bool = False,
    male_only: bool = False,
    female_only: bool = False,
    progress_callback = None
):
    """Run the full retraining pipeline with progress reporting."""
    start_time = time.time()
    
    logger.info("=" * 60)
    logger.info("FULL MODEL RETRAINING PIPELINE")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    results = {}
    current_progress = 0
    
    # Calculate progress increments based on what we're doing
    num_genders = 0
    if not male_only:
        num_genders += 1
    if not female_only:
        num_genders += 1
    
    ingest_pct = 20 if not skip_ingest else 0
    elo_pct = 10 if not skip_elo else 0
    per_gender_pct = (100 - ingest_pct - elo_pct) / num_genders if num_genders > 0 else 0
    
    # Helper to report progress
    def report_progress(step_name):
        nonlocal current_progress
        if progress_callback:
            progress_callback(int(current_progress), step_name)
    
    # Step 1: Ingest data
    if not skip_ingest:
        report_progress("Step 1/7: Ingesting data from Cricsheet...")
        results['ingest'] = step_ingest_data()
        current_progress += ingest_pct
        report_progress("Data ingestion complete")
    else:
        logger.info("Skipping data ingestion (--skip-ingest)")
    
    # Step 2: Calculate ELO
    if not skip_elo:
        report_progress("Step 2/7: Calculating ELO ratings...")
        results['elo'] = step_calculate_elo()
        current_progress += elo_pct
        report_progress("ELO calculation complete")
    else:
        logger.info("Skipping ELO calculation (--skip-elo)")
    
    # Determine which genders to process
    genders = []
    if not female_only:
        genders.append('male')
    if not male_only:
        genders.append('female')
    
    for idx, gender in enumerate(genders, 1):
        logger.info(f"\n*** Processing {gender.upper()} data ***\n")
        
        # Track start time for this gender's training
        gender_start_time = time.time()
        
        # Steps 3-7 for each gender
        step_increment = per_gender_pct / 6  # 6 sub-steps per gender
        
        report_progress(f"Step 3/{gender.upper()}: Building player distributions...")
        results[f'player_dist_{gender}'] = step_build_player_distributions(gender)
        current_progress += step_increment
        
        report_progress(f"Step 4/{gender.upper()}: Building venue statistics...")
        results[f'venue_stats_{gender}'] = step_build_venue_stats(gender)
        current_progress += step_increment
        
        report_progress(f"Step 5/{gender.upper()}: Generating training data...")
        results[f'training_data_{gender}'] = step_generate_training_data(gender)
        current_progress += step_increment
        
        report_progress(f"Step 6/{gender.upper()}: Training neural network model...")
        results[f'train_{gender}'] = step_train_model(gender)
        current_progress += step_increment
        
        report_progress(f"Step 7/{gender.upper()}: Validating model...")
        results[f'validate_{gender}'] = step_validate_model(gender)
        current_progress += step_increment
        
        # Save model version metadata
        report_progress(f"Saving {gender.upper()} model version metadata...")
        gender_end_time = datetime.now()
        gender_start_datetime = datetime.fromtimestamp(gender_start_time)
        results[f'version_{gender}'] = save_model_version_metadata(gender, gender_start_datetime, gender_end_time)
        current_progress += step_increment
    
    elapsed = time.time() - start_time
    
    # Final progress report
    if progress_callback:
        progress_callback(100, "Pipeline complete!")
    
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Total time: {elapsed/60:.1f} minutes")
    logger.info("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Full model retraining pipeline')
    parser.add_argument('--skip-ingest', action='store_true', help='Skip data ingestion step')
    parser.add_argument('--skip-elo', action='store_true', help='Skip ELO calculation step')
    parser.add_argument('--male-only', action='store_true', help='Only train male model')
    parser.add_argument('--female-only', action='store_true', help='Only train female model')
    
    args = parser.parse_args()
    
    results = run_full_pipeline(
        skip_ingest=args.skip_ingest,
        skip_elo=args.skip_elo,
        male_only=args.male_only,
        female_only=args.female_only
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

