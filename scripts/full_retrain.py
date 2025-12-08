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


def run_full_pipeline(
    skip_ingest: bool = False,
    skip_elo: bool = False,
    male_only: bool = False,
    female_only: bool = False
):
    """Run the full retraining pipeline."""
    start_time = time.time()
    
    logger.info("=" * 60)
    logger.info("FULL MODEL RETRAINING PIPELINE")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    results = {}
    
    # Step 1: Ingest data
    if not skip_ingest:
        results['ingest'] = step_ingest_data()
    else:
        logger.info("Skipping data ingestion (--skip-ingest)")
    
    # Step 2: Calculate ELO
    if not skip_elo:
        results['elo'] = step_calculate_elo()
    else:
        logger.info("Skipping ELO calculation (--skip-elo)")
    
    # Determine which genders to process
    genders = []
    if not female_only:
        genders.append('male')
    if not male_only:
        genders.append('female')
    
    for gender in genders:
        logger.info(f"\n*** Processing {gender.upper()} data ***\n")
        
        # Steps 3-7 for each gender
        results[f'player_dist_{gender}'] = step_build_player_distributions(gender)
        results[f'venue_stats_{gender}'] = step_build_venue_stats(gender)
        results[f'training_data_{gender}'] = step_generate_training_data(gender)
        results[f'train_{gender}'] = step_train_model(gender)
        results[f'validate_{gender}'] = step_validate_model(gender)
    
    elapsed = time.time() - start_time
    
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

