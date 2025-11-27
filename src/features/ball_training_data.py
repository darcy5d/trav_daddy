"""
Ball Prediction Training Data Generator.

Creates training dataset for the Ball Prediction Neural Network.

For each delivery in the database, creates a feature vector containing:
- Match state: innings, over, balls, runs, wickets, target (if 2nd innings)
- Batter statistics: historical outcome distribution (8 features)
- Bowler statistics: historical outcome distribution (8 features)
- Phase: powerplay/middle/death (3 one-hot features)
- Required rate (if chasing)
- Venue features: scoring_factor, boundary_rate, wicket_rate, has_reliable_data (4 features)

Target: outcome class (0, 1, 2, 3, 4, 6, W)

Total features: 29 (6 + 3 + 8 + 8 + 4)
"""

import logging
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import DATABASE_PATH
from src.data.database import get_connection
from src.features.player_distributions import PlayerDistributionBuilder
from src.features.venue_stats import VenueStatsBuilder

logger = logging.getLogger(__name__)


def get_innings_phase(over: int) -> Tuple[int, int, int]:
    """
    Get one-hot encoded phase.
    
    Returns: (is_powerplay, is_middle, is_death)
    """
    if over < 6:
        return (1, 0, 0)  # Powerplay
    elif over < 15:
        return (0, 1, 0)  # Middle
    else:
        return (0, 0, 1)  # Death


def outcome_to_class(runs_batter: int, is_wicket: bool) -> int:
    """
    Convert delivery outcome to class label.
    
    Classes:
    0 = dot ball (0 runs, no wicket)
    1 = single (1 run)
    2 = two runs
    3 = three runs
    4 = four runs
    5 = six runs
    6 = wicket
    """
    if is_wicket:
        return 6  # Wicket
    elif runs_batter == 0:
        return 0  # Dot
    elif runs_batter == 1:
        return 1  # Single
    elif runs_batter == 2:
        return 2  # Two
    elif runs_batter == 3:
        return 3  # Three
    elif runs_batter == 4:
        return 4  # Four
    else:
        return 5  # Six (or more)


def class_to_outcome_name(class_idx: int) -> str:
    """Convert class index to human-readable name."""
    names = ['Dot', 'Single', 'Two', 'Three', 'Four', 'Six', 'Wicket']
    return names[class_idx]


class BallTrainingDataGenerator:
    """
    Generates training data for ball prediction neural network.
    """
    
    def __init__(self, format_type: str = 'T20', gender: str = 'male'):
        self.format_type = format_type
        self.gender = gender
        self.player_distributions = None
        self.venue_stats = None
        
    def load_player_distributions(self, min_balls: int = 10):
        """Load or build player distributions."""
        dist_path = Path(f'data/processed/player_distributions_{self.format_type.lower()}_{self.gender}.pkl')
        
        if dist_path.exists():
            logger.info(f"Loading player distributions from {dist_path}")
            self.player_distributions = PlayerDistributionBuilder.load(str(dist_path))
        else:
            logger.info("Building player distributions...")
            self.player_distributions = PlayerDistributionBuilder(self.format_type, self.gender)
            self.player_distributions.build_from_database(min_balls)
            self.player_distributions.save(str(dist_path))
    
    def load_venue_stats(self):
        """Load or build venue statistics."""
        venue_path = Path(f'data/processed/venue_stats_{self.format_type.lower()}_{self.gender}.pkl')
        
        if venue_path.exists():
            logger.info(f"Loading venue statistics from {venue_path}")
            self.venue_stats = VenueStatsBuilder.load(str(venue_path))
        else:
            logger.info("Building venue statistics...")
            self.venue_stats = VenueStatsBuilder(self.format_type, self.gender)
            self.venue_stats.build_from_database()
            self.venue_stats.save(str(venue_path))
    
    def generate_training_data(self, limit: int = None) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Generate training data from deliveries table.
        
        Returns:
            X: Feature matrix (n_samples, n_features) - 29 features
            y: Target vector (n_samples,) - class labels
            df: DataFrame with metadata (match_id, innings, etc.)
        """
        if self.player_distributions is None:
            self.load_player_distributions()
        
        if self.venue_stats is None:
            self.load_venue_stats()
        
        conn = get_connection()
        cursor = conn.cursor()
        
        logger.info("Querying deliveries with match context...")
        
        # Query deliveries with match and innings context (including venue_id)
        query = """
            SELECT 
                d.delivery_id,
                d.innings_id,
                d.over_number,
                d.ball_number,
                d.batter_id,
                d.bowler_id,
                d.runs_batter,
                d.is_wicket,
                i.innings_number,
                i.total_runs as innings_runs_so_far,
                i.total_wickets as innings_wickets_so_far,
                m.match_id,
                m.date,
                m.venue_id
            FROM deliveries d
            JOIN innings i ON d.innings_id = i.innings_id
            JOIN matches m ON i.match_id = m.match_id
            WHERE m.match_type = ? AND m.gender = ?
            ORDER BY m.date, d.delivery_id
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query, (self.format_type, self.gender))
        rows = cursor.fetchall()
        
        logger.info(f"Processing {len(rows)} deliveries...")
        
        # We need running totals per innings
        # Compute these as we iterate
        
        features = []
        targets = []
        metadata = []
        
        # Track innings state
        current_innings_id = None
        innings_runs = 0
        innings_wickets = 0
        innings_balls = 0
        first_innings_score = None
        
        for row in tqdm(rows, desc="Building features"):
            innings_id = row['innings_id']
            
            # Reset on new innings
            if innings_id != current_innings_id:
                # Save first innings score for second innings target
                if current_innings_id is not None and row['innings_number'] == 2:
                    # Look up first innings total
                    cursor.execute("""
                        SELECT total_runs FROM innings 
                        WHERE match_id = ? AND innings_number = 1
                    """, (row['match_id'],))
                    first_innings = cursor.fetchone()
                    first_innings_score = first_innings['total_runs'] if first_innings else None
                else:
                    first_innings_score = None
                
                current_innings_id = innings_id
                innings_runs = 0
                innings_wickets = 0
                innings_balls = 0
            
            # Get current state BEFORE this delivery
            over = row['over_number']
            ball = row['ball_number']
            
            # Calculate target/required rate for 2nd innings
            target = None
            required_rate = 0.0
            if row['innings_number'] == 2 and first_innings_score is not None:
                target = first_innings_score + 1
                balls_remaining = 120 - innings_balls
                if balls_remaining > 0:
                    runs_needed = target - innings_runs
                    required_rate = runs_needed * 6 / balls_remaining if runs_needed > 0 else 0
            
            # Phase features
            is_powerplay, is_middle, is_death = get_innings_phase(over)
            
            # Get player distributions
            batter_dist = self.player_distributions.get_batter_vector(row['batter_id'])
            bowler_dist = self.player_distributions.get_bowler_vector(row['bowler_id'])
            
            # Get venue features (4 features)
            venue_features = self.venue_stats.get_venue_features(row['venue_id'])
            
            # Build feature vector (29 total features)
            feature = np.concatenate([
                # Match state (6 features)
                np.array([
                    row['innings_number'],      # 1 or 2
                    over,                        # 0-19
                    innings_balls,               # balls bowled in innings so far
                    innings_runs,                # runs scored so far
                    innings_wickets,             # wickets lost so far
                    required_rate,               # required rate (0 if first innings)
                ]),
                # Phase one-hot (3 features)
                np.array([is_powerplay, is_middle, is_death]),
                # Batter distribution (8 features)
                batter_dist,
                # Bowler distribution (8 features)
                bowler_dist,
                # Venue features (4 features)
                venue_features
            ])
            
            features.append(feature)
            
            # Target: outcome class
            target_class = outcome_to_class(row['runs_batter'], row['is_wicket'])
            targets.append(target_class)
            
            # Metadata
            metadata.append({
                'delivery_id': row['delivery_id'],
                'match_id': row['match_id'],
                'innings_number': row['innings_number'],
                'over': over,
                'ball': ball,
                'batter_id': row['batter_id'],
                'bowler_id': row['bowler_id'],
                'venue_id': row['venue_id'],
                'date': row['date']
            })
            
            # Update innings state AFTER recording
            innings_balls += 1
            innings_runs += row['runs_batter']
            if row['is_wicket']:
                innings_wickets += 1
        
        conn.close()
        
        X = np.array(features, dtype=np.float32)
        y = np.array(targets, dtype=np.int64)
        df_meta = pd.DataFrame(metadata)
        
        logger.info(f"Generated {len(X)} samples with {X.shape[1]} features")
        
        return X, y, df_meta
    
    def save_training_data(self, output_dir: str = 'data/processed'):
        """Generate and save training data."""
        X, y, df_meta = self.generate_training_data()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save as numpy arrays
        np.save(output_path / f'ball_X_{self.format_type.lower()}_{self.gender}.npy', X)
        np.save(output_path / f'ball_y_{self.format_type.lower()}_{self.gender}.npy', y)
        df_meta.to_csv(output_path / f'ball_meta_{self.format_type.lower()}_{self.gender}.csv', index=False)
        
        logger.info(f"Saved training data to {output_path}")
        
        return X, y, df_meta


def analyze_class_distribution(y: np.ndarray):
    """Analyze and print class distribution."""
    print("\n" + "=" * 60)
    print("OUTCOME CLASS DISTRIBUTION")
    print("=" * 60)
    
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    for cls, count in zip(unique, counts):
        name = class_to_outcome_name(cls)
        pct = count * 100 / total
        bar = "█" * int(pct * 2)
        print(f"  {name:8}: {count:>8,} ({pct:>5.1f}%) {bar}")


def main(format_type: str = 'T20', gender: str = 'male'):
    """Generate ball prediction training data."""
    generator = BallTrainingDataGenerator(format_type, gender)
    
    print("=" * 70)
    print(f"BALL PREDICTION TRAINING DATA GENERATOR ({format_type} {gender.upper()})")
    print("=" * 70)
    
    X, y, df_meta = generator.save_training_data()
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    # Feature names for reference (29 features)
    feature_names = [
        'innings_number', 'over', 'balls_bowled', 'runs', 'wickets', 'required_rate',
        'is_powerplay', 'is_middle', 'is_death',
        'bat_p0', 'bat_p1', 'bat_p2', 'bat_p3', 'bat_p4', 'bat_p6', 'bat_pW', 'bat_pEx',
        'bowl_p0', 'bowl_p1', 'bowl_p2', 'bowl_p3', 'bowl_p4', 'bowl_p6', 'bowl_pW', 'bowl_pEx',
        'venue_scoring_factor', 'venue_boundary_rate', 'venue_wicket_rate', 'venue_reliable'
    ]
    
    print(f"\nFeature names ({len(feature_names)} features):")
    for i, name in enumerate(feature_names):
        print(f"  {i:2}: {name}")
    
    analyze_class_distribution(y)
    
    # Show sample
    print("\n" + "=" * 60)
    print("SAMPLE FEATURE VECTOR (first delivery)")
    print("=" * 60)
    for i, (name, val) in enumerate(zip(feature_names, X[0])):
        print(f"  {name:15}: {val:.4f}")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    parser = argparse.ArgumentParser(description='Generate ball prediction training data')
    parser.add_argument('--format', default='T20', choices=['T20', 'ODI'])
    parser.add_argument('--gender', default='male', choices=['male', 'female'])
    args = parser.parse_args()
    
    main(args.format, args.gender)


