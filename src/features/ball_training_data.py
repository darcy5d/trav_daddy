"""
Ball Prediction Training Data Generator.

Creates training dataset for the Ball Prediction Neural Network.

For each delivery in the database, creates a feature vector containing:
- Match state: innings, over, balls, runs, wickets, target (if 2nd innings) - 6 features
- Phase: powerplay/middle/death - 3 one-hot features
- Batter statistics: historical outcome distribution - 8 features
- Bowler statistics: historical outcome distribution - 8 features
- Venue features: scoring_factor, boundary_rate, wicket_rate, has_reliable_data - 4 features
- Team ELO features: batting_team_elo, bowling_team_elo, team_elo_diff - 3 features
- Player ELO features: batter_elo, bowler_elo - 2 features

Target: outcome class (0, 1, 2, 3, 4, 6, W)

Total features: 34 (6 + 3 + 8 + 8 + 4 + 3 + 2)

Note: ELO features use HISTORICAL ELOs at match date, not current ELOs.
This ensures the model learns from temporally-consistent data.
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


# =============================================================================
# ELO Lookup Functions (for temporally-consistent training data)
# =============================================================================

def build_team_elo_lookup(cursor, format_type: str, gender: str) -> Dict:
    """
    Pre-load team ELO history into memory for fast lookup.
    
    Returns dict: {(team_id, year_month): elo}
    Uses monthly snapshots for efficiency.
    """
    logger.info("Loading team ELO history for fast lookup...")
    
    cursor.execute("""
        SELECT team_id, date, elo
        FROM team_elo_history
        WHERE format = ? AND gender = ?
        ORDER BY team_id, date
    """, (format_type, gender))
    
    rows = cursor.fetchall()
    
    # Build lookup: for each team, store ELO at each date
    team_elo_by_date = {}
    for row in rows:
        team_id = row['team_id']
        date_str = str(row['date'])[:10]  # YYYY-MM-DD
        elo = row['elo']
        
        if team_id not in team_elo_by_date:
            team_elo_by_date[team_id] = []
        team_elo_by_date[team_id].append((date_str, elo))
    
    logger.info(f"Loaded ELO history for {len(team_elo_by_date)} teams")
    return team_elo_by_date


def build_player_elo_lookup(cursor, format_type: str, gender: str) -> Dict:
    """
    Pre-load player ELO history into memory for fast lookup.
    
    Returns dict: {(player_id, year_month): {'batting': elo, 'bowling': elo}}
    """
    logger.info("Loading player ELO history for fast lookup...")
    
    cursor.execute("""
        SELECT player_id, date, batting_elo, bowling_elo
        FROM player_elo_history
        WHERE format = ? AND gender = ?
        ORDER BY player_id, date
    """, (format_type, gender))
    
    rows = cursor.fetchall()
    
    player_elo_by_date = {}
    for row in rows:
        player_id = row['player_id']
        date_str = str(row['date'])[:10]
        
        if player_id not in player_elo_by_date:
            player_elo_by_date[player_id] = []
        player_elo_by_date[player_id].append((
            date_str, 
            row['batting_elo'], 
            row['bowling_elo']
        ))
    
    logger.info(f"Loaded ELO history for {len(player_elo_by_date)} players")
    return player_elo_by_date


def get_team_elo_at_date(team_elo_lookup: Dict, team_id: int, match_date: str) -> float:
    """
    Get team ELO as of match date using pre-loaded lookup.
    
    Uses binary search for efficiency.
    """
    if team_id not in team_elo_lookup:
        return 1500.0
    
    history = team_elo_lookup[team_id]
    
    # Find most recent ELO before match_date
    best_elo = 1500.0
    for date_str, elo in history:
        if date_str <= match_date:
            best_elo = elo
        else:
            break  # History is sorted, so we can stop
    
    return best_elo


def get_player_elo_at_date(
    player_elo_lookup: Dict, 
    player_id: int, 
    match_date: str, 
    elo_type: str = 'batting'
) -> float:
    """
    Get player ELO as of match date using pre-loaded lookup.
    
    elo_type: 'batting' or 'bowling'
    """
    if player_id not in player_elo_lookup:
        return 1500.0
    
    history = player_elo_lookup[player_id]
    
    best_batting = 1500.0
    best_bowling = 1500.0
    
    for date_str, batting_elo, bowling_elo in history:
        if date_str <= match_date:
            best_batting = batting_elo
            best_bowling = bowling_elo
        else:
            break
    
    return best_batting if elo_type == 'batting' else best_bowling


def normalize_elo(elo: float) -> float:
    """Normalize ELO to roughly [-2, 2] range."""
    return (elo - 1500) / 200


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
    
    Features (34 total):
    - Match state: 6 features
    - Phase: 3 features (one-hot)
    - Batter distribution: 8 features
    - Bowler distribution: 8 features
    - Venue: 4 features
    - Team ELO: 3 features (batting_team, bowling_team, diff)
    - Player ELO: 2 features (batter, bowler)
    """
    
    def __init__(self, format_type: str = 'T20', gender: str = 'male'):
        self.format_type = format_type
        self.gender = gender
        self.player_distributions = None
        self.venue_stats = None
        self.team_elo_lookup = None
        self.player_elo_lookup = None
        
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
    
    def load_elo_lookups(self, cursor):
        """Load ELO history lookups for fast access during training data generation."""
        if self.team_elo_lookup is None:
            self.team_elo_lookup = build_team_elo_lookup(cursor, self.format_type, self.gender)
        if self.player_elo_lookup is None:
            self.player_elo_lookup = build_player_elo_lookup(cursor, self.format_type, self.gender)
    
    def generate_training_data(self, limit: int = None) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Generate training data from deliveries table.
        
        Returns:
            X: Feature matrix (n_samples, n_features) - 34 features
            y: Target vector (n_samples,) - class labels
            df: DataFrame with metadata (match_id, innings, etc.)
        """
        if self.player_distributions is None:
            self.load_player_distributions()
        
        if self.venue_stats is None:
            self.load_venue_stats()
        
        conn = get_connection()
        cursor = conn.cursor()
        
        # Load ELO lookups for fast historical lookup
        self.load_elo_lookups(cursor)
        
        logger.info("Querying deliveries with match context...")
        
        # Query deliveries with match and innings context (including team IDs for ELO)
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
                i.batting_team_id,
                i.bowling_team_id,
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
            
            # Get ELO features (5 features) - using HISTORICAL ELOs at match date
            match_date = str(row['date'])[:10]  # YYYY-MM-DD
            
            # Team ELOs
            batting_team_elo = get_team_elo_at_date(
                self.team_elo_lookup, row['batting_team_id'], match_date
            )
            bowling_team_elo = get_team_elo_at_date(
                self.team_elo_lookup, row['bowling_team_id'], match_date
            )
            team_elo_diff = batting_team_elo - bowling_team_elo
            
            # Player ELOs
            batter_elo = get_player_elo_at_date(
                self.player_elo_lookup, row['batter_id'], match_date, 'batting'
            )
            bowler_elo = get_player_elo_at_date(
                self.player_elo_lookup, row['bowler_id'], match_date, 'bowling'
            )
            
            # Build feature vector (34 total features)
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
                venue_features,
                # Team ELO features (3 features) - normalized
                np.array([
                    normalize_elo(batting_team_elo),
                    normalize_elo(bowling_team_elo),
                    team_elo_diff / 200  # raw difference, normalized
                ]),
                # Player ELO features (2 features) - normalized
                np.array([
                    normalize_elo(batter_elo),
                    normalize_elo(bowler_elo)
                ])
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
    
    # Feature names for reference (34 features)
    feature_names = [
        'innings_number', 'over', 'balls_bowled', 'runs', 'wickets', 'required_rate',
        'is_powerplay', 'is_middle', 'is_death',
        'bat_p0', 'bat_p1', 'bat_p2', 'bat_p3', 'bat_p4', 'bat_p6', 'bat_pW', 'bat_pEx',
        'bowl_p0', 'bowl_p1', 'bowl_p2', 'bowl_p3', 'bowl_p4', 'bowl_p6', 'bowl_pW', 'bowl_pEx',
        'venue_scoring_factor', 'venue_boundary_rate', 'venue_wicket_rate', 'venue_reliable',
        'batting_team_elo', 'bowling_team_elo', 'team_elo_diff',
        'batter_elo', 'bowler_elo'
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


