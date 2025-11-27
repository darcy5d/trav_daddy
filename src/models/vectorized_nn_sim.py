"""
Vectorized Neural Network Match Simulator.

Simulates N matches simultaneously using batched NN predictions.
Target: 1000 matches in ~10 seconds (50x speedup over sequential).

Key optimization: Instead of simulating 1 match at a time (240K NN calls for 1000 matches),
we simulate ALL matches in parallel (240 batched NN calls total).

Now includes venue features (29 total input features).
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
import pickle
import time

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import tensorflow as tf
from tensorflow import keras

from src.features.venue_stats import VenueStatsBuilder

logger = logging.getLogger(__name__)

# Outcome classes
NUM_CLASSES = 7
OUTCOME_RUNS = np.array([0, 1, 2, 3, 4, 6, 0])  # Class -> runs (6=wicket)
OUTCOME_NAMES = ['Dot', 'Single', 'Two', 'Three', 'Four', 'Six', 'Wicket']

# Bias correction: Real T20 data shows chasing team wins ~52-55%
# We balance this by giving first innings a slight scoring boost
FIRST_INNINGS_SCORE_BONUS = 1.05  # 5% bonus to first innings scores


class VectorizedNNSimulator:
    """
    Vectorized match simulator using batched neural network predictions.
    
    Simulates N matches simultaneously by:
    1. Tracking state for all N matches as numpy arrays
    2. Making ONE batched NN call per ball across all matches
    3. Vectorized state updates using numpy operations
    """
    
    def __init__(
        self,
        model_path: str = None,
        player_dist_path: str = None,
        venue_stats_path: str = None,
        format_type: str = 'T20',
        gender: str = 'male'
    ):
        """
        Initialize the vectorized NN simulator.
        
        Args:
            model_path: Path to trained model (auto-generated if None)
            player_dist_path: Path to player distributions (auto-generated if None)
            venue_stats_path: Path to venue stats (auto-generated if None)
            format_type: Match format ('T20', 'ODI')
            gender: 'male' or 'female'
        """
        self.format_type = format_type
        self.gender = gender
        
        # Auto-generate paths based on format and gender if not provided
        if model_path is None:
            model_path = f'data/processed/ball_prediction_model_{format_type.lower()}_{gender}.keras'
        if player_dist_path is None:
            player_dist_path = f'data/processed/player_distributions_{format_type.lower()}_{gender}.pkl'
        if venue_stats_path is None:
            venue_stats_path = f'data/processed/venue_stats_{format_type.lower()}_{gender}.pkl'
        
        # Load model
        self.model = keras.models.load_model(model_path)
        
        # Load normalizer
        normalizer_path = model_path.replace('.keras', '_normalizer.pkl')
        with open(normalizer_path, 'rb') as f:
            norm = pickle.load(f)
        self.mean = norm['mean']
        self.std = norm['std']
        
        # Load player distributions
        with open(player_dist_path, 'rb') as f:
            data = pickle.load(f)
        
        # Build player distribution lookup (player_id -> 8-element vector)
        self.batter_dists = {}
        for pid, d in data['batter_distributions'].items():
            self.batter_dists[int(pid)] = np.array(d['prob_vector'], dtype=np.float32)
        
        self.bowler_dists = {}
        for pid, d in data['bowler_distributions'].items():
            self.bowler_dists[int(pid)] = np.array(d['prob_vector'], dtype=np.float32)
        
        # Default distributions for unknown players
        self.default_bat_dist = np.array([0.35, 0.32, 0.08, 0.02, 0.12, 0.06, 0.04, 0.01], dtype=np.float32)
        self.default_bowl_dist = np.array([0.38, 0.32, 0.07, 0.01, 0.10, 0.05, 0.05, 0.02], dtype=np.float32)
        
        # Load venue statistics
        try:
            self.venue_stats = VenueStatsBuilder.load(venue_stats_path)
            logger.info(f"Loaded venue stats for {len(self.venue_stats.venue_stats)} venues")
        except Exception as e:
            logger.warning(f"Could not load venue stats: {e}")
            self.venue_stats = None
        
        # Default venue features (neutral)
        self.default_venue_features = np.array([1.0, 0.22, 0.042, 0.0], dtype=np.float32)
        
        logger.info(f"VectorizedNNSimulator initialized with {len(self.batter_dists)} batters, {len(self.bowler_dists)} bowlers")
    
    def get_batter_dist(self, player_id: int) -> np.ndarray:
        return self.batter_dists.get(player_id, self.default_bat_dist)
    
    def get_bowler_dist(self, player_id: int) -> np.ndarray:
        return self.bowler_dists.get(player_id, self.default_bowl_dist)
    
    def get_venue_features(self, venue_id: Optional[int]) -> np.ndarray:
        """Get 4-element venue feature vector."""
        if self.venue_stats is None or venue_id is None:
            return self.default_venue_features
        return self.venue_stats.get_venue_features(venue_id)
    
    def simulate_matches(
        self,
        n_matches: int,
        team1_batter_ids: List[int],
        team1_bowler_ids: List[int],
        team2_batter_ids: List[int],
        team2_bowler_ids: List[int],
        max_overs: int = 20,
        venue_id: Optional[int] = None
    ) -> Dict:
        """
        Simulate N matches in parallel using vectorized operations.
        
        Args:
            n_matches: Number of matches to simulate
            team1_batter_ids: List of 11 batter player IDs for team 1
            team1_bowler_ids: List of 5 bowler player IDs for team 1
            team2_batter_ids: List of 11 batter player IDs for team 2
            team2_bowler_ids: List of 5 bowler player IDs for team 2
            max_overs: Maximum overs per innings (20 for T20)
            venue_id: Optional venue ID for venue-specific effects
        
        Returns:
            Dict with simulation results
        """
        max_balls = max_overs * 6
        
        # Pre-compute player distribution matrices
        team1_bat_dists = np.array([self.get_batter_dist(pid) for pid in team1_batter_ids])  # (11, 8)
        team1_bowl_dists = np.array([self.get_bowler_dist(pid) for pid in team1_bowler_ids])  # (5, 8)
        team2_bat_dists = np.array([self.get_batter_dist(pid) for pid in team2_batter_ids])  # (11, 8)
        team2_bowl_dists = np.array([self.get_bowler_dist(pid) for pid in team2_bowler_ids])  # (5, 8)
        
        # Get venue features (4 features)
        venue_features = self.get_venue_features(venue_id)
        
        # ========== FIRST INNINGS (Team 1 bats, Team 2 bowls) ==========
        first_runs, first_wickets = self._simulate_innings_vectorized(
            n_matches=n_matches,
            batting_dists=team1_bat_dists,
            bowling_dists=team2_bowl_dists,
            innings_number=1,
            target=None,
            max_balls=max_balls,
            venue_features=venue_features
        )
        
        # ========== SECOND INNINGS (Team 2 bats, Team 1 bowls) ==========
        # Apply first innings bonus to balance win rates
        # Real T20 data shows batting first teams win ~45-48%
        adjusted_first_runs = (first_runs * FIRST_INNINGS_SCORE_BONUS).astype(np.int32)
        
        # Target is adjusted first innings score + 1
        targets = adjusted_first_runs + 1
        
        second_runs, second_wickets = self._simulate_innings_vectorized(
            n_matches=n_matches,
            batting_dists=team2_bat_dists,
            bowling_dists=team1_bowl_dists,
            innings_number=2,
            target=targets,
            max_balls=max_balls,
            venue_features=venue_features
        )
        
        # Determine winners
        team2_wins = second_runs >= targets
        team1_wins = ~team2_wins
        
        return {
            'n_matches': n_matches,
            'team1_win_prob': team1_wins.mean(),
            'team2_win_prob': team2_wins.mean(),
            'avg_team1_score': first_runs.mean(),
            'avg_team2_score': second_runs.mean(),
            'std_team1_score': first_runs.std(),
            'std_team2_score': second_runs.std(),
            'avg_team1_wickets': first_wickets.mean(),
            'avg_team2_wickets': second_wickets.mean(),
            'team1_score_range': (np.percentile(first_runs, 5), np.percentile(first_runs, 95)),
            'team2_score_range': (np.percentile(second_runs, 5), np.percentile(second_runs, 95)),
            'team1_scores': first_runs,
            'team2_scores': second_runs,
        }
    
    def _simulate_innings_vectorized(
        self,
        n_matches: int,
        batting_dists: np.ndarray,  # (11, 8)
        bowling_dists: np.ndarray,  # (5, 8)
        innings_number: int,
        target: Optional[np.ndarray],  # (n_matches,) or None
        max_balls: int = 120,
        venue_features: Optional[np.ndarray] = None  # (4,) venue feature vector
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate innings for all N matches simultaneously.
        
        Returns:
            (runs, wickets) arrays of shape (n_matches,)
        """
        # Default venue features if not provided
        if venue_features is None:
            venue_features = self.default_venue_features
        # State arrays for all matches
        runs = np.zeros(n_matches, dtype=np.int32)
        wickets = np.zeros(n_matches, dtype=np.int32)
        balls = np.zeros(n_matches, dtype=np.int32)
        
        # Current batter index for each match (0-10)
        current_batter = np.zeros(n_matches, dtype=np.int32)
        
        # Track which matches are still active
        active = np.ones(n_matches, dtype=bool)
        
        for ball_idx in range(max_balls):
            if not active.any():
                break
            
            over = ball_idx // 6
            
            # Determine phase (one-hot)
            if over < 6:
                phase = np.array([1, 0, 0], dtype=np.float32)
            elif over < 15:
                phase = np.array([0, 1, 0], dtype=np.float32)
            else:
                phase = np.array([0, 0, 1], dtype=np.float32)
            
            # Calculate required rate for active matches
            if target is not None:
                balls_remaining = max_balls - balls
                runs_needed = target - runs
                required_rate = np.where(
                    balls_remaining > 0,
                    np.maximum(0, runs_needed * 6 / balls_remaining),
                    0
                ).astype(np.float32)
            else:
                required_rate = np.zeros(n_matches, dtype=np.float32)
            
            # Get batter distributions for current batters
            # Clip to valid range (0-10)
            safe_batter_idx = np.clip(current_batter, 0, 10)
            batter_dist = batting_dists[safe_batter_idx]  # (n_matches, 8)
            
            # Get bowler distribution (simple rotation by over)
            bowler_idx = over % len(bowling_dists)
            bowler_dist = np.tile(bowling_dists[bowler_idx], (n_matches, 1))  # (n_matches, 8)
            
            # Build feature matrix (n_matches, 29)
            features = np.column_stack([
                np.full(n_matches, innings_number, dtype=np.float32),  # innings
                np.full(n_matches, over, dtype=np.float32),            # over
                balls.astype(np.float32),                               # balls bowled
                runs.astype(np.float32),                                # runs
                wickets.astype(np.float32),                             # wickets
                required_rate,                                          # required rate
                np.tile(phase, (n_matches, 1)),                         # phase (3)
                batter_dist,                                            # batter dist (8)
                bowler_dist,                                            # bowler dist (8)
                np.tile(venue_features, (n_matches, 1)),               # venue features (4)
            ])
            
            # Normalize features
            features_norm = (features - self.mean) / self.std
            
            # Batched NN prediction (single call for all matches!)
            proba = self.model.predict(features_norm, verbose=0, batch_size=n_matches)  # (n_matches, 7)
            
            # Sample outcomes for all matches
            outcomes = self._vectorized_sample(proba)  # (n_matches,)
            
            # Update state only for active matches
            runs_scored = OUTCOME_RUNS[outcomes]
            is_wicket = (outcomes == 6)
            
            runs = np.where(active, runs + runs_scored, runs)
            wickets = np.where(active & is_wicket, wickets + 1, wickets)
            balls = np.where(active, balls + 1, balls)
            
            # Move to next batter on wicket
            current_batter = np.where(
                active & is_wicket & (wickets < 10),
                np.minimum(wickets + 1, 10),
                current_batter
            )
            
            # Check termination conditions
            all_out = wickets >= 10
            if target is not None:
                target_reached = runs >= target
            else:
                target_reached = np.zeros(n_matches, dtype=bool)
            active = active & ~all_out & ~target_reached
        
        return runs, wickets
    
    def _vectorized_sample(self, proba: np.ndarray) -> np.ndarray:
        """
        Vectorized sampling from probability distributions.
        
        Args:
            proba: (n_samples, n_classes) probability matrix
        
        Returns:
            (n_samples,) array of sampled class indices
        """
        # Cumulative probabilities
        cumprob = np.cumsum(proba, axis=1)
        
        # Random values
        r = np.random.random(len(proba))[:, np.newaxis]
        
        # Find first cumprob >= r
        outcomes = (cumprob < r).sum(axis=1)
        
        # Clip to valid range
        return np.clip(outcomes, 0, NUM_CLASSES - 1)


def benchmark_vectorized_simulator():
    """Benchmark the vectorized simulator."""
    print("=" * 70)
    print("VECTORIZED NN SIMULATOR BENCHMARK")
    print("=" * 70)
    
    simulator = VectorizedNNSimulator()
    
    # Create dummy team IDs (will use default distributions for unknowns)
    team1_batters = list(range(1, 12))
    team1_bowlers = list(range(100, 105))
    team2_batters = list(range(200, 211))
    team2_bowlers = list(range(300, 305))
    
    # Warmup
    print("\nWarming up (10 matches)...")
    _ = simulator.simulate_matches(10, team1_batters, team1_bowlers, team2_batters, team2_bowlers)
    
    # Benchmark different batch sizes
    for n in [100, 500, 1000]:
        print(f"\nBenchmarking {n} matches...")
        start = time.time()
        results = simulator.simulate_matches(n, team1_batters, team1_bowlers, team2_batters, team2_bowlers)
        elapsed = time.time() - start
        
        print(f"  Time: {elapsed:.2f}s ({elapsed/n*1000:.1f}ms per match)")
        print(f"  Team1 Win: {results['team1_win_prob']:.1%}")
        print(f"  Avg Scores: {results['avg_team1_score']:.1f} vs {results['avg_team2_score']:.1f}")
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    benchmark_vectorized_simulator()

