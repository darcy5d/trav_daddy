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

# NumPy BLAS threading for parallel matrix operations
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
import pickle
import time
import multiprocessing

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import tensorflow as tf
from tensorflow import keras

# Configure TensorFlow threading for Apple M2 Pro
# Use multiple threads for parallel operations
N_CPU_CORES = multiprocessing.cpu_count()
tf.config.threading.set_inter_op_parallelism_threads(max(4, N_CPU_CORES // 2))
tf.config.threading.set_intra_op_parallelism_threads(max(4, N_CPU_CORES // 2))

# Enable Metal GPU acceleration (Apple Silicon)
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        logging.info(f"Metal GPU enabled: {physical_devices}")
except Exception as e:
    logging.warning(f"Could not configure Metal GPU: {e}")

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
        venue_id: Optional[int] = None,
        use_toss: bool = False,
        toss_field_prob: float = 0.65
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
            use_toss: If True, simulate toss for each match (50/50 winner, then choose bat/field)
            toss_field_prob: Probability winner chooses to field (default 0.65 for T20)
        
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
        
        if use_toss:
            # ========== TOSS SIMULATION (per-match) ==========
            # Step 1: 50/50 who wins toss
            team1_wins_toss = np.random.random(n_matches) < 0.5  # (n_matches,)
            
            # Step 2: Winner chooses bat (1-toss_field_prob) or field (toss_field_prob)
            winner_chooses_field = np.random.random(n_matches) < toss_field_prob  # (n_matches,)
            
            # Step 3: Determine who bats first
            # Team1 bats first if: (Team1 wins toss AND chooses bat) OR (Team2 wins toss AND chooses field)
            team1_bats_first = (team1_wins_toss & ~winner_chooses_field) | (~team1_wins_toss & winner_chooses_field)
            
            # Split into two groups
            team1_first_mask = team1_bats_first
            team2_first_mask = ~team1_bats_first
            n_team1_first = team1_first_mask.sum()
            n_team2_first = team2_first_mask.sum()
            
            # Initialize result arrays
            team1_scores = np.zeros(n_matches, dtype=np.int32)
            team2_scores = np.zeros(n_matches, dtype=np.int32)
            team1_wins = np.zeros(n_matches, dtype=bool)
            
            # ========== Simulate matches where Team 1 bats first ==========
            if n_team1_first > 0:
                first_runs_t1, _ = self._simulate_innings_vectorized(
                    n_matches=n_team1_first,
                    batting_dists=team1_bat_dists,
                    bowling_dists=team2_bowl_dists,
                    innings_number=1,
                    target=None,
                    max_balls=max_balls,
                    venue_features=venue_features
                )
                adjusted_first_runs_t1 = (first_runs_t1 * FIRST_INNINGS_SCORE_BONUS).astype(np.int32)
                targets_t1 = adjusted_first_runs_t1 + 1
                
                second_runs_t1, _ = self._simulate_innings_vectorized(
                    n_matches=n_team1_first,
                    batting_dists=team2_bat_dists,
                    bowling_dists=team1_bowl_dists,
                    innings_number=2,
                    target=targets_t1,
                    max_balls=max_balls,
                    venue_features=venue_features
                )
                
                team1_scores[team1_first_mask] = first_runs_t1
                team2_scores[team1_first_mask] = second_runs_t1
                team1_wins[team1_first_mask] = second_runs_t1 < targets_t1
            
            # ========== Simulate matches where Team 2 bats first ==========
            if n_team2_first > 0:
                first_runs_t2, _ = self._simulate_innings_vectorized(
                    n_matches=n_team2_first,
                    batting_dists=team2_bat_dists,
                    bowling_dists=team1_bowl_dists,
                    innings_number=1,
                    target=None,
                    max_balls=max_balls,
                    venue_features=venue_features
                )
                adjusted_first_runs_t2 = (first_runs_t2 * FIRST_INNINGS_SCORE_BONUS).astype(np.int32)
                targets_t2 = adjusted_first_runs_t2 + 1
                
                second_runs_t2, _ = self._simulate_innings_vectorized(
                    n_matches=n_team2_first,
                    batting_dists=team1_bat_dists,
                    bowling_dists=team2_bowl_dists,
                    innings_number=2,
                    target=targets_t2,
                    max_balls=max_balls,
                    venue_features=venue_features
                )
                
                # Note: Team 2 batted first, so their score is first_runs_t2
                team2_scores[team2_first_mask] = first_runs_t2
                team1_scores[team2_first_mask] = second_runs_t2
                team1_wins[team2_first_mask] = second_runs_t2 >= targets_t2
            
            toss_stats = {
                'team1_won_toss_pct': team1_wins_toss.mean(),
                'chose_field_pct': winner_chooses_field.mean(),
                'team1_batted_first_pct': team1_bats_first.mean()
            }
        else:
            # No toss - Team 1 always bats first
            first_runs, first_wickets = self._simulate_innings_vectorized(
                n_matches=n_matches,
                batting_dists=team1_bat_dists,
                bowling_dists=team2_bowl_dists,
                innings_number=1,
                target=None,
                max_balls=max_balls,
                venue_features=venue_features
            )
            
            adjusted_first_runs = (first_runs * FIRST_INNINGS_SCORE_BONUS).astype(np.int32)
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
            
            team1_scores = first_runs
            team2_scores = second_runs
            team1_wins = second_runs < targets
            toss_stats = None
        
        results = {
            'n_matches': n_matches,
            'team1_win_prob': team1_wins.mean(),
            'team2_win_prob': (~team1_wins).mean(),
            'avg_team1_score': team1_scores.mean(),
            'avg_team2_score': team2_scores.mean(),
            'std_team1_score': team1_scores.std(),
            'std_team2_score': team2_scores.std(),
            'team1_score_range': (np.percentile(team1_scores, 5), np.percentile(team1_scores, 95)),
            'team2_score_range': (np.percentile(team2_scores, 5), np.percentile(team2_scores, 95)),
            'team1_scores': team1_scores,
            'team2_scores': team2_scores,
        }
        
        if toss_stats:
            results['toss_stats'] = toss_stats
        
        return results
    
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
        
        # Current batter index for each match (clip to available batters)
        current_batter = np.zeros(n_matches, dtype=np.int32)
        max_batter_idx = len(batting_dists) - 1  # Handle teams with <11 batters
        
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
            # Clip to valid range (0 to num_batters-1)
            safe_batter_idx = np.clip(current_batter, 0, max_batter_idx)
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
            
            # Move to next batter on wicket (clip to available batters)
            current_batter = np.where(
                active & is_wicket & (wickets < 10),
                np.minimum(wickets + 1, max_batter_idx),
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
    
    def simulate_detailed_match(
        self,
        team1_batter_ids: List[int],
        team1_bowler_ids: List[int],
        team2_batter_ids: List[int],
        team2_bowler_ids: List[int],
        max_overs: int = 20,
        venue_id: Optional[int] = None,
        team1_bats_first: bool = True
    ) -> Dict:
        """
        Simulate ONE match with full ball-by-ball tracking for scorecard display.
        
        Returns detailed per-player batting and bowling stats.
        """
        max_balls = max_overs * 6
        venue_features = self.get_venue_features(venue_id)
        
        if team1_bats_first:
            # First innings: Team 1 bats
            inn1_batting, inn1_bowling = self._simulate_innings_detailed(
                batter_ids=team1_batter_ids,
                bowler_ids=team2_bowler_ids,
                innings_number=1,
                target=None,
                max_balls=max_balls,
                venue_features=venue_features
            )
            team1_total = sum(b['runs'] for b in inn1_batting)
            team1_wickets = sum(1 for b in inn1_batting if b['dismissal'] != 'not out')
            
            # Second innings: Team 2 chases
            target = int(team1_total * FIRST_INNINGS_SCORE_BONUS) + 1
            inn2_batting, inn2_bowling = self._simulate_innings_detailed(
                batter_ids=team2_batter_ids,
                bowler_ids=team1_bowler_ids,
                innings_number=2,
                target=target,
                max_balls=max_balls,
                venue_features=venue_features
            )
            team2_total = sum(b['runs'] for b in inn2_batting)
            team2_wickets = sum(1 for b in inn2_batting if b['dismissal'] != 'not out')
            
            return {
                'team1_batting': inn1_batting,
                'team1_bowling': inn2_bowling,
                'team2_batting': inn2_batting,
                'team2_bowling': inn1_bowling,
                'team1_total': team1_total,
                'team1_wickets': team1_wickets,
                'team1_overs': sum(b['balls'] for b in inn1_batting) / 6,
                'team2_total': team2_total,
                'team2_wickets': team2_wickets,
                'team2_overs': sum(b['balls'] for b in inn2_batting) / 6,
                'team1_won': team2_total < target,
                'target': target,
                'result': self._format_result(team1_total, team1_wickets, team2_total, team2_wickets, target, True)
            }
        else:
            # Team 2 bats first
            inn1_batting, inn1_bowling = self._simulate_innings_detailed(
                batter_ids=team2_batter_ids,
                bowler_ids=team1_bowler_ids,
                innings_number=1,
                target=None,
                max_balls=max_balls,
                venue_features=venue_features
            )
            team2_total = sum(b['runs'] for b in inn1_batting)
            team2_wickets = sum(1 for b in inn1_batting if b['dismissal'] != 'not out')
            
            target = int(team2_total * FIRST_INNINGS_SCORE_BONUS) + 1
            inn2_batting, inn2_bowling = self._simulate_innings_detailed(
                batter_ids=team1_batter_ids,
                bowler_ids=team2_bowler_ids,
                innings_number=2,
                target=target,
                max_balls=max_balls,
                venue_features=venue_features
            )
            team1_total = sum(b['runs'] for b in inn2_batting)
            team1_wickets = sum(1 for b in inn2_batting if b['dismissal'] != 'not out')
            
            return {
                'team1_batting': inn2_batting,
                'team1_bowling': inn1_bowling,
                'team2_batting': inn1_batting,
                'team2_bowling': inn2_bowling,
                'team1_total': team1_total,
                'team1_wickets': team1_wickets,
                'team1_overs': sum(b['balls'] for b in inn2_batting) / 6,
                'team2_total': team2_total,
                'team2_wickets': team2_wickets,
                'team2_overs': sum(b['balls'] for b in inn1_batting) / 6,
                'team1_won': team1_total >= target,
                'target': target,
                'result': self._format_result(team1_total, team1_wickets, team2_total, team2_wickets, target, False)
            }
    
    def _format_result(self, t1_score, t1_wkts, t2_score, t2_wkts, target, t1_batted_first):
        """Format the match result string."""
        if t1_batted_first:
            if t2_score >= target:
                return f"Team 2 won by {10 - t2_wkts} wickets"
            else:
                return f"Team 1 won by {target - 1 - t2_score} runs"
        else:
            if t1_score >= target:
                return f"Team 1 won by {10 - t1_wkts} wickets"
            else:
                return f"Team 2 won by {target - 1 - t1_score} runs"
    
    def _simulate_innings_detailed(
        self,
        batter_ids: List[int],
        bowler_ids: List[int],
        innings_number: int,
        target: Optional[int],
        max_balls: int = 120,
        venue_features: Optional[np.ndarray] = None
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Simulate one innings with detailed ball-by-ball tracking.
        
        CRICKET LAW: Each batter can only bat ONCE per innings.
        
        Returns:
            (batting_cards, bowling_cards)
        """
        if venue_features is None:
            venue_features = self.default_venue_features
        
        # VALIDATION: Check for duplicate batter IDs (cricket law violation)
        unique_batter_ids = set(batter_ids)
        if len(unique_batter_ids) != len(batter_ids):
            # Find duplicates for logging
            from collections import Counter
            counts = Counter(batter_ids)
            duplicates = {pid: count for pid, count in counts.items() if count > 1}
            logger.warning(f"DUPLICATE BATTER IDS DETECTED: {duplicates}")
            # Deduplicate, keeping order
            seen = set()
            deduped = []
            for pid in batter_ids:
                if pid not in seen:
                    deduped.append(pid)
                    seen.add(pid)
            batter_ids = deduped
            logger.info(f"Deduplicated to {len(batter_ids)} batters: {batter_ids}")
        
        # Handle fewer than 11 batters (shouldn't happen, but be resilient)
        if len(batter_ids) < 11:
            logger.warning(f"Only {len(batter_ids)} batters provided, expected 11")
        
        # Initialize batting stats for each UNIQUE batter
        batting = [{
            'player_id': pid,
            'runs': 0,
            'balls': 0,
            'fours': 0,
            'sixes': 0,
            'dismissal': 'not out',
            'dismissed_by': None,
            'batting_position': i + 1  # 1-indexed batting position
        } for i, pid in enumerate(batter_ids)]
        
        # Initialize bowling stats
        bowling = [{
            'player_id': pid,
            'balls': 0,
            'runs': 0,
            'wickets': 0,
            'dots': 0,
            'fours': 0,
            'sixes': 0
        } for pid in bowler_ids]
        
        # Get distribution matrices
        bat_dists = np.array([self.get_batter_dist(pid) for pid in batter_ids])
        bowl_dists = np.array([self.get_bowler_dist(pid) for pid in bowler_ids])
        
        # State tracking (using INDICES into the batting array, not player_ids)
        runs = 0
        wickets = 0
        current_batter_idx = 0  # Batter #1 (index 0) on strike
        non_striker_idx = 1     # Batter #2 (index 1) at non-striker's end
        next_batter_in = 2      # Batter #3 (index 2) is next to come in
        
        # Track which batters have been at the crease (can't bat again once out)
        batters_used = {0, 1}  # Openers are at the crease
        
        for ball_idx in range(max_balls):
            if wickets >= 10:
                break
            if target is not None and runs >= target:
                break
            
            over = ball_idx // 6
            ball_in_over = ball_idx % 6
            
            # Pick bowler (rotate each over)
            bowler_idx = over % len(bowler_ids)
            
            # Calculate required rate
            if target is not None:
                balls_remaining = max_balls - ball_idx
                runs_needed = target - runs
                req_rate = max(0, runs_needed * 6 / balls_remaining) if balls_remaining > 0 else 0
            else:
                req_rate = 0
            
            # Phase
            if over < 6:
                phase = np.array([1, 0, 0], dtype=np.float32)
            elif over < 15:
                phase = np.array([0, 1, 0], dtype=np.float32)
            else:
                phase = np.array([0, 0, 1], dtype=np.float32)
            
            # Build features
            batter_dist = bat_dists[current_batter_idx]
            bowler_dist = bowl_dists[bowler_idx]
            
            features = np.array([[
                innings_number,
                over,
                ball_idx,
                runs,
                wickets,
                req_rate,
                *phase,
                *batter_dist,
                *bowler_dist,
                *venue_features
            ]], dtype=np.float32)
            
            # Normalize and predict
            features_norm = (features - self.mean) / self.std
            proba = self.model.predict(features_norm, verbose=0)
            outcome = self._vectorized_sample(proba)[0]
            
            # Update batting stats
            batting[current_batter_idx]['balls'] += 1
            bowling[bowler_idx]['balls'] += 1
            
            if outcome == 0:  # Dot
                bowling[bowler_idx]['dots'] += 1
            elif outcome == 1:  # Single
                runs += 1
                batting[current_batter_idx]['runs'] += 1
                bowling[bowler_idx]['runs'] += 1
                # Rotate strike
                current_batter_idx, non_striker_idx = non_striker_idx, current_batter_idx
            elif outcome == 2:  # Two
                runs += 2
                batting[current_batter_idx]['runs'] += 2
                bowling[bowler_idx]['runs'] += 2
            elif outcome == 3:  # Three
                runs += 3
                batting[current_batter_idx]['runs'] += 3
                bowling[bowler_idx]['runs'] += 3
                current_batter_idx, non_striker_idx = non_striker_idx, current_batter_idx
            elif outcome == 4:  # Four
                runs += 4
                batting[current_batter_idx]['runs'] += 4
                batting[current_batter_idx]['fours'] += 1
                bowling[bowler_idx]['runs'] += 4
                bowling[bowler_idx]['fours'] += 1
            elif outcome == 5:  # Six
                runs += 6
                batting[current_batter_idx]['runs'] += 6
                batting[current_batter_idx]['sixes'] += 1
                bowling[bowler_idx]['runs'] += 6
                bowling[bowler_idx]['sixes'] += 1
            elif outcome == 6:  # Wicket
                bowling[bowler_idx]['wickets'] += 1
                batting[current_batter_idx]['dismissal'] = 'bowled'
                batting[current_batter_idx]['dismissed_by'] = bowler_ids[bowler_idx]
                wickets += 1
                
                # Bring in next batter from pavilion (non-striker stays at their end)
                # Cricket law: each batter can only bat once
                if wickets < 10 and next_batter_in < len(batter_ids):
                    # Verify the next batter hasn't already batted
                    if next_batter_in in batters_used:
                        logger.error(f"BUG: Batter index {next_batter_in} already used! batters_used={batters_used}")
                    
                    current_batter_idx = next_batter_in
                    batters_used.add(next_batter_in)
                    next_batter_in += 1
            
            # Rotate strike at end of over
            if ball_in_over == 5 and outcome not in [1, 3]:
                current_batter_idx, non_striker_idx = non_striker_idx, current_batter_idx
        
        # Calculate derived stats
        for b in batting:
            b['strike_rate'] = round(b['runs'] * 100 / b['balls'], 1) if b['balls'] > 0 else 0
        
        for b in bowling:
            overs_bowled = b['balls'] // 6 + (b['balls'] % 6) / 10  # 2.3 format
            b['overs'] = round(overs_bowled, 1)
            b['economy'] = round(b['runs'] * 6 / b['balls'], 2) if b['balls'] > 0 else 0
        
        return batting, bowling
    
    def calculate_expected_stats(
        self,
        team1_batter_ids: List[int],
        team1_bowler_ids: List[int],
        team2_batter_ids: List[int],
        team2_bowler_ids: List[int],
        n_simulations: int = 100,
        venue_id: Optional[int] = None
    ) -> Dict:
        """
        Calculate expected (average) player stats across multiple simulations.
        
        Runs n_simulations detailed matches and aggregates stats.
        """
        # Accumulators for Team 1 batting
        t1_bat_stats = [{
            'player_id': pid,
            'total_runs': 0,
            'total_balls': 0,
            'total_fours': 0,
            'total_sixes': 0,
            'times_out': 0,
            'simulations': 0
        } for pid in team1_batter_ids]
        
        # Accumulators for Team 2 batting
        t2_bat_stats = [{
            'player_id': pid,
            'total_runs': 0,
            'total_balls': 0,
            'total_fours': 0,
            'total_sixes': 0,
            'times_out': 0,
            'simulations': 0
        } for pid in team2_batter_ids]
        
        # Accumulators for bowling (team1 bowling = team2's bowlers facing team1)
        t1_bowl_stats = [{
            'player_id': pid,
            'total_balls': 0,
            'total_runs': 0,
            'total_wickets': 0,
            'total_dots': 0,
            'simulations': 0
        } for pid in team1_bowler_ids]
        
        t2_bowl_stats = [{
            'player_id': pid,
            'total_balls': 0,
            'total_runs': 0,
            'total_wickets': 0,
            'total_dots': 0,
            'simulations': 0
        } for pid in team2_bowler_ids]
        
        for _ in range(n_simulations):
            # 50% chance each team bats first
            t1_bats_first = np.random.random() < 0.5
            
            result = self.simulate_detailed_match(
                team1_batter_ids, team1_bowler_ids,
                team2_batter_ids, team2_bowler_ids,
                venue_id=venue_id,
                team1_bats_first=t1_bats_first
            )
            
            # Aggregate batting stats
            for i, b in enumerate(result['team1_batting']):
                t1_bat_stats[i]['total_runs'] += b['runs']
                t1_bat_stats[i]['total_balls'] += b['balls']
                t1_bat_stats[i]['total_fours'] += b['fours']
                t1_bat_stats[i]['total_sixes'] += b['sixes']
                if b['dismissal'] != 'not out':
                    t1_bat_stats[i]['times_out'] += 1
                t1_bat_stats[i]['simulations'] += 1
            
            for i, b in enumerate(result['team2_batting']):
                t2_bat_stats[i]['total_runs'] += b['runs']
                t2_bat_stats[i]['total_balls'] += b['balls']
                t2_bat_stats[i]['total_fours'] += b['fours']
                t2_bat_stats[i]['total_sixes'] += b['sixes']
                if b['dismissal'] != 'not out':
                    t2_bat_stats[i]['times_out'] += 1
                t2_bat_stats[i]['simulations'] += 1
            
            # Aggregate bowling stats (team1_bowling is the bowlers from team1 facing team2)
            for i, b in enumerate(result['team1_bowling']):
                t1_bowl_stats[i]['total_balls'] += b['balls']
                t1_bowl_stats[i]['total_runs'] += b['runs']
                t1_bowl_stats[i]['total_wickets'] += b['wickets']
                t1_bowl_stats[i]['total_dots'] += b['dots']
                t1_bowl_stats[i]['simulations'] += 1
            
            for i, b in enumerate(result['team2_bowling']):
                t2_bowl_stats[i]['total_balls'] += b['balls']
                t2_bowl_stats[i]['total_runs'] += b['runs']
                t2_bowl_stats[i]['total_wickets'] += b['wickets']
                t2_bowl_stats[i]['total_dots'] += b['dots']
                t2_bowl_stats[i]['simulations'] += 1
        
        # Calculate averages
        def avg_batting(stats):
            n = stats['simulations']
            if n == 0:
                return None
            return {
                'player_id': stats['player_id'],
                'avg_runs': round(stats['total_runs'] / n, 1),
                'avg_balls': round(stats['total_balls'] / n, 1),
                'avg_fours': round(stats['total_fours'] / n, 1),
                'avg_sixes': round(stats['total_sixes'] / n, 1),
                'dismissal_rate': round(stats['times_out'] / n * 100, 1),
                'strike_rate': round(stats['total_runs'] * 100 / stats['total_balls'], 1) if stats['total_balls'] > 0 else 0
            }
        
        def avg_bowling(stats):
            n = stats['simulations']
            if n == 0:
                return None
            return {
                'player_id': stats['player_id'],
                'avg_overs': round(stats['total_balls'] / n / 6, 1),
                'avg_runs': round(stats['total_runs'] / n, 1),
                'avg_wickets': round(stats['total_wickets'] / n, 2),
                'avg_dots': round(stats['total_dots'] / n, 1),
                'economy': round(stats['total_runs'] * 6 / stats['total_balls'], 2) if stats['total_balls'] > 0 else 0
            }
        
        return {
            'team1_batting_expected': [avg_batting(s) for s in t1_bat_stats if s['simulations'] > 0],
            'team2_batting_expected': [avg_batting(s) for s in t2_bat_stats if s['simulations'] > 0],
            'team1_bowling_expected': [avg_bowling(s) for s in t1_bowl_stats if s['simulations'] > 0],
            'team2_bowling_expected': [avg_bowling(s) for s in t2_bowl_stats if s['simulations'] > 0],
            'n_simulations': n_simulations
        }


# Global simulator instances for worker processes (avoid re-initialization)
_worker_simulators = {}

def _run_simulation_chunk(args):
    """
    Worker function for parallel simulation.
    
    Creates its own simulator instance (TensorFlow models can't be shared across processes).
    """
    (n_matches, team1_batters, team1_bowlers, team2_batters, team2_bowlers, 
     venue_id, use_toss, toss_field_prob, gender, worker_id) = args
    
    global _worker_simulators
    
    # Create simulator for this worker if not exists
    if gender not in _worker_simulators:
        _worker_simulators[gender] = VectorizedNNSimulator(gender=gender)
    
    simulator = _worker_simulators[gender]
    
    results = simulator.simulate_matches(
        n_matches,
        team1_batters,
        team1_bowlers,
        team2_batters,
        team2_bowlers,
        venue_id=venue_id,
        use_toss=use_toss,
        toss_field_prob=toss_field_prob
    )
    
    return {
        'team1_scores': results['team1_scores'].tolist(),
        'team2_scores': results['team2_scores'].tolist(),
        'toss_stats': results.get('toss_stats'),
        'worker_id': worker_id
    }


def run_parallel_simulations(
    n_simulations: int,
    team1_batters: List[int],
    team1_bowlers: List[int],
    team2_batters: List[int],
    team2_bowlers: List[int],
    venue_id: Optional[int] = None,
    use_toss: bool = False,
    toss_field_prob: float = 0.65,
    gender: str = 'male',
    n_workers: Optional[int] = None,
    progress_callback = None
) -> Dict:
    """
    Run simulations in parallel across multiple CPU cores.
    
    Uses ProcessPoolExecutor to distribute work across cores,
    bypassing Python's GIL for true parallelism.
    
    Args:
        n_simulations: Total number of simulations
        n_workers: Number of parallel workers (default: CPU count - 2)
        progress_callback: Optional callback(completed, total) for progress updates
        
    Returns:
        Combined results from all workers
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    if n_workers is None:
        n_workers = max(2, multiprocessing.cpu_count() - 2)
    
    # Split work across workers
    chunk_size = n_simulations // n_workers
    remainder = n_simulations % n_workers
    
    # Create argument tuples for each worker
    worker_args = []
    for i in range(n_workers):
        n_chunk = chunk_size + (1 if i < remainder else 0)
        if n_chunk > 0:
            worker_args.append((
                n_chunk,
                team1_batters,
                team1_bowlers,
                team2_batters,
                team2_bowlers,
                venue_id,
                use_toss,
                toss_field_prob,
                gender,
                i  # worker_id
            ))
    
    # Run in parallel
    all_team1_scores = []
    all_team2_scores = []
    toss_stats_accum = {'team1_won_toss': 0, 'chose_field': 0, 'team1_batted_first': 0, 'total': 0}
    completed = 0
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_run_simulation_chunk, args) for args in worker_args]
        
        for future in as_completed(futures):
            result = future.result()
            all_team1_scores.extend(result['team1_scores'])
            all_team2_scores.extend(result['team2_scores'])
            
            # Accumulate toss stats
            if use_toss and result.get('toss_stats'):
                ts = result['toss_stats']
                n = len(result['team1_scores'])
                toss_stats_accum['team1_won_toss'] += ts['team1_won_toss_pct'] * n
                toss_stats_accum['chose_field'] += ts['chose_field_pct'] * n
                toss_stats_accum['team1_batted_first'] += ts['team1_batted_first_pct'] * n
                toss_stats_accum['total'] += n
            
            completed += len(result['team1_scores'])
            if progress_callback:
                progress_callback(completed, n_simulations)
    
    # Convert to numpy arrays
    team1_scores = np.array(all_team1_scores)
    team2_scores = np.array(all_team2_scores)
    
    # Build final results
    results = {
        'team1_scores': team1_scores,
        'team2_scores': team2_scores,
        'team1_win_pct': float((team1_scores > team2_scores).mean() * 100),
        'team2_win_pct': float((team2_scores > team1_scores).mean() * 100),
        'n_workers': n_workers
    }
    
    if use_toss and toss_stats_accum['total'] > 0:
        t = toss_stats_accum['total']
        results['toss_stats'] = {
            'team1_won_toss_pct': toss_stats_accum['team1_won_toss'] / t,
            'chose_field_pct': toss_stats_accum['chose_field'] / t,
            'team1_batted_first_pct': toss_stats_accum['team1_batted_first'] / t
        }
    
    return results


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

