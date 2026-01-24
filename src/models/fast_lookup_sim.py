"""
Fast Calibrated Distribution Lookup Simulator.

Uses pre-computed outcome distributions from historical data,
avoiding neural network inference entirely.

Target: 1000 matches in ~0.1 seconds (1000x+ faster than NN).

Now includes venue effects for distribution adjustment.
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
import pickle
import time

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.matchups import MatchupDatabase
from src.features.venue_stats import VenueStatsBuilder

logger = logging.getLogger(__name__)

# H2H blending weight (when H2H data is available)
H2H_BLEND_WEIGHT = 0.4  # 40% H2H, 60% base distribution

# Outcome classes
NUM_CLASSES = 7
OUTCOME_RUNS = np.array([0, 1, 2, 3, 4, 6, 0])  # Class -> runs (6=wicket)
OUTCOME_NAMES = ['Dot', 'Single', 'Two', 'Three', 'Four', 'Six', 'Wicket']

# Bias correction: Real T20 data shows chasing team wins ~52-55%
# We balance this by giving first innings a slight scoring boost
FIRST_INNINGS_SCORE_BONUS = 1.05  # 5% bonus to first innings scores


class FastLookupSimulator:
    """
    Ultra-fast match simulator using pre-computed calibrated distributions.
    
    Key features:
    1. No neural network - pure numpy operations
    2. Pre-computed distributions by (phase, wickets, pressure)
    3. Player "archetypes" to capture batting/bowling styles
    4. Fully vectorized for parallel match simulation
    """
    
    # Calibrated distributions from historical T20 data
    # Format: (phase, pressure_level) -> probability distribution
    # Phases: 0=powerplay, 1=middle, 2=death
    # Pressure: 0=low (<8 RRR), 1=medium (8-12 RRR), 2=high (>12 RRR)
    
    BASE_DISTRIBUTIONS = {
        # Powerplay (overs 1-6) - More boundaries, fewer wickets
        (0, 0): np.array([0.34, 0.32, 0.07, 0.01, 0.14, 0.07, 0.05]),  # Low pressure
        (0, 1): np.array([0.32, 0.30, 0.06, 0.01, 0.16, 0.08, 0.07]),  # Medium pressure
        (0, 2): np.array([0.30, 0.28, 0.05, 0.01, 0.18, 0.10, 0.08]),  # High pressure
        
        # Middle overs (7-15) - More dots, controlled batting
        (1, 0): np.array([0.38, 0.32, 0.08, 0.01, 0.11, 0.06, 0.04]),
        (1, 1): np.array([0.35, 0.30, 0.07, 0.01, 0.13, 0.08, 0.06]),
        (1, 2): np.array([0.32, 0.28, 0.06, 0.01, 0.15, 0.10, 0.08]),
        
        # Death overs (16-20) - High scoring, more wickets
        (2, 0): np.array([0.28, 0.28, 0.07, 0.01, 0.16, 0.13, 0.07]),
        (2, 1): np.array([0.25, 0.26, 0.06, 0.01, 0.18, 0.15, 0.09]),
        (2, 2): np.array([0.22, 0.24, 0.05, 0.01, 0.20, 0.18, 0.10]),
    }
    
    def __init__(
        self, 
        player_dist_path: str = None,
        venue_stats_path: str = None,
        use_h2h: bool = True,
        format_type: str = 'T20',
        gender: str = 'male'
    ):
        """
        Initialize with player distributions, venue stats, and H2H matchups.
        
        Args:
            player_dist_path: Path to player distributions (auto-generated if None)
            venue_stats_path: Path to venue stats (auto-generated if None)
            use_h2h: Whether to use head-to-head matchup data
            format_type: Match format ('T20', 'ODI')
            gender: 'male' or 'female'
        """
        self.format_type = format_type
        self.gender = gender
        self.use_h2h = use_h2h
        
        # Auto-generate paths based on format and gender if not provided
        if player_dist_path is None:
            player_dist_path = f'data/processed/player_distributions_{format_type.lower()}_{gender}.pkl'
        if venue_stats_path is None:
            venue_stats_path = f'data/processed/venue_stats_{format_type.lower()}_{gender}.pkl'
        
        # Load H2H matchup database
        if use_h2h:
            self.h2h_db = MatchupDatabase('T20', 'male')
            self.h2h_db.load_from_database()
        else:
            self.h2h_db = None
        
        # Load player distributions for player-specific adjustments
        try:
            with open(player_dist_path, 'rb') as f:
                data = pickle.load(f)
            
            self.batter_dists = {}
            for pid, d in data['batter_distributions'].items():
                self.batter_dists[int(pid)] = np.array(d['prob_vector'][:7], dtype=np.float32)
            
            self.bowler_dists = {}
            for pid, d in data['bowler_distributions'].items():
                self.bowler_dists[int(pid)] = np.array(d['prob_vector'][:7], dtype=np.float32)
            
            logger.info(f"Loaded {len(self.batter_dists)} batter, {len(self.bowler_dists)} bowler distributions")
        except Exception as e:
            logger.warning(f"Could not load player distributions: {e}")
            self.batter_dists = {}
            self.bowler_dists = {}
        
        # Load venue statistics
        try:
            self.venue_stats = VenueStatsBuilder.load(venue_stats_path)
            logger.info(f"Loaded venue stats for {len(self.venue_stats.venue_stats)} venues")
        except Exception as e:
            logger.warning(f"Could not load venue stats: {e}")
            self.venue_stats = None
        
        # Default distributions
        self.default_bat_dist = np.array([0.35, 0.32, 0.08, 0.02, 0.12, 0.06, 0.05], dtype=np.float32)
        self.default_bowl_dist = np.array([0.38, 0.32, 0.07, 0.01, 0.10, 0.05, 0.07], dtype=np.float32)
        
        # Pre-compute cumulative distributions for fast sampling
        self.cum_dists = {k: np.cumsum(v) for k, v in self.BASE_DISTRIBUTIONS.items()}
        
        # Track H2H usage stats
        self.h2h_hits = 0
        self.h2h_misses = 0
    
    def get_batter_dist(self, player_id) -> np.ndarray:
        """Get batter distribution, converting ID to int if needed."""
        try:
            pid = int(player_id)
        except (ValueError, TypeError):
            return self.default_bat_dist
        return self.batter_dists.get(pid, self.default_bat_dist)
    
    def get_bowler_dist(self, player_id) -> np.ndarray:
        """Get bowler distribution, converting ID to int if needed."""
        try:
            pid = int(player_id)
        except (ValueError, TypeError):
            return self.default_bowl_dist
        return self.bowler_dists.get(pid, self.default_bowl_dist)
    
    def get_h2h_dist(self, batter_id: int, bowler_id: int) -> Optional[np.ndarray]:
        """Get H2H distribution if available (≥25 balls faced)."""
        if not self.use_h2h or self.h2h_db is None:
            return None
        return self.h2h_db.get_h2h_distribution_array(batter_id, bowler_id)
    
    def adjust_distribution_for_venue(
        self, 
        base_dist: np.ndarray, 
        venue_id: Optional[int]
    ) -> np.ndarray:
        """
        Adjust a ball outcome distribution based on venue characteristics.
        
        High-scoring venues: more boundaries, fewer dots
        Low-scoring venues: more dots, fewer boundaries
        """
        if self.venue_stats is None or venue_id is None:
            return base_dist
        return self.venue_stats.adjust_distribution_for_venue(base_dist, venue_id)
    
    def build_h2h_matrix(
        self,
        batter_ids: List[int],
        bowler_ids: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pre-compute H2H distributions for all batter-bowler pairs.
        
        Returns:
            h2h_dists: (n_batters, n_bowlers, 7) array of H2H distributions
            h2h_available: (n_batters, n_bowlers) boolean array
        """
        n_batters = len(batter_ids)
        n_bowlers = len(bowler_ids)
        
        h2h_dists = np.zeros((n_batters, n_bowlers, 7), dtype=np.float32)
        h2h_available = np.zeros((n_batters, n_bowlers), dtype=bool)
        
        if not self.use_h2h:
            return h2h_dists, h2h_available
        
        for i, bat_id in enumerate(batter_ids):
            for j, bowl_id in enumerate(bowler_ids):
                h2h = self.get_h2h_dist(bat_id, bowl_id)
                if h2h is not None:
                    h2h_dists[i, j] = h2h
                    h2h_available[i, j] = True
        
        return h2h_dists, h2h_available
    
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
        Ultra-fast: No neural network, pure numpy.
        Now with H2H matchup blending, venue effects, and optional toss simulation!
        
        Args:
            use_toss: If True, simulate toss for each match (50/50 winner, then choose bat/field)
            toss_field_prob: Probability winner chooses to field (default 0.65 for T20)
        """
        max_balls = max_overs * 6
        
        # Reset H2H stats
        self.h2h_hits = 0
        self.h2h_misses = 0
        
        # Pre-compute player distribution matrices
        # Apply venue adjustment to distributions
        team1_bat_dists = np.array([
            self.adjust_distribution_for_venue(self.get_batter_dist(pid), venue_id)
            for pid in team1_batter_ids
        ])
        team1_bowl_dists = np.array([
            self.adjust_distribution_for_venue(self.get_bowler_dist(pid), venue_id)
            for pid in team1_bowler_ids
        ])
        team2_bat_dists = np.array([
            self.adjust_distribution_for_venue(self.get_batter_dist(pid), venue_id)
            for pid in team2_batter_ids
        ])
        team2_bowl_dists = np.array([
            self.adjust_distribution_for_venue(self.get_bowler_dist(pid), venue_id)
            for pid in team2_bowler_ids
        ])
        
        # Pre-compute H2H matrices (also apply venue adjustments)
        # Team1 batters vs Team2 bowlers
        h2h_1v2, h2h_1v2_avail = self.build_h2h_matrix(team1_batter_ids, team2_bowler_ids)
        if venue_id is not None:
            for i in range(h2h_1v2.shape[0]):
                for j in range(h2h_1v2.shape[1]):
                    if h2h_1v2_avail[i, j]:
                        h2h_1v2[i, j] = self.adjust_distribution_for_venue(h2h_1v2[i, j], venue_id)
        
        # Team2 batters vs Team1 bowlers
        h2h_2v1, h2h_2v1_avail = self.build_h2h_matrix(team2_batter_ids, team1_bowler_ids)
        if venue_id is not None:
            for i in range(h2h_2v1.shape[0]):
                for j in range(h2h_2v1.shape[1]):
                    if h2h_2v1_avail[i, j]:
                        h2h_2v1[i, j] = self.adjust_distribution_for_venue(h2h_2v1[i, j], venue_id)
        
        if use_toss:
            # ========== TOSS SIMULATION (per-match) ==========
            # Step 1: 50/50 who wins toss
            team1_wins_toss = np.random.random(n_matches) < 0.5
            
            # Step 2: Winner chooses bat (1-toss_field_prob) or field (toss_field_prob)
            winner_chooses_field = np.random.random(n_matches) < toss_field_prob
            
            # Step 3: Determine who bats first
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
                    h2h_dists=h2h_1v2,
                    h2h_available=h2h_1v2_avail,
                    target=None,
                    max_balls=max_balls
                )
                adjusted_first_runs_t1 = (first_runs_t1 * FIRST_INNINGS_SCORE_BONUS).astype(np.int32)
                targets_t1 = adjusted_first_runs_t1 + 1
                
                second_runs_t1, _ = self._simulate_innings_vectorized(
                    n_matches=n_team1_first,
                    batting_dists=team2_bat_dists,
                    bowling_dists=team1_bowl_dists,
                    h2h_dists=h2h_2v1,
                    h2h_available=h2h_2v1_avail,
                    target=targets_t1,
                    max_balls=max_balls
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
                    h2h_dists=h2h_2v1,
                    h2h_available=h2h_2v1_avail,
                    target=None,
                    max_balls=max_balls
                )
                adjusted_first_runs_t2 = (first_runs_t2 * FIRST_INNINGS_SCORE_BONUS).astype(np.int32)
                targets_t2 = adjusted_first_runs_t2 + 1
                
                second_runs_t2, _ = self._simulate_innings_vectorized(
                    n_matches=n_team2_first,
                    batting_dists=team1_bat_dists,
                    bowling_dists=team2_bowl_dists,
                    h2h_dists=h2h_1v2,
                    h2h_available=h2h_1v2_avail,
                    target=targets_t2,
                    max_balls=max_balls
                )
                
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
                h2h_dists=h2h_1v2,
                h2h_available=h2h_1v2_avail,
                target=None,
                max_balls=max_balls
            )
            
            adjusted_first_runs = (first_runs * FIRST_INNINGS_SCORE_BONUS).astype(np.int32)
            targets = adjusted_first_runs + 1
            
            second_runs, second_wickets = self._simulate_innings_vectorized(
                n_matches=n_matches,
                batting_dists=team2_bat_dists,
                bowling_dists=team1_bowl_dists,
                h2h_dists=h2h_2v1,
                h2h_available=h2h_2v1_avail,
                target=targets,
                max_balls=max_balls
            )
            
            team1_scores = first_runs
            team2_scores = second_runs
            team1_wins = second_runs < targets
            toss_stats = None
        
        # H2H usage stats
        total_h2h = self.h2h_hits + self.h2h_misses
        h2h_rate = self.h2h_hits / total_h2h if total_h2h > 0 else 0
        
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
            'h2h_hits': self.h2h_hits,
            'h2h_misses': self.h2h_misses,
            'h2h_rate': h2h_rate,
        }
        
        if toss_stats:
            results['toss_stats'] = toss_stats
        
        return results
    
    def _simulate_innings_vectorized(
        self,
        n_matches: int,
        batting_dists: np.ndarray,
        bowling_dists: np.ndarray,
        h2h_dists: np.ndarray,
        h2h_available: np.ndarray,
        target: Optional[np.ndarray],
        max_balls: int = 120
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate innings for all N matches simultaneously with H2H blending."""
        
        # State arrays
        runs = np.zeros(n_matches, dtype=np.int32)
        wickets = np.zeros(n_matches, dtype=np.int32)
        balls = np.zeros(n_matches, dtype=np.int32)
        current_batter = np.zeros(n_matches, dtype=np.int32)
        active = np.ones(n_matches, dtype=bool)
        
        for ball_idx in range(max_balls):
            if not active.any():
                break
            
            n_active = active.sum()
            
            # Determine phase
            over = ball_idx // 6
            if over < 6:
                phase = 0  # Powerplay
            elif over < 15:
                phase = 1  # Middle
            else:
                phase = 2  # Death
            
            # Calculate pressure level
            if target is not None:
                balls_remaining = max_balls - balls[active]
                runs_needed = target[active] - runs[active]
                required_rate = np.where(
                    balls_remaining > 0,
                    runs_needed * 6 / balls_remaining,
                    0
                )
                pressure = np.where(required_rate < 8, 0, np.where(required_rate < 12, 1, 2))
            else:
                pressure = np.zeros(n_active, dtype=np.int32)
            
            # Get base distributions for each active match
            base_dist = self.BASE_DISTRIBUTIONS[(phase, 0)]
            
            # Get batter distributions for current batters
            safe_batter_idx = np.clip(current_batter[active], 0, len(batting_dists) - 1)
            batter_dist = batting_dists[safe_batter_idx]
            
            # Bowler rotation
            bowler_idx = over % len(bowling_dists)
            bowler_dist = bowling_dists[bowler_idx]
            
            # Blend: base (40%) + batter (35%) + bowler (25%)
            combined_dist = 0.40 * base_dist + 0.35 * batter_dist + 0.25 * bowler_dist
            
            # Apply H2H blending where available
            if self.use_h2h and h2h_available.any():
                active_indices = np.where(active)[0]
                for i, match_idx in enumerate(active_indices):
                    bat_idx = int(current_batter[match_idx])
                    if bat_idx < h2h_available.shape[0] and bowler_idx < h2h_available.shape[1]:
                        if h2h_available[bat_idx, bowler_idx]:
                            # Blend with H2H data
                            h2h_dist = h2h_dists[bat_idx, bowler_idx]
                            combined_dist[i] = (1 - H2H_BLEND_WEIGHT) * combined_dist[i] + H2H_BLEND_WEIGHT * h2h_dist
                            self.h2h_hits += 1
                        else:
                            self.h2h_misses += 1
                    else:
                        self.h2h_misses += 1
            
            # Apply pressure adjustment
            for i, p in enumerate(pressure):
                if p > 0:
                    pressure_dist = self.BASE_DISTRIBUTIONS[(phase, int(p))]
                    combined_dist[i] = 0.6 * combined_dist[i] + 0.4 * pressure_dist
            
            # Normalize
            combined_dist = combined_dist / combined_dist.sum(axis=-1, keepdims=True)
            
            # Sample outcomes
            outcomes = self._vectorized_sample(combined_dist)
            
            # Update states
            runs_scored = OUTCOME_RUNS[outcomes]
            is_wicket = (outcomes == 6)
            
            # Apply updates only to active matches
            active_indices = np.where(active)[0]
            runs[active_indices] += runs_scored
            wickets[active_indices] += is_wicket.astype(np.int32)
            balls[active_indices] += 1
            
            # New batter on wicket
            wicket_matches = active_indices[is_wicket]
            for idx in wicket_matches:
                if wickets[idx] < 10:
                    current_batter[idx] = min(wickets[idx] + 1, 10)
            
            # Check termination
            all_out = wickets >= 10
            if target is not None:
                target_reached = runs >= target
            else:
                target_reached = np.zeros(n_matches, dtype=bool)
            active = active & ~all_out & ~target_reached
        
        return runs, wickets
    
    def _vectorized_sample(self, proba: np.ndarray) -> np.ndarray:
        """Vectorized sampling from probability distributions."""
        if proba.ndim == 1:
            proba = proba.reshape(1, -1)
        
        cumprob = np.cumsum(proba, axis=1)
        r = np.random.random(len(proba))[:, np.newaxis]
        outcomes = (cumprob < r).sum(axis=1)
        return np.clip(outcomes, 0, NUM_CLASSES - 1)


def benchmark_lookup_simulator():
    """Benchmark the fast lookup simulator."""
    print("=" * 70)
    print("FAST LOOKUP SIMULATOR BENCHMARK")
    print("=" * 70)
    
    simulator = FastLookupSimulator()
    
    # Create dummy team IDs
    team1_batters = list(range(1, 12))
    team1_bowlers = list(range(100, 105))
    team2_batters = list(range(200, 211))
    team2_bowlers = list(range(300, 305))
    
    # Warmup
    print("\nWarming up...")
    _ = simulator.simulate_matches(100, team1_batters, team1_bowlers, team2_batters, team2_bowlers)
    
    # Benchmark different batch sizes
    for n in [100, 1000, 10000]:
        print(f"\nBenchmarking {n} matches...")
        start = time.time()
        results = simulator.simulate_matches(n, team1_batters, team1_bowlers, team2_batters, team2_bowlers)
        elapsed = time.time() - start
        
        print(f"  Time: {elapsed*1000:.1f}ms ({elapsed/n*1000:.3f}ms per match)")
        print(f"  Team1 Win: {results['team1_win_prob']:.1%}")
        print(f"  Avg Scores: {results['avg_team1_score']:.1f} vs {results['avg_team2_score']:.1f}")
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    benchmark_lookup_simulator()

