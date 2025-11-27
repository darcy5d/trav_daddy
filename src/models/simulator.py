"""
Monte Carlo Ball-by-Ball Match Simulation Engine.

Simulates cricket matches delivery by delivery using:
- HYBRID outcome probabilities: H2H data (≥25 balls) or ELO-based fallback
- Smart captain bowling selection with 4-over max enforcement
- Phase-specific adjustments (powerplay, middle, death)
- Match situation context (wickets, required rate)
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random
import numpy as np
from collections import defaultdict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import SIMULATION_CONFIG
from src.models.matchups import get_matchup_db, MatchupDatabase, MatchupStats
from src.models.bowling_captain import (
    SmartCaptain, BowlingTracker, Bowler, MatchState,
    InningsPhase, get_innings_phase
)

logger = logging.getLogger(__name__)


@dataclass
class Player:
    """Represents a player with ELO ratings."""
    player_id: int
    name: str
    batting_elo: float = 1500.0
    bowling_elo: float = 1500.0


@dataclass
class InningsState:
    """Current state of an innings."""
    score: int = 0
    wickets: int = 0
    balls: int = 0
    target: Optional[int] = None
    
    @property
    def overs(self) -> float:
        """Current over in decimal format."""
        return self.balls // 6 + (self.balls % 6) / 10
    
    @property
    def overs_completed(self) -> int:
        """Number of completed overs."""
        return self.balls // 6
    
    @property
    def run_rate(self) -> float:
        """Current run rate."""
        overs = self.balls / 6
        return self.score / overs if overs > 0 else 0
    
    @property
    def required_rate(self) -> Optional[float]:
        """Required run rate if chasing."""
        if self.target is None:
            return None
        runs_needed = self.target - self.score
        balls_remaining = 120 - self.balls  # T20
        if balls_remaining <= 0:
            return float('inf')
        return runs_needed * 6 / balls_remaining


class OutcomeDistribution:
    """
    Stores and samples from outcome probability distributions.
    
    Supports:
    - Head-to-head historical data (when ≥25 balls faced)
    - ELO-based fallback distributions (keyed by phase and elo_bucket)
    """
    
    # Default distributions based on T20 cricket averages
    DEFAULT_DISTRIBUTIONS = {
        'powerplay': {
            'very_low':  {'0': 0.38, '1': 0.30, '2': 0.06, '3': 0.01, '4': 0.12, '6': 0.06, 'W': 0.07},
            'low':       {'0': 0.36, '1': 0.31, '2': 0.06, '3': 0.01, '4': 0.13, '6': 0.07, 'W': 0.06},
            'even':      {'0': 0.34, '1': 0.32, '2': 0.07, '3': 0.01, '4': 0.14, '6': 0.07, 'W': 0.05},
            'high':      {'0': 0.32, '1': 0.32, '2': 0.07, '3': 0.01, '4': 0.15, '6': 0.08, 'W': 0.05},
            'very_high': {'0': 0.30, '1': 0.32, '2': 0.07, '3': 0.01, '4': 0.16, '6': 0.10, 'W': 0.04},
        },
        'middle': {
            'very_low':  {'0': 0.42, '1': 0.30, '2': 0.07, '3': 0.01, '4': 0.09, '6': 0.05, 'W': 0.06},
            'low':       {'0': 0.40, '1': 0.31, '2': 0.07, '3': 0.01, '4': 0.10, '6': 0.06, 'W': 0.05},
            'even':      {'0': 0.38, '1': 0.32, '2': 0.08, '3': 0.01, '4': 0.11, '6': 0.06, 'W': 0.04},
            'high':      {'0': 0.36, '1': 0.32, '2': 0.08, '3': 0.01, '4': 0.12, '6': 0.07, 'W': 0.04},
            'very_high': {'0': 0.34, '1': 0.32, '2': 0.08, '3': 0.01, '4': 0.13, '6': 0.09, 'W': 0.03},
        },
        'death': {
            'very_low':  {'0': 0.32, '1': 0.26, '2': 0.06, '3': 0.01, '4': 0.14, '6': 0.12, 'W': 0.09},
            'low':       {'0': 0.30, '1': 0.27, '2': 0.06, '3': 0.01, '4': 0.15, '6': 0.13, 'W': 0.08},
            'even':      {'0': 0.28, '1': 0.28, '2': 0.07, '3': 0.01, '4': 0.16, '6': 0.13, 'W': 0.07},
            'high':      {'0': 0.26, '1': 0.28, '2': 0.07, '3': 0.01, '4': 0.17, '6': 0.15, 'W': 0.06},
            'very_high': {'0': 0.24, '1': 0.28, '2': 0.07, '3': 0.01, '4': 0.18, '6': 0.17, 'W': 0.05},
        }
    }
    
    def __init__(
        self,
        distributions: Optional[Dict] = None,
        matchup_db: Optional[MatchupDatabase] = None
    ):
        """Initialize with custom or default distributions."""
        self.distributions = distributions or self.DEFAULT_DISTRIBUTIONS
        self.matchup_db = matchup_db
        
        # Track H2H usage for analysis
        self.h2h_hits = 0
        self.h2h_misses = 0
    
    def get_elo_bucket(self, elo_diff: float) -> str:
        """Convert ELO difference to bucket."""
        if elo_diff < -150:
            return 'very_low'
        elif elo_diff < -50:
            return 'low'
        elif elo_diff < 50:
            return 'even'
        elif elo_diff < 150:
            return 'high'
        else:
            return 'very_high'
    
    def get_phase(self, over_number: int) -> str:
        """Get innings phase from over number."""
        if over_number < 6:
            return 'powerplay'
        elif over_number < 15:
            return 'middle'
        else:
            return 'death'
    
    def get_h2h_distribution(
        self,
        batter_id: int,
        bowler_id: int
    ) -> Optional[Dict[str, float]]:
        """
        Get H2H outcome distribution if sufficient data exists.
        
        Returns None if:
        - No matchup database configured
        - Less than 25 balls faced between batter and bowler
        """
        if self.matchup_db is None:
            return None
        
        return self.matchup_db.get_h2h_distribution(batter_id, bowler_id)
    
    def get_elo_distribution(
        self,
        batter_elo: float,
        bowler_elo: float,
        over_number: int
    ) -> Dict[str, float]:
        """Get ELO-based outcome distribution (fallback)."""
        elo_diff = batter_elo - bowler_elo
        bucket = self.get_elo_bucket(elo_diff)
        phase = self.get_phase(over_number)
        
        return self.distributions[phase][bucket].copy()
    
    def sample_outcome(
        self,
        batter: Player,
        bowler: 'Bowler',
        over_number: int,
        pressure_adjustment: float = 0.0,
        use_h2h: bool = True
    ) -> str:
        """
        Sample an outcome from the HYBRID distribution.
        
        Strategy:
        1. If H2H data exists (≥25 balls), use historical distribution
        2. Otherwise, fall back to ELO-based distribution
        
        Args:
            batter: Batter player object
            bowler: Bowler object (from bowling_captain)
            over_number: Current over (0-19 for T20)
            pressure_adjustment: Adjustment factor for pressure situations
            use_h2h: Whether to attempt H2H lookup (can disable for testing)
        
        Returns:
            Outcome string: '0', '1', '2', '3', '4', '6', or 'W'
        """
        probs = None
        
        # TRY H2H FIRST
        if use_h2h:
            h2h_dist = self.get_h2h_distribution(batter.player_id, bowler.player_id)
            if h2h_dist is not None:
                probs = h2h_dist.copy()
                self.h2h_hits += 1
        
        # FALLBACK TO ELO
        if probs is None:
            probs = self.get_elo_distribution(
                batter.batting_elo,
                bowler.bowling_elo,
                over_number
            )
            self.h2h_misses += 1
        
        # Apply pressure adjustment (increases wicket probability under pressure)
        if pressure_adjustment > 0:
            wicket_boost = pressure_adjustment * 0.02  # Up to 2% more wickets
            probs['W'] = min(0.15, probs['W'] + wicket_boost)
            
            # Reduce other probabilities proportionally
            total_other = 1 - probs['W']
            other_keys = [k for k in probs if k != 'W']
            original_other = sum(probs[k] for k in other_keys)
            for k in other_keys:
                probs[k] = probs[k] * total_other / original_other
        
        # Sample
        outcomes = list(probs.keys())
        probabilities = list(probs.values())
        
        return np.random.choice(outcomes, p=probabilities)
    
    def get_h2h_stats(self) -> Dict[str, int]:
        """Get statistics on H2H vs ELO usage."""
        total = self.h2h_hits + self.h2h_misses
        return {
            'h2h_hits': self.h2h_hits,
            'h2h_misses': self.h2h_misses,
            'h2h_rate': self.h2h_hits / total if total > 0 else 0
        }


def player_to_bowler(player: Player) -> Bowler:
    """Convert Player to Bowler for captain selection."""
    return Bowler(
        player_id=player.player_id,
        name=player.name,
        bowling_elo=player.bowling_elo
    )


class MatchSimulator:
    """
    Monte Carlo match simulation engine with H2H and T20 bowling rules.
    
    Features:
    - Hybrid outcome distributions (H2H when ≥25 balls, else ELO)
    - Smart captain bowler selection
    - T20 4-over max per bowler enforcement
    - Phase-aware bowling (powerplay, middle, death)
    """
    
    def __init__(
        self,
        outcome_distribution: Optional[OutcomeDistribution] = None,
        num_simulations: int = SIMULATION_CONFIG['num_simulations'],
        format_type: str = 'T20',
        gender: str = 'male',
        use_smart_captain: bool = True,
        use_h2h: bool = True
    ):
        # Load matchup database if H2H is enabled
        matchup_db = None
        if use_h2h:
            matchup_db = get_matchup_db(format_type, gender)
            matchup_db.load_from_database()
        
        self.outcome_dist = outcome_distribution or OutcomeDistribution(matchup_db=matchup_db)
        self.num_simulations = num_simulations
        self.format_type = format_type
        self.gender = gender
        self.use_smart_captain = use_smart_captain
        self.use_h2h = use_h2h
    
    def simulate_delivery(
        self,
        batter: Player,
        bowler: Bowler,
        state: InningsState,
        over_number: int
    ) -> Tuple[int, bool]:
        """
        Simulate a single delivery.
        
        Uses HYBRID distribution:
        - H2H stats if batter has faced this bowler ≥25 times
        - ELO-based distribution otherwise
        
        Returns:
            Tuple of (runs_scored, is_wicket)
        """
        # Calculate pressure
        pressure = 0.0
        if state.target is not None:
            req_rate = state.required_rate
            if req_rate and req_rate > 10:
                pressure = min(1.0, (req_rate - 10) / 6)
        
        # Sample outcome (HYBRID: H2H or ELO)
        outcome = self.outcome_dist.sample_outcome(
            batter,
            bowler,
            over_number,
            pressure,
            use_h2h=self.use_h2h
        )
        
        if outcome == 'W':
            return 0, True
        else:
            return int(outcome), False
    
    def simulate_innings(
        self,
        batting_xi: List[Player],
        bowling_attack: List[Player],
        target: Optional[int] = None,
        max_overs: int = 20
    ) -> InningsState:
        """
        Simulate a complete innings with T20 bowling rules.
        
        Features:
        - Smart captain bowler selection
        - 4-over max per bowler enforcement
        - H2H matchup-aware bowling choices
        
        Args:
            batting_xi: List of 11 batters in order
            bowling_attack: List of available bowlers (need at least 5)
            target: Target score if chasing (None for first innings)
            max_overs: Maximum overs in innings (20 for T20)
        
        Returns:
            Final innings state
        """
        state = InningsState(target=target)
        
        current_batter_idx = 0
        non_striker_idx = 1
        max_balls = max_overs * 6
        
        # Convert players to bowlers and create smart captain
        bowlers = [player_to_bowler(p) for p in bowling_attack[:5]]
        
        if self.use_smart_captain:
            captain = SmartCaptain(
                bowling_attack=bowlers,
                format_type=self.format_type,
                gender=self.gender
            )
        else:
            captain = None
        
        for over in range(max_overs):
            # Select bowler
            if self.use_smart_captain and captain:
                # Get current batters at crease
                current_batter_ids = [
                    batting_xi[current_batter_idx].player_id,
                    batting_xi[non_striker_idx].player_id
                ]
                
                match_state = MatchState(
                    total_runs=state.score,
                    wickets=state.wickets,
                    overs_completed=over,
                    balls_in_over=0,
                    target=target,
                    run_rate=state.run_rate,
                    required_rate=state.required_rate
                )
                
                # Smart bowler selection
                bowler = captain.select_bowler(
                    current_batter_ids,
                    match_state,
                    over + 1  # 1-indexed
                )
            else:
                # Simple rotation fallback
                bowler = bowlers[over % len(bowlers)]
            
            # Bowl the over
            for ball in range(6):
                if state.wickets >= 10:
                    # Record partial over
                    if self.use_smart_captain and captain:
                        # Only record if bowled at least some balls
                        if ball > 0:
                            captain.tracker.balls_in_current_over[bowler.player_id] = ball
                    return state
                
                if target is not None and state.score >= target:
                    if self.use_smart_captain and captain:
                        if ball > 0:
                            captain.tracker.balls_in_current_over[bowler.player_id] = ball
                    return state
                
                # Get current batter
                batter = batting_xi[current_batter_idx]
                
                # Simulate delivery (HYBRID distribution)
                runs, is_wicket = self.simulate_delivery(
                    batter, bowler, state, over
                )
                
                state.score += runs
                state.balls += 1
                
                if is_wicket:
                    state.wickets += 1
                    if state.wickets < 10:
                        # Next batter comes in
                        current_batter_idx = min(10, state.wickets + 1)
                
                # Rotate strike for odd runs
                if runs % 2 == 1:
                    current_batter_idx, non_striker_idx = non_striker_idx, current_batter_idx
            
            # End of over: record and rotate strike
            if self.use_smart_captain and captain:
                captain.record_over_complete(bowler.player_id)
            
            current_batter_idx, non_striker_idx = non_striker_idx, current_batter_idx
        
        return state
    
    def simulate_match(
        self,
        team1_batting: List[Player],
        team1_bowling: List[Player],
        team2_batting: List[Player],
        team2_bowling: List[Player],
        team1_bats_first: bool = True
    ) -> Dict:
        """
        Simulate a single match.
        
        Returns:
            Dict with match result details
        """
        if team1_bats_first:
            first_batting = team1_batting
            first_bowling = team2_bowling
            second_batting = team2_batting
            second_bowling = team1_bowling
        else:
            first_batting = team2_batting
            first_bowling = team1_bowling
            second_batting = team1_batting
            second_bowling = team2_bowling
        
        # First innings
        first_innings = self.simulate_innings(first_batting, first_bowling)
        
        # Second innings (chasing)
        target = first_innings.score + 1
        second_innings = self.simulate_innings(second_batting, second_bowling, target=target)
        
        # Determine winner
        if second_innings.score >= target:
            # Chasing team won
            winner = 2 if team1_bats_first else 1
            margin = 10 - second_innings.wickets
            margin_type = 'wickets'
        else:
            # Setting team won
            winner = 1 if team1_bats_first else 2
            margin = first_innings.score - second_innings.score
            margin_type = 'runs'
        
        return {
            'team1_score': first_innings.score if team1_bats_first else second_innings.score,
            'team2_score': second_innings.score if team1_bats_first else first_innings.score,
            'team1_wickets': first_innings.wickets if team1_bats_first else second_innings.wickets,
            'team2_wickets': second_innings.wickets if team1_bats_first else first_innings.wickets,
            'winner': winner,
            'margin': margin,
            'margin_type': margin_type
        }
    
    def run_simulations(
        self,
        team1_batting: List[Player],
        team1_bowling: List[Player],
        team2_batting: List[Player],
        team2_bowling: List[Player],
        team1_bats_first: bool = True,
        n_simulations: Optional[int] = None
    ) -> Dict:
        """
        Run Monte Carlo simulations.
        
        Returns:
            Dict with aggregated predictions:
            - team1_win_prob: P(team1 wins)
            - avg_team1_score, avg_team2_score
            - score_ranges: Confidence intervals
            - h2h_usage: Stats on H2H vs ELO distribution usage
        """
        n = n_simulations or self.num_simulations
        
        # Reset H2H stats
        self.outcome_dist.h2h_hits = 0
        self.outcome_dist.h2h_misses = 0
        
        results = []
        for i in range(n):
            if i % 1000 == 0 and i > 0:
                logger.debug(f"Simulation {i+1}/{n}")
            
            result = self.simulate_match(
                team1_batting, team1_bowling,
                team2_batting, team2_bowling,
                team1_bats_first
            )
            results.append(result)
        
        # Aggregate results
        team1_wins = sum(1 for r in results if r['winner'] == 1)
        team1_scores = [r['team1_score'] for r in results]
        team2_scores = [r['team2_score'] for r in results]
        
        return {
            'team1_win_prob': team1_wins / n,
            'team2_win_prob': 1 - team1_wins / n,
            'n_simulations': n,
            'avg_team1_score': np.mean(team1_scores),
            'avg_team2_score': np.mean(team2_scores),
            'team1_score_std': np.std(team1_scores),
            'team2_score_std': np.std(team2_scores),
            'team1_score_range': (np.percentile(team1_scores, 5), np.percentile(team1_scores, 95)),
            'team2_score_range': (np.percentile(team2_scores, 5), np.percentile(team2_scores, 95)),
            'h2h_usage': self.outcome_dist.get_h2h_stats()
        }


def create_dummy_players(
    batting_elo_avg: float,
    bowling_elo_avg: float,
    n_batters: int = 11,
    n_bowlers: int = 5,
    variation: float = 50
) -> Tuple[List[Player], List[Player]]:
    """Create dummy players with given average ELOs."""
    batters = [
        Player(
            player_id=i,
            name=f"Batter{i}",
            batting_elo=batting_elo_avg + random.uniform(-variation, variation)
        )
        for i in range(n_batters)
    ]
    
    bowlers = [
        Player(
            player_id=i + 100,
            name=f"Bowler{i}",
            bowling_elo=bowling_elo_avg + random.uniform(-variation, variation)
        )
        for i in range(n_bowlers)
    ]
    
    return batters, bowlers


def quick_simulation(
    team1_batting_elo: float,
    team1_bowling_elo: float,
    team2_batting_elo: float,
    team2_bowling_elo: float,
    n_simulations: int = 1000,
    use_h2h: bool = False,  # Default off for dummy players
    use_smart_captain: bool = True
) -> Dict:
    """
    Quick match simulation with average team ELOs.
    
    Useful for testing without full player data.
    Note: H2H is disabled by default since dummy players won't have matchup data.
    """
    team1_batters, team1_bowlers = create_dummy_players(
        team1_batting_elo, team1_bowling_elo
    )
    team2_batters, team2_bowlers = create_dummy_players(
        team2_batting_elo, team2_bowling_elo
    )
    
    simulator = MatchSimulator(
        num_simulations=n_simulations,
        use_h2h=use_h2h,
        use_smart_captain=use_smart_captain
    )
    
    return simulator.run_simulations(
        team1_batters, team1_bowlers,
        team2_batters, team2_bowlers
    )


def load_team_from_db(
    team_name: str,
    format_type: str = 'T20',
    gender: str = 'male',
    n_batters: int = 11,
    n_bowlers: int = 5
) -> Tuple[List[Player], List[Player]]:
    """
    Load real team players from database.
    
    Returns batting XI and bowling attack with their ELO ratings.
    """
    from src.data.database import get_connection
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get team ID
    cursor.execute("SELECT team_id FROM teams WHERE name = ?", (team_name,))
    row = cursor.fetchone()
    if not row:
        raise ValueError(f"Team '{team_name}' not found in database")
    team_id = row['team_id']
    
    # Get ELO column names
    batting_col = f'batting_elo_{format_type.lower()}_{gender}'
    bowling_col = f'bowling_elo_{format_type.lower()}_{gender}'
    
    # Get top batters by batting ELO
    cursor.execute(f"""
        SELECT DISTINCT p.player_id, p.name, 
               COALESCE(e.{batting_col}, 1500) as batting_elo,
               COALESCE(e.{bowling_col}, 1500) as bowling_elo
        FROM players p
        JOIN player_match_stats pms ON p.player_id = pms.player_id
        JOIN matches m ON pms.match_id = m.match_id
        LEFT JOIN player_current_elo e ON p.player_id = e.player_id
        WHERE m.match_type = ? AND m.gender = ?
        AND (m.team1_id = ? OR m.team2_id = ?)
        AND pms.runs_scored > 0
        GROUP BY p.player_id
        ORDER BY COALESCE(e.{batting_col}, 1500) DESC
        LIMIT ?
    """, (format_type, gender, team_id, team_id, n_batters))
    
    batters = [
        Player(
            player_id=row['player_id'],
            name=row['name'],
            batting_elo=row['batting_elo'],
            bowling_elo=row['bowling_elo']
        )
        for row in cursor.fetchall()
    ]
    
    # Get top bowlers by bowling ELO
    cursor.execute(f"""
        SELECT DISTINCT p.player_id, p.name,
               COALESCE(e.{batting_col}, 1500) as batting_elo,
               COALESCE(e.{bowling_col}, 1500) as bowling_elo
        FROM players p
        JOIN player_match_stats pms ON p.player_id = pms.player_id
        JOIN matches m ON pms.match_id = m.match_id
        LEFT JOIN player_current_elo e ON p.player_id = e.player_id
        WHERE m.match_type = ? AND m.gender = ?
        AND (m.team1_id = ? OR m.team2_id = ?)
        AND pms.overs_bowled > 0
        GROUP BY p.player_id
        ORDER BY COALESCE(e.{bowling_col}, 1500) DESC
        LIMIT ?
    """, (format_type, gender, team_id, team_id, n_bowlers))
    
    bowlers = [
        Player(
            player_id=row['player_id'],
            name=row['name'],
            batting_elo=row['batting_elo'],
            bowling_elo=row['bowling_elo']
        )
        for row in cursor.fetchall()
    ]
    
    conn.close()
    
    # Pad if needed
    while len(batters) < n_batters:
        batters.append(Player(-len(batters), f"Batter{len(batters)+1}", 1450, 1400))
    while len(bowlers) < n_bowlers:
        bowlers.append(Player(-len(bowlers)-100, f"Bowler{len(bowlers)+1}", 1400, 1450))
    
    return batters, bowlers


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    print("=" * 70)
    print("MONTE CARLO MATCH SIMULATION WITH H2H & SMART CAPTAIN")
    print("=" * 70)
    
    # Test 1: Quick simulation with dummy players (no H2H)
    print("\n" + "=" * 70)
    print("TEST 1: Dummy Players (Strong vs Weaker team)")
    print("=" * 70)
    
    result = quick_simulation(
        team1_batting_elo=1600,
        team1_bowling_elo=1580,
        team2_batting_elo=1500,
        team2_bowling_elo=1500,
        n_simulations=5000,
        use_h2h=False,  # Dummy players have no H2H data
        use_smart_captain=True
    )
    
    print(f"\nTeam 1 Win Probability: {result['team1_win_prob']:.1%}")
    print(f"Team 2 Win Probability: {result['team2_win_prob']:.1%}")
    print(f"\nAverage Scores:")
    print(f"  Team 1: {result['avg_team1_score']:.1f} (+/- {result['team1_score_std']:.1f})")
    print(f"  Team 2: {result['avg_team2_score']:.1f} (+/- {result['team2_score_std']:.1f})")
    print(f"\n90% Score Ranges:")
    print(f"  Team 1: {result['team1_score_range'][0]:.0f} - {result['team1_score_range'][1]:.0f}")
    print(f"  Team 2: {result['team2_score_range'][0]:.0f} - {result['team2_score_range'][1]:.0f}")
    
    # Test 2: Try loading real teams (with H2H)
    print("\n" + "=" * 70)
    print("TEST 2: Real Teams - India vs Australia (with H2H)")
    print("=" * 70)
    
    try:
        india_batting, india_bowling = load_team_from_db("India", "T20", "male")
        australia_batting, australia_bowling = load_team_from_db("Australia", "T20", "male")
        
        print(f"\nIndia Top 5 Batters:")
        for p in india_batting[:5]:
            print(f"  {p.name}: Bat ELO={p.batting_elo:.0f}")
        
        print(f"\nAustralia Top 5 Bowlers:")
        for p in australia_bowling[:5]:
            print(f"  {p.name}: Bowl ELO={p.bowling_elo:.0f}")
        
        # Run simulation with H2H enabled
        simulator = MatchSimulator(
            num_simulations=5000,
            use_h2h=True,
            use_smart_captain=True
        )
        
        result2 = simulator.run_simulations(
            india_batting, india_bowling,
            australia_batting, australia_bowling
        )
        
        print(f"\n{'='*70}")
        print("INDIA vs AUSTRALIA SIMULATION RESULTS")
        print("=" * 70)
        print(f"\nIndia Win Probability:     {result2['team1_win_prob']:.1%}")
        print(f"Australia Win Probability: {result2['team2_win_prob']:.1%}")
        print(f"\nAverage Scores:")
        print(f"  India:     {result2['avg_team1_score']:.1f} (+/- {result2['team1_score_std']:.1f})")
        print(f"  Australia: {result2['avg_team2_score']:.1f} (+/- {result2['team2_score_std']:.1f})")
        
        # H2H usage stats
        h2h = result2['h2h_usage']
        print(f"\nH2H Matchup Usage:")
        print(f"  Deliveries with H2H data: {h2h['h2h_hits']:,}")
        print(f"  Deliveries using ELO fallback: {h2h['h2h_misses']:,}")
        print(f"  H2H usage rate: {h2h['h2h_rate']:.1%}")
        
    except Exception as e:
        print(f"\nCould not load real teams: {e}")
        print("(This is expected if database doesn't have India/Australia)")
