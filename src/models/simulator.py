"""
Monte Carlo Ball-by-Ball Match Simulator.

Simulates cricket matches ball-by-ball using probability distributions
derived from historical data and player ELO ratings.

This approach is inspired by the article:
"Predicting T20 Cricket Matches with a Ball Simulation Model"
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import SIMULATION_CONFIG, ELO_CONFIG
from src.data.database import get_db_connection
from src.elo.calculator import EloCalculator

logger = logging.getLogger(__name__)


@dataclass
class BallOutcome:
    """Represents the outcome of a single delivery."""
    runs_batter: int = 0
    runs_extras: int = 0
    is_wicket: bool = False
    is_wide: bool = False
    is_noball: bool = False
    wicket_type: Optional[str] = None


@dataclass
class BatterState:
    """State of a batter during simulation."""
    player_id: int
    name: str
    batting_elo: float
    runs: int = 0
    balls: int = 0
    fours: int = 0
    sixes: int = 0
    is_out: bool = False


@dataclass 
class BowlerState:
    """State of a bowler during simulation."""
    player_id: int
    name: str
    bowling_elo: float
    overs: float = 0
    runs: int = 0
    wickets: int = 0
    balls_bowled: int = 0


@dataclass
class InningsState:
    """State of an innings during simulation."""
    batting_team: str
    bowling_team: str
    total_runs: int = 0
    wickets: int = 0
    overs: float = 0
    balls_in_over: int = 0
    target: Optional[int] = None
    batters: List[BatterState] = field(default_factory=list)
    bowlers: List[BowlerState] = field(default_factory=list)
    current_batter_idx: int = 0
    current_bowler_idx: int = 0
    

class BallProbabilityModel:
    """
    Model for calculating ball outcome probabilities.
    
    Uses historical data and player ELO ratings to determine
    probability distributions for each delivery.
    """
    
    def __init__(self):
        self.elo_calc = EloCalculator()
        
        # Base probability distributions (can be refined with data)
        # These are approximate distributions for T20 cricket
        self.base_run_probs = {
            'T20': {
                'powerplay': [0.32, 0.28, 0.05, 0.01, 0.18, 0.00, 0.08, 0.08],  # 0,1,2,3,4,5,6,wicket
                'middle': [0.38, 0.30, 0.06, 0.01, 0.12, 0.00, 0.06, 0.07],
                'death': [0.30, 0.25, 0.08, 0.02, 0.15, 0.00, 0.12, 0.08]
            },
            'ODI': {
                'powerplay': [0.35, 0.32, 0.08, 0.02, 0.12, 0.00, 0.04, 0.07],
                'middle': [0.42, 0.30, 0.08, 0.02, 0.08, 0.00, 0.03, 0.07],
                'death': [0.32, 0.28, 0.08, 0.02, 0.14, 0.00, 0.08, 0.08]
            }
        }
        
        # Wide/Noball probabilities
        self.extras_prob = {
            'T20': {'wide': 0.03, 'noball': 0.01},
            'ODI': {'wide': 0.02, 'noball': 0.01}
        }
    
    def get_phase(self, over: int, match_format: str) -> str:
        """Determine the phase of the innings."""
        if match_format == 'T20':
            if over < 6:
                return 'powerplay'
            elif over < 15:
                return 'middle'
            else:
                return 'death'
        else:  # ODI
            if over < 10:
                return 'powerplay'
            elif over < 40:
                return 'middle'
            else:
                return 'death'
    
    def adjust_probabilities(
        self,
        base_probs: List[float],
        batter_elo: float,
        bowler_elo: float,
        match_situation: Dict[str, Any]
    ) -> List[float]:
        """
        Adjust base probabilities based on player ELOs and situation.
        
        Args:
            base_probs: Base probability distribution [0,1,2,3,4,5,6,wicket]
            batter_elo: Batter's ELO rating
            bowler_elo: Bowler's ELO rating
            match_situation: Current match state
            
        Returns:
            Adjusted probability distribution
        """
        probs = list(base_probs)
        initial_elo = ELO_CONFIG['initial_rating']
        
        # ELO differential effect
        elo_diff = batter_elo - bowler_elo
        adjustment_factor = 1 + (elo_diff / 800)  # ~+/-25% at 200 ELO diff
        
        # Adjust scoring probabilities (indices 1-6)
        for i in range(1, 7):
            probs[i] *= adjustment_factor
        
        # Adjust wicket probability (inverse effect)
        probs[7] /= adjustment_factor
        
        # Situation-based adjustments
        if 'required_rate' in match_situation:
            req_rate = match_situation['required_rate']
            if req_rate > 12:  # High pressure chase
                # More aggressive = more boundaries but also more wickets
                probs[4] *= 1.2  # More 4s
                probs[6] *= 1.3  # More 6s
                probs[7] *= 1.15  # More wickets
        
        # Normalize
        total = sum(probs)
        return [p / total for p in probs]
    
    def simulate_ball(
        self,
        batter_elo: float,
        bowler_elo: float,
        match_format: str,
        over: int,
        match_situation: Dict[str, Any]
    ) -> BallOutcome:
        """
        Simulate a single ball delivery.
        
        Args:
            batter_elo: Batter's ELO rating
            bowler_elo: Bowler's ELO rating
            match_format: 'T20' or 'ODI'
            over: Current over number
            match_situation: Current match state
            
        Returns:
            BallOutcome for this delivery
        """
        # Check for extras first
        if np.random.random() < self.extras_prob[match_format]['wide']:
            return BallOutcome(runs_extras=1, is_wide=True)
        
        if np.random.random() < self.extras_prob[match_format]['noball']:
            # Noball - still need to determine runs
            pass
        
        # Get phase-based probabilities
        phase = self.get_phase(over, match_format)
        base_probs = self.base_run_probs[match_format][phase]
        
        # Adjust for player quality and situation
        probs = self.adjust_probabilities(
            base_probs, batter_elo, bowler_elo, match_situation
        )
        
        # Sample outcome
        outcomes = [0, 1, 2, 3, 4, 5, 6, 'wicket']
        outcome = np.random.choice(range(len(outcomes)), p=probs)
        
        if outcome == 7:  # Wicket
            return BallOutcome(is_wicket=True, wicket_type='caught')
        
        runs = outcomes[outcome]
        return BallOutcome(runs_batter=runs)


class MatchSimulator:
    """
    Monte Carlo match simulator.
    
    Runs multiple simulations to estimate match outcome probabilities.
    """
    
    def __init__(self):
        self.prob_model = BallProbabilityModel()
        self.elo_calc = EloCalculator()
        self.num_simulations = SIMULATION_CONFIG['num_simulations']
    
    def simulate_innings(
        self,
        batting_team_elos: List[float],
        bowling_team_elos: List[float],
        match_format: str,
        target: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Simulate a complete innings.
        
        Args:
            batting_team_elos: List of batting ELOs for each batter
            bowling_team_elos: List of bowling ELOs for bowlers
            match_format: 'T20' or 'ODI'
            target: Target score (for second innings)
            
        Returns:
            Innings result dictionary
        """
        max_overs = 20 if match_format == 'T20' else 50
        max_wickets = 10
        
        total_runs = 0
        wickets = 0
        balls = 0
        batter_idx = 0
        
        # Cycle through bowlers
        bowler_elos = bowling_team_elos[:5]  # Top 5 bowlers
        bowler_idx = 0
        
        while balls < max_overs * 6 and wickets < max_wickets:
            over = balls // 6
            
            # Current batter and bowler
            batter_elo = batting_team_elos[min(batter_idx, len(batting_team_elos) - 1)]
            bowler_elo = bowler_elos[bowler_idx % len(bowler_elos)]
            
            # Match situation
            overs_remaining = max_overs - over
            situation = {
                'overs_remaining': overs_remaining,
                'wickets_remaining': max_wickets - wickets,
                'current_score': total_runs
            }
            
            if target:
                runs_needed = target - total_runs
                balls_remaining = (max_overs * 6) - balls
                situation['required_rate'] = (runs_needed / balls_remaining) * 6 if balls_remaining > 0 else 99
            
            # Simulate ball
            outcome = self.prob_model.simulate_ball(
                batter_elo, bowler_elo, match_format, over, situation
            )
            
            # Process outcome
            if outcome.is_wide:
                total_runs += 1
                # Wide doesn't count as a ball
            else:
                balls += 1
                total_runs += outcome.runs_batter + outcome.runs_extras
                
                if outcome.is_wicket:
                    wickets += 1
                    batter_idx += 1
            
            # Change bowler at end of over
            if balls % 6 == 0 and balls > 0:
                bowler_idx += 1
            
            # Check if target reached
            if target and total_runs >= target:
                break
        
        return {
            'runs': total_runs,
            'wickets': wickets,
            'balls': balls,
            'overs': balls / 6,
            'won_chase': total_runs >= target if target else None
        }
    
    def simulate_match(
        self,
        team1_batting_elos: List[float],
        team1_bowling_elos: List[float],
        team2_batting_elos: List[float],
        team2_bowling_elos: List[float],
        match_format: str,
        team1_bats_first: bool = True
    ) -> Dict[str, Any]:
        """
        Simulate a complete match.
        
        Args:
            team1_batting_elos: Team 1 batting ELOs
            team1_bowling_elos: Team 1 bowling ELOs
            team2_batting_elos: Team 2 batting ELOs
            team2_bowling_elos: Team 2 bowling ELOs
            match_format: 'T20' or 'ODI'
            team1_bats_first: Whether team 1 bats first
            
        Returns:
            Match result dictionary
        """
        if team1_bats_first:
            first_bat = team1_batting_elos
            first_bowl = team2_bowling_elos
            second_bat = team2_batting_elos
            second_bowl = team1_bowling_elos
            first_team = 'team1'
        else:
            first_bat = team2_batting_elos
            first_bowl = team1_bowling_elos
            second_bat = team1_batting_elos
            second_bowl = team2_bowling_elos
            first_team = 'team2'
        
        # First innings
        innings1 = self.simulate_innings(first_bat, first_bowl, match_format)
        target = innings1['runs'] + 1
        
        # Second innings
        innings2 = self.simulate_innings(second_bat, second_bowl, match_format, target)
        
        # Determine winner
        if innings2['runs'] >= target:
            winner = 'team2' if first_team == 'team1' else 'team1'
            win_type = 'wickets'
            margin = 10 - innings2['wickets']
        else:
            winner = first_team
            win_type = 'runs'
            margin = innings1['runs'] - innings2['runs']
        
        return {
            'winner': winner,
            'win_type': win_type,
            'margin': margin,
            'innings1': innings1,
            'innings2': innings2,
            'team1_bats_first': team1_bats_first
        }
    
    def run_monte_carlo(
        self,
        team1_batting_elos: List[float],
        team1_bowling_elos: List[float],
        team2_batting_elos: List[float],
        team2_bowling_elos: List[float],
        match_format: str,
        n_simulations: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for match prediction.
        
        Args:
            team1_batting_elos: Team 1 batting ELOs
            team1_bowling_elos: Team 1 bowling ELOs
            team2_batting_elos: Team 2 batting ELOs
            team2_bowling_elos: Team 2 bowling ELOs
            match_format: 'T20' or 'ODI'
            n_simulations: Number of simulations (default from config)
            
        Returns:
            Simulation results with win probabilities
        """
        if n_simulations is None:
            n_simulations = self.num_simulations
        
        results = {
            'team1_wins': 0,
            'team2_wins': 0,
            'team1_scores': [],
            'team2_scores': [],
            'margins': []
        }
        
        for i in range(n_simulations):
            # Randomly determine who bats first (50/50 for now)
            team1_bats_first = np.random.random() < 0.5
            
            match_result = self.simulate_match(
                team1_batting_elos, team1_bowling_elos,
                team2_batting_elos, team2_bowling_elos,
                match_format, team1_bats_first
            )
            
            if match_result['winner'] == 'team1':
                results['team1_wins'] += 1
            else:
                results['team2_wins'] += 1
            
            # Track scores
            if team1_bats_first:
                results['team1_scores'].append(match_result['innings1']['runs'])
                results['team2_scores'].append(match_result['innings2']['runs'])
            else:
                results['team1_scores'].append(match_result['innings2']['runs'])
                results['team2_scores'].append(match_result['innings1']['runs'])
            
            results['margins'].append(match_result['margin'])
        
        # Calculate probabilities and statistics
        team1_prob = results['team1_wins'] / n_simulations
        team2_prob = results['team2_wins'] / n_simulations
        
        return {
            'team1_win_probability': team1_prob,
            'team2_win_probability': team2_prob,
            'team1_expected_score': np.mean(results['team1_scores']),
            'team2_expected_score': np.mean(results['team2_scores']),
            'team1_score_std': np.std(results['team1_scores']),
            'team2_score_std': np.std(results['team2_scores']),
            'average_margin': np.mean(results['margins']),
            'n_simulations': n_simulations,
            'confidence_interval': self._calculate_ci(team1_prob, n_simulations)
        }
    
    def _calculate_ci(self, p: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for probability."""
        from scipy import stats
        z = stats.norm.ppf((1 + confidence) / 2)
        margin = z * np.sqrt(p * (1 - p) / n)
        return (max(0, p - margin), min(1, p + margin))
    
    def predict_match(
        self,
        conn,
        team1_id: int,
        team2_id: int,
        match_format: str,
        match_date: datetime,
        team1_players: Optional[List[int]] = None,
        team2_players: Optional[List[int]] = None,
        n_simulations: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Predict match outcome using Monte Carlo simulation.
        
        Args:
            conn: Database connection
            team1_id: Team 1 ID
            team2_id: Team 2 ID
            match_format: 'T20' or 'ODI'
            match_date: Date for ELO lookup
            team1_players: Optional player IDs for team 1
            team2_players: Optional player IDs for team 2
            n_simulations: Number of simulations
            
        Returns:
            Prediction results
        """
        # Get player ELOs (or use team average if players not specified)
        if team1_players:
            team1_batting = [
                self.elo_calc.get_player_rating(conn, pid, match_format, 'batting', match_date)
                for pid in team1_players
            ]
            team1_bowling = [
                self.elo_calc.get_player_rating(conn, pid, match_format, 'bowling', match_date)
                for pid in team1_players
            ]
        else:
            # Use team ELO as proxy
            team_elo = self.elo_calc.get_team_rating(conn, team1_id, match_format, match_date)
            team1_batting = [team_elo] * 11
            team1_bowling = [team_elo] * 11
        
        if team2_players:
            team2_batting = [
                self.elo_calc.get_player_rating(conn, pid, match_format, 'batting', match_date)
                for pid in team2_players
            ]
            team2_bowling = [
                self.elo_calc.get_player_rating(conn, pid, match_format, 'bowling', match_date)
                for pid in team2_players
            ]
        else:
            team_elo = self.elo_calc.get_team_rating(conn, team2_id, match_format, match_date)
            team2_batting = [team_elo] * 11
            team2_bowling = [team_elo] * 11
        
        # Run simulation
        return self.run_monte_carlo(
            team1_batting, team1_bowling,
            team2_batting, team2_bowling,
            match_format, n_simulations
        )


def simulate_match_cli(
    team1_name: str,
    team2_name: str,
    match_format: str = 'T20',
    n_simulations: int = 1000
):
    """
    Command-line interface for match simulation.
    """
    simulator = MatchSimulator()
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Find teams
        cursor.execute("SELECT team_id FROM teams WHERE name = ?", (team1_name,))
        row = cursor.fetchone()
        if not row:
            print(f"Team not found: {team1_name}")
            return
        team1_id = row['team_id']
        
        cursor.execute("SELECT team_id FROM teams WHERE name = ?", (team2_name,))
        row = cursor.fetchone()
        if not row:
            print(f"Team not found: {team2_name}")
            return
        team2_id = row['team_id']
        
        # Run simulation
        print(f"\nSimulating {team1_name} vs {team2_name} ({match_format})")
        print(f"Running {n_simulations:,} simulations...")
        
        results = simulator.predict_match(
            conn, team1_id, team2_id, match_format,
            datetime.now(), n_simulations=n_simulations
        )
        
        print("\n" + "=" * 50)
        print("MATCH PREDICTION RESULTS")
        print("=" * 50)
        print(f"\n{team1_name}: {results['team1_win_probability']*100:.1f}% win probability")
        print(f"{team2_name}: {results['team2_win_probability']*100:.1f}% win probability")
        print(f"\nExpected scores:")
        print(f"  {team1_name}: {results['team1_expected_score']:.0f} "
              f"(±{results['team1_score_std']:.0f})")
        print(f"  {team2_name}: {results['team2_expected_score']:.0f} "
              f"(±{results['team2_score_std']:.0f})")
        
        ci = results['confidence_interval']
        print(f"\n95% confidence interval: {ci[0]*100:.1f}% - {ci[1]*100:.1f}%")
        print("=" * 50)


def main():
    """Run match simulation."""
    logging.basicConfig(level=logging.INFO)
    
    import argparse
    parser = argparse.ArgumentParser(description='Monte Carlo match simulation')
    parser.add_argument('--team1', required=True, help='Team 1 name')
    parser.add_argument('--team2', required=True, help='Team 2 name')
    parser.add_argument('--format', default='T20', choices=['T20', 'ODI'])
    parser.add_argument('--simulations', type=int, default=1000)
    args = parser.parse_args()
    
    simulate_match_cli(args.team1, args.team2, args.format, args.simulations)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

