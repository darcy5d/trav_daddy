"""
Neural Network-Powered Monte Carlo Match Simulator.

This simulator uses the trained Ball Prediction Neural Network to generate
realistic match simulations by predicting each delivery's outcome.

Following the approach from:
https://towardsdatascience.com/predicting-t20-cricket-matches-with-a-ball-simulation-model-1e9cae5dea22/
"""

import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import pickle

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.ball_prediction_nn import BallPredictionModel, OUTCOME_NAMES
from src.features.player_distributions import PlayerDistributionBuilder
from src.models.bowling_captain import SmartCaptain, Bowler, MatchState, BowlingTracker

logger = logging.getLogger(__name__)


@dataclass
class Player:
    """Represents a cricket player."""
    player_id: int
    name: str
    batting_distribution: np.ndarray = None  # 8-element probability vector
    bowling_distribution: np.ndarray = None  # 8-element probability vector
    
    def __post_init__(self):
        if self.batting_distribution is None:
            # Default batting distribution
            self.batting_distribution = np.array([0.35, 0.32, 0.08, 0.02, 0.12, 0.06, 0.04, 0.01])
        if self.bowling_distribution is None:
            # Default bowling distribution
            self.bowling_distribution = np.array([0.38, 0.32, 0.07, 0.01, 0.10, 0.05, 0.05, 0.02])


@dataclass
class InningsState:
    """Current state of an innings."""
    runs: int = 0
    wickets: int = 0
    balls: int = 0
    target: Optional[int] = None
    
    @property
    def overs(self) -> float:
        return self.balls // 6 + (self.balls % 6) / 10
    
    @property
    def overs_completed(self) -> int:
        return self.balls // 6
    
    @property
    def required_rate(self) -> float:
        if self.target is None:
            return 0.0
        balls_remaining = 120 - self.balls
        if balls_remaining <= 0:
            return float('inf')
        runs_needed = self.target - self.runs
        return max(0, runs_needed * 6 / balls_remaining)


@dataclass
class MatchResult:
    """Result of a simulated match."""
    team1_runs: int
    team1_wickets: int
    team2_runs: int
    team2_wickets: int
    winner: int  # 1 or 2
    margin: int
    margin_type: str  # 'runs' or 'wickets'


class NNMatchSimulator:
    """
    Monte Carlo match simulator powered by neural network ball predictions.
    
    For each delivery:
    1. Build feature vector (match state + batter/bowler distributions)
    2. Pass through trained NN to get outcome probabilities
    3. Sample from the probability distribution
    4. Update match state
    """
    
    def __init__(
        self,
        model_path: str = 'data/processed/ball_prediction_model.keras',
        distributions_path: str = 'data/processed/player_distributions_t20_male.pkl',
        use_smart_captain: bool = True
    ):
        self.model = BallPredictionModel(model_path)
        self.model.load()
        
        self.player_distributions = PlayerDistributionBuilder.load(distributions_path)
        self.use_smart_captain = use_smart_captain
        
        logger.info("NN Match Simulator initialized")
    
    def _build_feature_vector(
        self,
        innings_number: int,
        over: int,
        balls_bowled: int,
        runs: int,
        wickets: int,
        required_rate: float,
        batter: Player,
        bowler: Player
    ) -> np.ndarray:
        """Build feature vector for ball prediction."""
        # Phase one-hot
        if over < 6:
            phase = [1, 0, 0]  # Powerplay
        elif over < 15:
            phase = [0, 1, 0]  # Middle
        else:
            phase = [0, 0, 1]  # Death
        
        # Get player distributions
        batter_dist = batter.batting_distribution
        bowler_dist = bowler.bowling_distribution
        
        # Build feature vector (25 features)
        features = np.concatenate([
            np.array([innings_number, over, balls_bowled, runs, wickets, required_rate]),
            np.array(phase),
            batter_dist,
            bowler_dist
        ])
        
        return features.astype(np.float32)
    
    def simulate_delivery(
        self,
        state: InningsState,
        batter: Player,
        bowler: Player,
        innings_number: int
    ) -> Tuple[int, bool]:
        """
        Simulate a single delivery using the neural network.
        
        Returns:
            (runs_scored, is_wicket)
        """
        features = self._build_feature_vector(
            innings_number=innings_number,
            over=state.balls // 6,
            balls_bowled=state.balls,
            runs=state.runs,
            wickets=state.wickets,
            required_rate=state.required_rate,
            batter=batter,
            bowler=bowler
        )
        
        # Get prediction from NN and sample
        outcome_class = self.model.sample_outcome(features)
        
        # Convert to runs/wicket
        return self.model.class_to_runs_wicket(outcome_class)
    
    def simulate_innings(
        self,
        batting_xi: List[Player],
        bowling_attack: List[Player],
        innings_number: int,
        target: Optional[int] = None,
        max_overs: int = 20
    ) -> InningsState:
        """
        Simulate a complete innings.
        
        Args:
            batting_xi: List of 11 batters in order
            bowling_attack: List of bowlers
            innings_number: 1 or 2
            target: Target to chase (None if batting first)
            max_overs: Maximum overs (20 for T20)
        
        Returns:
            Final innings state
        """
        state = InningsState(target=target)
        
        current_batter_idx = 0
        non_striker_idx = 1
        
        # Set up bowling with smart captain
        if self.use_smart_captain:
            bowlers = [
                Bowler(
                    player_id=p.player_id,
                    name=p.name,
                    bowling_elo=1500  # Default ELO
                )
                for p in bowling_attack[:5]
            ]
            captain = SmartCaptain(bowlers, format_type='T20', gender='male')
        
        for over in range(max_overs):
            # Select bowler
            if self.use_smart_captain:
                batter_ids = [
                    batting_xi[current_batter_idx].player_id,
                    batting_xi[non_striker_idx].player_id
                ]
                match_state = MatchState(
                    total_runs=state.runs,
                    wickets=state.wickets,
                    overs_completed=over,
                    target=target,
                    required_rate=state.required_rate
                )
                selected_bowler = captain.select_bowler(batter_ids, match_state, over + 1)
                bowler = next(p for p in bowling_attack if p.player_id == selected_bowler.player_id)
            else:
                bowler = bowling_attack[over % len(bowling_attack)]
            
            # Bowl the over
            for ball in range(6):
                # Check termination conditions
                if state.wickets >= 10:
                    if self.use_smart_captain:
                        captain.record_over_complete(bowler.player_id)
                    return state
                
                if target is not None and state.runs >= target:
                    if self.use_smart_captain:
                        captain.record_over_complete(bowler.player_id)
                    return state
                
                # Current batter
                batter = batting_xi[current_batter_idx]
                
                # Simulate delivery
                runs, is_wicket = self.simulate_delivery(
                    state, batter, bowler, innings_number
                )
                
                state.runs += runs
                state.balls += 1
                
                if is_wicket:
                    state.wickets += 1
                    if state.wickets < 10:
                        current_batter_idx = min(10, state.wickets + 1)
                
                # Rotate strike for odd runs
                if runs % 2 == 1:
                    current_batter_idx, non_striker_idx = non_striker_idx, current_batter_idx
            
            # End of over
            if self.use_smart_captain:
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
    ) -> MatchResult:
        """Simulate a complete match."""
        if team1_bats_first:
            first_batting, first_bowling = team1_batting, team2_bowling
            second_batting, second_bowling = team2_batting, team1_bowling
        else:
            first_batting, first_bowling = team2_batting, team1_bowling
            second_batting, second_bowling = team1_batting, team2_bowling
        
        # First innings
        first_innings = self.simulate_innings(first_batting, first_bowling, innings_number=1)
        
        # Second innings (chase)
        target = first_innings.runs + 1
        second_innings = self.simulate_innings(
            second_batting, second_bowling, 
            innings_number=2, 
            target=target
        )
        
        # Determine winner
        if second_innings.runs >= target:
            winner = 2 if team1_bats_first else 1
            margin = 10 - second_innings.wickets
            margin_type = 'wickets'
        else:
            winner = 1 if team1_bats_first else 2
            margin = first_innings.runs - second_innings.runs
            margin_type = 'runs'
        
        return MatchResult(
            team1_runs=first_innings.runs if team1_bats_first else second_innings.runs,
            team1_wickets=first_innings.wickets if team1_bats_first else second_innings.wickets,
            team2_runs=second_innings.runs if team1_bats_first else first_innings.runs,
            team2_wickets=second_innings.wickets if team1_bats_first else first_innings.wickets,
            winner=winner,
            margin=margin,
            margin_type=margin_type
        )
    
    def run_simulations(
        self,
        team1_batting: List[Player],
        team1_bowling: List[Player],
        team2_batting: List[Player],
        team2_bowling: List[Player],
        n_simulations: int = 1000,
        team1_bats_first: bool = True
    ) -> Dict:
        """
        Run Monte Carlo simulations.
        
        Returns aggregate statistics.
        """
        results = []
        
        for i in range(n_simulations):
            result = self.simulate_match(
                team1_batting, team1_bowling,
                team2_batting, team2_bowling,
                team1_bats_first
            )
            results.append(result)
        
        # Aggregate
        team1_wins = sum(1 for r in results if r.winner == 1)
        team1_scores = [r.team1_runs for r in results]
        team2_scores = [r.team2_runs for r in results]
        team1_wickets = [r.team1_wickets for r in results]
        team2_wickets = [r.team2_wickets for r in results]
        
        return {
            'team1_win_prob': team1_wins / n_simulations,
            'team2_win_prob': 1 - team1_wins / n_simulations,
            'n_simulations': n_simulations,
            'avg_team1_score': np.mean(team1_scores),
            'avg_team2_score': np.mean(team2_scores),
            'std_team1_score': np.std(team1_scores),
            'std_team2_score': np.std(team2_scores),
            'avg_team1_wickets': np.mean(team1_wickets),
            'avg_team2_wickets': np.mean(team2_wickets),
            'team1_score_range': (np.percentile(team1_scores, 5), np.percentile(team1_scores, 95)),
            'team2_score_range': (np.percentile(team2_scores, 5), np.percentile(team2_scores, 95)),
        }


def load_team_players(
    team_name: str,
    distributions: PlayerDistributionBuilder,
    n_batters: int = 11,
    n_bowlers: int = 5
) -> Tuple[List[Player], List[Player]]:
    """
    Load team players with their historical distributions.
    """
    from src.data.database import get_connection
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get team ID
    cursor.execute("SELECT team_id FROM teams WHERE name = ?", (team_name,))
    row = cursor.fetchone()
    if not row:
        raise ValueError(f"Team '{team_name}' not found")
    team_id = row['team_id']
    
    # Get recent players for this team (by batting appearances)
    cursor.execute("""
        SELECT DISTINCT p.player_id, p.name
        FROM players p
        JOIN player_match_stats pms ON p.player_id = pms.player_id
        JOIN matches m ON pms.match_id = m.match_id
        WHERE m.match_type = 'T20' AND m.gender = 'male'
        AND pms.team_id = ?
        AND pms.runs_scored > 0
        GROUP BY p.player_id
        ORDER BY MAX(m.date) DESC, SUM(pms.runs_scored) DESC
        LIMIT ?
    """, (team_id, n_batters))
    
    batters = []
    for row in cursor.fetchall():
        bat_dist = distributions.get_batter_vector(row['player_id'])
        bowl_dist = distributions.get_bowler_vector(row['player_id'])
        batters.append(Player(
            player_id=row['player_id'],
            name=row['name'],
            batting_distribution=bat_dist,
            bowling_distribution=bowl_dist
        ))
    
    # Get bowlers
    cursor.execute("""
        SELECT DISTINCT p.player_id, p.name
        FROM players p
        JOIN player_match_stats pms ON p.player_id = pms.player_id
        JOIN matches m ON pms.match_id = m.match_id
        WHERE m.match_type = 'T20' AND m.gender = 'male'
        AND pms.team_id = ?
        AND pms.overs_bowled > 0
        GROUP BY p.player_id
        ORDER BY MAX(m.date) DESC, SUM(pms.wickets_taken) DESC
        LIMIT ?
    """, (team_id, n_bowlers))
    
    bowlers = []
    for row in cursor.fetchall():
        bat_dist = distributions.get_batter_vector(row['player_id'])
        bowl_dist = distributions.get_bowler_vector(row['player_id'])
        bowlers.append(Player(
            player_id=row['player_id'],
            name=row['name'],
            batting_distribution=bat_dist,
            bowling_distribution=bowl_dist
        ))
    
    conn.close()
    
    # Pad if needed
    while len(batters) < n_batters:
        batters.append(Player(
            player_id=-len(batters),
            name=f"Batter {len(batters)+1}"
        ))
    while len(bowlers) < n_bowlers:
        bowlers.append(Player(
            player_id=-len(bowlers)-100,
            name=f"Bowler {len(bowlers)+1}"
        ))
    
    return batters, bowlers


def main():
    """Test the NN-powered simulator."""
    print("=" * 70)
    print("NEURAL NETWORK MATCH SIMULATOR TEST")
    print("=" * 70)
    
    # Initialize
    simulator = NNMatchSimulator()
    distributions = simulator.player_distributions
    
    # Load teams
    print("\nLoading India vs Australia...")
    
    try:
        india_bat, india_bowl = load_team_players("India", distributions)
        aus_bat, aus_bowl = load_team_players("Australia", distributions)
        
        print(f"\nIndia Batting XI:")
        for i, p in enumerate(india_bat[:5]):
            print(f"  {i+1}. {p.name}")
        print("  ...")
        
        print(f"\nAustralia Bowling Attack:")
        for p in aus_bowl:
            print(f"  - {p.name}")
        
        # Run simulations
        print("\n" + "=" * 70)
        print("Running 1000 Monte Carlo Simulations...")
        print("=" * 70)
        
        results = simulator.run_simulations(
            india_bat, india_bowl,
            aus_bat, aus_bowl,
            n_simulations=1000
        )
        
        print(f"\n{'='*70}")
        print("INDIA vs AUSTRALIA - SIMULATION RESULTS")
        print("=" * 70)
        print(f"\n  India Win Probability:     {results['team1_win_prob']:.1%}")
        print(f"  Australia Win Probability: {results['team2_win_prob']:.1%}")
        print(f"\n  Average Scores:")
        print(f"    India:     {results['avg_team1_score']:.1f}/{results['avg_team1_wickets']:.1f}")
        print(f"    Australia: {results['avg_team2_score']:.1f}/{results['avg_team2_wickets']:.1f}")
        print(f"\n  Score Ranges (90% CI):")
        print(f"    India:     {results['team1_score_range'][0]:.0f} - {results['team1_score_range'][1]:.0f}")
        print(f"    Australia: {results['team2_score_range'][0]:.0f} - {results['team2_score_range'][1]:.0f}")
        
    except Exception as e:
        print(f"\nError loading teams: {e}")
        print("Running with dummy teams instead...")
        
        # Create dummy teams
        batters1 = [Player(i, f"Batter1_{i}") for i in range(11)]
        bowlers1 = [Player(i+100, f"Bowler1_{i}") for i in range(5)]
        batters2 = [Player(i+200, f"Batter2_{i}") for i in range(11)]
        bowlers2 = [Player(i+300, f"Bowler2_{i}") for i in range(5)]
        
        results = simulator.run_simulations(
            batters1, bowlers1,
            batters2, bowlers2,
            n_simulations=1000
        )
        
        print(f"\n  Team 1 Win Probability: {results['team1_win_prob']:.1%}")
        print(f"  Team 2 Win Probability: {results['team2_win_prob']:.1%}")
        print(f"\n  Average Scores:")
        print(f"    Team 1: {results['avg_team1_score']:.1f}")
        print(f"    Team 2: {results['avg_team2_score']:.1f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()


