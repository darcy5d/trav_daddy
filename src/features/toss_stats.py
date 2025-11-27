"""
Toss Statistics and Simulation.

Provides historical toss decision rates and a simple toss simulator.
"""

import logging
import random
from typing import Dict, Tuple, Optional
from pathlib import Path
import pickle

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.database import get_connection

logger = logging.getLogger(__name__)


class TossStatistics:
    """
    Build and query historical toss statistics from database.
    """
    
    def __init__(self):
        self.decision_rates: Dict[Tuple[str, str], Dict[str, float]] = {}
        self.toss_winner_advantage: Dict[Tuple[str, str], float] = {}
    
    def build_from_database(self):
        """
        Build toss statistics from historical match data.
        """
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get toss decision rates by format and gender
        for format_type in ['T20', 'ODI']:
            for gender in ['male', 'female']:
                cursor.execute("""
                    SELECT 
                        toss_decision,
                        COUNT(*) as count
                    FROM matches
                    WHERE match_type = ? AND gender = ? 
                    AND toss_decision IS NOT NULL
                    GROUP BY toss_decision
                """, (format_type, gender))
                
                rows = cursor.fetchall()
                total = sum(row['count'] for row in rows)
                
                if total > 0:
                    rates = {}
                    for row in rows:
                        rates[row['toss_decision']] = row['count'] / total
                    
                    # Ensure both bat and field exist
                    rates.setdefault('bat', 0.0)
                    rates.setdefault('field', 0.0)
                    
                    self.decision_rates[(format_type, gender)] = rates
                    logger.info(f"{format_type} {gender}: bat={rates['bat']:.2%}, field={rates['field']:.2%} ({total} matches)")
        
        # Get toss winner advantage (win rate when winning toss)
        for format_type in ['T20', 'ODI']:
            for gender in ['male', 'female']:
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN toss_winner_id = winner_id THEN 1 ELSE 0 END) as toss_winner_won
                    FROM matches
                    WHERE match_type = ? AND gender = ?
                    AND toss_winner_id IS NOT NULL
                    AND winner_id IS NOT NULL
                """, (format_type, gender))
                
                row = cursor.fetchone()
                if row and row['total'] > 0:
                    advantage = row['toss_winner_won'] / row['total']
                    self.toss_winner_advantage[(format_type, gender)] = advantage
                    logger.info(f"{format_type} {gender} toss winner wins: {advantage:.2%}")
        
        conn.close()
    
    def get_decision_rates(
        self, 
        format_type: str = 'T20', 
        gender: str = 'male'
    ) -> Dict[str, float]:
        """
        Get historical bat/field election rates.
        
        Returns:
            {'bat': 0.35, 'field': 0.65}
        """
        key = (format_type, gender)
        if key in self.decision_rates:
            return self.decision_rates[key]
        
        # Default rates if no data
        if format_type == 'T20':
            return {'bat': 0.35, 'field': 0.65}  # T20 teams prefer to chase
        else:
            return {'bat': 0.50, 'field': 0.50}  # ODI more balanced
    
    def get_toss_winner_advantage(
        self,
        format_type: str = 'T20',
        gender: str = 'male'
    ) -> float:
        """
        Get historical win rate when winning the toss.
        
        Returns:
            Win probability (e.g., 0.52 means 52% win rate)
        """
        key = (format_type, gender)
        return self.toss_winner_advantage.get(key, 0.50)
    
    def save(self, path: str = 'data/processed/toss_stats.pkl'):
        """Save toss statistics to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'decision_rates': self.decision_rates,
                'toss_winner_advantage': self.toss_winner_advantage
            }, f)
        logger.info(f"Saved toss statistics to {path}")
    
    @classmethod
    def load(cls, path: str = 'data/processed/toss_stats.pkl') -> 'TossStatistics':
        """Load toss statistics from file."""
        stats = cls()
        with open(path, 'rb') as f:
            data = pickle.load(f)
        stats.decision_rates = data['decision_rates']
        stats.toss_winner_advantage = data['toss_winner_advantage']
        return stats


class TossSimulator:
    """
    Simulate toss outcomes based on historical data.
    """
    
    def __init__(self, stats: Optional[TossStatistics] = None):
        self.stats = stats
        if self.stats is None:
            # Try to load from file, else build
            stats_path = Path('data/processed/toss_stats.pkl')
            if stats_path.exists():
                self.stats = TossStatistics.load(str(stats_path))
            else:
                self.stats = TossStatistics()
                self.stats.build_from_database()
                self.stats.save(str(stats_path))
    
    def simulate_toss(
        self,
        team1_id: int,
        team2_id: int,
        format_type: str = 'T20',
        gender: str = 'male'
    ) -> Tuple[int, str, int]:
        """
        Simulate toss outcome.
        
        Args:
            team1_id: First team ID
            team2_id: Second team ID
            format_type: 'T20' or 'ODI'
            gender: 'male' or 'female'
        
        Returns:
            (toss_winner_id, decision, batting_first_id):
            - toss_winner_id: Which team won the toss
            - decision: 'bat' or 'field'
            - batting_first_id: Which team bats first
        """
        # 50/50 toss winner
        toss_winner = random.choice([team1_id, team2_id])
        toss_loser = team2_id if toss_winner == team1_id else team1_id
        
        # Historical bat/field probability
        rates = self.stats.get_decision_rates(format_type, gender)
        decision = random.choices(
            ['bat', 'field'], 
            weights=[rates['bat'], rates['field']]
        )[0]
        
        # Determine who bats first
        if decision == 'bat':
            batting_first = toss_winner
        else:
            batting_first = toss_loser
        
        return toss_winner, decision, batting_first
    
    def get_batting_order(
        self,
        team1_id: int,
        team2_id: int,
        team1_batters: list,
        team1_bowlers: list,
        team2_batters: list,
        team2_bowlers: list,
        format_type: str = 'T20',
        gender: str = 'male'
    ) -> Tuple[dict, dict]:
        """
        Simulate toss and return batting/bowling order for both innings.
        
        Returns:
            (first_innings, second_innings) where each is:
            {
                'team_id': int,
                'batters': list,
                'bowlers_against': list  # The opposition bowlers
            }
        """
        _, _, batting_first_id = self.simulate_toss(
            team1_id, team2_id, format_type, gender
        )
        
        if batting_first_id == team1_id:
            first_innings = {
                'team_id': team1_id,
                'batters': team1_batters,
                'bowlers_against': team2_bowlers
            }
            second_innings = {
                'team_id': team2_id,
                'batters': team2_batters,
                'bowlers_against': team1_bowlers
            }
        else:
            first_innings = {
                'team_id': team2_id,
                'batters': team2_batters,
                'bowlers_against': team1_bowlers
            }
            second_innings = {
                'team_id': team1_id,
                'batters': team1_batters,
                'bowlers_against': team2_bowlers
            }
        
        return first_innings, second_innings


def build_toss_stats():
    """Build and save toss statistics."""
    print("=" * 60)
    print("TOSS STATISTICS BUILDER")
    print("=" * 60)
    
    stats = TossStatistics()
    stats.build_from_database()
    stats.save()
    
    print("\nSummary:")
    for (fmt, gender), rates in stats.decision_rates.items():
        advantage = stats.toss_winner_advantage.get((fmt, gender), 0.5)
        print(f"  {fmt} {gender}:")
        print(f"    - Elect to bat: {rates['bat']:.1%}")
        print(f"    - Elect to field: {rates['field']:.1%}")
        print(f"    - Toss winner advantage: {advantage:.1%}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    build_toss_stats()

