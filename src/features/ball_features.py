"""
Ball-Level Feature Engineering for Cricket Simulation.

Creates features for predicting ball-by-ball outcomes:
- Batter ELO vs Bowler ELO matchup
- Innings phase (powerplay, middle, death)
- Match situation (score, wickets, required rate)
"""

import logging
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import DATABASE_PATH
from src.data.database import get_connection

logger = logging.getLogger(__name__)


def get_innings_phase(over_number: int, format_type: str = 'T20') -> str:
    """Determine innings phase based on over number."""
    if format_type == 'T20':
        if over_number < 6:
            return 'powerplay'
        elif over_number < 15:
            return 'middle'
        else:
            return 'death'
    else:  # ODI
        if over_number < 10:
            return 'powerplay'
        elif over_number < 40:
            return 'middle'
        else:
            return 'death'


def get_pressure_index(
    current_score: int,
    wickets_fallen: int,
    balls_bowled: int,
    target: Optional[int] = None,
    format_type: str = 'T20'
) -> float:
    """
    Calculate pressure index (0-1) based on match situation.
    
    Higher values = more pressure on batting team.
    """
    max_balls = 120 if format_type == 'T20' else 300
    balls_remaining = max(1, max_balls - balls_bowled)
    
    # Wicket pressure (more wickets = more pressure)
    wicket_pressure = wickets_fallen / 10
    
    if target is not None:
        # Chasing: pressure based on required rate
        runs_needed = target - current_score
        if runs_needed <= 0:
            return 0.0  # Already won
        
        required_rate = runs_needed * 6 / balls_remaining
        
        # Normalize RRR (8 RPO is normal, 12+ is very high pressure)
        if format_type == 'T20':
            rr_pressure = min(1.0, max(0, (required_rate - 6) / 8))
        else:
            rr_pressure = min(1.0, max(0, (required_rate - 4) / 6))
        
        return 0.6 * rr_pressure + 0.4 * wicket_pressure
    else:
        # First innings: pressure mainly from wickets
        return wicket_pressure


class BallFeatureBuilder:
    """Builds ball-level features for simulation model training."""
    
    def __init__(self):
        self.conn = None
    
    def connect(self):
        """Establish database connection."""
        self.conn = get_connection()
        return self
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        return self.connect()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def get_player_elo_at_date(
        self,
        player_id: int,
        match_date: str,
        format_type: str,
        gender: str,
        elo_type: str  # 'batting' or 'bowling'
    ) -> float:
        """Get player ELO rating just before a match date."""
        cursor = self.conn.cursor()
        
        cursor.execute(f"""
            SELECT {elo_type}_elo FROM player_elo_history
            WHERE player_id = ? AND format = ? AND gender = ? AND date < ?
            ORDER BY date DESC, elo_id DESC
            LIMIT 1
        """, (player_id, format_type, gender, match_date))
        
        row = cursor.fetchone()
        return row[0] if row else 1500.0
    
    def categorize_outcome(self, delivery: Dict) -> str:
        """Categorize delivery outcome for simulation."""
        if delivery['is_wicket']:
            return 'W'
        
        runs = delivery['runs_batter']
        if runs == 0:
            return '0'
        elif runs == 1:
            return '1'
        elif runs == 2:
            return '2'
        elif runs == 3:
            return '3'
        elif runs == 4:
            return '4'
        elif runs >= 6:
            return '6'
        else:
            return str(runs)
    
    def build_delivery_features(
        self,
        format_type: str = 'T20',
        gender: str = 'male',
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Build features dataset for all deliveries."""
        cursor = self.conn.cursor()
        
        # Get deliveries with match context
        query = """
            SELECT 
                d.delivery_id,
                d.innings_id,
                d.over_number,
                d.ball_number,
                d.batter_id,
                d.bowler_id,
                d.runs_batter,
                d.runs_total,
                d.is_wicket,
                d.wicket_type,
                i.match_id,
                i.innings_number,
                i.batting_team_id,
                i.bowling_team_id,
                i.target_runs,
                m.date,
                m.match_type,
                m.gender
            FROM deliveries d
            JOIN innings i ON d.innings_id = i.innings_id
            JOIN matches m ON i.match_id = m.match_id
            WHERE m.match_type = ? AND m.gender = ?
            ORDER BY m.date, d.innings_id, d.over_number, d.ball_number
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query, (format_type, gender))
        deliveries = cursor.fetchall()
        
        logger.info(f"Processing {len(deliveries)} deliveries...")
        
        # Track running totals per innings
        innings_state = {}
        features = []
        
        for i, d in enumerate(deliveries):
            if i % 50000 == 0:
                logger.info(f"Processing delivery {i+1}/{len(deliveries)}")
            
            innings_id = d['innings_id']
            
            # Initialize innings state if new
            if innings_id not in innings_state:
                innings_state[innings_id] = {
                    'score': 0,
                    'wickets': 0,
                    'balls': 0
                }
            
            state = innings_state[innings_id]
            
            # Get player ELOs (use current ELO for speed, could use historical)
            batter_elo = self.get_player_elo_at_date(
                d['batter_id'], d['date'], format_type, gender, 'batting'
            ) if i < 10000 else 1500  # Limit ELO lookups for speed
            
            bowler_elo = self.get_player_elo_at_date(
                d['bowler_id'], d['date'], format_type, gender, 'bowling'
            ) if i < 10000 else 1500
            
            # Calculate features
            phase = get_innings_phase(d['over_number'], format_type)
            pressure = get_pressure_index(
                state['score'],
                state['wickets'],
                state['balls'],
                d['target_runs'],
                format_type
            )
            
            # ELO matchup
            elo_diff = batter_elo - bowler_elo
            
            # Outcome
            outcome = self.categorize_outcome(d)
            
            features.append({
                'delivery_id': d['delivery_id'],
                'match_id': d['match_id'],
                'innings_number': d['innings_number'],
                'over_number': d['over_number'],
                'ball_number': d['ball_number'],
                'batter_elo': batter_elo,
                'bowler_elo': bowler_elo,
                'elo_diff': elo_diff,
                'elo_diff_bucket': self.bucket_elo_diff(elo_diff),
                'phase': phase,
                'current_score': state['score'],
                'wickets_fallen': state['wickets'],
                'balls_bowled': state['balls'],
                'is_chasing': 1 if d['target_runs'] else 0,
                'target': d['target_runs'],
                'pressure_index': pressure,
                'outcome': outcome,
                'runs': d['runs_batter'],
                'is_wicket': d['is_wicket']
            })
            
            # Update state
            state['score'] += d['runs_total']
            state['balls'] += 1
            if d['is_wicket']:
                state['wickets'] += 1
        
        return pd.DataFrame(features)
    
    def bucket_elo_diff(self, elo_diff: float) -> str:
        """Bucket ELO difference into categories."""
        if elo_diff < -150:
            return 'very_low'  # Batter much weaker
        elif elo_diff < -50:
            return 'low'
        elif elo_diff < 50:
            return 'even'
        elif elo_diff < 150:
            return 'high'
        else:
            return 'very_high'  # Batter much stronger


def analyze_outcome_distributions(
    format_type: str = 'T20',
    gender: str = 'male'
) -> Dict:
    """
    Analyze historical outcome distributions by ELO matchup and phase.
    
    Returns probability tables for simulation.
    """
    with BallFeatureBuilder() as builder:
        df = builder.build_delivery_features(format_type, gender, limit=100000)
    
    logger.info(f"Analyzing {len(df)} deliveries...")
    
    # Calculate outcome distributions by ELO bucket and phase
    distributions = {}
    
    outcomes = ['0', '1', '2', '3', '4', '6', 'W']
    
    for phase in ['powerplay', 'middle', 'death']:
        distributions[phase] = {}
        
        for elo_bucket in ['very_low', 'low', 'even', 'high', 'very_high']:
            subset = df[(df['phase'] == phase) & (df['elo_diff_bucket'] == elo_bucket)]
            
            if len(subset) < 100:
                # Not enough data, use overall phase distribution
                subset = df[df['phase'] == phase]
            
            if len(subset) > 0:
                probs = {}
                for outcome in outcomes:
                    probs[outcome] = (subset['outcome'] == outcome).mean()
                
                distributions[phase][elo_bucket] = probs
    
    return distributions


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # Analyze outcome distributions
    print("Analyzing T20 Men's outcome distributions...")
    distributions = analyze_outcome_distributions('T20', 'male')
    
    print("\nOutcome Probabilities by Phase and ELO Matchup:")
    print("=" * 70)
    
    for phase in ['powerplay', 'middle', 'death']:
        print(f"\n{phase.upper()}")
        print("-" * 50)
        
        for elo_bucket in ['very_low', 'even', 'very_high']:
            if elo_bucket in distributions[phase]:
                probs = distributions[phase][elo_bucket]
                prob_str = " ".join([f"{k}:{v:.1%}" for k, v in probs.items()])
                print(f"  {elo_bucket:12} | {prob_str}")

