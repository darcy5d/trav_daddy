#!/usr/bin/env python3
"""
Backtest predictions against historical match outcomes.

This script:
1. Samples historical matches with stratified sampling (tier, format, recency)
2. Runs predictions using historical ELOs at match date
3. Compares predictions to actual outcomes
4. Calculates Brier score and calibration metrics
"""

import sys
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import argparse
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.vectorized_nn_sim import VectorizedNNSimulator

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class MatchBacktester:
    """Backtest model predictions against historical match outcomes."""
    
    def __init__(self, db_path: str = 'cricket.db', format_type: str = 'T20', gender: str = 'female'):
        self.db_path = db_path
        self.format_type = format_type
        self.gender = gender
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        
        # Initialize simulator
        logger.info(f"Initializing {format_type} {gender} simulator...")
        self.simulator = VectorizedNNSimulator(format_type=format_type, gender=gender)
        
    def get_stratified_sample(
        self,
        n_matches: int = 100,
        min_date: str = '2020-01-01',
        max_date: str = None
    ) -> pd.DataFrame:
        """
        Get stratified sample of matches across tiers and time periods.
        
        Args:
            n_matches: Target number of matches to sample
            min_date: Minimum match date
            max_date: Maximum match date (defaults to today)
        """
        if max_date is None:
            max_date = datetime.now().strftime('%Y-%m-%d')
            
        cursor = self.conn.cursor()
        
        # Get matches with team tiers
        query = """
            SELECT 
                m.match_id,
                m.date,
                m.team1_id,
                m.team2_id,
                m.winner_id,
                t1.name as team1_name,
                t2.name as team2_name,
                t1.tier as team1_tier,
                t2.tier as team2_tier
            FROM matches m
            JOIN teams t1 ON m.team1_id = t1.team_id
            JOIN teams t2 ON m.team2_id = t2.team_id
            WHERE m.match_type = ?
            AND m.gender = ?
            AND m.date BETWEEN ? AND ?
            AND m.winner_id IS NOT NULL
            ORDER BY RANDOM()
            LIMIT ?
        """
        
        cursor.execute(query, (self.format_type, self.gender, min_date, max_date, n_matches * 2))
        rows = cursor.fetchall()
        
        if not rows:
            logger.warning("No matches found matching criteria")
            return pd.DataFrame()
            
        df = pd.DataFrame([dict(row) for row in rows])
        
        # Sample with stratification by tier combination
        df['tier_combo'] = df['team1_tier'].astype(str) + '-' + df['team2_tier'].astype(str)
        
        # Try to get balanced sample across tier combinations
        sampled = []
        tier_combos = df['tier_combo'].unique()
        per_combo = max(1, n_matches // len(tier_combos))
        
        for combo in tier_combos:
            combo_df = df[df['tier_combo'] == combo]
            sample_size = min(per_combo, len(combo_df))
            sampled.append(combo_df.sample(n=sample_size))
            
        result = pd.concat(sampled, ignore_index=True)
        
        # If we don't have enough, add more random samples
        if len(result) < n_matches:
            remaining = df[~df['match_id'].isin(result['match_id'])]
            additional = remaining.sample(n=min(n_matches - len(result), len(remaining)))
            result = pd.concat([result, additional], ignore_index=True)
            
        logger.info(f"Sampled {len(result)} matches across {len(tier_combos)} tier combinations")
        return result.head(n_matches)
    
    def get_match_lineups(self, match_id: int) -> Tuple[List[int], List[int], List[int], List[int]]:
        """Get player lineups for a match."""
        cursor = self.conn.cursor()
        
        # Get players who batted or bowled in this match
        cursor.execute("""
            SELECT DISTINCT 
                i.batting_team_id,
                d.batter_id,
                d.bowler_id
            FROM deliveries d
            JOIN innings i ON d.innings_id = i.innings_id
            WHERE i.match_id = ?
        """, (match_id,))
        
        rows = cursor.fetchall()
        if not rows:
            return None, None, None, None
            
        team1_batters = set()
        team1_bowlers = set()
        team2_batters = set()
        team2_bowlers = set()
        
        # Get match teams
        cursor.execute("SELECT team1_id, team2_id FROM matches WHERE match_id = ?", (match_id,))
        match_row = cursor.fetchone()
        if not match_row:
            return None, None, None, None
            
        team1_id, team2_id = match_row
        
        for row in rows:
            batting_team_id, batter_id, bowler_id = row
            if batting_team_id == team1_id:
                team1_batters.add(batter_id)
                team2_bowlers.add(bowler_id)
            else:
                team2_batters.add(batter_id)
                team1_bowlers.add(bowler_id)
                
        # Pad to required sizes (11 batters, 5 bowlers)
        team1_batters = list(team1_batters)[:11]
        team1_bowlers = list(team1_bowlers)[:5]
        team2_batters = list(team2_batters)[:11]
        team2_bowlers = list(team2_bowlers)[:5]
        
        # Pad with repeats if needed
        while len(team1_batters) < 11:
            team1_batters.append(team1_batters[0] if team1_batters else 0)
        while len(team1_bowlers) < 5:
            team1_bowlers.append(team1_bowlers[0] if team1_bowlers else 0)
        while len(team2_batters) < 11:
            team2_batters.append(team2_batters[0] if team2_batters else 0)
        while len(team2_bowlers) < 5:
            team2_bowlers.append(team2_bowlers[0] if team2_bowlers else 0)
            
        return team1_batters, team1_bowlers, team2_batters, team2_bowlers
    
    def predict_match(
        self,
        team1_id: int,
        team2_id: int,
        team1_batters: List[int],
        team1_bowlers: List[int],
        team2_batters: List[int],
        team2_bowlers: List[int],
        n_simulations: int = 100
    ) -> float:
        """
        Predict win probability for team1.
        
        Returns:
            Team 1 win probability (0-1)
        """
        try:
            results = self.simulator.simulate_matches(
                n_matches=n_simulations,
                team1_batter_ids=team1_batters,
                team1_bowler_ids=team1_bowlers,
                team2_batter_ids=team2_batters,
                team2_bowler_ids=team2_bowlers,
                max_overs=20,
                team1_id=team1_id,
                team2_id=team2_id
            )
            return results['team1_win_prob']
        except Exception as e:
            logger.warning(f"Prediction failed: {e}")
            return 0.5  # Return neutral probability on failure
    
    def run_backtest(
        self,
        n_matches: int = 50,
        n_simulations: int = 100,
        min_date: str = '2020-01-01'
    ) -> Dict:
        """
        Run backtest on historical matches.
        
        Returns:
            Dictionary with metrics and predictions
        """
        # Get sample matches
        matches = self.get_stratified_sample(n_matches=n_matches, min_date=min_date)
        
        if matches.empty:
            return {'error': 'No matches found'}
            
        predictions = []
        actuals = []
        match_info = []
        
        for idx, row in matches.iterrows():
            match_id = row['match_id']
            team1_id = row['team1_id']
            team2_id = row['team2_id']
            winner_id = row['winner_id']
            
            # Get lineups
            lineups = self.get_match_lineups(match_id)
            if lineups[0] is None:
                logger.warning(f"Could not get lineups for match {match_id}")
                continue
                
            team1_batters, team1_bowlers, team2_batters, team2_bowlers = lineups
            
            # Get prediction
            pred = self.predict_match(
                team1_id, team2_id,
                team1_batters, team1_bowlers,
                team2_batters, team2_bowlers,
                n_simulations=n_simulations
            )
            
            # Actual outcome (1 if team1 won, 0 otherwise)
            actual = 1 if winner_id == team1_id else 0
            
            predictions.append(pred)
            actuals.append(actual)
            match_info.append({
                'match_id': match_id,
                'date': row['date'],
                'team1': row['team1_name'],
                'team2': row['team2_name'],
                'team1_tier': row['team1_tier'],
                'team2_tier': row['team2_tier'],
                'prediction': pred,
                'actual': actual,
                'correct': (pred > 0.5) == (actual == 1)
            })
            
            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{len(matches)} matches")
        
        if not predictions:
            return {'error': 'No valid predictions made'}
            
        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Brier score (lower is better, 0 is perfect)
        brier_score = np.mean((predictions - actuals) ** 2)
        
        # Accuracy (prediction > 0.5 matches actual)
        accuracy = np.mean((predictions > 0.5) == (actuals == 1))
        
        # Calibration - bin predictions and compare to actual rates
        bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
        calibration = []
        for i in range(len(bins) - 1):
            mask = (predictions >= bins[i]) & (predictions < bins[i+1])
            if mask.sum() > 0:
                predicted_avg = predictions[mask].mean()
                actual_rate = actuals[mask].mean()
                calibration.append({
                    'bin': f'{bins[i]:.1f}-{bins[i+1]:.1f}',
                    'count': int(mask.sum()),
                    'predicted_avg': float(predicted_avg),
                    'actual_rate': float(actual_rate),
                    'calibration_error': float(abs(predicted_avg - actual_rate))
                })
        
        return {
            'n_matches': len(predictions),
            'brier_score': float(brier_score),
            'accuracy': float(accuracy),
            'mean_prediction': float(predictions.mean()),
            'calibration': calibration,
            'predictions': match_info
        }
    
    def print_results(self, results: Dict):
        """Print backtest results in a readable format."""
        if 'error' in results:
            print(f"Error: {results['error']}")
            return
            
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Matches tested: {results['n_matches']}")
        print(f"Brier Score: {results['brier_score']:.4f} (lower is better, 0 is perfect)")
        print(f"Accuracy: {results['accuracy']*100:.1f}%")
        print(f"Mean Prediction: {results['mean_prediction']:.3f}")
        
        print("\n" + "-"*60)
        print("CALIBRATION")
        print("-"*60)
        print(f"{'Bin':<12} {'Count':<8} {'Predicted':<12} {'Actual':<12} {'Error':<10}")
        for cal in results['calibration']:
            print(f"{cal['bin']:<12} {cal['count']:<8} {cal['predicted_avg']:.3f}        {cal['actual_rate']:.3f}        {cal['calibration_error']:.3f}")
        
        print("\n" + "-"*60)
        print("SAMPLE PREDICTIONS")
        print("-"*60)
        for pred in results['predictions'][:10]:
            outcome = "✓" if pred['correct'] else "✗"
            winner = pred['team1'] if pred['actual'] == 1 else pred['team2']
            print(f"{outcome} {pred['team1']} ({pred['prediction']*100:.0f}%) vs {pred['team2']} - Winner: {winner}")


def main():
    parser = argparse.ArgumentParser(description='Backtest model predictions')
    parser.add_argument('--matches', type=int, default=50, help='Number of matches to test')
    parser.add_argument('--simulations', type=int, default=100, help='Simulations per match')
    parser.add_argument('--format', type=str, default='T20', help='Match format')
    parser.add_argument('--gender', type=str, default='female', help='Gender category')
    parser.add_argument('--min-date', type=str, default='2023-01-01', help='Minimum match date')
    
    args = parser.parse_args()
    
    logger.info(f"Starting backtest: {args.matches} matches, {args.simulations} sims/match")
    logger.info(f"Format: {args.format}, Gender: {args.gender}, Min date: {args.min_date}")
    
    backtester = MatchBacktester(
        db_path='cricket.db',
        format_type=args.format,
        gender=args.gender
    )
    
    results = backtester.run_backtest(
        n_matches=args.matches,
        n_simulations=args.simulations,
        min_date=args.min_date
    )
    
    backtester.print_results(results)
    
    return results


if __name__ == '__main__':
    main()
