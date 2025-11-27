"""
Match-Level Feature Engineering for Cricket Prediction.

Creates features for predicting match outcomes using:
- Team ELO ratings at match date
- Head-to-head records
- Recent form
- Venue factors
- Toss information
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import DATABASE_PATH
from src.data.database import get_connection

logger = logging.getLogger(__name__)


class MatchFeatureBuilder:
    """Builds match-level features for ML models."""
    
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
    
    def get_team_elo_at_date(
        self, 
        team_id: int, 
        match_date: str, 
        format_type: str, 
        gender: str
    ) -> float:
        """Get team ELO rating just before a specific match date."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT elo FROM team_elo_history
            WHERE team_id = ? AND format = ? AND gender = ? AND date < ?
            ORDER BY date DESC, elo_id DESC
            LIMIT 1
        """, (team_id, format_type, gender, match_date))
        
        row = cursor.fetchone()
        return row['elo'] if row else 1500.0
    
    def get_team_elo_momentum(
        self,
        team_id: int,
        match_date: str,
        format_type: str,
        gender: str,
        months: int = 3
    ) -> float:
        """Get ELO change over last N months (momentum/trend)."""
        cursor = self.conn.cursor()
        
        # Current ELO
        current_elo = self.get_team_elo_at_date(team_id, match_date, format_type, gender)
        
        # ELO from N months ago
        past_date = (datetime.strptime(match_date, '%Y-%m-%d') - timedelta(days=months*30)).strftime('%Y-%m-%d')
        past_elo = self.get_team_elo_at_date(team_id, past_date, format_type, gender)
        
        return current_elo - past_elo
    
    def get_head_to_head(
        self,
        team1_id: int,
        team2_id: int,
        match_date: str,
        format_type: str,
        gender: str,
        lookback_matches: int = 20
    ) -> Dict:
        """Get head-to-head record between two teams."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT winner_id
            FROM matches
            WHERE ((team1_id = ? AND team2_id = ?) OR (team1_id = ? AND team2_id = ?))
            AND match_type = ? AND gender = ? AND date < ?
            ORDER BY date DESC
            LIMIT ?
        """, (team1_id, team2_id, team2_id, team1_id, format_type, gender, match_date, lookback_matches))
        
        rows = cursor.fetchall()
        
        if not rows:
            return {'team1_wins': 0, 'team2_wins': 0, 'total': 0, 'team1_win_rate': 0.5}
        
        team1_wins = sum(1 for r in rows if r['winner_id'] == team1_id)
        team2_wins = sum(1 for r in rows if r['winner_id'] == team2_id)
        total = len(rows)
        
        return {
            'team1_wins': team1_wins,
            'team2_wins': team2_wins,
            'total': total,
            'team1_win_rate': team1_wins / total if total > 0 else 0.5
        }
    
    def get_recent_form(
        self,
        team_id: int,
        match_date: str,
        format_type: str,
        gender: str,
        last_n_matches: int = 5
    ) -> float:
        """Get team's win rate in last N matches."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT winner_id
            FROM matches
            WHERE (team1_id = ? OR team2_id = ?)
            AND match_type = ? AND gender = ? AND date < ?
            AND winner_id IS NOT NULL
            ORDER BY date DESC
            LIMIT ?
        """, (team_id, team_id, format_type, gender, match_date, last_n_matches))
        
        rows = cursor.fetchall()
        
        if not rows:
            return 0.5
        
        wins = sum(1 for r in rows if r['winner_id'] == team_id)
        return wins / len(rows)
    
    def get_venue_stats(
        self,
        team_id: int,
        venue_id: int,
        match_date: str,
        format_type: str,
        gender: str
    ) -> Dict:
        """Get team's performance at specific venue."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT winner_id
            FROM matches
            WHERE (team1_id = ? OR team2_id = ?)
            AND venue_id = ? AND match_type = ? AND gender = ? AND date < ?
            AND winner_id IS NOT NULL
        """, (team_id, team_id, venue_id, format_type, gender, match_date))
        
        rows = cursor.fetchall()
        
        if not rows:
            return {'matches_at_venue': 0, 'win_rate_at_venue': 0.5}
        
        wins = sum(1 for r in rows if r['winner_id'] == team_id)
        return {
            'matches_at_venue': len(rows),
            'win_rate_at_venue': wins / len(rows)
        }
    
    def get_venue_characteristics(
        self,
        venue_id: int,
        match_date: str,
        format_type: str,
        gender: str
    ) -> Dict:
        """Get venue scoring patterns and characteristics."""
        cursor = self.conn.cursor()
        
        # Get historical scores at this venue
        cursor.execute("""
            SELECT 
                AVG(i.total_runs) as avg_score,
                AVG(i.total_wickets) as avg_wickets,
                AVG(CASE WHEN i.innings_number = 1 THEN i.total_runs END) as avg_1st_innings,
                COUNT(DISTINCT m.match_id) as venue_matches
            FROM innings i
            JOIN matches m ON i.match_id = m.match_id
            WHERE m.venue_id = ? AND m.match_type = ? AND m.gender = ? AND m.date < ?
        """, (venue_id, format_type, gender, match_date))
        
        row = cursor.fetchone()
        
        # Get chase success rate at venue
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN win_type = 'wickets' THEN 1 ELSE 0 END) as chase_wins
            FROM matches
            WHERE venue_id = ? AND match_type = ? AND gender = ? AND date < ?
            AND winner_id IS NOT NULL
        """, (venue_id, format_type, gender, match_date))
        
        chase_row = cursor.fetchone()
        
        # Default values if no data
        avg_score = row['avg_score'] if row and row['avg_score'] else 155.0  # T20 average
        avg_wickets = row['avg_wickets'] if row and row['avg_wickets'] else 6.5
        avg_1st = row['avg_1st_innings'] if row and row['avg_1st_innings'] else 160.0
        venue_matches = row['venue_matches'] if row and row['venue_matches'] else 0
        
        chase_rate = 0.5
        if chase_row and chase_row['total'] and chase_row['total'] > 0:
            chase_rate = chase_row['chase_wins'] / chase_row['total']
        
        return {
            'venue_avg_score': avg_score,
            'venue_avg_wickets': avg_wickets,
            'venue_avg_1st_innings': avg_1st,
            'venue_matches_total': venue_matches,
            'venue_chase_win_rate': chase_rate,
            'venue_is_high_scoring': 1 if avg_score > 160 else 0,
            'venue_is_low_scoring': 1 if avg_score < 140 else 0
        }
    
    def is_home_game(
        self,
        team_name: str,
        venue_city: Optional[str]
    ) -> bool:
        """Determine if team is playing at home based on venue city."""
        if not venue_city:
            return False
        
        home_cities = {
            'India': ['Mumbai', 'Delhi', 'Kolkata', 'Chennai', 'Bangalore', 'Hyderabad', 
                     'Ahmedabad', 'Rajkot', 'Mohali', 'Lucknow', 'Guwahati', 'Indore', 
                     'Nagpur', 'Visakhapatnam', 'Dharamsala', 'Thiruvananthapuram'],
            'Australia': ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide', 
                         'Hobart', 'Canberra', 'Cairns'],
            'England': ['London', 'Manchester', 'Birmingham', 'Leeds', 'Southampton', 
                       'Bristol', 'Cardiff', 'Nottingham', 'Chester-le-Street'],
            'South Africa': ['Johannesburg', 'Cape Town', 'Durban', 'Centurion', 
                            'Port Elizabeth', 'Bloemfontein', 'Paarl', 'Benoni'],
            'Pakistan': ['Lahore', 'Karachi', 'Rawalpindi', 'Multan', 'Faisalabad'],
            'New Zealand': ['Auckland', 'Wellington', 'Christchurch', 'Hamilton', 
                           'Napier', 'Mount Maunganui'],
            'West Indies': ['Bridgetown', 'Kingston', 'Port of Spain', "St George's", 
                           'Basseterre', 'Gros Islet', 'North Sound'],
            'Sri Lanka': ['Colombo', 'Kandy', 'Pallekele', 'Hambantota', 'Dambulla'],
            'Bangladesh': ['Dhaka', 'Chittagong', 'Sylhet', 'Mirpur'],
            'Zimbabwe': ['Harare', 'Bulawayo'],
            'Afghanistan': ['Kabul', 'Sharjah'],  # Often play in UAE
            'Ireland': ['Dublin', 'Belfast', 'Malahide'],
        }
        
        team_cities = home_cities.get(team_name, [])
        return venue_city in team_cities
    
    def get_team_batting_strength(
        self,
        team_id: int,
        match_id: int,
        format_type: str,
        gender: str,
        match_date: str
    ) -> Dict:
        """Get aggregated batting ELOs for team's players in a match."""
        cursor = self.conn.cursor()
        
        # Get players who played for this team in this match
        bat_col = f'batting_elo_{format_type.lower()}_{gender}'
        
        cursor.execute(f"""
            SELECT pms.player_id, COALESCE(pce.{bat_col}, 1500) as batting_elo
            FROM player_match_stats pms
            LEFT JOIN player_current_elo pce ON pms.player_id = pce.player_id
            WHERE pms.match_id = ? AND pms.team_id = ?
        """, (match_id, team_id))
        
        rows = cursor.fetchall()
        
        if not rows:
            return {
                'batting_elo_sum': 1500 * 11,
                'batting_elo_avg': 1500,
                'batting_elo_max': 1500
            }
        
        elos = [r['batting_elo'] for r in rows]
        
        return {
            'batting_elo_sum': sum(elos),
            'batting_elo_avg': np.mean(elos),
            'batting_elo_max': max(elos)
        }
    
    def get_team_bowling_strength(
        self,
        team_id: int,
        match_id: int,
        format_type: str,
        gender: str,
        match_date: str
    ) -> Dict:
        """Get aggregated bowling ELOs for team's players in a match."""
        cursor = self.conn.cursor()
        
        bowl_col = f'bowling_elo_{format_type.lower()}_{gender}'
        
        cursor.execute(f"""
            SELECT pms.player_id, COALESCE(pce.{bowl_col}, 1500) as bowling_elo
            FROM player_match_stats pms
            LEFT JOIN player_current_elo pce ON pms.player_id = pce.player_id
            WHERE pms.match_id = ? AND pms.team_id = ?
            AND pms.overs_bowled > 0
        """, (match_id, team_id))
        
        rows = cursor.fetchall()
        
        if not rows:
            return {
                'bowling_elo_sum': 1500 * 5,
                'bowling_elo_avg': 1500,
                'bowling_elo_max': 1500
            }
        
        elos = [r['bowling_elo'] for r in rows]
        
        return {
            'bowling_elo_sum': sum(elos),
            'bowling_elo_avg': np.mean(elos),
            'bowling_elo_max': max(elos)
        }
    
    def build_features_for_match(self, match_id: int) -> Optional[Dict]:
        """Build all features for a single match."""
        cursor = self.conn.cursor()
        
        # Get match info
        cursor.execute("""
            SELECT m.*, t1.name as team1_name, t2.name as team2_name
            FROM matches m
            JOIN teams t1 ON m.team1_id = t1.team_id
            JOIN teams t2 ON m.team2_id = t2.team_id
            WHERE m.match_id = ?
        """, (match_id,))
        
        match = cursor.fetchone()
        if not match:
            return None
        
        match_date = match['date']
        format_type = match['match_type']
        gender = match['gender']
        team1_id = match['team1_id']
        team2_id = match['team2_id']
        
        # Team ELOs at match date
        team1_elo = self.get_team_elo_at_date(team1_id, match_date, format_type, gender)
        team2_elo = self.get_team_elo_at_date(team2_id, match_date, format_type, gender)
        
        # ELO momentum
        team1_momentum = self.get_team_elo_momentum(team1_id, match_date, format_type, gender)
        team2_momentum = self.get_team_elo_momentum(team2_id, match_date, format_type, gender)
        
        # Head to head
        h2h = self.get_head_to_head(team1_id, team2_id, match_date, format_type, gender)
        
        # Recent form
        team1_form = self.get_recent_form(team1_id, match_date, format_type, gender)
        team2_form = self.get_recent_form(team2_id, match_date, format_type, gender)
        
        # Venue stats
        venue_id = match['venue_id']
        team1_venue = self.get_venue_stats(team1_id, venue_id, match_date, format_type, gender) if venue_id else {'matches_at_venue': 0, 'win_rate_at_venue': 0.5}
        team2_venue = self.get_venue_stats(team2_id, venue_id, match_date, format_type, gender) if venue_id else {'matches_at_venue': 0, 'win_rate_at_venue': 0.5}
        
        # Venue characteristics (scoring patterns)
        venue_chars = self.get_venue_characteristics(venue_id, match_date, format_type, gender) if venue_id else {
            'venue_avg_score': 155.0, 'venue_avg_wickets': 6.5, 'venue_avg_1st_innings': 160.0,
            'venue_matches_total': 0, 'venue_chase_win_rate': 0.5,
            'venue_is_high_scoring': 0, 'venue_is_low_scoring': 0
        }
        
        # Get venue city for home game detection
        cursor.execute("SELECT city FROM venues WHERE venue_id = ?", (venue_id,))
        venue_row = cursor.fetchone()
        venue_city = venue_row['city'] if venue_row else None
        
        # Home game flags
        is_team1_home = self.is_home_game(match['team1_name'], venue_city)
        is_team2_home = self.is_home_game(match['team2_name'], venue_city)
        
        # Team composition (batting/bowling strength)
        team1_batting = self.get_team_batting_strength(team1_id, match_id, format_type, gender, match_date)
        team2_batting = self.get_team_batting_strength(team2_id, match_id, format_type, gender, match_date)
        team1_bowling = self.get_team_bowling_strength(team1_id, match_id, format_type, gender, match_date)
        team2_bowling = self.get_team_bowling_strength(team2_id, match_id, format_type, gender, match_date)
        
        # Toss features
        toss_winner_is_team1 = 1 if match['toss_winner_id'] == team1_id else 0
        chose_to_bat = 1 if match['toss_decision'] == 'bat' else 0
        
        # Target variable
        if match['winner_id'] is None:
            target = None  # No result
        else:
            target = 1 if match['winner_id'] == team1_id else 0
        
        return {
            # Identifiers
            'match_id': match_id,
            'date': match_date,
            'format': format_type,
            'gender': gender,
            'team1_name': match['team1_name'],
            'team2_name': match['team2_name'],
            
            # Team ELO features
            'team1_elo': team1_elo,
            'team2_elo': team2_elo,
            'elo_diff': team1_elo - team2_elo,
            'elo_diff_abs': abs(team1_elo - team2_elo),
            
            # ELO momentum
            'team1_momentum': team1_momentum,
            'team2_momentum': team2_momentum,
            'momentum_diff': team1_momentum - team2_momentum,
            
            # Head to head
            'h2h_team1_wins': h2h['team1_wins'],
            'h2h_team2_wins': h2h['team2_wins'],
            'h2h_total': h2h['total'],
            'h2h_team1_win_rate': h2h['team1_win_rate'],
            
            # Recent form
            'team1_recent_form': team1_form,
            'team2_recent_form': team2_form,
            'form_diff': team1_form - team2_form,
            
            # Team venue stats
            'team1_venue_matches': team1_venue['matches_at_venue'],
            'team2_venue_matches': team2_venue['matches_at_venue'],
            'team1_venue_win_rate': team1_venue['win_rate_at_venue'],
            'team2_venue_win_rate': team2_venue['win_rate_at_venue'],
            
            # Venue characteristics (NEW)
            'venue_avg_score': venue_chars['venue_avg_score'],
            'venue_avg_wickets': venue_chars['venue_avg_wickets'],
            'venue_chase_win_rate': venue_chars['venue_chase_win_rate'],
            'venue_is_high_scoring': venue_chars['venue_is_high_scoring'],
            'venue_is_low_scoring': venue_chars['venue_is_low_scoring'],
            
            # Home advantage (NEW)
            'is_team1_home': 1 if is_team1_home else 0,
            'is_team2_home': 1 if is_team2_home else 0,
            'home_advantage_team1': 1 if is_team1_home and not is_team2_home else 0,
            'home_advantage_team2': 1 if is_team2_home and not is_team1_home else 0,
            'is_neutral_venue': 1 if not is_team1_home and not is_team2_home else 0,
            
            # Team composition - batting
            'team1_batting_elo_avg': team1_batting['batting_elo_avg'],
            'team2_batting_elo_avg': team2_batting['batting_elo_avg'],
            'team1_batting_elo_max': team1_batting['batting_elo_max'],
            'team2_batting_elo_max': team2_batting['batting_elo_max'],
            
            # Team composition - bowling
            'team1_bowling_elo_avg': team1_bowling['bowling_elo_avg'],
            'team2_bowling_elo_avg': team2_bowling['bowling_elo_avg'],
            'team1_bowling_elo_max': team1_bowling['bowling_elo_max'],
            'team2_bowling_elo_max': team2_bowling['bowling_elo_max'],
            
            # Toss
            'toss_winner_is_team1': toss_winner_is_team1,
            'chose_to_bat': chose_to_bat,
            
            # Target
            'team1_won': target
        }
    
    def build_dataset(
        self,
        format_type: Optional[str] = None,
        gender: Optional[str] = None,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Build feature dataset for all matches matching criteria."""
        cursor = self.conn.cursor()
        
        query = "SELECT match_id FROM matches WHERE winner_id IS NOT NULL"
        params = []
        
        if format_type:
            query += " AND match_type = ?"
            params.append(format_type)
        
        if gender:
            query += " AND gender = ?"
            params.append(gender)
        
        if min_date:
            query += " AND date >= ?"
            params.append(min_date)
        
        if max_date:
            query += " AND date <= ?"
            params.append(max_date)
        
        query += " ORDER BY date"
        
        cursor.execute(query, params)
        match_ids = [row['match_id'] for row in cursor.fetchall()]
        
        logger.info(f"Building features for {len(match_ids)} matches...")
        
        features = []
        for i, match_id in enumerate(match_ids):
            if i % 100 == 0:
                logger.info(f"Processing match {i+1}/{len(match_ids)}")
            
            match_features = self.build_features_for_match(match_id)
            if match_features and match_features['team1_won'] is not None:
                features.append(match_features)
        
        df = pd.DataFrame(features)
        logger.info(f"Built dataset with {len(df)} matches and {len(df.columns)} features")
        
        return df


def build_training_dataset(
    format_type: str = 'T20',
    gender: str = 'male',
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """Convenience function to build and optionally save training dataset."""
    with MatchFeatureBuilder() as builder:
        df = builder.build_dataset(format_type=format_type, gender=gender)
    
    if output_path:
        df.to_csv(output_path, index=False)
        logger.info(f"Saved dataset to {output_path}")
    
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # Build T20 Men's dataset
    df = build_training_dataset(format_type='T20', gender='male')
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFeature columns:\n{df.columns.tolist()}")
    print(f"\nSample row:\n{df.iloc[0]}")

