"""
Feature Engineering Module for Cricket Match Prediction.

Creates features for match outcome prediction including:
- Team features (ELO, recent form, head-to-head)
- Player features (batting/bowling ELO, career stats)
- Match context features (venue, toss, conditions)
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import FEATURE_CONFIG, ELO_CONFIG
from src.data.database import get_db_connection
from src.elo.calculator import EloCalculator

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering for cricket match prediction.
    
    Creates comprehensive feature sets for training and prediction.
    """
    
    def __init__(self):
        self.elo_calc = EloCalculator()
        self.recent_form_matches = FEATURE_CONFIG['recent_form_matches']
        self.min_matches = FEATURE_CONFIG['min_matches_for_stats']
    
    def get_team_features(
        self,
        conn,
        team_id: int,
        match_format: str,
        as_of_date: datetime,
        opponent_id: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Get features for a team.
        
        Args:
            conn: Database connection
            team_id: Team ID
            match_format: 'T20' or 'ODI'
            as_of_date: Date to calculate features for
            opponent_id: Optional opponent for head-to-head features
            
        Returns:
            Dictionary of team features
        """
        cursor = conn.cursor()
        features = {}
        
        # ELO rating
        elo = self.elo_calc.get_team_rating(conn, team_id, match_format, as_of_date)
        features['team_elo'] = elo
        
        # ELO trajectory (change over last 3 months)
        three_months_ago = as_of_date - timedelta(days=90)
        old_elo = self.elo_calc.get_team_rating(conn, team_id, match_format, three_months_ago)
        features['team_elo_trend'] = elo - old_elo
        
        # Recent form (win rate in last N matches)
        cursor.execute("""
            SELECT 
                COUNT(*) as matches,
                SUM(CASE WHEN winner_id = ? THEN 1 ELSE 0 END) as wins
            FROM matches
            WHERE (team1_id = ? OR team2_id = ?)
            AND match_type = ?
            AND date < ?
            ORDER BY date DESC
            LIMIT ?
        """, (team_id, team_id, team_id, match_format, as_of_date, self.recent_form_matches))
        
        row = cursor.fetchone()
        if row and row['matches'] > 0:
            features['team_recent_win_rate'] = row['wins'] / row['matches']
            features['team_recent_matches'] = row['matches']
        else:
            features['team_recent_win_rate'] = 0.5
            features['team_recent_matches'] = 0
        
        # Head-to-head record against opponent
        if opponent_id:
            cursor.execute("""
                SELECT 
                    COUNT(*) as matches,
                    SUM(CASE WHEN winner_id = ? THEN 1 ELSE 0 END) as wins
                FROM matches
                WHERE ((team1_id = ? AND team2_id = ?) OR (team1_id = ? AND team2_id = ?))
                AND match_type = ?
                AND date < ?
            """, (team_id, team_id, opponent_id, opponent_id, team_id, match_format, as_of_date))
            
            row = cursor.fetchone()
            if row and row['matches'] > 0:
                features['h2h_win_rate'] = row['wins'] / row['matches']
                features['h2h_matches'] = row['matches']
            else:
                features['h2h_win_rate'] = 0.5
                features['h2h_matches'] = 0
        
        # Average score in recent matches
        cursor.execute("""
            SELECT AVG(i.total_runs) as avg_score
            FROM innings i
            JOIN matches m ON i.match_id = m.match_id
            WHERE i.batting_team_id = ?
            AND m.match_type = ?
            AND m.date < ?
            ORDER BY m.date DESC
            LIMIT ?
        """, (team_id, match_format, as_of_date, self.recent_form_matches))
        
        row = cursor.fetchone()
        features['team_avg_score'] = row['avg_score'] if row and row['avg_score'] else (
            150 if match_format == 'T20' else 250
        )
        
        return features
    
    def get_player_features(
        self,
        conn,
        player_id: int,
        match_format: str,
        as_of_date: datetime
    ) -> Dict[str, float]:
        """
        Get features for a player.
        
        Args:
            conn: Database connection
            player_id: Player ID
            match_format: 'T20' or 'ODI'
            as_of_date: Date to calculate features for
            
        Returns:
            Dictionary of player features
        """
        cursor = conn.cursor()
        features = {}
        
        # ELO ratings
        features['batting_elo'] = self.elo_calc.get_player_rating(
            conn, player_id, match_format, 'batting', as_of_date
        )
        features['bowling_elo'] = self.elo_calc.get_player_rating(
            conn, player_id, match_format, 'bowling', as_of_date
        )
        features['overall_elo'] = self.elo_calc.get_player_rating(
            conn, player_id, match_format, 'overall', as_of_date
        )
        
        # Career stats
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT pms.match_id) as matches,
                SUM(pms.runs_scored) as total_runs,
                SUM(pms.balls_faced) as total_balls,
                SUM(pms.batting_innings) as innings,
                SUM(CASE WHEN pms.not_out THEN 0 ELSE 1 END) as outs,
                SUM(pms.fours_hit) as fours,
                SUM(pms.sixes_hit) as sixes,
                SUM(pms.wickets_taken) as wickets,
                SUM(pms.runs_conceded) as runs_conceded,
                SUM(pms.overs_bowled) as overs_bowled
            FROM player_match_stats pms
            JOIN matches m ON pms.match_id = m.match_id
            WHERE pms.player_id = ?
            AND m.match_type = ?
            AND m.date < ?
        """, (player_id, match_format, as_of_date))
        
        row = cursor.fetchone()
        
        if row and row['matches'] and row['matches'] >= self.min_matches:
            # Batting stats
            if row['innings'] and row['innings'] > 0:
                outs = row['outs'] if row['outs'] else row['innings']
                features['batting_average'] = row['total_runs'] / max(outs, 1)
                features['strike_rate'] = (row['total_runs'] / max(row['total_balls'], 1)) * 100
                features['boundary_rate'] = (row['fours'] + row['sixes']) / max(row['total_balls'], 1)
            else:
                features['batting_average'] = 0
                features['strike_rate'] = 0
                features['boundary_rate'] = 0
            
            # Bowling stats
            if row['overs_bowled'] and row['overs_bowled'] > 0:
                features['bowling_average'] = row['runs_conceded'] / max(row['wickets'], 1)
                features['economy'] = row['runs_conceded'] / row['overs_bowled']
                features['bowling_strike_rate'] = (row['overs_bowled'] * 6) / max(row['wickets'], 1)
            else:
                features['bowling_average'] = 50
                features['economy'] = 10
                features['bowling_strike_rate'] = 100
            
            features['matches_played'] = row['matches']
        else:
            # Default values for new/unknown players
            features['batting_average'] = 20
            features['strike_rate'] = 100 if match_format == 'T20' else 75
            features['boundary_rate'] = 0.1
            features['bowling_average'] = 35
            features['economy'] = 8 if match_format == 'T20' else 5
            features['bowling_strike_rate'] = 30
            features['matches_played'] = 0
        
        # Recent form (last 5 matches)
        cursor.execute("""
            SELECT 
                AVG(pms.runs_scored) as avg_runs,
                AVG(CASE WHEN pms.balls_faced > 0 
                    THEN pms.runs_scored * 100.0 / pms.balls_faced 
                    ELSE 0 END) as avg_sr
            FROM player_match_stats pms
            JOIN matches m ON pms.match_id = m.match_id
            WHERE pms.player_id = ?
            AND m.match_type = ?
            AND m.date < ?
            ORDER BY m.date DESC
            LIMIT 5
        """, (player_id, match_format, as_of_date))
        
        row = cursor.fetchone()
        if row and row['avg_runs']:
            features['recent_avg_runs'] = row['avg_runs']
            features['recent_avg_sr'] = row['avg_sr'] or 0
        else:
            features['recent_avg_runs'] = features['batting_average']
            features['recent_avg_sr'] = features['strike_rate']
        
        return features
    
    def get_team_composition_features(
        self,
        conn,
        player_ids: List[int],
        match_format: str,
        as_of_date: datetime
    ) -> Dict[str, float]:
        """
        Get aggregated features for a team's playing XI.
        
        Args:
            conn: Database connection
            player_ids: List of 11 player IDs
            match_format: 'T20' or 'ODI'
            as_of_date: Date to calculate features for
            
        Returns:
            Dictionary of team composition features
        """
        features = {}
        
        batting_elos = []
        bowling_elos = []
        overall_elos = []
        experiences = []
        
        for player_id in player_ids:
            player_features = self.get_player_features(
                conn, player_id, match_format, as_of_date
            )
            batting_elos.append(player_features['batting_elo'])
            bowling_elos.append(player_features['bowling_elo'])
            overall_elos.append(player_features['overall_elo'])
            experiences.append(player_features['matches_played'])
        
        # Aggregate batting strength
        features['team_batting_elo_sum'] = sum(batting_elos)
        features['team_batting_elo_avg'] = np.mean(batting_elos)
        features['team_batting_elo_max'] = max(batting_elos)
        
        # Top order (positions 1-4) vs lower order
        features['top_order_batting_elo'] = np.mean(batting_elos[:4]) if len(batting_elos) >= 4 else np.mean(batting_elos)
        features['lower_order_batting_elo'] = np.mean(batting_elos[7:]) if len(batting_elos) >= 8 else np.mean(batting_elos[4:])
        
        # Bowling strength
        features['team_bowling_elo_sum'] = sum(bowling_elos)
        features['team_bowling_elo_avg'] = np.mean(bowling_elos)
        features['team_bowling_elo_max'] = max(bowling_elos)
        
        # Overall team strength
        features['team_overall_elo_avg'] = np.mean(overall_elos)
        
        # Experience
        features['team_experience_sum'] = sum(experiences)
        features['team_experience_avg'] = np.mean(experiences)
        
        # All-rounder depth (players with both ELOs above threshold)
        threshold = ELO_CONFIG['initial_rating']
        all_rounders = sum(
            1 for bat, bowl in zip(batting_elos, bowling_elos)
            if bat > threshold and bowl > threshold
        )
        features['all_rounder_count'] = all_rounders
        
        return features
    
    def get_venue_features(
        self,
        conn,
        venue_id: int,
        match_format: str,
        as_of_date: datetime
    ) -> Dict[str, float]:
        """
        Get features for a venue.
        
        Args:
            conn: Database connection
            venue_id: Venue ID
            match_format: 'T20' or 'ODI'
            as_of_date: Date to calculate features for
            
        Returns:
            Dictionary of venue features
        """
        cursor = conn.cursor()
        features = {}
        
        # Average first innings score at venue
        cursor.execute("""
            SELECT 
                AVG(i.total_runs) as avg_first_innings,
                COUNT(DISTINCT m.match_id) as matches
            FROM innings i
            JOIN matches m ON i.match_id = m.match_id
            WHERE m.venue_id = ?
            AND m.match_type = ?
            AND i.innings_number = 1
            AND m.date < ?
        """, (venue_id, match_format, as_of_date))
        
        row = cursor.fetchone()
        if row and row['matches'] and row['matches'] >= 3:
            features['venue_avg_first_score'] = row['avg_first_innings']
            features['venue_matches'] = row['matches']
        else:
            features['venue_avg_first_score'] = 160 if match_format == 'T20' else 270
            features['venue_matches'] = 0
        
        # First batting win rate at venue
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN m.winner_id = i.batting_team_id THEN 1 ELSE 0 END) as first_bat_wins
            FROM matches m
            JOIN innings i ON m.match_id = i.match_id AND i.innings_number = 1
            WHERE m.venue_id = ?
            AND m.match_type = ?
            AND m.winner_id IS NOT NULL
            AND m.date < ?
        """, (venue_id, match_format, as_of_date))
        
        row = cursor.fetchone()
        if row and row['total'] and row['total'] >= 3:
            features['venue_first_bat_win_rate'] = row['first_bat_wins'] / row['total']
        else:
            features['venue_first_bat_win_rate'] = 0.45  # Slight advantage to chasing
        
        return features
    
    def create_match_features(
        self,
        conn,
        team1_id: int,
        team2_id: int,
        venue_id: int,
        match_format: str,
        match_date: datetime,
        toss_winner_id: Optional[int] = None,
        toss_decision: Optional[str] = None,
        team1_players: Optional[List[int]] = None,
        team2_players: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        Create full feature set for a match prediction.
        
        Args:
            conn: Database connection
            team1_id: First team ID
            team2_id: Second team ID
            venue_id: Venue ID
            match_format: 'T20' or 'ODI'
            match_date: Date of match
            toss_winner_id: ID of toss winner
            toss_decision: 'bat' or 'field'
            team1_players: Optional list of player IDs for team 1
            team2_players: Optional list of player IDs for team 2
            
        Returns:
            Dictionary of all features
        """
        features = {}
        
        # Team 1 features
        team1_features = self.get_team_features(
            conn, team1_id, match_format, match_date, team2_id
        )
        for key, value in team1_features.items():
            features[f'team1_{key}'] = value
        
        # Team 2 features
        team2_features = self.get_team_features(
            conn, team2_id, match_format, match_date, team1_id
        )
        for key, value in team2_features.items():
            features[f'team2_{key}'] = value
        
        # Differential features
        features['elo_diff'] = team1_features['team_elo'] - team2_features['team_elo']
        features['form_diff'] = team1_features['team_recent_win_rate'] - team2_features['team_recent_win_rate']
        
        # Team composition features (if player lists provided)
        if team1_players and len(team1_players) == 11:
            team1_comp = self.get_team_composition_features(
                conn, team1_players, match_format, match_date
            )
            for key, value in team1_comp.items():
                features[f'team1_{key}'] = value
        
        if team2_players and len(team2_players) == 11:
            team2_comp = self.get_team_composition_features(
                conn, team2_players, match_format, match_date
            )
            for key, value in team2_comp.items():
                features[f'team2_{key}'] = value
            
            # Composition differentials
            if team1_players:
                features['batting_elo_diff'] = (
                    features['team1_team_batting_elo_avg'] - 
                    features['team2_team_batting_elo_avg']
                )
                features['bowling_elo_diff'] = (
                    features['team1_team_bowling_elo_avg'] - 
                    features['team2_team_bowling_elo_avg']
                )
        
        # Venue features
        venue_features = self.get_venue_features(
            conn, venue_id, match_format, match_date
        )
        for key, value in venue_features.items():
            features[key] = value
        
        # Toss features
        if toss_winner_id:
            features['toss_winner_is_team1'] = 1 if toss_winner_id == team1_id else 0
            features['toss_elected_bat'] = 1 if toss_decision == 'bat' else 0
            features['team1_batting_first'] = 1 if (
                (toss_winner_id == team1_id and toss_decision == 'bat') or
                (toss_winner_id == team2_id and toss_decision == 'field')
            ) else 0
        
        # Match format indicator
        features['is_t20'] = 1 if match_format == 'T20' else 0
        
        return features


def create_training_dataset(
    match_format: str = 'T20',
    min_date: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create training dataset from historical matches.
    
    Args:
        match_format: 'T20' or 'ODI'
        min_date: Minimum date for matches (YYYY-MM-DD)
        
    Returns:
        Tuple of (features_array, labels_array, feature_names)
    """
    from tqdm import tqdm
    
    engineer = FeatureEngineer()
    
    features_list = []
    labels = []
    feature_names = None
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Get matches with results
        query = """
            SELECT 
                match_id, team1_id, team2_id, venue_id, date,
                toss_winner_id, toss_decision, winner_id
            FROM matches
            WHERE match_type = ?
            AND winner_id IS NOT NULL
        """
        params = [match_format]
        
        if min_date:
            query += " AND date >= ?"
            params.append(min_date)
        
        query += " ORDER BY date"
        
        cursor.execute(query, params)
        matches = cursor.fetchall()
        
        logger.info(f"Processing {len(matches)} matches for feature extraction...")
        
        for match in tqdm(matches, desc="Extracting features"):
            try:
                match_date = match['date']
                if isinstance(match_date, str):
                    match_date = datetime.strptime(match_date, '%Y-%m-%d')
                
                features = engineer.create_match_features(
                    conn,
                    team1_id=match['team1_id'],
                    team2_id=match['team2_id'],
                    venue_id=match['venue_id'],
                    match_format=match_format,
                    match_date=match_date,
                    toss_winner_id=match['toss_winner_id'],
                    toss_decision=match['toss_decision']
                )
                
                if feature_names is None:
                    feature_names = list(features.keys())
                
                features_list.append([features[name] for name in feature_names])
                
                # Label: 1 if team1 won, 0 if team2 won
                labels.append(1 if match['winner_id'] == match['team1_id'] else 0)
                
            except Exception as e:
                logger.warning(f"Error processing match {match['match_id']}: {e}")
                continue
    
    X = np.array(features_list)
    y = np.array(labels)
    
    logger.info(f"Created dataset with {len(X)} samples and {len(feature_names)} features")
    
    return X, y, feature_names


def main():
    """Test feature engineering."""
    logging.basicConfig(level=logging.INFO)
    
    print("Creating training dataset...")
    X, y, feature_names = create_training_dataset(match_format='T20')
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Label distribution: {np.bincount(y)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

