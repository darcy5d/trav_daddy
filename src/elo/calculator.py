"""
ELO Rating Calculator for Cricket Teams and Players.

Implements a comprehensive ELO rating system that tracks:
- Team ELO ratings (separate for T20 and ODI)
- Player batting ELO
- Player bowling ELO
- Player overall ELO

Ratings are updated after each match and monthly snapshots are stored.
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
import math

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import ELO_CONFIG, DATABASE_PATH
from src.data.database import get_db_connection, get_connection

logger = logging.getLogger(__name__)


@dataclass
class EloRating:
    """Represents an ELO rating with metadata."""
    rating: float
    matches_played: int = 0
    last_match_date: Optional[datetime] = None


class EloCalculator:
    """
    ELO Rating Calculator for cricket matches.
    
    Uses the standard ELO formula with adjustments for cricket-specific scenarios.
    """
    
    def __init__(self):
        self.initial_rating = ELO_CONFIG['initial_rating']
        self.k_factor_team = ELO_CONFIG['k_factor_team']
        self.k_factor_batting = ELO_CONFIG['k_factor_player_batting']
        self.k_factor_bowling = ELO_CONFIG['k_factor_player_bowling']
        self.rating_floor = ELO_CONFIG['rating_floor']
        self.rating_ceiling = ELO_CONFIG['rating_ceiling']
        
        # Cache for current ratings
        self._team_ratings: Dict[int, Dict[str, float]] = {}
        self._player_ratings: Dict[int, Dict[str, Dict[str, float]]] = {}
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected score for player/team A against B.
        
        Uses the standard ELO expected score formula:
        E_A = 1 / (1 + 10^((R_B - R_A) / 400))
        
        Args:
            rating_a: Rating of player/team A
            rating_b: Rating of player/team B
            
        Returns:
            Expected score (probability of winning) for A
        """
        exponent = (rating_b - rating_a) / 400.0
        return 1.0 / (1.0 + math.pow(10, exponent))
    
    def calculate_new_rating(
        self,
        old_rating: float,
        expected: float,
        actual: float,
        k_factor: float
    ) -> float:
        """
        Calculate new rating after a match/event.
        
        R_new = R_old + K * (S - E)
        
        Args:
            old_rating: Current rating
            expected: Expected score (0-1)
            actual: Actual score (0-1)
            k_factor: K-factor for this calculation
            
        Returns:
            New rating (bounded by floor and ceiling)
        """
        new_rating = old_rating + k_factor * (actual - expected)
        return max(self.rating_floor, min(self.rating_ceiling, new_rating))
    
    def get_team_rating(
        self,
        conn,
        team_id: int,
        match_format: str,
        as_of_date: Optional[datetime] = None
    ) -> float:
        """
        Get team rating at a specific point in time.
        
        Args:
            conn: Database connection
            team_id: Team ID
            match_format: 'T20' or 'ODI'
            as_of_date: Date to get rating for (latest if None)
            
        Returns:
            ELO rating
        """
        cursor = conn.cursor()
        
        elo_column = 'elo_t20' if match_format == 'T20' else 'elo_odi'
        
        if as_of_date:
            cursor.execute(f"""
                SELECT {elo_column} FROM team_elo_history
                WHERE team_id = ? AND date <= ?
                ORDER BY date DESC, elo_id DESC
                LIMIT 1
            """, (team_id, as_of_date))
        else:
            cursor.execute(f"""
                SELECT {elo_column} FROM team_current_elo
                WHERE team_id = ?
            """, (team_id,))
        
        row = cursor.fetchone()
        return row[0] if row else self.initial_rating
    
    def get_player_rating(
        self,
        conn,
        player_id: int,
        match_format: str,
        rating_type: str = 'overall',
        as_of_date: Optional[datetime] = None
    ) -> float:
        """
        Get player rating at a specific point in time.
        
        Args:
            conn: Database connection
            player_id: Player ID
            match_format: 'T20' or 'ODI'
            rating_type: 'batting', 'bowling', or 'overall'
            as_of_date: Date to get rating for (latest if None)
            
        Returns:
            ELO rating
        """
        cursor = conn.cursor()
        
        elo_column = f'{rating_type}_elo'
        
        if as_of_date:
            cursor.execute(f"""
                SELECT {elo_column} FROM player_elo_history
                WHERE player_id = ? AND format = ? AND date <= ?
                ORDER BY date DESC, elo_id DESC
                LIMIT 1
            """, (player_id, match_format, as_of_date))
        else:
            format_suffix = '_t20' if match_format == 'T20' else '_odi'
            column = f'{rating_type}_elo{format_suffix}'
            cursor.execute(f"""
                SELECT {column} FROM player_current_elo
                WHERE player_id = ?
            """, (player_id,))
        
        row = cursor.fetchone()
        return row[0] if row else self.initial_rating
    
    def update_team_ratings(
        self,
        conn,
        match_id: int,
        team1_id: int,
        team2_id: int,
        winner_id: Optional[int],
        match_format: str,
        match_date: datetime
    ) -> Tuple[float, float]:
        """
        Update team ELO ratings after a match.
        
        Args:
            conn: Database connection
            match_id: Match ID
            team1_id: First team ID
            team2_id: Second team ID
            winner_id: Winning team ID (None for ties/no result)
            match_format: 'T20' or 'ODI'
            match_date: Date of match
            
        Returns:
            Tuple of (team1_new_rating, team2_new_rating)
        """
        cursor = conn.cursor()
        
        # Get current ratings
        rating1 = self.get_team_rating(conn, team1_id, match_format, match_date)
        rating2 = self.get_team_rating(conn, team2_id, match_format, match_date)
        
        # Calculate expected scores
        expected1 = self.expected_score(rating1, rating2)
        expected2 = 1 - expected1
        
        # Determine actual scores
        if winner_id == team1_id:
            actual1, actual2 = 1.0, 0.0
        elif winner_id == team2_id:
            actual1, actual2 = 0.0, 1.0
        else:
            # Tie or no result
            actual1, actual2 = 0.5, 0.5
        
        # Calculate new ratings
        new_rating1 = self.calculate_new_rating(rating1, expected1, actual1, self.k_factor_team)
        new_rating2 = self.calculate_new_rating(rating2, expected2, actual2, self.k_factor_team)
        
        # Determine column names based on format
        elo_col = 'elo_t20' if match_format == 'T20' else 'elo_odi'
        change_col = 'elo_t20_change' if match_format == 'T20' else 'elo_odi_change'
        date_col = 'last_t20_match_date' if match_format == 'T20' else 'last_odi_match_date'
        
        # Get the other format's rating for history insert
        other_elo_col = 'elo_odi' if match_format == 'T20' else 'elo_t20'
        
        for team_id, new_rating, old_rating in [
            (team1_id, new_rating1, rating1),
            (team2_id, new_rating2, rating2)
        ]:
            change = new_rating - old_rating
            
            # Get other format rating
            other_rating = self.get_team_rating(
                conn, team_id,
                'ODI' if match_format == 'T20' else 'T20',
                match_date
            )
            
            # Insert into history
            if match_format == 'T20':
                cursor.execute("""
                    INSERT INTO team_elo_history (
                        team_id, date, match_id, elo_t20, elo_odi,
                        elo_t20_change, elo_odi_change
                    ) VALUES (?, ?, ?, ?, ?, ?, 0)
                """, (team_id, match_date, match_id, new_rating, other_rating, change))
            else:
                cursor.execute("""
                    INSERT INTO team_elo_history (
                        team_id, date, match_id, elo_t20, elo_odi,
                        elo_t20_change, elo_odi_change
                    ) VALUES (?, ?, ?, ?, ?, 0, ?)
                """, (team_id, match_date, match_id, other_rating, new_rating, change))
            
            # Update current rating
            cursor.execute(f"""
                INSERT INTO team_current_elo (team_id, {elo_col}, {date_col})
                VALUES (?, ?, ?)
                ON CONFLICT(team_id) DO UPDATE SET
                    {elo_col} = excluded.{elo_col},
                    {date_col} = excluded.{date_col},
                    updated_at = CURRENT_TIMESTAMP
            """, (team_id, new_rating, match_date))
        
        return new_rating1, new_rating2
    
    def update_player_batting_rating(
        self,
        conn,
        player_id: int,
        match_id: int,
        match_format: str,
        match_date: datetime,
        runs_scored: int,
        balls_faced: int,
        not_out: bool,
        bowling_team_avg_elo: float
    ) -> float:
        """
        Update player batting ELO after an innings.
        
        The update is based on:
        - Runs scored vs expected (considering bowling strength)
        - Strike rate relative to format norms
        - Match context (opponent strength)
        """
        cursor = conn.cursor()
        
        # Get current rating
        current_rating = self.get_player_rating(
            conn, player_id, match_format, 'batting', match_date
        )
        
        if balls_faced == 0:
            return current_rating  # No update if didn't face a ball
        
        # Calculate performance score (0-1 scale)
        # Based on runs scored, strike rate, and opponent quality
        
        # Expected runs based on balls faced and format
        avg_sr = 130 if match_format == 'T20' else 85  # Average strike rates
        expected_runs = balls_faced * (avg_sr / 100)
        
        # Adjust for opponent bowling strength
        bowling_adjustment = (bowling_team_avg_elo - self.initial_rating) / 400
        expected_runs *= (1 - bowling_adjustment * 0.1)
        
        # Performance ratio
        performance = runs_scored / max(expected_runs, 1)
        
        # Normalize to 0-1 scale (cap at 2x expected = 1.0)
        actual_score = min(1.0, performance / 2.0)
        
        # Expected score based on own rating vs bowling team
        expected_score = self.expected_score(current_rating, bowling_team_avg_elo)
        
        # Calculate new rating
        new_rating = self.calculate_new_rating(
            current_rating, expected_score, actual_score, self.k_factor_batting
        )
        
        return new_rating
    
    def update_player_bowling_rating(
        self,
        conn,
        player_id: int,
        match_id: int,
        match_format: str,
        match_date: datetime,
        overs_bowled: float,
        runs_conceded: int,
        wickets_taken: int,
        batting_team_avg_elo: float
    ) -> float:
        """
        Update player bowling ELO after bowling.
        
        The update is based on:
        - Economy rate vs expected
        - Wickets taken
        - Opponent batting strength
        """
        cursor = conn.cursor()
        
        current_rating = self.get_player_rating(
            conn, player_id, match_format, 'bowling', match_date
        )
        
        if overs_bowled == 0:
            return current_rating
        
        # Expected economy based on format
        avg_economy = 8.0 if match_format == 'T20' else 5.5
        
        # Adjust for opponent batting strength
        batting_adjustment = (batting_team_avg_elo - self.initial_rating) / 400
        expected_economy = avg_economy * (1 + batting_adjustment * 0.1)
        
        # Actual economy
        actual_economy = runs_conceded / overs_bowled if overs_bowled > 0 else avg_economy
        
        # Economy performance (lower is better)
        economy_score = max(0, min(1, (expected_economy - actual_economy + 4) / 8))
        
        # Wicket bonus (average 1 wicket per 4 overs in T20, 1 per 8 in ODI)
        expected_wickets = overs_bowled / (4 if match_format == 'T20' else 8)
        wicket_score = min(1, wickets_taken / max(expected_wickets, 0.5))
        
        # Combined performance
        actual_score = 0.6 * economy_score + 0.4 * wicket_score
        
        expected_score = self.expected_score(current_rating, batting_team_avg_elo)
        
        new_rating = self.calculate_new_rating(
            current_rating, expected_score, actual_score, self.k_factor_bowling
        )
        
        return new_rating
    
    def update_player_ratings_for_match(
        self,
        conn,
        match_id: int,
        match_format: str,
        match_date: datetime
    ):
        """
        Update all player ratings for a match.
        
        Args:
            conn: Database connection
            match_id: Match ID
            match_format: 'T20' or 'ODI'
            match_date: Date of match
        """
        cursor = conn.cursor()
        
        # Get match info for team ELOs
        cursor.execute("""
            SELECT team1_id, team2_id FROM matches WHERE match_id = ?
        """, (match_id,))
        match_row = cursor.fetchone()
        if not match_row:
            return
        
        team1_id, team2_id = match_row['team1_id'], match_row['team2_id']
        
        # Get team ratings for the match
        team1_elo = self.get_team_rating(conn, team1_id, match_format, match_date)
        team2_elo = self.get_team_rating(conn, team2_id, match_format, match_date)
        
        # Get all player stats for this match
        cursor.execute("""
            SELECT 
                pms.*,
                CASE WHEN pms.team_id = ? THEN ? ELSE ? END as opponent_batting_elo,
                CASE WHEN pms.team_id = ? THEN ? ELSE ? END as opponent_bowling_elo
            FROM player_match_stats pms
            WHERE pms.match_id = ?
        """, (team1_id, team2_elo, team1_elo, team1_id, team2_elo, team1_elo, match_id))
        
        player_stats = cursor.fetchall()
        
        for stats in player_stats:
            player_id = stats['player_id']
            
            # Get current ratings
            batting_elo = self.get_player_rating(conn, player_id, match_format, 'batting', match_date)
            bowling_elo = self.get_player_rating(conn, player_id, match_format, 'bowling', match_date)
            
            new_batting_elo = batting_elo
            new_bowling_elo = bowling_elo
            
            # Update batting ELO if batted
            if stats['batting_innings'] > 0:
                new_batting_elo = self.update_player_batting_rating(
                    conn, player_id, match_id, match_format, match_date,
                    stats['runs_scored'], stats['balls_faced'], stats['not_out'],
                    stats['opponent_bowling_elo']
                )
            
            # Update bowling ELO if bowled
            if stats['bowling_innings'] > 0 and stats['overs_bowled'] > 0:
                new_bowling_elo = self.update_player_bowling_rating(
                    conn, player_id, match_id, match_format, match_date,
                    stats['overs_bowled'], stats['runs_conceded'], stats['wickets_taken'],
                    stats['opponent_batting_elo']
                )
            
            # Calculate overall ELO (weighted average based on contribution)
            # Weight batting/bowling based on balls faced/bowled ratio
            total_balls = stats['balls_faced'] + (stats['overs_bowled'] * 6)
            if total_balls > 0:
                bat_weight = stats['balls_faced'] / total_balls
                bowl_weight = (stats['overs_bowled'] * 6) / total_balls
            else:
                bat_weight = bowl_weight = 0.5
            
            new_overall_elo = (new_batting_elo * bat_weight + new_bowling_elo * bowl_weight)
            
            # Insert into history
            cursor.execute("""
                INSERT INTO player_elo_history (
                    player_id, date, match_id, format,
                    batting_elo, bowling_elo, overall_elo,
                    batting_elo_change, bowling_elo_change, overall_elo_change
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                player_id, match_date, match_id, match_format,
                new_batting_elo, new_bowling_elo, new_overall_elo,
                new_batting_elo - batting_elo,
                new_bowling_elo - bowling_elo,
                new_overall_elo - (batting_elo * bat_weight + bowling_elo * bowl_weight)
            ))
            
            # Update current ELO
            format_suffix = '_t20' if match_format == 'T20' else '_odi'
            date_col = 'last_t20_match_date' if match_format == 'T20' else 'last_odi_match_date'
            
            cursor.execute(f"""
                INSERT INTO player_current_elo (
                    player_id,
                    batting_elo{format_suffix}, bowling_elo{format_suffix}, overall_elo{format_suffix},
                    {date_col}
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(player_id) DO UPDATE SET
                    batting_elo{format_suffix} = excluded.batting_elo{format_suffix},
                    bowling_elo{format_suffix} = excluded.bowling_elo{format_suffix},
                    overall_elo{format_suffix} = excluded.overall_elo{format_suffix},
                    {date_col} = excluded.{date_col},
                    updated_at = CURRENT_TIMESTAMP
            """, (player_id, new_batting_elo, new_bowling_elo, new_overall_elo, match_date))
    
    def create_monthly_snapshots(self, conn, year_month: str):
        """
        Create monthly ELO snapshots for historical lookups.
        
        Args:
            conn: Database connection
            year_month: Year-month string (e.g., '2023-06')
        """
        cursor = conn.cursor()
        
        # Parse year-month
        year, month = map(int, year_month.split('-'))
        
        # Get last day of month
        if month == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)
        last_day = next_month - timedelta(days=1)
        
        # Create team snapshots
        cursor.execute("""
            INSERT INTO team_elo_history (team_id, date, elo_t20, elo_odi, is_monthly_snapshot)
            SELECT team_id, ?, elo_t20, elo_odi, TRUE
            FROM team_current_elo
        """, (last_day,))
        
        # Create player snapshots (for both formats)
        for format_type in ['T20', 'ODI']:
            suffix = '_t20' if format_type == 'T20' else '_odi'
            cursor.execute(f"""
                INSERT INTO player_elo_history (
                    player_id, date, format,
                    batting_elo, bowling_elo, overall_elo,
                    is_monthly_snapshot
                )
                SELECT 
                    player_id, ?, ?,
                    batting_elo{suffix}, bowling_elo{suffix}, overall_elo{suffix},
                    TRUE
                FROM player_current_elo
            """, (last_day, format_type))
        
        logger.info(f"Created monthly snapshots for {year_month}")


def calculate_all_elos(force_recalculate: bool = False):
    """
    Calculate ELO ratings for all matches in chronological order.
    
    Args:
        force_recalculate: If True, clear existing ratings and recalculate
    """
    from tqdm import tqdm
    
    calculator = EloCalculator()
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        if force_recalculate:
            logger.info("Clearing existing ELO data...")
            cursor.execute("DELETE FROM team_elo_history")
            cursor.execute("DELETE FROM player_elo_history")
            cursor.execute("DELETE FROM team_current_elo")
            cursor.execute("DELETE FROM player_current_elo")
        
        # Get all matches in chronological order
        cursor.execute("""
            SELECT match_id, match_type, date, team1_id, team2_id, winner_id
            FROM matches
            ORDER BY date, match_id
        """)
        matches = cursor.fetchall()
        
        if not matches:
            logger.warning("No matches found in database")
            return
        
        logger.info(f"Calculating ELO ratings for {len(matches)} matches...")
        
        current_month = None
        
        for match in tqdm(matches, desc="Calculating ELOs"):
            match_id = match['match_id']
            match_format = match['match_type']
            match_date = match['date']
            team1_id = match['team1_id']
            team2_id = match['team2_id']
            winner_id = match['winner_id']
            
            # Parse date
            if isinstance(match_date, str):
                match_date = datetime.strptime(match_date, '%Y-%m-%d').date()
            
            # Check if we need to create monthly snapshot
            year_month = match_date.strftime('%Y-%m')
            if current_month and year_month != current_month:
                calculator.create_monthly_snapshots(conn, current_month)
            current_month = year_month
            
            # Update team ratings
            calculator.update_team_ratings(
                conn, match_id, team1_id, team2_id, winner_id,
                match_format, match_date
            )
            
            # Update player ratings
            calculator.update_player_ratings_for_match(
                conn, match_id, match_format, match_date
            )
        
        # Create final monthly snapshot
        if current_month:
            calculator.create_monthly_snapshots(conn, current_month)
        
        logger.info("ELO calculation complete!")
        
        # Print summary
        cursor.execute("SELECT COUNT(*) FROM team_elo_history")
        team_records = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM player_elo_history")
        player_records = cursor.fetchone()[0]
        
        logger.info(f"Team ELO records: {team_records}")
        logger.info(f"Player ELO records: {player_records}")


def print_rankings(format_type: str = 'T20', limit: int = 20):
    """Print current team and player rankings."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        elo_col = 'elo_t20' if format_type == 'T20' else 'elo_odi'
        
        print(f"\n{'='*50}")
        print(f"TOP {limit} TEAM RANKINGS ({format_type})")
        print('='*50)
        
        cursor.execute(f"""
            SELECT t.name, e.{elo_col} as elo
            FROM team_current_elo e
            JOIN teams t ON e.team_id = t.team_id
            ORDER BY e.{elo_col} DESC
            LIMIT ?
        """, (limit,))
        
        for i, row in enumerate(cursor.fetchall(), 1):
            print(f"{i:2}. {row['name']:30} {row['elo']:.0f}")
        
        # Player batting rankings
        bat_col = f'batting_elo_{format_type.lower()}'
        
        print(f"\n{'='*50}")
        print(f"TOP {limit} BATTING RANKINGS ({format_type})")
        print('='*50)
        
        cursor.execute(f"""
            SELECT p.name, e.{bat_col} as elo
            FROM player_current_elo e
            JOIN players p ON e.player_id = p.player_id
            WHERE e.{bat_col} != 1500
            ORDER BY e.{bat_col} DESC
            LIMIT ?
        """, (limit,))
        
        for i, row in enumerate(cursor.fetchall(), 1):
            print(f"{i:2}. {row['name']:30} {row['elo']:.0f}")
        
        # Player bowling rankings
        bowl_col = f'bowling_elo_{format_type.lower()}'
        
        print(f"\n{'='*50}")
        print(f"TOP {limit} BOWLING RANKINGS ({format_type})")
        print('='*50)
        
        cursor.execute(f"""
            SELECT p.name, e.{bowl_col} as elo
            FROM player_current_elo e
            JOIN players p ON e.player_id = p.player_id
            WHERE e.{bowl_col} != 1500
            ORDER BY e.{bowl_col} DESC
            LIMIT ?
        """, (limit,))
        
        for i, row in enumerate(cursor.fetchall(), 1):
            print(f"{i:2}. {row['name']:30} {row['elo']:.0f}")


def main():
    """Main function to calculate ELO ratings."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    import argparse
    parser = argparse.ArgumentParser(description='Calculate ELO ratings')
    parser.add_argument('--force', action='store_true', help='Force recalculation')
    parser.add_argument('--rankings', action='store_true', help='Print rankings')
    parser.add_argument('--format', default='T20', choices=['T20', 'ODI'], help='Format for rankings')
    args = parser.parse_args()
    
    if args.rankings:
        print_rankings(args.format)
    else:
        calculate_all_elos(force_recalculate=args.force)
        print_rankings('T20')
        print_rankings('ODI')
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

