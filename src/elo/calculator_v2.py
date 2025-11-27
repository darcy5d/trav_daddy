"""
ELO Rating Calculator V2 - With Format AND Gender Separation.

Maintains separate ELO ratings for:
- T20 Men
- T20 Women
- ODI Men
- ODI Women

Each combination tracks team and player (batting/bowling/overall) ratings.
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import math

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import ELO_CONFIG, DATABASE_PATH
from src.data.database import get_db_connection

logger = logging.getLogger(__name__)


class EloCalculatorV2:
    """
    ELO Rating Calculator with format and gender separation.
    """
    
    def __init__(self):
        self.initial_rating = ELO_CONFIG['initial_rating']
        self.k_factor_team = ELO_CONFIG['k_factor_team']
        self.k_factor_batting = ELO_CONFIG['k_factor_player_batting']
        self.k_factor_bowling = ELO_CONFIG['k_factor_player_bowling']
        self.rating_floor = ELO_CONFIG['rating_floor']
        self.rating_ceiling = ELO_CONFIG['rating_ceiling']
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player/team A against B."""
        exponent = (rating_b - rating_a) / 400.0
        return 1.0 / (1.0 + math.pow(10, exponent))
    
    def calculate_new_rating(
        self,
        old_rating: float,
        expected: float,
        actual: float,
        k_factor: float
    ) -> float:
        """Calculate new rating after a match/event."""
        new_rating = old_rating + k_factor * (actual - expected)
        return max(self.rating_floor, min(self.rating_ceiling, new_rating))
    
    def get_team_rating(
        self,
        conn,
        team_id: int,
        match_format: str,
        gender: str,
        as_of_date: Optional[datetime] = None
    ) -> float:
        """Get team rating for specific format and gender."""
        cursor = conn.cursor()
        
        if as_of_date:
            cursor.execute("""
                SELECT elo FROM team_elo_history
                WHERE team_id = ? AND format = ? AND gender = ? AND date <= ?
                ORDER BY date DESC, elo_id DESC
                LIMIT 1
            """, (team_id, match_format, gender, as_of_date))
        else:
            col = f'elo_{match_format.lower()}_{gender}'
            cursor.execute(f"""
                SELECT {col} FROM team_current_elo
                WHERE team_id = ?
            """, (team_id,))
        
        row = cursor.fetchone()
        return row[0] if row else self.initial_rating
    
    def get_player_rating(
        self,
        conn,
        player_id: int,
        match_format: str,
        gender: str,
        rating_type: str = 'overall',
        as_of_date: Optional[datetime] = None
    ) -> float:
        """Get player rating for specific format, gender, and type."""
        cursor = conn.cursor()
        
        if as_of_date:
            cursor.execute(f"""
                SELECT {rating_type}_elo FROM player_elo_history
                WHERE player_id = ? AND format = ? AND gender = ? AND date <= ?
                ORDER BY date DESC, elo_id DESC
                LIMIT 1
            """, (player_id, match_format, gender, as_of_date))
        else:
            col = f'{rating_type}_elo_{match_format.lower()}_{gender}'
            cursor.execute(f"""
                SELECT {col} FROM player_current_elo
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
        gender: str,
        match_date: datetime
    ) -> Tuple[float, float]:
        """Update team ELO ratings after a match."""
        cursor = conn.cursor()
        
        # Get current ratings
        rating1 = self.get_team_rating(conn, team1_id, match_format, gender, match_date)
        rating2 = self.get_team_rating(conn, team2_id, match_format, gender, match_date)
        
        # Calculate expected scores
        expected1 = self.expected_score(rating1, rating2)
        expected2 = 1 - expected1
        
        # Determine actual scores
        if winner_id == team1_id:
            actual1, actual2 = 1.0, 0.0
        elif winner_id == team2_id:
            actual1, actual2 = 0.0, 1.0
        else:
            actual1, actual2 = 0.5, 0.5
        
        # Calculate new ratings
        new_rating1 = self.calculate_new_rating(rating1, expected1, actual1, self.k_factor_team)
        new_rating2 = self.calculate_new_rating(rating2, expected2, actual2, self.k_factor_team)
        
        # Insert into history
        for team_id, new_rating, old_rating in [
            (team1_id, new_rating1, rating1),
            (team2_id, new_rating2, rating2)
        ]:
            change = new_rating - old_rating
            cursor.execute("""
                INSERT INTO team_elo_history (
                    team_id, date, match_id, format, gender, elo, elo_change
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (team_id, match_date, match_id, match_format, gender, new_rating, change))
            
            # Update current rating
            col = f'elo_{match_format.lower()}_{gender}'
            date_col = f'last_{match_format.lower()}_{gender}_date'
            
            cursor.execute(f"""
                INSERT INTO team_current_elo (team_id, {col}, {date_col})
                VALUES (?, ?, ?)
                ON CONFLICT(team_id) DO UPDATE SET
                    {col} = excluded.{col},
                    {date_col} = excluded.{date_col},
                    updated_at = CURRENT_TIMESTAMP
            """, (team_id, new_rating, match_date))
        
        return new_rating1, new_rating2
    
    def update_player_ratings_for_match(
        self,
        conn,
        match_id: int,
        match_format: str,
        gender: str,
        match_date: datetime
    ):
        """Update all player ratings for a match."""
        cursor = conn.cursor()
        
        # Get match info
        cursor.execute("""
            SELECT team1_id, team2_id FROM matches WHERE match_id = ?
        """, (match_id,))
        match_row = cursor.fetchone()
        if not match_row:
            return
        
        team1_id, team2_id = match_row['team1_id'], match_row['team2_id']
        
        # Get team ratings
        team1_elo = self.get_team_rating(conn, team1_id, match_format, gender, match_date)
        team2_elo = self.get_team_rating(conn, team2_id, match_format, gender, match_date)
        
        # Get all player stats
        cursor.execute("""
            SELECT 
                pms.*,
                CASE WHEN pms.team_id = ? THEN ? ELSE ? END as opponent_elo
            FROM player_match_stats pms
            WHERE pms.match_id = ?
        """, (team1_id, team2_elo, team1_elo, match_id))
        
        player_stats = cursor.fetchall()
        
        for stats in player_stats:
            player_id = stats['player_id']
            
            # Get current ratings
            batting_elo = self.get_player_rating(
                conn, player_id, match_format, gender, 'batting', match_date
            )
            bowling_elo = self.get_player_rating(
                conn, player_id, match_format, gender, 'bowling', match_date
            )
            
            new_batting_elo = batting_elo
            new_bowling_elo = bowling_elo
            
            # Update batting ELO
            if stats['batting_innings'] > 0 and stats['balls_faced'] > 0:
                avg_sr = 130 if match_format == 'T20' else 85
                expected_runs = stats['balls_faced'] * (avg_sr / 100)
                opponent_adjustment = (stats['opponent_elo'] - self.initial_rating) / 400
                expected_runs *= (1 - opponent_adjustment * 0.1)
                
                performance = stats['runs_scored'] / max(expected_runs, 1)
                actual_score = min(1.0, performance / 2.0)
                expected_score = self.expected_score(batting_elo, stats['opponent_elo'])
                
                new_batting_elo = self.calculate_new_rating(
                    batting_elo, expected_score, actual_score, self.k_factor_batting
                )
            
            # Update bowling ELO
            if stats['overs_bowled'] and stats['overs_bowled'] > 0:
                avg_economy = 8.0 if match_format == 'T20' else 5.5
                opponent_adjustment = (stats['opponent_elo'] - self.initial_rating) / 400
                expected_economy = avg_economy * (1 + opponent_adjustment * 0.1)
                
                actual_economy = stats['runs_conceded'] / stats['overs_bowled']
                economy_score = max(0, min(1, (expected_economy - actual_economy + 4) / 8))
                
                expected_wickets = stats['overs_bowled'] / (4 if match_format == 'T20' else 8)
                wicket_score = min(1, stats['wickets_taken'] / max(expected_wickets, 0.5))
                
                actual_score = 0.6 * economy_score + 0.4 * wicket_score
                expected_score = self.expected_score(bowling_elo, stats['opponent_elo'])
                
                new_bowling_elo = self.calculate_new_rating(
                    bowling_elo, expected_score, actual_score, self.k_factor_bowling
                )
            
            # Calculate overall
            total_balls = stats['balls_faced'] + (stats['overs_bowled'] * 6 if stats['overs_bowled'] else 0)
            if total_balls > 0:
                bat_weight = stats['balls_faced'] / total_balls
                bowl_weight = 1 - bat_weight
            else:
                bat_weight = bowl_weight = 0.5
            
            new_overall_elo = (new_batting_elo * bat_weight + new_bowling_elo * bowl_weight)
            
            # Insert into history
            cursor.execute("""
                INSERT INTO player_elo_history (
                    player_id, date, match_id, format, gender,
                    batting_elo, bowling_elo, overall_elo,
                    batting_elo_change, bowling_elo_change, overall_elo_change
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                player_id, match_date, match_id, match_format, gender,
                new_batting_elo, new_bowling_elo, new_overall_elo,
                new_batting_elo - batting_elo,
                new_bowling_elo - bowling_elo,
                new_overall_elo - (batting_elo * bat_weight + bowling_elo * bowl_weight)
            ))
            
            # Update current ELO
            suffix = f'{match_format.lower()}_{gender}'
            date_col = f'last_{suffix}_date'
            
            cursor.execute(f"""
                INSERT INTO player_current_elo (
                    player_id,
                    batting_elo_{suffix}, bowling_elo_{suffix}, overall_elo_{suffix},
                    {date_col}
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(player_id) DO UPDATE SET
                    batting_elo_{suffix} = excluded.batting_elo_{suffix},
                    bowling_elo_{suffix} = excluded.bowling_elo_{suffix},
                    overall_elo_{suffix} = excluded.overall_elo_{suffix},
                    {date_col} = excluded.{date_col},
                    updated_at = CURRENT_TIMESTAMP
            """, (player_id, new_batting_elo, new_bowling_elo, new_overall_elo, match_date))
    
    def create_monthly_snapshots(self, conn, year_month: str, format_type: str, gender: str):
        """Create monthly ELO snapshots for a specific format/gender."""
        cursor = conn.cursor()
        
        year, month = map(int, year_month.split('-'))
        if month == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)
        last_day = next_month - timedelta(days=1)
        
        col = f'elo_{format_type.lower()}_{gender}'
        
        # Team snapshots
        cursor.execute(f"""
            INSERT INTO team_elo_history (team_id, date, format, gender, elo, is_monthly_snapshot)
            SELECT team_id, ?, ?, ?, {col}, TRUE
            FROM team_current_elo
            WHERE {col} != 1500
        """, (last_day, format_type, gender))
        
        # Player snapshots
        bat_col = f'batting_elo_{format_type.lower()}_{gender}'
        bowl_col = f'bowling_elo_{format_type.lower()}_{gender}'
        overall_col = f'overall_elo_{format_type.lower()}_{gender}'
        
        cursor.execute(f"""
            INSERT INTO player_elo_history (
                player_id, date, format, gender,
                batting_elo, bowling_elo, overall_elo,
                is_monthly_snapshot
            )
            SELECT 
                player_id, ?, ?, ?,
                {bat_col}, {bowl_col}, {overall_col},
                TRUE
            FROM player_current_elo
            WHERE {bat_col} != 1500 OR {bowl_col} != 1500
        """, (last_day, format_type, gender))


def calculate_all_elos_v2(force_recalculate: bool = False):
    """Calculate ELO ratings with format and gender separation."""
    from tqdm import tqdm
    
    calculator = EloCalculatorV2()
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Apply new schema
        logger.info("Applying updated schema...")
        schema_path = Path(__file__).parent.parent / "data" / "schema_v2.sql"
        schema_sql = schema_path.read_text()
        conn.executescript(schema_sql)
        
        # Get all matches ordered chronologically
        cursor.execute("""
            SELECT match_id, match_type, gender, date, team1_id, team2_id, winner_id
            FROM matches
            ORDER BY date, match_id
        """)
        matches = cursor.fetchall()
        
        if not matches:
            logger.warning("No matches found")
            return
        
        logger.info(f"Calculating ELO ratings for {len(matches)} matches...")
        
        # Track months for snapshots by format/gender
        current_months = {}  # (format, gender) -> current_month
        
        for match in tqdm(matches, desc="Calculating ELOs"):
            match_id = match['match_id']
            match_format = match['match_type']
            gender = match['gender']
            match_date = match['date']
            team1_id = match['team1_id']
            team2_id = match['team2_id']
            winner_id = match['winner_id']
            
            if isinstance(match_date, str):
                match_date = datetime.strptime(match_date, '%Y-%m-%d').date()
            
            key = (match_format, gender)
            year_month = match_date.strftime('%Y-%m')
            
            # Monthly snapshot
            if key in current_months and year_month != current_months[key]:
                calculator.create_monthly_snapshots(conn, current_months[key], match_format, gender)
            current_months[key] = year_month
            
            # Update team ratings
            calculator.update_team_ratings(
                conn, match_id, team1_id, team2_id, winner_id,
                match_format, gender, match_date
            )
            
            # Update player ratings
            calculator.update_player_ratings_for_match(
                conn, match_id, match_format, gender, match_date
            )
        
        # Final snapshots
        for (fmt, gen), month in current_months.items():
            calculator.create_monthly_snapshots(conn, month, fmt, gen)
        
        logger.info("ELO calculation complete!")
        
        # Print summary
        for fmt in ['T20', 'ODI']:
            for gen in ['male', 'female']:
                cursor.execute("""
                    SELECT COUNT(*) FROM team_elo_history 
                    WHERE format = ? AND gender = ? AND NOT is_monthly_snapshot
                """, (fmt, gen))
                count = cursor.fetchone()[0]
                logger.info(f"{fmt} {gen}: {count} team rating updates")


def print_rankings_v2(format_type: str = 'T20', gender: str = 'male', limit: int = 15):
    """Print current rankings for specific format and gender."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        col = f'elo_{format_type.lower()}_{gender}'
        
        print(f"\n{'='*50}")
        print(f"TOP {limit} TEAM RANKINGS ({format_type} {gender.upper()})")
        print('='*50)
        
        cursor.execute(f"""
            SELECT t.name, e.{col} as elo
            FROM team_current_elo e
            JOIN teams t ON e.team_id = t.team_id
            WHERE e.{col} != 1500
            ORDER BY e.{col} DESC
            LIMIT ?
        """, (limit,))
        
        for i, row in enumerate(cursor.fetchall(), 1):
            print(f"{i:2}. {row['name']:30} {row['elo']:.0f}")
        
        # Batting
        bat_col = f'batting_elo_{format_type.lower()}_{gender}'
        
        print(f"\n{'='*50}")
        print(f"TOP {limit} BATTING RANKINGS ({format_type} {gender.upper()})")
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
        
        # Bowling
        bowl_col = f'bowling_elo_{format_type.lower()}_{gender}'
        
        print(f"\n{'='*50}")
        print(f"TOP {limit} BOWLING RANKINGS ({format_type} {gender.upper()})")
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
    parser = argparse.ArgumentParser(description='Calculate ELO ratings (V2 with gender separation)')
    parser.add_argument('--force', action='store_true', help='Force recalculation')
    parser.add_argument('--rankings', action='store_true', help='Print rankings')
    parser.add_argument('--format', default='T20', choices=['T20', 'ODI'])
    parser.add_argument('--gender', default='male', choices=['male', 'female'])
    args = parser.parse_args()
    
    if args.rankings:
        print_rankings_v2(args.format, args.gender)
    else:
        calculate_all_elos_v2(force_recalculate=args.force)
        
        # Print all rankings
        for fmt in ['T20', 'ODI']:
            for gen in ['male', 'female']:
                print_rankings_v2(fmt, gen, limit=10)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

