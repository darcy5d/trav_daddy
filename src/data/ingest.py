"""
Data ingestion pipeline for Cricsheet JSON data.

Parses JSON match files and populates the SQLite database.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import RAW_DATA_DIR, MIN_MATCH_DATE
from src.data.database import (
    get_db_connection,
    init_database,
    DatabaseManager,
    print_database_summary
)
from src.data.downloader import get_json_files

logger = logging.getLogger(__name__)


class CricsheetParser:
    """
    Parser for Cricsheet JSON match files.
    
    Handles the extraction and normalization of match data from
    Cricsheet's JSON format.
    """
    
    def __init__(self):
        self.db_manager = DatabaseManager()
    
    def parse_match_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Parse a single Cricsheet JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Parsed match data dictionary or None if parsing fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract cricsheet ID from filename
            cricsheet_id = file_path.stem
            
            return {
                'cricsheet_id': cricsheet_id,
                'meta': data.get('meta', {}),
                'info': data.get('info', {}),
                'innings': data.get('innings', [])
            }
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return None
    
    def get_match_type(self, info: Dict) -> Optional[str]:
        """Determine match type (T20 or ODI)."""
        match_type = info.get('match_type', '').upper()
        
        if match_type in ['T20', 'T20I']:
            return 'T20'
        elif match_type in ['ODI', 'OD']:
            return 'ODI'
        
        # Check overs per innings
        overs = info.get('overs')
        if overs:
            if overs == 20:
                return 'T20'
            elif overs == 50:
                return 'ODI'
        
        return None
    
    def parse_date(self, info: Dict) -> Optional[datetime]:
        """Parse match date from info."""
        dates = info.get('dates', [])
        if dates:
            try:
                # Take the first date (for multi-day matches)
                date_str = dates[0]
                return datetime.strptime(date_str, '%Y-%m-%d').date()
            except (ValueError, IndexError):
                pass
        return None
    
    def get_winner(self, info: Dict) -> tuple:
        """
        Extract winner information.
        
        Returns:
            Tuple of (winner_name, win_type, win_margin)
        """
        outcome = info.get('outcome', {})
        
        if 'winner' in outcome:
            winner = outcome['winner']
            
            if 'by' in outcome:
                by = outcome['by']
                if 'runs' in by:
                    return (winner, 'runs', by['runs'])
                elif 'wickets' in by:
                    return (winner, 'wickets', by['wickets'])
            
            return (winner, 'wickets', None)
        
        if 'result' in outcome:
            result = outcome['result']
            if result == 'tie':
                return (None, 'tie', None)
            elif result == 'no result':
                return (None, 'no result', None)
            elif result == 'draw':
                return (None, 'draw', None)
        
        return (None, None, None)
    
    def get_player_id_mapping(self, info: Dict) -> Dict[str, str]:
        """
        Extract player registry ID mapping from match info.
        
        Returns:
            Dictionary mapping player names to registry IDs
        """
        registry = info.get('registry', {})
        people = registry.get('people', {})
        return people  # Maps name -> registry_id


class MatchIngestor:
    """
    Handles the ingestion of parsed match data into the database.
    """
    
    def __init__(self):
        self.parser = CricsheetParser()
        self.db_manager = DatabaseManager()
        self.stats = {
            'matches_processed': 0,
            'matches_skipped': 0,
            'matches_failed': 0,
            'deliveries_inserted': 0
        }
    
    def ingest_match(self, conn, match_data: Dict[str, Any]) -> bool:
        """
        Ingest a single match into the database.
        
        Args:
            conn: Database connection
            match_data: Parsed match data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cricsheet_id = match_data['cricsheet_id']
            info = match_data['info']
            innings_data = match_data['innings']
            
            # Check if match already exists
            if self.db_manager.match_exists(conn, cricsheet_id):
                return False
            
            # Get match type
            match_type = self.parser.get_match_type(info)
            if not match_type:
                logger.debug(f"Skipping {cricsheet_id}: Unknown match type")
                return False
            
            # Parse date
            match_date = self.parser.parse_date(info)
            if not match_date:
                logger.debug(f"Skipping {cricsheet_id}: No date found")
                return False
            
            # Check date filter
            min_date = datetime.strptime(MIN_MATCH_DATE, '%Y-%m-%d').date()
            if match_date < min_date:
                return False
            
            # Get teams
            teams = info.get('teams', [])
            if len(teams) != 2:
                logger.debug(f"Skipping {cricsheet_id}: Invalid teams")
                return False
            
            team1_id = self.db_manager.get_or_create_team(conn, teams[0])
            team2_id = self.db_manager.get_or_create_team(conn, teams[1])
            
            # Get venue
            venue_name = info.get('venue', 'Unknown')
            city = info.get('city')
            venue_id = self.db_manager.get_or_create_venue(conn, venue_name, city)
            
            # Get toss info
            toss = info.get('toss', {})
            toss_winner_name = toss.get('winner')
            toss_winner_id = None
            if toss_winner_name:
                toss_winner_id = self.db_manager.get_or_create_team(conn, toss_winner_name)
            toss_decision = toss.get('decision')
            
            # Get winner
            winner_name, win_type, win_margin = self.parser.get_winner(info)
            winner_id = None
            if winner_name:
                winner_id = self.db_manager.get_or_create_team(conn, winner_name)
            
            # Get player ID mapping
            player_mapping = self.parser.get_player_id_mapping(info)
            
            # Get player of match
            pom_names = info.get('player_of_match', [])
            pom_id = None
            if pom_names:
                pom_name = pom_names[0]
                pom_registry_id = player_mapping.get(pom_name)
                pom_id = self.db_manager.get_or_create_player(conn, pom_name, pom_registry_id)
            
            # Insert match
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO matches (
                    cricsheet_id, match_type, date, venue_id,
                    team1_id, team2_id, toss_winner_id, toss_decision,
                    winner_id, win_type, win_margin, player_of_match_id,
                    overs_per_innings, event_name, gender
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cricsheet_id, match_type, match_date, venue_id,
                team1_id, team2_id, toss_winner_id, toss_decision,
                winner_id, win_type, win_margin, pom_id,
                20 if match_type == 'T20' else 50,
                info.get('event', {}).get('name'),
                info.get('gender', 'male')
            ))
            match_id = cursor.lastrowid
            
            # Insert innings and deliveries
            for innings_num, innings in enumerate(innings_data, 1):
                self._ingest_innings(conn, match_id, innings_num, innings, 
                                    teams, player_mapping)
            
            # Calculate and insert player match stats
            self._calculate_player_stats(conn, match_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error ingesting match {match_data.get('cricsheet_id', 'unknown')}: {e}")
            raise
    
    def _ingest_innings(
        self,
        conn,
        match_id: int,
        innings_num: int,
        innings: Dict,
        teams: List[str],
        player_mapping: Dict[str, str]
    ):
        """Ingest a single innings with all deliveries."""
        cursor = conn.cursor()
        
        # Get batting/bowling teams
        batting_team = innings.get('team')
        if not batting_team:
            return
        
        bowling_team = teams[1] if batting_team == teams[0] else teams[0]
        
        batting_team_id = self.db_manager.get_or_create_team(conn, batting_team)
        bowling_team_id = self.db_manager.get_or_create_team(conn, bowling_team)
        
        # Calculate innings totals from overs
        overs_data = innings.get('overs', [])
        total_runs = 0
        total_wickets = 0
        total_extras = 0
        total_overs = 0
        
        # Insert innings (will update totals after processing deliveries)
        cursor.execute("""
            INSERT INTO innings (
                match_id, innings_number, batting_team_id, bowling_team_id
            ) VALUES (?, ?, ?, ?)
        """, (match_id, innings_num, batting_team_id, bowling_team_id))
        innings_id = cursor.lastrowid
        
        # Process each over
        for over_data in overs_data:
            over_num = over_data.get('over', 0)
            deliveries = over_data.get('deliveries', [])
            
            ball_num = 0
            for delivery in deliveries:
                ball_num += 1
                
                # Get player IDs
                batter_name = delivery.get('batter')
                bowler_name = delivery.get('bowler')
                non_striker_name = delivery.get('non_striker')
                
                batter_id = self.db_manager.get_or_create_player(
                    conn, batter_name, player_mapping.get(batter_name)
                )
                bowler_id = self.db_manager.get_or_create_player(
                    conn, bowler_name, player_mapping.get(bowler_name)
                )
                non_striker_id = None
                if non_striker_name:
                    non_striker_id = self.db_manager.get_or_create_player(
                        conn, non_striker_name, player_mapping.get(non_striker_name)
                    )
                
                # Parse runs
                runs = delivery.get('runs', {})
                runs_batter = runs.get('batter', 0)
                runs_extras = runs.get('extras', 0)
                runs_total = runs.get('total', 0)
                
                total_runs += runs_total
                total_extras += runs_extras
                
                # Parse extras breakdown
                extras = delivery.get('extras', {})
                extras_wides = extras.get('wides', 0)
                extras_noballs = extras.get('noballs', 0)
                extras_byes = extras.get('byes', 0)
                extras_legbyes = extras.get('legbyes', 0)
                extras_penalty = extras.get('penalty', 0)
                
                # Parse wicket
                is_wicket = False
                wicket_type = None
                dismissed_player_id = None
                fielder1_id = None
                fielder2_id = None
                
                wickets = delivery.get('wickets', [])
                if wickets:
                    wicket = wickets[0]  # Take first wicket
                    is_wicket = True
                    wicket_type = wicket.get('kind')
                    total_wickets += 1
                    
                    dismissed_name = wicket.get('player_out')
                    if dismissed_name:
                        dismissed_player_id = self.db_manager.get_or_create_player(
                            conn, dismissed_name, player_mapping.get(dismissed_name)
                        )
                    
                    fielders = wicket.get('fielders', [])
                    if fielders:
                        if isinstance(fielders[0], dict):
                            fielder1_name = fielders[0].get('name')
                        else:
                            fielder1_name = fielders[0]
                        if fielder1_name:
                            fielder1_id = self.db_manager.get_or_create_player(
                                conn, fielder1_name, player_mapping.get(fielder1_name)
                            )
                        if len(fielders) > 1:
                            if isinstance(fielders[1], dict):
                                fielder2_name = fielders[1].get('name')
                            else:
                                fielder2_name = fielders[1]
                            if fielder2_name:
                                fielder2_id = self.db_manager.get_or_create_player(
                                    conn, fielder2_name, player_mapping.get(fielder2_name)
                                )
                
                # Boundaries
                is_four = runs_batter == 4 and runs_extras == 0
                is_six = runs_batter == 6 and runs_extras == 0
                
                # Insert delivery
                cursor.execute("""
                    INSERT INTO deliveries (
                        innings_id, over_number, ball_number,
                        batter_id, bowler_id, non_striker_id,
                        runs_batter, runs_extras, runs_total,
                        extras_wides, extras_noballs, extras_byes,
                        extras_legbyes, extras_penalty,
                        is_wicket, wicket_type, dismissed_player_id,
                        fielder1_id, fielder2_id,
                        is_boundary_four, is_boundary_six
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    innings_id, over_num, ball_num,
                    batter_id, bowler_id, non_striker_id,
                    runs_batter, runs_extras, runs_total,
                    extras_wides, extras_noballs, extras_byes,
                    extras_legbyes, extras_penalty,
                    is_wicket, wicket_type, dismissed_player_id,
                    fielder1_id, fielder2_id,
                    is_four, is_six
                ))
                
                self.stats['deliveries_inserted'] += 1
            
            total_overs = over_num + 1
        
        # Update innings totals
        cursor.execute("""
            UPDATE innings SET
                total_runs = ?,
                total_wickets = ?,
                total_overs = ?,
                total_extras = ?,
                is_complete = ?
            WHERE innings_id = ?
        """, (total_runs, total_wickets, total_overs, total_extras, True, innings_id))
    
    def _calculate_player_stats(self, conn, match_id: int):
        """Calculate aggregated player statistics for a match."""
        cursor = conn.cursor()
        
        # Get all innings for this match
        cursor.execute("""
            SELECT innings_id, batting_team_id, bowling_team_id
            FROM innings WHERE match_id = ?
        """, (match_id,))
        innings_rows = cursor.fetchall()
        
        player_stats = {}  # player_id -> stats dict
        
        for innings_row in innings_rows:
            innings_id = innings_row['innings_id']
            batting_team_id = innings_row['batting_team_id']
            bowling_team_id = innings_row['bowling_team_id']
            
            # Get all deliveries for this innings
            cursor.execute("""
                SELECT * FROM deliveries WHERE innings_id = ?
            """, (innings_id,))
            deliveries = cursor.fetchall()
            
            # Track batting positions
            batting_order = []
            
            for delivery in deliveries:
                batter_id = delivery['batter_id']
                bowler_id = delivery['bowler_id']
                
                # Initialize player stats
                for pid, team_id in [(batter_id, batting_team_id), (bowler_id, bowling_team_id)]:
                    if pid not in player_stats:
                        player_stats[pid] = {
                            'team_id': team_id,
                            'batting_innings': 0,
                            'runs_scored': 0,
                            'balls_faced': 0,
                            'fours_hit': 0,
                            'sixes_hit': 0,
                            'not_out': True,
                            'batting_position': None,
                            'bowling_innings': 0,
                            'overs_bowled': 0,
                            'runs_conceded': 0,
                            'wickets_taken': 0,
                            'maidens': 0,
                            'wides_bowled': 0,
                            'noballs_bowled': 0,
                            'dots_bowled': 0,
                            'catches': 0,
                            'run_outs': 0,
                            'stumpings': 0
                        }
                
                # Batting position tracking
                if batter_id not in batting_order:
                    batting_order.append(batter_id)
                    player_stats[batter_id]['batting_position'] = len(batting_order)
                    player_stats[batter_id]['batting_innings'] = 1
                
                # Batting stats
                runs_batter = delivery['runs_batter']
                player_stats[batter_id]['runs_scored'] += runs_batter
                
                # Count ball faced (not wides or noballs for batter)
                if not delivery['extras_wides']:
                    player_stats[batter_id]['balls_faced'] += 1
                
                if delivery['is_boundary_four']:
                    player_stats[batter_id]['fours_hit'] += 1
                if delivery['is_boundary_six']:
                    player_stats[batter_id]['sixes_hit'] += 1
                
                # Bowling stats
                player_stats[bowler_id]['bowling_innings'] = 1
                player_stats[bowler_id]['runs_conceded'] += delivery['runs_total']
                player_stats[bowler_id]['wides_bowled'] += delivery['extras_wides']
                player_stats[bowler_id]['noballs_bowled'] += delivery['extras_noballs']
                
                if delivery['runs_total'] == 0:
                    player_stats[bowler_id]['dots_bowled'] += 1
                
                # Wickets (bowling credit)
                if delivery['is_wicket']:
                    wicket_type = delivery['wicket_type']
                    dismissed_id = delivery['dismissed_player_id']
                    
                    # Mark batter as out
                    if dismissed_id and dismissed_id in player_stats:
                        player_stats[dismissed_id]['not_out'] = False
                    
                    # Bowling wickets (excluding run outs)
                    if wicket_type and wicket_type.lower() != 'run out':
                        player_stats[bowler_id]['wickets_taken'] += 1
                    
                    # Fielding stats
                    fielder1_id = delivery['fielder1_id']
                    if fielder1_id:
                        if fielder1_id not in player_stats:
                            player_stats[fielder1_id] = {
                                'team_id': bowling_team_id,
                                'batting_innings': 0, 'runs_scored': 0, 'balls_faced': 0,
                                'fours_hit': 0, 'sixes_hit': 0, 'not_out': True, 'batting_position': None,
                                'bowling_innings': 0, 'overs_bowled': 0, 'runs_conceded': 0,
                                'wickets_taken': 0, 'maidens': 0, 'wides_bowled': 0, 'noballs_bowled': 0,
                                'dots_bowled': 0, 'catches': 0, 'run_outs': 0, 'stumpings': 0
                            }
                        
                        if wicket_type == 'caught':
                            player_stats[fielder1_id]['catches'] += 1
                        elif wicket_type == 'run out':
                            player_stats[fielder1_id]['run_outs'] += 1
                        elif wicket_type == 'stumped':
                            player_stats[fielder1_id]['stumpings'] += 1
        
        # Calculate overs bowled from balls bowled (need to query)
        cursor.execute("""
            SELECT bowler_id, COUNT(*) as balls,
                   SUM(CASE WHEN extras_wides > 0 OR extras_noballs > 0 THEN 0 ELSE 1 END) as legal_balls
            FROM deliveries d
            JOIN innings i ON d.innings_id = i.innings_id
            WHERE i.match_id = ?
            GROUP BY bowler_id
        """, (match_id,))
        
        for row in cursor.fetchall():
            bowler_id = row['bowler_id']
            if bowler_id in player_stats:
                legal_balls = row['legal_balls']
                overs = legal_balls // 6 + (legal_balls % 6) / 10.0
                player_stats[bowler_id]['overs_bowled'] = overs
        
        # Insert player match stats
        for player_id, stats in player_stats.items():
            cursor.execute("""
                INSERT OR REPLACE INTO player_match_stats (
                    match_id, player_id, team_id,
                    batting_innings, runs_scored, balls_faced, fours_hit, sixes_hit,
                    not_out, batting_position,
                    bowling_innings, overs_bowled, runs_conceded, wickets_taken,
                    maidens, wides_bowled, noballs_bowled, dots_bowled,
                    catches, run_outs, stumpings
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                match_id, player_id, stats['team_id'],
                stats['batting_innings'], stats['runs_scored'], stats['balls_faced'],
                stats['fours_hit'], stats['sixes_hit'], stats['not_out'], stats['batting_position'],
                stats['bowling_innings'], stats['overs_bowled'], stats['runs_conceded'],
                stats['wickets_taken'], stats['maidens'], stats['wides_bowled'],
                stats['noballs_bowled'], stats['dots_bowled'],
                stats['catches'], stats['run_outs'], stats['stumpings']
            ))


def ingest_matches(
    formats: Optional[List[str]] = None,
    limit: Optional[int] = None,
    force_reingest: bool = False
) -> Dict[str, int]:
    """
    Ingest cricket matches from downloaded JSON files.
    
    Args:
        formats: List of formats to ingest (e.g., ["t20i", "odi"])
        limit: Maximum number of matches to ingest (for testing)
        force_reingest: If True, reingest all matches
        
    Returns:
        Dictionary with ingestion statistics
    """
    if formats is None:
        formats = ["t20i", "odi"]
    
    # Initialize database
    init_database()
    
    ingestor = MatchIngestor()
    parser = CricsheetParser()
    
    total_processed = 0
    total_skipped = 0
    total_failed = 0
    
    for format_name in formats:
        logger.info(f"Processing {format_name.upper()} matches...")
        
        json_files = get_json_files(format_name)
        
        if not json_files:
            logger.warning(f"No JSON files found for {format_name}")
            continue
        
        # Apply limit if specified
        if limit:
            json_files = json_files[:limit]
        
        with get_db_connection() as conn:
            for file_path in tqdm(json_files, desc=f"{format_name.upper()}"):
                try:
                    match_data = parser.parse_match_file(file_path)
                    
                    if match_data is None:
                        total_failed += 1
                        continue
                    
                    if ingestor.ingest_match(conn, match_data):
                        total_processed += 1
                    else:
                        total_skipped += 1
                        
                except Exception as e:
                    logger.error(f"Failed to ingest {file_path.name}: {e}")
                    total_failed += 1
                    conn.rollback()
    
    stats = {
        'matches_processed': total_processed,
        'matches_skipped': total_skipped,
        'matches_failed': total_failed,
        'deliveries_inserted': ingestor.stats['deliveries_inserted']
    }
    
    logger.info(f"\nIngestion complete:")
    logger.info(f"  Matches processed: {total_processed}")
    logger.info(f"  Matches skipped: {total_skipped}")
    logger.info(f"  Matches failed: {total_failed}")
    logger.info(f"  Deliveries inserted: {ingestor.stats['deliveries_inserted']}")
    
    return stats


def main():
    """Main function to run data ingestion."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    logger.info("Starting data ingestion...")
    
    # Check if data has been downloaded
    t20_files = get_json_files("t20i")
    odi_files = get_json_files("odi")
    
    if not t20_files and not odi_files:
        logger.error("No data files found. Please run the downloader first:")
        logger.error("  python -m src.data.downloader")
        return 1
    
    # Run ingestion
    stats = ingest_matches()
    
    # Print database summary
    print_database_summary()
    
    return 0 if stats['matches_processed'] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

