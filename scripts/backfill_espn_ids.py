"""
Backfill ESPN Player IDs from ESPN Cricinfo.

This script:
1. Fetches current T20 matches from ESPN
2. Extracts player squads with ESPN IDs
3. Matches players to database using name matching
4. Updates database with ESPN player IDs
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.espn_scraper import ESPNCricInfoScraper
from src.data.database import get_connection
from config import DATABASE_PATH

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def backfill_espn_ids():
    """
    Backfill ESPN player IDs by scraping current matches.
    """
    logger.info("Starting ESPN ID backfill...")
    
    # Initialize scraper
    scraper = ESPNCricInfoScraper(request_delay=1.0)
    
    # Get upcoming T20 matches (next 7 days to get good coverage)
    logger.info("Fetching T20 schedule...")
    matches = scraper.get_t20_schedule(hours_ahead=168)  # 7 days
    logger.info(f"Found {len(matches)} upcoming matches")
    
    # Track updates
    total_players_found = 0
    total_players_matched = 0
    total_ids_updated = 0
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Process each match
    for i, match in enumerate(matches, 1):
        logger.info(f"\n[{i}/{len(matches)}] Processing: {match.title}")
        
        try:
            # Get match details (includes squads)
            details = scraper.get_match_details(match.match_url)
            
            if not details:
                logger.warning(f"Could not fetch details for {match.title}")
                continue
            
            # Match teams to database
            team1_db = scraper.match_team_to_db(details.team1, details.gender) if details.team1 else None
            team2_db = scraper.match_team_to_db(details.team2, details.gender) if details.team2 else None
            
            # Process both teams
            for team_num, (espn_team, db_team) in enumerate([
                (details.team1, team1_db),
                (details.team2, team2_db)
            ], 1):
                if not espn_team or not db_team:
                    continue
                
                team_name = db_team[1] if db_team else "Unknown"
                logger.info(f"  Team {team_num}: {team_name} - {len(espn_team.players)} players")
                
                # Match players to database
                matched_players = scraper.match_players_to_db(espn_team, team_name, details.gender)
                
                total_players_found += len(matched_players)
                
                for player in matched_players:
                    if not player.db_player_id:
                        continue  # Not matched
                    
                    total_players_matched += 1
                    
                    # Check if ESPN ID already set
                    cursor.execute(
                        "SELECT espn_player_id FROM players WHERE player_id = ?",
                        (player.db_player_id,)
                    )
                    row = cursor.fetchone()
                    
                    if row and row['espn_player_id']:
                        # Already has ESPN ID
                        if row['espn_player_id'] != player.espn_id:
                            logger.warning(
                                f"    ESPN ID mismatch for {player.name}: "
                                f"DB has {row['espn_player_id']}, ESPN has {player.espn_id}"
                            )
                        continue
                    
                    # Update ESPN ID
                    cursor.execute(
                        "UPDATE players SET espn_player_id = ? WHERE player_id = ?",
                        (player.espn_id, player.db_player_id)
                    )
                    total_ids_updated += 1
                    logger.info(f"    ✓ Updated {player.name} (DB ID: {player.db_player_id}) with ESPN ID: {player.espn_id}")
            
            # Commit after each match
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error processing match {match.title}: {e}")
            continue
    
    conn.close()
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("BACKFILL SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total players found in ESPN: {total_players_found}")
    logger.info(f"Total players matched to DB: {total_players_matched}")
    logger.info(f"Total ESPN IDs updated: {total_ids_updated}")
    logger.info(f"Match rate: {total_players_matched/total_players_found*100:.1f}%")
    
    # Verify
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as total, COUNT(espn_player_id) as with_espn FROM players")
    row = cursor.fetchone()
    conn.close()
    
    logger.info(f"\nDatabase status:")
    logger.info(f"  Total players: {row['total']}")
    logger.info(f"  Players with ESPN ID: {row['with_espn']}")
    logger.info(f"  Coverage: {row['with_espn']/row['total']*100:.1f}%")


if __name__ == "__main__":
    backfill_espn_ids()

