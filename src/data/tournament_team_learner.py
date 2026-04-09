"""
Tournament Team Learning System

On-demand learning for tournament-specific team codes (like ZF, ZE, ZI, ZG) 
that aren't in the permanent CREX team directory but can be resolved via 
match detail pages.

Key features:
- Identifies unresolved cryptic codes in schedule
- Learns team names by fetching match detail pages (rate-limited)
- Caches tournament team mappings for future use
- Integrates with existing team variant system
"""

import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from src.api.crex_scraper import CREXScraper, CREXMatch

from src.data.database import get_connection

logger = logging.getLogger(__name__)


class TournamentTeamLearner:
    """
    Learns tournament-specific team codes by mining match detail pages.
    
    Designed to work alongside the comprehensive team variants system to handle
    dynamic/tournament codes that don't appear in the permanent CREX directory.
    """
    
    def __init__(self, max_learn_per_session: int = 5):
        """
        Initialize learner with rate limiting.
        
        Args:
            max_learn_per_session: Max teams to learn per schedule fetch (rate limiting)
        """
        self.max_learn_per_session = max_learn_per_session
        self._ensure_schema_ready()
    
    def _ensure_schema_ready(self):
        """Ensure tournament learning fields exist in variants table."""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            # Check if tournament fields exist
            cursor.execute("PRAGMA table_info(crex_team_variants)")
            columns = {row[1] for row in cursor.fetchall()}
            
            # Add tournament learning fields if missing
            if 'is_tournament_team' not in columns:
                cursor.execute("ALTER TABLE crex_team_variants ADD COLUMN is_tournament_team BOOLEAN DEFAULT FALSE")
                cursor.execute("ALTER TABLE crex_team_variants ADD COLUMN learned_from_match_id TEXT")
                cursor.execute("ALTER TABLE crex_team_variants ADD COLUMN tournament_series TEXT")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_variants_tournament ON crex_team_variants(is_tournament_team)")
                conn.commit()
                logger.info("Added tournament learning fields to variants table")
            
            conn.close()
            
        except sqlite3.OperationalError as e:
            logger.debug(f"Tournament schema check failed: {e}")
    
    def identify_unresolved_codes(self, matches: List['CREXMatch']) -> List[str]:
        """
        Find team fkeys that are still showing as cryptic codes in schedule.
        
        Args:
            matches: List of schedule matches
            
        Returns:
            List of fkeys that need resolution (showing as themselves in display names)
        """
        unresolved = set()
        
        for match in matches:
            # Team name is cryptic if it's the same as the fkey (or very short/uppercase)
            if (match.team1_id and match.team1_name and 
                (match.team1_name == match.team1_id or 
                 (len(match.team1_name) <= 3 and match.team1_name.isupper()))):
                unresolved.add(match.team1_id)
            
            if (match.team2_id and match.team2_name and  
                (match.team2_name == match.team2_id or
                 (len(match.team2_name) <= 3 and match.team2_name.isupper()))):
                unresolved.add(match.team2_id)
        
        return list(unresolved)
    
    def learn_tournament_teams(
        self, unresolved_fkeys: List[str], matches: List['CREXMatch'], scraper: 'CREXScraper'
    ) -> Dict[str, str]:
        """
        Learn tournament team names by fetching match details for unresolved codes.
        
        Args:
            unresolved_fkeys: Team fkeys needing resolution
            matches: Schedule matches (to find representative matches)
            scraper: CREX scraper for match detail fetching
            
        Returns:
            Dict mapping fkey to learned team name
        """
        learned_mappings = {}
        
        if not unresolved_fkeys:
            return learned_mappings
        
        # Limit learning per session for rate limiting
        fkeys_to_learn = unresolved_fkeys[:self.max_learn_per_session]
        
        logger.info(f"Learning {len(fkeys_to_learn)} tournament teams: {fkeys_to_learn}")
        
        # Find representative matches for each unresolved fkey
        fkey_to_matches = {}
        for match in matches:
            if match.team1_id in fkeys_to_learn:
                if match.team1_id not in fkey_to_matches:
                    fkey_to_matches[match.team1_id] = []
                fkey_to_matches[match.team1_id].append(match)
            
            if match.team2_id in fkeys_to_learn:  
                if match.team2_id not in fkey_to_matches:
                    fkey_to_matches[match.team2_id] = []
                fkey_to_matches[match.team2_id].append(match)
        
        # Learn from match details
        learned_count = 0
        for fkey in fkeys_to_learn:
            if fkey not in fkey_to_matches:
                continue
                
            if learned_count >= self.max_learn_per_session:
                break
            
            # Use first match as learning source
            representative_match = fkey_to_matches[fkey][0]
            
            try:
                logger.debug(f"Learning {fkey} from {representative_match.match_url}")
                match_details = scraper.get_match_details(representative_match.match_url)
                
                if match_details and match_details.team1_name and match_details.team2_name:
                    # Map fkey to the correct team name
                    learned_name = None
                    if (match_details.team1_id == fkey or 
                        representative_match.team1_id == fkey):
                        learned_name = match_details.team1_name
                    elif (match_details.team2_id == fkey or  
                          representative_match.team2_id == fkey):
                        learned_name = match_details.team2_name
                    
                    if learned_name and learned_name != fkey:
                        learned_mappings[fkey] = learned_name
                        self._cache_tournament_team(
                            fkey, learned_name, representative_match, scraper
                        )
                        learned_count += 1
                        logger.info(f"✅ Learned {fkey} → {learned_name}")
                    else:
                        logger.debug(f"❌ Could not extract name for {fkey}")
                
            except Exception as e:
                logger.debug(f"Failed to learn {fkey}: {e}")
                continue
        
        logger.info(f"Successfully learned {len(learned_mappings)} tournament teams")
        return learned_mappings
    
    def _cache_tournament_team(
        self, fkey: str, learned_name: str, source_match: 'CREXMatch', scraper: 'CREXScraper'
    ):
        """Cache learned tournament team with database validation."""
        from src.api.crex_scraper import CREXTeam
        
        # Create temp team for database validation
        temp_team = CREXTeam(
            crex_id=fkey,
            name=learned_name,
            abbreviation=fkey
        )
        
        # Validate against database
        db_match = scraper.match_team_to_db(temp_team, source_match.gender)
        
        # Store in variants table as tournament team
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO crex_team_variants
                (fkey, full_name, short_name, parent_team, team_type, gender,
                 db_team_id, db_team_name, match_confidence, source, 
                 is_tournament_team, learned_from_match_id, tournament_series,
                 created_at, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fkey, learned_name, fkey, learned_name,  # Tournament teams are their own "parent"
                'tournament', source_match.gender,
                db_match[0] if db_match else None,
                db_match[1] if db_match else None, 
                1.0 if db_match else None,
                'tournament_learning',
                True, source_match.crex_id, source_match.series_id,
                datetime.utcnow().isoformat(), datetime.utcnow().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except sqlite3.OperationalError as e:
            logger.debug(f"Failed to cache tournament team {fkey}: {e}")
    
    def get_learning_stats(self) -> Dict:
        """Get statistics about tournament team learning."""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as tournament_teams,
                    COUNT(CASE WHEN db_team_id IS NOT NULL THEN 1 END) as matched,
                    COUNT(DISTINCT tournament_series) as tournaments
                FROM crex_team_variants  
                WHERE is_tournament_team = TRUE
            """)
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'tournament_teams': row[0],
                    'matched_to_db': row[1],
                    'tournaments': row[2],
                    'match_rate': (row[1] / row[0] * 100) if row[0] > 0 else 0.0
                }
            
        except sqlite3.OperationalError:
            pass
        
        return {'tournament_teams': 0, 'matched_to_db': 0, 'tournaments': 0, 'match_rate': 0.0}