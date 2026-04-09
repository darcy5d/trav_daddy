"""
CREX Team Variant Resolution System

Resolves CREX team fkeys (like "1FO", "Q") to proper display names while preserving
database integration through parent team mapping for training data.

Key principles:
- Display: Show full variant names ("Hong Kong A", "Australia Women")  
- Database: Use parent team training data ("Hong Kong", "Australia")
- Warnings: Flag when variants have no training data or unknown parent teams
"""

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.api.crex_scraper import CREXScraper, CREXTeam

from src.data.database import get_connection

logger = logging.getLogger(__name__)


@dataclass  
class TeamVariantResult:
    """Result from team variant resolution with database integration."""
    fkey: str
    full_name: str              # Display name: "Hong Kong A", "Australia Women"
    parent_team: str           # Database lookup name: "Hong Kong", "Australia" 
    team_type: str             # 'main', 'a-team', 'women', 'u19', etc.
    gender: str                # 'male', 'female'
    db_team_id: Optional[int] = None        # Parent team's database ID
    db_team_name: Optional[str] = None      # Parent team's database name
    match_confidence: float = 1.0           # Confidence in parent mapping
    uses_parent_data: bool = True           # True for variants using parent training data
    needs_default_elo: bool = False         # True when no database match at all
    
    def __post_init__(self):
        """Set derived fields."""
        self.uses_parent_data = self.team_type != 'main'
        self.needs_default_elo = self.db_team_id is None


class CREXTeamVariantResolver:
    """
    Resolves CREX team fkeys to variants with proper database integration.
    
    Maintains display accuracy ("Hong Kong A") while using appropriate training data
    (Hong Kong's T20 history for Hong Kong A predictions).
    """
    
    def __init__(self):
        self._ensure_table_exists()
    
    def _ensure_table_exists(self):
        """Ensure the team variants table exists."""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT fkey FROM crex_team_variants LIMIT 1")
            conn.close()
        except sqlite3.OperationalError:
            logger.warning("crex_team_variants table not found - run migration script")
    
    def resolve_team(self, fkey: str, gender: str = None) -> Optional[TeamVariantResult]:
        """
        Resolve CREX fkey to team variant with database integration.
        
        Args:
            fkey: CREX team fkey (e.g., "1FO", "Q")
            gender: Optional gender hint for ambiguous cases
            
        Returns:
            TeamVariantResult with display name and database mapping
        """
        if not fkey:
            return None
        
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            # Look up variant in database (includes both directory and tournament teams)
            if gender:
                cursor.execute("""
                    SELECT fkey, full_name, parent_team, team_type, gender,
                           db_team_id, db_team_name, match_confidence,
                           COALESCE(is_tournament_team, FALSE) as is_tournament
                    FROM crex_team_variants
                    WHERE fkey = ? AND gender = ?
                """, (fkey, gender))
            else:
                cursor.execute("""
                    SELECT fkey, full_name, parent_team, team_type, gender,
                           db_team_id, db_team_name, match_confidence,
                           COALESCE(is_tournament_team, FALSE) as is_tournament
                    FROM crex_team_variants
                    WHERE fkey = ?
                    ORDER BY CASE WHEN gender = 'male' THEN 1 ELSE 2 END
                    LIMIT 1
                """, (fkey,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return TeamVariantResult(
                    fkey=row[0],
                    full_name=row[1],
                    parent_team=row[2], 
                    team_type=row[3],
                    gender=row[4],
                    db_team_id=row[5],
                    db_team_name=row[6],
                    match_confidence=row[7] or 1.0
                )
            
            return None
            
        except sqlite3.OperationalError as e:
            logger.debug(f"Team variant lookup failed for {fkey}: {e}")
            return None
    
    def resolve_with_learning(
        self, fkey: str, gender: str, matches: List['CREXMatch'], scraper: 'CREXScraper'
    ) -> Optional[TeamVariantResult]:
        """
        Resolve team with fallback to tournament learning.
        
        1. Try variants database (permanent + cached tournament teams)
        2. If miss, trigger on-demand learning from match details
        """
        # First try normal resolution
        result = self.resolve_team(fkey, gender)
        if result:
            return result
        
        # If not found, try to learn from current matches
        from src.data.tournament_team_learner import TournamentTeamLearner
        learner = TournamentTeamLearner(max_learn_per_session=1)  # Learn just this one
        
        learned_mappings = learner.learn_tournament_teams([fkey], matches, scraper)
        
        if fkey in learned_mappings:
            # Try resolution again after learning
            return self.resolve_team(fkey, gender)
        
        return None
    
    def bulk_resolve_teams(self, fkeys: List[str], gender: str = 'male') -> Dict[str, TeamVariantResult]:
        """
        Batch resolve multiple team fkeys for performance.
        
        Args:
            fkeys: List of CREX team fkeys
            gender: Default gender for lookups
            
        Returns:
            Dict mapping fkey to TeamVariantResult
        """
        results = {}
        
        if not fkeys:
            return results
        
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            # Batch query with gender preference
            placeholders = ','.join(['?'] * len(fkeys))
            cursor.execute(f"""
                SELECT fkey, full_name, parent_team, team_type, gender,
                       db_team_id, db_team_name, match_confidence
                FROM crex_team_variants
                WHERE fkey IN ({placeholders})
                ORDER BY fkey, CASE WHEN gender = ? THEN 1 ELSE 2 END
            """, fkeys + [gender])
            
            rows = cursor.fetchall()
            conn.close()
            
            # Process results, taking first match per fkey (gender-preferred)
            seen_fkeys = set()
            for row in rows:
                fkey = row[0]
                if fkey in seen_fkeys:
                    continue  # Already have preferred match
                seen_fkeys.add(fkey)
                
                results[fkey] = TeamVariantResult(
                    fkey=row[0],
                    full_name=row[1], 
                    parent_team=row[2],
                    team_type=row[3],
                    gender=row[4],
                    db_team_id=row[5],
                    db_team_name=row[6], 
                    match_confidence=row[7] or 1.0
                )
            
        except sqlite3.OperationalError as e:
            logger.debug(f"Bulk team variant lookup failed: {e}")
        
        return results
    
    def populate_variants_from_directory(self, scraper: 'CREXScraper') -> int:
        """
        Populate team variants table from comprehensive CREX directory scrape.
        
        Returns:
            Number of variants populated
        """
        from src.data.crex_team_directory import CREXTeamDirectoryScraper
        
        logger.info("Populating team variants from CREX directory...")
        
        # Scrape comprehensive team variants
        directory_scraper = CREXTeamDirectoryScraper(request_delay=0.3)
        variants = directory_scraper.scrape_all_teams()
        
        if not variants:
            logger.error("No team variants found from directory")
            return 0
        
        # Validate each variant against database and populate
        populated_count = 0
        conn = get_connection()
        
        try:
            cursor = conn.cursor()
            
            for variant in variants:
                # Create temporary team for database validation
                from src.api.crex_scraper import CREXTeam
                temp_team = CREXTeam(
                    crex_id=variant.fkey,
                    name=variant.parent_team,  # Use parent for database lookup
                    abbreviation=variant.fkey
                )
                
                # Validate through existing pipeline
                db_match = scraper.match_team_to_db(temp_team, variant.gender)
                
                # Store variant with validation results  
                db_team_id = db_match[0] if db_match else None
                db_team_name = db_match[1] if db_match else None
                confidence = 1.0 if db_match else None
                
                cursor.execute("""
                    INSERT OR REPLACE INTO crex_team_variants
                    (fkey, full_name, short_name, parent_team, team_type, gender,
                     db_team_id, db_team_name, match_confidence, source, 
                     created_at, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    variant.fkey, variant.full_name, variant.short_name,
                    variant.parent_team, variant.team_type, variant.gender,
                    db_team_id, db_team_name, confidence, 'directory',
                    datetime.utcnow().isoformat(), datetime.utcnow().isoformat()
                ))
                
                populated_count += 1
                
                if db_match:
                    logger.debug(f"✅ {variant.fkey}: {variant.full_name} → DB:{db_team_name}")
                else:
                    logger.debug(f"⚠️ {variant.fkey}: {variant.full_name} → No DB match (parent: {variant.parent_team})")
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to populate variants: {e}")
            conn.rollback()
        finally:
            conn.close()
        
        logger.info(f"Populated {populated_count} team variants")
        return populated_count
    
    def get_variant_stats(self) -> Dict:
        """Get statistics about team variant coverage and database matches."""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    team_type,
                    COUNT(*) as total,
                    COUNT(CASE WHEN db_team_id IS NOT NULL THEN 1 END) as matched,
                    COUNT(CASE WHEN db_team_id IS NULL THEN 1 END) as unmatched
                FROM crex_team_variants
                GROUP BY team_type
                ORDER BY total DESC
            """)
            
            type_stats = {}
            for row in cursor.fetchall():
                type_stats[row[0]] = {
                    'total': row[1],
                    'matched': row[2],
                    'unmatched': row[3],
                    'match_rate': (row[2] / row[1] * 100) if row[1] > 0 else 0
                }
            
            cursor.execute("""
                SELECT COUNT(*) as total_variants,
                       COUNT(CASE WHEN db_team_id IS NOT NULL THEN 1 END) as total_matched
                FROM crex_team_variants
            """)
            
            row = cursor.fetchone()
            overall = {
                'total_variants': row[0],
                'total_matched': row[1],
                'overall_match_rate': (row[1] / row[0] * 100) if row[0] > 0 else 0
            }
            
            conn.close()
            
            return {
                'overall': overall,
                'by_type': type_stats
            }
            
        except sqlite3.OperationalError as e:
            logger.debug(f"Variant stats query failed: {e}")
            return {'overall': {'total_variants': 0}, 'by_type': {}}
    
    def get_unmatched_variants(self, limit: int = 20) -> List[Dict]:
        """Get team variants with no database match (need manual attention)."""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT fkey, full_name, parent_team, team_type, gender
                FROM crex_team_variants  
                WHERE db_team_id IS NULL
                ORDER BY team_type, full_name
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [
                {
                    'fkey': row[0],
                    'full_name': row[1],
                    'parent_team': row[2],
                    'team_type': row[3],
                    'gender': row[4]
                }
                for row in rows
            ]
            
        except sqlite3.OperationalError as e:
            logger.debug(f"Unmatched variants query failed: {e}")
            return []