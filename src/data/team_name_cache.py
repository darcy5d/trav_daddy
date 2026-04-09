"""
CREX Team Name Cache (Legacy)

DEPRECATED: This module is replaced by team_variant_resolver.py which provides
comprehensive team variant handling (A-teams, Women's teams, U19, etc.) with
proper parent team database integration.

For backward compatibility during migration, this module wraps the new system.
"""

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.api.crex_scraper import CREXScraper, CREXTeam

from src.data.database import get_connection

logger = logging.getLogger(__name__)


@dataclass
class CachedTeamResult:
    """Result from team cache lookup with database integration info."""
    fkey: str
    gender: str
    display_name: str
    db_team_id: Optional[int] = None  # None = no database match
    db_team_name: Optional[str] = None
    match_confidence: Optional[float] = None
    needs_default_elo: bool = True  # True when db_team_id is None
    source: str = 'unknown'
    last_validated: Optional[datetime] = None
    
    def __post_init__(self):
        self.needs_default_elo = self.db_team_id is None


class CREXTeamCache:
    """
    Persistent cache for CREX team fkey → display name + database match mappings.
    
    Enhances schedule display with readable team names while preserving all existing
    team matching, warning, and default ELO behavior for the prediction pipeline.
    """
    
    def __init__(self):
        self._ensure_table_exists()
    
    def _ensure_table_exists(self):
        """Ensure the cache table exists (handles schema migration)."""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            # Test if table exists and has correct schema
            cursor.execute("SELECT fkey, gender, display_name FROM crex_team_cache LIMIT 1")
            conn.close()
        except sqlite3.OperationalError:
            # Table doesn't exist or has wrong schema - should be handled by schema migration
            logger.warning("crex_team_cache table not found - ensure database schema is up to date")
    
    def get_display_name(self, fkey: str, gender: str = 'male') -> Optional[str]:
        """Quick lookup for display name only (for UI performance)."""
        if not fkey:
            return None
            
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT display_name 
                FROM crex_team_cache 
                WHERE fkey = ? AND gender = ?
            """, (fkey, gender))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                # Update last_seen timestamp
                self._update_last_seen(fkey, gender)
                return row[0]
                
            return None
            
        except sqlite3.OperationalError as e:
            logger.debug(f"Cache lookup failed for {fkey}: {e}")
            return None
    
    def get_cached_team(self, fkey: str, gender: str = 'male') -> Optional[CachedTeamResult]:
        """Full lookup with database integration info."""
        if not fkey:
            return None
            
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT fkey, gender, display_name, db_team_id, db_team_name, 
                       match_confidence, source, last_validated
                FROM crex_team_cache 
                WHERE fkey = ? AND gender = ?
            """, (fkey, gender))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                # Update last_seen timestamp
                self._update_last_seen(fkey, gender)
                
                return CachedTeamResult(
                    fkey=row[0],
                    gender=row[1], 
                    display_name=row[2],
                    db_team_id=row[3],
                    db_team_name=row[4],
                    match_confidence=row[5],
                    source=row[6],
                    last_validated=datetime.fromisoformat(row[7]) if row[7] else None
                )
                
            return None
            
        except sqlite3.OperationalError as e:
            logger.debug(f"Cache lookup failed for {fkey}: {e}")
            return None
    
    def bulk_resolve_teams(
        self, fkey_list: List[str], gender: str, scraper: 'CREXScraper'
    ) -> Dict[str, CachedTeamResult]:
        """Batch lookup for multiple teams (optimized for schedule display)."""
        results = {}
        
        if not fkey_list:
            return results
        
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            # Batch query for existing cache entries
            placeholders = ','.join(['?'] * len(fkey_list))
            cursor.execute(f"""
                SELECT fkey, gender, display_name, db_team_id, db_team_name, 
                       match_confidence, source, last_validated
                FROM crex_team_cache 
                WHERE fkey IN ({placeholders}) AND gender = ?
            """, fkey_list + [gender])
            
            rows = cursor.fetchall()
            conn.close()
            
            # Process cached results
            cached_fkeys = set()
            for row in rows:
                result = CachedTeamResult(
                    fkey=row[0],
                    gender=row[1],
                    display_name=row[2], 
                    db_team_id=row[3],
                    db_team_name=row[4],
                    match_confidence=row[5],
                    source=row[6],
                    last_validated=datetime.fromisoformat(row[7]) if row[7] else None
                )
                results[row[0]] = result
                cached_fkeys.add(row[0])
            
            # Update last_seen for found entries
            if cached_fkeys:
                self._bulk_update_last_seen(list(cached_fkeys), gender)
            
            # For cache misses, try to populate from abbreviations 
            missing_fkeys = [fkey for fkey in fkey_list if fkey not in cached_fkeys]
            if missing_fkeys:
                self._populate_from_abbreviations(missing_fkeys, gender, scraper, results)
                
        except sqlite3.OperationalError as e:
            logger.debug(f"Bulk cache lookup failed: {e}")
            
        return results
    
    def cache_team_from_crex(
        self, fkey: str, crex_team: 'CREXTeam', gender: str, scraper: 'CREXScraper'
    ) -> CachedTeamResult:
        """Cache team from CREXTeam object with database validation."""
        display_name = crex_team.name or crex_team.abbreviation or fkey
        
        # Validate against database using existing pipeline
        db_match = scraper.match_team_to_db(crex_team, gender)
        
        return self._store_validated_mapping(
            fkey, display_name, db_match, 'crex_team', gender
        )
    
    def cache_team_from_names(
        self, fkey: str, display_name: str, gender: str, source: str, scraper: 'CREXScraper'
    ) -> CachedTeamResult:
        """Cache team from display name string with database validation."""
        from src.api.crex_scraper import CREXTeam
        
        # Create temporary team for validation
        temp_team = CREXTeam(
            crex_id=fkey,
            name=display_name,
            abbreviation=fkey
        )
        
        # Validate against database
        db_match = scraper.match_team_to_db(temp_team, gender)
        
        return self._store_validated_mapping(
            fkey, display_name, db_match, source, gender
        )
    
    def populate_from_match_detail(self, match_detail: 'CREXMatch', scraper: 'CREXScraper'):
        """Extract team mappings from match detail (already validated through match_team_to_db)."""
        if match_detail.team1 and match_detail.team1_id:
            self.cache_team_from_crex(
                match_detail.team1_id, match_detail.team1, match_detail.gender, scraper
            )
        
        if match_detail.team2 and match_detail.team2_id:
            self.cache_team_from_crex(
                match_detail.team2_id, match_detail.team2, match_detail.gender, scraper
            )
    
    def populate_from_global_num(self, fx: dict, scraper: 'CREXScraper'):
        """Extract team mapping from API row global_num hint."""
        gn = fx.get('global_num')
        if not isinstance(gn, dict) or not gn.get('fr'):
            return
            
        fri = (gn.get('fri') or '').lstrip('^')
        fr_name = str(gn.get('fr')).strip()
        
        if not (fri and fr_name):
            return
        
        # Determine gender from fx
        g = fx.get('g')
        gender = 'female' if g == 0 else 'male'
        
        self.cache_team_from_names(fri, fr_name, gender, 'global_num', scraper)
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics for monitoring and optimization."""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_entries,
                    COUNT(CASE WHEN db_team_id IS NOT NULL THEN 1 END) as matched_entries,
                    COUNT(CASE WHEN db_team_id IS NULL THEN 1 END) as unknown_entries,
                    AVG(match_confidence) as avg_confidence,
                    COUNT(DISTINCT source) as source_count
                FROM crex_team_cache
            """)
            
            row = cursor.fetchone()
            
            cursor.execute("""
                SELECT source, COUNT(*) as count
                FROM crex_team_cache  
                GROUP BY source
            """)
            
            sources = dict(cursor.fetchall())
            conn.close()
            
            if row:
                total = row[0] or 0
                matched = row[1] or 0
                return {
                    'total_entries': total,
                    'matched_entries': matched,
                    'unknown_entries': row[2] or 0,
                    'match_rate': (matched / total * 100) if total > 0 else 0.0,
                    'avg_confidence': row[3] or 0.0,
                    'sources': sources
                }
            
        except sqlite3.OperationalError as e:
            logger.debug(f"Cache stats failed: {e}")
            
        return {'total_entries': 0, 'matched_entries': 0, 'unknown_entries': 0, 'match_rate': 0.0}
    
    def get_unknown_teams(self, limit: int = 50) -> List[Dict]:
        """Get list of teams with no database match (for manual review)."""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT fkey, display_name, gender, source, last_seen
                FROM crex_team_cache
                WHERE db_team_id IS NULL
                ORDER BY last_seen DESC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [
                {
                    'fkey': row[0],
                    'display_name': row[1], 
                    'gender': row[2],
                    'source': row[3],
                    'last_seen': row[4]
                }
                for row in rows
            ]
            
        except sqlite3.OperationalError as e:
            logger.debug(f"Unknown teams query failed: {e}")
            return []
    
    def invalidate_team(self, fkey: str, gender: str = None):
        """Force re-validation on next lookup."""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            if gender:
                cursor.execute("""
                    UPDATE crex_team_cache 
                    SET last_validated = NULL 
                    WHERE fkey = ? AND gender = ?
                """, (fkey, gender))
            else:
                cursor.execute("""
                    UPDATE crex_team_cache 
                    SET last_validated = NULL 
                    WHERE fkey = ?
                """, (fkey,))
            
            conn.commit()
            conn.close()
            
        except sqlite3.OperationalError as e:
            logger.debug(f"Cache invalidation failed: {e}")
    
    # Private helper methods
    
    def _store_validated_mapping(
        self, fkey: str, display_name: str, db_match: Optional[Tuple[int, str]], 
        source: str, gender: str
    ) -> CachedTeamResult:
        """Store mapping with database validation results."""
        db_team_id = db_match[0] if db_match else None
        db_team_name = db_match[1] if db_match else None
        confidence = 1.0 if db_match else None
        
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO crex_team_cache 
                (fkey, gender, display_name, db_team_id, db_team_name, 
                 match_confidence, source, last_validated, last_seen, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fkey, gender, display_name, db_team_id, db_team_name,
                confidence, source, datetime.utcnow().isoformat(),
                datetime.utcnow().isoformat(), datetime.utcnow().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Cached team {fkey} -> {display_name} (db_match: {bool(db_match)})")
            
        except sqlite3.OperationalError as e:
            logger.warning(f"Failed to cache team {fkey}: {e}")
        
        return CachedTeamResult(
            fkey=fkey,
            gender=gender,
            display_name=display_name,
            db_team_id=db_team_id,
            db_team_name=db_team_name,
            match_confidence=confidence,
            source=source,
            last_validated=datetime.utcnow()
        )
    
    def _update_last_seen(self, fkey: str, gender: str):
        """Update last_seen timestamp for cache hit tracking."""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE crex_team_cache 
                SET last_seen = ? 
                WHERE fkey = ? AND gender = ?
            """, (datetime.utcnow().isoformat(), fkey, gender))
            
            conn.commit()
            conn.close()
            
        except sqlite3.OperationalError:
            pass  # Non-critical update
    
    def _bulk_update_last_seen(self, fkey_list: List[str], gender: str):
        """Bulk update last_seen timestamps."""
        if not fkey_list:
            return
            
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            placeholders = ','.join(['?'] * len(fkey_list))
            cursor.execute(f"""
                UPDATE crex_team_cache 
                SET last_seen = ? 
                WHERE fkey IN ({placeholders}) AND gender = ?
            """, [datetime.utcnow().isoformat()] + fkey_list + [gender])
            
            conn.commit()
            conn.close()
            
        except sqlite3.OperationalError:
            pass  # Non-critical update
    
    def _populate_from_abbreviations(
        self, fkey_list: List[str], gender: str, scraper: 'CREXScraper', results: Dict
    ):
        """Try to populate cache misses from known abbreviations."""
        # Get team abbreviations from scraper (need to extract from match_team_to_db method)
        known_abbrevs = self._get_known_abbreviations()
        
        for fkey in fkey_list:
            fkey_lower = fkey.lower()
            if fkey_lower in known_abbrevs:
                display_name = known_abbrevs[fkey_lower]
                cached_result = self.cache_team_from_names(
                    fkey, display_name, gender, 'abbreviation', scraper
                )
                results[fkey] = cached_result
                logger.debug(f"Populated cache from abbreviation: {fkey} -> {display_name}")
    
    def populate_initial_mappings(self, scraper: 'CREXScraper'):
        """Seed cache from existing abbreviations, validating against database."""
        logger.info("Populating initial team name cache mappings...")
        
        known_abbrevs = self._get_known_abbreviations()
        populated_count = 0
        
        for fkey_lower, display_name in known_abbrevs.items():
            for gender in ['male', 'female']:
                # Check if already cached
                existing = self.get_cached_team(fkey_lower.upper(), gender)
                if existing:
                    continue
                
                try:
                    # Cache with validation
                    self.cache_team_from_names(fkey_lower.upper(), display_name, gender, 'abbreviation', scraper)
                    populated_count += 1
                except Exception as e:
                    logger.debug(f"Failed to populate {fkey_lower}: {e}")
        
        logger.info(f"Populated {populated_count} initial team mappings")
    
    def _get_known_abbreviations(self) -> Dict[str, str]:
        """Extract known abbreviations mapping (simplified version from scraper)."""
        # This is a subset of the full abbreviations from CREXScraper.match_team_to_db
        return {
            # International teams (common ones likely to appear in CREX)
            'hk': 'Hong Kong', 'bhu': 'Bhutan', 'mas': 'Malaysia', 'sin': 'Singapore',
            'tha': 'Thailand', 'mya': 'Myanmar', 'jpn': 'Japan', 'kor': 'Korea',
            'idn': 'Indonesia', 'uga': 'Uganda', 'tan': 'Tanzania', 'rwa': 'Rwanda',
            'nig': 'Nigeria', 'bot': 'Botswana', 'gha': 'Ghana', 'cam': 'Cameroon',
            'qat': 'Qatar', 'bhr': 'Bahrain', 'kwt': 'Kuwait', 'sau': 'Saudi Arabia',
            'mdv': 'Maldives', 'irn': 'Iran', 'ger': 'Germany', 'aut': 'Austria',
            'ita': 'Italy', 'fra': 'France', 'bel': 'Belgium', 'esp': 'Spain',
            # Associate nations
            'ber': 'Bermuda', 'cay': 'Cayman Islands', 'bar': 'Barbados',
            'jam': 'Jamaica', 'guy': 'Guyana', 'cyp': 'Cyprus', 'mlt': 'Malta',
            # Women's Super Smash (likely to appear in CREX)
            'ahw': 'Auckland', 'wbw': 'Wellington', 'cmw': 'Canterbury',
            'chw': 'Central Districts', 'nbw': 'Northern Districts', 'osw': 'Otago',
            # Common additional mappings
            'pak': 'Pakistan', 'ind': 'India', 'aus': 'Australia', 'eng': 'England',
            'nz': 'New Zealand', 'sa': 'South Africa', 'wi': 'West Indies',
            'sl': 'Sri Lanka', 'ban': 'Bangladesh', 'afg': 'Afghanistan',
            'zim': 'Zimbabwe', 'ire': 'Ireland', 'sco': 'Scotland', 'ned': 'Netherlands',
            'uae': 'United Arab Emirates', 'nam': 'Namibia', 'oma': 'Oman',
            'usa': 'United States of America', 'can': 'Canada', 'ken': 'Kenya',
            'nep': 'Nepal', 'png': 'Papua New Guinea',
        }