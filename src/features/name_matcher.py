"""
Player Name Matcher.

Matches player names from external sources (Cricket Data API) to our database.
Handles challenges like:
- "Amanda-Jade Wellington" -> "A Wellington"
- "Laura Wolvaardt" -> "L Wolvaardt"
- "Megan Schutt" -> "ML Schutt"

Uses fuzzy matching and team context to find the best match.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from difflib import SequenceMatcher

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.database import get_connection

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of a name match."""
    player_id: int
    db_name: str
    api_name: str
    score: float
    method: str  # 'exact', 'surname', 'fuzzy', 'initials'


class PlayerNameMatcher:
    """
    Matches player names from API to database player IDs.
    
    Uses multiple strategies:
    1. Exact match
    2. Surname match (last word)
    3. Initials + surname match
    4. Fuzzy match with threshold
    """
    
    FUZZY_THRESHOLD = 0.7
    
    def __init__(self):
        self._player_cache: Dict[int, Dict] = {}
        self._team_players: Dict[str, List[Dict]] = {}
        self._initialized = False
    
    def _initialize(self):
        """Load all players from database."""
        if self._initialized:
            return
        
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get all players with their teams (for both male and female)
        cursor.execute("""
            SELECT DISTINCT 
                p.player_id, 
                p.name,
                p.espn_player_id,
                t.name as team_name,
                m.gender
            FROM players p
            JOIN player_match_stats pms ON p.player_id = pms.player_id
            JOIN teams t ON pms.team_id = t.team_id
            JOIN matches m ON pms.match_id = m.match_id
            WHERE m.match_type = 'T20'
        """)
        
        for row in cursor.fetchall():
            player = {
                'player_id': row['player_id'],
                'name': row['name'],
                'espn_player_id': row['espn_player_id'],
                'team_name': row['team_name'],
                'gender': row['gender']
            }
            
            self._player_cache[row['player_id']] = player
            
            # Index by team
            team = row['team_name']
            if team not in self._team_players:
                self._team_players[team] = []
            
            # Avoid duplicates in team list
            if not any(p['player_id'] == player['player_id'] for p in self._team_players[team]):
                self._team_players[team].append(player)
        
        conn.close()
        self._initialized = True
        
        logger.info(f"Loaded {len(self._player_cache)} players from {len(self._team_players)} teams")
    
    def _normalize_name(self, name: str) -> str:
        """Normalize a name for comparison."""
        # Remove hyphens, extra spaces, lowercase
        name = name.lower()
        name = re.sub(r'[-]', ' ', name)
        name = re.sub(r'\s+', ' ', name).strip()
        return name
    
    def _get_surname(self, name: str) -> str:
        """Extract surname (last word) from name."""
        parts = name.strip().split()
        return parts[-1].lower() if parts else ''
    
    def _get_initials_surname(self, name: str) -> Tuple[str, str]:
        """Get initials and surname from full name."""
        parts = name.strip().split()
        if not parts:
            return '', ''
        
        surname = parts[-1].lower()
        initials = ''.join(p[0].lower() for p in parts[:-1])
        return initials, surname
    
    def _fuzzy_score(self, name1: str, name2: str) -> float:
        """Calculate fuzzy match score between two names."""
        return SequenceMatcher(None, self._normalize_name(name1), self._normalize_name(name2)).ratio()
    
    def _match_db_name_format(self, api_name: str, db_name: str) -> float:
        """
        Score how well API name matches DB name format.
        
        DB format is typically: "A Wellington" (initial + surname)
        API format is: "Amanda-Jade Wellington" (full name)
        """
        api_parts = api_name.replace('-', ' ').split()
        db_parts = db_name.split()
        
        if not api_parts or not db_parts:
            return 0.0
        
        api_surname = api_parts[-1].lower()
        db_surname = db_parts[-1].lower()
        
        # Surname must match
        if api_surname != db_surname:
            return 0.0
        
        # Check if DB initials match API first names
        db_initials = ''.join(p[0].lower() for p in db_parts[:-1])
        api_initials = ''.join(p[0].lower() for p in api_parts[:-1])
        
        if db_initials == api_initials:
            return 1.0
        
        # Partial initial match
        if db_initials and api_initials.startswith(db_initials[0]):
            return 0.9
        
        return 0.5  # Surname only match
    
    def find_player(
        self, 
        api_name: str, 
        team_name: str = None,
        role: str = None,
        espn_player_id: int = None
    ) -> Optional[MatchResult]:
        """
        Find database player matching API name.
        
        Args:
            api_name: Player name from Cricket Data API or ESPN
            team_name: Optional team name to narrow search
            role: Optional role (Batsman, Bowler, etc.) for disambiguation
            espn_player_id: Optional ESPN player ID for exact matching
            
        Returns:
            MatchResult if found, None otherwise
        """
        self._initialize()
        
        # Strategy 0: ESPN ID match (most reliable)
        if espn_player_id:
            for player in self._player_cache.values():
                if player.get('espn_player_id') == espn_player_id:
                    logger.debug(f"ESPN ID match: {api_name} -> {player['name']} (ID: {espn_player_id})")
                    return MatchResult(
                        player_id=player['player_id'],
                        db_name=player['name'],
                        api_name=api_name,
                        score=1.0,
                        method='espn_id'
                    )
        
        # Normalize team name for lookup
        team_key = None
        if team_name:
            # Handle "Adelaide Strikers Women" -> "Adelaide Strikers"
            team_key = team_name.replace(' Women', '').strip()
        
        # Get candidate players
        if team_key and team_key in self._team_players:
            candidates = self._team_players[team_key]
        else:
            candidates = list(self._player_cache.values())
        
        best_match = None
        best_score = 0.0
        best_method = ''
        
        for player in candidates:
            db_name = player['name']
            
            # Strategy 1: Exact match
            if self._normalize_name(api_name) == self._normalize_name(db_name):
                return MatchResult(
                    player_id=player['player_id'],
                    db_name=db_name,
                    api_name=api_name,
                    score=1.0,
                    method='exact'
                )
            
            # Strategy 2: DB name format match (initial + surname)
            format_score = self._match_db_name_format(api_name, db_name)
            if format_score > best_score:
                best_score = format_score
                best_match = player
                best_method = 'initials_surname'
            
            # Strategy 3: Surname only match (lower priority)
            if self._get_surname(api_name) == self._get_surname(db_name):
                if best_score < 0.5:
                    best_score = 0.5
                    best_match = player
                    best_method = 'surname'
            
            # Strategy 4: Fuzzy match
            fuzzy = self._fuzzy_score(api_name, db_name)
            if fuzzy > best_score and fuzzy >= self.FUZZY_THRESHOLD:
                best_score = fuzzy
                best_match = player
                best_method = 'fuzzy'
        
        if best_match and best_score >= 0.5:
            return MatchResult(
                player_id=best_match['player_id'],
                db_name=best_match['name'],
                api_name=api_name,
                score=best_score,
                method=best_method
            )
        
        logger.debug(f"No match found for player: {api_name} (team: {team_name})")
        return None
    
    def match_squad(
        self, 
        api_players: List[Dict], 
        team_name: str
    ) -> List[Dict]:
        """
        Match a full squad from API to database players.
        
        Args:
            api_players: List of player dicts from API (with 'name', 'role')
            team_name: Team name
            
        Returns:
            List of dicts with player_id, api_name, db_name, role, matched
        """
        results = []
        
        for api_player in api_players:
            name = api_player.get('name', '')
            role = api_player.get('role', '')
            
            match = self.find_player(name, team_name, role)
            
            if match:
                results.append({
                    'player_id': match.player_id,
                    'api_name': name,
                    'db_name': match.db_name,
                    'role': role,
                    'matched': True,
                    'match_score': match.score,
                    'match_method': match.method
                })
            else:
                results.append({
                    'player_id': None,
                    'api_name': name,
                    'db_name': None,
                    'role': role,
                    'matched': False,
                    'match_score': 0.0,
                    'match_method': None
                })
        
        matched = sum(1 for r in results if r['matched'])
        logger.info(f"Matched {matched}/{len(results)} players for {team_name}")
        
        return results
    
    def get_team_name_mapping(self) -> Dict[str, str]:
        """
        Get mapping from API team names to DB team names.
        
        API uses "Adelaide Strikers Women", DB uses "Adelaide Strikers"
        """
        return {
            'Adelaide Strikers Women': 'Adelaide Strikers',
            'Brisbane Heat Women': 'Brisbane Heat',
            'Hobart Hurricanes Women': 'Hobart Hurricanes',
            'Melbourne Renegades Women': 'Melbourne Renegades',
            'Melbourne Stars Women': 'Melbourne Stars',
            'Perth Scorchers Women': 'Perth Scorchers',
            'Sydney Sixers Women': 'Sydney Sixers',
            'Sydney Thunder Women': 'Sydney Thunder',
        }


if __name__ == "__main__":
    # Test the matcher
    logging.basicConfig(level=logging.INFO)
    
    matcher = PlayerNameMatcher()
    
    # Test individual matches
    test_cases = [
        ("Amanda-Jade Wellington", "Adelaide Strikers"),
        ("Laura Wolvaardt", "Adelaide Strikers"),
        ("Megan Schutt", "Adelaide Strikers"),
        ("Darcie Brown", "Adelaide Strikers"),
        ("Tahlia McGrath", "Adelaide Strikers"),
        ("Sophie Ecclestone", "Adelaide Strikers"),
        ("Jess Jonassen", "Brisbane Heat"),
        ("Grace Harris", "Brisbane Heat"),
    ]
    
    print("=" * 60)
    print("NAME MATCHING TEST")
    print("=" * 60)
    
    for api_name, team in test_cases:
        result = matcher.find_player(api_name, team)
        if result:
            print(f"\n'{api_name}' -> '{result.db_name}' (ID: {result.player_id})")
            print(f"  Score: {result.score:.2f}, Method: {result.method}")
        else:
            print(f"\n'{api_name}' -> NO MATCH FOUND")

