"""
Lineup Service.

Combines Cricket Data API with our database to provide:
1. Upcoming WBBL match fixtures
2. Match squads with matched player IDs
3. Recent playing XI for each team
4. Smart player suggestions based on ELO
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.database import get_connection
from src.api.cricket_data_client import CricketDataClient, WBBLMatch, TeamSquad
from src.features.name_matcher import PlayerNameMatcher

logger = logging.getLogger(__name__)


@dataclass
class MatchedPlayer:
    """Player with DB match info."""
    player_id: Optional[int]
    api_name: str
    db_name: Optional[str]
    role: str
    batting_style: Optional[str]
    bowling_style: Optional[str]
    matched: bool
    elo_batting: Optional[float] = None
    elo_bowling: Optional[float] = None
    recent_form: Optional[str] = None  # e.g., "Played last match"


@dataclass
class MatchFixture:
    """WBBL match fixture with full squad data."""
    match_id: str
    date: str
    team1_name: str
    team2_name: str
    team1_db_name: str
    team2_db_name: str
    venue: Optional[str]
    venue_db_id: Optional[int]  # Matched venue ID from our database
    venue_db_name: Optional[str]  # Venue name from our database
    status: str
    is_upcoming: bool
    team1_squad: List[MatchedPlayer]
    team2_squad: List[MatchedPlayer]
    team1_recent_xi: List[int]  # Player IDs from last match
    team2_recent_xi: List[int]


class LineupService:
    """
    Service for getting WBBL match lineups.
    
    Combines:
    - Cricket Data API for fixtures and squads
    - Our database for player IDs and ELO ratings
    - Name matching for linking API to DB
    """
    
    def __init__(self):
        self.api_client = CricketDataClient()
        self.name_matcher = PlayerNameMatcher()
        self._team_name_map = self.name_matcher.get_team_name_mapping()
    
    def _get_db_team_name(self, api_team_name: str) -> str:
        """Convert API team name to DB team name."""
        return self._team_name_map.get(api_team_name, api_team_name.replace(' Women', ''))
    
    def find_venue_in_db(self, api_venue: str) -> Tuple[Optional[int], Optional[str]]:
        """
        Find a venue in our database matching the API venue name.
        
        Args:
            api_venue: Venue string from API (e.g., "Adelaide Oval, Adelaide")
            
        Returns:
            Tuple of (venue_id, venue_name) or (None, None) if not found
        """
        if not api_venue:
            return None, None
        
        conn = get_connection()
        cursor = conn.cursor()
        
        # Try exact match first
        cursor.execute("""
            SELECT venue_id, name, city
            FROM venues
            WHERE name || ', ' || city = ?
            OR name = ?
            LIMIT 1
        """, (api_venue, api_venue.split(',')[0].strip()))
        
        result = cursor.fetchone()
        
        if not result:
            # Try partial match on venue name
            venue_name_part = api_venue.split(',')[0].strip()
            cursor.execute("""
                SELECT venue_id, name, city
                FROM venues
                WHERE name LIKE ?
                LIMIT 1
            """, (f'%{venue_name_part}%',))
            result = cursor.fetchone()
        
        if not result:
            # Try matching by city (fallback for variations like "WACA Ground" vs "W.A.C.A. Ground")
            city_part = api_venue.split(',')[-1].strip() if ',' in api_venue else None
            if city_part:
                # Get most common venue in that city for women's T20
                cursor.execute("""
                    SELECT v.venue_id, v.name, v.city, COUNT(m.match_id) as match_count
                    FROM venues v
                    JOIN matches m ON m.venue_id = v.venue_id
                    WHERE v.city = ? AND m.gender = 'female'
                    GROUP BY v.venue_id
                    ORDER BY match_count DESC
                    LIMIT 1
                """, (city_part,))
                result = cursor.fetchone()
        
        conn.close()
        
        if result:
            full_name = f"{result['name']}, {result['city']}" if result['city'] else result['name']
            logger.info(f"Matched venue '{api_venue}' -> {result['venue_id']}: {full_name}")
            return result['venue_id'], full_name
        
        logger.warning(f"Could not match venue: {api_venue}")
        return None, None
    
    def get_upcoming_fixtures(self, limit: int = 10) -> List[WBBLMatch]:
        """
        Get upcoming WBBL matches.
        
        Returns:
            List of upcoming WBBLMatch objects
        """
        return self.api_client.get_upcoming_wbbl_matches()[:limit]
    
    def get_recent_lineup(self, team_name: str) -> Tuple[List[int], str]:
        """
        Get the playing XI from a team's most recent WBBL match.
        
        Args:
            team_name: DB team name (e.g., "Adelaide Strikers")
            
        Returns:
            Tuple of (list of player IDs, match date)
        """
        conn = get_connection()
        cursor = conn.cursor()
        
        # Find most recent WBBL match for this team
        cursor.execute("""
            SELECT m.match_id, m.date, t.team_id
            FROM matches m
            JOIN teams t ON t.team_id IN (m.team1_id, m.team2_id)
            WHERE t.name = ? 
            AND m.event_name LIKE '%Big Bash%'
            AND m.gender = 'female'
            ORDER BY m.date DESC
            LIMIT 1
        """, (team_name,))
        
        result = cursor.fetchone()
        if not result:
            conn.close()
            return [], ''
        
        match_id = result['match_id']
        match_date = result['date']
        team_id = result['team_id']
        
        # Get playing XI for that match
        cursor.execute("""
            SELECT p.player_id, p.name, pms.batting_position
            FROM player_match_stats pms
            JOIN players p ON pms.player_id = p.player_id
            WHERE pms.match_id = ? AND pms.team_id = ?
            ORDER BY pms.batting_position NULLS LAST
        """, (match_id, team_id))
        
        players = cursor.fetchall()
        conn.close()
        
        player_ids = [p['player_id'] for p in players]
        logger.info(f"{team_name}: {len(player_ids)} players from {match_date}")
        
        return player_ids, match_date
    
    def get_player_elo(self, player_id: int) -> Tuple[Optional[float], Optional[float]]:
        """Get batting and bowling ELO for a player."""
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT batting_elo_t20_female, bowling_elo_t20_female
            FROM player_current_elo
            WHERE player_id = ?
        """, (player_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return result['batting_elo_t20_female'], result['bowling_elo_t20_female']
        return None, None
    
    def get_match_with_squads(self, match_id: str) -> Optional[MatchFixture]:
        """
        Get full fixture data with matched squads.
        
        Args:
            match_id: API match ID
            
        Returns:
            MatchFixture with all squad data
        """
        # Get upcoming matches to find this one
        matches = self.api_client.get_wbbl_matches()
        match = next((m for m in matches if m.id == match_id), None)
        
        if not match:
            logger.warning(f"Match not found: {match_id}")
            return None
        
        # Get squads from API
        squads = self.api_client.get_match_squad(match_id)
        
        if not squads:
            logger.warning(f"No squads for match: {match_id}")
            return None
        
        # Get DB team names
        team1_db = self._get_db_team_name(match.team1)
        team2_db = self._get_db_team_name(match.team2)
        
        # Get recent lineups from our DB
        team1_recent, team1_date = self.get_recent_lineup(team1_db)
        team2_recent, team2_date = self.get_recent_lineup(team2_db)
        
        # Match squads to DB players
        team1_squad_data = squads.get(match.team1, TeamSquad(match.team1, '', []))
        team2_squad_data = squads.get(match.team2, TeamSquad(match.team2, '', []))
        
        team1_matched = self._match_squad_to_db(team1_squad_data, team1_db, team1_recent)
        team2_matched = self._match_squad_to_db(team2_squad_data, team2_db, team2_recent)
        
        # Match venue to our database
        venue_db_id, venue_db_name = self.find_venue_in_db(match.venue)
        
        return MatchFixture(
            match_id=match_id,
            date=match.date,
            team1_name=match.team1,
            team2_name=match.team2,
            team1_db_name=team1_db,
            team2_db_name=team2_db,
            venue=match.venue,
            venue_db_id=venue_db_id,
            venue_db_name=venue_db_name,
            status=match.status,
            is_upcoming=match.is_upcoming,
            team1_squad=team1_matched,
            team2_squad=team2_matched,
            team1_recent_xi=team1_recent,
            team2_recent_xi=team2_recent
        )
    
    def _match_squad_to_db(
        self, 
        api_squad: TeamSquad, 
        db_team_name: str,
        recent_xi: List[int]
    ) -> List[MatchedPlayer]:
        """Match API squad to database players with ELO."""
        matched_players = []
        
        for api_player in api_squad.players:
            # Try to match
            match = self.name_matcher.find_player(
                api_player.name, 
                db_team_name,
                api_player.role
            )
            
            if match:
                batting_elo, bowling_elo = self.get_player_elo(match.player_id)
                in_recent = match.player_id in recent_xi
                
                matched_players.append(MatchedPlayer(
                    player_id=match.player_id,
                    api_name=api_player.name,
                    db_name=match.db_name,
                    role=api_player.role,
                    batting_style=api_player.batting_style,
                    bowling_style=api_player.bowling_style,
                    matched=True,
                    elo_batting=batting_elo,
                    elo_bowling=bowling_elo,
                    recent_form="Played last match" if in_recent else "In squad"
                ))
            else:
                matched_players.append(MatchedPlayer(
                    player_id=None,
                    api_name=api_player.name,
                    db_name=None,
                    role=api_player.role,
                    batting_style=api_player.batting_style,
                    bowling_style=api_player.bowling_style,
                    matched=False
                ))
        
        return matched_players
    
    def suggest_lineup(self, squad: List[MatchedPlayer], recent_xi: List[int]) -> List[int]:
        """
        Suggest a playing XI from squad.
        
        Priority:
        1. Players from recent XI who are in squad
        2. Fill remaining spots with highest ELO players by role
        
        Returns:
            List of player IDs for suggested XI
        """
        suggested = []
        used_ids = set()
        
        # First: add players from recent XI who are in squad
        squad_ids = {p.player_id for p in squad if p.player_id}
        for pid in recent_xi:
            if pid in squad_ids and len(suggested) < 11:
                suggested.append(pid)
                used_ids.add(pid)
        
        # Second: fill with highest ELO players not yet selected
        remaining = [p for p in squad if p.player_id and p.player_id not in used_ids]
        
        # Sort by combined ELO (batting + bowling)
        def combined_elo(p: MatchedPlayer) -> float:
            bat = p.elo_batting or 1500
            bowl = p.elo_bowling or 1500
            return bat + bowl
        
        remaining.sort(key=combined_elo, reverse=True)
        
        for player in remaining:
            if len(suggested) >= 11:
                break
            suggested.append(player.player_id)
        
        return suggested


def get_service() -> LineupService:
    """Get a configured LineupService instance."""
    return LineupService()


if __name__ == "__main__":
    # Test the service
    logging.basicConfig(level=logging.INFO)
    
    service = LineupService()
    
    print("=" * 70)
    print("WBBL LINEUP SERVICE TEST")
    print("=" * 70)
    
    # Get upcoming fixtures
    print("\nUpcoming WBBL Matches:")
    print("-" * 50)
    fixtures = service.get_upcoming_fixtures(5)
    
    for f in fixtures:
        print(f"\n{f.date}: {f.team1} vs {f.team2}")
        print(f"  ID: {f.id}")
    
    # Get full data for first fixture with squad
    if fixtures:
        for f in fixtures:
            if f.has_squad:
                print(f"\n{'='*70}")
                print(f"LOADING FULL DATA FOR: {f.name}")
                print("=" * 70)
                
                match = service.get_match_with_squads(f.id)
                
                if match:
                    print(f"\n{match.team1_name} ({match.team1_db_name}):")
                    print(f"  Recent XI from {service.get_recent_lineup(match.team1_db_name)[1]}")
                    for p in match.team1_squad[:8]:
                        status = "✓" if p.matched else "✗"
                        elo = f"ELO: {p.elo_batting:.0f}/{p.elo_bowling:.0f}" if p.elo_batting else ""
                        print(f"  {status} {p.api_name} -> {p.db_name or 'NOT FOUND'} {elo}")
                    
                    # Suggest lineup
                    suggested = service.suggest_lineup(match.team1_squad, match.team1_recent_xi)
                    print(f"\n  Suggested XI ({len(suggested)} players): {suggested[:5]}...")
                
                break

