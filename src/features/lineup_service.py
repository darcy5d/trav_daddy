"""
Lineup Service.

Combines Cricket Data API with our database to provide:
1. Upcoming T20 match fixtures (all competitions)
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
from src.api.cricket_data_client import (
    CricketDataClient, 
    WBBLMatch, 
    TeamSquad,
    T20Match,
    T20Series,
    detect_gender
)
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
    """T20 match fixture with full squad data (works for any competition)."""
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
    gender: str = 'female'  # 'male' or 'female'
    series_id: Optional[str] = None
    series_name: Optional[str] = None


class LineupService:
    """
    Service for getting T20 match lineups (all competitions).
    
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
    
    def find_venue_in_db(self, api_venue: str, gender: str = 'female') -> Tuple[Optional[int], Optional[str]]:
        """
        Find a venue in our database matching the API venue name.
        
        Args:
            api_venue: Venue string from API (e.g., "Adelaide Oval, Adelaide")
            gender: 'male' or 'female' for fallback venue lookup
            
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
                # Get most common venue in that city for specified gender
                cursor.execute("""
                    SELECT v.venue_id, v.name, v.city, COUNT(m.match_id) as match_count
                    FROM venues v
                    JOIN matches m ON m.venue_id = v.venue_id
                    WHERE v.city = ? AND m.gender = ?
                    GROUP BY v.venue_id
                    ORDER BY match_count DESC
                    LIMIT 1
                """, (city_part, gender))
                result = cursor.fetchone()
        
        conn.close()
        
        if result:
            full_name = f"{result['name']}, {result['city']}" if result['city'] else result['name']
            logger.info(f"Matched venue '{api_venue}' -> {result['venue_id']}: {full_name}")
            return result['venue_id'], full_name
        
        logger.warning(f"Could not match venue: {api_venue}")
        return None, None
    
    def get_upcoming_fixtures(self, limit: int = 10, gender: str = 'female') -> List[WBBLMatch]:
        """
        Get upcoming WBBL matches (legacy method for backward compatibility).
        
        Args:
            limit: Maximum number of fixtures to return
            gender: 'male' or 'female' (ignored for WBBL, always female)
            
        Returns:
            List of upcoming WBBLMatch objects
        """
        return self.api_client.get_upcoming_wbbl_matches()[:limit]
    
    def get_t20_series(self, gender: Optional[str] = None) -> List[T20Series]:
        """
        Get list of active T20 series.
        
        Args:
            gender: Optional filter - 'male' or 'female'
            
        Returns:
            List of T20Series objects
        """
        return self.api_client.get_t20_series(gender=gender)
    
    def get_series_fixtures(
        self, 
        series_id: str, 
        series_name: str = '',
        days_ahead: int = 14
    ) -> List[T20Match]:
        """
        Get upcoming fixtures for a specific T20 series.
        
        Args:
            series_id: The series ID from the API
            series_name: Optional series name for gender detection
            days_ahead: How many days ahead to include
            
        Returns:
            List of upcoming T20Match objects
        """
        return self.api_client.get_upcoming_series_matches(
            series_id=series_id,
            series_name=series_name,
            days_ahead=days_ahead
        )
    
    def get_recent_lineup(self, team_name: str, gender: str = 'female') -> Tuple[List[int], str]:
        """
        Get the playing XI from a team's most recent T20 match.
        
        Args:
            team_name: DB team name (e.g., "Adelaide Strikers", "Mumbai Indians")
            gender: 'male' or 'female'
            
        Returns:
            Tuple of (list of player IDs, match date)
        """
        conn = get_connection()
        cursor = conn.cursor()
        
        # Find most recent T20 match for this team
        cursor.execute("""
            SELECT m.match_id, m.date, t.team_id
            FROM matches m
            JOIN teams t ON t.team_id IN (m.team1_id, m.team2_id)
            WHERE t.name = ? 
            AND m.match_type = 'T20'
            AND m.gender = ?
            ORDER BY m.date DESC
            LIMIT 1
        """, (team_name, gender))
        
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
        logger.info(f"{team_name} ({gender}): {len(player_ids)} players from {match_date}")
        
        return player_ids, match_date
    
    def get_player_elo(self, player_id: int, gender: str = 'female') -> Tuple[Optional[float], Optional[float]]:
        """
        Get batting and bowling ELO for a player.
        
        Args:
            player_id: The player ID
            gender: 'male' or 'female'
            
        Returns:
            Tuple of (batting_elo, bowling_elo) or (None, None)
        """
        conn = get_connection()
        cursor = conn.cursor()
        
        batting_col = f'batting_elo_t20_{gender}'
        bowling_col = f'bowling_elo_t20_{gender}'
        
        cursor.execute(f"""
            SELECT {batting_col}, {bowling_col}
            FROM player_current_elo
            WHERE player_id = ?
        """, (player_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return result[batting_col], result[bowling_col]
        return None, None
    
    def get_match_with_squads(self, match_id: str, series_id: str = None) -> Optional[MatchFixture]:
        """
        Get full fixture data with matched squads.
        
        Works with any T20 competition (WBBL, BBL, IPL, etc.)
        
        Args:
            match_id: API match ID
            series_id: Optional series ID to search within
            
        Returns:
            MatchFixture with all squad data
        """
        # Try to find the match - first in WBBL (legacy), then via match_info API
        match = None
        gender = 'female'  # Default
        series_name = ''
        
        # Try WBBL first (for backward compatibility)
        wbbl_matches = self.api_client.get_wbbl_matches()
        match = next((m for m in wbbl_matches if m.id == match_id), None)
        
        if match:
            gender = 'female'
            series_name = 'WBBL'
        else:
            # Try to get match info directly from API
            match_info = self.api_client.get_match_info(match_id)
            if match_info:
                # Create a temporary match object from match_info
                teams = match_info.get('teams', [])
                team1 = teams[0] if teams else ''
                team2 = teams[1] if len(teams) > 1 else ''
                
                # Detect gender from team names
                gender = detect_gender(f"{team1} {team2}")
                series_name = match_info.get('series', '')
                
                # Create a WBBLMatch-like object for compatibility
                class TempMatch:
                    def __init__(self, data):
                        self.id = data.get('id', match_id)
                        self.name = data.get('name', '')
                        self.date = data.get('date', '')
                        self.status = data.get('status', '')
                        self.team1 = team1
                        self.team2 = team2
                        self.venue = data.get('venue')
                        self.has_squad = True
                        self.is_upcoming = 'won' not in self.status.lower()
                
                match = TempMatch(match_info)
        
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
        
        # Get recent lineups from our DB (gender-aware)
        team1_recent, team1_date = self.get_recent_lineup(team1_db, gender=gender)
        team2_recent, team2_date = self.get_recent_lineup(team2_db, gender=gender)
        
        # Match squads to DB players
        team1_squad_data = squads.get(match.team1, TeamSquad(match.team1, '', []))
        team2_squad_data = squads.get(match.team2, TeamSquad(match.team2, '', []))
        
        team1_matched = self._match_squad_to_db(team1_squad_data, team1_db, team1_recent, gender=gender)
        team2_matched = self._match_squad_to_db(team2_squad_data, team2_db, team2_recent, gender=gender)
        
        # Match venue to our database
        venue_db_id, venue_db_name = self.find_venue_in_db(match.venue, gender=gender)
        
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
            team2_recent_xi=team2_recent,
            gender=gender,
            series_id=series_id,
            series_name=series_name
        )
    
    def _match_squad_to_db(
        self, 
        api_squad: TeamSquad, 
        db_team_name: str,
        recent_xi: List[int],
        gender: str = 'female'
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
                batting_elo, bowling_elo = self.get_player_elo(match.player_id, gender=gender)
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
    
    def _get_top_players_by_role(
        self, 
        team_name: str, 
        role: str,  # 'batter' or 'bowler'
        count: int, 
        exclude_ids: set,
        gender: str = 'female'
    ) -> List[MatchedPlayer]:
        """
        Get highest ELO players for a specific role from team's historical data.
        
        Args:
            team_name: DB team name (e.g., "Sydney Sixers")
            role: 'batter' or 'bowler'
            count: Number of players needed
            exclude_ids: Player IDs already selected
            gender: 'male' or 'female'
            
        Returns:
            List of MatchedPlayer objects from DB
        """
        if count <= 0:
            return []
        
        conn = get_connection()
        cursor = conn.cursor()
        
        # Determine which ELO column to use
        elo_col = f'batting_elo_t20_{gender}' if role == 'batter' else f'bowling_elo_t20_{gender}'
        
        # Build exclusion clause
        if exclude_ids:
            exclude_clause = f"AND p.player_id NOT IN ({','.join(str(pid) for pid in exclude_ids)})"
        else:
            exclude_clause = ""
        
        # Query players who have played for this team, ordered by relevant ELO
        # We look at player_match_stats to find players who played for this team
        query = f"""
            SELECT DISTINCT 
                p.player_id, 
                p.name,
                COALESCE(e.{elo_col}, 1500) as elo,
                COALESCE(e.batting_elo_t20_{gender}, 1500) as batting_elo,
                COALESCE(e.bowling_elo_t20_{gender}, 1500) as bowling_elo
            FROM players p
            JOIN player_match_stats pms ON p.player_id = pms.player_id
            JOIN matches m ON pms.match_id = m.match_id
            JOIN teams t ON pms.team_id = t.team_id
            LEFT JOIN player_current_elo e ON p.player_id = e.player_id
            WHERE t.name = ? 
            AND m.gender = ?
            AND m.match_type = 'T20'
            {exclude_clause}
            GROUP BY p.player_id
            ORDER BY elo DESC
            LIMIT ?
        """
        
        cursor.execute(query, (team_name, gender, count))
        rows = cursor.fetchall()
        conn.close()
        
        players = []
        for row in rows:
            players.append(MatchedPlayer(
                player_id=row['player_id'],
                api_name='',
                db_name=row['name'],
                role='Batter' if role == 'batter' else 'Bowler',
                batting_style=None,
                bowling_style=None,
                matched=True,
                elo_batting=row['batting_elo'],
                elo_bowling=row['bowling_elo'],
                recent_form="Auto-filled from team history"
            ))
        
        logger.info(f"Auto-filled {len(players)} {role}s for {team_name} from DB")
        return players
    
    def ensure_playing_xi(
        self, 
        matched_players: List[MatchedPlayer], 
        team_db_name: str,
        gender: str = 'female'
    ) -> Tuple[List[MatchedPlayer], Dict]:
        """
        Ensure we have exactly 11 UNIQUE valid players with proper role balance.
        
        CRITICAL: Each player can only appear ONCE in the XI (cricket law).
        
        Target composition:
        - At least 6 recognized batters (top order + wicket-keeper)
        - At least 4 recognized bowlers
        
        Args:
            matched_players: List of players from API with DB matches
            team_db_name: DB team name for fallback queries
            gender: 'male' or 'female'
            
        Returns:
            Tuple of (List of 11 UNIQUE MatchedPlayer, dict with fill info)
        """
        # STEP 1: Deduplicate input - each player_id can only appear once
        seen_ids = set()
        unique_players = []
        duplicates_removed = 0
        
        for p in matched_players:
            if p.player_id and p.player_id not in seen_ids:
                unique_players.append(p)
                seen_ids.add(p.player_id)
            elif p.player_id:
                duplicates_removed += 1
        
        if duplicates_removed > 0:
            logger.warning(f"Removed {duplicates_removed} duplicate player(s) from input")
        
        valid = unique_players
        
        fill_info = {
            'original_count': len(valid),
            'duplicates_removed': duplicates_removed,
            'batters_added': 0,
            'bowlers_added': 0,
            'auto_filled': []
        }
        
        # Already have 11+ unique players, return first 11
        if len(valid) >= 11:
            return valid[:11], fill_info
        
        needed = 11 - len(valid)
        used_ids = seen_ids.copy()  # Use the already-seen IDs for exclusion
        
        # Categorize players by role
        batter_roles = {'Batter', 'Top order Batter', 'Opening Batter', 
                       'WK-Batter', 'Wicketkeeper Batter', 'Batting Allrounder'}
        bowler_roles = {'Bowler', 'Bowling Allrounder'}
        
        batters = [p for p in valid if p.role in batter_roles]
        bowlers = [p for p in valid if p.role in bowler_roles]
        
        # Target: 6 batters, 5 bowlers (typical T20 balance)
        target_batters = 6
        target_bowlers = 5
        
        # Determine what we need to fill
        batters_needed = max(0, target_batters - len(batters))
        bowlers_needed = max(0, target_bowlers - len(bowlers))
        
        # Don't overfill - respect total needed
        if batters_needed + bowlers_needed > needed:
            # Prioritize what we're more short on
            if len(batters) < len(bowlers):
                batters_needed = min(batters_needed, needed)
                bowlers_needed = needed - batters_needed
            else:
                bowlers_needed = min(bowlers_needed, needed)
                batters_needed = needed - bowlers_needed
        
        # Fill batters first
        if batters_needed > 0:
            fill_batters = self._get_top_players_by_role(
                team_db_name, 'batter', batters_needed, used_ids, gender
            )
            valid.extend(fill_batters)
            used_ids.update(p.player_id for p in fill_batters)
            fill_info['batters_added'] = len(fill_batters)
            fill_info['auto_filled'].extend([p.db_name for p in fill_batters])
        
        # Fill bowlers
        remaining_needed = 11 - len(valid)
        if remaining_needed > 0:
            fill_bowlers = self._get_top_players_by_role(
                team_db_name, 'bowler', remaining_needed, used_ids, gender
            )
            valid.extend(fill_bowlers)
            fill_info['bowlers_added'] = len(fill_bowlers)
            fill_info['auto_filled'].extend([p.db_name for p in fill_bowlers])
        
        # Final safety: if still not 11, fill with any players by overall ELO
        remaining_needed = 11 - len(valid)
        if remaining_needed > 0:
            # Update used_ids with any bowlers we added
            used_ids.update(p.player_id for p in valid if p.player_id)
            # Get any players with best combined ELO
            any_players = self._get_top_players_by_role(
                team_db_name, 'batter', remaining_needed, used_ids, gender
            )
            valid.extend(any_players)
            fill_info['auto_filled'].extend([p.db_name for p in any_players])
        
        # FINAL VALIDATION: Ensure all 11 are unique
        final_xi = valid[:11]
        final_ids = [p.player_id for p in final_xi]
        if len(set(final_ids)) != len(final_ids):
            logger.error(f"DUPLICATE PLAYER IDS IN FINAL XI: {final_ids}")
            # Deduplicate again as last resort
            seen = set()
            deduped = []
            for p in final_xi:
                if p.player_id not in seen:
                    deduped.append(p)
                    seen.add(p.player_id)
            final_xi = deduped
            logger.warning(f"After final dedup: {len(final_xi)} unique players")
        
        logger.info(f"Playing XI for {team_db_name}: {len(final_xi)} unique players "
                   f"(batters added: {fill_info['batters_added']}, "
                   f"bowlers added: {fill_info['bowlers_added']})")
        
        return final_xi, fill_info


def get_service() -> LineupService:
    """Get a configured LineupService instance."""
    return LineupService()


if __name__ == "__main__":
    # Test the service
    logging.basicConfig(level=logging.INFO)
    
    service = LineupService()
    
    print("=" * 70)
    print("T20 LINEUP SERVICE TEST")
    print("=" * 70)
    
    # Test T20 series listing
    print("\n--- Women's T20 Series ---")
    women_series = service.get_t20_series(gender='female')
    for s in women_series[:5]:
        print(f"  {s.name} | {s.t20_count} T20s")
    
    print("\n--- Men's T20 Series ---")
    men_series = service.get_t20_series(gender='male')
    for s in men_series[:5]:
        print(f"  {s.name} | {s.t20_count} T20s")
    
    # Get upcoming WBBL fixtures (legacy method)
    print("\n" + "=" * 70)
    print("Upcoming WBBL Matches (Legacy):")
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
                    print(f"  Gender: {match.gender}")
                    print(f"  Recent XI from {service.get_recent_lineup(match.team1_db_name, match.gender)[1]}")
                    for p in match.team1_squad[:8]:
                        status = "✓" if p.matched else "✗"
                        elo = f"ELO: {p.elo_batting:.0f}/{p.elo_bowling:.0f}" if p.elo_batting else ""
                        print(f"  {status} {p.api_name} -> {p.db_name or 'NOT FOUND'} {elo}")
                    
                    # Suggest lineup
                    suggested = service.suggest_lineup(match.team1_squad, match.team1_recent_xi)
                    print(f"\n  Suggested XI ({len(suggested)} players): {suggested[:5]}...")
                
                break

