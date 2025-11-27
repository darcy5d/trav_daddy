"""
Cricket Data API Client.

Client for cricketdata.org API to fetch:
- Current/upcoming matches
- Series information (WBBL)
- Match squads

API Documentation: https://cricketdata.org/how-to-use-cricket-data-api.aspx
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import requests
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import CRICKET_DATA_API_KEY, CRICKET_DATA_BASE_URL

logger = logging.getLogger(__name__)


@dataclass
class WBBLMatch:
    """Represents a WBBL match."""
    id: str
    name: str
    date: str
    status: str
    team1: str
    team2: str
    venue: Optional[str] = None
    has_squad: bool = False
    is_upcoming: bool = False
    
    @classmethod
    def from_api(cls, data: dict) -> 'WBBLMatch':
        """Create from API response."""
        teams = data.get('teams', [])
        team1 = teams[0] if teams else data.get('name', '').split(' vs ')[0].strip()
        team2 = teams[1] if len(teams) > 1 else ''
        
        # Check if upcoming
        status = data.get('status', '')
        is_upcoming = 'Match starts at' in status or status == ''
        
        return cls(
            id=data.get('id', ''),
            name=data.get('name', ''),
            date=data.get('date', ''),
            status=status,
            team1=team1,
            team2=team2,
            venue=data.get('venue'),
            has_squad=data.get('hasSquad', False),
            is_upcoming=is_upcoming
        )


@dataclass
class Player:
    """Represents a player from API."""
    id: str
    name: str
    role: str
    batting_style: Optional[str] = None
    bowling_style: Optional[str] = None
    country: Optional[str] = None
    
    @classmethod
    def from_api(cls, data: dict) -> 'Player':
        """Create from API response."""
        return cls(
            id=data.get('id', ''),
            name=data.get('name', ''),
            role=data.get('role', 'Unknown'),
            batting_style=data.get('battingStyle'),
            bowling_style=data.get('bowlingStyle'),
            country=data.get('country')
        )


@dataclass
class TeamSquad:
    """Represents a team's squad."""
    team_name: str
    short_name: str
    players: List[Player]
    
    @classmethod
    def from_api(cls, data: dict) -> 'TeamSquad':
        """Create from API response."""
        players = [Player.from_api(p) for p in data.get('players', [])]
        return cls(
            team_name=data.get('teamName', ''),
            short_name=data.get('shortname', ''),
            players=players
        )


class CricketDataClient:
    """
    Client for Cricket Data API (cricketdata.org).
    
    Provides methods to fetch WBBL fixtures and squad data.
    """
    
    WBBL_SERIES_SEARCH = "Big Bash"
    
    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or CRICKET_DATA_API_KEY
        self.base_url = base_url or CRICKET_DATA_BASE_URL
        
        if not self.api_key:
            raise ValueError("Cricket Data API key not configured. Set CRICKET_DATA_API_KEY in .env")
    
    def _request(self, endpoint: str, params: dict = None) -> dict:
        """Make API request."""
        url = f"{self.base_url}/{endpoint}"
        
        if params is None:
            params = {}
        params['apikey'] = self.api_key
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') != 'success':
                logger.warning(f"API returned non-success: {data.get('info')}")
            
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_wbbl_series(self) -> Optional[dict]:
        """
        Find the current WBBL series.
        
        Returns:
            Series info dict or None if not found
        """
        data = self._request('series', {'search': self.WBBL_SERIES_SEARCH})
        
        if 'data' not in data:
            return None
        
        # Find WBBL specifically
        for series in data['data']:
            name = series.get('name', '').lower()
            if 'women' in name and 'big bash' in name:
                logger.info(f"Found WBBL series: {series.get('name')} (ID: {series.get('id')})")
                return series
        
        return None
    
    def get_wbbl_matches(self, series_id: str = None) -> List[WBBLMatch]:
        """
        Get all WBBL matches for the current season.
        
        Args:
            series_id: Optional series ID. If not provided, will search for WBBL.
            
        Returns:
            List of WBBLMatch objects
        """
        # Find series if not provided
        if not series_id:
            series = self.get_wbbl_series()
            if not series:
                logger.warning("Could not find WBBL series")
                return []
            series_id = series['id']
        
        # Get series info with matches
        data = self._request('series_info', {'id': series_id})
        
        if 'data' not in data:
            return []
        
        match_list = data['data'].get('matchList', [])
        matches = [WBBLMatch.from_api(m) for m in match_list]
        
        logger.info(f"Found {len(matches)} WBBL matches")
        return matches
    
    def get_upcoming_wbbl_matches(self) -> List[WBBLMatch]:
        """
        Get only upcoming WBBL matches (not yet played).
        
        Returns:
            List of upcoming WBBLMatch objects, sorted by date
        """
        matches = self.get_wbbl_matches()
        upcoming = [m for m in matches if m.is_upcoming]
        
        # Sort by date
        upcoming.sort(key=lambda m: m.date)
        
        logger.info(f"Found {len(upcoming)} upcoming WBBL matches")
        return upcoming
    
    def get_match_squad(self, match_id: str) -> Dict[str, TeamSquad]:
        """
        Get squad for a specific match.
        
        Args:
            match_id: The match ID from the API
            
        Returns:
            Dict with team names as keys and TeamSquad as values
        """
        data = self._request('match_squad', {'id': match_id})
        
        if 'data' not in data:
            logger.warning(f"No squad data for match {match_id}")
            return {}
        
        squads = {}
        for team_data in data['data']:
            squad = TeamSquad.from_api(team_data)
            squads[squad.team_name] = squad
            logger.info(f"Loaded {len(squad.players)} players for {squad.team_name}")
        
        return squads
    
    def get_match_info(self, match_id: str) -> Optional[dict]:
        """
        Get detailed info for a specific match.
        
        Args:
            match_id: The match ID
            
        Returns:
            Match info dict or None
        """
        data = self._request('match_info', {'id': match_id})
        return data.get('data')


# Convenience function
def get_client() -> CricketDataClient:
    """Get a configured CricketDataClient instance."""
    return CricketDataClient()


if __name__ == "__main__":
    # Test the client
    logging.basicConfig(level=logging.INFO)
    
    client = CricketDataClient()
    
    print("=" * 60)
    print("WBBL Upcoming Matches")
    print("=" * 60)
    
    upcoming = client.get_upcoming_wbbl_matches()
    for match in upcoming[:5]:
        print(f"\n{match.date}: {match.name}")
        print(f"  Teams: {match.team1} vs {match.team2}")
        print(f"  Status: {match.status}")
        print(f"  Has Squad: {match.has_squad}")
        
        if match.has_squad:
            print(f"\n  Loading squad for: {match.id}")
            squads = client.get_match_squad(match.id)
            for team_name, squad in squads.items():
                print(f"\n  {team_name} ({len(squad.players)} players):")
                for p in squad.players[:5]:
                    print(f"    - {p.name} ({p.role})")
                if len(squad.players) > 5:
                    print(f"    ... and {len(squad.players) - 5} more")
            break  # Just show one to save API calls

