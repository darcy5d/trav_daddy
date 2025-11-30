"""
Cricket Data API Client.

Client for cricketdata.org API to fetch:
- Current/upcoming matches
- Series information (all T20 competitions)
- Match squads

API Documentation: https://cricketdata.org/how-to-use-cricket-data-api.aspx
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import requests
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import CRICKET_DATA_API_KEY, CRICKET_DATA_BASE_URL

logger = logging.getLogger(__name__)


@dataclass
class WBBLMatch:
    """Represents a WBBL match (legacy, use T20Match for new code)."""
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
class T20Series:
    """Represents a T20 series/competition."""
    id: str
    name: str
    start_date: str
    end_date: str
    t20_count: int
    gender: str  # 'male' or 'female'
    odi_count: int = 0
    test_count: int = 0
    match_count: int = 0
    
    @classmethod
    def from_api(cls, data: dict) -> 'T20Series':
        """Create from API response."""
        name = data.get('name', '')
        return cls(
            id=data.get('id', ''),
            name=name,
            start_date=data.get('startDate', ''),
            end_date=data.get('endDate', ''),
            t20_count=data.get('t20', 0),
            gender=detect_gender(name),
            odi_count=data.get('odi', 0),
            test_count=data.get('test', 0),
            match_count=data.get('matches', 0)
        )


@dataclass
class T20Match:
    """Represents a T20 match from any competition."""
    id: str
    name: str
    date: str
    status: str
    team1: str
    team2: str
    series_id: str
    series_name: str
    gender: str  # 'male' or 'female'
    venue: Optional[str] = None
    has_squad: bool = False
    is_upcoming: bool = False
    date_time_gmt: Optional[str] = None
    
    @classmethod
    def from_api(cls, data: dict, series_id: str = '', series_name: str = '') -> 'T20Match':
        """Create from API response."""
        teams = data.get('teams', [])
        team1 = teams[0] if teams else data.get('name', '').split(' vs ')[0].strip()
        team2 = teams[1] if len(teams) > 1 else ''
        
        # Check if match is selectable (upcoming or in progress today)
        status = data.get('status', '').lower()
        match_date = data.get('date', '')
        
        # Match is upcoming/selectable if:
        # 1. Status explicitly says it's starting
        # 2. Status is empty (unknown)
        # 3. Match is today and not ended (toss done, in progress, etc.)
        # 4. Status doesn't indicate the match has ended
        is_ended = any(word in status for word in ['won by', 'tied', 'draw', 'no result', 'abandoned'])
        is_today = match_date == datetime.now().strftime('%Y-%m-%d')
        
        is_upcoming = (
            'match starts at' in status 
            or status == '' 
            or 'not started' in status
            or (is_today and not is_ended)  # Today's match, not ended = still selectable
        )
        
        # Detect gender from teams or series name
        gender = detect_gender(f"{team1} {team2} {series_name}")
        
        return cls(
            id=data.get('id', ''),
            name=data.get('name', ''),
            date=data.get('date', ''),
            status=status,
            team1=team1,
            team2=team2,
            series_id=series_id or data.get('series_id', ''),
            series_name=series_name,
            gender=gender,
            venue=data.get('venue'),
            has_squad=data.get('hasSquad', False),
            is_upcoming=is_upcoming,
            date_time_gmt=data.get('dateTimeGMT')
        )
    
    def to_wbbl_match(self) -> WBBLMatch:
        """Convert to legacy WBBLMatch for backward compatibility."""
        return WBBLMatch(
            id=self.id,
            name=self.name,
            date=self.date,
            status=self.status,
            team1=self.team1,
            team2=self.team2,
            venue=self.venue,
            has_squad=self.has_squad,
            is_upcoming=self.is_upcoming
        )


# Gender detection keywords
WOMEN_KEYWORDS = ['women', 'wbbl', 'wpl', 'female', "women's", 'wpsl', 'wt20']


def detect_gender(text: str) -> str:
    """
    Detect gender from series/team name.
    
    Args:
        text: Series name, team name, or combination
        
    Returns:
        'female' if women's cricket detected, 'male' otherwise
    """
    text_lower = text.lower()
    return 'female' if any(kw in text_lower for kw in WOMEN_KEYWORDS) else 'male'


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
    
    # =========================================================================
    # Generic T20 Methods (for all competitions)
    # =========================================================================
    
    def get_t20_series(self, gender: Optional[str] = None) -> List[T20Series]:
        """
        Get all active T20 series.
        
        Args:
            gender: Optional filter - 'male' or 'female'
            
        Returns:
            List of T20Series objects with t20_count > 0
        """
        # Track series by ID to avoid duplicates
        series_by_id = {}
        
        # Fetch main series list
        data = self._request('series')
        if 'data' in data:
            for s in data['data']:
                if s.get('t20', 0) > 0:
                    series = T20Series.from_api(s)
                    series_by_id[series.id] = series
        
        # Search for known leagues that might not be in first page of results
        # These are major competitions that users expect to see
        # Note: Use actual names as API search doesn't recognize abbreviations
        search_leagues = []
        if gender == 'female' or gender is None:
            search_leagues.extend(['Womens Big Bash', 'Women Premier League', 'hundred women', 'fairbreak'])
        if gender == 'male' or gender is None:
            search_leagues.extend(['Big Bash League', 'Indian Premier League', 'Pakistan Super League', 'Caribbean Premier League', 'sa20', 'hundred'])
        
        for league in search_leagues:
            try:
                data = self._request('series', {'search': league})
                if 'data' in data:
                    for s in data['data']:
                        if s.get('t20', 0) > 0 and s.get('id') not in series_by_id:
                            series = T20Series.from_api(s)
                            # Only add if gender matches (when filtering)
                            if gender is None or series.gender == gender:
                                series_by_id[series.id] = series
            except Exception as e:
                logger.warning(f"Failed to search for {league}: {e}")
        
        # Convert to list and filter by gender if specified
        all_series = list(series_by_id.values())
        if gender:
            all_series = [s for s in all_series if s.gender == gender]
        
        # Sort by start date (most recent first)
        all_series.sort(key=lambda x: x.start_date, reverse=True)
        
        logger.info(f"Found {len(all_series)} T20 series" + 
                   (f" for {gender}" if gender else ""))
        return all_series
    
    def get_series_matches(self, series_id: str, series_name: str = '') -> List[T20Match]:
        """
        Get all matches for a specific series.
        
        Args:
            series_id: The series ID from the API
            series_name: Optional series name for gender detection
            
        Returns:
            List of T20Match objects (all matches in series)
        """
        data = self._request('series_info', {'id': series_id})
        
        if 'data' not in data:
            logger.warning(f"Failed to fetch series info for {series_id}")
            return []
        
        series_data = data['data']
        series_name = series_name or series_data.get('info', {}).get('name', '')
        match_list = series_data.get('matchList', [])
        
        matches = [
            T20Match.from_api(m, series_id=series_id, series_name=series_name)
            for m in match_list
        ]
        
        logger.info(f"Found {len(matches)} matches in series {series_name or series_id}")
        return matches
    
    def get_upcoming_series_matches(
        self, 
        series_id: str, 
        series_name: str = '',
        days_ahead: int = 14
    ) -> List[T20Match]:
        """
        Get upcoming matches for a specific series.
        
        Args:
            series_id: The series ID
            series_name: Optional series name for gender detection
            days_ahead: How many days ahead to include (default 14)
            
        Returns:
            List of upcoming T20Match objects, sorted by date
        """
        all_matches = self.get_series_matches(series_id, series_name)
        
        today = datetime.now().date()
        cutoff = today + timedelta(days=days_ahead)
        
        upcoming = []
        for m in all_matches:
            # Check if upcoming by status
            if not m.is_upcoming:
                continue
            
            # Also check date if available
            try:
                match_date = datetime.strptime(m.date, '%Y-%m-%d').date()
                if match_date < today:
                    continue
                if match_date > cutoff:
                    continue
            except (ValueError, TypeError):
                pass  # Keep match if date parsing fails
            
            upcoming.append(m)
        
        # Sort by date
        upcoming.sort(key=lambda m: m.date)
        
        logger.info(f"Found {len(upcoming)} upcoming matches in next {days_ahead} days")
        return upcoming
    
    def get_all_upcoming_t20_matches(
        self, 
        gender: Optional[str] = None,
        days_ahead: int = 7
    ) -> List[T20Match]:
        """
        Get all upcoming T20 matches across all active series.
        
        This makes multiple API calls (one per active series).
        Consider caching results.
        
        Args:
            gender: Optional filter - 'male' or 'female'
            days_ahead: How many days ahead to include
            
        Returns:
            List of upcoming T20Match objects from all series, sorted by date
        """
        series_list = self.get_t20_series(gender=gender)
        
        all_matches = []
        for series in series_list:
            try:
                matches = self.get_upcoming_series_matches(
                    series.id, 
                    series.name,
                    days_ahead=days_ahead
                )
                all_matches.extend(matches)
            except Exception as e:
                logger.warning(f"Failed to fetch matches for {series.name}: {e}")
        
        # Sort all by date
        all_matches.sort(key=lambda m: m.date)
        
        logger.info(f"Found {len(all_matches)} total upcoming T20 matches")
        return all_matches
    
    def get_upcoming_matches_24h(self) -> Dict[str, List[T20Match]]:
        """
        Get all T20 matches in the next 24 hours, grouped by series.
        
        Uses /currentMatches API for live/in-progress matches, supplemented with
        series-specific lookups for major leagues (WBBL, BBL) to catch scheduled matches.
        
        Returns:
            Dict mapping series_name to list of T20Match objects
            Each match includes gender for model selection
        """
        from datetime import timezone
        
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=24)
        today_str = now.strftime('%Y-%m-%d')
        tomorrow_str = (now + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Track match IDs to avoid duplicates
        seen_match_ids = set()
        matches_by_series = {}
        
        # PART 1: Use currentMatches for live/in-progress matches
        data = self._request('currentMatches', {'type': 't20'})
        
        if 'data' not in data:
            logger.warning("Failed to fetch current matches")
            return {}
        
        matches_by_series = {}
        
        for match_data in data['data']:
            # IMPORTANT: Filter by matchType - API doesn't always respect type=t20 param
            match_type = match_data.get('matchType', '').lower()
            if match_type != 't20':
                logger.debug(f"Skipping non-T20 match: {match_data.get('name')} (type: {match_type})")
                continue
            
            # Only include matches from today or tomorrow
            match_date = match_data.get('date', '')
            if match_date not in [today_str, tomorrow_str]:
                continue
            
            # Create T20Match object
            match = T20Match.from_api(match_data)
            
            # Skip finished matches
            if not match.is_upcoming:
                logger.debug(f"Skipping finished match: {match.name}")
                continue
            
            # Get series name (or default)
            series_name = match_data.get('series', '') or match_data.get('name', 'Unknown Series')
            # Clean up series name if it includes match details
            if ' vs ' in series_name:
                series_name = 'T20 Matches'
            
            # Format start time
            start_time = ''
            if match.date_time_gmt:
                try:
                    dt = datetime.fromisoformat(match.date_time_gmt.replace('Z', '+00:00'))
                    start_time = dt.strftime('%H:%M GMT')
                except:
                    start_time = match.status
            else:
                start_time = match.status or 'TBD'
            
            # Track to avoid duplicates
            seen_match_ids.add(match.id)
            
            # Group by series
            if series_name not in matches_by_series:
                matches_by_series[series_name] = {
                    'gender': match.gender,
                    'series_id': match.series_id,
                    'matches': []
                }
            
            matches_by_series[series_name]['matches'].append(match)
        
        # PART 2: Supplement with major leagues to catch scheduled matches not in currentMatches
        # These leagues often have tomorrow's matches scheduled but not yet in currentMatches
        major_leagues = [
            ('Womens Big Bash', 'WBBL 2025', 'female'),
            ('Big Bash League', 'BBL 2025-26', 'male'),
        ]
        
        for search_term, display_name, gender in major_leagues:
            try:
                # Find the series
                series_data = self._request('series', {'search': search_term})
                if 'data' not in series_data or not series_data['data']:
                    continue
                
                # Get the most recent series (first result)
                series = series_data['data'][0]
                series_id = series.get('id')
                
                # Get matches for this series
                matches_data = self._request('series_info', {'id': series_id})
                if 'data' not in matches_data:
                    continue
                
                match_list = matches_data['data'].get('matchList', [])
                
                for m in match_list:
                    match_id = m.get('id')
                    if match_id in seen_match_ids:
                        continue  # Already have this match
                    
                    match_date = m.get('date', '')
                    if match_date not in [today_str, tomorrow_str]:
                        continue
                    
                    # Check if within 24 hours
                    date_time_gmt = m.get('dateTimeGMT')
                    if date_time_gmt:
                        try:
                            match_dt = datetime.fromisoformat(date_time_gmt.replace('Z', '+00:00'))
                            if not (now <= match_dt <= cutoff):
                                continue
                        except:
                            pass
                    
                    # Create match object
                    match = T20Match.from_api(m, series_id=series_id, series_name=display_name)
                    
                    # Only include upcoming matches
                    if not match.is_upcoming:
                        continue
                    
                    seen_match_ids.add(match_id)
                    
                    if display_name not in matches_by_series:
                        matches_by_series[display_name] = {
                            'gender': gender,
                            'series_id': series_id,
                            'matches': []
                        }
                    
                    matches_by_series[display_name]['matches'].append(match)
                    
            except Exception as e:
                logger.warning(f"Failed to fetch {search_term} matches: {e}")
        
        # Sort matches within each series by time
        for series_data in matches_by_series.values():
            series_data['matches'].sort(key=lambda x: x.date_time_gmt or x.date)
        
        total = sum(len(s['matches']) for s in matches_by_series.values())
        logger.info(f"Found {total} matches in next 24h across {len(matches_by_series)} series")
        return matches_by_series


# Convenience function
def get_client() -> CricketDataClient:
    """Get a configured CricketDataClient instance."""
    return CricketDataClient()


if __name__ == "__main__":
    # Test the client
    logging.basicConfig(level=logging.INFO)
    
    client = CricketDataClient()
    
    # =========================================================================
    # Test new T20 Series methods
    # =========================================================================
    print("=" * 70)
    print("T20 SERIES EXPLORATION")
    print("=" * 70)
    
    print("\n--- All T20 Series ---")
    all_series = client.get_t20_series()
    print(f"Total T20 series: {len(all_series)}")
    
    print("\n--- Women's T20 Series ---")
    women_series = client.get_t20_series(gender='female')
    for s in women_series[:5]:
        print(f"  {s.name} | {s.t20_count} T20s | {s.start_date} to {s.end_date}")
    
    print("\n--- Men's T20 Series ---")
    men_series = client.get_t20_series(gender='male')
    for s in men_series[:5]:
        print(f"  {s.name} | {s.t20_count} T20s | {s.start_date} to {s.end_date}")
    
    # Test getting matches from a specific series
    if women_series:
        print("\n" + "=" * 70)
        print(f"UPCOMING MATCHES: {women_series[0].name}")
        print("=" * 70)
        
        upcoming = client.get_upcoming_series_matches(
            women_series[0].id, 
            women_series[0].name,
            days_ahead=14
        )
        for m in upcoming[:5]:
            print(f"\n{m.date}: {m.name}")
            print(f"  Teams: {m.team1} vs {m.team2}")
            print(f"  Gender: {m.gender}")
            print(f"  Status: {m.status}")
            print(f"  Has Squad: {m.has_squad}")
    
    # =========================================================================
    # Legacy WBBL test (backward compatibility)
    # =========================================================================
    print("\n" + "=" * 70)
    print("WBBL Upcoming Matches (Legacy Method)")
    print("=" * 70)
    
    upcoming = client.get_upcoming_wbbl_matches()
    for match in upcoming[:3]:
        print(f"\n{match.date}: {match.name}")
        print(f"  Teams: {match.team1} vs {match.team2}")
        print(f"  Status: {match.status}")
        print(f"  Has Squad: {match.has_squad}")

