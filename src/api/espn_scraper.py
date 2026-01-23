"""
ESPN Cricinfo Web Scraper.

Scrapes T20 match schedules, venues, and squad data from ESPN Cricinfo.
Uses requests + BeautifulSoup - no JavaScript/Playwright needed since
ESPN uses Next.js which embeds complete data in __NEXT_DATA__ script tags.

Key pages:
- Schedule: /live-cricket-match-schedule-fixtures?quick_class_id=t20
- Match Squads: /series/{slug}/match-squads (contains venue + FULL squad JSON with 15-20 players)
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from functools import lru_cache

import requests
from bs4 import BeautifulSoup

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.database import get_connection
from src.data.venue_normalizer import venue_similarity, normalize_venue_name
from src.features.name_matcher import PlayerNameMatcher

logger = logging.getLogger(__name__)


@dataclass
class ESPNPlayer:
    """Player data from ESPN."""
    espn_id: int
    name: str
    long_name: str
    role: str  # 'C' (captain), 'WK' (wicketkeeper), 'P' (player)
    playing_roles: List[str]  # e.g., ['bowler'], ['wicketkeeper batter']
    batting_styles: List[str]  # e.g., ['rhb', 'lhb']
    bowling_styles: List[str]  # e.g., ['rmf', 'sla']
    is_overseas: bool = False
    db_player_id: Optional[int] = None  # Matched database player ID


@dataclass
class ESPNTeam:
    """Team data from ESPN."""
    espn_id: int
    name: str
    long_name: str
    abbreviation: str
    players: List[ESPNPlayer] = field(default_factory=list)
    db_team_id: Optional[int] = None  # Matched database team ID


@dataclass
class ESPNVenue:
    """Venue data from ESPN."""
    name: str
    small_name: str
    town: str
    country: str
    db_venue_id: Optional[int] = None  # Matched database venue ID


@dataclass
class ESPNMatch:
    """Match data from ESPN."""
    espn_id: int
    slug: str
    title: str  # Clean "Team1 vs Team2" format
    series_name: str
    series_id: int
    status: str
    start_date: str  # YYYY-MM-DD
    start_time: Optional[str]  # HH:MM AM/PM (local time from ESPN)
    date_time_gmt: Optional[str]  # Full ISO datetime in GMT
    match_url: str
    team1_name: str = ''  # Parsed team name from schedule
    team2_name: str = ''  # Parsed team name from schedule
    match_type: str = ''  # e.g., "4th Match", "Final"
    venue_city: str = ''  # Venue city from schedule (e.g., "Sydney")
    team1: Optional[ESPNTeam] = None  # Full team data from match details
    team2: Optional[ESPNTeam] = None  # Full team data from match details
    venue: Optional[ESPNVenue] = None
    gender: str = 'male'  # Detected from series/team names
    has_squads: bool = False


class ESPNCricInfoScraper:
    """
    Scraper for ESPN Cricinfo T20 match data.
    
    Usage:
        scraper = ESPNCricInfoScraper()
        matches = scraper.get_t20_schedule()
        match_details = scraper.get_match_details(matches[0].match_url)
    """
    
    BASE_URL = "https://www.espncricinfo.com"
    SCHEDULE_URL = f"{BASE_URL}/live-cricket-match-schedule-fixtures"
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    
    # Keywords to detect women's cricket
    WOMEN_KEYWORDS = ['women', 'wbbl', 'wpl', "women's", 'wpsl', 'wt20', 'female']
    
    def __init__(self, request_delay: float = 1.0):
        """
        Initialize scraper.
        
        Args:
            request_delay: Seconds to wait between requests (rate limiting)
        """
        self.request_delay = request_delay
        self._last_request_time = 0
        self._name_matcher = None
    
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request_time = time.time()
    
    def _fetch(self, url: str) -> Optional[str]:
        """
        Fetch a URL with rate limiting and error handling.
        
        Args:
            url: URL to fetch
            
        Returns:
            HTML content or None on error
        """
        self._rate_limit()
        
        try:
            response = requests.get(url, headers=self.HEADERS, timeout=15)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None
    
    def _parse_next_data(self, html: str) -> Optional[dict]:
        """
        Parse __NEXT_DATA__ JSON from ESPN page.
        
        Args:
            html: Page HTML content
            
        Returns:
            Parsed JSON data or None
        """
        soup = BeautifulSoup(html, 'lxml')
        script = soup.find('script', {'id': '__NEXT_DATA__'})
        
        if not script:
            logger.warning("No __NEXT_DATA__ script found")
            return None
        
        try:
            return json.loads(script.get_text())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse __NEXT_DATA__: {e}")
            return None
    
    def _detect_gender(self, text: str) -> str:
        """Detect gender from series/team names."""
        text_lower = text.lower()
        return 'female' if any(kw in text_lower for kw in self.WOMEN_KEYWORDS) else 'male'
    
    def get_t20_schedule(self, hours_ahead: int = 24, hours_behind: int = 3) -> List[ESPNMatch]:
        """
        Get upcoming T20 matches from schedule page.
        
        Args:
            hours_ahead: Only include matches starting within this many hours (default 24)
            hours_behind: Also include matches from this many hours ago (for in-progress, default 3)
            
        Returns:
            List of ESPNMatch objects (without squad details), sorted by datetime
        """
        url = f"{self.SCHEDULE_URL}?quick_class_id=t20"
        html = self._fetch(url)
        
        if not html:
            return []
        
        soup = BeautifulSoup(html, 'lxml')
        matches = []
        
        now = datetime.now(timezone.utc)
        earliest = now - timedelta(hours=hours_behind)
        latest = now + timedelta(hours=hours_ahead)
        
        # Find all match links
        for link in soup.select('a[href*="/series/"]'):
            href = link.get('href', '')
            
            # Only match URLs that look like match pages (live-cricket-score or match-preview or match-squads)
            if '/live-cricket-score' not in href and '/match-preview' not in href and '/match-squads' not in href:
                continue
            
            text = link.get_text(' ', strip=True)
            if not text:
                continue
            
            # Parse the match info from rich text format
            match = self._parse_schedule_entry(text, href)
            if match:
                # Skip matches without proper datetime (required for filtering)
                if not match.date_time_gmt:
                    logger.debug(f"Skipping match without datetime: {match.title}")
                    continue
                
                # Filter out non-T20 formats (Test, 4-day, First-class, ODI)
                text_lower = text.lower()
                series_lower = match.series_name.lower()
                match_type_lower = match.match_type.lower()
                
                # Non-T20 keywords to filter
                non_t20_keywords = [
                    'test', 'ashes', '4-day', '4 day', 'four day', 'first-class', 'first class',
                    'day 1', 'day 2', 'day 3', 'day 4', 'day 5', 'ranji', 'sheffield shield',
                    'county championship', 'csa 4', 'day series'
                ]
                
                if any(term in text_lower or term in series_lower for term in non_t20_keywords):
                    logger.debug(f"Skipping non-T20 match: {match.title} ({match.series_name})")
                    continue
                
                # Check match_type for Test format
                if 'test' in match_type_lower:
                    logger.debug(f"Skipping Test match: {match.title}")
                    continue
                
                # Filter by date/time
                try:
                    dt = datetime.fromisoformat(match.date_time_gmt.replace('Z', '+00:00'))
                    # Skip matches outside our time window
                    if dt < earliest or dt > latest:
                        continue
                except ValueError:
                    continue  # Skip matches with unparseable dates
                
                matches.append(match)
        
        # Sort by datetime (matches with dates first, then those without)
        def sort_key(m):
            if m.date_time_gmt:
                try:
                    return datetime.fromisoformat(m.date_time_gmt.replace('Z', '+00:00'))
                except ValueError:
                    pass
            return datetime.max.replace(tzinfo=timezone.utc)
        
        matches.sort(key=sort_key)
        
        logger.info(f"Found {len(matches)} T20 matches in schedule (window: -{hours_behind}h to +{hours_ahead}h)")
        return matches
    
    def _parse_schedule_entry(self, text: str, href: str) -> Optional[ESPNMatch]:
        """
        Parse a schedule entry from the schedule page.
        
        Rich text format from ESPN:
        "Sydney Sixer v Adelaide Striker 4th Match (N), Sydney, December 17, 2025 Big Bash League 7:15 PM"
        
        Args:
            text: Link text content
            href: Link URL
            
        Returns:
            ESPNMatch or None
        """
        # Extract match ID from URL
        # Format: /series/series-slug-1234567/team1-vs-team2-match-1234568/live-cricket-score (or /match-preview or /match-squads)
        match_id_match = re.search(r'-(\d+)/(?:live-cricket-score|match-preview|match-squads)', href)
        series_id_match = re.search(r'/series/[^/]+-(\d+)/', href)
        
        if not match_id_match:
            return None
        
        espn_id = int(match_id_match.group(1))
        series_id = int(series_id_match.group(1)) if series_id_match else 0
        
        # Extract match slug
        match_slug = href.split('/')[-2] if '/' in href else ''
        
        # Initialize variables
        team1_name = ''
        team2_name = ''
        match_type = ''
        venue_city = ''
        start_date = ''
        start_time = ''
        date_time_gmt = None
        series_name = ''
        
        # Parse ESPN schedule text format
        # Actual format: "8:15 AM 8:15 am GMT 7:15 pm Local Sydney Sixers vs Adelaide Strikers 4th Match (N), Sydney, December 17, 2025"
        # Or with "Not covered live": "6:30 AM 6:30 am GMT 1:30 pm Local Not covered live Myanmar Women vs Singapore Women..."
        
        # First, extract the GMT time from the prefix
        gmt_time_match = re.search(r'(\d{1,2}:\d{2})\s*(am|pm)\s*GMT', text, re.I)
        if gmt_time_match:
            start_time = gmt_time_match.group(1) + ' ' + gmt_time_match.group(2).upper()
        
        # Strip the time prefixes to get the match info
        # Remove: "8:15 AM 8:15 am GMT 7:15 pm Local" or "8:15 AM 8:15 am GMT 7:15 pm Local Not covered live"
        clean_text = re.sub(r'^\d{1,2}:\d{2}\s*[AP]M\s*\d{1,2}:\d{2}\s*[ap]m\s*GMT\s*\d{1,2}:\d{2}\s*[ap]m\s*Local\s*(?:Not covered live\s*)?', '', text, flags=re.I)
        clean_text = clean_text.strip()
        
        # Now parse the clean text using a smarter approach
        # Format: "Team1 vs Team2 MatchType, [GroupInfo,] Venue, Month Day, Year..."
        # Examples: 
        #   "Sydney Sixers vs Adelaide Strikers 4th Match (N), Sydney, December 17, 2025"
        #   "Myanmar Women vs Singapore Women Int'l 6th Match, Group A, Bangkok, December 17, 2025"
        
        # Step 1: Find the date first (Month Day, Year)
        date_pattern = r'(\w+\s+\d{1,2},\s*\d{4})'
        date_match = re.search(date_pattern, clean_text)
        
        if date_match:
            date_str = date_match.group(1).strip()
            before_date = clean_text[:date_match.start()].rstrip(', ')
            
            # Step 2: Split by comma to get parts: [teams+matchtype, ..., venue]
            parts = [p.strip() for p in before_date.split(', ')]
            
            if len(parts) >= 2:
                # Last part is venue city
                venue_city = parts[-1]
                # First part contains teams and match type
                teams_and_type = parts[0]
                
                # Step 3: Parse teams and match type from first part
                # Pattern: "Team1 vs Team2 MatchType" or "Team1 v Team2 MatchType"
                teams_pattern = r'^(.+?)\s+vs?\s+(.+?)\s+((?:Int\'?l\s+)?(?:\d+(?:st|nd|rd|th)\s+)?(?:Match|T20I|T20|Final|Semi-final|Eliminator|Qualifier|place).*)$'
                teams_match = re.match(teams_pattern, teams_and_type, re.I)
                
                if teams_match:
                    team1_name = teams_match.group(1).strip()
                    team2_name = teams_match.group(2).strip()
                    match_type = teams_match.group(3).strip()
                    
                    # Extract series name from URL slug
                    series_match = re.search(r'/series/([^/]+)-\d+/', href)
                    series_slug = series_match.group(1) if series_match else ''
                    series_name = series_slug.replace('-', ' ').title()
                    
                    # Parse date and time to GMT ISO format
                    try:
                        if start_time:
                            dt_str = f"{date_str} {start_time}"
                            dt = datetime.strptime(dt_str, "%B %d, %Y %I:%M %p")
                        else:
                            dt_str = date_str
                            dt = datetime.strptime(date_str, "%B %d, %Y")
                        start_date = dt.strftime("%Y-%m-%d")
                        date_time_gmt = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                    except ValueError as e:
                        logger.debug(f"Failed to parse date/time: {dt_str} - {e}")
        else:
            # Fallback: parse from URL slug
            series_match = re.search(r'/series/([^/]+)-\d+/', href)
            series_slug = series_match.group(1) if series_match else ''
            series_name = series_slug.replace('-', ' ').title()
            
            # Parse team names from slug
            slug_without_id = re.sub(r'-\d+$', '', match_slug)
            
            if '-vs-' in slug_without_id:
                parts = slug_without_id.split('-vs-')
                team1_name = parts[0].replace('-', ' ').title()
                team2_part = parts[1] if len(parts) > 1 else ''
                
                # Extract match type
                match_type_match = re.search(r'(\d+(?:st|nd|rd|th)-match|final|eliminator|qualifier|semi-final|3rd-place-play-off|group-[a-z])', team2_part, re.I)
                if match_type_match:
                    raw_type = match_type_match.group(1).replace('-', ' ')
                    words = raw_type.split()
                    formatted_words = []
                    for word in words:
                        if re.match(r'^\d+(st|nd|rd|th)$', word, re.I):
                            formatted_words.append(word.lower())
                        else:
                            formatted_words.append(word.capitalize())
                    match_type = ' '.join(formatted_words)
                    team2_name = team2_part[:match_type_match.start()].rstrip('-').replace('-', ' ').title()
                else:
                    team2_name = team2_part.replace('-', ' ').title()
            
            # Try to extract time from text (old format fallback)
            ampm_match = re.search(r'(\d{1,2}:\d{2})\s*(am|pm)\s*GMT', text, re.I)
            if ampm_match:
                start_time = ampm_match.group(1) + ' ' + ampm_match.group(2).upper()
            else:
                time_match = re.search(r'(\d{1,2}:\d{2})\s*[AP]M', text, re.I)
                if time_match:
                    start_time = time_match.group(0)
        
        # Clean up team names
        team1_name = team1_name.strip()
        team2_name = team2_name.strip()
        
        # Handle TBA entries
        if team1_name.lower().startswith('tba'):
            team1_name = 'TBA'
        if team2_name.lower().startswith('tba'):
            team2_name = 'TBA'
        
        # Build clean title
        if team1_name and team2_name:
            title = f"{team1_name} vs {team2_name}"
        else:
            title = match_slug.replace('-', ' ').title()[:60]
        
        # Detect gender
        gender = self._detect_gender(text + series_name)
        
        # Build match URL for details page - use match-squads to get FULL squad (15-20 players)
        match_url = self.BASE_URL + href.replace('/live-cricket-score', '/match-squads')
        
        return ESPNMatch(
            espn_id=espn_id,
            slug=match_slug,
            title=title,
            series_name=series_name,
            series_id=series_id,
            status='scheduled',
            start_date=start_date,
            start_time=start_time,
            date_time_gmt=date_time_gmt,
            match_url=match_url,
            team1_name=team1_name,
            team2_name=team2_name,
            match_type=match_type,
            venue_city=venue_city,
            gender=gender
        )
    
    def get_match_details(self, match_url: str) -> Optional[ESPNMatch]:
        """
        Get full match details including venue and squads.
        
        Args:
            match_url: URL to match-squads page (for full squad) or match-preview page
            
        Returns:
            ESPNMatch with full details or None
        """
        html = self._fetch(match_url)
        if not html:
            return None
        
        data = self._parse_next_data(html)
        if not data:
            return None
        
        try:
            app_props = data.get('props', {}).get('appPageProps', {})
            match_data = app_props.get('data', {}).get('match', {})
            content = app_props.get('data', {}).get('content', {})
            
            if not match_data:
                logger.warning(f"No match data in response for {match_url}")
                return None
            
            return self._parse_match_data(match_data, content, match_url)
            
        except Exception as e:
            logger.error(f"Error parsing match data from {match_url}: {e}")
            return None
    
    def _parse_match_data(
        self, 
        match_data: dict, 
        content: dict, 
        match_url: str
    ) -> ESPNMatch:
        """Parse match data from __NEXT_DATA__ JSON."""
        
        # Basic match info
        espn_id = match_data.get('objectId', 0)
        slug = match_data.get('slug', '')
        title = match_data.get('title', '')
        status = match_data.get('statusText', 'scheduled')
        
        # Series info
        series = match_data.get('series', {})
        series_name = series.get('name', '')
        series_id = series.get('objectId', 0)
        
        # Timing
        start_date = match_data.get('startDate', '')
        start_time = match_data.get('startTime', '')
        
        # Build ISO datetime
        date_time_gmt = None
        if start_date and start_time:
            date_time_gmt = f"{start_date}T{start_time}Z"
        elif start_date:
            date_time_gmt = f"{start_date}T00:00:00Z"
        
        # Gender detection
        gender = self._detect_gender(series_name + title)
        
        # Parse venue
        venue = self._parse_venue(match_data.get('ground', {}))
        
        # Parse teams and squads
        teams_data = match_data.get('teams', [])
        team1 = None
        team2 = None
        
        # Get squad data from content
        match_players = content.get('matchPlayers', {})
        team_players = match_players.get('teamPlayers', []) if isinstance(match_players, dict) else []
        
        if len(teams_data) >= 1:
            team1 = self._parse_team(teams_data[0], team_players)
        if len(teams_data) >= 2:
            team2 = self._parse_team(teams_data[1], team_players)
        
        has_squads = bool(team_players and (
            (team1 and team1.players) or (team2 and team2.players)
        ))
        
        return ESPNMatch(
            espn_id=espn_id,
            slug=slug,
            title=title,
            series_name=series_name,
            series_id=series_id,
            status=status,
            start_date=start_date,
            start_time=start_time,
            date_time_gmt=date_time_gmt,
            match_url=match_url,
            team1=team1,
            team2=team2,
            venue=venue,
            gender=gender,
            has_squads=has_squads
        )
    
    def _parse_venue(self, ground_data: dict) -> Optional[ESPNVenue]:
        """Parse venue from ground data."""
        if not ground_data:
            return None
        
        name = ground_data.get('name', '')
        small_name = ground_data.get('smallName', '')
        
        # Town can be nested dict or string
        town_data = ground_data.get('town', {})
        town = town_data.get('name', '') if isinstance(town_data, dict) else str(town_data)
        
        # Country can be nested dict or string
        country_data = ground_data.get('country', {})
        country = country_data.get('name', '') if isinstance(country_data, dict) else str(country_data)
        
        return ESPNVenue(
            name=name,
            small_name=small_name,
            town=town,
            country=country
        )
    
    def _parse_team(
        self, 
        team_data: dict, 
        all_team_players: List[dict]
    ) -> Optional[ESPNTeam]:
        """Parse team and its players."""
        if not team_data:
            return None
        
        # Team data might be nested under 'team' key
        if 'team' in team_data:
            team_info = team_data['team']
        else:
            team_info = team_data
        
        espn_id = team_info.get('objectId', team_info.get('id', 0))
        name = team_info.get('name', '')
        long_name = team_info.get('longName', name)
        abbreviation = team_info.get('abbreviation', '')
        
        # Find this team's players from the squad data
        players = []
        for team_entry in all_team_players:
            team_obj = team_entry.get('team', {})
            if team_obj.get('objectId') == espn_id or team_obj.get('id') == espn_id:
                # Found matching team's player list
                for player_entry in team_entry.get('players', []):
                    player = self._parse_player(player_entry)
                    if player:
                        players.append(player)
                break
        
        return ESPNTeam(
            espn_id=espn_id,
            name=name,
            long_name=long_name,
            abbreviation=abbreviation,
            players=players
        )
    
    def _parse_player(self, player_entry: dict) -> Optional[ESPNPlayer]:
        """Parse player from squad entry."""
        player_data = player_entry.get('player', {})
        if not player_data:
            return None
        
        espn_id = player_data.get('objectId', player_data.get('id', 0))
        name = player_data.get('name', '')
        long_name = player_data.get('longName', name)
        
        role = player_entry.get('playerRoleType', 'P')
        playing_roles = player_data.get('playingRoles', [])
        batting_styles = player_data.get('battingStyles', [])
        bowling_styles = player_data.get('bowlingStyles', [])
        is_overseas = player_entry.get('isOverseas', False)
        
        return ESPNPlayer(
            espn_id=espn_id,
            name=name,
            long_name=long_name,
            role=role,
            playing_roles=playing_roles,
            batting_styles=batting_styles,
            bowling_styles=bowling_styles,
            is_overseas=is_overseas
        )
    
    # =========================================================================
    # Database Matching Methods
    # =========================================================================
    
    def match_venue_to_db(
        self, 
        venue: ESPNVenue, 
        gender: str = 'male'
    ) -> Optional[Tuple[int, str]]:
        """
        Match ESPN venue to database venue.
        
        Args:
            venue: ESPN venue data
            gender: 'male' or 'female' for venue filtering
            
        Returns:
            (venue_id, venue_name) or None
        """
        if not venue:
            return None
        
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get all venues with enough matches, ordered by match count (prefer more popular venues)
        cursor.execute("""
            SELECT v.venue_id, v.name, v.city, v.country, COUNT(*) as match_count
            FROM venues v
            JOIN matches m ON m.venue_id = v.venue_id
            WHERE m.match_type = 'T20' AND m.gender = ?
            GROUP BY v.venue_id
            HAVING COUNT(*) >= 3
            ORDER BY match_count DESC
        """, (gender,))
        
        db_venues = cursor.fetchall()
        conn.close()
        
        best_match = None
        best_score = 0.0
        best_match_count = 0
        
        espn_venue_name = venue.name
        
        for row in db_venues:
            db_name = row['name']
            match_count = row['match_count']
            score = venue_similarity(espn_venue_name, db_name)
            
            # Boost score if city matches
            if venue.town and row['city']:
                if venue.town.lower() in row['city'].lower():
                    score += 0.1
            
            # Update if score is better, OR if score is equal but this venue has more matches
            if score > best_score or (score == best_score and match_count > best_match_count):
                best_score = score
                best_match = (row['venue_id'], row['name'])
                best_match_count = match_count
        
        if best_match and best_score >= 0.7:
            logger.info(f"Matched venue '{espn_venue_name}' to '{best_match[1]}' (score: {best_score:.2f})")
            return best_match
        
        logger.warning(f"No database match for venue: {espn_venue_name}")
        return None
    
    def match_team_to_db(
        self, 
        team: ESPNTeam, 
        gender: str = 'male'
    ) -> Optional[Tuple[int, str]]:
        """
        Match ESPN team to database team.
        
        Args:
            team: ESPN team data
            gender: 'male' or 'female' for team filtering
            
        Returns:
            (team_id, team_name) or None
        """
        if not team:
            return None
        
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT t.team_id, t.name
            FROM teams t
            JOIN matches m ON t.team_id IN (m.team1_id, m.team2_id)
            WHERE m.match_type = 'T20' AND m.gender = ?
        """, (gender,))
        
        db_teams = cursor.fetchall()
        conn.close()
        
        espn_names = [team.long_name, team.name, team.abbreviation]
        
        best_match = None
        best_score = 0.0
        
        for row in db_teams:
            db_name = row['name']
            
            for espn_name in espn_names:
                # Normalize for comparison
                espn_norm = espn_name.lower().replace(' women', '').strip()
                db_norm = db_name.lower().strip()
                
                # Exact match
                if espn_norm == db_norm:
                    return (row['team_id'], row['name'])
                
                # Fuzzy match
                from difflib import SequenceMatcher
                score = SequenceMatcher(None, espn_norm, db_norm).ratio()
                
                if score > best_score:
                    best_score = score
                    best_match = (row['team_id'], row['name'])
        
        if best_match and best_score >= 0.7:
            logger.info(f"Matched team '{team.long_name}' to '{best_match[1]}' (score: {best_score:.2f})")
            return best_match
        
        logger.warning(f"No database match for team: {team.long_name}")
        return None
    
    def match_players_to_db(
        self, 
        team: ESPNTeam, 
        db_team_name: str,
        gender: str = 'male'
    ) -> List[ESPNPlayer]:
        """
        Match ESPN players to database players.
        
        Args:
            team: ESPN team with players
            db_team_name: Matched database team name
            gender: 'male' or 'female'
            
        Returns:
            Players with db_player_id populated where matched
        """
        if not team or not team.players:
            return []
        
        if self._name_matcher is None:
            self._name_matcher = PlayerNameMatcher()
        
        matched_players = []
        
        for player in team.players:
            result = self._name_matcher.find_player(
                player.long_name or player.name,
                db_team_name,
                espn_player_id=player.espn_id
            )
            
            if result:
                player.db_player_id = result.player_id
                logger.debug(f"Matched '{player.name}' to '{result.db_name}' (ID: {result.player_id}, method: {result.method})")
            
            matched_players.append(player)
        
        matched_count = sum(1 for p in matched_players if p.db_player_id)
        logger.info(f"Matched {matched_count}/{len(matched_players)} players for {db_team_name}")
        
        return matched_players


# =========================================================================
# Convenience Functions
# =========================================================================

_scraper_instance = None


def get_scraper() -> ESPNCricInfoScraper:
    """Get or create singleton scraper instance."""
    global _scraper_instance
    if _scraper_instance is None:
        _scraper_instance = ESPNCricInfoScraper()
    return _scraper_instance


if __name__ == "__main__":
    # Test the scraper
    logging.basicConfig(level=logging.INFO)
    
    scraper = ESPNCricInfoScraper(request_delay=0.5)
    
    print("=" * 70)
    print("ESPN CRICINFO SCRAPER TEST")
    print("=" * 70)
    
    # Test 1: Get schedule
    print("\n--- T20 Schedule (next 24h) ---")
    matches = scraper.get_t20_schedule(hours_ahead=24)
    print(f"Found {len(matches)} matches")
    
    for m in matches[:5]:
        print(f"\n  {m.title[:50]}...")
        print(f"    Series: {m.series_name}")
        print(f"    Gender: {m.gender}")
        print(f"    URL: {m.match_url[:60]}...")
    
    # Test 2: Get match details for first match
    if matches:
        print("\n\n--- Match Details ---")
        details = scraper.get_match_details(matches[0].match_url)
        
        if details:
            print(f"\nMatch: {details.title}")
            print(f"Series: {details.series_name}")
            print(f"Date: {details.start_date} {details.start_time or ''}")
            
            if details.venue:
                print(f"\nVenue: {details.venue.name}")
                print(f"  City: {details.venue.town}")
                print(f"  Country: {details.venue.country}")
            
            for i, team in enumerate([details.team1, details.team2], 1):
                if team:
                    print(f"\nTeam {i}: {team.long_name}")
                    print(f"  Players: {len(team.players)}")
                    for p in team.players[:5]:
                        role_str = f"({p.role})" if p.role != 'P' else ''
                        roles = ', '.join(p.playing_roles) if p.playing_roles else 'Unknown'
                        print(f"    - {p.name} {role_str}: {roles}")

