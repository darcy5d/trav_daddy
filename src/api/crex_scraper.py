"""
CREX Cricket Scraper.

Scrapes T20 match schedules, venues, and squad data from crex.com.
Replaces the ESPN Cricinfo scraper with a more reliable HTML-based approach.

Key pages:
- Schedule: https://crex.com/schedule
- Match Info: https://crex.com/scoreboard/{match_id}/{series_id}/{match_type}/{team1_id}/{team2_id}/{slug}/info
- Live Match: https://crex.com/scoreboard/.../live
"""

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.database import get_connection
from src.data.venue_normalizer import venue_similarity
from src.features.name_matcher import PlayerNameMatcher

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class CREXPlayer:
    """Player data from CREX."""
    crex_id: str  # e.g., "92", "E9U" - extracted from player URL
    name: str  # Full name
    short_name: str  # Abbreviated name shown on cards
    role: str  # "Batter", "Bowler", "All Rounder", "WK"
    is_captain: bool = False
    is_wicketkeeper: bool = False
    is_overseas: bool = False
    db_player_id: Optional[int] = None  # Matched database player ID


@dataclass
class CREXTeam:
    """Team data from CREX."""
    crex_id: str  # e.g., "N0", "MY", "2F"
    name: str  # Full team name
    abbreviation: str  # Short code
    players: List[CREXPlayer] = field(default_factory=list)
    db_team_id: Optional[int] = None  # Matched database team ID


@dataclass
class CREXVenue:
    """Venue data from CREX."""
    name: str  # Full venue name
    city: str  # City (extracted from name if possible)
    db_venue_id: Optional[int] = None  # Matched database venue ID


@dataclass
class CREXVenueStats:
    """Venue statistics from CREX match info page."""
    matches_played: int = 0
    win_bat_first_pct: float = 0.0
    win_bowl_first_pct: float = 0.0
    avg_first_innings: float = 0.0
    avg_second_innings: float = 0.0
    highest_total: Optional[str] = None
    lowest_total: Optional[str] = None
    highest_chased: Optional[str] = None
    pace_wickets: int = 0
    spin_wickets: int = 0
    pace_pct: float = 0.0
    spin_pct: float = 0.0


@dataclass
class CREXMatch:
    """Match data from CREX."""
    crex_id: str  # Match ID from URL (e.g., "ZEE", "VI4")
    series_id: str  # Series ID from URL (e.g., "2BL", "1QU")
    slug: str  # URL slug
    title: str  # "Team1 vs Team2"
    team1_name: str
    team2_name: str
    team1_id: str  # CREX team ID
    team2_id: str  # CREX team ID
    match_type: str  # "4th Match", "Qualifier 2", "Final"
    series_name: str
    format_type: str  # "T20", "ODI"
    status: str  # "upcoming", "live", "completed"
    start_date: Optional[str] = None  # YYYY-MM-DD
    start_time: Optional[str] = None  # HH:MM AM/PM
    date_time_gmt: Optional[str] = None  # ISO format
    match_url: str = ''
    venue: Optional[CREXVenue] = None
    venue_stats: Optional[CREXVenueStats] = None
    gender: str = 'male'
    team1: Optional[CREXTeam] = None
    team2: Optional[CREXTeam] = None
    toss_winner: Optional[str] = None
    toss_decision: Optional[str] = None  # "bat" or "field"
    playing_xi_available: bool = False
    has_squads: bool = False


# ============================================================================
# CREX Scraper Class
# ============================================================================

class CREXScraper:
    """
    Scraper for CREX cricket match data.
    
    Usage:
        scraper = CREXScraper()
        matches = scraper.get_schedule()
        match_details = scraper.get_match_details(matches[0].match_url)
    """
    
    BASE_URL = "https://crex.com"
    SCHEDULE_URL = f"{BASE_URL}/schedule"
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    
    # Keywords to detect women's cricket
    WOMEN_KEYWORDS = ['women', 'wbbl', 'wpl', "women's", 'wpsl', 'wt20', 'female', '-w', ' w ']
    
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
    
    def _fetch(self, url: str, timeout: int = 15) -> Optional[str]:
        """
        Fetch a URL with rate limiting and error handling.
        
        Args:
            url: URL to fetch
            timeout: Request timeout in seconds
            
        Returns:
            HTML content or None on error
        """
        self._rate_limit()
        
        try:
            response = requests.get(url, headers=self.HEADERS, timeout=timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None
    
    def _detect_gender(self, text: str) -> str:
        """Detect gender from team/series names."""
        text_lower = text.lower()
        return 'female' if any(kw in text_lower for kw in self.WOMEN_KEYWORDS) else 'male'
    
    def _extract_city_from_venue(self, venue_name: str) -> str:
        """Extract city from venue name like 'The Wanderers Stadium, Johannesburg'."""
        if ',' in venue_name:
            return venue_name.split(',')[-1].strip()
        return ''
    
    def _parse_crex_datetime(self, date_str: str, time_str: Optional[str] = None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parse CREX date/time formats.
        
        CREX is based in India, so all times are in IST (UTC+5:30).
        We convert to UTC for date_time_gmt.
        
        Args:
            date_str: e.g., "Saturday, 24 January, 10:00 AM" or "Friday, 23 January"
            time_str: Optional separate time string
            
        Returns:
            (start_date YYYY-MM-DD, start_time HH:MM AM/PM in IST, date_time_gmt ISO in UTC)
        """
        try:
            # Try to extract time from date_str if it contains it
            time_match = re.search(r'(\d{1,2}:\d{2}\s*(?:AM|PM))', date_str, re.I)
            if time_match:
                time_str = time_match.group(1)
                date_str = date_str[:time_match.start()].strip().rstrip(',')
            
            # Parse date: "Saturday, 24 January" or "Friday, 23 January"
            # Remove day name
            date_str = re.sub(r'^[A-Za-z]+,\s*', '', date_str)
            
            # Try different date formats
            current_year = datetime.now().year
            
            # "24 January" format
            date_match = re.match(r'(\d{1,2})\s+([A-Za-z]+)', date_str)
            if date_match:
                day = int(date_match.group(1))
                month_name = date_match.group(2)
                
                # Parse month
                try:
                    dt = datetime.strptime(f"{day} {month_name} {current_year}", "%d %B %Y")
                except ValueError:
                    dt = datetime.strptime(f"{day} {month_name} {current_year}", "%d %b %Y")
                
                # If date is in the past (more than 30 days ago), assume next year
                if dt < datetime.now() - timedelta(days=30):
                    dt = dt.replace(year=current_year + 1)
                
                start_date = dt.strftime("%Y-%m-%d")
                
                # Parse time if available (CREX times are in IST = UTC+5:30)
                start_time = None
                date_time_gmt = None
                
                if time_str:
                    start_time = time_str.strip()
                    try:
                        time_obj = datetime.strptime(time_str.strip(), "%I:%M %p")
                        dt = dt.replace(hour=time_obj.hour, minute=time_obj.minute)
                        
                        # Convert IST to UTC (subtract 5 hours 30 minutes)
                        # IST is UTC+5:30
                        dt_utc = dt - timedelta(hours=5, minutes=30)
                        date_time_gmt = dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
                    except ValueError:
                        pass
                
                return start_date, start_time, date_time_gmt
        
        except Exception as e:
            logger.debug(f"Failed to parse date/time '{date_str}' / '{time_str}': {e}")
        
        return None, None, None

    # =========================================================================
    # Schedule Parsing
    # =========================================================================
    
    def get_schedule(self, formats: List[str] = None) -> List[CREXMatch]:
        """
        Get upcoming matches from CREX schedule page.
        
        Args:
            formats: List of formats to include, e.g., ['T20', 'ODI']. None = all.
            
        Returns:
            List of CREXMatch objects with basic info
        """
        html = self._fetch(self.SCHEDULE_URL)
        if not html:
            return []
        
        soup = BeautifulSoup(html, 'html.parser')
        matches = []
        
        # Find all match links - they point to /scoreboard/...
        for link in soup.select('a[href*="/scoreboard/"]'):
            href = link.get('href', '')
            
            # Skip if not a valid match URL
            if '/scoreboard/' not in href:
                continue
            
            # Parse match from the link
            match = self._parse_schedule_entry(link, href)
            if match:
                # Filter by format if specified
                if formats and match.format_type not in formats:
                    continue
                matches.append(match)
        
        logger.info(f"Found {len(matches)} matches from CREX schedule")
        return matches
    
    def _parse_schedule_entry(self, link_element, href: str) -> Optional[CREXMatch]:
        """
        Parse a match entry from the schedule page.
        
        Args:
            link_element: BeautifulSoup element for the match link
            href: Link URL
            
        Returns:
            CREXMatch or None
        """
        try:
            # URL format: /scoreboard/{match_id}/{series_id}/{match_type}/{team1_id}/{team2_id}/{slug}/{page}
            # Example: /scoreboard/ZEE/2BL/4th-Match/2F/WP/bhu-w-vs-mas-w-4th-match-bhutan-womens-t20i-tri-series-2026/info
            # Note: split('/') gives ['', 'scoreboard', 'ZEE', ...]
            
            parts = href.strip('/').split('/')
            if len(parts) < 7:
                return None
            
            # Extract IDs from URL (after stripping leading slash)
            # parts = ['scoreboard', 'ZEE', '2BL', '4th-Match', '2F', 'WP', 'slug', 'info']
            match_id = parts[1] if len(parts) > 1 else ''
            series_id = parts[2] if len(parts) > 2 else ''
            match_type_raw = parts[3] if len(parts) > 3 else ''
            team1_id = parts[4] if len(parts) > 4 else ''
            team2_id = parts[5] if len(parts) > 5 else ''
            slug = parts[6] if len(parts) > 6 else ''
            
            # Clean up match type
            match_type = match_type_raw.replace('-', ' ').title()
            
            # Extract team names from the link content
            # Look for team name images or text
            team1_name = ''
            team2_name = ''
            
            # Try to find team images (they have alt text with team names)
            team_images = link_element.find_all('img', src=lambda x: x and 'Teams' in x)
            if len(team_images) >= 2:
                # First image is team1, last is team2
                team1_name = team_images[0].get('alt', '')
                team2_name = team_images[-1].get('alt', '')
            
            # Fallback: extract from slug
            if not team1_name or not team2_name:
                slug_match = re.match(r'([^-]+(?:-[^-]+)*)-vs-([^-]+(?:-[^-]+)*)', slug)
                if slug_match:
                    team1_name = slug_match.group(1).replace('-', ' ').title()
                    team2_name = slug_match.group(2).replace('-', ' ').title()
            
            # Get full text content for parsing additional info
            text = link_element.get_text(' ', strip=True)
            
            # Extract series name from text
            # Format usually includes series name after match type
            series_name = ''
            format_type = 'T20'  # Default
            
            # Look for format indicators
            text_lower = text.lower()
            if 'odi' in text_lower:
                format_type = 'ODI'
            elif 't20' in text_lower or 't20i' in text_lower:
                format_type = 'T20'
            
            # Try to extract series name from slug
            # Example: bhutan-womens-t20i-tri-series-2026
            series_match = re.search(r'(?:match|final|qualifier|eliminator|playoff)[-\s]+(.+?)(?:-\d{4})?$', slug.replace('-', ' '), re.I)
            if series_match:
                series_name = series_match.group(1).strip().title()
            else:
                # Fallback: use part of slug after team names
                slug_parts = slug.split('-vs-')
                if len(slug_parts) > 1:
                    after_teams = slug_parts[1]
                    # Remove team2 name and match type
                    series_name = re.sub(r'^[a-z-]+-(?:\d+(?:st|nd|rd|th)-)?match-', '', after_teams, flags=re.I)
                    series_name = series_name.replace('-', ' ').title()
            
            # Detect status from text
            status = 'upcoming'
            if 'live' in text_lower:
                status = 'live'
            elif 'won' in text_lower or 'lost' in text_lower:
                status = 'completed'
            
            # Extract time if shown (e.g., "3:30 PM")
            time_match = re.search(r'(\d{1,2}:\d{2}\s*(?:AM|PM))', text, re.I)
            start_time = time_match.group(1) if time_match else None
            
            # Detect gender
            gender = self._detect_gender(text + team1_name + team2_name + slug)
            
            # Build match URL (always use /info page for details)
            match_url = urljoin(self.BASE_URL, href)
            if not match_url.endswith('/info'):
                match_url = re.sub(r'/(?:live|scorecard)$', '/info', match_url)
                if not match_url.endswith('/info'):
                    match_url = match_url.rstrip('/') + '/info'
            
            # Build title
            title = f"{team1_name} vs {team2_name}" if team1_name and team2_name else slug.replace('-', ' ').title()
            
            return CREXMatch(
                crex_id=match_id,
                series_id=series_id,
                slug=slug,
                title=title,
                team1_name=team1_name,
                team2_name=team2_name,
                team1_id=team1_id,
                team2_id=team2_id,
                match_type=match_type,
                series_name=series_name,
                format_type=format_type,
                status=status,
                start_time=start_time,
                match_url=match_url,
                gender=gender
            )
        
        except Exception as e:
            logger.debug(f"Failed to parse schedule entry: {e}")
            return None

    # =========================================================================
    # Match Details Parsing
    # =========================================================================
    
    def get_match_details(self, match_url: str) -> Optional[CREXMatch]:
        """
        Get full match details including venue and squads from match info page.
        
        Args:
            match_url: URL to match info page
            
        Returns:
            CREXMatch with full details or None
        """
        # Ensure we're fetching the /info page
        if not match_url.endswith('/info'):
            match_url = re.sub(r'/(?:live|scorecard)$', '/info', match_url)
            if not match_url.endswith('/info'):
                match_url = match_url.rstrip('/') + '/info'
        
        html = self._fetch(match_url)
        if not html:
            return None
        
        soup = BeautifulSoup(html, 'html.parser')
        
        try:
            return self._parse_match_info_page(soup, match_url)
        except Exception as e:
            logger.error(f"Error parsing match info from {match_url}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _parse_match_info_page(self, soup: BeautifulSoup, match_url: str) -> CREXMatch:
        """Parse match info page HTML."""
        
        # Extract IDs from URL
        # URL: https://crex.com/scoreboard/ZEE/2BL/4th-Match/2F/WP/slug/info
        # After removing protocol and domain: scoreboard/ZEE/2BL/4th-Match/2F/WP/slug/info
        url_path = match_url.replace('https://crex.com/', '').replace('http://crex.com/', '')
        parts = url_path.strip('/').split('/')
        
        # parts = ['scoreboard', 'ZEE', '2BL', '4th-Match', '2F', 'WP', 'slug', 'info']
        match_id = parts[1] if len(parts) > 1 else ''
        series_id = parts[2] if len(parts) > 2 else ''
        match_type_raw = parts[3] if len(parts) > 3 else ''
        team1_id = parts[4] if len(parts) > 4 else ''
        team2_id = parts[5] if len(parts) > 5 else ''
        slug = parts[6] if len(parts) > 6 else ''
        
        match_type = match_type_raw.replace('-', ' ').title()
        
        # Get page title/header for match info
        # Format: "BHU-W vs MAS-W, 4th T20, BHU-W Tri-Series 2026 info"
        title_element = soup.find('h1')
        title_text = title_element.get_text(strip=True) if title_element else ''
        
        # Parse title for teams and series
        title_match = re.match(r'(.+?)\s+vs\s+(.+?),\s*(.+?),\s*(.+?)\s*info', title_text, re.I)
        if title_match:
            team1_name = title_match.group(1).strip()
            team2_name = title_match.group(2).strip()
            match_type = title_match.group(3).strip()
            series_name = title_match.group(4).strip()
        else:
            # Fallback: try to extract from other elements
            team1_name = ''
            team2_name = ''
            series_name = ''
            
            # Look for team images
            team_images = soup.select('img[src*="Teams"]')
            if len(team_images) >= 2:
                team1_name = team_images[0].get('alt', '')
                team2_name = team_images[1].get('alt', '')
        
        # Try to find full team names in page content (for franchise teams)
        # CREX pages have full names like "Paarl Royals   Sunrisers Eastern Cape  Match info"
        page_text = soup.get_text()
        
        # Common franchise team name patterns (endings)
        team_patterns = [
            r'([\w\s]+(?:Royals|Cape|Sixers|Thunder|Stars|Warriors|Sunrisers|Strikers|Heat|Hurricanes|Renegades|Titans|Lions|Dolphins|Knights|Capitals|Kings|Challengers|Super Giants|Lucknow|Gujarat|Punjab|Rajasthan|Chennai|Mumbai|Kolkata|Delhi|Hyderabad|Bangalore))',
        ]
        
        # Look for pattern "Team1 vs Team2" or "Team1   Team2" with full names
        for pattern in team_patterns:
            vs_match = re.search(pattern + r'\s+(?:vs\.?|v)\s+' + pattern, page_text, re.I)
            if vs_match:
                team1_full = vs_match.group(1).strip()
                team2_full = vs_match.group(2).strip()
                # Only use if longer than current names (abbreviations are short)
                if len(team1_full) > len(team1_name):
                    logger.debug(f"Found full team name: {team1_name} -> {team1_full}")
                    team1_name = team1_full
                if len(team2_full) > len(team2_name):
                    logger.debug(f"Found full team name: {team2_name} -> {team2_full}")
                    team2_name = team2_full
                break
        
        # Also try extracting from the section right after the H1 title
        # Pattern: "...info Team1FullName   Team2FullName  Match info..."
        if len(team1_name) <= 5 or len(team2_name) <= 5:  # Still using abbreviations
            info_match = re.search(r'info\s+([\w\s]+?)\s{2,}([\w\s]+?)\s+Match info', page_text)
            if info_match:
                team1_candidate = info_match.group(1).strip()
                team2_candidate = info_match.group(2).strip()
                if len(team1_candidate) > 3 and len(team1_candidate) > len(team1_name):
                    team1_name = team1_candidate
                if len(team2_candidate) > 3 and len(team2_candidate) > len(team2_name):
                    team2_name = team2_candidate
                logger.debug(f"Extracted full names from header: {team1_name} vs {team2_name}")
        
        # Detect format
        format_type = 'T20'
        if 'odi' in match_type.lower():
            format_type = 'ODI'
        
        # Detect gender
        gender = self._detect_gender(title_text + team1_name + team2_name)
        
        # Parse date/time and venue
        # Look for text like "Saturday, 24 January, 10:00 AM" and venue below it
        date_str = None
        venue_name = None
        
        # Find the series info section - usually contains date and venue
        all_text = soup.get_text(' ', strip=True)
        
        # Date pattern: "Day, DD Month, HH:MM AM/PM" or "Day, DD Month"
        date_match = re.search(r'([A-Za-z]+day,\s*\d{1,2}\s+[A-Za-z]+(?:,\s*\d{1,2}:\d{2}\s*(?:AM|PM))?)', all_text, re.I)
        if date_match:
            date_str = date_match.group(1)
        
        # Venue is usually on its own line near the date
        # Look for venue patterns (Stadium, Ground, Oval, etc.)
        venue_patterns = [
            r'([A-Za-z\s]+(?:Stadium|Ground|Oval|Arena|Park|Centre|Center)[^,\n]*(?:,\s*[A-Za-z\s]+)?)',
            r'(The\s+[A-Za-z\s]+,\s*[A-Za-z]+)',
        ]
        
        for pattern in venue_patterns:
            venue_match = re.search(pattern, all_text)
            if venue_match:
                venue_name = venue_match.group(1).strip()
                break
        
        # Parse date/time
        start_date, start_time, date_time_gmt = None, None, None
        if date_str:
            start_date, start_time, date_time_gmt = self._parse_crex_datetime(date_str)
        
        # Create venue object
        venue = None
        if venue_name:
            city = self._extract_city_from_venue(venue_name)
            venue = CREXVenue(name=venue_name, city=city)
        
        # Parse venue stats
        venue_stats = self._parse_venue_stats(soup)
        
        # Parse squads from static HTML first
        team1 = self._parse_squad(soup, team1_id, team1_name, is_first_team=True)
        team2 = self._parse_squad(soup, team2_id, team2_name, is_first_team=False)
        
        # If one team is missing, try Playwright to get both squads
        if (not team1 or not team1.players) or (not team2 or not team2.players):
            logger.info("Static HTML missing squad data for one team, trying Playwright...")
            pw_team1, pw_team2 = self.fetch_squads_with_playwright(
                match_url, team1_name, team2_name, team1_id, team2_id
            )
            
            # Use Playwright results if we got them
            if pw_team1 and pw_team1.players:
                team1 = pw_team1
            if pw_team2 and pw_team2.players:
                team2 = pw_team2
        
        has_squads = bool((team1 and team1.players) or (team2 and team2.players))
        
        # Build title
        title = f"{team1_name} vs {team2_name}" if team1_name and team2_name else slug.replace('-', ' ').title()
        
        return CREXMatch(
            crex_id=match_id,
            series_id=series_id,
            slug=slug,
            title=title,
            team1_name=team1_name,
            team2_name=team2_name,
            team1_id=team1_id,
            team2_id=team2_id,
            match_type=match_type,
            series_name=series_name,
            format_type=format_type,
            status='upcoming',
            start_date=start_date,
            start_time=start_time,
            date_time_gmt=date_time_gmt,
            match_url=match_url,
            venue=venue,
            venue_stats=venue_stats,
            gender=gender,
            team1=team1,
            team2=team2,
            has_squads=has_squads
        )
    
    def _parse_venue_stats(self, soup: BeautifulSoup) -> Optional[CREXVenueStats]:
        """Parse venue statistics from match info page."""
        try:
            stats = CREXVenueStats()
            
            text = soup.get_text(' ', strip=True)
            
            # Matches played
            matches_match = re.search(r'(\d+)\s*Matches', text)
            if matches_match:
                stats.matches_played = int(matches_match.group(1))
            
            # Win percentages
            bat_first_match = re.search(r'Win Bat first\s*(\d+)%', text)
            if bat_first_match:
                stats.win_bat_first_pct = float(bat_first_match.group(1))
            
            bowl_first_match = re.search(r'Win Bowl first\s*(\d+)%', text)
            if bowl_first_match:
                stats.win_bowl_first_pct = float(bowl_first_match.group(1))
            
            # Average innings scores
            avg_1st_match = re.search(r'Avg 1st Inns\s*(\d+)', text)
            if avg_1st_match:
                stats.avg_first_innings = float(avg_1st_match.group(1))
            
            avg_2nd_match = re.search(r'Avg 2(?:nd|st) Inns\s*(\d+)', text)
            if avg_2nd_match:
                stats.avg_second_innings = float(avg_2nd_match.group(1))
            
            # Pace vs Spin
            pace_match = re.search(r'Pace\s*(\d+)\s*Wkt\s*(\d+)%', text)
            if pace_match:
                stats.pace_wickets = int(pace_match.group(1))
                stats.pace_pct = float(pace_match.group(2))
            
            spin_match = re.search(r'Spin\s*(\d+)\s*Wkt', text)
            if spin_match:
                stats.spin_wickets = int(spin_match.group(1))
                stats.spin_pct = 100 - stats.pace_pct if stats.pace_pct else 0
            
            return stats if stats.matches_played > 0 else None
        
        except Exception as e:
            logger.debug(f"Failed to parse venue stats: {e}")
            return None
    
    def _parse_squad(self, soup: BeautifulSoup, team_id: str, team_name: str, is_first_team: bool) -> Optional[CREXTeam]:
        """
        Parse squad from match info page.
        
        NOTE: CREX uses JavaScript tabs to show squad data. The initial HTML only
        contains ONE team's squad (usually the second team in the URL). The other
        team's data is loaded via JavaScript when the tab is clicked.
        
        This method now:
        1. Extracts all players from the HTML
        2. Identifies which team they belong to from the tab structure
        3. Returns the full squad for the matching team, None for the other
        """
        try:
            players = []
            
            # Find all player links
            player_links = soup.select('a[href*="/player/"]')
            
            # Find which team's tab is active/visible
            # Look for pattern: "Squads SEC PR" where first abbreviation after "Squads" is active
            page_text = soup.get_text(' ', strip=True)
            squad_match = re.search(r'Squads?\s+([A-Z]{2,5})\s+([A-Z]{2,5})', page_text)
            
            active_team_abbr = None
            if squad_match:
                # The first abbreviation after "Squads" is the active/visible tab
                active_team_abbr = squad_match.group(1)
                logger.debug(f"Active squad tab: {active_team_abbr}")
            
            # Parse all players
            for link in player_links:
                href = link.get('href', '')
                
                # Extract player ID from URL: /player/quinton-de-kock-92 -> "92"
                player_match = re.search(r'/player/[^/]+-([A-Za-z0-9]+)$', href)
                if not player_match:
                    continue
                
                crex_player_id = player_match.group(1)
                
                # Get player name from link text or title
                name = link.get('title', '') or link.get_text(strip=True)
                
                # Clean up name - remove role text that got concatenated
                name = re.sub(r'(Batter|Bowler|All Rounder|Wicket Keeper|✈️|\(C\)|\(WK\))+', '', name).strip()
                
                if not name:
                    # Extract from URL: quinton-de-kock -> Quinton De Kock
                    name_from_url = href.split('/')[-1].rsplit('-', 1)[0]
                    name = name_from_url.replace('-', ' ').title()
                
                # Get short name (shown on card)
                short_name = name.split()[0] if name else 'Unknown'
                
                # Look for role in parent element ONLY (not grandparent)
                # Grandparent contains all players' text and causes false matches
                parent = link.parent
                parent_text = parent.get_text(' ', strip=True) if parent else ''
                
                # Use only parent text for role detection
                context_text = parent_text
                
                # Detect role from the parent text (e.g., "Q d Kock (WK) Batter")
                role = 'Batter'  # Default
                if 'All Rounder' in context_text or 'All-Rounder' in context_text:
                    role = 'All Rounder'
                elif 'Bowler' in context_text:
                    role = 'Bowler'
                elif 'Batter' in context_text:
                    role = 'Batter'
                
                # Detect wicketkeeper from (WK) marker in THIS player's text
                is_wicketkeeper = '(WK)' in context_text
                if is_wicketkeeper:
                    role = 'WK'
                
                # Detect captain from (C) marker
                is_captain = '(C)' in context_text
                
                # Detect overseas (airplane emoji) in this player's text
                is_overseas = '✈️' in context_text or '✈' in context_text
                
                player = CREXPlayer(
                    crex_id=crex_player_id,
                    name=name,
                    short_name=short_name,
                    role=role,
                    is_captain=is_captain,
                    is_wicketkeeper=is_wicketkeeper,
                    is_overseas=is_overseas
                )
                
                # Avoid duplicates
                if not any(p.crex_id == player.crex_id for p in players):
                    players.append(player)
            
            logger.debug(f"Parsed {len(players)} players from HTML")
            
            # Determine if this team's data is in the HTML
            # The active tab abbreviation should match the team_id
            # team_id comes from the URL (e.g., "MY", "N0")
            
            # Check if this team's data is visible (active tab matches)
            is_visible_team = False
            if active_team_abbr:
                # Compare abbreviations (team_id might be e.g., "N0" for SEC)
                if team_id == active_team_abbr or team_id.upper() == active_team_abbr:
                    is_visible_team = True
                # Also check if team_name abbreviation matches
                team_abbr = ''.join([w[0] for w in team_name.split()]).upper()
                if team_abbr == active_team_abbr:
                    is_visible_team = True
            
            # If we can't determine from tabs, use team_name to match player names
            if not is_visible_team and players:
                # Try to identify team from known player names
                # Check if any player in our parsed list matches what we expect for this team
                # This is a heuristic - we check if we can match players to the team
                is_visible_team = not is_first_team  # Default: second team is usually visible
            
            if is_visible_team and players:
                logger.info(f"CREX HTML contains {len(players)} players for {team_name}")
                return CREXTeam(
                    crex_id=team_id,
                    name=team_name,
                    abbreviation=team_id,
                    players=players
                )
            else:
                logger.info(f"CREX HTML does not contain squad data for {team_name} (loaded via JavaScript)")
                return None
        
        except Exception as e:
            logger.debug(f"Failed to parse squad for {team_name}: {e}")
            return None

    def fetch_squads_with_playwright(self, match_url: str, team1_name: str, team2_name: str,
                                      team1_id: str, team2_id: str) -> Tuple[Optional[CREXTeam], Optional[CREXTeam]]:
        """
        Use Playwright to render the page and click both squad tabs to get all players.
        
        Args:
            match_url: URL to the match info page
            team1_name: Name of team 1
            team2_name: Name of team 2
            team1_id: CREX ID of team 1 (e.g., "MY" for Paarl Royals)
            team2_id: CREX ID of team 2 (e.g., "N0" for Sunrisers EC)
            
        Returns:
            Tuple of (team1, team2) CREXTeam objects with full squads
        """
        try:
            from playwright.sync_api import sync_playwright
            
            logger.info(f"Using Playwright to fetch squads for {team1_name} vs {team2_name}")
            
            teams_data = {}  # {tab_name: [players]}
            
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                
                # Navigate to match page (use domcontentloaded for faster loading)
                page.goto(match_url, wait_until='domcontentloaded', timeout=60000)
                # Wait a bit for JavaScript to render
                page.wait_for_timeout(2000)
                
                # Scroll to load any lazy content
                page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
                page.wait_for_timeout(500)
                
                # Wait for squad section to load
                try:
                    page.wait_for_selector('a[href*="/player/"]', timeout=10000)
                except:
                    logger.warning("No player links found on page")
                    browser.close()
                    return None, None
                
                # Find the squad tab buttons (they're <button> elements with team abbreviations)
                tab_buttons = page.query_selector_all('button')
                
                squad_tabs = []
                for tab in tab_buttons:
                    text = tab.inner_text().strip()
                    # Look for team abbreviations (SEC, PR, BHU-W, etc.) - short uppercase text
                    if 1 <= len(text) <= 10 and (text.isupper() or '-' in text):
                        # Make sure it's clickable (visible)
                        if tab.is_visible():
                            squad_tabs.append((tab, text))
                
                logger.debug(f"Found {len(squad_tabs)} squad tabs: {[t[1] for t in squad_tabs]}")
                
                # Click each tab and extract players
                for tab, tab_name in squad_tabs:
                    try:
                        tab.click()
                        page.wait_for_timeout(800)  # Wait for tab content to load
                        
                        # Extract players from current view
                        players = self._extract_players_from_page(page)
                        teams_data[tab_name] = players
                        logger.debug(f"Tab '{tab_name}': {len(players)} players")
                        
                    except Exception as e:
                        logger.debug(f"Failed to process tab {tab_name}: {e}")
                
                browser.close()
            
            # Match tabs to team names
            # team1_id (e.g., "MY") might match tab "PR" via abbreviation
            team1_players = []
            team2_players = []
            
            # Create abbreviation mappings
            team1_abbr = ''.join([w[0] for w in team1_name.split() if w]).upper()
            team2_abbr = ''.join([w[0] for w in team2_name.split() if w]).upper()
            
            for tab_name, players in teams_data.items():
                # Try to match tab to team
                tab_upper = tab_name.upper().replace('-', '')
                
                # Match by tab name, team ID, or abbreviation
                if (tab_upper == team1_id.upper() or 
                    tab_upper == team1_abbr or 
                    team1_name.upper().startswith(tab_upper)):
                    team1_players = players
                    logger.debug(f"Tab '{tab_name}' matched to team1: {team1_name}")
                elif (tab_upper == team2_id.upper() or 
                      tab_upper == team2_abbr or 
                      team2_name.upper().startswith(tab_upper)):
                    team2_players = players
                    logger.debug(f"Tab '{tab_name}' matched to team2: {team2_name}")
                else:
                    # Try partial matching
                    if any(part.upper() in tab_upper or tab_upper in part.upper() 
                           for part in team1_name.split()):
                        team1_players = players
                    elif any(part.upper() in tab_upper or tab_upper in part.upper() 
                             for part in team2_name.split()):
                        team2_players = players
            
            # If we only got one set of players, it might be for the first/default tab
            if not team1_players and not team2_players and teams_data:
                # Assign to the teams in order
                tab_names = list(teams_data.keys())
                if len(tab_names) >= 2:
                    team2_players = teams_data[tab_names[0]]  # First tab is usually visible by default
                    team1_players = teams_data[tab_names[1]]
                elif len(tab_names) == 1:
                    team2_players = teams_data[tab_names[0]]
            
            # Create team objects
            team1 = None
            team2 = None
            
            if team1_players:
                team1 = CREXTeam(
                    crex_id=team1_id,
                    name=team1_name,
                    abbreviation=team1_id,
                    players=team1_players
                )
                logger.info(f"Playwright: {team1_name} has {len(team1_players)} players")
            
            if team2_players:
                team2 = CREXTeam(
                    crex_id=team2_id,
                    name=team2_name,
                    abbreviation=team2_id,
                    players=team2_players
                )
                logger.info(f"Playwright: {team2_name} has {len(team2_players)} players")
            
            return team1, team2
            
        except ImportError:
            logger.warning("Playwright not installed, cannot fetch squads via JavaScript rendering")
            return None, None
        except Exception as e:
            logger.error(f"Playwright squad fetch failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _extract_players_from_page(self, page) -> List[CREXPlayer]:
        """Extract player data from current Playwright page state."""
        players = []
        
        # Find all player links
        player_links = page.query_selector_all('a[href*="/player/"]')
        
        for link in player_links:
            try:
                href = link.get_attribute('href') or ''
                
                # Extract player ID from URL
                player_match = re.search(r'/player/[^/]+-([A-Za-z0-9]+)$', href)
                if not player_match:
                    continue
                
                crex_player_id = player_match.group(1)
                
                # Get parent element text for role detection
                parent = link.evaluate('el => el.parentElement ? el.parentElement.innerText : ""')
                
                # Get player name (clean it)
                name_raw = link.inner_text().strip()
                name = re.sub(r'(Batter|Bowler|All Rounder|Wicket Keeper|✈️|\(C\)|\(WK\))+', '', name_raw).strip()
                
                if not name:
                    name_from_url = href.split('/')[-1].rsplit('-', 1)[0]
                    name = name_from_url.replace('-', ' ').title()
                
                short_name = name.split()[0] if name else 'Unknown'
                
                # Detect role
                context_text = parent or name_raw
                role = 'Batter'
                if 'All Rounder' in context_text or 'All-Rounder' in context_text:
                    role = 'All Rounder'
                elif 'Bowler' in context_text:
                    role = 'Bowler'
                elif 'Batter' in context_text:
                    role = 'Batter'
                
                is_wicketkeeper = '(WK)' in context_text
                if is_wicketkeeper:
                    role = 'WK'
                
                is_captain = '(C)' in context_text
                is_overseas = '✈️' in context_text or '✈' in context_text
                
                player = CREXPlayer(
                    crex_id=crex_player_id,
                    name=name,
                    short_name=short_name,
                    role=role,
                    is_captain=is_captain,
                    is_wicketkeeper=is_wicketkeeper,
                    is_overseas=is_overseas
                )
                
                # Avoid duplicates
                if not any(p.crex_id == player.crex_id for p in players):
                    players.append(player)
                    
            except Exception as e:
                logger.debug(f"Failed to extract player: {e}")
                continue
        
        return players

    # =========================================================================
    # Live Match / Toss Parsing
    # =========================================================================
    
    def get_live_match(self, match_url: str) -> Optional[CREXMatch]:
        """
        Get live match details including toss and playing XI.
        
        Args:
            match_url: URL to match page (will convert to /live)
            
        Returns:
            CREXMatch with toss/playing XI info or None
        """
        # Convert to live page URL
        live_url = re.sub(r'/(?:info|scorecard)$', '/live', match_url)
        if not live_url.endswith('/live'):
            live_url = live_url.rstrip('/') + '/live'
        
        html = self._fetch(live_url, timeout=20)
        if not html:
            return None
        
        soup = BeautifulSoup(html, 'html.parser')
        
        try:
            # First get basic match info
            match = self.get_match_details(match_url)
            if not match:
                return None
            
            # Parse toss info
            toss_winner, toss_decision = self._parse_toss(soup)
            if toss_winner:
                match.toss_winner = toss_winner
                match.toss_decision = toss_decision
                match.status = 'live'
            
            # Parse playing XI if available
            playing_xi = self._parse_playing_xi(soup)
            if playing_xi:
                match.playing_xi_available = True
            
            return match
        
        except Exception as e:
            logger.error(f"Error parsing live match from {live_url}: {e}")
            return None
    
    def _parse_toss(self, soup: BeautifulSoup) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse toss result from live page.
        
        Looks for text like "IND opt to bowl" or "NZ won the toss and elected to bat"
        
        Returns:
            (toss_winner, toss_decision) or (None, None)
        """
        try:
            text = soup.get_text(' ', strip=True)
            
            # Pattern 1: "TEAM opt to bat/bowl"
            opt_match = re.search(r'([A-Z]{2,4}(?:-W)?)\s+opt\s+to\s+(bat|bowl|field)', text, re.I)
            if opt_match:
                winner = opt_match.group(1)
                decision = opt_match.group(2).lower()
                if decision == 'field':
                    decision = 'bowl'
                return winner, decision
            
            # Pattern 2: "TEAM won the toss and elected to bat/bowl"
            toss_match = re.search(r'([A-Za-z\s]+)\s+won\s+the\s+toss\s+and\s+(?:elected|opted|chose)\s+to\s+(bat|bowl|field)', text, re.I)
            if toss_match:
                winner = toss_match.group(1).strip()
                decision = toss_match.group(2).lower()
                if decision == 'field':
                    decision = 'bowl'
                return winner, decision
            
            return None, None
        
        except Exception as e:
            logger.debug(f"Failed to parse toss: {e}")
            return None, None
    
    def _parse_playing_xi(self, soup: BeautifulSoup) -> Optional[Dict[str, List[CREXPlayer]]]:
        """
        Parse playing XI from live page.
        
        Returns:
            Dict with 'team1' and 'team2' lists of players, or None
        """
        try:
            # Look for "Playing XI" section
            text = soup.get_text(' ', strip=True)
            
            # Check if playing XI is mentioned
            if 'Playing XI' not in text and 'playing xi' not in text.lower():
                return None
            
            # For now, return indication that XI is available
            # Full parsing would require identifying the specific structure
            return {'available': True}
        
        except Exception as e:
            logger.debug(f"Failed to parse playing XI: {e}")
            return None

    # =========================================================================
    # Database Matching Methods
    # =========================================================================
    
    def match_venue_to_db(self, venue: CREXVenue, gender: str = 'male') -> Optional[Tuple[int, str]]:
        """
        Match CREX venue to database venue.
        
        Args:
            venue: CREX venue data
            gender: 'male' or 'female' for venue filtering
            
        Returns:
            (venue_id, venue_name) or None
        """
        if not venue:
            return None
        
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get all venues with enough matches, ordered by match count
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
        
        for row in db_venues:
            db_name = row['name']
            score = venue_similarity(venue.name, db_name)
            
            # Boost score if city matches
            if venue.city and row['city']:
                if venue.city.lower() in row['city'].lower():
                    score += 0.1
            
            if score > best_score:
                best_score = score
                best_match = (row['venue_id'], row['name'])
        
        if best_match and best_score >= 0.7:
            logger.info(f"Matched venue '{venue.name}' to '{best_match[1]}' (score: {best_score:.2f})")
            return best_match
        
        logger.warning(f"No database match for venue: {venue.name}")
        return None
    
    def match_team_to_db(self, team: CREXTeam, gender: str = 'male') -> Optional[Tuple[int, str]]:
        """
        Match CREX team to database team.
        
        Args:
            team: CREX team data
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
        
        team_names = [team.name, team.abbreviation]
        
        # NZ Super Smash team rebrandings (new name -> old database name)
        nz_rebrand_aliases = {
            'northern brave': 'northern districts',
            'southern brave': 'southern districts', 
            'central stags': 'central districts',
            'otago volts': 'otago',
            'auckland aces': 'auckland',
            'wellington firebirds': 'wellington',
            'canterbury kings': 'canterbury',
        }
        
        # Add aliases for rebranded teams
        for alias_from, alias_to in nz_rebrand_aliases.items():
            for name in list(team_names):
                name_norm = name.lower().replace(' women', '').replace('-w', '').strip()
                if alias_from in name_norm:
                    team_names.append(alias_to)
                    team_names.append(alias_to + ' women')
        
        # Direction words that must match exactly
        directions = {'northern', 'southern', 'eastern', 'western', 'central'}
        
        def get_direction(name):
            """Extract direction word from team name."""
            words = name.lower().split()
            for word in words:
                if word in directions:
                    return word
            return None
        
        best_match = None
        best_score = 0.0
        
        for row in db_teams:
            db_name = row['name']
            db_direction = get_direction(db_name)
            
            for crex_name in team_names:
                # Normalize for comparison
                crex_norm = crex_name.lower().replace(' women', '').replace('-w', '').strip()
                db_norm = db_name.lower().strip()
                
                crex_direction = get_direction(crex_name)
                
                # If both have direction words, they MUST match (Northern != Southern)
                if crex_direction and db_direction:
                    if crex_direction != db_direction:
                        continue  # Skip - wrong direction
                
                # Exact match
                if crex_norm == db_norm:
                    return (row['team_id'], row['name'])
                
                # Fuzzy match
                from difflib import SequenceMatcher
                score = SequenceMatcher(None, crex_norm, db_norm).ratio()
                
                if score > best_score:
                    best_score = score
                    best_match = (row['team_id'], row['name'])
        
        if best_match and best_score >= 0.8:  # Increased threshold from 0.7 to 0.8
            logger.info(f"Matched team '{team.name}' to '{best_match[1]}' (score: {best_score:.2f})")
            return best_match
        
        logger.warning(f"No database match for team: {team.name}")
        return None
    
    def match_players_to_db(self, team: CREXTeam, db_team_name: str, gender: str = 'male') -> List[CREXPlayer]:
        """
        Match CREX players to database players.
        
        Args:
            team: CREX team with players
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
            # Try matching with team filter first
            result = self._name_matcher.find_player(
                player.name,
                db_team_name
            )
            
            # If no match with team filter, try without (for franchise teams)
            # Players may have played for other teams in our database
            if not result and db_team_name:
                result = self._name_matcher.find_player(
                    player.name,
                    None  # Search all players
                )
                if result:
                    logger.debug(f"Matched '{player.name}' to '{result.db_name}' via global search (not found in {db_team_name})")
            
            if result:
                player.db_player_id = result.player_id
                logger.debug(f"Matched '{player.name}' to '{result.db_name}' (ID: {result.player_id})")
            else:
                logger.warning(f"Could not match player: {player.name} (team: {db_team_name})")
            
            matched_players.append(player)
        
        matched_count = sum(1 for p in matched_players if p.db_player_id)
        logger.info(f"Matched {matched_count}/{len(matched_players)} players for {db_team_name}")
        
        return matched_players


# ============================================================================
# Convenience Functions
# ============================================================================

_scraper_instance = None


def get_crex_scraper() -> CREXScraper:
    """Get or create singleton scraper instance."""
    global _scraper_instance
    if _scraper_instance is None:
        _scraper_instance = CREXScraper()
    return _scraper_instance


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    # Test the scraper
    logging.basicConfig(level=logging.INFO)
    
    scraper = CREXScraper(request_delay=0.5)
    
    print("=" * 70)
    print("CREX SCRAPER TEST")
    print("=" * 70)
    
    # Test 1: Get schedule
    print("\n--- T20 Schedule ---")
    matches = scraper.get_schedule(formats=['T20'])
    print(f"Found {len(matches)} T20 matches")
    
    for m in matches[:5]:
        print(f"\n  {m.title}")
        print(f"    Series: {m.series_name}")
        print(f"    Match: {m.match_type}")
        print(f"    Gender: {m.gender}")
        print(f"    Time: {m.start_time or 'TBD'}")
        print(f"    URL: {m.match_url[:60]}...")
    
    # Test 2: Get match details for first match with a URL
    if matches:
        for match in matches[:3]:
            if match.match_url:
                print(f"\n\n--- Match Details: {match.title} ---")
                details = scraper.get_match_details(match.match_url)
                
                if details:
                    print(f"\nMatch: {details.title}")
                    print(f"Series: {details.series_name}")
                    print(f"Type: {details.match_type}")
                    print(f"Date: {details.start_date} {details.start_time or ''}")
                    
                    if details.venue:
                        print(f"\nVenue: {details.venue.name}")
                        print(f"  City: {details.venue.city}")
                    
                    if details.venue_stats:
                        print(f"\nVenue Stats:")
                        print(f"  Matches: {details.venue_stats.matches_played}")
                        print(f"  Bat First Win: {details.venue_stats.win_bat_first_pct}%")
                        print(f"  Avg 1st Innings: {details.venue_stats.avg_first_innings}")
                    
                    for i, team in enumerate([details.team1, details.team2], 1):
                        if team:
                            print(f"\nTeam {i}: {team.name}")
                            print(f"  Players: {len(team.players)}")
                            for p in team.players[:5]:
                                captain_str = " (C)" if p.is_captain else ""
                                overseas_str = " ✈️" if p.is_overseas else ""
                                print(f"    - {p.name}{captain_str}{overseas_str}: {p.role}")
                
                break  # Just test one match
