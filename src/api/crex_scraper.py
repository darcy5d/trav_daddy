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
from src.data.venue_normalizer import venue_similarity, get_canonical_venue_name, normalize_venue_name as normalize_venue_global
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
    # Note: Be careful with abbreviations - "WI" is West Indies, not Women's Indicator
    # Use patterns that are clearly women's cricket, not ambiguous abbreviations
    WOMEN_KEYWORDS = ['women', 'wbbl', 'wpl', "women's", 'wpsl', 'wt20', 'female', 'womens']
    # Suffix patterns for team names (e.g., "Australia-W", "IND-W")
    WOMEN_SUFFIXES = ['-w', ' w$', 'w$']  # -w or ending with W after space
    
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
    
    def parse_url_team_codes(self, match_url: str) -> dict:
        """
        Parse team codes from URL for verification.
        
        URL format: .../scoreboard/{match_id}/{series_id}/{match_type}/{team1_id}/{team2_id}/{slug}/info
        Slug format: team1-vs-team2-match-type-series-name
        
        Returns:
            dict with 'team1_id', 'team2_id', 'team1_slug', 'team2_slug', 'slug'
        """
        url_path = match_url.replace('https://crex.com/', '').replace('http://crex.com/', '')
        parts = url_path.strip('/').split('/')
        
        result = {
            'team1_id': parts[4] if len(parts) > 4 else '',
            'team2_id': parts[5] if len(parts) > 5 else '',
            'slug': parts[6] if len(parts) > 6 else '',
            'team1_slug': '',
            'team2_slug': '',
        }
        
        # Parse team codes from slug (e.g., "ahw-vs-wbw-final-womens-super-smash-2025-26")
        slug = result['slug']
        vs_match = re.match(r'^([a-z0-9-]+?)-vs-([a-z0-9-]+?)-', slug, re.I)
        if vs_match:
            result['team1_slug'] = vs_match.group(1).upper()
            result['team2_slug'] = vs_match.group(2).upper()
        
        return result
    
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
        
        # Check for clear women's keywords
        if any(kw in text_lower for kw in self.WOMEN_KEYWORDS):
            return 'female'
        
        # Check for women's team suffixes like "AUS-W", "IND W", "Australia W"
        # But NOT "WI" (West Indies) or standalone "W" in middle of text
        import re
        # Pattern: word boundary followed by -w or space+w at end of a team name segment
        # Examples that SHOULD match: "aus-w", "ind-w", "australia w vs bhutan w"
        # Examples that should NOT match: "wi", "west indies", "w south africa"
        if re.search(r'\b[a-z]+-w\b', text_lower):  # e.g., "aus-w", "bhu-w"
            return 'female'
        if re.search(r'\b[a-z]+\s+w\s+vs\b', text_lower):  # e.g., "australia w vs"
            return 'female'
        if re.search(r'\bvs\s+[a-z]+\s+w\b', text_lower):  # e.g., "vs bhutan w"
            return 'female'
        
        return 'male'
    
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
        from bs4 import NavigableString
        
        html = self._fetch(self.SCHEDULE_URL)
        if not html:
            return []
        
        soup = BeautifulSoup(html, 'html.parser')
        matches = []
        
        # Date pattern for headers like "Sat, 24 Jan 2026"
        date_pattern = re.compile(r'^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*,?\s+\d{1,2}\s+[A-Za-z]+(?:\s+\d{4})?$', re.I)
        
        # Track current date as we traverse the DOM
        current_date = None
        match_hrefs_seen = set()
        
        # Traverse all elements in document order
        for element in soup.body.descendants if soup.body else []:
            if isinstance(element, NavigableString):
                text = str(element).strip()
                # Check if this is a date header
                if date_pattern.match(text):
                    current_date = text
            elif hasattr(element, 'name') and element.name == 'a':
                href = element.get('href', '')
                if '/scoreboard/' in href and href not in match_hrefs_seen:
                    match_hrefs_seen.add(href)
                    
                    # Parse match from the link with current date
                    match = self._parse_schedule_entry(element, href, current_date)
                    if match:
                        # Filter by format if specified
                        if formats and match.format_type not in formats:
                            continue
                        matches.append(match)
        
        logger.info(f"Found {len(matches)} matches from CREX schedule")
        return matches
    
    def _parse_schedule_entry(self, link_element, href: str, date_str: str = None) -> Optional[CREXMatch]:
        """
        Parse a match entry from the schedule page.
        
        Args:
            link_element: BeautifulSoup element for the match link
            href: Link URL
            date_str: Date string from section header (e.g., "Sat, 24 Jan 2026")
            
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
            # CREX schedule times are in GMT (confirmed via supersmash.co.nz)
            time_match = re.search(r'(\d{1,2}:\d{2}\s*(?:AM|PM))', text, re.I)
            start_time = time_match.group(1) if time_match else None
            
            # Parse date and build GMT timestamp
            # Date shown is venue local date, time is GMT
            # For venues ahead of GMT (Asia/Pacific), GMT date may be previous day
            start_date = None
            date_time_gmt = None
            if date_str:
                # Parse the venue local date
                parsed_date, _, _ = self._parse_crex_datetime(date_str)
                start_date = parsed_date
                
                if start_time and parsed_date:
                    # Time is already GMT - just need to figure out the GMT date
                    # For venues ahead of GMT (NZ, AU, Asia), if GMT time is PM,
                    # the GMT date is typically the previous day
                    try:
                        time_obj = datetime.strptime(start_time.strip(), "%I:%M %p")
                        gmt_hour = time_obj.hour
                        gmt_minute = time_obj.minute
                        
                        # Parse venue date
                        venue_date = datetime.strptime(parsed_date, "%Y-%m-%d")
                        
                        # If GMT time is in the evening (after 6 PM), the GMT date
                        # is likely the previous day for Asia-Pacific venues
                        # This handles NZ (UTC+13), AU (UTC+10/11), India (UTC+5:30), etc.
                        if gmt_hour >= 18:  # 6 PM or later in GMT
                            gmt_date = venue_date - timedelta(days=1)
                        else:
                            gmt_date = venue_date
                        
                        # Build the GMT timestamp
                        gmt_datetime = gmt_date.replace(hour=gmt_hour, minute=gmt_minute)
                        date_time_gmt = gmt_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")
                        
                        # Also set start_date to GMT date for consistency
                        start_date = gmt_date.strftime("%Y-%m-%d")
                    except ValueError as e:
                        logger.debug(f"Failed to parse time '{start_time}': {e}")
            
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
                start_date=start_date,
                start_time=start_time,
                date_time_gmt=date_time_gmt,
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
        # Stop at "Team Form" or other section boundaries
        venue_patterns = [
            # Match venue name + optional city, stop at Team Form or other sections
            r'([A-Za-z\s\-\']+(?:Stadium|Ground|Oval|Arena|Park|Centre|Center|International Cricket Ground)(?:,\s*[A-Za-z\s\-]+)?)\s*(?:Team Form|Head to Head|Match|$)',
            r'(The\s+[A-Za-z\s\-\']+,\s*[A-Za-z]+)\s*(?:Team Form|Head to Head|Match|$)',
        ]
        
        for pattern in venue_patterns:
            venue_match = re.search(pattern, all_text)
            if venue_match:
                venue_name = venue_match.group(1).strip()
                # Clean up any trailing whitespace or punctuation
                venue_name = re.sub(r'\s+$', '', venue_name)
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
            abbreviation_found = False
            
            if active_team_abbr:
                abbreviation_found = True
                active_abbr_clean = active_team_abbr.replace('-W', '').replace('-M', '').upper()
                
                # Compare abbreviations (team_id might be e.g., "N0" for SEC)
                if team_id.upper() == active_abbr_clean:
                    is_visible_team = True
                
                # Also check if team_name abbreviation matches (first letters)
                team_abbr = ''.join([w[0] for w in team_name.split()]).upper()
                if team_abbr == active_abbr_clean:
                    is_visible_team = True
                
                # Also try common country/team abbreviations
                team_name_lower = team_name.lower().replace(' women', '').replace(' men', '')
                common_abbrs = {
                    'south africa': 'SA', 'west indies': 'WI', 'new zealand': 'NZ',
                    'sri lanka': 'SL', 'bangladesh': 'BAN', 'afghanistan': 'AFG',
                    'australia': 'AUS', 'england': 'ENG', 'india': 'IND',
                    'pakistan': 'PAK', 'zimbabwe': 'ZIM', 'ireland': 'IRE',
                    'scotland': 'SCO', 'netherlands': 'NED', 'malaysia': 'MAS',
                    'bhutan': 'BHU', 'nepal': 'NEP', 'oman': 'OMA', 'uae': 'UAE',
                    'usa': 'USA', 'canada': 'CAN', 'namibia': 'NAM', 'kenya': 'KEN',
                }
                for name_pattern, abbr in common_abbrs.items():
                    if name_pattern in team_name_lower:
                        if abbr == active_abbr_clean:
                            is_visible_team = True
                            logger.debug(f"Matched '{team_name}' to active tab '{active_team_abbr}' via common abbr '{abbr}'")
                        break
            
            # Only use fallback if we couldn't detect ANY active tab abbreviation
            # If we found an abbreviation but it doesn't match this team, 
            # this team's data is NOT visible (the other team's data is visible)
            if not abbreviation_found and not is_visible_team and players:
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
        
        Uses the CREX page structure where:
        - Tab buttons have class 'playingxi-button' with team abbreviations (e.g., 'IRE', 'ITA')
        - The 'selected' class indicates the active tab
        - Player cards are in 'playingxi-card' sections
        
        This function extracts {tab_abbr: players} mapping and uses the abbreviation
        to match tabs to teams more accurately.
        
        Args:
            match_url: URL to the match info page
            team1_name: Name of team 1 from URL
            team2_name: Name of team 2 from URL
            team1_id: CREX ID of team 1
            team2_id: CREX ID of team 2
            
        Returns:
            Tuple of (team1, team2) CREXTeam objects with full squads.
        """
        try:
            from playwright.sync_api import sync_playwright
            
            logger.info(f"[PLAYWRIGHT] Fetching squads for {team1_name} vs {team2_name}")
            
            squads = {}  # {tab_abbr: players} mapping
            
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                
                # Navigate to match page (use domcontentloaded for faster loading)
                page.goto(match_url, wait_until='domcontentloaded', timeout=60000)
                # Wait for JavaScript to render
                page.wait_for_timeout(2000)
                
                # Scroll to load any lazy content
                page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
                page.wait_for_timeout(500)
                
                # Wait for squad section to load
                try:
                    page.wait_for_selector('a[href*="/player/"]', timeout=10000)
                except:
                    logger.warning("[PLAYWRIGHT] No player links found on page")
                    browser.close()
                    return None, None
                
                # Find the squad tab buttons using the specific CREX class
                # CREX uses: <button class="playingxi-button selected"> IRE </button>
                tab_buttons = page.query_selector_all('button.playingxi-button')
                
                if not tab_buttons:
                    # Fallback: try generic button search
                    tab_buttons = page.query_selector_all('button')
                    tab_buttons = [btn for btn in tab_buttons if btn.is_visible() and 
                                   len(btn.inner_text().strip()) <= 10]
                
                squad_tabs = []
                for tab in tab_buttons:
                    text = tab.inner_text().strip()
                    if text and len(text) <= 10:
                        squad_tabs.append((tab, text))
                
                logger.info(f"[PLAYWRIGHT] Found {len(squad_tabs)} squad tabs: {[t[1] for t in squad_tabs]}")
                
                # Click each tab and extract players, keyed by tab abbreviation
                for tab, tab_abbr in squad_tabs:
                    try:
                        tab.click()
                        page.wait_for_timeout(800)  # Wait for tab content to load
                        
                        # Extract players from current view
                        players = self._extract_players_from_page(page)
                        if players:
                            squads[tab_abbr] = players
                            logger.info(f"[PLAYWRIGHT] Tab '{tab_abbr}': {len(players)} players extracted")
                        
                    except Exception as e:
                        logger.debug(f"[PLAYWRIGHT] Failed to process tab {tab_abbr}: {e}")
                
                browser.close()
            
            if not squads:
                logger.warning("[PLAYWRIGHT] No squads could be extracted from any tabs")
                return None, None
            
            # Now match tabs to teams using abbreviation matching
            # Build abbreviation-to-team mapping
            team1_abbrs = self._generate_team_abbreviations(team1_name, team1_id)
            team2_abbrs = self._generate_team_abbreviations(team2_name, team2_id)
            
            logger.info(f"[PLAYWRIGHT] Team1 '{team1_name}' abbrs: {team1_abbrs}")
            logger.info(f"[PLAYWRIGHT] Team2 '{team2_name}' abbrs: {team2_abbrs}")
            
            team1_players = None
            team2_players = None
            team1_tab = None
            team2_tab = None
            
            for tab_abbr, players in squads.items():
                tab_upper = tab_abbr.upper().replace('-', '').replace(' ', '')
                
                # Check if this tab matches team1
                if any(abbr == tab_upper for abbr in team1_abbrs):
                    team1_players = players
                    team1_tab = tab_abbr
                    logger.info(f"[PLAYWRIGHT] Tab '{tab_abbr}' matched to team1 '{team1_name}'")
                # Check if this tab matches team2
                elif any(abbr == tab_upper for abbr in team2_abbrs):
                    team2_players = players
                    team2_tab = tab_abbr
                    logger.info(f"[PLAYWRIGHT] Tab '{tab_abbr}' matched to team2 '{team2_name}'")
            
            # If matching failed, fall back to remaining unmatched tabs
            # IMPORTANT: Don't reuse a tab that was already matched to another team!
            matched_tabs = set()
            if team1_tab:
                matched_tabs.add(team1_tab)
            if team2_tab:
                matched_tabs.add(team2_tab)
            
            # Get unmatched tabs in order
            unmatched_tabs = [(tab, players) for tab, players in squads.items() if tab not in matched_tabs]
            
            if team1_players is None and unmatched_tabs:
                team1_tab, team1_players = unmatched_tabs.pop(0)
                logger.warning(f"[PLAYWRIGHT] No match for team1, using unmatched tab '{team1_tab}'")
            if team2_players is None and unmatched_tabs:
                team2_tab, team2_players = unmatched_tabs.pop(0)
                logger.warning(f"[PLAYWRIGHT] No match for team2, using unmatched tab '{team2_tab}'")
            
            # Create team objects
            team1 = None
            team2 = None
            
            if team1_players:
                team1 = CREXTeam(
                    crex_id=team1_id,
                    name=team1_name,
                    abbreviation=team1_tab or team1_id,
                    players=team1_players
                )
                logger.info(f"[PLAYWRIGHT] Team1 '{team1_name}' (tab: {team1_tab}): {len(team1_players)} players")
            
            if team2_players:
                team2 = CREXTeam(
                    crex_id=team2_id,
                    name=team2_name,
                    abbreviation=team2_tab or team2_id,
                    players=team2_players
                )
                logger.info(f"[PLAYWRIGHT] Team2 '{team2_name}' (tab: {team2_tab}): {len(team2_players)} players")
            
            logger.info(f"[PLAYWRIGHT] NOTE: API layer will verify via player affiliations if needed.")
            
            return team1, team2
            
        except ImportError:
            logger.warning("[PLAYWRIGHT] Playwright not installed, cannot fetch squads via JavaScript rendering")
            return None, None
        except Exception as e:
            logger.error(f"[PLAYWRIGHT] Squad fetch failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _generate_team_abbreviations(self, team_name: str, team_id: str) -> List[str]:
        """
        Generate possible abbreviations for a team name.
        
        Args:
            team_name: Full team name (e.g., "Ireland", "Perth Scorchers")
            team_id: CREX team ID
            
        Returns:
            List of possible abbreviations in uppercase
        """
        abbrs = set()
        
        # Add the team ID
        abbrs.add(team_id.upper().replace('-', ''))
        
        # Standard country/team abbreviations
        KNOWN_ABBRS = {
            'ireland': ['IRE', 'IRL'],
            'italy': ['ITA', 'ITALY'],
            'australia': ['AUS'],
            'india': ['IND'],
            'england': ['ENG'],
            'pakistan': ['PAK'],
            'new zealand': ['NZ', 'NZL'],
            'south africa': ['SA', 'RSA', 'SAF'],
            'west indies': ['WI', 'WIN'],
            'sri lanka': ['SL', 'SRI'],
            'bangladesh': ['BAN', 'BD'],
            'afghanistan': ['AFG'],
            'zimbabwe': ['ZIM'],
            'scotland': ['SCO'],
            'netherlands': ['NED', 'NL'],
            'nepal': ['NEP'],
            'oman': ['OMA', 'OMN'],
            'uae': ['UAE'],
            'usa': ['USA'],
            'canada': ['CAN'],
            'hong kong': ['HK', 'HKG'],
            'malaysia': ['MAS', 'MAL', 'MYS'],
            'bhutan': ['BHU'],
            'perth scorchers': ['PRS', 'PS', 'PERTH'],
            'sydney sixers': ['SYS', 'SS', 'SIXERS'],
            'sydney thunder': ['SYT', 'ST', 'THUNDER'],
            'melbourne stars': ['MLS', 'STARS'],
            'melbourne renegades': ['MLR', 'RENEGADES'],
            'brisbane heat': ['BRH', 'HEAT'],
            'hobart hurricanes': ['HOB', 'HURRICANES'],
            'adelaide strikers': ['ADS', 'STRIKERS'],
            'northern brave': ['NB', 'NBW', 'BRAVE'],
            'canterbury': ['CTB', 'CMW', 'CANTERBURY'],  # Note: NOT 'CAN' - that's Canada
            'auckland': ['AKL', 'AHW', 'AUCKLAND'],  # AHW = Auckland Hearts Women
            'wellington': ['WEL', 'WBW', 'WELLINGTON'],  # WBW = Wellington Blaze Women
            'otago': ['OTA', 'OSW', 'OTAGO'],  # OSW = Otago Sparks Women
            'central stags': ['CS', 'STAGS'],
            'central districts': ['CD', 'CHW', 'CENTRAL'],  # CHW = Central Hinds Women
            # NZ Women's Super Smash franchise names
            'auckland hearts': ['AHW', 'AH', 'HEARTS'],
            'northern brave women': ['NBW', 'NB', 'BRAVE'],
            'wellington blaze': ['WBW', 'WB', 'BLAZE'],
            'canterbury magicians': ['CMW', 'CM', 'MAGICIANS'],
            'central hinds': ['CHW', 'CH', 'HINDS'],
            'otago sparks': ['OSW', 'OS', 'SPARKS'],
        }
        
        # Check known abbreviations
        name_lower = team_name.lower().replace(' women', '').replace('-w', '').strip()
        if name_lower in KNOWN_ABBRS:
            abbrs.update(KNOWN_ABBRS[name_lower])
        
        # Also check partial matches for franchise teams (e.g., "Auckland Women" should match "auckland")
        # BUT: Skip partial matching for short names (<=4 chars) to avoid false positives
        # e.g., "CAN" is a substring of both "canada" and "canterbury" - we don't want to match both!
        if len(name_lower) > 4:
            for known_name, known_abbrs in KNOWN_ABBRS.items():
                # Only match if the known_name is contained in our team name (not vice versa)
                # This allows "Auckland Hearts" to match "auckland", but prevents
                # "can" from matching "canterbury"
                if known_name in name_lower:
                    abbrs.update(known_abbrs)
        
        # Reserved country abbreviations - never auto-generate these for non-country teams
        # These are used by international teams and should not collide with domestic teams
        RESERVED_COUNTRY_ABBRS = {
            'AUS', 'IND', 'ENG', 'PAK', 'NZ', 'SA', 'WI', 'SL', 'BAN', 'AFG',
            'ZIM', 'IRE', 'SCO', 'NED', 'UAE', 'NAM', 'OMA', 'USA', 'CAN', 'KEN',
            'NEP', 'PNG', 'HK', 'MAS', 'SIN', 'BHU', 'MYA', 'JPN', 'KOR', 'THA',
            'UGA', 'TAN', 'RWA', 'NIG', 'BOT', 'GHA', 'GER', 'ITA', 'FRA', 'BER',
        }
        
        # Check if this team name is a country (if so, it can use reserved abbrs)
        is_country_team = name_lower in KNOWN_ABBRS and any(
            abbr in RESERVED_COUNTRY_ABBRS for abbr in KNOWN_ABBRS.get(name_lower, [])
        )
        
        # Generate from first letters of words
        words = team_name.replace('-', ' ').split()
        if words:
            # First 3 letters of first word
            auto_abbr_3 = words[0][:3].upper()
            if is_country_team or auto_abbr_3 not in RESERVED_COUNTRY_ABBRS:
                abbrs.add(auto_abbr_3)
            
            # First letter of each word
            abbrs.add(''.join(w[0] for w in words if w).upper())
            
            # First 2-3 letters
            if len(words[0]) >= 2:
                auto_abbr_2 = words[0][:2].upper()
                if is_country_team or auto_abbr_2 not in RESERVED_COUNTRY_ABBRS:
                    abbrs.add(auto_abbr_2)
        
        return list(abbrs)
    
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
    
    def get_player_team_affiliation(self, player_db_id: int, format_type: str = 'T20', 
                                     gender: str = 'male') -> Optional[Dict]:
        """
        Query which team a player most recently played for in the database.
        
        This is used to verify team assignments from CREX scraping. After matching
        players to the database, we can determine the actual team identity by
        checking which team the majority of players are affiliated with.
        
        Args:
            player_db_id: Database player ID
            format_type: 'T20' or 'ODI'
            gender: 'male' or 'female'
            
        Returns:
            Dict with team_id, team_name, match_count or None if not found
        """
        if not player_db_id:
            return None
        
        conn = get_connection()
        cursor = conn.cursor()
        
        try:
            # Query match_players + matches to find most recent team_id for this player
            # We look at the most recent matches to determine current affiliation
            cursor.execute("""
                SELECT pms.team_id, t.name as team_name, COUNT(*) as match_count,
                       MAX(m.date) as last_match_date
                FROM player_match_stats pms
                JOIN matches m ON m.match_id = pms.match_id
                JOIN teams t ON t.team_id = pms.team_id
                WHERE pms.player_id = ? 
                  AND m.match_type = ? 
                  AND m.gender = ?
                GROUP BY pms.team_id
                ORDER BY MAX(m.date) DESC, match_count DESC
                LIMIT 1
            """, (player_db_id, format_type, gender))
            
            result = cursor.fetchone()
            
            if result:
                return {
                    'team_id': result['team_id'],
                    'team_name': result['team_name'],
                    'match_count': result['match_count'],
                    'last_match_date': result['last_match_date']
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting player team affiliation for player_id={player_db_id}: {e}")
            return None
        finally:
            conn.close()
    
    def get_squad_team_affiliations(self, players: List[CREXPlayer], format_type: str = 'T20',
                                     gender: str = 'male') -> Dict[int, int]:
        """
        For a list of players, determine how many belong to each team.
        
        This is the core logic for affirmative team assignment. After scraping
        players and matching them to DB IDs, we count which database team each
        player belongs to. The team with the majority of players is the actual team.
        
        Args:
            players: List of CREXPlayer objects with db_player_id populated
            format_type: 'T20' or 'ODI'
            gender: 'male' or 'female'
            
        Returns:
            Dict mapping team_id -> count of players affiliated with that team
        """
        affiliations = {}  # team_id -> count
        
        for player in players:
            if not player.db_player_id:
                continue
                
            affiliation = self.get_player_team_affiliation(
                player.db_player_id, format_type, gender
            )
            
            if affiliation:
                team_id = affiliation['team_id']
                affiliations[team_id] = affiliations.get(team_id, 0) + 1
        
        return affiliations
    
    def determine_team_from_players(self, players: List[CREXPlayer], format_type: str = 'T20',
                                     gender: str = 'male') -> Optional[Dict]:
        """
        Determine which team a squad belongs to based on player affiliations.
        
        This is the authoritative team assignment function. It looks at all matched
        players and returns the team that the majority of them belong to.
        
        Args:
            players: List of CREXPlayer objects with db_player_id populated
            format_type: 'T20' or 'ODI'
            gender: 'male' or 'female'
            
        Returns:
            Dict with team_id, team_name, player_count, total_matched, confidence
            or None if no affiliation could be determined
        """
        if not players:
            return None
        
        affiliations = self.get_squad_team_affiliations(players, format_type, gender)
        
        if not affiliations:
            return None
        
        # Find the team with the most players
        best_team_id = max(affiliations, key=affiliations.get)
        best_count = affiliations[best_team_id]
        total_matched = sum(affiliations.values())
        
        # Get team name
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM teams WHERE team_id = ?", (best_team_id,))
        result = cursor.fetchone()
        conn.close()
        
        team_name = result['name'] if result else f"Team {best_team_id}"
        
        # Calculate confidence (proportion of matched players from this team)
        confidence = best_count / total_matched if total_matched > 0 else 0
        
        return {
            'team_id': best_team_id,
            'team_name': team_name,
            'player_count': best_count,
            'total_matched': total_matched,
            'confidence': confidence,
            'all_affiliations': affiliations  # For debugging mixed squads
        }
    
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
        
        # Clean venue name - strip common sponsor prefixes and suffixes
        def local_normalize_venue_name(name: str) -> str:
            if not name:
                return ""
            # Strip common sponsor/naming prefixes (2-3 letter codes often sponsors)
            # E.g., "AM Gelephu" -> "Gelephu", "JW Marriott Stadium" -> "Marriott Stadium"
            cleaned = re.sub(r'^[A-Z]{1,3}\s+', '', name.strip())
            # Also strip "The " prefix
            cleaned = re.sub(r'^The\s+', '', cleaned, flags=re.I)
            # Remove trailing notes like "Team Form" that CREX sometimes includes
            cleaned = re.sub(r'\s+Team Form$', '', cleaned, flags=re.I)
            return cleaned
        
        venue_name_clean = local_normalize_venue_name(venue.name)
        
        # Try canonical alias lookup first (e.g., "Optus Stadium" -> "Perth Stadium")
        canonical_name = get_canonical_venue_name(venue.name)
        canonical_clean = get_canonical_venue_name(venue_name_clean)
        
        # Create list of names to try matching
        names_to_try = [venue.name]
        if venue_name_clean != venue.name:
            names_to_try.append(venue_name_clean)
        if canonical_name != venue.name:
            names_to_try.append(canonical_name)
        if canonical_clean != venue_name_clean and canonical_clean not in names_to_try:
            names_to_try.append(canonical_clean)
        
        best_match = None
        best_score = 0.0
        
        for try_name in names_to_try:
            for row in db_venues:
                db_name = row['name']
                db_name_clean = local_normalize_venue_name(db_name)
                
                # Try both original and cleaned names
                score1 = venue_similarity(try_name, db_name)
                score2 = venue_similarity(try_name, db_name_clean)
                # Also try matching against the database name's canonical form
                db_canonical = get_canonical_venue_name(db_name)
                score3 = venue_similarity(try_name, db_canonical) if db_canonical != db_name else 0
                score = max(score1, score2, score3)
                
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
        # Includes both men's and women's franchise names
        nz_rebrand_aliases = {
            # Men's
            'northern brave': 'northern districts',
            'southern brave': 'southern districts', 
            'central stags': 'central districts',
            'otago volts': 'otago',
            'auckland aces': 'auckland',
            'wellington firebirds': 'wellington',
            'canterbury kings': 'canterbury',
            # Women's Super Smash franchise names
            'auckland hearts': 'auckland',
            'auckland hearts women': 'auckland',
            'wellington blaze': 'wellington',
            'wellington blaze women': 'wellington',
            'canterbury magicians': 'canterbury',
            'canterbury magicians women': 'canterbury',
            'central hinds': 'central districts',
            'central hinds women': 'central districts',
            'northern brave women': 'northern districts',
            'otago sparks': 'otago',
            'otago sparks women': 'otago',
        }
        
        # Common cricket team abbreviations (CREX abbreviation -> full name)
        team_abbreviations = {
            # International teams
            'aus': 'australia', 'ind': 'india', 'eng': 'england', 'pak': 'pakistan',
            'nz': 'new zealand', 'sa': 'south africa', 'wi': 'west indies',
            'sl': 'sri lanka', 'ban': 'bangladesh', 'afg': 'afghanistan',
            'zim': 'zimbabwe', 'ire': 'ireland', 'sco': 'scotland', 'ned': 'netherlands',
            'uae': 'united arab emirates', 'nam': 'namibia', 'oma': 'oman',
            'usa': 'united states of america', 'can': 'canada', 'ken': 'kenya',
            'nep': 'nepal', 'png': 'papua new guinea', 'hk': 'hong kong',
            'tha': 'thailand', 'mas': 'malaysia', 'sin': 'singapore', 'bhu': 'bhutan',
            'mya': 'myanmar', 'jpn': 'japan', 'kor': 'korea', 'idn': 'indonesia',
            'uga': 'uganda', 'tan': 'tanzania', 'rwa': 'rwanda', 'nig': 'nigeria',
            'bot': 'botswana', 'gha': 'ghana', 'cam': 'cameroon', 'moz': 'mozambique',
            'mwi': 'malawi', 'les': 'lesotho', 'esw': 'eswatini', 'swz': 'swaziland',
            'ger': 'germany', 'aut': 'austria', 'ita': 'italy', 'fra': 'france',
            'bel': 'belgium', 'esp': 'spain', 'por': 'portugal', 'cze': 'czech republic',
            'ber': 'bermuda', 'cay': 'cayman islands', 'bar': 'barbados',
            'jam': 'jamaica', 'tto': 'trinidad and tobago', 'guy': 'guyana',
            'lwi': 'leeward islands', 'wwi': 'windward islands',
            # NZ domestic (men's)
            # Note: 'can' is reserved for Canada (country), use 'ctb' for Canterbury
            'akl': 'auckland', 'wel': 'wellington', 'ctb': 'canterbury', 'ota': 'otago',
            'cd': 'central districts', 'nd': 'northern districts',
            # NZ Women's Super Smash abbreviations
            'ahw': 'auckland',  # Auckland Hearts Women
            'wbw': 'wellington',  # Wellington Blaze Women
            'cmw': 'canterbury',  # Canterbury Magicians Women
            'chw': 'central districts',  # Central Hinds Women
            'nbw': 'northern districts',  # Northern Brave Women
            'osw': 'otago',  # Otago Sparks Women
            'wfw': 'wellington',
            # Australia domestic
            'nsw': 'new south wales', 'vic': 'victoria', 'qld': 'queensland',
            'tas': 'tasmania', 'wa': 'western australia',
            'ss': 'sydney sixers', 'st': 'sydney thunder', 'ps': 'perth scorchers',
            'ms': 'melbourne stars', 'mr': 'melbourne renegades', 'bh': 'brisbane heat',
            'hs': 'hobart hurricanes', 'as': 'adelaide strikers',
        }
        
        # Add aliases for rebranded teams
        for alias_from, alias_to in nz_rebrand_aliases.items():
            for name in list(team_names):
                name_norm = name.lower().replace(' women', '').replace('-w', '').strip()
                if alias_from in name_norm:
                    team_names.append(alias_to)
                    team_names.append(alias_to + ' women')
        
        # Expand abbreviations to full team names
        for name in list(team_names):
            name_norm = name.lower().replace(' women', '').replace('-w', '').strip()
            if name_norm in team_abbreviations:
                full_name = team_abbreviations[name_norm]
                team_names.append(full_name)
                team_names.append(full_name + ' women')
                logger.debug(f"Expanded abbreviation '{name_norm}' to '{full_name}'")
        
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
