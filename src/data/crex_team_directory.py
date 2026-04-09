"""
CREX Team Directory Scraper

Scrapes the comprehensive team directory from https://crex.com/team to extract
all fkey → team variant mappings, preserving distinctions between main teams,
A-teams, Women's teams, U19 teams, etc.

Usage:
    scraper = CREXTeamDirectoryScraper()
    variants = scraper.scrape_all_teams()
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class TeamVariant:
    """A team variant from CREX directory."""
    fkey: str              # CREX team fkey (e.g., "1FO", "Q", "1I")
    full_name: str         # Complete name (e.g., "Hong Kong A", "Australia Women")
    short_name: str        # Short display (e.g., "HK-A", "AUS-W") 
    parent_team: str       # Parent team name (e.g., "Hong Kong", "Australia")
    team_type: str         # 'main', 'a-team', 'women', 'u19', 'u19-women', 'special'
    gender: str            # 'male', 'female' (affects database matching)
    url: str              # Source URL for reference
    
    def __post_init__(self):
        """Infer gender from team type and name if not explicitly set."""
        if not hasattr(self, '_gender_inferred'):
            if self.team_type in ('women', 'u19-women') or 'women' in self.full_name.lower():
                self.gender = 'female'
            else:
                self.gender = 'male'
            self._gender_inferred = True


class CREXTeamDirectoryScraper:
    """Scraper for CREX comprehensive team directory."""
    
    BASE_URL = "https://crex.com"
    TEAM_LIST_URL = f"{BASE_URL}/team"
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    
    def __init__(self, request_delay: float = 0.5):
        """Initialize scraper with rate limiting."""
        self.request_delay = request_delay
        self._last_request_time = 0
    
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request_time = time.time()
    
    def _fetch(self, url: str) -> Optional[str]:
        """Fetch URL with rate limiting and error handling."""
        self._rate_limit()
        try:
            response = requests.get(url, headers=self.HEADERS, timeout=15)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None
    
    def scrape_team_list_page(self) -> List[Dict[str, str]]:
        """
        Scrape the main team directory page for ALL team links systematically.
        
        Returns:
            List of dicts with 'name', 'url', 'fkey' extracted from team links
        """
        html = self._fetch(self.TEAM_LIST_URL)
        if not html:
            return []
        
        soup = BeautifulSoup(html, 'html.parser')
        teams = []
        
        # Strategy 1: Find explicit team links with pattern /team/{slug}-{fkey}
        team_links = soup.find_all('a', href=re.compile(r'/team/[^/]+-[A-Za-z0-9]+/?$'))
        
        for link in team_links:
            href = link.get('href', '').strip()
            if not href:
                continue
                
            # Extract fkey from URL: /team/hong-kong-a-1FO → fkey="1FO"
            fkey_match = re.search(r'-([A-Za-z0-9]+)/?$', href)
            if not fkey_match:
                continue
                
            fkey = fkey_match.group(1)
            
            # Get team name from link text
            name = link.get_text(strip=True)
            if not name:
                # Extract from URL slug as fallback
                slug_part = href.split('/')[-1].rsplit('-', 1)[0]
                name = slug_part.replace('-', ' ').title()
            
            if name and fkey:
                teams.append({
                    'name': name,
                    'url': urljoin(self.BASE_URL, href),
                    'fkey': fkey,
                    'slug': href.split('/')[-1] if '/' in href else href
                })
        
        # Strategy 2: Also look for any other patterns that might contain teams
        # Sometimes teams are in different HTML structures
        all_links = soup.find_all('a', href=re.compile(r'/team/'))
        
        for link in all_links:
            href = link.get('href', '').strip()
            if not href or any(existing['url'].endswith(href) for existing in teams):
                continue  # Skip if already found
            
            # Look for any /team/{something}-{alphanumeric} pattern
            if re.match(r'/team/[^/]+-[A-Za-z0-9]+/?$', href):
                fkey_match = re.search(r'-([A-Za-z0-9]+)/?$', href)
                if fkey_match:
                    fkey = fkey_match.group(1)
                    name = link.get_text(strip=True)
                    
                    if not name:
                        slug_part = href.split('/')[-1].rsplit('-', 1)[0] 
                        name = slug_part.replace('-', ' ').title()
                    
                    if name and fkey:
                        teams.append({
                            'name': name,
                            'url': urljoin(self.BASE_URL, href),
                            'fkey': fkey,
                            'slug': href.split('/')[-1] if '/' in href else href
                        })
        
        # Strategy 3: Extract ALL team links from page text/HTML and test them
        # Look for any mention of team URLs in the HTML source
        all_team_urls = re.findall(r'(?:href=["\']/team/[^"\']+|/team/[a-z0-9-]+-[A-Za-z0-9]+)', html)
        
        additional_teams = []
        for url_match in all_team_urls:
            # Clean the URL
            url = url_match.replace('href="', '').replace("href='", '').strip()
            if url.startswith('/team/'):
                fkey_match = re.search(r'-([A-Za-z0-9]+)/?$', url)
                if fkey_match and not any(existing['url'].endswith(url) for existing in teams):
                    fkey = fkey_match.group(1)
                    slug_part = url.split('/')[-1].rsplit('-', 1)[0]
                    name = slug_part.replace('-', ' ').title()
                    
                    additional_teams.append({
                        'name': name,
                        'url': urljoin(self.BASE_URL, url),
                        'fkey': fkey,  
                        'slug': url.split('/')[-1] if '/' in url else url
                    })
        
        teams.extend(additional_teams)
        
        # Remove duplicates by fkey
        seen_fkeys = set()
        unique_teams = []
        for team in teams:
            if team['fkey'] not in seen_fkeys:
                seen_fkeys.add(team['fkey'])
                unique_teams.append(team)
        
        logger.info(f"Found {len(unique_teams)} unique teams in directory (was {len(teams)} with duplicates)")
        return unique_teams
    
    def classify_team_variant(self, name: str, fkey: str) -> Dict[str, str]:
        """
        Classify team into variant type and determine parent team.
        
        Returns:
            Dict with 'team_type', 'parent_team', 'gender', 'short_name'
        """
        name_lower = name.lower().strip()
        
        # Initialize classification
        result = {
            'team_type': 'main',
            'parent_team': name,
            'gender': 'male',
            'short_name': fkey
        }
        
        # Women's teams
        if 'women' in name_lower or name.upper().endswith('-W'):
            result['gender'] = 'female'
            if 'u19' in name_lower or 'under 19' in name_lower:
                result['team_type'] = 'u19-women'
                # Extract parent: "Australia U19-Women" → "Australia"  
                parent = re.sub(r'\s+u19[\s-]*women?.*$', '', name, flags=re.I).strip()
            elif ' a ' in name_lower or name_lower.endswith(' a'):
                result['team_type'] = 'a-team'  
                # Extract parent: "Hong Kong A Women" → "Hong Kong"
                parent = re.sub(r'\s+a(?:\s+women)?.*$', '', name, flags=re.I).strip()
            else:
                result['team_type'] = 'women'
                # Extract parent: "Australia Women" → "Australia"
                parent = re.sub(r'\s+women.*$', '', name, flags=re.I).strip()
            result['parent_team'] = parent
            
        # U19 teams (male)
        elif 'u19' in name_lower or 'under 19' in name_lower:
            result['team_type'] = 'u19'
            # Extract parent: "Australia U19" → "Australia"
            parent = re.sub(r'\s+u19.*$|under\s+19.*$', '', name, flags=re.I).strip()
            result['parent_team'] = parent
            
        # A-teams (male)  
        elif ' a ' in name_lower or name_lower.endswith(' a'):
            result['team_type'] = 'a-team'
            # Extract parent: "Hong Kong A" → "Hong Kong", "Malaysia A" → "Malaysia"
            parent = re.sub(r'\s+a$', '', name, flags=re.I).strip()
            result['parent_team'] = parent
            
        # Special teams (PM's XI, etc.)
        elif any(special in name_lower for special in ['xi', 'eleven', "pm's", 'prime minister']):
            result['team_type'] = 'special'
            # Extract parent: "Australia PM-XI" → "Australia"  
            parent = re.sub(r'\s+pm[\s-]*xi.*$|xi.*$', '', name, flags=re.I).strip()
            result['parent_team'] = parent
        
        # Generate short name
        if result['team_type'] == 'women':
            result['short_name'] = f"{fkey}-W"
        elif result['team_type'] == 'u19':
            result['short_name'] = f"{fkey}-U19"
        elif result['team_type'] == 'u19-women':
            result['short_name'] = f"{fkey}-WU19"
        elif result['team_type'] == 'a-team':
            result['short_name'] = f"{fkey}-A"
        else:
            result['short_name'] = fkey
            
        return result
    
    def scrape_team_page(self, team_slug: str) -> Optional[Dict[str, str]]:
        """
        Scrape individual team page to get full name and metadata.
        
        Args:
            team_slug: Team slug like "hong-kong-a-1FO"
            
        Returns:
            Dict with 'name', 'fkey', 'url' or None if failed
        """
        url = f"{self.BASE_URL}/team/{team_slug}"
        html = self._fetch(url)
        if not html:
            return None
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract full name from H1 heading
        h1 = soup.find('h1')
        full_name = h1.get_text(strip=True) if h1 else ''
        
        # Extract fkey from URL
        fkey_match = re.search(r'-([A-Za-z0-9]+)/?$', team_slug)
        fkey = fkey_match.group(1) if fkey_match else ''
        
        if not full_name or not fkey:
            logger.debug(f"Could not extract name/fkey from {url}")
            return None
        
        return {
            'name': full_name,
            'fkey': fkey, 
            'url': url,
            'slug': team_slug
        }
    
    def scrape_known_team_variants(self) -> List[TeamVariant]:
        """
        Scrape known team variants by testing probable URLs.
        
        More reliable than depending on the main directory page which doesn't
        show all teams (notably missing A-teams).
        """
        logger.info("Scraping known CREX team variants...")
        
        # Build list of probable team variant URLs based on patterns observed
        known_patterns = self._generate_known_team_patterns()
        
        variants = []
        found_count = 0
        
        for slug in known_patterns:
            team_data = self.scrape_team_page(slug)
            if not team_data:
                continue
                
            found_count += 1
            name = team_data['name']
            fkey = team_data['fkey']
            url = team_data['url']
            
            # Classify team variant
            classification = self.classify_team_variant(name, fkey)
            
            variant = TeamVariant(
                fkey=fkey,
                full_name=name,
                short_name=classification['short_name'],
                parent_team=classification['parent_team'],
                team_type=classification['team_type'],
                gender=classification['gender'],
                url=url
            )
            
            variants.append(variant)
            logger.debug(f"Found {fkey}: {name} → {classification['team_type']} (parent: {classification['parent_team']})")
        
        # Also scrape main directory for any additional teams
        directory_teams = self.scrape_team_list_page()
        for team_data in directory_teams:
            if any(v.fkey == team_data['fkey'] for v in variants):
                continue  # Already have this one
            
            classification = self.classify_team_variant(team_data['name'], team_data['fkey'])
            variant = TeamVariant(
                fkey=team_data['fkey'],
                full_name=team_data['name'],
                short_name=classification['short_name'],
                parent_team=classification['parent_team'],
                team_type=classification['team_type'],
                gender=classification['gender'],
                url=team_data['url']
            )
            variants.append(variant)
        
        # Summary statistics
        type_counts = {}
        for v in variants:
            type_counts[v.team_type] = type_counts.get(v.team_type, 0) + 1
        
        logger.info(f"Found {found_count} teams via direct URLs, {len(variants)} total variants:")
        for team_type, count in sorted(type_counts.items()):
            logger.info(f"  {team_type}: {count}")
        
        return variants
    
    def _generate_known_team_patterns(self) -> List[str]:
        """Generate list of probable team variant URLs to test."""
        patterns = []
        
        # Major international teams with common variants
        countries = [
            ('australia', 'Q'), ('india', 'R'), ('england', 'S'), ('pakistan', 'P'),
            ('new-zealand', 'NZ'), ('south-africa', 'SA'), ('west-indies', 'WI'),
            ('sri-lanka', 'SL'), ('bangladesh', 'BAN'), ('afghanistan', 'AF'),
            ('zimbabwe', 'ZIM'), ('ireland', 'IRE'), ('scotland', 'SC'),
            ('netherlands', 'NED'), ('uae', 'UAE'), ('hong-kong', '13'),
            ('malaysia', 'MY'), ('singapore', 'SG'), ('thailand', 'TH'),
            ('oman', 'OM'), ('qatar', 'QA'), ('bahrain', 'BH'), ('kuwait', 'KW'),
            ('nepal', 'NP'), ('myanmar', 'MM'), ('bhutan', 'BH2')
        ]
        
        # Known specific team variants from user examples and investigation
        specific_variants = [
            'hong-kong-a-1FO',      # Hong Kong A  
            'malaysia-a-1FN',       # Malaysia A
            'australia-women-1I',   # Australia Women
            'australia-u19-8M',     # Australia U19
            'australia-u19women-N9', # Australia U19-Women  
            'australia-pmxi-10D',   # Australia PM-XI
            'australia-a-women-WK', # Australia A Women
            # Add more as we discover them
        ]
        
        patterns.extend(specific_variants)
        
        # For major countries, generate common variant patterns
        for country, base_fkey in countries[:10]:  # Limit to avoid too many requests
            # Generate probable variant URLs
            variants = [
                f'{country}-a-{base_fkey}A',     # A-team
                f'{country}-women-{base_fkey}W', # Women's team  
                f'{country}-u19-{base_fkey}U',   # U19 team
            ]
            patterns.extend(variants)
        
        return patterns
    
    def extract_teams_from_embedded_json(self, html: str) -> List[Dict[str, str]]:
        """
        Extract complete team catalog from script#app-root-state JSON.
        
        Uses same decoding infrastructure as schedule extraction to get all 1,737+ teams.
        """
        soup = BeautifulSoup(html, 'html.parser')
        script = soup.find('script', id='app-root-state')
        
        if not script or not script.string:
            logger.error("No app-root-state script found")
            return []
        
        # Reuse existing JSON decoding from scraper
        from src.api.crex_scraper import CREXScraper
        temp_scraper = CREXScraper()
        state = temp_scraper._decode_crex_app_root_state(script.string)
        
        if not state:
            logger.error("Could not decode app-root-state JSON")
            return []
        
        # Extract team mapping data
        team_key = "https://oc.crickapi.com/mapping/getHomeMapDatateamwise"
        team_data = state.get(team_key, {})
        
        if not isinstance(team_data, dict):
            logger.error(f"Team mapping data not found or invalid format")
            return []
        
        team_rows = team_data.get('t', [])
        if not isinstance(team_rows, list):
            logger.error("Team rows not found in 't' field")
            return []
        
        logger.info(f"Found {len(team_rows)} teams in embedded JSON")
        
        # Convert to our format
        teams = []
        for team_row in team_rows:
            if not isinstance(team_row, dict):
                continue
                
            fkey = team_row.get('f_key', '').strip()
            full_name = team_row.get('n', '').strip()
            short_name = team_row.get('sn', '').strip()
            
            if fkey and full_name:
                teams.append({
                    'fkey': fkey,
                    'name': full_name,
                    'short_name': short_name,
                    'url': f"{self.BASE_URL}/team/{full_name.lower().replace(' ', '-')}-{fkey}",
                    'source': 'embedded_json'
                })
        
        logger.info(f"Extracted {len(teams)} valid teams from embedded JSON")
        return teams
    
    def scrape_comprehensive_teams(self) -> List[TeamVariant]:
        """
        Scrape ALL teams using embedded JSON data (1,737+ teams).
        
        Replaces the limited DOM link approach with comprehensive JSON extraction.
        """
        logger.info("Scraping comprehensive CREX team directory from embedded JSON...")
        
        # Get the team directory page  
        html = self._fetch(self.TEAM_LIST_URL)
        if not html:
            logger.error("Failed to fetch team directory page")
            return []
        
        # Extract all teams from embedded JSON
        teams_data = self.extract_teams_from_embedded_json(html)
        
        if not teams_data:
            logger.error("No teams extracted from embedded JSON")
            return []
        
        # Convert to TeamVariant objects with classification
        variants = []
        seen_fkeys = set()
        
        for team_data in teams_data:
            fkey = team_data['fkey']
            name = team_data['name']
            
            # Skip duplicates
            if fkey in seen_fkeys:
                continue
            seen_fkeys.add(fkey)
            
            # Classify team variant
            classification = self.classify_team_variant(name, fkey)
            
            variant = TeamVariant(
                fkey=fkey,
                full_name=name,
                short_name=team_data.get('short_name', fkey),
                parent_team=classification['parent_team'],
                team_type=classification['team_type'],
                gender=classification['gender'],
                url=team_data['url']
            )
            
            variants.append(variant)
        
        # Summary statistics
        type_counts = {}
        for v in variants:
            type_counts[v.team_type] = type_counts.get(v.team_type, 0) + 1
        
        logger.info(f"Classified {len(variants)} comprehensive team variants:")
        for team_type, count in sorted(type_counts.items()):
            logger.info(f"  {team_type}: {count}")
        
        return variants
    
    def scrape_all_teams(self) -> List[TeamVariant]:
        """
        Scrape comprehensive team variants using embedded JSON extraction.
        
        Returns:
            List of TeamVariant objects with comprehensive coverage (1,737+ teams)
        """
        return self.scrape_comprehensive_teams()
    
    def export_variants_to_csv(self, variants: List[TeamVariant], filepath: str):
        """Export team variants to CSV for manual review."""
        import csv
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['fkey', 'full_name', 'short_name', 'parent_team', 'team_type', 'gender', 'url'])
            
            for v in variants:
                writer.writerow([
                    v.fkey, v.full_name, v.short_name, 
                    v.parent_team, v.team_type, v.gender, v.url
                ])
        
        logger.info(f"Exported {len(variants)} team variants to {filepath}")


# Utility functions for integration

def get_team_variant_by_fkey(variants: List[TeamVariant], fkey: str) -> Optional[TeamVariant]:
    """Find team variant by CREX fkey."""
    for variant in variants:
        if variant.fkey == fkey:
            return variant
    return None


def group_variants_by_parent(variants: List[TeamVariant]) -> Dict[str, List[TeamVariant]]:
    """Group team variants by parent team."""
    groups = {}
    for variant in variants:
        parent = variant.parent_team
        if parent not in groups:
            groups[parent] = []
        groups[parent].append(variant)
    return groups