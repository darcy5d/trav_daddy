"""
Venue Normalization Module

Provides functions to normalize venue names, detect duplicates,
and canonicalize venue data for consistent storage.
"""

import re
from typing import Tuple, Optional, List
from difflib import SequenceMatcher


def normalize_venue_name(name: str) -> str:
    """
    Normalize a venue name for comparison.
    
    Args:
        name: Original venue name
        
    Returns:
        Normalized name for matching
    """
    if not name:
        return ""
    
    # Convert to lowercase
    normalized = name.lower().strip()
    
    # Remove city suffix (e.g., ", Adelaide" or ", Perth")
    # This is the key step for matching "Adelaide Oval" to "Adelaide Oval, Adelaide"
    if ', ' in normalized:
        parts = normalized.split(', ')
        # Keep only the venue name part (first part) if it looks like a venue
        if len(parts) >= 1:
            normalized = parts[0]
    
    # Remove common sponsor prefixes (e.g., "PM Optus Stadium" -> "Optus Stadium")
    # These are typically 2-3 letter codes at the start
    sponsor_prefixes = [
        'pm ',      # Prime Minister's XI match prefix
        'kfc ',     # KFC BBL sponsorship
        'rebel ',   # Rebel WBBL sponsorship
        'weber ',   # Weber BBL sponsorship
        'alinta ',  # Alinta Energy
        'marsh ',   # Marsh One-Day Cup
        'jlt ',     # JLT Cup
        'kia ',     # Kia Oval
        'emirates ',  # Emirates Old Trafford
        'lg ',      # LG sponsorship
        'dlf ',     # DLF IPL
        'pepsi ',   # Pepsi IPL
        'vivo ',    # Vivo IPL
        'tata ',    # Tata IPL
    ]
    
    for prefix in sponsor_prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break  # Only remove one prefix
    
    # Remove common venue type suffixes
    suffixes_to_remove = [
        ' international cricket stadium',
        ' international cricket ground',
        ' cricket ground',
        ' cricket stadium',
        ' cricket club',
        ' cricket oval',
        ' international',
        ' stadium',
        ' ground',
        ' oval',
        ' park',
    ]
    
    for suffix in suffixes_to_remove:
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)]
            break  # Only remove one suffix
    
    # Remove punctuation and extra spaces
    normalized = re.sub(r'[^\w\s]', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized


def extract_canonical_name(name: str, city: Optional[str] = None) -> str:
    """
    Extract canonical venue name by removing redundant city suffix.
    
    Args:
        name: Original venue name
        city: City name (if known)
        
    Returns:
        Canonical name for display
    """
    if not name:
        return name
    
    canonical = name.strip()
    
    # Remove city suffix if present (e.g., "Adelaide Oval, Adelaide" -> "Adelaide Oval")
    if city:
        suffix = f", {city}"
        if canonical.endswith(suffix):
            canonical = canonical[:-len(suffix)]
    
    # Also try common variations
    city_suffixes = [
        f", {city}",
        f" {city}",
        f" - {city}",
    ]
    
    if city:
        for suffix in city_suffixes:
            if canonical.lower().endswith(suffix.lower()):
                canonical = canonical[:-len(suffix)]
                break
    
    return canonical.strip()


def venue_similarity(name1: str, name2: str) -> float:
    """
    Calculate similarity score between two venue names.
    
    Args:
        name1: First venue name
        name2: Second venue name
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    norm1 = normalize_venue_name(name1)
    norm2 = normalize_venue_name(name2)
    
    if not norm1 or not norm2:
        return 0.0
    
    # Exact match after normalization
    if norm1 == norm2:
        return 1.0
    
    # Use SequenceMatcher for fuzzy comparison
    return SequenceMatcher(None, norm1, norm2).ratio()


def find_best_match(
    venue_name: str,
    existing_venues: List[Tuple[int, str, str]],  # (venue_id, name, city)
    threshold: float = 0.85
) -> Optional[Tuple[int, str, str, float]]:
    """
    Find best matching venue from a list of existing venues.
    
    Args:
        venue_name: Name to match
        existing_venues: List of (venue_id, name, city) tuples
        threshold: Minimum similarity score to consider a match
        
    Returns:
        Best match as (venue_id, name, city, score) or None
    """
    if not venue_name or not existing_venues:
        return None
    
    best_match = None
    best_score = 0.0
    
    for venue_id, name, city in existing_venues:
        score = venue_similarity(venue_name, name)
        
        if score > best_score:
            best_score = score
            best_match = (venue_id, name, city, score)
    
    if best_match and best_match[3] >= threshold:
        return best_match
    
    return None


def extract_city_from_venue(venue_name: str) -> Optional[str]:
    """
    Try to extract city from venue name suffix.
    
    Args:
        venue_name: Full venue name
        
    Returns:
        Extracted city or None
    """
    if not venue_name:
        return None
    
    # Check for ", City" pattern
    if ', ' in venue_name:
        parts = venue_name.rsplit(', ', 1)
        if len(parts) == 2:
            potential_city = parts[1].strip()
            # Filter out obviously non-city suffixes
            non_city = ['international', 'stadium', 'ground', 'oval', 'cricket']
            if potential_city.lower() not in non_city:
                return potential_city
    
    return None


def extract_state_from_venue(venue_name: str, city: Optional[str] = None, country: Optional[str] = None) -> Optional[str]:
    """
    Infer state/province from venue string and location hints.

    This is intentionally conservative: return a value only when confidence is high.
    """
    if not venue_name:
        return None

    text = venue_name.lower().strip()
    city_lower = (city or '').lower().strip()
    country_lower = (country or '').lower().strip()

    # Explicit state names in venue text (mostly Australia where this is common).
    explicit_state_hints = {
        'new south wales': 'New South Wales',
        'western australia': 'Western Australia',
        'south australia': 'South Australia',
        'queensland': 'Queensland',
        'victoria': 'Victoria',
        'tasmania': 'Tasmania',
        'australian capital territory': 'Australian Capital Territory',
    }
    for hint, state in explicit_state_hints.items():
        if hint in text:
            return state

    # City-to-state mappings for common cricket locations.
    city_to_state = {
        'sydney': 'New South Wales',
        'newcastle': 'New South Wales',
        'perth': 'Western Australia',
        'adelaide': 'South Australia',
        'brisbane': 'Queensland',
        'gold coast': 'Queensland',
        'melbourne': 'Victoria',
        'geelong': 'Victoria',
        'hobart': 'Tasmania',
        'launceston': 'Tasmania',
        'canberra': 'Australian Capital Territory',
    }
    if city_lower in city_to_state:
        return city_to_state[city_lower]

    # If country is unknown, avoid over-guessing from generic city names.
    if country_lower and country_lower != 'australia':
        return None

    for city_hint, state in city_to_state.items():
        if city_hint in text:
            return state

    return None


# Known venue aliases - maps variations to canonical names
VENUE_ALIASES = {
    # Australia
    "w.a.c.a. ground": "WACA Ground",
    "waca ground": "WACA Ground",
    "waca": "WACA Ground",
    "w a c a": "WACA Ground",                               # normalized form of "W.A.C.A. Ground"
    "western australia cricket association": "WACA Ground", # normalized form of full DB name
    "optus stadium": "Perth Stadium",
    "optus": "Perth Stadium",
    "pm optus stadium": "Perth Stadium",
    "perth stadium": "Perth Stadium",
    "mcg": "Melbourne Cricket Ground",
    "scg": "Sydney Cricket Ground",
    "gabba": "Brisbane Cricket Ground",
    "woolloongabba": "Brisbane Cricket Ground",
    "brisbane cricket ground": "Brisbane Cricket Ground",
    "marvel stadium": "Marvel Stadium",
    "docklands stadium": "Marvel Stadium",
    "etihad stadium": "Marvel Stadium",
    "adelaide oval": "Adelaide Oval",
    "bellerive oval": "Bellerive Oval",
    "blundstone arena": "Bellerive Oval",
    "manuka oval": "Manuka Oval",
    "junction oval": "Junction Oval",
    "karen rolton oval": "Karen Rolton Oval",
    "north sydney oval": "North Sydney Oval",
    "cricket central": "Cricket Central",
    "sydney olympic": "Cricket Central",   # "Sydney Olympic Park" normalizes to "sydney olympic"
    # India
    "eden gardens": "Eden Gardens",
    "wankhede stadium": "Wankhede Stadium",
    "wankhede": "Wankhede Stadium",
    "chinnaswamy stadium": "M. Chinnaswamy Stadium",
    "m chinnaswamy stadium": "M. Chinnaswamy Stadium",
    "feroz shah kotla": "Arun Jaitley Stadium",
    "arun jaitley stadium": "Arun Jaitley Stadium",
    "narendra modi stadium": "Narendra Modi Stadium",
    "motera stadium": "Narendra Modi Stadium",
    # England
    "lords": "Lord's",
    "lord's": "Lord's",
    "the oval": "The Oval",
    "kia oval": "The Oval",
    "kennington oval": "The Oval",
    "trent bridge": "Trent Bridge",
    "old trafford": "Old Trafford",
    "emirates old trafford": "Old Trafford",
    "edgbaston": "Edgbaston",
    "headingley": "Headingley",
    "sophia gardens": "Sophia Gardens",
    "county ground bristol": "County Ground, Bristol",
    "county ground taunton": "County Ground, Taunton",
    # New Zealand
    "basin reserve": "Basin Reserve",
    "the basin reserve": "Basin Reserve",
    # South Africa
    "uplands college": "Uplands College, White River",
    "uplands college white river": "Uplands College, White River",
    "uplands college white river mpumalanga": "Uplands College, White River",
    # Pakistan
    "gaddafi stadium": "Gaddafi Stadium",
    "national stadium karachi": "National Stadium, Karachi",
    "national bank cricket arena": "National Stadium, Karachi",
    "national bank cricket arena karachi": "National Stadium, Karachi",
    "rawalpindi cricket stadium": "Rawalpindi Cricket Stadium",
    # Sri Lanka
    "r premadasa stadium": "R. Premadasa Stadium",
    "premadasa stadium": "R. Premadasa Stadium",
    "pallekele international cricket stadium": "Pallekele International Cricket Stadium",
    # Bangladesh
    "sher e bangla national stadium": "Shere Bangla National Stadium",
    "shere bangla national stadium": "Shere Bangla National Stadium",
    "zahur ahmed chowdhury stadium": "Zahur Ahmed Chowdhury Stadium",
    # UAE
    "dubai international stadium": "Dubai International Cricket Stadium",
    "abu dhabi cricket stadium": "Sheikh Zayed Stadium",
    "sheikh zayed stadium": "Sheikh Zayed Stadium",
    "sharjah cricket stadium": "Sharjah Cricket Stadium",
    # West Indies / Caribbean
    "kensington oval": "Kensington Oval",
    "queen's park oval": "Queen's Park Oval",
    "daren sammy national cricket stadium": "Daren Sammy National Cricket Stadium",
    "sir vivian richards stadium": "Sir Vivian Richards Stadium",
    "sabina park": "Sabina Park",
    "providence stadium": "Providence Stadium",
    "warner park": "Warner Park",
    # Zimbabwe
    "harare sports club": "Harare Sports Club",
    "queens sports club": "Queens Sports Club",
    # Afghanistan neutral home
    "kabul international cricket stadium": "Kabul International Cricket Stadium",
}


def get_canonical_venue_name(name: str) -> str:
    """
    Get canonical venue name, checking aliases.
    
    Args:
        name: Original venue name
        
    Returns:
        Canonical name if alias exists, otherwise original
    """
    if not name:
        return name
    
    normalized = normalize_venue_name(name)
    
    if normalized in VENUE_ALIASES:
        return VENUE_ALIASES[normalized]
    
    return name

