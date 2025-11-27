"""
Cricket Data API Exploration Script.

Tests various endpoints of the Cricket Data API (cricketdata.org)
to understand what data is available for WBBL fixtures and squads.

API Documentation: https://cricketdata.org/how-to-use-cricket-data-api.aspx
"""

import os
import sys
import json
import requests
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CRICKET_DATA_API_KEY, CRICKET_DATA_BASE_URL

def test_endpoint(endpoint: str, params: dict = None, description: str = "") -> dict:
    """Test an API endpoint and print results."""
    url = f"{CRICKET_DATA_BASE_URL}/{endpoint}"
    
    # Always include API key
    if params is None:
        params = {}
    params['apikey'] = CRICKET_DATA_API_KEY
    
    print(f"\n{'='*60}")
    print(f"Testing: {endpoint}")
    if description:
        print(f"Description: {description}")
    print(f"URL: {url}")
    print(f"Params: {params}")
    print('='*60)
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Print summary
        if 'status' in data:
            print(f"Status: {data['status']}")
        if 'info' in data:
            print(f"Info: {data['info']}")
        
        # Print data summary
        if 'data' in data:
            if isinstance(data['data'], list):
                print(f"Results: {len(data['data'])} items")
                if data['data']:
                    print(f"First item keys: {list(data['data'][0].keys())}")
                    print("\nFirst 3 items:")
                    for item in data['data'][:3]:
                        print(json.dumps(item, indent=2))
            else:
                print(f"Data: {json.dumps(data['data'], indent=2)[:1000]}")
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"ERROR: {e}")
        return {}


def find_wbbl_series(data: dict) -> list:
    """Find WBBL-related series from series list."""
    wbbl_series = []
    if 'data' in data and isinstance(data['data'], list):
        for series in data['data']:
            name = series.get('name', '').lower()
            if 'wbbl' in name or 'women' in name and 'big bash' in name:
                wbbl_series.append(series)
    return wbbl_series


def main():
    print("Cricket Data API Explorer")
    print(f"API Key: {CRICKET_DATA_API_KEY[:8]}...{CRICKET_DATA_API_KEY[-4:]}")
    print(f"Base URL: {CRICKET_DATA_BASE_URL}")
    
    # Test 1: Current Matches
    current = test_endpoint(
        "currentMatches",
        description="Get all current/upcoming matches"
    )
    
    # Filter for WBBL matches
    if current and 'data' in current:
        wbbl_matches = [
            m for m in current['data'] 
            if 'wbbl' in m.get('name', '').lower() 
            or 'wbbl' in m.get('series', '').lower()
            or "women's big bash" in m.get('name', '').lower()
        ]
        print(f"\n>>> Found {len(wbbl_matches)} WBBL matches in current matches")
        for m in wbbl_matches[:5]:
            print(f"  - {m.get('name')}: {m.get('status')} (ID: {m.get('id')})")
    
    # Test 2: Series Search
    series_data = test_endpoint(
        "series",
        params={"search": "WBBL"},
        description="Search for WBBL series"
    )
    
    # Test 3: Get all series (to find WBBL)
    all_series = test_endpoint(
        "series",
        description="Get all current series"
    )
    
    wbbl_series = find_wbbl_series(all_series)
    print(f"\n>>> Found {len(wbbl_series)} WBBL-related series:")
    for s in wbbl_series:
        print(f"  - {s.get('name')} (ID: {s.get('id')})")
    
    # Test 4: If we found a WBBL series, get its matches
    if wbbl_series:
        series_id = wbbl_series[0].get('id')
        print(f"\n>>> Fetching matches for series: {wbbl_series[0].get('name')}")
        
        series_info = test_endpoint(
            "series_info",
            params={"id": series_id},
            description=f"Get matches in series {series_id}"
        )
        
        # If we found matches, try to get squad for one
        if series_info and 'data' in series_info:
            matches = series_info['data'].get('matchList', [])
            if matches:
                # Find an upcoming match
                upcoming = [m for m in matches if m.get('status') in ['Match not started', 'upcoming', '']]
                if upcoming:
                    match_id = upcoming[0].get('id')
                    print(f"\n>>> Fetching squad for match: {upcoming[0].get('name')}")
                    
                    squad_data = test_endpoint(
                        "match_squad",
                        params={"id": match_id},
                        description=f"Get squad for match {match_id}"
                    )
    
    # Test 5: Match Info endpoint
    if current and 'data' in current and current['data']:
        sample_match = current['data'][0]
        match_id = sample_match.get('id')
        
        test_endpoint(
            "match_info",
            params={"id": match_id},
            description=f"Get detailed info for match {match_id}"
        )
    
    print("\n" + "="*60)
    print("API EXPLORATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

