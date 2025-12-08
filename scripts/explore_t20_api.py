"""
T20 API Exploration Script.

Tests Cricket Data API endpoints to understand:
1. What T20 series are currently active
2. Does /matches include future scheduled matches?
3. What does /currentMatches actually return?
4. Best strategy for getting "next 24 hours" matches

Run: python scripts/explore_t20_api.py
"""

import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from config import CRICKET_DATA_API_KEY, CRICKET_DATA_BASE_URL


def api_call(endpoint: str, params: dict = None) -> dict:
    """Make API call and return response."""
    url = f"{CRICKET_DATA_BASE_URL}/{endpoint}"
    if params is None:
        params = {}
    params['apikey'] = CRICKET_DATA_API_KEY
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"ERROR calling {endpoint}: {e}")
        return {}


def explore_series():
    """Explore the /series endpoint to understand available T20 series."""
    print("\n" + "="*70)
    print("1. EXPLORING /series ENDPOINT")
    print("="*70)
    
    data = api_call('series')
    
    if data.get('status') != 'success':
        print(f"API Error: {data.get('info', 'Unknown error')}")
        return []
    
    all_series = data.get('data', [])
    print(f"Total series returned: {len(all_series)}")
    
    # Filter for T20 series
    t20_series = [s for s in all_series if s.get('t20', 0) > 0]
    print(f"Series with T20 matches: {len(t20_series)}")
    
    # Categorize by gender
    women_series = []
    men_series = []
    
    women_keywords = ['women', 'wbbl', 'wpl', 'female', "women's", 'wpsl']
    
    for s in t20_series:
        name_lower = s.get('name', '').lower()
        if any(kw in name_lower for kw in women_keywords):
            women_series.append(s)
        else:
            men_series.append(s)
    
    print(f"\nWomen's T20 series: {len(women_series)}")
    print(f"Men's T20 series: {len(men_series)}")
    
    print("\n--- Women's T20 Series ---")
    for s in women_series[:10]:
        print(f"  {s.get('name')} | T20s: {s.get('t20')} | {s.get('startDate')} to {s.get('endDate')}")
    
    print("\n--- Men's T20 Series (top 10) ---")
    for s in men_series[:10]:
        print(f"  {s.get('name')} | T20s: {s.get('t20')} | {s.get('startDate')} to {s.get('endDate')}")
    
    return t20_series


def explore_matches():
    """Explore the /matches endpoint to see if it includes future matches."""
    print("\n" + "="*70)
    print("2. EXPLORING /matches ENDPOINT")
    print("="*70)
    
    data = api_call('matches')
    
    if data.get('status') != 'success':
        print(f"API Error: {data.get('info', 'Unknown error')}")
        return
    
    all_matches = data.get('data', [])
    print(f"Total matches returned: {len(all_matches)}")
    
    # Filter for T20 only
    t20_matches = [m for m in all_matches if m.get('matchType') == 't20']
    print(f"T20 matches: {len(t20_matches)}")
    
    # Analyze dates
    today = datetime.now().date()
    future_matches = []
    today_matches = []
    past_matches = []
    
    for m in t20_matches:
        date_str = m.get('date', '')
        try:
            match_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            if match_date > today:
                future_matches.append(m)
            elif match_date == today:
                today_matches.append(m)
            else:
                past_matches.append(m)
        except:
            pass
    
    print(f"\nT20 matches by date:")
    print(f"  Past: {len(past_matches)}")
    print(f"  Today: {len(today_matches)}")
    print(f"  Future: {len(future_matches)}")
    
    if future_matches:
        print(f"\n✅ /matches DOES include future matches!")
        print(f"\n--- Future T20 matches (next 5) ---")
        future_matches.sort(key=lambda m: m.get('date', ''))
        for m in future_matches[:5]:
            print(f"  {m.get('date')} | {m.get('name')} | Status: {m.get('status')}")
    else:
        print(f"\n❌ /matches does NOT include future matches")
    
    # Check status values
    statuses = defaultdict(int)
    for m in t20_matches[:100]:
        statuses[m.get('status', 'N/A')] += 1
    
    print(f"\n--- Match status values (from first 100 T20s) ---")
    for status, count in sorted(statuses.items(), key=lambda x: -x[1]):
        print(f"  {status}: {count}")
    
    return future_matches


def explore_current_matches():
    """Explore the /currentMatches endpoint."""
    print("\n" + "="*70)
    print("3. EXPLORING /currentMatches ENDPOINT")
    print("="*70)
    
    data = api_call('currentMatches')
    
    if data.get('status') != 'success':
        print(f"API Error: {data.get('info', 'Unknown error')}")
        return
    
    all_matches = data.get('data', [])
    print(f"Total 'current' matches: {len(all_matches)}")
    
    # Filter for T20
    t20_matches = [m for m in all_matches if m.get('matchType') == 't20']
    print(f"T20 'current' matches: {len(t20_matches)}")
    
    # Analyze what's included
    today = datetime.now().date()
    tomorrow = today + timedelta(days=1)
    
    today_count = 0
    tomorrow_count = 0
    future_count = 0
    
    print(f"\n--- All 'current' T20 matches ---")
    for m in t20_matches:
        date_str = m.get('date', '')
        status = m.get('status', 'N/A')
        name = m.get('name', 'N/A')
        
        try:
            match_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            if match_date == today:
                today_count += 1
                day_label = "TODAY"
            elif match_date == tomorrow:
                tomorrow_count += 1
                day_label = "TOMORROW"
            else:
                future_count += 1
                day_label = "FUTURE"
        except:
            day_label = "UNKNOWN"
        
        print(f"  [{day_label}] {date_str} | {name[:50]} | Status: {status[:40]}")
    
    print(f"\n--- Summary ---")
    print(f"  Today: {today_count}")
    print(f"  Tomorrow: {tomorrow_count}")
    print(f"  Future: {future_count}")
    
    if tomorrow_count > 0 or future_count > 0:
        print(f"\n✅ /currentMatches includes scheduled matches!")
    else:
        print(f"\n⚠️  /currentMatches appears to be live/today only")


def explore_series_info(series_id: str, series_name: str):
    """Explore /series_info for a specific series."""
    print("\n" + "="*70)
    print(f"4. EXPLORING /series_info for: {series_name}")
    print("="*70)
    
    data = api_call('series_info', {'id': series_id})
    
    if data.get('status') != 'success':
        print(f"API Error: {data.get('info', 'Unknown error')}")
        return
    
    series_data = data.get('data', {})
    matches = series_data.get('matchList', [])
    print(f"Total matches in series: {len(matches)}")
    
    # Analyze dates
    today = datetime.now().date()
    upcoming = []
    completed = []
    
    for m in matches:
        date_str = m.get('date', '')
        status = m.get('status', '')
        try:
            match_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            if match_date >= today and 'won' not in status.lower():
                upcoming.append(m)
            else:
                completed.append(m)
        except:
            pass
    
    print(f"Completed: {len(completed)}")
    print(f"Upcoming: {len(upcoming)}")
    
    if upcoming:
        print(f"\n--- Upcoming matches ---")
        upcoming.sort(key=lambda m: m.get('date', ''))
        for m in upcoming[:10]:
            print(f"  {m.get('date')} | {m.get('name')}")
            print(f"    Status: {m.get('status')}")
            print(f"    Has Squad: {m.get('hasSquad', False)}")


def main():
    print("="*70)
    print("T20 API EXPLORATION")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"API: {CRICKET_DATA_BASE_URL}")
    print("="*70)
    
    # 1. Explore series
    t20_series = explore_series()
    
    # 2. Explore matches endpoint
    future_matches = explore_matches()
    
    # 3. Explore current matches
    explore_current_matches()
    
    # 4. Pick a women's series to drill into (WBBL if available)
    women_keywords = ['women', 'wbbl', 'wpl', 'female']
    women_t20 = [s for s in t20_series 
                 if any(kw in s.get('name', '').lower() for kw in women_keywords)]
    
    if women_t20:
        # Try to find WBBL specifically
        wbbl = next((s for s in women_t20 if 'big bash' in s.get('name', '').lower()), None)
        if wbbl:
            explore_series_info(wbbl['id'], wbbl['name'])
        else:
            explore_series_info(women_t20[0]['id'], women_t20[0]['name'])
    
    # Summary and recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    if future_matches:
        print("""
✅ GOOD NEWS: /matches endpoint includes future matches!

RECOMMENDED STRATEGY:
1. Use /matches (one call) to get ALL matches
2. Filter client-side for:
   - matchType = 't20'
   - date within desired range (next 24h, next week, etc.)
   - gender (detect from team names)

This is MORE EFFICIENT than calling /series_info for each series!
""")
    else:
        print("""
⚠️  /matches endpoint appears historical-only.

RECOMMENDED STRATEGY:
1. Use /series to get active T20 series (one call, cache daily)
2. Use /series_info for selected series (on-demand)
3. Filter matches by date

Consider pre-fetching popular series (WBBL, BBL, IPL) on page load.
""")


if __name__ == "__main__":
    main()






