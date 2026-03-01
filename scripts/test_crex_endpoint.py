#!/usr/bin/env python3
"""
Test CREX data pipeline end-to-end.

This script:
1. Fetches upcoming matches from CREX
2. Gets match details including squads
3. Shows whether players have db_player_id set

Run from project root:
    python scripts/test_crex_endpoint.py
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.crex_scraper import CREXScraper, get_crex_scraper


def test_crex_pipeline():
    print("=" * 70)
    print("🔍 CREX DATA PIPELINE TEST")
    print("=" * 70)
    
    # Create scraper
    scraper = CREXScraper(request_delay=0.5)
    
    # Step 1: Get upcoming matches
    print("\n📅 Fetching upcoming T20 matches from CREX...")
    print("-" * 70)
    
    try:
        matches = scraper.get_schedule(formats=['T20'])
        print(f"Found {len(matches)} upcoming T20 matches")
        
        if not matches:
            print("❌ No matches found!")
            return
        
        # Show first few matches
        for i, m in enumerate(matches[:5]):
            print(f"  {i+1}. {m.title}")
            print(f"     Series: {m.series_name}")
            print(f"     Date: {m.start_date}")
            print(f"     URL: {m.match_url}")
            print()
        
    except Exception as e:
        print(f"❌ Error fetching matches: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Get full match details for first match
    if matches:
        print("\n📊 Fetching full details for first match...")
        print("-" * 70)
        
        first_match = matches[0]
        
        try:
            match_details = scraper.get_match_details(first_match.match_url)
            
            if match_details:
                print(f"Match: {match_details.title}")
                print(f"Gender: {match_details.gender}")
                print(f"Venue: {match_details.venue.name if match_details.venue else 'Unknown'}")
                
                # Check Team 1
                print(f"\n📋 Team 1: {match_details.team1.name if match_details.team1 else 'None'}")
                if match_details.team1:
                    players = match_details.team1.players
                    print(f"   Players from CREX: {len(players)}")
                    
                    if players:
                        # Match players to database
                        team_match = scraper.match_team_to_db(match_details.team1, match_details.gender)
                        team_name = team_match[1] if team_match else None
                        print(f"   DB team match: {team_name}")
                        
                        scraper.match_players_to_db(match_details.team1, team_name, match_details.gender)
                        
                        matched = sum(1 for p in players if p.db_player_id)
                        print(f"   Matched to DB: {matched}/{len(players)}")
                        
                        print("\n   Player details:")
                        for p in players[:8]:
                            status = "✅" if p.db_player_id else "❌"
                            print(f"     {status} {p.name:<25} ID={p.db_player_id} (crex={p.crex_id})")
                    else:
                        print("   ⚠️  NO PLAYERS scraped from CREX!")
                        print("   This is likely due to JavaScript rendering issues.")
                
                # Check Team 2
                print(f"\n📋 Team 2: {match_details.team2.name if match_details.team2 else 'None'}")
                if match_details.team2:
                    players = match_details.team2.players
                    print(f"   Players from CREX: {len(players)}")
                    
                    if players:
                        # Match players to database
                        team_match = scraper.match_team_to_db(match_details.team2, match_details.gender)
                        team_name = team_match[1] if team_match else None
                        print(f"   DB team match: {team_name}")
                        
                        scraper.match_players_to_db(match_details.team2, team_name, match_details.gender)
                        
                        matched = sum(1 for p in players if p.db_player_id)
                        print(f"   Matched to DB: {matched}/{len(players)}")
                        
                        print("\n   Player details:")
                        for p in players[:8]:
                            status = "✅" if p.db_player_id else "❌"
                            print(f"     {status} {p.name:<25} ID={p.db_player_id} (crex={p.crex_id})")
                    else:
                        print("   ⚠️  NO PLAYERS scraped from CREX!")
                        print("   This is likely due to JavaScript rendering issues.")
                
            else:
                print("❌ Could not fetch match details!")
                
        except Exception as e:
            print(f"❌ Error fetching match details: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 DIAGNOSIS")
    print("=" * 70)
    print("""
If players show ❌ (no db_player_id):
  - Name matching failed for those players
  - They will use default distributions → 50/50 results

If teams show "NO PLAYERS scraped":
  - CREX uses JavaScript to load squad tabs
  - Playwright may not be installed or working
  - Fallback to database squad should kick in

Check:
  1. Is Playwright installed? Try: pip install playwright && playwright install
  2. Are player names in a format the matcher can handle?
  3. Check server logs when loading a CREX match in the UI
""")


if __name__ == '__main__':
    test_crex_pipeline()
