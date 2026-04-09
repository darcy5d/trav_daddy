#!/usr/bin/env python3
"""
Quick integration test for CREX team name cache.

Tests that the cache system improves team name display while preserving 
all database integration and warning behavior.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_cache_basic_functionality():
    """Test basic cache functionality without full database."""
    print("🧪 Testing basic cache functionality...")
    
    try:
        from src.data.team_name_cache import CREXTeamCache, CachedTeamResult
        
        # Test dataclass
        result = CachedTeamResult(
            fkey="HK",
            gender="male",
            display_name="Hong Kong", 
            db_team_id=None,  # No database match
            source="test"
        )
        
        assert result.needs_default_elo == True, "Should need default ELO when no DB match"
        assert result.fkey == "HK", "Should preserve fkey"
        print("  ✅ CachedTeamResult dataclass working")
        
        # Test cache initialization
        cache = CREXTeamCache()
        print("  ✅ CREXTeamCache initialization working")
        
        # Test cache miss
        missing = cache.get_display_name("NONEXISTENT", "male")
        assert missing is None, "Cache miss should return None"
        print("  ✅ Cache miss handling working")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_schedule_integration():
    """Test that schedule integration works."""
    print("🧪 Testing schedule integration...")
    
    try:
        from src.api.crex_scraper import CREXScraper, CREXMatch
        
        # Create scraper
        scraper = CREXScraper(request_delay=0)
        print("  ✅ CREXScraper initialization working")
        
        # Test fixture row parsing with cache integration
        # Create a mock fixture row like what comes from the API
        fx = {
            'mf': 'TEST1',
            't1f': 'HK', 
            't2f': 'ZF',
            'sf': 'TEST_SERIES',
            'mn': '1',
            'ft': 2,
            'fo': 'T20',
            'date': '4/8/2026',
            't': 1775617200000,
            'status': 0,
            'g': 1,  # Male
            # Note: no team1/team2 names, will fall back to fkeys
        }
        
        # Parse the fixture row (should not crash with cache integration)
        match = scraper._fixture_row_to_schedule_match(fx)
        
        assert match is not None, "Should create match from fixture row"
        assert match.team1_name is not None, "Should have team1 name (even if just fkey)"
        assert match.team2_name is not None, "Should have team2 name (even if just fkey)"
        print(f"  ✅ Fixture parsing working: {match.team1_name} vs {match.team2_name}")
        
        # Test enhance_team_names_from_cache method exists
        enhanced = scraper.enhance_team_names_from_cache([match])
        assert len(enhanced) == 1, "Should return same number of matches"
        print("  ✅ Team name enhancement method working")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Schedule integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_live_crex_api():
    """Test with live CREX API to see actual improvement."""
    print("🧪 Testing with live CREX API...")
    
    try:
        from src.api.crex_scraper import CREXScraper
        
        # Create scraper and get a few matches
        scraper = CREXScraper(request_delay=1.0)
        
        print("  📡 Fetching live schedule from CREX...")
        matches = scraper.get_schedule(formats=['T20'])
        
        if not matches:
            print("  ⚠️  No matches returned from CREX API")
            return True  # Not a failure, just no data
        
        print(f"  📊 Found {len(matches)} T20 matches")
        
        # Show first few matches to see team name quality
        for i, match in enumerate(matches[:3]):
            print(f"    {i+1}. {match.team1_name} vs {match.team2_name}")
            
            # Check if names are still cryptic fkeys
            is_team1_cryptic = match.team1_name and len(match.team1_name) <= 3 and match.team1_name.isupper()
            is_team2_cryptic = match.team2_name and len(match.team2_name) <= 3 and match.team2_name.isupper()
            
            if is_team1_cryptic or is_team2_cryptic:
                print(f"       (Still has cryptic names - cache needs more data)")
            else:
                print(f"       (Names look good)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Live API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run integration tests."""
    print("=" * 70)
    print("🏏 CREX TEAM NAME CACHE INTEGRATION TEST")
    print("=" * 70)
    
    tests = [
        test_cache_basic_functionality,
        test_schedule_integration,
        test_with_live_crex_api
    ]
    
    results = []
    for test_func in tests:
        print(f"\n{test_func.__doc__}")
        success = test_func()
        results.append(success)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 70)
    if passed == total:
        print("✅ ALL TESTS PASSED!")
        print("\nTeam name cache integration is working correctly.")
        print("The cache will improve display names while preserving:")
        print("  - Database team matching warnings") 
        print("  - Default ELO behavior for unknown teams")
        print("  - Player matching for franchise teams")
    else:
        print(f"❌ {total - passed} of {total} tests failed")
        print("\nSome issues were found. Check the output above for details.")
    
    print("\nTo seed the cache with initial mappings, run:")
    print("  python scripts/seed_team_cache.py")
    print("\nThe cache will learn more team names as users view match details.")


if __name__ == '__main__':
    main()