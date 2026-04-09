#!/usr/bin/env python3
"""
Seed CREX team name cache with initial mappings.

This script populates the team name cache with known abbreviation mappings,
validating each one against the database to preserve all warning/flagging behavior.

Run from project root:
    python scripts/seed_team_cache.py
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.crex_scraper import CREXScraper
from src.data.team_name_cache import CREXTeamCache

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Seed the team name cache with initial abbreviation mappings."""
    print("=" * 70)
    print("🏏 CREX TEAM NAME CACHE SEEDING")
    print("=" * 70)
    
    try:
        # Initialize cache and scraper
        cache = CREXTeamCache()
        scraper = CREXScraper(request_delay=0.1)  # Short delay for seeding
        
        print("\n📊 Cache statistics before seeding:")
        stats = cache.get_cache_stats()
        print(f"  Total entries: {stats['total_entries']}")
        print(f"  Matched to DB: {stats['matched_entries']}")
        print(f"  Unknown teams: {stats['unknown_entries']}")
        print(f"  Match rate: {stats['match_rate']:.1f}%")
        
        # Populate initial mappings
        print("\n🌱 Seeding initial team mappings...")
        cache.populate_initial_mappings(scraper)
        
        # Show updated statistics
        print("\n📊 Cache statistics after seeding:")
        stats = cache.get_cache_stats()
        print(f"  Total entries: {stats['total_entries']}")
        print(f"  Matched to DB: {stats['matched_entries']}")
        print(f"  Unknown teams: {stats['unknown_entries']}")
        print(f"  Match rate: {stats['match_rate']:.1f}%")
        print(f"  Sources: {', '.join(f'{k}({v})' for k, v in stats['sources'].items())}")
        
        # Show unknown teams for manual review
        if stats['unknown_entries'] > 0:
            print(f"\n⚠️  Teams not found in database (may need manual addition):")
            unknown = cache.get_unknown_teams(limit=10)
            for team in unknown[:5]:  # Show first 5
                print(f"  - {team['fkey']} → {team['display_name']} ({team['gender']})")
            if len(unknown) > 5:
                print(f"  ... and {len(unknown) - 5} more")
        
        print("\n✅ Cache seeding completed successfully!")
        print("\nThe cache will continue to learn from:")
        print("  - Match detail pages (when users view match info)")
        print("  - global_num hints in API responses")
        print("  - Manual mappings (when added)")
        
    except Exception as e:
        print(f"\n❌ Cache seeding failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()