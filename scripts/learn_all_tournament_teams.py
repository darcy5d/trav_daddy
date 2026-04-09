#!/usr/bin/env python3
"""
Learn ALL current unresolved tournament team codes.

This script identifies all cryptic team codes in the current CREX schedule
and learns their proper names by mining match detail pages.

Run from project root:
    python scripts/learn_all_tournament_teams.py
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.crex_scraper import CREXScraper
from src.data.tournament_team_learner import TournamentTeamLearner

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Learn all current unresolved tournament teams."""
    print("=" * 70)
    print("🏏 LEARN ALL TOURNAMENT TEAMS")
    print("=" * 70)
    
    try:
        # Initialize systems
        scraper = CREXScraper(request_delay=0.3)  
        learner = TournamentTeamLearner(max_learn_per_session=20)  # High limit for batch learning
        
        print("\n📡 Getting current CREX schedule...")
        matches = scraper.get_schedule(formats=None)  # Get all matches
        
        print(f"📊 Found {len(matches)} total matches")
        
        # Identify all unresolved codes
        print("\n🔍 Identifying unresolved team codes...")
        unresolved = learner.identify_unresolved_codes(matches)
        
        print(f"Found {len(unresolved)} unresolved codes: {unresolved[:10]}{'...' if len(unresolved) > 10 else ''}")
        
        # Show current state
        print("\n📋 Current cryptic match examples:")
        cryptic_matches = []
        for match in matches[:20]:  # Check first 20 matches
            team1_cryptic = match.team1_name == match.team1_id
            team2_cryptic = match.team2_name == match.team2_id
            
            if team1_cryptic or team2_cryptic:
                cryptic_matches.append(match)
                status_icon = '🟢' if match.status == 'live' else '🔵'
                print(f"   {match.team1_name} vs {match.team2_name} {status_icon}")
                print(f"     IDs: {match.team1_id} vs {match.team2_id}")
        
        if not unresolved:
            print("✅ No unresolved codes found! All teams are properly named.")
            return
        
        # Learn tournament teams in batches
        print(f"\n🎓 Learning tournament teams...")
        print(f"Will process {min(len(unresolved), learner.max_learn_per_session)} codes this session")
        
        learned_mappings = learner.learn_tournament_teams(unresolved, matches, scraper)
        
        print(f"\n📈 Learning results:")
        print(f"  Successfully learned: {len(learned_mappings)} teams")
        
        for fkey, name in learned_mappings.items():
            print(f"    ✅ {fkey} → {name}")
        
        failed_count = min(len(unresolved), learner.max_learn_per_session) - len(learned_mappings)
        if failed_count > 0:
            print(f"  Failed to learn: {failed_count} teams")
        
        # Show updated statistics
        print("\n📊 Tournament learning statistics:")
        stats = learner.get_learning_stats()
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.1f}{'%' if 'rate' in k else ''}")
            else:
                print(f"  {k}: {v}")
        
        # Test the improved schedule
        if learned_mappings:
            print("\n🧪 Testing improved schedule display...")
            improved_matches = scraper.get_schedule(formats=['T20'])[:5]
            
            print("Updated first 5 T20 matches:")
            for i, match in enumerate(improved_matches):
                team1_improved = match.team1_name != match.team1_id  
                team2_improved = match.team2_name != match.team2_id
                
                improvement = '✅✅' if (team1_improved and team2_improved) else '✅❌' if (team1_improved or team2_improved) else '❌❌'
                status_icon = '🟢' if match.status == 'live' else '🔵'
                
                print(f"  {i+1}. {improvement} {match.team1_name} vs {match.team2_name} {status_icon}")
        
        # Show what still needs work
        remaining_unresolved = [code for code in unresolved if code not in learned_mappings]
        if remaining_unresolved:
            print(f"\n⏳ Still need to learn: {len(remaining_unresolved)} codes")
            print("Run this script again to learn more, or they'll be learned gradually during normal usage.")
        else:
            print("\n🎉 All identified codes have been learned!")
        
        print("\n✅ Tournament team learning completed!")
        
    except Exception as e:
        print(f"\n❌ Learning failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()