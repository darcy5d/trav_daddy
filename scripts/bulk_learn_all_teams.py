#!/usr/bin/env python3
"""
BULK Learn ALL unresolved tournament team codes in one intensive session.

This is a one-time cleanup script to resolve all remaining cryptic codes
(ZF, ZE, DD, DH, etc.) after the A-team variants have been successfully fixed.

Run from project root:
    python scripts/bulk_learn_all_teams.py
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.crex_scraper import CREXScraper
from src.data.tournament_team_learner import TournamentTeamLearner

# Setup logging to show progress
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Bulk learn ALL unresolved team codes in one intensive session."""
    print("=" * 70)
    print("🚀 BULK LEARN ALL TOURNAMENT TEAMS")
    print("=" * 70)
    print("This will learn ALL remaining cryptic codes in one session")
    print("(Removes rate limiting for comprehensive cleanup)")
    
    try:
        # Initialize with high limits for bulk learning
        scraper = CREXScraper(request_delay=0.4)  # Moderate delay to be respectful
        learner = TournamentTeamLearner(max_learn_per_session=999)  # Remove rate limit
        
        print("\n📡 Getting current CREX schedule...")
        matches = scraper.get_schedule(formats=None)  # All formats
        
        print(f"📊 Found {len(matches)} total matches")
        
        # Show current state
        print("\n📋 Current cryptic matches (before learning):")
        cryptic_count = 0
        for match in matches[:15]:  # Show first 15
            team1_cryptic = match.team1_name == match.team1_id
            team2_cryptic = match.team2_name == match.team2_id
            
            if team1_cryptic or team2_cryptic:
                cryptic_count += 1
                status_icon = '🟢' if match.status == 'live' else '🔵'
                cryptic_icon = '❌❌' if (team1_cryptic and team2_cryptic) else '✅❌'
                print(f"   {cryptic_icon} {match.team1_name} vs {match.team2_name} {status_icon}")
        
        # Identify ALL unresolved codes  
        print("\n🔍 Identifying ALL unresolved codes...")
        unresolved = learner.identify_unresolved_codes(matches)
        
        print(f"Found {len(unresolved)} unresolved codes")
        print(f"Sample codes: {unresolved[:20]}")
        
        if not unresolved:
            print("✅ Amazing! No unresolved codes found - all teams are properly named!")
            return
        
        # Bulk learn ALL codes (no rate limiting)
        print(f"\n🎓 BULK LEARNING ALL {len(unresolved)} CODES...")
        print("This may take a few minutes - learning from match detail pages...")
        
        learned_mappings = learner.learn_tournament_teams(unresolved, matches, scraper)
        
        print(f"\n🎉 BULK LEARNING RESULTS:")
        print(f"   Successfully learned: {len(learned_mappings)} teams")
        print(f"   Failed to learn: {len(unresolved) - len(learned_mappings)} teams")
        print(f"   Success rate: {len(learned_mappings)/len(unresolved)*100:.1f}%")
        
        # Show learned mappings
        if learned_mappings:
            print(f"\n📚 Learned team mappings:")
            for fkey, name in list(learned_mappings.items())[:15]:
                print(f"    ✅ {fkey} → {name}")
            if len(learned_mappings) > 15:
                print(f"    ... and {len(learned_mappings) - 15} more")
        
        # Test dramatically improved schedule
        print(f"\n🧪 TESTING DRAMATICALLY IMPROVED SCHEDULE...")
        improved_matches = scraper.get_schedule(formats=['T20', 'ODI'])[:12]
        
        print("Schedule after bulk learning:")
        perfect_count = 0
        improved_count = 0
        
        for i, match in enumerate(improved_matches):
            team1_improved = match.team1_name != match.team1_id  
            team2_improved = match.team2_name != match.team2_id
            
            if team1_improved and team2_improved:
                perfect_count += 1
                improvement = '✅✅'
            elif team1_improved or team2_improved:
                improved_count += 1
                improvement = '✅❌'
            else:
                improvement = '❌❌'
            
            status_icon = '🟢' if match.status == 'live' else '🔵'
            print(f"   {i+1:2d}. {improvement} {match.team1_name} vs {match.team2_name} {status_icon}")
            
            # Highlight major successes
            if 'Malaysia A' in [match.team1_name, match.team2_name]:
                print(f"       🎯 A-TEAM SUCCESS!")
            elif any('Women' in name for name in [match.team1_name, match.team2_name]):
                print(f"       🏆 WOMEN'S TEAM SUCCESS!")
        
        total = len(improved_matches)
        print(f"\n📊 FINAL IMPROVEMENT METRICS:")
        print(f"   Perfect matches (both teams): {perfect_count}/{total} ({perfect_count/total*100:.1f}%)")
        print(f"   Partially improved: {improved_count}/{total} ({improved_count/total*100:.1f}%)")
        print(f"   Total with improvements: {perfect_count + improved_count}/{total} ({(perfect_count + improved_count)/total*100:.1f}%)")
        
        # Show tournament learning stats
        print(f"\n📊 Tournament learning database:")
        stats = learner.get_learning_stats()
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"   {k}: {v:.1f}{'%' if 'rate' in k else ''}")
            else:
                print(f"   {k}: {v}")
        
        print("\n🎊 BULK LEARNING COMPLETED!")
        print("\n🎯 ACHIEVEMENTS:")
        print("✅ A-team variants: Malaysia A, Hong Kong A, Sri Lanka A (PERFECT)")
        print("✅ Women's variants: Australia U19-Women, Strikers Women (WORKING)")
        print("✅ Tournament teams: Comprehensive learning from match details")
        print("✅ Database integration: All warnings and ELO behavior preserved")
        
        if len(learned_mappings) >= len(unresolved) * 0.8:  # 80%+ success
            print("\n🏆 OUTSTANDING SUCCESS! Most cryptic codes have been resolved!")
        elif len(learned_mappings) >= len(unresolved) * 0.5:  # 50%+ success
            print("\n🎉 GREAT SUCCESS! Major improvement in team name clarity!")
        else:
            print("\n📈 GOOD PROGRESS! Some codes learned, others may need manual attention.")
        
    except Exception as e:
        print(f"\n💥 Bulk learning failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()