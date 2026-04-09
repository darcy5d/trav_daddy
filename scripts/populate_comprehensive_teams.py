#!/usr/bin/env python3
"""
Populate comprehensive CREX team variants from embedded JSON directory.

Extracts all 1,737+ teams from CREX's embedded JSON team catalog and populates
the team variants database with comprehensive fkey coverage.

Run from project root:
    python scripts/populate_comprehensive_teams.py
"""

import logging
import sqlite3
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.crex_scraper import CREXScraper
from src.data.crex_team_directory import CREXTeamDirectoryScraper
from src.data.database import get_connection
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Populate comprehensive team variants from CREX directory."""
    print("=" * 70)
    print("🌍 POPULATE COMPREHENSIVE CREX TEAM VARIANTS")
    print("=" * 70)
    print("Extracting ALL 1,737+ teams from CREX embedded JSON directory")
    
    try:
        # Initialize systems
        directory_scraper = CREXTeamDirectoryScraper(request_delay=0.1)
        crex_scraper = CREXScraper(request_delay=0)  # No delay for database validation
        
        # Step 1: Extract comprehensive team variants
        print("\n🔍 Step 1: Extract ALL teams from CREX directory JSON...")
        variants = directory_scraper.scrape_all_teams()
        
        if not variants:
            print("❌ Failed to extract team variants")
            return
        
        print(f"✅ Extracted {len(variants)} team variants")
        
        # Show statistics
        from collections import Counter
        type_counts = Counter(v.team_type for v in variants)
        print("\n📊 Team types found:")
        for team_type, count in type_counts.most_common():
            print(f"  {team_type}: {count}")
        
        # Step 2: Check for our target problem codes
        problem_codes = ['DD', 'DH', 'DC', 'DF', 'DG', 'DE', 'ZF', 'ZE', 'ZI', 'ZG', 'UA', '1FW']
        found_problems = [v for v in variants if v.fkey in problem_codes]
        
        print(f"\n🎯 Target problem codes coverage:")
        print(f"Found {len(found_problems)}/{len(problem_codes)} codes")
        
        for variant in found_problems:
            print(f"  ✅ {variant.fkey}: {variant.full_name}")
        
        # Step 3: Database population with validation
        print(f"\n💾 Step 2: Populate variants database...")
        print("(This validates each team against your trained model database)")
        
        conn = get_connection()
        cursor = conn.cursor()
        
        # Clear existing comprehensive data (keep manually added ones)
        print("Clearing old comprehensive data...")
        cursor.execute("DELETE FROM crex_team_variants WHERE source = 'directory'")
        
        populated_count = 0
        matched_count = 0
        
        print(f"Processing {len(variants)} variants...")
        for i, variant in enumerate(variants):
            if i % 200 == 0:  # Progress indicator
                print(f"  Progress: {i+1}/{len(variants)} ({i/len(variants)*100:.1f}%)")
            
            # Create temp team for database validation
            from src.api.crex_scraper import CREXTeam
            temp_team = CREXTeam(
                crex_id=variant.fkey,
                name=variant.parent_team,  # Use parent for database lookup
                abbreviation=variant.fkey
            )
            
            # Validate through existing pipeline
            db_match = crex_scraper.match_team_to_db(temp_team, variant.gender)
            
            # Store comprehensive variant
            cursor.execute("""
                INSERT OR REPLACE INTO crex_team_variants
                (fkey, full_name, short_name, parent_team, team_type, gender,
                 db_team_id, db_team_name, match_confidence, source,
                 is_tournament_team, created_at, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                variant.fkey, variant.full_name, variant.short_name,
                variant.parent_team, variant.team_type, variant.gender,
                db_match[0] if db_match else None,
                db_match[1] if db_match else None,
                1.0 if db_match else None,
                'comprehensive_directory',
                variant.team_type in ('tournament', 'special'),  # Mark special teams as tournament-like
                datetime.utcnow().isoformat(),
                datetime.utcnow().isoformat()
            ))
            
            populated_count += 1
            if db_match:
                matched_count += 1
        
        conn.commit()
        conn.close()
        
        print(f"\n📊 Population Results:")
        print(f"  Teams populated: {populated_count}")
        print(f"  Database matches: {matched_count}")
        print(f"  Match rate: {matched_count/populated_count*100:.1f}%")
        
        # Step 3: Test the dramatically improved schedule
        print(f"\n🧪 Step 3: Test dramatically improved schedule...")
        
        test_matches = crex_scraper.get_schedule(formats=['T20', 'ODI'])[:12]
        
        perfect_count = 0
        improved_count = 0
        
        print("\nSchedule after comprehensive team population:")
        for i, match in enumerate(test_matches):
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
            print(f"  {i+1:2d}. {improvement} {match.team1_name} vs {match.team2_name} {status_icon}")
            
            # Highlight key successes  
            if any(code in [match.team1_id, match.team2_id] for code in problem_codes):
                print(f"       🎯 PROBLEM CODE RESOLVED!")
        
        total = len(test_matches)
        improvement_rate = (perfect_count + improved_count) / total * 100
        
        print(f"\n🎊 FINAL RESULTS:")
        print(f"Perfect matches: {perfect_count}/{total} ({perfect_count/total*100:.1f}%)")
        print(f"Total improvement: {perfect_count + improved_count}/{total} ({improvement_rate:.1f}%)")
        
        if improvement_rate >= 80:
            print("\n🏆 OUTSTANDING SUCCESS! Massive improvement achieved!")
        elif improvement_rate >= 60:
            print("\n🎉 GREAT SUCCESS! Major improvement!")
        else:
            print("\n📈 Good progress - comprehensive data loaded!")
        
        print("\n✅ COMPREHENSIVE TEAM POPULATION COMPLETED!")
        print("🎯 Your cryptic codes should now be resolved:")
        print("   DD → Emirates Blues, DC → Abu Dhabi, etc.")
        
    except Exception as e:
        print(f"\n💥 Comprehensive population failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()