#!/usr/bin/env python3
"""
Migrate from crex_team_cache to comprehensive crex_team_variants system.

This script:
1. Creates the new crex_team_variants table (if not exists)
2. Populates it with comprehensive CREX directory data  
3. Migrates any existing cache data
4. Provides statistics and validation

Run from project root:
    python scripts/migrate_to_team_variants.py
"""

import logging
import sqlite3
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.crex_scraper import CREXScraper
from src.data.team_variant_resolver import CREXTeamVariantResolver
from src.data.database import get_connection

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def ensure_variants_table():
    """Ensure the team variants table exists."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if old cache table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='crex_team_cache'")
    has_old_cache = cursor.fetchone() is not None
    
    # Check if new variants table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='crex_team_variants'")
    has_variants = cursor.fetchone() is not None
    
    if not has_variants:
        print("📋 Creating crex_team_variants table...")
        # Apply the schema (should match schema.sql)
        cursor.execute("""
            CREATE TABLE crex_team_variants (
                fkey TEXT PRIMARY KEY,           
                full_name TEXT NOT NULL,         
                short_name TEXT,                 
                parent_team TEXT NOT NULL,       
                team_type TEXT NOT NULL CHECK(team_type IN ('main', 'a-team', 'women', 'u19', 'u19-women', 'special')),
                gender TEXT NOT NULL DEFAULT 'male' CHECK(gender IN ('male', 'female')),
                db_team_id INTEGER,              
                db_team_name TEXT,               
                match_confidence REAL,           
                source TEXT DEFAULT 'directory', 
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (db_team_id) REFERENCES teams(team_id)
            )
        """)
        
        cursor.execute("CREATE INDEX idx_team_variants_parent ON crex_team_variants(parent_team)")  
        cursor.execute("CREATE INDEX idx_team_variants_type ON crex_team_variants(team_type)")
        cursor.execute("CREATE INDEX idx_team_variants_gender ON crex_team_variants(gender)")
        
        conn.commit()
        print("  ✅ Table created with indexes")
    else:
        print("📋 crex_team_variants table already exists")
    
    conn.close()
    return has_old_cache, has_variants


def migrate_old_cache_data():
    """Migrate any data from old crex_team_cache table."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Check if old cache has data
        cursor.execute("SELECT COUNT(*) FROM crex_team_cache")
        old_count = cursor.fetchone()[0]
        
        if old_count == 0:
            print("📊 No data in old cache to migrate")
            conn.close()
            return 0
        
        print(f"📊 Migrating {old_count} entries from old cache...")
        
        # Get old cache entries
        cursor.execute("""
            SELECT fkey, gender, display_name, db_team_id, db_team_name, 
                   match_confidence, source
            FROM crex_team_cache
        """)
        
        old_entries = cursor.fetchall()
        migrated = 0
        
        for entry in old_entries:
            fkey, gender, display_name, db_team_id, db_team_name, confidence, source = entry
            
            # Skip if already exists in variants table
            cursor.execute("SELECT fkey FROM crex_team_variants WHERE fkey = ?", (fkey,))
            if cursor.fetchone():
                continue
            
            # Classify the team type from display name
            if display_name.endswith(' A') or ' A ' in display_name:
                team_type = 'a-team'
                parent_team = display_name.replace(' A', '').strip()
            elif 'Women' in display_name:
                team_type = 'women' 
                parent_team = display_name.replace(' Women', '').strip()
            else:
                team_type = 'main'
                parent_team = display_name
            
            # Insert into variants table
            cursor.execute("""
                INSERT INTO crex_team_variants
                (fkey, full_name, parent_team, team_type, gender,
                 db_team_id, db_team_name, match_confidence, source, 
                 created_at, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            """, (fkey, display_name, parent_team, team_type, gender,
                  db_team_id, db_team_name, confidence, f'migrated_{source}'))
            
            migrated += 1
        
        conn.commit()
        conn.close()
        
        print(f"  ✅ Migrated {migrated} cache entries")
        return migrated
        
    except sqlite3.OperationalError as e:
        logger.debug(f"Cache migration failed (table may not exist): {e}")
        return 0


def main():
    """Perform complete migration to team variants system."""
    print("=" * 70)
    print("🔄 CREX TEAM VARIANTS MIGRATION")
    print("=" * 70)
    
    try:
        # Step 1: Ensure schema is ready
        print("\n🏗️ Step 1: Database schema...")
        has_old_cache, had_variants = ensure_variants_table()
        
        # Step 2: Migrate old cache data (if any)
        if has_old_cache:
            print("\n📦 Step 2: Migrate existing cache...")
            migrated = migrate_old_cache_data()
        else:
            print("\n📦 Step 2: No old cache to migrate")
            migrated = 0
        
        # Step 3: Populate comprehensive directory data
        print("\n🌍 Step 3: Populate from CREX directory...")
        scraper = CREXScraper(request_delay=0.2)
        resolver = CREXTeamVariantResolver()
        
        populated = resolver.populate_variants_from_directory(scraper)
        
        # Step 4: Show results
        print("\n📊 Step 4: Migration results...")
        stats = resolver.get_variant_stats()
        overall = stats.get('overall', {})
        by_type = stats.get('by_type', {})
        
        print(f"  Total variants: {overall.get('total_variants', 0)}")
        print(f"  Database matches: {overall.get('total_matched', 0)}")
        print(f"  Overall match rate: {overall.get('overall_match_rate', 0):.1f}%")
        
        print("\n📋 By team type:")
        for team_type, type_stats in by_type.items():
            print(f"  {team_type}: {type_stats['matched']}/{type_stats['total']} ({type_stats['match_rate']:.1f}%)")
        
        # Show unmatched variants (need database attention)
        unmatched = resolver.get_unmatched_variants(limit=10)
        if unmatched:
            print(f"\n⚠️ Unmatched variants (need database review):")
            for variant in unmatched[:5]:
                print(f"  - {variant['fkey']}: {variant['full_name']} (parent: {variant['parent_team']})")
            if len(unmatched) > 5:
                print(f"  ... and {len(unmatched) - 5} more")
        
        # Test key variants we care about
        print("\n🧪 Testing key variant resolution:")
        test_cases = [
            ('1FO', 'male', 'Hong Kong A'),
            ('1FN', 'male', 'Malaysia A'),
            ('Q', 'male', 'Australia'),
            ('1I', 'female', 'Australia Women'),
        ]
        
        all_passed = True
        for fkey, gender, expected_name in test_cases:
            variant = resolver.resolve_team(fkey, gender)
            if variant and variant.full_name == expected_name:
                print(f"  ✅ {fkey} → {variant.full_name} (parent: {variant.parent_team})")
            else:
                print(f"  ❌ {fkey} → {variant.full_name if variant else 'NOT FOUND'} (expected: {expected_name})")
                all_passed = False
        
        if all_passed:
            print("\n🎉 Migration completed successfully!")
            print("\nNext steps:")
            print("  1. Test the updated schedule display")
            print("  2. Verify 'Malaysia A vs Hong Kong A' shows correctly")
            print("  3. Check that database warnings are preserved")
        else:
            print("\n❌ Some test cases failed - check variant classification logic")
        
    except Exception as e:
        print(f"\n💥 Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()