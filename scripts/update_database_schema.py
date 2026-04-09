#!/usr/bin/env python3
"""
Update database schema with new tables (like crex_team_cache).

This script applies any missing schema changes to bring the database up to date.

Run from project root:
    python scripts/update_database_schema.py
"""

import logging
import sqlite3
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.database import get_connection

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_table_exists(cursor, table_name):
    """Check if a table exists in the database."""
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name=?
    """, (table_name,))
    return cursor.fetchone() is not None


def apply_crex_team_cache_schema(cursor):
    """Apply the crex_team_cache table schema."""
    print("📋 Adding crex_team_cache table...")
    
    # Create the table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS crex_team_cache (
            fkey TEXT NOT NULL,              -- CREX team fkey (e.g., "1FN", "ZF") 
            gender TEXT NOT NULL DEFAULT 'male',  -- Gender context ('male' or 'female')
            display_name TEXT NOT NULL,      -- Human readable name (e.g., "Hong Kong A")
            db_team_id INTEGER,              -- Matched database team ID (NULL if no match)
            db_team_name TEXT,               -- Matched database team name (NULL if no match)  
            match_confidence REAL,           -- Confidence from match_team_to_db (NULL if no match)
            source TEXT NOT NULL,            -- How we learned this: 'match_detail', 'global_num', 'manual', 'abbreviation'
            last_validated TIMESTAMP,        -- When we last ran match_team_to_db validation
            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            PRIMARY KEY (fkey, gender),
            FOREIGN KEY (db_team_id) REFERENCES teams(team_id)
        )
    """)
    
    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_crex_team_cache_fkey ON crex_team_cache(fkey)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_crex_team_cache_db_team ON crex_team_cache(db_team_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_crex_team_cache_source ON crex_team_cache(source)")
    
    print("  ✅ crex_team_cache table created with indexes")


def main():
    """Apply database schema updates."""
    print("=" * 70)
    print("🗃️  DATABASE SCHEMA UPDATE")
    print("=" * 70)
    
    try:
        # Get database connection
        conn = get_connection()
        cursor = conn.cursor()
        
        print("\n🔍 Checking existing schema...")
        
        # Check what tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        existing_tables = {row[0] for row in cursor.fetchall()}
        
        print(f"  Found {len(existing_tables)} existing tables")
        
        # Apply updates for missing tables
        updates_applied = 0
        
        if 'crex_team_cache' not in existing_tables:
            apply_crex_team_cache_schema(cursor)
            updates_applied += 1
        else:
            print("📋 crex_team_cache table already exists")
            
            # Verify it has correct schema
            cursor.execute("PRAGMA table_info(crex_team_cache)")
            columns = {row[1] for row in cursor.fetchall()}  # row[1] is column name
            expected_columns = {'fkey', 'gender', 'display_name', 'db_team_id', 'db_team_name', 
                              'match_confidence', 'source', 'last_validated', 'last_seen', 'created_at'}
            
            missing_columns = expected_columns - columns
            if missing_columns:
                print(f"  ⚠️  Missing columns: {missing_columns}")
                print("  Consider dropping and recreating table if schema is outdated")
            else:
                print("  ✅ Schema looks correct")
        
        # Commit changes
        conn.commit()
        conn.close()
        
        print(f"\n🎉 Database schema update completed!")
        print(f"  Applied {updates_applied} updates")
        
        if updates_applied > 0:
            print("\n📝 Next steps:")
            print("  1. Run 'python scripts/seed_team_cache.py' to populate initial mappings")
            print("  2. Test the improved team names in upcoming matches")
        else:
            print("\n✅ Database schema is up to date")
        
    except sqlite3.Error as e:
        print(f"\n❌ Database error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()