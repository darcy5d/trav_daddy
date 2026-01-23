#!/usr/bin/env python3
"""
Migration script to add tier columns to teams table.
This fixes the "no such column: tier" error when using ELO calculator V3.
"""

import sqlite3
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATABASE_PATH
from src.data.database import get_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Check if a column exists in a table."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    return column in columns


def migrate_add_tier_columns():
    """Add tier columns to teams table if they don't exist."""
    logger.info("Starting migration: Adding tier columns to teams table")
    
    try:
        conn = get_connection(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Check and add tier column
        if not column_exists(conn, 'teams', 'tier'):
            logger.info("Adding 'tier' column to teams table...")
            cursor.execute("""
                ALTER TABLE teams ADD COLUMN tier INTEGER DEFAULT 3 CHECK(tier BETWEEN 1 AND 5)
            """)
            logger.info("✓ Added 'tier' column")
        else:
            logger.info("'tier' column already exists")
        
        # Check and add tier_last_reviewed column
        if not column_exists(conn, 'teams', 'tier_last_reviewed'):
            logger.info("Adding 'tier_last_reviewed' column to teams table...")
            cursor.execute("""
                ALTER TABLE teams ADD COLUMN tier_last_reviewed DATE
            """)
            logger.info("✓ Added 'tier_last_reviewed' column")
        else:
            logger.info("'tier_last_reviewed' column already exists")
        
        # Check and add tier_notes column
        if not column_exists(conn, 'teams', 'tier_notes'):
            logger.info("Adding 'tier_notes' column to teams table...")
            cursor.execute("""
                ALTER TABLE teams ADD COLUMN tier_notes TEXT
            """)
            logger.info("✓ Added 'tier_notes' column")
        else:
            logger.info("'tier_notes' column already exists")
        
        # Ensure all teams have tier set (default to 3 if NULL)
        logger.info("Setting default tier values for teams...")
        cursor.execute("UPDATE teams SET tier = 3 WHERE tier IS NULL")
        rows_updated = cursor.rowcount
        logger.info(f"✓ Updated {rows_updated} teams with default tier value")
        
        # Create index for fast tier lookups
        logger.info("Creating index for tier lookups...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_teams_tier ON teams(tier)")
        logger.info("✓ Created index on tier column")
        
        # Create tournament_tiers table if it doesn't exist
        logger.info("Creating tournament_tiers table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tournament_tiers (
                tournament_pattern TEXT PRIMARY KEY,
                base_tier INTEGER CHECK(base_tier BETWEEN 1 AND 5) NOT NULL,
                notes TEXT
            )
        """)
        
        # Insert tournament tier patterns
        tournament_data = [
            # World Cups and Champions Events (Tier 1)
            ('%T20 World Cup%', 1, 'Premier global tournament'),
            ('%World Cup%', 1, 'Premier global tournament'),
            ('%Champions Trophy%', 1, 'Premier ICC event'),
            ('%World Twenty20%', 1, 'Legacy World Cup naming'),
            ('%ICC Men\'s T20 World Cup%', 1, 'Full ICC World Cup title'),
            ('%ICC Women\'s T20 World Cup%', 1, 'Full ICC World Cup title'),
            # Bilateral International (Full Members) - Tier 2
            ('%tour of%', 2, 'Bilateral international series'),
            ('%T20I Series%', 2, 'T20 International series'),
            ('%Triangular%', 2, 'Multi-nation tournament'),
            ('%Tri-Series%', 2, 'Tri-nation series'),
            ('%Tri-Nation%', 2, 'Tri-nation series'),
            # Premier Franchise Leagues - Tier 3
            ('%Indian Premier League%', 3, 'IPL'),
            ('%Big Bash League%', 3, 'BBL'),
            ('%Caribbean Premier League%', 3, 'CPL'),
            ('%Pakistan Super League%', 3, 'PSL'),
            ('%The Hundred%', 3, 'The Hundred'),
            ('%Super Smash%', 3, 'New Zealand domestic T20'),
            ('%Bangladesh Premier League%', 3, 'BPL'),
            ('%Lanka Premier League%', 3, 'LPL'),
            ('%Women\'s Premier League%', 3, 'WPL India'),
            # Regional/Associate Tournaments - Tier 4
            ('%Africa%', 4, 'African regional'),
            ('%Asia Cup%', 4, 'Asian regional'),
            ('%ACC%', 4, 'Asian Cricket Council'),
            ('%Continental Cup%', 4, 'Associate regional'),
            ('%ICC World Cup Qualifier%', 4, 'World Cup qualifying'),
            ('%East Asia%', 4, 'Regional Asian'),
            ('%Europe%', 4, 'European regional'),
            # Domestic/Minor Leagues - Tier 5
            ('%County%', 5, 'English county cricket'),
            ('%Trophy%', 5, 'Domestic trophy'),
            ('%Challenge%', 5, 'Domestic challenge'),
            ('%T20 Blast%', 5, 'English domestic T20'),
            ('%Vitality Blast%', 5, 'English domestic T20'),
            ('%Syed Mushtaq Ali%', 5, 'Indian domestic T20'),
            ('%Inter-Provincial%', 5, 'Irish domestic'),
            # Catch-all patterns (lowest priority)
            ('%Cup%', 5, 'Generic cup tournament'),
            ('%Series%', 4, 'Generic series'),
            ('%Tournament%', 4, 'Generic tournament'),
        ]
        
        cursor.executemany(
            "INSERT OR IGNORE INTO tournament_tiers VALUES (?, ?, ?)",
            tournament_data
        )
        logger.info(f"✓ Inserted tournament tier patterns")
        
        # Create promotion_review_flags table if it doesn't exist
        logger.info("Creating promotion_review_flags table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS promotion_review_flags (
                flag_id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_id INTEGER NOT NULL REFERENCES teams(team_id),
                format TEXT NOT NULL CHECK(format IN ('T20', 'ODI')),
                gender TEXT NOT NULL CHECK(gender IN ('male', 'female')),
                current_tier INTEGER NOT NULL CHECK(current_tier BETWEEN 1 AND 5),
                suggested_tier INTEGER NOT NULL CHECK(suggested_tier BETWEEN 1 AND 5),
                trigger_reason TEXT NOT NULL,
                current_elo REAL NOT NULL,
                months_at_ceiling INTEGER,
                cross_tier_record TEXT,
                flagged_date DATE DEFAULT CURRENT_DATE,
                reviewed BOOLEAN DEFAULT FALSE,
                reviewed_date DATE,
                reviewer_notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(team_id, format, gender, reviewed)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_promotion_flags_pending 
            ON promotion_review_flags(reviewed, flagged_date)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_promotion_flags_team 
            ON promotion_review_flags(team_id, format, gender)
        """)
        logger.info("✓ Created promotion_review_flags table")
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Migration completed successfully!")
        logger.info("The database now supports the tiered ELO system (V3)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = migrate_add_tier_columns()
    sys.exit(0 if success else 1)
