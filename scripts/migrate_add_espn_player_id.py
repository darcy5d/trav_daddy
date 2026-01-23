#!/usr/bin/env python3
"""
Migration script to add espn_player_id column to players table.
This column is needed for matching players from ESPN Cricinfo data.
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


def migrate_add_espn_player_id():
    """Add espn_player_id column to players table if it doesn't exist."""
    logger.info("Starting migration: Adding espn_player_id column to players table")
    
    try:
        conn = get_connection(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Check if column exists
        if column_exists(conn, 'players', 'espn_player_id'):
            logger.info("✅ 'espn_player_id' column already exists")
            conn.close()
            return True
        
        # Add espn_player_id column
        logger.info("Adding 'espn_player_id' column to players table...")
        cursor.execute("""
            ALTER TABLE players ADD COLUMN espn_player_id INTEGER
        """)
        logger.info("✓ Added 'espn_player_id' column")
        
        # Create index for fast lookups
        logger.info("Creating index for ESPN player ID lookups...")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_players_espn_id ON players(espn_player_id)
        """)
        logger.info("✓ Created index on espn_player_id column")
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Migration completed successfully!")
        logger.info("The database now supports ESPN player ID matching")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = migrate_add_espn_player_id()
    sys.exit(0 if success else 1)
