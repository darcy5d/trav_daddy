#!/usr/bin/env python3
"""
Migration script to update ELO tables to V2 schema.
This adds format and gender columns to support the tiered ELO calculator V3.
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


def migrate_elo_tables_to_v2():
    """Migrate ELO tables from V1 to V2 schema."""
    logger.info("Starting migration: ELO tables V1 -> V2")
    logger.info("This will recreate the ELO tables with format+gender separation")
    
    try:
        conn = get_connection(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Drop old ELO tables
        logger.info("Dropping old ELO tables...")
        cursor.execute("DROP TABLE IF EXISTS team_elo_history")
        cursor.execute("DROP TABLE IF EXISTS player_elo_history")
        cursor.execute("DROP TABLE IF EXISTS team_current_elo")
        cursor.execute("DROP TABLE IF EXISTS player_current_elo")
        
        # Drop old views
        logger.info("Dropping old views...")
        cursor.execute("DROP VIEW IF EXISTS team_rankings_t20")
        cursor.execute("DROP VIEW IF EXISTS team_rankings_odi")
        cursor.execute("DROP VIEW IF EXISTS player_batting_rankings_t20")
        cursor.execute("DROP VIEW IF EXISTS player_bowling_rankings_t20")
        
        # Create new team_elo_history table
        logger.info("Creating new team_elo_history table...")
        cursor.execute("""
            CREATE TABLE team_elo_history (
                elo_id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_id INTEGER NOT NULL REFERENCES teams(team_id),
                date DATE NOT NULL,
                match_id INTEGER REFERENCES matches(match_id),
                format TEXT NOT NULL CHECK(format IN ('T20', 'ODI')),
                gender TEXT NOT NULL CHECK(gender IN ('male', 'female')),
                elo REAL DEFAULT 1500,
                elo_change REAL DEFAULT 0,
                is_monthly_snapshot BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("CREATE INDEX idx_team_elo_team_date ON team_elo_history(team_id, date)")
        cursor.execute("CREATE INDEX idx_team_elo_format_gender ON team_elo_history(format, gender)")
        cursor.execute("CREATE INDEX idx_team_elo_monthly ON team_elo_history(is_monthly_snapshot, date)")
        
        # Create new player_elo_history table
        logger.info("Creating new player_elo_history table...")
        cursor.execute("""
            CREATE TABLE player_elo_history (
                elo_id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER NOT NULL REFERENCES players(player_id),
                date DATE NOT NULL,
                match_id INTEGER REFERENCES matches(match_id),
                format TEXT NOT NULL CHECK(format IN ('T20', 'ODI')),
                gender TEXT NOT NULL CHECK(gender IN ('male', 'female')),
                batting_elo REAL DEFAULT 1500,
                bowling_elo REAL DEFAULT 1500,
                overall_elo REAL DEFAULT 1500,
                batting_elo_change REAL DEFAULT 0,
                bowling_elo_change REAL DEFAULT 0,
                overall_elo_change REAL DEFAULT 0,
                is_monthly_snapshot BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("CREATE INDEX idx_player_elo_player_date ON player_elo_history(player_id, date)")
        cursor.execute("CREATE INDEX idx_player_elo_format_gender ON player_elo_history(format, gender)")
        cursor.execute("CREATE INDEX idx_player_elo_monthly ON player_elo_history(is_monthly_snapshot, date)")
        
        # Create new team_current_elo table
        logger.info("Creating new team_current_elo table...")
        cursor.execute("""
            CREATE TABLE team_current_elo (
                team_id INTEGER PRIMARY KEY REFERENCES teams(team_id),
                elo_t20_male REAL DEFAULT 1500,
                elo_t20_female REAL DEFAULT 1500,
                elo_odi_male REAL DEFAULT 1500,
                elo_odi_female REAL DEFAULT 1500,
                last_t20_male_date DATE,
                last_t20_female_date DATE,
                last_odi_male_date DATE,
                last_odi_female_date DATE,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create new player_current_elo table
        logger.info("Creating new player_current_elo table...")
        cursor.execute("""
            CREATE TABLE player_current_elo (
                player_id INTEGER PRIMARY KEY REFERENCES players(player_id),
                batting_elo_t20_male REAL DEFAULT 1500,
                bowling_elo_t20_male REAL DEFAULT 1500,
                overall_elo_t20_male REAL DEFAULT 1500,
                batting_elo_t20_female REAL DEFAULT 1500,
                bowling_elo_t20_female REAL DEFAULT 1500,
                overall_elo_t20_female REAL DEFAULT 1500,
                batting_elo_odi_male REAL DEFAULT 1500,
                bowling_elo_odi_male REAL DEFAULT 1500,
                overall_elo_odi_male REAL DEFAULT 1500,
                batting_elo_odi_female REAL DEFAULT 1500,
                bowling_elo_odi_female REAL DEFAULT 1500,
                overall_elo_odi_female REAL DEFAULT 1500,
                last_t20_male_date DATE,
                last_t20_female_date DATE,
                last_odi_male_date DATE,
                last_odi_female_date DATE,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create new views
        logger.info("Creating new ranking views...")
        
        cursor.execute("""
            CREATE VIEW team_rankings_t20_male AS
            SELECT 
                t.name as team_name,
                e.elo_t20_male as elo,
                e.last_t20_male_date as last_match
            FROM team_current_elo e
            JOIN teams t ON e.team_id = t.team_id
            WHERE e.elo_t20_male != 1500
            ORDER BY e.elo_t20_male DESC
        """)
        
        cursor.execute("""
            CREATE VIEW team_rankings_t20_female AS
            SELECT 
                t.name as team_name,
                e.elo_t20_female as elo,
                e.last_t20_female_date as last_match
            FROM team_current_elo e
            JOIN teams t ON e.team_id = t.team_id
            WHERE e.elo_t20_female != 1500
            ORDER BY e.elo_t20_female DESC
        """)
        
        cursor.execute("""
            CREATE VIEW team_rankings_odi_male AS
            SELECT 
                t.name as team_name,
                e.elo_odi_male as elo,
                e.last_odi_male_date as last_match
            FROM team_current_elo e
            JOIN teams t ON e.team_id = t.team_id
            WHERE e.elo_odi_male != 1500
            ORDER BY e.elo_odi_male DESC
        """)
        
        cursor.execute("""
            CREATE VIEW team_rankings_odi_female AS
            SELECT 
                t.name as team_name,
                e.elo_odi_female as elo,
                e.last_odi_female_date as last_match
            FROM team_current_elo e
            JOIN teams t ON e.team_id = t.team_id
            WHERE e.elo_odi_female != 1500
            ORDER BY e.elo_odi_female DESC
        """)
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Migration completed successfully!")
        logger.info("ELO tables now support format+gender separation for V3 calculator")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = migrate_elo_tables_to_v2()
    sys.exit(0 if success else 1)
