"""
Database connection and initialization module.

Handles SQLite database setup and provides connection utilities.
"""

import sqlite3
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Generator

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import DATABASE_PATH

logger = logging.getLogger(__name__)

# Path to schema file
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Get a database connection with optimized settings.
    
    Args:
        db_path: Optional path to database file. Uses default if not provided.
        
    Returns:
        SQLite connection object
    """
    if db_path is None:
        db_path = DATABASE_PATH
    
    conn = sqlite3.connect(str(db_path))
    
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON")
    
    # Optimize for performance
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA cache_size = -64000")  # 64MB cache
    conn.execute("PRAGMA temp_store = MEMORY")
    
    # Return rows as dictionaries
    conn.row_factory = sqlite3.Row
    
    return conn


@contextmanager
def get_db_connection(db_path: Optional[Path] = None) -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager for database connections.
    
    Args:
        db_path: Optional path to database file.
        
    Yields:
        SQLite connection object
    """
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        conn.close()


def init_database(db_path: Optional[Path] = None, force_recreate: bool = False) -> bool:
    """
    Initialize the database with the schema.
    
    Args:
        db_path: Optional path to database file.
        force_recreate: If True, drop existing tables and recreate.
        
    Returns:
        True if successful, False otherwise
    """
    if db_path is None:
        db_path = DATABASE_PATH
    
    try:
        # Check if database already exists
        db_exists = db_path.exists()
        
        if db_exists and not force_recreate:
            logger.info(f"Database already exists at {db_path}")
            return True
        
        if force_recreate and db_exists:
            logger.warning("Recreating database - all existing data will be lost!")
            db_path.unlink()
        
        # Read schema file
        if not SCHEMA_PATH.exists():
            logger.error(f"Schema file not found: {SCHEMA_PATH}")
            return False
        
        schema_sql = SCHEMA_PATH.read_text()
        
        # Create database and execute schema
        with get_db_connection(db_path) as conn:
            conn.executescript(schema_sql)
            logger.info(f"Database initialized at {db_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False


def get_table_counts(db_path: Optional[Path] = None) -> dict:
    """
    Get row counts for all tables in the database.
    
    Args:
        db_path: Optional path to database file.
        
    Returns:
        Dictionary mapping table names to row counts
    """
    tables = [
        "teams", "players", "venues", "matches", "innings",
        "deliveries", "player_match_stats", "team_elo_history",
        "player_elo_history", "team_current_elo", "player_current_elo"
    ]
    
    counts = {}
    
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                counts[table] = cursor.fetchone()[0]
            except sqlite3.OperationalError:
                counts[table] = 0
    
    return counts


def print_database_summary(db_path: Optional[Path] = None):
    """Print a summary of the database contents."""
    counts = get_table_counts(db_path)
    
    logger.info("\n" + "=" * 50)
    logger.info("DATABASE SUMMARY")
    logger.info("=" * 50)
    
    for table, count in counts.items():
        logger.info(f"{table:25} {count:>10,} rows")
    
    logger.info("=" * 50)


class DatabaseManager:
    """
    Database manager class for easier access to common operations.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DATABASE_PATH
    
    def get_or_create_team(self, conn: sqlite3.Connection, name: str) -> int:
        """Get team ID, creating if necessary."""
        cursor = conn.cursor()
        
        # Try to find existing team
        cursor.execute("SELECT team_id FROM teams WHERE name = ?", (name,))
        row = cursor.fetchone()
        
        if row:
            return row[0]
        
        # Create new team
        cursor.execute(
            "INSERT INTO teams (name) VALUES (?)",
            (name,)
        )
        return cursor.lastrowid
    
    def get_or_create_player(
        self,
        conn: sqlite3.Connection,
        name: str,
        registry_id: Optional[str] = None
    ) -> int:
        """Get player ID, creating if necessary."""
        cursor = conn.cursor()
        
        # Try to find by registry_id first (most reliable)
        if registry_id:
            cursor.execute(
                "SELECT player_id FROM players WHERE registry_id = ?",
                (registry_id,)
            )
            row = cursor.fetchone()
            if row:
                return row[0]
        
        # Try to find by name
        cursor.execute(
            "SELECT player_id FROM players WHERE name = ?",
            (name,)
        )
        row = cursor.fetchone()
        
        if row:
            # Update registry_id if we now have it
            if registry_id:
                cursor.execute(
                    "UPDATE players SET registry_id = ? WHERE player_id = ?",
                    (registry_id, row[0])
                )
            return row[0]
        
        # Create new player
        cursor.execute(
            "INSERT INTO players (name, registry_id) VALUES (?, ?)",
            (name, registry_id)
        )
        return cursor.lastrowid
    
    def get_or_create_venue(
        self,
        conn: sqlite3.Connection,
        name: str,
        city: Optional[str] = None
    ) -> int:
        """
        Get venue ID, creating if necessary.
        
        Uses fuzzy matching to find existing venues and populates
        country, canonical_name, and region fields.
        """
        from src.data.venue_normalizer import (
            normalize_venue_name,
            extract_canonical_name,
            venue_similarity,
            extract_city_from_venue
        )
        from src.data.country_mapping import (
            get_country_for_venue,
            get_region_for_country
        )
        
        cursor = conn.cursor()
        
        # Try exact match first
        if city:
            cursor.execute(
                "SELECT venue_id FROM venues WHERE name = ? AND city = ?",
                (name, city)
            )
        else:
            cursor.execute(
                "SELECT venue_id FROM venues WHERE name = ?",
                (name,)
            )
        
        row = cursor.fetchone()
        if row:
            return row[0]
        
        # Try to extract city from venue name if not provided
        if not city:
            city = extract_city_from_venue(name)
        
        # Try fuzzy matching against existing venues
        # Only match if city matches or one city is NULL
        if city:
            # Get venues in the same city
            cursor.execute(
                "SELECT venue_id, name, city FROM venues WHERE city = ? OR city IS NULL",
                (city,)
            )
        else:
            # If no city provided, only match venues without city or with extracted city
            cursor.execute(
                "SELECT venue_id, name, city FROM venues WHERE city IS NULL"
            )
        
        similar_venues = cursor.fetchall()
        
        # Check for similar venues
        for existing in similar_venues:
            similarity = venue_similarity(name, existing['name'])
            if similarity >= 0.85:
                logger.debug(f"Fuzzy matched venue: '{name}' -> '{existing['name']}' (score: {similarity:.2f})")
                return existing['venue_id']
        
        # No match found - create new venue with enriched data
        country = get_country_for_venue(name, city)
        region = get_region_for_country(country) if country != "Unknown" else None
        canonical = extract_canonical_name(name, city)
        
        cursor.execute(
            """INSERT INTO venues (name, city, country, canonical_name, region) 
               VALUES (?, ?, ?, ?, ?)""",
            (name, city, country if country != "Unknown" else None, canonical, region)
        )
        return cursor.lastrowid
    
    def match_exists(self, conn: sqlite3.Connection, cricsheet_id: str) -> bool:
        """Check if a match has already been ingested."""
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM matches WHERE cricsheet_id = ?",
            (cricsheet_id,)
        )
        return cursor.fetchone() is not None


def main():
    """Initialize the database."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    success = init_database()
    
    if success:
        print_database_summary()
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())

