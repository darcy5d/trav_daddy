#!/usr/bin/env python3
"""
Venue Migration Script

Adds canonical_name and region columns to venues table,
and populates country data using the country_mapping module.

Usage:
    python scripts/migrate_venues.py
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.database import get_connection
from src.data.country_mapping import (
    get_country_for_venue, 
    get_region_for_country,
    get_location_for_venue
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def add_columns_if_missing(conn):
    """Add canonical_name and region columns if they don't exist."""
    cursor = conn.cursor()
    
    # Check existing columns
    cursor.execute("PRAGMA table_info(venues)")
    columns = {row[1] for row in cursor.fetchall()}
    
    if 'canonical_name' not in columns:
        logger.info("Adding canonical_name column to venues table")
        cursor.execute("ALTER TABLE venues ADD COLUMN canonical_name TEXT")
    
    if 'region' not in columns:
        logger.info("Adding region column to venues table")
        cursor.execute("ALTER TABLE venues ADD COLUMN region TEXT")
    
    conn.commit()


def populate_country_data(conn):
    """Populate country field for venues that don't have one."""
    cursor = conn.cursor()
    
    # Get venues without country
    cursor.execute("""
        SELECT venue_id, name, city 
        FROM venues 
        WHERE country IS NULL OR country = ''
    """)
    venues = cursor.fetchall()
    
    updated = 0
    for venue in venues:
        country = get_country_for_venue(venue['name'], venue['city'])
        if country != "Unknown":
            cursor.execute(
                "UPDATE venues SET country = ? WHERE venue_id = ?",
                (country, venue['venue_id'])
            )
            updated += 1
            logger.debug(f"Set country for {venue['name']}: {country}")
    
    conn.commit()
    logger.info(f"Updated country for {updated} venues")


def populate_region_data(conn):
    """Populate region field for West Indies venues."""
    cursor = conn.cursor()
    
    # Get all venues with their country
    cursor.execute("""
        SELECT venue_id, name, city, country 
        FROM venues 
        WHERE country IS NOT NULL
    """)
    venues = cursor.fetchall()
    
    updated = 0
    for venue in venues:
        country = venue['country']
        if country:
            region = get_region_for_country(country)
            if region:
                cursor.execute(
                    "UPDATE venues SET region = ? WHERE venue_id = ?",
                    (region, venue['venue_id'])
                )
                updated += 1
    
    conn.commit()
    logger.info(f"Set region for {updated} West Indies venues")


def populate_canonical_names(conn):
    """Create canonical names by removing redundant city suffix."""
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT venue_id, name, city 
        FROM venues 
        WHERE canonical_name IS NULL
    """)
    venues = cursor.fetchall()
    
    updated = 0
    for venue in venues:
        name = venue['name']
        city = venue['city']
        
        # Create canonical name by removing city suffix if present
        canonical = name
        if city and name.endswith(f", {city}"):
            canonical = name[:-len(f", {city}")]
        
        cursor.execute(
            "UPDATE venues SET canonical_name = ? WHERE venue_id = ?",
            (canonical, venue['venue_id'])
        )
        updated += 1
    
    conn.commit()
    logger.info(f"Set canonical_name for {updated} venues")


def print_summary(conn):
    """Print summary of venue data after migration."""
    cursor = conn.cursor()
    
    # Count by country
    cursor.execute("""
        SELECT country, COUNT(*) as cnt 
        FROM venues 
        GROUP BY country 
        ORDER BY cnt DESC 
        LIMIT 15
    """)
    
    print("\n" + "=" * 50)
    print("VENUE MIGRATION SUMMARY")
    print("=" * 50)
    
    print("\nTop 15 countries by venue count:")
    for row in cursor.fetchall():
        country = row['country'] or 'Unknown'
        print(f"  {country}: {row['cnt']}")
    
    # Count with regions
    cursor.execute("SELECT COUNT(*) FROM venues WHERE region IS NOT NULL")
    with_region = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM venues WHERE country IS NULL OR country = ''")
    no_country = cursor.fetchone()[0]
    
    print(f"\nVenues with region (West Indies): {with_region}")
    print(f"Venues still without country: {no_country}")
    print("=" * 50)


def main():
    """Run venue migration."""
    logger.info("=" * 60)
    logger.info("VENUE MIGRATION")
    logger.info("=" * 60)
    
    conn = get_connection()
    
    try:
        # Step 1: Add columns if missing
        add_columns_if_missing(conn)
        
        # Step 2: Populate country data
        populate_country_data(conn)
        
        # Step 3: Populate region data
        populate_region_data(conn)
        
        # Step 4: Create canonical names
        populate_canonical_names(conn)
        
        # Print summary
        print_summary(conn)
        
        logger.info("Migration completed successfully!")
        
    finally:
        conn.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())



