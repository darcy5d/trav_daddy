#!/usr/bin/env python3
"""
Baseline Snapshot Script

Captures current state metrics for the cricket database and models.
Run this BEFORE making changes to compare before/after.

Usage:
    python scripts/capture_baseline.py
    
Output:
    data/baseline_YYYYMMDD_HHMMSS.json
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.database import get_connection
from src.data.country_mapping import get_country_for_city

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def get_match_stats(conn) -> dict:
    """Get match statistics."""
    cursor = conn.cursor()
    
    # Total matches by gender
    cursor.execute("""
        SELECT gender, COUNT(*) as count,
               MIN(date) as earliest,
               MAX(date) as latest
        FROM matches
        GROUP BY gender
    """)
    
    matches = {}
    for row in cursor.fetchall():
        matches[row['gender']] = {
            'count': row['count'],
            'earliest': row['earliest'],
            'latest': row['latest']
        }
    
    # Total matches
    cursor.execute("SELECT COUNT(*) FROM matches")
    total = cursor.fetchone()[0]
    
    return {
        'total': total,
        'by_gender': matches
    }


def get_venue_stats(conn) -> dict:
    """Get venue statistics including duplicates and unknown countries."""
    cursor = conn.cursor()
    
    # Total venues
    cursor.execute("SELECT COUNT(*) FROM venues")
    total = cursor.fetchone()[0]
    
    # Get all venues with city
    cursor.execute("""
        SELECT venue_id, name, city, country
        FROM venues
        ORDER BY name
    """)
    
    venues = cursor.fetchall()
    
    # Count venues by country
    country_counts = defaultdict(int)
    unknown_venues = []
    
    for v in venues:
        # Try to get country from city mapping if not in DB
        country = v['country']
        if not country:
            country = get_country_for_city(v['city']) if v['city'] else "Unknown"
        
        if country == "Unknown":
            unknown_venues.append({
                'venue_id': v['venue_id'],
                'name': v['name'],
                'city': v['city']
            })
        
        country_counts[country] += 1
    
    # Detect potential duplicates (same name, different city formatting)
    name_groups = defaultdict(list)
    for v in venues:
        # Normalize name for grouping
        base_name = v['name'].split(',')[0].strip()
        name_groups[base_name].append({
            'venue_id': v['venue_id'],
            'name': v['name'],
            'city': v['city']
        })
    
    # Find groups with multiple venues (potential duplicates)
    duplicate_groups = []
    for base_name, group in name_groups.items():
        if len(group) > 1:
            duplicate_groups.append({
                'base_name': base_name,
                'count': len(group),
                'venues': group
            })
    
    # Sort by count descending
    duplicate_groups.sort(key=lambda x: x['count'], reverse=True)
    
    return {
        'total': total,
        'by_country': dict(country_counts),
        'unknown_country_count': len(unknown_venues),
        'unknown_venues': unknown_venues,
        'potential_duplicate_groups': len(duplicate_groups),
        'duplicate_details': duplicate_groups[:20]  # Top 20 duplicate groups
    }


def get_player_stats(conn) -> dict:
    """Get player statistics."""
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM players")
    total = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM players WHERE registry_id IS NOT NULL")
    with_registry = cursor.fetchone()[0]
    
    return {
        'total': total,
        'with_registry_id': with_registry,
        'without_registry_id': total - with_registry
    }


def get_team_stats(conn) -> dict:
    """Get team statistics."""
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM teams")
    total = cursor.fetchone()[0]
    
    # Teams with ELO
    cursor.execute("SELECT COUNT(*) FROM team_current_elo")
    with_elo = cursor.fetchone()[0]
    
    return {
        'total': total,
        'with_elo': with_elo
    }


def get_table_row_counts(conn) -> dict:
    """Get row counts for all tables."""
    tables = [
        "teams", "players", "venues", "matches", "innings",
        "deliveries", "player_match_stats", "team_elo_history",
        "player_elo_history", "team_current_elo", "player_current_elo"
    ]
    
    counts = {}
    cursor = conn.cursor()
    
    for table in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = cursor.fetchone()[0]
        except Exception:
            counts[table] = 0
    
    return counts


def check_model_files() -> dict:
    """Check which model files exist."""
    processed_dir = project_root / "data" / "processed"
    
    model_files = [
        "ball_prediction_model_t20_female.keras",
        "ball_prediction_model_t20_male.keras",
        "ball_prediction_model_t20_female_normalizer.pkl",
        "ball_prediction_model_t20_male_normalizer.pkl",
        "player_distributions_t20_female.pkl",
        "player_distributions_t20_male.pkl",
        "venue_stats_t20_female.pkl",
        "venue_stats_t20_male.pkl",
    ]
    
    results = {}
    for f in model_files:
        path = processed_dir / f
        if path.exists():
            stat = path.stat()
            results[f] = {
                'exists': True,
                'size_mb': round(stat.st_size / (1024 * 1024), 2),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
            }
        else:
            results[f] = {'exists': False}
    
    return results


def check_raw_data() -> dict:
    """Check raw data files."""
    raw_dir = project_root / "data" / "raw"
    
    results = {}
    
    for zip_file in raw_dir.glob("*.zip"):
        stat = zip_file.stat()
        results[zip_file.name] = {
            'size_mb': round(stat.st_size / (1024 * 1024), 2),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
        }
    
    # Count JSON files in each folder
    for folder in raw_dir.iterdir():
        if folder.is_dir():
            json_count = len(list(folder.glob("*.json")))
            results[folder.name] = {'json_files': json_count}
    
    return results


def main():
    """Capture baseline snapshot."""
    logger.info("=" * 60)
    logger.info("CAPTURING BASELINE SNAPSHOT")
    logger.info("=" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    snapshot = {
        'timestamp': datetime.now().isoformat(),
        'version': '1.0',
    }
    
    # Database stats
    logger.info("Connecting to database...")
    conn = get_connection()
    
    try:
        logger.info("Getting match statistics...")
        snapshot['matches'] = get_match_stats(conn)
        
        logger.info("Getting venue statistics...")
        snapshot['venues'] = get_venue_stats(conn)
        
        logger.info("Getting player statistics...")
        snapshot['players'] = get_player_stats(conn)
        
        logger.info("Getting team statistics...")
        snapshot['teams'] = get_team_stats(conn)
        
        logger.info("Getting table row counts...")
        snapshot['table_counts'] = get_table_row_counts(conn)
        
    finally:
        conn.close()
    
    # Model files
    logger.info("Checking model files...")
    snapshot['model_files'] = check_model_files()
    
    # Raw data
    logger.info("Checking raw data...")
    snapshot['raw_data'] = check_raw_data()
    
    # Save snapshot
    output_dir = project_root / "data"
    output_file = output_dir / f"baseline_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(snapshot, f, indent=2)
    
    logger.info(f"\nSnapshot saved to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("BASELINE SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 MATCHES")
    print(f"   Total: {snapshot['matches']['total']:,}")
    for gender, stats in snapshot['matches']['by_gender'].items():
        print(f"   {gender}: {stats['count']:,} ({stats['earliest']} to {stats['latest']})")
    
    print(f"\n🏟️  VENUES")
    print(f"   Total: {snapshot['venues']['total']:,}")
    print(f"   Unknown country: {snapshot['venues']['unknown_country_count']}")
    print(f"   Potential duplicate groups: {snapshot['venues']['potential_duplicate_groups']}")
    
    if snapshot['venues']['duplicate_details']:
        print(f"\n   Top duplicates:")
        for d in snapshot['venues']['duplicate_details'][:5]:
            print(f"     • {d['base_name']}: {d['count']} variants")
    
    print(f"\n👤 PLAYERS")
    print(f"   Total: {snapshot['players']['total']:,}")
    print(f"   With registry ID: {snapshot['players']['with_registry_id']:,}")
    
    print(f"\n🏏 TEAMS")
    print(f"   Total: {snapshot['teams']['total']:,}")
    print(f"   With ELO: {snapshot['teams']['with_elo']:,}")
    
    print("\n" + "=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())



