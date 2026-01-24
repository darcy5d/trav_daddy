"""
Full recalculation of ELO ratings with tiered system.

This script:
1. Backs up current ELO tables
2. Applies tiered schema changes
3. Applies team tier classifications
4. Clears team_elo_history and player_elo_history
5. Resets team_current_elo and player_current_elo to tier-based initials
6. Runs calculator_v3 on all 2019-2025 matches chronologically
7. Generates promotion flags
8. Creates validation report
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_db_connection
from src.elo.calculator_v3 import EloCalculatorV3, calculate_all_elos_v3
from config import DATABASE_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def backup_current_database():
    """Create timestamped backup of database before recalculation."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = DATABASE_PATH.parent / f"cricket_backup_tiered_elo_{timestamp}.db"
    
    logger.info(f"Creating backup: {backup_path}")
    shutil.copy(DATABASE_PATH, backup_path)
    logger.info(f"✓ Backup created: {backup_path}")
    
    return backup_path


def apply_schema_changes(conn):
    """Apply tiered ELO schema changes."""
    logger.info("Applying tiered ELO schema...")
    cursor = conn.cursor()
    
    try:
        # Check if tier column exists
        cursor.execute("PRAGMA table_info(teams)")
        columns = {row[1] for row in cursor.fetchall()}
        
        # Add tier columns if they don't exist
        if 'tier' not in columns:
            cursor.execute("ALTER TABLE teams ADD COLUMN tier INTEGER DEFAULT 3")
            logger.info("  Added tier column")
        else:
            logger.info("  tier column already exists")
        
        if 'tier_last_reviewed' not in columns:
            cursor.execute("ALTER TABLE teams ADD COLUMN tier_last_reviewed DATE")
            logger.info("  Added tier_last_reviewed column")
        
        if 'tier_notes' not in columns:
            cursor.execute("ALTER TABLE teams ADD COLUMN tier_notes TEXT")
            logger.info("  Added tier_notes column")
        
        # Create index if not exists
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_teams_tier ON teams(tier)")
        
        # Create tournament_tiers table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tournament_tiers (
              tournament_pattern TEXT PRIMARY KEY,
              base_tier INTEGER CHECK(base_tier BETWEEN 1 AND 5) NOT NULL,
              notes TEXT
            )
        """)
        
        # Insert tournament patterns (hardcoded to avoid parsing issues)
        tournament_inserts = [
            ('%T20 World Cup%', 1, 'Premier global tournament'),
            ('%World Cup%', 1, 'Premier global tournament'),
            ('%Champions Trophy%', 1, 'Premier ICC event'),
            ('%World Twenty20%', 1, 'Legacy World Cup naming'),
            ('%ICC Men''s T20 World Cup%', 1, 'Full ICC World Cup title'),
            ('%ICC Women''s T20 World Cup%', 1, 'Full ICC World Cup title'),
            ('%tour of%', 2, 'Bilateral international series'),
            ('%T20I Series%', 2, 'T20 International series'),
            ('%Triangular%', 2, 'Multi-nation tournament'),
            ('%Tri-Series%', 2, 'Tri-nation series'),
            ('%Tri-Nation%', 2, 'Tri-nation series'),
            ('%Indian Premier League%', 3, 'IPL'),
            ('%Big Bash League%', 3, 'BBL'),
            ('%Caribbean Premier League%', 3, 'CPL'),
            ('%Pakistan Super League%', 3, 'PSL'),
            ('%The Hundred%', 3, 'The Hundred'),
            ('%Super Smash%', 3, 'New Zealand domestic T20'),
            ('%Bangladesh Premier League%', 3, 'BPL'),
            ('%Lanka Premier League%', 3, 'LPL'),
            ('%Women''s Premier League%', 3, 'WPL India'),
            ('%Africa%', 4, 'African regional'),
            ('%Asia Cup%', 4, 'Asian regional'),
            ('%ACC%', 4, 'Asian Cricket Council'),
            ('%Continental Cup%', 4, 'Associate regional'),
            ('%ICC World Cup Qualifier%', 4, 'World Cup qualifying'),
            ('%East Asia%', 4, 'Regional Asian'),
            ('%Europe%', 4, 'European regional'),
            ('%County%', 5, 'English county cricket'),
            ('%Trophy%', 5, 'Domestic trophy'),
            ('%Challenge%', 5, 'Domestic challenge'),
            ('%T20 Blast%', 5, 'English domestic T20'),
            ('%Vitality Blast%', 5, 'English domestic T20'),
            ('%Syed Mushtaq Ali%', 5, 'Indian domestic T20'),
            ('%Inter-Provincial%', 5, 'Irish domestic'),
            ('%Cup%', 5, 'Generic cup tournament'),
            ('%Series%', 4, 'Generic series'),
            ('%Tournament%', 4, 'Generic tournament'),
        ]
        
        for pattern, tier, notes in tournament_inserts:
            cursor.execute("""
                INSERT OR IGNORE INTO tournament_tiers (tournament_pattern, base_tier, notes)
                VALUES (?, ?, ?)
            """, (pattern, tier, notes))
        
        # Create promotion_review_flags table if not exists
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
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_promotion_flags_pending ON promotion_review_flags(reviewed, flagged_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_promotion_flags_team ON promotion_review_flags(team_id, format, gender)")
        
        conn.commit()
        logger.info("✓ Schema applied successfully")
        return True
    except Exception as e:
        logger.error(f"Error applying schema: {e}")
        conn.rollback()
        return False


def apply_team_classifications(conn):
    """Apply team tier classifications using classify_teams_v2.py logic."""
    logger.info("Applying team tier classifications...")
    
    # Import the classification function from our new script
    try:
        from scripts.classify_teams_v2 import classify_team_tier
    except ImportError:
        # Try alternate import path
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "classify_teams_v2", 
            Path(__file__).parent / "classify_teams_v2.py"
        )
        classify_teams_v2 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(classify_teams_v2)
        classify_team_tier = classify_teams_v2.classify_team_tier
    
    try:
        cursor = conn.cursor()
        
        # Get all teams
        cursor.execute("SELECT team_id, name FROM teams")
        teams = cursor.fetchall()
        
        tier_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        for team in teams:
            new_tier = classify_team_tier(team['name'])
            tier_counts[new_tier] += 1
            
            cursor.execute(
                "UPDATE teams SET tier = ?, tier_last_reviewed = datetime('now') WHERE team_id = ?",
                (new_tier, team['team_id'])
            )
        
        conn.commit()
        
        logger.info("✓ Team classifications applied:")
        for tier, count in tier_counts.items():
            tier_name = {
                1: "Full ICC Members",
                2: "Associate Nations",
                3: "Premier Leagues (IPL, BBL, etc.)",
                4: "Other Leagues & Major Domestic",
                5: "Regional/Development Cricket"
            }.get(tier, f"Tier {tier}")
            logger.info(f"  Tier {tier} ({tier_name}): {count} teams")
        
        return True
    except Exception as e:
        logger.error(f"Error applying classifications: {e}")
        import traceback
        traceback.print_exc()
        return False


def reset_elo_tables(conn):
    """Clear ELO history and reset current ELO to tier-based initials."""
    logger.info("Resetting ELO tables...")
    cursor = conn.cursor()
    
    # Clear history
    logger.info("  Clearing ELO history...")
    cursor.execute("DELETE FROM team_elo_history")
    cursor.execute("DELETE FROM player_elo_history")
    
    # Delete promotion flags (they'll be regenerated)
    cursor.execute("DELETE FROM promotion_review_flags")
    
    # Reset team current ELO based on tiers
    logger.info("  Resetting team ELO to tier-based initials...")
    for tier, initial_elo in EloCalculatorV3.TIER_INITIAL_RATINGS.items():
        for format_type in ['t20', 'odi']:
            for gender in ['male', 'female']:
                col = f'elo_{format_type}_{gender}'
                cursor.execute(f"""
                    UPDATE team_current_elo
                    SET {col} = ?
                    WHERE team_id IN (SELECT team_id FROM teams WHERE tier = ?)
                """, (initial_elo, tier))
    
    # Reset player current ELO to 1500 (simple approach)
    logger.info("  Resetting player ELO to 1500...")
    for format_type in ['t20', 'odi']:
        for gender in ['male', 'female']:
            for rating_type in ['batting', 'bowling', 'overall']:
                col = f'{rating_type}_elo_{format_type}_{gender}'
                cursor.execute(f"UPDATE player_current_elo SET {col} = 1500")
    
    conn.commit()
    logger.info("✓ ELO tables reset complete")


def print_summary(conn):
    """Print summary of recalculation."""
    cursor = conn.cursor()
    
    print("\n" + "="*70)
    print("TIERED ELO RECALCULATION SUMMARY")
    print("="*70)
    
    # Team ELO updates by format/gender
    print("\nTeam ELO Updates:")
    for fmt in ['T20', 'ODI']:
        for gen in ['male', 'female']:
            cursor.execute("""
                SELECT COUNT(*) FROM team_elo_history 
                WHERE format = ? AND gender = ? AND NOT is_monthly_snapshot
            """, (fmt, gen))
            count = cursor.fetchone()[0]
            print(f"  {fmt} {gen}: {count} updates")
    
    # Top teams by tier
    print("\nTop 5 Teams by Tier (T20 Male):")
    for tier in [1, 2, 3]:
        cursor.execute("""
            SELECT t.name, t.tier, e.elo_t20_male
            FROM team_current_elo e
            JOIN teams t ON e.team_id = t.team_id
            WHERE t.tier = ? AND e.elo_t20_male != 1500
            ORDER BY e.elo_t20_male DESC
            LIMIT 5
        """, (tier,))
        
        tier_name = {1: "Elite", 2: "Full Members", 3: "Top Associates/Franchises"}[tier]
        print(f"\n  Tier {tier} ({tier_name}):")
        for row in cursor.fetchall():
            print(f"    {row['name']:35} {row['elo_t20_male']:.0f}")
    
    # Promotion flags
    cursor.execute("SELECT COUNT(*) FROM promotion_review_flags WHERE reviewed = FALSE")
    flag_count = cursor.fetchone()[0]
    print(f"\n{flag_count} promotion review flags generated")
    
    if flag_count > 0:
        print("\nSample Promotion Flags:")
        cursor.execute("""
            SELECT t.name, prf.current_tier, prf.suggested_tier, prf.trigger_reason
            FROM promotion_review_flags prf
            JOIN teams t ON prf.team_id = t.team_id
            WHERE prf.reviewed = FALSE
            LIMIT 5
        """)
        for row in cursor.fetchall():
            print(f"  {row['name']:30} Tier {row['current_tier']} → {row['suggested_tier']}")
            print(f"    Reason: {row['trigger_reason']}")
    
    print("\n" + "="*70)


def main():
    """Main recalculation function."""
    import argparse
    parser = argparse.ArgumentParser(description='Recalculate tiered ELO ratings')
    parser.add_argument('--reset', action='store_true', 
                        help='Skip confirmation prompt and run immediately')
    parser.add_argument('--no-backup', action='store_true',
                        help='Skip database backup (for testing)')
    args = parser.parse_args()
    
    print("="*70)
    print("TIERED ELO SYSTEM - FULL HISTORICAL RECALCULATION")
    print("="*70)
    print("\nThis will:")
    print("  1. Backup current database")
    print("  2. Apply tiered ELO schema")
    print("  3. Classify all teams into 5 tiers")
    print("  4. Reset all ELO ratings")
    print("  5. Recalculate ~11,000 matches chronologically")
    print("  6. Generate promotion review flags")
    print("\nEstimated time: 5-10 minutes")
    
    if not args.reset:
        proceed = input("\nProceed? (yes/no): ").strip().lower()
        if proceed != 'yes':
            print("Aborted.")
            return 1
    else:
        print("\n[--reset flag set, proceeding automatically]")
    
    print()
    
    # Step 1: Backup
    print("\n" + "-"*70)
    print("STEP 1: Creating backup...")
    print("-"*70)
    if args.no_backup:
        print("  [--no-backup flag set, skipping backup]")
        backup_path = None
    else:
        backup_path = backup_current_database()
    
    # Step 2: Apply schema changes
    print("\n" + "-"*70)
    print("STEP 2: Applying schema changes...")
    print("-"*70)
    with get_db_connection() as conn:
        if not apply_schema_changes(conn):
            print("\n❌ Schema application failed. Check logs.")
            return 1
    
    # Step 3: Apply team classifications
    print("\n" + "-"*70)
    print("STEP 3: Applying team tier classifications...")
    print("-"*70)
    with get_db_connection() as conn:
        if not apply_team_classifications(conn):
            print("\n❌ Team classification failed. Check logs.")
            return 1
    
    # Step 4: Reset ELO data
    print("\n" + "-"*70)
    print("STEP 4: Resetting ELO data...")
    print("-"*70)
    with get_db_connection() as conn:
        reset_elo_tables(conn)
    
    # Step 5: Recalculate all ELOs
    print("\n" + "-"*70)
    print("STEP 5: Recalculating all ELOs with tiered system...")
    print("-"*70)
    print("This will take 5-10 minutes for ~11,000 matches...")
    print()
    
    calculate_all_elos_v3(force_recalculate=True)
    
    # Step 6: Print summary
    print("\n" + "-"*70)
    print("STEP 6: Generating summary...")
    print("-"*70)
    with get_db_connection() as conn:
        print_summary(conn)
    
    print("\n" + "="*70)
    print("✓ RECALCULATION COMPLETE!")
    print("="*70)
    if backup_path:
        print(f"\nBackup saved: {backup_path}")
    print("\nNext steps:")
    print("  1. Review rankings: python -m src.elo.calculator_v3 --rankings")
    print("  2. Check promotion flags: curl http://localhost:5001/api/admin/promotion-flags")
    print("  3. Start Flask server to view UI: python app/main.py")
    if backup_path:
        print("  4. If issues occur, restore from backup")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

