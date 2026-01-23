#!/usr/bin/env python3
"""
Schema validation script for Cricket Predictor database.
Validates that all required tables and columns exist for the tiered ELO system.
"""

import sqlite3
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATABASE_PATH
from src.data.database import get_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# Define required schema
REQUIRED_TABLES = {
    'teams': [
        'team_id', 'name', 'country_code', 'is_international', 'team_type',
        'tier', 'tier_last_reviewed', 'tier_notes', 'created_at'
    ],
    'players': [
        'player_id', 'name', 'registry_id', 'espn_player_id', 'country', 
        'batting_style', 'bowling_style', 'is_active', 'created_at', 'updated_at'
    ],
    'venues': [
        'venue_id', 'name', 'city', 'country', 'canonical_name',
        'region', 'created_at'
    ],
    'matches': [
        'match_id', 'cricsheet_id', 'match_type', 'date', 'venue_id',
        'team1_id', 'team2_id', 'toss_winner_id', 'toss_decision',
        'winner_id', 'win_type', 'win_margin', 'player_of_match_id',
        'overs_per_innings', 'match_number', 'event_name', 'gender', 'created_at'
    ],
    'innings': [
        'innings_id', 'match_id', 'innings_number', 'batting_team_id',
        'bowling_team_id', 'total_runs', 'total_wickets', 'total_overs',
        'total_extras', 'target_runs', 'is_complete', 'declared', 'created_at'
    ],
    'deliveries': [
        'delivery_id', 'innings_id', 'over_number', 'ball_number',
        'batter_id', 'bowler_id', 'non_striker_id', 'runs_batter',
        'runs_extras', 'runs_total', 'extras_wides', 'extras_noballs',
        'extras_byes', 'extras_legbyes', 'extras_penalty', 'is_wicket',
        'wicket_type', 'dismissed_player_id', 'fielder1_id', 'fielder2_id',
        'is_boundary_four', 'is_boundary_six', 'created_at'
    ],
    'player_match_stats': [
        'stat_id', 'match_id', 'player_id', 'team_id', 'batting_innings',
        'runs_scored', 'balls_faced', 'fours_hit', 'sixes_hit', 'not_out',
        'batting_position', 'bowling_innings', 'overs_bowled', 'runs_conceded',
        'wickets_taken', 'maidens', 'wides_bowled', 'noballs_bowled',
        'dots_bowled', 'catches', 'run_outs', 'stumpings', 'created_at'
    ],
    'team_elo_history': [
        'elo_id', 'team_id', 'date', 'match_id', 'format', 'gender',
        'elo', 'elo_change', 'is_monthly_snapshot', 'created_at'
    ],
    'player_elo_history': [
        'elo_id', 'player_id', 'date', 'match_id', 'format', 'gender',
        'batting_elo', 'bowling_elo', 'overall_elo', 'batting_elo_change',
        'bowling_elo_change', 'overall_elo_change', 'is_monthly_snapshot', 'created_at'
    ],
    'team_current_elo': [
        'team_id', 'elo_t20_male', 'elo_t20_female', 'elo_odi_male',
        'elo_odi_female', 'last_t20_male_date', 'last_t20_female_date',
        'last_odi_male_date', 'last_odi_female_date', 'updated_at'
    ],
    'player_current_elo': [
        'player_id', 'batting_elo_t20_male', 'bowling_elo_t20_male',
        'overall_elo_t20_male', 'batting_elo_t20_female', 'bowling_elo_t20_female',
        'overall_elo_t20_female', 'batting_elo_odi_male', 'bowling_elo_odi_male',
        'overall_elo_odi_male', 'batting_elo_odi_female', 'bowling_elo_odi_female',
        'overall_elo_odi_female', 'last_t20_male_date', 'last_t20_female_date',
        'last_odi_male_date', 'last_odi_female_date', 'updated_at'
    ],
    'tournament_tiers': [
        'tournament_pattern', 'base_tier', 'notes'
    ],
    'promotion_review_flags': [
        'flag_id', 'team_id', 'format', 'gender', 'current_tier',
        'suggested_tier', 'trigger_reason', 'current_elo', 'months_at_ceiling',
        'cross_tier_record', 'flagged_date', 'reviewed', 'reviewed_date',
        'reviewer_notes', 'created_at'
    ],
    'model_versions': [
        'id', 'model_name', 'gender', 'format_type', 'model_path',
        'normalizer_path', 'data_earliest_date', 'data_latest_date',
        'training_samples', 'training_duration_seconds', 'model_size_mb',
        'accuracy_metrics', 'is_active', 'notes', 'created_at'
    ]
}

REQUIRED_INDEXES = [
    ('teams', 'idx_teams_tier'),
    ('players', 'idx_players_name'),
    ('players', 'idx_players_registry_id'),
    ('players', 'idx_players_espn_id'),
    ('matches', 'idx_matches_date'),
    ('matches', 'idx_matches_type'),
    ('matches', 'idx_matches_teams'),
    ('innings', 'idx_innings_match'),
    ('deliveries', 'idx_deliveries_innings'),
    ('deliveries', 'idx_deliveries_batter'),
    ('deliveries', 'idx_deliveries_bowler'),
    ('team_elo_history', 'idx_team_elo_team_date'),
    ('team_elo_history', 'idx_team_elo_format_gender'),
    ('player_elo_history', 'idx_player_elo_player_date'),
    ('player_elo_history', 'idx_player_elo_format_gender'),
    ('promotion_review_flags', 'idx_promotion_flags_pending'),
    ('promotion_review_flags', 'idx_promotion_flags_team'),
]

REQUIRED_VIEWS = [
    'match_summary',
    'team_rankings_t20_male',
    'team_rankings_t20_female',
    'team_rankings_odi_male',
    'team_rankings_odi_female',
    'player_batting_rankings_t20_male',
    'player_batting_rankings_t20_female',
    'player_bowling_rankings_t20_male',
    'player_bowling_rankings_t20_female',
]


def get_table_columns(conn: sqlite3.Connection, table_name: str) -> List[str]:
    """Get list of columns in a table."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    return [row[1] for row in cursor.fetchall()]


def get_table_indexes(conn: sqlite3.Connection, table_name: str) -> List[str]:
    """Get list of indexes on a table."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA index_list({table_name})")
    return [row[1] for row in cursor.fetchall()]


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Check if a table exists."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name=?
    """, (table_name,))
    return cursor.fetchone() is not None


def view_exists(conn: sqlite3.Connection, view_name: str) -> bool:
    """Check if a view exists."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='view' AND name=?
    """, (view_name,))
    return cursor.fetchone() is not None


def index_exists(conn: sqlite3.Connection, index_name: str) -> bool:
    """Check if an index exists."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='index' AND name=?
    """, (index_name,))
    return cursor.fetchone() is not None


def validate_schema(db_path: Path = DATABASE_PATH) -> Tuple[bool, List[str]]:
    """
    Validate database schema against requirements.
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    try:
        conn = get_connection(db_path)
        cursor = conn.cursor()
        
        logger.info("Starting schema validation...")
        logger.info(f"Database: {db_path}")
        
        # Check tables and columns
        logger.info("\n📋 Validating tables and columns...")
        for table_name, required_columns in REQUIRED_TABLES.items():
            if not table_exists(conn, table_name):
                issues.append(f"❌ Missing table: {table_name}")
                logger.error(f"  ❌ Table '{table_name}' does not exist")
                continue
            
            existing_columns = get_table_columns(conn, table_name)
            missing_columns = set(required_columns) - set(existing_columns)
            
            if missing_columns:
                for col in missing_columns:
                    issue = f"❌ Missing column: {table_name}.{col}"
                    issues.append(issue)
                    logger.error(f"  {issue}")
            else:
                logger.info(f"  ✅ Table '{table_name}' has all required columns")
        
        # Check indexes
        logger.info("\n🔍 Validating indexes...")
        for table_name, index_name in REQUIRED_INDEXES:
            if not index_exists(conn, index_name):
                issue = f"⚠️  Missing index: {index_name} on {table_name}"
                issues.append(issue)
                logger.warning(f"  {issue}")
            else:
                logger.info(f"  ✅ Index '{index_name}' exists")
        
        # Check views
        logger.info("\n👁️  Validating views...")
        for view_name in REQUIRED_VIEWS:
            if not view_exists(conn, view_name):
                issue = f"⚠️  Missing view: {view_name}"
                issues.append(issue)
                logger.warning(f"  {issue}")
            else:
                logger.info(f"  ✅ View '{view_name}' exists")
        
        # Check tournament_tiers has data
        logger.info("\n🏆 Validating tournament tier data...")
        cursor.execute("SELECT COUNT(*) FROM tournament_tiers")
        tier_count = cursor.fetchone()[0]
        if tier_count == 0:
            issue = "⚠️  tournament_tiers table is empty (should have ~45 patterns)"
            issues.append(issue)
            logger.warning(f"  {issue}")
        else:
            logger.info(f"  ✅ tournament_tiers has {tier_count} patterns")
        
        # Check teams have tier values
        logger.info("\n🎯 Validating team tier classifications...")
        cursor.execute("SELECT COUNT(*) FROM teams WHERE tier IS NULL")
        null_tier_count = cursor.fetchone()[0]
        if null_tier_count > 0:
            issue = f"⚠️  {null_tier_count} teams have NULL tier values"
            issues.append(issue)
            logger.warning(f"  {issue}")
        else:
            cursor.execute("SELECT tier, COUNT(*) FROM teams GROUP BY tier ORDER BY tier")
            tier_distribution = cursor.fetchall()
            logger.info("  ✅ All teams have tier assignments:")
            tier_names = {
                1: 'Elite Full Members',
                2: 'Full Members',
                3: 'Top Associates/Premier Franchises',
                4: 'Associates/Regional',
                5: 'Emerging/Domestic'
            }
            for tier, count in tier_distribution:
                logger.info(f"     Tier {tier} ({tier_names.get(tier, 'Unknown')}): {count} teams")
        
        conn.close()
        
        # Summary
        logger.info("\n" + "="*70)
        if not issues:
            logger.info("✅ SCHEMA VALIDATION PASSED")
            logger.info("   All required tables, columns, and data are present!")
            logger.info("="*70)
            return True, []
        else:
            logger.error("❌ SCHEMA VALIDATION FAILED")
            logger.error(f"   Found {len(issues)} issue(s):")
            for issue in issues:
                logger.error(f"   {issue}")
            logger.info("="*70)
            return False, issues
        
    except Exception as e:
        logger.error(f"❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False, [f"Validation error: {str(e)}"]


def print_fix_suggestions(issues: List[str]):
    """Print suggestions for fixing schema issues."""
    if not issues:
        return
    
    logger.info("\n" + "="*70)
    logger.info("💡 SUGGESTED FIXES:")
    logger.info("="*70)
    
    if any("Missing table" in issue or "Missing column" in issue for issue in issues):
        logger.info("\n1. Reset database with updated schema:")
        logger.info("   - In GUI: Go to Training tab → 'Reset Database' → 'Full Retrain'")
        logger.info("   - Or run: python scripts/migrate_add_tier_columns.py")
        logger.info("   - Then run: python scripts/migrate_elo_tables_to_v2.py")
    
    if any("tournament_tiers" in issue for issue in issues):
        logger.info("\n2. Populate tournament tier patterns:")
        logger.info("   - Run: python scripts/migrate_add_tier_columns.py")
    
    if any("tier" in issue.lower() and "NULL" in issue for issue in issues):
        logger.info("\n3. Apply team tier classifications:")
        logger.info("   - Run tier classification script")
        logger.info("   - Or manually assign tiers to teams")
    
    if any("Missing index" in issue or "Missing view" in issue for issue in issues):
        logger.info("\n4. Recreate indexes and views:")
        logger.info("   - These are warnings and won't prevent operation")
        logger.info("   - But may impact query performance")


if __name__ == "__main__":
    is_valid, issues = validate_schema()
    
    if not is_valid:
        print_fix_suggestions(issues)
        sys.exit(1)
    
    sys.exit(0)
