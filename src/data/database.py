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
            extract_city_from_venue,
            extract_state_from_venue,
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
        state = extract_state_from_venue(name, city, country if country != "Unknown" else None)

        cursor.execute("PRAGMA table_info(venues)")
        venue_columns = {row[1] for row in cursor.fetchall()}
        has_state_column = 'state' in venue_columns
        
        if has_state_column:
            cursor.execute(
                """INSERT INTO venues (name, city, country, state, canonical_name, region) 
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (name, city, country if country != "Unknown" else None, state, canonical, region)
            )
        else:
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


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Return True if `table.column` exists. SQLite doesn't have IF NOT EXISTS
    for ALTER TABLE ADD COLUMN, so callers do the check themselves."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cursor.fetchall())


def init_franchise_tables(db_path: Optional[Path] = None) -> bool:
    """
    Schema V4: franchise grouping + cross-source identifiers + merge proposals.

    Idempotent. Safe to call on every app startup. Adds:
      - team_groups, team_external_ids, team_merge_proposals tables
      - teams.franchise_id and teams.canonical_team_id columns
      - Self-grouping for any teams that haven't been backfilled yet (so the
        runtime resolver always has a valid mapping). Explicit franchise
        unifications (RCB, Capitals, Punjab Kings) are applied separately by
        scripts/backfill_franchises.py so app startup never silently rewrites
        ratings.
    """
    if db_path is None:
        db_path = DATABASE_PATH

    schema_path = Path(__file__).parent / "schema_v4_franchise.sql"
    if not schema_path.exists():
        logger.error(f"Schema file not found: {schema_path}")
        return False

    try:
        schema_sql = schema_path.read_text()

        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()

            conn.executescript(schema_sql)

            # Add columns to the existing teams table if missing.
            if not _column_exists(conn, "teams", "franchise_id"):
                cursor.execute(
                    "ALTER TABLE teams ADD COLUMN franchise_id INTEGER REFERENCES team_groups(group_id)"
                )
                logger.info("Added teams.franchise_id column")
            if not _column_exists(conn, "teams", "canonical_team_id"):
                cursor.execute(
                    "ALTER TABLE teams ADD COLUMN canonical_team_id INTEGER REFERENCES teams(team_id)"
                )
                logger.info("Added teams.canonical_team_id column")

            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_teams_franchise ON teams(franchise_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_teams_canonical ON teams(canonical_team_id)"
            )

            # Self-group any teams that don't yet have a franchise. This keeps
            # the runtime resolver total: every team always points somewhere.
            # Explicit multi-id franchise unifications happen in the backfill
            # script (Phase B), not here.
            cursor.execute(
                """
                SELECT team_id, name, country_code, team_type
                FROM teams
                WHERE franchise_id IS NULL
                """
            )
            ungrouped = cursor.fetchall()
            for row in ungrouped:
                team_id = row["team_id"]
                name = row["name"]
                country = row["country_code"]
                ttype = row["team_type"] or "domestic"
                # Map teams.team_type -> team_groups.group_type. Anything we
                # don't recognise lands in 'domestic' so the CHECK doesn't fire.
                gtype = ttype if ttype in (
                    "franchise", "international", "domestic"
                ) else "domestic"
                if gtype == "international":
                    gtype = "national"

                cursor.execute(
                    """
                    INSERT OR IGNORE INTO team_groups (
                        canonical_name, group_type, country_code, notes
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (name, gtype, country, "Auto-created during V4 migration (self-group)"),
                )
                cursor.execute(
                    "SELECT group_id FROM team_groups WHERE canonical_name = ?",
                    (name,),
                )
                grp = cursor.fetchone()
                if not grp:
                    continue
                group_id = grp["group_id"]

                cursor.execute(
                    """
                    UPDATE teams
                    SET franchise_id = ?, canonical_team_id = ?
                    WHERE team_id = ?
                    """,
                    (group_id, team_id, team_id),
                )

            if ungrouped:
                logger.info(
                    f"Self-grouped {len(ungrouped)} teams under their own franchise rows"
                )

            logger.info("Franchise schema (V4) initialized")

        return True

    except Exception as e:
        logger.error(f"Failed to initialize franchise schema: {e}")
        import traceback

        traceback.print_exc()
        return False


def init_betting_tables(db_path: Optional[Path] = None) -> bool:
    """Wave 5 Phase 6b + Wave 5.7: bet_ledger schema (V5 + V6 paper-bet columns).

    Idempotent. Safe to call on every app startup. Creates `bet_ledger` from
    schema_v5_betting.sql, then adds Wave 5.7 paper-bet columns (bet_kind,
    strategy_label, bankroll_at_proposal, bankroll_after_settle) via
    ALTER TABLE ADD COLUMN if they don't already exist.
    """
    if db_path is None:
        db_path = DATABASE_PATH

    schema_path = Path(__file__).parent / "schema_v5_betting.sql"
    if not schema_path.exists():
        logger.error(f"Schema file not found: {schema_path}")
        return False
    v6_path = Path(__file__).parent / "schema_v6_paper_betting.sql"

    try:
        schema_sql = schema_path.read_text()
        with get_db_connection(db_path) as conn:
            conn.executescript(schema_sql)

            # Wave 5.7: add paper-bet columns idempotently
            paper_columns = [
                ("bet_kind",              "TEXT DEFAULT 'real'"),
                ("strategy_label",        "TEXT"),
                ("bankroll_at_proposal",  "REAL"),
                ("bankroll_after_settle", "REAL"),
                # Wave 5.7b: phase tag - 'pre_toss' (default) or 'post_toss'
                # for bets placed AFTER the toss outcome was known. Lets
                # us A/B-compare pre-toss vs post-toss strategy ROI.
                ("phase",                 "TEXT DEFAULT 'pre_toss'"),
                # Confirmed XI hash at bet time (when post-toss). Helps
                # debugging which lineup the model used.
                ("xi_signature",          "TEXT"),
                # Toss outcome captured at bet time (post-toss only).
                ("toss_winner_team_id",   "INTEGER"),
                ("toss_chose_to",         "TEXT"),
                # Wave 5.8.2: scheduled match start (ISO8601 UTC) at the
                # moment we placed the bet. Lets the UI show "T-X.Xh to
                # kickoff" per bet and post-hoc-bucket bets by actual
                # time-to-match relative to the paper-trading comparison.
                ("kickoff_at",            "TEXT"),
                # Wave 5.9.1: compact JSON snapshot of active model versions
                # at bet time, e.g. '{"t20_male":"male_t20_20260416",...}'.
                # Enables per-model-version calibration grouping in rollups.
                ("model_snapshot",        "TEXT"),
            ]
            for col_name, col_type in paper_columns:
                if not _column_exists(conn, "bet_ledger", col_name):
                    conn.execute(f"ALTER TABLE bet_ledger ADD COLUMN {col_name} {col_type}")
                    logger.info(f"Added bet_ledger.{col_name} column")

            # V6 schema script (just CREATE INDEX IF NOT EXISTS lines)
            if v6_path.exists():
                conn.executescript(v6_path.read_text())

            logger.info("Betting schema (V5 + V6 paper) initialized")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize betting schema: {e}")
        import traceback
        traceback.print_exc()
        return False


def init_cashout_columns(db_path: Optional[Path] = None) -> bool:
    """Wave 5.10: add in-game cashout columns + index to bet_ledger.

    Idempotent. Safe to call on every app startup.
    """
    if db_path is None:
        db_path = DATABASE_PATH

    cashout_columns = [
        ("cashout_triggered_at",  "TEXT"),
        ("cashout_price",         "REAL"),
        ("cashout_pnl_usdc",      "REAL"),
        ("cashout_threshold_used","REAL"),
        ("cashout_order_id",      "TEXT"),
        ("cashout_reason",        "TEXT"),
    ]

    schema_path = Path(__file__).parent / "schema_v8_cashout.sql"

    try:
        with get_db_connection(db_path) as conn:
            for col_name, col_type in cashout_columns:
                if not _column_exists(conn, "bet_ledger", col_name):
                    conn.execute(
                        f"ALTER TABLE bet_ledger ADD COLUMN {col_name} {col_type}"
                    )
                    logger.info(f"Added bet_ledger.{col_name} column")

            # Apply schema file for the partial index (idempotent).
            if schema_path.exists():
                conn.executescript(schema_path.read_text())

            logger.info("Cashout columns (V8) initialized")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize cashout columns: {e}")
        import traceback
        traceback.print_exc()
        return False


def init_crex_xi_cache(db_path: Optional[Path] = None) -> bool:
    """Wave 5.11: create crex_xi_cache table + index.

    Idempotent. Safe to call on every app startup.
    """
    if db_path is None:
        db_path = DATABASE_PATH

    schema_path = Path(__file__).parent / "schema_v9_crex_xi_cache.sql"
    if not schema_path.exists():
        logger.error(f"Schema file not found: {schema_path}")
        return False

    try:
        schema_sql = schema_path.read_text()
        with get_db_connection(db_path) as conn:
            conn.executescript(schema_sql)
            logger.info("CREX XI cache table (V9) initialized")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize CREX XI cache table: {e}")
        import traceback
        traceback.print_exc()
        return False


def init_twap_tables(db_path: Optional[Path] = None) -> bool:
    """Wave 5.9: TWAP order execution tables (order_plans + order_chunks).

    Idempotent. Safe to call on every app startup.
    """
    if db_path is None:
        db_path = DATABASE_PATH

    schema_path = Path(__file__).parent / "schema_v7_twap.sql"
    if not schema_path.exists():
        logger.error(f"Schema file not found: {schema_path}")
        return False

    try:
        schema_sql = schema_path.read_text()
        with get_db_connection(db_path) as conn:
            conn.executescript(schema_sql)
            logger.info("TWAP tables (V7) initialized")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize TWAP tables: {e}")
        import traceback
        traceback.print_exc()
        return False


def init_order_history(db_path: Optional[Path] = None) -> bool:
    """Wave 5.13: order_history audit table + bet_ledger reason/category cols.

    Idempotent. Safe to call on every app startup. Adds:
      - order_history table (every Polymarket order id we have ever posted)
      - bet_ledger.cancel_reason / .error_category / .reconciled_at columns
    """
    if db_path is None:
        db_path = DATABASE_PATH

    schema_path = Path(__file__).parent / "schema_v10_order_history.sql"
    if not schema_path.exists():
        logger.error(f"Schema file not found: {schema_path}")
        return False

    audit_columns = [
        ("cancel_reason",   "TEXT"),
        ("error_category",  "TEXT"),
        ("reconciled_at",   "TEXT"),
    ]

    try:
        schema_sql = schema_path.read_text()
        with get_db_connection(db_path) as conn:
            conn.executescript(schema_sql)
            for col_name, col_type in audit_columns:
                if not _column_exists(conn, "bet_ledger", col_name):
                    conn.execute(
                        f"ALTER TABLE bet_ledger ADD COLUMN {col_name} {col_type}"
                    )
                    logger.info(f"Added bet_ledger.{col_name} column")
            logger.info("Order history (V10) initialized")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize order_history schema: {e}")
        import traceback
        traceback.print_exc()
        return False


def init_model_versions_table(db_path: Optional[Path] = None) -> bool:
    """
    Initialize the model_versions table.
    
    Args:
        db_path: Optional path to database file.
        
    Returns:
        True if successful, False otherwise
    """
    if db_path is None:
        db_path = DATABASE_PATH
    
    try:
        schema_path = Path(__file__).parent / "schema_model_versions.sql"
        
        if not schema_path.exists():
            logger.error(f"Schema file not found: {schema_path}")
            return False
        
        schema_sql = schema_path.read_text()
        
        with get_db_connection(db_path) as conn:
            conn.executescript(schema_sql)
            logger.info("Model versions table initialized")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize model_versions table: {e}")
        return False


def get_model_versions(
    db_path: Optional[Path] = None,
    gender: Optional[str] = None,
    format_type: Optional[str] = None,
    active_only: bool = False
) -> list:
    """
    Get model versions from the database.
    
    Args:
        db_path: Optional path to database file.
        gender: Filter by gender ('male' or 'female').
        format_type: Filter by format ('T20' or 'ODI').
        active_only: If True, only return active models.
        
    Returns:
        List of model version dictionaries
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        query = "SELECT * FROM model_versions WHERE 1=1"
        params = []
        
        if gender:
            query += " AND gender = ?"
            params.append(gender)
        
        if format_type:
            query += " AND format_type = ?"
            params.append(format_type)
        
        if active_only:
            query += " AND is_active = 1"
        
        query += " ORDER BY created_at DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        return [dict(row) for row in rows]


def get_active_model_snapshot(conn=None, db_path: Optional[Path] = None) -> str:
    """Return a compact JSON string of currently active model versions.

    Keyed by '{format_type}_{gender}'.lower(), e.g.:
        '{"t20_male":"male_t20_20260416","odi_male":"male_odi_20260416",...}'

    Used to stamp each bet_ledger row so calibration can be grouped by the
    exact model instance that made the prediction. Pass an open connection
    or db_path; if neither, uses the default DATABASE_PATH.
    """
    import json as _json

    def _fetch(c):
        c.execute(
            "SELECT format_type, gender, model_name FROM model_versions "
            "WHERE is_active = 1 ORDER BY format_type, gender"
        )
        return {
            f"{row[0].lower()}_{row[1].lower()}": row[2]
            for row in c.fetchall()
        }

    if conn is not None:
        snapshot = _fetch(conn.cursor())
    else:
        with get_db_connection(db_path) as _conn:
            snapshot = _fetch(_conn.cursor())

    return _json.dumps(snapshot, sort_keys=True)


def save_model_version(
    model_name: str,
    gender: str,
    format_type: str,
    model_path: str,
    normalizer_path: str,
    data_earliest_date: Optional[str] = None,
    data_latest_date: Optional[str] = None,
    training_samples: Optional[int] = None,
    training_duration_seconds: Optional[int] = None,
    model_size_mb: Optional[float] = None,
    accuracy_metrics: Optional[str] = None,
    is_active: bool = False,
    notes: Optional[str] = None,
    db_path: Optional[Path] = None
) -> int:
    """
    Save a new model version to the database.
    
    Args:
        model_name: Unique name for the model version
        gender: 'male' or 'female'
        format_type: 'T20' or 'ODI'
        model_path: Path to .keras model file
        normalizer_path: Path to normalizer .pkl file
        data_earliest_date: Earliest match date in training data
        data_latest_date: Latest match date in training data
        training_samples: Number of samples (deliveries) used
        training_duration_seconds: Training duration in seconds
        model_size_mb: Model file size in MB
        accuracy_metrics: JSON string with accuracy metrics
        is_active: Whether this is the currently active model
        notes: Optional notes
        db_path: Optional path to database file
        
    Returns:
        Model version ID
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # If setting this as active, deactivate others for the same gender/format
        if is_active:
            cursor.execute(
                "UPDATE model_versions SET is_active = 0 WHERE gender = ? AND format_type = ?",
                (gender, format_type)
            )
        
        cursor.execute(
            """INSERT INTO model_versions (
                model_name, gender, format_type, model_path, normalizer_path,
                data_earliest_date, data_latest_date, training_samples,
                training_duration_seconds, model_size_mb, accuracy_metrics,
                is_active, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                model_name, gender, format_type, model_path, normalizer_path,
                data_earliest_date, data_latest_date, training_samples,
                training_duration_seconds, model_size_mb, accuracy_metrics,
                is_active, notes
            )
        )
        
        return cursor.lastrowid


def set_active_model(model_id: int, db_path: Optional[Path] = None) -> bool:
    """
    Set a specific model version as active.
    
    Args:
        model_id: ID of the model to activate
        db_path: Optional path to database file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            
            # Get the model's gender and format
            cursor.execute(
                "SELECT gender, format_type FROM model_versions WHERE id = ?",
                (model_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                logger.error(f"Model version {model_id} not found")
                return False
            
            gender, format_type = row['gender'], row['format_type']
            
            # Deactivate all models for this gender/format
            cursor.execute(
                "UPDATE model_versions SET is_active = 0 WHERE gender = ? AND format_type = ?",
                (gender, format_type)
            )
            
            # Activate the specified model
            cursor.execute(
                "UPDATE model_versions SET is_active = 1 WHERE id = ?",
                (model_id,)
            )
            
            logger.info(f"Set model {model_id} as active for {gender} {format_type}")
            return True
            
    except Exception as e:
        logger.error(f"Failed to set active model: {e}")
        return False


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

