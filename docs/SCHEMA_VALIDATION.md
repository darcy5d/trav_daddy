# Database Schema Validation

This document describes the schema validation system that ensures your database has all required tables, columns, and data before model training.

## Overview

The schema validation system checks:
- ✅ All required tables exist
- ✅ All required columns exist in each table
- ✅ Required indexes exist (for performance)
- ✅ Required views exist (for queries)
- ✅ Tournament tier patterns are populated
- ✅ Teams have tier assignments

## Running Validation

### Automatic Validation

Schema validation runs **automatically** before every training session:
- When you click "Full Retrain" in the GUI
- When you run `scripts/full_retrain.py`

If validation fails, training will abort with a clear error message.

### Manual Validation

You can manually check the schema anytime:

```bash
# Option 1: Python script
python scripts/validate_schema.py

# Option 2: Shell script
./scripts/check_schema.sh
```

## What Gets Validated

### Core Tables
- `teams` - Team information with tier classifications
- `players` - Player registry
- `venues` - Match venues
- `matches` - Match metadata
- `innings` - Innings data
- `deliveries` - Ball-by-ball data
- `player_match_stats` - Aggregated player statistics

### ELO System Tables (V2 Schema)
- `team_elo_history` - Team ELO over time (format+gender separated)
- `player_elo_history` - Player ELO over time (format+gender separated)
- `team_current_elo` - Current team ELO ratings
- `player_current_elo` - Current player ELO ratings

### Tiered ELO System Tables (V3)
- `tournament_tiers` - Tournament tier mappings (~45 patterns)
- `promotion_review_flags` - Teams flagged for tier review

### Supporting Tables
- `model_versions` - ML model version tracking

### Performance Indexes
- Team, player, and match lookup indexes
- ELO history indexes for fast queries
- Tier classification indexes

### Views
- Match summary views
- Team ranking views (by format and gender)
- Player ranking views (batting and bowling)

## Common Issues and Fixes

### Issue: Missing Table or Column

**Cause:** Database was created with an old schema version

**Fix:**
```bash
# Option 1: Reset database in GUI
# Go to Training tab → "Reset Database" → "Full Retrain"

# Option 2: Run migrations manually
python scripts/migrate_add_tier_columns.py
python scripts/migrate_elo_tables_to_v2.py
```

### Issue: Empty tournament_tiers Table

**Cause:** Tournament patterns weren't populated during database creation

**Fix:**
```bash
python scripts/migrate_add_tier_columns.py
```

### Issue: Teams Have NULL Tier Values

**Cause:** Team tier classifications haven't been applied

**Fix:**
```bash
# Apply tier classifications
python -c "
from src.data.database import get_connection
from config import DATABASE_PATH
from pathlib import Path

conn = get_connection(DATABASE_PATH)
sql = Path('src/data/team_tier_classifications.sql').read_text()
conn.executescript(sql)
conn.commit()
conn.close()
print('✅ Team tiers applied')
"
```

### Issue: Missing Indexes or Views

**Cause:** These are warnings, not critical errors

**Impact:** Queries may be slower but will still work

**Fix:** Not urgent, but can recreate database from updated schema if desired

## Schema Versions

### V1 (Legacy)
- Single ELO rating per team (no gender separation)
- No tier system

### V2 (Current)
- ELO ratings separated by format AND gender
- Tables: `team_elo_history`, `player_elo_history` with format/gender columns
- Views for each format/gender combination

### V3 (Tiered System)
- Builds on V2
- Adds tier classifications to teams (1-5)
- Adds tournament tier mappings
- Adds promotion review system
- Cross-tier ELO normalization

## Testing After Schema Changes

After making schema changes, always:

1. **Validate the schema:**
   ```bash
   python scripts/validate_schema.py
   ```

2. **Test a training run:**
   ```bash
   python scripts/full_retrain.py --male-only
   ```

3. **Check the GUI:**
   - Visit http://127.0.0.1:5001/training
   - Click "Check Database Status"
   - Verify all tables show correct row counts

## Continuous Integration

For CI/CD pipelines, add schema validation as a required check:

```bash
#!/bin/bash
set -e

# Validate schema
python scripts/validate_schema.py

# If validation passes, proceed with tests
python -m pytest tests/
```

## Troubleshooting

### Validation Hangs or Times Out

**Possible causes:**
- Database is locked by another process
- Database file is corrupted
- Disk I/O issues

**Fix:**
1. Stop the Flask server if running
2. Close any DB browser applications
3. Try validation again
4. If still failing, restore from backup

### Validation Passes But Training Still Fails

**Possible causes:**
- Data integrity issues (not schema issues)
- Logic bugs in training code
- Resource constraints (memory, disk space)

**Debug steps:**
1. Check logs for specific error
2. Verify data quality:
   ```bash
   python -c "
   from src.data.database import print_database_summary
   print_database_summary()
   "
   ```
3. Try training with reduced dataset (`--male-only`)

## Maintenance

### Keeping Schema Up to Date

When adding new features:

1. Update `src/data/schema.sql` with new tables/columns
2. Update `scripts/validate_schema.py` with new requirements
3. Create migration script if needed (e.g., `migrate_*.py`)
4. Update this documentation

### Schema Evolution Strategy

- **Backward compatible changes:** Add new columns with defaults
- **Breaking changes:** Create new migration script and version
- **Testing:** Always validate schema before committing changes

## See Also

- [Database Schema](DATABASE_SCHEMA.md) - Full schema documentation
- [ELO System](ELO_SYSTEM_EXPLAINED.md) - ELO calculation details
- [Validation Report](VALIDATION_REPORT.md) - Model validation methodology
