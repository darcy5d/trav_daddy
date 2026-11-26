#!/usr/bin/env bash
# Rolling harvest: every 10 min, check all_t20_associates*.txt for URLs not yet
# in ca_archive.db and harvest any batch of >=20 new ones.
set -euo pipefail

PROJ=/Users/darcy5d/Desktop/DD_AI_models/indias_dad
VENV="$PROJ/venv311/bin/python"
DB="$PROJ/ca_archive.db"
PENDING="$PROJ/data/raw/cricketarchive/rolling_harvest_pending.txt"
LOG="$PROJ/logs/ca_rolling_harvest.log"
CHECK_INTERVAL=600   # 10 minutes
MIN_BATCH=20         # harvest when at least this many new URLs found

log() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "$LOG"; }

log "Rolling harvest watcher started."

while true; do
    sleep "$CHECK_INTERVAL"

    # Collect all discovered URLs from all worker output files
    DISC_URLS=$(cat "$PROJ"/data/raw/cricketarchive/all_t20_w*.txt \
                    "$PROJ/data/raw/cricketarchive/all_t20_associates.txt" \
                2>/dev/null | sort -u)

    TOTAL=$(echo "$DISC_URLS" | grep -c '/Archive/Scorecards/' 2>/dev/null || echo 0)

    # Find which are already in the DB
    NEW_URLS=$("$VENV" - <<PYEOF
import sqlite3, re, sys
from pathlib import Path

discovered = set()
for line in """$DISC_URLS""".splitlines():
    line = line.strip()
    if '/Archive/Scorecards/' in line:
        discovered.add(line)

db = Path("$PROJ/ca_archive.db")
if not db.exists():
    print('\n'.join(discovered))
    sys.exit(0)

conn = sqlite3.connect(str(db))
existing = set(r[0] for r in conn.execute("SELECT source_url FROM ca_matches").fetchall())
conn.close()

new = discovered - existing
for u in sorted(new):
    print(u)
PYEOF
)

    N_NEW=$(echo "$NEW_URLS" | grep '/Archive/Scorecards/' 2>/dev/null | wc -l | tr -d ' ')
    N_NEW=${N_NEW:-0}
    log "Discovered: $TOTAL urls total | $N_NEW not yet in DB"

    if [ "$N_NEW" -ge "$MIN_BATCH" ]; then
        log "Harvesting $N_NEW new scorecards..."
        echo "$NEW_URLS" | grep '/Archive/Scorecards/' > "$PENDING"
        cd "$PROJ"
        "$VENV" scripts/ca_harvest_parallel.py --list "$PENDING" 2>&1 | tee -a "$LOG"
        log "Harvest batch complete."
    else
        log "Below threshold ($MIN_BATCH), waiting..."
    fi
done
