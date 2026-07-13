#!/usr/bin/env bash
# Runs every 4 hours: prints discovery progress + Thread A AUC sweep.
# Designed to run detached; output goes to logs/ca_reporter.log
set -euo pipefail

PROJ=/Users/darcy5d/Desktop/DD_AI_models/indias_dad
VENV="$PROJ/venv311/bin/python"
LOG="$PROJ/logs/ca_reporter.log"
DISC_PID_FILE="$PROJ/logs/ca_discover_associates.pid"
DISC_LOG="$PROJ/logs/ca_discover_associates.log"
DISC_PROGRESS="$PROJ/data/raw/cricketarchive/all_t20_discovery_progress.json"
DISC_OUT="$PROJ/data/raw/cricketarchive/all_t20_associates.txt"

INTERVAL_SEC=$((4 * 3600))   # 4 hours

log() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "$LOG"; }

separator() {
    echo "" | tee -a "$LOG"
    echo "════════════════════════════════════════════════════════════════" | tee -a "$LOG"
    echo "  UPDATE  $(date '+%A %d %b %Y  %H:%M %Z')" | tee -a "$LOG"
    echo "════════════════════════════════════════════════════════════════" | tee -a "$LOG"
}

report_discovery() {
    log "--- DISCOVERY STATUS ---"
    for i in 1 2 3; do
        PF="$PROJ/logs/ca_discover_w${i}.pid"
        if [ -f "$PF" ]; then
            W_PID=$(cat "$PF")
            kill -0 "$W_PID" 2>/dev/null && STATUS="ALIVE" || STATUS="DONE/DEAD"
            LAST=$(tail -1 "$PROJ/logs/ca_discover_w${i}.log" 2>/dev/null || echo "no log")
            log "  Worker $i PID $W_PID: $STATUS | $LAST"
        fi
    done
    # Harvest watcher
    if [ -f "$PROJ/logs/ca_rolling_harvest.pid" ]; then
        H_PID=$(cat "$PROJ/logs/ca_rolling_harvest.pid")
        kill -0 "$H_PID" 2>/dev/null && log "  Harvester PID $H_PID: ALIVE" || log "  Harvester: DEAD"
    fi

    "$VENV" - <<'PYEOF' 2>&1 | tee -a "$LOG"
import json, sys
from pathlib import Path
proj = Path('/Users/darcy5d/Desktop/DD_AI_models/indias_dad')
cache = proj / 'data/raw/cricketarchive'

# Collect all discovered URLs across all worker files
all_urls = set()
for f in cache.glob('all_t20_w*.txt'):
    all_urls.update(l.strip() for l in f.read_text().splitlines() if l.strip())
# Also include the original associates file
af = cache / 'all_t20_associates.txt'
if af.exists():
    all_urls.update(l.strip() for l in af.read_text().splitlines() if l.strip())

# Count what's already in DB
import sqlite3
db = proj / 'data/processed/ca_archive.db'
in_db = 0
if db.exists():
    conn = sqlite3.connect(str(db))
    in_db = conn.execute("SELECT COUNT(*) FROM ca_matches").fetchone()[0]
    conn.close()

print(f"  Total discovered URLs : {len(all_urls)}")
print(f"  Matches in DB         : {in_db}")
print(f"  Pending harvest       : {len(all_urls) - in_db} (approx)")

# Per-worker progress
for i in range(1, 4):
    pf = cache / f'all_t20_w{i}.txt'
    n = sum(1 for l in pf.read_text().splitlines() if l.strip()) if pf.exists() else 0
    print(f"  Worker {i} output      : {n} URLs")
PYEOF
}

report_auc() {
    log "--- THREAD A AUC SWEEP ---"
    cd "$PROJ"
    "$VENV" scripts/ca_thread_a_signal.py --sweep 2>&1 | tee -a "$LOG"
}

# ── Main loop ──────────────────────────────────────────────────────────────
log "Reporter started. Will report every 4 hours. CTRL-C to stop."

while true; do
    separator
    report_discovery
    echo "" | tee -a "$LOG"
    report_auc
    echo "" | tee -a "$LOG"
    log "Next update in 4 hours ($(date -v+4H '+%H:%M' 2>/dev/null || date -d '+4 hours' '+%H:%M'))."
    sleep "$INTERVAL_SEC"
done
