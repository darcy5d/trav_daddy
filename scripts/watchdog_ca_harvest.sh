#!/bin/bash
# Watchdog for ca_harvest_parallel (Tranche 3).
# Runs every 5 min via cron. Restarts the harvest if it is not running.
# Self-terminates when the events file is exhausted (harvest exits clean).
#
# Cron entry (added by agent):
#   */5 * * * * cd /Users/darcy5d/Desktop/DD_AI_models/indias_dad && bash scripts/watchdog_ca_harvest.sh >> logs/ca_harvest_watchdog.log 2>&1  # CA harvest watchdog (Tranche 3)

cd /Users/darcy5d/Desktop/DD_AI_models/indias_dad

EVENTS=data/raw/cricketarchive/tranche3_events.txt
LOG=logs/ca_harvest_t3.log
PID_FILE=logs/ca_harvest_t3.pid
PYTHON=venv311/bin/python

# If already running, do nothing.
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "[$(date)] watchdog: harvest running (pid $PID) — OK"
        exit 0
    fi
    echo "[$(date)] watchdog: stale pid $PID — restarting"
fi

# Events file must exist and have content.
if [ ! -s "$EVENTS" ]; then
    echo "[$(date)] watchdog: no events file — nothing to do"
    exit 0
fi

# Launch.
nohup $PYTHON -u scripts/ca_harvest_parallel.py \
    --list "$EVENTS" --workers 3 >> "$LOG" 2>&1 &
echo $! > "$PID_FILE"
echo "[$(date)] watchdog: started harvest pid $(cat $PID_FILE)"
