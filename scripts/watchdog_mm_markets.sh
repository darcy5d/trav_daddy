#!/usr/bin/env bash
# Keep the paper-only order-book snapshot collector healthy.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PID_FILE="$ROOT/logs/scan_mm_markets.pid"
STATUS_FILE="$ROOT/logs/scan_mm_markets_status.json"
LOG_FILE="$ROOT/logs/scan_mm_markets.log"
WATCHDOG_LOG="$ROOT/logs/watchdog_mm_markets.log"
mkdir -p "$ROOT/logs"

is_healthy() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(<"$PID_FILE")"
  kill -0 "$pid" 2>/dev/null || return 1
  [[ -f "$STATUS_FILE" ]] || return 1
  python3 - "$STATUS_FILE" <<'PY'
import json, sys
from datetime import datetime, timezone
try:
    status = json.load(open(sys.argv[1]))
    written = datetime.fromisoformat(status["written_at_utc"].replace("Z", "+00:00"))
    age = (datetime.now(timezone.utc) - written).total_seconds()
    # A sweep walks every live book. 20 minutes covers a slow pass; the
    # collector writes a heartbeat before the first sweep and after each one.
    raise SystemExit(0 if status.get("db_ok") and age < 1200 else 1)
except Exception:
    raise SystemExit(1)
PY
}

if is_healthy; then
  exit 0
fi

if [[ -f "$PID_FILE" ]]; then
  stale_pid="$(<"$PID_FILE")"
  kill "$stale_pid" 2>/dev/null || true
  rm -f "$PID_FILE"
fi

echo "$(date -u +%FT%TZ) restarting MM snapshot collector" >> "$WATCHDOG_LOG"
# System python3 does not have this repo's dependencies. Every other cron
# job uses the Apple Silicon virtualenv; the collector has to as well.
nohup "$ROOT/venv311/bin/python" "$ROOT/scripts/scan_mm_markets.py" \
  --loop --interval-min 2 --hours-ahead 48 --max-stale-hours 6 \
  --markets moneyline >> "$LOG_FILE" 2>&1 &
