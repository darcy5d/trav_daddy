#!/usr/bin/env bash
# Watchdog that restarts the Flask dashboard/API (port 5001) if it has stopped.
#
# Intended to run every few minutes from cron. Idempotent. Cheap. Mirrors
# scripts/watchdog_twap_daemon.sh so "everything" self-heals, not just the
# TWAP daemon. Flask has no other supervisor, so without this it stays down
# after a crash/reboot until manually restarted.
#
# Decision order (conservative — a duplicate Flask just fails to bind 5001):
#   1. If something is already serving on the port -> exit (Flask is fine).
#   2. Otherwise, start a fresh Flask process.
#
# Usage:
#     scripts/watchdog_flask.sh

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO_ROOT}/venv311/bin/python"
LOG="${REPO_ROOT}/logs/flask.log"
WATCHDOG_LOG="${REPO_ROOT}/logs/watchdog_flask.log"
PID_FILE="${REPO_ROOT}/logs/flask.pid"
PORT="${FLASK_PORT:-5001}"

mkdir -p "${REPO_ROOT}/logs"

ts() { date -u "+%Y-%m-%dT%H:%M:%SZ"; }

port_alive() {
    # Returns 0 if something is accepting connections on the Flask port.
    #
    # Uses a TCP socket probe rather than `lsof`/`pgrep`: under cron/launchd
    # (and in sandboxes) macOS can deny process/port enumeration ("sysmond
    # service not found" / "Cannot get process list"), which made the old
    # lsof check a false-negative and caused the watchdog to spawn duplicate
    # Flasks that then failed to bind. A socket connect needs no such perms.
    "${PYTHON}" - "${PORT}" <<'PYEOF' >/dev/null 2>&1
import socket, sys
port = int(sys.argv[1])
s = socket.socket(); s.settimeout(2)
sys.exit(0 if s.connect_ex(("127.0.0.1", port)) == 0 else 1)
PYEOF
}

# 1. If the port is already served, Flask (or something) is up — leave it.
if port_alive; then
    NEW_PID="$(pgrep -f 'flask --app app.main' 2>/dev/null | head -1 || true)"
    [[ -n "${NEW_PID}" ]] && echo "${NEW_PID}" > "${PID_FILE}"
    exit 0
fi

# 2. Nothing listening: start a fresh Flask. Clean stale PID file first.
rm -f "${PID_FILE}"
echo "[$(ts)] flask not running, starting on :${PORT}..." >> "${WATCHDOG_LOG}"

cd "${REPO_ROOT}" || {
    echo "[$(ts)] cd failed: ${REPO_ROOT}" >> "${WATCHDOG_LOG}"
    exit 1
}

FLASK_PORT="${PORT}" AUTO_INSTALL_PLAYWRIGHT_CHROMIUM=false \
nohup "${PYTHON}" -m flask --app app.main run \
  --port "${PORT}" --host 127.0.0.1 \
  >> "${LOG}" 2>&1 < /dev/null &
disown 2>/dev/null || true

# Flask import is heavy (models, TF, etc.); give it ample time before the
# health check or we log a false "did not come up" while it is still importing.
sleep 30

if port_alive; then
    NEW_PID="$(pgrep -f 'flask --app app.main' 2>/dev/null | head -1 || true)"
    [[ -n "${NEW_PID}" ]] && echo "${NEW_PID}" > "${PID_FILE}"
    echo "[$(ts)] flask restarted (pid=${NEW_PID:-unknown})" >> "${WATCHDOG_LOG}"
else
    echo "[$(ts)] WARN flask did not come up on :${PORT} after spawn" >> "${WATCHDOG_LOG}"
fi
