#!/usr/bin/env bash
# Watchdog that restarts the Flask dashboard/API (port 5001) if it has stopped.
#
# Intended to run every few minutes from cron. Idempotent. Cheap. Mirrors
# scripts/watchdog_twap_daemon.sh so "everything" self-heals, not just the
# TWAP daemon. Flask has no other supervisor, so without this it stays down
# after a crash/reboot until manually restarted.
#
# Decision order (conservative — a duplicate Flask just fails to bind 5001):
#   1. If the port is served AND /healthz reports the DB is reachable -> exit.
#      If the port is served but the DB is wedged (disk I/O error), kill it and
#      restart (subject to an anti-flap cooldown).
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

# DB-health hardening (Wave 6): detect a wedged-but-alive Flask (process-level
# SQLite "disk I/O error" state) and restart it, not just a dead port.
HEALTH_MARKER="${REPO_ROOT}/logs/flask_health_restart.ts"
HEALTH_LOG_WINDOW_MIN=4          # log-scan lookback for backup signal
HEALTH_COOLDOWN_MIN=10           # anti-flap: min gap between health restarts

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

db_probe() {
    # Probe the in-process DB health endpoint. Prints exactly one token:
    #   healthy     -> /healthz answered db_ok=true (HTTP 200)
    #   unhealthy   -> /healthz answered db_ok=false (HTTP 503) — DB is wedged
    #   unreachable -> no/garbled response (still booting, hung, or too slow)
    "${PYTHON}" - "${PORT}" <<'PYEOF' 2>/dev/null
import sys, json, urllib.request, urllib.error
port = sys.argv[1]
url = f"http://127.0.0.1:{port}/healthz"
try:
    with urllib.request.urlopen(url, timeout=5) as r:
        data = json.loads(r.read().decode("utf-8", "replace"))
        print("healthy" if data.get("db_ok") else "unhealthy")
except urllib.error.HTTPError as e:
    try:
        data = json.loads(e.read().decode("utf-8", "replace"))
        print("unhealthy" if data.get("db_ok") is False else "unreachable")
    except Exception:
        print("unhealthy" if e.code == 503 else "unreachable")
except Exception:
    print("unreachable")
PYEOF
}

recent_db_errors() {
    # Returns 0 if a 'disk I/O error' line appears in LOG within the last
    # HEALTH_LOG_WINDOW_MIN minutes. Time-bounded so old, already-recovered
    # errors near the tail don't trigger a restart. Log timestamps are local
    # ("%Y-%m-%d %H:%M:%S"), matched against local now().
    "${PYTHON}" - "${LOG}" "${HEALTH_LOG_WINDOW_MIN}" <<'PYEOF' >/dev/null 2>&1
import sys, os, re, datetime
path, minutes = sys.argv[1], int(sys.argv[2])
if not os.path.exists(path):
    sys.exit(1)
cutoff = datetime.datetime.now() - datetime.timedelta(minutes=minutes)
with open(path, "rb") as f:
    try:
        f.seek(-2_000_000, os.SEEK_END)   # only scan the last ~2MB
    except OSError:
        f.seek(0)
    lines = f.read().decode("utf-8", "replace").splitlines()
pat = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
for ln in lines:
    if "disk I/O error" not in ln:
        continue
    m = pat.match(ln)
    if not m:
        continue
    try:
        t = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        continue
    if t >= cutoff:
        sys.exit(0)
sys.exit(1)
PYEOF
}

health_restart_allowed() {
    # Anti-flap: allow a health-triggered restart only if the last one was more
    # than HEALTH_COOLDOWN_MIN ago (so a genuinely broken DB doesn't thrash).
    [[ -f "${HEALTH_MARKER}" ]] || return 0
    local last now
    last="$(cat "${HEALTH_MARKER}" 2>/dev/null || echo 0)"
    now="$(date +%s)"
    [[ "${last}" =~ ^[0-9]+$ ]] || return 0
    (( now - last >= HEALTH_COOLDOWN_MIN * 60 ))
}

mark_health_restart() { date +%s > "${HEALTH_MARKER}"; }

# 1. If the port is already served, Flask (or something) is up. Confirm it can
#    actually reach the DB before declaring it healthy; a wedged Flask keeps the
#    port open but every request 500s with "disk I/O error".
if port_alive; then
    NEW_PID="$(pgrep -f 'flask --app app.main' 2>/dev/null | head -1 || true)"
    [[ -n "${NEW_PID}" ]] && echo "${NEW_PID}" > "${PID_FILE}"

    PROBE="$(db_probe)"
    UNHEALTHY=0
    REASON=""
    if [[ "${PROBE}" == "unhealthy" ]]; then
        UNHEALTHY=1
        REASON="/healthz reported db_ok=false"
    elif [[ "${PROBE}" != "healthy" ]] && recent_db_errors; then
        # Probe unreachable/too-slow AND the log shows fresh disk I/O errors:
        # treat as wedged. A bare unreachable with no recent errors is left
        # alone (likely still importing TF on a cold start).
        UNHEALTHY=1
        REASON="/healthz unreachable + recent disk I/O errors in flask.log"
    fi

    if [[ "${UNHEALTHY}" -eq 0 ]]; then
        exit 0
    fi

    if ! health_restart_allowed; then
        echo "[$(ts)] WARN flask DB-unhealthy (${REASON}) but within ${HEALTH_COOLDOWN_MIN}m anti-flap window; not restarting" >> "${WATCHDOG_LOG}"
        exit 0
    fi

    echo "[$(ts)] flask DB-unhealthy (${REASON}); killing for restart" >> "${WATCHDOG_LOG}"
    mark_health_restart
    pkill -f 'flask --app app.main' 2>/dev/null || true
    sleep 3
    # Fall through to the start-fresh block below.
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
