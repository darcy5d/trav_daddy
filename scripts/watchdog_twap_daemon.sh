#!/usr/bin/env bash
# Watchdog that restarts the TWAP execution daemon if it has stopped.
#
# Intended to run every few minutes from cron. Idempotent. Cheap.
#
# Decision order (must be very conservative — false positives spawn a
# duplicate daemon, which causes 403s when both clients hit the CLOB):
#
#   1. If the PID file has a PID that is alive -> health-gate it. If the
#      daemon's heartbeat reports the DB is reachable, exit (fine). If the DB
#      is wedged (disk I/O error), kill it and restart (subject to anti-flap).
#   2. Otherwise, if `pgrep` finds a running paper_bet_auto_post_toss.py
#      process, write its PID to the PID file and health-gate it as above.
#   3. Otherwise, start a fresh daemon.
#
# Usage:
#     scripts/watchdog_twap_daemon.sh

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO_ROOT}/venv311/bin/python"
SCRIPT="${REPO_ROOT}/scripts/paper_bet_auto_post_toss.py"
LOG="${REPO_ROOT}/logs/paper_auto_post_toss.log"
WATCHDOG_LOG="${REPO_ROOT}/logs/watchdog_twap.log"
PID_FILE="${REPO_ROOT}/logs/paper_auto_post_toss.pid"
STATUS_FILE="${REPO_ROOT}/logs/paper_auto_post_toss_status.json"

# DB-health hardening (Wave 6): detect a wedged-but-alive daemon (process-level
# SQLite "disk I/O error" state) and restart it, not just a dead process.
HEALTH_MARKER="${REPO_ROOT}/logs/daemon_health_restart.ts"
HEALTH_STALE_MIN=6               # heartbeat older than this => daemon hung
HEALTH_LOG_WINDOW_MIN=4          # log-scan lookback for backup signal
HEALTH_COOLDOWN_MIN=10           # anti-flap: min gap between health restarts

mkdir -p "${REPO_ROOT}/logs"

ts() { date -u "+%Y-%m-%dT%H:%M:%SZ"; }

is_alive() {
    # Returns 0 if the given pid is alive and belongs to the daemon script.
    local pid="$1"
    [[ -z "${pid}" ]] && return 1
    [[ ! "${pid}" =~ ^[0-9]+$ ]] && return 1
    if ! kill -0 "${pid}" 2>/dev/null; then
        return 1
    fi
    # Confirm the cmdline matches — defends against pid recycling onto an
    # unrelated process. macOS ps output is `command` (full argv).
    if ps -p "${pid}" -o command= 2>/dev/null | grep -q "paper_bet_auto_post_toss.py"; then
        return 0
    fi
    return 1
}

heartbeat_issue() {
    # Reads the daemon's heartbeat status JSON and prints a human reason string
    # if the daemon is DB-unhealthy (db_ok=false) or its heartbeat is stale
    # (hung / not writing status). Prints nothing when healthy or when there is
    # not yet enough info (e.g. freshly started, no status file).
    "${PYTHON}" - "${STATUS_FILE}" "${HEALTH_STALE_MIN}" <<'PYEOF' 2>/dev/null
import sys, os, json, datetime
path, stale_min = sys.argv[1], int(sys.argv[2])
if not os.path.exists(path):
    sys.exit(0)  # no heartbeat yet -> not enough info to call unhealthy
try:
    with open(path) as f:
        d = json.load(f)
except Exception:
    sys.exit(0)
if d.get("db_ok") is False:
    print("db_ok=false: " + str(d.get("db_error"))[:180])
    sys.exit(0)
w = d.get("written_at_utc")
if w:
    try:
        t = datetime.datetime.fromisoformat(w)
        if t.tzinfo is None:
            t = t.replace(tzinfo=datetime.timezone.utc)
        age = (datetime.datetime.now(datetime.timezone.utc) - t).total_seconds()
        if age > stale_min * 60:
            print(f"heartbeat stale ({int(age)}s old)")
    except Exception:
        pass
sys.exit(0)
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

health_gate() {
    # Called with a known-alive daemon PID. If the daemon is DB-unhealthy,
    # kill it (subject to the anti-flap cooldown) so the start-fresh block
    # below relaunches it. Exits 0 directly when healthy.
    local pid="$1" reason=""
    reason="$(heartbeat_issue)"
    if [[ -z "${reason}" ]] && recent_db_errors; then
        reason="recent disk I/O errors in paper_auto_post_toss.log"
    fi

    if [[ -z "${reason}" ]]; then
        exit 0  # healthy
    fi

    if ! health_restart_allowed; then
        echo "[$(ts)] WARN daemon DB-unhealthy (${reason}) but within ${HEALTH_COOLDOWN_MIN}m anti-flap window; not restarting" >> "${WATCHDOG_LOG}"
        exit 0
    fi

    echo "[$(ts)] daemon DB-unhealthy (${reason}); killing pid=${pid} for restart" >> "${WATCHDOG_LOG}"
    mark_health_restart
    kill "${pid}" 2>/dev/null || true
    sleep 2
    pkill -f "paper_bet_auto_post_toss.py" 2>/dev/null || true
    rm -f "${PID_FILE}"
    # Return so the caller falls through to the start-fresh block.
}

# 1. Trust the PID file first. If alive, health-gate before declaring it fine:
#    a wedged daemon keeps running (and writing its heartbeat) but every DB op
#    fails with "disk I/O error", so liveness alone is not enough.
if [[ -f "${PID_FILE}" ]]; then
    PID="$(cat "${PID_FILE}" 2>/dev/null)"
    if is_alive "${PID}"; then
        health_gate "${PID}"  # exits 0 if healthy; else kills + falls through
    fi
fi

# 2. Hunt for an existing process even if the PID file is missing or stale.
EXISTING_PID="$(pgrep -f "paper_bet_auto_post_toss.py" 2>/dev/null | head -1 || true)"
if [[ -n "${EXISTING_PID}" ]] && is_alive "${EXISTING_PID}"; then
    echo "${EXISTING_PID}" > "${PID_FILE}"
    echo "[$(ts)] reclaimed PID file for existing daemon pid=${EXISTING_PID}" >> "${WATCHDOG_LOG}"
    health_gate "${EXISTING_PID}"  # exits 0 if healthy; else kills + falls through
fi

# 3. Nothing alive: start a fresh daemon. Clean stale PID file first.
rm -f "${PID_FILE}"
echo "[$(ts)] daemon not running, starting..." >> "${WATCHDOG_LOG}"

cd "${REPO_ROOT}" || {
    echo "[$(ts)] cd failed: ${REPO_ROOT}" >> "${WATCHDOG_LOG}"
    exit 1
}

AUTO_INSTALL_PLAYWRIGHT_CHROMIUM=false \
nohup "${PYTHON}" "${SCRIPT}" \
  --poll-interval 90 --lookback-min 45 --lookahead-min 30 --also-live \
  >> "${LOG}" 2>&1 < /dev/null &
disown 2>/dev/null || true

sleep 3

NEW_PID="$(pgrep -f "paper_bet_auto_post_toss.py" 2>/dev/null | head -1 || true)"
if [[ -n "${NEW_PID}" ]] && is_alive "${NEW_PID}"; then
    echo "[$(ts)] daemon restarted (pid=${NEW_PID})" >> "${WATCHDOG_LOG}"
else
    echo "[$(ts)] WARN daemon did not stay up after spawn" >> "${WATCHDOG_LOG}"
fi
