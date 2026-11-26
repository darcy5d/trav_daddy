#!/usr/bin/env bash
# Start/stop a side-by-side scratch Flask instance running the code on disk.
#
#   .cursor/skills/verify-indias-dad/scripts/scratch.sh start  [PORT]   # default 5091
#   .cursor/skills/verify-indias-dad/scripts/scratch.sh status [PORT]
#   .cursor/skills/verify-indias-dad/scripts/scratch.sh stop   [PORT]
#
# The scratch instance:
#   - never uses port 5001 (the live dashboard, supervised by cron watchdog);
#   - is launched as `python -m flask run` with FLASK_APP set, so it does NOT
#     match the watchdog's `pkill -f 'flask --app app.main'` pattern;
#   - has every wallet/CLOB credential blanked and BETTING_MODE=OFF,
#     BETTING_KILL_SWITCH=1, so it cannot sign or place orders;
#   - still shares cricket.db / betting tables with live (process isolation,
#     not data isolation) — drive it with drive.py, which blocks write POSTs.
# stop only kills the PID this script recorded, after checking its command line.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON="${REPO_ROOT}/venv311/bin/python"
STATE_DIR="${REPO_ROOT}/tmp/verify-indias-dad"
EVIDENCE_DIR="${REPO_ROOT}/logs/verify"
ACTION="${1:-status}"
PORT="${2:-5091}"
PID_FILE="${STATE_DIR}/scratch-${PORT}.pid"

if [[ "${PORT}" == "5001" ]]; then
    echo "refusing: 5001 is the live instance" >&2
    exit 2
fi

port_busy() { lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN -t >/dev/null 2>&1; }
our_pid() {
    [[ -f "${PID_FILE}" ]] || return 1
    local pid; pid="$(cat "${PID_FILE}")"
    ps -o command= -p "${pid}" 2>/dev/null | grep -q "flask run --port ${PORT}" || return 1
    echo "${pid}"
}

case "${ACTION}" in
start)
    if pid="$(our_pid)"; then echo "already running: pid ${pid} on :${PORT}"; exit 0; fi
    if port_busy; then echo "port ${PORT} is in use by something else; pick another port" >&2; exit 1; fi
    mkdir -p "${STATE_DIR}" "${EVIDENCE_DIR}"
    LOG="${EVIDENCE_DIR}/scratch-${PORT}-$(date +%Y%m%d-%H%M%S).log"
    cd "${REPO_ROOT}" || exit 1
    # New session so agent shells that reap their process group on exit don't kill it.
    env FLASK_APP=app.main \
        BETTING_MODE=OFF BETTING_KILL_SWITCH=1 \
        POLYGON_PRIVATE_KEY= POLYMARKET_PRIVATE_KEY= \
        POLYMARKET_API_KEY= POLYMARKET_API_SECRET= POLYMARKET_PASSPHRASE= \
        AUTO_INSTALL_PLAYWRIGHT_CHROMIUM=false PYTHONDONTWRITEBYTECODE=1 \
        "${PYTHON}" - "${PYTHON}" "${PORT}" "${LOG}" "${PID_FILE}" <<'PYEOF'
import subprocess, sys
py, port, log, pid_file = sys.argv[1:5]
with open(log, "ab") as out:
    p = subprocess.Popen([py, "-m", "flask", "run", "--port", port, "--host", "127.0.0.1"],
                         stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                         start_new_session=True)
open(pid_file, "w").write(str(p.pid))
PYEOF
    echo "started pid $(cat "${PID_FILE}") on :${PORT}, log ${LOG}; waiting for /healthz (TensorFlow import is slow)..."
    for _ in $(seq 1 90); do
        if curl -sf -m 3 "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1; then
            echo "READY http://127.0.0.1:${PORT}"
            exit 0
        fi
        our_pid >/dev/null || { echo "scratch process died; see ${LOG}" >&2; rm -f "${PID_FILE}"; exit 1; }
        sleep 2
    done
    echo "not ready after 180s; see ${LOG} (still running, stop it with: $0 stop ${PORT})" >&2
    exit 1
    ;;
status)
    if pid="$(our_pid)"; then
        echo "running: pid ${pid} on :${PORT}"
        curl -s -m 5 "http://127.0.0.1:${PORT}/healthz"; echo
    else
        echo "not running on :${PORT}"
    fi
    ;;
stop)
    if ! pid="$(our_pid)"; then
        echo "no scratch instance recorded for :${PORT}"
        rm -f "${PID_FILE}"
        exit 0
    fi
    kill "${pid}"
    for _ in $(seq 1 10); do kill -0 "${pid}" 2>/dev/null || break; sleep 1; done
    kill -0 "${pid}" 2>/dev/null && kill -9 "${pid}"
    rm -f "${PID_FILE}"
    rmdir "${STATE_DIR}" 2>/dev/null || true
    echo "stopped pid ${pid}; evidence kept in ${EVIDENCE_DIR}"
    ;;
*)
    echo "usage: $0 start|status|stop [PORT]" >&2
    exit 2
    ;;
esac
