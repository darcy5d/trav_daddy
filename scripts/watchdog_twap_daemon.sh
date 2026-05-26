#!/usr/bin/env bash
# Watchdog that restarts the TWAP execution daemon if it has stopped.
#
# Intended to run every few minutes from cron. Idempotent. Cheap.
#
# Decision order (must be very conservative — false positives spawn a
# duplicate daemon, which causes 403s when both clients hit the CLOB):
#
#   1. If the PID file has a PID that is alive -> exit (daemon is fine).
#   2. Otherwise, if `pgrep` finds a running paper_bet_auto_post_toss.py
#      process, write its PID to the PID file and exit. This recovers
#      from cases where the PID file was lost but the daemon is alive.
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

# 1. Trust the PID file first.
if [[ -f "${PID_FILE}" ]]; then
    PID="$(cat "${PID_FILE}" 2>/dev/null)"
    if is_alive "${PID}"; then
        exit 0
    fi
fi

# 2. Hunt for an existing process even if the PID file is missing or stale.
EXISTING_PID="$(pgrep -f "paper_bet_auto_post_toss.py" 2>/dev/null | head -1 || true)"
if [[ -n "${EXISTING_PID}" ]] && is_alive "${EXISTING_PID}"; then
    echo "${EXISTING_PID}" > "${PID_FILE}"
    echo "[$(ts)] reclaimed PID file for existing daemon pid=${EXISTING_PID}" >> "${WATCHDOG_LOG}"
    exit 0
fi

# 3. Nothing alive: start a fresh daemon. Clean stale PID file first.
rm -f "${PID_FILE}"
echo "[$(ts)] daemon not running, starting..." >> "${WATCHDOG_LOG}"

cd "${REPO_ROOT}" || {
    echo "[$(ts)] cd failed: ${REPO_ROOT}" >> "${WATCHDOG_LOG}"
    exit 1
}

AUTO_INSTALL_PLAYWRIGHT_CHROMIUM=false \
nohup "${PYTHON}" "${SCRIPT}" >> "${LOG}" 2>&1 < /dev/null &
disown 2>/dev/null || true

sleep 3

NEW_PID="$(pgrep -f "paper_bet_auto_post_toss.py" 2>/dev/null | head -1 || true)"
if [[ -n "${NEW_PID}" ]] && is_alive "${NEW_PID}"; then
    echo "[$(ts)] daemon restarted (pid=${NEW_PID})" >> "${WATCHDOG_LOG}"
else
    echo "[$(ts)] WARN daemon did not stay up after spawn" >> "${WATCHDOG_LOG}"
fi
