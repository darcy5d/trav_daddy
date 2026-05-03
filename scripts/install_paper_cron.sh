#!/usr/bin/env bash
# Wave 5.7 + follow-up: install two cron jobs.
#   1. Hourly paper-bet scan + reconcile (paper_bet_daily.py)
#   2. Daily Cricsheet data refresh + ELO recalc (daily_data_refresh.py)
#       at 11:00 UTC, AFTER most matches end and BEFORE the next-day
#       Polymarket window opens.
# Both entries are idempotent - running this script twice does not add
# duplicates. --uninstall removes both.
#
# Usage:
#     bash scripts/install_paper_cron.sh
#     bash scripts/install_paper_cron.sh --uninstall

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO_ROOT}/venv311/bin/python"
ENV_PREFIX="AUTO_INSTALL_PLAYWRIGHT_CHROMIUM=false"

# Entry 1: hourly paper-bet scan
PAPER_SCRIPT="${REPO_ROOT}/scripts/paper_bet_daily.py"
PAPER_LOG="${REPO_ROOT}/logs/paper_daily.log"
PAPER_TAG="# paper-trade scan (Wave 5.7)"
PAPER_LINE="0 * * * * cd ${REPO_ROOT} && ${ENV_PREFIX} ${PYTHON} ${PAPER_SCRIPT} --hours-ahead 96 >> ${PAPER_LOG} 2>&1 ${PAPER_TAG}"

# Entry 2: daily data refresh
REFRESH_SCRIPT="${REPO_ROOT}/scripts/daily_data_refresh.py"
REFRESH_LOG="${REPO_ROOT}/logs/daily_data_refresh.log"
REFRESH_TAG="# daily data refresh (Wave 5.7 follow-up)"
REFRESH_LINE="0 11 * * * cd ${REPO_ROOT} && ${ENV_PREFIX} ${PYTHON} ${REFRESH_SCRIPT} >> ${REFRESH_LOG} 2>&1 ${REFRESH_TAG}"

mkdir -p "${REPO_ROOT}/logs"

# Pull the existing crontab (or empty if none); strip any prior lines we added
existing="$(crontab -l 2>/dev/null || true)"
filtered="$(echo "${existing}" | grep -vF "${PAPER_TAG}" | grep -vF "${REFRESH_TAG}" || true)"

if [[ "${1:-}" == "--uninstall" ]]; then
    if [[ -n "${filtered}" ]]; then
        echo "${filtered}" | crontab -
    else
        crontab -r 2>/dev/null || true
    fi
    echo "Removed paper-trade and daily-refresh cron entries."
    exit 0
fi

new_crontab="${filtered}
${PAPER_LINE}
${REFRESH_LINE}"
echo "${new_crontab}" | crontab -

echo "Installed cron entries:"
echo "    ${PAPER_LINE}"
echo "    ${REFRESH_LINE}"
echo
echo "Schedule:"
echo "  - paper-bet scan: every hour at :00 (next run within the hour)"
echo "  - data refresh:   daily at 11:00 UTC"
echo
echo "Logs: ${PAPER_LOG}"
echo "      ${REFRESH_LOG}"
echo
echo "To verify:    crontab -l | grep -E '(paper-trade|daily data refresh)'"
echo "To uninstall: bash scripts/install_paper_cron.sh --uninstall"
