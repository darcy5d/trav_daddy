#!/usr/bin/env bash
# Wave 5.7: install a cron job that runs the paper-bet scan + reconcile
# every hour. Idempotent - running this twice won't add duplicate entries.
#
# Usage:
#     bash scripts/install_paper_cron.sh
#     bash scripts/install_paper_cron.sh --uninstall

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO_ROOT}/venv311/bin/python"
SCRIPT="${REPO_ROOT}/scripts/paper_bet_daily.py"
LOGFILE="${REPO_ROOT}/logs/paper_daily.log"
CRON_TAG="# paper-trade scan (Wave 5.7)"
CRON_LINE="0 * * * * cd ${REPO_ROOT} && AUTO_INSTALL_PLAYWRIGHT_CHROMIUM=false ${PYTHON} ${SCRIPT} --hours-ahead 96 >> ${LOGFILE} 2>&1 ${CRON_TAG}"

mkdir -p "${REPO_ROOT}/logs"

# Pull the existing crontab (or empty if none); strip any prior line we added
existing="$(crontab -l 2>/dev/null || true)"
filtered="$(echo "${existing}" | grep -vF "${CRON_TAG}" || true)"

if [[ "${1:-}" == "--uninstall" ]]; then
    if [[ -n "${filtered}" ]]; then
        echo "${filtered}" | crontab -
    else
        crontab -r 2>/dev/null || true
    fi
    echo "Removed paper-trade cron entry."
    exit 0
fi

new_crontab="${filtered}
${CRON_LINE}"
echo "${new_crontab}" | crontab -

echo "Installed paper-trade cron entry:"
echo "    ${CRON_LINE}"
echo
echo "Will run every hour at :00. First run starts within the hour."
echo "Logs: ${LOGFILE}"
echo
echo "To verify:    crontab -l | grep 'paper-trade'"
echo "To uninstall: bash scripts/install_paper_cron.sh --uninstall"
