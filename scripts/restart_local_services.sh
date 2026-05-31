#!/usr/bin/env bash
# Restart Flask (5001) + TWAP/auto-toss daemon. Run: bash scripts/restart_local_services.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs

pkill -f 'paper_bet_auto_post_toss.py' 2>/dev/null || true
pkill -f 'flask --app app.main' 2>/dev/null || true
rm -f logs/paper_auto_post_toss.pid
sleep 2

export AUTO_INSTALL_PLAYWRIGHT_CHROMIUM=false
nohup "${REPO_ROOT}/venv311/bin/python" scripts/paper_bet_auto_post_toss.py \
  --poll-interval 90 --lookback-min 45 --lookahead-min 30 --also-live \
  >> logs/paper_auto_post_toss.log 2>&1 &

export FLASK_PORT=5001
nohup "${REPO_ROOT}/venv311/bin/python" -m flask --app app.main run \
  --port 5001 --host 127.0.0.1 >> logs/flask.log 2>&1 &

echo "Waiting for startup..."
sleep 25

echo "=== processes ==="
pgrep -fl 'paper_bet_auto_post_toss' || echo "(no daemon)"
pgrep -fl 'flask --app app.main' || echo "(no flask)"

echo "=== geoblock ==="
curl -s "http://127.0.0.1:5001/api/betting/geoblock" | "${REPO_ROOT}/venv311/bin/python" -m json.tool || echo "Flask not responding yet"

if [[ -f logs/paper_auto_post_toss_status.json ]]; then
  echo "=== daemon status ==="
  head -12 logs/paper_auto_post_toss_status.json
fi
