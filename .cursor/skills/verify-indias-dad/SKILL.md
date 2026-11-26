---
name: verify-indias-dad
description: Drive the indias_dad cricket predictor / Polymarket betting Flask web UI (127.0.0.1:5001 live, or a scratch instance) in a real Chromium via Playwright to prove a change works — predict, bulk-predict, rankings, team-explorer, training, live-betting pages. Use after changing app/main.py, app/templates/, src/, or config.py, or whenever you need screenshot/network evidence of UI or API behavior. Read-only by default; never places bets.
---

# Verify indias_dad

The user-facing surface is a Flask web app (`app/main.py`, Jinja templates in `app/templates/`) rendered in a browser. Secondary surfaces: JSON APIs under `/api/*` (what the pages call) and cron scripts in `scripts/` (not covered here). Drive the UI with `scripts/drive.py` (Playwright, Chromium already installed in `venv311`).

All commands run from the repo root `~/Desktop/DD_AI_models/indias_dad` with `venv311/bin/python`. `S=.cursor/skills/verify-indias-dad/scripts`.

## Hard rules (this app moves real money)

- The live instance on **:5001 is production**: `BETTING_MODE` is `AUTO` with a funded Polymarket wallet, cron jobs (`crontab -l`) scan/place/cash-out bets, and `scripts/watchdog_flask.sh` restarts Flask every 5 min if it looks down. **Never** restart, kill, or reconfigure it, never `pkill`/`killall` anything, never run `scripts/restart_local_services.sh`, never edit `.env`.
- Never trigger these, on any instance: `/api/betting/place`, `/kill-switch` (it rewrites `.env`), `/mode`, `/reconcile`, `/scale-up`, `/post-toss-scan`, any `/api/paper/*` POST (start/stop poller, scans, resolve/fetch XI), `/api/training/*` POSTs (download/reset-database/retrain/tune/activate), `/api/team-explorer/proposals*` POSTs, `/api/admin/*` POSTs. Do not run `scripts/live_bet_scan.py`, `redeem_*`, `reconcile_*`, `inplay_cashout_scan.py`, or `npm run redeem-winners`.
- `drive.py` enforces this: it aborts every non-GET request except the compute-only allow-list (`/api/simulate`, `/api/simulate-stream`, `/api/players/by-ids`, `/api/team/optimize`, `/api/integrations/polymarket/compare/{batch,multi,multi/batch}`) and logs each block. Don't bypass it with curl. `--allow-post` is refused on :5001; on a scratch instance, only allow a write route if you are verifying that exact route and you accept that **scratch shares the live databases** (`cricket.db`, betting tables).
- Do not double-drive: one Playwright session per instance at a time; the live UI is also the user's.

## Launch

Two targets — choose by what you changed:

1. **Live :5001 (default, nothing to launch).** Already running (`flask --app app.main run --port 5001`, no reloader). `TEMPLATES_AUTO_RELOAD` is on, so **template/JS edits are live on the next page load**, but **Python edits (`app/main.py`, `src/`, `config.py`) are NOT** until someone else restarts it. `doctor.py` prints `STALE` when that's the case.
2. **Scratch instance (for Python changes).** Runs the code on disk, side by side:

```bash
$S/scratch.sh start 5091      # prints READY http://127.0.0.1:5091 (TensorFlow import: ~20-90s)
$S/scratch.sh status 5091
```

Scratch is started as `python -m flask run` with `FLASK_APP=app.main` (so the watchdog's `pkill -f 'flask --app app.main'` can't hit it), with wallet/CLOB credentials blanked, `BETTING_MODE=OFF`, `BETTING_KILL_SWITCH=1`. Its log goes to `logs/verify/scratch-<port>-<ts>.log`; PID in `tmp/verify-indias-dad/scratch-<port>.pid`. Port 5001 is refused. Startup runs idempotent schema migrations on the shared DB. Teardown: `$S/scratch.sh stop 5091`.

## Doctor (run first, and whenever anything looks off)

```bash
venv311/bin/python $S/doctor.py            # live :5001 + any recorded scratch instance
venv311/bin/python $S/doctor.py 5091       # a specific port
```

Read-only. Shows listener PID/command/start time, `/healthz` (`{"db_ok":true}` expected), whether the process predates the newest `.py` change (`STALE`), git branch/HEAD, and whether Playwright imports. `DOCTOR OK` / exit 0 means drive it. `/healthz` 503 = SQLite wedged — report it, don't fix it (the watchdog owns that).

## Drive

```bash
venv311/bin/python $S/drive.py [--base http://127.0.0.1:5091] [--run-id NAME] [--timeout-ms 60000] [--headed] STEP...
```

Steps run in order and stop at the first failure: `goto:/path`, `wait:SEL`, `waitgone:SEL`, `waittext:SEL~REGEX`, `click:SEL`, `select:SEL=VALUE_OR_LABEL`, `fill:SEL=VALUE`, `text:SEL`, `count:SEL`, `expect:SEL~REGEX`, `expectcount:SEL>=N`, `dialogs:accept|dismiss` (answer later `alert`/`confirm`; default dismiss), `shot:NAME`, `sleep:MS`. Selectors are Playwright selectors; the templates use stable `id`s — prefer them. Quote steps containing `#`, `~`, `>`, or spaces.

Example (verified on live and scratch, ~6s): team rankings render with real data and react to a filter. Player lists on that page are much slower (see `features/rankings.md`).

```bash
venv311/bin/python $S/drive.py --run-id rankings-check goto:/rankings \
  "waitgone:#team-rankings .spinner" "expectcount:#team-rankings .ranking-row>=10" \
  "expect:#tier-stats-banner~T1" "select:#gender-select=female" \
  "waitgone:#team-rankings .spinner" "expectcount:#team-rankings .ranking-row>=5" shot:rankings
```

Per-feature recipes, selectors, and end states live in `features/` — read `features/README.md` and the file for the area you changed, and cover every entry point it lists.

## Evidence

Each run writes `logs/verify/<run-id>/` (gitignored, survives cleanup; an existing `<run-id>` is never overwritten — the run goes to `<run-id>-2`, and the last output line prints the real path):

- `result.json` — every step with pass/fail, captured text/counts, timings, blocked requests, page errors.
- `network.jsonl` — method/URL/status of each response, plus `"blocked": true` entries.
- `console.log` — browser console, `[pageerror]`, and every dialog with how it was answered.
- `*.png` — `shot:` screenshots; `FAILED.png` at the failing step.

Proof standards: drive the real page the user uses, not the JSON API alone (use `curl` on GET APIs only to corroborate). Capture the action and the resulting state — screenshot plus a `text:`/`expect:` of the value that changed. For backend changes, also show the request in `network.jsonl` with its status, and for anything that should persist, read it back with a read-only `sqlite3 -readonly cricket.db "SELECT ..."` (never write). Compare `page_errors` against the baseline in the feature file; new page errors are regressions. If you verified on live, say whether doctor reported `STALE` (if so, Python changes were not exercised).

## Cleanup

```bash
$S/scratch.sh stop 5091    # kills only the PID it recorded, after checking its command line
```

Drive runs leave no processes (the browser closes per run). Never kill by name, never touch PID 5001's process. Cleanup removes `tmp/verify-indias-dad/`; it never deletes `logs/verify/` — confirm evidence is still there with `ls logs/verify/<run-id>/`.

## Helpers

| Script | Invocation |
|---|---|
| `scripts/doctor.py` | `venv311/bin/python $S/doctor.py [PORT ...]` |
| `scripts/scratch.sh` | `$S/scratch.sh start\|status\|stop [PORT=5091]` |
| `scripts/drive.py` | `venv311/bin/python $S/drive.py [--base URL] [--run-id ID] STEP...` |

Keep `features/` honest as the app changes: `/maintain-verification-skill`.
