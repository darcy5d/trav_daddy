# Service Resilience Plan — kill the Flask/daemon restart fragility

## Context

During the 2026-06-28 restart (to pick up `BETTING_TWAP_MAX_CHASE_PP`), Flask
refused to come back up for ~40 minutes. Root cause was **not** the chase-cap
change; it was a latent fragility in how our local services start. This plan
records the failure, the root cause, and a phased fix.

## What actually happened

1. Killed the daemon + Flask to restart them with the new env var.
2. Processes spawned from the agent shell were **reaped** when the shell exited
   (macOS session/process-group reaping). `nohup`/`disown` alone was not enough;
   only a double-fork (`os.setsid()` + reparent to launchd) survived.
3. Flask's heavy `import tensorflow` (which pulls in `keras` → `sklearn`)
   repeatedly **deadlocked** with `OSError: [Errno 11] Resource deadlock avoided`
   (and sometimes an indefinite hang).

## Root cause

Two compounding problems:

### A. Bytecode-cache recompilation race (the deadlock)

- The traceback always died/hung in `importlib ... get_code -> get_data`, i.e.
  Python reading the **`.py` source** because the `__pycache__/*.pyc` was missing
  or stale and had to be recompiled.
- We run **many** TensorFlow-importing entrypoints, several of them overlapping:
  - `inplay_cashout_scan.py` — every 3 min, **x2** staggered (cron)
  - `live_bet_scan.py` — every 30 min (cron)
  - `paper_bet_daily.py` — every 30 min (cron)
  - post-toss scans spawned by the daemon
  - Flask (`app.main` imports `tensorflow` at module load)
- When two processes import the same heavy module while the `.pyc` is absent,
  both recompile and write the same cache files concurrently, and the OS file
  locking deadlocks (`EDEADLK`). My repeated kill-mid-import cycles **invalidated
  the cache**, which is why the normally-rare race became near-deterministic.
- Confirmed: standalone `import sklearn` succeeded 6/6 when run alone, but
  `app.main` / `import tensorflow` deadlocked whenever it collided with a cron
  scan's import. Even `python -m compileall` deadlocked while scans were running.

### B. The watchdog can't detect a hung (alive-but-not-listening) Flask

- `scripts/watchdog_flask.sh` only probes whether **port 5001 is accepting
  connections**. A Flask process that is deadlocked *during import* is alive but
  not listening. The watchdog:
  - does not kill it, and
  - on the next tick spawns **another** Flask (rm's pidfile, starts fresh),
  - which then collides on imports again → pileup of deadlocked processes.

### C. Minor: missing fork-safety env

- Every other cron entry sets `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`;
  `watchdog_flask.sh` does not. Low impact but inconsistent.

## Goals / acceptance criteria

- A restart of Flask (or daemon) **always** converges to a healthy state within
  a bounded time (target: < 3 min), with no manual intervention.
- No accumulation of zombie/deadlocked Flask processes, ever.
- Concurrent heavy imports never deadlock on the bytecode cache.
- Services survive the spawning shell exiting (no agent-shell reaping).
- Failures are observable (logged stack on hang; "down for > N min" signal).

---

## Phase 1 — Quick wins (low risk, do first)

1. **Precompile bytecode, eliminate the recompile race (root cause A).**
   - Add `scripts/precompile_heavy_imports.sh` that runs, single-process:
     `python -m compileall -q tensorflow keras sklearn scipy pandas numpy`
     (resolve site-packages paths from the venv).
   - Run it: (a) once now, in a quiet window with **no scans running**; (b) from
     cron daily at a quiet hour (e.g. 03:30 UTC), guarded by the global import
     lock from Phase 2 so it never runs while a scan imports.
   - Result: imports find valid `.pyc`, never re-read source, no write contention.

2. **Harden `watchdog_flask.sh` (root cause B).**
   - Before deciding to spawn, if the port is **not** listening, **kill any
     existing `flask --app app.main` process** (it is by definition hung or
     mid-start) and remove its pidfile. This prevents zombie pileup.
   - Add a bounded startup wait with a single retry inside the tick: spawn, wait
     up to ~75s for the port; if it binds, done; if not, kill the spawn and let
     the next cron tick try again (clean slate each time).
   - Add `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` to the spawn env (parity with
     other cron jobs).
   - Apply the same alive-but-idle guard logic to `watchdog_twap_daemon.sh` for
     symmetry (it already checks cmdline, but should also detect a wedged daemon).

3. **Single-owner lock on the watchdogs.**
   - Wrap each watchdog body in `flock -n logs/watchdog_flask.lock` (and the
     twap one) so two overlapping cron ticks can't both act and double-spawn.

## Phase 2 — Serialize heavy imports (structural, kills the race for good)

4. **Global "heavy import" mutex.**
   - Add a tiny helper (e.g. `src/utils/heavy_import_lock.py`) that acquires an
     advisory `flock` on `logs/heavy_import.lock` around `import tensorflow`.
   - Have every TF-importing entrypoint (`app/main.py`, `live_bet_scan.py`,
     `live_bet_post_toss_scan.py`, `paper_bet_*`, `inplay_cashout_scan.py`,
     daemon-spawned scans) acquire it **before** importing TF, release after.
   - Effect: only one process initializes/compiles TF at a time; others wait a
     few seconds instead of deadlocking. Combined with Phase 1 precompile, waits
     become near-zero.

5. **Set thread-limit env globally** for the services (reduce OpenMP/Accelerate
   init contention and CPU thrash from many parallel TF procs):
   `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`
   for the scan/cron entrypoints (Flask can keep more if it needs them).

6. **Reduce import storms.** Re-evaluate `inplay_cashout_scan` cadence
   (every 3 min x2 = 4 TF imports/6 min). Consider a long-lived cashout daemon
   (like the TWAP daemon) that imports TF **once** instead of re-importing every
   run. This removes the dominant source of concurrent imports.

## Phase 3 — Proper supervision (remove the nohup/double-fork hack)

7. **Migrate to launchd LaunchAgents** for Flask and the daemon:
   - `~/Library/LaunchAgents/com.indiasdad.flask.plist` and `...daemon.plist`
     with `KeepAlive=true`, `RunAtLoad=true`, stdout/stderr to `logs/`.
   - launchd runs in a persistent session (no agent-shell reaping) and restarts
     on crash automatically — replaces the double-fork detach entirely.
   - Keep a **liveness** check the watchdog can't get from launchd: a small cron
     (or launchd `StartInterval`) that probes the Flask port and, if dead/hung,
     `launchctl kickstart -k` the service. launchd handles persistence; the probe
     handles deadlock recovery.
   - Retire `restart_local_services.sh`'s `nohup &` path in favor of
     `launchctl kickstart`.

## Phase 4 — Observability

8. **Auto-capture hangs.** Wrap the heavy import in the entrypoints with
   `faulthandler.dump_traceback_later(60)` so any future deadlock writes a stack
   to the log instead of hanging silently.
9. **Down-alert.** Watchdog writes a `last_healthy` timestamp; a cheap check
   surfaces "Flask down > 10 min" (log line the dashboard or a notification can
   pick up).

## Rollout / testing

- Implement Phase 1 first; verify by force-restarting Flask 10x in a loop while
  cron scans run and confirming it always binds within the deadline with zero
  leftover processes.
- Add Phase 2 mutex; re-run the same stress loop — expect zero `EDEADLK`.
- Phase 3 only after 1–2 are proven; validate launchd restart by `kill -9` of
  both services and confirming automatic recovery.
- Each phase is independently shippable; stop after Phase 2 if it proves
  sufficient in practice.

## Out of scope / notes

- The chase-cap change (`BETTING_TWAP_MAX_CHASE_PP=6`) was unrelated to this
  fragility and is already live.
- `.env`, `venv*/`, and `.DS_Store` remain untracked/ignored.
