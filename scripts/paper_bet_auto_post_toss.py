#!/usr/bin/env python3
"""Wave 5.7d: auto-detect toss + fetch confirmed XI + trigger post-toss scan.

Long-running daemon that polls Polymarket for upcoming fixtures and CREX
for their toss outcomes. When toss is detected for a fixture we haven't
already post-toss-bet on, it:
    1. Fetches the confirmed XI from CREX (`fetch_xi_from_crex()`)
    2. Resolves names to DB player_ids (same logic as /api/paper/resolve-xi)
    3. Spawns `paper_bet_post_toss_scan.py` as a subprocess with the
       toss outcome + resolved XIs

Polling cadence: 90 seconds between iterations. Fixtures outside the
T-45min to T+30min window are skipped (toss happens ~30min before kickoff;
a 30min after-buffer catches late-resolving markets).

Idempotency:
    - Daemon-level: in-memory "already-handled" set per fixture_key.
      Resets when daemon restarts.
    - DB-level: skip fixtures that already have a 'post_toss' bet from
      any strategy (covers daemon restart + manual UI submissions).

Logging:
    - Per-iteration summary -> logs/paper_auto_post_toss.log
    - Liveness heartbeat   -> logs/paper_auto_post_toss_status.json

Lifecycle:
    - Single instance enforced via PID file at logs/paper_auto_post_toss.pid
    - Graceful shutdown on SIGTERM / Ctrl-C
    - Crash-safe: each iteration is independently try/except

Usage:
    venv311/bin/python scripts/paper_bet_auto_post_toss.py
        [--poll-interval 90] [--lookback-min 45] [--lookahead-min 30]
        [--once]   # run one iteration and exit (cron-friendly)
        [--dry-run] # don't spawn post-toss-scan, just log what would happen
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_connection
from src.integrations.polymarket import PolymarketClient
from src.integrations.polymarket.upcoming import (
    find_upcoming_cricket_events, attach_db_team_ids,
)
from src.integrations.polymarket.crex_xi import (
    find_crex_match_for_fixture, fetch_xi_from_crex,
)
from src.integrations.odds.polymarket_compare import PolymarketComparisonService
from src.api.crex_scraper import CREXScraper
from src.utils.name_matcher import match_abbreviated_name


REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
STATUS_FILE = LOG_DIR / "paper_auto_post_toss_status.json"
PID_FILE = LOG_DIR / "paper_auto_post_toss.pid"
SCRIPT_LOG = LOG_DIR / "paper_auto_post_toss.log"
POST_TOSS_SCAN_LOG = LOG_DIR / "paper_post_toss.log"
LIVE_POST_TOSS_SCAN_LOG = LOG_DIR / "live_post_toss.log"
VENV_PYTHON = REPO_ROOT / "venv311" / "bin" / "python"
POST_TOSS_SCAN_SCRIPT = REPO_ROOT / "scripts" / "paper_bet_post_toss_scan.py"
LIVE_POST_TOSS_SCAN_SCRIPT = REPO_ROOT / "scripts" / "live_bet_post_toss_scan.py"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(SCRIPT_LOG),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# Module-level shutdown flag set by signal handlers
_SHUTDOWN = False


def _handle_signal(signum, frame):
    global _SHUTDOWN
    logger.info(f"Received signal {signum}, shutting down after current iteration")
    _SHUTDOWN = True


def _write_status(state: Dict[str, Any]) -> None:
    state["written_at_utc"] = datetime.now(timezone.utc).isoformat()
    try:
        with STATUS_FILE.open("w") as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as exc:
        logger.warning(f"Failed to write status file: {exc}")


def _has_post_toss_bet_for_fixture(fixture_key: str) -> bool:
    """Has ANY strategy already placed a post_toss bet for this fixture?"""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT 1 FROM bet_ledger
            WHERE bet_kind = 'paper'
              AND fixture_key = ?
              AND COALESCE(phase, 'pre_toss') = 'post_toss'
            LIMIT 1
            """,
            (fixture_key,),
        )
        return cur.fetchone() is not None


def _get_team_candidates(team_id: int, fmt: str, gender: str) -> List[Tuple[int, str]]:
    """DB players who've appeared for this team since 2023 - candidate pool
    for fuzzy-matching CREX names to player_ids."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT DISTINCT pms.player_id, p.name
            FROM player_match_stats pms
            JOIN players p ON p.player_id = pms.player_id
            JOIN matches m ON m.match_id = pms.match_id
            WHERE pms.team_id = ?
              AND m.match_type = ? AND m.gender = ?
              AND m.date >= '2023-01-01'
            """,
            (team_id, fmt, gender),
        )
        return [(r["player_id"], r["name"]) for r in cur.fetchall()]


def _resolve_xi_to_ids(
    xi_names: List[str], team_id: int, fmt: str, gender: str,
) -> Tuple[List[int], int, int]:
    """Resolve a list of player names to DB player_ids using fuzzy matching.

    Returns (player_ids, n_matched, n_input). When all 11 match we have a
    fully-confirmed XI we can pass to the post-toss scan. If fewer match,
    caller can decide whether to skip XI override (fall back to recent-XI
    proxy in the simulator) or skip the auto-trigger entirely.
    """
    if not xi_names:
        return [], 0, 0
    candidates = _get_team_candidates(team_id, fmt, gender)
    if not candidates:
        return [], 0, len(xi_names)
    ids: List[int] = []
    matched = 0
    for raw in xi_names:
        raw = (raw or "").strip()
        if not raw:
            continue
        m = match_abbreviated_name(raw, candidates, threshold=0.55)
        if m:
            ids.append(m[0])
            matched += 1
    return ids, matched, len(xi_names)


def _detect_toss_winner_side(
    toss_winner_str: str, fixture: Dict[str, Any], crex_match,
) -> Optional[str]:
    """Map the CREX-parsed toss winner string to 'team1' or 'team2'.

    `toss_winner_str` may be a full team name ('Punjab Kings'), an
    abbreviation from CREX live page ('PBKS'), or various other shapes.
    We try multiple matching strategies in order of confidence.
    """
    if not toss_winner_str:
        return None
    cmp = PolymarketComparisonService.label_matches_team

    # Strategy 1: token-overlap against PM team names (handles full names)
    pm_t1 = fixture.get("team1_db_name") or ""
    pm_t2 = fixture.get("team2_db_name") or ""
    if pm_t1 and cmp(toss_winner_str, pm_t1):
        return "team1"
    if pm_t2 and cmp(toss_winner_str, pm_t2):
        return "team2"

    # Strategy 2: token-overlap against CREX team names (different spelling)
    if crex_match:
        # Reverse-derive which CREX team corresponds to which PM team
        if cmp(crex_match.team1_name, pm_t1):
            crex_t1_for_pm_t1, crex_t2_for_pm_t2 = crex_match.team1_name, crex_match.team2_name
        else:
            crex_t1_for_pm_t1, crex_t2_for_pm_t2 = crex_match.team2_name, crex_match.team1_name
        if cmp(toss_winner_str, crex_t1_for_pm_t1):
            return "team1"
        if cmp(toss_winner_str, crex_t2_for_pm_t2):
            return "team2"

    # Strategy 3: lowercase substring match (handles "Punjab" vs "Punjab Kings")
    tw_low = toss_winner_str.lower()
    if pm_t1 and (tw_low in pm_t1.lower() or pm_t1.lower() in tw_low):
        return "team1"
    if pm_t2 and (tw_low in pm_t2.lower() or pm_t2.lower() in tw_low):
        return "team2"

    # Strategy 4: known-aliases table for major leagues. CREX live page
    # uses specific team codes that don't always derive from team names
    # via initial-letter rules.
    KNOWN_ALIASES = {
        # IPL
        "csk": "Chennai Super Kings", "che": "Chennai Super Kings",
        "rcb": "Royal Challengers Bengaluru", "roy": "Royal Challengers Bengaluru",
        "mi":  "Mumbai Indians", "mum": "Mumbai Indians",
        "kkr": "Kolkata Knight Riders", "kol": "Kolkata Knight Riders",
        "dc":  "Delhi Capitals", "del": "Delhi Capitals",
        "rr":  "Rajasthan Royals", "raj": "Rajasthan Royals",
        "pbks": "Punjab Kings", "pk": "Punjab Kings", "pun": "Punjab Kings",
        "srh": "Sunrisers Hyderabad", "sun": "Sunrisers Hyderabad",
        "lsg": "Lucknow Super Giants", "luc": "Lucknow Super Giants",
        "gt":  "Gujarat Titans", "guj": "Gujarat Titans",
        # PSL
        "isu": "Islamabad United", "isl": "Islamabad United",
        "psz": "Peshawar Zalmi", "pes": "Peshawar Zalmi",
        "krk": "Karachi Kings", "kar": "Karachi Kings",
        "lqt": "Lahore Qalandars", "lah": "Lahore Qalandars",
        "mlt": "Multan Sultans", "mul": "Multan Sultans",
        "qlc": "Quetta Gladiators", "qg":  "Quetta Gladiators",
        "hyd": "Hyderabad Kingsmen",  # PSL franchise (different from SRH)
        "raw": "Rawalpindi Pindiz", "rwp": "Rawalpindi Pindiz",
    }

    tw_lower = toss_winner_str.lower().replace("-", "").replace(" ", "").strip()
    if tw_lower in KNOWN_ALIASES:
        canonical = KNOWN_ALIASES[tw_lower]
        if cmp(canonical, pm_t1):
            return "team1"
        if cmp(canonical, pm_t2):
            return "team2"

    # Strategy 5: derived abbreviation match (handles "PK" vs "Punjab Kings")
    def _abbrevs_for(team_name: str) -> Set[str]:
        toks = [t for t in team_name.split() if t]
        out: Set[str] = set()
        if len(toks) >= 2:
            out.add("".join(t[0].upper() for t in toks))
            out.add("".join(t[:2].upper() for t in toks if len(t) >= 2))
        if toks:
            out.add(toks[0][:3].upper())
            out.add(toks[0][:4].upper())
        if len(toks) >= 1:
            first = toks[0]
            if len(first) >= 4:
                out.add((first[0] + first[2] + (toks[1][0] if len(toks) >= 2 else "") + (toks[1][1] if len(toks) >= 2 and len(toks[1]) > 1 else "")).upper())
        return {a for a in out if len(a) >= 2}

    tw_upper = toss_winner_str.upper().replace("-", "").replace(" ", "")
    if tw_upper in _abbrevs_for(pm_t1):
        return "team1"
    if tw_upper in _abbrevs_for(pm_t2):
        return "team2"
    return None


def _spawn_scan_subprocess(
    script_path: Path,
    log_path: Path,
    fixture: Dict[str, Any],
    toss_winner_side: str,
    chose_to: str,
    team1_xi: Optional[List[int]],
    team2_xi: Optional[List[int]],
    dry_run: bool,
    log_tag: str,
) -> Optional[int]:
    """Shared spawner for paper / live post-toss scan subprocesses.
    Both scripts have identical CLIs (fixture-key, toss-winner, chose-to,
    optional XI overrides), so a single helper avoids drift between the
    paper and live spawn paths.
    """
    cmd = [
        str(VENV_PYTHON), str(script_path),
        "--fixture-key", fixture["fixture_key"],
        "--toss-winner", toss_winner_side,
        "--chose-to", chose_to,
    ]
    if team1_xi:
        cmd.extend(["--team1-xi", ",".join(str(int(x)) for x in team1_xi)])
    if team2_xi:
        cmd.extend(["--team2-xi", ",".join(str(int(x)) for x in team2_xi)])
    cmd_str = " ".join(cmd)
    if dry_run:
        logger.info(f"  [DRY-RUN] would spawn ({log_tag}): {cmd_str}")
        return None
    logger.info(f"  Spawning ({log_tag}): {cmd_str}")
    with log_path.open("a") as logf:
        logf.write(f"\n\n=== auto-toss {log_tag} spawn at {datetime.now(timezone.utc).isoformat()} ===\n")
        logf.write(f"=== cmd: {cmd_str} ===\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT), stdout=logf,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
    return proc.pid


def _spawn_post_toss_scan(
    fixture: Dict[str, Any],
    toss_winner_side: str,
    chose_to: str,
    team1_xi: Optional[List[int]] = None,
    team2_xi: Optional[List[int]] = None,
    dry_run: bool = False,
) -> Optional[int]:
    """Spawn the PAPER post-toss scan as a detached subprocess.
    Returns the PID, or None if dry_run."""
    return _spawn_scan_subprocess(
        script_path=POST_TOSS_SCAN_SCRIPT,
        log_path=POST_TOSS_SCAN_LOG,
        fixture=fixture,
        toss_winner_side=toss_winner_side,
        chose_to=chose_to,
        team1_xi=team1_xi,
        team2_xi=team2_xi,
        dry_run=dry_run,
        log_tag="paper",
    )


def _spawn_live_post_toss_scan(
    fixture: Dict[str, Any],
    toss_winner_side: str,
    chose_to: str,
    team1_xi: Optional[List[int]] = None,
    team2_xi: Optional[List[int]] = None,
    dry_run: bool = False,
) -> Optional[int]:
    """Spawn the LIVE post-toss scan as a detached subprocess.
    The live scan has its own internal mode/kill-switch checks so we don't
    duplicate them here; if BETTING_MODE != AUTO the spawned process will
    log and exit cleanly without placing any bets.
    Returns the PID, or None if dry_run."""
    return _spawn_scan_subprocess(
        script_path=LIVE_POST_TOSS_SCAN_SCRIPT,
        log_path=LIVE_POST_TOSS_SCAN_LOG,
        fixture=fixture,
        toss_winner_side=toss_winner_side,
        chose_to=chose_to,
        team1_xi=team1_xi,
        team2_xi=team2_xi,
        dry_run=dry_run,
        log_tag="live",
    )


def process_fixture(
    fixture: Dict[str, Any],
    handled_set: Set[str],
    dry_run: bool = False,
    also_live: bool = False,
) -> Dict[str, Any]:
    """Process one upcoming fixture: check toss, fetch XI, spawn scan if applicable.

    If `also_live` is True, ALSO spawn the live post-toss scan so the same
    toss event triggers both paper analytics and real-money placement.
    The two subprocesses run independently and write to separate log files
    (logs/paper_post_toss.log and logs/live_post_toss.log)."""
    fixture_key = fixture["fixture_key"]
    result = {"fixture_key": fixture_key, "action": None, "reason": None}

    # Fast in-memory dedup
    if fixture_key in handled_set:
        result["action"] = "skip"
        result["reason"] = "already-handled-this-session"
        return result

    # DB-level dedup
    if _has_post_toss_bet_for_fixture(fixture_key):
        result["action"] = "skip"
        result["reason"] = "post_toss-bet-already-in-db"
        handled_set.add(fixture_key)
        return result

    # Find CREX match (uses 15min schedule cache)
    crex_match = find_crex_match_for_fixture(fixture)
    if crex_match is None:
        result["action"] = "skip"
        result["reason"] = "no-CREX-match"
        return result
    if not crex_match.match_url:
        result["action"] = "skip"
        result["reason"] = "no-CREX-url"
        return result

    # Check the live page for toss
    scraper = CREXScraper()
    try:
        live = scraper.get_live_match(crex_match.match_url)
    except Exception as exc:
        result["action"] = "error"
        result["reason"] = f"CREX live fetch failed: {exc}"
        return result
    if live is None or not live.toss_winner:
        result["action"] = "wait"
        result["reason"] = "toss-not-yet-detected"
        return result

    # Map toss winner to team1 / team2
    side = _detect_toss_winner_side(live.toss_winner, fixture, crex_match)
    if side is None:
        result["action"] = "error"
        result["reason"] = f"could-not-map-toss-winner '{live.toss_winner}' to team1/team2"
        return result

    # Decision: bat or bowl. _parse_toss returns "bowl" instead of "field"
    chose_raw = (live.toss_decision or "").lower()
    if chose_raw in ("bat",):
        chose_to = "bat"
    elif chose_raw in ("bowl", "field"):
        chose_to = "field"
    else:
        result["action"] = "error"
        result["reason"] = f"unknown-toss-decision '{live.toss_decision}'"
        return result

    # Pull confirmed XI (best-effort - if it fails we still trigger scan
    # without XI override)
    xi_data = None
    try:
        xi_data = fetch_xi_from_crex(fixture)
    except Exception as exc:
        logger.warning(f"  fetch_xi_from_crex failed for {fixture_key}: {exc}")

    team1_xi: Optional[List[int]] = None
    team2_xi: Optional[List[int]] = None
    fmt = fixture["format"]
    gender = fixture["gender"]
    if xi_data:
        t1_ids, t1_matched, t1_total = _resolve_xi_to_ids(
            xi_data.get("team1_xi", []), fixture["team1_id"], fmt, gender,
        )
        t2_ids, t2_matched, t2_total = _resolve_xi_to_ids(
            xi_data.get("team2_xi", []), fixture["team2_id"], fmt, gender,
        )
        # Only override XI if we resolved AT LEAST 9 of 11 (avoid bad data)
        if t1_matched >= 9 and t1_total >= 9:
            team1_xi = t1_ids[:11]
        if t2_matched >= 9 and t2_total >= 9:
            team2_xi = t2_ids[:11]
        logger.info(
            f"  XI resolved for {fixture_key}: "
            f"team1 {t1_matched}/{t1_total} (override={'yes' if team1_xi else 'no'}), "
            f"team2 {t2_matched}/{t2_total} (override={'yes' if team2_xi else 'no'})"
        )

    # Spawn the paper post-toss scan
    pid = _spawn_post_toss_scan(
        fixture, side, chose_to,
        team1_xi=team1_xi, team2_xi=team2_xi, dry_run=dry_run,
    )
    # Optionally spawn the LIVE post-toss scan in parallel. Independent
    # subprocess so a paper-side crash can't block the live placement
    # (and vice versa). The live script's own mode/kill-switch checks
    # decide whether the spawned process actually places anything.
    live_pid: Optional[int] = None
    if also_live:
        try:
            live_pid = _spawn_live_post_toss_scan(
                fixture, side, chose_to,
                team1_xi=team1_xi, team2_xi=team2_xi, dry_run=dry_run,
            )
        except Exception as exc:
            logger.error(f"  live post-toss spawn failed for {fixture_key}: {exc}")
    handled_set.add(fixture_key)
    result["action"] = "spawned" if pid else "dry-run"
    result["pid"] = pid
    result["live_pid"] = live_pid
    result["toss_winner_side"] = side
    result["chose_to"] = chose_to
    result["team1_xi_override"] = bool(team1_xi)
    result["team2_xi_override"] = bool(team2_xi)
    result["crex_toss_winner_raw"] = live.toss_winner
    return result


def run_single_iteration(
    handled_set: Set[str],
    lookback_min: int,
    lookahead_min: int,
    dry_run: bool = False,
    also_live: bool = False,
) -> Dict[str, Any]:
    """One scan iteration. Returns a per-iteration summary dict."""
    iter_summary: Dict[str, Any] = {
        "iter_started_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixtures_seen": 0,
        "fixtures_in_window": 0,
        "fixtures_processed": [],
        "also_live": also_live,
    }

    try:
        c = PolymarketClient()
        events = find_upcoming_cricket_events(c, hours_ahead=24, include_started=True)
        mapped = attach_db_team_ids(events)
    except Exception as exc:
        logger.error(f"Polymarket fixture fetch failed: {exc}")
        iter_summary["error"] = f"polymarket-fetch: {exc}"
        return iter_summary

    iter_summary["fixtures_seen"] = len(mapped)
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(minutes=lookahead_min)  # already-started up to N min after kickoff
    window_end = now + timedelta(minutes=lookback_min)     # haven't started yet, within N min before kickoff

    in_window = []
    for fx in mapped:
        if not (fx.get("team1_id") and fx.get("team2_id")):
            continue
        if not fx.get("moneyline"):
            continue
        kickoff = fx["scheduled_start_estimate"]
        if window_start <= kickoff <= window_end:
            in_window.append(fx)

    iter_summary["fixtures_in_window"] = len(in_window)

    for fx in in_window:
        try:
            res = process_fixture(fx, handled_set, dry_run=dry_run, also_live=also_live)
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error(f"  process_fixture crashed for {fx.get('fixture_key')}: {exc}\n{tb}")
            res = {"fixture_key": fx.get("fixture_key"), "action": "error", "reason": f"crash: {exc}"}
        iter_summary["fixtures_processed"].append(res)

        emoji = {"spawned": "✓", "dry-run": "[dry]", "skip": "·", "wait": "...", "error": "!"}
        marker = emoji.get(res.get("action") or "", "?")
        live_tag = " +live" if res.get("live_pid") else ""
        logger.info(
            f"  {marker} {fx['team1_db_name']} vs {fx['team2_db_name']}{live_tag}: "
            f"{res.get('action')} ({res.get('reason') or '-'})"
        )

    iter_summary["iter_finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    return iter_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-detect toss and trigger post-toss scans")
    parser.add_argument("--poll-interval", type=int, default=90, help="Seconds between polls")
    parser.add_argument("--lookback-min", type=int, default=45, help="Look at fixtures up to N min before kickoff")
    parser.add_argument("--lookahead-min", type=int, default=30, help="Also look at fixtures up to N min AFTER kickoff (toss may be late)")
    parser.add_argument("--once", action="store_true", help="Run a single iteration and exit (cron-friendly)")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually spawn post-toss scans, just log")
    parser.add_argument("--also-live", action="store_true",
                        help=("Also spawn the LIVE post-toss scan (scripts/live_bet_post_toss_scan.py) "
                              "for each detected toss. The live scan respects BETTING_MODE / kill switch / "
                              "BETTING_LIVE_STRATEGIES, so leaving it on without configuring real betting "
                              "is safe (it'll just log 'mode_off' and exit)."))
    args = parser.parse_args()

    # PID-file lifecycle (only for daemon mode, not --once)
    if not args.once:
        if PID_FILE.exists():
            try:
                pid_str = PID_FILE.read_text().strip()
                if pid_str:
                    other_pid = int(pid_str)
                    try:
                        os.kill(other_pid, 0)  # check if still alive
                        logger.error(
                            f"Daemon already running (pid {other_pid}). To restart, kill it first or "
                            f"`rm {PID_FILE}` if it's stale."
                        )
                        return 1
                    except OSError:
                        logger.warning(f"Stale PID file (pid {other_pid} not running), reclaiming")
            except (ValueError, OSError):
                pass
        PID_FILE.write_text(str(os.getpid()))
        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)

    handled_set: Set[str] = set()
    n_iters = 0
    n_spawns_total = 0
    n_live_spawns_total = 0

    logger.info(
        f"Auto-toss daemon starting (poll_interval={args.poll_interval}s, "
        f"window=[T-{args.lookback_min}m, T+{args.lookahead_min}m], "
        f"dry_run={args.dry_run}, once={args.once}, also_live={args.also_live})"
    )

    try:
        while not _SHUTDOWN:
            iter_started = time.time()
            n_iters += 1
            try:
                iter_summary = run_single_iteration(
                    handled_set, args.lookback_min, args.lookahead_min,
                    args.dry_run, args.also_live,
                )
                spawns_this_iter = sum(
                    1 for f in iter_summary.get("fixtures_processed", [])
                    if f.get("action") == "spawned"
                )
                live_spawns_this_iter = sum(
                    1 for f in iter_summary.get("fixtures_processed", [])
                    if f.get("live_pid")
                )
                n_spawns_total += spawns_this_iter
                n_live_spawns_total += live_spawns_this_iter
            except Exception as exc:
                tb = traceback.format_exc()
                logger.error(f"Iteration {n_iters} crashed: {exc}\n{tb}")
                iter_summary = {"error": str(exc)}
                spawns_this_iter = 0
                live_spawns_this_iter = 0

            elapsed = time.time() - iter_started
            _write_status({
                "pid": os.getpid(),
                "running": True,
                "iter_count": n_iters,
                "spawns_total": n_spawns_total,
                "spawns_this_iter": spawns_this_iter,
                "live_spawns_total": n_live_spawns_total,
                "live_spawns_this_iter": live_spawns_this_iter,
                "iter_elapsed_s": round(elapsed, 1),
                "handled_session": sorted(handled_set),
                "last_iter": iter_summary,
                "poll_interval": args.poll_interval,
                "lookback_min": args.lookback_min,
                "lookahead_min": args.lookahead_min,
                "dry_run": args.dry_run,
                "also_live": args.also_live,
            })

            if args.once:
                logger.info(
                    f"--once: exiting after 1 iteration ({elapsed:.1f}s, "
                    f"{spawns_this_iter} paper spawns, {live_spawns_this_iter} live spawns)"
                )
                break

            sleep_s = max(0, args.poll_interval - elapsed)
            logger.info(
                f"Iteration {n_iters} done in {elapsed:.1f}s "
                f"(spawned {spawns_this_iter} paper, {live_spawns_this_iter} live). "
                f"Sleeping {sleep_s:.1f}s..."
            )
            # Sleep in 1s chunks so signal handlers get a chance
            slept = 0.0
            while slept < sleep_s and not _SHUTDOWN:
                time.sleep(min(1.0, sleep_s - slept))
                slept += 1.0
    finally:
        _write_status({
            "pid": os.getpid(),
            "running": False,
            "iter_count": n_iters,
            "spawns_total": n_spawns_total,
            "live_spawns_total": n_live_spawns_total,
            "also_live": args.also_live,
            "shutdown_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        if not args.once and PID_FILE.exists():
            try:
                if PID_FILE.read_text().strip() == str(os.getpid()):
                    PID_FILE.unlink()
            except Exception:
                pass

    logger.info(
        f"Daemon exiting after {n_iters} iterations "
        f"({n_spawns_total} total paper spawns, {n_live_spawns_total} total live spawns)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
