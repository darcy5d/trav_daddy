#!/usr/bin/env python3
"""Phase 2: Polymarket competition discovery + data-coverage gate.

Answers "which cricket competitions is Polymarket currently listing, do we
recognise them, and do we have enough data to model them?" — so we can onboard
well-covered leagues (paper-first) and avoid betting blind on the rest.

For every upcoming/live Polymarket cricket fixture it:
  1. Discovers fixtures (Gamma /events) and groups them by tournament prefix.
  2. Flags each prefix as mapped (TOURNAMENT_PREFIX_MAP) or UNMAPPED.
  3. Resolves each fixture's teams to cricket.db team_ids (fuzzy title match).
  4. Scores data coverage per league from cricket.db:
       - team-resolution rate (labels -> DB team_id)
       - historical match count + recency for the resolved teams (format+gender)
       - player-vocab coverage: fraction of recent squad players present in the
         V2/V3 model vocab (data/models/{v2,v3}/vocabs.json).
  5. Emits a per-league verdict: GO / PAPER-ONLY / NO-GO.

Usage:
    venv311/bin/python scripts/poly_competition_coverage.py
    venv311/bin/python scripts/poly_competition_coverage.py --hours-ahead 336 --json out.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import BETTING_CONFIG
from src.data.database import get_connection
from src.integrations.polymarket import PolymarketClient
from src.integrations.polymarket.upcoming import (
    TOURNAMENT_PREFIX_MAP,
    attach_db_team_ids,
    find_upcoming_cricket_events,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Coverage thresholds (tunable) ------------------------------------------
RECENCY_DAYS = 540          # most-recent match for the league's teams within this
HISTORY_DAYS = 1095         # window for counting historical matches / squads
GO_MIN_TEAM_MATCH_RATE = 0.90
GO_MIN_MEDIAN_MATCHES = 30
GO_MIN_VOCAB_COVERAGE = 0.60
PAPER_MIN_TEAM_MATCH_RATE = 0.50
PAPER_MIN_MEDIAN_MATCHES = 8


def _load_vocab_ids() -> Tuple[Set[str], Set[str]]:
    """Return (player_vocab_ids, team_vocab_ids) merged across V2 and V3."""
    players: Set[str] = set()
    teams: Set[str] = set()
    for ver in ("v2", "v3"):
        p = Path(f"data/models/{ver}/vocabs.json")
        if not p.exists():
            continue
        try:
            v = json.loads(p.read_text())
        except Exception as exc:
            logger.warning("Could not read %s: %s", p, exc)
            continue
        for key in ("batter", "bowler"):
            players.update(str(k) for k in (v.get(key) or {}).keys())
        teams.update(str(k) for k in (v.get("team") or {}).keys())
    return players, teams


def _team_profile(
    conn,
    team_id: int,
    fmt: str,
    gender: str,
    vocab_players: Set[str],
    vocab_teams: Set[str],
) -> Dict[str, Any]:
    """Per-team data coverage in cricket.db for a given format/gender."""
    cur = conn.cursor()
    recent_cut = (datetime.now(timezone.utc) - timedelta(days=HISTORY_DAYS)).date().isoformat()

    cur.execute(
        """
        SELECT COUNT(*) AS n, MAX(date) AS last_date
        FROM matches
        WHERE (team1_id = ? OR team2_id = ?) AND match_type = ? AND gender = ?
        """,
        (team_id, team_id, fmt, gender),
    )
    row = cur.fetchone()
    n_matches = (row["n"] if row else 0) or 0
    last_date = row["last_date"] if row else None

    # Player-vocab coverage: distinct players who turned out for this team in
    # recent matches of this format/gender, and how many are in the model vocab.
    cur.execute(
        """
        SELECT DISTINCT pms.player_id
        FROM player_match_stats pms
        JOIN matches m ON m.match_id = pms.match_id
        WHERE pms.team_id = ? AND m.match_type = ? AND m.gender = ? AND m.date >= ?
        """,
        (team_id, fmt, gender, recent_cut),
    )
    player_ids = [str(r["player_id"]) for r in cur.fetchall()]
    n_players = len(player_ids)
    n_in_vocab = sum(1 for pid in player_ids if pid in vocab_players)

    return {
        "team_id": team_id,
        "n_matches": n_matches,
        "last_date": last_date,
        "n_players_recent": n_players,
        "n_players_in_vocab": n_in_vocab,
        "team_in_vocab": str(team_id) in vocab_teams,
    }


def _classify(cov: Dict[str, Any]) -> Tuple[str, List[str]]:
    """GO / PAPER-ONLY / NO-GO from a league coverage summary."""
    reasons: List[str] = []
    if not cov["mapped"]:
        reasons.append("prefix UNMAPPED in TOURNAMENT_PREFIX_MAP")
    rate = cov["team_match_rate"]
    med = cov["median_matches"]
    vocab = cov["vocab_coverage"]
    recent_ok = cov["most_recent_within_recency"]

    if (
        cov["mapped"]
        and rate >= GO_MIN_TEAM_MATCH_RATE
        and med >= GO_MIN_MEDIAN_MATCHES
        and vocab >= GO_MIN_VOCAB_COVERAGE
        and recent_ok
    ):
        return "GO", ["meets all GO thresholds"]

    if (
        rate >= PAPER_MIN_TEAM_MATCH_RATE
        and med >= PAPER_MIN_MEDIAN_MATCHES
        and cov["n_teams_resolved"] > 0
    ):
        if rate < GO_MIN_TEAM_MATCH_RATE:
            reasons.append(f"team-match rate {rate:.0%} < GO {GO_MIN_TEAM_MATCH_RATE:.0%}")
        if med < GO_MIN_MEDIAN_MATCHES:
            reasons.append(f"median matches {med:.0f} < GO {GO_MIN_MEDIAN_MATCHES}")
        if vocab < GO_MIN_VOCAB_COVERAGE:
            reasons.append(f"vocab coverage {vocab:.0%} < GO {GO_MIN_VOCAB_COVERAGE:.0%}")
        if not recent_ok:
            reasons.append(f"no match within {RECENCY_DAYS}d")
        return "PAPER-ONLY", reasons or ["below GO thresholds"]

    if rate < PAPER_MIN_TEAM_MATCH_RATE:
        reasons.append(f"team-match rate {rate:.0%} < {PAPER_MIN_TEAM_MATCH_RATE:.0%}")
    if med < PAPER_MIN_MEDIAN_MATCHES:
        reasons.append(f"median matches {med:.0f} < {PAPER_MIN_MEDIAN_MATCHES}")
    return "NO-GO", reasons or ["insufficient data"]


def build_report(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    vocab_players, vocab_teams = _load_vocab_ids()
    live_excluded = {p.lower() for p in (BETTING_CONFIG.get("live_exclude_prefixes") or [])}

    by_prefix: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ev in events:
        by_prefix[ev["tournament_prefix"]].append(ev)

    report: List[Dict[str, Any]] = []
    with get_connection() as conn:
        for prefix, fixtures in sorted(by_prefix.items()):
            fmt = fixtures[0].get("format", "T20")
            gender = fixtures[0].get("gender", "male")
            tour_name = fixtures[0].get("tournament_name", prefix)

            # Resolve unique teams across this league's fixtures.
            resolved_ids: Set[int] = set()
            label_slots = 0
            resolved_slots = 0
            for ev in fixtures:
                for side in (1, 2):
                    label = ev.get(f"team{side}_label")
                    tid = ev.get(f"team{side}_id")
                    if label:
                        label_slots += 1
                        if tid is not None:
                            resolved_slots += 1
                            resolved_ids.add(int(tid))

            profiles = [
                _team_profile(conn, tid, fmt, gender, vocab_players, vocab_teams)
                for tid in sorted(resolved_ids)
            ]

            match_counts = [p["n_matches"] for p in profiles]
            total_players = sum(p["n_players_recent"] for p in profiles)
            players_in_vocab = sum(p["n_players_in_vocab"] for p in profiles)
            last_dates = [p["last_date"] for p in profiles if p["last_date"]]
            most_recent = max(last_dates) if last_dates else None
            recency_cut = (datetime.now(timezone.utc) - timedelta(days=RECENCY_DAYS)).date().isoformat()

            cov = {
                "prefix": prefix,
                "tournament_name": tour_name,
                "format": fmt,
                "gender": gender,
                "mapped": prefix in TOURNAMENT_PREFIX_MAP,
                "live_excluded": prefix.lower() in live_excluded,
                "n_fixtures": len(fixtures),
                "sample_fixtures": [f["fixture_key"] for f in fixtures[:4]],
                "n_team_slots": label_slots,
                "n_teams_resolved": len(resolved_ids),
                "team_match_rate": (resolved_slots / label_slots) if label_slots else 0.0,
                "median_matches": float(median(match_counts)) if match_counts else 0.0,
                "vocab_coverage": (players_in_vocab / total_players) if total_players else 0.0,
                "most_recent_match": most_recent,
                "most_recent_within_recency": bool(most_recent and most_recent >= recency_cut),
                "teams_in_vocab": sum(1 for p in profiles if p["team_in_vocab"]),
                "team_profiles": profiles,
            }
            cov["verdict"], cov["verdict_reasons"] = _classify(cov)
            report.append(cov)

    order = {"GO": 0, "PAPER-ONLY": 1, "NO-GO": 2}
    report.sort(key=lambda c: (order.get(c["verdict"], 9), -c["n_fixtures"]))
    return report


def print_report(report: List[Dict[str, Any]]) -> None:
    print()
    print("=" * 100)
    print("POLYMARKET CRICKET COMPETITION COVERAGE")
    print("=" * 100)
    hdr = f"{'PREFIX':<16}{'VERDICT':<12}{'FIX':>4}  {'MAP':<4}{'LIVEX':<6}{'TEAMS':>7}{'MATCH%':>8}{'MEDmt':>7}{'VOCAB%':>8}  {'NAME'}"
    print(hdr)
    print("-" * 100)
    for c in report:
        print(
            f"{c['prefix']:<16}"
            f"{c['verdict']:<12}"
            f"{c['n_fixtures']:>4}  "
            f"{('Y' if c['mapped'] else 'NO'):<4}"
            f"{('Y' if c['live_excluded'] else '-'):<6}"
            f"{c['n_teams_resolved']:>3}/{c['n_team_slots']:<3}"
            f"{c['team_match_rate']*100:>7.0f}%"
            f"{c['median_matches']:>7.0f}"
            f"{c['vocab_coverage']*100:>7.0f}%"
            f"  {c['tournament_name'][:34]} ({c['format']}/{c['gender']})"
        )
    print("-" * 100)
    for c in report:
        print(f"\n[{c['verdict']}] {c['prefix']} — {c['tournament_name']} ({c['format']}/{c['gender']})")
        print(f"    fixtures={c['n_fixtures']} sample={c['sample_fixtures']}")
        print(f"    most_recent_match={c['most_recent_match']}  teams_in_vocab={c['teams_in_vocab']}/{c['n_teams_resolved']}")
        print(f"    reasons: {'; '.join(c['verdict_reasons'])}")

    counts = defaultdict(int)
    for c in report:
        counts[c["verdict"]] += 1
    print("\n" + "=" * 100)
    print(f"SUMMARY: {dict(counts)}  across {len(report)} live competition prefix(es)")
    print("=" * 100)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours-ahead", type=float, default=336.0,
                        help="Discovery horizon in hours (default 336 = 14 days).")
    parser.add_argument("--include-started", action="store_true", default=True,
                        help="Include in-play fixtures (default true).")
    parser.add_argument("--json", type=str, default=None,
                        help="Optional path to write the full report as JSON.")
    args = parser.parse_args()

    logger.info("Discovering Polymarket cricket fixtures (horizon=%.0fh)…", args.hours_ahead)
    client = PolymarketClient()
    events = find_upcoming_cricket_events(
        client, hours_ahead=args.hours_ahead, include_started=args.include_started,
    )
    if not events:
        logger.warning("No upcoming cricket fixtures discovered (network issue or empty board).")
        return 1
    logger.info("Discovered %d fixture(s); resolving DB teams…", len(events))
    events = attach_db_team_ids(events)

    report = build_report(events)
    print_report(report)

    if args.json:
        # Drop nested profiles' verbosity but keep them; JSON is for tooling.
        Path(args.json).write_text(json.dumps(report, indent=2, default=str))
        logger.info("Wrote JSON report to %s", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
