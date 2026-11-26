#!/usr/bin/env python3
"""CricketArchive pilot harvester (read-only, no DB writes).

Walks a CricketArchive series page end-to-end and proves the full pipeline:
    series -> scorecards -> (meta + batting + bowling) -> ball-by-ball commentary

For each match it parses the scorecard and both innings of commentary, then
self-validates the ball-by-ball translation by reconciling commentary totals
against the scorecard innings totals (runs + wickets). It also collects the
CA player-ID -> name map (priority B raw material) across the series.

Writes a structured JSON summary to data/processed/cricketarchive/ for review.
Pages are served from the on-disk cache, so re-runs hit no network.

Usage:
    venv311/bin/python scripts/cricketarchive_pilot.py
    venv311/bin/python scripts/cricketarchive_pilot.py --series <event_url>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CRICKETARCHIVE_CONFIG, PROCESSED_DATA_DIR
from src.api.cricketarchive import auth
from src.api.cricketarchive.fetcher import PoliteFetcher
from src.api.cricketarchive import parsers as P

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ca_pilot")

DEFAULT_SERIES = "https://cricketarchive.com/Archive/Events/41/India_in_Ireland_2026.html"


def _batting_order_surnames(innings) -> list[str]:
    """Surnames in batting order, for commentary non-striker reconstruction."""
    return [b.name.split()[-1] for b in innings.batting]


def process_match(fetcher: PoliteFetcher, url: str) -> dict:
    sc = P.parse_scorecard(fetcher.get(url), url)
    result = {
        "scorecard_id": sc.ca_id,
        "title": sc.title,
        "date": sc.match_date,
        "label": sc.match_label,
        "ground": sc.ground,
        "result": sc.result,
        "teams": sc.teams,
        "players_seen": len(sc.players_seen),
        "innings": [],
        "validation": [],
    }

    for idx, comm_url in enumerate(sc.commentary_urls):
        order = (_batting_order_surnames(sc.innings[idx])
                 if idx < len(sc.innings) else None)
        deliveries = P.parse_commentary(fetcher.get(comm_url), comm_url, order)
        runs = sum(d.runs_total for d in deliveries)
        wkts = sum(1 for d in deliveries if d.is_wicket)
        legal = sum(1 for d in deliveries
                    if d.extras_wides == 0 and d.extras_noballs == 0)
        exp_runs = sc.innings[idx].total_runs if idx < len(sc.innings) else None
        exp_wkts = sc.innings[idx].total_wickets if idx < len(sc.innings) else None
        ok = (runs == exp_runs) and (exp_wkts is None or wkts == exp_wkts)
        result["validation"].append({
            "innings": idx + 1,
            "deliveries": len(deliveries),
            "legal_balls": legal,
            "runs": runs, "expected_runs": exp_runs,
            "wickets": wkts, "expected_wickets": exp_wkts,
            "reconciles": ok,
        })
        result["innings"].append({
            "batting_team": sc.innings[idx].batting_team if idx < len(sc.innings) else None,
            "deliveries": [asdict(d) for d in deliveries],
        })
    return result, sc.players_seen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--series", default=DEFAULT_SERIES)
    ap.add_argument("--login", action="store_true",
                    help="use an authenticated session (data pages are public, so off by default)")
    args = ap.parse_args()

    session = auth.ensure_session() if args.login else auth._new_session()
    fetcher = PoliteFetcher(session)

    matches = P.parse_event_matches(fetcher.get(args.series), args.series)
    logger.info("Series has %d scorecards", len(matches))

    all_players: dict[str, str] = {}
    out = {"series": args.series, "matches": []}
    all_ok = True
    for m in matches:
        res, players = process_match(fetcher, m.url)
        all_players.update(players)
        out["matches"].append(res)
        for v in res["validation"]:
            flag = "OK" if v["reconciles"] else "MISMATCH"
            if not v["reconciles"]:
                all_ok = False
            logger.info("  %s inn%d: %d deliveries, %d runs (exp %s), %d wkts (exp %s) -> %s",
                        res["scorecard_id"], v["innings"], v["deliveries"],
                        v["runs"], v["expected_runs"], v["wickets"],
                        v["expected_wickets"], flag)
    out["players_seen"] = all_players

    out_dir = Path(PROCESSED_DATA_DIR) / "cricketarchive"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pilot_output.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 80)
    print(f"Matches processed : {len(matches)}")
    print(f"Unique CA players : {len(all_players)}")
    print(f"All reconcile     : {all_ok}")
    print(f"Output            : {out_path}")
    print("=" * 80)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
