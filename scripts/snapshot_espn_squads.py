#!/usr/bin/env python3
"""Snapshot ESPN squads for upcoming limited-overs matches, and the played XI
of recently finished ones, into data/espn_squads/ (see espn_squads.py).

ESPN drops squads once a match starts, so this has to run ahead of matches;
the LaunchAgent com.indiasdad.espn-squads runs it every 3 hours. One summary
request per event, 0.5 s apart. Never writes to cricket.db.

    python scripts/snapshot_espn_squads.py [--days-ahead 3] [--days-back 3]
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.integrations.polymarket import espn_fixture_venues as efv  # noqa: E402
from src.integrations.polymarket import espn_squads as es  # noqa: E402

# ESPN gives some 10-over leagues (e.g. Abu Dhabi T10) eventType ODI.
TEN_OVER = re.compile(r"\bT10\b|10-over|ten-over", re.I)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("snapshot_espn_squads")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days-ahead", type=int, default=3)
    ap.add_argument("--days-back", type=int, default=3)
    ap.add_argument("--root", default=str(es.SNAPSHOT_DIR))
    args = ap.parse_args()
    root = Path(args.root)
    now = datetime.now(timezone.utc)
    today = date.today()

    events = {}
    for k in range(-args.days_back, args.days_ahead + 1):
        for ev in efv.fetch_scorepanel(today + timedelta(days=k)):
            events.setdefault(ev["espn_id"], ev)

    import requests
    s = requests.Session()
    n_sq = n_xi = n_req = 0
    for eid, ev in events.items():
        fmt = efv.event_format(ev)
        if fmt == "other" or efv.event_is_youth(ev) or TEN_OVER.search(f"{ev.get('league')} {ev.get('name')}"):
            continue
        st = efv.parse_utc(ev.get("start"))
        if st is None:
            continue
        rec = es.load_event(eid, root)
        upcoming = st > now
        if not upcoming and (rec is None or rec.get("played_xi")):
            continue
        if not upcoming and now - st < timedelta(hours=4):
            continue
        summary = es.fetch_summary(ev["league_id"], eid, session=s)
        n_req += 1
        time.sleep(0.5)
        if summary is None:
            continue
        rec = rec or {"espn_id": eid, "league_id": ev["league_id"], "league": ev.get("league"),
                      "teams": ev.get("teams"), "class": ev.get("class"), "fmt": fmt,
                      "gender": efv.event_gender(ev), "snapshots": []}
        rec["start"] = st.strftime("%Y-%m-%dT%H:%M:%SZ")
        if upcoming:
            sq = es.parse_squads(summary)
            if not sq:
                continue
            last = rec["snapshots"][-1]["squads"] if rec["snapshots"] else None
            if sq != last:
                rec["snapshots"].append({"fetched_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "squads": sq})
                n_sq += 1
            es.save_event(rec, root)
        else:
            xi = es.parse_played_xi(summary)
            if xi:
                rec["played_xi"] = xi
                rec["played_xi_fetched_at"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
                es.save_event(rec, root)
                n_xi += 1
    logger.info(f"{len(events)} events, {n_req} summaries, {n_sq} new squad snapshots, {n_xi} played XIs")


if __name__ == "__main__":
    main()
