#!/usr/bin/env python3
"""Harvest CricketArchive scorecards (+ ball-by-ball) into the isolated ca_archive.db.

This NEVER touches cricket.db. It walks a series/event page (or a single
scorecard, or a list file), parses each scorecard and its commentary innings,
and writes everything to ca_archive.db. Idempotent per scorecard. Polite +
cached via the shared fetcher, so re-runs hit no network.

Keep harvests BOUNDED — start with a series/season slice, validate, then scale.

Usage:
    venv311/bin/python scripts/ca_harvest.py --series <event_url>
    venv311/bin/python scripts/ca_harvest.py --scorecard <scorecard_url>
    venv311/bin/python scripts/ca_harvest.py --list urls.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.cricketarchive import auth, parsers as P, store, identity as I
from src.api.cricketarchive.fetcher import PoliteFetcher

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ca_harvest")


def harvest_scorecard(conn, fetcher: PoliteFetcher, url: str, cache_only: bool = False) -> dict:
    sc = P.parse_scorecard(fetcher.get(url, cache_only=cache_only), url)
    deliveries_by_innings = {}
    reports_by_innings = {}
    recon = []
    for idx, comm_url in enumerate(sc.commentary_urls):
        if idx >= len(sc.innings):
            continue
        dels = P.parse_commentary(fetcher.get(comm_url, cache_only=cache_only), comm_url)
        dels, report = I.resolve_innings(sc.innings[idx], dels)
        deliveries_by_innings[idx + 1] = dels
        reports_by_innings[idx + 1] = report
        runs = sum(d.runs_total for d in dels)
        wkts = sum(1 for d in dels if d.is_wicket)
        exp_r = sc.innings[idx].total_runs
        exp_w = sc.innings[idx].total_wickets
        recon.append((idx + 1, runs, exp_r, wkts, exp_w,
                      runs == exp_r and (exp_w is None or wkts == exp_w),
                      report["identity_verified"]))
    store.write_scorecard(conn, sc, deliveries_by_innings, reports_by_innings)
    return {"scorecard_id": sc.ca_id, "title": sc.title, "date": sc.match_date,
            "innings": len(sc.innings), "bbb": bool(deliveries_by_innings),
            "recon": recon,
            "identity_ok": all(r[6] for r in recon) if recon else True}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--series", help="event/series page URL")
    g.add_argument("--scorecard", help="single scorecard URL")
    g.add_argument("--list", help="file with one scorecard URL per line")
    ap.add_argument("--cache-only", action="store_true",
                    help="re-parse only already-cached pages (no network); for "
                         "applying parser fixes offline and skipping missing pages")
    args = ap.parse_args()

    fetcher = PoliteFetcher(auth._new_session())  # data pages are public

    def expand(url: str) -> list[str]:
        """Event/series URL -> its scorecard URLs; scorecard URL -> itself."""
        if "/Archive/Events/" in url:
            try:
                ms = P.parse_event_matches(fetcher.get(url, cache_only=args.cache_only), url)
                logger.info("Series %s -> %d scorecards", url.rsplit("/", 1)[-1], len(ms))
                return [m.url for m in ms]
            except Exception as e:  # noqa: BLE001
                logger.error("Failed to expand series %s: %s", url, e)
                return []
        return [url]

    if args.series:
        urls = expand(args.series)
    elif args.scorecard:
        urls = [args.scorecard]
    else:
        lines = [ln.strip() for ln in Path(args.list).read_text().splitlines() if ln.strip()]
        urls = []
        for ln in lines:
            urls.extend(expand(ln))

    logger.info("Harvesting %d scorecard(s) into ca_archive.db%s",
                len(urls), " [cache-only]" if args.cache_only else "")
    n_ok = n_bbb = n_skip = n_fail = 0
    from src.api.cricketarchive.fetcher import CacheMissError
    with store.get_connection() as conn:
        store.init_db(conn)
        for u in urls:
            try:
                res = harvest_scorecard(conn, fetcher, u, cache_only=args.cache_only)
                conn.commit()
                recon_ok = all(r[5] for r in res["recon"]) if res["recon"] else True
                n_ok += 1
                n_bbb += 1 if res["bbb"] else 0
                if not res["bbb"]:
                    flag = "no-bbb"
                elif recon_ok and res["identity_ok"]:
                    flag = "bbb-OK+id"
                elif recon_ok:
                    flag = "bbb-OK,id-UNVERIFIED"
                else:
                    flag = "bbb-MISMATCH"
                logger.info("  %s %s (%s) [%s]", res["scorecard_id"], res["title"], res["date"], flag)
            except CacheMissError:
                n_skip += 1  # cache-only mode: page not cached, skip quietly
            except RuntimeError as e:
                if "cap" in str(e).lower():     # daily request cap -> stop cleanly (resumable)
                    logger.warning("Daily request cap reached — stopping cleanly; "
                                   "re-run to resume. (%s)", e)
                    break
                n_fail += 1
                logger.error("  FAILED %s -> %s: %s", u, type(e).__name__, e)
            except Exception as e:  # noqa: BLE001
                n_fail += 1
                logger.error("  FAILED %s -> %s: %s", u, type(e).__name__, e)

    print("\n" + "=" * 70)
    print(f"Harvested: {n_ok}/{len(urls)} scorecards ({n_bbb} with ball-by-ball)"
          + (f"  skipped(uncached): {n_skip}" if n_skip else "")
          + (f"  failed: {n_fail}" if n_fail else ""))
    print("Store    : ca_archive.db")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
