#!/usr/bin/env python3
"""Parallel CricketArchive harvester — N worker threads for faster data collection.

Design:
* Each worker thread has its own PoliteFetcher (own requests.Session, own throttle
  state). Threads are fully independent; the disk cache is the coordination mechanism
  — a URL that one thread already cached is served instantly to any other.
* A shared threading.Lock serialises writes to the daily request-counter file so the
  cap is correctly enforced across all workers combined.
* DB writes are serialised in the main thread (fetching is the bottleneck, not writes).
* Idempotent: already-harvested scorecards are skipped by the store (upsert). Already-
  fetched pages are served from disk (no network). Safe to interrupt and restart.

Throughput (rough): N workers × 1/(avg_delay_s) ≈ 3 workers × 1/4.5s ≈ 40 req/min.
The serial harvester at the same delay does ≈ 13 req/min — roughly 3× speedup.

Usage:
    # Kill the running serial harvest first, then:
    CRICKETARCHIVE_MIN_DELAY=3 CRICKETARCHIVE_MAX_DELAY=6 \\
    CRICKETARCHIVE_MAX_PER_DAY=20000 \\
    venv311/bin/python scripts/ca_harvest_parallel.py \\
        --list data/raw/cricketarchive/all_t20_events.txt \\
        --workers 3 \\
        2>&1 | tee data/raw/cricketarchive/harvest_parallel.log
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import sqlite3
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.cricketarchive import auth, parsers as P, store, identity as I
from src.api.cricketarchive.fetcher import PoliteFetcher, CacheMissError

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ca_harvest_parallel")

# Thread-local storage so each thread gets exactly one PoliteFetcher (created lazily).
_thread_local = threading.local()
# Set by main() before the pool starts; all worker threads read this.
_shared_counter_lock: threading.Lock | None = None


def _get_fetcher() -> PoliteFetcher:
    """Return (or create) the PoliteFetcher for the calling thread."""
    if not hasattr(_thread_local, "fetcher"):
        _thread_local.fetcher = PoliteFetcher(
            auth._new_session(),
            counter_lock=_shared_counter_lock,
        )
        logger.debug("Thread %s: new PoliteFetcher created", threading.current_thread().name)
    return _thread_local.fetcher


def _expand_url(fetcher: PoliteFetcher, url: str) -> list[str]:
    """Event URL → list of scorecard URLs; scorecard URL → [itself]."""
    if "/Archive/Events/" in url:
        try:
            ms = P.parse_event_matches(fetcher.get(url), url)
            return [m.url for m in ms]
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to expand event %s: %s", url, e)
            return []
    return [url]


# ---------------------------------------------------------------------------
# Worker: fetch + parse (no DB access)
# ---------------------------------------------------------------------------

def _fetch_and_parse(url: str) -> dict:
    """Fetch + parse one scorecard URL. Returns a result dict for the main thread.

    Never touches the database. Returns one of:
        {"ok": True,  "sc": ..., "deliveries": ..., "reports": ..., ...}
        {"skip": True, "url": url}        # cache_only miss (not used here, safety net)
        {"cap": True}                      # daily cap hit
        {"error": str, "url": url}         # any other failure
    """
    fetcher = _get_fetcher()
    try:
        sc = P.parse_scorecard(fetcher.get(url), url)
        deliveries_by_innings: dict[int, list] = {}
        reports_by_innings: dict[int, dict] = {}
        recon: list[tuple] = []
        for idx, comm_url in enumerate(sc.commentary_urls):
            if idx >= len(sc.innings):
                continue
            dels = P.parse_commentary(fetcher.get(comm_url), comm_url)
            dels, report = I.resolve_innings(sc.innings[idx], dels)
            deliveries_by_innings[idx + 1] = dels
            reports_by_innings[idx + 1] = report
            runs = sum(d.runs_total for d in dels)
            wkts = sum(1 for d in dels if d.is_wicket)
            exp_r = sc.innings[idx].total_runs
            exp_w = sc.innings[idx].total_wickets
            recon.append((
                idx + 1, runs, exp_r, wkts, exp_w,
                runs == exp_r and (exp_w is None or wkts == exp_w),
                report["identity_verified"],
            ))
        return {
            "ok": True,
            "sc": sc,
            "deliveries": deliveries_by_innings,
            "reports": reports_by_innings,
            "scorecard_id": sc.ca_id,
            "title": sc.title,
            "date": sc.match_date,
            "bbb": bool(deliveries_by_innings),
            "recon": recon,
            "identity_ok": all(r[6] for r in recon) if recon else True,
        }
    except CacheMissError:
        return {"skip": True, "url": url}
    except RuntimeError as e:
        if "cap" in str(e).lower():
            return {"cap": True}
        return {"error": str(e), "url": url}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "url": url}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    global _shared_counter_lock

    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", help="file with one event/scorecard URL per line")
    g.add_argument("--scorecard", help="single scorecard URL")
    ap.add_argument("--workers", type=int, default=3,
                    help="number of parallel fetcher threads (default: 3)")
    args = ap.parse_args()

    _shared_counter_lock = threading.Lock()

    # ---- Phase 1: expand event URLs → flat scorecard list (fast, mostly cached) ---
    expand_fetcher = PoliteFetcher(auth._new_session(), counter_lock=_shared_counter_lock)

    if args.scorecard:
        lines = [args.scorecard]
    else:
        lines = [ln.strip() for ln in Path(args.list).read_text().splitlines()
                 if ln.strip() and not ln.startswith("#")]

    logger.info("Expanding %d event/scorecard URLs (mostly cached) ...", len(lines))
    scorecard_urls: list[str] = []
    for ln in lines:
        scorecard_urls.extend(_expand_url(expand_fetcher, ln))
    logger.info(
        "Expanded to %d scorecard URLs — starting %d workers",
        len(scorecard_urls), args.workers,
    )

    # ---- Phase 2: parallel fetch+parse, serial DB write -----------------------
    n_ok = n_bbb = n_fail = n_skip = 0
    capped = False

    # Open a single long-lived connection (all writes from main thread — no threading issues).
    from config import CRICKETARCHIVE_CONFIG
    db_path = CRICKETARCHIVE_CONFIG["archive_db_path"]
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # allows readers while writing
    store.init_db(conn)

    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers,
            thread_name_prefix="ca-worker",
        ) as pool:
            futures = {pool.submit(_fetch_and_parse, u): u for u in scorecard_urls}
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()

                if res.get("cap"):
                    logger.warning(
                        "Daily request cap reached — stopping cleanly. "
                        "Re-run tomorrow (or raise CRICKETARCHIVE_MAX_PER_DAY) to resume."
                    )
                    capped = True
                    # Cancel remaining futures so threads wind down politely.
                    for f in futures:
                        f.cancel()
                    break

                if res.get("skip"):
                    n_skip += 1
                    continue

                if "error" in res:
                    n_fail += 1
                    logger.error("  FAILED %s: %s", res["url"], res["error"])
                    continue

                # Serial DB write (fast — not the bottleneck).
                store.write_scorecard(conn, res["sc"], res["deliveries"], res["reports"])
                conn.commit()
                n_ok += 1
                n_bbb += 1 if res["bbb"] else 0

                recon_ok = all(r[5] for r in res["recon"]) if res["recon"] else True
                if not res["bbb"]:
                    flag = "no-bbb"
                elif recon_ok and res["identity_ok"]:
                    flag = "bbb-OK+id"
                elif recon_ok:
                    flag = "bbb-OK,id-UNVERIFIED"
                else:
                    flag = "bbb-MISMATCH"

                logger.info(
                    "  %s %s (%s) [%s]",
                    res["scorecard_id"], res["title"], res["date"], flag,
                )
    finally:
        conn.close()

    sep = "=" * 70
    print(f"\n{sep}")
    total = len(scorecard_urls)
    print(
        f"Harvested : {n_ok}/{total} scorecards  ({n_bbb} with ball-by-ball)"
        + (f"  failed: {n_fail}" if n_fail else "")
        + (f"  skipped(uncached): {n_skip}" if n_skip else "")
        + ("  [CAP REACHED — resume tomorrow]" if capped else "")
    )
    print(f"Workers   : {args.workers}")
    print(f"Store     : {db_path}")
    print(sep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
