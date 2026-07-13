#!/usr/bin/env python3
"""Discover missing premium-franchise events — Tranche 3 — direct URL probing.

We know CricketArchive's event page format:
  https://cricketarchive.com/Archive/Events/{batch}/{Competition_Name}_{Year}.html

Batch numbers follow a seasonal pattern (observed from existing data):
  27 = 2015/16    28 = 2016    29 = 2016/17    30 = 2017/18
  31 = 2018       32 = 2018/19  33 = 2019/20    34 = 2020/21
  35 = 2021/22    36 = 2022     37 = 2022/23    38 = 2023/24
  39 = 2024/25    40 = 2025/26  41 = 2025/26+

This script probes a ±3 window around the expected batch for each
missing competition, which is fast (cache hits for pages we've seen;
a handful of fresh fetches for new ones).

Coverage gaps targeted
======================
  ILT20        2022/23 – 2025/26    (entirely missing)
  MLC          2023 – 2025          (entirely missing)
  BPL          2012 – 2025          (entirely missing)
  APL          2018 – 2024          (entirely missing)
  GT20 Canada  2018 – 2024          (entirely missing)
  LPL          2020, 2021, 2025     (partial; 2022-2024 already ingested)
  PSL          2017/18, 2018/19, 2020/21, 2022/23, 2024/25  (partial)
  WPL          2023/24              (2022/23, 2024/25, 2025/26 already ingested)
  International T20Is via season index pages

Output
======
  data/raw/cricketarchive/tranche3_events.txt
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CRICKETARCHIVE_CONFIG
from src.api.cricketarchive import auth, parsers as P
from src.api.cricketarchive.fetcher import PoliteFetcher

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ca_discover_t3")

BASE = "https://cricketarchive.com"

# ---------------------------------------------------------------------------
# Candidates: (label, batch_range, filename_stem)
# filename_stem is the part after the batch folder, without .html
# ---------------------------------------------------------------------------
DIRECT_CANDIDATES: list[tuple[str, range, str]] = [
    # ILT20 — starts 2022/23
    ("ILT20 2022/23",   range(36, 40), "International_League_T20_2022-23"),
    ("ILT20 2023/24",   range(37, 41), "International_League_T20_2023-24"),
    ("ILT20 2024/25",   range(38, 42), "International_League_T20_2024-25"),
    ("ILT20 2025/26",   range(39, 43), "International_League_T20_2025-26"),
    # MLC — starts 2023
    ("MLC 2023",        range(36, 40), "Major_League_Cricket_2023"),
    ("MLC 2024",        range(38, 42), "Major_League_Cricket_2024"),
    ("MLC 2025",        range(39, 43), "Major_League_Cricket_2025"),
    # LPL — 2020/21/25 missing; 2022-2024 already ingested
    ("LPL 2020",        range(32, 36), "Lanka_Premier_League_2020"),
    ("LPL 2021",        range(33, 37), "Lanka_Premier_League_2021"),
    ("LPL 2025",        range(39, 43), "Lanka_Premier_League_2025"),
    # BPL — entirely missing
    ("BPL 2012",        range(16, 22), "Bangladesh_Premier_League_2012"),
    ("BPL 2013",        range(18, 24), "Bangladesh_Premier_League_2012-13"),
    ("BPL 2013/14",     range(18, 24), "Bangladesh_Premier_League_2013-14"),
    ("BPL 2015/16",     range(24, 30), "Bangladesh_Premier_League_2015-16"),
    ("BPL 2016/17",     range(27, 32), "Bangladesh_Premier_League_2016-17"),
    ("BPL 2017/18",     range(29, 34), "Bangladesh_Premier_League_2017-18"),
    ("BPL 2019",        range(31, 36), "Bangladesh_Premier_League_2019"),
    ("BPL 2019/20",     range(31, 36), "Bangladesh_Premier_League_2019-20"),
    ("BPL 2022",        range(34, 39), "Bangladesh_Premier_League_2022"),
    ("BPL 2022/23",     range(36, 40), "Bangladesh_Premier_League_2022-23"),
    ("BPL 2023/24",     range(37, 42), "Bangladesh_Premier_League_2023-24"),
    ("BPL 2024/25",     range(38, 43), "Bangladesh_Premier_League_2024-25"),
    # APL — entirely missing
    ("APL 2018",        range(31, 36), "Afghanistan_Premier_League_2018"),
    ("APL 2023",        range(36, 41), "Afghanistan_Premier_League_2023"),
    ("APL 2024",        range(38, 43), "Afghanistan_Premier_League_2024"),
    # GT20 Canada — entirely missing
    ("GT20 2018",       range(31, 36), "Global_T20_Canada_2018"),
    ("GT20 2019",       range(32, 37), "Global_T20_Canada_2019"),
    ("GT20 2022",       range(35, 40), "Global_T20_Canada_2022"),
    ("GT20 2023",       range(37, 41), "Global_T20_Canada_2023"),
    ("GT20 2024",       range(38, 43), "Global_T20_Canada_2024"),
    # PSL gaps
    ("PSL 2017/18",     range(29, 33), "HBL_Pakistan_Super_League_2017-18"),
    ("PSL 2018/19",     range(31, 35), "HBL_Pakistan_Super_League_2018-19"),
    ("PSL 2020/21",     range(33, 37), "HBL_Pakistan_Super_League_2020-21"),
    ("PSL 2022/23",     range(36, 40), "HBL_Pakistan_Super_League_2022-23"),
    ("PSL 2024/25",     range(38, 42), "HBL_Pakistan_Super_League_2024-25"),
    # WPL 2023/24 gap
    ("WPL 2023/24",     range(37, 41), "Womens_Premier_League_2023-24"),
]

# ---------------------------------------------------------------------------
# Season-index based discovery for international T20Is
# ---------------------------------------------------------------------------
BASE_COUNTRIES_T20I = ["ENG", "AUS", "IND", "PAK", "SA", "WI", "NZ", "SRI"]
T20I_YEARS = list(range(2019, 2027))
T20I_RX = re.compile(
    r"(Twenty20 International|T20 International|T20I Series)\b", re.I
)


def _season_urls(country: str, year: int) -> list[str]:
    yy = str(year + 1)[2:]
    return [
        f"{BASE}/Archive/Seasons/{year}-{year+1}_{country}.html",
        f"{BASE}/Archive/Seasons/{year}-{yy}_{country}.html",
        f"{BASE}/Archive/Seasons/{year}_{country}.html",
        f"{BASE}/Archive/Seasons/{year+1}_{country}.html",
    ]


def _find_event_links(html: str, name_rx: re.Pattern) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    found = []
    for a in soup.find_all("a"):
        text = a.get_text(" ", strip=True)
        href = a.get("href") or ""
        if "/Archive/Events/" in href and name_rx.search(text):
            found.append(urljoin(BASE, href))
    return found


def discover_international(fetcher: PoliteFetcher) -> list[tuple[str, str, int]]:
    """Scan season pages for T20I bilateral series links."""
    results: list[tuple[str, str, int]] = []
    seen: set[str] = set()
    for country in BASE_COUNTRIES_T20I:
        for year in T20I_YEARS:
            for url in _season_urls(country, year):
                try:
                    html = fetcher.get(url)
                except Exception:  # noqa: BLE001
                    continue
                links = _find_event_links(html, T20I_RX)
                for link in links:
                    if link in seen:
                        continue
                    seen.add(link)
                    try:
                        n = len(P.parse_event_matches(fetcher.get(link), link))
                    except Exception:  # noqa: BLE001
                        n = 0
                    label = f"T20I/{country}/{year}"
                    logger.info("  %-30s %d  %s", label, n, link)
                    results.append((label, link, n))
                if links:
                    break
    return results


def discover_direct(fetcher: PoliteFetcher) -> list[tuple[str, str, int]]:
    """Probe direct event page URLs for known missing competitions."""
    results: list[tuple[str, str, int]] = []
    seen: set[str] = set()
    for label, batch_range, stem in DIRECT_CANDIDATES:
        found = False
        for batch in batch_range:
            url = f"{BASE}/Archive/Events/{batch}/{stem}.html"
            if url in seen:
                continue
            try:
                html = fetcher.get(url)
                n = len(P.parse_event_matches(html, url))
            except Exception:  # noqa: BLE001
                continue
            if n > 0:
                seen.add(url)
                logger.info("  %-25s %d matches  %s", label, n, url)
                results.append((label, url, n))
                found = True
                break
        if not found:
            logger.debug("  %-25s NOT FOUND in batches %s", label, list(batch_range))
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["direct", "t20i", "all"], default="all",
                    help="'direct' = probe franchise gaps; 't20i' = scan T20I bilaterals; 'all' = both")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        print("Direct probes:")
        for label, br, stem in DIRECT_CANDIDATES:
            print(f"  {label:<25}  batch {br.start}-{br.stop-1}  {stem}.html")
        print(f"\nT20I countries: {BASE_COUNTRIES_T20I} years={T20I_YEARS}")
        return 0

    fetcher = PoliteFetcher(auth._new_session())
    all_results: list[tuple[str, str, int]] = []

    if args.mode in ("direct", "all"):
        logger.info("=== Phase 1: direct franchise URL probing ===")
        all_results.extend(discover_direct(fetcher))

    if args.mode in ("t20i", "all"):
        logger.info("=== Phase 2: T20I bilateral discovery ===")
        all_results.extend(discover_international(fetcher))

    # Write output (append to existing file if it exists)
    out_dir = Path(CRICKETARCHIVE_CONFIG["cache_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tranche3_events.txt"

    existing: set[str] = set()
    if out_path.exists():
        existing = set(out_path.read_text().splitlines())

    good_urls = [u for (_, u, n) in all_results if n > 0 and u not in existing]
    all_good = sorted(existing | set(good_urls))
    out_path.write_text("\n".join(all_good) + ("\n" if all_good else ""))

    print("\n" + "=" * 80)
    print("TRANCHE 3 — DISCOVERY RESULTS")
    print("=" * 80)
    print(f"  {'LABEL':<25} {'N':>6}  URL")
    print("  " + "-" * 70)
    for label, url, n in sorted(all_results, key=lambda x: -x[2]):
        status = "OK  " if n > 0 else "SKIP"
        print(f"  {status} {label:<23} {n:>6}  {url}")
    print("  " + "-" * 70)
    new_count = len(good_urls)
    total_new = sum(n for _, _, n in all_results if n > 0)
    print(f"  New event URLs: {new_count}  |  Total event URLs in file: {len(all_good)}"
          f"  |  New scorecards to harvest: ~{total_new}")
    print(f"  Output: {out_path}")
    print("=" * 80)
    return 0 if good_urls else 1


if __name__ == "__main__":
    sys.exit(main())
