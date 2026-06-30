#!/usr/bin/env python3
"""Discover ALL T20 scorecards on CricketArchive (men's; optional women's).

CA exposes a per-country-per-season list of every T20 at
``/Archive/Seasons/Seasonal_Averages/{COUNTRY}/{SEASON}_tt_Match_List.html``
(``_wtt_`` for women's). This enumerator iterates countries x seasons, tries the
plausible season-token formats (calendar "2023" and split "2022-23"/"2023-24"),
collects every scorecard URL, dedupes, and writes a seed file for ca_harvest.py.

It is polite (shared hardened fetcher: rate-limited, 403 backoff, disk cache),
daily-capped, and RESUMABLE: completed (country, year) pairs are tracked so a
re-run continues where it stopped. Invalid combos 404 and are skipped.

SCALE: full members alone span ~20-30k T20s over 2003-2026; "all" (incl.
associates + women's) is larger. This only discovers URLs; the scorecard harvest
(ca_harvest.py --list) is the long pole.

Usage:
    venv311/bin/python scripts/ca_discover_all_t20.py --countries full --start 2008 --end 2026
    venv311/bin/python scripts/ca_discover_all_t20.py --countries all --include-women
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from urllib.parse import urljoin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CRICKETARCHIVE_CONFIG
from src.api.cricketarchive import auth
from src.api.cricketarchive.fetcher import PoliteFetcher

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ca_all_t20")

SEASON_BASE = "https://cricketarchive.com/Archive/Seasons/"

FULL_MEMBERS = ["IND", "AUS", "ENG", "RSA", "PAK", "NZ", "SL", "WI", "BAN", "ZIM", "IRE", "AFG"]
MAJOR_ASSOCIATES = ["NED", "SCO", "UAE", "NEP", "USA", "CAN", "HK", "OMA", "PNG", "NAM",
                    "KEN", "JER", "GUE", "BER", "ITA", "GER", "DEN", "SGP", "MAS", "THA",
                    "QAT", "KSA", "BHR", "KWT", "KUW", "KOR", "JPN", "KEN", "KEN"]
# de-dup while preserving order
def _uniq(seq):
    seen = set(); out = []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


def _season_tokens(year: int) -> list:
    yy = str(year)[2:]; pyy = str(year - 1)[2:]
    return [str(year), f"{year-1}-{yy}", f"{year}-{str(year+1)[2:]}",
            f"{year-1}-{year}", f"{year}-{year+1}"]


def _match_list_url(country: str, token: str, women: bool) -> str:
    kind = "wtt" if women else "tt"
    return urljoin(SEASON_BASE, f"Seasonal_Averages/{country}/{token}_{kind}_Match_List.html")


SCORECARD_RX = re.compile(r"/Archive/Scorecards/\d+/\d+\.html")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--countries", default="full", help="'full' | 'all' | comma list of codes")
    ap.add_argument("--start", type=int, default=2008)
    ap.add_argument("--end", type=int, default=2026)
    ap.add_argument("--include-women", action="store_true")
    ap.add_argument("--out", default=str(Path(CRICKETARCHIVE_CONFIG["cache_dir"]) / "all_t20_events.txt"))
    args = ap.parse_args()

    if args.countries == "full":
        countries = FULL_MEMBERS
    elif args.countries == "all":
        countries = _uniq(FULL_MEMBERS + MAJOR_ASSOCIATES)
    else:
        countries = [c.strip().upper() for c in args.countries.split(",") if c.strip()]

    cache_dir = Path(CRICKETARCHIVE_CONFIG["cache_dir"])
    progress_path = cache_dir / "all_t20_discovery_progress.json"
    out_path = Path(args.out)
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"done": [], "counts": {}}
    done = set(tuple(x) for x in progress["done"])
    scorecards = set()
    if out_path.exists():
        scorecards.update(l.strip() for l in out_path.read_text().splitlines() if l.strip())

    fetcher = PoliteFetcher(auth._new_session())
    kinds = [False, True] if args.include_women else [False]

    try:
        for country in countries:
            for year in range(args.start, args.end + 1):
                for women in kinds:
                    key = (country, year, women)
                    if list(key) in progress["done"] or key in done:
                        continue
                    found_token = None; n = 0
                    for token in _season_tokens(year):
                        url = _match_list_url(country, token, women)
                        try:
                            html = fetcher.get(url)
                        except Exception:  # noqa: BLE001 (404/throttle-exhausted -> skip token)
                            continue
                        cards = set(urljoin("https://cricketarchive.com", m)
                                    for m in SCORECARD_RX.findall(html))
                        if cards:
                            scorecards |= cards; found_token = token; n = len(cards)
                            break
                    done.add(key)
                    progress["done"].append(list(key))
                    if found_token:
                        progress["counts"][f"{country}/{year}/{'w' if women else 'm'}"] = n
                        logger.info("%s %s %s -> %d (token %s)  [running total %d]",
                                    country, year, "W" if women else "M", n, found_token, len(scorecards))
                    # checkpoint every few
                    if len(progress["done"]) % 10 == 0:
                        progress_path.write_text(json.dumps(progress))
                        out_path.write_text("\n".join(sorted(scorecards)) + "\n")
    finally:
        progress_path.write_text(json.dumps(progress))
        out_path.write_text("\n".join(sorted(scorecards)) + "\n")

    total = sum(progress["counts"].values())
    print("\n" + "=" * 72)
    print(f"ALL-T20 DISCOVERY  countries={args.countries} years={args.start}-{args.end} "
          f"women={args.include_women}")
    print(f"  (country,year,kind) probed: {len(done)}   with matches: {len(progress['counts'])}")
    print(f"  unique T20 scorecards discovered: {len(scorecards)}   (sum per-season {total})")
    print(f"  seed -> {out_path}")
    print(f"  progress -> {progress_path} (resumable)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
