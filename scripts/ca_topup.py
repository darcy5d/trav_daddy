#!/usr/bin/env python3
"""Incremental top-up: find and harvest T20 scorecards played since our last crawl.

Unlike the full discovery script, this tool:
  - Re-probes every country for the specified seasons (default: current + prior year)
  - IGNORES the existing progress cache so newly played matches are always found
  - Diffs discovered URLs against ca_archive.db to find only genuinely new ones
  - Optionally harvests them immediately (--harvest)

Run this weekly (or after any major tournament) to keep the DB current.

Usage:
    # Discover + harvest everything from the last 2 years
    venv311/bin/python scripts/ca_topup.py --harvest

    # Just discover, write new URLs to file
    venv311/bin/python scripts/ca_topup.py --out data/raw/cricketarchive/topup_new.txt

    # Specific years only
    venv311/bin/python scripts/ca_topup.py --years 2026 --harvest

    # Single country (fast check)
    venv311/bin/python scripts/ca_topup.py --countries ENG,AUS --years 2026 --harvest
"""

from __future__ import annotations

import argparse
import logging
import re
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CRICKETARCHIVE_CONFIG
from src.api.cricketarchive import auth
from src.api.cricketarchive.fetcher import PoliteFetcher

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ca_topup")

SEASON_BASE = "https://cricketarchive.com/Archive/Seasons/"
SCORECARD_RX = re.compile(r"/Archive/Scorecards/\d+/\d+\.html")

# Full country list — same as ca_discover_all_t20.py
ALL_COUNTRIES = [
    # Full members
    "IND", "AUS", "ENG", "RSA", "PAK", "NZ", "SL", "WI", "BAN", "ZIM", "IRE", "AFG",
    # Associates with active T20 programmes
    "NED", "SCO", "UAE", "NEP", "USA", "CAN", "HK", "OMA", "PNG", "NAM",
    "KEN", "JER", "GUE", "BER", "ITA", "GER", "DEN", "SGP", "MAS", "THA",
    "QAT", "KSA", "BHR", "KWT", "KOR", "JPN",
    "UGA", "TAN", "NIG", "GHA", "RWA", "BOT", "ZAM", "SLE", "CMR",
    "ETH", "LES", "MAU", "MOZ", "SWZ", "MLW", "CIV", "SEN",
    "BAH", "CAY", "BLZ", "ARG", "BRA", "CHI", "PAN", "MEX", "TCI",
    "SUR", "CRC", "VIN", "LCA", "GRN", "ANT", "DOM",
    "MLD", "BHU", "MYA", "PHI", "KUW",
    "SAM", "VAN", "FIJ", "COK", "TON", "INS",
    "NOR", "AUT", "FRA", "SPN", "ROM", "BEL", "GIB", "IOM",
    "CZE", "SWE", "FIN", "POR", "GRC", "CYP", "MLT", "TUR",
    "EST", "LAT", "LTU", "HUN", "POL", "SVK", "CRO", "BUL", "SRB",
]


def _season_tokens(year: int) -> list[str]:
    yy = str(year)[2:]
    pyy = str(year - 1)[2:]
    return [
        str(year),
        f"{year - 1}-{yy}",
        f"{year}-{str(year + 1)[2:]}",
        f"{year - 1}-{year}",
        f"{year}-{year + 1}",
    ]


def _match_list_url(country: str, token: str, women: bool) -> str:
    kind = "wtt" if women else "tt"
    return urljoin(SEASON_BASE, f"Seasonal_Averages/{country}/{token}_{kind}_Match_List.html")


def discover(countries: list[str], years: list[int], include_women: bool,
             fetcher: PoliteFetcher) -> set[str]:
    """Return all scorecard URLs found across countries × years."""
    discovered: set[str] = set()
    kinds = [False, True] if include_women else [False]
    total = len(countries) * len(years) * len(kinds)
    done = 0

    for country in countries:
        for year in years:
            for women in kinds:
                done += 1
                label = f"{country}/{year}/{'W' if women else 'M'}"
                found = False
                for token in _season_tokens(year):
                    url = _match_list_url(country, token, women)
                    try:
                        html = fetcher.get(url)
                        cards = set(
                            urljoin("https://cricketarchive.com", m)
                            for m in SCORECARD_RX.findall(html)
                        )
                        if cards:
                            discovered |= cards
                            logger.info("%s -> %d scorecards (token %s)  [%d/%d]",
                                        label, len(cards), token, done, total)
                            found = True
                            break
                    except Exception:
                        continue
                if not found:
                    logger.debug("%s -> no matches", label)

    return discovered


def get_existing_urls(db_path: Path) -> set[str]:
    if not db_path.exists():
        return set()
    conn = sqlite3.connect(str(db_path))
    urls = set(r[0] for r in conn.execute("SELECT source_url FROM ca_matches").fetchall()
               if r[0])
    conn.close()
    return urls


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", default=None,
                    help="Comma-separated years to probe, e.g. '2026' or '2025,2026' "
                         "(default: current year + prior year)")
    ap.add_argument("--countries", default="all",
                    help="'all' | comma list of 3-letter codes")
    ap.add_argument("--include-women", action="store_true", default=True,
                    help="Include women's matches (default: True)")
    ap.add_argument("--no-women", action="store_true",
                    help="Skip women's matches")
    ap.add_argument("--harvest", action="store_true",
                    help="Harvest newly discovered scorecards immediately")
    ap.add_argument("--out", default=None,
                    help="Write new scorecard URLs to this file "
                         "(default: data/raw/cricketarchive/topup_YYYYMMDD.txt)")
    ap.add_argument("--db", default=None,
                    help="Path to ca_archive.db (auto-detected if not set)")
    args = ap.parse_args()

    # ── Resolve years ─────────────────────────────────────────────
    now = datetime.now()
    if args.years:
        years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
    else:
        years = [now.year - 1, now.year]

    # ── Resolve countries ─────────────────────────────────────────
    if args.countries == "all":
        countries = ALL_COUNTRIES
    else:
        countries = [c.strip().upper() for c in args.countries.split(",") if c.strip()]

    include_women = args.include_women and not args.no_women

    # ── Resolve DB path ───────────────────────────────────────────
    proj = Path(__file__).resolve().parent.parent
    if args.db:
        db_path = Path(args.db)
    else:
        # Try both locations
        for candidate in [proj / "ca_archive.db",
                          proj / "data" / "processed" / "ca_archive.db"]:
            if candidate.exists():
                db_path = candidate
                break
        else:
            db_path = proj / "ca_archive.db"

    # ── Output path ───────────────────────────────────────────────
    cache_dir = Path(CRICKETARCHIVE_CONFIG["cache_dir"])
    out_path = Path(args.out) if args.out else \
        cache_dir / f"topup_{now.strftime('%Y%m%d')}.txt"

    logger.info("Top-up config:")
    logger.info("  Years     : %s", years)
    logger.info("  Countries : %d", len(countries))
    logger.info("  Women     : %s", include_women)
    logger.info("  DB        : %s", db_path)
    logger.info("  Out       : %s", out_path)

    # ── Discover ──────────────────────────────────────────────────
    fetcher = PoliteFetcher(auth._new_session())
    discovered = discover(countries, years, include_women, fetcher)
    logger.info("Discovery complete: %d total scorecards found", len(discovered))

    # ── Diff against DB ───────────────────────────────────────────
    existing = get_existing_urls(db_path)
    new_urls = sorted(discovered - existing)
    logger.info("Already in DB: %d  |  New: %d", len(existing), len(new_urls))

    if not new_urls:
        logger.info("DB is up to date — nothing to harvest.")
        return 0

    out_path.write_text("\n".join(new_urls) + "\n")
    logger.info("Wrote %d new URLs to %s", len(new_urls), out_path)

    # ── Optionally harvest ────────────────────────────────────────
    if args.harvest:
        logger.info("Harvesting %d new scorecards...", len(new_urls))
        result = subprocess.run(
            [sys.executable, str(proj / "scripts" / "ca_harvest_parallel.py"),
             "--list", str(out_path)],
            cwd=str(proj),
        )
        if result.returncode != 0:
            logger.error("Harvest exited with code %d", result.returncode)
            return result.returncode
        logger.info("Harvest complete.")
    else:
        logger.info("Run with --harvest to ingest these scorecards, or:")
        logger.info("  venv311/bin/python scripts/ca_harvest_parallel.py --list %s", out_path)

    print("\n" + "=" * 72)
    print(f"TOP-UP COMPLETE  {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"  New scorecards found : {len(new_urls)}")
    print(f"  DB before            : {len(existing)}")
    print(f"  DB after (est.)      : {len(existing) + len(new_urls)}")
    print(f"  URL list             : {out_path}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
