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
import os
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
from src.api.cricketarchive.fetcher import NotAuthenticatedError, PoliteFetcher

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ca_topup")

SEASON_BASE = "https://cricketarchive.com/Archive/Seasons/"
SCORECARD_RX = re.compile(r"/Archive/Scorecards/\d+/\d+\.html")

# Countries probed by the top-up.
#
# These are CricketArchive's own country slugs, NOT IOC/ISO codes. The site puts
# the slug in the URL it publishes for each country:
#   Archive/Seasons/2026_BDESH.html  ->  Seasonal_Averages/BDESH/2026_tt_Match_List.html
# On 7-8 Nov 2026 the list still held ISO codes, so 76 of 107 countries 404'd on
# every list and silently contributed nothing. Every code below was re-checked
# against the site; the audit and the old -> new table are in the commit that
# introduced this block.
#
# Rules for editing:
#   * VERIFIED_COUNTRIES holds codes that have been seen to return cards. A wrong
#     code there is silent - it just yields nothing, so never guess.
#   * Never delete a country to "clean up": dropping one is an invisible coverage
#     hole. A code we could not resolve moves to UNVERIFIED_COUNTRIES.
#   * No duplicates, and no code in both blocks.
VERIFIED_COUNTRIES = [
    # ── Full members ──────────────────────────────────────────────
    "IND", "AUS", "ENG", "RSA", "PAK", "NZ", "SL", "WI", "BDESH", "ZIM",
    "IRELAND", "AFG",
    # ── Associates with active T20 programmes (all verified) ──────
    "NL", "SCOT", "UAE", "NEP", "USA", "CAN", "HKG", "OMAN", "PNG", "NAMIB",
    "KENYA", "JERSEY", "GUERN", "BMDA", "ITALY", "GERMANY", "DNMRK", "SPORE",
    "MAL", "THAI", "QTR", "BAH", "SKOREA", "JAPAN", "UGAN", "TZN", "NGA",
    "GHA", "RWN", "BOT", "SWAZ", "MLW", "FIJI", "CAY", "ARG", "BRAZIL",
    "CHI", "PNM", "MEXICO", "CRICA", "BHU", "INA",
    "NOR", "AUSTRIA", "SPAIN", "ROM", "BELGIUM", "GIBR", "IOM", "CZK",
    "SWE", "FIN", "PORT", "GREECE", "CYP", "MALTA", "EST", "HUN", "SLOV", "CRT",
    "BULG", "SER",
]

# Probed by the top-up but NOT proven to return cards. Kept so the coverage gap
# stays visible instead of disappearing. Probed exactly like the verified codes;
# a country that starts publishing a list starts working again with no edit.
UNVERIFIED_COUNTRIES = [
    # ── No T20 programme found on the archive ─────────────────────
    # Real cricket countries, but their Seasonal_Averages T20 list 404s for
    # 2024-2026 and their season index page has no T20 link: either the archive
    # has no T20 for them or the code is wrong. Kept so the gap stays visible.
    "KSA",   # UNVERIFIED - Saudi Arabia is not in the season index at all
    "SLE", "CMR", "ETH", "LES", "MAU", "MOZ", "CIV", "SEN",  # UNVERIFIED
    "BLZ", "TCI", "VIN", "LCA", "GRN", "ANT", "DOM",  # UNVERIFIED
    "MYA", "SAM", "VAN", "COK", "TON", "LAT", "LTU", "POL", "SVK", "TUR",  # UNVERIFIED
    # ── Dead slugs: no T20 list published under any token ─────────
    # Re-checked 8 Nov 2026. These return HTTP 404 on every Seasonal_Averages
    # token _season_tokens() generates for 2025-2027, for both tt and wtt, and
    # Archive/Seasons/<year>_<SLUG>.html does not exist either, so the archive
    # publishes no list for the country at all. The pre-correction codes we used
    # before (ZAM, SUR, MLD, KUW, FRA) are equally dead, which is what proves
    # this is not simply another wrong-slug mistake.
    #
    # They are NOT dark, though: their matches are still discovered through
    # OPPONENTS' lists, which is how ca_archive.db already holds Zambia (22 men,
    # newest 2026-10-25), Kuwait (30 men + 24 women), France (23), Philippines
    # (19 + 24), Maldives (18) and Suriname (14). Probing them directly therefore
    # costs 6 countries x 8 tokens x 2 genders of 404s per run and finds nothing
    # the opponents do not already give us. They stay listed here so the probed
    # set stays complete and someone re-checking coverage can see they were
    # considered, not forgotten.
    "ZAMBIA", "SURINAM", "MDV", "FRANCE", "KWT", "PHI",  # UNVERIFIED
]

# Everything the top-up probes, in one order. Built from the two blocks above so
# the split cannot drift out of sync with what is actually fetched.
ALL_COUNTRIES = VERIFIED_COUNTRIES + UNVERIFIED_COUNTRIES


# Season tokens to try for each year, in priority order.
#
# `year` is the "end" year of a season: 2025 means "season ending in 2025", so
# the archive's own spellings for it are 2025 (calendar), 2024-25 and 2025-26.
#
# MEASURED 8 Nov 2026, two independent ways. "Live" is the authoritative one:
# every token was requested over the network for 25 countries (full members,
# associates, the codes corrected on 7-8 Nov, and the six dead slugs as dud
# controls), both genders, 400 requests. The cache column counts every one of
# the 106 country slugs whose page was ever fetched successfully with cards.
#
#   token      live: countries yielding (of 25)   cache: 200+cards (of 106)
#   2025       tt 14, wtt  9                      tt 36, wtt 21
#   2024-25    tt 16, wtt 15                      tt 27, wtt 24
#   2025-26    tt 12, wtt 13                      tt 54, wtt 42
#   2026       tt 14, wtt 12                      tt 34, wtt 23
#   2026-27    tt 10, wtt  6                      tt 12, wtt  7
#   2024-2025  tt  0, wtt  0                      tt  0, wtt  0   PRUNED
#   2025-2026  tt  0, wtt  0                      tt  0, wtt  0   PRUNED
#   2026-2027  tt  0, wtt  0                      tt  0, wtt  0   PRUNED
#
# Every live 200 carried cards, so the 50 probes of each pruned token were 50
# clean 404s. The gap backfill that same day is a third confirmation: it fetched
# PNM/NEP/PNG/TZN with force=True across all five old tokens, and again the only
# hits were 2025, 2024-25, 2025-26, 2026 and 2026-27.
#
# The pruned spellings are the season written out in full at both ends. They are
# not merely rare: the archive's own season index publishes only 2024, 2024-25,
# 2025, 2025-26, 2026 and 2026-27, so that spelling is not in its URL namespace
# at all. Dropping them removes 3 of 8 requests per (country, year, gender),
# about 37% of discovery wall-clock, without touching a list that ever returned
# cards. Tests pin this exact set for both years.
#
# Do not shrink this further without new measurements. 2026-27 looks like a
# fringe spelling but carries a whole season for several countries: in the 8 Nov
# gap backfill NEP/2026/M got 2 cards from token 2026 and 15 more from 2026-27,
# and IND/AUS/RSA 2026-27 lists are full of matches.
TOKENS_BY_YEAR = {
    2025: ["2025", "2024-25", "2025-26"],
    2026: ["2026", "2025-26", "2026-27"],
}


def _season_tokens(year: int) -> list[str]:
    """Season tokens to try for `year`.

    Any year outside TOKENS_BY_YEAR still gets the full 5-token spread, because
    the pruned spellings were only shown to be dead for 2025 and 2026. The
    discover() loop tries every token it is given and never stops early: the
    first token with cards is not the whole season.
    """
    if year in TOKENS_BY_YEAR:
        return list(TOKENS_BY_YEAR[year])
    return [
        str(year),
        f"{year - 1}-{str(year)[2:]}",
        f"{year}-{str(year + 1)[2:]}",
        f"{year - 1}-{year}",
        f"{year}-{year + 1}",
    ]


def _match_list_url(country: str, token: str, women: bool) -> str:
    kind = "wtt" if women else "tt"
    return urljoin(SEASON_BASE, f"Seasonal_Averages/{country}/{token}_{kind}_Match_List.html")


class ArchiveAccessError(RuntimeError):
    """Raised when a match list is rejected, so the weekly job stops instead of crawling the paywall."""


def other_topup_pids(ps_output: str, my_pid: int, parent_pid: int) -> list[int]:
    """Python ca_topup processes other than this one and the shell that launched it."""
    found: list[int] = []
    for line in ps_output.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        cmd = parts[2]
        if pid in (my_pid, parent_pid) or ppid == my_pid:
            continue
        if "ca_topup.py" not in cmd:
            continue
        exe = Path(cmd.split(None, 1)[0]).name.lower()
        if "python" in exe:
            found.append(pid)
    return found


def uniq_in_order(seq):
    """De-duplicate while keeping first-seen order, for stable warning text."""
    seen = set()
    out = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def discover(countries: list[str], years: list[int], include_women: bool,
             fetcher: PoliteFetcher) -> set[str]:
    """Return all scorecard URLs found across countries × years."""
    discovered: set[str] = set()
    kinds = [False, True] if include_women else [False]
    total = len(countries) * len(years) * len(kinds)
    done = 0
    # A wrong CricketArchive country code (BAN vs BDESH) 404s on every list and
    # used to be skipped silently, so a whole country could vanish for months.
    # Track it and say so at the end.
    #
    # These are per (country, year, gender) probes, because that is the unit that
    # can come back empty. The warning below is about countries, so it reports
    # distinct countries: a country with cards in one gender but not the other is
    # NOT empty, and listing the same country once per probe made a single bad
    # code look like four bad countries.
    empty_countries: list[str] = []
    probes_with_cards: set[str] = set()
    # Per-country list-request tallies, so "404 on every list" means exactly
    # that: every token of every probe for that country came back 404. Counting
    # only "saw a 404 somewhere" would call a country a wrong code even when some
    # of its lists loaded fine and were simply empty.
    list_requests: dict[str, int] = {}
    list_404s: dict[str, int] = {}

    for country in countries:
        for year in years:
            for women in kinds:
                done += 1
                label = f"{country}/{year}/{'W' if women else 'M'}"
                found = False
                for token in _season_tokens(year):
                    url = _match_list_url(country, token, women)
                    list_requests[country] = list_requests.get(country, 0) + 1
                    try:
                        # Match lists change as the season is played. The HTML
                        # cache is forever, so a weekly run must refetch these
                        # index pages. Scorecards stay cached in the harvester.
                        html = fetcher.get(url, force=True)
                        cards = set(
                            urljoin("https://cricketarchive.com", m)
                            for m in SCORECARD_RX.findall(html)
                        )
                        if cards:
                            discovered |= cards
                            logger.info("%s -> %d scorecards (token %s)  [%d/%d]",
                                        label, len(cards), token, done, total)
                            found = True
                            # Keep going. The first token that has any cards
                            # is not the whole season: IND 2026 and 2025-26
                            # are different lists.
                    except Exception as exc:
                        text = str(exc)
                        if ("403" in text or "paywall" in text.lower()
                                or isinstance(exc, NotAuthenticatedError)):
                            raise ArchiveAccessError(
                                "Archive fetch was rejected "
                                f"({text[:180]}). The saved browser session is not being "
                                "accepted. Headless login cannot replace it. Open a "
                                "scorecard in the Cursor browser while signed in, then "
                                "re-run. Stopping so this does not continue across "
                                "every country."
                            ) from exc
                        if "404" in text:
                            list_404s[country] = list_404s.get(country, 0) + 1
                        continue
                if found:
                    probes_with_cards.add(country)
                else:
                    empty_countries.append(country)
                    logger.debug("%s -> no matches", label)

    # Distinct countries, in first-seen order, so one country cannot appear four
    # times just because it was probed for two years × two genders.
    empty = uniq_in_order(empty_countries)
    # A country is only empty if NO probe of it returned cards. A country with
    # men's cards but no women's list is incomplete, not absent; reporting it as
    # "no scorecards" contradicts the log line that just printed its cards.
    empty = [c for c in empty if c not in probes_with_cards]

    if empty:
        # A country whose every list 404'd is the signature of a wrong country
        # code, not an empty season: BAN 404s while BDESH returns scorecards.
        suspect = sorted(
            c for c in empty
            if list_requests.get(c) and list_404s.get(c, 0) == list_requests[c]
        )
        logger.warning(
            "No scorecards for %d of %d countries: %s",
            len(empty), len(countries), ", ".join(empty),
        )
        if suspect:
            logger.warning(
                "These %d countries returned HTTP 404 on every list, which usually "
                "means a wrong CricketArchive code rather than an empty season: %s. "
                "Check them against the site before trusting coverage.",
                len(suspect), ", ".join(suspect),
            )

    return discovered


def format_elapsed(started: datetime, finished: datetime) -> str:
    """Human-readable duration between two naive local timestamps."""
    seconds = max(0, int((finished - started).total_seconds()))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def render_summary_lines(
    started: datetime,
    finished: datetime,
    new_urls: list[str],
    existing_count: int,
    out_path: Path,
) -> list[str]:
    """The final banner.

    Two timestamps on purpose: `started` names the run (and the output file),
    `finished` says when the work actually ended. Reporting only `started` here
    made a run that began 00:24 and finished 09:14 advertise 00:24.
    """
    return [
        "=" * 72,
        f"TOP-UP COMPLETE  {finished.strftime('%Y-%m-%d %H:%M')}"
        f"  (started {started.strftime('%Y-%m-%d %H:%M')},"
        f" elapsed {format_elapsed(started, finished)})",
        f"  New scorecards found : {len(new_urls)}",
        f"  DB before            : {existing_count}",
        f"  DB after (est.)      : {existing_count + len(new_urls)}",
        f"  URL list             : {out_path}",
        "=" * 72,
    ]


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

    # Sunday 03:00 starts a second process. If this harvest is still walking
    # the match lists, leave it alone rather than writing ca_archive.db twice.
    try:
        ps_output = subprocess.check_output(
            ["ps", "-ax", "-o", "pid=,ppid=,command="], text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        ps_output = ""
    others = other_topup_pids(ps_output, os.getpid(), os.getppid())
    if others:
        logger.info(
            "Another ca_topup is already running (pid %s). "
            "Leaving it alone so this run does not start a second harvest.",
            ", ".join(str(pid) for pid in others),
        )
        return 0

    # ── Resolve years ─────────────────────────────────────────────
    # `started` only labels the run (it names the default output file). The
    # banner must report the real finish time, not this.
    started = datetime.now()
    if args.years:
        years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
    else:
        years = [started.year - 1, started.year]

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
        cache_dir / f"topup_{started.strftime('%Y%m%d')}.txt"

    logger.info("Top-up config:")
    logger.info("  Years     : %s", years)
    logger.info("  Countries : %d", len(countries))
    logger.info("  Women     : %s", include_women)
    logger.info("  DB        : %s", db_path)
    logger.info("  Out       : %s", out_path)

    # ── Discover ──────────────────────────────────────────────────
    # Match-list pages 403 without pigeon_access. ensure_session() reuses the
    # saved browser session or refreshes it from the Cursor browser. It does
    # not fall back to headless login, which never receives that cookie.
    try:
        fetcher = PoliteFetcher(auth.ensure_session())
        discovered = discover(countries, years, include_women, fetcher)
    except (RuntimeError, ArchiveAccessError) as exc:
        logger.error("%s", exc)
        return 1
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

    print("\n" + "\n".join(render_summary_lines(
        started=started,
        finished=datetime.now(),
        new_urls=new_urls,
        existing_count=len(existing),
        out_path=out_path,
    )))

    return 0


if __name__ == "__main__":
    sys.exit(main())
