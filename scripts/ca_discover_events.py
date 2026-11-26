#!/usr/bin/env python3
"""Discover + verify IPL and Big Bash season event-page URLs on CricketArchive.

Event-path numbers (e.g. /Archive/Events/37/Indian_Premier_League_2023.html)
are not guessable, and Big Bash event names carry a changing sponsor prefix, so
we resolve each season via CA's per-season index pages
(/Archive/Seasons/{season}_{country}.html), which link to the correctly-named
event regardless of sponsor. Each resolved event URL is verified by parsing it
and confirming it yields scorecards.

Output: data/raw/cricketarchive/ipl_bbl_events.txt (one event URL per line),
consumed by `ca_harvest.py --list`.

Usage:
    venv311/bin/python scripts/ca_discover_events.py
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CRICKETARCHIVE_CONFIG
from src.api.cricketarchive import auth, parsers as P
from src.api.cricketarchive.fetcher import PoliteFetcher, NotAuthenticatedError

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ca_discover")

BASE = "https://cricketarchive.com"
IPL_YEARS = list(range(2019, 2027))          # 2019..2026 (8 IPL seasons)
BBL_START_YEARS = list(range(2018, 2026))    # 2018/19 .. 2025/26 (8 BBL seasons)


def _season_index_candidates(country: str, y: int) -> list[str]:
    """Candidate season-index URLs (CA naming varies for split seasons)."""
    if country == "IND":  # India keyed by calendar year
        return [f"{BASE}/Archive/Seasons/{y}_IND.html"]
    # Australia: split season y/(y+1) - try the common encodings
    yy = str(y + 1)[2:]
    return [
        f"{BASE}/Archive/Seasons/{y}-{y+1}_AUS.html",
        f"{BASE}/Archive/Seasons/{y}-{yy}_AUS.html",
        f"{BASE}/Archive/Seasons/{y}_AUS.html",
        f"{BASE}/Archive/Seasons/{y+1}_AUS.html",
    ]


def _find_event_link(html: str, name_rx: re.Pattern) -> str | None:
    soup = BeautifulSoup(html, "lxml")
    for a in soup.find_all("a"):
        text = a.get_text(" ", strip=True)
        href = a.get("href") or ""
        if "/Archive/Events/" in href and name_rx.search(text):
            return urljoin(BASE, href)
    return None


def _resolve(fetcher: PoliteFetcher, candidates: list[str], name_rx: re.Pattern) -> str | None:
    for url in candidates:
        try:
            html = fetcher.get(url)
        except (NotAuthenticatedError, Exception) as e:  # noqa: BLE001
            logger.debug("  season index miss %s (%s)", url, type(e).__name__)
            continue
        link = _find_event_link(html, name_rx)
        if link:
            return link
    return None


def _probe_bbl_event(fetcher: PoliteFetcher, y: int, yy: str,
                     batch_range: range = range(28, 45)) -> str | None:
    """Probe the stable BBL event-name across plausible batch numbers."""
    for n in batch_range:
        url = f"{BASE}/Archive/Events/{n}/KFC_Twenty20_Big_Bash_{y}-{yy}.html"
        try:
            if P.parse_event_matches(fetcher.get(url), url):
                return url
        except Exception:  # noqa: BLE001
            continue
    return None


def _verify(fetcher: PoliteFetcher, event_url: str) -> int:
    """Return scorecard count for an event URL (0 = failed/empty)."""
    try:
        matches = P.parse_event_matches(fetcher.get(event_url), event_url)
        return len(matches)
    except Exception as e:  # noqa: BLE001
        logger.warning("  verify failed %s: %s", event_url, e)
        return 0


def main() -> int:
    fetcher = PoliteFetcher(auth._new_session())
    resolved: list[tuple[str, str, int]] = []   # (label, url, n_matches)

    # IPL
    for y in IPL_YEARS:
        rx = re.compile(rf"Indian Premier League {y}\b")
        url = _resolve(fetcher, _season_index_candidates("IND", y), rx)
        if not url:
            logger.warning("IPL %s: NOT FOUND", y)
            continue
        n = _verify(fetcher, url)
        resolved.append((f"IPL {y}", url, n))
        logger.info("IPL %s -> %s (%d matches)", y, url, n)

    # Big Bash (men's) - exclude Women's Big Bash
    for y in BBL_START_YEARS:
        yy = str(y + 1)[2:]
        rx = re.compile(rf"(?<!Women's )Big Bash {y}[/-]{yy}\b")
        url = _resolve(fetcher, _season_index_candidates("AUS", y), rx)
        if not url:
            # Fallback: the event name is stable ("KFC_Twenty20_Big_Bash_{y}-{yy}")
            # even when the season-index page isn't where we expect. Probe the
            # handful of plausible batch numbers directly.
            url = _probe_bbl_event(fetcher, y, yy)
        if not url:
            logger.warning("BBL %s/%s: NOT FOUND", y, yy)
            continue
        n = _verify(fetcher, url)
        resolved.append((f"BBL {y}/{yy}", url, n))
        logger.info("BBL %s/%s -> %s (%d matches)", y, yy, url, n)

    out_dir = Path(CRICKETARCHIVE_CONFIG["cache_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_path = out_dir / "ipl_bbl_events.txt"
    good = [u for (_, u, n) in resolved if n > 0]
    seed_path.write_text("\n".join(good) + ("\n" if good else ""))

    print("\n" + "=" * 78)
    print(f"{'SEASON':<14} {'MATCHES':>8}  URL")
    print("-" * 78)
    for label, url, n in resolved:
        print(f"{label:<14} {n:>8}  {url}")
    total = sum(n for _, _, n in resolved)
    print("-" * 78)
    print(f"Seasons resolved: {len(resolved)}    Total matches: {total}")
    print(f"Seed file: {seed_path}  ({len(good)} event URLs)")
    print("=" * 78)
    return 0 if resolved else 1


if __name__ == "__main__":
    sys.exit(main())
