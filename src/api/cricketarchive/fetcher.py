"""Polite, cached, robots-aware fetcher for CricketArchive.

Design goals (see CRICKETARCHIVE_CONFIG in config.py for the why):
* Fetch each URL at most ONCE, ever — every response is cached to disk keyed by
  URL, so re-parsing / re-runs never re-hit their servers.
* Honour the robots.txt ``User-Agent: *`` disallow list.
* Rate-limit hard (randomised delay) and cap requests per day, so we never look
  "detrimental to the use and enjoyment of the Site" under their ToS.
* Detect the paywall / logged-out state and fail loudly rather than caching junk.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from datetime import date
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests

logger = logging.getLogger(__name__)


class NotAuthenticatedError(RuntimeError):
    """Raised when a fetched page is the paywall / sign-in wall, not real data."""


class RobotsDisallowedError(RuntimeError):
    """Raised when a URL is disallowed by robots.txt for User-Agent: *."""


# Markers that indicate we got bounced to the paywall instead of content.
_PAYWALL_MARKERS = (
    "To continue reading, sign in",
    "Premium Access - CricketArchive",
    "choose an access option below",
)


class PoliteFetcher:
    def __init__(self, session: requests.Session, config: Optional[dict] = None):
        if config is None:
            from config import CRICKETARCHIVE_CONFIG
            config = CRICKETARCHIVE_CONFIG
        self.cfg = config
        self.session = session
        self.cache_dir = Path(config["cache_dir"]) / "html"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._counter_path = Path(config["cache_dir"]) / "request_counter.json"
        self._robots: dict[str, RobotFileParser] = {}
        self._last_fetch_ts = 0.0

    # -- robots -----------------------------------------------------------------
    def _robots_for(self, url: str) -> RobotFileParser:
        host = urlparse(url).netloc
        rp = self._robots.get(host)
        if rp is None:
            rp = RobotFileParser()
            robots_url = f"{urlparse(url).scheme}://{host}/robots.txt"
            try:
                resp = self.session.get(robots_url, timeout=self.cfg["request_timeout_sec"])
                rp.parse(resp.text.splitlines())
            except Exception as e:  # be conservative but don't crash the run
                logger.warning("Could not load robots.txt for %s: %s", host, e)
                rp.parse([])  # empty => allow all
            self._robots[host] = rp
        return rp

    def _robots_allows(self, url: str) -> bool:
        # We present as a normal browser, so the User-Agent: * group applies.
        return self._robots_for(url).can_fetch("*", url)

    # -- cache ------------------------------------------------------------------
    def _cache_path(self, url: str) -> Path:
        h = hashlib.sha1(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}.html"

    def _cached(self, url: str) -> Optional[str]:
        p = self._cache_path(url)
        return p.read_text(encoding="utf-8", errors="replace") if p.exists() else None

    # -- daily request cap ------------------------------------------------------
    def _check_and_bump_counter(self) -> None:
        today = date.today().isoformat()
        data = {"date": today, "count": 0}
        if self._counter_path.exists():
            try:
                data = json.loads(self._counter_path.read_text())
            except Exception:
                pass
        if data.get("date") != today:
            data = {"date": today, "count": 0}
        if data["count"] >= self.cfg["max_requests_per_day"]:
            raise RuntimeError(
                f"Daily request cap reached ({self.cfg['max_requests_per_day']}). "
                "Resume tomorrow or raise CRICKETARCHIVE_MAX_PER_DAY."
            )
        data["count"] += 1
        self._counter_path.write_text(json.dumps(data))

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_fetch_ts
        delay = random.uniform(self.cfg["min_delay_sec"], self.cfg["max_delay_sec"])
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_fetch_ts = time.time()

    # -- main entrypoint --------------------------------------------------------
    def get(self, url: str, force: bool = False) -> str:
        """Return page HTML, from cache when available; otherwise fetch politely."""
        if not force:
            cached = self._cached(url)
            if cached is not None:
                return cached

        if not self._robots_allows(url):
            raise RobotsDisallowedError(f"robots.txt disallows: {url}")

        self._check_and_bump_counter()
        self._throttle()

        logger.info("FETCH %s", url)
        resp = self.session.get(url, timeout=self.cfg["request_timeout_sec"], allow_redirects=True)
        resp.raise_for_status()
        html = resp.text

        if any(m in html for m in _PAYWALL_MARKERS):
            raise NotAuthenticatedError(
                f"Got paywall/sign-in for {url} — session expired? Re-run with force_login."
            )

        self._cache_path(url).write_text(html, encoding="utf-8")
        return html
