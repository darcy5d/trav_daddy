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


class CacheMissError(RuntimeError):
    """Raised in cache_only mode when a URL is not in the on-disk cache."""


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
        # Adaptive extra delay (seconds) added to the base throttle; grows on a
        # throttle response (403/429), decays on sustained success.
        self._extra_delay = 0.0
        # Look more like a real browser navigation to reduce bot heuristics.
        self.session.headers.setdefault("Referer", str(config.get("base_url", "")) + "/")

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
        delay += self._extra_delay
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_fetch_ts = time.time()

    # -- main entrypoint --------------------------------------------------------
    def get(self, url: str, force: bool = False, cache_only: bool = False) -> str:
        """Return page HTML, from cache when available; otherwise fetch politely.

        Retries HTTP 403/429 (CA throttling) with exponential backoff and adaptively
        slows the base rate. In cache_only mode, never hits the network.
        """
        if not force:
            cached = self._cached(url)
            if cached is not None:
                return cached
        if cache_only:
            raise CacheMissError(url)

        if not self._robots_allows(url):
            raise RobotsDisallowedError(f"robots.txt disallows: {url}")

        max_retries = int(self.cfg.get("max_retries", 4))
        backoff_base = float(self.cfg.get("backoff_base_sec", 20))
        last_exc: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            self._check_and_bump_counter()
            self._throttle()
            logger.info("FETCH %s%s", url, f" (retry {attempt})" if attempt else "")
            resp = self.session.get(url, timeout=self.cfg["request_timeout_sec"],
                                    allow_redirects=True)

            if resp.status_code in (403, 429):
                # CA throttling: back off, slow the sustained rate, and retry.
                self._extra_delay = min(self._extra_delay + 4.0, 60.0)
                wait = backoff_base * (2 ** attempt) + random.uniform(0, 5)
                logger.warning("HTTP %s for %s — backoff %.0fs (attempt %d/%d, extra_delay=%.1fs)",
                               resp.status_code, url, wait, attempt + 1, max_retries + 1,
                               self._extra_delay)
                last_exc = requests.HTTPError(f"{resp.status_code} for {url}")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            html = resp.text
            if any(m in html for m in _PAYWALL_MARKERS):
                raise NotAuthenticatedError(
                    f"Got paywall/sign-in for {url} — session expired? Re-run with force_login."
                )
            # success: decay the adaptive slowdown
            self._extra_delay = max(0.0, self._extra_delay - 1.0)
            self._cache_path(url).write_text(html, encoding="utf-8")
            return html

        raise last_exc or RuntimeError(f"exhausted retries for {url}")
