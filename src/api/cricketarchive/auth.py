"""Authentication for CricketArchive.

CricketArchive's paywall (my.cricketarchive.com, "Powered by Pigeon") is a simple
email/password form; once signed in, the session cookie unlocks the data pages on
cricketarchive.com. We log in ONCE via Playwright, persist the browser
``storage_state`` (cookies) to a gitignored file, and reuse those cookies in a
fast ``requests.Session`` for all subsequent page fetches.

Credentials come from .env via ``config.CRICKETARCHIVE_CONFIG`` and are NEVER
logged. Two login paths:

* ``login_playwright`` — headless, fills the form from .env (default).
* ``login_manual_assist`` — headed browser; you sign in by hand (handles
  captcha / SSO / "remember me"). Useful if the automated form flow breaks.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)


def _cfg() -> dict:
    from config import CRICKETARCHIVE_CONFIG
    return CRICKETARCHIVE_CONFIG


def _auth_state_fresh(path: Path, max_age_hours: float) -> bool:
    if not path.exists():
        return False
    age_hours = (time.time() - path.stat().st_mtime) / 3600.0
    return age_hours < max_age_hours


def _load_cookies_into_session(session: requests.Session, state_path: Path) -> int:
    """Copy cookies from a Playwright storage_state JSON into a requests Session."""
    data = json.loads(state_path.read_text())
    n = 0
    for c in data.get("cookies", []):
        # requests cookies don't accept all Playwright fields; map the essentials.
        session.cookies.set(
            name=c["name"],
            value=c["value"],
            domain=c.get("domain", "").lstrip("."),
            path=c.get("path", "/"),
        )
        n += 1
    return n


def _new_session() -> requests.Session:
    cfg = _cfg()
    s = requests.Session()
    s.headers.update({
        "User-Agent": cfg["user_agent"],
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
    })
    return s


def login_playwright(state_path: Optional[Path] = None, headless: bool = True) -> Path:
    """Automated login: fill the email/password form, save storage_state.

    Returns the path to the saved auth-state JSON. Raises RuntimeError on failure.
    """
    cfg = _cfg()
    state_path = Path(state_path or cfg["auth_state_path"])
    username, password = cfg.get("username"), cfg.get("password")
    if not username or not password:
        raise RuntimeError(
            "CRICKETARCHIVE_USERNAME / CRICKETARCHIVE_PASSWORD not set in .env"
        )
    state_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise RuntimeError(
            "playwright not available; run `playwright install chromium`"
        ) from e

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(user_agent=cfg["user_agent"])
        page = context.new_page()
        page.goto(cfg["login_url"], wait_until="domcontentloaded")
        # The Pigeon paywall form: email + password textboxes + "Sign in" button.
        page.get_by_placeholder("Your Email").fill(username)
        page.get_by_placeholder("Your Password").fill(password)
        page.get_by_role("button", name="Sign in").click()
        # Wait for the session to settle (auth cookie set / redirect away from form).
        try:
            page.wait_for_load_state("networkidle", timeout=20000)
        except Exception:
            pass
        time.sleep(2.0)
        context.storage_state(path=str(state_path))
        browser.close()

    logger.info("Saved CricketArchive auth state -> %s", state_path)
    return state_path


def login_manual_assist(state_path: Optional[Path] = None) -> Path:
    """Headed login: open a browser, you sign in by hand, then we save cookies."""
    cfg = _cfg()
    state_path = Path(state_path or cfg["auth_state_path"])
    state_path.parent.mkdir(parents=True, exist_ok=True)

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(user_agent=cfg["user_agent"])
        page = context.new_page()
        page.goto(cfg["login_url"], wait_until="domcontentloaded")
        input(
            "\n>>> Sign in to CricketArchive in the opened browser window, "
            "then press Enter here to save the session... "
        )
        context.storage_state(path=str(state_path))
        browser.close()

    logger.info("Saved CricketArchive auth state -> %s", state_path)
    return state_path


def ensure_session(force_login: bool = False, headless: bool = True) -> requests.Session:
    """Return a requests.Session carrying a valid CricketArchive login.

    Reuses a fresh saved auth-state if present; otherwise logs in via Playwright.
    """
    cfg = _cfg()
    state_path = Path(cfg["auth_state_path"])
    if force_login or not _auth_state_fresh(state_path, cfg["auth_max_age_hours"]):
        login_playwright(state_path=state_path, headless=headless)

    session = _new_session()
    n = _load_cookies_into_session(session, state_path)
    logger.info("CricketArchive session ready (%d cookies)", n)
    return session
