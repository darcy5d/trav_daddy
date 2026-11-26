"""Authentication for CricketArchive.

CricketArchive's paywall (my.cricketarchive.com, "Powered by Pigeon") is an
email/password form. That headless login reaches My Account, but archive
scorecards stay on the paywall until the browser also holds ``pigeon_access``.
Headless Playwright does not receive that cookie. The weekly job therefore
reuses the saved browser session, and refreshes it from the Cursor browser's
cookie jar when that jar has a newer unexpired ``pigeon_access``.

Credentials come from .env via ``config.CRICKETARCHIVE_CONFIG`` and are NEVER
logged. Cookie values are never logged either. Two login paths:

* ``login_playwright`` — headless, fills the form from .env (default). It must
  not replace a saved session that still has an unexpired archive cookie.
* ``login_manual_assist`` — headed browser; you sign in by hand (handles
  captcha / SSO / "remember me"). Useful if the automated form flow breaks.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Archive pages 403 without this cookie. The account form never sets it; the
# site issues it when the signed-in browser then opens an archive page.
ARCHIVE_COOKIE = "pigeon_access"
ARCHIVE_PROBE_URL = "https://cricketarchive.com/Archive/Scorecards/1460/1460099.html"
# Refresh while this much life remains, so a failed run still has days left.
REFRESH_MARGIN_SECS = 48 * 3600
# Chromium stores expires_utc as microseconds since 1601-01-01 UTC.
_CHROMIUM_EPOCH_OFFSET = 11644473600
_SAMESITE = {-1: "Lax", 0: "None", 1: "Lax", 2: "Strict"}


def _cfg() -> dict:
    from config import CRICKETARCHIVE_CONFIG
    return CRICKETARCHIVE_CONFIG


def cursor_browser_cookie_db() -> Path:
    """Cookie jar for the Cursor in-app browser, which is where archive access lives."""
    return (
        Path.home()
        / "Library/Application Support/Cursor/Partitions/cursor-browser/Cookies"
    )


def _load_state(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    cookies = data.get("cookies") or []
    return [c for c in cookies if isinstance(c, dict) and c.get("name") and c.get("value")]


def _write_state(path: Path, cookies: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"cookies": cookies, "origins": []})
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(payload)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _find_cookie(cookies: list[dict], name: str) -> Optional[dict]:
    for cookie in cookies:
        if cookie.get("name") == name:
            return cookie
    return None


def _expiry_unix(cookie: Optional[dict]) -> Optional[float]:
    if not cookie:
        return None
    exp = cookie.get("expires")
    if not isinstance(exp, (int, float)) or exp < 0:
        return None
    return float(exp)


def _is_expired(cookie: Optional[dict], now: float) -> bool:
    """A missing expiry is a session cookie we can still send. A past expiry is dead."""
    exp = _expiry_unix(cookie)
    if exp is None:
        return False
    return exp <= now


def refresh_due(cookie: Optional[dict], now: Optional[float] = None, margin_secs: float = REFRESH_MARGIN_SECS) -> bool:
    """True when the archive cookie is missing, expired, or inside the refresh margin."""
    now = time.time() if now is None else now
    if cookie is None or _is_expired(cookie, now):
        return True
    exp = _expiry_unix(cookie)
    if exp is None:
        return True
    return (exp - now) <= margin_secs


def _expiry_text(cookie: Optional[dict]) -> str:
    exp = _expiry_unix(cookie)
    if exp is None:
        return "no expiry timestamp"
    return datetime.fromtimestamp(exp).astimezone().isoformat(timespec="seconds")


def _materially_newer(saved: Optional[dict], imported: dict) -> bool:
    if saved is None:
        return True
    if saved.get("value") != imported.get("value"):
        return True
    saved_exp = _expiry_unix(saved) or -1
    imported_exp = _expiry_unix(imported) or -1
    return imported_exp > 0 and imported_exp > saved_exp + 1


def choose_session_cookies(
    saved: list[dict],
    imported: list[dict],
    now: Optional[float] = None,
) -> tuple[Optional[list[dict]], str]:
    """Pick the cookie set the weekly job should send.

    Returns ``(cookies, reason)``. ``reason`` is ``refreshed`` when the Cursor
    browser jar supplied a newer unexpired archive cookie, ``reused`` when the
    saved file is still good, and ``expired`` or ``missing`` when neither is.
    """
    now = time.time() if now is None else now
    saved_archive = _find_cookie(saved, ARCHIVE_COOKIE)
    imported_archive = _find_cookie(imported, ARCHIVE_COOKIE)
    imported_ok = imported_archive is not None and not _is_expired(imported_archive, now)
    saved_ok = saved_archive is not None and not _is_expired(saved_archive, now)

    if imported_ok and _materially_newer(saved_archive, imported_archive):
        return imported, "refreshed"
    if saved_ok:
        return saved, "reused"
    if imported_ok:
        return imported, "refreshed"
    if saved_archive is not None or imported_archive is not None:
        return None, "expired"
    return None, "missing"


def _chromium_expires_to_unix(expires_utc: int) -> float:
    if not expires_utc:
        return -1.0
    return (expires_utc / 1_000_000) - _CHROMIUM_EPOCH_OFFSET


def _cookie_value_usable(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    return not value.startswith(("v10", "v11"))


def read_cursor_browser_cookies(db_path: Path) -> list[dict]:
    """Read plaintext CricketArchive cookies from a Chromium cookie DB.

    Returns an empty list when the file is absent. Raises if it cannot be read.
    Encrypted values are skipped. Values are not logged.
    """
    if not db_path.exists():
        return []

    tmp_dir = tempfile.mkdtemp(prefix="ca-cookies-")
    try:
        dst = Path(tmp_dir) / "Cookies"
        shutil.copy2(db_path, dst)
        for ext in ("-wal", "-shm"):
            sidecar = Path(str(db_path) + ext)
            if sidecar.exists():
                shutil.copy2(sidecar, Path(str(dst) + ext))
        conn = sqlite3.connect(f"file:{dst}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                """
                SELECT name, value, host_key, path, expires_utc,
                       is_secure, is_httponly, samesite
                FROM cookies
                WHERE host_key LIKE '%cricketarchive.com%'
                """
            ).fetchall()
        finally:
            conn.close()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    cookies: list[dict] = []
    for name, value, host, path, expires_utc, secure, httponly, samesite in rows:
        if not _cookie_value_usable(value):
            continue
        cookies.append({
            "name": name,
            "value": value,
            "domain": host,
            "path": path or "/",
            "expires": _chromium_expires_to_unix(int(expires_utc or 0)),
            "httpOnly": bool(httponly),
            "secure": bool(secure),
            "sameSite": _SAMESITE.get(int(samesite or 0), "Lax"),
        })
    return cookies


def _load_cookies_into_session(session: requests.Session, cookies: list[dict]) -> int:
    """Copy Playwright-shaped cookies into a requests Session."""
    n = 0
    for c in cookies:
        kwargs = {}
        exp = _expiry_unix(c)
        if exp is not None:
            kwargs["expires"] = exp
        session.cookies.set(
            name=c["name"],
            value=c["value"],
            domain=c.get("domain", "").lstrip("."),
            path=c.get("path", "/"),
            **kwargs,
        )
        n += 1
    return n


def _keep_saved_archive_cookie(state_path: Path, new_cookies: list[dict]) -> bool:
    """True when a fresh login lacks the archive cookie and the saved one is still good."""
    now = time.time()
    new_archive = _find_cookie(new_cookies, ARCHIVE_COOKIE)
    old_archive = _find_cookie(_load_state(state_path), ARCHIVE_COOKIE)
    new_ok = new_archive is not None and not _is_expired(new_archive, now)
    old_ok = old_archive is not None and not _is_expired(old_archive, now)
    return old_ok and not new_ok


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
        # The page also has "Sign in with a passkey"; exact avoids that button.
        page.get_by_role("button", name="Sign in", exact=True).click()
        # Wait for the session to settle (auth cookie set / redirect away from form).
        try:
            page.wait_for_load_state("networkidle", timeout=20000)
        except Exception:
            pass
        time.sleep(2.0)
        # The account login does not issue pigeon_access. Opening one archive
        # page in this same browser does. A missing cookie must not replace a
        # saved session: _commit_login_state keeps the old file in that case.
        try:
            page.goto(ARCHIVE_PROBE_URL, wait_until="domcontentloaded", timeout=30000)
            try:
                page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:
                pass
            time.sleep(1.0)
        except Exception:
            logger.warning(
                "Signed in, but the archive page did not load. "
                "The saved session is kept unless this login received %s.",
                ARCHIVE_COOKIE,
            )
        tmp_path = state_path.with_suffix(".login.tmp")
        context.storage_state(path=str(tmp_path))
        browser.close()

    return _commit_login_state(state_path, tmp_path)


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
        tmp_path = state_path.with_suffix(".login.tmp")
        context.storage_state(path=str(tmp_path))
        browser.close()

    return _commit_login_state(state_path, tmp_path)


def _commit_login_state(state_path: Path, tmp_path: Path) -> Path:
    """Install a login snapshot unless it would drop a still-valid archive cookie."""
    try:
        new_cookies = _load_state(tmp_path)
        if _keep_saved_archive_cookie(state_path, new_cookies):
            kept = _find_cookie(_load_state(state_path), ARCHIVE_COOKIE)
            logger.warning(
                "Headless login did not receive %s. Kept the saved browser session; "
                "it expires %s.",
                ARCHIVE_COOKIE,
                _expiry_text(kept),
            )
            return state_path
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, state_path)
    finally:
        if tmp_path.exists() and tmp_path != state_path:
            # replace already moved it on the success path; this cleans a rejected login.
            if not state_path.exists() or tmp_path.exists():
                try:
                    if tmp_path.exists() and tmp_path.resolve() != state_path.resolve():
                        tmp_path.unlink()
                except OSError:
                    pass

    archive = _find_cookie(_load_state(state_path), ARCHIVE_COOKIE)
    if archive is None or _is_expired(archive, time.time()):
        logger.warning(
            "Headless login did not receive %s. Archive pages will stay on the paywall "
            "until the Cursor browser session is imported.",
            ARCHIVE_COOKIE,
        )
    else:
        logger.info("Saved CricketArchive auth state -> %s", state_path)
    return state_path


def ensure_session(force_login: bool = False, headless: bool = True) -> requests.Session:
    """Return a requests.Session that can read CricketArchive scorecards.

    Reuses the saved browser session. When the Cursor browser cookie jar has a
    newer unexpired ``pigeon_access``, that jar replaces the saved file. A stale
    file is not refreshed by headless login: that login never receives the
    archive cookie, so it would turn a working session into a paywall session.
    Raises RuntimeError, after an error log, when the archive cookie is missing
    or expired.
    """
    cfg = _cfg()
    state_path = Path(cfg["auth_state_path"])
    saved = _load_state(state_path)

    if force_login:
        try:
            login_playwright(state_path=state_path, headless=headless)
            saved = _load_state(state_path)
        except Exception as exc:
            logger.warning("Headless CricketArchive login failed (%s).", type(exc).__name__)

    imported: list[dict] = []
    try:
        imported = read_cursor_browser_cookies(cursor_browser_cookie_db())
    except Exception as exc:
        logger.warning(
            "Could not read the Cursor browser cookie jar (%s). "
            "Using the saved session if it still has the archive cookie.",
            type(exc).__name__,
        )

    chosen, reason = choose_session_cookies(saved, imported)
    if chosen is None:
        archive = _find_cookie(saved, ARCHIVE_COOKIE) or _find_cookie(imported, ARCHIVE_COOKIE)
        state = "missing" if archive is None else f"expired at {_expiry_text(archive)}"
        logger.error(
            "CricketArchive archive cookie %s is %s. "
            "Headless email/password login does not receive this cookie, so archive "
            "pages would return the paywall. Open a scorecard in the Cursor browser "
            "while signed in, then run the top-up again. Not fetching archive pages.",
            ARCHIVE_COOKIE,
            state,
        )
        raise RuntimeError(f"CricketArchive archive cookie {ARCHIVE_COOKIE} is {state}")

    archive = _find_cookie(chosen, ARCHIVE_COOKIE)
    if reason == "refreshed":
        _write_state(state_path, chosen)
        logger.info(
            "Refreshed CricketArchive session from the Cursor browser; "
            "%s expires %s.",
            ARCHIVE_COOKIE,
            _expiry_text(archive),
        )
    else:
        logger.info(
            "Reusing saved CricketArchive session; %s expires %s. "
            "Headless login is not used because it does not receive this cookie.",
            ARCHIVE_COOKIE,
            _expiry_text(archive),
        )

    session = _new_session()
    n = _load_cookies_into_session(session, chosen)
    logger.info("CricketArchive session ready (%d cookies)", n)
    return session
