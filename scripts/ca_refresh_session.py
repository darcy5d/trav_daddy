#!/usr/bin/env python3
"""Refresh the CricketArchive archive cookie before it expires.

The account form never sets pigeon_access. login_playwright signs in and then
opens one archive page in that same browser, which is what issues the cookie.
The saved file is replaced only when the new snapshot has an unexpired one.

Run by LaunchAgent every 6 hours. A cookie lasts about 7 days; this refreshes
once less than 48 hours remain, so a failed run still leaves days of margin.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CRICKETARCHIVE_CONFIG
from src.api.cricketarchive import auth

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ca_refresh_session")

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "ca_session_refresh.log"


def _file_log() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(LOG_PATH)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)


def main(argv: list[str] | None = None) -> int:
    _file_log()
    force = "--force" in (argv if argv is not None else sys.argv[1:])
    state_path = Path(CRICKETARCHIVE_CONFIG["auth_state_path"])
    current = auth._find_cookie(auth._load_state(state_path), auth.ARCHIVE_COOKIE)
    if not force and not auth.refresh_due(current):
        logger.info(
            "Archive cookie still valid; expires %s. Next refresh when under 48h remain.",
            auth._expiry_text(current),
        )
        return 0

    logger.info(
        "Refreshing CricketArchive archive cookie (expires %s).",
        auth._expiry_text(current),
    )
    try:
        auth.login_playwright(state_path=state_path, headless=True)
    except Exception as exc:
        logger.error("CricketArchive login failed (%s). Saved session was not replaced.", type(exc).__name__)
        return 1

    fresh = auth._find_cookie(auth._load_state(state_path), auth.ARCHIVE_COOKIE)
    if fresh is None or auth._is_expired(fresh, time.time()):
        logger.error(
            "Login did not obtain an unexpired %s. Archive pages would stay on the paywall. "
            "Not scheduling further browser work.",
            auth.ARCHIVE_COOKIE,
        )
        return 1

    session = auth._new_session()
    auth._load_cookies_into_session(session, auth._load_state(state_path))
    try:
        response = session.get(auth.ARCHIVE_PROBE_URL, timeout=30, allow_redirects=True)
    except Exception as exc:
        logger.error("Scorecard check failed (%s). Cookie expires %s.", type(exc).__name__, auth._expiry_text(fresh))
        return 1

    ok = response.status_code == 200 and "innings" in response.text.lower()
    logger.info(
        "Scorecard check HTTP %s innings=%s; %s expires %s.",
        response.status_code,
        ok,
        auth.ARCHIVE_COOKIE,
        auth._expiry_text(fresh),
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
