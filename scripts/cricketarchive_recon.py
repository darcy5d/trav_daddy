#!/usr/bin/env python3
"""CricketArchive recon / login smoke-test.

Logs in (Playwright, from .env), then fetches a handful of representative pages
for the pilot series (India in Ireland 2026) into the on-disk cache and prints a
short summary. Proves auth + the polite fetcher work end-to-end before we build
the parsers/backfill on top.

Usage:
    venv311/bin/python scripts/cricketarchive_recon.py            # reuse saved session
    venv311/bin/python scripts/cricketarchive_recon.py --force-login
    venv311/bin/python scripts/cricketarchive_recon.py --manual   # headed manual login
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.cricketarchive import auth
from src.api.cricketarchive.fetcher import PoliteFetcher, NotAuthenticatedError

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# Pilot: India in Ireland 2026 (2nd Twenty20) and its series page.
PILOT_URLS = {
    "series":       "https://cricketarchive.com/Archive/Events/41/India_in_Ireland_2026.html",
    "scorecard":    "https://cricketarchive.com/Archive/Scorecards/1446/1446777.html",
    "commentary1":  "https://cricketarchive.com/Archive/Scorecards/1446/1446777/1446777_commentary_i1_page.html",
    "commentary2":  "https://cricketarchive.com/Archive/Scorecards/1446/1446777/1446777_commentary_i2_page.html",
    "player_public":"https://cricketarchive.com/Archive/Players/0/984/984.html",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force-login", action="store_true")
    ap.add_argument("--manual", action="store_true", help="headed manual login")
    args = ap.parse_args()

    if args.manual:
        auth.login_manual_assist()
        session = auth.ensure_session()  # reuse the just-saved state
    else:
        session = auth.ensure_session(force_login=args.force_login, headless=True)

    fetcher = PoliteFetcher(session)

    print("\n" + "=" * 84)
    print("CRICKETARCHIVE RECON")
    print("=" * 84)
    ok = True
    for name, url in PILOT_URLS.items():
        try:
            html = fetcher.get(url)
            title = ""
            if "<title>" in html:
                title = html.split("<title>", 1)[1].split("</title>", 1)[0].strip()
            print(f"  [OK]   {name:<14} {len(html):>8,d} bytes   title={title!r}")
        except NotAuthenticatedError as e:
            ok = False
            print(f"  [AUTH] {name:<14} PAYWALLED — {e}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  [ERR]  {name:<14} {type(e).__name__}: {e}")

    print("=" * 84)
    print("Cache dir:", fetcher.cache_dir)
    print("Result:", "ALL OK" if ok else "SOME FAILED — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
