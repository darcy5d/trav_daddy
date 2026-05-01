"""Wave 5.7c: pull confirmed Playing XI from CREX for a Polymarket fixture.

Reuses the existing CREX scraper infrastructure:
    1. `crex.get_schedule()` - lists upcoming CREX matches with team names + dates
    2. `crex.fetch_squads_with_playwright()` - clicks the per-team
       'playingxi-button' tabs and extracts 11 players each

The pipeline:
    polymarket_fixture (team1, team2, scheduled_start_utc)
        --> match to CREX match by (team-pair, date)
        --> fetch_squads_with_playwright(crex_url, ...)
        --> CREX player names per team
        --> fuzzy-match against DB players for that team
        --> 11 player_ids per team

Notes:
    - CREX may show a "predicted XI" before toss and "confirmed XI" after
      toss. We use whatever is currently rendered. Caller should check the
      timestamp to know how reliable the data is.
    - Playwright takes ~5-15s to render the page + click tabs. Run as a
      subprocess from Flask routes to avoid blocking the request thread.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from src.api.crex_scraper import CREXScraper, CREXMatch, CREXPlayer
from src.integrations.odds.polymarket_compare import PolymarketComparisonService

logger = logging.getLogger(__name__)

# Cache get_schedule() results at module level to avoid the ~50s series-name
# enrichment cost on every UI click. CREX's schedule is stable enough
# (team-pair + date mapping doesn't change pre-match) that a 15-min TTL
# is fine, and we still get the fresh status of any in-progress match if
# the user re-opens the modal a while later.
_SCHEDULE_CACHE: Dict[Tuple[str, ...], Dict[str, Any]] = {}
_SCHEDULE_CACHE_TTL = timedelta(minutes=15)


def _normalize(s: str) -> str:
    """Lower + collapse non-alnum -> space + strip."""
    if not s:
        return ""
    s = re.sub(r"[^A-Za-z0-9 ]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _team_pair_matches(
    pm_t1: str, pm_t2: str, crex_t1: str, crex_t2: str,
) -> bool:
    """Check if a CREX match's team pair corresponds to our Polymarket fixture.

    Uses the same token-overlap matcher we already use for moneyline outcomes.
    Tries both orderings (CREX may list teams in either order vs Polymarket).
    """
    cmp = PolymarketComparisonService.label_matches_team
    fwd = cmp(crex_t1, pm_t1) and cmp(crex_t2, pm_t2)
    rev = cmp(crex_t1, pm_t2) and cmp(crex_t2, pm_t1)
    return fwd or rev


def _get_cached_schedule(
    formats: List[str], hours_ahead: int = 72,
    max_attempts: int = 3,
) -> List[CREXMatch]:
    """Module-level cache around CREXScraper.get_schedule() with retry.

    CREX's schedule endpoint occasionally returns an empty array with no
    error (~1s response), then a successful 19-match response on the next
    call (~50s). We retry up to `max_attempts` times with a 5s backoff
    when the first attempt returns empty.

    The first SUCCESSFUL call on a cold cache pays the ~50s series-
    enrichment cost. Subsequent calls within TTL are instant.
    """
    key = tuple(sorted(formats))
    now = datetime.now(timezone.utc)
    cached = _SCHEDULE_CACHE.get(key)
    if cached and (now - cached["fetched_at"]) < _SCHEDULE_CACHE_TTL:
        return cached["schedule"]

    schedule: List[CREXMatch] = []
    for attempt in range(1, max_attempts + 1):
        scraper = CREXScraper()
        schedule = scraper.get_schedule(
            formats=formats, hours_ahead=max(hours_ahead, 96), hours_behind=3,
        )
        if schedule:
            break
        if attempt < max_attempts:
            logger.warning(
                f"CREX get_schedule returned 0 matches on attempt {attempt}; "
                f"retrying after 5s..."
            )
            time.sleep(5)

    if schedule:
        _SCHEDULE_CACHE[key] = {"schedule": schedule, "fetched_at": now}
    else:
        logger.error(
            f"CREX get_schedule returned 0 matches after {max_attempts} attempts. "
            "Schedule endpoint is currently unavailable; XI auto-fetch will be unavailable."
        )
    return schedule


def find_crex_match_for_fixture(
    pm_fixture: Dict[str, Any],
    formats: Optional[List[str]] = None,
    hours_ahead: int = 72,
) -> Optional[CREXMatch]:
    """Locate the CREX schedule entry that corresponds to a Polymarket fixture.

    Args:
        pm_fixture: must contain team1_db_name, team2_db_name, scheduled_start_utc (datetime), format
        formats: list of CREX format filters; defaults derived from pm_fixture['format']
        hours_ahead: schedule lookahead window for the CREX query

    Returns:
        CREXMatch with `match_url` populated, or None if no match could be paired.
    """
    pm_t1 = pm_fixture.get("team1_db_name") or pm_fixture.get("team1_label") or ""
    pm_t2 = pm_fixture.get("team2_db_name") or pm_fixture.get("team2_label") or ""
    if not pm_t1 or not pm_t2:
        return None
    # upcoming.py uses 'scheduled_start_estimate'; some callers might pass
    # 'scheduled_start_utc' instead. Accept either.
    pm_start = (pm_fixture.get("scheduled_start_estimate")
                or pm_fixture.get("scheduled_start_utc"))
    if pm_start is None:
        return None
    if isinstance(pm_start, str):
        try:
            pm_start = datetime.fromisoformat(pm_start.replace("Z", "+00:00"))
        except ValueError:
            return None
    pm_start_date = pm_start.astimezone(timezone.utc).strftime("%Y-%m-%d")

    pm_format = pm_fixture.get("format") or "T20"
    if formats is None:
        formats = ["T20"] if pm_format == "T20" else ["ODI", "T20"]

    schedule = _get_cached_schedule(formats, hours_ahead)

    for m in schedule:
        if not (m.team1_name and m.team2_name):
            continue
        if not _team_pair_matches(pm_t1, pm_t2, m.team1_name, m.team2_name):
            continue
        # Date check (allow ±1 day for timezone slop / overnight matches)
        if m.start_date:
            try:
                m_date = datetime.fromisoformat(m.start_date).date()
                pm_date = datetime.fromisoformat(pm_start_date).date()
                if abs((m_date - pm_date).days) > 1:
                    continue
            except (ValueError, AttributeError):
                pass
        return m
    return None


def fetch_xi_from_crex(
    pm_fixture: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Pull the current Playing XI from CREX for this Polymarket fixture.

    Returns:
        {
            "crex_match": CREXMatch (with team1/team2 having .crex_id),
            "team1_xi": List[str],   # CREX player names for team1
            "team2_xi": List[str],   # CREX player names for team2
            "fetched_at_utc": ISO string
        }
        or None on failure.
    """
    crex_match = find_crex_match_for_fixture(pm_fixture)
    if crex_match is None:
        logger.info(f"No CREX match found for fixture {pm_fixture.get('fixture_key')}")
        return None

    if not crex_match.match_url:
        logger.warning(f"CREX match has no match_url for fixture {pm_fixture.get('fixture_key')}")
        return None

    pm_t1 = pm_fixture.get("team1_db_name") or pm_fixture.get("team1_label") or ""
    pm_t2 = pm_fixture.get("team2_db_name") or pm_fixture.get("team2_label") or ""

    # Determine which CREX team corresponds to which Polymarket team
    cmp = PolymarketComparisonService.label_matches_team
    if cmp(crex_match.team1_name, pm_t1):
        crex_team_for_pm_t1 = crex_match.team1_name
        crex_team_for_pm_t2 = crex_match.team2_name
        crex_id_for_pm_t1 = crex_match.team1_id
        crex_id_for_pm_t2 = crex_match.team2_id
    else:
        crex_team_for_pm_t1 = crex_match.team2_name
        crex_team_for_pm_t2 = crex_match.team1_name
        crex_id_for_pm_t1 = crex_match.team2_id
        crex_id_for_pm_t2 = crex_match.team1_id

    # CREX uses team1/team2 in the URL as ordered there; pass both
    scraper = CREXScraper()
    try:
        crex_t1, crex_t2 = scraper.fetch_squads_with_playwright(
            crex_match.match_url,
            crex_match.team1_name, crex_match.team2_name,
            crex_match.team1_id, crex_match.team2_id,
        )
    except Exception as exc:
        logger.error(f"CREX fetch_squads_with_playwright failed: {exc}")
        return None

    if crex_t1 is None and crex_t2 is None:
        logger.warning(f"CREX returned no squads for {crex_match.match_url}")
        return None

    # Re-orient CREX teams so the FIRST element matches our pm_t1
    pm_t1_squad: Optional[Any] = None
    pm_t2_squad: Optional[Any] = None
    for crex_team in (crex_t1, crex_t2):
        if crex_team is None:
            continue
        if cmp(crex_team.name, pm_t1):
            pm_t1_squad = crex_team
        elif cmp(crex_team.name, pm_t2):
            pm_t2_squad = crex_team

    def _player_names(squad) -> Tuple[List[str], List[str]]:
        """Return (first_11, full_squad). CREX's playingxi-button tab
        displays the Playing XI first followed by the rest of the squad,
        so the first 11 entries are typically the confirmed XI when CREX
        has it published, or the predicted XI before toss."""
        if not squad or not getattr(squad, "players", None):
            return [], []
        all_names = [p.name for p in squad.players if p.name]
        return all_names[:11], all_names

    t1_xi, t1_full = _player_names(pm_t1_squad)
    t2_xi, t2_full = _player_names(pm_t2_squad)
    return {
        "crex_match_url": crex_match.match_url,
        "crex_team1_name": crex_match.team1_name,
        "crex_team2_name": crex_match.team2_name,
        "crex_match_id": crex_match.crex_id,
        "crex_status": crex_match.status,
        "team1_xi": t1_xi,
        "team2_xi": t2_xi,
        "team1_full_squad": t1_full,
        "team2_full_squad": t2_full,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
    }
