"""Wave 5.7: discover upcoming Polymarket cricket events for paper-trading.

Pulls open Polymarket cricket events via Gamma /events?tag_slug=cricket and
groups sub-markets by fixture so each upcoming match comes back as a single
record with its moneyline + side markets attached.

Slug taxonomy (observed Apr 2026):
    Moneyline:   cricipl-luc-kkr-2026-04-26          (tournament, t1, t2, date)
    Toss double: cricipl-luc-kkr-2026-04-26-toss-match-double-luc
    Most sixes:  cricipl-luc-kkr-2026-04-26-most-sixes-luc
    Top batter:  cricipl-luc-kkr-2026-04-26-team-top-batter-luc

Tournament prefix examples: cricipl, cricpsl, cricbbl, criclcl, cricwpl,
cricint (international), cricwt (women's tournaments). Generally any slug
starting with "cric" + 2-5 lowercase letters.

The match-date in the slug is the TRUE start day. endDate on the Gamma event
is the settlement deadline (typically a few days after match).
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.integrations.polymarket import PolymarketClient
from src.integrations.odds.polymarket_compare import (
    PolymarketComparisonService,
    _coerce_list,
    _coerce_float,
    normalize_team_name,
)

logger = logging.getLogger(__name__)

# Match a moneyline slug: prefix-team1-team2-YYYY-MM-DD (no further suffix).
# Polymarket uses two prefix conventions:
#   "cric*"  for major T20 league fixtures (cricipl, cricpsl, criclcl, ...)
#   "crint"  for ALL internationals (men ODI, women T20, U19, etc.)
# Both share the same team-team-date suffix shape; differentiate format/gender
# downstream from the event title.
# Team tokens are usually 3-letter abbrevs (luc, kkr) but internationals often
# add a digit for duplicate ISO codes in the same series (wst2, lka2, bgd3).
_TEAM_SLUG = r"[a-z0-9]{2,8}"
SLUG_MONEYLINE_RE = re.compile(
    rf"^(?P<prefix>cr[a-z0-9]{{2,14}})-(?P<t1>{_TEAM_SLUG})-(?P<t2>{_TEAM_SLUG})-(?P<date>\d{{4}}-\d{{2}}-\d{{2}})$"
)
SLUG_SUBMARKET_RE = re.compile(
    rf"^(?P<prefix>cr[a-z0-9]{{2,14}})-(?P<t1>{_TEAM_SLUG})-(?P<t2>{_TEAM_SLUG})-(?P<date>\d{{4}}-\d{{2}}-\d{{2}})-(?P<suffix>.+)$"
)

# Static prefix -> (cricket_format, gender, human_name) for unambiguous prefixes.
# `crint` is title-dependent and resolved by `_infer_format_gender_from_title`.
TOURNAMENT_PREFIX_MAP = {
    "cricipl":   ("T20", "male",   "Indian Premier League"),
    "cricpsl":   ("T20", "male",   "Pakistan Super League"),
    "cricbbl":   ("T20", "male",   "Big Bash League"),
    "cricwbbl":  ("T20", "female", "Women's Big Bash"),
    "cricwpl":   ("T20", "female", "Women's Premier League"),
    "criclcl":   ("T20", "male",   "Legends Cricket League"),
    "crictbcl":  ("T20", "male",   "Brisbane Champions League"),
    "crint":     None,  # title-dependent; see _infer_format_gender_from_title
    "cricint":   ("T20", "male",   "International (T20I)"),
    "cricodi":   ("ODI", "male",   "ODI International"),
    "cricwt":    ("T20", "female", "Women's T20"),
    "cricwodi":  ("ODI", "female", "Women's ODI"),
    "cricbifa":   ("T20", "female", "BIFA Women's"),
    "cricwt20":   ("T20", "female", "Women's T20"),
    "cricmlc":    ("T20", "male",   "Major League Cricket"),
    "crict20blast":  ("T20", "male",   "Vitality T20 Blast"),
    # Polymarket's observed women's-Blast prefix is "crict20blastw" (suffix -w),
    # not "crict20wblast"; keep both so either taxonomy resolves.
    "crict20blastw": ("T20", "female", "Women's Vitality T20 Blast"),
    "crict20wblast": ("T20", "female", "Women's T20 Blast"),
}


def _infer_format_gender_from_title(title: str, prefix: str) -> Tuple[str, str, str]:
    """Resolve (format, gender, tournament_name) for prefixes whose static
    map entry is None (currently just 'crint' which covers many shapes).

    Heuristics from observed Polymarket titles:
        "ICC Cricket World Cup League Two: ..." -> ODI men
        "ODI Series ...: ..." -> ODI (gender from 'Women' in title)
        "T20 Challenge Trophy, Women: ..." -> T20 women
        "T20 Series ...: ..." -> T20 men (or women if title contains 'Women')
        "T20I Series ...: ..." -> T20 men/women
        "ICC Women's ..." -> T20/ODI women
    """
    static = TOURNAMENT_PREFIX_MAP.get(prefix)
    if isinstance(static, tuple):
        return static

    title_lower = (title or "").lower()
    is_women = (
        "women" in title_lower or "women's" in title_lower
        or " w " in title_lower or title_lower.endswith(" w")
        or "u19" in title_lower
    )
    gender = "female" if is_women else "male"

    # Format detection: explicit format tokens dominate, otherwise infer from
    # tournament name (World Cup League Two = ODI; T20 Challenge Trophy = T20)
    fmt = "T20"
    if "odi" in title_lower or "world cup league" in title_lower or "50-over" in title_lower:
        fmt = "ODI"
    elif "test" in title_lower:
        fmt = "TEST"
    elif "t10" in title_lower:
        fmt = "T20"  # treat T10 as T20 for sim purposes

    # Tournament name = portion before ":" (or full title if no colon)
    tour_name = title.split(":", 1)[0].strip() if title and ":" in title else (title or prefix.upper())
    if len(tour_name) > 50:
        tour_name = tour_name[:47] + "..."
    return fmt, gender, tour_name


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        d = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return d.astimezone(timezone.utc) if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        return None


def _classify_market_within_event(
    question: str,
    outcomes: List[Any],
    event_slug_kind: str,
) -> str:
    """Within the bare-slug event, multiple markets coexist. Tell them apart
    by outcomes + question text, falling back to the slug-derived event kind.
    """
    q_lower = (question or "").lower()
    outcome_strs = [str(o).lower() for o in outcomes]
    is_yes_no = set(outcome_strs) == {"yes", "no"}

    # Sub-market events (slug suffix already classified): trust the slug
    if event_slug_kind in ("toss_match_double", "most_sixes", "top_batter", "toss_winner"):
        return event_slug_kind

    # Inside a moneyline-ish bare-slug event:
    if "completed match" in q_lower or "completed-match" in q_lower:
        return "completed_match"
    if "who wins" in q_lower:
        return "moneyline_alt"
    if not is_yes_no and len(outcome_strs) == 2:
        # Two team-name outcomes -> THIS is the real moneyline
        return "moneyline"
    # Other Yes/No markets in a bare-slug event are fallback
    return "other"


def _classify_slug(slug: str) -> Optional[Dict[str, str]]:
    """Parse a Polymarket cricket slug into its components."""
    if not slug:
        return None
    m = SLUG_MONEYLINE_RE.match(slug)
    if m:
        d = m.groupdict()
        d["kind"] = "moneyline"
        return d
    m = SLUG_SUBMARKET_RE.match(slug)
    if m:
        d = m.groupdict()
        suffix = d["suffix"]
        if "toss-match-double" in suffix:
            kind = "toss_match_double"
        elif "most-sixes" in suffix:
            kind = "most_sixes"
        elif "team-top-batter" in suffix:
            kind = "top_batter"
        elif suffix == "completed-match":
            kind = "moneyline"  # alternate moneyline name
        elif "toss-winner" in suffix:
            kind = "toss_winner"
        elif suffix in ("more-markets", "more-markets-winner"):
            kind = "other"
        else:
            kind = "other"
        d["kind"] = kind
        return d
    return None


def find_upcoming_cricket_events(
    client: PolymarketClient,
    hours_ahead: float = 96.0,
    page_size: int = 100,
    max_offset: int = 2000,
    include_started: bool = False,
) -> List[Dict[str, Any]]:
    """Return open cricket fixtures with kickoff in the next `hours_ahead` hours.

    Each entry has:
        fixture_key                 - "{prefix}-{t1}-{t2}-{date}"
        tournament_prefix           - e.g. "cricipl"
        tournament_name             - e.g. "Indian Premier League"
        format                      - "T20" or "ODI"
        gender                      - "male" or "female"
        team1_abbr, team2_abbr      - 3-letter abbrevs from slug
        team1_label, team2_label    - human names from event title
        match_date                  - "YYYY-MM-DD"
        scheduled_start_estimate    - datetime (best-effort: endDate minus 6h buffer)
        moneyline                   - dict or None: {market_id, outcomes, token_ids}
        side_markets                - dict[kind] -> list of side-market events
    """
    now = datetime.now(timezone.utc)
    horizon = now.timestamp() + hours_ahead * 3600

    # Pull all open cricket events.
    # Polymarket uses two separate tags: "cricket" (international / IPL / PSL)
    # and "t20-blast" (Vitality T20 Blast).  Query both so we don't miss any.
    TAGS_TO_QUERY = ["cricket", "t20-blast"]
    raw_events: List[Dict[str, Any]] = []
    seen_event_ids: set = set()
    for tag_slug in TAGS_TO_QUERY:
        for offset in range(0, max_offset, page_size):
            try:
                rows = client.get_events(
                    limit=page_size,
                    offset=offset,
                    tag_slug=tag_slug,
                    closed=False,
                    active=True,
                )
            except Exception as exc:
                logger.warning(f"Polymarket Gamma /events open fetch failed (tag={tag_slug}): {exc}")
                break
            if not isinstance(rows, list) or not rows:
                break
            for ev in rows:
                evid = str(ev.get("id") or ev.get("slug") or "")
                if evid in seen_event_ids:
                    continue
                seen_event_ids.add(evid)
                raw_events.append(ev)
            if len(rows) < page_size:
                break

    # Group by fixture key derived from slug
    by_fixture: Dict[str, Dict[str, Any]] = {}
    for ev in raw_events:
        slug = ev.get("slug") or ""
        info = _classify_slug(slug)
        if info is None:
            continue
        prefix = info["prefix"]
        t1, t2, date = info["t1"], info["t2"], info["date"]
        kind = info["kind"]
        # Canonicalise the team-pair so reverse listings (e.g. cricipl-gt-csk and
        # cricipl-csk-gt on the same day) collapse into one fixture
        pair = tuple(sorted([t1, t2]))
        fixture_key = f"{prefix}-{pair[0]}-{pair[1]}-{date}"

        # Estimate match start. Priority order:
        #   1. Per-market `gameStartTime` from Gamma (most accurate;
        #      newly observed post-2026-04 migration).
        #   2. Default by prefix: IPL/PSL evening 14:00 UTC, BBL 09:00 UTC.
        # The slug date alone is misleading - Polymarket uses Pacific
        # Time for slug naming, so e.g. crint-idn-mys-2026-04-30 has
        # gameStartTime = 2026-05-01T02:00 UTC (~12 hr offset).
        prefix = info["prefix"]
        match_start_est = None
        for m in ev.get("markets") or []:
            if not isinstance(m, dict):
                continue
            gst_str = m.get("gameStartTime") or m.get("game_start_time")
            if gst_str:
                try:
                    parsed = datetime.fromisoformat(str(gst_str).replace(" +00", "+00:00").replace(" UTC", "+00:00").replace("Z", "+00:00"))
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=timezone.utc)
                    match_start_est = parsed.astimezone(timezone.utc)
                    break
                except (ValueError, AttributeError):
                    pass
        if match_start_est is None:
            if prefix in ("cricbbl", "cricwbbl"):
                default_start_utc = "T09:00:00Z"
            else:
                default_start_utc = "T14:00:00Z"
            match_start_est = _parse_iso(date + default_start_utc)

        if match_start_est is None:
            continue
        if match_start_est.timestamp() > horizon:
            continue
        if not include_started and match_start_est.timestamp() < now.timestamp() - 600:
            continue

        # Initialise fixture record. Resolve format/gender from the title
        # for ambiguous prefixes (`crint` is ODI men, T20 women, etc.).
        if fixture_key not in by_fixture:
            fmt, gender, tour_name = _infer_format_gender_from_title(
                ev.get("title") or "", prefix,
            )
            by_fixture[fixture_key] = {
                "fixture_key": fixture_key,
                "tournament_prefix": prefix,
                "tournament_name": tour_name,
                "format": fmt,
                "gender": gender,
                "team1_abbr": t1,
                "team2_abbr": t2,
                "team1_label": None,
                "team2_label": None,
                "match_date": date,
                "scheduled_start_estimate": match_start_est,
                "moneyline": None,
                "side_markets": defaultdict(list),
                "raw_events": [],
            }
        rec = by_fixture[fixture_key]
        # Bump scheduled_start_estimate to the EARLIEST estimate across markets
        if match_start_est < rec["scheduled_start_estimate"]:
            rec["scheduled_start_estimate"] = match_start_est

        # Parse out the team labels from the event title (first kind-of fixture seen)
        title = ev.get("title") or ""
        if rec["team1_label"] is None or rec["team2_label"] is None:
            t1_label, t2_label = _parse_team_names_from_title(title)
            if t1_label and t2_label:
                rec["team1_label"] = t1_label
                rec["team2_label"] = t2_label

        # Extract this event's market(s).
        # NOTE: for the bare-slug event (kind="moneyline" by slug pattern),
        # the EVENT actually contains multiple markets:
        #   - the real moneyline (outcomes=[team1_name, team2_name])
        #   - "Completed match?" (outcomes=[Yes, No])
        #   - "Who wins?" alternate moneyline
        # We classify each market individually inside the event.
        for m in ev.get("markets") or []:
            if not isinstance(m, dict):
                continue
            outcomes = _coerce_list(m.get("outcomes")) or []
            token_ids = _coerce_list(m.get("clobTokenIds") or m.get("clobTokenIDs")) or []
            outcome_prices_raw = m.get("outcomePrices")
            last_prices: List[Optional[float]] = []
            if isinstance(outcome_prices_raw, str):
                try:
                    parsed = json.loads(outcome_prices_raw)
                    if isinstance(parsed, list):
                        last_prices = [_coerce_float(v) for v in parsed]
                except json.JSONDecodeError:
                    pass
            elif isinstance(outcome_prices_raw, list):
                last_prices = [_coerce_float(v) for v in outcome_prices_raw]

            outcomes_with_meta = []
            for idx, label in enumerate(outcomes):
                outcomes_with_meta.append({
                    "label": str(label),
                    "token_id": str(token_ids[idx]) if idx < len(token_ids) else None,
                    "last_price": last_prices[idx] if idx < len(last_prices) else None,
                })

            # Reclassify the SPECIFIC market based on its outcomes + question text
            market_kind = _classify_market_within_event(
                question=m.get("question") or "",
                outcomes=outcomes,
                event_slug_kind=kind,
            )

            market_record = {
                "kind": market_kind,
                "event_slug": slug,
                "event_id": str(ev.get("id") or ""),
                "market_id": str(m.get("id") or m.get("conditionId") or ""),
                "question": m.get("question"),
                "outcomes": outcomes_with_meta,
                "raw_market": m,
            }
            if market_kind == "moneyline":
                # Prefer the cleanest moneyline (team-name outcomes); only overwrite
                # if rec.moneyline is None (first match) or current is alternate
                if rec["moneyline"] is None:
                    rec["moneyline"] = market_record
            elif market_kind == "completed_match":
                rec["completed_match"] = market_record
            elif market_kind == "moneyline_alt":
                # Alternate moneyline ("Who wins?") - keep as fallback
                if rec.get("moneyline_alt") is None:
                    rec["moneyline_alt"] = market_record
            else:
                rec["side_markets"][market_kind].append(market_record)
        rec["raw_events"].append(ev)

    # Convert defaultdicts and sort
    out = []
    for rec in by_fixture.values():
        rec["side_markets"] = dict(rec["side_markets"])
        out.append(rec)
    return sorted(out, key=lambda r: r["scheduled_start_estimate"])


def _parse_team_names_from_title(title: str) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort: extract two team names from event title.

    Examples:
        'Pakistan Super League: Karachi Kings vs Hyderabad Kingsmen - Toss'
        'Indian Premier League: Lucknow Super Giants vs Kolkata Knight Riders'
    """
    if not title:
        return None, None
    # Strip tournament prefix (everything before first ':')
    if ":" in title:
        title = title.split(":", 1)[1].strip()
    # Strip sub-market suffix (everything after ' - ')
    if " - " in title:
        title = title.split(" - ", 1)[0].strip()
    for sep in (" vs. ", " vs ", " v "):
        if sep in title:
            parts = title.split(sep, 1)
            if len(parts) == 2:
                t1, t2 = parts[0].strip(), parts[1].strip()
                if t1 and t2:
                    return t1, t2
    return None, None


def map_team_label_to_db(label: str, fmt: str, gender: str, conn) -> Optional[Dict[str, Any]]:
    """Map a Polymarket team label to a DB team_id via fuzzy matching."""
    if not label:
        return None
    cur = conn.cursor()
    norm_label = normalize_team_name(label).strip()
    if not norm_label:
        return None
    cur.execute("SELECT team_id, name, team_type FROM teams WHERE LOWER(name) = ?", (norm_label,))
    row = cur.fetchone()
    if row:
        return dict(row)
    tokens = [t for t in norm_label.split() if len(t) >= 3]
    if not tokens:
        return None
    distinctive = max(tokens, key=len)
    cur.execute(
        """
        SELECT t.team_id, t.name, t.team_type, COUNT(m.match_id) AS recent_matches
        FROM teams t
        LEFT JOIN matches m ON (m.team1_id = t.team_id OR m.team2_id = t.team_id)
            AND m.match_type = ? AND m.gender = ? AND m.date >= '2024-01-01'
        WHERE LOWER(t.name) LIKE ?
        GROUP BY t.team_id
        ORDER BY recent_matches DESC, LENGTH(t.name) ASC
        LIMIT 5
        """,
        (fmt, gender, f"%{distinctive}%"),
    )
    candidates = [dict(r) for r in cur.fetchall()]
    if not candidates:
        return None
    return candidates[0]


def attach_db_team_ids(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach team1_id + team2_id (DB ids) to each upcoming fixture by fuzzy
    matching the labels parsed from event titles.

    Format/gender for each fixture is taken from TOURNAMENT_PREFIX_MAP entries
    on the fixture itself.
    """
    from src.data.database import get_connection, get_db_connection
    out = []
    with get_db_connection() as conn:
        for ev in events:
            fmt = ev.get("format", "T20")
            gender = ev.get("gender", "male")
            t1_match = map_team_label_to_db(ev["team1_label"], fmt, gender, conn) if ev.get("team1_label") else None
            t2_match = map_team_label_to_db(ev["team2_label"], fmt, gender, conn) if ev.get("team2_label") else None
            ev_out = dict(ev)
            ev_out["team1_id"] = t1_match["team_id"] if t1_match else None
            ev_out["team1_db_name"] = t1_match["name"] if t1_match else None
            ev_out["team2_id"] = t2_match["team_id"] if t2_match else None
            ev_out["team2_db_name"] = t2_match["name"] if t2_match else None
            out.append(ev_out)
    return out
