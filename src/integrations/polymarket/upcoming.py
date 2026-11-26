"""Wave 5.7: discover upcoming Polymarket cricket events for paper-trading.

Pulls open Polymarket cricket events via Gamma /events?tag_slug=cricket and
groups sub-markets by fixture so each upcoming match comes back as a single
record with its moneyline + side markets attached.

Slug taxonomy (observed Apr 2026):
    Moneyline:   cricipl-luc-kkr-2026-04-26          (tournament, t1, t2, date)
    Toss double: cricipl-luc-kkr-2026-04-26-toss-match-double-luc
    Most sixes:  cricipl-luc-kkr-2026-04-26-most-sixes-luc
    Top batter:  cricipl-luc-kkr-2026-04-26-team-top-batter-luc

Tournament prefix examples: cricipl, cricpsl, cricbbl, cricwbbl, crwpl20 (WPL),
crint (all internationals). The authoritative list is Gamma `/sports`
(cricket entries carry tag 517); TOURNAMENT_PREFIX_MAP mirrors it.

The match-date in the slug is the TRUE start day. endDate on the Gamma event
is the settlement deadline (typically a few days after match).
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
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

# Static prefix -> (cricket_format, gender, human_name), mirroring the cricket
# entries of Gamma `/sports` (refreshed 2026-11-07; 88 `cr*` prefixes).
# None = title-dependent (internationals and the legacy per-country containers,
# which mix formats/genders); see `_infer_format_gender_from_title`.
# T10 / Hundred / sixes formats are routed to the T20 model; first-class and
# Test cricket is "TEST" (3-way market, no model, never a 2-way moneyline).
TOURNAMENT_PREFIX_MAP = {
    # --- Internationals and legacy country containers (title-dependent) ---
    "crint":            None,
    "craus":            None,
    "crban":            None,
    "creng":            None,
    "crind":            None,
    "crnew":            None,
    "crpak":            None,
    "crsou":            None,
    "cruae":            None,
    "crbtnmlyhkg20":    None,
    "cru19wc":          None,
    "crafgwi20":        ("T20", "male",   "Afghanistan vs West Indies (T20I)"),
    "crwt20wcgq":       ("T20", "female", "Women's T20 World Cup Qualifier"),
    # --- Men's T20 franchise / domestic leagues ---
    "cricipl":          ("T20", "male",   "Indian Premier League"),
    "cricpsl":          ("T20", "male",   "Pakistan Super League"),
    "cricbbl":          ("T20", "male",   "Big Bash League"),
    "cricbpl":          ("T20", "male",   "Bangladesh Premier League"),
    "criccpl":          ("T20", "male",   "Caribbean Premier League"),
    "cricsa20":         ("T20", "male",   "SA20"),
    "cricilt20":        ("T20", "male",   "International League T20"),
    "cricmlc":          ("T20", "male",   "Major League Cricket"),
    "crict20lpl":       ("T20", "male",   "Lanka Premier League"),
    "crict20blast":     ("T20", "male",   "Vitality T20 Blast"),
    "crichundred":      ("T20", "male",   "The Hundred"),
    "cricnpl":          ("T20", "male",   "Nepal Premier League"),
    "cricnt20c":        ("T20", "male",   "National T20 Cup"),
    "cricpakt20cup":    ("T20", "male",   "Pakistan National T20 Cup"),
    "criclcl":          ("T20", "male",   "Legends League Cricket"),
    "cricwcl":          ("T20", "male",   "World Championship of Legends"),
    "crictbcl":         ("T20", "male",   "T20 Brisbane Champions League"),
    "cricgsl":          ("T20", "male",   "Global Super League"),
    "crichksixes":      ("T20", "male",   "Hong Kong Sixes"),
    "cricapl":          ("T20", "male",   "Andhra Premier League"),
    "cricassam":        ("T20", "male",   "Assam Premier League"),
    "cricdpl":          ("T20", "male",   "Delhi Premier League"),
    "cricetpl":         ("T20", "male",   "European T20 Premier League"),
    "cricfalcons":      ("T20", "male",   "Falcons Super League T20"),
    "cricjcl":          ("T20", "male",   "JCL T20"),
    "crickcct20":       ("T20", "male",   "KCC T20 Summer League"),
    "crickcl":          ("T20", "male",   "Kerala Cricket League T20"),
    "crickerala":       ("T20", "male",   "Kuwait Kerala Premier League T20"),
    "crickpl":          ("T20", "male",   "KPL Indo Nepal T20"),
    "cricmaharaja":     ("T20", "male",   "Maharaja T20 Trophy"),
    "cricmajort20":     ("T20", "male",   "Major Clubs T20"),
    "cricmcat20":       ("T20", "male",   "Singer-MCA Super Premier T20"),
    "cricmukono":       ("T20", "male",   "Mukono Super Smash T20"),
    "cricnsk":          ("T20", "male",   "NSK Trophy State T20"),
    "cricppl":          ("T20", "male",   "Pondicherry Premier League"),
    "cricsherepunjab":  ("T20", "male",   "Sher-E-Punjab T20"),
    "cricshpageeza":    ("T20", "male",   "Shpageeza Cricket League"),
    "crict20max":       ("T20", "male",   "KFC T20 Max"),
    "crictelangana":    ("T20", "male",   "Telangana T20"),
    "crictnpl":         ("T20", "male",   "Tamil Nadu Premier League"),
    "crictopend":       ("T20", "male",   "Top End T20 Series"),
    "cricupt20":        ("T20", "male",   "Uttar Pradesh T20"),
    "cricecsbg":        ("T20", "male",   "ECS Bulgaria"),
    "cricecsch":        ("T20", "male",   "ECS Switzerland"),
    "cricecseng":       ("T20", "male",   "ECS England"),
    "cricecspt":        ("T20", "male",   "ECS Portugal"),
    # NZ Super Smash runs men's and women's competitions under one prefix.
    "cricsm":           None,
    # --- T10 (routed to the T20 model) ---
    "cricadt10":        ("T20", "male",   "Abu Dhabi T10 League"),
    "criccoppat10":     ("T20", "male",   "Coppa il Mondo T10"),
    "cricgermant10":    ("T20", "male",   "German Super League Hamburg T10"),
    "cricnrwt10":       ("T20", "male",   "German Super League NRW T10"),
    "cricjclt10":       ("T20", "male",   "JCL T10"),
    "cricthunderbolt":  ("T20", "male",   "Thunderbolt T10 League"),
    # --- Women's T20 ---
    "cricwbbl":         ("T20", "female", "Women's Big Bash League"),
    "crwpl20":          ("T20", "female", "Women's Premier League"),
    "crict20blastw":    ("T20", "female", "Vitality T20 Blast Women"),
    "crict20blastl2w":  ("T20", "female", "Vitality T20 Blast Women League 2"),
    "crichundredw":     ("T20", "female", "The Hundred Women"),
    "criccsat20w":      ("T20", "female", "CSA T20 Women"),
    "cricdplw":         ("T20", "female", "Women's Delhi Premier League T20"),
    "crichkt20w":       ("T20", "female", "Hong Kong T20 Premier League Women"),
    "cricmaharani":     ("T20", "female", "Maharani Trophy KSCA T20"),
    "crict20plw":       ("T20", "female", "T20 Premier League Women"),
    # --- 50-over (List A) ---
    "cricodc":          ("ODI", "male",   "Metro Bank One Day Cup"),
    "cricinterprov":    ("ODI", "male",   "Inter-Provincial One Day"),
    "cricomand50":      ("ODI", "male",   "Oman D50"),
    "cricdhakadpl":     ("ODI", "male",   "Dhaka Premier Division Cricket League"),
    "cricodcw":         ("ODI", "female", "Metro Bank One Day Cup Women"),
    "cricodcl2w":       ("ODI", "female", "Metro Bank One Day Cup Women League 2"),
    "cricwncl":         ("ODI", "female", "Women's National Cricket League"),
    "crwncl":           ("ODI", "female", "Women's National Cricket League"),
    # --- First-class ---
    "cricranji":        ("TEST", "male",  "Ranji Trophy"),
    "cricshield":       ("TEST", "male",  "Sheffield Shield"),
    "cricss":           ("TEST", "male",  "Sheffield Shield"),
    "cricps":           ("TEST", "male",  "Plunket Shield"),
}

# Age-group / development-side markers in Polymarket team labels or titles.
# The DB has only senior teams, so these fixtures must never be matched to a
# senior side (and are never bet).
_U19_RE = re.compile(r"\bu(?:nder)?[\s\-]?19s?\b", re.I)
_DEV_SIDE_RE = re.compile(
    r"(?:\S\s+A|^England Lions|\bEmerging|\bAcademy|\bHigh Performance|\bDevelopment XI)"
    r"(?:\s+(?:Women|W))?\s*$",
    re.I,
)


def infer_age_group(title: str, *labels: Optional[str]) -> str:
    """'u19', 'a_team' or 'senior' for a fixture, from its title and team labels."""
    texts = [title or ""] + [l or "" for l in labels]
    if any(_U19_RE.search(t) for t in texts):
        return "u19"
    if any(l and _DEV_SIDE_RE.search(l.strip()) for l in labels):
        return "a_team"
    if re.search(r"\bA Team\b|\bunofficial (?:odi|test)s?\b", title or "", re.I):
        return "a_team"
    return "senior"


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
    # Age group says nothing about gender: "ACC Men's U19 Asia Cup" is male.
    is_women = bool(re.search(r"\bwom[ae]n(?:'s)?\b|\bwomens\b|\bw\b", title_lower))
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


# Gamma `sportsMarketType` -> our market kind. A "moneyline" with Yes/No
# outcomes is one leg of a 3-way (win/draw/win) Test or first-class market.
SPORTS_MARKET_TYPE_MAP = {
    "moneyline": "moneyline",
    "cricket_toss_winner": "toss_winner",
    "cricket_completed_match": "completed_match",
    "cricket_team_top_batter": "top_batter",
    "cricket_most_sixes": "most_sixes",
    "cricket_toss_match_double": "toss_match_double",
    "cricket_match_to_go_till": "match_to_go_till",
}


def _classify_market_within_event(
    question: str,
    outcomes: List[Any],
    event_slug_kind: str,
    sports_market_type: Optional[str] = None,
) -> str:
    """Within the bare-slug event, multiple markets coexist. Gamma's
    `sportsMarketType` is authoritative; question text / outcomes and the
    slug-derived event kind are a fallback for markets that lack it.
    """
    q_lower = (question or "").lower()
    outcome_strs = [str(o).lower() for o in outcomes]
    is_yes_no = set(outcome_strs) == {"yes", "no"}

    smt_kind = SPORTS_MARKET_TYPE_MAP.get((sports_market_type or "").strip().lower())
    if smt_kind == "moneyline":
        return "moneyline" if (not is_yes_no and len(outcome_strs) == 2) else "moneyline_3way_leg"
    if smt_kind:
        return smt_kind
    if sports_market_type:
        return "other"

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
                "age_group": (
                    "u19" if any(re.search(r"u19\d*$", t) for t in (t1, t2))
                    else infer_age_group(ev.get("title") or "")
                ),
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
                if rec["age_group"] == "senior":
                    rec["age_group"] = infer_age_group(title, t1_label, t2_label)

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
                sports_market_type=m.get("sportsMarketType"),
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


# Fuzzy fallback acceptance floor (SequenceMatcher ratio on the label core vs
# the DB name core). High on purpose: a wrong team is far worse than no match,
# and known spelling variants belong in team_external_ids instead.
MIN_FUZZY_CONFIDENCE = 0.90
# Runner-up must trail the best fuzzy candidate by at least this much.
MIN_FUZZY_MARGIN = 0.05
_LABEL_STOPWORDS = {"and", "the", "of", "cc", "cricket", "club"}
_GENDER_SUFFIX_RE = re.compile(r"\s+(?:women|w)$")


def polymarket_alias_key(label: str) -> str:
    """Normalised, gender-neutral key used for team_external_ids lookups."""
    return _GENDER_SUFFIX_RE.sub("", normalize_team_name(label or "").strip())


def _name_core(text: str) -> str:
    toks = [t for t in polymarket_alias_key(text).split() if t not in _LABEL_STOPWORDS]
    return " ".join(toks)


def _team_gender_matches(cur, team_id: int, gender: str) -> int:
    cur.execute(
        "SELECT COUNT(*) FROM matches WHERE (team1_id = ? OR team2_id = ?) AND gender = ?",
        (team_id, team_id, gender),
    )
    return int(cur.fetchone()[0])


def _row_to_match(row, method: str, confidence: float) -> Dict[str, Any]:
    return {
        "team_id": row[0], "name": row[1], "team_type": row[2],
        "match_method": method, "confidence": confidence,
    }


def map_team_label_to_db(label: str, fmt: str, gender: str, conn) -> Optional[Dict[str, Any]]:
    """Map a Polymarket team label to a DB team, failing closed.

    Order: explicit alias (team_external_ids, source='polymarket') -> exact
    name (label, or label minus a Women/W suffix) -> strict fuzzy fallback
    (>= MIN_FUZZY_CONFIDENCE, unambiguous). Every path also requires the team
    to have match history in the fixture's gender; otherwise returns None.
    """
    if not label:
        return None
    cur = conn.cursor()
    key = polymarket_alias_key(label)
    if not key:
        return None

    cur.execute(
        """
        SELECT t.team_id, t.name, t.team_type
        FROM team_external_ids e
        JOIN teams t ON t.franchise_id = e.group_id
        WHERE e.source = 'polymarket' AND e.external_id = ?
        ORDER BY (t.team_id = t.canonical_team_id) DESC, t.team_id
        """,
        (key,),
    )
    for row in cur.fetchall():
        if _team_gender_matches(cur, row[0], gender) > 0:
            return _row_to_match(row, "alias", 1.0)

    full = normalize_team_name(label).strip()
    for name in dict.fromkeys((full, key)):
        cur.execute("SELECT team_id, name, team_type FROM teams WHERE LOWER(name) = ?", (name,))
        row = cur.fetchone()
        if row and _team_gender_matches(cur, row[0], gender) > 0:
            return _row_to_match(row, "exact", 1.0)

    core = _name_core(label)
    tokens = [t for t in core.split() if len(t) >= 3]
    if not tokens:
        return None
    clauses = " OR ".join(["LOWER(name) LIKE ?"] * len(tokens))
    cur.execute(
        f"SELECT team_id, name, team_type FROM teams WHERE {clauses}",
        [f"%{t}%" for t in tokens],
    )
    scored = sorted(
        ((SequenceMatcher(a=core, b=_name_core(r[1])).ratio(), r) for r in cur.fetchall()),
        key=lambda x: -x[0],
    )
    if not scored or scored[0][0] < MIN_FUZZY_CONFIDENCE:
        return None
    if len(scored) > 1 and scored[0][0] - scored[1][0] < MIN_FUZZY_MARGIN:
        return None
    best_score, best = scored[0]
    if _team_gender_matches(cur, best[0], gender) == 0:
        return None
    return _row_to_match(best, "fuzzy", round(best_score, 3))


def attach_db_team_ids(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach team1_id + team2_id (DB ids) to each upcoming fixture.

    Fixtures are left unmatched (ids None, `unmatched_reason` set) when they
    are U19 / A-team / development fixtures (the DB has only senior sides),
    when either label fails `map_team_label_to_db`, or when both labels
    resolve to the same team. Every scanner skips fixtures without both ids.
    """
    from src.data.database import get_db_connection
    out = []
    with get_db_connection() as conn:
        for ev in events:
            fmt = ev.get("format", "T20")
            gender = ev.get("gender", "male")
            ev_out = dict(ev)
            age_group = ev.get("age_group", "senior")
            if age_group != "senior":
                t1_match = t2_match = None
                reason = f"age_group={age_group}"
            else:
                t1_match = map_team_label_to_db(ev["team1_label"], fmt, gender, conn) if ev.get("team1_label") else None
                t2_match = map_team_label_to_db(ev["team2_label"], fmt, gender, conn) if ev.get("team2_label") else None
                reason = None
                if t1_match and t2_match and t1_match["team_id"] == t2_match["team_id"]:
                    t1_match = t2_match = None
                    reason = "both labels resolved to the same team"
                elif not (t1_match and t2_match):
                    reason = "team label unresolved"
            ev_out["team1_id"] = t1_match["team_id"] if t1_match else None
            ev_out["team1_db_name"] = t1_match["name"] if t1_match else None
            ev_out["team1_match_method"] = t1_match["match_method"] if t1_match else None
            ev_out["team2_id"] = t2_match["team_id"] if t2_match else None
            ev_out["team2_db_name"] = t2_match["name"] if t2_match else None
            ev_out["team2_match_method"] = t2_match["match_method"] if t2_match else None
            ev_out["unmatched_reason"] = reason
            out.append(ev_out)
    return out
