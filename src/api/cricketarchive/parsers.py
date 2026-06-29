"""HTML parsers for CricketArchive pages.

All parsers take raw HTML (from the cached fetcher) + the source URL and return
the dataclasses in ``models``. CricketArchive tables are class-less and keyed by
their header row, so we identify them by content, not CSS.

Pages handled:
* parse_event_matches  -> /Archive/Events/...        (series -> scorecard URLs)
* parse_scorecard      -> /Archive/Scorecards/{f}/{id}.html
* parse_player_bio     -> /Archive/Players/{f}/{id}/{id}.html
* parse_commentary     -> .../{id}_commentary_iN_page.html  (ball-by-ball)
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from .models import (
    CAPlayer, CAEventMatch, CAScorecard, CAInnings,
    CABattingEntry, CABowlingEntry, CADelivery,
)

BASE = "https://cricketarchive.com"

_MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}


# ---------------------------------------------------------------------------
# ID + value helpers
# ---------------------------------------------------------------------------
def _player_id(href: str) -> Optional[str]:
    m = re.search(r"/Players/\d+/(\d+)/", href or "")
    return m.group(1) if m else None


def _team_id(href: str) -> Optional[str]:
    m = re.search(r"/Teams/\d+/(\d+)/", href or "")
    return m.group(1) if m else None


def _ground_id(href: str) -> Optional[str]:
    m = re.search(r"/Grounds/\d+/(\d+)\.html", href or "")
    return m.group(1) if m else None


def _scorecard_id(url: str) -> Optional[str]:
    m = re.search(r"/Scorecards/\d+/(\d+)\.html", url or "")
    return m.group(1) if m else None


def _int(s) -> Optional[int]:
    if s is None:
        return None
    s = str(s).strip()
    return int(s) if re.fullmatch(r"-?\d+", s) else None


def _float(s) -> Optional[float]:
    if s is None:
        return None
    s = str(s).strip()
    return float(s) if re.fullmatch(r"-?\d+(\.\d+)?", s) else None


def _clean_name(name: str) -> str:
    """Strip CricketArchive captain/keeper markers (*, +) from a batter name."""
    return name.lstrip("*+").strip()


def _parse_date(text: str) -> Tuple[Optional[str], Optional[str]]:
    """'... on 28th June 2026 (20-over match)' -> ('2026-06-28', raw)."""
    if not text:
        return None, None
    m = re.search(r"(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\s+(\d{4})", text)
    if not m:
        return None, text
    day, mon, year = int(m.group(1)), m.group(2), int(m.group(3))
    if mon not in _MONTHS:
        return None, text
    return f"{year:04d}-{_MONTHS[mon]:02d}-{day:02d}", text


def _row_cells(tr) -> List[str]:
    return [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]


# ---------------------------------------------------------------------------
# Event / series page
# ---------------------------------------------------------------------------
def parse_event_matches(html: str, url: str) -> List[CAEventMatch]:
    soup = BeautifulSoup(html, "lxml")
    rx = re.compile(r"/Archive/Scorecards/\d+/\d+\.html$")
    out, seen = [], set()
    for a in soup.find_all("a"):
        href = a.get("href") or ""
        if not rx.search(href):
            continue
        full = urljoin(BASE, href)
        sid = _scorecard_id(full)
        if sid in seen:
            continue
        seen.add(sid)
        out.append(CAEventMatch(
            scorecard_id=sid, url=full,
            teams_text=a.get_text(" ", strip=True) or None,
        ))
    return out


# ---------------------------------------------------------------------------
# Player profile
# ---------------------------------------------------------------------------
def parse_player_bio(html: str, url: str) -> CAPlayer:
    soup = BeautifulSoup(html, "lxml")
    ca_id = _player_id(url) or ""
    player = CAPlayer(ca_id=ca_id, url=url)

    heading = soup.find(["h1", "h2"])
    if heading:
        player.display_name = heading.get_text(" ", strip=True) or None

    table = soup.find("table")
    if table:
        for tr in table.find_all("tr"):
            cells = _row_cells(tr)
            if len(cells) < 2 or not cells[0].endswith(":"):
                continue
            label, value = cells[0][:-1].strip(), cells[1].strip()
            player.raw_fields[label] = value
            low = label.lower()
            if low == "full name":
                player.full_name = value
            elif low == "born":
                d, raw = _parse_date(value)
                player.born_date, player.born_raw = d, raw
                place = value.split(",", 1)[1].strip() if "," in value else None
                player.born_place = place
            elif low == "died":
                player.died_date, _ = _parse_date(value)
            elif low == "batting":
                player.batting_style = value
            elif low == "bowling":
                player.bowling_styles = [s.strip() for s in value.split(",") if s.strip()]
            elif low == "teams":
                player.teams = [t.strip() for t in value.split(";") if t.strip()]
    return player


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------
def _is_batting_header(cells: List[str]) -> bool:
    return (len(cells) >= 2 and cells[0].endswith("innings")
            and "Runs" in cells and "Balls" in cells)


def _is_bowling_header(cells: List[str]) -> bool:
    return "Overs" in cells and "Mdns" in cells and "Wkts" in cells


def _record_links(tr, players_seen: dict) -> List[str]:
    """Return player CA ids for links in a row; also populate players_seen."""
    ids = []
    for a in tr.find_all("a"):
        href = a.get("href") or ""
        pid = _player_id(href)
        if pid:
            ids.append(pid)
            players_seen.setdefault(pid, _clean_name(a.get_text(" ", strip=True)))
    return ids


def _parse_batting_table(table, players_seen: dict) -> List[CABattingEntry]:
    rows = table.find_all("tr")
    entries: List[CABattingEntry] = []
    for tr in rows[1:]:
        cells = _row_cells(tr)
        if not cells or not cells[0]:
            continue
        head = cells[0]
        if head in ("Extras", "Total") or head.startswith("Fall of wickets"):
            continue
        link_ids = _record_links(tr, players_seen)
        e = CABattingEntry(name=_clean_name(head))
        e.ca_id = link_ids[0] if link_ids else None
        e.dismissal = cells[1] if len(cells) > 1 else None
        # dismissal grammar -> fielder / bowler ids (link_ids[0] is the batter)
        dis = (e.dismissal or "")
        rest = link_ids[1:]
        if dis.startswith("c ") and " b " in dis and len(rest) >= 2:
            e.fielder_ca_id, e.bowler_ca_id = rest[0], rest[1]
        elif "run out" in dis and rest:
            e.fielder_ca_id = rest[0]
        elif rest:  # b / lbw b / st _ b -> bowler is last link
            e.bowler_ca_id = rest[-1]
        # numeric stats: name, dismissal, runs, balls, mins, 4s, 6s, dots, sr
        if len(cells) >= 9 and cells[1] not in ("did not bat",):
            e.runs = _int(cells[2]); e.balls = _int(cells[3]); e.mins = _int(cells[4])
            e.fours = _int(cells[5]); e.sixes = _int(cells[6]); e.dots = _int(cells[7])
            e.strike_rate = _float(cells[8])
        entries.append(e)
    return entries


def _parse_bowling_table(table, players_seen: dict) -> List[CABowlingEntry]:
    rows = table.find_all("tr")
    entries: List[CABowlingEntry] = []
    for tr in rows[1:]:
        cells = _row_cells(tr)
        if not cells or not cells[0]:
            continue
        link_ids = _record_links(tr, players_seen)
        e = CABowlingEntry(name=_clean_name(cells[0]))
        e.ca_id = link_ids[0] if link_ids else None
        # name, overs, mdns, runs, wkts, wides, noballs, dots, 4s, 6s, sr, econ
        c = cells + [None] * (12 - len(cells))
        e.overs = (c[1] or "").strip() or None
        e.maidens = _int(c[2]); e.runs = _int(c[3]); e.wickets = _int(c[4])
        e.wides = _int(c[5]); e.noballs = _int(c[6]); e.dots = _int(c[7])
        e.fours = _int(c[8]); e.sixes = _int(c[9]); e.econ = _float(c[11])
        entries.append(e)
    return entries


def _extract_fow(table) -> Optional[str]:
    txt = table.get_text("\n", strip=True)
    m = re.search(r"Fall of wickets:\s*(.+)", txt, re.S)
    if not m:
        return None
    return re.sub(r"\s+", " ", m.group(1)).strip()


def parse_scorecard(html: str, url: str) -> CAScorecard:
    soup = BeautifulSoup(html, "lxml")
    sc = CAScorecard(ca_id=_scorecard_id(url) or "", url=url)

    tables = soup.find_all("table")
    if not tables:
        return sc

    # --- meta table (first) ---
    meta = tables[0]
    for tr in meta.find_all("tr"):
        cells = _row_cells(tr)
        links = tr.find_all("a")
        if not cells:
            continue
        # title row: ["tt16997 itt4001", "Ireland v India"] with team links
        if len(cells) >= 2 and " v " in cells[1] and any(_team_id(a.get("href") or "") for a in links):
            sc.alt_ids = cells[0].split()
            sc.title = cells[1]
            for a in links:
                tid = _team_id(a.get("href") or "")
                if tid:
                    sc.teams.append((a.get_text(" ", strip=True), tid))
            continue
        label = cells[0].strip()
        value = cells[1].strip() if len(cells) > 1 else ""
        if not label and links:  # competition row
            sc.competition = value or (links[0].get_text(" ", strip=True))
            sc.competition_url = urljoin(BASE, links[0].get("href") or "")
            m = re.search(r"\(([^)]+)\)\s*$", value)
            if m:
                sc.match_label = m.group(1)
        elif label == "Venue":
            if links:
                sc.ground = links[0].get_text(" ", strip=True)
                sc.ground_ca_id = _ground_id(links[0].get("href") or "")
            sc.match_date, sc.match_date_raw = _parse_date(value)
            mo = re.search(r"\((\d+)-over", value)
            if mo:
                sc.overs_per_innings = int(mo.group(1))
        elif label == "Balls per over":
            sc.balls_per_over = _int(value)
        elif label == "Toss":
            sc.toss = value
        elif label == "Result":
            sc.result = value
        elif label in ("Umpires", "TV umpire", "Referee", "Scorers", "Reserve Umpire"):
            sc.officials[label] = [
                (a.get_text(" ", strip=True), _player_id(a.get("href") or "")) for a in links
            ]
        elif label == "Player of the Match":
            sc.player_of_match = value
            if links:
                sc.player_of_match_ca_id = _player_id(links[0].get("href") or "")
        elif label == "Ball-by-ball":
            sc.commentary_urls = [urljoin(BASE, a.get("href") or "") for a in links]

    # record officials + potm into players_seen
    for role_list in sc.officials.values():
        for nm, pid in role_list:
            if pid:
                sc.players_seen.setdefault(pid, nm)
    if sc.player_of_match_ca_id:
        sc.players_seen.setdefault(sc.player_of_match_ca_id, sc.player_of_match)

    # --- batting/bowling tables -> innings (batting followed by bowling) ---
    current: Optional[CAInnings] = None
    for t in tables[1:]:
        first = t.find("tr")
        if not first:
            continue
        head = _row_cells(first)
        if _is_batting_header(head):
            current = CAInnings(batting_team=head[0].replace(" innings", "").strip())
            current.batting = _parse_batting_table(t, sc.players_seen)
            current.fall_of_wickets = _extract_fow(t)
            # extras / total rows live in the batting table
            for tr in t.find_all("tr"):
                cells = _row_cells(tr)
                if cells and cells[0] == "Extras" and len(cells) > 1:
                    current.extras_text = cells[1]
                elif cells and cells[0] == "Total" and len(cells) > 1:
                    current.total_text = cells[1]
                    if len(cells) > 2:
                        current.total_runs = _int(cells[2])
                    mw = re.search(r"(\d+)\s+wickets?", cells[1])
                    if mw:
                        current.total_wickets = int(mw.group(1))
                    elif "all out" in cells[1].lower():
                        current.total_wickets = 10
            sc.innings.append(current)
        elif _is_bowling_header(head) and current is not None:
            current.bowling = _parse_bowling_table(t, sc.players_seen)

    return sc


# ---------------------------------------------------------------------------
# Ball-by-ball commentary -> CADelivery (our deliveries shape)
# ---------------------------------------------------------------------------
_BALL_RE = re.compile(r"^b\d+$")


def _parse_runs_token(tok: str) -> dict:
    """Parse a CA commentary runs cell into an extras-aware run dict.

    Handles simple tokens ('4', '1lb', '2w', '1nb', '1b') and compound ones
    separated by ';' / whitespace, e.g. '1; 1nb' (1 off the bat on a no-ball) or
    '4; 1w'. Bare suffixes ('nb', 'w') default to 1.
    """
    d = dict(runs_batter=0, runs_extras=0, runs_total=0, extras_wides=0,
             extras_noballs=0, extras_byes=0, extras_legbyes=0,
             is_boundary_four=False, is_boundary_six=False)
    for part in re.split(r"[;\s]+", tok.strip()):
        if not part:
            continue
        m = re.match(r"(\d+)?(lb|nb|b|w)?$", part)
        if not m or (m.group(1) is None and m.group(2) is None):
            continue
        g1, suf = m.group(1), m.group(2)
        if suf is None:                       # plain runs off the bat
            n = int(g1)
            d["runs_batter"] += n
            if n == 4:
                d["is_boundary_four"] = True
            elif n == 6:
                d["is_boundary_six"] = True
        else:
            n = int(g1) if g1 else 1
            key = {"lb": "extras_legbyes", "b": "extras_byes",
                   "w": "extras_wides", "nb": "extras_noballs"}[suf]
            d[key] += n
    d["runs_extras"] = (d["extras_wides"] + d["extras_noballs"]
                        + d["extras_byes"] + d["extras_legbyes"])
    d["runs_total"] = d["runs_batter"] + d["runs_extras"]
    return d


_DISMISS_RE = re.compile(
    r"^(?P<who>.+?)\s+(?P<how>c & b|c|lbw b|st .*? b|run out|b|hit wicket|stumped)\b"
)


def _parse_dismissal(text: str) -> Tuple[Optional[str], Optional[str]]:
    """'Calitz c Tilak Varma b Dubey 37 (...)' -> ('Calitz', 'caught')."""
    m = _DISMISS_RE.match(text.strip())
    if not m:
        return None, None
    who = m.group("who").strip()
    how = m.group("how")
    kind = {
        "c": "caught", "c & b": "caught and bowled", "lbw b": "lbw",
        "b": "bowled", "run out": "run out", "hit wicket": "hit wicket",
        "stumped": "stumped",
    }.get(how, "stumped" if how.startswith("st ") else how)
    return who, kind


def parse_commentary(html: str, url: str,
                     batting_order: Optional[List[str]] = None) -> List[CADelivery]:
    """Parse a commentary innings page into ordered CADelivery rows.

    ``batting_order`` (surnames as they appear in commentary, in batting order)
    lets us reconstruct the non-striker via a simple at-crease pair tracker.
    """
    soup = BeautifulSoup(html, "lxml")
    deliveries: List[CADelivery] = []

    # at-crease pair state for non-striker reconstruction
    order = list(batting_order or [])
    next_idx = 0
    pair: List[str] = []
    if order:
        pair = [order[0], order[1]] if len(order) > 1 else [order[0]]
        next_idx = len(pair)

    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            cells = _row_cells(tr)
            if len(cells) < 4:
                continue
            over_tok, ball_tok, runs_tok, desc = cells[0], cells[1], cells[2], cells[3]

            # delivery row
            if over_tok.isdigit() and _BALL_RE.match(ball_tok) and " to " in desc:
                bowler, batter = [x.strip() for x in desc.split(" to ", 1)]
                rd = _parse_runs_token(runs_tok)
                dlv = CADelivery(
                    over_number=int(over_tok),
                    ball_number=int(ball_tok[1:]),
                    batter=batter, bowler=bowler,
                    commentary_raw=f"{runs_tok} {desc}",
                    **rd,
                )
                # non-striker = the other batter currently at crease
                if pair:
                    if batter in pair:
                        others = [p for p in pair if p != batter]
                        dlv.non_striker = others[0] if others else None
                    else:
                        # order/desync: best-effort, keep partner if any
                        dlv.non_striker = pair[0] if pair and pair[0] != batter else None
                deliveries.append(dlv)
                continue

            # dismissal row: first cells empty, last cell carries the text
            if (not over_tok and not ball_tok and desc and deliveries
                    and (" b " in f" {desc} " or "run out" in desc
                         or desc.split()[0:1] and (" c " in f" {desc} " or "lbw" in desc
                                                   or "st " in desc))):
                who, kind = _parse_dismissal(desc)
                last = deliveries[-1]
                last.is_wicket = True
                last.wicket_type = kind
                last.dismissed_player = who
                # update at-crease pair
                if who and who in pair and order:
                    pair = [p for p in pair if p != who]
                    if next_idx < len(order):
                        pair.append(order[next_idx]); next_idx += 1

    return deliveries
