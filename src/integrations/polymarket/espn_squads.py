"""ESPN match squads, snapshotted before each match, for the pre-toss XI selector.

ESPN's match summary (same unauthenticated site API as the scorepanel used for
venues) lists each side's announced squad until the match starts. Afterwards
the `squads` block disappears and `rosters` holds the played XI, so squads
cannot be recovered for past matches: they must be captured beforehand.
`scripts/snapshot_espn_squads.py` does that on a schedule and later records
the played XI, which `scripts/eval_xi_squads.py` uses to test the squad-aware
selector out of sample.

    https://site.api.espn.com/apis/site/v2/sports/cricket/{league_id}/summary?event={espn_id}

ESPN athlete ids are Cricinfo ids; Cricsheet's people register maps them to
`players.registry_id`. Snapshots live as one JSON file per event under
`data/espn_squads/`, outside cricket.db.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[3]
SNAPSHOT_DIR = REPO / "data" / "espn_squads"
REGISTER_PATH = REPO / "data" / "external" / "cricsheet_people.csv"
REGISTER_URL = "https://cricsheet.org/register/people.csv"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/cricket/{lid}/summary?event={eid}"
UA = {"User-Agent": "Mozilla/5.0 (squad lookup)"}
MIN_SQUAD = 12
MIN_TEAM_OVERLAP = 5


def fetch_summary(league_id, espn_id, session=None, retries: int = 2) -> Optional[dict]:
    import requests

    s = session or requests.Session()
    for attempt in range(retries):
        try:
            r = s.get(SUMMARY_URL.format(lid=league_id, eid=espn_id), headers=UA, timeout=20)
            if r.status_code == 200:
                return r.json()
        except Exception as exc:
            logger.debug(f"summary {espn_id} attempt {attempt}: {exc}")
        time.sleep(1.5 * 2 ** attempt)
    return None


def _athlete_id(entry: dict) -> Optional[str]:
    a = entry.get("athlete", entry)
    return str(a["id"]) if a.get("id") is not None else None


def parse_squads(summary: dict) -> Dict[str, List[str]]:
    """{ESPN team name: [athlete ids]} for sides with an announced squad."""
    out = {}
    for t in summary.get("squads") or []:
        ids = [i for i in (_athlete_id(a) for a in t.get("athletes") or []) if i]
        if ids:
            out[(t.get("team") or {}).get("displayName", "")] = ids
    return out


def parse_played_xi(summary: dict) -> Dict[str, List[str]]:
    """{ESPN team name: [athlete ids]} for sides whose roster is exactly 11 starters."""
    out = {}
    for r in summary.get("rosters") or []:
        ro = r.get("roster") or []
        ids = [i for i in (_athlete_id(p) for p in ro if p.get("starter")) if i]
        if len(ro) == 11 and len(ids) == 11:
            out[(r.get("team") or {}).get("displayName", "")] = ids
    return out


def ball_bowled(summary: dict) -> bool:
    """True once a competitor linescore shows overs bowled.

    A Live status with no linescore is not a ball. Player linescores and the
    squad list are not the board.
    """
    if not isinstance(summary, dict):
        return False
    for comp in (summary.get("header") or {}).get("competitions") or []:
        for side in comp.get("competitors") or []:
            for ls in side.get("linescores") or []:
                try:
                    if ls.get("overs") is not None and float(ls["overs"]) > 0:
                        return True
                except (TypeError, ValueError):
                    continue
    return False


def confirmed_rosters(summary: dict) -> Optional[Dict[str, List[str]]]:
    """{team name: [athlete ids]} when both rosters are an XI and no ball is bowled.

    A side counts only when its `roster` list has length exactly 11. Fifteen
    and eighteen are the squad, including when every player is `starter`.
    Squads, the toss note and a Live flag are ignored. None unless there are
    exactly two such sides. A linescore with overs bowled closes the window
    even if the lists are now 11.
    """
    if not isinstance(summary, dict) or ball_bowled(summary):
        return None
    rows = summary.get("rosters") or []
    if len(rows) != 2:
        return None
    out: Dict[str, List[str]] = {}
    for r in rows:
        roster = r.get("roster") or []
        if len(roster) != 11:
            return None
        ids = [i for i in (_athlete_id(p) for p in roster) if i]
        name = ((r.get("team") or {}).get("displayName")) or ""
        if len(ids) != 11 or len(set(ids)) != 11 or not name or name in out:
            return None
        out[name] = ids
    return out if len(out) == 2 else None


# ---------------------------------------------------------------------------
# Snapshot store
# ---------------------------------------------------------------------------

def _path(espn_id, root: Path) -> Path:
    return Path(root) / f"{espn_id}.json"


def load_event(espn_id, root: Path = SNAPSHOT_DIR) -> Optional[dict]:
    try:
        return json.loads(_path(espn_id, root).read_text())
    except (OSError, ValueError):
        return None


def save_event(rec: dict, root: Path = SNAPSHOT_DIR) -> None:
    p = _path(rec["espn_id"], root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    os.replace(tmp, p)


def iter_events(root: Path = SNAPSHOT_DIR) -> Iterable[dict]:
    for p in sorted(Path(root).glob("*.json")):
        try:
            yield json.loads(p.read_text())
        except ValueError:
            continue


def pre_start_squads(rec: dict) -> Dict[str, List[str]]:
    """Latest squad per side fetched before the scheduled start."""
    out: Dict[str, List[str]] = {}
    start = rec.get("start") or ""
    for snap in sorted(rec.get("snapshots", []), key=lambda s: s["fetched_at"]):
        if start and snap["fetched_at"] >= start:
            continue
        out.update(snap["squads"])
    return out


# ---------------------------------------------------------------------------
# Cricinfo id -> players.player_id
# ---------------------------------------------------------------------------

def refresh_register(path: Path = REGISTER_PATH, max_age_days: float = 7.0) -> Path:
    try:
        if time.time() - path.stat().st_mtime < max_age_days * 86400:
            return path
    except OSError:
        pass
    import requests

    r = requests.get(REGISTER_URL, headers=UA, timeout=60)
    r.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(r.content)
    os.replace(tmp, path)
    return path


_ID_MAP: Dict[str, tuple] = {}


def cricinfo_player_map(db_path, register: Path = REGISTER_PATH) -> Dict[str, int]:
    """{Cricinfo/ESPN athlete id: players.player_id}, cached on the register's mtime."""
    import pandas as pd

    try:
        mt = register.stat().st_mtime
    except OSError:
        return {}
    key = f"{db_path}:{register}"
    hit = _ID_MAP.get(key)
    if hit and hit[0] == mt:
        return hit[1]
    reg = pd.read_csv(register, dtype=str)
    c = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    pid = dict(c.execute("SELECT registry_id, player_id FROM players WHERE registry_id IS NOT NULL"))
    c.close()
    out: Dict[str, int] = {}
    for col in ("key_cricinfo", "key_cricinfo_2", "key_cricinfo_3"):
        if col not in reg:
            continue
        for ci, ident in zip(reg[col], reg["identifier"]):
            if isinstance(ci, str) and ident in pid:
                out.setdefault(ci.split(".")[0], int(pid[ident]))
    _ID_MAP[key] = (mt, out)
    return out


def map_ids(ids: Iterable[str], id_map: Dict[str, int]) -> List[int]:
    return [id_map[i] for i in ids if i in id_map]


def squad_for_team(known_players: Set[int], gender: str, fmt: str, day,
                   id_map: Dict[str, int], root: Path = SNAPSHOT_DIR,
                   max_hours: float = 30.0) -> Optional[List[int]]:
    """Mapped pre-start squad for the side whose recent players are `known_players`.

    The side is identified by player overlap rather than team name: among
    snapshotted events of this gender/format starting within `max_hours` of
    `day`, the squad sharing the most players with `known_players` (at least
    MIN_TEAM_OVERLAP) wins. None when nothing qualifies.
    """
    from src.integrations.polymarket.espn_fixture_venues import parse_utc

    anchor = day if isinstance(day, datetime) else datetime(day.year, day.month, day.day, 12)
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=timezone.utc)
    best, best_n = None, MIN_TEAM_OVERLAP - 1
    for rec in iter_events(root):
        if rec.get("gender") != gender or rec.get("fmt") != fmt:
            continue
        st = parse_utc(rec.get("start"))
        if st is None or abs((st - anchor).total_seconds()) / 3600 > max_hours:
            continue
        for ids in pre_start_squads(rec).values():
            sq = map_ids(ids, id_map)
            n = len(known_players & set(sq))
            if len(sq) >= MIN_SQUAD and n > best_n:
                best, best_n = sq, n
    return best
