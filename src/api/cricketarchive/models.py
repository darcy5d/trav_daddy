"""Dataclasses for CricketArchive entities.

Mirror the shape of CREXPlayer / ESPNPlayer in the sibling scrapers so the
downstream name-matching / backfill code feels familiar. Fields default to None
because CricketArchive pages are sparsely/variably populated across players.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CAPlayer:
    """A CricketArchive player profile.

    ``ca_id`` is the stable numeric id from the URL
    ``/Archive/Players/{folder}/{ca_id}/{ca_id}.html`` — this is the canonical
    key we want to attach to our local ``players`` rows.
    """

    ca_id: str
    url: str
    full_name: Optional[str] = None        # "Charles Henry Palmer"
    display_name: Optional[str] = None      # page heading, e.g. "Charles Palmer"
    born_date: Optional[str] = None         # ISO 'YYYY-MM-DD' when parseable, else raw
    born_raw: Optional[str] = None          # original "15th May 1919, Old Hill, ..."
    born_place: Optional[str] = None
    died_date: Optional[str] = None
    batting_style: Optional[str] = None     # "Right-hand batter"
    bowling_styles: List[str] = field(default_factory=list)
    playing_role: Optional[str] = None
    teams: List[str] = field(default_factory=list)
    # Every label/value row we found, verbatim, so we never silently drop data
    # and can mine new fields later without re-fetching.
    raw_fields: Dict[str, str] = field(default_factory=dict)
    # Resolved during backfill:
    db_player_id: Optional[int] = None


@dataclass
class CAEventMatch:
    """A match link harvested from an /Archive/Events/ series page."""

    scorecard_id: str               # canonical id from the URL, e.g. "1446777"
    url: str
    label: Optional[str] = None     # e.g. "2nd Twenty20"
    teams_text: Optional[str] = None
    match_date: Optional[str] = None


@dataclass
class CADelivery:
    """One ball, reconstructed from CA commentary -> our `deliveries` shape."""

    over_number: int                # 0-indexed over (as CA shows it)
    ball_number: int                # legal-ball ordinal within the over
    batter: str                     # striker (name as shown)
    bowler: str
    non_striker: Optional[str] = None   # reconstructed via strike rotation
    runs_batter: int = 0
    runs_extras: int = 0
    runs_total: int = 0
    extras_wides: int = 0
    extras_noballs: int = 0
    extras_byes: int = 0
    extras_legbyes: int = 0
    extras_penalty: int = 0
    is_wicket: bool = False
    wicket_type: Optional[str] = None
    dismissed_player: Optional[str] = None
    fielder1: Optional[str] = None
    fielder2: Optional[str] = None
    is_boundary_four: bool = False
    is_boundary_six: bool = False
    commentary_raw: Optional[str] = None


@dataclass
class CABattingEntry:
    name: str
    ca_id: Optional[str] = None
    dismissal: Optional[str] = None         # "c Sharma b Rana", "not out", "did not bat"
    bowler_ca_id: Optional[str] = None
    fielder_ca_id: Optional[str] = None
    runs: Optional[int] = None
    balls: Optional[int] = None
    mins: Optional[int] = None
    fours: Optional[int] = None
    sixes: Optional[int] = None
    dots: Optional[int] = None
    strike_rate: Optional[float] = None


@dataclass
class CABowlingEntry:
    name: str
    ca_id: Optional[str] = None
    overs: Optional[str] = None
    maidens: Optional[int] = None
    runs: Optional[int] = None
    wickets: Optional[int] = None
    wides: Optional[int] = None
    noballs: Optional[int] = None
    dots: Optional[int] = None
    fours: Optional[int] = None
    sixes: Optional[int] = None
    econ: Optional[float] = None


@dataclass
class CAInnings:
    batting_team: Optional[str] = None
    batting: List[CABattingEntry] = field(default_factory=list)
    bowling: List[CABowlingEntry] = field(default_factory=list)
    extras_text: Optional[str] = None       # "(2 lb, 1 nb, 3 w)"
    total_text: Optional[str] = None        # "(8 wickets, innings closed, ...)"
    total_runs: Optional[int] = None
    total_wickets: Optional[int] = None
    fall_of_wickets: Optional[str] = None


@dataclass
class CAScorecard:
    """A CricketArchive scorecard. CA DOES carry ball-by-ball (see commentary_urls)."""

    ca_id: str                              # canonical scorecard id from URL
    url: str
    title: Optional[str] = None             # "Ireland v India"
    alt_ids: List[str] = field(default_factory=list)   # e.g. ["tt16997","itt4001"]
    competition: Optional[str] = None
    competition_url: Optional[str] = None
    match_label: Optional[str] = None       # "2nd Twenty20"
    match_date: Optional[str] = None        # ISO if parseable
    match_date_raw: Optional[str] = None
    overs_per_innings: Optional[int] = None
    ground: Optional[str] = None
    ground_ca_id: Optional[str] = None
    balls_per_over: Optional[int] = None
    toss: Optional[str] = None
    result: Optional[str] = None
    teams: List[tuple] = field(default_factory=list)        # [(name, ca_team_id)]
    officials: Dict[str, list] = field(default_factory=dict)  # role -> [(name, ca_id)]
    player_of_match: Optional[str] = None
    player_of_match_ca_id: Optional[str] = None
    commentary_urls: List[str] = field(default_factory=list)
    innings: List[CAInnings] = field(default_factory=list)
    # Every player encountered on the card -> their CA id (priority B: ID map).
    players_seen: Dict[str, str] = field(default_factory=dict)  # ca_id -> name
    raw_meta: Dict[str, str] = field(default_factory=dict)
