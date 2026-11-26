"""CREX women's short codes map onto the fixture's own teams.

2026-11-14 08:30Z: the auto daemon saw toss winner 'ADS-W' for Perth Scorchers
vs Adelaide Strikers and returned None, so the post-toss scan never started.
"""

import pytest

from src.api.crex_scraper import CREXScraper, CREX_TEAM_CODES
from scripts.paper_bet_auto_post_toss import _detect_toss_winner_side


# CREX franchise code -> the team name the daemon already stores.
WBBL_CODES = {
    "ADS-W": "Adelaide Strikers",
    "PRS-W": "Perth Scorchers",
    "PS-W": "Perth Scorchers",
    "SYS-W": "Sydney Sixers",
    "SS-W": "Sydney Sixers",
    "SYT-W": "Sydney Thunder",
    "ST-W": "Sydney Thunder",
    "MLS-W": "Melbourne Stars",
    "MLR-W": "Melbourne Renegades",
    "BRH-W": "Brisbane Heat",
    "HEAT-W": "Brisbane Heat",
    "HOB-W": "Hobart Hurricanes",
}


def _fixture(team1, team2):
    return {"team1_db_name": team1, "team2_db_name": team2}


class _Crex:
    def __init__(self, team1_name, team2_name):
        self.team1_name = team1_name
        self.team2_name = team2_name


def test_ads_w_maps_onto_adelaide_for_the_logged_fixture():
    """The toss line that blocked tonight's scan."""
    fixture = _fixture("Perth Scorchers", "Adelaide Strikers")
    assert _detect_toss_winner_side("ADS-W", fixture, None) == "team2"
    assert _detect_toss_winner_side("ads-w", fixture, None) == "team2"
    assert _detect_toss_winner_side("ADSW", fixture, None) == "team2"


def test_ads_w_follows_whichever_side_adelaide_is_on():
    fixture = _fixture("Adelaide Strikers", "Perth Scorchers")
    assert _detect_toss_winner_side("ADS-W", fixture, None) == "team1"


@pytest.mark.parametrize("code,team", sorted(WBBL_CODES.items()))
def test_wbbl_womens_code_maps_to_its_franchise(code, team):
    others = sorted({name for name in WBBL_CODES.values() if name != team})
    for other in others:
        assert _detect_toss_winner_side(code, _fixture(team, other), None) == "team1"
        assert _detect_toss_winner_side(code, _fixture(other, team), None) == "team2"
        women = _fixture(f"{other} Women", f"{team} Women")
        assert _detect_toss_winner_side(code, women, None) == "team2"


def test_melbourne_codes_do_not_share_a_side():
    fixture = _fixture("Melbourne Stars", "Melbourne Renegades")
    assert _detect_toss_winner_side("MLS-W", fixture, None) == "team1"
    assert _detect_toss_winner_side("MLR-W", fixture, None) == "team2"


def test_womens_code_uses_crex_name_when_the_db_name_is_only_the_nickname():
    fixture = _fixture("Scorchers", "Strikers")
    crex = _Crex("Perth Scorchers Women", "Adelaide Strikers Women")
    assert _detect_toss_winner_side("ADS-W", fixture, crex) == "team2"
    assert _detect_toss_winner_side("PRS-W", fixture, crex) == "team1"


def test_unknown_code_stays_unmapped_and_a_real_code_maps():
    fixture = _fixture("Perth Scorchers", "Adelaide Strikers")
    assert _detect_toss_winner_side("XYZ-W", fixture, None) is None
    assert _detect_toss_winner_side("ADS", fixture, None) == "team2"


# What _generate_team_abbreviations emits. None of these is the CREX toss code.
GENERATED = {
    "Uganda": {"U", "UG", "UGA"},
    "Rwanda": {"R", "RW"},
    "Biratnagar Kings": {"BI", "BIR", "BK"},
    "Lumbini Lions": {"LL", "LU", "LUM"},
}

NPL_CODES = {
    "BK": "Biratnagar Kings",
    "LBL": "Lumbini Lions",
    "CWR": "Chitwan Rhinos",
    "JKB": "Janakpur Bolts",
    "KMG": "Kathmandu Gurkhas",
    "KNY": "Karnali Yaks",
    "PKA": "Pokhara Avengers",
    "SPR": "Sudurpaschim Royals",
}


def _generated(name):
    return {abbr.upper() for abbr in CREXScraper()._generate_team_abbreviations(name, "") if abbr}


def test_daemon_does_not_generate_ugn_or_lbl():
    for name, expected in GENERATED.items():
        assert expected <= _generated(name)
    assert "UGN" not in _generated("Uganda")
    assert "UGN" not in _generated("Uganda Women")
    assert "RWA" not in _generated("Rwanda")
    assert "LBL" not in _generated("Lumbini Lions")


def test_ugn_w_maps_onto_uganda_not_rwanda():
    fixture = _fixture("Uganda", "Rwanda")
    assert _detect_toss_winner_side("UGN-W", fixture, None) == "team1"
    assert _detect_toss_winner_side("ugn-w", fixture, None) == "team1"
    women = _fixture("Rwanda Women", "Uganda Women")
    assert _detect_toss_winner_side("UGN-W", women, None) == "team2"


def test_lbl_maps_onto_lumbini_not_biratnagar():
    fixture = _fixture("Biratnagar Kings", "Lumbini Lions")
    assert _detect_toss_winner_side("LBL", fixture, None) == "team2"
    assert _detect_toss_winner_side("lbl", fixture, None) == "team2"
    swapped = _fixture("Lumbini Lions", "Biratnagar Kings")
    assert _detect_toss_winner_side("LBL", swapped, None) == "team1"


@pytest.mark.parametrize("code,team", sorted(NPL_CODES.items()))
def test_npl_code_maps_to_its_franchise(code, team):
    others = sorted({name for name in NPL_CODES.values() if name != team})
    for other in others:
        assert _detect_toss_winner_side(code, _fixture(team, other), None) == "team1"
        assert _detect_toss_winner_side(code, _fixture(other, team), None) == "team2"


def test_kathmandu_gorkhas_spelling_still_takes_kmg():
    fixture = _fixture("Karnali Yaks", "Kathmandu Gorkhas")
    assert _detect_toss_winner_side("KMG", fixture, None) == "team2"


def test_table_codes_the_generator_omits_still_map():
    """RWA, TAN, and the rest of CREX_TEAM_CODES fail the same way as UGN and LBL."""
    for code, name in CREX_TEAM_CODES.items():
        if code.upper() in _generated(name):
            continue
        displayed = name.title()
        assert _detect_toss_winner_side(code.upper(), _fixture(displayed, "Decoy Athletic"), None) == "team1"
        assert _detect_toss_winner_side(
            code.upper() + "-W", _fixture("Decoy Athletic", displayed + " Women"), None,
        ) == "team2"


def test_full_name_and_ipl_alias_still_map():
    wbbl = _fixture("Perth Scorchers", "Adelaide Strikers")
    assert _detect_toss_winner_side("Adelaide Strikers", wbbl, None) == "team2"
    ipl = _fixture("Punjab Kings", "Mumbai Indians")
    assert _detect_toss_winner_side("PBKS", ipl, None) == "team1"
    assert _detect_toss_winner_side("MI", ipl, None) == "team2"
