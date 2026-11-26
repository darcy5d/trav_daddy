"""Pin the CricketArchive country codes used by scripts/ca_topup.py.

The 7-8 Nov 2026 top-up discovered zero scorecards for 76 of 107 countries
because `ALL_COUNTRIES` held IOC/ISO codes. CricketArchive does not use those.
The site publishes its own slug for each country in every season index page:

    Archive/Seasons/2026.html
      -> Archive/Seasons/2026_BDESH.html          (one page per country)
      -> Archive/Seasons/Seasonal_Averages/BDESH/2026_tt_Match_List.html

so the slug in that link is the code the top-up must use. Each mapping below was
confirmed by fetching `Seasonal_Averages/<CODE>/<token>_tt_Match_List.html` and,
where the code existed but had no T20 list, by reading the country season page to
find which token it actually publishes.

These tests are offline: they assert the shape of the list, not the site.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.ca_topup import (  # noqa: E402
    ALL_COUNTRIES,
    UNVERIFIED_COUNTRIES,
    VERIFIED_COUNTRIES,
    _match_list_url,
)

# old code -> the code CricketArchive actually serves
CORRECTIONS = {
    "IRE": "IRELAND",
    "NED": "NL",
    "SCO": "SCOT",
    "HK": "HKG",
    "OMA": "OMAN",
    "NAM": "NAMIB",
    "KEN": "KENYA",
    "JER": "JERSEY",
    "GUE": "GUERN",
    "BER": "BMDA",
    "ITA": "ITALY",
    "GER": "GERMANY",
    "DEN": "DNMRK",
    "SGP": "SPORE",
    "MAS": "MAL",
    "THA": "THAI",
    "QAT": "QTR",
    "BHR": "BAH",      # Bahrain is BAH on the archive; BHM is Bahamas
    "KOR": "SKOREA",
    "JPN": "JAPAN",
    "UGA": "UGAN",
    "TAN": "TZN",
    "NIG": "NGA",
    "RWA": "RWN",
    "ZAM": "ZAMBIA",
    "SWZ": "SWAZ",
    "FIJ": "FIJI",
    "BRA": "BRAZIL",
    "PAN": "PNM",
    "MEX": "MEXICO",
    "SUR": "SURINAM",
    "CRC": "CRICA",
    "MLD": "MDV",
    "INS": "INA",
    "AUT": "AUSTRIA",
    "FRA": "FRANCE",
    "SPN": "SPAIN",
    "BEL": "BELGIUM",
    "GIB": "GIBR",
    "CZE": "CZK",
    "POR": "PORT",
    "GRC": "GREECE",
    "MLT": "MALTA",
    "CRO": "CRT",
    "BUL": "BULG",
    "SRB": "SER",
    "KUW": "KWT",      # Kuwait was listed twice under two spellings (KWT and KUW)
}

# Real cricketing countries for which no T20 match list could be found. They are
# deliberately kept so the coverage gap stays visible rather than disappearing.
# Note SVK is Slovakia and SLOV is Slovenia: two different countries, so SLOV is
# added alongside SVK rather than replacing it.
UNRESOLVED_KEPT = {
    "KSA", "SLE", "CMR", "ETH", "LES", "MAU", "MOZ", "CIV", "SEN",
    "BLZ", "TCI", "VIN", "LCA", "GRN", "ANT", "DOM",
    "MYA", "SAM", "VAN", "COK", "TON", "LAT", "LTU", "POL", "SVK", "TUR",
}

# Re-checked 8 Nov 2026: 404 on every Seasonal_Averages token for 2025-2027, both
# genders, and no Archive/Seasons/<year>_<SLUG>.html either. Their old codes
# (ZAM, SUR, MLD, KUW, FRA) are dead too. Their matches still reach the archive
# through opponents' lists, so moving them here loses no coverage.
DEAD_SLUGS = {"ZAMBIA", "SURINAM", "MDV", "FRANCE", "KWT", "PHI"}

# Counts are pinned so a silent drop or a "tidy up" delete fails loudly. Adding a
# country is expected to bump one of these numbers deliberately.
EXPECTED_VERIFIED = 74
EXPECTED_UNVERIFIED = 32


def test_no_duplicates():
    duplicates = sorted({c for c in ALL_COUNTRIES if ALL_COUNTRIES.count(c) > 1})
    assert duplicates == [], f"duplicate country codes: {duplicates}"


def test_each_block_is_internally_duplicate_free():
    for name, block in (("VERIFIED", VERIFIED_COUNTRIES),
                        ("UNVERIFIED", UNVERIFIED_COUNTRIES)):
        dupes = sorted({c for c in block if block.count(c) > 1})
        assert dupes == [], f"duplicate codes in {name}: {dupes}"


def test_blocks_do_not_overlap():
    both = sorted(set(VERIFIED_COUNTRIES) & set(UNVERIFIED_COUNTRIES))
    assert both == [], f"codes in both blocks: {both}"


def test_all_countries_is_the_two_blocks_and_nothing_else():
    assert ALL_COUNTRIES == VERIFIED_COUNTRIES + UNVERIFIED_COUNTRIES


def test_block_sizes_are_pinned():
    # A code vanishing is the failure mode that hid for months; a count makes it
    # a test failure instead of a quieter discovery run.
    assert len(VERIFIED_COUNTRIES) == EXPECTED_VERIFIED
    assert len(UNVERIFIED_COUNTRIES) == EXPECTED_UNVERIFIED
    assert len(ALL_COUNTRIES) == EXPECTED_VERIFIED + EXPECTED_UNVERIFIED


def test_dead_slugs_live_in_the_unverified_block():
    assert DEAD_SLUGS <= set(UNVERIFIED_COUNTRIES)
    still_verified = sorted(DEAD_SLUGS & set(VERIFIED_COUNTRIES))
    assert still_verified == [], f"dead slugs still probed as verified: {still_verified}"


def test_dead_slugs_are_kept_not_deleted():
    # The whole point of moving them is that the gap stays visible.
    for code in DEAD_SLUGS:
        assert code in ALL_COUNTRIES, f"{code} deleted instead of moved"


def test_no_empty_or_malformed_codes():
    bad = [c for c in ALL_COUNTRIES if not c or c != c.strip() or not c.isalnum()]
    assert bad == [], f"bad country codes: {bad}"


def test_all_codes_are_uppercase():
    bad = [c for c in ALL_COUNTRIES if c != c.upper()]
    assert bad == [], f"lowercase country codes: {bad}"


def test_bangladesh_is_bdesh_not_ban():
    assert "BDESH" in ALL_COUNTRIES
    assert "BAN" not in ALL_COUNTRIES


@pytest.mark.parametrize("old,new", sorted(CORRECTIONS.items()))
def test_corrected_code_is_present_and_old_code_is_gone(old, new):
    assert new in ALL_COUNTRIES, f"{new} (was {old}) missing from ALL_COUNTRIES"
    if old != new:
        assert old not in ALL_COUNTRIES, f"stale {old} still in ALL_COUNTRIES"


def test_full_members_use_archive_spelling():
    for code in ("IND", "AUS", "ENG", "RSA", "PAK", "NZ", "SL", "WI", "BDESH",
                 "ZIM", "IRELAND", "AFG"):
        assert code in ALL_COUNTRIES, code


def test_bahamas_is_not_confused_with_bahrain():
    # BAH is Bahrain on the archive, BHM is Bahamas. Swapping them silently
    # probes the wrong country, so pin both.
    assert "BAH" in ALL_COUNTRIES
    assert "BAH" in VERIFIED_COUNTRIES
    assert "BHM" not in ALL_COUNTRIES


def test_unresolved_countries_are_kept():
    missing = sorted(UNRESOLVED_KEPT - set(ALL_COUNTRIES))
    assert missing == [], f"dropped without resolving: {missing}"


def test_match_list_url_shape():
    url = _match_list_url("BDESH", "2025-26", women=False)
    assert url == (
        "https://cricketarchive.com/Archive/Seasons/"
        "Seasonal_Averages/BDESH/2025-26_tt_Match_List.html"
    )
    assert _match_list_url("IND", "2026", women=True).endswith(
        "/IND/2026_wtt_Match_List.html"
    )
