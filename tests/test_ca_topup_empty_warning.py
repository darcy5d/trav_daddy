"""The end-of-run "no scorecards" warning must count countries, not probes.

`discover()` walks countries × years × genders, so one country can come back
empty four times (2025, 2026, men, women). The old warning appended to
`empty_countries` once per empty probe, so a single dead code printed as
"4 of 4 countries" even when every other country was fine — and a country with
men's cards but no women's list was reported as having no scorecards at all.

These tests drive `discover()` with a fake fetcher so they stay offline: the
fake answers with a scorecard link or an HTTP 404 based on a fixture map.
"""

import logging
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.ca_topup import _season_tokens, discover, uniq_in_order  # noqa: E402

BASE = "https://cricketarchive.com"


class FakeFetcher:
    """Answers match lists from a map of {(country, year, women): n_cards}.

    Anything absent from the map 404s, which is what a wrong country code does.
    `status_200_empty` names countries whose lists all load but hold no cards —
    an empty season, not a bad code.
    """

    status_200_empty: set[str] = set()

    def __init__(self, cards: dict[tuple[str, int, bool], int]):
        self.cards = cards
        self.requests: list[str] = []

    def get(self, url: str, force: bool = False, cache_only: bool = False) -> str:
        self.requests.append(url)
        country = url.split("/Seasonal_Averages/")[1].split("/")[0]
        token = url.rsplit("/", 1)[1].split("_")[0]
        women = "_wtt_" in url
        # A token belongs to a year if it is one of the tokens that year probes.
        years = [y for y in (2024, 2025, 2026) if token in _season_tokens(y)]
        for year in years:
            n = self.cards.get((country, year, women))
            if n:
                links = "".join(
                    f'<a href="/Archive/Scorecards/{1400 + i}/{900000 + i}.html">x</a>'
                    for i in range(n)
                )
                return f"<html>{links}</html>"
        if country in self.status_200_empty:
            return "<html><body>No matches in this season.</body></html>"
        resp = requests.Response()
        resp.status_code = 404
        resp.url = url
        raise requests.HTTPError(f"404 Client Error: Not Found for url: {url}",
                                 response=resp)


@pytest.fixture()
def warnings(caplog):
    caplog.set_level(logging.WARNING, logger="ca_topup")
    return caplog


def _warning_text(caplog) -> str:
    return "\n".join(r.getMessage() for r in caplog.records
                     if r.levelno >= logging.WARNING)


def test_empty_country_is_reported_once_not_once_per_probe(caplog):
    # ZAMBIA has nothing anywhere; IND has cards in every probe. Two years × two
    # genders would give ZAMBIA four entries under the old code.
    cards = {(c, y, w): 2 for c in ("IND",) for y in (2025, 2026) for w in (False, True)}
    discover(["ZAMBIA", "IND"], [2025, 2026], True, FakeFetcher(cards))

    text = _warning_text(caplog)
    assert "No scorecards for 1 of 2 countries" in text, text
    listed = text.split("countries: ", 1)[1].splitlines()[0]
    assert listed == "ZAMBIA", f"empty list should name ZAMBIA once, got: {listed}"
    assert "IND" not in listed, listed


def test_distinct_count_is_correct_with_several_empty_countries(caplog):
    cards = {(c, y, w): 1 for c in ("IND", "AUS") for y in (2025, 2026) for w in (False, True)}
    discover(["ZAMBIA", "IND", "KWT", "AUS"], [2025, 2026], True, FakeFetcher(cards))

    text = _warning_text(caplog)
    assert "No scorecards for 2 of 4 countries" in text, text
    assert "ZAMBIA" in text and "KWT" in text


def test_country_with_cards_in_one_gender_only_is_not_reported_empty(caplog):
    # Men's list has matches, women's list 404s. The country is incomplete, not
    # absent, and must not appear in the "no scorecards" warning.
    cards = {("NAMIB", 2025, False): 3, ("NAMIB", 2026, False): 3}
    discover(["NAMIB"], [2025, 2026], True, FakeFetcher(cards))

    text = _warning_text(caplog)
    assert "NAMIB" not in text, f"half-covered country reported as empty:\n{text}"
    assert "No scorecards for" not in text, text


def test_all_empty_is_reported_with_suspect_list(caplog):
    # Every probe 404s: that is the wrong-code signature and both warnings fire.
    discover(["ZAMBIA", "KWT"], [2025, 2026], True, FakeFetcher({}))

    text = _warning_text(caplog)
    assert "No scorecards for 2 of 2 countries" in text, text
    assert "returned HTTP 404 on every list" in text, text
    assert text.count("ZAMBIA") == 2, f"ZAMBIA should appear once per warning:\n{text}"


def test_cards_in_one_gender_are_still_collected(caplog):
    # The distinctness fix must not throw away URLs that were found.
    cards = {("NAMIB", 2025, False): 3}
    found = discover(["NAMIB"], [2025], True, FakeFetcher(cards))
    assert len(found) == 3, found


def test_suspect_list_needs_404_on_every_list(caplog):
    # A country whose lists load but are empty is not a wrong-code suspect, even
    # though some of its tokens 404. Only an all-404 country is.
    #   EMPTYOK: every token 200, zero cards -> empty, not suspect
    #   DEADCODE: every token 404           -> empty, suspect
    fetcher = FakeFetcher({})
    fetcher.status_200_empty = {"EMPTYOK"}
    discover(["EMPTYOK", "DEADCODE"], [2025], False, fetcher)

    text = _warning_text(caplog)
    assert "No scorecards for 2 of 2 countries" in text, text
    suspect_part = text.split("rather than an empty season: ", 1)[1]
    assert "DEADCODE" in suspect_part, text
    assert "EMPTYOK" not in suspect_part, f"empty season called a wrong code:\n{text}"


def test_uniq_in_order_preserves_first_seen_order():
    assert uniq_in_order(["B", "A", "B", "C", "A"]) == ["B", "A", "C"]
    assert uniq_in_order([]) == []


def test_every_token_is_still_probed_when_a_year_yields_early():
    # Guards the "keep going" behaviour: discovering cards on token 1 of a year
    # must not short-circuit the remaining tokens for that country/year.
    cards = {("IND", 2026, False): 2}
    fetcher = FakeFetcher(cards)
    discover(["IND"], [2026], False, fetcher)
    tokens_probed = {u.rsplit("/", 1)[1].split("_")[0] for u in fetcher.requests}
    assert set(_season_tokens(2026)) == tokens_probed, (tokens_probed, _season_tokens(2026))
