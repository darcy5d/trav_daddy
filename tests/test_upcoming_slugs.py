"""Slug parsing for Polymarket upcoming cricket fixtures."""

from src.integrations.polymarket.upcoming import _classify_slug


def test_classify_crint_digit_team_codes():
    info = _classify_slug("crint-wst2-lka2-2026-06-03")
    assert info is not None
    assert info["prefix"] == "crint"
    assert info["t1"] == "wst2"
    assert info["t2"] == "lka2"
    assert info["kind"] == "moneyline"


def test_classify_crint_womens_digit_codes():
    info = _classify_slug("crint-bgd3-nld4-2026-05-31")
    assert info is not None
    assert info["t1"] == "bgd3"
    assert info["t2"] == "nld4"


def test_classify_crint_submarket_more_markets():
    info = _classify_slug("crint-wst2-lka2-2026-06-03-more-markets")
    assert info is not None
    assert info["kind"] == "other"
    assert info["t1"] == "wst2"


def test_classify_legacy_three_letter_still_works():
    info = _classify_slug("cricipl-luc-kkr-2026-04-26")
    assert info is not None
    assert info["kind"] == "moneyline"
