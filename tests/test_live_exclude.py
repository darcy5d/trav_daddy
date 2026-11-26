"""Unit tests for the live-only tournament-prefix exclusion guard
(scripts/live_bet_scan._live_excluded_prefix).

County T20 Blast is excluded from LIVE placement (paper-confirm window) but must
still flow to paper scanners, which never call this guard.
"""

from __future__ import annotations

from unittest.mock import patch

from scripts.live_bet_scan import _live_excluded_prefix


def _cfg(prefixes):
    return {"live_exclude_prefixes": prefixes}


def test_excluded_prefix_blocks():
    with patch("config.BETTING_CONFIG", _cfg(["crict20blast", "crict20blastw"])):
        assert _live_excluded_prefix({"tournament_prefix": "crict20blast"}) is True
        assert _live_excluded_prefix({"tournament_prefix": "crict20blastw"}) is True


def test_excluded_prefix_is_case_insensitive():
    with patch("config.BETTING_CONFIG", _cfg(["crict20blast"])):
        assert _live_excluded_prefix({"tournament_prefix": "CRICT20BLAST"}) is True


def test_non_excluded_prefix_allowed():
    with patch("config.BETTING_CONFIG", _cfg(["crict20blast", "crict20blastw"])):
        assert _live_excluded_prefix({"tournament_prefix": "crint"}) is False
        assert _live_excluded_prefix({"tournament_prefix": "cricipl"}) is False
        assert _live_excluded_prefix({"tournament_prefix": "cricmlc"}) is False


def test_empty_exclude_list_allows_everything():
    with patch("config.BETTING_CONFIG", _cfg([])):
        assert _live_excluded_prefix({"tournament_prefix": "crict20blast"}) is False


def test_missing_prefix_fails_closed():
    with patch("config.BETTING_CONFIG", _cfg(["crict20blast"])):
        assert _live_excluded_prefix({"tournament_prefix": None}) is True
        assert _live_excluded_prefix({}) is True


def test_unlisted_prefix_is_live_excluded():
    with patch("config.BETTING_CONFIG", _cfg([])):
        assert _live_excluded_prefix({"tournament_prefix": "cricbrandnewleague"}) is True
        assert _live_excluded_prefix({"tournament_prefix": "cricadt10"}) is True


def test_policy_live_false_is_excluded_even_if_not_in_list():
    policy = {"crint": {"live": True}, "cricwbbl": {"live": False}}
    with patch("config.BETTING_CONFIG", _cfg([])), \
            patch("config.BETTING_LEAGUE_POLICY", policy):
        assert _live_excluded_prefix({"tournament_prefix": "crint"}) is False
        assert _live_excluded_prefix({"tournament_prefix": "cricwbbl"}) is True
        assert _live_excluded_prefix({"tournament_prefix": "cricipl"}) is True
