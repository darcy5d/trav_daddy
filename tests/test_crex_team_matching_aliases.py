"""Unit tests for CREX team matching aliases/rebrands."""

import unittest
from unittest.mock import patch

from src.api.crex_scraper import CREXScraper, CREXTeam


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, *_args, **_kwargs):
        return None

    def fetchall(self):
        return self._rows


class _FakeConnection:
    def __init__(self, rows):
        self._rows = rows

    def cursor(self):
        return _FakeCursor(self._rows)

    def close(self):
        return None


class TeamAliasMatchingTests(unittest.TestCase):
    @patch("src.api.crex_scraper.get_connection")
    def test_bengaluru_prefers_bangalore_alias_target(self, mock_get_connection):
        # Simulate DB containing both legacy and renamed forms.
        rows = [
            {"team_id": 283, "name": "Royal Challengers Bengaluru"},
            {"team_id": 111, "name": "Royal Challengers Bangalore"},
        ]
        mock_get_connection.return_value = _FakeConnection(rows)
        scraper = CREXScraper(request_delay=0.0)
        team = CREXTeam(
            crex_id="RCB",
            name="Royal Challengers Bengaluru",
            abbreviation="RCB",
        )

        matched = scraper.match_team_to_db(team, gender="male")
        self.assertIsNotNone(matched)
        self.assertEqual(matched[0], 111)
        self.assertEqual(matched[1], "Royal Challengers Bangalore")

    @patch("src.api.crex_scraper.get_connection")
    def test_truncated_royal_challengers_name_maps_to_bangalore(self, mock_get_connection):
        rows = [
            {"team_id": 111, "name": "Royal Challengers Bangalore"},
            {"team_id": 118, "name": "Rajasthan Royals"},
        ]
        mock_get_connection.return_value = _FakeConnection(rows)
        scraper = CREXScraper(request_delay=0.0)
        team = CREXTeam(
            crex_id="RCB",
            name="Royal Challengers",
            abbreviation="RCB",
        )

        matched = scraper.match_team_to_db(team, gender="male")
        self.assertIsNotNone(matched)
        self.assertEqual(matched[0], 111)
        self.assertEqual(matched[1], "Royal Challengers Bangalore")


if __name__ == "__main__":
    unittest.main()
