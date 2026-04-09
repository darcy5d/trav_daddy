"""Unit tests for CREX schedule row parsing (CrickAPI JSON shape)."""
import unittest

from src.api.crex_scraper import CREXScraper


class CrickapiFixtureRowTests(unittest.TestCase):
    def setUp(self):
        self.scraper = CREXScraper(request_delay=0.0)

    def test_synthetic_link_and_nf_match_key(self):
        row = {
            't1f': 'M',
            't2f': 'KB',
            'sf': '1PW',
            'mn': '52',
            'nf': '118X',
            'ft': 2,
            'fo': 'T20',
            'date': '5/9/2026',
            't': 1778335200000,
            'g': 1,
            'status': 0,
        }
        m = self.scraper._fixture_row_to_schedule_match(row)
        self.assertIsNotNone(m)
        self.assertEqual(m.crex_id, '118X')
        self.assertIn('match-updates-118X', m.match_url)
        self.assertEqual(m.format_type, 'T20')
        self.assertEqual(m.status, 'upcoming')

    def test_t20i_not_mislabeled_odi(self):
        row = {
            'mf': '11ZZ',
            't1f': 'A',
            't2f': 'B',
            'sf': '1ZZ',
            'mn': '1',
            'ft': 2,
            'fo': 'T20I',
            'date': '4/8/2026',
            't': 1775617200000,
            'status': 0,
            'g': 1,
        }
        m = self.scraper._fixture_row_to_schedule_match(row)
        self.assertEqual(m.format_type, 'T20')

    def test_live_status_from_crickapi(self):
        row = {
            'mf': '11FW',
            't1f': '1FN',
            't2f': '1FO',
            'sf': '2GD',
            'mn': '2',
            'ft': 1,
            'fo': 'null',
            'date': '4/8/2026',
            't': 1775611800000,
            'status': 1,
            'g': 1,
            'global_num': {'fr': 'Hong Kong A', 'fri': '^1FO'},
        }
        m = self.scraper._fixture_row_to_schedule_match(row)
        self.assertEqual(m.status, 'live')
        self.assertEqual(m.team2_name, 'Hong Kong A')

    def test_flatten_dict_response(self):
        data = {
            '2026/5/9': [
                {'mf': 'X1', 't1f': 'A', 't2f': 'B', 'sf': 'S', 'mn': '1', 'ft': 2, 'fo': 'T20',
                 'date': '5/9/2026', 't': 1778335200000, 'status': 0, 'g': 1},
            ]
        }
        flat = CREXScraper._flatten_get_fixture_response(data)
        self.assertEqual(len(flat), 1)
        self.assertEqual(flat[0]['mf'], 'X1')


if __name__ == '__main__':
    unittest.main()
