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

    def test_reorders_by_venue_country_when_clear(self):
        row = {
            'mf': '11AA',
            't1f': 'IND',
            't2f': 'AUS',
            'team1': 'India Women',
            'team2': 'Australia Women',
            'sf': '1ZZ',
            'mn': '3',
            'ft': 2,
            'fo': 'T20I',
            'date': '4/8/2026',
            't': 1775617200000,
            'status': 0,
            'g': 0,
            'venue': 'Adelaide Oval, Adelaide',
        }
        m = self.scraper._fixture_row_to_schedule_match(row)
        self.assertEqual(m.team1_name, 'Australia Women')
        self.assertEqual(m.team2_name, 'India Women')
        self.assertEqual(m.team1_id, 'AUS')
        self.assertEqual(m.team2_id, 'IND')

    def test_reorders_australia_domestic_by_venue_hint(self):
        row = {
            'mf': '11AB',
            't1f': 'VIC',
            't2f': 'WA',
            'team1': 'Victoria',
            'team2': 'Western Australia',
            'sf': '1ZZ',
            'mn': '4',
            'ft': 2,
            'fo': 'T20',
            'date': '4/8/2026',
            't': 1775617200000,
            'status': 0,
            'g': 1,
            'venue': 'WACA Ground, Perth',
        }
        m = self.scraper._fixture_row_to_schedule_match(row)
        self.assertEqual(m.team1_name, 'Western Australia')
        self.assertEqual(m.team2_name, 'Victoria')
        self.assertEqual(m.team1_id, 'WA')
        self.assertEqual(m.team2_id, 'VIC')

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

    def test_womens_label_overrides_incorrect_gender_flag(self):
        row = {
            'mf': '11HT',
            't1f': '15N',
            't2f': '15P',
            'team1': 'Durham Women',
            'team2': 'Lancashire Women',
            'n': "England Women's One Day Cup 2026",
            'sf': '2L9',
            'mn': '5',
            'ft': 1,
            'fo': 'ODI',
            'date': '4/11/2026',
            't': 1775923200000,
            'status': 0,
            'g': 1,  # Upstream occasionally flags these as male.
        }
        m = self.scraper._fixture_row_to_schedule_match(row)
        self.assertEqual(m.gender, 'female')

    def test_post_enrichment_gender_correction_from_series_name(self):
        """
        Tests that gender is corrected AFTER series name enrichment.
        
        Scenario: Raw fixture has empty series name and g=1 (male).
        After series name is enriched to "Metrobank Women One Day Cup",
        gender should be corrected to female.
        """
        from src.api.crex_scraper import CREXMatch
        
        # Simulate a match that was parsed with empty series name and wrong gender
        match = CREXMatch(
            crex_id='TEST1',
            series_id='2FL',
            slug='test-match',
            title='Essex vs Kent',
            team1_name='Essex',
            team2_name='Kent',
            team1_id='ESS',
            team2_id='KNT',
            match_type='1st Match',
            series_name='',  # Empty initially
            format_type='ODI',
            status='upcoming',
            start_date='2026-04-12',
            start_time='10:00',
            date_time_gmt='2026-04-12T10:00:00Z',
            match_url='https://crex.live/scoreboard/test',
            venue=None,
            gender='male',  # Wrong - set from g=1 flag
        )
        
        # Simulate series name enrichment
        match.series_name = 'Metrobank Women One Day Cup 2026'
        
        # Run the post-enrichment gender recheck
        self.scraper._recheck_gender_from_series([match])
        
        # Gender should now be corrected
        self.assertEqual(match.gender, 'female')

    def test_post_enrichment_leaves_mens_matches_unchanged(self):
        """Verifies that men's matches are not incorrectly changed to women's."""
        from src.api.crex_scraper import CREXMatch
        
        match = CREXMatch(
            crex_id='TEST2',
            series_id='2FM',
            slug='test-mens-match',
            title='Essex vs Kent',
            team1_name='Essex',
            team2_name='Kent',
            team1_id='ESS',
            team2_id='KNT',
            match_type='1st Match',
            series_name='County Championship Division 1 2026',  # Men's competition
            format_type='TEST',
            status='upcoming',
            start_date='2026-04-12',
            start_time='10:00',
            date_time_gmt='2026-04-12T10:00:00Z',
            match_url='https://crex.live/scoreboard/test',
            venue=None,
            gender='male',
        )
        
        # Run the post-enrichment gender recheck
        self.scraper._recheck_gender_from_series([match])
        
        # Gender should remain male
        self.assertEqual(match.gender, 'male')


if __name__ == '__main__':
    unittest.main()
