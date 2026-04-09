"""Tests for CREX team name cache functionality."""
import unittest
import tempfile
import os
from unittest.mock import Mock, patch

from src.data.team_name_cache import CREXTeamCache, CachedTeamResult
from src.api.crex_scraper import CREXScraper, CREXTeam


class TestCREXTeamCache(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.cache = CREXTeamCache()
        self.scraper = Mock(spec=CREXScraper)
    
    def test_cached_team_result_dataclass(self):
        """Test CachedTeamResult dataclass behavior."""
        # Test with database match
        result = CachedTeamResult(
            fkey="HK", 
            gender="male",
            display_name="Hong Kong",
            db_team_id=123,
            db_team_name="Hong Kong",
            match_confidence=0.95,
            source="abbreviation"
        )
        
        self.assertFalse(result.needs_default_elo)
        self.assertEqual(result.fkey, "HK")
        self.assertEqual(result.display_name, "Hong Kong")
        
        # Test without database match
        result_no_match = CachedTeamResult(
            fkey="ZF",
            gender="male", 
            display_name="Zarfistan",
            source="global_num"
        )
        
        self.assertTrue(result_no_match.needs_default_elo)
        self.assertIsNone(result_no_match.db_team_id)
    
    def test_display_name_fallback(self):
        """Test display name lookup with cache miss fallback."""
        # Cache miss should return None
        result = self.cache.get_display_name("MISSING", "male")
        self.assertIsNone(result)
    
    def test_cache_team_from_names_validation(self):
        """Test caching team from display name with database validation."""
        # Mock match_team_to_db to return a database match
        self.scraper.match_team_to_db.return_value = (123, "Hong Kong")
        
        # Cache team with validation
        result = self.cache.cache_team_from_names(
            "HK", "Hong Kong", "male", "test", self.scraper
        )
        
        # Should have database match info
        self.assertEqual(result.fkey, "HK")
        self.assertEqual(result.display_name, "Hong Kong")
        self.assertEqual(result.db_team_id, 123)
        self.assertEqual(result.db_team_name, "Hong Kong")
        self.assertFalse(result.needs_default_elo)
        
        # Verify scraper was called for validation
        self.scraper.match_team_to_db.assert_called_once()
        call_args = self.scraper.match_team_to_db.call_args[0]
        crex_team = call_args[0]
        self.assertEqual(crex_team.crex_id, "HK")
        self.assertEqual(crex_team.name, "Hong Kong")
        self.assertEqual(call_args[1], "male")  # gender
    
    def test_cache_team_no_database_match(self):
        """Test caching team that doesn't match database (preserves warnings)."""
        # Mock match_team_to_db to return no match 
        self.scraper.match_team_to_db.return_value = None
        
        # Cache team with validation
        result = self.cache.cache_team_from_names(
            "ZF", "Zarfistan", "male", "test", self.scraper
        )
        
        # Should NOT have database match info
        self.assertEqual(result.fkey, "ZF")
        self.assertEqual(result.display_name, "Zarfistan")
        self.assertIsNone(result.db_team_id)
        self.assertIsNone(result.db_team_name)
        self.assertTrue(result.needs_default_elo)  # Will trigger default ELO behavior
    
    def test_global_num_population(self):
        """Test cache population from API global_num hint."""
        fx = {
            'global_num': {
                'fr': 'Hong Kong A',
                'fri': '^HK'  # Note the ^ prefix
            },
            'g': 1  # Male gender
        }
        
        # Mock database validation
        self.scraper.match_team_to_db.return_value = None  # No database match
        
        # Should populate cache
        self.cache.populate_from_global_num(fx, self.scraper)
        
        # Verify validation was called
        self.scraper.match_team_to_db.assert_called_once()
        call_args = self.scraper.match_team_to_db.call_args[0] 
        crex_team = call_args[0]
        self.assertEqual(crex_team.crex_id, "HK")  # ^ prefix stripped
        self.assertEqual(crex_team.name, "Hong Kong A")
    
    def test_bulk_resolve_teams(self):
        """Test batch team resolution for performance."""
        # Mock abbreviation lookup for cache miss
        with patch.object(self.cache, '_populate_from_abbreviations') as mock_populate:
            result = self.cache.bulk_resolve_teams(["HK", "ZF"], "male", self.scraper)
            
            # Should attempt to populate missing teams
            mock_populate.assert_called_once()
    
    def test_database_integration_preservation(self):
        """Test that cache preserves existing database integration behavior."""
        # The key insight: cache NEVER bypasses match_team_to_db
        # It only improves display names, all validation still goes through existing pipeline
        
        # Mock scenario: team has cached display name but no database match
        self.scraper.match_team_to_db.return_value = None
        
        result = self.cache.cache_team_from_names(
            "NEW", "New Team", "male", "test", self.scraper
        )
        
        # Cache stores the "no match" result
        self.assertEqual(result.display_name, "New Team")
        self.assertIsNone(result.db_team_id)
        self.assertTrue(result.needs_default_elo)
        
        # When this team is used in predictions:
        # 1. Schedule shows "New Team" (improved UX)
        # 2. match_team_to_db still gets called with CREXTeam(name="New Team", ...)
        # 3. Still returns None -> warnings + default ELO (existing behavior)
        
        # The cache doesn't change the match_team_to_db pipeline, only improves inputs to it


class TestCacheIntegration(unittest.TestCase):
    """Integration tests with the scraper."""
    
    def test_cache_improves_schedule_display(self):
        """Test that cache integration improves schedule team names."""
        # This is tested via the enhance_team_names_from_cache method
        # which was added to CREXScraper
        
        scraper = CREXScraper(request_delay=0)
        
        # Mock matches with cryptic team IDs 
        from src.api.crex_scraper import CREXMatch
        matches = [
            CREXMatch(
                crex_id="TEST1",
                series_id="S1", 
                slug="test-match",
                title="HK vs ZF",
                team1_name="HK",  # Cryptic
                team2_name="ZF",  # Cryptic
                team1_id="HK",
                team2_id="ZF",
                match_type="Test Match",
                series_name="Test Series",
                format_type="T20",
                status="upcoming",
                gender="male"
            )
        ]
        
        # The enhance_team_names_from_cache method should improve these
        # (actual improvement depends on cache content, but method should not crash)
        enhanced = scraper.enhance_team_names_from_cache(matches)
        self.assertEqual(len(enhanced), 1)
        self.assertEqual(enhanced[0].crex_id, "TEST1")


if __name__ == '__main__':
    unittest.main()