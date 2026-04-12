"""Unit tests for venue alias normalization."""

import unittest

from src.data.venue_normalizer import get_canonical_venue_name


class VenueAliasTests(unittest.TestCase):
    def test_national_bank_cricket_arena_maps_to_national_stadium(self):
        self.assertEqual(
            get_canonical_venue_name("National Bank Cricket Arena, Karachi"),
            "National Stadium, Karachi",
        )


if __name__ == "__main__":
    unittest.main()
