#!/usr/bin/env python3
"""
Add Uplands College, White River (Mpumalanga, South Africa) to venues.

This ground hosts Men's and Women's cricket including ODIs and T20s.
Run once to ensure the venue exists for CREX/ESPN matching.

Usage:
    python scripts/add_uplands_venue.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.database import get_connection


def main():
    conn = get_connection()
    cursor = conn.cursor()

    # Check if venue already exists
    cursor.execute(
        "SELECT venue_id FROM venues WHERE name = ? AND city = ?",
        ("Uplands College, White River", "White River")
    )
    if cursor.fetchone():
        print("Uplands College, White River already exists in venues.")
        conn.close()
        return 0

    cursor.execute(
        """
        INSERT INTO venues (name, city, country, canonical_name, region)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            "Uplands College, White River",
            "White River",
            "South Africa",
            "Uplands College, White River",
            None,
        )
    )
    conn.commit()
    venue_id = cursor.lastrowid
    conn.close()
    print(f"Added venue: Uplands College, White River (venue_id={venue_id})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
