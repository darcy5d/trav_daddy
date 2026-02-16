#!/usr/bin/env python3
"""
One-time migration script to manually insert Afghanistan into the database.

Background: Cricsheet removed all Afghanistan match data in Nov 2024 as a political
protest over the ICC's treatment of Afghan women's cricket. This means Afghanistan
was never ingested into our database, despite being an ICC Full Member and one of
the strongest T20I sides (2024 T20 WC semi-finalists).

This script:
1. Inserts Afghanistan into the teams table with tier 1
2. Sets a calibrated starting ELO (~1700 T20 male) based on ICC rankings
3. Updates the country field for known Afghan international players
"""

import sys
import os
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATABASE_PATH


# Afghan international players known to exist in the DB from franchise cricket
AFGHAN_PLAYERS = [
    'Rashid Khan',
    'Rahmanullah Gurbaz',
    'Ibrahim Zadran',
    'Fazalhaq Farooqi',
    'Naveen-ul-Haq',
    'Gulbadin Naib',
    'Najibullah Zadran',
    'Mujeeb Ur Rahman',
    'Mohammad Nabi',
    'Azmatullah Omarzai',
    'Noor Ahmad',
    'Karim Janat',
    'Qais Ahmad',
    'Darwish Rasooli',
    'Sediqullah Atal',
    'Usman Ghani',
    'Hazratullah Zazai',
    'Asghar Afghan',
    'Sharafuddin Ashraf',
    'Fareed Ahmad',
    'Samiullah Shinwari',
    'Hashmatullah Shahidi',
    'Nangeyalia Kharote',
    'Wahidullah Zadran',
    'Masood Gurbaz',
    'Sahel Zadran',
    'Zahid Zadran',
]

# Calibrated ELO ratings for Afghanistan
# T20 male: ~1700 (between West Indies 1698 and Sri Lanka 1704)
# ODI male: ~1650 (strong ODI side)
# T20/ODI female: 1500 (no women's team due to Taliban ban)
AFG_ELO_T20_MALE = 1700.0
AFG_ELO_ODI_MALE = 1650.0
AFG_ELO_T20_FEMALE = 1500.0
AFG_ELO_ODI_FEMALE = 1500.0


def main():
    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Check if Afghanistan already exists
    cursor.execute("SELECT team_id, name, tier FROM teams WHERE name = 'Afghanistan'")
    existing = cursor.fetchone()
    if existing:
        print(f"Afghanistan already exists: team_id={existing['team_id']}, tier={existing['tier']}")
        print("Skipping team insertion. Will still update player countries.")
        afg_team_id = existing['team_id']
    else:
        # Insert Afghanistan into teams table
        cursor.execute("""
            INSERT INTO teams (name, country_code, is_international, team_type, tier, tier_notes)
            VALUES ('Afghanistan', 'AFG', 1, 'international', 1,
                    'Full member - manual entry (Cricsheet data withheld Nov 2024)')
        """)
        afg_team_id = cursor.lastrowid
        print(f"Inserted Afghanistan: team_id={afg_team_id}, tier=1")

    # Insert/update ELO ratings
    cursor.execute("SELECT team_id FROM team_current_elo WHERE team_id = ?", (afg_team_id,))
    if cursor.fetchone():
        cursor.execute("""
            UPDATE team_current_elo
            SET elo_t20_male = ?, elo_odi_male = ?,
                elo_t20_female = ?, elo_odi_female = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE team_id = ?
        """, (AFG_ELO_T20_MALE, AFG_ELO_ODI_MALE,
              AFG_ELO_T20_FEMALE, AFG_ELO_ODI_FEMALE, afg_team_id))
        print(f"Updated ELO: T20m={AFG_ELO_T20_MALE}, ODIm={AFG_ELO_ODI_MALE}")
    else:
        cursor.execute("""
            INSERT INTO team_current_elo (team_id, elo_t20_male, elo_odi_male,
                                          elo_t20_female, elo_odi_female)
            VALUES (?, ?, ?, ?, ?)
        """, (afg_team_id, AFG_ELO_T20_MALE, AFG_ELO_ODI_MALE,
              AFG_ELO_T20_FEMALE, AFG_ELO_ODI_FEMALE))
        print(f"Inserted ELO: T20m={AFG_ELO_T20_MALE}, ODIm={AFG_ELO_ODI_MALE}")

    # Update player country fields for known Afghan internationals
    updated_count = 0
    not_found = []
    for player_name in AFGHAN_PLAYERS:
        cursor.execute(
            "SELECT player_id FROM players WHERE name = ?", (player_name,)
        )
        row = cursor.fetchone()
        if row:
            cursor.execute(
                "UPDATE players SET country = 'Afghanistan' WHERE player_id = ?",
                (row['player_id'],)
            )
            updated_count += 1
        else:
            not_found.append(player_name)

    print(f"\nUpdated country='Afghanistan' for {updated_count} players")
    if not_found:
        print(f"Players not found in DB (may not have franchise data): {not_found}")

    conn.commit()

    # Verification
    print("\n--- Verification ---")
    cursor.execute(
        "SELECT team_id, name, tier, team_type, tier_notes FROM teams WHERE name = 'Afghanistan'"
    )
    team = cursor.fetchone()
    print(f"Team: id={team['team_id']}, name={team['name']}, tier={team['tier']}, "
          f"type={team['team_type']}")
    print(f"Notes: {team['tier_notes']}")

    cursor.execute(
        "SELECT elo_t20_male, elo_odi_male FROM team_current_elo WHERE team_id = ?",
        (team['team_id'],)
    )
    elo = cursor.fetchone()
    print(f"ELO: T20m={elo['elo_t20_male']}, ODIm={elo['elo_odi_male']}")

    cursor.execute(
        "SELECT COUNT(*) as cnt FROM players WHERE country = 'Afghanistan'"
    )
    cnt = cursor.fetchone()['cnt']
    print(f"Players with country='Afghanistan': {cnt}")

    conn.close()
    print("\nDone.")


if __name__ == '__main__':
    main()
