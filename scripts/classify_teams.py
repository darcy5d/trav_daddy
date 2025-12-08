#!/usr/bin/env python3
"""
Classify teams into categories: international, franchise, domestic.

This script adds a team_type column to the teams table and classifies
each team based on known patterns and lists.
"""

import sqlite3
from pathlib import Path

# Known international teams (national teams)
INTERNATIONAL_TEAMS = {
    # Full ICC Members
    'Afghanistan', 'Australia', 'Bangladesh', 'England', 'India', 'Ireland',
    'New Zealand', 'Pakistan', 'South Africa', 'Sri Lanka', 'West Indies', 'Zimbabwe',
    
    # Associate Members and other nations
    'Argentina', 'Austria', 'Bahamas', 'Bahrain', 'Belgium', 'Belize', 'Bermuda',
    'Bhutan', 'Botswana', 'Brazil', 'Bulgaria', 'Cambodia', 'Cameroon', 'Canada',
    'Cayman Islands', 'Chile', 'China', 'Cook Islands', 'Costa Rica', 'Croatia',
    'Cyprus', 'Czech Republic', 'Denmark', 'Ecuador', 'Eswatini', 'Estonia', 'Fiji',
    'Finland', 'France', 'Gambia', 'Germany', 'Ghana', 'Gibraltar', 'Greece', 
    'Guernsey', 'Hong Kong', 'Hungary', 'Indonesia', 'Isle of Man', 'Israel', 
    'Italy', 'Japan', 'Jersey', 'Kenya', 'Kuwait', 'Lesotho', 'Luxembourg',
    'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Mexico', 'Mongolia',
    'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nepal', 'Netherlands',
    'Nigeria', 'Norway', 'Oman', 'Panama', 'Papua New Guinea', 'Peru', 'Philippines',
    'Poland', 'Portugal', 'Qatar', 'Romania', 'Rwanda', 'Samoa', 'Saudi Arabia',
    'Scotland', 'Serbia', 'Sierra Leone', 'Singapore', 'Slovenia', 'South Korea',
    'Spain', 'Sweden', 'Switzerland', 'Tanzania', 'Thailand', 'Turkey', 
    'Turks and Caicos Islands', 'Uganda', 'United Arab Emirates', 
    'United States of America', 'Vanuatu', 'Zambia',
    
    # Alternative names
    'USA', 'UAE', 'PNG', 'U.A.E.', 'U.S.A.',
}

# Known franchise/league teams (patterns and specific names)
FRANCHISE_PATTERNS = [
    # IPL
    'Chennai Super Kings', 'Mumbai Indians', 'Royal Challengers', 'Kolkata Knight Riders',
    'Sunrisers Hyderabad', 'Rajasthan Royals', 'Delhi Capitals', 'Delhi Daredevils',
    'Punjab Kings', 'Kings XI Punjab', 'Lucknow Super Giants', 'Gujarat Titans',
    'Deccan Chargers', 'Kochi Tuskers Kerala', 'Pune Warriors', 'Rising Pune',
    'Gujarat Lions',
    
    # BBL
    'Adelaide Strikers', 'Brisbane Heat', 'Hobart Hurricanes', 'Melbourne Renegades',
    'Melbourne Stars', 'Perth Scorchers', 'Sydney Sixers', 'Sydney Thunder',
    
    # WBBL (same as BBL but for women)
    
    # PSL
    'Karachi Kings', 'Lahore Qalandars', 'Islamabad United', 'Peshawar Zalmi',
    'Quetta Gladiators', 'Multan Sultans',
    
    # CPL
    'Barbados Royals', 'Barbados Tridents', 'Guyana Amazon Warriors', 'Jamaica Tallawahs',
    'St Kitts and Nevis Patriots', 'St Lucia Kings', 'St Lucia Zouks', 'Trinbago Knight Riders',
    'Trinidad and Tobago Red Steel', 'Antigua and Barbuda Falcons',
    
    # The Hundred
    'Birmingham Phoenix', 'London Spirit', 'Manchester Originals', 'Northern Superchargers',
    'Oval Invincibles', 'Southern Brave', 'Trent Rockets', 'Welsh Fire',
    
    # SA20 / MSL / T20 Challenge
    'Cape Town Blitz', 'Durban Heat', 'Jozi Stars', 'Nelson Mandela Bay Giants',
    'Paarl Rocks', 'Tshwane Spartans', 'MI Cape Town', 'Joburg Super Kings',
    'Durban Super Giants', 'Pretoria Capitals', 'Paarl Royals', 'Sunrisers Eastern Cape',
    
    # BPL
    'Chittagong Kings', 'Chittagong Vikings', 'Comilla Victorians', 'Dhaka Dominators',
    'Dhaka Dynamites', 'Dhaka Gladiators', 'Khulna Tigers', 'Khulna Titans',
    'Rajshahi Kings', 'Rajshahi Royals', 'Rangpur Rangers', 'Rangpur Riders',
    'Sylhet Sixers', 'Sylhet Strikers', 'Sylhet Sunrisers', 'Barisal Bulls',
    'Chattogram Challengers', 'Cumilla Warriors', 'Fortune Barishal',
    
    # ILT20
    'Abu Dhabi Knight Riders', 'Desert Vipers', 'Dubai Capitals', 'Gulf Giants',
    'MI Emirates', 'Sharjah Warriors',
    
    # LPL
    'Colombo Kings', 'Colombo Stars', 'Colombo Strikers', 'Dambulla Aura',
    'Dambulla Giants', 'Dambulla Sixers', 'Dambulla Viiking', 'Galle Gladiators',
    'Galle Marvels', 'Jaffna Kings', 'Jaffna Stallions', 'Kandy Falcons',
    'Kandy Tuskers', 'B-Love Kandy',
    
    # Nepal Premier League
    'Biratnagar Kings', 'Biratnagar Warriors', 'Chitwan Rhinos', 'Chitwan Tigers',
    'Kathmandu Kings', 'Lalitpur Patriots', 'Pokhara Avengers', 'Pokhara Rhinos',
    'Sudurpashchim Royals',
    
    # Other franchise leagues
    'Ace Capital Cricket Club', 'Barmy Army',
    
    # Women's franchise teams
    'Yorkshire Diamonds', 'Western Storm', 'Loughborough Lightning', 'Lancashire Thunder',
    'Surrey Stars', 'Southern Vipers', 'Northern Diamonds', 'Lightning', 'Thunder',
    'Fire', 'Sunrisers', 'Sapphires', 'Spirit',
]

# Indian domestic teams (states)
INDIAN_DOMESTIC = {
    'Andhra', 'Arunachal Pradesh', 'Assam', 'Baroda', 'Bengal', 'Bihar', 'Chandigarh',
    'Chhattisgarh', 'Delhi', 'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh',
    'Hyderabad', 'Jammu and Kashmir', 'Jharkhand', 'Karnataka', 'Kerala',
    'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Mumbai',
    'Nagaland', 'Odisha', 'Pondicherry', 'Puducherry', 'Punjab', 'Rajasthan',
    'Saurashtra', 'Services', 'Sikkim', 'Tamil Nadu', 'Tripura', 'Uttar Pradesh',
    'Uttarakhand', 'Vidarbha', 'Railways',
}

# Australian domestic teams (states)
AUSTRALIAN_DOMESTIC = {
    'New South Wales', 'Queensland', 'South Australia', 'Tasmania', 'Victoria',
    'Western Australia', 'ACT', 'Australian Capital Territory',
}

# English county teams
ENGLISH_COUNTIES = {
    'Derbyshire', 'Durham', 'Essex', 'Glamorgan', 'Gloucestershire', 'Hampshire',
    'Kent', 'Lancashire', 'Leicestershire', 'Middlesex', 'Northamptonshire',
    'Nottinghamshire', 'Somerset', 'Surrey', 'Sussex', 'Warwickshire',
    'Worcestershire', 'Yorkshire', 'Birmingham Bears', 'Notts Outlaws',
    'Northants Steelbacks', 'Leicestershire Foxes', 'Durham Jets',
}

# South African domestic
SA_DOMESTIC = {
    'Boland', 'Border', 'Cape Cobras', 'Dolphins', 'Eastern Province', 'Easterns',
    'Free State', 'Gauteng', 'Griqualand West', 'KwaZulu-Natal', 'Lions',
    'North West', 'Northern Cape', 'Northerns', 'Titans', 'Warriors',
    'Western Province',
}

# New Zealand domestic
NZ_DOMESTIC = {
    'Auckland', 'Canterbury', 'Central Districts', 'Northern Districts', 'Otago',
    'Wellington',
}

# West Indies domestic
WI_DOMESTIC = {
    'Barbados', 'Combined Campuses and Colleges', 'Guyana', 'Jamaica',
    'Leeward Islands', 'Trinidad and Tobago', 'Windward Islands',
}

# Pakistan domestic
PAK_DOMESTIC = {
    'Baluchistan', 'Central Punjab', 'Federal Areas', 'Khyber Pakhtunkhwa',
    'Northern', 'Sindh', 'Southern Punjab',
}

# Sri Lanka domestic (club teams)
SL_CLUBS = {
    'Badureliya Sports Club', 'Bloomfield Cricket and Athletic Club',
    'Burgher Recreation Club', 'Chilaw Marians Cricket Club', 
    'Colombo Cricket Club', 'Colts Cricket Club', 'Galle Cricket Club',
    'Moors Sports Club', 'Nondescripts Cricket Club', 'Panadura Sports Club',
    'Ragama Cricket Club', 'Saracens Sports Club', 'Sinhalese Sports Club',
    'Sri Lanka Air Force Sports Club', 'Sri Lanka Army Sports Club',
    'Sri Lanka Navy Sports Club', 'Sri Lanka Ports Authority Cricket Club',
    'Tamil Union Cricket and Athletic Club',
}

# Bangladesh domestic
BD_DOMESTIC = {
    'Abahani Limited', 'Brothers Union', 'Dhaka Division', 'Gazi Group Cricketers',
    'Khulna Division', 'Legends of Rupganj', 'Mohammedan Sporting Club',
    'Old DOHS Sports Club', 'Prime Bank Cricket Club', 'Prime Doleshwar',
    'Rajshahi Division', 'Sheikh Jamal Dhanmondi Club', 'Sylhet Division',
    'Uttara Sporting Club', 'Victoria Sporting Club',
}


def classify_team(team_name: str) -> str:
    """Classify a team as international, franchise, or domestic."""
    
    # Check international first
    if team_name in INTERNATIONAL_TEAMS:
        return 'international'
    
    # Check domestic lists BEFORE franchise (to catch "Barbados" before "Barbados Royals" pattern)
    domestic_sets = [
        INDIAN_DOMESTIC, AUSTRALIAN_DOMESTIC, ENGLISH_COUNTIES, SA_DOMESTIC,
        NZ_DOMESTIC, WI_DOMESTIC, PAK_DOMESTIC, SL_CLUBS, BD_DOMESTIC
    ]
    
    for domestic_set in domestic_sets:
        if team_name in domestic_set:
            return 'domestic'
    
    # Check franchise patterns (exact matches first)
    for pattern in FRANCHISE_PATTERNS:
        if team_name == pattern:
            return 'franchise'
    
    # Then partial matches for franchise
    for pattern in FRANCHISE_PATTERNS:
        if pattern.lower() in team_name.lower() or team_name.lower() in pattern.lower():
            # Make sure it's not an exact match to a domestic team
            return 'franchise'
    
    # Heuristics for remaining teams
    
    # Franchise indicators in name
    franchise_indicators = [
        'Kings', 'Royals', 'Warriors', 'Titans', 'Giants', 'Strikers',
        'Hurricanes', 'Heat', 'Scorchers', 'Sixers', 'Thunder', 'Stars',
        'Renegades', 'Knight Riders', 'Super Kings', 'Capitals', 'Qalandars',
        'Gladiators', 'Sultans', 'United', 'Falcons', 'Stallions', 'Rhinos',
        'Patriots', 'Tallawahs', 'Zouks', 'Phoenix', 'Spirit', 'Originals',
        'Superchargers', 'Invincibles', 'Brave', 'Rockets', 'Fire', 'Vipers',
        'Sunrisers', 'Lightning', 'Diamonds', 'Storm', 'Marvels',
    ]
    
    for indicator in franchise_indicators:
        if indicator in team_name:
            return 'franchise'
    
    # Default to domestic for unclassified (likely regional/club teams)
    return 'domestic'


def main():
    db_path = Path(__file__).parent.parent / 'cricket.db'
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    print("Adding team_type column to teams table...")
    
    # Check if column exists
    cursor.execute("PRAGMA table_info(teams)")
    columns = [col['name'] for col in cursor.fetchall()]
    
    if 'team_type' not in columns:
        cursor.execute("ALTER TABLE teams ADD COLUMN team_type TEXT DEFAULT 'domestic'")
        print("  Column added.")
    else:
        print("  Column already exists.")
    
    # Get all teams
    cursor.execute("SELECT team_id, name FROM teams")
    teams = cursor.fetchall()
    
    print(f"\nClassifying {len(teams)} teams...")
    
    counts = {'international': 0, 'franchise': 0, 'domestic': 0}
    
    for team in teams:
        team_type = classify_team(team['name'])
        counts[team_type] += 1
        
        cursor.execute(
            "UPDATE teams SET team_type = ? WHERE team_id = ?",
            (team_type, team['team_id'])
        )
    
    conn.commit()
    
    print("\n=== Classification Summary ===")
    for team_type, count in counts.items():
        print(f"  {team_type}: {count}")
    
    # Show some examples
    print("\n=== Sample Classifications ===")
    for team_type in ['international', 'franchise', 'domestic']:
        cursor.execute(
            "SELECT name FROM teams WHERE team_type = ? ORDER BY name LIMIT 10",
            (team_type,)
        )
        teams = cursor.fetchall()
        print(f"\n{team_type.upper()}:")
        for t in teams:
            print(f"  - {t['name']}")
    
    conn.close()
    print("\nDone!")


if __name__ == '__main__':
    main()

