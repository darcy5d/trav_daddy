#!/usr/bin/env python3
"""
Classify teams into tiers 1-5 for ELO K-factor adjustment.

Tier System:
- Tier 1: Full ICC members (national teams)
- Tier 2: Associate nations with T20I status
- Tier 3: Premier domestic leagues (IPL, WPL, BBL, WBBL, The Hundred, CPL, SA20)
- Tier 4: Other domestic leagues (PSL, BPL, Super Smash, LPL, etc.)
- Tier 5: Regional/development cricket, club teams

This tier classification is used to adjust K-factors in ELO calculations:
- Beating a lower-tier team should give less ELO than beating an equal-tier team
- This prevents ELO inflation from winning against weak opposition
"""

import sqlite3
from pathlib import Path


# =============================================================================
# TIER 1: Full ICC Members (12 teams + major variations)
# =============================================================================
TIER_1_TEAMS = {
    # Full Members
    'Afghanistan', 'Australia', 'Bangladesh', 'England', 'India', 'Ireland',
    'New Zealand', 'Pakistan', 'South Africa', 'Sri Lanka', 'West Indies', 'Zimbabwe',
}


# =============================================================================
# TIER 2: Associate Nations with T20I Status
# =============================================================================
TIER_2_TEAMS = {
    # Top Associates / Recent World Cup participants
    'Canada', 'Hong Kong', 'Kenya', 'Namibia', 'Nepal', 'Netherlands', 'Oman',
    'Papua New Guinea', 'Scotland', 'Uganda', 'United Arab Emirates', 
    'United States of America',
    
    # Other established associates
    'Bermuda', 'Germany', 'Italy', 'Jersey', 'Malaysia', 'Singapore', 'Thailand',
    
    # Alternate names
    'USA', 'UAE', 'PNG', 'U.A.E.', 'U.S.A.',
}


# =============================================================================
# TIER 3: Premier Domestic Leagues
# =============================================================================
TIER_3_FRANCHISE_PATTERNS = [
    # IPL teams
    'Chennai Super Kings', 'Mumbai Indians', 'Royal Challengers', 'Kolkata Knight Riders',
    'Sunrisers Hyderabad', 'Rajasthan Royals', 'Delhi Capitals', 'Delhi Daredevils',
    'Punjab Kings', 'Kings XI Punjab', 'Lucknow Super Giants', 'Gujarat Titans',
    
    # WPL teams
    'Delhi Capitals Women', 'Gujarat Giants', 'Mumbai Indians Women',
    'Royal Challengers Bengaluru Women', 'UP Warriorz',
    
    # BBL teams
    'Adelaide Strikers', 'Brisbane Heat', 'Hobart Hurricanes', 'Melbourne Renegades',
    'Melbourne Stars', 'Perth Scorchers', 'Sydney Sixers', 'Sydney Thunder',
    
    # The Hundred teams
    'Birmingham Phoenix', 'London Spirit', 'Manchester Originals', 'Northern Superchargers',
    'Oval Invincibles', 'Southern Brave', 'Trent Rockets', 'Welsh Fire',
    
    # CPL teams
    'Barbados Royals', 'Guyana Amazon Warriors', 'Jamaica Tallawahs',
    'St Kitts and Nevis Patriots', 'St Lucia Kings', 'Trinbago Knight Riders',
    'Antigua and Barbuda Falcons',
    
    # SA20 teams
    'MI Cape Town', 'Joburg Super Kings', 'Durban Super Giants', 
    'Pretoria Capitals', 'Paarl Royals', 'Sunrisers Eastern Cape',
]


# =============================================================================
# TIER 4: Other Domestic Leagues
# =============================================================================
TIER_4_FRANCHISE_PATTERNS = [
    # PSL teams
    'Karachi Kings', 'Lahore Qalandars', 'Islamabad United', 'Peshawar Zalmi',
    'Quetta Gladiators', 'Multan Sultans',
    
    # BPL teams
    'Chittagong Kings', 'Chittagong Vikings', 'Comilla Victorians', 'Dhaka Dominators',
    'Dhaka Dynamites', 'Khulna Tigers', 'Khulna Titans', 'Rangpur Riders',
    'Sylhet Strikers', 'Fortune Barishal', 'Chattogram Challengers', 'Cumilla Warriors',
    
    # LPL teams
    'Colombo Stars', 'Colombo Strikers', 'Dambulla Giants', 'Dambulla Aura',
    'Galle Gladiators', 'Galle Marvels', 'Jaffna Kings', 'Kandy Falcons', 'B-Love Kandy',
    
    # ILT20 teams
    'Abu Dhabi Knight Riders', 'Desert Vipers', 'Dubai Capitals', 'Gulf Giants',
    'MI Emirates', 'Sharjah Warriors',
    
    # Super Smash (NZ)
    'Auckland', 'Canterbury', 'Central Districts', 'Northern Districts', 'Otago', 'Wellington',
    
    # English Women's domestic (Charlotte Edwards Cup, etc.)
    'Southern Vipers', 'Western Storm', 'Lightning', 'Thunder', 'Central Sparks',
    'Northern Diamonds', 'South East Stars', 'Sunrisers',
    'The Blaze',
    
    # Nepal Premier League
    'Biratnagar Kings', 'Chitwan Tigers', 'Kathmandu Kings', 'Lalitpur Patriots',
    'Pokhara Rhinos', 'Sudurpashchim Royals',
    
    # Women's Caribbean Premier League
    'Barbados Royals Women', 'Guyana Amazon Warriors Women', 'Trinbago Knight Riders Women',
]


# =============================================================================
# TIER 5 PATTERNS: Regional/Development Cricket
# =============================================================================
TIER_5_TEAMS = {
    # Developing nations
    'Argentina', 'Austria', 'Bahamas', 'Bahrain', 'Belgium', 'Belize', 'Bhutan',
    'Botswana', 'Brazil', 'Bulgaria', 'Cambodia', 'Cameroon', 'Cayman Islands',
    'Chile', 'China', 'Cook Islands', 'Costa Rica', 'Croatia', 'Cyprus',
    'Czech Republic', 'Denmark', 'Ecuador', 'Eswatini', 'Estonia', 'Fiji',
    'Finland', 'France', 'Gambia', 'Ghana', 'Gibraltar', 'Greece', 'Guernsey',
    'Hungary', 'Indonesia', 'Isle of Man', 'Israel', 'Japan', 'Kuwait',
    'Lesotho', 'Luxembourg', 'Malawi', 'Maldives', 'Mali', 'Malta', 'Mexico',
    'Mongolia', 'Morocco', 'Mozambique', 'Myanmar', 'Nigeria', 'Norway',
    'Panama', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Romania',
    'Rwanda', 'Samoa', 'Saudi Arabia', 'Serbia', 'Sierra Leone', 'Slovenia',
    'South Korea', 'Spain', 'Sweden', 'Switzerland', 'Tanzania', 'Turkey',
    'Turks and Caicos Islands', 'Vanuatu', 'Zambia',
}


# =============================================================================
# Domestic Cricket Teams (assign to Tier 4 or 5)
# =============================================================================

# Tier 4: Major domestic cricket (strong first-class cricket nations)
TIER_4_DOMESTIC = {
    # Indian states
    'Andhra', 'Baroda', 'Bengal', 'Delhi', 'Gujarat', 'Haryana', 'Himachal Pradesh',
    'Hyderabad', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Mumbai',
    'Punjab', 'Rajasthan', 'Saurashtra', 'Tamil Nadu', 'Uttar Pradesh', 'Vidarbha',
    'Railways', 'Services',
    
    # Australian states
    'New South Wales', 'Queensland', 'South Australia', 'Tasmania', 'Victoria',
    'Western Australia', 'ACT',
    
    # English counties (T20 Blast)
    'Birmingham Bears', 'Derbyshire', 'Durham', 'Essex', 'Glamorgan', 'Gloucestershire',
    'Hampshire', 'Kent', 'Lancashire', 'Leicestershire', 'Middlesex', 'Northamptonshire',
    'Nottinghamshire', 'Somerset', 'Surrey', 'Sussex', 'Warwickshire', 'Worcestershire',
    'Yorkshire', 'Notts Outlaws', 'Northants Steelbacks',
    
    # South African provincial
    'Cape Cobras', 'Dolphins', 'Knights', 'Lions', 'Titans', 'Warriors', 
    'Western Province', 'Boland', 'Border', 'Eastern Province', 'Easterns',
    'Free State', 'Gauteng', 'Griqualand West', 'KwaZulu-Natal', 'North West',
    'Northern Cape', 'Northerns',
    
    # West Indies domestic
    'Barbados', 'Guyana', 'Jamaica', 'Leeward Islands', 'Trinidad and Tobago',
    'Windward Islands', 'Combined Campuses and Colleges',
    
    # Pakistan domestic
    'Baluchistan', 'Central Punjab', 'Khyber Pakhtunkhwa', 'Northern', 'Sindh',
    'Southern Punjab', 'Federal Areas',
}

# Tier 5: Minor domestic and club cricket
TIER_5_DOMESTIC = {
    # Minor Indian states
    'Arunachal Pradesh', 'Assam', 'Bihar', 'Chandigarh', 'Chhattisgarh',
    'Goa', 'Jammu and Kashmir', 'Jharkhand', 'Manipur', 'Meghalaya', 'Mizoram',
    'Nagaland', 'Odisha', 'Pondicherry', 'Puducherry', 'Sikkim', 'Tripura', 'Uttarakhand',
    
    # Sri Lanka club cricket
    'Badureliya Sports Club', 'Bloomfield Cricket and Athletic Club',
    'Burgher Recreation Club', 'Chilaw Marians Cricket Club',
    'Colombo Cricket Club', 'Colts Cricket Club', 'Galle Cricket Club',
    'Moors Sports Club', 'Nondescripts Cricket Club', 'Panadura Sports Club',
    'Ragama Cricket Club', 'Saracens Sports Club', 'Sinhalese Sports Club',
    'Sri Lanka Air Force Sports Club', 'Sri Lanka Army Sports Club',
    'Sri Lanka Navy Sports Club', 'Sri Lanka Ports Authority Cricket Club',
    'Tamil Union Cricket and Athletic Club',
    
    # Bangladesh domestic/club
    'Abahani Limited', 'Brothers Union', 'Dhaka Division', 'Gazi Group Cricketers',
    'Khulna Division', 'Legends of Rupganj', 'Mohammedan Sporting Club',
    'Old DOHS Sports Club', 'Prime Bank Cricket Club', 'Prime Doleshwar',
    'Rajshahi Division', 'Sheikh Jamal Dhanmondi Club', 'Sylhet Division',
    'Uttara Sporting Club', 'Victoria Sporting Club',
    
    # Other
    'Barmy Army', 'Ace Capital Cricket Club',
}


def classify_team_tier(team_name: str) -> int:
    """
    Classify a team into tier 1-5.
    
    Returns:
        int: Tier number (1-5)
    """
    
    # Tier 1: Full ICC members
    if team_name in TIER_1_TEAMS:
        return 1
    
    # Tier 2: Top associates
    if team_name in TIER_2_TEAMS:
        return 2
    
    # Check domestic BEFORE franchise to avoid "Delhi" matching "Delhi Capitals"
    # Tier 4: Major domestic cricket
    if team_name in TIER_4_DOMESTIC:
        return 4
    
    # Tier 5: Minor domestic and club cricket
    if team_name in TIER_5_DOMESTIC:
        return 5
    
    # Tier 5: Specific developing nations
    if team_name in TIER_5_TEAMS:
        return 5
    
    # Tier 3: Premier franchise leagues (exact match ONLY to avoid partial matches)
    for pattern in TIER_3_FRANCHISE_PATTERNS:
        # Exact match or franchise name contains team name (but not the reverse for short names)
        if team_name == pattern:
            return 3
        # For longer team names, allow partial matching
        if len(team_name) > 10 and pattern.lower() in team_name.lower():
            return 3
    
    # Tier 4: Other franchise leagues
    for pattern in TIER_4_FRANCHISE_PATTERNS:
        if team_name == pattern:
            return 4
        if len(team_name) > 10 and pattern.lower() in team_name.lower():
            return 4
    
    # Heuristics for remaining teams
    
    # Check for common franchise indicators (likely Tier 4)
    franchise_indicators = [
        'Kings', 'Royals', 'Warriors', 'Titans', 'Giants', 'Strikers',
        'Hurricanes', 'Heat', 'Scorchers', 'Sixers', 'Thunder', 'Stars',
        'Renegades', 'Knight Riders', 'Super Kings', 'Capitals', 'Qalandars',
        'Gladiators', 'Sultans', 'United', 'Falcons', 'Stallions', 'Rhinos',
        'Patriots', 'Tallawahs', 'Phoenix', 'Spirit', 'Originals',
        'Superchargers', 'Invincibles', 'Brave', 'Rockets', 'Fire', 'Vipers',
        'Sunrisers', 'Lightning', 'Diamonds', 'Storm', 'Marvels',
    ]
    
    for indicator in franchise_indicators:
        if indicator in team_name:
            return 4  # Default franchise to Tier 4
    
    # Check for "Women" suffix (same tier as base team)
    if ' Women' in team_name or ' W' in team_name:
        base_name = team_name.replace(' Women', '').replace(' W', '').strip()
        base_tier = classify_team_tier(base_name)
        return base_tier
    
    # Default unclassified teams to Tier 5
    return 5


def main():
    db_path = Path(__file__).parent.parent / 'cricket.db'
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    print("=" * 70)
    print("TEAM TIER CLASSIFICATION")
    print("=" * 70)
    
    # Get all teams
    cursor.execute("SELECT team_id, name, tier FROM teams ORDER BY name")
    teams = cursor.fetchall()
    
    print(f"\nClassifying {len(teams)} teams into tiers 1-5...")
    
    tier_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    changes = []
    
    for team in teams:
        old_tier = team['tier']
        new_tier = classify_team_tier(team['name'])
        tier_counts[new_tier] += 1
        
        if old_tier != new_tier:
            changes.append((team['name'], old_tier, new_tier))
        
        cursor.execute(
            "UPDATE teams SET tier = ?, tier_last_reviewed = datetime('now') WHERE team_id = ?",
            (new_tier, team['team_id'])
        )
    
    conn.commit()
    
    print("\n" + "=" * 70)
    print("CLASSIFICATION SUMMARY")
    print("=" * 70)
    for tier, count in tier_counts.items():
        tier_desc = {
            1: "Full ICC Members",
            2: "Associate Nations",
            3: "Premier Leagues (IPL, BBL, CPL, etc.)",
            4: "Other Leagues & Major Domestic",
            5: "Regional/Development Cricket"
        }
        print(f"  Tier {tier} ({tier_desc[tier]}): {count} teams")
    
    print(f"\n  Total changes: {len(changes)}")
    
    # Show sample teams from each tier
    print("\n" + "=" * 70)
    print("SAMPLE TEAMS BY TIER")
    print("=" * 70)
    
    for tier in range(1, 6):
        cursor.execute(
            "SELECT name FROM teams WHERE tier = ? ORDER BY name LIMIT 10",
            (tier,)
        )
        teams = cursor.fetchall()
        print(f"\nTier {tier}:")
        for t in teams:
            print(f"  - {t['name']}")
    
    conn.close()
    print("\n" + "=" * 70)
    print("Done! Teams classified into tiers.")
    print("=" * 70)


if __name__ == '__main__':
    main()
