-- ============================================================================
-- TEAM TIER CLASSIFICATIONS
-- ============================================================================
-- Manual classification of teams into 5 tiers for cross-pool ELO normalization
-- 
-- Tier 1: Elite Full Members (Top 6 international teams)
-- Tier 2: Full Members (Remaining ICC full member nations)
-- Tier 3: Top Associates + Premier Franchises (IPL, BBL, CPL, PSL, etc.)
-- Tier 4: Associates + Regional Franchises
-- Tier 5: Emerging Nations + Minor Domestic Leagues

-- ============================================================================
-- TIER 1: Elite Full Members
-- ============================================================================
-- Top 6 international cricket teams with consistent World Cup performance

UPDATE teams SET tier = 1, tier_notes = 'Elite full member' 
WHERE name IN (
  'India',
  'Australia',
  'England',
  'New Zealand',
  'South Africa',
  'Pakistan'
);

-- ============================================================================
-- TIER 2: Full Members
-- ============================================================================
-- Remaining ICC full member nations

UPDATE teams SET tier = 2, tier_notes = 'Full member' 
WHERE name IN (
  'West Indies',
  'Sri Lanka',
  'Bangladesh',
  'Afghanistan',
  'Ireland',
  'Zimbabwe'
);

-- ============================================================================
-- TIER 3: Top Associates + Premier Franchises
-- ============================================================================

-- Top Associate Nations
UPDATE teams SET tier = 3, tier_notes = 'Top associate nation'
WHERE name IN (
  'United Arab Emirates',
  'Nepal',
  'Namibia',
  'Oman',
  'Scotland',
  'Netherlands',
  'Hong Kong',
  'Papua New Guinea',
  'United States of America',
  'Canada'
);

-- Indian Premier League (IPL)
UPDATE teams SET tier = 3, tier_notes = 'IPL franchise'
WHERE name IN (
  'Mumbai Indians',
  'Chennai Super Kings',
  'Royal Challengers Bangalore',
  'Kolkata Knight Riders',
  'Delhi Capitals',
  'Rajasthan Royals',
  'Punjab Kings',
  'Sunrisers Hyderabad',
  'Gujarat Titans',
  'Lucknow Super Giants',
  'Rising Pune Supergiant',
  'Pune Warriors',
  'Kochi Tuskers Kerala',
  'Deccan Chargers'
);

-- Big Bash League (BBL)
UPDATE teams SET tier = 3, tier_notes = 'BBL franchise'
WHERE name IN (
  'Sydney Sixers',
  'Perth Scorchers',
  'Adelaide Strikers',
  'Melbourne Stars',
  'Sydney Thunder',
  'Hobart Hurricanes',
  'Melbourne Renegades',
  'Brisbane Heat'
);

-- Caribbean Premier League (CPL)
UPDATE teams SET tier = 3, tier_notes = 'CPL franchise'
WHERE name IN (
  'Trinbago Knight Riders',
  'Guyana Amazon Warriors',
  'Barbados Royals',
  'Jamaica Tallawahs',
  'St Kitts and Nevis Patriots',
  'St Lucia Kings',
  'Antigua and Barbuda Falcons',
  'Barbados Tridents',
  'St Lucia Zouks'
);

-- Pakistan Super League (PSL)
UPDATE teams SET tier = 3, tier_notes = 'PSL franchise'
WHERE name IN (
  'Islamabad United',
  'Karachi Kings',
  'Lahore Qalandars',
  'Multan Sultans',
  'Peshawar Zalmi',
  'Quetta Gladiators'
);

-- Women's Premier League (WPL)
UPDATE teams SET tier = 3, tier_notes = 'WPL franchise'
WHERE name IN (
  'Mumbai Indians (w)',
  'Delhi Capitals (w)',
  'Royal Challengers Bangalore (w)',
  'UP Warriorz',
  'Gujarat Giants'
);

-- Women's Big Bash League (WBBL) - Premier franchises
UPDATE teams SET tier = 3, tier_notes = 'WBBL franchise'
WHERE name LIKE '%Scorchers (w)%'
   OR name LIKE '%Sixers (w)%'
   OR name LIKE '%Thunder (w)%'
   OR name LIKE '%Stars (w)%'
   OR name LIKE '%Renegades (w)%'
   OR name LIKE '%Strikers (w)%'
   OR name LIKE '%Heat (w)%'
   OR name LIKE '%Hurricanes (w)%';

-- The Hundred
UPDATE teams SET tier = 3, tier_notes = 'The Hundred franchise'
WHERE name IN (
  'Oval Invincibles',
  'Manchester Originals',
  'Northern Superchargers',
  'London Spirit',
  'Birmingham Phoenix',
  'Trent Rockets',
  'Southern Brave',
  'Welsh Fire'
);

-- Bangladesh Premier League (BPL)
UPDATE teams SET tier = 3, tier_notes = 'BPL franchise'
WHERE name IN (
  'Comilla Victorians',
  'Dhaka Dominators',
  'Fortune Barishal',
  'Rangpur Riders',
  'Chattogram Challengers',
  'Sylhet Strikers',
  'Khulna Tigers',
  'Durdanto Dhaka'
);

-- Lanka Premier League (LPL)
UPDATE teams SET tier = 3, tier_notes = 'LPL franchise'
WHERE name IN (
  'Jaffna Kings',
  'Colombo Stars',
  'Dambulla Giants',
  'Galle Gladiators',
  'Kandy Falcons',
  'B-Love Kandy'
);

-- ============================================================================
-- TIER 4: Associates + Regional Franchises
-- ============================================================================

-- Associate Nations
UPDATE teams SET tier = 4, tier_notes = 'Associate nation'
WHERE name IN (
  'Kenya',
  'Uganda',
  'Tanzania',
  'Botswana',
  'Ghana',
  'Nigeria',
  'Rwanda',
  'Malawi',
  'Mozambique',
  'Swaziland',
  'Malaysia',
  'Singapore',
  'Thailand',
  'Myanmar',
  'Indonesia',
  'Philippines',
  'Kuwait',
  'Bahrain',
  'Saudi Arabia',
  'Qatar',
  'Maldives',
  'Iran',
  'Jersey',
  'Guernsey',
  'Isle of Man',
  'Denmark',
  'Norway',
  'Sweden',
  'Vanuatu',
  'Samoa',
  'Fiji',
  'Argentina',
  'Brazil',
  'Chile',
  'Mexico',
  'Bermuda',
  'Cayman Islands',
  'Bahamas',
  'Belize',
  'Panama',
  'Costa Rica',
  'China',
  'Japan',
  'South Korea',
  'Cambodia',
  'Bhutan',
  'Cook Islands'
);

-- Regional domestic franchises (South Africa, India, New Zealand, etc.)
UPDATE teams SET tier = 4, tier_notes = 'Regional domestic franchise'
WHERE name IN (
  'Titans',
  'Lions',
  'Dolphins',
  'Warriors',
  'Knights',
  'Cape Cobras',
  'Paarl Royals',
  'Joburg Super Kings',
  'Durban''s Super Giants',
  'Pretoria Capitals',
  'MI Cape Town',
  'Sunrisers Eastern Cape',
  'Auckland',
  'Canterbury',
  'Central Districts',
  'Northern Districts',
  'Otago',
  'Wellington',
  'Mumbai',
  'Delhi',
  'Karnataka',
  'Tamil Nadu',
  'Punjab',
  'Bengal',
  'Rajasthan',
  'Haryana',
  'Maharashtra',
  'Gujarat',
  'Andhra',
  'Vidarbha',
  'Baroda',
  'Saurashtra',
  'Jharkhand',
  'Madhya Pradesh',
  'Kerala',
  'Hyderabad',
  'Uttar Pradesh'
);

-- ============================================================================
-- TIER 5: Emerging Nations + Minor Domestic Leagues
-- ============================================================================

-- Emerging European Nations
UPDATE teams SET tier = 5, tier_notes = 'Emerging nation'
WHERE name IN (
  'Malta',
  'Austria',
  'Belgium',
  'Bulgaria',
  'Croatia',
  'Cyprus',
  'Estonia',
  'Finland',
  'France',
  'Germany',
  'Hungary',
  'Italy',
  'Luxembourg',
  'Romania',
  'Spain',
  'Switzerland',
  'Czechia',
  'Greece',
  'Portugal',
  'Serbia',
  'Turkey',
  'Israel'
);

-- English County Cricket (T20 Blast / Vitality Blast)
UPDATE teams SET tier = 5, tier_notes = 'English county'
WHERE name IN (
  'Somerset',
  'Surrey',
  'Lancashire',
  'Yorkshire',
  'Hampshire',
  'Kent',
  'Middlesex',
  'Essex',
  'Nottinghamshire',
  'Warwickshire',
  'Durham',
  'Derbyshire',
  'Gloucestershire',
  'Leicestershire',
  'Northamptonshire',
  'Sussex',
  'Worcestershire',
  'Glamorgan',
  'Birmingham Bears',
  'Lancashire Lightning',
  'Yorkshire Vikings'
);

-- Irish Inter-Provincial
UPDATE teams SET tier = 5, tier_notes = 'Irish provincial'
WHERE name IN (
  'Leinster Lightning',
  'Munster Reds',
  'Northern Knights',
  'North West Warriors'
);

-- Sri Lankan domestic
UPDATE teams SET tier = 5, tier_notes = 'Sri Lankan domestic'
WHERE name LIKE '%Cricket Club%'
   OR name LIKE '%Cricket and Athletic Club%'
   OR name LIKE '%Sports Club%';

-- ============================================================================
-- DEFAULT TIER FOR UNCLASSIFIED TEAMS
-- ============================================================================
-- Set remaining teams to tier 4 (associate/regional level) as default
-- Teams can be manually adjusted or flagged for review

UPDATE teams 
SET tier = 4, 
    tier_notes = 'Default classification (associate/regional)'
WHERE tier IS NULL OR tier = 3;  -- 3 is the default from schema

-- Special case: Women's teams that mirror men's teams
-- Ensure they have same tier as their men's counterpart would have
UPDATE teams 
SET tier = 3, tier_notes = 'Premier women''s franchise'
WHERE tier = 4 
  AND (name LIKE '%(w)%' OR name LIKE '%Women%')
  AND (name LIKE '%Indians%' OR name LIKE '%Super Kings%' OR name LIKE '%Knight Riders%' 
       OR name LIKE '%Capitals%' OR name LIKE '%Challengers%' OR name LIKE '%Royals%'
       OR name LIKE '%Scorchers%' OR name LIKE '%Sixers%' OR name LIKE '%Thunder%'
       OR name LIKE '%Stars%' OR name LIKE '%Heat%' OR name LIKE '%Hurricanes%');

-- ============================================================================
-- VERIFICATION QUERIES (commented out, run separately if needed)
-- ============================================================================

-- SELECT tier, COUNT(*) as team_count FROM teams GROUP BY tier ORDER BY tier;
-- SELECT tier, name FROM teams WHERE tier = 1 ORDER BY name;
-- SELECT tier, name FROM teams WHERE tier = 2 ORDER BY name;
-- SELECT name FROM teams WHERE tier IS NULL;






