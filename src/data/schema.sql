-- Cricket Match Predictor Database Schema
-- SQLite database for storing match data, player stats, and ELO ratings

-- ============================================================================
-- CORE TABLES
-- ============================================================================

-- Teams table
CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    country_code TEXT,
    is_international BOOLEAN DEFAULT TRUE,
    team_type TEXT DEFAULT 'domestic' CHECK(team_type IN ('international', 'franchise', 'domestic')),
    tier INTEGER DEFAULT 3 CHECK(tier BETWEEN 1 AND 5),
    tier_last_reviewed DATE,
    tier_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for fast tier lookups
CREATE INDEX IF NOT EXISTS idx_teams_tier ON teams(tier);

-- Players table
CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    registry_id TEXT UNIQUE,  -- Cricsheet unique identifier
    espn_player_id INTEGER,  -- ESPN Cricinfo player ID
    country TEXT,
    batting_style TEXT,  -- right-hand bat, left-hand bat
    bowling_style TEXT,  -- right-arm fast, left-arm spin, etc.
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for player name lookups
CREATE INDEX IF NOT EXISTS idx_players_name ON players(name);
CREATE INDEX IF NOT EXISTS idx_players_registry_id ON players(registry_id);
CREATE INDEX IF NOT EXISTS idx_players_espn_id ON players(espn_player_id);

-- Venues table
CREATE TABLE IF NOT EXISTS venues (
    venue_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    city TEXT,
    country TEXT,
    canonical_name TEXT,  -- Standardized name for display (e.g., "Adelaide Oval" not "Adelaide Oval, Adelaide")
    region TEXT,  -- Supra-national grouping (e.g., "West Indies" for Caribbean venues)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, city)
);

-- ============================================================================
-- MATCH TABLES
-- ============================================================================

-- Matches table
CREATE TABLE IF NOT EXISTS matches (
    match_id INTEGER PRIMARY KEY AUTOINCREMENT,
    cricsheet_id TEXT UNIQUE,  -- Original Cricsheet file identifier
    match_type TEXT NOT NULL CHECK(match_type IN ('T20', 'ODI')),
    date DATE NOT NULL,
    venue_id INTEGER REFERENCES venues(venue_id),
    team1_id INTEGER NOT NULL REFERENCES teams(team_id),
    team2_id INTEGER NOT NULL REFERENCES teams(team_id),
    toss_winner_id INTEGER REFERENCES teams(team_id),
    toss_decision TEXT CHECK(toss_decision IN ('bat', 'field')),
    winner_id INTEGER REFERENCES teams(team_id),
    win_type TEXT,  -- 'runs', 'wickets', 'tie', 'no result', 'draw'
    win_margin INTEGER,
    player_of_match_id INTEGER REFERENCES players(player_id),
    overs_per_innings INTEGER,  -- 20 for T20, 50 for ODI
    match_number INTEGER,  -- Match number in series/tournament
    event_name TEXT,  -- Tournament or series name
    gender TEXT DEFAULT 'male' CHECK(gender IN ('male', 'female')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for match queries
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(date);
CREATE INDEX IF NOT EXISTS idx_matches_type ON matches(match_type);
CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(team1_id, team2_id);
CREATE INDEX IF NOT EXISTS idx_matches_winner ON matches(winner_id);

-- Innings table
CREATE TABLE IF NOT EXISTS innings (
    innings_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL REFERENCES matches(match_id),
    innings_number INTEGER NOT NULL CHECK(innings_number IN (1, 2, 3, 4)),
    batting_team_id INTEGER NOT NULL REFERENCES teams(team_id),
    bowling_team_id INTEGER NOT NULL REFERENCES teams(team_id),
    total_runs INTEGER DEFAULT 0,
    total_wickets INTEGER DEFAULT 0,
    total_overs REAL DEFAULT 0,  -- Decimal for partial overs
    total_extras INTEGER DEFAULT 0,
    target_runs INTEGER,  -- For chasing innings
    is_complete BOOLEAN DEFAULT FALSE,
    declared BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(match_id, innings_number)
);

CREATE INDEX IF NOT EXISTS idx_innings_match ON innings(match_id);

-- ============================================================================
-- BALL-BY-BALL DATA
-- ============================================================================

-- Deliveries (ball-by-ball) table
CREATE TABLE IF NOT EXISTS deliveries (
    delivery_id INTEGER PRIMARY KEY AUTOINCREMENT,
    innings_id INTEGER NOT NULL REFERENCES innings(innings_id),
    over_number INTEGER NOT NULL,
    ball_number INTEGER NOT NULL,
    batter_id INTEGER NOT NULL REFERENCES players(player_id),
    bowler_id INTEGER NOT NULL REFERENCES players(player_id),
    non_striker_id INTEGER REFERENCES players(player_id),
    
    -- Runs
    runs_batter INTEGER DEFAULT 0,
    runs_extras INTEGER DEFAULT 0,
    runs_total INTEGER DEFAULT 0,
    
    -- Extras breakdown
    extras_wides INTEGER DEFAULT 0,
    extras_noballs INTEGER DEFAULT 0,
    extras_byes INTEGER DEFAULT 0,
    extras_legbyes INTEGER DEFAULT 0,
    extras_penalty INTEGER DEFAULT 0,
    
    -- Wicket information
    is_wicket BOOLEAN DEFAULT FALSE,
    wicket_type TEXT,  -- bowled, caught, lbw, run out, stumped, hit wicket, etc.
    dismissed_player_id INTEGER REFERENCES players(player_id),
    fielder1_id INTEGER REFERENCES players(player_id),
    fielder2_id INTEGER REFERENCES players(player_id),
    
    -- Additional info
    is_boundary_four BOOLEAN DEFAULT FALSE,
    is_boundary_six BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(innings_id, over_number, ball_number)
);

-- Indexes for delivery queries
CREATE INDEX IF NOT EXISTS idx_deliveries_innings ON deliveries(innings_id);
CREATE INDEX IF NOT EXISTS idx_deliveries_batter ON deliveries(batter_id);
CREATE INDEX IF NOT EXISTS idx_deliveries_bowler ON deliveries(bowler_id);
CREATE INDEX IF NOT EXISTS idx_deliveries_over ON deliveries(innings_id, over_number);

-- ============================================================================
-- PLAYER STATISTICS (AGGREGATED)
-- ============================================================================

-- Player match statistics (per player per match)
CREATE TABLE IF NOT EXISTS player_match_stats (
    stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL REFERENCES matches(match_id),
    player_id INTEGER NOT NULL REFERENCES players(player_id),
    team_id INTEGER NOT NULL REFERENCES teams(team_id),
    
    -- Batting stats
    batting_innings INTEGER DEFAULT 0,
    runs_scored INTEGER DEFAULT 0,
    balls_faced INTEGER DEFAULT 0,
    fours_hit INTEGER DEFAULT 0,
    sixes_hit INTEGER DEFAULT 0,
    not_out BOOLEAN DEFAULT FALSE,
    batting_position INTEGER,
    
    -- Bowling stats
    bowling_innings INTEGER DEFAULT 0,
    overs_bowled REAL DEFAULT 0,
    runs_conceded INTEGER DEFAULT 0,
    wickets_taken INTEGER DEFAULT 0,
    maidens INTEGER DEFAULT 0,
    wides_bowled INTEGER DEFAULT 0,
    noballs_bowled INTEGER DEFAULT 0,
    dots_bowled INTEGER DEFAULT 0,
    
    -- Fielding stats
    catches INTEGER DEFAULT 0,
    run_outs INTEGER DEFAULT 0,
    stumpings INTEGER DEFAULT 0,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(match_id, player_id)
);

CREATE INDEX IF NOT EXISTS idx_player_stats_match ON player_match_stats(match_id);
CREATE INDEX IF NOT EXISTS idx_player_stats_player ON player_match_stats(player_id);

-- ============================================================================
-- ELO RATING TABLES (V2 - with format and gender separation)
-- ============================================================================

-- Team ELO history (with gender separation)
CREATE TABLE IF NOT EXISTS team_elo_history (
    elo_id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id INTEGER NOT NULL REFERENCES teams(team_id),
    date DATE NOT NULL,
    match_id INTEGER REFERENCES matches(match_id),
    format TEXT NOT NULL CHECK(format IN ('T20', 'ODI')),
    gender TEXT NOT NULL CHECK(gender IN ('male', 'female')),
    elo REAL DEFAULT 1500,
    elo_change REAL DEFAULT 0,
    is_monthly_snapshot BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_team_elo_team_date ON team_elo_history(team_id, date);
CREATE INDEX IF NOT EXISTS idx_team_elo_format_gender ON team_elo_history(format, gender);
CREATE INDEX IF NOT EXISTS idx_team_elo_monthly ON team_elo_history(is_monthly_snapshot, date);

-- Player ELO history (with gender separation)
CREATE TABLE IF NOT EXISTS player_elo_history (
    elo_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL REFERENCES players(player_id),
    date DATE NOT NULL,
    match_id INTEGER REFERENCES matches(match_id),
    format TEXT NOT NULL CHECK(format IN ('T20', 'ODI')),
    gender TEXT NOT NULL CHECK(gender IN ('male', 'female')),
    
    -- ELO ratings
    batting_elo REAL DEFAULT 1500,
    bowling_elo REAL DEFAULT 1500,
    overall_elo REAL DEFAULT 1500,
    
    -- Changes from previous
    batting_elo_change REAL DEFAULT 0,
    bowling_elo_change REAL DEFAULT 0,
    overall_elo_change REAL DEFAULT 0,
    
    is_monthly_snapshot BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_player_elo_player_date ON player_elo_history(player_id, date);
CREATE INDEX IF NOT EXISTS idx_player_elo_format_gender ON player_elo_history(format, gender);
CREATE INDEX IF NOT EXISTS idx_player_elo_monthly ON player_elo_history(is_monthly_snapshot, date);

-- Current ELO ratings for teams (denormalized for quick lookups)
-- Now with 4 combinations: T20_male, T20_female, ODI_male, ODI_female
CREATE TABLE IF NOT EXISTS team_current_elo (
    team_id INTEGER PRIMARY KEY REFERENCES teams(team_id),
    -- T20
    elo_t20_male REAL DEFAULT 1500,
    elo_t20_female REAL DEFAULT 1500,
    -- ODI
    elo_odi_male REAL DEFAULT 1500,
    elo_odi_female REAL DEFAULT 1500,
    -- Last match dates
    last_t20_male_date DATE,
    last_t20_female_date DATE,
    last_odi_male_date DATE,
    last_odi_female_date DATE,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Current ELO ratings for players (denormalized for quick lookups)
CREATE TABLE IF NOT EXISTS player_current_elo (
    player_id INTEGER PRIMARY KEY REFERENCES players(player_id),
    -- T20 Male
    batting_elo_t20_male REAL DEFAULT 1500,
    bowling_elo_t20_male REAL DEFAULT 1500,
    overall_elo_t20_male REAL DEFAULT 1500,
    -- T20 Female
    batting_elo_t20_female REAL DEFAULT 1500,
    bowling_elo_t20_female REAL DEFAULT 1500,
    overall_elo_t20_female REAL DEFAULT 1500,
    -- ODI Male
    batting_elo_odi_male REAL DEFAULT 1500,
    bowling_elo_odi_male REAL DEFAULT 1500,
    overall_elo_odi_male REAL DEFAULT 1500,
    -- ODI Female
    batting_elo_odi_female REAL DEFAULT 1500,
    bowling_elo_odi_female REAL DEFAULT 1500,
    overall_elo_odi_female REAL DEFAULT 1500,
    -- Last match dates
    last_t20_male_date DATE,
    last_t20_female_date DATE,
    last_odi_male_date DATE,
    last_odi_female_date DATE,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- View for match summary with team names
CREATE VIEW IF NOT EXISTS match_summary AS
SELECT 
    m.match_id,
    m.cricsheet_id,
    m.match_type,
    m.date,
    v.name as venue_name,
    v.city as venue_city,
    t1.name as team1_name,
    t2.name as team2_name,
    tw.name as toss_winner_name,
    m.toss_decision,
    w.name as winner_name,
    m.win_type,
    m.win_margin,
    m.event_name
FROM matches m
LEFT JOIN venues v ON m.venue_id = v.venue_id
LEFT JOIN teams t1 ON m.team1_id = t1.team_id
LEFT JOIN teams t2 ON m.team2_id = t2.team_id
LEFT JOIN teams tw ON m.toss_winner_id = tw.team_id
LEFT JOIN teams w ON m.winner_id = w.team_id;

-- Views for team rankings (by format and gender)
CREATE VIEW IF NOT EXISTS team_rankings_t20_male AS
SELECT 
    t.name as team_name,
    e.elo_t20_male as elo,
    e.last_t20_male_date as last_match
FROM team_current_elo e
JOIN teams t ON e.team_id = t.team_id
WHERE e.elo_t20_male != 1500
ORDER BY e.elo_t20_male DESC;

CREATE VIEW IF NOT EXISTS team_rankings_t20_female AS
SELECT 
    t.name as team_name,
    e.elo_t20_female as elo,
    e.last_t20_female_date as last_match
FROM team_current_elo e
JOIN teams t ON e.team_id = t.team_id
WHERE e.elo_t20_female != 1500
ORDER BY e.elo_t20_female DESC;

CREATE VIEW IF NOT EXISTS team_rankings_odi_male AS
SELECT 
    t.name as team_name,
    e.elo_odi_male as elo,
    e.last_odi_male_date as last_match
FROM team_current_elo e
JOIN teams t ON e.team_id = t.team_id
WHERE e.elo_odi_male != 1500
ORDER BY e.elo_odi_male DESC;

CREATE VIEW IF NOT EXISTS team_rankings_odi_female AS
SELECT 
    t.name as team_name,
    e.elo_odi_female as elo,
    e.last_odi_female_date as last_match
FROM team_current_elo e
JOIN teams t ON e.team_id = t.team_id
WHERE e.elo_odi_female != 1500
ORDER BY e.elo_odi_female DESC;

-- Views for player batting rankings
CREATE VIEW IF NOT EXISTS player_batting_rankings_t20_male AS
SELECT 
    p.name as player_name,
    p.country,
    e.batting_elo_t20_male as elo,
    e.last_t20_male_date as last_match
FROM player_current_elo e
JOIN players p ON e.player_id = p.player_id
WHERE e.batting_elo_t20_male != 1500
ORDER BY e.batting_elo_t20_male DESC;

CREATE VIEW IF NOT EXISTS player_batting_rankings_t20_female AS
SELECT 
    p.name as player_name,
    p.country,
    e.batting_elo_t20_female as elo,
    e.last_t20_female_date as last_match
FROM player_current_elo e
JOIN players p ON e.player_id = p.player_id
WHERE e.batting_elo_t20_female != 1500
ORDER BY e.batting_elo_t20_female DESC;

-- Views for player bowling rankings
CREATE VIEW IF NOT EXISTS player_bowling_rankings_t20_male AS
SELECT 
    p.name as player_name,
    p.country,
    e.bowling_elo_t20_male as elo,
    e.last_t20_male_date as last_match
FROM player_current_elo e
JOIN players p ON e.player_id = p.player_id
WHERE e.bowling_elo_t20_male != 1500
ORDER BY e.bowling_elo_t20_male DESC;

CREATE VIEW IF NOT EXISTS player_bowling_rankings_t20_female AS
SELECT 
    p.name as player_name,
    p.country,
    e.bowling_elo_t20_female as elo,
    e.last_t20_female_date as last_match
FROM player_current_elo e
JOIN players p ON e.player_id = p.player_id
WHERE e.bowling_elo_t20_female != 1500
ORDER BY e.bowling_elo_t20_female DESC;

-- ============================================================================
-- TIERED ELO SYSTEM TABLES
-- ============================================================================

-- Tournament tiers mapping table
CREATE TABLE IF NOT EXISTS tournament_tiers (
    tournament_pattern TEXT PRIMARY KEY,
    base_tier INTEGER CHECK(base_tier BETWEEN 1 AND 5) NOT NULL,
    notes TEXT
);

-- Populate tournament tier patterns
INSERT OR IGNORE INTO tournament_tiers VALUES 
    -- World Cups and Champions Events (Tier 1)
    ('%T20 World Cup%', 1, 'Premier global tournament'),
    ('%World Cup%', 1, 'Premier global tournament'),
    ('%Champions Trophy%', 1, 'Premier ICC event'),
    ('%World Twenty20%', 1, 'Legacy World Cup naming'),
    ('%ICC Men''s T20 World Cup%', 1, 'Full ICC World Cup title'),
    ('%ICC Women''s T20 World Cup%', 1, 'Full ICC World Cup title'),
    -- Bilateral International (Full Members) - Tier 2
    ('%tour of%', 2, 'Bilateral international series'),
    ('%T20I Series%', 2, 'T20 International series'),
    ('%Triangular%', 2, 'Multi-nation tournament'),
    ('%Tri-Series%', 2, 'Tri-nation series'),
    ('%Tri-Nation%', 2, 'Tri-nation series'),
    -- Premier Franchise Leagues - Tier 3
    ('%Indian Premier League%', 3, 'IPL'),
    ('%Big Bash League%', 3, 'BBL'),
    ('%Caribbean Premier League%', 3, 'CPL'),
    ('%Pakistan Super League%', 3, 'PSL'),
    ('%The Hundred%', 3, 'The Hundred'),
    ('%Super Smash%', 3, 'New Zealand domestic T20'),
    ('%Bangladesh Premier League%', 3, 'BPL'),
    ('%Lanka Premier League%', 3, 'LPL'),
    ('%Women''s Premier League%', 3, 'WPL India'),
    -- Regional/Associate Tournaments - Tier 4
    ('%Africa%', 4, 'African regional'),
    ('%Asia Cup%', 4, 'Asian regional'),
    ('%ACC%', 4, 'Asian Cricket Council'),
    ('%Continental Cup%', 4, 'Associate regional'),
    ('%ICC World Cup Qualifier%', 4, 'World Cup qualifying'),
    ('%East Asia%', 4, 'Regional Asian'),
    ('%Europe%', 4, 'European regional'),
    -- Domestic/Minor Leagues - Tier 5
    ('%County%', 5, 'English county cricket'),
    ('%Trophy%', 5, 'Domestic trophy'),
    ('%Challenge%', 5, 'Domestic challenge'),
    ('%T20 Blast%', 5, 'English domestic T20'),
    ('%Vitality Blast%', 5, 'English domestic T20'),
    ('%Syed Mushtaq Ali%', 5, 'Indian domestic T20'),
    ('%Inter-Provincial%', 5, 'Irish domestic'),
    -- Catch-all patterns (lowest priority)
    ('%Cup%', 5, 'Generic cup tournament'),
    ('%Series%', 4, 'Generic series'),
    ('%Tournament%', 4, 'Generic tournament');

-- Promotion review flags table
CREATE TABLE IF NOT EXISTS promotion_review_flags (
    flag_id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id INTEGER NOT NULL REFERENCES teams(team_id),
    format TEXT NOT NULL CHECK(format IN ('T20', 'ODI')),
    gender TEXT NOT NULL CHECK(gender IN ('male', 'female')),
    current_tier INTEGER NOT NULL CHECK(current_tier BETWEEN 1 AND 5),
    suggested_tier INTEGER NOT NULL CHECK(suggested_tier BETWEEN 1 AND 5),
    trigger_reason TEXT NOT NULL,
    current_elo REAL NOT NULL,
    months_at_ceiling INTEGER,
    cross_tier_record TEXT,
    flagged_date DATE DEFAULT CURRENT_DATE,
    reviewed BOOLEAN DEFAULT FALSE,
    reviewed_date DATE,
    reviewer_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(team_id, format, gender, reviewed)
);

CREATE INDEX IF NOT EXISTS idx_promotion_flags_pending ON promotion_review_flags(reviewed, flagged_date);
CREATE INDEX IF NOT EXISTS idx_promotion_flags_team ON promotion_review_flags(team_id, format, gender);

