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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Players table
CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    registry_id TEXT UNIQUE,  -- Cricsheet unique identifier
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
-- ELO RATING TABLES
-- ============================================================================

-- Team ELO history
CREATE TABLE IF NOT EXISTS team_elo_history (
    elo_id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id INTEGER NOT NULL REFERENCES teams(team_id),
    date DATE NOT NULL,
    match_id INTEGER REFERENCES matches(match_id),  -- Match that triggered update
    elo_t20 REAL DEFAULT 1500,
    elo_odi REAL DEFAULT 1500,
    elo_t20_change REAL DEFAULT 0,  -- Change from previous rating
    elo_odi_change REAL DEFAULT 0,
    is_monthly_snapshot BOOLEAN DEFAULT FALSE,  -- For monthly summaries
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_team_elo_team_date ON team_elo_history(team_id, date);
CREATE INDEX IF NOT EXISTS idx_team_elo_monthly ON team_elo_history(is_monthly_snapshot, date);

-- Player ELO history
CREATE TABLE IF NOT EXISTS player_elo_history (
    elo_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL REFERENCES players(player_id),
    date DATE NOT NULL,
    match_id INTEGER REFERENCES matches(match_id),
    format TEXT NOT NULL CHECK(format IN ('T20', 'ODI')),
    
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
CREATE INDEX IF NOT EXISTS idx_player_elo_format ON player_elo_history(format);
CREATE INDEX IF NOT EXISTS idx_player_elo_monthly ON player_elo_history(is_monthly_snapshot, date);

-- Current ELO ratings (denormalized for quick lookups)
CREATE TABLE IF NOT EXISTS team_current_elo (
    team_id INTEGER PRIMARY KEY REFERENCES teams(team_id),
    elo_t20 REAL DEFAULT 1500,
    elo_odi REAL DEFAULT 1500,
    last_t20_match_date DATE,
    last_odi_match_date DATE,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS player_current_elo (
    player_id INTEGER PRIMARY KEY REFERENCES players(player_id),
    batting_elo_t20 REAL DEFAULT 1500,
    bowling_elo_t20 REAL DEFAULT 1500,
    overall_elo_t20 REAL DEFAULT 1500,
    batting_elo_odi REAL DEFAULT 1500,
    bowling_elo_odi REAL DEFAULT 1500,
    overall_elo_odi REAL DEFAULT 1500,
    last_t20_match_date DATE,
    last_odi_match_date DATE,
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

-- View for current team rankings
CREATE VIEW IF NOT EXISTS team_rankings_t20 AS
SELECT 
    t.name as team_name,
    e.elo_t20 as elo,
    e.last_t20_match_date as last_match
FROM team_current_elo e
JOIN teams t ON e.team_id = t.team_id
ORDER BY e.elo_t20 DESC;

CREATE VIEW IF NOT EXISTS team_rankings_odi AS
SELECT 
    t.name as team_name,
    e.elo_odi as elo,
    e.last_odi_match_date as last_match
FROM team_current_elo e
JOIN teams t ON e.team_id = t.team_id
ORDER BY e.elo_odi DESC;

-- View for player batting rankings (T20)
CREATE VIEW IF NOT EXISTS player_batting_rankings_t20 AS
SELECT 
    p.name as player_name,
    p.country,
    e.batting_elo_t20 as elo,
    e.last_t20_match_date as last_match
FROM player_current_elo e
JOIN players p ON e.player_id = p.player_id
WHERE e.batting_elo_t20 != 1500  -- Only players with ratings
ORDER BY e.batting_elo_t20 DESC;

-- View for player bowling rankings (T20)
CREATE VIEW IF NOT EXISTS player_bowling_rankings_t20 AS
SELECT 
    p.name as player_name,
    p.country,
    e.bowling_elo_t20 as elo,
    e.last_t20_match_date as last_match
FROM player_current_elo e
JOIN players p ON e.player_id = p.player_id
WHERE e.bowling_elo_t20 != 1500
ORDER BY e.bowling_elo_t20 DESC;

