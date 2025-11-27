-- Cricket Match Predictor Database Schema V2
-- Now with separate ELO ratings for each format AND gender combination

-- ============================================================================
-- ELO RATING TABLES (Updated for format + gender separation)
-- ============================================================================

-- Drop old ELO tables
DROP TABLE IF EXISTS team_elo_history;
DROP TABLE IF EXISTS player_elo_history;
DROP TABLE IF EXISTS team_current_elo;
DROP TABLE IF EXISTS player_current_elo;

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
-- VIEWS FOR RANKINGS (Updated)
-- ============================================================================

-- Drop old views
DROP VIEW IF EXISTS team_rankings_t20;
DROP VIEW IF EXISTS team_rankings_odi;
DROP VIEW IF EXISTS player_batting_rankings_t20;
DROP VIEW IF EXISTS player_bowling_rankings_t20;

-- Team rankings by format and gender
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

-- Player batting rankings
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

-- Player bowling rankings
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

