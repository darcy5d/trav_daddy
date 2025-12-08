# Cricket Match Predictor - Database Schema & ELO Methodology

## Overview

This document describes the SQLite database schema used to store cricket match data from [Cricsheet](https://cricsheet.org/) and the ELO rating methodology for teams and players.

**Data Coverage:**
- **Date Range:** 2019-01-01 to present
- **Formats:** T20 Internationals, One Day Internationals (ODI)
- **Gender:** Both men's and women's cricket
- **Source:** Cricsheet JSON data files

---

## Table of Contents

1. [Core Tables](#1-core-tables)
2. [Match Tables](#2-match-tables)
3. [Ball-by-Ball Data](#3-ball-by-ball-data)
4. [Player Statistics](#4-player-statistics)
5. [ELO Rating Tables](#5-elo-rating-tables)
6. [ELO Methodology](#6-elo-methodology)
7. [Data Quality Notes](#7-data-quality-notes)

---

## 1. Core Tables

### teams
Stores cricket teams (national and international).

| Column | Type | Description |
|--------|------|-------------|
| `team_id` | INTEGER | Primary key, auto-increment |
| `name` | TEXT | Team name (e.g., "India", "Australia") |
| `country_code` | TEXT | ISO country code (optional) |
| `is_international` | BOOLEAN | Whether it's an international team |
| `created_at` | TIMESTAMP | Record creation time |

**Note:** The same team (e.g., "India") is used for both men's and women's matches. Gender is tracked at the match level.

---

### players
Stores all cricketers who have participated in matches.

| Column | Type | Description |
|--------|------|-------------|
| `player_id` | INTEGER | Primary key, auto-increment |
| `name` | TEXT | Player's full name |
| `registry_id` | TEXT | Cricsheet unique identifier (UUID) |
| `country` | TEXT | Player's country/team association |
| `batting_style` | TEXT | "right-hand bat", "left-hand bat" |
| `bowling_style` | TEXT | "right-arm fast", "left-arm spin", etc. |
| `is_active` | BOOLEAN | Whether player is currently active |
| `created_at` | TIMESTAMP | Record creation time |
| `updated_at` | TIMESTAMP | Last update time |

**Example:**
```
player_id: 2069
name: "V Kohli"
registry_id: "ba607b88"
country: "India"
batting_style: "right-hand bat"
```

---

### venues
Stores match venues/grounds with hierarchical location data.

| Column | Type | Description |
|--------|------|-------------|
| `venue_id` | INTEGER | Primary key, auto-increment |
| `name` | TEXT | Venue name (e.g., "Melbourne Cricket Ground") |
| `city` | TEXT | City location |
| `country` | TEXT | Country (auto-populated from city mapping) |
| `canonical_name` | TEXT | Standardized name for display (removes city suffix) |
| `region` | TEXT | Supra-national grouping (e.g., "West Indies" for Caribbean venues) |
| `created_at` | TIMESTAMP | Record creation time |

**Venue Normalization:**
- 600+ venues across 75+ countries
- Fuzzy matching during ingestion prevents duplicates
- West Indies venues grouped under region with country subdivisions
- Country auto-populated using `src/data/country_mapping.py`

---

## 2. Match Tables

### matches
The central table storing match metadata.

| Column | Type | Description |
|--------|------|-------------|
| `match_id` | INTEGER | Primary key, auto-increment |
| `cricsheet_id` | TEXT | Original Cricsheet file ID (e.g., "1442989") |
| `match_type` | TEXT | "T20" or "ODI" |
| `date` | DATE | Match date |
| `venue_id` | INTEGER | FK to venues |
| `team1_id` | INTEGER | FK to teams |
| `team2_id` | INTEGER | FK to teams |
| `toss_winner_id` | INTEGER | FK to teams |
| `toss_decision` | TEXT | "bat" or "field" |
| `winner_id` | INTEGER | FK to teams (NULL for ties/no result) |
| `win_type` | TEXT | "runs", "wickets", "tie", "no result" |
| `win_margin` | INTEGER | Margin of victory |
| `player_of_match_id` | INTEGER | FK to players |
| `overs_per_innings` | INTEGER | 20 for T20, 50 for ODI |
| `match_number` | INTEGER | Match number in series |
| `event_name` | TEXT | Tournament/series name |
| `gender` | TEXT | "male" or "female" |
| `created_at` | TIMESTAMP | Record creation time |

**Key Constraint:** `match_type IN ('T20', 'ODI')`, `gender IN ('male', 'female')`

---

### innings
Stores innings-level summary data.

| Column | Type | Description |
|--------|------|-------------|
| `innings_id` | INTEGER | Primary key |
| `match_id` | INTEGER | FK to matches |
| `innings_number` | INTEGER | 1, 2, 3, or 4 |
| `batting_team_id` | INTEGER | FK to teams |
| `bowling_team_id` | INTEGER | FK to teams |
| `total_runs` | INTEGER | Total runs scored |
| `total_wickets` | INTEGER | Wickets lost |
| `total_overs` | REAL | Overs completed (e.g., 19.4) |
| `total_extras` | INTEGER | Extra runs |
| `target_runs` | INTEGER | Target for chasing team |
| `is_complete` | BOOLEAN | Whether innings completed |
| `declared` | BOOLEAN | Whether innings was declared |

---

## 3. Ball-by-Ball Data

### deliveries
The most granular table - every ball bowled in every match.

| Column | Type | Description |
|--------|------|-------------|
| `delivery_id` | INTEGER | Primary key |
| `innings_id` | INTEGER | FK to innings |
| `over_number` | INTEGER | Over number (0-19 for T20, 0-49 for ODI) |
| `ball_number` | INTEGER | Ball within over (1-6+) |
| `batter_id` | INTEGER | FK to players (striker) |
| `bowler_id` | INTEGER | FK to players |
| `non_striker_id` | INTEGER | FK to players |
| **Runs** | | |
| `runs_batter` | INTEGER | Runs scored by batter |
| `runs_extras` | INTEGER | Extra runs |
| `runs_total` | INTEGER | Total runs from delivery |
| **Extras Breakdown** | | |
| `extras_wides` | INTEGER | Wide runs |
| `extras_noballs` | INTEGER | No-ball runs |
| `extras_byes` | INTEGER | Byes |
| `extras_legbyes` | INTEGER | Leg byes |
| `extras_penalty` | INTEGER | Penalty runs |
| **Wicket Info** | | |
| `is_wicket` | BOOLEAN | Whether wicket fell |
| `wicket_type` | TEXT | "bowled", "caught", "lbw", "run out", etc. |
| `dismissed_player_id` | INTEGER | FK to players |
| `fielder1_id` | INTEGER | FK to players (catcher/fielder) |
| `fielder2_id` | INTEGER | FK to players (for run outs) |
| **Boundaries** | | |
| `is_boundary_four` | BOOLEAN | Hit a four |
| `is_boundary_six` | BOOLEAN | Hit a six |

**Expected Record Counts:**
- T20 match: ~240 deliveries (20 overs × 2 innings × 6 balls)
- ODI match: ~600 deliveries (50 overs × 2 innings × 6 balls)

---

## 4. Player Statistics

### player_match_stats
Aggregated per-player-per-match statistics (computed from deliveries).

| Column | Type | Description |
|--------|------|-------------|
| `stat_id` | INTEGER | Primary key |
| `match_id` | INTEGER | FK to matches |
| `player_id` | INTEGER | FK to players |
| `team_id` | INTEGER | FK to teams |
| **Batting Stats** | | |
| `batting_innings` | INTEGER | Number of innings batted |
| `runs_scored` | INTEGER | Total runs scored |
| `balls_faced` | INTEGER | Balls faced |
| `fours_hit` | INTEGER | Number of 4s |
| `sixes_hit` | INTEGER | Number of 6s |
| `not_out` | BOOLEAN | Whether remained not out |
| `batting_position` | INTEGER | Position in batting order |
| **Bowling Stats** | | |
| `bowling_innings` | INTEGER | Number of innings bowled |
| `overs_bowled` | REAL | Overs bowled |
| `runs_conceded` | INTEGER | Runs given away |
| `wickets_taken` | INTEGER | Wickets taken |
| `maidens` | INTEGER | Maiden overs |
| `wides_bowled` | INTEGER | Wides bowled |
| `noballs_bowled` | INTEGER | No-balls bowled |
| `dots_bowled` | INTEGER | Dot balls bowled |
| **Fielding Stats** | | |
| `catches` | INTEGER | Catches taken |
| `run_outs` | INTEGER | Run out involvements |
| `stumpings` | INTEGER | Stumpings (for keepers) |

---

## 5. ELO Rating Tables

### team_current_elo
Current (latest) ELO ratings for teams - denormalized for fast lookups.

| Column | Type | Description |
|--------|------|-------------|
| `team_id` | INTEGER | Primary key, FK to teams |
| `elo_t20_male` | REAL | T20 Men's ELO rating |
| `elo_t20_female` | REAL | T20 Women's ELO rating |
| `elo_odi_male` | REAL | ODI Men's ELO rating |
| `elo_odi_female` | REAL | ODI Women's ELO rating |
| `last_t20_male_date` | DATE | Last T20 men's match |
| `last_t20_female_date` | DATE | Last T20 women's match |
| `last_odi_male_date` | DATE | Last ODI men's match |
| `last_odi_female_date` | DATE | Last ODI women's match |
| `updated_at` | TIMESTAMP | Last update time |

---

### player_current_elo
Current ELO ratings for players (12 rating columns).

| Column | Type | Description |
|--------|------|-------------|
| `player_id` | INTEGER | Primary key, FK to players |
| **T20 Men** | | |
| `batting_elo_t20_male` | REAL | Batting rating |
| `bowling_elo_t20_male` | REAL | Bowling rating |
| `overall_elo_t20_male` | REAL | Combined rating |
| **T20 Women** | | |
| `batting_elo_t20_female` | REAL | Batting rating |
| `bowling_elo_t20_female` | REAL | Bowling rating |
| `overall_elo_t20_female` | REAL | Combined rating |
| **ODI Men** | | |
| `batting_elo_odi_male` | REAL | Batting rating |
| `bowling_elo_odi_male` | REAL | Bowling rating |
| `overall_elo_odi_male` | REAL | Combined rating |
| **ODI Women** | | |
| `batting_elo_odi_female` | REAL | Batting rating |
| `bowling_elo_odi_female` | REAL | Bowling rating |
| `overall_elo_odi_female` | REAL | Combined rating |
| `last_*_date` | DATE | Last match dates (4 columns) |

---

### team_elo_history
Historical team ELO ratings with monthly snapshots.

| Column | Type | Description |
|--------|------|-------------|
| `elo_id` | INTEGER | Primary key |
| `team_id` | INTEGER | FK to teams |
| `date` | DATE | Date of rating |
| `match_id` | INTEGER | Match that triggered update (NULL for snapshots) |
| `format` | TEXT | "T20" or "ODI" |
| `gender` | TEXT | "male" or "female" |
| `elo` | REAL | ELO rating at this point |
| `elo_change` | REAL | Change from previous rating |
| `is_monthly_snapshot` | BOOLEAN | Whether this is a month-end snapshot |
| `created_at` | TIMESTAMP | Record creation time |

---

### player_elo_history
Historical player ELO ratings.

| Column | Type | Description |
|--------|------|-------------|
| `elo_id` | INTEGER | Primary key |
| `player_id` | INTEGER | FK to players |
| `date` | DATE | Date of rating |
| `match_id` | INTEGER | Match that triggered update |
| `format` | TEXT | "T20" or "ODI" |
| `gender` | TEXT | "male" or "female" |
| `batting_elo` | REAL | Batting rating |
| `bowling_elo` | REAL | Bowling rating |
| `overall_elo` | REAL | Combined rating |
| `batting_elo_change` | REAL | Change in batting |
| `bowling_elo_change` | REAL | Change in bowling |
| `overall_elo_change` | REAL | Change in overall |
| `is_monthly_snapshot` | BOOLEAN | Month-end snapshot flag |

---

## 6. ELO Methodology

### 6.1 Overview

We maintain **4 independent ELO systems** for proper comparison:
1. T20 Men
2. T20 Women
3. ODI Men
4. ODI Women

Each player/team is rated only against opponents in the same format and gender category.

### 6.2 Configuration Parameters

```python
ELO_CONFIG = {
    "initial_rating": 1500,      # Starting rating for new teams/players
    "k_factor_team": 32,         # How much team ratings change per match
    "k_factor_player_batting": 20,
    "k_factor_player_bowling": 20,
    "rating_floor": 1000,        # Minimum possible rating
    "rating_ceiling": 2500,      # Maximum possible rating
}
```

### 6.3 Team ELO Calculation

**Expected Score Formula:**
```
E = 1 / (1 + 10^((R_opponent - R_team) / 400))
```

Where:
- `E` = Expected probability of winning (0 to 1)
- `R_team` = Team's current rating
- `R_opponent` = Opponent's current rating

**Rating Update Formula:**
```
R_new = R_old + K × (S - E)
```

Where:
- `K` = K-factor (32 for teams)
- `S` = Actual score (1 for win, 0 for loss, 0.5 for tie)
- `E` = Expected score

**Example:**
- India (1850) vs England (1700)
- Expected: E = 1 / (1 + 10^((1700-1850)/400)) = 0.70 (India 70% favored)
- If India wins: R_new = 1850 + 32 × (1 - 0.70) = 1859.6
- If India loses: R_new = 1850 + 32 × (0 - 0.70) = 1827.6

### 6.4 Player Batting ELO

Batting ELO is updated based on **performance vs expectation**:

1. **Calculate Expected Runs:**
   ```
   expected_runs = balls_faced × (avg_strike_rate / 100)
   ```
   - T20 avg_strike_rate = 130
   - ODI avg_strike_rate = 85

2. **Adjust for Opponent Strength:**
   ```
   opponent_adjustment = (opponent_team_elo - 1500) / 400
   expected_runs *= (1 - opponent_adjustment × 0.1)
   ```

3. **Calculate Performance Score:**
   ```
   performance = runs_scored / expected_runs
   actual_score = min(1.0, performance / 2.0)
   ```

4. **Update Rating:**
   ```
   R_new = R_old + K × (actual_score - expected_score)
   ```

### 6.5 Player Bowling ELO

Bowling ELO combines economy rate and wicket-taking:

1. **Economy Score (60% weight):**
   ```
   avg_economy = 8.0 (T20) or 5.5 (ODI)
   actual_economy = runs_conceded / overs_bowled
   economy_score = (expected_economy - actual_economy + 4) / 8
   ```

2. **Wicket Score (40% weight):**
   ```
   expected_wickets = overs_bowled / (4 for T20, 8 for ODI)
   wicket_score = wickets_taken / expected_wickets
   ```

3. **Combined Score:**
   ```
   actual_score = 0.6 × economy_score + 0.4 × wicket_score
   ```

### 6.6 Overall Player ELO

Weighted combination based on involvement:
```
total_balls = balls_faced + (overs_bowled × 6)
bat_weight = balls_faced / total_balls
bowl_weight = 1 - bat_weight
overall_elo = batting_elo × bat_weight + bowling_elo × bowl_weight
```

### 6.7 Monthly Snapshots

At the end of each month, we store snapshot records (`is_monthly_snapshot = TRUE`) to enable:
- Historical trend analysis
- Time-series visualization
- Point-in-time rating lookups

---

## 7. Data Quality Notes

### 7.1 Known Issues

1. **Recent Matches May Be Incomplete:**
   - Matches downloaded shortly after occurrence may have partial player data
   - Filter: Require ≥18 players per match for ML training

2. **Player Name Variations:**
   - Same player may appear with slight name differences
   - Use `registry_id` for definitive player matching

3. **Missing Data:**
   - Some matches lack toss information
   - Abandoned matches may have minimal data

### 7.2 Data Validation Queries

**Check match completeness:**
```sql
SELECT match_id, COUNT(*) as players
FROM player_match_stats
GROUP BY match_id
HAVING players < 18;
```

**Verify ball counts:**
```sql
SELECT m.match_type,
       AVG(d.cnt) as avg_deliveries
FROM (
    SELECT i.match_id, COUNT(*) as cnt
    FROM deliveries d
    JOIN innings i ON d.innings_id = i.innings_id
    GROUP BY i.match_id
) d
JOIN matches m ON d.match_id = m.match_id
GROUP BY m.match_type;
-- Expected: T20 ~240, ODI ~520
```

### 7.3 Current Data Summary

| Category | Count |
|----------|-------|
| Total Matches | ~11,300 |
| T20 Men | ~8,130 |
| T20 Women | ~3,172 |
| Total Players | ~9,620 |
| Total Venues | ~614 |
| Countries | ~75 |
| Total Teams | ~353 |

**Raw Data (Cricsheet):**
| Dataset | Match Files |
|---------|-------------|
| all_male | 16,708 |
| all_female | 3,967 |

---

## Appendix: Entity Relationship Diagram

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│  teams   │     │ players  │     │  venues  │
└────┬─────┘     └────┬─────┘     └────┬─────┘
     │                │                │
     │    ┌───────────┴───────────┐    │
     │    │                       │    │
     ▼    ▼                       ▼    ▼
┌────────────────────────────────────────────┐
│                  matches                    │
│  (team1_id, team2_id, winner_id, venue_id) │
└─────────────────────┬──────────────────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
    ┌──────────┐         ┌─────────────────────┐
    │ innings  │         │ player_match_stats  │
    └────┬─────┘         └─────────────────────┘
         │
         ▼
   ┌────────────┐
   │ deliveries │
   │ (per ball) │
   └────────────┘

┌─────────────────┐     ┌─────────────────┐
│ team_current_elo│     │player_current_elo│
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ team_elo_history│     │player_elo_history│
│ (with monthly   │     │ (with monthly   │
│  snapshots)     │     │  snapshots)     │
└─────────────────┘     └─────────────────┘
```

---

*Last updated: November 2025*

