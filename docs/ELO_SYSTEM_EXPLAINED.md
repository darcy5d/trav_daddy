# ELO Rating System Explained

A comprehensive guide to how the Cricket Match Predictor calculates and maintains ELO ratings for teams and players.

---

## Table of Contents
1. [Overview](#overview)
2. [Basic ELO Formula](#basic-elo-formula)
3. [Multi-Dimensional ELO Structure](#multi-dimensional-elo-structure)
4. [Team ELO Calculations](#team-elo-calculations)
5. [Player ELO Calculations](#player-elo-calculations)
6. [Monthly Snapshots & Temporal Tracking](#monthly-snapshots--temporal-tracking)
7. [Database Structure](#database-structure)
8. [Configuration Parameters](#configuration-parameters)
9. [How It All Fits Together](#how-it-all-fits-together)

---

## Overview

The ELO rating system is a method for calculating the relative skill levels of players/teams in head-to-head competitions. Originally developed for chess, it's been adapted here for cricket with several enhancements:

- **Separate ratings by Format**: T20 and ODI are different games requiring different skills
- **Separate ratings by Gender**: Men's and Women's cricket are tracked independently
- **Player specialization**: Batting, Bowling, and Overall ratings for each player
- **Temporal tracking**: Monthly snapshots preserve historical ratings for any point in time
- **Context-aware updates**: Performance is evaluated relative to opponent strength

---

## Basic ELO Formula

### 1. Expected Score
The probability that Team/Player A will beat Team/Player B:

```
Expected_A = 1 / (1 + 10^((Rating_B - Rating_A) / 400))
```

**Example:**
- Team A has ELO 1600
- Team B has ELO 1500
- Expected_A = 1 / (1 + 10^((1500-1600)/400)) = 1 / (1 + 10^(-0.25)) = 0.64 (64% chance to win)

### 2. Rating Update
After a match, the new rating is:

```
New_Rating = Old_Rating + K * (Actual - Expected)
```

Where:
- **K** = K-factor (how much ratings change per match)
- **Actual** = 1.0 for win, 0.5 for tie/no result, 0.0 for loss
- **Expected** = The calculated expected score

**Example:**
- Team A (1600) beats Team B (1500)
- Expected_A = 0.64
- Actual_A = 1.0
- K = 32 (team K-factor)
- New_Rating_A = 1600 + 32 * (1.0 - 0.64) = 1600 + 11.52 = **1611.52**
- New_Rating_B = 1500 + 32 * (0.0 - 0.36) = 1500 - 11.52 = **1488.48**

### 3. Boundaries
Ratings are constrained:
- **Floor**: 1000 (minimum rating)
- **Ceiling**: 2500 (maximum rating)

---

## Multi-Dimensional ELO Structure

The system maintains **16 separate ELO dimensions**:

### Format × Gender = 4 Combinations
| Format | Gender | Team ELO | Player ELO (Bat/Bowl/Overall) |
|--------|--------|----------|-------------------------------|
| T20 | Male | ✓ | ✓ ✓ ✓ |
| T20 | Female | ✓ | ✓ ✓ ✓ |
| ODI | Male | ✓ | ✓ ✓ ✓ |
| ODI | Female | ✓ | ✓ ✓ ✓ |

**Why separate?**
- A team great at T20 might not dominate ODI (different strategies)
- Women's cricket has different player pools and competitive balance
- Prevents rating pollution across unrelated competitions

**Example:**
- India (Men's T20): 1650 ELO
- India (Women's T20): 1580 ELO
- India (Men's ODI): 1620 ELO
- India (Women's ODI): 1560 ELO

These are **completely independent** ratings.

---

## Team ELO Calculations

### Match Processing Flow

```
1. Retrieve current ELO for both teams (for specific format/gender)
2. Calculate expected scores for each team
3. Determine actual scores (1.0 win, 0.5 tie, 0.0 loss)
4. Update ratings using K-factor of 32
5. Store in history table (with match_id and date)
6. Update current ELO lookup table
```

### Code Example (Simplified)

```python
# Get current ratings (as of match date)
rating1 = get_team_rating(team1_id, 'T20', 'male', match_date)  # e.g., 1600
rating2 = get_team_rating(team2_id, 'T20', 'male', match_date)  # e.g., 1500

# Expected scores
expected1 = 1 / (1 + 10**((rating2 - rating1) / 400))  # 0.64
expected2 = 1 - expected1  # 0.36

# Actual (team1 won)
actual1 = 1.0
actual2 = 0.0

# New ratings
new_rating1 = rating1 + 32 * (actual1 - expected1)  # 1611.52
new_rating2 = rating2 + 32 * (actual2 - expected2)  # 1488.48
```

### Team K-Factor
**K = 32** means:
- Maximum gain/loss per match: ±32 points (against equal opponent)
- Moderate volatility: allows ratings to change but prevents wild swings
- Fair for teams that play 10-30 matches per year

---

## Player ELO Calculations

Player ratings are more complex because we track **three dimensions**:
1. **Batting ELO** - Scoring runs
2. **Bowling ELO** - Taking wickets and economy
3. **Overall ELO** - Weighted combination

### Batting ELO Update

**Formula:**
```python
# Expected runs based on balls faced and format
avg_strike_rate = 130 for T20, 85 for ODI
expected_runs = balls_faced * (avg_strike_rate / 100)

# Adjust for opponent team strength
opponent_adjustment = (opponent_team_elo - 1500) / 400
expected_runs *= (1 - opponent_adjustment * 0.1)

# Performance score (capped at 1.0)
performance = runs_scored / max(expected_runs, 1)
actual_score = min(1.0, performance / 2.0)

# Expected score against opponent
expected_score = 1 / (1 + 10^((opponent_team_elo - batting_elo) / 400))

# Update
new_batting_elo = batting_elo + K_batting * (actual_score - expected_score)
```

**Example:**
- Player batting ELO: 1550
- Opponent team ELO: 1600
- Balls faced: 30
- Runs scored: 45
- Expected runs: 30 * 1.30 = 39 (adjusted for opponent: 38)
- Performance: 45/38 = 1.18 → actual_score = 0.59
- Expected_score: 0.47
- Change: 20 * (0.59 - 0.47) = +2.4 → **New ELO: 1552.4**

### Bowling ELO Update

**Formula:**
```python
# Economy score (0-1 scale)
avg_economy = 8.0 for T20, 5.5 for ODI
expected_economy = avg_economy * (1 + opponent_adjustment * 0.1)
actual_economy = runs_conceded / overs_bowled
economy_score = (expected_economy - actual_economy + 4) / 8  # normalized

# Wicket score (0-1 scale)
expected_wickets = overs_bowled / (4 for T20, 8 for ODI)
wicket_score = min(1.0, wickets_taken / max(expected_wickets, 0.5))

# Combined bowling performance (60% economy, 40% wickets)
actual_score = 0.6 * economy_score + 0.4 * wicket_score

# Update
new_bowling_elo = bowling_elo + K_bowling * (actual_score - expected_score)
```

**Example:**
- Bowler ELO: 1520
- Opponent team ELO: 1580
- Overs: 4, Runs conceded: 28, Wickets: 2
- Actual economy: 28/4 = 7.0
- Expected economy: 8.1 (adjusted for opponent)
- Economy score: (8.1 - 7.0 + 4) / 8 = 0.64
- Wicket score: 2 / 1 = 1.0 (capped)
- Actual: 0.6 * 0.64 + 0.4 * 1.0 = 0.78
- Expected: 0.45
- Change: 20 * (0.78 - 0.45) = +6.6 → **New ELO: 1526.6**

### Overall ELO

**Weighted by participation:**
```python
total_balls = balls_faced + (overs_bowled * 6)
bat_weight = balls_faced / total_balls
bowl_weight = 1 - bat_weight

overall_elo = (batting_elo * bat_weight) + (bowling_elo * bowl_weight)
```

**Example:**
- Batting ELO: 1620, Bowling ELO: 1480
- Balls faced: 35, Overs bowled: 3 (18 balls)
- Total: 53 balls
- Weights: 35/53 = 66% bat, 34% bowl
- Overall: 1620 * 0.66 + 1480 * 0.34 = **1572.8**

### Player K-Factors
**K_batting = 20, K_bowling = 20**
- Lower than team K (32) because players play more frequently
- Prevents excessive volatility from individual performances
- Still allows recognition of form changes

---

## Monthly Snapshots & Temporal Tracking

### Why Snapshots?

**Problem:** ELO ratings constantly change, but we need historical ratings for:
- Training models (what was a player's ELO *at the time* of a match?)
- Historical rankings (who was #1 in March 2023?)
- Trend analysis (how did a team's rating evolve?)

**Solution:** Create monthly snapshots at the end of each month.

### How It Works

```
January 2024:
  Match 1 (Jan 5):  Team A: 1500 → 1512
  Match 2 (Jan 12): Team A: 1512 → 1508
  Match 3 (Jan 28): Team A: 1508 → 1515
  [End of Month Snapshot]: Team A = 1515 (Jan 31, 2024)

February 2024:
  Match 4 (Feb 3):  Team A: 1515 → 1522
  ... more matches ...
  [End of Month Snapshot]: Team A = 1538 (Feb 29, 2024)
```

### Database Storage

**Two types of history records:**

1. **Match Records** (`is_monthly_snapshot = FALSE`)
   - Created after every match
   - Linked to `match_id`
   - Shows rating change from specific game

2. **Monthly Snapshots** (`is_monthly_snapshot = TRUE`)
   - Created at month-end
   - No `match_id` (not tied to a game)
   - Represents final rating for that month

### Query Examples

```sql
-- Get a team's rating on a specific date
SELECT elo FROM team_elo_history
WHERE team_id = 5 
  AND format = 'T20' 
  AND gender = 'male'
  AND date <= '2024-03-15'
ORDER BY date DESC, elo_id DESC
LIMIT 1;

-- Get monthly ranking history
SELECT date, elo 
FROM team_elo_history
WHERE team_id = 5 
  AND format = 'T20'
  AND gender = 'male'
  AND is_monthly_snapshot = TRUE
ORDER BY date;
```

### Calculation Flow

```python
current_months = {}  # Track (format, gender) -> current_month

for match in matches (chronological order):
    format = match.format
    gender = match.gender
    date = match.date
    year_month = date.strftime('%Y-%m')  # e.g., '2024-03'
    
    key = (format, gender)
    
    # Month changed? Create snapshot for previous month
    if key in current_months and year_month != current_months[key]:
        create_monthly_snapshot(current_months[key], format, gender)
    
    current_months[key] = year_month
    
    # Update ratings for this match
    update_team_ratings(...)
    update_player_ratings(...)

# Final snapshots for last month
for (format, gender), month in current_months.items():
    create_monthly_snapshot(month, format, gender)
```

---

## Database Structure

### Tables

#### 1. `team_elo_history`
Stores every team rating change.

| Column | Type | Description |
|--------|------|-------------|
| `elo_id` | INTEGER | Primary key |
| `team_id` | INTEGER | Foreign key to teams |
| `date` | DATE | Match date or snapshot date |
| `match_id` | INTEGER | NULL for snapshots |
| `format` | TEXT | 'T20' or 'ODI' |
| `gender` | TEXT | 'male' or 'female' |
| `elo` | REAL | New rating value |
| `elo_change` | REAL | Points gained/lost |
| `is_monthly_snapshot` | BOOLEAN | TRUE for end-of-month records |

**Indexes:**
- `(team_id, date)` - Fast historical lookups
- `(format, gender)` - Filter by competition type
- `(is_monthly_snapshot, date)` - Quick snapshot queries

#### 2. `player_elo_history`
Stores every player rating change (batting, bowling, overall).

| Column | Type | Description |
|--------|------|-------------|
| `elo_id` | INTEGER | Primary key |
| `player_id` | INTEGER | Foreign key to players |
| `date` | DATE | Match date or snapshot date |
| `match_id` | INTEGER | NULL for snapshots |
| `format` | TEXT | 'T20' or 'ODI' |
| `gender` | TEXT | 'male' or 'female' |
| `batting_elo` | REAL | Batting rating |
| `bowling_elo` | REAL | Bowling rating |
| `overall_elo` | REAL | Combined rating |
| `batting_elo_change` | REAL | Batting change |
| `bowling_elo_change` | REAL | Bowling change |
| `overall_elo_change` | REAL | Overall change |
| `is_monthly_snapshot` | BOOLEAN | TRUE for end-of-month |

#### 3. `team_current_elo`
Fast lookup table for latest ratings (denormalized).

| Column | Type | Description |
|--------|------|-------------|
| `team_id` | INTEGER | Primary key |
| `elo_t20_male` | REAL | Current T20 men's rating |
| `elo_t20_female` | REAL | Current T20 women's rating |
| `elo_odi_male` | REAL | Current ODI men's rating |
| `elo_odi_female` | REAL | Current ODI women's rating |
| `last_t20_male_date` | DATE | Last T20 men's match |
| `last_t20_female_date` | DATE | Last T20 women's match |
| `last_odi_male_date` | DATE | Last ODI men's match |
| `last_odi_female_date` | DATE | Last ODI women's match |

**Why this table?**
- **Performance**: Getting current rankings without scanning history
- **API efficiency**: `/api/rankings/teams` queries this table only
- **Updated after every match**: Kept in sync with history table

#### 4. `player_current_elo`
Fast lookup for current player ratings.

| Column | Type | Description |
|--------|------|-------------|
| `player_id` | INTEGER | Primary key |
| `batting_elo_t20_male` | REAL | Current T20 men's batting |
| `bowling_elo_t20_male` | REAL | Current T20 men's bowling |
| `overall_elo_t20_male` | REAL | Current T20 men's overall |
| ... | ... | (12 more columns for other combinations) |

---

## Configuration Parameters

From `config.py`:

```python
ELO_CONFIG = {
    "initial_rating": 1500,           # Starting point for all new teams/players
    "k_factor_team": 32,              # Team rating volatility
    "k_factor_player_batting": 20,    # Player batting volatility
    "k_factor_player_bowling": 20,    # Player bowling volatility
    "rating_floor": 1000,             # Minimum possible rating
    "rating_ceiling": 2500,           # Maximum possible rating
}
```

### Parameter Tuning

| Parameter | Effect of Increasing | Current Value | Rationale |
|-----------|----------------------|---------------|-----------|
| `initial_rating` | New teams/players start higher | 1500 | Industry standard |
| `k_factor_team` | Ratings change faster per match | 32 | Moderate - allows growth but not wild swings |
| `k_factor_player_batting` | Batting ratings more volatile | 20 | Lower than team (players play more often) |
| `k_factor_player_bowling` | Bowling ratings more volatile | 20 | Lower than team |
| `rating_floor` | Prevents ratings from going too low | 1000 | Protects new/weak players |
| `rating_ceiling` | Caps maximum rating | 2500 | Prevents inflation over time |

---

## How It All Fits Together

### Full System Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    1. INGESTION PHASE                            │
│  Raw match data → Database (matches, deliveries, player_stats)   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                 2. ELO CALCULATION PHASE                         │
│  python -m src.elo.calculator_v2                                 │
│                                                                   │
│  Process ALL matches in chronological order:                     │
│    For each match:                                               │
│      - Get current ELO (team & players) as of match date         │
│      - Calculate expected scores                                 │
│      - Determine actual scores (who won? player performance?)    │
│      - Update ratings using K-factors                            │
│      - Store in history tables                                   │
│      - Update current_elo tables                                 │
│      - Create monthly snapshot if month changed                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  3. FEATURE ENGINEERING                          │
│  Use ELO ratings as features for ML model training:             │
│    - Batter ELO (as of match date)                               │
│    - Bowler ELO (as of match date)                               │
│    - Team ELO difference                                         │
│    → Include in ball_training_data.py                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   4. MODEL TRAINING                              │
│  Neural network learns to predict ball outcomes considering:     │
│    - Current match situation (score, wickets, overs)             │
│    - Batter skill (ELO)                                          │
│    - Bowler skill (ELO)                                          │
│    - Venue characteristics                                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   5. PREDICTION PHASE                            │
│  For upcoming match:                                             │
│    1. Load current ELO ratings for selected teams/players        │
│    2. Run Monte Carlo simulation with Neural Network             │
│    3. For each simulated ball:                                   │
│         - Input: situation + batter_elo + bowler_elo + venue     │
│         - Output: probability distribution of outcomes           │
│    4. Aggregate 10,000 simulations → win probability             │
└─────────────────────────────────────────────────────────────────┘
```

### Timeline Example: India vs Australia

```
2024-01-15: Match played
├─ Pre-match:
│  ├─ Query current_elo tables for team/player ratings
│  └─ Display in UI for context
│
├─ Simulation:
│  ├─ For each ball in simulation:
│  │   └─ Use current batter_elo, bowler_elo as NN inputs
│  └─ Generate win probabilities
│
└─ Post-match (after result):
   ├─ ELO calculator processes match:
   │  ├─ Team: India 1650→1658 (+8), Australia 1620→1612 (-8)
   │  ├─ Batter: Kohli 1680→1685 (+5)
   │  └─ Bowler: Starc 1590→1588 (-2)
   │
   ├─ Insert into team_elo_history (match_id=12345)
   ├─ Insert into player_elo_history (match_id=12345)
   ├─ Update team_current_elo
   └─ Update player_current_elo

2024-01-31: End of month
└─ Create monthly snapshots:
   ├─ India T20 male: 1658 (date=2024-01-31, is_monthly_snapshot=TRUE)
   └─ All other active teams/players
```

### API Integration

```python
# Frontend requests rankings
GET /api/rankings/teams?format=T20&gender=male

# Backend queries
SELECT t.name, e.elo_t20_male 
FROM team_current_elo e
JOIN teams t ON e.team_id = t.team_id
WHERE e.elo_t20_male != 1500  -- Active teams
ORDER BY e.elo_t20_male DESC

# Returns:
[
  {"name": "India", "elo": 1658},
  {"name": "Australia", "elo": 1612},
  ...
]
```

---

## Key Insights

### 1. Temporal Accuracy
Using `as_of_date` parameter ensures we use **historical ratings** when:
- Training the model (ratings at time of match)
- Analyzing past performance
- Creating training datasets

**Without this:** A player's 2024 rating would pollute 2019 training data ❌  
**With this:** We use their actual 2019 rating ✓

### 2. Separate Dimensions
A player can be:
- Elite in Men's T20 (1800 ELO)
- Average in Men's ODI (1520 ELO)
- Not rated in Women's (not applicable)

This prevents "skill leakage" between unrelated competitions.

### 3. Dynamic vs Static
**Most systems:** Single ELO that changes over time  
**This system:** 
- Current ELO (fast lookup)
- Historical ELO (preserved in snapshots)
- Match-by-match ELO (full audit trail)

All three are maintained simultaneously.

### 4. Opponent Adjustment
Player performance is judged **relative to opponent strength**:
- 50 runs vs 1700 ELO bowling attack → bigger ELO gain
- 50 runs vs 1300 ELO bowling attack → smaller ELO gain

This prevents "stat padding" against weak teams.

---

## Maintenance & Recalculation

### When to Recalculate?

Run `python -m src.elo.calculator_v2` when:
1. New matches ingested
2. Database reset (Day 0)
3. ELO parameters changed in `config.py`

### Full Recalculation Time
- **~9,000 T20 matches** (male + female)
- **Processing:** ~2-3 minutes
- **Result:** All history and current tables updated

### Viewing Rankings

```bash
# Print current rankings
python -m src.elo.calculator_v2 --rankings --format T20 --gender male

# Or via web UI
http://localhost:5001/rankings
```

---

## Summary

The ELO system provides a **dynamic, context-aware, temporally-accurate** measure of team and player skill across multiple dimensions:

✓ **16 separate rating tracks** (4 format×gender combos, each with team + 3 player types)  
✓ **Match-by-match updates** preserve full history  
✓ **Monthly snapshots** enable historical queries  
✓ **Opponent-adjusted** performance scoring  
✓ **Denormalized current tables** for fast API responses  
✓ **Temporal accuracy** for ML training  

This powers the prediction engine by providing accurate skill assessments for the Monte Carlo simulation's neural network inputs.

---

*For implementation details, see `src/elo/calculator_v2.py` and `src/data/schema_v2.sql`*

