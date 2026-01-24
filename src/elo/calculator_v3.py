"""
ELO Rating Calculator V3 - Tiered Cross-Pool Normalization.

Maintains separate ELO ratings for:
- T20 Men / T20 Women / ODI Men / ODI Women

NEW in V3:
- Tiered K-factors (5 tiers)
- Cross-pool asymmetric K-factor adjustments
- Prestige-adjusted expected scores
- Tier ceiling/floor enforcement
- Hybrid tournament classification
- Automatic promotion review triggers
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import math

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import ELO_CONFIG, DATABASE_PATH
from src.data.database import get_db_connection

logger = logging.getLogger(__name__)


class EloCalculatorV3:
    """
    ELO Rating Calculator with tiered cross-pool normalization.
    
    Enhancements over V2:
    - Tiered K-factors (5 tiers)
    - Cross-pool asymmetric updates
    - Prestige-adjusted expected scores
    - Tier ceiling/floor enforcement
    - Hybrid tournament classification
    - Automatic promotion review triggers
    """
    
    # Tier-based K-factors
    TIER_K_FACTORS = {
        1: 40,  # Elite full members (World Cup level impact)
        2: 32,  # Full members (baseline)
        3: 24,  # Top associates / premier franchises
        4: 20,  # Associates / regional franchises
        5: 16,  # Emerging / minor leagues
    }
    
    # Tier ceilings (maximum ELO per tier)
    TIER_CEILINGS = {
        1: 2500,  # No effective ceiling for elite
        2: 1950,  # Full members max
        3: 1800,  # Top associates max
        4: 1700,  # Associates max
        5: 1600,  # Emerging max
    }
    
    # Tier floors (minimum ELO per tier)
    TIER_FLOORS = {
        1: 1550,  # Elite teams maintain prestige
        2: 1450,
        3: 1350,
        4: 1250,
        5: 1150,
    }
    
    # Player K-factor adjustment based on opponent team tier
    # When playing against weaker teams, K-factor is reduced
    PLAYER_TIER_K_ADJUSTMENT = {
        1: 1.0,   # Full K vs Tier 1 (Australia, India, England)
        2: 0.8,   # 80% K vs Tier 2 (Associates)
        3: 0.6,   # 60% K vs Tier 3 (Franchise leagues)
        4: 0.4,   # 40% K vs Tier 4 (Minor leagues)
        5: 0.2,   # 20% K vs Tier 5 (Development)
    }
    
    # Initial ratings by tier
    TIER_INITIAL_RATINGS = {
        1: 1650,
        2: 1550,
        3: 1450,
        4: 1350,
        5: 1250,
    }
    
    def __init__(self):
        self.initial_rating = ELO_CONFIG['initial_rating']
        self.k_factor_batting = ELO_CONFIG['k_factor_player_batting']
        self.k_factor_bowling = ELO_CONFIG['k_factor_player_bowling']
        self.rating_floor = ELO_CONFIG['rating_floor']
        self.rating_ceiling = ELO_CONFIG['rating_ceiling']
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player/team A against B."""
        exponent = (rating_b - rating_a) / 400.0
        return 1.0 / (1.0 + math.pow(10, exponent))
    
    def calculate_new_rating(
        self,
        old_rating: float,
        expected: float,
        actual: float,
        k_factor: float
    ) -> float:
        """Calculate new rating after a match/event."""
        new_rating = old_rating + k_factor * (actual - expected)
        return max(self.rating_floor, min(self.rating_ceiling, new_rating))
    
    def get_cross_pool_k_factors(
        self,
        team1_tier: int,
        team2_tier: int,
        base_k1: float,
        base_k2: float
    ) -> Tuple[float, float]:
        """
        Apply asymmetric K-factor adjustments for cross-tier matches.
        
        Logic:
        - Same tier: No adjustment (1.0, 1.0)
        - 1 tier gap: Slight asymmetry (0.85, 1.15)
        - 2 tier gap: Moderate asymmetry (0.6, 1.4)
        - 3+ tier gap: Heavy asymmetry (0.4, 1.6)
        
        Effect: Higher-tier teams gain less and lose more in upsets.
        """
        tier_gap = abs(team1_tier - team2_tier)
        
        if tier_gap == 0:
            return base_k1, base_k2
        
        # Determine which is higher tier (lower number = higher tier)
        if team1_tier < team2_tier:
            higher_k, lower_k = base_k1, base_k2
            team1_is_higher = True
        else:
            higher_k, lower_k = base_k2, base_k1
            team1_is_higher = False
        
        # Asymmetry multipliers
        if tier_gap == 1:
            mult_higher, mult_lower = 0.85, 1.15
        elif tier_gap == 2:
            mult_higher, mult_lower = 0.6, 1.4
        else:  # >= 3
            mult_higher, mult_lower = 0.4, 1.6
        
        adj_higher = higher_k * mult_higher
        adj_lower = lower_k * mult_lower
        
        # Return in original order
        return (adj_higher, adj_lower) if team1_is_higher else (adj_lower, adj_higher)
    
    def get_prestige_adjusted_expected(
        self,
        rating1: float,
        rating2: float,
        tier1: int,
        tier2: int
    ) -> Tuple[float, float]:
        """
        Calculate expected scores with tier prestige boost.
        
        Logic: Higher-tier teams get small boost (0.04 per tier gap)
        Max adjustment: ±0.15 (clamped)
        
        Effect: India (Tier 1) vs Uganda (Tier 4) 
        → India gets +0.12 expected score boost beyond ELO difference
        """
        base_expected1 = self.expected_score(rating1, rating2)
        
        tier_gap = tier1 - tier2  # Negative if team1 is higher tier
        prestige_adjustment = 0.04 * tier_gap
        
        # Clamp to ±0.15 max
        prestige_adjustment = max(-0.15, min(0.15, prestige_adjustment))
        
        adjusted_expected1 = base_expected1 + prestige_adjustment
        adjusted_expected1 = max(0.05, min(0.95, adjusted_expected1))
        
        return adjusted_expected1, 1 - adjusted_expected1
    
    def get_match_tier(
        self,
        conn,
        tournament_name: str,
        team1_tier: int,
        team2_tier: int
    ) -> int:
        """
        Determine effective match tier using hybrid approach.
        
        Logic:
        1. Get series base tier from tournament_tiers table
        2. If teams are both higher tier than series, upgrade match by 1
        3. If teams are both lower tier than series, downgrade match by 1
        4. Clamp to valid range [1, 5]
        
        Examples:
        - World Cup (base=1) + India vs Australia (both tier 1) → Match tier 1
        - World Cup (base=1) + Uganda vs Kenya (both tier 4) → Match tier 2 (downgraded)
        - Bilateral (base=2) + Somerset vs Surrey (both tier 5) → Match tier 3 (downgraded)
        """
        cursor = conn.cursor()
        
        # Get base tier from tournament pattern matching
        cursor.execute("""
            SELECT base_tier FROM tournament_tiers
            WHERE ? LIKE tournament_pattern
            ORDER BY LENGTH(tournament_pattern) DESC
            LIMIT 1
        """, (tournament_name,))
        
        row = cursor.fetchone()
        base_tier = row['base_tier'] if row else 3  # Default to tier 3
        
        # Apply team-based adjustment
        avg_team_tier = (team1_tier + team2_tier) / 2
        
        if avg_team_tier < base_tier - 0.5:
            # Both teams higher quality → upgrade by 1
            match_tier = base_tier - 1
        elif avg_team_tier > base_tier + 0.5:
            # Both teams lower quality → downgrade by 1
            match_tier = base_tier + 1
        else:
            # Teams match series level
            match_tier = base_tier
        
        return max(1, min(5, match_tier))  # Clamp [1, 5]
    
    def get_tournament_weight(self, match_tier: int) -> float:
        """
        Convert match tier to K-factor multiplier.
        
        Logic: Higher importance matches have higher volatility
        - Tier 1 (World Cup): 1.3x multiplier
        - Tier 2 (Bilateral): 1.0x (baseline)
        - Tier 3 (Franchise): 0.9x
        - Tier 4 (Regional): 0.8x
        - Tier 5 (Minor): 0.7x
        """
        weights = {1: 1.3, 2: 1.0, 3: 0.9, 4: 0.8, 5: 0.7}
        return weights.get(match_tier, 1.0)
    
    def apply_tier_boundaries(self, new_rating: float, tier: int) -> float:
        """Enforce tier ceiling/floor."""
        ceiling = self.TIER_CEILINGS[tier]
        floor = self.TIER_FLOORS[tier]
        return max(floor, min(ceiling, new_rating))
    
    def get_team_rating(
        self,
        conn,
        team_id: int,
        match_format: str,
        gender: str,
        as_of_date: Optional[datetime] = None
    ) -> float:
        """Get team rating for specific format and gender."""
        cursor = conn.cursor()
        
        if as_of_date:
            cursor.execute("""
                SELECT elo FROM team_elo_history
                WHERE team_id = ? AND format = ? AND gender = ? AND date <= ?
                ORDER BY date DESC, elo_id DESC
                LIMIT 1
            """, (team_id, match_format, gender, as_of_date))
        else:
            col = f'elo_{match_format.lower()}_{gender}'
            cursor.execute(f"""
                SELECT {col} FROM team_current_elo
                WHERE team_id = ?
            """, (team_id,))
        
        row = cursor.fetchone()
        return row[0] if row else self.initial_rating
    
    def get_player_rating(
        self,
        conn,
        player_id: int,
        match_format: str,
        gender: str,
        rating_type: str = 'overall',
        as_of_date: Optional[datetime] = None
    ) -> float:
        """Get player rating for specific format, gender, and type."""
        cursor = conn.cursor()
        
        if as_of_date:
            cursor.execute(f"""
                SELECT {rating_type}_elo FROM player_elo_history
                WHERE player_id = ? AND format = ? AND gender = ? AND date <= ?
                ORDER BY date DESC, elo_id DESC
                LIMIT 1
            """, (player_id, match_format, gender, as_of_date))
        else:
            col = f'{rating_type}_elo_{match_format.lower()}_{gender}'
            cursor.execute(f"""
                SELECT {col} FROM player_current_elo
                WHERE player_id = ?
            """, (player_id,))
        
        row = cursor.fetchone()
        return row[0] if row else self.initial_rating
    
    def update_team_ratings(
        self,
        conn,
        match_id: int,
        team1_id: int,
        team2_id: int,
        winner_id: Optional[int],
        match_format: str,
        gender: str,
        match_date: datetime,
        tournament_name: str
    ) -> Tuple[float, float]:
        """
        Update team ELO with full tiered cross-pool logic.
        
        Flow:
        1. Get team tiers
        2. Get current ratings (as of match date)
        3. Determine match tier (hybrid tournament + teams)
        4. Get base K-factors from team tiers
        5. Apply tournament weight multiplier
        6. Apply cross-pool asymmetry
        7. Calculate prestige-adjusted expected scores
        8. Update ratings with tier boundaries
        9. Store in history
        10. Check for promotion review triggers
        """
        cursor = conn.cursor()
        
        # Step 1: Get team tiers
        cursor.execute("SELECT tier FROM teams WHERE team_id = ?", (team1_id,))
        tier1_row = cursor.fetchone()
        tier1 = tier1_row['tier'] if tier1_row and tier1_row['tier'] else 3
        
        cursor.execute("SELECT tier FROM teams WHERE team_id = ?", (team2_id,))
        tier2_row = cursor.fetchone()
        tier2 = tier2_row['tier'] if tier2_row and tier2_row['tier'] else 3
        
        # Step 2: Current ratings
        rating1 = self.get_team_rating(conn, team1_id, match_format, gender, match_date)
        rating2 = self.get_team_rating(conn, team2_id, match_format, gender, match_date)
        
        # Step 3: Match tier
        match_tier = self.get_match_tier(conn, tournament_name, tier1, tier2)
        
        # Step 4: Base K-factors
        base_k1 = self.TIER_K_FACTORS[tier1]
        base_k2 = self.TIER_K_FACTORS[tier2]
        
        # Step 5: Tournament weight
        tournament_weight = self.get_tournament_weight(match_tier)
        base_k1 *= tournament_weight
        base_k2 *= tournament_weight
        
        # Step 6: Cross-pool asymmetry
        k1, k2 = self.get_cross_pool_k_factors(tier1, tier2, base_k1, base_k2)
        
        # Step 7: Expected scores with prestige
        expected1, expected2 = self.get_prestige_adjusted_expected(
            rating1, rating2, tier1, tier2
        )
        
        # Step 8: Actual scores
        if winner_id == team1_id:
            actual1, actual2 = 1.0, 0.0
        elif winner_id == team2_id:
            actual1, actual2 = 0.0, 1.0
        else:
            actual1, actual2 = 0.5, 0.5
        
        # Calculate new ratings
        new_rating1 = self.calculate_new_rating(rating1, expected1, actual1, k1)
        new_rating2 = self.calculate_new_rating(rating2, expected2, actual2, k2)
        
        # Apply tier boundaries
        new_rating1 = self.apply_tier_boundaries(new_rating1, tier1)
        new_rating2 = self.apply_tier_boundaries(new_rating2, tier2)
        
        # Step 9: Store in history
        for team_id, new_rating, old_rating in [
            (team1_id, new_rating1, rating1),
            (team2_id, new_rating2, rating2)
        ]:
            change = new_rating - old_rating
            cursor.execute("""
                INSERT INTO team_elo_history (
                    team_id, date, match_id, format, gender, elo, elo_change
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (team_id, match_date, match_id, match_format, gender, new_rating, change))
            
            # Update current rating
            col = f'elo_{match_format.lower()}_{gender}'
            date_col = f'last_{match_format.lower()}_{gender}_date'
            
            cursor.execute(f"""
                INSERT INTO team_current_elo (team_id, {col}, {date_col})
                VALUES (?, ?, ?)
                ON CONFLICT(team_id) DO UPDATE SET
                    {col} = excluded.{col},
                    {date_col} = excluded.{date_col},
                    updated_at = CURRENT_TIMESTAMP
            """, (team_id, new_rating, match_date))
        
        # Step 10: Check promotion triggers
        self.check_promotion_triggers(conn, team1_id, tier1, new_rating1, match_format, gender)
        self.check_promotion_triggers(conn, team2_id, tier2, new_rating2, match_format, gender)
        
        return new_rating1, new_rating2
    
    def check_promotion_triggers(
        self,
        conn,
        team_id: int,
        current_tier: int,
        current_elo: float,
        match_format: str,
        gender: str
    ):
        """
        Check if team should be flagged for tier review.
        
        Triggers:
        1. At ceiling for 6+ months
        2. Strong record vs higher tier (40%+ wins over 10+ matches)
        3. At floor for 6+ months (demotion)
        """
        cursor = conn.cursor()
        
        ceiling = self.TIER_CEILINGS[current_tier]
        floor = self.TIER_FLOORS[current_tier]
        
        # Trigger 1: Near ceiling
        if current_elo >= ceiling - 30:
            months_at_ceiling = self._count_months_at_threshold(
                conn, team_id, match_format, gender, ceiling - 30, above=True
            )
            
            if months_at_ceiling >= 6 and current_tier > 1:
                self._create_promotion_flag(
                    conn, team_id, match_format, gender,
                    current_tier, current_tier - 1,
                    f"At ceiling ({current_elo:.0f}) for {months_at_ceiling} months",
                    current_elo, months_at_ceiling, None
                )
        
        # Trigger 2: Strong vs higher tier
        if current_tier > 1:
            higher_tier = current_tier - 1
            record = self._get_cross_tier_record(
                conn, team_id, higher_tier, match_format, gender, lookback_months=12
            )
            
            if record['matches'] >= 10 and record['win_pct'] >= 0.40:
                self._create_promotion_flag(
                    conn, team_id, match_format, gender,
                    current_tier, higher_tier,
                    f"Strong vs Tier {higher_tier}: {record['wins']}-{record['losses']} ({record['win_pct']:.0%})",
                    current_elo, None, f"{record['wins']}-{record['losses']}"
                )
        
        # Trigger 3: At floor (demotion)
        if current_elo <= floor + 30:
            months_at_floor = self._count_months_at_threshold(
                conn, team_id, match_format, gender, floor + 30, above=False
            )
            
            if months_at_floor >= 6 and current_tier < 5:
                self._create_promotion_flag(
                    conn, team_id, match_format, gender,
                    current_tier, current_tier + 1,
                    f"At floor ({current_elo:.0f}) for {months_at_floor} months (demotion)",
                    current_elo, months_at_floor, None
                )
    
    def _count_months_at_threshold(
        self,
        conn,
        team_id: int,
        format_type: str,
        gender: str,
        threshold: float,
        above: bool
    ) -> int:
        """Count consecutive months team has been above/below threshold."""
        cursor = conn.cursor()
        
        operator = '>=' if above else '<='
        
        cursor.execute(f"""
            SELECT date, elo 
            FROM team_elo_history
            WHERE team_id = ? AND format = ? AND gender = ?
            ORDER BY date DESC
            LIMIT 180
        """, (team_id, format_type, gender))
        
        rows = cursor.fetchall()
        if not rows:
            return 0
        
        # Count consecutive months
        current_month = None
        months_count = 0
        
        for row in rows:
            date_str = row['date']
            elo = row['elo']
            
            if above and elo < threshold:
                break
            if not above and elo > threshold:
                break
            
            month = date_str[:7] if isinstance(date_str, str) else date_str.strftime('%Y-%m')
            
            if month != current_month:
                months_count += 1
                current_month = month
        
        return months_count
    
    def _get_cross_tier_record(
        self,
        conn,
        team_id: int,
        opponent_tier: int,
        format_type: str,
        gender: str,
        lookback_months: int = 12
    ) -> Dict:
        """Get team's record against teams from a specific tier."""
        cursor = conn.cursor()
        
        lookback_date = (datetime.now() - timedelta(days=lookback_months * 30)).date()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as matches,
                SUM(CASE WHEN m.winner_id = ? THEN 1 ELSE 0 END) as wins
            FROM matches m
            JOIN teams t1 ON m.team1_id = t1.team_id
            JOIN teams t2 ON m.team2_id = t2.team_id
            WHERE m.date >= ?
              AND m.match_type = ?
              AND m.gender = ?
              AND (
                  (m.team1_id = ? AND t2.tier = ?) OR
                  (m.team2_id = ? AND t1.tier = ?)
              )
        """, (team_id, lookback_date, format_type, gender, 
              team_id, opponent_tier, team_id, opponent_tier))
        
        row = cursor.fetchone()
        matches = row['matches'] if row else 0
        wins = row['wins'] if row and row['wins'] is not None else 0
        losses = matches - wins
        win_pct = wins / matches if matches > 0 else 0
        
        return {
            'matches': matches,
            'wins': wins,
            'losses': losses,
            'win_pct': win_pct
        }
    
    def _create_promotion_flag(
        self, conn, team_id, format_type, gender, current_tier, suggested_tier,
        reason, current_elo, months_at_ceiling, cross_tier_record
    ):
        """Insert or update promotion flag (UPSERT to avoid duplicates)."""
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO promotion_review_flags (
                team_id, format, gender, current_tier, suggested_tier,
                trigger_reason, current_elo, months_at_ceiling, cross_tier_record
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(team_id, format, gender, reviewed) 
            DO UPDATE SET
                suggested_tier = excluded.suggested_tier,
                trigger_reason = excluded.trigger_reason,
                current_elo = excluded.current_elo,
                flagged_date = CURRENT_DATE
        """, (team_id, format_type, gender, current_tier, suggested_tier,
              reason, current_elo, months_at_ceiling, cross_tier_record))
    
    def update_player_ratings_for_match(
        self,
        conn,
        match_id: int,
        match_format: str,
        gender: str,
        match_date: datetime
    ):
        """
        Update all player ratings for a match with tier-adjusted K-factors.
        
        NEW in V3: Player K-factors are adjusted based on opponent team tier.
        Performances against weaker teams give less ELO than against strong teams.
        """
        cursor = conn.cursor()
        
        # Get match info
        cursor.execute("""
            SELECT team1_id, team2_id FROM matches WHERE match_id = ?
        """, (match_id,))
        match_row = cursor.fetchone()
        if not match_row:
            return
        
        team1_id, team2_id = match_row['team1_id'], match_row['team2_id']
        
        # Get team tiers for K-factor adjustment
        cursor.execute("SELECT tier FROM teams WHERE team_id = ?", (team1_id,))
        tier1_row = cursor.fetchone()
        team1_tier = tier1_row['tier'] if tier1_row and tier1_row['tier'] else 3
        
        cursor.execute("SELECT tier FROM teams WHERE team_id = ?", (team2_id,))
        tier2_row = cursor.fetchone()
        team2_tier = tier2_row['tier'] if tier2_row and tier2_row['tier'] else 3
        
        # Get team ratings
        team1_elo = self.get_team_rating(conn, team1_id, match_format, gender, match_date)
        team2_elo = self.get_team_rating(conn, team2_id, match_format, gender, match_date)
        
        # Get all player stats with opponent tier info
        cursor.execute("""
            SELECT 
                pms.*,
                CASE WHEN pms.team_id = ? THEN ? ELSE ? END as opponent_elo,
                CASE WHEN pms.team_id = ? THEN ? ELSE ? END as opponent_tier
            FROM player_match_stats pms
            WHERE pms.match_id = ?
        """, (team1_id, team2_elo, team1_elo, team1_id, team2_tier, team1_tier, match_id))
        
        player_stats = cursor.fetchall()
        
        for stats in player_stats:
            player_id = stats['player_id']
            opponent_tier = stats['opponent_tier']
            
            # Get tier-adjusted K-factors
            tier_k_adjustment = self.PLAYER_TIER_K_ADJUSTMENT.get(opponent_tier, 0.5)
            adjusted_k_batting = self.k_factor_batting * tier_k_adjustment
            adjusted_k_bowling = self.k_factor_bowling * tier_k_adjustment
            
            # Get current ratings
            batting_elo = self.get_player_rating(
                conn, player_id, match_format, gender, 'batting', match_date
            )
            bowling_elo = self.get_player_rating(
                conn, player_id, match_format, gender, 'bowling', match_date
            )
            
            new_batting_elo = batting_elo
            new_bowling_elo = bowling_elo
            
            # Update batting ELO with tier-adjusted K-factor
            if stats['batting_innings'] > 0 and stats['balls_faced'] > 0:
                avg_sr = 130 if match_format == 'T20' else 85
                expected_runs = stats['balls_faced'] * (avg_sr / 100)
                opponent_adjustment = (stats['opponent_elo'] - self.initial_rating) / 400
                expected_runs *= (1 - opponent_adjustment * 0.1)
                
                performance = stats['runs_scored'] / max(expected_runs, 1)
                actual_score = min(1.0, performance / 2.0)
                expected_score = self.expected_score(batting_elo, stats['opponent_elo'])
                
                new_batting_elo = self.calculate_new_rating(
                    batting_elo, expected_score, actual_score, adjusted_k_batting
                )
            
            # Update bowling ELO with tier-adjusted K-factor
            if stats['overs_bowled'] and stats['overs_bowled'] > 0:
                avg_economy = 8.0 if match_format == 'T20' else 5.5
                opponent_adjustment = (stats['opponent_elo'] - self.initial_rating) / 400
                expected_economy = avg_economy * (1 + opponent_adjustment * 0.1)
                
                actual_economy = stats['runs_conceded'] / stats['overs_bowled']
                economy_score = max(0, min(1, (expected_economy - actual_economy + 4) / 8))
                
                expected_wickets = stats['overs_bowled'] / (4 if match_format == 'T20' else 8)
                wicket_score = min(1, stats['wickets_taken'] / max(expected_wickets, 0.5))
                
                actual_score = 0.6 * economy_score + 0.4 * wicket_score
                expected_score = self.expected_score(bowling_elo, stats['opponent_elo'])
                
                new_bowling_elo = self.calculate_new_rating(
                    bowling_elo, expected_score, actual_score, adjusted_k_bowling
                )
            
            # Calculate overall
            total_balls = stats['balls_faced'] + (stats['overs_bowled'] * 6 if stats['overs_bowled'] else 0)
            if total_balls > 0:
                bat_weight = stats['balls_faced'] / total_balls
                bowl_weight = 1 - bat_weight
            else:
                bat_weight = bowl_weight = 0.5
            
            new_overall_elo = (new_batting_elo * bat_weight + new_bowling_elo * bowl_weight)
            
            # Insert into history
            cursor.execute("""
                INSERT INTO player_elo_history (
                    player_id, date, match_id, format, gender,
                    batting_elo, bowling_elo, overall_elo,
                    batting_elo_change, bowling_elo_change, overall_elo_change
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                player_id, match_date, match_id, match_format, gender,
                new_batting_elo, new_bowling_elo, new_overall_elo,
                new_batting_elo - batting_elo,
                new_bowling_elo - bowling_elo,
                new_overall_elo - (batting_elo * bat_weight + bowling_elo * bowl_weight)
            ))
            
            # Update current ELO
            suffix = f'{match_format.lower()}_{gender}'
            date_col = f'last_{suffix}_date'
            
            cursor.execute(f"""
                INSERT INTO player_current_elo (
                    player_id,
                    batting_elo_{suffix}, bowling_elo_{suffix}, overall_elo_{suffix},
                    {date_col}
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(player_id) DO UPDATE SET
                    batting_elo_{suffix} = excluded.batting_elo_{suffix},
                    bowling_elo_{suffix} = excluded.bowling_elo_{suffix},
                    overall_elo_{suffix} = excluded.overall_elo_{suffix},
                    {date_col} = excluded.{date_col},
                    updated_at = CURRENT_TIMESTAMP
            """, (player_id, new_batting_elo, new_bowling_elo, new_overall_elo, match_date))
    
    def create_monthly_snapshots(self, conn, year_month: str, format_type: str, gender: str):
        """Create monthly ELO snapshots for a specific format/gender."""
        cursor = conn.cursor()
        
        year, month = map(int, year_month.split('-'))
        if month == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)
        last_day = next_month - timedelta(days=1)
        
        col = f'elo_{format_type.lower()}_{gender}'
        
        # Team snapshots
        cursor.execute(f"""
            INSERT INTO team_elo_history (team_id, date, format, gender, elo, is_monthly_snapshot)
            SELECT team_id, ?, ?, ?, {col}, TRUE
            FROM team_current_elo
            WHERE {col} != 1500
        """, (last_day, format_type, gender))
        
        # Player snapshots
        bat_col = f'batting_elo_{format_type.lower()}_{gender}'
        bowl_col = f'bowling_elo_{format_type.lower()}_{gender}'
        overall_col = f'overall_elo_{format_type.lower()}_{gender}'
        
        cursor.execute(f"""
            INSERT INTO player_elo_history (
                player_id, date, format, gender,
                batting_elo, bowling_elo, overall_elo,
                is_monthly_snapshot
            )
            SELECT 
                player_id, ?, ?, ?,
                {bat_col}, {bowl_col}, {overall_col},
                TRUE
            FROM player_current_elo
            WHERE {bat_col} != 1500 OR {bowl_col} != 1500
        """, (last_day, format_type, gender))


def calculate_all_elos_v3(force_recalculate: bool = False):
    """
    Calculate ELO ratings with tiered cross-pool system.
    
    Processes all matches chronologically with:
    - Tier-based K-factors
    - Cross-pool asymmetry
    - Prestige adjustments
    - Hybrid tournament classification
    - Promotion review triggers
    - Monthly snapshots
    """
    from tqdm import tqdm
    
    calculator = EloCalculatorV3()
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Get all matches chronologically
        cursor.execute("""
            SELECT 
                match_id, match_type, gender, date, 
                team1_id, team2_id, winner_id, event_name
            FROM matches
            ORDER BY date, match_id
        """)
        matches = cursor.fetchall()
        
        if not matches:
            logger.warning("No matches found")
            return
        
        logger.info(f"Calculating tiered ELO ratings for {len(matches)} matches...")
        
        current_months = {}  # (format, gender) -> current_month
        
        for match in tqdm(matches, desc="Calculating Tiered ELOs"):
            match_id = match['match_id']
            match_format = match['match_type']
            gender = match['gender']
            match_date = match['date']
            team1_id = match['team1_id']
            team2_id = match['team2_id']
            winner_id = match['winner_id']
            tournament_name = match['event_name'] or ''
            
            if isinstance(match_date, str):
                match_date = datetime.strptime(match_date, '%Y-%m-%d').date()
            
            key = (match_format, gender)
            year_month = match_date.strftime('%Y-%m')
            
            # Monthly snapshot
            if key in current_months and year_month != current_months[key]:
                calculator.create_monthly_snapshots(conn, current_months[key], match_format, gender)
            current_months[key] = year_month
            
            # Update team ratings (NEW: with tournament name)
            calculator.update_team_ratings(
                conn, match_id, team1_id, team2_id, winner_id,
                match_format, gender, match_date, tournament_name
            )
            
            # Update player ratings (unchanged from V2)
            calculator.update_player_ratings_for_match(
                conn, match_id, match_format, gender, match_date
            )
        
        # Final snapshots
        for (fmt, gen), month in current_months.items():
            calculator.create_monthly_snapshots(conn, month, fmt, gen)
        
        logger.info("Tiered ELO calculation complete!")
        
        # Print summary by tier
        print_tiered_summary(conn)


def print_tiered_summary(conn):
    """Print summary of tiered ELO calculations."""
    cursor = conn.cursor()
    
    print("\n" + "="*60)
    print("TIERED ELO CALCULATION SUMMARY")
    print("="*60)
    
    for fmt in ['T20', 'ODI']:
        for gen in ['male', 'female']:
            cursor.execute("""
                SELECT COUNT(*) FROM team_elo_history 
                WHERE format = ? AND gender = ? AND NOT is_monthly_snapshot
            """, (fmt, gen))
            count = cursor.fetchone()[0]
            print(f"{fmt} {gen}: {count} team rating updates")
    
    # Print promotion flags generated
    cursor.execute("SELECT COUNT(*) FROM promotion_review_flags WHERE reviewed = FALSE")
    flag_count = cursor.fetchone()[0]
    print(f"\n{flag_count} promotion review flags generated")


def print_rankings_v3(format_type: str = 'T20', gender: str = 'male', tier_filter: int = None, limit: int = 15):
    """Print current rankings for specific format, gender, and optionally tier."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        col = f'elo_{format_type.lower()}_{gender}'
        
        tier_clause = f"AND t.tier = {tier_filter}" if tier_filter else ""
        
        print(f"\n{'='*60}")
        print(f"TOP {limit} TEAM RANKINGS ({format_type} {gender.upper()})")
        if tier_filter:
            print(f"Tier {tier_filter} only")
        print('='*60)
        
        cursor.execute(f"""
            SELECT t.name, t.tier, e.{col} as elo
            FROM team_current_elo e
            JOIN teams t ON e.team_id = t.team_id
            WHERE e.{col} != 1500 {tier_clause}
            ORDER BY e.{col} DESC
            LIMIT ?
        """, (limit,))
        
        for i, row in enumerate(cursor.fetchall(), 1):
            print(f"{i:2}. [T{row['tier']}] {row['name']:30} {row['elo']:.0f}")


def main():
    """Main function to calculate tiered ELO ratings."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    import argparse
    parser = argparse.ArgumentParser(description='Calculate Tiered ELO ratings (V3)')
    parser.add_argument('--force', action='store_true', help='Force recalculation')
    parser.add_argument('--rankings', action='store_true', help='Print rankings')
    parser.add_argument('--format', default='T20', choices=['T20', 'ODI'])
    parser.add_argument('--gender', default='male', choices=['male', 'female'])
    parser.add_argument('--tier', type=int, choices=[1,2,3,4,5], help='Filter by tier')
    args = parser.parse_args()
    
    if args.rankings:
        print_rankings_v3(args.format, args.gender, args.tier)
    else:
        calculate_all_elos_v3(force_recalculate=args.force)
        
        # Print tier-separated rankings
        for fmt in ['T20', 'ODI']:
            for gen in ['male', 'female']:
                print_rankings_v3(fmt, gen, tier_filter=1, limit=10)
                print_rankings_v3(fmt, gen, tier_filter=2, limit=10)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

