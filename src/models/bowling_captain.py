"""
Smart Captain Bowling Selection System.

Implements realistic T20 bowling rules:
- Max 4 overs per bowler
- Intelligent bowler selection based on:
  - Phase of innings (powerplay, middle, death)
  - Batter weakness (H2H data)
  - Bowling ELO
  - Match situation
"""

import logging
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.matchups import get_matchup_db, MatchupStats

logger = logging.getLogger(__name__)


class InningsPhase(Enum):
    """Phases of a T20 innings."""
    POWERPLAY = "powerplay"   # Overs 1-6
    MIDDLE = "middle"         # Overs 7-15
    DEATH = "death"           # Overs 16-20


def get_innings_phase(over: int) -> InningsPhase:
    """Determine phase based on over number (1-indexed)."""
    if over <= 6:
        return InningsPhase.POWERPLAY
    elif over <= 15:
        return InningsPhase.MIDDLE
    else:
        return InningsPhase.DEATH


@dataclass
class Bowler:
    """Represents a bowler in the simulation."""
    player_id: int
    name: str
    bowling_elo: float = 1500.0
    is_spinner: bool = False
    is_death_specialist: bool = False
    is_powerplay_specialist: bool = False
    
    # Categorization based on economy rates (optional)
    economy_powerplay: float = 8.0
    economy_middle: float = 7.5
    economy_death: float = 9.0


@dataclass
class BowlingTracker:
    """
    Tracks bowling overs and enforces T20 maximum.
    
    T20 Rules:
    - Each bowler can bowl max 4 overs
    - Team needs at least 5 bowlers to complete 20 overs
    """
    
    bowlers: List[Bowler]
    max_overs: int = 4  # T20 rule
    overs_bowled: Dict[int, int] = field(default_factory=dict)
    balls_in_current_over: Dict[int, int] = field(default_factory=dict)
    
    def __post_init__(self):
        # Initialize tracking for all bowlers
        for bowler in self.bowlers:
            self.overs_bowled[bowler.player_id] = 0
            self.balls_in_current_over[bowler.player_id] = 0
    
    def get_available_bowlers(self) -> List[Bowler]:
        """Get bowlers who haven't reached their max overs."""
        return [
            b for b in self.bowlers
            if self.overs_bowled[b.player_id] < self.max_overs
        ]
    
    def record_ball(self, bowler_id: int, is_legal: bool = True):
        """
        Record a delivery bowled.
        
        Args:
            bowler_id: ID of the bowler
            is_legal: True if legal delivery (not wide/no-ball)
        """
        if is_legal:
            self.balls_in_current_over[bowler_id] += 1
            
            # Check if over is complete
            if self.balls_in_current_over[bowler_id] >= 6:
                self.overs_bowled[bowler_id] += 1
                self.balls_in_current_over[bowler_id] = 0
    
    def complete_over(self, bowler_id: int):
        """Mark an over as complete for a bowler."""
        self.overs_bowled[bowler_id] += 1
        self.balls_in_current_over[bowler_id] = 0
    
    def get_overs_remaining(self, bowler_id: int) -> int:
        """Get number of overs remaining for a bowler."""
        return self.max_overs - self.overs_bowled[bowler_id]
    
    def get_total_overs_available(self) -> int:
        """Get total overs available from all bowlers."""
        return sum(
            self.max_overs - self.overs_bowled[b.player_id]
            for b in self.bowlers
        )
    
    def can_complete_innings(self, overs_remaining: int) -> bool:
        """Check if we have enough bowling to complete the innings."""
        return self.get_total_overs_available() >= overs_remaining


@dataclass
class MatchState:
    """Current state of the match for captain decisions."""
    total_runs: int = 0
    wickets: int = 0
    overs_completed: int = 0
    balls_in_over: int = 0
    target: Optional[int] = None  # None if batting first
    run_rate: float = 0.0
    required_rate: Optional[float] = None
    
    @property
    def is_chasing(self) -> bool:
        return self.target is not None
    
    @property
    def is_under_pressure(self) -> bool:
        """Match situation is tense."""
        if self.is_chasing:
            return self.required_rate and self.required_rate > 10
        else:
            # Batting first, lost several wickets
            return self.wickets >= 5 and self.overs_completed < 15


class SmartCaptain:
    """
    Smart bowler selection logic.
    
    Considers:
    - Phase of innings
    - Bowler's record against current batters
    - Bowling ELO
    - Overs remaining (strategy)
    - Match situation
    """
    
    def __init__(
        self,
        bowling_attack: List[Bowler],
        format_type: str = 'T20',
        gender: str = 'male'
    ):
        self.tracker = BowlingTracker(bowlers=bowling_attack)
        self.matchup_db = get_matchup_db(format_type, gender)
        self.format_type = format_type
        self.last_bowler_id: Optional[int] = None
        
    def select_bowler(
        self,
        current_batter_ids: List[int],
        match_state: MatchState,
        current_over: int
    ) -> Bowler:
        """
        Select the best bowler for the current situation.
        
        Args:
            current_batter_ids: IDs of batters currently at the crease
            match_state: Current match state
            current_over: Current over number (1-indexed)
        
        Returns:
            Selected bowler
        """
        available = self.tracker.get_available_bowlers()
        
        if not available:
            raise ValueError("No bowlers available - should not happen in valid T20!")
        
        # Can't bowl consecutive overs
        if self.last_bowler_id:
            available = [b for b in available if b.player_id != self.last_bowler_id]
            
            # Edge case: only one bowler left with overs
            if not available:
                available = self.tracker.get_available_bowlers()
        
        phase = get_innings_phase(current_over)
        
        # Score each available bowler
        bowler_scores = []
        
        for bowler in available:
            score = self._calculate_bowler_score(
                bowler,
                current_batter_ids,
                phase,
                match_state
            )
            bowler_scores.append((bowler, score))
        
        # Select bowler using weighted random (higher score = more likely)
        selected = self._weighted_selection(bowler_scores)
        
        return selected
    
    def _calculate_bowler_score(
        self,
        bowler: Bowler,
        batter_ids: List[int],
        phase: InningsPhase,
        match_state: MatchState
    ) -> float:
        """
        Calculate a score for how suitable a bowler is.
        
        Higher score = better choice.
        """
        score = bowler.bowling_elo  # Base score from ELO
        
        # === H2H BONUS ===
        # If bowler has good record against current batters
        for batter_id in batter_ids:
            h2h = self.matchup_db.get_matchup(batter_id, bowler.player_id)
            
            if h2h and h2h.has_sufficient_data:
                # Bonus for high dismissal rate
                if h2h.dismissal_rate > 0.08:  # >8% dismissal rate
                    score += 100  # Strong bonus
                elif h2h.dismissal_rate > 0.05:  # >5%
                    score += 50
                
                # Bonus for low strike rate against
                if h2h.strike_rate < 100:
                    score += 30
                elif h2h.strike_rate < 120:
                    score += 15
                
                # Penalty for being dominated
                if h2h.strike_rate > 160:
                    score -= 50
        
        # === PHASE BONUSES ===
        if phase == InningsPhase.DEATH:
            if bowler.is_death_specialist:
                score += 80
            # Penalize high death economy bowlers
            if bowler.economy_death > 10:
                score -= 30
                
        elif phase == InningsPhase.POWERPLAY:
            if bowler.is_powerplay_specialist:
                score += 60
            # Spinners less effective in powerplay (generally)
            if bowler.is_spinner:
                score -= 20
                
        elif phase == InningsPhase.MIDDLE:
            # Spinners shine in middle overs
            if bowler.is_spinner:
                score += 40
        
        # === STRATEGIC OVERS MANAGEMENT ===
        overs_left = self.tracker.get_overs_remaining(bowler.player_id)
        
        # Save death specialists for death overs
        if phase != InningsPhase.DEATH and bowler.is_death_specialist:
            if overs_left <= 2:
                score -= 40  # Save them for death
        
        # === MATCH SITUATION ===
        if match_state.is_under_pressure:
            # Under pressure, prefer experienced/higher ELO bowlers
            if bowler.bowling_elo > 1600:
                score += 30
        
        return score
    
    def _weighted_selection(
        self,
        bowler_scores: List[Tuple[Bowler, float]]
    ) -> Bowler:
        """
        Select bowler using weighted probability.
        
        Not deterministic to simulate real captain decisions.
        Higher scored bowlers have higher probability but not guaranteed.
        """
        if not bowler_scores:
            raise ValueError("No bowlers to select from!")
        
        # Normalize scores to probabilities
        min_score = min(s for _, s in bowler_scores)
        
        # Shift scores to be positive
        adjusted = [(b, s - min_score + 100) for b, s in bowler_scores]
        total = sum(s for _, s in adjusted)
        
        # Random selection weighted by score
        r = random.random() * total
        cumulative = 0
        
        for bowler, score in adjusted:
            cumulative += score
            if r <= cumulative:
                return bowler
        
        # Fallback (shouldn't reach here)
        return bowler_scores[0][0]
    
    def record_over_complete(self, bowler_id: int):
        """Record that a bowler completed an over."""
        self.tracker.complete_over(bowler_id)
        self.last_bowler_id = bowler_id
    
    def record_ball(self, bowler_id: int, is_legal: bool = True):
        """Record a single delivery."""
        self.tracker.record_ball(bowler_id, is_legal)


def create_bowling_attack_from_db(
    team_id: int,
    match_date: str,
    format_type: str = 'T20',
    gender: str = 'male',
    num_bowlers: int = 5
) -> List[Bowler]:
    """
    Create bowling attack from database for a team.
    
    Uses player ELO ratings and assigns specialist roles based on stats.
    """
    from src.data.database import get_connection
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get bowlers from the team (simplified - uses bowling ELO)
    # In production, you'd want actual squad selection
    
    cursor.execute("""
        SELECT DISTINCT p.player_id, p.name
        FROM players p
        JOIN player_match_stats pms ON p.player_id = pms.player_id
        JOIN matches m ON pms.match_id = m.match_id
        WHERE m.match_type = ? AND m.gender = ?
        AND (m.team1_id = ? OR m.team2_id = ?)
        AND m.date <= ?
        AND pms.overs_bowled > 0
        GROUP BY p.player_id
        ORDER BY SUM(pms.wickets_taken) DESC
        LIMIT ?
    """, (format_type, gender, team_id, team_id, match_date, num_bowlers))
    
    rows = cursor.fetchall()
    
    # Get bowling ELOs
    bowlers = []
    elo_col = f'bowling_elo_{format_type.lower()}_{gender}'
    
    for row in rows:
        # Try to get current ELO
        cursor.execute(f"""
            SELECT {elo_col} FROM player_current_elo 
            WHERE player_id = ?
        """, (row['player_id'],))
        elo_row = cursor.fetchone()
        
        bowling_elo = elo_row[0] if elo_row and elo_row[0] else 1500.0
        
        bowler = Bowler(
            player_id=row['player_id'],
            name=row['name'],
            bowling_elo=bowling_elo
        )
        bowlers.append(bowler)
    
    conn.close()
    
    # Ensure we have at least 5 bowlers
    if len(bowlers) < 5:
        logger.warning(f"Only found {len(bowlers)} bowlers for team {team_id}, padding with defaults")
        while len(bowlers) < 5:
            bowlers.append(Bowler(
                player_id=-len(bowlers),
                name=f"Bowler {len(bowlers)+1}",
                bowling_elo=1450.0
            ))
    
    return bowlers


def demo_captain_selection():
    """Demonstrate the smart captain bowler selection."""
    print("=" * 70)
    print("SMART CAPTAIN BOWLER SELECTION DEMO")
    print("=" * 70)
    
    # Create sample bowling attack
    bowlers = [
        Bowler(1, "Jasprit Bumrah", 1750, is_death_specialist=True),
        Bowler(2, "Mohammed Shami", 1680, is_powerplay_specialist=True),
        Bowler(3, "Ravindra Jadeja", 1620, is_spinner=True),
        Bowler(4, "Yuzvendra Chahal", 1640, is_spinner=True),
        Bowler(5, "Hardik Pandya", 1550),
    ]
    
    captain = SmartCaptain(bowlers)
    
    # Simulate bowling selections across an innings
    batter_ids = [100, 101]  # Placeholder batter IDs
    
    print("\nSimulating bowler selection for 20 overs:\n")
    print(f"{'Over':<6} {'Phase':<12} {'Selected':<20} {'Overs Remaining':<20}")
    print("-" * 60)
    
    for over in range(1, 21):
        match_state = MatchState(
            total_runs=over * 8,  # ~8 runs per over
            wickets=min(over // 4, 9),
            overs_completed=over - 1
        )
        
        selected = captain.select_bowler(batter_ids, match_state, over)
        phase = get_innings_phase(over)
        
        # Record over complete
        captain.record_over_complete(selected.player_id)
        
        overs_remaining = ", ".join(
            f"{b.name[:10]}:{captain.tracker.get_overs_remaining(b.player_id)}"
            for b in bowlers
        )
        
        print(f"{over:<6} {phase.value:<12} {selected.name:<20} ...")
    
    print("\n" + "=" * 70)
    print("FINAL BOWLING FIGURES")
    print("=" * 70)
    for bowler in bowlers:
        overs = captain.tracker.overs_bowled[bowler.player_id]
        print(f"  {bowler.name:<20}: {overs} overs")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    demo_captain_selection()

