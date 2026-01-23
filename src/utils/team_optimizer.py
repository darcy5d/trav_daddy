"""
Team Selection Optimizer and Validator.

Implements constraint-based team selection for cricket XIs.
Ensures balanced teams with proper role distribution.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class TeamConstraints:
    """Constraints for valid cricket team selection."""
    min_keepers: int = 1
    max_keepers: int = 1
    min_specialist_bowlers: int = 4  # Pure bowlers (not all-rounders)
    min_bowling_options: int = 5  # Total players who can bowl
    max_batters: int = 7  # Pure batters (not all-rounders or keepers)
    total_players: int = 11


@dataclass
class TeamSelection:
    """Result of team validation with detailed breakdown."""
    keepers: List[Dict]
    batters: List[Dict]
    allrounders: List[Dict]
    bowlers: List[Dict]
    total_players: int
    total_bowling_options: int
    is_valid: bool
    validation_errors: List[str]
    warnings: List[str]


def validate_team(players: List[Dict], constraints: Optional[TeamConstraints] = None) -> TeamSelection:
    """
    Validate a team of players meets cricket team constraints.
    
    Args:
        players: List of player dicts with 'role_category' and 'is_bowling_option' fields
        constraints: TeamConstraints object (uses defaults if None)
    
    Returns:
        TeamSelection object with validation results
    """
    if constraints is None:
        constraints = TeamConstraints()
    
    # Group players by role category
    keepers = [p for p in players if p.get('role_category') == 'KEEPER']
    batters = [p for p in players if p.get('role_category') == 'BATTER']
    allrounders = [p for p in players if p.get('role_category') == 'ALLROUNDER']
    bowlers = [p for p in players if p.get('role_category') == 'BOWLER']
    
    # Count totals
    total_players = len(players)
    bowling_options = len([p for p in players if p.get('is_bowling_option', False)])
    specialist_bowlers = len(bowlers)
    
    # Validation checks
    errors = []
    warnings = []
    
    # Check total players
    if total_players != constraints.total_players:
        errors.append(f"Must select exactly {constraints.total_players} players (currently {total_players})")
    
    # Check keepers
    if len(keepers) < constraints.min_keepers:
        errors.append(f"Must have at least {constraints.min_keepers} wicketkeeper (currently {len(keepers)})")
    elif len(keepers) > constraints.max_keepers:
        errors.append(f"Cannot have more than {constraints.max_keepers} wicketkeeper (currently {len(keepers)})")
    
    # Check specialist bowlers
    if specialist_bowlers < constraints.min_specialist_bowlers:
        errors.append(f"Must have at least {constraints.min_specialist_bowlers} specialist bowlers (currently {specialist_bowlers})")
    
    # Check total bowling options
    if bowling_options < constraints.min_bowling_options:
        errors.append(f"Must have at least {constraints.min_bowling_options} bowling options (currently {bowling_options})")
    
    # Check batters (warning only)
    if len(batters) > constraints.max_batters:
        warnings.append(f"Team has {len(batters)} specialist batters (recommended max: {constraints.max_batters})")
    
    # Check balance (warning only)
    if len(batters) < 3 and len(allrounders) == 0:
        warnings.append("Team may lack batting depth (only {len(batters)} specialist batters)")
    
    is_valid = len(errors) == 0
    
    return TeamSelection(
        keepers=keepers,
        batters=batters,
        allrounders=allrounders,
        bowlers=bowlers,
        total_players=total_players,
        total_bowling_options=bowling_options,
        is_valid=is_valid,
        validation_errors=errors,
        warnings=warnings
    )


def optimize_xi_from_squad(
    squad: List[Dict], 
    constraints: Optional[TeamConstraints] = None
) -> List[Dict]:
    """
    Select best XI from a squad using constraints and performance metrics.
    
    Algorithm:
    1. Select best keeper (by batting avg or recent form)
    2. Select 4 best specialist bowlers (by wickets/avg)
    3. Fill remaining spots with best batters/all-rounders
    4. Ensure at least 5 bowling options total
    
    Args:
        squad: List of player dicts with stats and role_category
        constraints: TeamConstraints object (uses defaults if None)
    
    Returns:
        List of 11 player dicts representing optimized XI
    """
    if constraints is None:
        constraints = TeamConstraints()
    
    logger.info(f"Optimizing XI from squad of {len(squad)} players")
    
    # Group squad by role
    keepers = [p for p in squad if p.get('role_category') == 'KEEPER']
    batters = [p for p in squad if p.get('role_category') == 'BATTER']
    allrounders = [p for p in squad if p.get('role_category') == 'ALLROUNDER']
    bowlers = [p for p in squad if p.get('role_category') == 'BOWLER']
    
    selected = []
    
    # Step 1: Select best keeper
    if keepers:
        # Sort by batting average (keepers who can bat are valuable)
        keepers_sorted = sorted(
            keepers, 
            key=lambda p: p.get('stats', {}).get('runs', 0), 
            reverse=True
        )
        selected.append(keepers_sorted[0])
        logger.info(f"Selected keeper: {keepers_sorted[0]['name']}")
    else:
        logger.warning("No wicketkeeper available in squad!")
    
    # Step 2: Select 4 best specialist bowlers
    bowlers_sorted = sorted(
        bowlers,
        key=lambda p: (
            p.get('stats', {}).get('wickets', 0),
            -p.get('stats', {}).get('runs', 0)  # Negative for descending
        ),
        reverse=True
    )
    
    num_bowlers_to_select = min(constraints.min_specialist_bowlers, len(bowlers_sorted))
    selected.extend(bowlers_sorted[:num_bowlers_to_select])
    logger.info(f"Selected {num_bowlers_to_select} specialist bowlers")
    
    # Step 3: Fill remaining spots with batters and all-rounders
    remaining_spots = constraints.total_players - len(selected)
    
    # Combine batters and all-rounders, sort by runs scored
    batters_and_allrounders = batters + allrounders
    batters_sorted = sorted(
        batters_and_allrounders,
        key=lambda p: p.get('stats', {}).get('runs', 0),
        reverse=True
    )
    
    # Prioritize all-rounders if we need more bowling options
    current_bowling_options = len([p for p in selected if p.get('is_bowling_option', False)])
    
    if current_bowling_options < constraints.min_bowling_options:
        # Need more bowling options - prioritize all-rounders
        bowling_allrounders = [p for p in allrounders if p.get('is_bowling_option', False)]
        bowling_allrounders_sorted = sorted(
            bowling_allrounders,
            key=lambda p: (
                p.get('stats', {}).get('wickets', 0),
                p.get('stats', {}).get('runs', 0)
            ),
            reverse=True
        )
        
        # Add bowling all-rounders first
        for ar in bowling_allrounders_sorted:
            if len(selected) >= constraints.total_players:
                break
            if ar not in selected:
                selected.append(ar)
                current_bowling_options += 1
                logger.info(f"Selected bowling all-rounder: {ar['name']}")
                if current_bowling_options >= constraints.min_bowling_options:
                    break
    
    # Fill remaining spots with best batters
    for player in batters_sorted:
        if len(selected) >= constraints.total_players:
            break
        if player not in selected:
            selected.append(player)
            logger.info(f"Selected batter/all-rounder: {player['name']}")
    
    # If still short, add any remaining players
    if len(selected) < constraints.total_players:
        remaining_players = [p for p in squad if p not in selected]
        remaining_sorted = sorted(
            remaining_players,
            key=lambda p: p.get('stats', {}).get('matches', 0),
            reverse=True
        )
        for player in remaining_sorted:
            if len(selected) >= constraints.total_players:
                break
            selected.append(player)
            logger.info(f"Selected additional player: {player['name']}")
    
    # Ensure we have exactly 11 players
    if len(selected) < constraints.total_players:
        logger.warning(f"Only selected {len(selected)} players, need {constraints.total_players}. Adding best remaining players.")
        remaining = [p for p in squad if p not in selected]
        remaining_sorted = sorted(
            remaining,
            key=lambda p: (
                p.get('stats', {}).get('matches', 0),
                p.get('stats', {}).get('runs', 0) + p.get('stats', {}).get('wickets', 0)
            ),
            reverse=True
        )
        for player in remaining_sorted:
            if len(selected) >= constraints.total_players:
                break
            selected.append(player)
            logger.info(f"Added remaining player to reach 11: {player['name']}")
    
    # If somehow we have more than 11, trim to best 11
    if len(selected) > constraints.total_players:
        logger.warning(f"Selected {len(selected)} players, trimming to {constraints.total_players}")
        # Keep the first 11 (already sorted by quality)
        selected = selected[:constraints.total_players]
    
    logger.info(f"Optimization complete: selected {len(selected)} players")
    
    # Validate the selection
    validation = validate_team(selected, constraints)
    if not validation.is_valid:
        logger.warning(f"Optimized team has validation errors: {validation.validation_errors}")
    
    return selected


def get_team_balance_score(players: List[Dict]) -> Dict[str, any]:
    """
    Calculate a balance score for a team.
    
    Returns metrics like:
    - batting_depth: How many players can bat
    - bowling_variety: Mix of pace/spin (if available)
    - all_rounder_count: Number of all-rounders
    - balance_score: Overall score (0-100)
    """
    keepers = len([p for p in players if p.get('role_category') == 'KEEPER'])
    batters = len([p for p in players if p.get('role_category') == 'BATTER'])
    allrounders = len([p for p in players if p.get('role_category') == 'ALLROUNDER'])
    bowlers = len([p for p in players if p.get('role_category') == 'BOWLER'])
    bowling_options = len([p for p in players if p.get('is_bowling_option', False)])
    
    # Simple balance score
    score = 0
    
    # Ideal: 1 keeper, 4-5 batters, 1-2 all-rounders, 4-5 bowlers
    if keepers == 1:
        score += 20
    if 4 <= batters <= 5:
        score += 20
    if 1 <= allrounders <= 2:
        score += 20
    if 4 <= bowlers <= 5:
        score += 20
    if bowling_options >= 5:
        score += 20
    
    return {
        'keepers': keepers,
        'batters': batters,
        'allrounders': allrounders,
        'bowlers': bowlers,
        'bowling_options': bowling_options,
        'balance_score': score
    }


