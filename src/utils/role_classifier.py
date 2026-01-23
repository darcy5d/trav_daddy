"""
Player Role Classification System.

Infers cricket player roles from statistics and ESPN data.
Used for intelligent team selection and UI organization.
"""

from enum import Enum
from typing import Dict, List, Optional


class PlayerRole(Enum):
    """Cricket player roles for team selection."""
    WICKETKEEPER = "Wicketkeeper"
    WICKETKEEPER_BATTER = "Wicketkeeper Batter"
    TOP_ORDER_BATTER = "Top Order Batter"
    MIDDLE_ORDER_BATTER = "Middle Order Batter"
    BATTING_ALLROUNDER = "Batting Allrounder"
    BOWLING_ALLROUNDER = "Bowling Allrounder"
    FAST_BOWLER = "Fast Bowler"
    SPIN_BOWLER = "Spinner"
    SPECIALIST_BOWLER = "Specialist Bowler"


class RoleCategory(Enum):
    """High-level role categories for team selection UI."""
    KEEPER = "KEEPER"
    BATTER = "BATTER"
    ALLROUNDER = "ALLROUNDER"
    BOWLER = "BOWLER"


def infer_role_from_stats(player_stats: Dict) -> PlayerRole:
    """
    Infer player role from career statistics.
    
    Args:
        player_stats: Dictionary with keys:
            - total_matches: int
            - runs_scored: int
            - balls_faced: int
            - overs_bowled: float
            - wickets_taken: int
            - stumpings: int
            - avg_batting_position: float (optional)
            - batting_avg: float (optional)
            - bowling_avg: float (optional)
    
    Returns:
        PlayerRole enum value
    """
    matches = player_stats.get('total_matches', 1)
    runs = player_stats.get('runs_scored', 0)
    balls_faced = player_stats.get('balls_faced', 1)
    overs = player_stats.get('overs_bowled', 0)
    wickets = player_stats.get('wickets_taken', 0)
    stumpings = player_stats.get('stumpings', 0)
    avg_position = player_stats.get('avg_batting_position', 5.5)
    
    # Calculate per-match metrics
    runs_per_match = runs / matches if matches > 0 else 0
    overs_per_match = overs / matches if matches > 0 else 0
    wickets_per_match = wickets / matches if matches > 0 else 0
    
    # Wicketkeeper detection (stumpings is the key indicator)
    if stumpings > 0:
        # If they also bat well, they're a batting keeper
        if runs_per_match > 15:
            return PlayerRole.WICKETKEEPER_BATTER
        return PlayerRole.WICKETKEEPER
    
    # Determine primary skill
    is_bowler = overs_per_match > 2.0  # Bowls 2+ overs per match
    is_batter = runs_per_match > 10  # Scores 10+ runs per match on average
    
    # All-rounder detection
    if is_bowler and is_batter:
        # Decide primary skill by contribution
        batting_score = runs_per_match
        bowling_score = wickets_per_match * 20  # Weight wickets heavily
        
        if batting_score > bowling_score * 1.5:
            return PlayerRole.BATTING_ALLROUNDER
        else:
            return PlayerRole.BOWLING_ALLROUNDER
    
    # Specialist bowler
    if is_bowler:
        # Distinguish fast vs spin (not possible from stats alone, default to specialist)
        if wickets_per_match > 1.0:
            return PlayerRole.SPECIALIST_BOWLER
        return PlayerRole.FAST_BOWLER  # Default assumption
    
    # Specialist batter
    if is_batter:
        # Top order if batting position < 4, otherwise middle order
        if avg_position < 4.0:
            return PlayerRole.TOP_ORDER_BATTER
        else:
            return PlayerRole.MIDDLE_ORDER_BATTER
    
    # Default: middle order batter (most common fallback)
    return PlayerRole.MIDDLE_ORDER_BATTER


def infer_role_from_espn(playing_roles: List[str]) -> PlayerRole:
    """
    Map ESPN's playing_roles to our PlayerRole enum.
    
    ESPN uses strings like: 'bowler', 'wicketkeeper batter', 
    'allrounder', 'batter', 'batting allrounder', 'bowling allrounder'
    
    Args:
        playing_roles: List of role strings from ESPN
    
    Returns:
        PlayerRole enum value
    """
    if not playing_roles:
        return PlayerRole.MIDDLE_ORDER_BATTER  # Default
    
    # Convert to lowercase for matching
    roles_lower = [r.lower() for r in playing_roles]
    roles_str = ' '.join(roles_lower)
    
    # Wicketkeeper
    if 'wicketkeeper' in roles_str or 'wk' in roles_str:
        return PlayerRole.WICKETKEEPER_BATTER
    
    # All-rounders
    if 'batting allrounder' in roles_str or 'batting all-rounder' in roles_str:
        return PlayerRole.BATTING_ALLROUNDER
    if 'bowling allrounder' in roles_str or 'bowling all-rounder' in roles_str:
        return PlayerRole.BOWLING_ALLROUNDER
    if 'allrounder' in roles_str or 'all-rounder' in roles_str:
        # Generic all-rounder, default to bowling all-rounder
        return PlayerRole.BOWLING_ALLROUNDER
    
    # Bowlers
    if 'bowler' in roles_str:
        # Try to distinguish fast vs spin (not always clear from ESPN)
        if 'spin' in roles_str:
            return PlayerRole.SPIN_BOWLER
        return PlayerRole.SPECIALIST_BOWLER
    
    # Batters (default)
    if 'batter' in roles_str or 'batsman' in roles_str:
        # Can't distinguish top vs middle order from ESPN alone
        return PlayerRole.TOP_ORDER_BATTER
    
    # Default fallback
    return PlayerRole.MIDDLE_ORDER_BATTER


def categorize_role(role: PlayerRole) -> RoleCategory:
    """
    Group detailed roles into high-level categories for UI.
    
    Args:
        role: PlayerRole enum value
    
    Returns:
        RoleCategory enum value
    """
    keeper_roles = {PlayerRole.WICKETKEEPER, PlayerRole.WICKETKEEPER_BATTER}
    batter_roles = {PlayerRole.TOP_ORDER_BATTER, PlayerRole.MIDDLE_ORDER_BATTER}
    allrounder_roles = {PlayerRole.BATTING_ALLROUNDER, PlayerRole.BOWLING_ALLROUNDER}
    bowler_roles = {PlayerRole.FAST_BOWLER, PlayerRole.SPIN_BOWLER, PlayerRole.SPECIALIST_BOWLER}
    
    if role in keeper_roles:
        return RoleCategory.KEEPER
    elif role in batter_roles:
        return RoleCategory.BATTER
    elif role in allrounder_roles:
        return RoleCategory.ALLROUNDER
    elif role in bowler_roles:
        return RoleCategory.BOWLER
    else:
        return RoleCategory.BATTER  # Default fallback


def get_role_display_name(role: PlayerRole) -> str:
    """Get user-friendly display name for a role."""
    return role.value


def is_bowling_option(role: PlayerRole) -> bool:
    """Check if a player can be counted as a bowling option."""
    bowling_roles = {
        PlayerRole.BOWLING_ALLROUNDER,
        PlayerRole.FAST_BOWLER,
        PlayerRole.SPIN_BOWLER,
        PlayerRole.SPECIALIST_BOWLER
    }
    return role in bowling_roles






