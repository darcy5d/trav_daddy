"""
Infer tier-appropriate default ELO for unmatched teams.

When a team cannot be matched to the database (e.g. Eastern Cape Iinyathi XI),
we default their ELO for simulation. Using 1500 (global average) is often wrong
for domestic/second-division competitions where both teams are below average.

This module infers a competition-appropriate default from:
- The matched opponent's tier (if one team is known)
- The series/tournament name (e.g. "CSA second division" -> tier 5)
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Tier initial ratings (from calculator_v3) - used when we infer tier from context
TIER_DEFAULT_ELO = {
    1: 1650,  # Elite full members
    2: 1550,  # Full members
    3: 1450,  # Top associates / premier franchises
    4: 1350,  # Regional domestic / associates
    5: 1250,  # Emerging / second division
}

# Fallback when we cannot infer tier
DEFAULT_FALLBACK = 1500.0


def infer_default_elo(
    series_name: Optional[str] = None,
    matched_opponent_team_id: Optional[int] = None,
    matched_opponent_tier: Optional[int] = None,
    format_type: str = "T20",
) -> float:
    """
    Infer a tier-appropriate default ELO for an unmatched team.

    Args:
        series_name: Tournament/series name (e.g. "CSA Second Division ODI")
        matched_opponent_team_id: Team ID of the matched opponent (for tier lookup)
        matched_opponent_tier: Pre-fetched tier of matched opponent (avoids DB call)
        format_type: 'T20' or 'ODI'

    Returns:
        Default ELO (e.g. 1250 for second division, 1350 for regional domestic)
    """
    inferred_tier = None

    # 1. Use matched opponent's tier if available (they're peers in same competition)
    if matched_opponent_tier is not None:
        inferred_tier = matched_opponent_tier
        # If series name suggests "second division", use one tier lower
        if series_name and _is_second_division(series_name):
            inferred_tier = min(5, matched_opponent_tier + 1)
        logger.info(
            f"[DEFAULT_ELO] Inferred from opponent tier {matched_opponent_tier}: "
            f"default ELO {TIER_DEFAULT_ELO.get(inferred_tier, DEFAULT_FALLBACK)}"
        )
        return float(TIER_DEFAULT_ELO.get(inferred_tier, DEFAULT_FALLBACK))

    # 2. Look up opponent tier from DB if we have team_id
    if matched_opponent_team_id is not None:
        try:
            from src.data.database import get_connection

            conn = get_connection()
            cur = conn.cursor()
            cur.execute("SELECT tier FROM teams WHERE team_id = ?", (matched_opponent_team_id,))
            row = cur.fetchone()
            conn.close()
            if row and row[0] is not None:
                return infer_default_elo(
                    series_name=series_name,
                    matched_opponent_tier=int(row[0]),
                    format_type=format_type,
                )
        except Exception as e:
            logger.warning(f"[DEFAULT_ELO] Could not look up opponent tier: {e}")

    # 3. Infer from series name patterns
    if series_name:
        sn_lower = series_name.lower()
        if _is_second_division(sn_lower):
            inferred_tier = 5  # Second division / emerging
        elif any(x in sn_lower for x in ["csa", "domestic", "provincial", "division 1"]):
            inferred_tier = 4  # Regional domestic first division
        elif any(x in sn_lower for x in ["ipl", "bbl", "cpl", "psl", "sa20", "hundred"]):
            inferred_tier = 3  # Premier franchise
        elif any(x in sn_lower for x in ["world cup", "champions trophy", "asia cup"]):
            inferred_tier = 2  # Major international
        if inferred_tier is not None:
            default = TIER_DEFAULT_ELO.get(inferred_tier, DEFAULT_FALLBACK)
            logger.info(
                f"[DEFAULT_ELO] Inferred from series '{series_name[:50]}...': "
                f"tier {inferred_tier} -> default ELO {default}"
            )
            return float(default)

    logger.info(f"[DEFAULT_ELO] No context - using fallback {DEFAULT_FALLBACK}")
    return DEFAULT_FALLBACK


def _is_second_division(name: str) -> bool:
    """Check if series name suggests second division / lower tier."""
    if not name:
        return False
    n = name.lower()
    return any(
        x in n
        for x in [
            "second division",
            "2nd division",
            "division 2",
            "div 2",
            "division two",
            "second div",
            "2nd div",
        ]
    )
