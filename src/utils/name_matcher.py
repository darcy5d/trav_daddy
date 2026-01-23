"""
Smart name matching for ESPN abbreviated names to database full names.

Handles cases like:
- "NA McSweeney" → "Nathan McSweeney"
- "TP Alsop" → "Tom Alsop"
- "XC Bartlett" → "Xavier Bartlett"
"""

import re
from typing import Optional, Tuple
from difflib import SequenceMatcher


def expand_initials(abbreviated_name: str) -> list[str]:
    """
    Generate possible expansions of abbreviated names.
    
    Args:
        abbreviated_name: e.g. "NA McSweeney", "TP Alsop"
    
    Returns:
        List of possible full name patterns for matching
    """
    # Split into parts
    parts = abbreviated_name.strip().split()
    if not parts:
        return [abbreviated_name]
    
    # Check if first part looks like initials (all uppercase, no spaces)
    first_part = parts[0]
    if not re.match(r'^[A-Z]{1,3}$', first_part):
        # Not initials, return as-is
        return [abbreviated_name]
    
    # Extract initials and surname
    initials = list(first_part)
    surname_parts = parts[1:]
    surname = ' '.join(surname_parts)
    
    # Generate patterns:
    # 1. "% McSweeney" - any first name(s) ending with McSweeney
    # 2. "N% McSweeney" - first name starting with N
    # 3. "% A% McSweeney" - first name + middle name starting with A
    
    patterns = []
    
    if len(initials) == 1:
        # Single initial: "N McSweeney"
        patterns.append(f"{initials[0]}% {surname}")
    elif len(initials) == 2:
        # Two initials: "NA McSweeney"
        patterns.append(f"{initials[0]}% {initials[1]}% {surname}")
        patterns.append(f"{initials[0]}% {surname}")  # Sometimes middle name is omitted
    elif len(initials) >= 3:
        # Three+ initials: rare but handle it
        patterns.append(f"{initials[0]}% {surname}")
    
    # Also add a generic pattern for the surname
    patterns.append(f"% {surname}")
    
    return patterns


def calculate_name_similarity(name1: str, name2: str) -> float:
    """
    Calculate similarity between two names (0.0 to 1.0).
    
    Uses SequenceMatcher for fuzzy string matching.
    """
    return SequenceMatcher(None, name1.lower(), name2.lower()).ratio()


def match_abbreviated_name(
    abbreviated_name: str,
    candidate_names: list[Tuple[int, str]],
    threshold: float = 0.6
) -> Optional[Tuple[int, str, float]]:
    """
    Match an abbreviated ESPN name to a list of candidate full names.
    
    Args:
        abbreviated_name: e.g. "NA McSweeney"
        candidate_names: List of (player_id, full_name) tuples from database
        threshold: Minimum similarity score (0.0 to 1.0)
    
    Returns:
        (player_id, matched_name, similarity_score) or None if no match
    """
    if not candidate_names:
        return None
    
    # Extract surname from abbreviated name (last part)
    abbrev_parts = abbreviated_name.strip().split()
    if not abbrev_parts:
        return None
    
    abbrev_surname = abbrev_parts[-1].lower()
    
    # First pass: filter candidates by surname match
    surname_matches = []
    for player_id, full_name in candidate_names:
        full_parts = full_name.strip().split()
        if not full_parts:
            continue
        
        full_surname = full_parts[-1].lower()
        
        # Check if surnames match (exact or very similar)
        surname_similarity = calculate_name_similarity(abbrev_surname, full_surname)
        if surname_similarity >= 0.85:  # High threshold for surname
            surname_matches.append((player_id, full_name, surname_similarity))
    
    if not surname_matches:
        return None
    
    # Second pass: check initial matching for first/middle names
    best_match = None
    best_score = 0.0
    
    abbrev_parts = abbreviated_name.strip().split()
    first_part = abbrev_parts[0] if abbrev_parts else ""
    
    # Check if first part is initials
    is_initials = re.match(r'^[A-Z]{1,3}$', first_part)
    
    for player_id, full_name, surname_sim in surname_matches:
        full_parts = full_name.strip().split()
        
        if is_initials:
            # Match initials to first letters of full name parts
            initials = list(first_part)
            full_initials = [part[0].upper() for part in full_parts[:-1]]  # Exclude surname
            
            # Check if initials match
            if len(initials) <= len(full_initials):
                initials_match = all(
                    init == full_init 
                    for init, full_init in zip(initials, full_initials)
                )
                
                if initials_match:
                    # Perfect match: initials + surname
                    score = 1.0
                    if score > best_score:
                        best_score = score
                        best_match = (player_id, full_name, score)
                else:
                    # Partial match: just surname
                    score = surname_sim * 0.7  # Penalize for initial mismatch
                    if score > best_score and score >= threshold:
                        best_score = score
                        best_match = (player_id, full_name, score)
            else:
                # Just surname match
                score = surname_sim * 0.7
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = (player_id, full_name, score)
        else:
            # No initials, use full string similarity
            score = calculate_name_similarity(abbreviated_name, full_name)
            if score > best_score and score >= threshold:
                best_score = score
                best_match = (player_id, full_name, score)
    
    return best_match


def normalize_name(name: str) -> str:
    """
    Normalize a name for comparison.
    
    - Lowercase
    - Remove extra whitespace
    - Remove special characters
    """
    # Remove special characters except spaces and hyphens
    name = re.sub(r'[^\w\s\-]', '', name)
    # Normalize whitespace
    name = ' '.join(name.split())
    return name.lower()






