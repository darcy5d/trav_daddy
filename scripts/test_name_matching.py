#!/usr/bin/env python3
"""
Test player name matching to diagnose why CREX players aren't being matched.

This script:
1. Shows sample players from the database
2. Tests name matching for common player names
3. Identifies potential matching issues

Run from project root:
    python scripts/test_name_matching.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.database import get_connection
from src.features.name_matcher import PlayerNameMatcher


def get_sample_players(gender='male', limit=20):
    """Get sample players from database."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT DISTINCT p.player_id, p.name, t.name as team_name
        FROM players p
        JOIN player_match_stats pms ON p.player_id = pms.player_id
        JOIN teams t ON pms.team_id = t.team_id
        JOIN matches m ON pms.match_id = m.match_id
        WHERE m.match_type = 'T20' AND m.gender = ?
        ORDER BY p.player_id
        LIMIT ?
    """, (gender, limit))
    
    players = cursor.fetchall()
    conn.close()
    return players


def get_teams_with_player_counts(gender='male'):
    """Get teams and their player counts."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT t.name, COUNT(DISTINCT pms.player_id) as player_count
        FROM teams t
        JOIN matches m ON t.team_id IN (m.team1_id, m.team2_id)
        JOIN player_match_stats pms ON pms.match_id = m.match_id
        WHERE m.match_type = 'T20' AND m.gender = ?
        GROUP BY t.team_id
        ORDER BY player_count DESC
        LIMIT 20
    """, (gender,))
    
    teams = cursor.fetchall()
    conn.close()
    return teams


def test_name_matching(test_names, team_name=None, gender='male'):
    """Test matching for a list of names."""
    matcher = PlayerNameMatcher()
    
    results = []
    for name in test_names:
        result = matcher.find_player(name, team_name)
        if result:
            results.append({
                'input': name,
                'matched_id': result.player_id,
                'matched_name': result.db_name,
                'score': result.score,
                'method': result.method
            })
        else:
            results.append({
                'input': name,
                'matched_id': None,
                'matched_name': None,
                'score': 0,
                'method': 'NO MATCH'
            })
    
    return results


def main():
    print("=" * 70)
    print("🔍 PLAYER NAME MATCHING DIAGNOSTIC")
    print("=" * 70)
    
    # Show database info
    print("\n📊 Database Sample Players (Male T20):")
    print("-" * 70)
    players = get_sample_players('male', 15)
    for p in players:
        print(f"  ID {p['player_id']:5d}: {p['name']:<30} ({p['team_name']})")
    
    print("\n📊 Teams with Most Players (Male T20):")
    print("-" * 70)
    teams = get_teams_with_player_counts('male')
    for t in teams[:10]:
        print(f"  {t['name']:<30} - {t['player_count']} players")
    
    # Test common name formats that CREX might use
    print("\n" + "=" * 70)
    print("🧪 TESTING NAME MATCHING")
    print("=" * 70)
    
    # These are examples of how CREX might name players vs how they appear in database
    test_cases = [
        # (CREX-style name, expected DB name pattern)
        "Virat Kohli",
        "V Kohli",
        "Rohit Sharma",
        "R Sharma",
        "Jasprit Bumrah",
        "J Bumrah",
        "Mohammad Rizwan",
        "M Rizwan",
        "Babar Azam",
        "B Azam",
        # Try some less common names
        "Shadab Khan",
        "Shaheen Afridi",
        "David Miller",
        "Quinton de Kock",
        "Glenn Maxwell",
    ]
    
    print("\n📝 Testing common player names (no team filter):")
    print("-" * 70)
    results = test_name_matching(test_cases)
    
    for r in results:
        status = "✅" if r['matched_id'] else "❌"
        if r['matched_id']:
            print(f"  {status} '{r['input']}' → ID {r['matched_id']}: '{r['matched_name']}' ({r['method']}, score={r['score']:.2f})")
        else:
            print(f"  {status} '{r['input']}' → NOT FOUND")
    
    # Test with team context
    print("\n📝 Testing with team context (India):")
    print("-" * 70)
    india_players = ["Virat Kohli", "Rohit Sharma", "Jasprit Bumrah", "Hardik Pandya"]
    results = test_name_matching(india_players, team_name="India")
    
    for r in results:
        status = "✅" if r['matched_id'] else "❌"
        if r['matched_id']:
            print(f"  {status} '{r['input']}' → ID {r['matched_id']}: '{r['matched_name']}' ({r['method']})")
        else:
            print(f"  {status} '{r['input']}' → NOT FOUND")
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 DIAGNOSIS SUMMARY")
    print("=" * 70)
    
    total_tests = len(test_cases)
    matched = sum(1 for r in results if r['matched_id'])
    
    print(f"""
Total tests: {total_tests}
Matched: {matched}/{total_tests} ({100*matched/total_tests:.1f}%)

If match rate is low:
1. Check if CREX player names differ significantly from database names
2. Database may use abbreviated names (e.g., "V Kohli" not "Virat Kohli")
3. CREX may use full names while DB uses initials

Next steps:
1. Run the Flask app and try loading a CREX match
2. Check server logs for "Matched X/Y players" messages
3. Compare CREX player names with database names above
""")


if __name__ == '__main__':
    main()
