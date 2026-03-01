#!/usr/bin/env python3
"""
Debug script to diagnose 50/50 simulation results.

This script checks:
1. Whether player IDs in simulation requests match database IDs
2. How many players are actually found in the distribution files
3. What fallback distributions are being used

Run this from the project root:
    python scripts/debug_simulation.py
"""

import pickle
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_player_distributions(gender: str = 'male'):
    """Load player distributions and return the ID sets."""
    dist_path = project_root / f'data/processed/player_distributions_t20_{gender}.pkl'
    
    if not dist_path.exists():
        print(f"❌ ERROR: Player distributions file not found: {dist_path}")
        return None, None
    
    with open(dist_path, 'rb') as f:
        data = pickle.load(f)
    
    batter_ids = set(int(pid) for pid in data['batter_distributions'].keys())
    bowler_ids = set(int(pid) for pid in data['bowler_distributions'].keys())
    
    return batter_ids, bowler_ids


def check_player_match_rates(test_ids: list, batter_ids: set, bowler_ids: set):
    """Check what percentage of test IDs are found in distributions."""
    test_ids_int = []
    for tid in test_ids:
        try:
            test_ids_int.append(int(tid))
        except (ValueError, TypeError):
            print(f"  ⚠️ Non-numeric ID: {tid} (type: {type(tid).__name__})")
    
    batter_matches = sum(1 for tid in test_ids_int if tid in batter_ids)
    bowler_matches = sum(1 for tid in test_ids_int if tid in bowler_ids)
    
    return {
        'total': len(test_ids),
        'numeric': len(test_ids_int),
        'batter_matches': batter_matches,
        'bowler_matches': bowler_matches,
        'match_rate_batter': batter_matches / len(test_ids_int) * 100 if test_ids_int else 0,
        'match_rate_bowler': bowler_matches / len(test_ids_int) * 100 if test_ids_int else 0,
    }


def print_sample_ids(batter_ids: set, bowler_ids: set, n: int = 10):
    """Print sample IDs from distributions."""
    print(f"\n📊 Sample batter IDs from distributions:")
    sample_batters = list(batter_ids)[:n]
    for bid in sample_batters:
        print(f"    {bid}")
    
    print(f"\n📊 Sample bowler IDs from distributions:")
    sample_bowlers = list(bowler_ids)[:n]
    for bid in sample_bowlers:
        print(f"    {bid}")


def main():
    print("=" * 60)
    print("🔍 SIMULATION DEBUG: Checking Player ID Matching")
    print("=" * 60)
    
    for gender in ['male', 'female']:
        print(f"\n{'='*60}")
        print(f"📋 Checking {gender.upper()} distributions...")
        print("=" * 60)
        
        batter_ids, bowler_ids = load_player_distributions(gender)
        
        if batter_ids is None:
            continue
        
        print(f"✅ Loaded {len(batter_ids)} batter distributions")
        print(f"✅ Loaded {len(bowler_ids)} bowler distributions")
        
        # Check ID ranges
        if batter_ids:
            min_id = min(batter_ids)
            max_id = max(batter_ids)
            print(f"📊 Batter ID range: {min_id} to {max_id}")
        
        if bowler_ids:
            min_id = min(bowler_ids)
            max_id = max(bowler_ids)
            print(f"📊 Bowler ID range: {min_id} to {max_id}")
        
        print_sample_ids(batter_ids, bowler_ids)
        
        # Test with some typical CREX-style IDs (usually smaller numbers)
        print("\n" + "-" * 40)
        print("🧪 Testing with typical CREX-style IDs (1-1000):")
        print("-" * 40)
        
        crex_style_ids = list(range(1, 100))
        result = check_player_match_rates(crex_style_ids, batter_ids, bowler_ids)
        print(f"  IDs 1-99: {result['batter_matches']}/99 match batters ({result['match_rate_batter']:.1f}%)")
        print(f"  IDs 1-99: {result['bowler_matches']}/99 match bowlers ({result['match_rate_bowler']:.1f}%)")
        
        # Check if any small IDs exist in distributions
        small_ids_in_dist = [bid for bid in batter_ids if bid < 1000]
        print(f"\n  Distribution IDs < 1000: {len(small_ids_in_dist)}")
        if small_ids_in_dist:
            print(f"  Sample small IDs: {sorted(small_ids_in_dist)[:10]}")
    
    print("\n" + "=" * 60)
    print("🎯 DIAGNOSIS SUMMARY")
    print("=" * 60)
    print("""
If you're getting 50/50 results, check:

1. PLAYER ID MISMATCH:
   - CREX uses its own ID system (often small integers)
   - Database uses ESPN IDs (often 6-7 digit numbers)
   - If db_player_id is null, frontend sends CREX ID instead
   
2. HOW TO DEBUG IN BROWSER:
   Open browser DevTools (F12) → Network tab
   Run a simulation and check the /api/simulate request
   Look at the Request Payload:
     - team1_batters: [...]
     - team2_batters: [...]
   
   If IDs are small numbers (< 1000), they're probably CREX IDs
   If IDs are large numbers (> 100000), they're ESPN IDs
   
3. HOW TO FIX:
   - Ensure db_player_id is populated during CREX → DB matching
   - Check logs for "Matched X/Y players" messages
   - Add better logging in the simulation endpoint
   
4. QUICK TEST:
   Add this console.log in predict.html before simulation:
   
   console.log('Team1 IDs:', uniqueTeam1Batters);
   console.log('Team2 IDs:', uniqueTeam2Batters);
   
   Then check if any match the distribution IDs shown above.
""")


if __name__ == '__main__':
    main()
