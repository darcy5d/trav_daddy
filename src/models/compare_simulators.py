"""
Simulator Comparison: Vectorized NN vs Fast Lookup.

Compares both simulators on:
1. Speed benchmarks
2. Score distribution realism (vs historical data)
3. Win probability calibration
4. Player-sensitivity (do player IDs matter?)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
from typing import Dict
import numpy as np
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.vectorized_nn_sim import VectorizedNNSimulator
from src.models.fast_lookup_sim import FastLookupSimulator
from src.data.database import get_connection

logger = logging.getLogger(__name__)


def get_historical_stats() -> Dict:
    """Get historical T20 match statistics for comparison."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Average scores for completed innings
    cursor.execute("""
        SELECT 
            AVG(i.total_runs) as avg_score,
            AVG(i.total_wickets) as avg_wickets,
            MIN(i.total_runs) as min_score,
            MAX(i.total_runs) as max_score,
            COUNT(*) as n_innings
        FROM innings i
        JOIN matches m ON i.match_id = m.match_id
        WHERE m.match_type = 'T20' AND m.gender = 'male'
        AND i.total_overs >= 18
    """)
    row = cursor.fetchone()
    
    # Get percentiles
    cursor.execute("""
        SELECT i.total_runs
        FROM innings i
        JOIN matches m ON i.match_id = m.match_id
        WHERE m.match_type = 'T20' AND m.gender = 'male'
        AND i.total_overs >= 18
        ORDER BY i.total_runs
    """)
    scores = [r['total_runs'] for r in cursor.fetchall()]
    
    conn.close()
    
    return {
        'avg_score': row['avg_score'],
        'std_score': np.std(scores),
        'avg_wickets': row['avg_wickets'],
        'min_score': row['min_score'],
        'max_score': row['max_score'],
        'n_innings': row['n_innings'],
        'p5_score': np.percentile(scores, 5),
        'p95_score': np.percentile(scores, 95),
        'scores': np.array(scores)
    }


def compare_simulators():
    """Compare both simulators head-to-head."""
    
    print("=" * 80)
    print("SIMULATOR COMPARISON: VECTORIZED NN vs FAST LOOKUP")
    print("=" * 80)
    
    # Get historical data
    print("\nLoading historical statistics...")
    hist = get_historical_stats()
    print(f"Historical T20 ({hist['n_innings']} innings):")
    print(f"  Avg Score: {hist['avg_score']:.1f} ± {hist['std_score']:.1f}")
    print(f"  90% Range: {hist['p5_score']:.0f} - {hist['p95_score']:.0f}")
    print(f"  Avg Wickets: {hist['avg_wickets']:.1f}")
    
    # Initialize simulators
    print("\nInitializing simulators...")
    nn_sim = VectorizedNNSimulator()
    lookup_sim = FastLookupSimulator()
    
    # Common test setup
    team1_batters = list(range(1, 12))
    team1_bowlers = list(range(100, 105))
    team2_batters = list(range(200, 211))
    team2_bowlers = list(range(300, 305))
    
    n_matches = 1000
    
    # ========== SPEED BENCHMARK ==========
    print("\n" + "=" * 80)
    print("SPEED BENCHMARK")
    print("=" * 80)
    
    # Warmup
    _ = nn_sim.simulate_matches(10, team1_batters, team1_bowlers, team2_batters, team2_bowlers)
    _ = lookup_sim.simulate_matches(10, team1_batters, team1_bowlers, team2_batters, team2_bowlers)
    
    # NN Simulator
    start = time.time()
    nn_results = nn_sim.simulate_matches(n_matches, team1_batters, team1_bowlers, team2_batters, team2_bowlers)
    nn_time = time.time() - start
    
    # Lookup Simulator
    start = time.time()
    lookup_results = lookup_sim.simulate_matches(n_matches, team1_batters, team1_bowlers, team2_batters, team2_bowlers)
    lookup_time = time.time() - start
    
    print(f"\n{n_matches} matches:")
    print(f"  Vectorized NN:  {nn_time:.3f}s ({nn_time/n_matches*1000:.2f}ms/match)")
    print(f"  Fast Lookup:    {lookup_time:.3f}s ({lookup_time/n_matches*1000:.2f}ms/match)")
    print(f"  Speedup:        {nn_time/lookup_time:.1f}x")
    
    # ========== DISTRIBUTION COMPARISON ==========
    print("\n" + "=" * 80)
    print("SCORE DISTRIBUTION COMPARISON")
    print("=" * 80)
    
    # Combine both innings scores for comparison
    nn_all_scores = np.concatenate([nn_results['team1_scores'], nn_results['team2_scores']])
    lookup_all_scores = np.concatenate([lookup_results['team1_scores'], lookup_results['team2_scores']])
    
    print(f"\n{'Metric':<20} {'Historical':>12} {'NN Sim':>12} {'Lookup Sim':>12}")
    print("-" * 60)
    
    metrics = [
        ('Avg Score', hist['avg_score'], nn_all_scores.mean(), lookup_all_scores.mean()),
        ('Std Dev', hist['std_score'], nn_all_scores.std(), lookup_all_scores.std()),
        ('5th Percentile', hist['p5_score'], np.percentile(nn_all_scores, 5), np.percentile(lookup_all_scores, 5)),
        ('95th Percentile', hist['p95_score'], np.percentile(nn_all_scores, 95), np.percentile(lookup_all_scores, 95)),
        ('Avg Wickets', hist['avg_wickets'], 
         (nn_results['avg_team1_wickets'] + nn_results['avg_team2_wickets'])/2,
         (lookup_results['avg_team1_wickets'] + lookup_results['avg_team2_wickets'])/2),
    ]
    
    nn_errors = []
    lookup_errors = []
    
    for name, hist_val, nn_val, lookup_val in metrics:
        nn_err = abs(nn_val - hist_val) / hist_val * 100
        lookup_err = abs(lookup_val - hist_val) / hist_val * 100
        nn_errors.append(nn_err)
        lookup_errors.append(lookup_err)
        
        print(f"{name:<20} {hist_val:>12.1f} {nn_val:>12.1f} {lookup_val:>12.1f}")
    
    print("-" * 60)
    print(f"{'Mean Abs % Error':<20} {'':>12} {np.mean(nn_errors):>11.1f}% {np.mean(lookup_errors):>11.1f}%")
    
    # ========== WIN PROBABILITY ==========
    print("\n" + "=" * 80)
    print("WIN PROBABILITY (Equal Teams)")
    print("=" * 80)
    
    print(f"\n  Expected (equal teams): ~50%")
    print(f"  NN Simulator:           {nn_results['team1_win_prob']:.1%} vs {nn_results['team2_win_prob']:.1%}")
    print(f"  Lookup Simulator:       {lookup_results['team1_win_prob']:.1%} vs {lookup_results['team2_win_prob']:.1%}")
    
    # ========== RECOMMENDATION ==========
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    
    print(f"""
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  SIMULATOR         │  SPEED          │  ACCURACY       │  USE CASE         │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │  Vectorized NN     │  {nn_time/n_matches*1000:.1f}ms/match    │  {np.mean(nn_errors):.1f}% error      │  Player-specific  │
    │  Fast Lookup       │  {lookup_time/n_matches*1000:.2f}ms/match   │  {np.mean(lookup_errors):.1f}% error      │  Quick estimates  │
    └─────────────────────────────────────────────────────────────────────────────┘
    
    VERDICT:
    """)
    
    if np.mean(lookup_errors) < np.mean(nn_errors) + 5:
        print("    ✓ Fast Lookup is recommended for most use cases")
        print(f"    ✓ {nn_time/lookup_time:.0f}x faster with similar accuracy")
        print("    ✓ Use NN only when player-specific predictions are critical")
    else:
        print("    ✓ Vectorized NN is more accurate")
        print(f"    ✓ Fast Lookup is {nn_time/lookup_time:.0f}x faster")
        print("    ✓ Choose based on accuracy vs speed tradeoff needed")
    
    return {
        'nn_results': nn_results,
        'lookup_results': lookup_results,
        'nn_time': nn_time,
        'lookup_time': lookup_time,
        'historical': hist
    }


def test_with_real_players():
    """Test with actual player IDs from database."""
    print("\n" + "=" * 80)
    print("TEST WITH REAL PLAYERS")
    print("=" * 80)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get top batters and bowlers
    cursor.execute("""
        SELECT p.player_id, p.name
        FROM players p
        JOIN player_match_stats pms ON p.player_id = pms.player_id
        JOIN matches m ON pms.match_id = m.match_id
        WHERE m.match_type = 'T20' AND m.gender = 'male'
        GROUP BY p.player_id
        ORDER BY SUM(pms.runs_scored) DESC
        LIMIT 22
    """)
    top_batters = [r['player_id'] for r in cursor.fetchall()]
    
    cursor.execute("""
        SELECT p.player_id, p.name
        FROM players p
        JOIN player_match_stats pms ON p.player_id = pms.player_id
        JOIN matches m ON pms.match_id = m.match_id
        WHERE m.match_type = 'T20' AND m.gender = 'male'
        AND pms.overs_bowled > 0
        GROUP BY p.player_id
        ORDER BY SUM(pms.wickets_taken) DESC
        LIMIT 10
    """)
    top_bowlers = [r['player_id'] for r in cursor.fetchall()]
    
    conn.close()
    
    if len(top_batters) >= 22 and len(top_bowlers) >= 10:
        team1_batters = top_batters[:11]
        team2_batters = top_batters[11:22]
        team1_bowlers = top_bowlers[:5]
        team2_bowlers = top_bowlers[5:10]
        
        nn_sim = VectorizedNNSimulator()
        lookup_sim = FastLookupSimulator()
        
        print("\nTop Players Match (1000 sims each):")
        
        nn_results = nn_sim.simulate_matches(1000, team1_batters, team1_bowlers, team2_batters, team2_bowlers)
        lookup_results = lookup_sim.simulate_matches(1000, team1_batters, team1_bowlers, team2_batters, team2_bowlers)
        
        print(f"\n  Vectorized NN:")
        print(f"    Team1 Win: {nn_results['team1_win_prob']:.1%}")
        print(f"    Avg Scores: {nn_results['avg_team1_score']:.1f} vs {nn_results['avg_team2_score']:.1f}")
        
        print(f"\n  Fast Lookup:")
        print(f"    Team1 Win: {lookup_results['team1_win_prob']:.1%}")
        print(f"    Avg Scores: {lookup_results['avg_team1_score']:.1f} vs {lookup_results['avg_team2_score']:.1f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    results = compare_simulators()
    test_with_real_players()

