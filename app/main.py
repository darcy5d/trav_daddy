"""
Flask Web Application for Cricket Match Predictor.

Provides a web interface and API for:
- Match predictions using fast lookup or NN simulators
- Team and player ELO rankings
- Match simulation
- Venue selection and toss simulation
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATABASE_PATH
from src.data.database import get_connection
from src.features.toss_stats import TossSimulator

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'cricket-predictor-secret-key'
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Lazy load simulators (they take time to initialize)
# Cache per gender to avoid re-initializing
_fast_simulators = {}  # {'male': simulator, 'female': simulator}
_nn_simulators = {}
_toss_simulator = None


def get_fast_simulator(gender: str = 'male'):
    """Get or initialize the fast lookup simulator for specified gender."""
    global _fast_simulators
    if gender not in _fast_simulators:
        from src.models.fast_lookup_sim import FastLookupSimulator
        _fast_simulators[gender] = FastLookupSimulator(use_h2h=True, gender=gender)
        logger.info(f"Fast lookup simulator initialized for {gender}")
    return _fast_simulators[gender]


def get_nn_simulator(gender: str = 'male'):
    """Get or initialize the NN simulator for specified gender."""
    global _nn_simulators
    if gender not in _nn_simulators:
        from src.models.vectorized_nn_sim import VectorizedNNSimulator
        _nn_simulators[gender] = VectorizedNNSimulator(gender=gender)
        logger.info(f"NN simulator initialized for {gender}")
    return _nn_simulators[gender]


def get_toss_simulator():
    """Get or initialize the toss simulator."""
    global _toss_simulator
    if _toss_simulator is None:
        _toss_simulator = TossSimulator()
        logger.info("Toss simulator initialized")
    return _toss_simulator


# ============================================================================
# Web Routes
# ============================================================================

@app.route('/')
def index():
    """Home page with quick stats."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get stats
        cursor.execute("SELECT COUNT(*) FROM matches WHERE match_type = 'T20'")
        t20_matches = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT player_id) FROM player_match_stats")
        total_players = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT team_id) FROM teams")
        total_teams = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM deliveries")
        total_deliveries = cursor.fetchone()[0]
        
        conn.close()
        
        stats = {
            't20_matches': t20_matches,
            'total_players': total_players,
            'total_teams': total_teams,
            'total_deliveries': total_deliveries
        }
        
        return render_template('index.html', stats=stats)
    except Exception as e:
        logger.error(f"Error loading index: {e}")
        return render_template('index.html', stats={})


@app.route('/predict')
def predict_page():
    """Match prediction page."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get teams
        cursor.execute("""
            SELECT DISTINCT t.team_id, t.name
            FROM teams t
            JOIN matches m ON t.team_id IN (m.team1_id, m.team2_id)
            WHERE m.match_type = 'T20' AND m.gender = 'male'
            ORDER BY t.name
        """)
        teams = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return render_template('predict.html', teams=teams)
    except Exception as e:
        logger.error(f"Error loading predict page: {e}")
        return render_template('predict.html', teams=[])


@app.route('/rankings')
def rankings_page():
    """ELO rankings page."""
    return render_template('rankings.html')


# ============================================================================
# API Routes
# ============================================================================

@app.route('/api/teams', methods=['GET'])
def get_teams():
    """
    Get list of teams for specified gender and type.
    
    Query params:
        gender: 'male' or 'female' (default: 'male')
        team_type: 'international', 'franchise', 'domestic', or 'all' (default: 'all')
    """
    try:
        gender = request.args.get('gender', 'male')
        team_type = request.args.get('team_type', 'all')
        
        conn = get_connection()
        cursor = conn.cursor()
        
        # Build query based on filters
        if team_type == 'all':
            cursor.execute("""
                SELECT DISTINCT t.team_id, t.name, t.team_type
                FROM teams t
                JOIN matches m ON t.team_id IN (m.team1_id, m.team2_id)
                WHERE m.match_type = 'T20' AND m.gender = ?
                ORDER BY t.name
            """, (gender,))
        elif team_type == 'club':
            # 'club' combines franchise and domestic
            cursor.execute("""
                SELECT DISTINCT t.team_id, t.name, t.team_type
                FROM teams t
                JOIN matches m ON t.team_id IN (m.team1_id, m.team2_id)
                WHERE m.match_type = 'T20' AND m.gender = ?
                  AND t.team_type IN ('franchise', 'domestic')
                ORDER BY t.name
            """, (gender,))
        else:
            cursor.execute("""
                SELECT DISTINCT t.team_id, t.name, t.team_type
                FROM teams t
                JOIN matches m ON t.team_id IN (m.team1_id, m.team2_id)
                WHERE m.match_type = 'T20' AND m.gender = ?
                  AND t.team_type = ?
                ORDER BY t.name
            """, (gender, team_type))
        
        teams = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return jsonify({
            'success': True, 
            'teams': teams, 
            'gender': gender,
            'team_type': team_type,
            'count': len(teams)
        })
    
    except Exception as e:
        logger.error(f"Error fetching teams: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/venues', methods=['GET'])
def get_venues():
    """
    Get list of venues with match counts for specified gender.
    
    Returns venues grouped hierarchically by country -> city -> venue.
    Consolidates duplicate venues (same venue with different name formats).
    """
    from src.data.country_mapping import get_country_for_venue, get_flag_for_country, get_region_for_country, get_location_for_venue
    from src.data.venue_normalizer import normalize_venue_name, extract_canonical_name
    
    try:
        gender = request.args.get('gender', 'male')
        min_matches = int(request.args.get('min_matches', 3))
        
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT v.venue_id, v.name, v.city, v.country, v.canonical_name,
                   COUNT(m.match_id) as match_count
            FROM venues v
            JOIN matches m ON m.venue_id = v.venue_id
            WHERE m.match_type = 'T20' AND m.gender = ?
            GROUP BY v.venue_id
            HAVING match_count >= ?
            ORDER BY v.city, v.name
        """, (gender, min_matches))
        
        raw_venues = cursor.fetchall()
        conn.close()
        
        # Consolidate duplicate venues by canonical name + city
        # Keep the venue with the most matches
        consolidated = {}  # key: (canonical_name, city) -> best venue
        
        for row in raw_venues:
            venue_id = row['venue_id']
            name = row['name']
            city = row['city']
            country = row['country']
            match_count = row['match_count']
            
            # If city or country is missing, try to get from venue name mapping
            if not city or not country:
                mapped_city, mapped_country = get_location_for_venue(name)
                if not city and mapped_city:
                    city = mapped_city
                if not country and mapped_country:
                    country = mapped_country
            
            # Fall back to venue-based country lookup
            if not country:
                country = get_country_for_venue(name, city)
            
            # Get canonical name (remove city suffix if present)
            canonical = row['canonical_name'] or extract_canonical_name(name, city)
            
            # Create consolidation key
            key = (normalize_venue_name(canonical), city or '')
            
            if key not in consolidated or match_count > consolidated[key]['match_count']:
                consolidated[key] = {
                    'venue_id': venue_id,
                    'name': canonical,  # Use canonical name for display
                    'original_name': name,
                    'city': city,
                    'country': country,
                    'match_count': match_count
                }
            else:
                # Add match count from duplicate
                consolidated[key]['match_count'] += match_count
        
        # Build hierarchical structure: country -> city -> venues
        countries = {}
        
        for venue_data in consolidated.values():
            venue_id = venue_data['venue_id']
            name = venue_data['name']
            city = venue_data['city'] or 'Unknown'
            country = venue_data['country']
            match_count = venue_data['match_count']
            
            # Check if this country should be grouped under West Indies
            region = get_region_for_country(country)
            display_country = region if region else country
            
            # Initialize country/region if not seen
            if display_country not in countries:
                countries[display_country] = {
                    'name': display_country,
                    'flag': get_flag_for_country(display_country),
                    'cities': {},
                    'is_region': region is not None
                }
            
            # For West Indies, include the actual country in city display
            display_city = f"{country} - {city}" if region else city
            
            # Initialize city if not seen
            if display_city not in countries[display_country]['cities']:
                countries[display_country]['cities'][display_city] = []
            
            # Add venue
            countries[display_country]['cities'][display_city].append({
                'venue_id': venue_id,
                'name': name,
                'city': city,
                'country': country,
                'region': region,
                'match_count': match_count
            })
        
        # Convert to sorted list structure
        result = []
        for country_name in sorted(countries.keys()):
            country_data = countries[country_name]
            cities_list = []
            
            for city_name in sorted(country_data['cities'].keys()):
                venues = sorted(country_data['cities'][city_name], key=lambda v: -v['match_count'])
                cities_list.append({
                    'name': city_name,
                    'venues': venues
                })
            
            result.append({
                'name': country_data['name'],
                'flag': country_data['flag'],
                'cities': cities_list
            })
        
        # Flat list (consolidated)
        flat_venues = list(consolidated.values())
        
        return jsonify({
            'success': True, 
            'venues': flat_venues,
            'venues_hierarchical': result,
            'gender': gender
        })
    
    except Exception as e:
        logger.error(f"Error fetching venues: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/players/<int:team_id>', methods=['GET'])
def get_team_players(team_id):
    """Get players for a specific team."""
    try:
        gender = request.args.get('gender', 'male')
        
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get batters (by runs scored)
        cursor.execute("""
            SELECT DISTINCT p.player_id, p.name, 'batter' as role,
                   SUM(pms.runs_scored) as total_runs
            FROM players p
            JOIN player_match_stats pms ON p.player_id = pms.player_id
            JOIN matches m ON pms.match_id = m.match_id
            WHERE pms.team_id = ? AND m.match_type = 'T20' AND m.gender = ?
            AND pms.runs_scored > 0
            GROUP BY p.player_id
            ORDER BY total_runs DESC
            LIMIT 15
        """, (team_id, gender))
        batters = [dict(row) for row in cursor.fetchall()]
        
        # Get bowlers (by wickets taken)
        cursor.execute("""
            SELECT DISTINCT p.player_id, p.name, 'bowler' as role,
                   SUM(pms.wickets_taken) as total_wickets
            FROM players p
            JOIN player_match_stats pms ON p.player_id = pms.player_id
            JOIN matches m ON pms.match_id = m.match_id
            WHERE pms.team_id = ? AND m.match_type = 'T20' AND m.gender = ?
            AND pms.overs_bowled > 0
            GROUP BY p.player_id
            ORDER BY total_wickets DESC
            LIMIT 10
        """, (team_id, gender))
        bowlers = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return jsonify({
            'success': True,
            'batters': batters,
            'bowlers': bowlers,
            'gender': gender
        })
    
    except Exception as e:
        logger.error(f"Error fetching players: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/simulate', methods=['POST'])
def simulate_match():
    """
    Run match simulation.
    
    Expected JSON:
    {
        "team1_batters": [player_ids],
        "team1_bowlers": [player_ids],
        "team2_batters": [player_ids],
        "team2_bowlers": [player_ids],
        "simulator": "fast" or "nn",
        "n_simulations": int,
        "venue_id": int (optional),
        "use_toss": bool (optional, default false),
        "gender": "male" or "female" (optional, default "male")
    }
    """
    try:
        data = request.get_json()
        
        team1_batters_raw = data.get('team1_batters', [])
        team1_bowlers_raw = data.get('team1_bowlers', [])
        team2_batters_raw = data.get('team2_batters', [])
        team2_bowlers_raw = data.get('team2_bowlers', [])
        simulator_type = data.get('simulator', 'nn')  # Default to NN for scorecard
        n_simulations = min(data.get('n_simulations', 1000), 10000)
        venue_id = data.get('venue_id')
        use_toss = data.get('use_toss', False)
        gender = data.get('gender', 'male')
        
        # CRICKET XI LOGIC: Batting order = top-order batters + bowlers at tail
        def build_batting_order(batters, bowlers):
            """Build proper batting order: batters[:6] + bowlers[:5] = 11 unique"""
            top_order = batters[:6] if len(batters) >= 6 else batters[:]
            tail = bowlers[:5] if len(bowlers) >= 5 else bowlers[:]
            batting_order = list(top_order) + list(tail)
            # Remove duplicates
            seen = set()
            unique_order = []
            for pid in batting_order:
                if pid not in seen:
                    unique_order.append(pid)
                    seen.add(pid)
            # Pad from remaining batters if needed
            if len(unique_order) < 11:
                for pid in batters[6:]:
                    if pid not in seen and len(unique_order) < 11:
                        unique_order.append(pid)
                        seen.add(pid)
            return unique_order[:11]
        
        team1_batters = build_batting_order(team1_batters_raw, team1_bowlers_raw)
        team2_batters = build_batting_order(team2_batters_raw, team2_bowlers_raw)
        
        # Get unique bowlers (no duplicates allowed - each bowler can only bowl once)
        team1_bowlers = list(dict.fromkeys(team1_bowlers_raw[:5]))  # Dedupe, keep order
        team2_bowlers = list(dict.fromkeys(team2_bowlers_raw[:5]))
        
        # If < 5 bowlers, add part-timers from batters (realistic cricket strategy)
        # Part-timers are batters who aren't already in the bowling list
        def fill_with_parttimers(bowlers, batters, min_bowlers=5):
            if len(bowlers) >= min_bowlers:
                return bowlers
            bowler_set = set(bowlers)
            for batter_id in batters:
                if batter_id not in bowler_set:
                    bowlers.append(batter_id)
                    bowler_set.add(batter_id)
                    if len(bowlers) >= min_bowlers:
                        break
            return bowlers
        
        team1_bowlers = fill_with_parttimers(team1_bowlers, team1_batters_raw)
        team2_bowlers = fill_with_parttimers(team2_bowlers, team2_batters_raw)
        
        # Get toss field probability from historical data
        toss_field_prob = 0.65  # Default T20 field preference
        if use_toss:
            try:
                toss_simulator = get_toss_simulator()
                rates = toss_simulator.stats.get_decision_rates('T20', gender)
                toss_field_prob = rates.get('field', 0.65)
            except:
                pass  # Use default
        
        # Select simulator for specified gender
        if simulator_type == 'nn':
            simulator = get_nn_simulator(gender)
        else:
            simulator = get_fast_simulator(gender)
        
        # Run simulation - toss is now simulated WITHIN each Monte Carlo iteration
        import time
        start = time.time()
        
        results = simulator.simulate_matches(
            n_simulations,
            team1_batters[:11],
            team1_bowlers[:5],
            team2_batters[:11],
            team2_bowlers[:5],
            venue_id=venue_id,
            use_toss=use_toss,
            toss_field_prob=toss_field_prob
        )
        
        elapsed = time.time() - start
        
        # Extract toss statistics from results if toss was simulated
        toss_info = None
        if use_toss and 'toss_stats' in results:
            ts = results['toss_stats']
            toss_info = {
                'team1_won_toss_pct': round(ts['team1_won_toss_pct'] * 100, 1),
                'chose_field_pct': round(ts['chose_field_pct'] * 100, 1),
                'team1_batted_first_pct': round(ts['team1_batted_first_pct'] * 100, 1),
                'note': 'Toss simulated independently for each of the {} matches'.format(n_simulations)
            }
        
        # Format response
        response = {
            'success': True,
            'team1_win_prob': round(results['team1_win_prob'] * 100, 1),
            'team2_win_prob': round(results['team2_win_prob'] * 100, 1),
            'avg_team1_score': round(results['avg_team1_score'], 1),
            'avg_team2_score': round(results['avg_team2_score'], 1),
            'team1_score_range': [round(results['team1_score_range'][0]), round(results['team1_score_range'][1])],
            'team2_score_range': [round(results['team2_score_range'][0]), round(results['team2_score_range'][1])],
            'n_simulations': n_simulations,
            'simulator': simulator_type,
            'elapsed_ms': round(elapsed * 1000, 1),
            'h2h_rate': round(results.get('h2h_rate', 0) * 100, 1) if 'h2h_rate' in results else None,
            'venue_id': venue_id,
            'toss_info': toss_info,
            'gender': gender
        }
        
        # Generate detailed scorecard for NN simulator
        if simulator_type == 'nn' and hasattr(simulator, 'simulate_detailed_match'):
            try:
                # Get player names for display
                player_names = _get_player_names(team1_batters + team2_batters + team1_bowlers + team2_bowlers)
                
                # Simulate one detailed match (use 50/50 bat first for representative scorecard)
                scorecard = simulator.simulate_detailed_match(
                    team1_batters[:11],
                    team1_bowlers[:5],
                    team2_batters[:11],
                    team2_bowlers[:5],
                    venue_id=venue_id,
                    team1_bats_first=True  # For scorecard, show Team 1 batting first
                )
                
                # Add player names to scorecard entries
                for b in scorecard['team1_batting']:
                    b['name'] = player_names.get(b['player_id'], f"Player {b['player_id']}")
                for b in scorecard['team2_batting']:
                    b['name'] = player_names.get(b['player_id'], f"Player {b['player_id']}")
                for b in scorecard['team1_bowling']:
                    b['name'] = player_names.get(b['player_id'], f"Player {b['player_id']}")
                for b in scorecard['team2_bowling']:
                    b['name'] = player_names.get(b['player_id'], f"Player {b['player_id']}")
                
                response['scorecard'] = scorecard
                
            except Exception as e:
                logger.warning(f"Could not generate detailed scorecard: {e}")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error simulating match: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/simulate-stream', methods=['POST'])
def simulate_match_stream():
    """
    Run match simulation with real-time progress updates via Server-Sent Events.
    
    Returns streaming response with progress updates and final results.
    """
    from flask import Response, stream_with_context
    import json
    import time
    
    data = request.get_json()
    
    team1_batters_raw = data.get('team1_batters', [])
    team1_bowlers_raw = data.get('team1_bowlers', [])
    team2_batters_raw = data.get('team2_batters', [])
    team2_bowlers_raw = data.get('team2_bowlers', [])
    simulator_type = data.get('simulator', 'nn')  # Default to NN for scorecard
    n_simulations = min(data.get('n_simulations', 1000), 10000)
    venue_id = data.get('venue_id')
    use_toss = data.get('use_toss', False)
    gender = data.get('gender', 'male')
    
    # CRICKET XI LOGIC: Batting order = top-order batters + bowlers at tail
    # This ensures 11 UNIQUE players (no duplicates possible)
    def build_batting_order(batters, bowlers):
        """
        Build proper cricket batting order:
        - Positions 1-6: Specialist batters (from batters list)
        - Positions 7-11: Bowlers who bat at tail (from bowlers list)
        Total: 11 unique players
        """
        seen = set()
        unique_order = []
        
        # 1. Add batters first (positions 1-6 typically)
        for pid in batters:
            if pid not in seen and len(unique_order) < 11:
                unique_order.append(pid)
                seen.add(pid)
        
        # 2. Add bowlers (they bat at tail, positions 7-11)
        for pid in bowlers:
            if pid not in seen and len(unique_order) < 11:
                unique_order.append(pid)
                seen.add(pid)
        
        # Log if we couldn't get 11 unique players
        if len(unique_order) < 11:
            logger.warning(f"Only {len(unique_order)} unique players for batting order! "
                          f"batters={batters}, bowlers={bowlers}")
        
        return unique_order
    
    # Build proper batting orders (11 unique players each)
    team1_batting_order = build_batting_order(team1_batters_raw, team1_bowlers_raw)
    team2_batting_order = build_batting_order(team2_batters_raw, team2_bowlers_raw)
    
    # Get unique bowlers (no duplicates allowed - each bowler can only bowl once)
    team1_bowlers = list(dict.fromkeys(team1_bowlers_raw[:5]))  # Dedupe, keep order
    team2_bowlers = list(dict.fromkeys(team2_bowlers_raw[:5]))
    
    # If < 5 bowlers, add part-timers from batters (realistic cricket strategy)
    def fill_with_parttimers(bowlers, batters, min_bowlers=5):
        if len(bowlers) >= min_bowlers:
            return bowlers
        bowler_set = set(bowlers)
        for batter_id in batters:
            if batter_id not in bowler_set:
                bowlers.append(batter_id)
                bowler_set.add(batter_id)
                if len(bowlers) >= min_bowlers:
                    break
        return bowlers
    
    team1_bowlers = fill_with_parttimers(team1_bowlers, team1_batters_raw)
    team2_bowlers = fill_with_parttimers(team2_bowlers, team2_batters_raw)
    
    logger.info(f"Team 1 batting order: {len(team1_batting_order)} unique players")
    logger.info(f"Team 2 batting order: {len(team2_batting_order)} unique players")
    
    def generate():
        try:
            # Get toss field probability
            toss_field_prob = 0.65
            if use_toss:
                try:
                    toss_simulator = get_toss_simulator()
                    rates = toss_simulator.stats.get_decision_rates('T20', gender)
                    toss_field_prob = rates.get('field', 0.65)
                except:
                    pass
            
            # Select simulator
            if simulator_type == 'nn':
                simulator = get_nn_simulator(gender)
            else:
                simulator = get_fast_simulator(gender)
            
            # Chunked simulation with progress updates
            # Larger chunks = better GPU/CPU utilization (TF batches more efficiently)
            chunk_size = 1000  # Increased from 500 for better throughput
            all_team1_scores = []
            all_team2_scores = []
            all_team1_wins = []
            completed = 0
            start_time = time.time()
            
            toss_stats_accum = {'team1_won_toss': 0, 'chose_field': 0, 'team1_batted_first': 0, 'total': 0}
            
            for chunk_start in range(0, n_simulations, chunk_size):
                chunk_n = min(chunk_size, n_simulations - chunk_start)
                
                # Run chunk with proper batting orders (11 unique players each)
                chunk_results = simulator.simulate_matches(
                    chunk_n,
                    team1_batting_order,  # 11 unique: batters[:6] + bowlers[:5]
                    team1_bowlers[:5],
                    team2_batting_order,  # 11 unique: batters[:6] + bowlers[:5]
                    team2_bowlers[:5],
                    venue_id=venue_id,
                    use_toss=use_toss,
                    toss_field_prob=toss_field_prob
                )
                
                # Accumulate results
                all_team1_scores.extend(chunk_results['team1_scores'].tolist())
                all_team2_scores.extend(chunk_results['team2_scores'].tolist())
                all_team1_wins.extend((chunk_results['team1_scores'] > chunk_results['team2_scores']).tolist())
                
                # Accumulate toss stats
                if use_toss and 'toss_stats' in chunk_results:
                    ts = chunk_results['toss_stats']
                    toss_stats_accum['team1_won_toss'] += ts['team1_won_toss_pct'] * chunk_n
                    toss_stats_accum['chose_field'] += ts['chose_field_pct'] * chunk_n
                    toss_stats_accum['team1_batted_first'] += ts['team1_batted_first_pct'] * chunk_n
                    toss_stats_accum['total'] += chunk_n
                
                completed += chunk_n
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (n_simulations - completed) / rate if rate > 0 else 0
                
                # Send progress update
                progress = {
                    'type': 'progress',
                    'completed': completed,
                    'total': n_simulations,
                    'elapsed_ms': int(elapsed * 1000),
                    'rate': round(rate, 1),
                    'eta_seconds': round(remaining, 1),
                    'pct': round(completed * 100 / n_simulations, 1)
                }
                yield f"data: {json.dumps(progress)}\n\n"
            
            # Calculate final stats
            import numpy as np
            team1_scores = np.array(all_team1_scores)
            team2_scores = np.array(all_team2_scores)
            team1_wins = np.array(all_team1_wins)
            
            elapsed = time.time() - start_time
            
            # Build toss info
            toss_info = None
            if use_toss and toss_stats_accum['total'] > 0:
                t = toss_stats_accum['total']
                toss_info = {
                    'team1_won_toss_pct': round(toss_stats_accum['team1_won_toss'] / t * 100, 1),
                    'chose_field_pct': round(toss_stats_accum['chose_field'] / t * 100, 1),
                    'team1_batted_first_pct': round(toss_stats_accum['team1_batted_first'] / t * 100, 1),
                    'note': f'Toss simulated for each of {n_simulations} matches'
                }
            
            # Build final result
            result = {
                'type': 'result',
                'success': True,
                'team1_win_prob': round(team1_wins.mean() * 100, 1),
                'team2_win_prob': round((~team1_wins).mean() * 100, 1),
                'avg_team1_score': round(team1_scores.mean(), 1),
                'avg_team2_score': round(team2_scores.mean(), 1),
                'team1_score_range': [int(np.percentile(team1_scores, 5)), int(np.percentile(team1_scores, 95))],
                'team2_score_range': [int(np.percentile(team2_scores, 5)), int(np.percentile(team2_scores, 95))],
                'n_simulations': n_simulations,
                'simulator': simulator_type,
                'elapsed_ms': round(elapsed * 1000, 1),
                'venue_id': venue_id,
                'toss_info': toss_info,
                'gender': gender
            }
            
            # Generate scorecard for NN simulator
            if simulator_type == 'nn' and hasattr(simulator, 'simulate_detailed_match'):
                try:
                    player_names = _get_player_names(team1_batting_order + team2_batting_order + team1_bowlers + team2_bowlers)
                    scorecard = simulator.simulate_detailed_match(
                        team1_batting_order,  # 11 unique players in batting order
                        team1_bowlers[:5],
                        team2_batting_order,  # 11 unique players in batting order
                        team2_bowlers[:5],
                        venue_id=venue_id,
                        team1_bats_first=True
                    )
                    for b in scorecard['team1_batting']:
                        b['name'] = player_names.get(b['player_id'], f"Player {b['player_id']}")
                    for b in scorecard['team2_batting']:
                        b['name'] = player_names.get(b['player_id'], f"Player {b['player_id']}")
                    for b in scorecard['team1_bowling']:
                        b['name'] = player_names.get(b['player_id'], f"Player {b['player_id']}")
                    for b in scorecard['team2_bowling']:
                        b['name'] = player_names.get(b['player_id'], f"Player {b['player_id']}")
                    result['scorecard'] = scorecard
                except Exception as e:
                    logger.warning(f"Could not generate scorecard: {e}")
            
            yield f"data: {json.dumps(result)}\n\n"
            
        except Exception as e:
            logger.error(f"Error in streaming simulation: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/api/rankings/months', methods=['GET'])
def get_available_months():
    """Get list of available months for historical ELO data."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT strftime('%Y-%m', date) as month
            FROM team_elo_history
            ORDER BY month DESC
            LIMIT 36
        """)
        
        months = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return jsonify({'success': True, 'months': months})
    
    except Exception as e:
        logger.error(f"Error fetching months: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/rankings/teams', methods=['GET'])
def get_team_rankings():
    """Get team ELO rankings with filters."""
    try:
        # Get parameters
        format_type = request.args.get('format', 'T20')
        gender = request.args.get('gender', 'male')
        min_matches = int(request.args.get('min_matches', 10))
        month = request.args.get('month')  # YYYY-MM or None for current
        
        conn = get_connection()
        cursor = conn.cursor()
        
        if month:
            # Get historical data for specific month (end of month snapshot)
            cursor.execute("""
                SELECT t.name, h.elo, 
                       (SELECT COUNT(*) FROM matches m 
                        WHERE (m.team1_id = t.team_id OR m.team2_id = t.team_id)
                          AND m.match_type = ? AND m.gender = ?
                          AND m.match_date <= date(? || '-01', '+1 month', '-1 day')) as match_count
                FROM team_elo_history h
                JOIN teams t ON h.team_id = t.team_id
                WHERE h.format = ? AND h.gender = ?
                  AND strftime('%Y-%m', h.date) = ?
                GROUP BY t.team_id
                HAVING match_count >= ?
                ORDER BY h.elo DESC
                LIMIT 30
            """, (format_type, gender, month, format_type, gender, month, min_matches))
        else:
            # Get current ELO
            elo_col = f"elo_{format_type.lower()}_{gender}"
            cursor.execute(f"""
                SELECT t.name, e.{elo_col} as elo,
                       (SELECT COUNT(*) FROM matches m 
                        WHERE (m.team1_id = t.team_id OR m.team2_id = t.team_id)
                          AND m.match_type = ? AND m.gender = ?) as match_count
                FROM team_current_elo e
                JOIN teams t ON e.team_id = t.team_id
                WHERE e.{elo_col} IS NOT NULL AND e.{elo_col} != 1500
                GROUP BY t.team_id
                HAVING match_count >= ?
                ORDER BY e.{elo_col} DESC
                LIMIT 30
            """, (format_type, gender, min_matches))
        
        rankings = [{'name': r[0], 'elo': round(r[1], 1), 'matches': r[2]} for r in cursor.fetchall()]
        conn.close()
        
        return jsonify({'success': True, 'rankings': rankings})
    
    except Exception as e:
        logger.error(f"Error fetching team rankings: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/rankings/batting', methods=['GET'])
def get_batting_rankings():
    """Get player batting ELO rankings with filters."""
    try:
        format_type = request.args.get('format', 'T20')
        gender = request.args.get('gender', 'male')
        min_matches = int(request.args.get('min_matches', 5))
        
        elo_col = f"batting_elo_{format_type.lower()}_{gender}"
        
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute(f"""
            SELECT p.name, p.country as team, e.{elo_col} as elo,
                   (SELECT COUNT(DISTINCT pms.match_id) FROM player_match_stats pms
                    JOIN matches m ON pms.match_id = m.match_id
                    WHERE pms.player_id = p.player_id 
                      AND m.match_type = ? AND m.gender = ?) as match_count
            FROM player_current_elo e
            JOIN players p ON e.player_id = p.player_id
            WHERE e.{elo_col} IS NOT NULL AND e.{elo_col} != 1500
            GROUP BY p.player_id
            HAVING match_count >= ?
            ORDER BY e.{elo_col} DESC
            LIMIT 30
        """, (format_type, gender, min_matches))
        
        rankings = [{'name': r[0], 'team': r[1] or '', 'elo': round(r[2], 1), 'matches': r[3]} for r in cursor.fetchall()]
        conn.close()
        
        return jsonify({'success': True, 'rankings': rankings})
    
    except Exception as e:
        logger.error(f"Error fetching batting rankings: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/rankings/bowling', methods=['GET'])
def get_bowling_rankings():
    """Get player bowling ELO rankings with filters."""
    try:
        format_type = request.args.get('format', 'T20')
        gender = request.args.get('gender', 'male')
        min_matches = int(request.args.get('min_matches', 5))
        
        elo_col = f"bowling_elo_{format_type.lower()}_{gender}"
        
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute(f"""
            SELECT p.name, p.country as team, e.{elo_col} as elo,
                   (SELECT COUNT(DISTINCT pms.match_id) FROM player_match_stats pms
                    JOIN matches m ON pms.match_id = m.match_id
                    WHERE pms.player_id = p.player_id 
                      AND m.match_type = ? AND m.gender = ?) as match_count
            FROM player_current_elo e
            JOIN players p ON e.player_id = p.player_id
            WHERE e.{elo_col} IS NOT NULL AND e.{elo_col} != 1500
            GROUP BY p.player_id
            HAVING match_count >= ?
            ORDER BY e.{elo_col} DESC
            LIMIT 30
        """, (format_type, gender, min_matches))
        
        rankings = [{'name': r[0], 'team': r[1] or '', 'elo': round(r[2], 1), 'matches': r[3]} for r in cursor.fetchall()]
        conn.close()
        
        return jsonify({'success': True, 'rankings': rankings})
    
    except Exception as e:
        logger.error(f"Error fetching bowling rankings: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# T20 Series & Fixture Endpoints (All Competitions)
# ============================================================================

_cricket_api_client = None
_lineup_service = None


def get_cricket_api_client():
    """Get or initialize the Cricket Data API client."""
    global _cricket_api_client
    if _cricket_api_client is None:
        from src.api.cricket_data_client import CricketDataClient
        _cricket_api_client = CricketDataClient()
        logger.info("Cricket Data API client initialized")
    return _cricket_api_client


def get_lineup_service():
    """Get or initialize the lineup service."""
    global _lineup_service
    if _lineup_service is None:
        from src.features.lineup_service import LineupService
        _lineup_service = LineupService()
        logger.info("Lineup service initialized")
    return _lineup_service


@app.route('/api/t20/series', methods=['GET'])
def get_t20_series():
    """
    Get list of active T20 series.
    
    Query params:
        gender: Optional filter - 'male' or 'female'
        
    Returns:
        List of T20 series with id, name, dates, match counts
    """
    try:
        gender = request.args.get('gender')  # None = all genders
        
        client = get_cricket_api_client()
        series_list = client.get_t20_series(gender=gender)
        
        return jsonify({
            'success': True,
            'series': [
                {
                    'id': s.id,
                    'name': s.name,
                    'start_date': s.start_date,
                    'end_date': s.end_date,
                    't20_count': s.t20_count,
                    'gender': s.gender
                }
                for s in series_list
            ],
            'gender_filter': gender,
            'total': len(series_list)
        })
    
    except Exception as e:
        logger.error(f"Error fetching T20 series: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/t20/series/<series_id>/fixtures', methods=['GET'])
def get_series_fixtures(series_id):
    """
    Get upcoming fixtures for a specific T20 series.
    
    Path params:
        series_id: The series ID from the API
        
    Query params:
        days_ahead: How many days ahead to include (default 14)
        
    Returns:
        List of upcoming T20 matches in the series
    """
    try:
        days_ahead = int(request.args.get('days_ahead', 14))
        series_name = request.args.get('series_name', '')
        
        client = get_cricket_api_client()
        matches = client.get_upcoming_series_matches(
            series_id=series_id,
            series_name=series_name,
            days_ahead=days_ahead
        )
        
        return jsonify({
            'success': True,
            'fixtures': [
                {
                    'id': m.id,
                    'name': m.name,
                    'date': m.date,
                    'team1': m.team1,
                    'team2': m.team2,
                    'status': m.status,
                    'venue': m.venue,
                    'gender': m.gender,
                    'series_id': m.series_id,
                    'series_name': m.series_name,
                    'has_squad': m.has_squad,
                    'is_upcoming': m.is_upcoming,
                    'date_time_gmt': m.date_time_gmt
                }
                for m in matches
            ],
            'series_id': series_id,
            'days_ahead': days_ahead,
            'total': len(matches)
        })
    
    except Exception as e:
        logger.error(f"Error fetching series fixtures: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/t20/upcoming', methods=['GET'])
def get_upcoming_matches():
    """
    Get all T20 matches in the next 24 hours, grouped by series.
    
    Returns matches from all competitions (men's and women's) with gender indicators.
    """
    try:
        client = get_cricket_api_client()
        matches_by_series = client.get_upcoming_matches_24h()
        
        # Format for frontend
        result = []
        total_matches = 0
        
        for series_name, series_data in matches_by_series.items():
            matches_list = []
            for m in series_data['matches']:
                matches_list.append({
                    'match_id': m.id,
                    'team1': m.team1,
                    'team2': m.team2,
                    'date_time_gmt': m.date_time_gmt,  # Send raw UTC for frontend to convert
                    'venue': m.venue or 'TBD',
                    'date': m.date,
                    'status': m.status,
                    'has_squad': m.has_squad
                })
                total_matches += 1
            
            result.append({
                'series_name': series_name,
                'series_id': series_data['series_id'],
                'gender': series_data['gender'],
                'matches': matches_list
            })
        
        # Sort by series name
        result.sort(key=lambda x: x['series_name'])
        
        return jsonify({
            'success': True,
            'matches_by_series': result,
            'total_matches': total_matches
        })
    
    except Exception as e:
        logger.error(f"Error fetching upcoming matches: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/t20/match/<match_id>', methods=['GET'])
def get_t20_match(match_id):
    """
    Get full T20 match data with squads.
    
    This is a generalized version of /api/wbbl/match/<match_id>
    that works with any T20 competition.
    
    Path params:
        match_id: The match ID from the API
        
    Returns:
        Match details with squad data and suggested XI
    """
    try:
        from src.api.cricket_data_client import CricketDataClient, detect_gender
        
        # First check if match exists and has squad data
        api_client = CricketDataClient()
        match_info = api_client.get_match_info(match_id)
        
        if not match_info:
            return jsonify({'success': False, 'error': 'Match not found in API'}), 404
        
        # Check if squad data is available
        if not match_info.get('hasSquad', False):
            # No squad data available - try to get recent lineups from our database
            teams = match_info.get('teams', [])
            team1 = teams[0] if teams else ''
            team2 = teams[1] if len(teams) > 1 else ''
            series_name = match_info.get('series', '')
            gender = detect_gender(f"{team1} {team2} {series_name}")
            
            # Try to find teams and their recent lineups in our database
            service = get_lineup_service()
            team1_db = service._get_db_team_name(team1)
            team2_db = service._get_db_team_name(team2)
            
            team1_recent_xi, team1_last_match = service.get_recent_lineup(team1_db, gender=gender)
            team2_recent_xi, team2_last_match = service.get_recent_lineup(team2_db, gender=gender)
            
            # Get player details for the recent XI
            def get_player_details(player_ids, gender):
                if not player_ids:
                    return []
                conn = get_connection()
                cursor = conn.cursor()
                placeholders = ','.join('?' * len(player_ids))
                elo_col_bat = f'batting_elo_t20_{gender}'
                elo_col_bowl = f'bowling_elo_t20_{gender}'
                cursor.execute(f"""
                    SELECT p.player_id, p.name,
                           COALESCE(e.{elo_col_bat}, 1500) as batting_elo,
                           COALESCE(e.{elo_col_bowl}, 1500) as bowling_elo
                    FROM players p
                    LEFT JOIN player_current_elo e ON p.player_id = e.player_id
                    WHERE p.player_id IN ({placeholders})
                """, player_ids)
                players = [dict(row) for row in cursor.fetchall()]
                conn.close()
                # Sort by original order
                id_to_player = {p['player_id']: p for p in players}
                return [id_to_player[pid] for pid in player_ids if pid in id_to_player]
            
            team1_players = get_player_details(team1_recent_xi, gender)
            team2_players = get_player_details(team2_recent_xi, gender)
            
            # Match venue
            venue_db_id, venue_db_name = service.find_venue_in_db(match_info.get('venue', ''), gender=gender)
            
            logger.info(f"No API squad - using DB recent XI: {team1_db} ({len(team1_recent_xi)} players), {team2_db} ({len(team2_recent_xi)} players)")
            
            return jsonify({
                'success': True,  # We can still proceed with DB data!
                'from_database': True,
                'message': f'Squad data not available from API. Using most recent lineups from database.',
                'match': {
                    'match_id': match_id,
                    'team1_name': team1,
                    'team2_name': team2,
                    'team1_db_name': team1_db,
                    'team2_db_name': team2_db,
                    'venue': match_info.get('venue', ''),
                    'venue_db_id': venue_db_id,
                    'venue_db_name': venue_db_name,
                    'date': match_info.get('date', ''),
                    'gender': gender,
                    'series': series_name,
                    'team1_recent_xi': team1_recent_xi,
                    'team2_recent_xi': team2_recent_xi,
                    'team1_last_match': team1_last_match,
                    'team2_last_match': team2_last_match,
                    'team1_players': team1_players,
                    'team2_players': team2_players
                }
            }), 200
        
        service = get_lineup_service()
        match = service.get_match_with_squads(match_id)
        
        if not match:
            return jsonify({'success': False, 'error': 'Failed to load match data'}), 500
        
        def player_to_dict(p):
            return {
                'player_id': p.player_id,
                'api_name': p.api_name,
                'db_name': p.db_name,
                'role': p.role,
                'batting_style': p.batting_style,
                'bowling_style': p.bowling_style,
                'matched': p.matched,
                'elo_batting': p.elo_batting,
                'elo_bowling': p.elo_bowling,
                'recent_form': p.recent_form
            }
        
        # Detect gender from team names
        from src.api.cricket_data_client import detect_gender
        gender = detect_gender(f"{match.team1_name} {match.team2_name}")
        
        # Get suggested XI from squads
        team1_suggested_ids = service.suggest_lineup(match.team1_squad, match.team1_recent_xi)
        team2_suggested_ids = service.suggest_lineup(match.team2_squad, match.team2_recent_xi)
        
        # Filter squad to get suggested players as MatchedPlayer objects
        team1_suggested_players = [p for p in match.team1_squad if p.player_id in team1_suggested_ids]
        team2_suggested_players = [p for p in match.team2_squad if p.player_id in team2_suggested_ids]
        
        # Ensure we have exactly 11 players with proper role balance
        team1_final, team1_fill_info = service.ensure_playing_xi(
            team1_suggested_players, match.team1_db_name, gender=gender
        )
        team2_final, team2_fill_info = service.ensure_playing_xi(
            team2_suggested_players, match.team2_db_name, gender=gender
        )
        
        return jsonify({
            'success': True,
            'match': {
                'match_id': match.match_id,
                'date': match.date,
                'team1_name': match.team1_name,
                'team2_name': match.team2_name,
                'team1_db_name': match.team1_db_name,
                'team2_db_name': match.team2_db_name,
                'venue': match.venue,
                'venue_db_id': match.venue_db_id,
                'venue_db_name': match.venue_db_name,
                'status': match.status,
                'is_upcoming': match.is_upcoming,
                'gender': gender,
                'team1_squad': [player_to_dict(p) for p in match.team1_squad],
                'team2_squad': [player_to_dict(p) for p in match.team2_squad],
                'team1_recent_xi': match.team1_recent_xi,
                'team2_recent_xi': match.team2_recent_xi,
                # Final playing XI with exactly 11 players
                'team1_suggested_xi': [p.player_id for p in team1_final],
                'team2_suggested_xi': [p.player_id for p in team2_final],
                # Full player info for suggested XI (for display)
                'team1_xi_players': [player_to_dict(p) for p in team1_final],
                'team2_xi_players': [player_to_dict(p) for p in team2_final],
                # Fill info for UI display
                'team1_fill_info': team1_fill_info,
                'team2_fill_info': team2_fill_info
            }
        })
    
    except Exception as e:
        logger.error(f"Error fetching T20 match: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# WBBL Fixture Endpoints (Legacy - kept for backward compatibility)
# ============================================================================


def _get_player_names(player_ids: list) -> dict:
    """Get player names for a list of player IDs."""
    if not player_ids:
        return {}
    
    conn = get_connection()
    cursor = conn.cursor()
    
    placeholders = ','.join('?' for _ in player_ids)
    cursor.execute(f"""
        SELECT player_id, name FROM players WHERE player_id IN ({placeholders})
    """, player_ids)
    
    result = {row['player_id']: row['name'] for row in cursor.fetchall()}
    conn.close()
    return result


@app.route('/api/wbbl/fixtures', methods=['GET'])
def get_wbbl_fixtures():
    """Get upcoming WBBL match fixtures."""
    try:
        service = get_lineup_service()
        fixtures = service.get_upcoming_fixtures(limit=10)
        
        return jsonify({
            'success': True,
            'fixtures': [
                {
                    'id': f.id,
                    'name': f.name,
                    'date': f.date,
                    'team1': f.team1,
                    'team2': f.team2,
                    'status': f.status,
                    'has_squad': f.has_squad
                }
                for f in fixtures
            ]
        })
    
    except Exception as e:
        logger.error(f"Error fetching WBBL fixtures: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/wbbl/match/<match_id>', methods=['GET'])
def get_wbbl_match(match_id):
    """Get full match data with squads, ensuring exactly 11 players per team."""
    try:
        service = get_lineup_service()
        match = service.get_match_with_squads(match_id)
        
        if not match:
            return jsonify({'success': False, 'error': 'Match not found'}), 404
        
        def player_to_dict(p):
            return {
                'player_id': p.player_id,
                'api_name': p.api_name,
                'db_name': p.db_name,
                'role': p.role,
                'batting_style': p.batting_style,
                'bowling_style': p.bowling_style,
                'matched': p.matched,
                'elo_batting': p.elo_batting,
                'elo_bowling': p.elo_bowling,
                'recent_form': p.recent_form
            }
        
        # Get suggested XI from squads
        team1_suggested_ids = service.suggest_lineup(match.team1_squad, match.team1_recent_xi)
        team2_suggested_ids = service.suggest_lineup(match.team2_squad, match.team2_recent_xi)
        
        # Filter squad to get suggested players as MatchedPlayer objects
        team1_suggested_players = [p for p in match.team1_squad if p.player_id in team1_suggested_ids]
        team2_suggested_players = [p for p in match.team2_squad if p.player_id in team2_suggested_ids]
        
        # Ensure we have exactly 11 players with proper role balance
        team1_final, team1_fill_info = service.ensure_playing_xi(
            team1_suggested_players, match.team1_db_name, gender='female'
        )
        team2_final, team2_fill_info = service.ensure_playing_xi(
            team2_suggested_players, match.team2_db_name, gender='female'
        )
        
        return jsonify({
            'success': True,
            'match': {
                'match_id': match.match_id,
                'date': match.date,
                'team1_name': match.team1_name,
                'team2_name': match.team2_name,
                'team1_db_name': match.team1_db_name,
                'team2_db_name': match.team2_db_name,
                'venue': match.venue,
                'venue_db_id': match.venue_db_id,
                'venue_db_name': match.venue_db_name,
                'status': match.status,
                'is_upcoming': match.is_upcoming,
                'team1_squad': [player_to_dict(p) for p in match.team1_squad],
                'team2_squad': [player_to_dict(p) for p in match.team2_squad],
                'team1_recent_xi': match.team1_recent_xi,
                'team2_recent_xi': match.team2_recent_xi,
                # Final playing XI with exactly 11 players
                'team1_suggested_xi': [p.player_id for p in team1_final],
                'team2_suggested_xi': [p.player_id for p in team2_final],
                # Full player info for suggested XI (for display)
                'team1_xi_players': [player_to_dict(p) for p in team1_final],
                'team2_xi_players': [player_to_dict(p) for p in team2_final],
                # Fill info for UI display
                'team1_fill_info': team1_fill_info,
                'team2_fill_info': team2_fill_info
            }
        })
    
    except Exception as e:
        logger.error(f"Error fetching WBBL match: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'error': 'Not found'}), 404
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}")
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'error': 'Internal server error'}), 500
    return render_template('500.html'), 500


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
