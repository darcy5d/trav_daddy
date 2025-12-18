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


@app.route('/api/team/<int:team_id>/recent-lineup', methods=['GET'])
def get_team_recent_lineup(team_id):
    """
    Get the most recent playing XI for a team from our database.
    
    This is used as a fallback when ESPN doesn't have squad data.
    """
    try:
        gender = request.args.get('gender', 'male')
        
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get the most recent match for this team
        cursor.execute("""
            SELECT m.match_id, m.date, m.team1_id, m.team2_id
            FROM matches m
            WHERE (m.team1_id = ? OR m.team2_id = ?)
              AND m.match_type = 'T20'
              AND m.gender = ?
            ORDER BY m.date DESC
            LIMIT 1
        """, (team_id, team_id, gender))
        
        recent_match = cursor.fetchone()
        
        if not recent_match:
            conn.close()
            return jsonify({
                'success': False,
                'error': 'No recent matches found for this team'
            })
        
        match_id = recent_match['match_id']
        match_date = recent_match['date']
        
        # Get the players who played in that match for this team
        # First get the innings for this team
        cursor.execute("""
            SELECT innings_id, batting_team_id
            FROM innings
            WHERE match_id = ?
              AND (batting_team_id = ? OR bowling_team_id = ?)
        """, (match_id, team_id, team_id))
        
        innings_rows = cursor.fetchall()
        
        # Get unique players from deliveries
        player_ids = set()
        for inn in innings_rows:
            innings_id = inn['innings_id']
            batting_team = inn['batting_team_id']
            
            if batting_team == team_id:
                # Get batters from this innings
                cursor.execute("""
                    SELECT DISTINCT batter_id FROM deliveries WHERE innings_id = ?
                    UNION
                    SELECT DISTINCT non_striker_id FROM deliveries WHERE innings_id = ? AND non_striker_id IS NOT NULL
                """, (innings_id, innings_id))
            else:
                # Get bowlers from this innings (we were bowling)
                cursor.execute("""
                    SELECT DISTINCT bowler_id FROM deliveries WHERE innings_id = ?
                """, (innings_id,))
            
            for row in cursor.fetchall():
                if row[0]:
                    player_ids.add(row[0])
        
        # Get player details
        if player_ids:
            placeholders = ','.join('?' * len(player_ids))
            cursor.execute(f"""
                SELECT player_id, name, batting_style, bowling_style
                FROM players
                WHERE player_id IN ({placeholders})
                ORDER BY name
            """, list(player_ids))
            players = [dict(row) for row in cursor.fetchall()]
        else:
            players = []
        
        # Get team name
        cursor.execute("SELECT name FROM teams WHERE team_id = ?", (team_id,))
        team_row = cursor.fetchone()
        team_name = team_row['name'] if team_row else 'Unknown'
        
        conn.close()
        
        return jsonify({
            'success': True,
            'team_id': team_id,
            'team_name': team_name,
            'recent_match_date': match_date,
            'recent_xi': players,
            'player_count': len(players)
        })
    
    except Exception as e:
        logger.error(f"Error fetching recent lineup for team {team_id}: {e}")
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


@app.route('/api/players/details', methods=['GET'])
def get_player_details():
    """Get details for specific player IDs."""
    try:
        player_ids = request.args.get('player_ids', '')
        if not player_ids:
            return jsonify({'success': False, 'error': 'No player IDs provided'}), 400
        
        # Parse comma-separated IDs
        player_id_list = [int(pid.strip()) for pid in player_ids.split(',') if pid.strip()]
        
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get player details
        placeholders = ','.join('?' * len(player_id_list))
        cursor.execute(f"""
            SELECT player_id, name, batting_style, bowling_style
            FROM players
            WHERE player_id IN ({placeholders})
        """, player_id_list)
        
        players_raw = cursor.fetchall()
        conn.close()
        
        # Infer playing role from batting and bowling styles
        players = []
        for row in players_raw:
            player_dict = dict(row)
            batting = player_dict.get('batting_style', '') or ''
            bowling = player_dict.get('bowling_style', '') or ''
            
            # Infer role
            if batting and bowling:
                player_dict['playing_role'] = 'Allrounder'
            elif bowling:
                player_dict['playing_role'] = 'Bowler'
            elif batting:
                player_dict['playing_role'] = 'Batter'
            else:
                player_dict['playing_role'] = ''
            
            players.append(player_dict)
        
        return jsonify({
            'success': True,
            'players': players
        })
    
    except Exception as e:
        logger.error(f"Error fetching player details: {e}")
        import traceback
        traceback.print_exc()
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
# ESPN Cricinfo Endpoints (Web Scraping)
# ============================================================================

_espn_scraper = None
_espn_cache = {}  # Simple in-memory cache
_espn_cache_time = {}

ESPN_CACHE_TTL = 30 * 60  # 30 minutes


def get_espn_scraper():
    """Get or initialize the ESPN scraper."""
    global _espn_scraper
    if _espn_scraper is None:
        from src.api.espn_scraper import ESPNCricInfoScraper
        _espn_scraper = ESPNCricInfoScraper(request_delay=1.0)
        logger.info("ESPN Cricinfo scraper initialized")
    return _espn_scraper


def _is_cache_valid(cache_key: str) -> bool:
    """Check if cache entry is still valid."""
    if cache_key not in _espn_cache_time:
        return False
    return (datetime.now().timestamp() - _espn_cache_time[cache_key]) < ESPN_CACHE_TTL


@app.route('/api/espn/upcoming', methods=['GET'])
def get_espn_upcoming():
    """
    Get upcoming T20 matches from ESPN Cricinfo.
    
    Query params:
        hours_ahead: How many hours ahead to include (default 24)
        refresh: Set to 'true' to bypass cache
        
    Returns:
        List of upcoming T20 matches with basic info
    """
    try:
        hours_ahead = int(request.args.get('hours_ahead', 24))
        force_refresh = request.args.get('refresh', '').lower() == 'true'
        cache_key = f'espn_schedule_{hours_ahead}'
        
        # Check cache (skip if force refresh)
        if not force_refresh and _is_cache_valid(cache_key):
            logger.info("Using cached ESPN schedule")
            return jsonify(_espn_cache[cache_key])
        
        scraper = get_espn_scraper()
        matches = scraper.get_t20_schedule(hours_ahead=hours_ahead)
        
        # Group by series
        series_dict = {}
        for m in matches:
            if m.series_name not in series_dict:
                series_dict[m.series_name] = {
                    'series_name': m.series_name,
                    'series_id': m.series_id,
                    'gender': m.gender,
                    'matches': []
                }
            
            series_dict[m.series_name]['matches'].append({
                'espn_id': m.espn_id,
                'title': m.title,
                'team1': m.team1_name,
                'team2': m.team2_name,
                'match_type': m.match_type,
                'slug': m.slug,
                'status': m.status,
                'start_date': m.start_date,  # YYYY-MM-DD
                'start_time': m.start_time,  # Local time string
                'date_time_gmt': m.date_time_gmt,  # ISO format GMT
                'venue_city': m.venue_city,  # City name from schedule
                'match_url': m.match_url,
                'gender': m.gender
            })
        
        result = {
            'success': True,
            'source': 'espn_cricinfo',
            'matches_by_series': list(series_dict.values()),
            'total_matches': len(matches),
            'hours_ahead': hours_ahead
        }
        
        # Cache result
        _espn_cache[cache_key] = result
        _espn_cache_time[cache_key] = datetime.now().timestamp()
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error fetching ESPN upcoming matches: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/espn/match', methods=['GET'])
def get_espn_match():
    """
    Get full match details from ESPN Cricinfo including venue and squads.
    
    Query params:
        url: The match-preview URL from ESPN
        
    Returns:
        Match details with venue and squad data, matched to database
    """
    try:
        match_url = request.args.get('url')
        if not match_url:
            return jsonify({'success': False, 'error': 'url parameter required'}), 400
        
        cache_key = f'espn_match_{match_url}'
        
        # Check cache (longer TTL for match details)
        if cache_key in _espn_cache and _is_cache_valid(cache_key):
            logger.info(f"Using cached ESPN match data")
            return jsonify(_espn_cache[cache_key])
        
        scraper = get_espn_scraper()
        match = scraper.get_match_details(match_url)
        
        if not match:
            return jsonify({'success': False, 'error': 'Failed to fetch match from ESPN'}), 404
        
        # Match venue to database
        venue_db = None
        if match.venue:
            venue_match = scraper.match_venue_to_db(match.venue, match.gender)
            if venue_match:
                venue_db = {'venue_id': venue_match[0], 'name': venue_match[1]}
        
        # Match teams to database
        team1_db = None
        team2_db = None
        if match.team1:
            team_match = scraper.match_team_to_db(match.team1, match.gender)
            if team_match:
                team1_db = {'team_id': team_match[0], 'name': team_match[1]}
                # Match players
                scraper.match_players_to_db(match.team1, team_match[1], match.gender)
        
        if match.team2:
            team_match = scraper.match_team_to_db(match.team2, match.gender)
            if team_match:
                team2_db = {'team_id': team_match[0], 'name': team_match[1]}
                # Match players
                scraper.match_players_to_db(match.team2, team_match[1], match.gender)
        
        def team_to_dict(team):
            if not team:
                return None
            return {
                'espn_id': team.espn_id,
                'name': team.name,
                'long_name': team.long_name,
                'abbreviation': team.abbreviation,
                'db_team_id': team.db_team_id,
                'players': [
                    {
                        'espn_id': p.espn_id,
                        'name': p.name,
                        'long_name': p.long_name,
                        'role': p.role,
                        'playing_roles': p.playing_roles,
                        'batting_styles': p.batting_styles,
                        'bowling_styles': p.bowling_styles,
                        'is_overseas': p.is_overseas,
                        'db_player_id': p.db_player_id
                    }
                    for p in team.players
                ]
            }
        
        result = {
            'success': True,
            'source': 'espn_cricinfo',
            'match': {
                'espn_id': match.espn_id,
                'title': match.title,
                'series_name': match.series_name,
                'series_id': match.series_id,
                'status': match.status,
                'start_date': match.start_date,
                'start_time': match.start_time,
                'date_time_gmt': match.date_time_gmt,
                'gender': match.gender,
                'has_squads': match.has_squads,
                'venue': {
                    'name': match.venue.name if match.venue else None,
                    'town': match.venue.town if match.venue else None,
                    'country': match.venue.country if match.venue else None,
                    'db_venue_id': venue_db['venue_id'] if venue_db else None,
                    'db_venue_name': venue_db['name'] if venue_db else None
                } if match.venue else None,
                'team1': team_to_dict(match.team1),
                'team2': team_to_dict(match.team2),
                'team1_db': team1_db,
                'team2_db': team2_db
            }
        }
        
        # Cache result
        _espn_cache[cache_key] = result
        _espn_cache_time[cache_key] = datetime.now().timestamp()
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error fetching ESPN match: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/espn/prefetch', methods=['POST'])
def prefetch_espn_matches():
    """
    Pre-fetch all match details for upcoming matches and cache them.
    Returns status of each match (venue found, squad availability).
    """
    try:
        hours_ahead = int(request.args.get('hours_ahead', 24))
        
        scraper = get_espn_scraper()
        matches = scraper.get_t20_schedule(hours_ahead=hours_ahead)
        
        results = []
        for m in matches:
            match_status = {
                'espn_id': m.espn_id,
                'title': f"{m.team1_name} vs {m.team2_name}",
                'match_url': m.match_url,
                'venue_found': False,
                'venue_name': None,
                'team1_squad': False,
                'team1_count': 0,
                'team2_squad': False,
                'team2_count': 0,
                'cached': False
            }
            
            # Check if already cached
            cache_key = f'espn_match_{m.match_url}'
            if _is_cache_valid(cache_key):
                cached_data = _espn_cache.get(cache_key, {})
                match_data = cached_data.get('match', {})
                venue = match_data.get('venue', {})
                team1 = match_data.get('team1', {})
                team2 = match_data.get('team2', {})
                
                match_status['cached'] = True
                match_status['venue_found'] = venue.get('db_venue_id') is not None
                match_status['venue_name'] = venue.get('name')
                match_status['team1_squad'] = len(team1.get('players', [])) > 0
                match_status['team1_count'] = len(team1.get('players', []))
                match_status['team2_squad'] = len(team2.get('players', [])) > 0
                match_status['team2_count'] = len(team2.get('players', []))
            else:
                # Fetch and cache the match details
                try:
                    details = scraper.get_match_details(m.match_url)
                    if details:
                        # Match venue to database
                        venue_db = None
                        if details.venue:
                            venue_match = scraper.match_venue_to_db(details.venue, details.gender)
                            if venue_match:
                                venue_db = {'venue_id': venue_match[0], 'name': venue_match[1]}
                        
                        # Match teams to database
                        team1_db = None
                        team2_db = None
                        if details.team1:
                            team_match = scraper.match_team_to_db(details.team1, details.gender)
                            if team_match:
                                team1_db = {'team_id': team_match[0], 'name': team_match[1]}
                                scraper.match_players_to_db(details.team1, team_match[1], details.gender)
                        
                        if details.team2:
                            team_match = scraper.match_team_to_db(details.team2, details.gender)
                            if team_match:
                                team2_db = {'team_id': team_match[0], 'name': team_match[1]}
                                scraper.match_players_to_db(details.team2, team_match[1], details.gender)
                        
                        # Build cache entry (same as get_espn_match)
                        def team_to_dict(team):
                            if not team:
                                return None
                            return {
                                'espn_id': team.espn_id,
                                'name': team.name,
                                'long_name': team.long_name,
                                'abbreviation': team.abbreviation,
                                'db_team_id': team.db_team_id,
                                'players': [
                                    {
                                        'espn_id': p.espn_id,
                                        'name': p.name,
                                        'long_name': p.long_name,
                                        'role': p.role,
                                        'playing_roles': p.playing_roles,
                                        'batting_styles': p.batting_styles,
                                        'bowling_styles': p.bowling_styles,
                                        'is_overseas': p.is_overseas,
                                        'db_player_id': p.db_player_id
                                    }
                                    for p in team.players
                                ]
                            }
                        
                        cache_result = {
                            'success': True,
                            'source': 'espn_cricinfo',
                            'match': {
                                'espn_id': details.espn_id,
                                'title': details.title,
                                'series_name': details.series_name,
                                'series_id': details.series_id,
                                'status': details.status,
                                'start_date': details.start_date,
                                'start_time': details.start_time,
                                'date_time_gmt': details.date_time_gmt,
                                'gender': details.gender,
                                'has_squads': details.has_squads,
                                'venue': {
                                    'name': details.venue.name if details.venue else None,
                                    'town': details.venue.town if details.venue else None,
                                    'country': details.venue.country if details.venue else None,
                                    'db_venue_id': venue_db['venue_id'] if venue_db else None,
                                    'db_venue_name': venue_db['name'] if venue_db else None
                                } if details.venue else None,
                                'team1': team_to_dict(details.team1),
                                'team2': team_to_dict(details.team2),
                                'team1_db': team1_db,
                                'team2_db': team2_db
                            }
                        }
                        
                        # Cache it
                        _espn_cache[cache_key] = cache_result
                        _espn_cache_time[cache_key] = datetime.now().timestamp()
                        
                        # Update status
                        match_status['cached'] = True
                        match_status['venue_found'] = venue_db is not None
                        match_status['venue_name'] = details.venue.name if details.venue else None
                        match_status['team1_squad'] = details.team1 and len(details.team1.players) > 0
                        match_status['team1_count'] = len(details.team1.players) if details.team1 else 0
                        match_status['team2_squad'] = details.team2 and len(details.team2.players) > 0
                        match_status['team2_count'] = len(details.team2.players) if details.team2 else 0
                        
                        logger.info(f"Pre-fetched: {match_status['title']} - Venue: {match_status['venue_found']}, T1: {match_status['team1_count']}, T2: {match_status['team2_count']}")
                except Exception as e:
                    logger.error(f"Error pre-fetching {m.match_url}: {e}")
            
            results.append(match_status)
        
        return jsonify({
            'success': True,
            'matches': results,
            'total': len(results),
            'cached_count': sum(1 for r in results if r['cached'])
        })
    
    except Exception as e:
        logger.error(f"Error in prefetch: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# Training / Data Management API Routes
# ============================================================================

@app.route('/training')
def training_page():
    """Data and training management page."""
    return render_template('training.html')


@app.route('/api/training/db-status', methods=['GET'])
def get_db_status():
    """
    Get database status including match counts and date ranges.
    
    Returns:
        JSON with database statistics by gender and format
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get T20 match statistics by gender
        cursor.execute("""
            SELECT 
                gender,
                COUNT(*) as total_matches,
                MIN(date) as earliest_date,
                MAX(date) as latest_date
            FROM matches
            WHERE match_type = 'T20'
            GROUP BY gender
        """)
        
        match_stats = {}
        for row in cursor.fetchall():
            gender = row['gender']
            match_stats[gender] = {
                'total_matches': row['total_matches'],
                'earliest_date': row['earliest_date'],
                'latest_date': row['latest_date'],
                'days_since_latest': (datetime.now().date() - datetime.strptime(row['latest_date'], '%Y-%m-%d').date()).days if row['latest_date'] else None
            }
        
        # Get overall statistics
        cursor.execute("SELECT COUNT(DISTINCT player_id) FROM player_match_stats")
        total_players = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT team_id) FROM teams")
        total_teams = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM deliveries")
        total_deliveries = cursor.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            'success': True,
            'match_stats': match_stats,
            'total_players': total_players,
            'total_teams': total_teams,
            'total_deliveries': total_deliveries
        })
        
    except Exception as e:
        logger.error(f"Error getting DB status: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/training/models', methods=['GET'])
def get_models():
    """
    Get all model versions and currently active models.
    
    Query params:
        gender: Filter by gender (optional)
        active_only: Return only active models (optional)
    
    Returns:
        JSON with list of model versions
    """
    try:
        from src.data.database import get_model_versions, init_model_versions_table
        
        # Ensure table exists
        init_model_versions_table()
        
        gender = request.args.get('gender')
        active_only = request.args.get('active_only', 'false').lower() == 'true'
        
        models = get_model_versions(
            gender=gender,
            format_type='T20',
            active_only=active_only
        )
        
        return jsonify({
            'success': True,
            'models': models
        })
        
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/training/models/<int:model_id>/activate', methods=['POST'])
def activate_model(model_id):
    """
    Set a specific model as active.
    
    Args:
        model_id: Model version ID
        
    Returns:
        JSON with success status
    """
    try:
        from src.data.database import set_active_model
        
        success = set_active_model(model_id)
        
        if success:
            return jsonify({'success': True, 'message': f'Model {model_id} activated'})
        else:
            return jsonify({'success': False, 'error': 'Failed to activate model'}), 400
            
    except Exception as e:
        logger.error(f"Error activating model: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/training/download', methods=['POST'])
def start_data_download():
    """
    Start background data download from Cricsheet.
    
    Request body:
        {
            "formats": ["all_male", "all_female"],
            "force_download": false
        }
        
    Returns:
        JSON with job ID for tracking progress
    """
    try:
        from src.utils.job_manager import start_job
        from src.data.downloader import download_cricsheet_data
        from src.data.ingest import ingest_matches
        
        data = request.get_json() or {}
        formats = data.get('formats', ['all_male', 'all_female'])
        force_download = data.get('force_download', False)
        
        def download_and_ingest():
            """Download and ingest data."""
            logger.info(f"Starting data download for formats: {formats}")
            
            # Download
            download_result = download_cricsheet_data(formats=formats, force_download=force_download)
            
            # Ingest
            logger.info("Starting data ingestion...")
            ingest_result = ingest_matches(formats=formats)
            
            return {
                'download': download_result,
                'ingest': ingest_result
            }
        
        job_id = start_job(
            name=f"Data Download ({', '.join(formats)})",
            func=download_and_ingest
        )
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Data download started'
        })
        
    except Exception as e:
        logger.error(f"Error starting download: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/training/retrain', methods=['POST'])
def start_model_retrain():
    """
    Start background model retraining.
    
    Request body:
        {
            "mode": "full" or "quick",
            "genders": ["male", "female"]
        }
        
    Returns:
        JSON with job ID for tracking progress
    """
    try:
        from src.utils.job_manager import start_job
        import sys
        from pathlib import Path
        
        # Add scripts to path
        scripts_dir = Path(__file__).resolve().parent.parent / 'scripts'
        sys.path.insert(0, str(scripts_dir))
        
        from full_retrain import run_full_pipeline
        
        data = request.get_json() or {}
        mode = data.get('mode', 'quick')
        genders = data.get('genders', ['male', 'female'])
        
        def retrain_models():
            """Retrain models based on mode."""
            logger.info(f"Starting model retrain: mode={mode}, genders={genders}")
            
            skip_ingest = (mode == 'quick')
            skip_elo = (mode == 'quick')
            male_only = ('male' in genders and 'female' not in genders)
            female_only = ('female' in genders and 'male' not in genders)
            
            result = run_full_pipeline(
                skip_ingest=skip_ingest,
                skip_elo=skip_elo,
                male_only=male_only,
                female_only=female_only
            )
            
            return result
        
        job_id = start_job(
            name=f"Model Retrain ({mode}, {', '.join(genders)})",
            func=retrain_models
        )
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Model retraining started'
        })
        
    except Exception as e:
        logger.error(f"Error starting retrain: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/training/job/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """
    Get status of a background job.
    
    Args:
        job_id: Job ID returned from start_data_download or start_model_retrain
        
    Returns:
        JSON with job status and progress
    """
    try:
        from src.utils.job_manager import get_job_status as get_status
        from pathlib import Path
        
        job_info = get_status(job_id)
        
        if job_info is None:
            return jsonify({'success': False, 'error': 'Job not found'}), 404
        
        # Convert any Path objects to strings for JSON serialization
        def convert_paths(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            else:
                return obj
        
        job_info = convert_paths(job_info)
        
        # Return recent logs only (last 100 lines)
        logs = job_info.get('logs', [])
        recent_logs = logs[-100:] if len(logs) > 100 else logs
        
        return jsonify({
            'success': True,
            'job': {
                **job_info,
                'logs': recent_logs,
                'total_log_lines': len(logs)
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
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
