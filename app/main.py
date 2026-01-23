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

# Configure TensorFlow threading BEFORE importing TensorFlow
# This must happen before any module imports TensorFlow
import multiprocessing
N_CPU_CORES = multiprocessing.cpu_count()

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Configure TensorFlow early (before any imports that might use it)
import tensorflow as tf
try:
    tf.config.threading.set_inter_op_parallelism_threads(max(4, N_CPU_CORES // 2))
    tf.config.threading.set_intra_op_parallelism_threads(max(4, N_CPU_CORES // 2))
except RuntimeError:
    pass  # Already configured

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATABASE_PATH
from src.data.database import get_connection
from src.features.toss_stats import TossSimulator

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'cricket-predictor-secret-key'
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
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
    Get the most recent playing XI for a team from our database with role categorization.
    
    This is used as a fallback when ESPN doesn't have squad data.
    """
    try:
        from src.utils.role_classifier import (
            infer_role_from_stats, categorize_role, 
            get_role_display_name, is_bowling_option
        )
        
        gender = request.args.get('gender', 'male')
        
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get the most recent match for this team
        cursor.execute("""
            SELECT m.match_id, m.date, m.team1_id, m.team2_id,
                   t1.name as team1_name, t2.name as team2_name
            FROM matches m
            JOIN teams t1 ON m.team1_id = t1.team_id
            JOIN teams t2 ON m.team2_id = t2.team_id
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
        opponent_name = recent_match['team2_name'] if recent_match['team1_id'] == team_id else recent_match['team1_name']
        
        # Get the players who played in that match with their stats
        cursor.execute("""
            SELECT 
                p.player_id,
                p.name,
                pms.batting_position,
                pms.runs_scored as match_runs,
                pms.wickets_taken as match_wickets,
                pms.stumpings as match_stumpings,
                -- Career stats for role inference
                (SELECT COUNT(DISTINCT match_id) FROM player_match_stats WHERE player_id = p.player_id) as total_matches,
                (SELECT SUM(runs_scored) FROM player_match_stats WHERE player_id = p.player_id) as career_runs,
                (SELECT SUM(balls_faced) FROM player_match_stats WHERE player_id = p.player_id) as career_balls,
                (SELECT SUM(overs_bowled) FROM player_match_stats WHERE player_id = p.player_id) as career_overs,
                (SELECT SUM(wickets_taken) FROM player_match_stats WHERE player_id = p.player_id) as career_wickets,
                (SELECT SUM(stumpings) FROM player_match_stats WHERE player_id = p.player_id) as career_stumpings,
                (SELECT AVG(CASE WHEN batting_position > 0 THEN batting_position END) FROM player_match_stats WHERE player_id = p.player_id) as avg_position
            FROM player_match_stats pms
            JOIN players p ON pms.player_id = p.player_id
            WHERE pms.match_id = ? AND pms.team_id = ?
            ORDER BY pms.batting_position, p.name
        """, (match_id, team_id))
        
        players_data = cursor.fetchall()
        
        # Get team name
        cursor.execute("SELECT name FROM teams WHERE team_id = ?", (team_id,))
        team_row = cursor.fetchone()
        team_name = team_row['name'] if team_row else 'Unknown'
        
        conn.close()
        
        # Build players list with roles
        players = []
        grouped_by_role = {
            'KEEPER': [],
            'BATTER': [],
            'ALLROUNDER': [],
            'BOWLER': []
        }
        
        for row in players_data:
            player_dict = dict(row)
            
            # Prepare stats for role inference
            stats = {
                'total_matches': player_dict.get('total_matches', 0),
                'runs_scored': player_dict.get('career_runs', 0),
                'balls_faced': player_dict.get('career_balls', 0),
                'overs_bowled': player_dict.get('career_overs', 0),
                'wickets_taken': player_dict.get('career_wickets', 0),
                'stumpings': player_dict.get('career_stumpings', 0),
                'avg_batting_position': player_dict.get('avg_position', 5.5)
            }
            
            # Infer role
            role = infer_role_from_stats(stats)
            role_category = categorize_role(role)
            
            player_info = {
                'player_id': player_dict['player_id'],
                'name': player_dict['name'],
                'role': get_role_display_name(role),
                'role_category': role_category.value,
                'is_bowling_option': is_bowling_option(role),
                'batting_position': player_dict.get('batting_position', 0),
                'match_performance': {
                    'runs': player_dict.get('match_runs', 0),
                    'wickets': player_dict.get('match_wickets', 0)
                },
                'stats': {
                    'matches': stats['total_matches'],
                    'runs': stats['runs_scored'],
                    'wickets': stats['wickets_taken']
                }
            }
            
            players.append(player_info)
            grouped_by_role[role_category.value].append(player_info)
        
        return jsonify({
            'success': True,
            'team_id': team_id,
            'team_name': team_name,
            'match_id': match_id,
            'match_date': match_date,
            'opponent': opponent_name,
            'players': players,
            'grouped_by_role': grouped_by_role,
            'player_count': len(players)
        })
    
    except Exception as e:
        logger.error(f"Error fetching recent lineup for team {team_id}: {e}")
        import traceback
        traceback.print_exc()
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


@app.route('/api/players/by-ids', methods=['POST'])
def get_players_by_ids():
    """
    Get player stats for a list of player IDs.
    Used for franchise teams where players are matched but team is not.
    
    Request body:
        player_ids: List of player IDs
        gender: 'male' or 'female'
        
    Returns:
        List of players with stats
    """
    try:
        data = request.get_json()
        player_ids = data.get('player_ids', [])
        gender = data.get('gender', 'male')
        
        if not player_ids:
            return jsonify({'success': False, 'error': 'No player IDs provided'}), 400
        
        conn = get_connection()
        cursor = conn.cursor()
        
        placeholders = ','.join(['?'] * len(player_ids))
        cursor.execute(f"""
            SELECT 
                p.player_id,
                p.name,
                SUM(pms.runs_scored) as total_runs,
                SUM(pms.balls_faced) as balls_faced,
                SUM(pms.wickets_taken) as total_wickets,
                SUM(pms.overs_bowled) as overs_bowled,
                COUNT(DISTINCT pms.match_id) as matches
            FROM players p
            LEFT JOIN player_match_stats pms ON p.player_id = pms.player_id
            LEFT JOIN matches m ON pms.match_id = m.match_id AND m.match_type = 'T20' AND m.gender = ?
            WHERE p.player_id IN ({placeholders})
            GROUP BY p.player_id
        """, (gender,) + tuple(player_ids))
        
        players = []
        for row in cursor.fetchall():
            players.append({
                'player_id': row['player_id'],
                'name': row['name'],
                'total_runs': row['total_runs'] or 0,
                'balls_faced': row['balls_faced'] or 0,
                'total_wickets': row['total_wickets'] or 0,
                'overs_bowled': row['overs_bowled'] or 0,
                'matches': row['matches'] or 0
            })
        
        conn.close()
        
        logger.info(f"Fetched stats for {len(players)} players")
        return jsonify({
            'success': True,
            'players': players
        })
    
    except Exception as e:
        logger.error(f"Error fetching players by IDs: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/players/<int:team_id>', methods=['GET'])
def get_team_players(team_id):
    """
    Get players with inferred roles - returns ALL players for the gender/format.
    Players who have played for this specific team are listed first.
    """
    try:
        gender = request.args.get('gender', 'male')
        
        from src.utils.role_classifier import (
            infer_role_from_stats, categorize_role, 
            get_role_display_name, is_bowling_option
        )
        
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get ALL players with T20 experience for this gender
        # Players who have played for this specific team will be prioritized
        cursor.execute("""
            SELECT 
                p.player_id,
                p.name,
                COUNT(DISTINCT pms.match_id) as total_matches,
                SUM(pms.runs_scored) as runs_scored,
                SUM(pms.balls_faced) as balls_faced,
                SUM(pms.overs_bowled) as overs_bowled,
                SUM(pms.wickets_taken) as wickets_taken,
                SUM(pms.stumpings) as stumpings,
                AVG(CASE WHEN pms.batting_position > 0 THEN pms.batting_position END) as avg_batting_position,
                CASE 
                    WHEN SUM(pms.not_out) > 0 THEN 
                        CAST(SUM(pms.runs_scored) AS FLOAT) / (COUNT(DISTINCT CASE WHEN pms.runs_scored > 0 THEN pms.match_id END) - SUM(pms.not_out))
                    ELSE 
                        CAST(SUM(pms.runs_scored) AS FLOAT) / NULLIF(COUNT(DISTINCT CASE WHEN pms.runs_scored > 0 THEN pms.match_id END), 0)
                END as batting_avg,
                -- Priority: 1 if played for this team, 0 otherwise
                CASE WHEN SUM(CASE WHEN pms.team_id = ? THEN 1 ELSE 0 END) > 0 THEN 1 ELSE 0 END as played_for_team
            FROM players p
            JOIN player_match_stats pms ON p.player_id = pms.player_id
            JOIN matches m ON pms.match_id = m.match_id
            WHERE m.match_type = 'T20' AND m.gender = ?
            GROUP BY p.player_id
            HAVING total_matches > 0
            ORDER BY played_for_team DESC, total_matches DESC, runs_scored DESC
            LIMIT 500
        """, (team_id, gender))
        
        players_data = cursor.fetchall()
        conn.close()
        
        # Process players and infer roles
        players = []
        for row in players_data:
            player_dict = dict(row)
            
            # Prepare stats for role inference
            stats = {
                'total_matches': player_dict.get('total_matches', 0),
                'runs_scored': player_dict.get('runs_scored', 0),
                'balls_faced': player_dict.get('balls_faced', 0),
                'overs_bowled': player_dict.get('overs_bowled', 0),
                'wickets_taken': player_dict.get('wickets_taken', 0),
                'stumpings': player_dict.get('stumpings', 0),
                'avg_batting_position': player_dict.get('avg_batting_position', 5.5),
                'batting_avg': player_dict.get('batting_avg', 0)
            }
            
            # Infer role
            role = infer_role_from_stats(stats)
            role_category = categorize_role(role)
            
            players.append({
                'player_id': player_dict['player_id'],
                'name': player_dict['name'],
                'role': get_role_display_name(role),
                'role_category': role_category.value,
                'is_bowling_option': is_bowling_option(role),
                'stats': {
                    'matches': stats['total_matches'],
                    'runs': stats['runs_scored'],
                    'wickets': stats['wickets_taken'],
                    'avg': round(stats['batting_avg'], 1) if stats['batting_avg'] else 0,
                    'stumpings': stats['stumpings']
                },
                'source': 'stats'
            })
        
        # Group players by role category
        grouped = {
            'KEEPER': [],
            'BATTER': [],
            'ALLROUNDER': [],
            'BOWLER': []
        }
        
        for player in players:
            category = player['role_category']
            grouped[category].append(player)
        
        return jsonify({
            'success': True,
            'players': players,
            'grouped_by_role': grouped,
            'gender': gender
        })
    
    except Exception as e:
        logger.error(f"Error fetching players: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500



@app.route('/api/team/waterfall-select', methods=['GET'])
def waterfall_team_selection():
    """
    External source-first waterfall selection hierarchy:
    1. CREX/ESPN XI (if available from match data)
    2. CREX/ESPN Squad Optimized (optimize best 11 from full squad)
    3. Database Last Match (recent lineup from database)
    4. Best Available (fallback from all T20 players)
    
    Auto-detects source from URL (crex.com or espn).
    """
    try:
        from src.utils.role_classifier import infer_role_from_stats, categorize_role, get_role_display_name, is_bowling_option
        from src.utils.team_optimizer import optimize_xi_from_squad, validate_team
        from src.features.name_matcher import PlayerNameMatcher
        
        team_id = request.args.get('team_id', type=int)
        team_num = request.args.get('team_num', type=int, default=1)
        match_url = request.args.get('match_url')
        gender = request.args.get('gender', 'male')
        
        if not team_id:
            return jsonify({'success': False, 'error': 'Missing team_id'}), 400
        
        logger.info(f"Waterfall selection for team {team_id} (team {team_num}), gender {gender}, match_url: {match_url}")
        
        # Step 1: Try external source data if match_url provided
        if match_url:
            try:
                # Auto-detect source from URL
                if 'crex.com' in match_url:
                    logger.info(f"Detected CREX URL, using CREX scraper")
                    scraper = get_crex_scraper()
                else:
                    logger.info(f"Using ESPN scraper for URL")
                    scraper = get_espn_scraper()
                
                match = scraper.get_match_details(match_url)
                
                if match:
                    # Determine which team (1 or 2)
                    team_data = match.team1 if team_num == 1 else match.team2
                    
                    # Detect source type for logging
                    source_type = 'CREX' if 'crex.com' in match_url else 'ESPN'
                    
                    if team_data and team_data.players:
                        logger.info(f"{source_type} squad found: {len(team_data.players)} players for team {team_num}")
                        
                        # Match venue and teams to database
                        if match.venue:
                            venue_match = scraper.match_venue_to_db(match.venue, match.gender)
                        team_match = None
                        if team_data:
                            team_match = scraper.match_team_to_db(team_data, match.gender)
                            # Always try to match players, even if team doesn't match
                            # For franchise teams, players are often international players in our database
                            team_name_for_matching = team_match[1] if team_match else None
                            scraper.match_players_to_db(team_data, team_name_for_matching, match.gender)
                        
                        # Step 1b: Optimize from External Squad
                        conn = get_connection()
                        cursor = conn.cursor()
                        
                        # Get database players matching squad (ESPN uses espn_id, CREX uses name matching)
                        # For ESPN, try espn_player_id lookup first
                        espn_player_ids = [getattr(p, 'espn_id', None) for p in team_data.players if getattr(p, 'espn_id', None)]
                        db_players = []
                        
                        if espn_player_ids:
                            placeholders = ','.join(['?'] * len(espn_player_ids))
                            cursor.execute(f"""
                                SELECT 
                                    p.player_id,
                                    p.name,
                                    p.espn_player_id,
                                    COUNT(DISTINCT pms.match_id) as total_matches,
                                    SUM(pms.runs_scored) as runs_scored,
                                    SUM(pms.balls_faced) as balls_faced,
                                    SUM(pms.overs_bowled) as overs_bowled,
                                    SUM(pms.wickets_taken) as wickets_taken,
                                    SUM(pms.stumpings) as stumpings,
                                    AVG(CASE WHEN pms.batting_position > 0 THEN pms.batting_position END) as avg_position
                                FROM players p
                                LEFT JOIN player_match_stats pms ON p.player_id = pms.player_id
                                LEFT JOIN matches m ON pms.match_id = m.match_id AND m.match_type = 'T20' AND m.gender = ?
                                WHERE p.espn_player_id IN ({placeholders})
                                GROUP BY p.player_id
                            """, (gender,) + tuple(espn_player_ids))
                            
                            # Convert sqlite3.Row objects to dicts
                            db_players = [dict(row) for row in cursor.fetchall()]
                        
                        # Also try name matching for players without db_player_id (primary method for CREX)
                        name_matcher = PlayerNameMatcher()
                        team_name = team_match[1] if team_match else None
                        for squad_player in team_data.players:
                            if not squad_player.db_player_id:
                                # Try to match by name
                                # Get external ID (espn_id or crex_id depending on source)
                                external_id = getattr(squad_player, 'espn_id', None) or getattr(squad_player, 'crex_id', None)
                                matched = name_matcher.find_player(
                                    squad_player.name,
                                    team_name=team_name,
                                    espn_player_id=external_id if source_type == 'ESPN' else None
                                )
                                if matched:
                                    squad_player.db_player_id = matched.player_id
                        
                        # Build squad from matched players
                        squad = []
                        db_players_dict = {p['player_id']: p for p in db_players}
                        
                        for squad_player in team_data.players:
                            db_id = squad_player.db_player_id
                            
                            if db_id:
                                # Get stats for this player (from query or fetch if name-matched)
                                player_row = db_players_dict.get(db_id)
                                
                                if not player_row:
                                    # Player was matched by name but not in query - fetch stats
                                    cursor.execute("""
                                        SELECT 
                                            p.player_id,
                                            p.name,
                                            COUNT(DISTINCT pms.match_id) as total_matches,
                                            SUM(pms.runs_scored) as runs_scored,
                                            SUM(pms.balls_faced) as balls_faced,
                                            SUM(pms.overs_bowled) as overs_bowled,
                                            SUM(pms.wickets_taken) as wickets_taken,
                                            SUM(pms.stumpings) as stumpings,
                                            AVG(CASE WHEN pms.batting_position > 0 THEN pms.batting_position END) as avg_position
                                        FROM players p
                                        LEFT JOIN player_match_stats pms ON p.player_id = pms.player_id
                                        LEFT JOIN matches m ON pms.match_id = m.match_id AND m.match_type = 'T20' AND m.gender = ?
                                        WHERE p.player_id = ?
                                        GROUP BY p.player_id
                                    """, (gender, db_id))
                                    row = cursor.fetchone()
                                    if row:
                                        player_row = dict(row)
                                
                                if player_row:
                                    stats = {
                                        'total_matches': player_row.get('total_matches', 0),
                                        'runs_scored': player_row.get('runs_scored', 0),
                                        'balls_faced': player_row.get('balls_faced', 0),
                                        'overs_bowled': player_row.get('overs_bowled', 0),
                                        'wickets_taken': player_row.get('wickets_taken', 0),
                                        'stumpings': player_row.get('stumpings', 0),
                                        'avg_batting_position': player_row.get('avg_position', 5.5)
                                    }
                                    
                                    role = infer_role_from_stats(stats)
                                    role_category = categorize_role(role)
                                    
                                    # Get external ID (espn_id or crex_id)
                                    external_id = getattr(squad_player, 'espn_id', None) or getattr(squad_player, 'crex_id', None)
                                    
                                    squad.append({
                                        'player_id': db_id,
                                        'name': player_row['name'],
                                        'role': get_role_display_name(role),
                                        'role_category': role_category.value,
                                        'is_bowling_option': is_bowling_option(role),
                                        'stats': {
                                            'runs': stats['runs_scored'],
                                            'wickets': stats['wickets_taken'],
                                            'matches': stats['total_matches']
                                        },
                                        'external_id': external_id  # Works for both ESPN and CREX
                                    })
                        
                        conn.close()
                        
                        if len(squad) >= 11:
                            # Optimize XI from external squad
                            optimized_xi = optimize_xi_from_squad(squad)
                            validation = validate_team(optimized_xi)
                            
                            # Ensure we have exactly 11 players
                            if len(optimized_xi) != 11:
                                logger.warning(f"Optimizer returned {len(optimized_xi)} players, expected 11. Using first 11.")
                                optimized_xi = optimized_xi[:11]
                            
                            logger.info(f"Optimized XI from {source_type} squad: {len(optimized_xi)} players")
                            
                            # Use source-specific labels for frontend
                            source_label = 'crex_squad_optimized' if source_type == 'CREX' else 'espn_squad_optimized'
                            
                            return jsonify({
                                'success': True,
                                'players': optimized_xi,
                                'source': source_label,
                                'confidence': 'high' if len(squad) >= 15 else 'medium',
                                'alternatives': [p for p in squad if p not in optimized_xi],
                                'validation': {
                                    'is_valid': validation.is_valid,
                                    'errors': validation.validation_errors,
                                    'warnings': validation.warnings
                                }
                            })
                        else:
                            logger.warning(f"{source_type} squad matched only {len(squad)} players, falling back to database")
            except Exception as e:
                logger.warning(f"Error fetching external match data: {e}")
                import traceback
                traceback.print_exc()
        
        # Step 2: Try database last match
        try:
            response = get_team_recent_lineup(team_id)
            data = response.get_json()
            
            if data.get('success') and data.get('players') and len(data['players']) >= 11:
                logger.info(f"Using recent lineup for team {team_id}")
                return jsonify({
                    'success': True,
                    'players': data['players'][:11],
                    'source': 'database_last_match',
                    'confidence': 'high',
                    'alternatives': data['players'][11:] if len(data['players']) > 11 else []
                })
        except Exception as e:
            logger.warning(f"Could not load recent lineup: {e}")
        
        # Step 3: Fallback - optimize from ALL available T20 players
        logger.info(f"No ESPN or database data, optimizing from all available players")
        
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get top 50 T20 players by experience (mix of batters and bowlers)
        cursor.execute("""
            SELECT 
                p.player_id,
                p.name,
                COUNT(DISTINCT pms.match_id) as total_matches,
                SUM(pms.runs_scored) as runs_scored,
                SUM(pms.balls_faced) as balls_faced,
                SUM(pms.overs_bowled) as overs_bowled,
                SUM(pms.wickets_taken) as wickets_taken,
                SUM(pms.stumpings) as stumpings,
                AVG(CASE WHEN pms.batting_position > 0 THEN pms.batting_position END) as avg_position
            FROM players p
            JOIN player_match_stats pms ON p.player_id = pms.player_id
            JOIN matches m ON pms.match_id = m.match_id
            WHERE m.match_type = 'T20' AND m.gender = ?
            GROUP BY p.player_id
            HAVING total_matches > 2
            ORDER BY total_matches DESC
            LIMIT 50
        """, (gender,))
        
        players_data = cursor.fetchall()
        conn.close()
        
        squad = []
        for row in players_data:
            player_dict = dict(row)
            stats = {
                'total_matches': player_dict.get('total_matches', 0),
                'runs_scored': player_dict.get('runs_scored', 0),
                'balls_faced': player_dict.get('balls_faced', 0),
                'overs_bowled': player_dict.get('overs_bowled', 0),
                'wickets_taken': player_dict.get('wickets_taken', 0),
                'stumpings': player_dict.get('stumpings', 0),
                'avg_batting_position': player_dict.get('avg_position', 5.5)
            }
            
            role = infer_role_from_stats(stats)
            role_category = categorize_role(role)
            
            squad.append({
                'player_id': player_dict['player_id'],
                'name': player_dict['name'],
                'role': get_role_display_name(role),
                'role_category': role_category.value,
                'is_bowling_option': is_bowling_option(role),
                'stats': {
                    'runs': stats['runs_scored'],
                    'wickets': stats['wickets_taken'],
                    'matches': stats['total_matches']
                },
                'source': 'fallback'
            })
        
        # Optimize XI from available players
        optimized_xi = optimize_xi_from_squad(squad)
        validation = validate_team(optimized_xi)
        
        logger.info(f"Optimized XI from {len(squad)} available players")
        
        return jsonify({
            'success': True,
            'players': optimized_xi,
                'source': 'fallback',
                'confidence': 'low',
            'alternatives': [p for p in squad if p not in optimized_xi],
            'validation': {
                'is_valid': validation.is_valid,
                'errors': validation.validation_errors,
                'warnings': validation.warnings
            }
        })
    
    except Exception as e:
        logger.error(f"Error in waterfall selection: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/team/optimize', methods=['POST'])
def optimize_team_from_squad():
    """
    Optimize XI selection from a squad of players.
    
    Request JSON:
    {
        "squad_player_ids": [1, 2, 3, ...],
        "team_id": 123,
        "gender": "male"
    }
    
    Response:
    {
        "success": true,
        "optimized_xi": [...11 players...],
        "validation": {...},
        "balance_score": {...}
    }
    """
    try:
        from src.utils.team_optimizer import optimize_xi_from_squad, validate_team, get_team_balance_score
        
        data = request.get_json()
        squad_ids = data.get('squad_player_ids', [])
        team_id = data.get('team_id')
        gender = data.get('gender', 'male')
        
        if not squad_ids:
            return jsonify({'success': False, 'error': 'No squad player IDs provided'}), 400
        
        # Fetch full player data for the squad
        # We'll use the existing /api/players endpoint logic
        from src.utils.role_classifier import (
            infer_role_from_stats, categorize_role, 
            get_role_display_name, is_bowling_option
        )
        
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get stats for all squad players
        placeholders = ','.join('?' * len(squad_ids))
        cursor.execute(f"""
            SELECT 
                p.player_id,
                p.name,
                COUNT(DISTINCT pms.match_id) as total_matches,
                SUM(pms.runs_scored) as runs_scored,
                SUM(pms.balls_faced) as balls_faced,
                SUM(pms.overs_bowled) as overs_bowled,
                SUM(pms.wickets_taken) as wickets_taken,
                SUM(pms.stumpings) as stumpings,
                AVG(CASE WHEN pms.batting_position > 0 THEN pms.batting_position END) as avg_batting_position
            FROM players p
            JOIN player_match_stats pms ON p.player_id = pms.player_id
            JOIN matches m ON pms.match_id = m.match_id
            WHERE p.player_id IN ({placeholders}) AND m.match_type = 'T20' AND m.gender = ?
            GROUP BY p.player_id
        """, squad_ids + [gender])
        
        squad_data = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        # Build squad list with roles
        squad = []
        for row in squad_data:
            player_dict = dict(row)
            stats = {
                'total_matches': player_dict.get('total_matches', 0),
                'runs_scored': player_dict.get('runs_scored', 0),
                'balls_faced': player_dict.get('balls_faced', 0),
                'overs_bowled': player_dict.get('overs_bowled', 0),
                'wickets_taken': player_dict.get('wickets_taken', 0),
                'stumpings': player_dict.get('stumpings', 0),
                'avg_batting_position': player_dict.get('avg_batting_position', 5.5)
            }
            
            role = infer_role_from_stats(stats)
            role_category = categorize_role(role)
            
            squad.append({
                'player_id': player_dict['player_id'],
                'name': player_dict['name'],
                'role': get_role_display_name(role),
                'role_category': role_category.value,
                'is_bowling_option': is_bowling_option(role),
                'stats': {
                    'matches': stats['total_matches'],
                    'runs': stats['runs_scored'],
                    'wickets': stats['wickets_taken']
                }
            })
        
        # Optimize XI from squad
        optimized_xi = optimize_xi_from_squad(squad)
        
        # Ensure we have exactly 11 players
        if len(optimized_xi) != 11:
            logger.warning(f"Optimizer returned {len(optimized_xi)} players, expected 11. Using first 11.")
            optimized_xi = optimized_xi[:11]
        
        # Validate the result
        validation = validate_team(optimized_xi)
        balance = get_team_balance_score(optimized_xi)
        
        return jsonify({
            'success': True,
            'optimized_xi': optimized_xi,
            'validation': {
                'is_valid': validation.is_valid,
                'errors': validation.validation_errors,
                'warnings': validation.warnings,
                'breakdown': {
                    'keepers': len(validation.keepers),
                    'batters': len(validation.batters),
                    'allrounders': len(validation.allrounders),
                    'bowlers': len(validation.bowlers),
                    'bowling_options': validation.total_bowling_options
                }
            },
            'balance_score': balance
        })
    
    except Exception as e:
        logger.error(f"Error optimizing team: {e}")
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
    """Get team ELO rankings with filters and tier information.
    
    Query parameters:
        format: 'T20' or 'ODI' (default: 'T20')
        gender: 'male' or 'female' (default: 'male')
        tier: Filter by tier (1-5, optional)
        min_matches: Minimum matches played (default: 10)
        month: Historical month 'YYYY-MM' or None for current (default: None)
        limit: Max results to return (default: 50)
    """
    try:
        # Get parameters
        format_type = request.args.get('format', 'T20')
        gender = request.args.get('gender', 'male')
        tier_filter = request.args.get('tier', type=int)  # Optional tier filter
        min_matches = int(request.args.get('min_matches', 10))
        month = request.args.get('month')  # YYYY-MM or None for current
        limit = int(request.args.get('limit', 50))
        
        conn = get_connection()
        cursor = conn.cursor()
        
        # Build tier filter clause
        tier_clause = f"AND t.tier = {tier_filter}" if tier_filter else ""
        
        if month:
            # Get historical data for specific month (end of month snapshot)
            cursor.execute(f"""
                SELECT t.team_id, t.name, t.tier, h.elo, 
                       (SELECT COUNT(*) FROM matches m 
                        WHERE (m.team1_id = t.team_id OR m.team2_id = t.team_id)
                          AND m.match_type = ? AND m.gender = ?
                          AND m.date <= date(? || '-01', '+1 month', '-1 day')) as match_count,
                       NULL as elo_change_30d,
                       (SELECT MAX(date) FROM team_elo_history h2 
                        WHERE h2.team_id = t.team_id AND h2.format = ? AND h2.gender = ?
                          AND h2.date <= date(? || '-01', '+1 month', '-1 day')) as last_match_date
                FROM team_elo_history h
                JOIN teams t ON h.team_id = t.team_id
                WHERE h.format = ? AND h.gender = ?
                  AND strftime('%Y-%m', h.date) = ?
                  {tier_clause}
                GROUP BY t.team_id
                HAVING match_count >= ?
                ORDER BY h.elo DESC
                LIMIT ?
            """, (format_type, gender, month, format_type, gender, month, 
                  format_type, gender, month, min_matches, limit))
        else:
            # Get current ELO with tier information and 30-day change
            elo_col = f"elo_{format_type.lower()}_{gender}"
            cursor.execute(f"""
                SELECT t.team_id, t.name, t.tier, e.{elo_col} as elo,
                       (SELECT COUNT(*) FROM matches m 
                        WHERE (m.team1_id = t.team_id OR m.team2_id = t.team_id)
                          AND m.match_type = ? AND m.gender = ?) as match_count,
                       (SELECT ROUND(e.{elo_col} - h_30d.elo, 1)
                        FROM team_elo_history h_30d
                        WHERE h_30d.team_id = t.team_id 
                          AND h_30d.format = ? AND h_30d.gender = ?
                          AND h_30d.date <= date('now', '-30 days')
                        ORDER BY h_30d.date DESC
                        LIMIT 1) as elo_change_30d,
                       e.last_{format_type.lower()}_{gender}_date as last_match_date
                FROM team_current_elo e
                JOIN teams t ON e.team_id = t.team_id
                WHERE e.{elo_col} IS NOT NULL AND e.{elo_col} != 1500
                  {tier_clause}
                GROUP BY t.team_id
                HAVING match_count >= ?
                ORDER BY e.{elo_col} DESC
                LIMIT ?
            """, (format_type, gender, format_type, gender, min_matches, limit))
        
        rankings = []
        for idx, row in enumerate(cursor.fetchall(), 1):
            rankings.append({
                'rank': idx,
                'team_id': row[0],
                'name': row[1],
                'tier': row[2] or 3,  # Default to tier 3 if NULL
                'elo': round(row[3], 1),
                'matches': row[4],
                'elo_change_30d': row[5] or 0,
                'last_match_date': row[6]
            })
        
        conn.close()
        
        return jsonify({
            'success': True,
            'rankings': rankings,
            'filters': {
                'format': format_type,
                'gender': gender,
                'tier': tier_filter,
                'month': month
            }
        })
    
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


@app.route('/api/rankings/tier-stats', methods=['GET'])
def get_tier_statistics():
    """Get statistics about team tier distribution."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get team counts by tier
        cursor.execute("""
            SELECT tier, COUNT(*) as team_count
            FROM teams
            WHERE tier IS NOT NULL
            GROUP BY tier
            ORDER BY tier
        """)
        
        tier_counts = {}
        tier_names = {
            1: "Elite Full Members",
            2: "Full Members",
            3: "Top Associates/Premier Franchises",
            4: "Associates/Regional",
            5: "Emerging/Domestic"
        }
        
        for row in cursor.fetchall():
            tier = row[0]
            tier_counts[tier] = {
                'tier': tier,
                'name': tier_names.get(tier, f"Tier {tier}"),
                'team_count': row[1]
            }
        
        # Get promotion flags count
        cursor.execute("SELECT COUNT(*) FROM promotion_review_flags WHERE reviewed = FALSE")
        pending_flags = cursor.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            'success': True,
            'tier_counts': list(tier_counts.values()),
            'pending_promotion_flags': pending_flags
        })
    
    except Exception as e:
        logger.error(f"Error fetching tier statistics: {e}")
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


def get_espn_scraper():
    """Get or initialize the ESPN scraper."""
    global _espn_scraper
    if _espn_scraper is None:
        from src.api.espn_scraper import ESPNCricInfoScraper
        _espn_scraper = ESPNCricInfoScraper(request_delay=1.0)
        logger.info("ESPN Cricinfo scraper initialized")
    return _espn_scraper


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
        
        # Always fetch fresh from ESPN
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
        
        # Always fetch fresh from ESPN (no caching)
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
            # Always try to match players, even if team doesn't match
            # For franchise teams like IPL/BBL, players are often international players in our database
            team_name_for_matching = team_match[1] if team_match else None
            scraper.match_players_to_db(match.team1, team_name_for_matching, match.gender)
        
        if match.team2:
            team_match = scraper.match_team_to_db(match.team2, match.gender)
            if team_match:
                team2_db = {'team_id': team_match[0], 'name': team_match[1]}
            # Always try to match players, even if team doesn't match
            team_name_for_matching = team_match[1] if team_match else None
            scraper.match_players_to_db(match.team2, team_name_for_matching, match.gender)
        
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
        
        # Check if playing XI is available (for now, always False - can be enhanced later)
        has_playing_xi = False  # ESPN doesn't always provide confirmed XI for upcoming matches
        
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
                'has_playing_xi': has_playing_xi,
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
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error fetching ESPN match: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# CREX Endpoints (Web Scraping - replaces ESPN)
# ============================================================================

_crex_scraper = None


def get_crex_scraper():
    """Get or initialize the CREX scraper."""
    global _crex_scraper
    if _crex_scraper is None:
        from src.api.crex_scraper import CREXScraper
        _crex_scraper = CREXScraper(request_delay=1.0)
        logger.info("CREX scraper initialized")
    return _crex_scraper


@app.route('/api/crex/upcoming', methods=['GET'])
def get_crex_upcoming():
    """
    Get upcoming T20 matches from CREX.
    
    Query params:
        format: 'T20' or 'ODI' (default 'T20')
        
    Returns:
        List of upcoming matches with basic info, grouped by series
    """
    try:
        format_type = request.args.get('format', 'T20')
        formats = [format_type] if format_type else None
        
        scraper = get_crex_scraper()
        matches = scraper.get_schedule(formats=formats)
        
        # Group by series
        series_dict = {}
        for m in matches:
            series_key = m.series_name or 'Other'
            if series_key not in series_dict:
                series_dict[series_key] = {
                    'series_name': m.series_name,
                    'series_id': m.series_id,
                    'gender': m.gender,
                    'matches': []
                }
            
            series_dict[series_key]['matches'].append({
                'crex_id': m.crex_id,
                'title': m.title,
                'team1': m.team1_name,
                'team2': m.team2_name,
                'team1_id': m.team1_id,
                'team2_id': m.team2_id,
                'match_type': m.match_type,
                'format': m.format_type,
                'slug': m.slug,
                'status': m.status,
                'start_date': m.start_date,
                'start_time': m.start_time,
                'date_time_gmt': m.date_time_gmt,
                'match_url': m.match_url,
                'gender': m.gender
            })
        
        result = {
            'success': True,
            'source': 'crex',
            'matches_by_series': list(series_dict.values()),
            'total_matches': len(matches)
        }
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error fetching CREX upcoming matches: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/crex/match', methods=['GET'])
def get_crex_match():
    """
    Get full match details from CREX including venue and squads.
    
    Query params:
        url: The match info URL from CREX
        
    Returns:
        Match details with venue, venue stats, and squad data matched to database
    """
    try:
        match_url = request.args.get('url')
        if not match_url:
            return jsonify({'success': False, 'error': 'url parameter required'}), 400
        
        scraper = get_crex_scraper()
        match = scraper.get_match_details(match_url)
        
        if not match:
            return jsonify({'success': False, 'error': 'Failed to fetch match from CREX'}), 404
        
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
            # Always try to match players, even if team doesn't match
            # For franchise teams like SA20/IPL, players are often international players in our database
            team_name_for_matching = team_match[1] if team_match else None
            scraper.match_players_to_db(match.team1, team_name_for_matching, match.gender)
        
        if match.team2:
            team_match = scraper.match_team_to_db(match.team2, match.gender)
            if team_match:
                team2_db = {'team_id': team_match[0], 'name': team_match[1]}
            # Always try to match players, even if team doesn't match
            team_name_for_matching = team_match[1] if team_match else None
            scraper.match_players_to_db(match.team2, team_name_for_matching, match.gender)
        
        def team_to_dict(team):
            if not team:
                return None
            return {
                'crex_id': team.crex_id,
                'name': team.name,
                'abbreviation': team.abbreviation,
                'db_team_id': team.db_team_id,
                'players': [
                    {
                        'crex_id': p.crex_id,
                        'name': p.name,
                        'short_name': p.short_name,
                        'role': p.role,
                        'is_captain': p.is_captain,
                        'is_wicketkeeper': p.is_wicketkeeper,
                        'is_overseas': p.is_overseas,
                        'db_player_id': p.db_player_id
                    }
                    for p in team.players
                ]
            }
        
        def venue_stats_to_dict(stats):
            if not stats:
                return None
            return {
                'matches_played': stats.matches_played,
                'win_bat_first_pct': stats.win_bat_first_pct,
                'win_bowl_first_pct': stats.win_bowl_first_pct,
                'avg_first_innings': stats.avg_first_innings,
                'avg_second_innings': stats.avg_second_innings,
                'pace_wickets': stats.pace_wickets,
                'spin_wickets': stats.spin_wickets,
                'pace_pct': stats.pace_pct,
                'spin_pct': stats.spin_pct
            }
        
        result = {
            'success': True,
            'source': 'crex',
            'match': {
                'crex_id': match.crex_id,
                'title': match.title,
                'series_name': match.series_name,
                'series_id': match.series_id,
                'match_type': match.match_type,
                'format': match.format_type,
                'status': match.status,
                'start_date': match.start_date,
                'start_time': match.start_time,
                'date_time_gmt': match.date_time_gmt,
                'gender': match.gender,
                'has_squads': match.has_squads,
                'venue': {
                    'name': match.venue.name if match.venue else None,
                    'city': match.venue.city if match.venue else None,
                    'db_venue_id': venue_db['venue_id'] if venue_db else None,
                    'db_venue_name': venue_db['name'] if venue_db else None
                } if match.venue else None,
                'venue_stats': venue_stats_to_dict(match.venue_stats),
                'team1': team_to_dict(match.team1),
                'team2': team_to_dict(match.team2),
                'team1_db': team1_db,
                'team2_db': team2_db,
                'toss_winner': match.toss_winner,
                'toss_decision': match.toss_decision
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error fetching CREX match: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/crex/live', methods=['GET'])
def get_crex_live_match():
    """
    Get live match details from CREX including toss and playing XI.
    
    Query params:
        url: The match URL from CREX
        
    Returns:
        Match details with toss info and playing XI if available
    """
    try:
        match_url = request.args.get('url')
        if not match_url:
            return jsonify({'success': False, 'error': 'url parameter required'}), 400
        
        scraper = get_crex_scraper()
        match = scraper.get_live_match(match_url)
        
        if not match:
            return jsonify({'success': False, 'error': 'Failed to fetch live match from CREX'}), 404
        
        result = {
            'success': True,
            'source': 'crex',
            'match': {
                'crex_id': match.crex_id,
                'title': match.title,
                'status': match.status,
                'toss_winner': match.toss_winner,
                'toss_decision': match.toss_decision,
                'playing_xi_available': match.playing_xi_available
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error fetching CREX live match: {e}")
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


@app.route('/api/training/reset-database', methods=['POST'])
def reset_database():
    """
    Reset the database (with backup) for a fresh start.
    
    Request body:
        {
            "confirm": true  # Must be true to proceed
        }
        
    Returns:
        JSON with success status
    """
    try:
        from pathlib import Path
        from datetime import datetime
        import shutil
        
        data = request.get_json() or {}
        
        if not data.get('confirm'):
            return jsonify({'success': False, 'error': 'Confirmation required'}), 400
        
        db_path = DATABASE_PATH
        
        # Create backup
        backup_name = f"cricket_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        backup_path = db_path.parent / 'backups' / backup_name
        backup_path.parent.mkdir(exist_ok=True)
        
        shutil.copy2(db_path, backup_path)
        logger.info(f"Database backed up to {backup_path}")
        
        # Delete current database
        db_path.unlink()
        logger.info(f"Deleted database: {db_path}")
        
        # Reinitialize with fresh schema
        from src.data.database import init_database, init_model_versions_table
        init_database()
        init_model_versions_table()
        
        logger.info("Database reset complete")
        
        return jsonify({
            'success': True,
            'message': 'Database reset successfully',
            'backup': str(backup_path)
        })
        
    except Exception as e:
        logger.error(f"Error resetting database: {e}")
        import traceback
        traceback.print_exc()
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
            """Download and ingest data with progress reporting."""
            from src.utils.job_manager import update_job_progress
            import threading
            
            # Get current job ID
            job_id = getattr(threading.current_thread(), 'job_id', None)
            
            logger.info(f"Starting data download for formats: {formats}")
            
            # Download - 0-50%
            if job_id:
                update_job_progress(job_id, 5, f"Downloading {', '.join(formats)} from Cricsheet...")
            
            download_result = download_cricsheet_data(formats=formats, force_download=force_download)
            
            if job_id:
                update_job_progress(job_id, 50, "Download complete. Starting data ingestion...")
            
            # Ingest - 50-100%
            logger.info("Starting data ingestion...")
            ingest_result = ingest_matches(formats=formats)
            
            if job_id:
                update_job_progress(job_id, 100, "Ingestion complete!")
            
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
            from src.utils.job_manager import update_job_progress, get_job_status
            import threading
            
            # Get current job ID from thread-local storage
            job_id = getattr(threading.current_thread(), 'job_id', None)
            
            logger.info(f"Starting model retrain: mode={mode}, genders={genders}, job_id={job_id}")
            
            skip_ingest = (mode == 'quick')
            skip_elo = (mode == 'quick')
            male_only = ('male' in genders and 'female' not in genders)
            female_only = ('female' in genders and 'male' not in genders)
            
            result = run_full_pipeline(
                skip_ingest=skip_ingest,
                skip_elo=skip_elo,
                male_only=male_only,
                female_only=female_only,
                progress_callback=lambda pct, step: update_job_progress(job_id, pct, step) if job_id else None
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
        
        # Convert any non-JSON-serializable objects for JSON serialization
        def make_json_serializable(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, (dict,)):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_serializable(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                # Complex object - convert to string representation
                return str(type(obj).__name__)
            else:
                return obj
        
        job_info = make_json_serializable(job_info)
        
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
# Admin: Promotion Review API Endpoints
# ============================================================================

@app.route('/api/admin/promotion-flags', methods=['GET'])
def get_promotion_flags():
    """
    Get all pending promotion review flags.
    
    Returns:
        JSON with list of teams flagged for tier review
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                prf.*,
                t.name as team_name
            FROM promotion_review_flags prf
            JOIN teams t ON prf.team_id = t.team_id
            WHERE prf.reviewed = FALSE
            ORDER BY prf.flagged_date DESC
        """)
        
        flags = []
        for row in cursor.fetchall():
            flags.append({
                'flag_id': row['flag_id'],
                'team_id': row['team_id'],
                'team_name': row['team_name'],
                'format': row['format'],
                'gender': row['gender'],
                'current_tier': row['current_tier'],
                'suggested_tier': row['suggested_tier'],
                'trigger_reason': row['trigger_reason'],
                'current_elo': row['current_elo'],
                'months_at_ceiling': row['months_at_ceiling'],
                'cross_tier_record': row['cross_tier_record'],
                'flagged_date': row['flagged_date']
            })
        
        conn.close()
        return jsonify({'success': True, 'flags': flags})
    
    except Exception as e:
        logger.error(f"Error getting promotion flags: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/admin/promotion-flags/<int:flag_id>/approve', methods=['POST'])
def approve_promotion(flag_id):
    """
    Approve a promotion/demotion and update team tier.
    
    Args:
        flag_id: Promotion flag ID
        
    Returns:
        JSON with success status
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get flag details
        cursor.execute("""
            SELECT * FROM promotion_review_flags WHERE flag_id = ?
        """, (flag_id,))
        flag = cursor.fetchone()
        
        if not flag:
            conn.close()
            return jsonify({'success': False, 'error': 'Flag not found'}), 404
        
        # Update team tier
        cursor.execute("""
            UPDATE teams 
            SET tier = ?, 
                tier_last_reviewed = CURRENT_DATE, 
                tier_notes = ?
            WHERE team_id = ?
        """, (
            flag['suggested_tier'],
            f"Promoted/relegated from tier {flag['current_tier']} on {datetime.now().strftime('%Y-%m-%d')}",
            flag['team_id']
        ))
        
        # Mark flag as reviewed
        cursor.execute("""
            UPDATE promotion_review_flags
            SET reviewed = TRUE, 
                reviewed_date = CURRENT_DATE,
                reviewer_notes = 'Approved - tier changed'
            WHERE flag_id = ?
        """, (flag_id,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Approved promotion flag {flag_id}: team {flag['team_id']} tier {flag['current_tier']} → {flag['suggested_tier']}")
        
        return jsonify({
            'success': True, 
            'message': f'Team tier updated from {flag["current_tier"]} to {flag["suggested_tier"]}'
        })
    
    except Exception as e:
        logger.error(f"Error approving promotion: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/admin/promotion-flags/<int:flag_id>/reject', methods=['POST'])
def reject_promotion(flag_id):
    """
    Reject a promotion flag without changing team tier.
    
    Args:
        flag_id: Promotion flag ID
        
    Request body:
        {
            "notes": "Reason for rejection"
        }
        
    Returns:
        JSON with success status
    """
    try:
        data = request.get_json() or {}
        notes = data.get('notes', 'Rejected by admin')
        
        conn = get_connection()
        cursor = conn.cursor()
        
        # Check flag exists
        cursor.execute("SELECT flag_id FROM promotion_review_flags WHERE flag_id = ?", (flag_id,))
        if not cursor.fetchone():
            conn.close()
            return jsonify({'success': False, 'error': 'Flag not found'}), 404
        
        # Mark flag as reviewed
        cursor.execute("""
            UPDATE promotion_review_flags
            SET reviewed = TRUE, 
                reviewed_date = CURRENT_DATE, 
                reviewer_notes = ?
            WHERE flag_id = ?
        """, (notes, flag_id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Rejected promotion flag {flag_id}: {notes}")
        
        return jsonify({'success': True, 'message': 'Promotion flag rejected'})
    
    except Exception as e:
        logger.error(f"Error rejecting promotion: {e}")
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
