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
    """Get list of all teams for specified gender."""
    try:
        gender = request.args.get('gender', 'male')
        
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT t.team_id, t.name
            FROM teams t
            JOIN matches m ON t.team_id IN (m.team1_id, m.team2_id)
            WHERE m.match_type = 'T20' AND m.gender = ?
            ORDER BY t.name
        """, (gender,))
        
        teams = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return jsonify({'success': True, 'teams': teams, 'gender': gender})
    
    except Exception as e:
        logger.error(f"Error fetching teams: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/venues', methods=['GET'])
def get_venues():
    """Get list of venues with match counts for specified gender."""
    try:
        gender = request.args.get('gender', 'male')
        
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT v.venue_id, v.name, v.city, v.country,
                   COUNT(m.match_id) as match_count
            FROM venues v
            JOIN matches m ON m.venue_id = v.venue_id
            WHERE m.match_type = 'T20' AND m.gender = ?
            GROUP BY v.venue_id
            HAVING match_count >= 5
            ORDER BY match_count DESC
            LIMIT 100
        """, (gender,))
        
        venues = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return jsonify({'success': True, 'venues': venues, 'gender': gender})
    
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
        
        team1_batters = data.get('team1_batters', [])
        team1_bowlers = data.get('team1_bowlers', [])
        team2_batters = data.get('team2_batters', [])
        team2_bowlers = data.get('team2_bowlers', [])
        simulator_type = data.get('simulator', 'fast')
        n_simulations = min(data.get('n_simulations', 1000), 10000)
        venue_id = data.get('venue_id')
        use_toss = data.get('use_toss', False)
        gender = data.get('gender', 'male')
        
        # Validate
        if len(team1_batters) < 11:
            team1_batters.extend([team1_batters[0]] * (11 - len(team1_batters)))
        if len(team2_batters) < 11:
            team2_batters.extend([team2_batters[0]] * (11 - len(team2_batters)))
        if len(team1_bowlers) < 5:
            team1_bowlers.extend([team1_bowlers[0]] * (5 - len(team1_bowlers)))
        if len(team2_bowlers) < 5:
            team2_bowlers.extend([team2_bowlers[0]] * (5 - len(team2_bowlers)))
        
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
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error simulating match: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


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
# WBBL Fixture Endpoints
# ============================================================================

_lineup_service = None

def get_lineup_service():
    """Get or initialize the lineup service."""
    global _lineup_service
    if _lineup_service is None:
        from src.features.lineup_service import LineupService
        _lineup_service = LineupService()
        logger.info("Lineup service initialized")
    return _lineup_service


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
    """Get full match data with squads."""
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
                'team1_suggested_xi': service.suggest_lineup(match.team1_squad, match.team1_recent_xi),
                'team2_suggested_xi': service.suggest_lineup(match.team2_squad, match.team2_recent_xi)
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
