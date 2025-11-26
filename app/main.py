"""
Flask Web Application for Cricket Match Predictor.

Provides a web interface and API for:
- Match predictions
- Team and player ELO rankings
- Match simulation
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FlaskConfig, DATABASE_PATH
from src.data.database import get_db_connection
from src.elo.calculator import EloCalculator
from src.features.engineer import FeatureEngineer
from src.models.simulator import MatchSimulator

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(FlaskConfig)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize components
elo_calculator = EloCalculator()
feature_engineer = FeatureEngineer()
simulator = MatchSimulator()


# ============================================================================
# Web Routes
# ============================================================================

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')


@app.route('/predict')
def predict_page():
    """Match prediction page."""
    return render_template('predict.html')


@app.route('/rankings')
def rankings_page():
    """ELO rankings page."""
    return render_template('rankings.html')


@app.route('/history')
def history_page():
    """Match history page."""
    return render_template('history.html')


# ============================================================================
# API Routes
# ============================================================================

@app.route('/api/teams', methods=['GET'])
def get_teams():
    """Get list of all teams with current ELO ratings."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    t.team_id,
                    t.name,
                    COALESCE(e.elo_t20, 1500) as elo_t20,
                    COALESCE(e.elo_odi, 1500) as elo_odi
                FROM teams t
                LEFT JOIN team_current_elo e ON t.team_id = e.team_id
                ORDER BY COALESCE(e.elo_t20, 1500) DESC
            """)
            
            teams = [dict(row) for row in cursor.fetchall()]
            
            return jsonify({'success': True, 'teams': teams})
    
    except Exception as e:
        logger.error(f"Error fetching teams: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/players', methods=['GET'])
def get_players():
    """Get list of players, optionally filtered by team."""
    team_id = request.args.get('team_id', type=int)
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            if team_id:
                # Get players who have played for this team
                cursor.execute("""
                    SELECT DISTINCT 
                        p.player_id,
                        p.name,
                        p.country,
                        COALESCE(e.batting_elo_t20, 1500) as batting_elo_t20,
                        COALESCE(e.bowling_elo_t20, 1500) as bowling_elo_t20
                    FROM players p
                    JOIN player_match_stats pms ON p.player_id = pms.player_id
                    LEFT JOIN player_current_elo e ON p.player_id = e.player_id
                    WHERE pms.team_id = ?
                    ORDER BY COALESCE(e.batting_elo_t20, 1500) DESC
                    LIMIT 50
                """, (team_id,))
            else:
                cursor.execute("""
                    SELECT 
                        p.player_id,
                        p.name,
                        p.country,
                        COALESCE(e.batting_elo_t20, 1500) as batting_elo_t20,
                        COALESCE(e.bowling_elo_t20, 1500) as bowling_elo_t20
                    FROM players p
                    LEFT JOIN player_current_elo e ON p.player_id = e.player_id
                    ORDER BY COALESCE(e.batting_elo_t20, 1500) DESC
                    LIMIT 100
                """)
            
            players = [dict(row) for row in cursor.fetchall()]
            
            return jsonify({'success': True, 'players': players})
    
    except Exception as e:
        logger.error(f"Error fetching players: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/rankings/teams', methods=['GET'])
def get_team_rankings():
    """Get team ELO rankings."""
    format_type = request.args.get('format', 'T20')
    limit = request.args.get('limit', 20, type=int)
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            elo_col = 'elo_t20' if format_type == 'T20' else 'elo_odi'
            
            cursor.execute(f"""
                SELECT 
                    t.team_id,
                    t.name,
                    e.{elo_col} as elo
                FROM team_current_elo e
                JOIN teams t ON e.team_id = t.team_id
                ORDER BY e.{elo_col} DESC
                LIMIT ?
            """, (limit,))
            
            rankings = [dict(row) for row in cursor.fetchall()]
            
            return jsonify({
                'success': True,
                'format': format_type,
                'rankings': rankings
            })
    
    except Exception as e:
        logger.error(f"Error fetching team rankings: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/rankings/players/batting', methods=['GET'])
def get_batting_rankings():
    """Get player batting ELO rankings."""
    format_type = request.args.get('format', 'T20')
    limit = request.args.get('limit', 20, type=int)
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            suffix = '_t20' if format_type == 'T20' else '_odi'
            
            cursor.execute(f"""
                SELECT 
                    p.player_id,
                    p.name,
                    p.country,
                    e.batting_elo{suffix} as elo
                FROM player_current_elo e
                JOIN players p ON e.player_id = p.player_id
                WHERE e.batting_elo{suffix} != 1500
                ORDER BY e.batting_elo{suffix} DESC
                LIMIT ?
            """, (limit,))
            
            rankings = [dict(row) for row in cursor.fetchall()]
            
            return jsonify({
                'success': True,
                'format': format_type,
                'rankings': rankings
            })
    
    except Exception as e:
        logger.error(f"Error fetching batting rankings: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/rankings/players/bowling', methods=['GET'])
def get_bowling_rankings():
    """Get player bowling ELO rankings."""
    format_type = request.args.get('format', 'T20')
    limit = request.args.get('limit', 20, type=int)
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            suffix = '_t20' if format_type == 'T20' else '_odi'
            
            cursor.execute(f"""
                SELECT 
                    p.player_id,
                    p.name,
                    p.country,
                    e.bowling_elo{suffix} as elo
                FROM player_current_elo e
                JOIN players p ON e.player_id = p.player_id
                WHERE e.bowling_elo{suffix} != 1500
                ORDER BY e.bowling_elo{suffix} DESC
                LIMIT ?
            """, (limit,))
            
            rankings = [dict(row) for row in cursor.fetchall()]
            
            return jsonify({
                'success': True,
                'format': format_type,
                'rankings': rankings
            })
    
    except Exception as e:
        logger.error(f"Error fetching bowling rankings: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/elo/team/<int:team_id>', methods=['GET'])
def get_team_elo_history(team_id):
    """Get ELO history for a team."""
    format_type = request.args.get('format', 'T20')
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            elo_col = 'elo_t20' if format_type == 'T20' else 'elo_odi'
            
            cursor.execute(f"""
                SELECT date, {elo_col} as elo
                FROM team_elo_history
                WHERE team_id = ?
                ORDER BY date
            """, (team_id,))
            
            history = [{'date': row['date'], 'elo': row['elo']} 
                      for row in cursor.fetchall()]
            
            return jsonify({
                'success': True,
                'team_id': team_id,
                'format': format_type,
                'history': history
            })
    
    except Exception as e:
        logger.error(f"Error fetching team ELO history: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/elo/player/<int:player_id>', methods=['GET'])
def get_player_elo_history(player_id):
    """Get ELO history for a player."""
    format_type = request.args.get('format', 'T20')
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT date, batting_elo, bowling_elo, overall_elo
                FROM player_elo_history
                WHERE player_id = ? AND format = ?
                ORDER BY date
            """, (player_id, format_type))
            
            history = [dict(row) for row in cursor.fetchall()]
            
            return jsonify({
                'success': True,
                'player_id': player_id,
                'format': format_type,
                'history': history
            })
    
    except Exception as e:
        logger.error(f"Error fetching player ELO history: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict_match():
    """
    Predict match outcome.
    
    Expected JSON body:
    {
        "team1_id": int,
        "team2_id": int,
        "format": "T20" or "ODI",
        "venue_id": int (optional),
        "team1_players": [int] (optional),
        "team2_players": [int] (optional)
    }
    """
    try:
        data = request.get_json()
        
        team1_id = data.get('team1_id')
        team2_id = data.get('team2_id')
        match_format = data.get('format', 'T20')
        venue_id = data.get('venue_id')
        team1_players = data.get('team1_players')
        team2_players = data.get('team2_players')
        
        if not team1_id or not team2_id:
            return jsonify({
                'success': False,
                'error': 'team1_id and team2_id are required'
            }), 400
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get team names
            cursor.execute("SELECT name FROM teams WHERE team_id = ?", (team1_id,))
            team1_name = cursor.fetchone()['name']
            
            cursor.execute("SELECT name FROM teams WHERE team_id = ?", (team2_id,))
            team2_name = cursor.fetchone()['name']
            
            # Run simulation
            results = simulator.predict_match(
                conn, team1_id, team2_id, match_format,
                datetime.now(),
                team1_players, team2_players,
                n_simulations=1000
            )
            
            return jsonify({
                'success': True,
                'prediction': {
                    'team1': {
                        'id': team1_id,
                        'name': team1_name,
                        'win_probability': results['team1_win_probability'],
                        'expected_score': results['team1_expected_score'],
                        'score_std': results['team1_score_std']
                    },
                    'team2': {
                        'id': team2_id,
                        'name': team2_name,
                        'win_probability': results['team2_win_probability'],
                        'expected_score': results['team2_expected_score'],
                        'score_std': results['team2_score_std']
                    },
                    'confidence_interval': results['confidence_interval'],
                    'format': match_format,
                    'n_simulations': results['n_simulations']
                }
            })
    
    except Exception as e:
        logger.error(f"Error predicting match: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/simulate', methods=['POST'])
def simulate_match():
    """
    Run detailed match simulation.
    
    Returns ball-by-ball simulation results.
    """
    try:
        data = request.get_json()
        
        team1_id = data.get('team1_id')
        team2_id = data.get('team2_id')
        match_format = data.get('format', 'T20')
        n_simulations = data.get('n_simulations', 100)
        
        with get_db_connection() as conn:
            results = simulator.predict_match(
                conn, team1_id, team2_id, match_format,
                datetime.now(), n_simulations=n_simulations
            )
            
            return jsonify({
                'success': True,
                'simulation': results
            })
    
    except Exception as e:
        logger.error(f"Error simulating match: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/search/players', methods=['GET'])
def search_players():
    """Search players by name."""
    query = request.args.get('q', '')
    
    if len(query) < 2:
        return jsonify({'success': True, 'players': []})
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    p.player_id,
                    p.name,
                    p.country,
                    COALESCE(e.batting_elo_t20, 1500) as batting_elo,
                    COALESCE(e.bowling_elo_t20, 1500) as bowling_elo
                FROM players p
                LEFT JOIN player_current_elo e ON p.player_id = e.player_id
                WHERE p.name LIKE ?
                ORDER BY COALESCE(e.batting_elo_t20, 1500) DESC
                LIMIT 20
            """, (f'%{query}%',))
            
            players = [dict(row) for row in cursor.fetchall()]
            
            return jsonify({'success': True, 'players': players})
    
    except Exception as e:
        logger.error(f"Error searching players: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_database_stats():
    """Get database statistics."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            cursor.execute("SELECT COUNT(*) FROM matches")
            stats['total_matches'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM teams")
            stats['total_teams'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM players")
            stats['total_players'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM deliveries")
            stats['total_deliveries'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT MIN(date), MAX(date) FROM matches")
            row = cursor.fetchone()
            stats['date_range'] = {'min': row[0], 'max': row[1]}
            
            return jsonify({'success': True, 'stats': stats})
    
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'error': 'Not found'}), 404
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    logger.error(f"Server error: {e}")
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'error': 'Internal server error'}), 500
    return render_template('500.html'), 500


# ============================================================================
# Main
# ============================================================================

def main():
    """Run the Flask application."""
    app.run(
        host=FlaskConfig.HOST,
        port=FlaskConfig.PORT,
        debug=FlaskConfig.DEBUG
    )


if __name__ == '__main__':
    main()

