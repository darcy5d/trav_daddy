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
# Disable XLA JIT compilation to avoid Apple Silicon Metal backend issues
# Must be set BEFORE importing TensorFlow
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'

# Configure TensorFlow threading BEFORE importing TensorFlow
# This must happen before any module imports TensorFlow
import multiprocessing
N_CPU_CORES = multiprocessing.cpu_count()

import logging
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Any, Tuple

# Configure TensorFlow early (before any imports that might use it)
import tensorflow as tf
try:
    tf.config.threading.set_inter_op_parallelism_threads(max(4, N_CPU_CORES // 2))
    tf.config.threading.set_intra_op_parallelism_threads(max(4, N_CPU_CORES // 2))
except RuntimeError:
    pass  # Already configured

# Disable XLA JIT at the optimizer level (Apple Silicon Metal + XLA can crash)
try:
    tf.config.optimizer.set_jit(False)
except (AttributeError, RuntimeError):
    pass

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATABASE_PATH, CRICSHEET_MATCHES_URL, FlaskConfig
from src.data.database import get_connection, get_db_connection
from src.features.toss_stats import TossSimulator
from src.api.crex_scraper import format_type_to_model_format
from src.integrations.credentials import get_market_credentials_status
from src.integrations.polymarket import PolymarketClient
from src.integrations.odds import PolymarketComparisonService

# Initialize Flask app
app = Flask(__name__)
app.secret_key = FlaskConfig.SECRET_KEY
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Run lightweight schema migrations on import so every entry point (web,
# scripts importing app.main) sees the franchise-grouping tables and columns.
# This is idempotent; safe to call repeatedly.
try:
    from src.data.database import init_franchise_tables as _init_franchise_tables

    _init_franchise_tables()
except Exception as _exc:  # pragma: no cover - logged but non-fatal at import time
    logger.warning(f"Franchise schema init failed at startup: {_exc}")

# Wave 5 Phase 6b: bet ledger schema (V5).
try:
    from src.data.database import init_betting_tables as _init_betting_tables

    _init_betting_tables()
except Exception as _exc:  # pragma: no cover - logged but non-fatal at import time
    logger.warning(f"Betting schema init failed at startup: {_exc}")

# Wave 5.13: order_history audit table + bet_ledger reason/category columns.
try:
    from src.data.database import init_order_history as _init_order_history

    _init_order_history()
except Exception as _exc:  # pragma: no cover - logged but non-fatal at import time
    logger.warning(f"Order history schema init failed at startup: {_exc}")

# Lazy load simulators (they take time to initialize)
# Cache per gender to avoid re-initializing
_fast_simulators = {}  # {'male': simulator, 'female': simulator}
_nn_simulators = {}
_toss_simulator = None

# Cricsheet source status cache (TTL 20 minutes)
_source_status_cache = None
_source_status_cache_expires = None
SOURCE_STATUS_CACHE_TTL_SECONDS = 20 * 60
_polymarket_client = None
_polymarket_compare_service = None


def get_fast_simulator(gender: str = 'male'):
    """Get or initialize the fast lookup simulator for specified gender."""
    global _fast_simulators
    if gender not in _fast_simulators:
        from src.models.fast_lookup_sim import FastLookupSimulator
        _fast_simulators[gender] = FastLookupSimulator(use_h2h=True, gender=gender)
        logger.info(f"Fast lookup simulator initialized for {gender}")
    return _fast_simulators[gender]


def get_nn_simulator(gender: str = 'male', format_type: str = 'T20'):
    """Get or initialize the NN simulator for the specified gender and format."""
    global _nn_simulators
    key = (gender, format_type.upper())
    if key not in _nn_simulators:
        from src.models.vectorized_nn_sim import VectorizedNNSimulator
        _nn_simulators[key] = VectorizedNNSimulator(gender=gender, format_type=format_type.upper())
        logger.info(f"NN simulator initialized for {gender}/{format_type.upper()}")
    return _nn_simulators[key]


def get_toss_simulator():
    """Get or initialize the toss simulator."""
    global _toss_simulator
    if _toss_simulator is None:
        _toss_simulator = TossSimulator()
        logger.info("Toss simulator initialized")
    return _toss_simulator


def get_polymarket_client() -> PolymarketClient:
    """Get or initialize Polymarket client."""
    global _polymarket_client
    if _polymarket_client is None:
        _polymarket_client = PolymarketClient()
        logger.info("Polymarket client initialized")
    return _polymarket_client


def get_polymarket_compare_service() -> PolymarketComparisonService:
    """Get or initialize Polymarket comparison service."""
    global _polymarket_compare_service
    if _polymarket_compare_service is None:
        _polymarket_compare_service = PolymarketComparisonService(client=get_polymarket_client())
        logger.info("Polymarket comparison service initialized")
    return _polymarket_compare_service


def has_active_model(gender: str, format_type: str) -> bool:
    """Return True if an active model exists for (gender, format_type)."""
    from src.data.database import get_model_versions, init_model_versions_table
    init_model_versions_table()
    fmt = (format_type or 'T20').upper()
    models = get_model_versions(gender=gender, format_type=fmt, active_only=True)
    return len(models) > 0


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


@app.route('/bulk-predict')
def bulk_predict_page():
    """Bulk match prediction page."""
    return render_template('bulk_predict.html')


@app.route('/rankings')
def rankings_page():
    """ELO rankings page."""
    return render_template('rankings.html')


@app.route('/data-explorer')
def data_explorer_page():
    """Venue data explorer page."""
    return render_template('data_explorer.html')


@app.route('/team-explorer')
def team_explorer_page():
    """Team / franchise explorer page (V4 franchise unification UI)."""
    return render_template('team_explorer.html')


# ============================================================================
# API Routes
# ============================================================================


def _fetch_venue_quality_rows(gender: str = 'male', min_matches: int = 0) -> List[Dict[str, Any]]:
    """
    Fetch venue rows with match counts for quality analysis.
    Uses LEFT JOIN to retain venues with zero matches when needed.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(venues)")
    venue_columns = {row["name"] for row in cursor.fetchall()}
    has_state_column = "state" in venue_columns

    state_select = "v.state AS state," if has_state_column else "'' AS state,"
    state_group = ", v.state" if has_state_column else ""
    state_order = ", v.state" if has_state_column else ""

    cursor.execute(
        f"""
        SELECT
            v.venue_id,
            v.name,
            v.city,
            v.country,
            {state_select}
            v.canonical_name,
            v.region,
            COUNT(m.match_id) AS match_count
        FROM venues v
        LEFT JOIN matches m
            ON m.venue_id = v.venue_id
           AND m.match_type = 'T20'
           AND m.gender = ?
        GROUP BY v.venue_id, v.name, v.city, v.country{state_group}, v.canonical_name, v.region
        HAVING COUNT(m.match_id) >= ?
        ORDER BY match_count DESC, v.country{state_order}, v.city, v.name
        """,
        (gender, max(0, min_matches)),
    )

    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def _build_venue_hierarchy(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build Country -> State -> Ground hierarchy for explorer UI."""
    hierarchy: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        country = row.get("country") or "Unknown"
        state = row.get("state") or "Unknown"
        city = row.get("city") or "Unknown"

        if country not in hierarchy:
            hierarchy[country] = {"name": country, "states": {}}
        if state not in hierarchy[country]["states"]:
            hierarchy[country]["states"][state] = {"name": state, "grounds": []}

        hierarchy[country]["states"][state]["grounds"].append(
            {
                "venue_id": row["venue_id"],
                "name": row["canonical_name"] or row["name"],
                "original_name": row["name"],
                "city": city,
                "match_count": row["match_count"],
            }
        )

    result = []
    for country_name in sorted(hierarchy.keys()):
        states = hierarchy[country_name]["states"]
        state_list = []
        for state_name in sorted(states.keys()):
            grounds = sorted(states[state_name]["grounds"], key=lambda g: (-g["match_count"], g["name"]))
            state_list.append({"name": state_name, "grounds": grounds})
        result.append({"name": country_name, "states": state_list})

    return result


def _find_venue_duplicates(
    rows: List[Dict[str, Any]], similarity_threshold: float = 0.9
) -> List[Dict[str, Any]]:
    """Find fuzzy duplicate candidates within country/state buckets."""
    from src.data.venue_normalizer import venue_similarity

    by_bucket: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        key = (
            (row.get("country") or "").strip().lower(),
            (row.get("state") or "").strip().lower(),
        )
        by_bucket.setdefault(key, []).append(row)

    pairs: List[Dict[str, Any]] = []
    for bucket_rows in by_bucket.values():
        n = len(bucket_rows)
        for i in range(n):
            for j in range(i + 1, n):
                a = bucket_rows[i]
                b = bucket_rows[j]
                score = venue_similarity(a["name"], b["name"])
                if score >= similarity_threshold:
                    pairs.append(
                        {
                            "score": round(score, 3),
                            "venue_id_1": a["venue_id"],
                            "name_1": a["name"],
                            "match_count_1": a["match_count"],
                            "venue_id_2": b["venue_id"],
                            "name_2": b["name"],
                            "match_count_2": b["match_count"],
                            "city_1": a.get("city"),
                            "city_2": b.get("city"),
                            "country": a.get("country") or b.get("country") or "Unknown",
                            "state": a.get("state") or b.get("state") or "Unknown",
                        }
                    )

    pairs.sort(
        key=lambda p: (
            -p["score"],
            -(p["match_count_1"] + p["match_count_2"]),
            p["name_1"].lower(),
            p["name_2"].lower(),
        )
    )
    return pairs


def _find_alias_gaps(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Return venues not directly covered by the alias map.
    This highlights likely mapping candidates, not guaranteed bugs.
    """
    from src.data.venue_normalizer import normalize_venue_name, VENUE_ALIASES

    gaps: List[Dict[str, Any]] = []
    for row in rows:
        normalized = normalize_venue_name(row["name"])
        canonical = normalize_venue_name(row.get("canonical_name") or row["name"])
        if normalized in VENUE_ALIASES or canonical in VENUE_ALIASES:
            continue
        gaps.append(
            {
                "venue_id": row["venue_id"],
                "name": row["name"],
                "canonical_name": row.get("canonical_name"),
                "normalized": normalized,
                "city": row.get("city"),
                "state": row.get("state"),
                "country": row.get("country"),
                "match_count": row.get("match_count", 0),
            }
        )

    gaps.sort(key=lambda x: (-x["match_count"], (x["country"] or ""), (x["name"] or "").lower()))
    return gaps


@app.route('/api/data-explorer/venues', methods=['GET'])
def data_explorer_venues():
    """Venue quality dataset for explorer UI."""
    try:
        gender = request.args.get('gender', 'male')
        min_matches = int(request.args.get('min_matches', 1))
        similarity = float(request.args.get('similarity', 0.9))
        similarity = max(0.7, min(0.99, similarity))

        rows = _fetch_venue_quality_rows(gender=gender, min_matches=min_matches)
        hierarchy = _build_venue_hierarchy(rows)
        duplicates = _find_venue_duplicates(rows, similarity_threshold=similarity)
        alias_gaps = _find_alias_gaps(rows)

        return jsonify(
            {
                "success": True,
                "gender": gender,
                "min_matches": min_matches,
                "similarity_threshold": similarity,
                "summary": {
                    "total_venues": len(rows),
                    "countries": len({(r.get("country") or "Unknown") for r in rows}),
                    "states": len({(r.get("state") or "Unknown") for r in rows}),
                    "duplicate_candidates": len(duplicates),
                    "alias_gaps": len(alias_gaps),
                },
                "hierarchy": hierarchy,
                "duplicate_candidates": duplicates[:200],
                "alias_gaps": alias_gaps[:300],
            }
        )
    except Exception as e:
        logger.error(f"Error building venue explorer payload: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================================
# Team Explorer (V4 franchise unification)
# ============================================================================
#
# Backs the Team Explorer tab. Surfaces franchise groupings, fuzzy duplicate
# candidates across teams, and a staged pending -> approved -> applied
# workflow for merging multiple teams.team_id rows under one franchise.
#
# Apply takes a timestamped DB backup, rewrites teams.franchise_id and
# teams.canonical_team_id in a single transaction, and invalidates the
# in-process FranchiseResolver cache. The actual ELO recalc that picks up
# the new groupings is intentionally manual (run scripts/dedupe_elo_history.py)
# because it can take 5+ minutes and shouldn't tie up an HTTP request.

def _team_explorer_franchise_rows() -> List[Dict[str, Any]]:
    """List every team_groups row with its members + headline ELO."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            g.group_id,
            g.canonical_name,
            g.group_type,
            g.country_code,
            g.notes,
            t.team_id,
            t.name AS team_name,
            t.team_type,
            t.tier,
            t.canonical_team_id,
            ce.elo_t20_male,
            ce.elo_t20_female,
            (
                SELECT COUNT(*) FROM matches m
                WHERE m.team1_id = t.team_id OR m.team2_id = t.team_id
            ) AS match_count
        FROM team_groups g
        LEFT JOIN teams t ON t.franchise_id = g.group_id
        LEFT JOIN team_current_elo ce ON ce.team_id = t.team_id
        ORDER BY g.canonical_name, t.name
        """
    )

    grouped: Dict[int, Dict[str, Any]] = {}
    for row in cur.fetchall():
        gid = row["group_id"]
        if gid not in grouped:
            grouped[gid] = {
                "group_id": gid,
                "canonical_name": row["canonical_name"],
                "group_type": row["group_type"],
                "country_code": row["country_code"],
                "notes": row["notes"],
                "members": [],
                "member_count": 0,
                "total_matches": 0,
                "canonical_elo_t20_male": None,
                "canonical_elo_t20_female": None,
            }
        if row["team_id"] is None:
            continue
        is_canonical = row["team_id"] == row["canonical_team_id"]
        grouped[gid]["members"].append(
            {
                "team_id": row["team_id"],
                "name": row["team_name"],
                "team_type": row["team_type"],
                "tier": row["tier"],
                "is_canonical": is_canonical,
                "match_count": row["match_count"] or 0,
                "elo_t20_male": row["elo_t20_male"],
                "elo_t20_female": row["elo_t20_female"],
            }
        )
        grouped[gid]["member_count"] += 1
        grouped[gid]["total_matches"] += row["match_count"] or 0
        if is_canonical:
            grouped[gid]["canonical_elo_t20_male"] = row["elo_t20_male"]
            grouped[gid]["canonical_elo_t20_female"] = row["elo_t20_female"]

    conn.close()
    return list(grouped.values())


def _team_explorer_duplicate_candidates(
    similarity_threshold: float = 0.82,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """Heuristic candidate pairs for merging.

    Heuristics (all low-cost):
      - Token-set ratio across canonical_name pairs that live in different
        team_groups (so we never propose a no-op merge).
      - Same country_code (or both NULL) so we don't propose
        "Royal Challengers Bengaluru" merged with "Trinidad Royals".
      - Excludes pairs whose ids are already in the same group via members.
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:
        logger.warning("rapidfuzz not installed; duplicate finder disabled")
        return []

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT g.group_id, g.canonical_name, g.group_type, g.country_code,
               COUNT(t.team_id) AS member_count,
               (
                   SELECT COUNT(*) FROM matches m
                   JOIN teams tt ON tt.team_id IN (m.team1_id, m.team2_id)
                   WHERE tt.franchise_id = g.group_id
               ) AS match_count
        FROM team_groups g
        LEFT JOIN teams t ON t.franchise_id = g.group_id
        GROUP BY g.group_id
        ORDER BY g.canonical_name
        """
    )
    groups = [dict(r) for r in cur.fetchall()]
    conn.close()

    candidates: List[Dict[str, Any]] = []
    for i, a in enumerate(groups):
        for b in groups[i + 1 :]:
            if a["group_id"] == b["group_id"]:
                continue
            # Same-country gate (treat both-NULL as a permissible pair)
            if a["country_code"] and b["country_code"] and a["country_code"] != b["country_code"]:
                continue
            # Length sanity: don't propose pairs where one is dramatically
            # longer than the other (filters out "Australia" vs "South Australia"
            # collisions that token_set_ratio happily scores 1.0).
            la, lb = len(a["canonical_name"]), len(b["canonical_name"])
            if max(la, lb) > 0 and min(la, lb) / max(la, lb) < 0.6:
                continue
            ratio = fuzz.ratio(a["canonical_name"], b["canonical_name"]) / 100.0
            token = fuzz.token_set_ratio(a["canonical_name"], b["canonical_name"]) / 100.0
            # Both signals must clear the bar. ratio handles typos and rebrand
            # spellings ("Bengaluru" vs "Bangalore"); token catches reorderings.
            if ratio < similarity_threshold or token < similarity_threshold:
                continue
            score = (ratio + token) / 2.0
            candidates.append(
                {
                    "group_a": {
                        "group_id": a["group_id"],
                        "name": a["canonical_name"],
                        "country_code": a["country_code"],
                        "member_count": a["member_count"],
                        "match_count": a["match_count"],
                    },
                    "group_b": {
                        "group_id": b["group_id"],
                        "name": b["canonical_name"],
                        "country_code": b["country_code"],
                        "member_count": b["member_count"],
                        "match_count": b["match_count"],
                    },
                    "similarity": round(score, 3),
                    "ratio": round(ratio, 3),
                    "token_set": round(token, 3),
                }
            )

    candidates.sort(key=lambda c: -c["similarity"])
    return candidates[:limit]


@app.route('/api/team-explorer/franchises', methods=['GET'])
def team_explorer_franchises():
    """Return all franchise groupings + members + headline ELO."""
    try:
        franchises = _team_explorer_franchise_rows()
        type_counts: Dict[str, int] = {}
        for g in franchises:
            type_counts[g["group_type"]] = type_counts.get(g["group_type"], 0) + 1
        return jsonify(
            {
                "success": True,
                "summary": {
                    "total_franchises": len(franchises),
                    "by_group_type": type_counts,
                    "multi_member": sum(1 for g in franchises if g["member_count"] > 1),
                },
                "franchises": franchises,
            }
        )
    except Exception as e:
        logger.error(f"Error building franchise list: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/team-explorer/duplicates', methods=['GET'])
def team_explorer_duplicates():
    """Heuristic merge candidate pairs."""
    try:
        threshold = float(request.args.get('similarity', 0.82))
        threshold = max(0.5, min(0.99, threshold))
        limit = int(request.args.get('limit', 200))
        candidates = _team_explorer_duplicate_candidates(
            similarity_threshold=threshold, limit=limit
        )
        return jsonify(
            {
                "success": True,
                "similarity_threshold": threshold,
                "count": len(candidates),
                "candidates": candidates,
            }
        )
    except Exception as e:
        logger.error(f"Error building duplicate candidates: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/team-explorer/proposals', methods=['GET'])
def team_explorer_list_proposals():
    """List merge proposals, optionally filtered by status."""
    try:
        status = request.args.get('status')
        conn = get_connection()
        cur = conn.cursor()
        if status:
            cur.execute(
                """
                SELECT p.*, st.name AS source_team_name, tg.canonical_name AS target_group_name,
                       tt.name AS target_canonical_team_name
                FROM team_merge_proposals p
                JOIN teams st ON st.team_id = p.source_team_id
                JOIN team_groups tg ON tg.group_id = p.target_group_id
                LEFT JOIN teams tt ON tt.team_id = p.target_canonical_team_id
                WHERE p.status = ?
                ORDER BY p.created_at DESC
                """,
                (status,),
            )
        else:
            cur.execute(
                """
                SELECT p.*, st.name AS source_team_name, tg.canonical_name AS target_group_name,
                       tt.name AS target_canonical_team_name
                FROM team_merge_proposals p
                JOIN teams st ON st.team_id = p.source_team_id
                JOIN team_groups tg ON tg.group_id = p.target_group_id
                LEFT JOIN teams tt ON tt.team_id = p.target_canonical_team_id
                ORDER BY p.created_at DESC
                LIMIT 200
                """
            )
        proposals = [dict(r) for r in cur.fetchall()]
        conn.close()
        return jsonify({"success": True, "count": len(proposals), "proposals": proposals})
    except Exception as e:
        logger.error(f"Error listing merge proposals: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/team-explorer/proposals', methods=['POST'])
def team_explorer_create_proposal():
    """Stage a new merge proposal.

    Body JSON: {
        source_team_id: int,
        target_group_id: int,
        target_canonical_team_id: int (optional - defaults to current canonical),
        rationale: str,
        fuzzy_score: float (optional)
    }
    """
    try:
        data = request.get_json() or {}
        source_team_id = data.get('source_team_id')
        target_group_id = data.get('target_group_id')
        if not source_team_id or not target_group_id:
            return jsonify(
                {"success": False, "error": "source_team_id and target_group_id required"}
            ), 400

        conn = get_connection()
        cur = conn.cursor()

        # Validate both ends exist.
        cur.execute("SELECT team_id, franchise_id FROM teams WHERE team_id = ?", (source_team_id,))
        src = cur.fetchone()
        if not src:
            conn.close()
            return jsonify({"success": False, "error": f"Unknown source_team_id={source_team_id}"}), 400

        cur.execute("SELECT group_id, canonical_name FROM team_groups WHERE group_id = ?", (target_group_id,))
        tgt = cur.fetchone()
        if not tgt:
            conn.close()
            return jsonify({"success": False, "error": f"Unknown target_group_id={target_group_id}"}), 400

        # No-op merges should not be staged.
        if src["franchise_id"] == target_group_id:
            conn.close()
            return jsonify(
                {
                    "success": False,
                    "error": (
                        f"team_id={source_team_id} is already in group {target_group_id} "
                        f"({tgt['canonical_name']}); nothing to merge"
                    ),
                }
            ), 400

        # Default target_canonical_team_id to whichever team_id currently owns
        # the target group's rating series.
        target_canonical = data.get('target_canonical_team_id')
        if not target_canonical:
            cur.execute(
                """
                SELECT canonical_team_id FROM teams
                WHERE franchise_id = ? AND canonical_team_id = team_id
                LIMIT 1
                """,
                (target_group_id,),
            )
            row = cur.fetchone()
            target_canonical = row['canonical_team_id'] if row else None

        cur.execute(
            """
            INSERT INTO team_merge_proposals (
                source_team_id, target_group_id, target_canonical_team_id,
                status, proposer, rationale, fuzzy_score
            ) VALUES (?, ?, ?, 'pending', ?, ?, ?)
            """,
            (
                source_team_id,
                target_group_id,
                target_canonical,
                data.get('proposer') or 'team-explorer-ui',
                data.get('rationale') or '',
                data.get('fuzzy_score'),
            ),
        )
        proposal_id = cur.lastrowid
        conn.commit()
        conn.close()

        return jsonify({"success": True, "proposal_id": proposal_id})
    except Exception as e:
        logger.error(f"Error creating merge proposal: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/team-explorer/proposals/<int:proposal_id>/decision', methods=['POST'])
def team_explorer_decide_proposal(proposal_id: int):
    """Approve or reject a pending proposal. Body: {decision: 'approve'|'reject'}."""
    try:
        data = request.get_json() or {}
        decision = (data.get('decision') or '').lower()
        if decision not in ('approve', 'reject'):
            return jsonify({"success": False, "error": "decision must be 'approve' or 'reject'"}), 400

        new_status = 'approved' if decision == 'approve' else 'rejected'
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE team_merge_proposals
            SET status = ?, decided_at = CURRENT_TIMESTAMP, decided_by = ?
            WHERE proposal_id = ? AND status = 'pending'
            """,
            (new_status, data.get('decided_by') or 'team-explorer-ui', proposal_id),
        )
        if cur.rowcount == 0:
            conn.close()
            return jsonify(
                {"success": False, "error": "Proposal not found or not pending"}
            ), 404
        conn.commit()
        conn.close()
        return jsonify({"success": True, "proposal_id": proposal_id, "status": new_status})
    except Exception as e:
        logger.error(f"Error deciding proposal {proposal_id}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/team-explorer/proposals/apply', methods=['POST'])
def team_explorer_apply_proposals():
    """Apply all approved proposals in a single transaction.

    Steps:
      1. Take a timestamped DB backup so the operator can roll back.
      2. For each approved proposal: rewrite the source team's franchise_id +
         canonical_team_id; if the target group's other members still point at
         a stale canonical, fix them too.
      3. Mark the proposals as 'applied'.
      4. Invalidate the FranchiseResolver cache.

    The user must run scripts/dedupe_elo_history.py separately to recompute
    ratings under the new groupings (it's a 5+ minute job).
    """
    try:
        import shutil

        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT proposal_id, source_team_id, target_group_id, target_canonical_team_id
                FROM team_merge_proposals
                WHERE status = 'approved'
                ORDER BY created_at
                """
            )
            approved = [dict(r) for r in cur.fetchall()]
            if not approved:
                return jsonify({"success": True, "applied": 0, "message": "No approved proposals to apply"})

            # Backup before writes. The DB is large; the copy can take a minute.
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = Path(str(DATABASE_PATH)).with_suffix(f'.db.bak.{ts}')
            logger.info(f"Backing up DB to {backup_path} before applying merges")
            shutil.copy2(str(DATABASE_PATH), str(backup_path))
            for suffix in ('-wal', '-shm'):
                sidecar = Path(str(DATABASE_PATH) + suffix)
                if sidecar.exists():
                    shutil.copy2(str(sidecar), str(backup_path) + suffix)

            try:
                cur.execute("BEGIN")
                applied_ids = []
                for p in approved:
                    source = p['source_team_id']
                    group = p['target_group_id']
                    canonical = p['target_canonical_team_id']

                    # Re-point the source team into the target group.
                    cur.execute(
                        """
                        UPDATE teams
                        SET franchise_id = ?, canonical_team_id = ?
                        WHERE team_id = ?
                        """,
                        (group, canonical or source, source),
                    )
                    # If a canonical was specified, force every member of the group
                    # to point at it (handles the case where the group previously
                    # had a different canonical owner).
                    if canonical:
                        cur.execute(
                            """
                            UPDATE teams
                            SET canonical_team_id = ?
                            WHERE franchise_id = ?
                            """,
                            (canonical, group),
                        )

                    cur.execute(
                        """
                        UPDATE team_merge_proposals
                        SET status = 'applied', applied_at = CURRENT_TIMESTAMP
                        WHERE proposal_id = ?
                        """,
                        (p['proposal_id'],),
                    )
                    applied_ids.append(p['proposal_id'])

                # Clean up orphan team_groups rows (the auto-created self-group
                # that the moved team used to belong to is now empty).
                cur.execute(
                    """
                    DELETE FROM team_groups
                    WHERE group_id NOT IN (SELECT franchise_id FROM teams WHERE franchise_id IS NOT NULL)
                      AND group_id NOT IN (SELECT group_id FROM team_external_ids WHERE group_id IS NOT NULL)
                      AND group_id NOT IN (SELECT target_group_id FROM team_merge_proposals)
                    """
                )
                orphans_removed = cur.rowcount

                cur.execute("COMMIT")
            except Exception as inner:
                cur.execute("ROLLBACK")
                logger.error(f"Apply rolled back: {inner}")
                return jsonify({"success": False, "error": f"Apply rolled back: {inner}"}), 500

        # Invalidate runtime caches so the next predict request sees the new
        # mapping. ELO recalc is intentionally manual (multi-minute job).
        from src.data.franchise_resolver import get_resolver
        get_resolver().invalidate()

        return jsonify(
            {
                "success": True,
                "applied": len(applied_ids),
                "applied_proposal_ids": applied_ids,
                "orphan_groups_removed": orphans_removed,
                "backup_path": str(backup_path),
                "next_step": (
                    "Run `python scripts/dedupe_elo_history.py --skip-backup` to "
                    "recompute team ELOs under the new franchise groupings."
                ),
            }
        )
    except Exception as e:
        logger.error(f"Error applying proposals: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


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
                            fmt = format_type_to_model_format(match.format_type) if getattr(match, 'format_type', None) else 'T20'
                            venue_match = scraper.match_venue_to_db(match.venue, match.gender, format_type=fmt)
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


@app.route('/api/prediction/available-models', methods=['GET'])
def get_available_prediction_models():
    """
    Return (gender, format) pairs for which an active model exists.
    Used by single and bulk predict to disable Run or show per-match model status.
    """
    try:
        from src.data.database import get_model_versions, init_model_versions_table
        init_model_versions_table()
        models = get_model_versions(active_only=True)
        seen = set()
        available = []
        for m in models:
            key = (m['gender'], (m.get('format_type') or 'T20').upper())
            if key not in seen:
                seen.add(key)
                available.append({'gender': key[0], 'format': key[1]})
        return jsonify({'success': True, 'available': available})
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        return jsonify({'success': False, 'error': str(e), 'available': []}), 500


def pad_with_db_fillers(batting_order, team_id, team_elo, gender, format_type):
    """
    Pad a short batting order to 11 using real DB players from the same team,
    ordered by batting ELO closest to the team's current rating.

    Only fires when the batting order has fewer than 11 players AND a team_id
    is known. Falls through silently if the team has no extra DB players.

    `team_elo` is used only as a fallback when the team's actual current ELO
    cannot be found in team_current_elo.  For matched teams the caller passes
    `None` (defaulting to 1500), which is wrong — always prefer the real ELO.

    Returns:
        (padded_order, filler_ids) — filler_ids is the list of IDs added.
    """
    if len(batting_order) >= 11 or not team_id:
        return batting_order, []

    n_needed = 11 - len(batting_order)
    already = set(batting_order)
    elo_col = f'batting_elo_{format_type.lower()}_{gender}'
    team_elo_col = f'elo_{format_type.lower()}_{gender}'

    # V4 franchise unification: resolve to canonical_team_id so rebrand-legacy
    # ids (e.g. 111 RCB Bangalore) read the unified franchise rating instead
    # of missing the row and falling back to 1500.
    from src.data.franchise_resolver import get_resolver
    canonical_team_id = get_resolver().canonical(team_id) or team_id

    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Always resolve the team's actual current ELO from the DB when possible;
        # the caller-supplied `team_elo` defaults to 1500 for matched teams which
        # would select their historically *best* players rather than their average.
        cursor.execute(
            f"SELECT COALESCE({team_elo_col}, ?) FROM team_current_elo WHERE team_id = ?",
            (team_elo, canonical_team_id)
        )
        row = cursor.fetchone()
        resolved_elo = row[0] if row else team_elo

        cursor.execute(f"""
            SELECT DISTINCT pms.player_id,
                   COALESCE(pce.{elo_col}, 1500) as player_elo,
                   ABS(COALESCE(pce.{elo_col}, 1500) - ?) as elo_diff
            FROM player_match_stats pms
            LEFT JOIN player_current_elo pce ON pms.player_id = pce.player_id
            JOIN matches m ON pms.match_id = m.match_id
            WHERE pms.team_id = ?
              AND m.match_type = ?
              AND m.gender = ?
            ORDER BY elo_diff ASC
            LIMIT ?
        """, (resolved_elo, team_id, format_type.upper(), gender, n_needed + len(already)))

        fillers = []
        for row in cursor.fetchall():
            if row['player_id'] not in already and len(fillers) < n_needed:
                fillers.append(row['player_id'])
                already.add(row['player_id'])
        conn.close()
    except Exception as e:
        logger.warning(f"[PADDING] Could not fetch DB fillers for team {team_id}: {e}")
        return batting_order, []

    if fillers:
        logger.info(f"[PADDING] Added {len(fillers)} DB filler player(s) for team {team_id} "
                    f"(ELO target {resolved_elo:.0f}, team ELO from DB): {fillers}")

    return batting_order + fillers, fillers


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
        "gender": "male" or "female" (optional, default "male"),
        "team1_id": int (optional, for ELO lookup),
        "team2_id": int (optional, for ELO lookup)
    }
    """
    try:
        data = request.get_json()
        
        # Convert IDs to integers - crucial for matching distribution dictionary keys
        # Frontend sends strings, but batter_dists/bowler_dists use integer keys
        def to_int_ids(id_list):
            result = []
            for x in id_list:
                try:
                    result.append(int(x))
                except (ValueError, TypeError):
                    logger.warning(f"[SIMULATION] Skipping non-integer ID: {x}")
            return result
        
        team1_batters_raw = to_int_ids(data.get('team1_batters', []))
        team1_bowlers_raw = to_int_ids(data.get('team1_bowlers', []))
        team2_batters_raw = to_int_ids(data.get('team2_batters', []))
        team2_bowlers_raw = to_int_ids(data.get('team2_bowlers', []))
        simulator_type = data.get('simulator', 'nn')  # Default to NN for scorecard
        n_simulations = min(data.get('n_simulations', 1000), 1000000)
        venue_id = data.get('venue_id')
        use_toss = data.get('use_toss', False)
        gender = data.get('gender', 'male')
        format_param = data.get('format', 'T20')
        model_format = format_type_to_model_format(format_param)

        if not has_active_model(gender, model_format):
            return jsonify({
                'success': False,
                'error': f"No active model for {gender} {model_format}. Please train a model on the Training page."
            }), 503

        # Get team IDs for ELO lookup (optional - will default to 1500 or tier-based if provided)
        team1_id = data.get('team1_id')
        team2_id = data.get('team2_id')
        team1_name = data.get('team1_name', 'Team 1')
        team2_name = data.get('team2_name', 'Team 2')
        team1_default_elo = data.get('team1_default_elo')
        team2_default_elo = data.get('team2_default_elo')
        series_name = data.get('series_name')
        if team1_id:
            try:
                team1_id = int(team1_id)
            except (ValueError, TypeError):
                team1_id = None
        if team2_id:
            try:
                team2_id = int(team2_id)
            except (ValueError, TypeError):
                team2_id = None
        if team1_default_elo is not None:
            try:
                team1_default_elo = float(team1_default_elo)
            except (ValueError, TypeError):
                team1_default_elo = None
        if team2_default_elo is not None:
            try:
                team2_default_elo = float(team2_default_elo)
            except (ValueError, TypeError):
                team2_default_elo = None

        # Compute tier-based default ELO for unmatched teams when not provided
        if not team1_id and team1_default_elo is None and (team2_id or series_name):
            from src.utils.default_elo import infer_default_elo
            team1_default_elo = infer_default_elo(
                series_name=series_name,
                matched_opponent_team_id=team2_id,
                format_type=format_param or 'T20',
            )
        if not team2_id and team2_default_elo is None and (team1_id or series_name):
            from src.utils.default_elo import infer_default_elo
            team2_default_elo = infer_default_elo(
                series_name=series_name,
                matched_opponent_team_id=team1_id,
                format_type=format_param or 'T20',
            )

        # Track data quality warnings
        data_warnings = []
        default1 = team1_default_elo if team1_default_elo is not None else 1500
        default2 = team2_default_elo if team2_default_elo is not None else 1500
        if not team1_id:
            data_warnings.append(
                f"'{team1_name}' has no database ID. Using default ELO ({int(default1)}). "
                f"Prediction may be unreliable."
            )
            logger.warning(f"[SIM WARNING] Team1 '{team1_name}' has no database ID -- using default ELO {default1}")
        if not team2_id:
            data_warnings.append(
                f"'{team2_name}' has no database ID. Using default ELO ({int(default2)}). "
                f"Prediction may be unreliable."
            )
            logger.warning(f"[SIM WARNING] Team2 '{team2_name}' has no database ID -- using default ELO {default2}")
        
        # DEBUG: Log incoming player IDs to diagnose 50/50 results
        logger.info(f"[SIMULATION DEBUG] Team1 batters: {team1_batters_raw[:5]}... ({len(team1_batters_raw)} total)")
        logger.info(f"[SIMULATION DEBUG] Team1 bowlers: {team1_bowlers_raw[:5]}... ({len(team1_bowlers_raw)} total)")
        logger.info(f"[SIMULATION DEBUG] Team2 batters: {team2_batters_raw[:5]}... ({len(team2_batters_raw)} total)")
        logger.info(f"[SIMULATION DEBUG] Team2 bowlers: {team2_bowlers_raw[:5]}... ({len(team2_bowlers_raw)} total)")
        
        # Check if IDs are likely CREX IDs vs database IDs
        # CREX IDs are typically strings like "crex_123" or large integers
        # Database IDs are sequential integers 1-10000
        all_ids = team1_batters_raw + team1_bowlers_raw + team2_batters_raw + team2_bowlers_raw
        string_ids = [x for x in all_ids if isinstance(x, str) and not x.isdigit()]
        if string_ids:
            logger.warning(f"[SIMULATION DEBUG] Found non-numeric IDs: {string_ids[:5]}... This may indicate CREX IDs not matched to database!")
        
        # Check ID range - database IDs are typically 1-10000
        numeric_ids = [int(x) for x in all_ids if str(x).isdigit()]
        if numeric_ids:
            min_id, max_id = min(numeric_ids), max(numeric_ids)
            logger.info(f"[SIMULATION DEBUG] ID range: {min_id} to {max_id}")
            if max_id > 100000:
                logger.warning(f"[SIMULATION DEBUG] Large IDs detected - these may be ESPN external IDs, not database IDs!")
        
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

        # Pad short batting orders with real DB players from the same team,
        # selected by ELO closest to the team's current rating.
        t1_pad_elo = team1_default_elo if team1_default_elo is not None else 1500
        t2_pad_elo = team2_default_elo if team2_default_elo is not None else 1500
        team1_batters, t1_fillers = pad_with_db_fillers(
            team1_batters, team1_id, t1_pad_elo, gender, model_format)
        team2_batters, t2_fillers = pad_with_db_fillers(
            team2_batters, team2_id, t2_pad_elo, gender, model_format)
        if t1_fillers:
            data_warnings.append(
                f"{team1_name}: {len(t1_fillers)} squad slot(s) filled from DB by ELO calibration "
                f"(squad had only {len(team1_batters) - len(t1_fillers)}/11 matched players)."
            )
        if t2_fillers:
            data_warnings.append(
                f"{team2_name}: {len(t2_fillers)} squad slot(s) filled from DB by ELO calibration "
                f"(squad had only {len(team2_batters) - len(t2_fillers)}/11 matched players)."
            )

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
        
        team1_bowlers = fill_with_parttimers(team1_bowlers, team1_batters)
        team2_bowlers = fill_with_parttimers(team2_bowlers, team2_batters)
        
        # Get toss field probability from historical data
        toss_field_prob = 0.65  # Default T20 field preference
        if use_toss:
            try:
                toss_simulator = get_toss_simulator()
                rates = toss_simulator.stats.get_decision_rates('T20', gender)
                toss_field_prob = rates.get('field', 0.65)
            except:
                pass  # Use default
        
        # Select simulator for specified gender and format
        if simulator_type == 'nn':
            simulator = get_nn_simulator(gender, model_format)
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
            toss_field_prob=toss_field_prob,
            team1_id=team1_id,
            team2_id=team2_id,
            team1_default_elo=team1_default_elo,
            team2_default_elo=team2_default_elo
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
        
        # V4: surface canonical franchise + dist quality alongside the headline
        # numbers so the Bulk Predict UI can render a small per-row badge.
        from src.data.franchise_resolver import get_resolver
        _r = get_resolver()
        franchise_info = {
            'team1': {
                'team_id': team1_id,
                'canonical_team_id': _r.canonical(team1_id),
                'franchise_id': _r.franchise(team1_id),
                'elo_used': results.get('team1_elo_used'),
            },
            'team2': {
                'team_id': team2_id,
                'canonical_team_id': _r.canonical(team2_id),
                'franchise_id': _r.franchise(team2_id),
                'elo_used': results.get('team2_elo_used'),
            },
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
            'gender': gender,
            'format': model_format,
            'data_warnings': data_warnings if data_warnings else None,
            'dist_quality': results.get('dist_quality'),
            'franchise_info': franchise_info,
        }

        # Wave 5 Phase 1-2: surface multi-market probabilities when the V2
        # simulator emits them. V1 results don't have team1_player_runs etc;
        # we just skip in that case so the response shape stays a strict superset.
        if 'team1_player_runs' in results:
            try:
                from src.models.market_outputs import (
                    derive_polymarket_market_probs,
                    market_summary_for_ui,
                )
                full_market_probs = derive_polymarket_market_probs(results)
                response['market_probs'] = full_market_probs
                response['market_probs_summary'] = market_summary_for_ui(full_market_probs)
            except Exception as exc:
                logger.warning(f"Failed to derive multi-market probs: {exc}")

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
                    team1_bats_first=True,  # For scorecard, show Team 1 batting first
                    team1_id=team1_id,
                    team2_id=team2_id,
                    team1_default_elo=team1_default_elo,
                    team2_default_elo=team2_default_elo
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
    
    # Convert IDs to integers - crucial for matching distribution dictionary keys
    # Frontend sends strings, but batter_dists/bowler_dists use integer keys
    def to_int_ids(id_list):
        result = []
        for x in id_list:
            try:
                result.append(int(x))
            except (ValueError, TypeError):
                logger.warning(f"[SIMULATION_STREAM] Skipping non-integer ID: {x}")
        return result
    
    team1_batters_raw = to_int_ids(data.get('team1_batters', []))
    team1_bowlers_raw = to_int_ids(data.get('team1_bowlers', []))
    team2_batters_raw = to_int_ids(data.get('team2_batters', []))
    team2_bowlers_raw = to_int_ids(data.get('team2_bowlers', []))
    simulator_type = data.get('simulator', 'nn')  # Default to NN for scorecard
    n_simulations = min(data.get('n_simulations', 1000), 1000000)
    venue_id = data.get('venue_id')
    use_toss = data.get('use_toss', False)
    gender = data.get('gender', 'male')
    format_param = data.get('format', 'T20')
    model_format = format_type_to_model_format(format_param)
    frontend_player_names = data.get('player_names', {})  # Names from frontend

    if not has_active_model(gender, model_format):
        return jsonify({
            'success': False,
            'error': f"No active model for {gender} {model_format}. Please train a model on the Training page."
        }), 503

    # Get team IDs for ELO lookup (optional - will default to 1500 or tier-based if provided)
    team1_id = data.get('team1_id')
    team2_id = data.get('team2_id')
    team1_name = data.get('team1_name', 'Team 1')
    team2_name = data.get('team2_name', 'Team 2')
    team1_default_elo = data.get('team1_default_elo')
    team2_default_elo = data.get('team2_default_elo')
    series_name = data.get('series_name')
    if team1_default_elo is not None:
        try:
            team1_default_elo = float(team1_default_elo)
        except (ValueError, TypeError):
            team1_default_elo = None
    if team2_default_elo is not None:
        try:
            team2_default_elo = float(team2_default_elo)
        except (ValueError, TypeError):
            team2_default_elo = None
    if team1_id:
        try:
            team1_id = int(team1_id)
        except (ValueError, TypeError):
            team1_id = None
    if team2_id:
        try:
            team2_id = int(team2_id)
        except (ValueError, TypeError):
            team2_id = None
    
    # Compute tier-based default ELO for unmatched teams when not provided
    if not team1_id and team1_default_elo is None and (team2_id or series_name):
        from src.utils.default_elo import infer_default_elo
        team1_default_elo = infer_default_elo(
            series_name=series_name,
            matched_opponent_team_id=team2_id,
            format_type=format_param or 'T20',
        )
    if not team2_id and team2_default_elo is None and (team1_id or series_name):
        from src.utils.default_elo import infer_default_elo
        team2_default_elo = infer_default_elo(
            series_name=series_name,
            matched_opponent_team_id=team1_id,
            format_type=format_param or 'T20',
        )

    # Track data quality warnings
    stream_data_warnings = []
    default1 = team1_default_elo if team1_default_elo is not None else 1500
    default2 = team2_default_elo if team2_default_elo is not None else 1500
    if not team1_id:
        stream_data_warnings.append(
            f"'{team1_name}' has no database ID. Using default ELO ({int(default1)}). "
            f"Prediction may be unreliable."
        )
        logger.warning(f"[SIM_STREAM WARNING] Team1 '{team1_name}' has no database ID -- using default ELO {default1}")
    if not team2_id:
        stream_data_warnings.append(
            f"'{team2_name}' has no database ID. Using default ELO ({int(default2)}). "
            f"Prediction may be unreliable."
        )
        logger.warning(f"[SIM_STREAM WARNING] Team2 '{team2_name}' has no database ID -- using default ELO {default2}")
    
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

    # Pad short batting orders with real DB players from the same team,
    # selected by ELO closest to the team's current rating.
    t1_pad_elo = team1_default_elo if team1_default_elo is not None else 1500
    t2_pad_elo = team2_default_elo if team2_default_elo is not None else 1500
    team1_batting_order, t1_fillers = pad_with_db_fillers(
        team1_batting_order, team1_id, t1_pad_elo, gender, model_format)
    team2_batting_order, t2_fillers = pad_with_db_fillers(
        team2_batting_order, team2_id, t2_pad_elo, gender, model_format)
    if t1_fillers:
        stream_data_warnings.append(
            f"{team1_name}: {len(t1_fillers)} squad slot(s) filled from DB by ELO calibration "
            f"(squad had only {len(team1_batting_order) - len(t1_fillers)}/11 matched players)."
        )
    if t2_fillers:
        stream_data_warnings.append(
            f"{team2_name}: {len(t2_fillers)} squad slot(s) filled from DB by ELO calibration "
            f"(squad had only {len(team2_batting_order) - len(t2_fillers)}/11 matched players)."
        )

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
    
    team1_bowlers = fill_with_parttimers(team1_bowlers, team1_batting_order)
    team2_bowlers = fill_with_parttimers(team2_bowlers, team2_batting_order)
    
    # ===== SIMULATION AUDIT LOG =====
    # Critical for debugging gender/model selection issues
    team1_name_audit = "unknown"
    team2_name_audit = "unknown"
    if team1_id or team2_id:
        try:
            conn_audit = get_connection()
            cur_audit = conn_audit.cursor()
            if team1_id:
                cur_audit.execute("SELECT name FROM teams WHERE team_id = ?", (team1_id,))
                row = cur_audit.fetchone()
                if row:
                    team1_name_audit = row[0]
            if team2_id:
                cur_audit.execute("SELECT name FROM teams WHERE team_id = ?", (team2_id,))
                row = cur_audit.fetchone()
                if row:
                    team2_name_audit = row[0]
            conn_audit.close()
        except Exception as e:
            logger.warning(f"[SIM_AUDIT] Could not look up team names: {e}")
    
    logger.info(f"[SIM_AUDIT] ========== SIMULATION REQUEST ==========")
    logger.info(f"[SIM_AUDIT] Gender: {gender} | Simulator: {simulator_type}")
    logger.info(f"[SIM_AUDIT] Team 1: {team1_name_audit} (id={team1_id}) | "
                f"Batters: {len(team1_batters_raw)} | Bowlers: {len(team1_bowlers_raw)} | "
                f"XI: {len(team1_batting_order)} unique")
    logger.info(f"[SIM_AUDIT] Team 2: {team2_name_audit} (id={team2_id}) | "
                f"Batters: {len(team2_batters_raw)} | Bowlers: {len(team2_bowlers_raw)} | "
                f"XI: {len(team2_batting_order)} unique")
    logger.info(f"[SIM_AUDIT] Venue ID: {venue_id} | N sims: {n_simulations} | Use toss: {use_toss}")
    logger.info(f"[SIM_AUDIT] Team 1 batting order IDs: {team1_batting_order}")
    logger.info(f"[SIM_AUDIT] Team 2 batting order IDs: {team2_batting_order}")
    logger.info(f"[SIM_AUDIT] Team 1 bowler IDs: {team1_bowlers}")
    logger.info(f"[SIM_AUDIT] Team 2 bowler IDs: {team2_bowlers}")
    
    # Log player names from frontend for easy identification
    if frontend_player_names:
        t1_names = [frontend_player_names.get(str(pid), f"?{pid}") for pid in team1_batting_order]
        t2_names = [frontend_player_names.get(str(pid), f"?{pid}") for pid in team2_batting_order]
        logger.info(f"[SIM_AUDIT] Team 1 XI names: {t1_names}")
        logger.info(f"[SIM_AUDIT] Team 2 XI names: {t2_names}")
    if not team1_id:
        logger.warning(f"[SIM_AUDIT] CAUTION: Team 1 '{team1_name_audit}' has no database ID -- using default ELO 1500. Results may be unreliable.")
    if not team2_id:
        logger.warning(f"[SIM_AUDIT] CAUTION: Team 2 '{team2_name_audit}' has no database ID -- using default ELO 1500. Results may be unreliable.")
    logger.info(f"[SIM_AUDIT] ========================================")
    
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
            
            # Select simulator for specified gender and format
            if simulator_type == 'nn':
                simulator = get_nn_simulator(gender, model_format)
            else:
                simulator = get_fast_simulator(gender)
            
            # Chunked simulation with progress updates
            # LARGE chunks for GPU throughput (M2 Pro handles big batches well)
            # Decouple sim chunk size from progress update frequency
            sim_chunk_size = 10000  # Big batches for TF/Metal GPU efficiency on M2 Pro
            progress_interval = max(1, n_simulations // 40)  # ~40 progress updates
            
            all_team1_scores = []
            all_team2_scores = []
            # Wave 5 Phase 1-2: per-chunk per-batter runs + per-team sixes,
            # concatenated across chunks for downstream Polymarket sub-market
            # probability extraction. Empty list when V1 simulator (no extras).
            all_team1_player_runs = []
            all_team2_player_runs = []
            all_team1_sixes = []
            all_team2_sixes = []
            total_team1_wins = 0  # Running count (avoid list sum)
            completed = 0
            last_progress_at = 0
            start_time = time.time()
            
            toss_stats_accum = {'team1_won_toss': 0, 'chose_field': 0, 'team1_batted_first': 0, 'total': 0}
            
            for chunk_start in range(0, n_simulations, sim_chunk_size):
                chunk_n = min(sim_chunk_size, n_simulations - chunk_start)
                
                # Run chunk with proper batting orders (11 unique players each)
                chunk_results = simulator.simulate_matches(
                    chunk_n,
                    team1_batting_order,  # 11 unique: batters[:6] + bowlers[:5]
                    team1_bowlers[:5],
                    team2_batting_order,  # 11 unique: batters[:6] + bowlers[:5]
                    team2_bowlers[:5],
                    venue_id=venue_id,
                    use_toss=use_toss,
                    toss_field_prob=toss_field_prob,
                    team1_id=team1_id,
                    team2_id=team2_id,
                    team1_default_elo=team1_default_elo,
                    team2_default_elo=team2_default_elo
                )
                
                # Accumulate results (keep as numpy, avoid .tolist())
                t1_scores = chunk_results['team1_scores']
                t2_scores = chunk_results['team2_scores']
                all_team1_scores.append(t1_scores)
                all_team2_scores.append(t2_scores)
                chunk_wins = int((t1_scores > t2_scores).sum())
                total_team1_wins += chunk_wins
                # V2-only multi-market arrays (V1 simulator chunks won't have these keys)
                if 'team1_player_runs' in chunk_results:
                    all_team1_player_runs.append(chunk_results['team1_player_runs'])
                    all_team2_player_runs.append(chunk_results['team2_player_runs'])
                    all_team1_sixes.append(chunk_results['team1_sixes'])
                    all_team2_sixes.append(chunk_results['team2_sixes'])
                
                # Accumulate toss stats
                if use_toss and 'toss_stats' in chunk_results:
                    ts = chunk_results['toss_stats']
                    toss_stats_accum['team1_won_toss'] += ts['team1_won_toss_pct'] * chunk_n
                    toss_stats_accum['chose_field'] += ts['chose_field_pct'] * chunk_n
                    toss_stats_accum['team1_batted_first'] += ts['team1_batted_first_pct'] * chunk_n
                    toss_stats_accum['total'] += chunk_n
                
                completed += chunk_n
                
                # Only send progress update at intervals (avoid SSE overhead)
                if completed - last_progress_at >= progress_interval or completed >= n_simulations:
                    last_progress_at = completed
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (n_simulations - completed) / rate if rate > 0 else 0
                    running_t1_win_pct = round(total_team1_wins / completed * 100, 1) if completed > 0 else 50.0
                    
                    progress = {
                        'type': 'progress',
                        'completed': completed,
                        'total': n_simulations,
                        'elapsed_ms': int(elapsed * 1000),
                        'rate': round(rate, 1),
                        'eta_seconds': round(remaining, 1),
                        'pct': round(completed * 100 / n_simulations, 1),
                        'running_team1_win_pct': running_t1_win_pct
                    }
                    yield f"data: {json.dumps(progress)}\n\n"
            
            # Calculate final stats
            import numpy as np
            team1_scores = np.concatenate(all_team1_scores)
            team2_scores = np.concatenate(all_team2_scores)
            team1_wins = team1_scores > team2_scores
            
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
            
            # V4: dist quality (carry the LAST chunk's snapshot since it's
            # constant across chunks for the same fixture) + canonical
            # franchise so the bulk-predict UI can render a per-row badge.
            from src.data.franchise_resolver import get_resolver
            _r = get_resolver()
            franchise_info = {
                'team1': {
                    'team_id': team1_id,
                    'canonical_team_id': _r.canonical(team1_id),
                    'franchise_id': _r.franchise(team1_id),
                    'elo_used': chunk_results.get('team1_elo_used'),
                },
                'team2': {
                    'team_id': team2_id,
                    'canonical_team_id': _r.canonical(team2_id),
                    'franchise_id': _r.franchise(team2_id),
                    'elo_used': chunk_results.get('team2_elo_used'),
                },
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
                'gender': gender,
                'format': model_format,
                'data_warnings': stream_data_warnings if stream_data_warnings else None,
                'dist_quality': chunk_results.get('dist_quality'),
                'franchise_info': franchise_info,
            }

            # Wave 5 Phase 1-2: surface multi-market probabilities when V2 sim was used.
            if all_team1_player_runs:
                try:
                    from src.models.market_outputs import (
                        derive_polymarket_market_probs,
                        market_summary_for_ui,
                    )
                    aggregated = {
                        'team1_win_prob': team1_wins.mean(),
                        'team1_player_runs': np.concatenate(all_team1_player_runs, axis=0),
                        'team2_player_runs': np.concatenate(all_team2_player_runs, axis=0),
                        'team1_sixes': np.concatenate(all_team1_sixes),
                        'team2_sixes': np.concatenate(all_team2_sixes),
                        'team1_batter_ids': chunk_results.get('team1_batter_ids', team1_batting_order),
                        'team2_batter_ids': chunk_results.get('team2_batter_ids', team2_batting_order),
                    }
                    full_market_probs = derive_polymarket_market_probs(aggregated)
                    result['market_probs'] = full_market_probs
                    result['market_probs_summary'] = market_summary_for_ui(full_market_probs)
                except Exception as exc:
                    logger.warning(f"Bulk predict: failed to derive multi-market probs: {exc}")

            # Generate scorecard for NN simulator
            if simulator_type == 'nn' and hasattr(simulator, 'simulate_detailed_match'):
                try:
                    # Get player names from database, then fall back to frontend-provided names
                    db_player_names = _get_player_names(team1_batting_order + team2_batting_order + team1_bowlers + team2_bowlers)
                    # Merge: DB names take precedence, then frontend names, then fallback
                    player_names = {**frontend_player_names}  # Start with frontend names
                    player_names.update({str(k): v for k, v in db_player_names.items()})  # Override with DB names
                    
                    scorecard = simulator.simulate_detailed_match(
                        team1_batting_order,  # 11 unique players in batting order
                        team1_bowlers[:5],
                        team2_batting_order,  # 11 unique players in batting order
                        team2_bowlers[:5],
                        venue_id=venue_id,
                        team1_bats_first=True,
                        team1_id=team1_id,
                        team2_id=team2_id,
                        team1_default_elo=team1_default_elo,
                        team2_default_elo=team2_default_elo
                    )
                    for b in scorecard['team1_batting']:
                        pid = str(b['player_id'])
                        b['name'] = player_names.get(pid, f"Player {pid}")
                    for b in scorecard['team2_batting']:
                        pid = str(b['player_id'])
                        b['name'] = player_names.get(pid, f"Player {pid}")
                    for b in scorecard['team1_bowling']:
                        pid = str(b['player_id'])
                        b['name'] = player_names.get(pid, f"Player {pid}")
                    for b in scorecard['team2_bowling']:
                        pid = str(b['player_id'])
                        b['name'] = player_names.get(pid, f"Player {pid}")
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
                    'has_squad': m.has_squad,
                    'format': 'T20',
                    'model_format': 'T20',
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
                'gender': m.gender,
                'format': 'T20',
                'model_format': 'T20',
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
    Get upcoming matches from CREX with time window filtering.
    
    Query params:
        format: 'T20', 'ODI', or omit / 'all' to include all formats (default: all).
        hours_ahead: Only include matches starting within this many hours (default: 36).
        hours_behind: Also include matches from this many hours ago (default: 3).
        
    Returns:
        List of upcoming matches with basic info, grouped by series
    """
    try:
        format_param = request.args.get('format')
        if format_param and format_param.lower() == 'all':
            format_param = None
        formats = [format_param] if format_param else None
        
        # Time window parameters (restored usability)
        hours_ahead = int(request.args.get('hours_ahead', 36))  # Default 36h as requested
        hours_behind = int(request.args.get('hours_behind', 3))

        scraper = get_crex_scraper()
        matches = scraper.get_schedule(formats=formats, hours_ahead=hours_ahead, hours_behind=hours_behind)

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
                'model_format': format_type_to_model_format(m.format_type),
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
        
        # Parse URL for verification - extract expected team codes from slug
        url_info = scraper.parse_url_team_codes(match_url)
        logger.info(f"[URL VERIFY] URL slug teams: {url_info['team1_slug']} vs {url_info['team2_slug']}")
        logger.info(f"[URL VERIFY] Scraped team names: {match.team1_name} vs {match.team2_name}")
        
        # Match venue to database
        venue_db = None
        if match.venue:
            fmt = format_type_to_model_format(match.format_type) if match.format_type else 'T20'
            venue_match = scraper.match_venue_to_db(match.venue, match.gender, format_type=fmt)
            if venue_match:
                venue_db = {'venue_id': venue_match[0], 'name': venue_match[1]}
        
        # Match teams to database
        team1_db = None
        team2_db = None
        
        # Helper function to load team from database
        def load_team_from_db(team_id, team_name, gender):
            """Load recent lineup from database for a team."""
            from src.utils.role_classifier import infer_role_from_stats
            
            conn = get_connection()
            cursor = conn.cursor()
            
            # Get the most recent match for this team
            cursor.execute("""
                SELECT m.match_id FROM matches m
                WHERE (m.team1_id = ? OR m.team2_id = ?)
                  AND m.match_type = 'T20' AND m.gender = ?
                ORDER BY m.date DESC LIMIT 1
            """, (team_id, team_id, gender))
            match_row = cursor.fetchone()
            
            if not match_row:
                conn.close()
                return []
            
            match_id = match_row['match_id']
            
            # Get players from that match
            cursor.execute("""
                SELECT 
                    p.player_id, p.name,
                    (SELECT SUM(overs_bowled) FROM player_match_stats WHERE player_id = p.player_id) as overs,
                    (SELECT SUM(wickets_taken) FROM player_match_stats WHERE player_id = p.player_id) as wickets,
                    (SELECT SUM(runs_scored) FROM player_match_stats WHERE player_id = p.player_id) as runs,
                    (SELECT SUM(balls_faced) FROM player_match_stats WHERE player_id = p.player_id) as balls,
                    (SELECT SUM(stumpings) FROM player_match_stats WHERE player_id = p.player_id) as stumpings
                FROM player_match_stats pms
                JOIN players p ON pms.player_id = p.player_id
                WHERE pms.match_id = ? AND pms.team_id = ?
                ORDER BY pms.batting_position
            """, (match_id, team_id))
            
            players = []
            for row in cursor.fetchall():
                stats = {
                    'total_matches': 1,
                    'runs_scored': row['runs'] or 0,
                    'balls_faced': row['balls'] or 0,
                    'overs_bowled': row['overs'] or 0,
                    'wickets_taken': row['wickets'] or 0,
                    'stumpings': row['stumpings'] or 0
                }
                role = infer_role_from_stats(stats)
                # role can be an enum or string, handle both
                role_name = role.value if hasattr(role, 'value') else str(role)
                role_str = 'WK' if 'keeper' in role_name.lower() else ('All Rounder' if 'all' in role_name.lower() else ('Bowler' if 'bowl' in role_name.lower() else 'Batter'))
                
                players.append({
                    'crex_id': None,
                    'name': row['name'],
                    'short_name': row['name'].split()[0] if row['name'] else 'Unknown',
                    'role': role_str,
                    'is_captain': False,
                    'is_wicketkeeper': 'keeper' in role_name.lower(),
                    'is_overseas': False,
                    'db_player_id': row['player_id']
                })
            
            conn.close()
            logger.info(f"Loaded {len(players)} players from database for {team_name}")
            return players
        
        if match.team1:
            logger.info(f"[CREX DEBUG] Team1 '{match.team1.name}' has {len(match.team1.players)} players from CREX scrape")
            if match.team1.players:
                logger.info(f"[CREX DEBUG] Team1 player names: {[p.name for p in match.team1.players[:5]]}...")
            
            team_match = scraper.match_team_to_db(match.team1, match.gender)
            if team_match:
                team1_db = {'team_id': team_match[0], 'name': team_match[1]}
                logger.info(f"[CREX DEBUG] Team1 matched to DB: {team_match[1]} (ID: {team_match[0]})")
            else:
                logger.warning(f"[CREX DEBUG] Team1 '{match.team1.name}' NOT matched to any DB team")
            
            # Always try to match players, even if team doesn't match
            # For franchise teams like SA20/IPL, players are often international players in our database
            team_name_for_matching = team_match[1] if team_match else None
            scraper.match_players_to_db(match.team1, team_name_for_matching, match.gender)
            
            # Log matching results
            matched_count = sum(1 for p in match.team1.players if p.db_player_id)
            logger.info(f"[CREX DEBUG] Team1 player matching: {matched_count}/{len(match.team1.players)} matched to DB")
            if matched_count == 0 and match.team1.players:
                logger.warning(f"[CREX DEBUG] NO players matched for Team1! Check name formats:")
                for p in match.team1.players[:3]:
                    logger.warning(f"[CREX DEBUG]   - '{p.name}' (db_player_id={p.db_player_id})")
        
        if match.team2:
            logger.info(f"[CREX DEBUG] Team2 '{match.team2.name}' has {len(match.team2.players)} players from CREX scrape")
            if match.team2.players:
                logger.info(f"[CREX DEBUG] Team2 player names: {[p.name for p in match.team2.players[:5]]}...")
            
            team_match = scraper.match_team_to_db(match.team2, match.gender)
            if team_match:
                team2_db = {'team_id': team_match[0], 'name': team_match[1]}
                logger.info(f"[CREX DEBUG] Team2 matched to DB: {team_match[1]} (ID: {team_match[0]})")
            else:
                logger.warning(f"[CREX DEBUG] Team2 '{match.team2.name}' NOT matched to any DB team")
            
            # Always try to match players, even if team doesn't match
            team_name_for_matching = team_match[1] if team_match else None
            scraper.match_players_to_db(match.team2, team_name_for_matching, match.gender)
            
            # Log matching results
            matched_count = sum(1 for p in match.team2.players if p.db_player_id)
            logger.info(f"[CREX DEBUG] Team2 player matching: {matched_count}/{len(match.team2.players)} matched to DB")
            if matched_count == 0 and match.team2.players:
                logger.warning(f"[CREX DEBUG] NO players matched for Team2! Check name formats:")
                for p in match.team2.players[:3]:
                    logger.warning(f"[CREX DEBUG]   - '{p.name}' (db_player_id={p.db_player_id})")
        
        # ============================================================
        # AFFIRMATIVE TEAM IDENTITY ASSIGNMENT
        # 
        # Instead of detecting swaps, we use player affiliations as the
        # authoritative source of truth for team identity.
        #
        # After matching all players to database, we check which team
        # each player has most recently played for. The team with the
        # majority of players from a squad becomes that squad's identity.
        #
        # This handles cases where CREX HTML has teams in wrong order
        # (e.g., Perth Scorchers players under Sydney Sixers tab).
        # ============================================================
        
        # Track if we detected and corrected a swap (for UI notification)
        team_swap_detected = False
        team_swap_details = {}
        
        def get_player_match_rate(players):
            """Calculate what % of players were matched to DB."""
            if not players:
                return 0.0
            matched = sum(1 for p in players if p.db_player_id)
            return matched / len(players)
        
        # Calculate player match rates
        team1_match_rate = get_player_match_rate(match.team1.players) if match.team1 and match.team1.players else 0
        team2_match_rate = get_player_match_rate(match.team2.players) if match.team2 and match.team2.players else 0
        
        logger.info(f"[TEAM ASSIGNMENT] Team1 match rate: {team1_match_rate:.0%}, Team2 match rate: {team2_match_rate:.0%}")
        
        # Use player affiliations to determine actual team identity
        squad1_team = None
        squad2_team = None
        
        if match.team1 and match.team1.players:
            squad1_team = scraper.determine_team_from_players(
                match.team1.players, 'T20', match.gender
            )
            if squad1_team:
                logger.info(f"[TEAM ASSIGNMENT] Squad1 affiliation: {squad1_team['player_count']}/{squad1_team['total_matched']} players from '{squad1_team['team_name']}' (confidence: {squad1_team['confidence']:.0%})")
                if squad1_team.get('all_affiliations') and len(squad1_team['all_affiliations']) > 1:
                    logger.warning(f"[TEAM ASSIGNMENT] Squad1 has mixed affiliations: {squad1_team['all_affiliations']}")
        
        if match.team2 and match.team2.players:
            squad2_team = scraper.determine_team_from_players(
                match.team2.players, 'T20', match.gender
            )
            if squad2_team:
                logger.info(f"[TEAM ASSIGNMENT] Squad2 affiliation: {squad2_team['player_count']}/{squad2_team['total_matched']} players from '{squad2_team['team_name']}' (confidence: {squad2_team['confidence']:.0%})")
                if squad2_team.get('all_affiliations') and len(squad2_team['all_affiliations']) > 1:
                    logger.warning(f"[TEAM ASSIGNMENT] Squad2 has mixed affiliations: {squad2_team['all_affiliations']}")
        
        # Now determine team assignments based on affiliations
        # IMPORTANT: Only use affiliations as FALLBACK when team name matching failed.
        # For domestic leagues (NZ Super Smash, etc.) players play for multiple teams
        # internationally, so their affiliations are scattered and NOT reliable for
        # determining their domestic team.
        #
        # Priority: Team name match > Player affiliation (fallback only)
        
        AFFILIATION_CONFIDENCE_THRESHOLD = 0.6  # At least 60% of matched players from same team
        AFFILIATION_MIN_PLAYERS = 5  # Need at least 5 players with affiliations to trust it
        
        # Determine final team1_db and team2_db based on affiliations
        original_team1_db = team1_db.copy() if team1_db else None
        original_team2_db = team2_db.copy() if team2_db else None
        
        # Squad 1: ONLY use affiliation if we couldn't match the team name
        if not team1_db and squad1_team and squad1_team['confidence'] >= AFFILIATION_CONFIDENCE_THRESHOLD:
            # No CREX match, but we have affiliation - use it as fallback
            if squad1_team.get('total_matched', 0) >= AFFILIATION_MIN_PLAYERS:
                logger.info(f"[TEAM ASSIGNMENT] Inferred Team1 as '{squad1_team['team_name']}' from player affiliations (fallback)")
                team1_db = {'team_id': squad1_team['team_id'], 'name': squad1_team['team_name']}
            else:
                logger.warning(f"[TEAM ASSIGNMENT] Squad1 affiliation low sample size ({squad1_team.get('total_matched', 0)} players) - not using")
        elif team1_db:
            # Team name match succeeded - trust it, ignore affiliations
            logger.info(f"[TEAM ASSIGNMENT] Team1 matched via name/abbreviation: '{team1_db['name']}' - trusting this match")
        
        # Squad 2: ONLY use affiliation if we couldn't match the team name
        if not team2_db and squad2_team and squad2_team['confidence'] >= AFFILIATION_CONFIDENCE_THRESHOLD:
            # No CREX match, but we have affiliation - use it as fallback
            if squad2_team.get('total_matched', 0) >= AFFILIATION_MIN_PLAYERS:
                logger.info(f"[TEAM ASSIGNMENT] Inferred Team2 as '{squad2_team['team_name']}' from player affiliations (fallback)")
                team2_db = {'team_id': squad2_team['team_id'], 'name': squad2_team['team_name']}
            else:
                logger.warning(f"[TEAM ASSIGNMENT] Squad2 affiliation low sample size ({squad2_team.get('total_matched', 0)} players) - not using")
        elif team2_db:
            # Team name match succeeded - trust it, ignore affiliations
            logger.info(f"[TEAM ASSIGNMENT] Team2 matched via name/abbreviation: '{team2_db['name']}' - trusting this match")
        
        # Edge case: Both squads assigned to the same team
        # This can happen when:
        # 1. CREX completely swapped the teams
        # 2. One squad's affiliation was below threshold but is still the correct team
        if team1_db and team2_db and team1_db['team_id'] == team2_db['team_id']:
            logger.error(f"[TEAM ASSIGNMENT] Both squads assigned to same team '{team1_db['name']}'! Resolving conflict...")
            
            # Use affiliation data to resolve the conflict
            # The squad with higher confidence keeps the team, the other gets its secondary affiliation
            squad1_conf = squad1_team['confidence'] if squad1_team else 0
            squad2_conf = squad2_team['confidence'] if squad2_team else 0
            
            resolved = False
            
            if squad1_conf >= squad2_conf and squad2_team and squad2_team.get('all_affiliations'):
                # Squad1 wins the conflict, Squad2 gets its best alternative team
                # Find the team in squad2's affiliations that is NOT the conflicting team
                for alt_team_id, count in sorted(squad2_team['all_affiliations'].items(), key=lambda x: -x[1]):
                    if alt_team_id != team1_db['team_id']:
                        # Get team name for this alternative
                        conn = get_connection()
                        cursor = conn.cursor()
                        cursor.execute("SELECT name FROM teams WHERE team_id = ?", (alt_team_id,))
                        result = cursor.fetchone()
                        conn.close()
                        
                        if result:
                            alt_team_name = result['name']
                            logger.info(f"[TEAM ASSIGNMENT] Resolving: Team2 reassigned to '{alt_team_name}' (next best affiliation with {count} players)")
                            team2_db = {'team_id': alt_team_id, 'name': alt_team_name}
                            team_swap_detected = True
                            team_swap_details['team2'] = {'from': team1_db['name'], 'to': alt_team_name}
                            resolved = True
                            break
            
            elif squad2_conf > squad1_conf and squad1_team and squad1_team.get('all_affiliations'):
                # Squad2 wins the conflict, Squad1 gets its best alternative team
                for alt_team_id, count in sorted(squad1_team['all_affiliations'].items(), key=lambda x: -x[1]):
                    if alt_team_id != team2_db['team_id']:
                        conn = get_connection()
                        cursor = conn.cursor()
                        cursor.execute("SELECT name FROM teams WHERE team_id = ?", (alt_team_id,))
                        result = cursor.fetchone()
                        conn.close()
                        
                        if result:
                            alt_team_name = result['name']
                            logger.info(f"[TEAM ASSIGNMENT] Resolving: Team1 reassigned to '{alt_team_name}' (next best affiliation with {count} players)")
                            team1_db = {'team_id': alt_team_id, 'name': alt_team_name}
                            team_swap_detected = True
                            team_swap_details['team1'] = {'from': team2_db['name'], 'to': alt_team_name}
                            resolved = True
                            break
            
            if not resolved:
                # Fallback: try to use original team names if they were different
                if original_team1_db and original_team2_db and original_team1_db['team_id'] != original_team2_db['team_id']:
                    logger.info(f"[TEAM ASSIGNMENT] Fallback: Swapping to original assignments")
                    team1_db, team2_db = original_team2_db, original_team1_db
                    match.team1, match.team2 = match.team2, match.team1
                    team_swap_detected = True
                    team_swap_details['full_swap'] = True
                else:
                    logger.error(f"[TEAM ASSIGNMENT] Could not resolve team conflict! Both squads remain as '{team1_db['name']}'")
                    team_swap_details['unresolved_conflict'] = True
        
        # Log final assignments
        if team_swap_detected:
            logger.warning(f"[TEAM ASSIGNMENT] SWAP DETECTED - Final: Team1='{team1_db['name'] if team1_db else 'Unknown'}', Team2='{team2_db['name'] if team2_db else 'Unknown'}'")
        else:
            logger.info(f"[TEAM ASSIGNMENT] Final: Team1='{team1_db['name'] if team1_db else 'Unknown'}', Team2='{team2_db['name'] if team2_db else 'Unknown'}'")
        
        # Propagate DB team IDs back to the CREXTeam objects
        if team1_db and match.team1:
            match.team1.db_team_id = team1_db['team_id']
        if team2_db and match.team2:
            match.team2.db_team_id = team2_db['team_id']
        
        # CREX uses JavaScript tabs for squads - only one team's data is in the HTML
        # If a team has no players, load from database as fallback
        if match.team1 and not match.team1.players and team1_db:
            logger.info(f"Loading {match.team1.name} squad from database (not in CREX HTML)")
            db_players = load_team_from_db(team1_db['team_id'], team1_db['name'], match.gender)
            # Convert to CREXPlayer format and add to team
            for p in db_players:
                from src.api.crex_scraper import CREXPlayer
                match.team1.players.append(CREXPlayer(
                    crex_id=p['crex_id'],
                    name=p['name'],
                    short_name=p['short_name'],
                    role=p['role'],
                    is_captain=p['is_captain'],
                    is_wicketkeeper=p['is_wicketkeeper'],
                    is_overseas=p['is_overseas'],
                    db_player_id=p['db_player_id']
                ))
        
        if match.team2 and not match.team2.players and team2_db:
            logger.info(f"Loading {match.team2.name} squad from database (not in CREX HTML)")
            db_players = load_team_from_db(team2_db['team_id'], team2_db['name'], match.gender)
            for p in db_players:
                from src.api.crex_scraper import CREXPlayer
                match.team2.players.append(CREXPlayer(
                    crex_id=p['crex_id'],
                    name=p['name'],
                    short_name=p['short_name'],
                    role=p['role'],
                    is_captain=p['is_captain'],
                    is_wicketkeeper=p['is_wicketkeeper'],
                    is_overseas=p['is_overseas'],
                    db_player_id=p['db_player_id']
                ))
        
        # Also handle case where team object is None (CREX couldn't create it)
        # First, try to match team names to database if we don't have team_db yet
        def find_team_in_db(team_name, gender):
            """Find team in database by name using improved matching."""
            from difflib import SequenceMatcher
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT t.team_id, t.name 
                FROM teams t
                JOIN matches m ON t.team_id IN (m.team1_id, m.team2_id)
                WHERE m.gender = ? AND m.match_type = 'T20'
            """, (gender,))
            teams = cursor.fetchall()
            conn.close()
            
            # Direction words that must match exactly
            directions = {'northern', 'southern', 'eastern', 'western', 'central'}
            
            def get_direction(name):
                words = name.lower().split()
                for word in words:
                    if word in directions:
                        return word
                return None
            
            search_direction = get_direction(team_name)
            
            best_match = None
            best_score = 0
            for row in teams:
                db_name = row['name']
                db_direction = get_direction(db_name)
                
                # If both have direction words, they MUST match
                if search_direction and db_direction:
                    if search_direction != db_direction:
                        continue  # Skip - wrong direction (Northern vs Southern)
                
                score = SequenceMatcher(None, team_name.lower(), db_name.lower()).ratio()
                
                # Require higher threshold (0.8 instead of 0.7)
                if score > best_score and score >= 0.8:
                    best_score = score
                    best_match = (row['team_id'], db_name)
            
            if best_match:
                logger.info(f"Matched team '{team_name}' to '{best_match[1]}' (score: {best_score:.2f})")
            
            return best_match
        
        if not team1_db and hasattr(match, 'team1_name') and match.team1_name:
            # First try the scraper's abbreviation-aware matching (handles USA, QAT, OMA, etc.)
            # Use team name as abbreviation too (NOT the CREX internal ID which can collide
            # with domestic league abbreviations, e.g. CREX ID "BH" -> Brisbane Heat not Bahrain)
            from src.api.crex_scraper import CREXTeam as _CREXTeam
            temp_team = _CREXTeam(
                crex_id=getattr(match, 'team1_id', ''),
                name=match.team1_name,
                abbreviation=match.team1_name
            )
            team_match = scraper.match_team_to_db(temp_team, match.gender)
            # Fall back to simpler fuzzy match if abbreviation matching didn't work
            if not team_match:
                team_match = find_team_in_db(match.team1_name, match.gender)
            if team_match:
                team1_db = {'team_id': team_match[0], 'name': team_match[1]}
                logger.info(f"Matched team name '{match.team1_name}' to DB: {team1_db['name']}")
        
        if not team2_db and hasattr(match, 'team2_name') and match.team2_name:
            # First try the scraper's abbreviation-aware matching (handles USA, QAT, OMA, etc.)
            from src.api.crex_scraper import CREXTeam as _CREXTeam
            temp_team = _CREXTeam(
                crex_id=getattr(match, 'team2_id', ''),
                name=match.team2_name,
                abbreviation=match.team2_name
            )
            team_match = scraper.match_team_to_db(temp_team, match.gender)
            # Fall back to simpler fuzzy match if abbreviation matching didn't work
            if not team_match:
                team_match = find_team_in_db(match.team2_name, match.gender)
            if team_match:
                team2_db = {'team_id': team_match[0], 'name': team_match[1]}
                logger.info(f"Matched team name '{match.team2_name}' to DB: {team2_db['name']}")
        
        if not match.team1 and team1_db:
            team_name = match.team1_name if hasattr(match, 'team1_name') else team1_db['name']
            logger.info(f"Creating {team_name} from database (no CREX data)")
            from src.api.crex_scraper import CREXTeam, CREXPlayer
            db_players = load_team_from_db(team1_db['team_id'], team1_db['name'], match.gender)
            if db_players:
                match.team1 = CREXTeam(
                    crex_id=getattr(match, 'team1_id', ''),
                    name=team_name,
                    abbreviation='',
                    players=[CREXPlayer(
                        crex_id=p['crex_id'],
                        name=p['name'],
                        short_name=p['short_name'],
                        role=p['role'],
                        is_captain=p['is_captain'],
                        is_wicketkeeper=p['is_wicketkeeper'],
                        is_overseas=p['is_overseas'],
                        db_player_id=p['db_player_id']
                    ) for p in db_players]
                )
        
        if not match.team2 and team2_db:
            team_name = match.team2_name if hasattr(match, 'team2_name') else team2_db['name']
            logger.info(f"Creating {team_name} from database (no CREX data)")
            from src.api.crex_scraper import CREXTeam, CREXPlayer
            db_players = load_team_from_db(team2_db['team_id'], team2_db['name'], match.gender)
            if db_players:
                match.team2 = CREXTeam(
                    crex_id=getattr(match, 'team2_id', ''),
                    name=team_name,
                    abbreviation='',
                    players=[CREXPlayer(
                        crex_id=p['crex_id'],
                        name=p['name'],
                        short_name=p['short_name'],
                        role=p['role'],
                        is_captain=p['is_captain'],
                        is_wicketkeeper=p['is_wicketkeeper'],
                        is_overseas=p['is_overseas'],
                        db_player_id=p['db_player_id']
                    ) for p in db_players]
                )
        
        def team_to_dict(team):
            if not team:
                return None
            # Look up DB team name if we have a db_team_id
            db_team_name = None
            if team.db_team_id:
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM teams WHERE team_id = ?", (team.db_team_id,))
                row = cursor.fetchone()
                conn.close()
                if row:
                    db_team_name = row['name']
            return {
                'crex_id': team.crex_id,
                'name': team.name,
                'abbreviation': team.abbreviation,
                'db_team_id': team.db_team_id,
                'db_team_name': db_team_name,
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
        
        # Collect any warnings from the scraper (auto-created teams, unmatched teams, etc.)
        match_warnings = list(scraper._warnings)
        scraper._warnings.clear()
        
        # Add warnings and compute tier-appropriate default ELO for unmatched teams
        from src.utils.default_elo import infer_default_elo

        default_elo_team1 = None
        default_elo_team2 = None
        if not team1_db:
            default_elo_team1 = infer_default_elo(
                series_name=match.series_name,
                matched_opponent_team_id=team2_db['team_id'] if team2_db else None,
                format_type=match.format_type or 'T20',
            )
            match_warnings.append(
                f"Team '{match.team1_name}' could not be identified. "
                f"Using default ELO ({int(default_elo_team1)}) based on competition tier. Prediction may be unreliable."
            )
        if not team2_db:
            default_elo_team2 = infer_default_elo(
                series_name=match.series_name,
                matched_opponent_team_id=team1_db['team_id'] if team1_db else None,
                format_type=match.format_type or 'T20',
            )
            match_warnings.append(
                f"Team '{match.team2_name}' could not be identified. "
                f"Using default ELO ({int(default_elo_team2)}) based on competition tier. Prediction may be unreliable."
            )

        result = {
            'success': True,
            'source': 'crex',
            'warnings': match_warnings,
            'match': {
                'crex_id': match.crex_id,
                'title': match.title,
                'team1_name': match.team1_name,
                'team2_name': match.team2_name,
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
                'default_elo_team1': default_elo_team1,
                'default_elo_team2': default_elo_team2,
                'toss_winner': match.toss_winner,
                'toss_decision': match.toss_decision,
                'team_swap_detected': team_swap_detected,
                'team_swap_details': team_swap_details if team_swap_detected else None,
                'url_verification': {
                    'team1_slug': url_info['team1_slug'],
                    'team2_slug': url_info['team2_slug'],
                    'team1_id': url_info['team1_id'],
                    'team2_id': url_info['team2_id'],
                }
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
# Integrations API Routes
# ============================================================================

@app.route('/api/integrations/credentials-status', methods=['GET'])
def get_integrations_credential_status():
    """
    Get safe credential readiness for external market integrations.

    Returns:
        JSON with provider readiness and masked previews only.
    """
    try:
        return jsonify(get_market_credentials_status())
    except Exception as e:
        logger.error(f"Error getting integration credential status: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/integrations/polymarket/health', methods=['GET'])
def polymarket_health():
    """Polymarket CLOB health check."""
    try:
        client = get_polymarket_client()
        payload = client.health_check()
        return jsonify(payload)
    except Exception as e:
        logger.error(f"Error in Polymarket health check: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/integrations/polymarket/markets', methods=['GET'])
def polymarket_markets():
    """Fetch Polymarket markets from public Gamma API."""
    try:
        client = get_polymarket_client()
        limit = int(request.args.get('limit', 20))
        active = request.args.get('active', 'true').lower() == 'true'
        closed = request.args.get('closed', 'false').lower() == 'true'
        markets = client.get_markets(limit=limit, active=active, closed=closed)
        return jsonify({
            'success': True,
            'count': len(markets) if isinstance(markets, list) else None,
            'markets': markets,
        })
    except Exception as e:
        logger.error(f"Error fetching Polymarket markets: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/integrations/polymarket/orderbook', methods=['GET'])
def polymarket_orderbook():
    """Fetch Polymarket CLOB order book for a token."""
    try:
        token_id = request.args.get('token_id')
        if not token_id:
            return jsonify({'success': False, 'error': 'token_id parameter required'}), 400
        client = get_polymarket_client()
        book = client.get_clob_order_book(token_id)
        return jsonify({'success': True, 'orderbook': book})
    except Exception as e:
        logger.error(f"Error fetching Polymarket orderbook: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/integrations/polymarket/compare', methods=['GET'])
def polymarket_compare():
    """Compare fixture model odds vs Polymarket implied odds."""
    try:
        team1 = (request.args.get('team1') or '').strip()
        team2 = (request.args.get('team2') or '').strip()
        if not team1 or not team2:
            return jsonify({'success': False, 'error': 'team1 and team2 parameters required'}), 400

        start_utc = request.args.get('start_utc')
        series = request.args.get('series')
        model_team1_win_pct = request.args.get('model_team1_win_pct', type=float)
        model_team2_win_pct = request.args.get('model_team2_win_pct', type=float)

        service = get_polymarket_compare_service()
        payload = service.compare_fixture(
            team1=team1,
            team2=team2,
            start_utc=start_utc,
            series=series,
            model_team1_win_pct=model_team1_win_pct,
            model_team2_win_pct=model_team2_win_pct,
        )
        return jsonify(payload)
    except Exception as e:
        logger.error(f"Error comparing Polymarket odds: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/integrations/polymarket/compare/batch', methods=['POST'])
def polymarket_compare_batch():
    """Batch Polymarket comparison for bulk row rendering."""
    try:
        payload = request.get_json(silent=True) or {}
        fixtures = payload.get('fixtures') or []
        if not isinstance(fixtures, list):
            return jsonify({'success': False, 'error': 'fixtures must be an array'}), 400

        service = get_polymarket_compare_service()
        result = service.compare_batch(fixtures)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error comparing batch Polymarket odds: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/integrations/polymarket/compare/multi', methods=['POST'])
def polymarket_compare_multi():
    """Wave 5: multi-market Polymarket comparison for a single fixture.

    Body:
        {
            "team1": str,
            "team2": str,
            "start_utc": str (optional),
            "series": str (optional),
            "market_probs": dict from derive_polymarket_market_probs (required)
        }

    Returns the 4-market structure (moneyline, top_batter, most_sixes,
    toss_match_double) with model probabilities, market prices, and edge
    in percentage points per outcome.
    """
    try:
        payload = request.get_json(silent=True) or {}
        team1 = (payload.get('team1') or '').strip()
        team2 = (payload.get('team2') or '').strip()
        if not team1 or not team2:
            return jsonify({'success': False, 'error': 'team1 and team2 required'}), 400
        market_probs = payload.get('market_probs') or {}
        if not market_probs:
            return jsonify({
                'success': False,
                'error': 'market_probs is required (from derive_polymarket_market_probs)',
            }), 400

        service = get_polymarket_compare_service()
        result = service.compare_fixture_multi(
            team1=team1,
            team2=team2,
            market_probs=market_probs,
            start_utc=payload.get('start_utc'),
            series=payload.get('series'),
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in multi-market Polymarket comparison: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/integrations/polymarket/compare/multi/batch', methods=['POST'])
def polymarket_compare_multi_batch():
    """Wave 5: batch multi-market Polymarket comparison.

    Body:
        {
            "fixtures": [
                {
                    "team1": str, "team2": str,
                    "start_utc": str?, "series": str?,
                    "market_probs": {moneyline:..., top_batter:..., most_sixes:..., toss_match_double:...},
                    "row_key": str?,
                },
                ...
            ]
        }
    """
    try:
        payload = request.get_json(silent=True) or {}
        fixtures = payload.get('fixtures') or []
        if not isinstance(fixtures, list):
            return jsonify({'success': False, 'error': 'fixtures must be an array'}), 400

        service = get_polymarket_compare_service()
        result = service.compare_batch_multi(fixtures)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in multi-market batch comparison: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# Wave 5 Phase 6c-d: Live Betting routes
# ============================================================================

@app.route('/live-betting')
def live_betting_page():
    """Wave 5 Phase 6d: Live Betting tab."""
    return render_template('live_betting.html')


@app.route('/api/betting/config', methods=['GET'])
def betting_config():
    """Current betting risk-gate state + caps + remaining-cap snapshot."""
    try:
        from src.integrations.polymarket.risk_gate import get_risk_status
        return jsonify(get_risk_status())
    except Exception as e:
        logger.error(f"Error getting betting config: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/betting/strategies', methods=['GET'])
def betting_strategies():
    """Wave 5.8: per-strategy bankroll + P&L rollup for the Live Betting UI.

    Returns starting bankroll, current bankroll, open exposure, realised P&L,
    ROI, win rate and bet counts for each strategy in BETTING_LIVE_STRATEGIES
    plus any retired strategy that still has historical real bets.
    """
    try:
        from src.integrations.polymarket.risk_gate import get_strategy_breakdown
        return jsonify(get_strategy_breakdown())
    except Exception as e:
        logger.error(f"Error getting betting strategies: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/betting/wallet', methods=['GET'])
def betting_wallet():
    """Wave 5.8: live on-chain USDC balance + approval status for the bot wallet.

    Includes wallet-driven deploy cap (portfolio × fraction) so the UI reflects
    top-ups automatically.
    """
    try:
        from src.integrations.polymarket import PolymarketClient
        from config import POLYMARKET_CONFIG
        from src.integrations.polymarket.live_bankroll import (
            get_max_deploy_usdc,
            get_portfolio_value,
        )
        pm = PolymarketClient()
        if not POLYMARKET_CONFIG.get('private_key'):
            return jsonify({
                'success': False,
                'error': 'POLYGON_PRIVATE_KEY not set in .env',
                'configured': False,
            })
        info = pm.get_usdc_balance()
        with get_db_connection() as conn:
            portfolio = get_portfolio_value(conn, pm=pm)
            max_deploy = get_max_deploy_usdc(conn, pm=pm)
        info['portfolio_value_usdc'] = round(portfolio, 2)
        info['max_deploy_usdc'] = round(max_deploy, 2)
        info['max_deposit_usdc'] = round(max_deploy, 2)  # legacy key for UI
        cash = float(info.get('balance_usdc') or 0.0)
        info['cash_pct_of_portfolio'] = (
            round(cash / portfolio * 100, 1) if portfolio > 0 else None
        )
        info['funded_vs_envelope_pct'] = info['cash_pct_of_portfolio']
        info['underfunded'] = portfolio > 0 and cash < 0.15 * portfolio
        info['success'] = True
        info['configured'] = True
        return jsonify(info)
    except Exception as e:
        logger.error(f"Error fetching wallet balance: {e}")
        return jsonify({'success': False, 'error': str(e), 'configured': True}), 500


@app.route('/api/betting/bankroll-history', methods=['GET'])
def betting_bankroll_history():
    """Bankroll-over-time per strategy for LIVE bets (bet_kind='real').
    
    Mirrors the paper trading endpoint but filters for real bets only.
    Computes running bankroll chronologically: starting + cumsum(pnl)
    ordered by settled_at.
    """
    try:
        from config import BETTING_CONFIG
        from src.integrations.polymarket.live_bankroll import get_strategy_bankroll

        live_strategies = BETTING_CONFIG.get('live_strategies', []) or []
        default_starting = float(BETTING_CONFIG.get('max_deposit_per_strategy_usdc', 100))

        def _strat_starting(name: str, conn) -> float:
            if name in live_strategies:
                bankroll = get_strategy_bankroll(name, conn)
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT COALESCE(SUM(pnl_realised_usdc), 0.0) AS realised
                    FROM bet_ledger
                    WHERE COALESCE(bet_kind, 'real') = 'real'
                      AND status = 'settled'
                      AND strategy_label = ?
                    """,
                    (name,),
                )
                realised = float(cur.fetchone()["realised"] or 0.0)
                return max(0.0, bankroll - realised)
            import os
            key = f"BETTING_MAX_DEPOSIT_{name.upper().replace('-', '_')}"
            v = os.getenv(key)
            return float(v) if v is not None else default_starting

        from collections import defaultdict

        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT strategy_label, settled_at, pnl_realised_usdc, side_label, fixture_key
                FROM bet_ledger
                WHERE COALESCE(bet_kind, 'real') = 'real' AND status = 'settled'
                ORDER BY strategy_label, settled_at ASC
                """
            )
            rows = [dict(r) for r in cur.fetchall()]

            by_strategy = {}
            for strat_name in live_strategies:
                base = _strat_starting(strat_name, conn)
                by_strategy[strat_name] = [{
                    "timestamp": None,
                    "bankroll": round(base, 2),
                    "side_label": "starting",
                    "fixture_key": "",
                    "pnl": 0.0,
                }]

            grouped = defaultdict(list)
            for r in rows:
                grouped[r["strategy_label"] or "(none)"].append(r)

            for name, lst in grouped.items():
                base = _strat_starting(name, conn)
                running = base
                if name not in by_strategy:
                    by_strategy[name] = [{
                        "timestamp": None,
                        "bankroll": round(base, 2),
                        "side_label": "starting",
                        "fixture_key": "",
                        "pnl": 0.0,
                    }]
                for r in lst:
                    pnl = float(r.get("pnl_realised_usdc") or 0.0)
                    running += pnl
                    by_strategy[name].append({
                        "timestamp": r["settled_at"],
                        "bankroll": round(running, 2),
                        "side_label": r.get("side_label"),
                        "fixture_key": r.get("fixture_key"),
                        "pnl": round(pnl, 2),
                    })

            starting_map = {
                name: round(_strat_starting(name, conn), 2)
                for name in set(live_strategies) | set(by_strategy.keys())
            }
        return jsonify({
            "success": True,
            "history": by_strategy,
            "live_strategies": list(live_strategies),
            "starting_by_strategy": starting_map,
            "wallet_driven": True,
        })
    except Exception as e:
        logger.error(f"Error in /api/betting/bankroll-history: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/betting/portfolio', methods=['GET'])
def betting_portfolio():
    """Wave 5.8: total portfolio value = wallet USDC + mark-to-market open positions.

    For each real + filled + not-settled bet in bet_ledger we compute the
    shares held (= fill_size_usdc / fill_price) and multiply by the current
    CLOB midpoint to get the present market value of those tokens.
    Portfolio value = wallet balance + sum of (shares * current midpoint).
    """
    try:
        from src.integrations.polymarket import PolymarketClient
        from config import POLYMARKET_CONFIG

        pm = PolymarketClient()
        if not POLYMARKET_CONFIG.get('private_key'):
            return jsonify({
                'success': False,
                'error': 'POLYGON_PRIVATE_KEY not set in .env',
                'configured': False,
            })

        # 1. Wallet cash
        wallet = pm.get_usdc_balance()
        wallet_cash = float(wallet.get('balance_usdc', 0.0))

        # 2. Open positions with their cost basis + token id
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT bet_id, polymarket_token_id AS token_id,
                       fill_size_usdc, fill_price, side_label, strategy_label
                FROM bet_ledger
                WHERE COALESCE(bet_kind, 'real') = 'real'
                  AND status = 'filled'
                  AND settled_at IS NULL
                  AND fill_price IS NOT NULL
                  AND fill_price > 0
                  AND polymarket_token_id IS NOT NULL
                  AND polymarket_token_id != ''
                """
            )
            positions = [dict(r) for r in cur.fetchall()]

        # 3. Batched midpoint query for all unique tokens
        unique_tokens = sorted({p['token_id'] for p in positions})
        midpoints: dict = {}
        if unique_tokens:
            try:
                midpoints = pm.get_token_midpoints(unique_tokens)
            except Exception as exc:
                logger.warning(f"portfolio: get_token_midpoints failed, falling back to cost basis: {exc}")
                midpoints = {}

        # 4. Mark each position to market; fallback to cost basis when
        #    the CLOB didn't quote a midpoint for that token.
        total_cost_basis = 0.0
        total_market_value = 0.0
        position_rows = []
        for p in positions:
            fill_price = float(p['fill_price'])
            cost_basis = float(p['fill_size_usdc'] or 0.0)
            shares = cost_basis / fill_price if fill_price > 0 else 0.0
            mid = midpoints.get(str(p['token_id']))
            if mid is None or mid <= 0:
                market_value = cost_basis  # fallback
                marked = False
            else:
                market_value = shares * mid
                marked = True
            total_cost_basis += cost_basis
            total_market_value += market_value
            position_rows.append({
                'bet_id': p['bet_id'],
                'strategy_label': p['strategy_label'],
                'side_label': p['side_label'],
                'shares': round(shares, 4),
                'fill_price': round(fill_price, 4),
                'cost_basis_usdc': round(cost_basis, 4),
                'mid_price': round(mid, 4) if mid is not None else None,
                'market_value_usdc': round(market_value, 4),
                'unrealised_pnl_usdc': round(market_value - cost_basis, 4),
                'marked_to_market': marked,
            })

        portfolio_value = wallet_cash + total_market_value
        unrealised_pnl = total_market_value - total_cost_basis

        return jsonify({
            'success': True,
            'configured': True,
            'wallet_cash_usdc': round(wallet_cash, 2),
            'open_positions_count': len(positions),
            'open_positions_cost_basis_usdc': round(total_cost_basis, 2),
            'open_positions_market_value_usdc': round(total_market_value, 2),
            'unrealised_pnl_usdc': round(unrealised_pnl, 2),
            'portfolio_value_usdc': round(portfolio_value, 2),
            'positions': position_rows,
        })
    except Exception as e:
        logger.error(f"Error fetching betting portfolio: {e}")
        return jsonify({'success': False, 'error': str(e), 'configured': True}), 500


@app.route('/api/betting/today', methods=['GET'])
def betting_today():
    """Today's REAL bets (UTC) for the Live Betting tab.

    Wave 5.8: filters bet_kind='real' so paper bets don't contaminate the
    /live-betting view. Paper bets are shown at /paper-trades.
    """
    try:
        from datetime import datetime, timezone
        today_iso = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT * FROM bet_ledger
                WHERE COALESCE(bet_kind, 'real') = 'real'
                  AND proposed_at >= ?
                ORDER BY proposed_at DESC
                LIMIT 100
                """,
                (today_iso,),
            )
            bets = [dict(r) for r in cur.fetchall()]
        return jsonify({'success': True, 'bets': bets})
    except Exception as e:
        logger.error(f"Error fetching today's bets: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


def _enrich_bets_with_sub_fills(conn, bets: List[Dict[str, Any]]) -> None:
    """Attach TWAP chunk sub-fills and cashout SELL detail to bet dicts in-place."""
    if not bets:
        return
    bet_ids = [b["bet_id"] for b in bets if b.get("bet_id") is not None]
    if not bet_ids:
        return

    chunks_by_bet: Dict[int, List[Dict[str, Any]]] = {bid: [] for bid in bet_ids}
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='order_chunks'"
        )
        if cur.fetchone():
            placeholders = ",".join("?" * len(bet_ids))
            cur.execute(
                f"""
                SELECT op.bet_ledger_id,
                       oc.chunk_index,
                       oc.limit_price,
                       oc.size_usdc,
                       oc.fill_price,
                       oc.fill_size_usdc,
                       oc.polymarket_order_id,
                       oc.status,
                       oc.placed_at,
                       oc.filled_at
                FROM order_plans op
                JOIN order_chunks oc ON oc.plan_id = op.plan_id
                WHERE op.bet_ledger_id IN ({placeholders})
                  AND oc.status IN ('filled', 'partially_filled', 'placed')
                ORDER BY op.bet_ledger_id, oc.chunk_index ASC
                """,
                bet_ids,
            )
            for row in cur.fetchall():
                r = dict(row)
                bid = r.pop("bet_ledger_id")
                r["kind"] = "TWAP"
                chunks_by_bet.setdefault(bid, []).append(r)
    except Exception:
        pass

    for bet in bets:
        bid = bet.get("bet_id")
        sub_fills = chunks_by_bet.get(bid, [])
        # Only expose multi-chunk TWAP detail (or any filled chunk with data)
        filled = [
            c for c in sub_fills
            if c.get("status") in ("filled", "partially_filled")
            and (c.get("fill_size_usdc") or c.get("fill_price"))
        ]
        # Always show TWAP entry fills when present — including single-chunk
        # plans (common for v3_marg) so cashout SELL rows aren't orphaned.
        bet["sub_fills"] = filled

        # Fallback: non-TWAP bets with a cashout but no chunk rows — synthesise
        # the entry fill from bet_ledger so BUY + SELL appear as a pair.
        if not bet["sub_fills"] and bet.get("cashout_triggered_at") and bet.get("fill_price"):
            bet["sub_fills"] = [{
                "kind": "BUY",
                "chunk_index": 0,
                "fill_price": bet["fill_price"],
                "fill_size_usdc": bet.get("fill_size_usdc") or bet.get("size_usdc"),
                "filled_at": bet.get("filled_at") or bet.get("placed_at"),
                "polymarket_order_id": bet.get("polymarket_order_id"),
            }]

        if bet.get("cashout_triggered_at") and bet.get("cashout_price"):
            fill_p = float(bet.get("fill_price") or 0)
            fill_sz = float(bet.get("fill_size_usdc") or bet.get("size_usdc") or 0)
            exit_p = float(bet["cashout_price"])
            proceeds = (fill_sz / fill_p * exit_p) if fill_p > 0 else None
            bet["cashout_sub"] = {
                "kind": "CASHOUT",
                "fill_price": exit_p,
                "fill_size_usdc": round(proceeds, 4) if proceeds is not None else None,
                "pnl_usdc": bet.get("cashout_pnl_usdc") or bet.get("pnl_realised_usdc"),
                "filled_at": bet.get("cashout_triggered_at"),
                "order_id": bet.get("cashout_order_id"),
            }


def _fixture_abbr_fallback(fixture_key: Optional[str]) -> tuple:
    """Last-resort team labels from slug tokens."""
    parts = (fixture_key or "").split("-")
    if len(parts) >= 6 and len(parts[-3]) == 4 and parts[-3].isdigit():
        return (parts[-5].upper(), parts[-4].upper())
    if len(parts) >= 4:
        return (parts[1].upper(), parts[2].upper())
    return ("?", "?")


def _teams_from_side_labels(side_labels: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """If bets exist on both sides, we already know both team names."""
    uniq = [s for s in dict.fromkeys(side_labels) if s]
    if len(uniq) >= 2:
        return uniq[0], uniq[1]
    if len(uniq) == 1:
        return uniq[0], None
    return None, None


def _resolve_fixture_meta(
    fixture_key: str,
    fixture_bets: List[Dict[str, Any]],
    gamma_by_market: Dict[str, Any],
) -> Dict[str, str]:
    """Resolve display names without extra HTTP — uses prefetched Gamma + side_labels."""
    from src.integrations.polymarket.bet_display_enrichment import (
        fixture_meta_cache_get,
        fixture_meta_cache_set,
        teams_from_gamma_market,
    )
    from src.integrations.odds.polymarket_compare import PolymarketComparisonService

    cached = fixture_meta_cache_get(fixture_key)
    if cached:
        return cached

    market_id = next(
        (b.get("polymarket_market_id") for b in fixture_bets if b.get("polymarket_market_id")),
        None,
    )
    side_labels = [b.get("side_label") for b in fixture_bets if b.get("side_label")]

    t1, t2 = None, None
    if market_id:
        market = gamma_by_market.get(str(market_id))
        if market is None:
            # Prefetch missed this fixture — fall back to single Gamma fetch.
            # Cached by bet_display_enrichment so a second call costs nothing.
            from src.integrations.polymarket.bet_display_enrichment import (
                get_gamma_market_cached,
            )
            market = get_gamma_market_cached(str(market_id))
        t1, t2 = teams_from_gamma_market(market)

    sl1, sl2 = _teams_from_side_labels(side_labels)
    if t1 is None and sl1:
        t1 = sl1
    if t2 is None and sl2:
        t2 = sl2
    elif t2 is None and sl1 and t1 and sl1 != t1:
        t2 = sl1

    svc = PolymarketComparisonService()

    # Fallback to fixture-key abbreviations. Important: when one slot is
    # already filled with a real team name, the fallback for the OTHER slot
    # must be the abbreviation that does NOT match the filled slot —
    # otherwise we end up with t1='Ivory Coast' and t2='IVO' (both the same
    # team) and the actual opponent is lost.
    if not t1 or not t2:
        fb1, fb2 = _fixture_abbr_fallback(fixture_key)

        def _abbr_matches(abbr: str, full: str) -> bool:
            """Match a 3-letter slug abbrev to a full team name.

            label_matches_team requires real token overlap (>= 4 chars) so
            it returns False for 3-letter slug abbrevs like 'ess' vs 'Essex'
            or 'cmr' vs 'Cameroon'. We need a cheap heuristic that catches
            those: case-insensitive prefix match on the normalised name,
            then fall back to substring containment.
            """
            if not abbr or not full:
                return False
            a = abbr.lower().strip()
            f = full.lower().strip()
            if not a or not f:
                return False
            # Strip non-alphanumeric from both
            import re
            a_clean = re.sub(r"[^a-z0-9]", "", a)
            f_clean = re.sub(r"[^a-z0-9]", "", f)
            if not a_clean or not f_clean:
                return False
            if f_clean.startswith(a_clean):
                return True
            # Token-prefix match: any whitespace-split token in `full` starts with `abbr`?
            for token in re.split(r"\s+", f):
                tok_clean = re.sub(r"[^a-z0-9]", "", token)
                if tok_clean and tok_clean.startswith(a_clean):
                    return True
            return False

        def _pick_other(filled: str, a: str, b: str) -> str:
            """Return whichever of (a, b) does NOT represent the already-filled team."""
            a_match = (svc.label_matches_team(a, filled) if a else False) or _abbr_matches(a, filled)
            b_match = (svc.label_matches_team(b, filled) if b else False) or _abbr_matches(b, filled)
            if a_match and not b_match:
                return b
            if b_match and not a_match:
                return a
            # Genuinely ambiguous: prefer the one that isn't a literal equal.
            if a and a != filled:
                return a
            return b

        if t1 and not t2:
            t2 = _pick_other(t1, fb1, fb2)
        elif t2 and not t1:
            t1 = _pick_other(t2, fb1, fb2)
        else:
            t1 = t1 or fb1
            t2 = t2 or fb2

    for sl in side_labels:
        if svc.label_matches_team(t2, sl) and not svc.label_matches_team(t1, sl):
            t1, t2 = t2, t1
            break

    display = f"{t1} v {t2}" if t1 and t2 else fixture_key
    meta = {"fixture_team1": t1, "fixture_team2": t2, "fixture_display": display}

    # Don't cache results where either team is still a slug abbreviation —
    # next request can retry Gamma and get the real names. Otherwise a single
    # transient Gamma timeout poisons the cache for the next 6 hours.
    import re
    looks_like_slug = lambda x: bool(x) and bool(re.fullmatch(r"[A-Z]{2,4}", str(x)))
    if not looks_like_slug(t1) and not looks_like_slug(t2):
        fixture_meta_cache_set(fixture_key, meta)
    return meta


def _enrich_bets_with_fixture_meta(
    bets: List[Dict[str, Any]],
    gamma_by_market: Dict[str, Any],
) -> None:
    """Attach human-readable fixture team names to bet dicts in-place."""
    if not bets:
        return

    by_fixture: Dict[str, List[Dict[str, Any]]] = {}
    for bet in bets:
        fk = bet.get("fixture_key") or "?"
        by_fixture.setdefault(fk, []).append(bet)

    meta_cache: Dict[str, Dict[str, str]] = {}
    for fk, fixture_bets in by_fixture.items():
        if fk not in meta_cache:
            meta_cache[fk] = _resolve_fixture_meta(fk, fixture_bets, gamma_by_market)
        for bet in fixture_bets:
            bet.update(meta_cache[fk])


def _match_winner_from_outcome(
    side_label: str,
    our_outcome: int,
    team1: str,
    team2: str,
) -> Optional[str]:
    """Return winning team name given our side's settle outcome."""
    from src.integrations.odds.polymarket_compare import PolymarketComparisonService
    svc = PolymarketComparisonService()
    our_is_t1 = svc.label_matches_team(team1, side_label)
    our_is_t2 = svc.label_matches_team(team2, side_label)
    if our_outcome == 1:
        return side_label
    if our_is_t1:
        return team2
    if our_is_t2:
        return team1
    return team2 if our_outcome == 0 else side_label


def _format_outcome_summary(bet: Dict[str, Any]) -> str:
    """Human-readable settlement narrative for live-betting meta row."""
    winner = bet.get("match_winner")
    winner_str = f"Winner: {winner}" if winner else "Winner: pending resolution"
    is_cashout = bool(bet.get("cashout_triggered_at"))
    delta = bet.get("counterfactual_delta")
    pnl = bet.get("pnl_realised_usdc")

    if is_cashout:
        if delta is not None:
            d = float(delta)
            if d > 0.01:
                return f"Left ${abs(d):.2f} on table vs holding to win · {winner_str}"
            if d < -0.01:
                return f"Saved ${abs(d):.2f} vs full loss · {winner_str}"
            return f"Cashout matched settlement · {winner_str}"
        return winner_str if winner else ""

    outcome = bet.get("settle_outcome")
    if outcome is not None:
        outcome = int(outcome)
    if outcome == 1:
        pnl_bit = f" (+${float(pnl):.2f})" if pnl is not None and float(pnl) > 0 else ""
        return f"Full win{pnl_bit} · {winner_str}"
    if outcome == 0:
        ate = f" — ate ${abs(float(pnl)):.2f}" if pnl is not None and float(pnl) < 0 else ""
        return f"Full loss{ate} · {winner_str}"
    return winner_str if winner else ""


def _enrich_bets_with_settlement_context(
    bets: List[Dict[str, Any]],
    gamma_by_market: Dict[str, Any],
) -> None:
    """Attach match winner and counterfactual hold-to-settlement PnL."""
    if not bets:
        return

    from src.integrations.polymarket.bet_display_enrichment import get_gamma_market_cached
    from src.integrations.polymarket.reconcile import (
        _compute_pnl_for_settled_bet,
        _resolve_via_gamma,
    )

    outcome_cache: Dict[str, Optional[int]] = {}

    def _gamma_market_for(mid: str) -> Optional[Dict[str, Any]]:
        if not mid:
            return None
        market = gamma_by_market.get(mid)
        if market is not None:
            return market
        return get_gamma_market_cached(mid)

    def _outcome_for_bet(bet: Dict[str, Any]) -> Optional[int]:
        if bet.get("cashout_triggered_at"):
            mid = str(bet.get("polymarket_market_id") or "")
            side = bet.get("side_label") or ""
            cache_key = f"{mid}:{side}"
            if cache_key not in outcome_cache:
                market = _gamma_market_for(mid)
                resolved = _resolve_via_gamma(market, side) if market else None
                outcome_cache[cache_key] = resolved[0] if resolved else None
            return outcome_cache[cache_key]
        so = bet.get("settle_outcome")
        if so is None:
            return None
        return int(so)

    for bet in bets:
        if bet.get("status") != "settled":
            continue

        match_outcome = _outcome_for_bet(bet)
        bet["match_settle_outcome"] = match_outcome

        team1 = bet.get("fixture_team1") or "?"
        team2 = bet.get("fixture_team2") or "?"
        side = bet.get("side_label") or ""

        if match_outcome is not None and side:
            bet["match_winner"] = _match_winner_from_outcome(
                side, match_outcome, team1, team2,
            )
        else:
            bet["match_winner"] = None

        fill_p = bet.get("fill_price")
        fill_sz = bet.get("fill_size_usdc") or bet.get("size_usdc")
        if match_outcome is not None and fill_p and fill_sz:
            hold_pnl = _compute_pnl_for_settled_bet(
                float(fill_p), float(fill_sz), float(match_outcome),
            )
            bet["hold_to_settlement_pnl"] = hold_pnl
            actual = bet.get("pnl_realised_usdc")
            if bet.get("cashout_triggered_at") and hold_pnl is not None and actual is not None:
                bet["counterfactual_delta"] = round(float(hold_pnl) - float(actual), 4)

        bet["outcome_summary"] = _format_outcome_summary(bet)


def _enrich_bets_with_sizing_context(bets: List[Dict[str, Any]]) -> None:
    """Attach Kelly / cap sizing notes for display."""
    if not bets:
        return

    from config import BETTING_CONFIG
    from src.integrations.polymarket.sizing import compute_sizing_context

    # Live wallet read once per strategy per request — not once per bet row.
    cap_by_strategy: Dict[str, Optional[float]] = {}
    static_cap = float(BETTING_CONFIG.get("max_per_bet_usdc") or 0) or None

    def _risk_cap(strategy_label: Optional[str], settled: bool) -> Optional[float]:
        if settled:
            # Historical rows use bankroll_at_proposal; skip live wallet for display notes.
            return None
        if not strategy_label:
            return static_cap
        if strategy_label not in cap_by_strategy:
            from src.integrations.polymarket.live_bankroll import get_max_per_bet_usdc
            with get_db_connection() as conn:
                cap_by_strategy[strategy_label] = get_max_per_bet_usdc(strategy_label, conn)
        return cap_by_strategy[strategy_label]

    for bet in bets:
        strategy_label = bet.get("strategy_label")
        settled = bet.get("status") == "settled"
        risk_cap = _risk_cap(strategy_label, settled)
        ctx = compute_sizing_context(bet, risk_gate_max_per_bet=risk_cap)
        if ctx:
            bet.update(ctx)


def _gamma_market_ids_for_enrichment(bets: List[Dict[str, Any]]) -> List[str]:
    """Market IDs that need Gamma prefetch.

    Three things need Gamma data:
      1. Cashout counterfactual P&L (needs settle_outcome) — covered when
         settle_outcome is missing.
      2. Open/proposed bet display — covered (no settle_outcome yet).
      3. Settled-bet display fixture_team1/team2 names — covered when the
         fixture_meta cache doesn't already have an entry. Otherwise the
         slug-abbreviation fallback ("NPL", "SVN", "GBR") leaks into the UI
         because the per-request get_gamma_market_cached fallback can race
         against the 4s timeout under load.
    """
    from src.integrations.polymarket.bet_display_enrichment import (
        fixture_meta_cache_get,
    )
    ids: List[str] = []
    seen = set()
    for bet in bets:
        mid = str(bet.get("polymarket_market_id") or "").strip()
        if not mid or mid in seen:
            continue
        fixture_key = bet.get("fixture_key") or ""
        has_cached_meta = fixture_meta_cache_get(fixture_key) is not None

        needs_gamma = True
        if bet.get("status") == "settled":
            if bet.get("cashout_triggered_at"):
                needs_gamma = bet.get("settle_outcome") is None or not has_cached_meta
            else:
                # Naturally-settled bets still need Gamma for fixture team names
                # when the meta cache hasn't been populated yet.
                needs_gamma = not has_cached_meta
        if needs_gamma:
            seen.add(mid)
            ids.append(mid)
    return ids


def _enrich_betting_bets(conn, bets: List[Dict[str, Any]]) -> None:
    """Run all bet enrichments for /api/betting/bets."""
    from src.integrations.polymarket.bet_display_enrichment import (
        get_gamma_market_cached,
        prefetch_gamma_markets,
    )

    _enrich_bets_with_sub_fills(conn, bets)

    market_ids = _gamma_market_ids_for_enrichment(bets)
    prefetch_gamma_markets(market_ids)

    gamma_by_market: Dict[str, Any] = {}
    for mid in market_ids:
        gamma_by_market[mid] = get_gamma_market_cached(mid)

    _enrich_bets_with_fixture_meta(bets, gamma_by_market)
    _enrich_bets_with_settlement_context(bets, gamma_by_market)
    _enrich_bets_with_sizing_context(bets)


@app.route('/api/betting/bets', methods=['GET'])
def betting_bets():
    """Wave 5.8.3: Live bets (card UI) — mirrors /api/paper/bets for real bets.

    Query: ?status=open|settled|all (default 'all'), ?limit=200, ?strategy=name

    Drives the card-style 'Open Live Bets' and 'Recent Settled' sections on
    /live-betting, analogous to /paper-trades but restricted to bet_kind='real'.
    """
    try:
        status = (request.args.get('status') or 'all').lower()
        strategy = request.args.get('strategy')
        try:
            limit = max(1, min(int(request.args.get('limit') or 200), 500))
        except ValueError:
            limit = 200

        where = ["COALESCE(bet_kind, 'real') = 'real'"]
        params: list = []
        if status == 'open':
            where.append("status NOT IN ('settled', 'cancelled', 'errored')")
            order_by = "proposed_at DESC"
        elif status == 'settled':
            where.append("status = 'settled'")
            order_by = "settled_at DESC"
        else:
            order_by = "proposed_at DESC"
        if strategy:
            where.append("strategy_label = ?")
            params.append(strategy)

        sql = f"""
            SELECT *
            FROM bet_ledger
            WHERE {' AND '.join(where)}
            ORDER BY {order_by}
            LIMIT ?
        """
        params.append(limit)

        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(sql, params)
            bets = [dict(r) for r in cur.fetchall()]
            _enrich_betting_bets(conn, bets)
        return jsonify({'success': True, 'bets': bets, 'n': len(bets)})
    except Exception as e:
        logger.error(f"Error in /api/betting/bets: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/betting/recent-settled', methods=['GET'])
def betting_recent_settled():
    """Last 50 settled REAL bets.

    Wave 5.8: filters bet_kind='real'; paper bets surfaced at /paper-trades.
    """
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT * FROM bet_ledger
                WHERE COALESCE(bet_kind, 'real') = 'real'
                  AND settled_at IS NOT NULL
                ORDER BY settled_at DESC
                LIMIT 50
                """
            )
            bets = [dict(r) for r in cur.fetchall()]
            _enrich_betting_bets(conn, bets)
        return jsonify({'success': True, 'bets': bets})
    except Exception as e:
        logger.error(f"Error fetching recent settled bets: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/betting/place', methods=['POST'])
def betting_place():
    """Manual bet placement endpoint.

    Body:
        {
            "fixture_key": str,
            "match_id": int (optional),
            "market_type": "moneyline" | "top_batter" | "most_sixes" | "toss_match_double",
            "polymarket_market_id": str,
            "polymarket_token_id": str,
            "side_label": str,
            "model_prob": float [0,1],
            "market_price_at_proposal": float [0,1],
            "side": "BUY" | "SELL",
            "size_usdc": float,
            "mode": "manual" | "auto" (default "manual")
        }
    """
    try:
        from src.integrations.polymarket.bet_placement import place_bet
        payload = request.get_json(silent=True) or {}
        result = place_bet(
            fixture_key=str(payload.get('fixture_key', '')),
            match_id=payload.get('match_id'),
            market_type=str(payload.get('market_type', '')),
            polymarket_market_id=str(payload.get('polymarket_market_id', '')),
            polymarket_token_id=str(payload.get('polymarket_token_id', '')),
            side_label=str(payload.get('side_label', '')),
            model_prob=float(payload.get('model_prob', 0)),
            market_price_at_proposal=float(payload.get('market_price_at_proposal', 0)),
            side=str(payload.get('side', 'BUY')),
            size_usdc=float(payload.get('size_usdc', 0)),
            requested_mode=str(payload.get('mode', 'manual')),
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error placing bet: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/betting/kill-switch', methods=['POST'])
def betting_kill_switch():
    """Engage / release the kill switch."""
    try:
        from src.integrations.polymarket.risk_gate import (
            engage_kill_switch, disengage_kill_switch,
        )
        payload = request.get_json(silent=True) or {}
        action = (payload.get('action') or 'engage').lower()
        if action == 'engage':
            result = engage_kill_switch()
        elif action == 'release':
            result = disengage_kill_switch()
        else:
            return jsonify({'success': False, 'error': f'Invalid action: {action}'}), 400
        return jsonify({'success': True, **result})
    except Exception as e:
        logger.error(f"Error toggling kill switch: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/betting/mode', methods=['POST'])
def betting_mode():
    """Toggle BETTING_MODE between OFF / MANUAL / AUTO."""
    try:
        from src.integrations.polymarket.risk_gate import set_mode
        payload = request.get_json(silent=True) or {}
        new_mode = (payload.get('mode') or 'OFF').upper()
        return jsonify(set_mode(new_mode))
    except Exception as e:
        logger.error(f"Error setting mode: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/betting/reconcile', methods=['POST'])
def betting_reconcile():
    """Trigger reconciliation: settle any matured bets via Polymarket prices-history."""
    try:
        from src.integrations.polymarket.reconcile import reconcile_pending_bets
        summary = reconcile_pending_bets()
        return jsonify({'success': True, **summary})
    except Exception as e:
        logger.error(f"Error reconciling bets: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/betting/upcoming', methods=['GET'])
def betting_upcoming():
    """Upcoming Polymarket cricket fixtures considered for LIVE bets.

    Identical Polymarket source as `/api/paper/upcoming` — both pages
    look at the same fixtures slate. Exposed under /api/betting/* so the
    live-betting page doesn't have to know about the paper namespace.
    Adds `pre_toss_real_bets` / `post_toss_real_bets` counts so the UI
    can show whether a fixture has already been bet on.
    """
    try:
        from src.integrations.polymarket import PolymarketClient
        from src.integrations.polymarket.upcoming import (
            find_upcoming_cricket_events, attach_db_team_ids,
        )
        try:
            hours_ahead = float(request.args.get('hours_ahead') or 96.0)
        except ValueError:
            hours_ahead = 96.0
        c = PolymarketClient()
        events = find_upcoming_cricket_events(c, hours_ahead=hours_ahead)
        mapped = attach_db_team_ids(events)

        # Pull live-bet counts per fixture/phase in one query — much
        # faster than N round-trips when there are many fixtures.
        fixture_keys = [ev["fixture_key"] for ev in mapped]
        bet_counts: dict = {}
        if fixture_keys:
            placeholders = ",".join(["?"] * len(fixture_keys))
            with get_db_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    f"""
                    SELECT fixture_key,
                           COALESCE(phase, 'pre_toss') AS phase,
                           COUNT(*) AS n
                    FROM bet_ledger
                    WHERE bet_kind = 'real'
                      AND status != 'errored'
                      AND fixture_key IN ({placeholders})
                    GROUP BY fixture_key, phase
                    """,
                    fixture_keys,
                )
                for r in cur.fetchall():
                    fk = r["fixture_key"]
                    if fk not in bet_counts:
                        bet_counts[fk] = {"pre_toss": 0, "post_toss": 0}
                    bet_counts[fk][r["phase"]] = int(r["n"])

        out = []
        for ev in mapped:
            ml = ev.get("moneyline") or {}
            counts = bet_counts.get(ev["fixture_key"], {"pre_toss": 0, "post_toss": 0})
            out.append({
                "fixture_key": ev["fixture_key"],
                "tournament_name": ev["tournament_name"],
                "tournament_prefix": ev.get("tournament_prefix"),
                "format": ev["format"],
                "gender": ev["gender"],
                "team1": ev.get("team1_db_name") or ev.get("team1_label"),
                "team2": ev.get("team2_db_name") or ev.get("team2_label"),
                "team1_id": ev.get("team1_id"),
                "team2_id": ev.get("team2_id"),
                "scheduled_start_utc": ev["scheduled_start_estimate"].isoformat(),
                "has_moneyline": bool(ev.get("moneyline")),
                "moneyline_outcomes": [
                    {"label": o.get("label"), "last_price": o.get("last_price")}
                    for o in (ml.get("outcomes") or [])
                ] if ml else [],
                "pre_toss_real_bets": counts["pre_toss"],
                "post_toss_real_bets": counts["post_toss"],
            })
        return jsonify({"success": True, "fixtures": out, "n": len(out)})
    except Exception as e:
        logger.error(f"Error in /api/betting/upcoming: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/betting/post-toss-scan', methods=['POST'])
def betting_post_toss_scan():
    """Trigger a LIVE post-toss re-scan for one fixture in the background.

    Spawns scripts/live_bet_post_toss_scan.py as a detached subprocess.
    The script runs V2/V3 sims with PINNED toss, applies LIVE strategy
    filters (BETTING_LIVE_STRATEGIES), and routes through `place_bet()`
    if any qualifying edge is found. Mode/kill-switch are enforced
    inside the script — calling this endpoint with BETTING_MODE=OFF is
    safe (the script logs and exits).

    Body: {fixture_key: str, toss_winner: 'team1'|'team2',
           chose_to: 'bat'|'field',
           team1_xi: [int,...] (optional), team2_xi: [int,...] (optional)}
    """
    try:
        import subprocess
        from pathlib import Path
        payload = request.get_json(silent=True) or {}
        fixture_key = str(payload.get("fixture_key") or "")
        toss_winner = str(payload.get("toss_winner") or "")
        chose_to = str(payload.get("chose_to") or "")
        if (
            not fixture_key
            or toss_winner not in ("team1", "team2")
            or chose_to not in ("bat", "field")
        ):
            return jsonify({
                "success": False,
                "error": "fixture_key + toss_winner (team1|team2) + chose_to (bat|field) required",
            }), 400

        repo_root = Path(__file__).resolve().parent.parent
        log_dir = repo_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "live_post_toss.log"
        venv_python = repo_root / "venv311" / "bin" / "python"
        script_path = repo_root / "scripts" / "live_bet_post_toss_scan.py"

        cmd = [
            str(venv_python), str(script_path),
            "--fixture-key", fixture_key,
            "--toss-winner", toss_winner,
            "--chose-to", chose_to,
        ]
        t1_xi = payload.get("team1_xi") or []
        t2_xi = payload.get("team2_xi") or []
        if t1_xi:
            cmd.extend(["--team1-xi", ",".join(str(int(x)) for x in t1_xi)])
        if t2_xi:
            cmd.extend(["--team2-xi", ",".join(str(int(x)) for x in t2_xi)])

        with log_path.open("a") as logf:
            logf.write(f"\n\n=== live post-toss spawn at {datetime.now(timezone.utc).isoformat()} ===\n")
            logf.write(f"=== cmd: {' '.join(cmd)} ===\n")
            logf.flush()
            proc = subprocess.Popen(
                cmd, cwd=str(repo_root), stdout=logf, stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        return jsonify({
            "success": True, "started": True, "pid": proc.pid,
            "log_path": str(log_path),
            "message": (
                "Live post-toss scan started in background. Poll "
                "/api/betting/post-toss-scan/status to track progress; "
                "any new bets appear in the live ledger when finished (~60s)."
            ),
        })
    except Exception as e:
        logger.error(f"Error in /api/betting/post-toss-scan: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/betting/post-toss-scan/status', methods=['GET'])
def betting_post_toss_scan_status():
    """Tail of the latest LIVE post-toss scan log."""
    try:
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent.parent
        log_path = repo_root / "logs" / "live_post_toss.log"
        if not log_path.exists():
            return jsonify({"success": True, "running": False, "tail": ""})
        with log_path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 4096))
            tail_bytes = f.read()
        tail = tail_bytes.decode("utf-8", errors="replace")
        # Match the "summary" sentinel printed by live_bet_post_toss_scan.main()
        running = "LIVE POST-TOSS SCAN SUMMARY" not in tail.split("=== live post-toss spawn")[-1]
        return jsonify({"success": True, "running": running, "tail": tail[-2000:]})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/betting/dashboard', methods=['GET'])
def betting_dashboard():
    """Wave 5 Phase 7: live ops dashboard data.

    Returns:
        - cumulative P&L timeline (one point per settled bet)
        - 10-bin calibration on settled bets (model_prob bucket vs actual win rate)
        - per-market-type realised ROI
        - rolling Brier on bets-placed (last 30, 60, all)
        - scale-up gate eligibility flag
    """
    try:
        import math
        with get_db_connection() as conn:
            cur = conn.cursor()
            # Wave 5.8: restrict to real bets; paper bets live at /paper-trades.
            cur.execute(
                """
                SELECT settled_at, market_type, model_prob, settle_outcome,
                       fill_size_usdc, pnl_realised_usdc
                FROM bet_ledger
                WHERE COALESCE(bet_kind, 'real') = 'real'
                  AND settled_at IS NOT NULL
                ORDER BY settled_at ASC
                """
            )
            settled = [dict(r) for r in cur.fetchall()]

        # Cumulative P&L timeline
        cum = 0.0
        timeline = []
        for b in settled:
            cum += float(b.get('pnl_realised_usdc') or 0)
            timeline.append({
                'settled_at': b['settled_at'],
                'cumulative_pnl_usdc': round(cum, 4),
            })

        # 10-bin calibration
        buckets = [(i / 10.0, (i + 1) / 10.0) for i in range(10)]
        calibration = []
        for lo, hi in buckets:
            in_bucket = [
                b for b in settled
                if b.get('model_prob') is not None and b.get('settle_outcome') is not None
                and lo <= float(b['model_prob']) < (hi if hi < 1.0 else hi + 1e-9)
            ]
            if not in_bucket:
                calibration.append({'lo': lo, 'hi': hi, 'n': 0, 'mean_pred': None, 'actual_win_rate': None})
                continue
            mean_pred = sum(float(b['model_prob']) for b in in_bucket) / len(in_bucket)
            wr = sum(int(b['settle_outcome']) for b in in_bucket) / len(in_bucket)
            calibration.append({
                'lo': lo, 'hi': hi, 'n': len(in_bucket),
                'mean_pred': round(mean_pred, 4), 'actual_win_rate': round(wr, 4),
            })

        # Per-market realised ROI
        per_market: dict = {}
        for b in settled:
            mt = b.get('market_type') or 'unknown'
            if mt not in per_market:
                per_market[mt] = {'n': 0, 'staked': 0.0, 'pnl': 0.0, 'wins': 0}
            per_market[mt]['n'] += 1
            per_market[mt]['staked'] += float(b.get('fill_size_usdc') or 0)
            per_market[mt]['pnl'] += float(b.get('pnl_realised_usdc') or 0)
            if (b.get('pnl_realised_usdc') or 0) > 0:
                per_market[mt]['wins'] += 1
        per_market_summary = {}
        for mt, d in per_market.items():
            per_market_summary[mt] = {
                'n': d['n'],
                'win_rate': round(d['wins'] / d['n'], 3) if d['n'] else None,
                'roi_pct': round(d['pnl'] / d['staked'] * 100, 2) if d['staked'] > 0 else 0.0,
                'total_pnl_usdc': round(d['pnl'], 2),
            }

        # Rolling Brier
        def _safe_log(p, eps=1e-6):
            import math as _m
            return _m.log(min(max(p, eps), 1.0 - eps))

        def _brier_window(items):
            terms = []
            for b in items:
                if b.get('model_prob') is None or b.get('settle_outcome') is None:
                    continue
                terms.append((float(b['model_prob']) - float(b['settle_outcome'])) ** 2)
            return round(sum(terms) / len(terms), 4) if terms else None

        rolling_brier = {
            'last_30': _brier_window(settled[-30:]),
            'last_60': _brier_window(settled[-60:]),
            'all': _brier_window(settled),
        }

        # Scale-up gate eligibility
        from src.integrations.polymarket.risk_gate import _is_scale_up_eligible
        scale_up_eligible = _is_scale_up_eligible(len(settled))

        return jsonify({
            'success': True,
            'n_settled': len(settled),
            'cumulative_pnl_timeline': timeline,
            'calibration_deciles': calibration,
            'per_market_summary': per_market_summary,
            'rolling_brier': rolling_brier,
            'scale_up_eligible': scale_up_eligible,
        })
    except Exception as e:
        logger.error(f"Error generating betting dashboard: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/paper-trades')
def paper_trades_page():
    """Wave 5.7: Paper trades dashboard."""
    return render_template('paper_trades.html')


@app.route('/api/paper/strategies', methods=['GET'])
def paper_strategies():
    """Per-strategy summary: bankroll, P&L, win rate, bet counts."""
    try:
        from src.integrations.polymarket.paper_strategies import STRATEGIES
        starting_by_name = {s.name: s.starting_bankroll_usdc for s in STRATEGIES}
        descriptions = {s.name: s.description for s in STRATEGIES}
        enabled = {s.name: s.enabled for s in STRATEGIES}

        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT strategy_label,
                       COUNT(*) AS n_bets,
                       SUM(CASE WHEN status = 'settled' THEN 1 ELSE 0 END) AS n_settled,
                       SUM(CASE WHEN status = 'settled' AND settle_outcome = 1 THEN 1 ELSE 0 END) AS n_won,
                       SUM(CASE WHEN status = 'settled' THEN pnl_realised_usdc ELSE 0 END) AS realised_pnl,
                       SUM(CASE WHEN status NOT IN ('settled', 'cancelled', 'errored') THEN size_usdc ELSE 0 END) AS open_size,
                       MIN(proposed_at) AS first_bet_at,
                       MAX(proposed_at) AS last_bet_at
                FROM bet_ledger
                WHERE bet_kind = 'paper'
                GROUP BY strategy_label
                """
            )
            rows = {r["strategy_label"]: dict(r) for r in cur.fetchall()}

        out = []
        for s in STRATEGIES:
            r = rows.get(s.name, {})
            n_bets = int(r.get("n_bets") or 0)
            n_settled = int(r.get("n_settled") or 0)
            n_won = int(r.get("n_won") or 0)
            realised = float(r.get("realised_pnl") or 0.0)
            open_size = float(r.get("open_size") or 0.0)
            bankroll = s.starting_bankroll_usdc + realised
            win_rate = (n_won / n_settled) if n_settled else None
            roi_pct = (realised / s.starting_bankroll_usdc * 100) if s.starting_bankroll_usdc else None
            out.append({
                "name":          s.name,
                "description":   s.description,
                "enabled":       s.enabled,
                "model_version": s.model_version,
                "min_edge_pp":   s.min_edge_pp,
                "starting":      s.starting_bankroll_usdc,
                "bankroll":      round(bankroll, 2),
                "realised_pnl":  round(realised, 2),
                "open_size":     round(open_size, 2),
                "roi_pct":       round(roi_pct, 2) if roi_pct is not None else None,
                "n_bets":        n_bets,
                "n_settled":     n_settled,
                "n_won":         n_won,
                "win_rate":      win_rate,
                "first_bet_at":  r.get("first_bet_at"),
                "last_bet_at":   r.get("last_bet_at"),
            })
        # Totals row
        total_starting = sum(x["starting"] for x in out)
        total_realised = sum(x["realised_pnl"] for x in out)
        total_bankroll = sum(x["bankroll"] for x in out)
        total_open = sum(x["open_size"] for x in out)
        return jsonify({
            "success": True,
            "strategies": out,
            "totals": {
                "starting": total_starting,
                "realised_pnl": round(total_realised, 2),
                "bankroll": round(total_bankroll, 2),
                "open_size": round(total_open, 2),
                "roi_pct": round((total_realised / total_starting * 100) if total_starting else 0, 2),
            },
        })
    except Exception as e:
        logger.error(f"Error in /api/paper/strategies: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/paper/bets', methods=['GET'])
def paper_bets():
    """Open + recent paper bets, ordered by proposed_at desc.

    Query: ?status=open|settled|all (default 'all'), ?limit=50, ?strategy=name
    """
    try:
        status = (request.args.get('status') or 'all').lower()
        strategy = request.args.get('strategy')
        try:
            limit = max(1, min(int(request.args.get('limit') or 100), 500))
        except ValueError:
            limit = 100

        where = ["bet_kind = 'paper'"]
        params = []
        if status == 'open':
            where.append("status NOT IN ('settled', 'cancelled', 'errored')")
        elif status == 'settled':
            where.append("status = 'settled'")
        if strategy:
            where.append("strategy_label = ?")
            params.append(strategy)

        sql = f"""
            SELECT *
            FROM bet_ledger
            WHERE {' AND '.join(where)}
            ORDER BY proposed_at DESC
            LIMIT ?
        """
        params.append(limit)

        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(sql, params)
            bets = [dict(r) for r in cur.fetchall()]
        return jsonify({"success": True, "bets": bets, "n": len(bets)})
    except Exception as e:
        logger.error(f"Error in /api/paper/bets: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/paper/bankroll-history', methods=['GET'])
def paper_bankroll_history():
    """Bankroll-over-time per strategy.

    Computes the running bankroll CHRONOLOGICALLY: starting + cumsum(pnl)
    ordered by settled_at. The stored `bankroll_after_settle` column is
    NOT used because it was sometimes filled in using the frozen
    bankroll_at_proposal (which can be stale if bets settle in a
    different order than they were placed); cumulative-sum is the
    canonical bankroll over time.
    """
    try:
        from src.integrations.polymarket.paper_strategies import STRATEGIES
        starting = {s.name: s.starting_bankroll_usdc for s in STRATEGIES}

        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT strategy_label, settled_at, pnl_realised_usdc, side_label, fixture_key
                FROM bet_ledger
                WHERE bet_kind = 'paper' AND status = 'settled'
                ORDER BY strategy_label, settled_at ASC
                """
            )
            rows = [dict(r) for r in cur.fetchall()]

        by_strategy = {}
        for s in STRATEGIES:
            by_strategy[s.name] = [{
                "timestamp": None, "bankroll": s.starting_bankroll_usdc,
                "side_label": "starting", "fixture_key": "",
                "pnl": 0.0,
            }]
        # Group rows by strategy
        from collections import defaultdict
        grouped = defaultdict(list)
        for r in rows:
            grouped[r["strategy_label"] or "(none)"].append(r)

        for name, lst in grouped.items():
            base = starting.get(name, 1000.0)
            running = base
            if name not in by_strategy:
                by_strategy[name] = [{
                    "timestamp": None, "bankroll": base,
                    "side_label": "starting", "fixture_key": "",
                    "pnl": 0.0,
                }]
            for r in lst:
                pnl = float(r.get("pnl_realised_usdc") or 0.0)
                running += pnl
                by_strategy[name].append({
                    "timestamp":   r["settled_at"],
                    "bankroll":    round(running, 2),
                    "side_label":  r.get("side_label"),
                    "fixture_key": r.get("fixture_key"),
                    "pnl":         round(pnl, 2),
                })
        return jsonify({"success": True, "history": by_strategy})
    except Exception as e:
        logger.error(f"Error in /api/paper/bankroll-history: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/paper/upcoming', methods=['GET'])
def paper_upcoming():
    """Upcoming Polymarket cricket fixtures we'll consider for paper bets."""
    try:
        from src.integrations.polymarket import PolymarketClient
        from src.integrations.polymarket.upcoming import (
            find_upcoming_cricket_events, attach_db_team_ids,
        )
        try:
            hours_ahead = float(request.args.get('hours_ahead') or 96.0)
        except ValueError:
            hours_ahead = 96.0
        c = PolymarketClient()
        events = find_upcoming_cricket_events(c, hours_ahead=hours_ahead)
        mapped = attach_db_team_ids(events)
        out = []
        for ev in mapped:
            ml = ev.get("moneyline") or {}
            out.append({
                "fixture_key": ev["fixture_key"],
                "tournament_name": ev["tournament_name"],
                "format": ev["format"],
                "gender": ev["gender"],
                "team1": ev.get("team1_db_name") or ev.get("team1_label"),
                "team2": ev.get("team2_db_name") or ev.get("team2_label"),
                "team1_id": ev.get("team1_id"),
                "team2_id": ev.get("team2_id"),
                "scheduled_start_utc": ev["scheduled_start_estimate"].isoformat(),
                "has_moneyline": bool(ev.get("moneyline")),
                "moneyline_outcomes": [
                    {"label": o.get("label"), "last_price": o.get("last_price")}
                    for o in (ml.get("outcomes") or [])
                ] if ml else [],
            })
        return jsonify({"success": True, "fixtures": out, "n": len(out)})
    except Exception as e:
        logger.error(f"Error in /api/paper/upcoming: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/paper/reconcile-only', methods=['POST'])
def paper_reconcile_only():
    """Lightweight: just reconcile settled markets, no expensive sim. Safe
    to call on every dashboard auto-refresh - hits Polymarket Gamma+CLOB
    only for unsettled bets and returns in 1-3 seconds."""
    try:
        from src.integrations.polymarket.reconcile import reconcile_pending_bets
        summary = reconcile_pending_bets()
        summary["errors"] = [
            {"bet_id": b, "msg": str(m)} for b, m in summary.get("errors", [])
        ]
        return jsonify({"success": True, "reconcile": summary})
    except Exception as e:
        logger.error(f"Error in /api/paper/reconcile-only: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/paper/scan-now', methods=['POST'])
def paper_scan_now():
    """Trigger a manual paper-bet scan + reconcile.

    Spawned as a SUBPROCESS so Flask's request thread isn't blocked by
    TensorFlow init / 3-minute sim run. Returns immediately with a
    'started' status; client should re-poll /api/paper/strategies (or
    just call refreshAll on a timer) to see results when the subprocess
    finishes.

    Latest subprocess log goes to logs/paper_scan_now.log. Latest scan
    summary is appended to data/paper_trading/daily_reports/*.json.
    """
    try:
        import subprocess
        from pathlib import Path
        payload = request.get_json(silent=True) or {}
        hours_ahead = float(payload.get("hours_ahead") or 96.0)

        repo_root = Path(__file__).resolve().parent.parent
        log_dir = repo_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "paper_scan_now.log"
        venv_python = repo_root / "venv311" / "bin" / "python"
        script_path = repo_root / "scripts" / "paper_bet_daily.py"

        cmd = [
            str(venv_python), str(script_path),
            "--hours-ahead", str(hours_ahead),
        ]
        # Detach: don't wait, redirect stdout/stderr to log file so the
        # parent doesn't keep a pipe open which would tether the child.
        with log_path.open("a") as logf:
            logf.write(f"\n\n=== scan-now spawn at {datetime.now(timezone.utc).isoformat()} ===\n")
            logf.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=str(repo_root),
                stdout=logf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        return jsonify({
            "success": True,
            "started": True,
            "pid": proc.pid,
            "log_path": str(log_path),
            "message": (
                "Scan started in background. The dashboard auto-refreshes "
                "every 60 seconds; new bets will appear when the scan finishes "
                "(typically 2-4 minutes)."
            ),
        })
    except Exception as e:
        logger.error(f"Error in /api/paper/scan-now: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/paper/resolve-xi', methods=['POST'])
def paper_resolve_xi():
    """Resolve a list of player names to DB player_ids, scoped to a team's
    recent appearances. Used by the post-toss UI form so operators can paste
    raw names ('V Kohli\\nF du Plessis\\n...') and get back resolved IDs.

    Body: {team_id: int, format: 'T20'|'ODI', gender: 'male'|'female',
           names: [str, str, ...]}
    """
    try:
        from src.utils.name_matcher import match_abbreviated_name
        payload = request.get_json(silent=True) or {}
        team_id = int(payload.get("team_id") or 0)
        fmt = str(payload.get("format") or "T20")
        gender = str(payload.get("gender") or "male")
        names = payload.get("names") or []
        if not team_id or not names:
            return jsonify({"success": False, "error": "team_id and names required"}), 400

        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT DISTINCT pms.player_id, p.name
                FROM player_match_stats pms
                JOIN players p ON p.player_id = pms.player_id
                JOIN matches m ON m.match_id = pms.match_id
                WHERE pms.team_id = ?
                  AND m.match_type = ? AND m.gender = ?
                  AND m.date >= '2023-01-01'
                """,
                (team_id, fmt, gender),
            )
            candidates = [(r["player_id"], r["name"]) for r in cur.fetchall()]

        results = []
        for raw in names:
            raw = (raw or "").strip()
            if not raw:
                results.append({"input": raw, "matched": False, "player_id": None, "name": None, "score": 0})
                continue
            match = match_abbreviated_name(raw, candidates, threshold=0.55)
            if match:
                pid, full_name, score = match
                results.append({
                    "input": raw, "matched": True, "player_id": pid,
                    "name": full_name, "score": round(score, 3),
                })
            else:
                results.append({
                    "input": raw, "matched": False, "player_id": None,
                    "name": None, "score": 0,
                })
        return jsonify({"success": True, "resolved": results})
    except Exception as e:
        logger.error(f"Error in /api/paper/resolve-xi: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/paper/auto-poller/status', methods=['GET'])
def paper_auto_poller_status():
    """Status of the auto-toss-detection daemon."""
    try:
        from pathlib import Path
        import json
        repo_root = Path(__file__).resolve().parent.parent
        status_file = repo_root / "logs" / "paper_auto_post_toss_status.json"
        pid_file = repo_root / "logs" / "paper_auto_post_toss.pid"

        # Check if PID is alive
        running = False
        live_pid = None
        if pid_file.exists():
            try:
                live_pid = int(pid_file.read_text().strip())
                import os as _os
                _os.kill(live_pid, 0)
                running = True
            except (ValueError, OSError):
                running = False
                live_pid = None

        status: dict = {"running": running, "pid": live_pid, "status_data": None}
        if status_file.exists():
            try:
                with status_file.open() as f:
                    status["status_data"] = json.load(f)
            except Exception:
                status["status_data"] = {"error": "could not parse status file"}
        return jsonify({"success": True, **status})
    except Exception as e:
        logger.error(f"Error in /api/paper/auto-poller/status: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/paper/auto-poller/start', methods=['POST'])
def paper_auto_poller_start():
    """Start the auto-toss-detection daemon as a detached subprocess.
    No-op if already running."""
    try:
        import subprocess
        from pathlib import Path
        import os as _os
        repo_root = Path(__file__).resolve().parent.parent
        pid_file = repo_root / "logs" / "paper_auto_post_toss.pid"
        log_dir = repo_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        venv_python = repo_root / "venv311" / "bin" / "python"
        script_path = repo_root / "scripts" / "paper_bet_auto_post_toss.py"

        if pid_file.exists():
            try:
                old_pid = int(pid_file.read_text().strip())
                _os.kill(old_pid, 0)
                return jsonify({"success": True, "already_running": True, "pid": old_pid})
            except (ValueError, OSError):
                pid_file.unlink(missing_ok=True)

        payload = request.get_json(silent=True) or {}
        poll_interval = int(payload.get("poll_interval") or 90)
        lookback_min = int(payload.get("lookback_min") or 45)
        lookahead_min = int(payload.get("lookahead_min") or 30)
        dry_run = bool(payload.get("dry_run") or False)
        # When also_live=true the daemon spawns BOTH the paper post-toss
        # scan AND the live post-toss scan for each detected toss. The
        # live scan respects BETTING_MODE / kill switch / live-strategy
        # whitelist independently, so this is safe to enable even on a
        # fully-OFF live system.
        also_live = bool(payload.get("also_live") or False)

        cmd = [
            str(venv_python), str(script_path),
            "--poll-interval", str(poll_interval),
            "--lookback-min", str(lookback_min),
            "--lookahead-min", str(lookahead_min),
        ]
        if dry_run:
            cmd.append("--dry-run")
        if also_live:
            cmd.append("--also-live")

        # Detach: redirect stdout/stderr so parent doesn't keep pipes open
        bootstrap_log = log_dir / "paper_auto_post_toss_bootstrap.log"
        with bootstrap_log.open("a") as logf:
            logf.write(f"\n=== boot at {datetime.now(timezone.utc).isoformat()} via Flask ===\n")
            logf.flush()
            proc = subprocess.Popen(
                cmd, cwd=str(repo_root), stdout=logf, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        return jsonify({
            "success": True, "started": True, "pid": proc.pid,
            "config": {
                "poll_interval": poll_interval, "lookback_min": lookback_min,
                "lookahead_min": lookahead_min, "dry_run": dry_run,
                "also_live": also_live,
            },
        })
    except Exception as e:
        logger.error(f"Error in /api/paper/auto-poller/start: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/paper/auto-poller/stop', methods=['POST'])
def paper_auto_poller_stop():
    """Send SIGTERM to the running daemon. No-op if not running."""
    try:
        from pathlib import Path
        import os as _os
        import signal as _signal
        repo_root = Path(__file__).resolve().parent.parent
        pid_file = repo_root / "logs" / "paper_auto_post_toss.pid"
        if not pid_file.exists():
            return jsonify({"success": True, "stopped": False, "reason": "not running"})
        try:
            pid = int(pid_file.read_text().strip())
        except (ValueError, OSError):
            pid_file.unlink(missing_ok=True)
            return jsonify({"success": True, "stopped": False, "reason": "stale-pid-file"})
        try:
            _os.kill(pid, _signal.SIGTERM)
            return jsonify({"success": True, "stopped": True, "pid": pid})
        except OSError as exc:
            return jsonify({"success": True, "stopped": False, "reason": f"kill failed: {exc}"})
    except Exception as e:
        logger.error(f"Error in /api/paper/auto-poller/stop: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/paper/fetch-crex-xi', methods=['POST'])
def paper_fetch_crex_xi():
    """Pull the current Playing XI from CREX for a given fixture.

    This is the Wave 5.7 wiring of the existing CREX scraper: it locates
    the matching CREX schedule entry by team-pair + date, then calls
    fetch_squads_with_playwright() to click the playingxi-button tabs and
    extract the XI per team. Names returned are CREX-format raw strings;
    the caller should follow up with /api/paper/resolve-xi to convert
    them to DB player_ids.

    Body: {fixture_key: str}

    Note: takes 5-15 seconds (Playwright + 2x tab clicks). Caller should
    show a loading spinner.
    """
    try:
        from src.integrations.polymarket import PolymarketClient
        from src.integrations.polymarket.upcoming import (
            find_upcoming_cricket_events, attach_db_team_ids,
        )
        from src.integrations.polymarket.crex_xi import fetch_xi_from_crex

        payload = request.get_json(silent=True) or {}
        fixture_key = str(payload.get("fixture_key") or "")
        if not fixture_key:
            return jsonify({"success": False, "error": "fixture_key required"}), 400

        c = PolymarketClient()
        events = find_upcoming_cricket_events(c, hours_ahead=168, include_started=True)
        mapped = attach_db_team_ids(events)
        fix = next((e for e in mapped if e["fixture_key"] == fixture_key), None)
        if fix is None:
            return jsonify({"success": False, "error": f"fixture {fixture_key} not found on Polymarket"}), 404

        crex_data = fetch_xi_from_crex(fix)
        if crex_data is None:
            return jsonify({
                "success": False,
                "error": (
                    "Could not locate a matching CREX entry for this fixture. "
                    "Either the match isn't in CREX's schedule yet (try again closer "
                    "to kickoff) or the team names didn't match. Use the manual "
                    "XI textareas instead."
                ),
            }), 404

        return jsonify({"success": True, "crex": crex_data})
    except Exception as e:
        logger.error(f"Error in /api/paper/fetch-crex-xi: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/paper/post-toss-scan', methods=['POST'])
def paper_post_toss_scan():
    """Trigger a post-toss re-scan for one fixture in the background.

    Body: {fixture_key: str, toss_winner: 'team1'|'team2',
           chose_to: 'bat'|'field',
           team1_xi: [int,...] (optional), team2_xi: [int,...] (optional)}
    """
    try:
        import subprocess
        from pathlib import Path
        payload = request.get_json(silent=True) or {}
        fixture_key = str(payload.get("fixture_key") or "")
        toss_winner = str(payload.get("toss_winner") or "")
        chose_to = str(payload.get("chose_to") or "")
        if not fixture_key or toss_winner not in ("team1", "team2") or chose_to not in ("bat", "field"):
            return jsonify({"success": False, "error": "fixture_key + toss_winner (team1|team2) + chose_to (bat|field) required"}), 400

        repo_root = Path(__file__).resolve().parent.parent
        log_dir = repo_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "paper_post_toss.log"
        venv_python = repo_root / "venv311" / "bin" / "python"
        script_path = repo_root / "scripts" / "paper_bet_post_toss_scan.py"

        cmd = [
            str(venv_python), str(script_path),
            "--fixture-key", fixture_key,
            "--toss-winner", toss_winner,
            "--chose-to", chose_to,
        ]
        t1_xi = payload.get("team1_xi") or []
        t2_xi = payload.get("team2_xi") or []
        if t1_xi:
            cmd.extend(["--team1-xi", ",".join(str(int(x)) for x in t1_xi)])
        if t2_xi:
            cmd.extend(["--team2-xi", ",".join(str(int(x)) for x in t2_xi)])

        with log_path.open("a") as logf:
            logf.write(f"\n\n=== post-toss spawn at {datetime.now(timezone.utc).isoformat()} ===\n")
            logf.write(f"=== cmd: {' '.join(cmd)} ===\n")
            logf.flush()
            proc = subprocess.Popen(
                cmd, cwd=str(repo_root), stdout=logf, stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        return jsonify({
            "success": True, "started": True, "pid": proc.pid,
            "log_path": str(log_path),
            "message": "Post-toss scan started in background. Polls /api/paper/post-toss-scan/status to track progress; new bets appear when finished (~60s).",
        })
    except Exception as e:
        logger.error(f"Error in /api/paper/post-toss-scan: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/paper/post-toss-scan/status', methods=['GET'])
def paper_post_toss_scan_status():
    """Tail of the latest post-toss scan log."""
    try:
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent.parent
        log_path = repo_root / "logs" / "paper_post_toss.log"
        if not log_path.exists():
            return jsonify({"success": True, "running": False, "tail": ""})
        with log_path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 4096))
            tail_bytes = f.read()
        tail = tail_bytes.decode("utf-8", errors="replace")
        running = "POST-TOSS SCAN SUMMARY" not in tail.split("=== post-toss spawn")[-1]
        return jsonify({"success": True, "running": running, "tail": tail[-2000:]})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/paper/scan-now/status', methods=['GET'])
def paper_scan_now_status():
    """Tail of the latest scan-now subprocess log so the UI can show
    progress without blocking Flask."""
    try:
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent.parent
        log_path = repo_root / "logs" / "paper_scan_now.log"
        if not log_path.exists():
            return jsonify({"success": True, "running": False, "tail": ""})
        # Read last 4 KB
        with log_path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 4096))
            tail_bytes = f.read()
        tail = tail_bytes.decode("utf-8", errors="replace")
        running = "PAPER TRADING - PER-STRATEGY SUMMARY" not in tail.split("=== scan-now spawn")[-1]
        return jsonify({"success": True, "running": running, "tail": tail[-2000:]})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/betting/scale-up', methods=['POST'])
def betting_scale_up():
    """Wave 5 Phase 7: graduate the envelope ($200 -> $500 -> $1000).

    Body:
        {"target": "500" | "1000"}
    Reads the dashboard scale_up_eligible flag; only allows graduation
    if eligible. Updates BETTING_MAX_DEPOSIT, BETTING_MAX_PER_BET,
    BETTING_MAX_PER_DAY in .env.
    """
    try:
        from src.integrations.polymarket.risk_gate import write_env_var, _is_scale_up_eligible
        from config import BETTING_CONFIG
        payload = request.get_json(silent=True) or {}
        target = str(payload.get('target', '500'))
        envelopes = {
            '500': {'deposit': 500, 'per_bet': 50, 'per_day': 100, 'max_loss_per_day': 75},
            '1000': {'deposit': 1000, 'per_bet': 100, 'per_day': 250, 'max_loss_per_day': 200},
        }
        if target not in envelopes:
            return jsonify({'success': False, 'error': f"Invalid target {target}; use '500' or '1000'."}), 400

        # Recompute eligibility live to avoid stale UI.
        # Wave 5.8: scale-up is based on REAL settled bets only.
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT COUNT(*) AS n FROM bet_ledger
                WHERE COALESCE(bet_kind, 'real') = 'real'
                  AND settled_at IS NOT NULL
                """
            )
            settled_count = int((cur.fetchone() or {'n': 0})['n'])
        if not _is_scale_up_eligible(settled_count):
            return jsonify({
                'success': False,
                'error': f"Not yet eligible: only {settled_count} settled bets.",
            }), 403

        env = envelopes[target]
        write_env_var('BETTING_MAX_DEPOSIT', str(env['deposit']))
        write_env_var('BETTING_MAX_PER_BET', str(env['per_bet']))
        write_env_var('BETTING_MAX_PER_DAY', str(env['per_day']))
        write_env_var('BETTING_MAX_LOSS_PER_DAY', str(env['max_loss_per_day']))
        BETTING_CONFIG['max_deposit_usdc'] = float(env['deposit'])
        BETTING_CONFIG['max_per_bet_usdc'] = float(env['per_bet'])
        BETTING_CONFIG['max_per_day_usdc'] = float(env['per_day'])
        BETTING_CONFIG['max_loss_per_day_usdc'] = float(env['max_loss_per_day'])
        return jsonify({'success': True, 'envelope': target, 'caps': env})
    except Exception as e:
        logger.error(f"Error in scale-up: {e}")
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

        # Optional: include Cricsheet source freshness (cached)
        source_status = None
        try:
            global _source_status_cache, _source_status_cache_expires
            now = datetime.now()
            if (
                _source_status_cache is not None
                and _source_status_cache_expires is not None
                and now < _source_status_cache_expires
            ):
                source_status = _source_status_cache
            else:
                from src.api.cricsheet_source_status import get_source_status
                source_status = get_source_status(CRICSHEET_MATCHES_URL)
                if source_status is not None:
                    _source_status_cache = source_status
                    _source_status_cache_expires = now + timedelta(seconds=SOURCE_STATUS_CACHE_TTL_SECONDS)
        except Exception as e:
            logger.debug("Source status fetch skipped or failed: %s", e)

        payload = {
            'success': True,
            'match_stats': match_stats,
            'total_players': total_players,
            'total_teams': total_teams,
            'total_deliveries': total_deliveries
        }
        if source_status is not None:
            payload['source_status'] = source_status
        return jsonify(payload)
        
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
        format_type = request.args.get('format') or None  # None = all formats
        active_only = request.args.get('active_only', 'false').lower() == 'true'

        models = get_model_versions(
            gender=gender,
            format_type=format_type,
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
        formats = data.get('formats', ['T20', 'ODI'])

        def retrain_models():
            """Retrain models based on mode."""
            from src.utils.job_manager import update_job_progress
            import threading

            job_id = getattr(threading.current_thread(), 'job_id', None)
            logger.info(
                f"Starting model retrain: mode={mode}, genders={genders}, "
                f"formats={formats}, job_id={job_id}"
            )

            skip_ingest = (mode == 'quick')
            skip_elo = (mode == 'quick')
            male_only = ('male' in genders and 'female' not in genders)
            female_only = ('female' in genders and 'male' not in genders)

            result = run_full_pipeline(
                skip_ingest=skip_ingest,
                skip_elo=skip_elo,
                male_only=male_only,
                female_only=female_only,
                formats=formats,
                progress_callback=lambda pct, step: update_job_progress(job_id, pct, step) if job_id else None,
            )

            return result

        job_id = start_job(
            name=f"Model Retrain ({mode}, {', '.join(genders)}, {', '.join(formats)})",
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


@app.route('/api/training/tune', methods=['POST'])
def start_hyperband_tune():
    """
    Start a Hyperband hyperparameter search for a given format/gender combination.

    Request body:
        {
            "format":    "T20" or "ODI"    (default: "T20")
            "gender":    "male" or "female" (default: "male")
            "overwrite": true | false       (default: false)
        }

    Returns:
        JSON with job_id for progress tracking
    """
    try:
        from src.utils.job_manager import start_job
        import sys
        from pathlib import Path

        scripts_dir = Path(__file__).resolve().parent.parent / 'scripts'
        sys.path.insert(0, str(scripts_dir))

        from tune_ball_predictor import run_tuner

        data = request.get_json() or {}
        fmt    = data.get('format',    'T20')
        gender = data.get('gender',    'male')
        overwrite = bool(data.get('overwrite', False))

        if fmt not in ('T20', 'ODI'):
            return jsonify({'success': False, 'error': f'Invalid format: {fmt}'}), 400
        if gender not in ('male', 'female'):
            return jsonify({'success': False, 'error': f'Invalid gender: {gender}'}), 400

        def run_tune_job():
            from src.utils.job_manager import update_job_progress
            import threading

            job_id = getattr(threading.current_thread(), 'job_id', None)

            result = run_tuner(
                format_type=fmt,
                gender=gender,
                overwrite=overwrite,
                progress_callback=lambda pct, msg: update_job_progress(job_id, pct, msg) if job_id else None,
            )
            return result

        job_id = start_job(
            name=f"Hyperband Tune ({fmt}/{gender})",
            func=run_tune_job,
        )

        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': f'Hyperband tuning started for {fmt}/{gender}',
        })

    except Exception as e:
        logger.error(f"Error starting hyperband tune: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/training/tune-results', methods=['GET'])
def get_tune_results():
    """
    Return saved best hyperparameters for all tuned format/gender combinations.

    Reads all best_hparams_*.json files from data/processed/.

    Returns:
        JSON dict keyed by "{format}_{gender}", e.g.
        { "t20_male": { "n_layers": 1, "units": 256, ... }, ... }
    """
    try:
        import json as _json
        from pathlib import Path

        processed_dir = Path(__file__).resolve().parent.parent / 'data' / 'processed'
        results = {}
        for hp_file in processed_dir.glob('best_hparams_*.json'):
            key = hp_file.stem.replace('best_hparams_', '')
            try:
                with open(hp_file) as f:
                    results[key] = _json.load(f)
            except Exception as e:
                logger.warning(f"Could not read {hp_file}: {e}")

        return jsonify({'success': True, 'results': results})

    except Exception as e:
        logger.error(f"Error fetching tune results: {e}")
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
# Startup Checks
# ============================================================================

def _check_playwright():
    """
    Verify the Playwright Chromium binary is installed and executable.

    The Python package (playwright) can be present in the venv while the
    browser binary is missing — this happens after a fresh pip install or
    when the cache directory is wiped.  When the binary is absent, CREX squad
    scraping silently falls back to static HTML only, which means some
    matches get incomplete squads with no visible error in the browser UI.

    Logs a prominent WARNING at startup so the problem is visible even when
    the terminal is not being watched.
    """
    auto_install = os.getenv('AUTO_INSTALL_PLAYWRIGHT_CHROMIUM', 'true').lower() == 'true'

    def _try_auto_install(reason: str) -> bool:
        """Attempt a one-shot Chromium install when runtime binary is missing."""
        if not auto_install:
            return False
        try:
            logger.warning(
                "[STARTUP] Attempting automatic Playwright Chromium install (%s). "
                "Set AUTO_INSTALL_PLAYWRIGHT_CHROMIUM=false to disable.",
                reason,
            )
            subprocess.run(
                [sys.executable, '-m', 'playwright', 'install', 'chromium'],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return True
        except Exception as install_error:
            logger.warning("[STARTUP] Automatic Chromium install failed: %s", install_error)
            return False

    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            # Just check the executable path — don't launch a full browser
            executable = p.chromium.executable_path
            if not Path(executable).exists():
                raise FileNotFoundError(executable)
        logger.info("[STARTUP] Playwright Chromium binary OK: %s", executable)
    except ImportError:
        installed = _try_auto_install('playwright package not importable')
        if installed:
            logger.info("[STARTUP] Playwright Chromium install completed after ImportError recovery attempt.")
        else:
            logger.warning(
                "[STARTUP] ⚠  Playwright is NOT installed in this environment. "
                "CREX squad scraping will fall back to static HTML only — squads may be "
                "incomplete. Fix: pip install playwright && playwright install chromium"
            )
    except Exception as e:
        installed = _try_auto_install(str(e))
        if installed:
            logger.info("[STARTUP] Playwright Chromium install completed after binary recovery attempt.")
            return
        logger.warning(
            "[STARTUP] ⚠  Playwright Chromium binary is MISSING or broken (%s). "
            "CREX squad scraping will fall back to static HTML only — squads may be "
            "incomplete. Fix: playwright install chromium",
            e,
        )


_check_playwright()


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', os.environ.get('FLASK_PORT', 5001)))
    app.run(host=FlaskConfig.HOST, port=port, debug=FlaskConfig.DEBUG)
