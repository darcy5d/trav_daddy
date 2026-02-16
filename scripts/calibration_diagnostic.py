#!/usr/bin/env python3
"""
Calibration Diagnostic Script.

Analyzes model prediction results from the CSV against:
- Actual win/loss outcomes
- Team ELO ratings from the database
- Player distribution data coverage
- Calibration curves by confidence bucket

Usage:
    python scripts/calibration_diagnostic.py --csv /path/to/results.csv
    python scripts/calibration_diagnostic.py  # uses default path
"""

import argparse
import csv
import logging
import pickle
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATABASE_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Default CSV path
DEFAULT_CSV = Path(__file__).resolve().parent.parent.parent / "trav_daddy_various_model_resutls.csv"


def parse_prediction(raw: str) -> float:
    """Parse prediction string to a float between 0 and 1. Returns None if unparseable."""
    if not raw or raw.strip() == '' or raw.strip().lower() == 'non-favourite':
        return None
    raw = raw.strip().replace('%', '')
    try:
        val = float(raw)
        # If > 1, treat as percentage
        if val > 1:
            val = val / 100.0
        return val
    except ValueError:
        return None


def load_csv(csv_path: str) -> list:
    """Load and parse the results CSV."""
    rows = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {
                'date': row.get('Date', '').strip(),
                'sportsbook': row.get('Sportsbook', '').strip(),
                'league': row.get('League', '').strip(),
                'model': row.get('Model', '').strip(),
                'pick': row.get('Pick', '').strip(),
                'home_away': row.get('vs', '').strip(),  # 'v' = home, '@' = away
                'opponent': row.get('Opponent', '').strip(),
                'prediction': parse_prediction(row.get('Prediction', '')),
                'bet_type': row.get('Type of Bet', '').strip(),
                'odds': None,
                'result': row.get('Result', '').strip().upper(),
                'wager': None,
                'profit_loss': None,
            }
            # Parse odds
            try:
                parsed['odds'] = float(row.get('Odds', '0').strip() or '0')
            except ValueError:
                pass
            # Parse wager
            try:
                wager_str = row.get('Wager', '$0').strip().replace('$', '').replace(',', '')
                parsed['wager'] = float(wager_str) if wager_str else 0
            except ValueError:
                pass
            # Parse profit/loss
            try:
                pl_str = row.get('Profit/Loss ($)', '$0').strip().replace('$', '').replace(',', '')
                parsed['profit_loss'] = float(pl_str) if pl_str else 0
            except ValueError:
                pass
            rows.append(parsed)
    return rows


def get_db_connection():
    """Get a read-only database connection."""
    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def load_player_dist_counts():
    """Load player distribution files and return counts of known players."""
    base = Path(__file__).resolve().parent.parent / "data" / "processed"
    counts = {}
    for gender_label, gender_key in [('male', 't20_male'), ('female', 't20_female')]:
        path = base / f"player_distributions_{gender_key}.pkl"
        if path.exists():
            with open(path, 'rb') as f:
                data = pickle.load(f)
            counts[gender_key] = {
                'batter_ids': set(int(k) for k in data.get('batter_distributions', {}).keys()),
                'bowler_ids': set(int(k) for k in data.get('bowler_distributions', {}).keys()),
            }
        else:
            counts[gender_key] = {'batter_ids': set(), 'bowler_ids': set()}
    return counts


def fuzzy_team_lookup(conn, name: str):
    """Try to find a team in the DB by name (exact, then LIKE)."""
    cursor = conn.cursor()

    # Clean up common suffixes
    clean = name.strip()
    clean = re.sub(r'\s*\((M|W)\)\s*$', '', clean)  # Remove (M) / (W) gender markers
    clean = re.sub(r'\s*Cricket Team\s*$', '', clean)
    clean = re.sub(r'\s*Men\'?s?\s*$', '', clean)
    clean = re.sub(r'\s*Women\'?s?\s*$', '', clean)

    # Try exact match
    cursor.execute("SELECT team_id, name, tier FROM teams WHERE name = ?", (clean,))
    row = cursor.fetchone()
    if row:
        return dict(row)

    # Try LIKE match
    cursor.execute("SELECT team_id, name, tier FROM teams WHERE name LIKE ?", (f"%{clean}%",))
    row = cursor.fetchone()
    if row:
        return dict(row)

    # Try with original name
    cursor.execute("SELECT team_id, name, tier FROM teams WHERE name LIKE ?", (f"%{name}%",))
    row = cursor.fetchone()
    if row:
        return dict(row)

    return None


def get_team_elo(conn, team_id: int, format_key: str = 't20', gender: str = 'male') -> float:
    """Get current team ELO."""
    cursor = conn.cursor()
    col = f'elo_{format_key}_{gender}'
    cursor.execute(f"SELECT {col} FROM team_current_elo WHERE team_id = ?", (team_id,))
    row = cursor.fetchone()
    return row[0] if row else 1500.0


def analyze_calibration(rows: list) -> dict:
    """
    Compute calibration metrics by confidence bucket.

    For each row:
    - prediction = model's P(pick wins)
    - If result == 'W', the pick won -> actual = 1
    - If result == 'L':
        - If bet_type == 'Lay Against', the pick WON (we lost the lay) -> actual = 1
        - If bet_type == 'Win', the pick lost -> actual = 0
    - If result == 'P', push -> exclude

    For predictions < 50%, the model actually favors the opponent.
    We convert: model_confidence = max(prediction, 1 - prediction)
    and model_correct = 1 if the favored team won, else 0.
    """
    buckets = defaultdict(lambda: {'correct': 0, 'total': 0, 'predictions': [], 'details': []})

    analyzed = []
    for row in rows:
        pred = row['prediction']
        result = row['result']

        # Skip rows without predictions or results
        if pred is None or result not in ('W', 'L'):
            continue

        # Determine if the PICK won
        if row['bet_type'] == 'Lay Against':
            # Lay against means we bet AGAINST the pick
            # L = the pick WON (we lost), W = the pick LOST (we won the lay)
            pick_won = (result == 'L')
        else:
            pick_won = (result == 'W')

        # Model's predicted probability for the pick to win
        model_p_pick = pred

        # Determine model confidence and whether the model's favored team won
        if model_p_pick >= 0.5:
            # Model favors the pick
            model_confidence = model_p_pick
            model_favored_won = pick_won
        else:
            # Model favors the opponent
            model_confidence = 1 - model_p_pick
            model_favored_won = not pick_won

        # Bucket by model confidence (10% buckets)
        bucket_key = min(int(model_confidence * 10), 9)  # 5-9
        bucket_label = f"{bucket_key * 10}-{bucket_key * 10 + 10}%"

        buckets[bucket_label]['total'] += 1
        if model_favored_won:
            buckets[bucket_label]['correct'] += 1
        buckets[bucket_label]['predictions'].append(model_confidence)
        buckets[bucket_label]['details'].append({
            'date': row['date'],
            'pick': row['pick'],
            'opponent': row['opponent'],
            'prediction': model_p_pick,
            'confidence': model_confidence,
            'favored_won': model_favored_won,
            'league': row['league'],
        })

        analyzed.append({
            **row,
            'pick_won': pick_won,
            'model_confidence': model_confidence,
            'model_favored_won': model_favored_won,
        })

    return {
        'buckets': dict(buckets),
        'analyzed': analyzed,
    }


def print_calibration_report(cal: dict, conn):
    """Print a comprehensive calibration report."""
    buckets = cal['buckets']
    analyzed = cal['analyzed']

    print("\n" + "=" * 80)
    print("CALIBRATION DIAGNOSTIC REPORT")
    print("=" * 80)

    # ---- Overall Stats ----
    total = len(analyzed)
    correct = sum(1 for r in analyzed if r['model_favored_won'])
    if total > 0:
        print(f"\nOverall: {correct}/{total} model-favored team won ({100 * correct / total:.1f}%)")
    else:
        print("\nNo predictions with results to analyze.")
        return

    # ---- Calibration by Bucket ----
    print(f"\n{'Bucket':<12} {'Correct':>8} {'Total':>6} {'Actual%':>8} {'AvgPred%':>9} {'Gap':>6}")
    print("-" * 55)

    for bucket in sorted(buckets.keys()):
        data = buckets[bucket]
        actual_pct = 100 * data['correct'] / data['total'] if data['total'] > 0 else 0
        avg_pred = 100 * sum(data['predictions']) / len(data['predictions']) if data['predictions'] else 0
        gap = actual_pct - avg_pred
        print(f"{bucket:<12} {data['correct']:>8} {data['total']:>6} {actual_pct:>7.1f}% {avg_pred:>8.1f}% {gap:>+5.1f}%")

    # ---- Worst Misses (high confidence, wrong) ----
    high_conf_wrong = [r for r in analyzed if r['model_confidence'] >= 0.85 and not r['model_favored_won']]
    high_conf_wrong.sort(key=lambda x: x['model_confidence'], reverse=True)

    if high_conf_wrong:
        print(f"\n{'=' * 80}")
        print(f"WORST MISSES: High confidence (>=85%) where model-favored team LOST")
        print(f"{'=' * 80}")
        print(f"{'Date':<18} {'Pick':<30} {'Opponent':<30} {'Pred':>6} {'League'}")
        print("-" * 100)
        for r in high_conf_wrong[:15]:
            pred_str = f"{r['prediction'] * 100:.1f}%" if r['prediction'] is not None else "N/A"
            print(f"{r['date']:<18} {r['pick'][:29]:<30} {r['opponent'][:29]:<30} {pred_str:>6} {r['league'][:30]}")

    # ---- Extreme predictions (>95%) ----
    extreme = [r for r in analyzed if r['model_confidence'] >= 0.95]
    extreme_wrong = [r for r in extreme if not r['model_favored_won']]

    print(f"\n{'=' * 80}")
    print(f"EXTREME PREDICTIONS (>=95% confidence)")
    print(f"{'=' * 80}")
    print(f"Total extreme predictions: {len(extreme)}")
    print(f"Extreme predictions wrong: {len(extreme_wrong)} ({100 * len(extreme_wrong) / len(extreme):.1f}%)" if extreme else "")
    if extreme_wrong:
        print(f"\nDetails of extreme misses:")
        for r in extreme_wrong:
            pred_str = f"{r['prediction'] * 100:.1f}%"
            print(f"  {r['date']} | {r['pick']} vs {r['opponent']} | Pred: {pred_str} | {r['league']}")

    # ---- ELO Analysis for suspicious predictions ----
    print(f"\n{'=' * 80}")
    print(f"TEAM ELO ANALYSIS (Current Ratings)")
    print(f"{'=' * 80}")

    # Collect unique teams from analyzed data
    team_names = set()
    for r in analyzed:
        team_names.add(r['pick'])
        team_names.add(r['opponent'])

    team_elos = {}
    team_tiers = {}
    for name in sorted(team_names):
        team_db = fuzzy_team_lookup(conn, name)
        if team_db:
            # Try male first, then female
            elo_m = get_team_elo(conn, team_db['team_id'], 't20', 'male')
            elo_f = get_team_elo(conn, team_db['team_id'], 't20', 'female')
            elo = elo_m if elo_m != 1500.0 else elo_f
            team_elos[name] = elo
            team_tiers[name] = team_db['tier']

    # Show predictions where ELO gap disagrees with model prediction
    print(f"\n{'Matchup':<55} {'Pred':>6} {'ELO Gap':>8} {'Tiers':>6} {'Result':>7}")
    print("-" * 90)

    disagreements = []
    for r in analyzed:
        pick_elo = team_elos.get(r['pick'], 1500)
        opp_elo = team_elos.get(r['opponent'], 1500)
        elo_gap = pick_elo - opp_elo
        pick_tier = team_tiers.get(r['pick'], '?')
        opp_tier = team_tiers.get(r['opponent'], '?')

        # ELO-implied probability for the pick
        elo_implied = 1 / (1 + 10 ** (-elo_gap / 400))

        pred = r['prediction']
        if pred is not None:
            pred_gap = abs(pred - elo_implied)
            if pred_gap > 0.20:  # >20% disagreement
                disagreements.append({
                    'matchup': f"{r['pick'][:25]} vs {r['opponent'][:25]}",
                    'pred': pred,
                    'elo_implied': elo_implied,
                    'elo_gap': elo_gap,
                    'tiers': f"T{pick_tier}vT{opp_tier}",
                    'result': 'W' if r['pick_won'] else 'L',
                    'date': r['date'],
                    'pred_gap': pred_gap,
                })

    disagreements.sort(key=lambda x: x['pred_gap'], reverse=True)
    for d in disagreements[:15]:
        print(f"{d['date'][:10]} {d['matchup']:<43} {d['pred'] * 100:>5.1f}% {d['elo_gap']:>+7.0f} {d['tiers']:>6} {d['result']:>7}"
              f"  (ELO implies {d['elo_implied'] * 100:.1f}%, gap={d['pred_gap'] * 100:.0f}%)")


def print_data_quality_report(conn, dist_counts: dict):
    """Report on data quality - which tiers have poorest distribution coverage."""
    cursor = conn.cursor()

    print(f"\n{'=' * 80}")
    print(f"DATA QUALITY: Player Distribution Coverage by Team Tier")
    print(f"{'=' * 80}")

    for gender_key, gender_label in [('t20_male', 'T20 Male'), ('t20_female', 'T20 Female')]:
        dists = dist_counts.get(gender_key, {'batter_ids': set(), 'bowler_ids': set()})
        all_known = dists['batter_ids'] | dists['bowler_ids']

        print(f"\n--- {gender_label} ---")
        print(f"Total players with distributions: {len(all_known)}")

        # Get players by team tier (using most recent match affiliations)
        format_type = 'T20'
        gender = 'male' if 'male' in gender_key else 'female'

        cursor.execute("""
            SELECT t.tier, COUNT(DISTINCT pms.player_id) as player_count,
                   GROUP_CONCAT(DISTINCT pms.player_id) as player_ids
            FROM player_match_stats pms
            JOIN matches m ON pms.match_id = m.match_id
            JOIN teams t ON pms.team_id = t.team_id
            WHERE m.match_type = ? AND m.gender = ?
              AND m.date >= '2024-01-01'
            GROUP BY t.tier
            ORDER BY t.tier
        """, (format_type, gender))

        print(f"\n{'Tier':<6} {'Players':>8} {'WithDist':>9} {'Coverage':>9}")
        print("-" * 35)

        for row in cursor.fetchall():
            tier = row['tier']
            total = row['player_count']
            player_ids_str = row['player_ids']
            if player_ids_str:
                pids = set(int(p) for p in player_ids_str.split(','))
                with_dist = len(pids & all_known)
            else:
                with_dist = 0
            coverage = 100 * with_dist / total if total > 0 else 0
            print(f"T{tier:<5} {total:>8} {with_dist:>9} {coverage:>8.1f}%")


def print_elo_tier_summary(conn):
    """Print ELO distribution by tier to show if ceilings/floors are being hit."""
    cursor = conn.cursor()

    from src.elo.calculator_v3 import EloCalculatorV3

    print(f"\n{'=' * 80}")
    print(f"ELO DISTRIBUTION BY TIER (T20 Male)")
    print(f"{'=' * 80}")

    print(f"\n{'Tier':<6} {'Count':>6} {'Min':>8} {'Avg':>8} {'Max':>8} {'Floor':>7} {'Ceil':>7} {'AtFloor':>8} {'AtCeil':>8}")
    print("-" * 75)

    for tier in range(1, 6):
        cursor.execute("""
            SELECT COUNT(*) as cnt,
                   MIN(e.elo_t20_male) as min_elo,
                   AVG(e.elo_t20_male) as avg_elo,
                   MAX(e.elo_t20_male) as max_elo
            FROM team_current_elo e
            JOIN teams t ON e.team_id = t.team_id
            WHERE t.tier = ? AND e.elo_t20_male != 1500
        """, (tier,))
        row = cursor.fetchone()

        floor_val = EloCalculatorV3.TIER_FLOORS[tier]
        ceil_val = EloCalculatorV3.TIER_CEILINGS[tier]

        # Count teams at floor/ceiling (within 30 points)
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN e.elo_t20_male <= ? + 30 THEN 1 ELSE 0 END) as at_floor,
                SUM(CASE WHEN e.elo_t20_male >= ? - 30 THEN 1 ELSE 0 END) as at_ceil
            FROM team_current_elo e
            JOIN teams t ON e.team_id = t.team_id
            WHERE t.tier = ? AND e.elo_t20_male != 1500
        """, (floor_val, ceil_val, tier))
        bounds = cursor.fetchone()

        if row['cnt'] and row['cnt'] > 0:
            print(f"T{tier:<5} {row['cnt']:>6} {row['min_elo']:>8.0f} {row['avg_elo']:>8.0f} {row['max_elo']:>8.0f}"
                  f" {floor_val:>7} {ceil_val:>7}"
                  f" {bounds['at_floor'] or 0:>8} {bounds['at_ceil'] or 0:>8}")
        else:
            print(f"T{tier:<5} {'(no data)':>6}")


def main():
    parser = argparse.ArgumentParser(description='Calibration Diagnostic')
    parser.add_argument('--csv', type=str, default=str(DEFAULT_CSV),
                        help='Path to results CSV file')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return 1

    logger.info(f"Loading CSV from: {csv_path}")
    rows = load_csv(str(csv_path))
    logger.info(f"Loaded {len(rows)} rows from CSV")

    # Filter to rows with predictions
    with_pred = [r for r in rows if r['prediction'] is not None]
    logger.info(f"Rows with model predictions: {len(with_pred)}")

    # Get DB connection
    conn = get_db_connection()

    # Load player distribution data
    dist_counts = load_player_dist_counts()

    # Run analyses
    cal = analyze_calibration(rows)
    print_calibration_report(cal, conn)
    print_data_quality_report(conn, dist_counts)
    print_elo_tier_summary(conn)

    conn.close()

    print(f"\n{'=' * 80}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'=' * 80}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
