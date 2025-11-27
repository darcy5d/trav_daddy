"""
Player Outcome Distributions.

Computes historical outcome distributions for each batter and bowler:
- For batters: P(0), P(1), P(2), P(3), P(4), P(6), P(W) across all balls faced
- For bowlers: P(0), P(1), P(2), P(3), P(4), P(6), P(W) across all balls bowled

These distributions are used as features for the Ball Prediction Neural Network.
"""

import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import DATABASE_PATH
from src.data.database import get_connection

logger = logging.getLogger(__name__)


@dataclass
class OutcomeDistribution:
    """Outcome distribution for a player."""
    player_id: int
    total_balls: int
    dots: int = 0      # 0 runs, no wicket
    singles: int = 0   # 1 run
    twos: int = 0      # 2 runs
    threes: int = 0    # 3 runs
    fours: int = 0     # 4 runs
    sixes: int = 0     # 6 runs
    wickets: int = 0   # Dismissal
    wides: int = 0     # Wide balls (bowler only)
    noballs: int = 0   # No balls (bowler only)
    
    def to_probability_vector(self) -> np.ndarray:
        """
        Convert to probability vector for NN input.
        
        Returns 8-element vector: [P(0), P(1), P(2), P(3), P(4), P(6), P(W), P(extra)]
        """
        if self.total_balls == 0:
            # Return uniform distribution for unknown players
            return np.array([0.35, 0.30, 0.08, 0.02, 0.12, 0.06, 0.05, 0.02])
        
        total = self.total_balls
        extras = self.wides + self.noballs
        
        return np.array([
            self.dots / total,
            self.singles / total,
            self.twos / total,
            self.threes / total,
            self.fours / total,
            self.sixes / total,
            self.wickets / total,
            extras / total if extras > 0 else 0.02  # Small default for extras
        ])
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'player_id': self.player_id,
            'total_balls': self.total_balls,
            'dots': self.dots,
            'singles': self.singles,
            'twos': self.twos,
            'threes': self.threes,
            'fours': self.fours,
            'sixes': self.sixes,
            'wickets': self.wickets,
            'wides': self.wides,
            'noballs': self.noballs,
            'prob_vector': self.to_probability_vector().tolist()
        }


class PlayerDistributionBuilder:
    """
    Builds and stores outcome distributions for all players.
    
    Distributions are computed from historical deliveries data.
    """
    
    def __init__(self, format_type: str = 'T20', gender: str = 'male'):
        self.format_type = format_type
        self.gender = gender
        self.batter_distributions: Dict[int, OutcomeDistribution] = {}
        self.bowler_distributions: Dict[int, OutcomeDistribution] = {}
        self._loaded = False
    
    def build_from_database(self, min_balls: int = 10):
        """
        Build distributions from deliveries table.
        
        Args:
            min_balls: Minimum balls to include a player (default 10)
        """
        conn = get_connection()
        
        logger.info(f"Building player distributions for {self.format_type} {self.gender}...")
        
        # Build BATTER distributions
        logger.info("Computing batter distributions...")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                d.batter_id,
                COUNT(*) as total_balls,
                SUM(CASE WHEN d.runs_batter = 0 AND d.is_wicket = 0 THEN 1 ELSE 0 END) as dots,
                SUM(CASE WHEN d.runs_batter = 1 THEN 1 ELSE 0 END) as singles,
                SUM(CASE WHEN d.runs_batter = 2 THEN 1 ELSE 0 END) as twos,
                SUM(CASE WHEN d.runs_batter = 3 THEN 1 ELSE 0 END) as threes,
                SUM(CASE WHEN d.runs_batter = 4 THEN 1 ELSE 0 END) as fours,
                SUM(CASE WHEN d.runs_batter >= 6 THEN 1 ELSE 0 END) as sixes,
                SUM(CASE WHEN d.is_wicket = 1 THEN 1 ELSE 0 END) as wickets
            FROM deliveries d
            JOIN innings i ON d.innings_id = i.innings_id
            JOIN matches m ON i.match_id = m.match_id
            WHERE m.match_type = ? AND m.gender = ?
            GROUP BY d.batter_id
            HAVING total_balls >= ?
        """, (self.format_type, self.gender, min_balls))
        
        for row in cursor.fetchall():
            dist = OutcomeDistribution(
                player_id=row['batter_id'],
                total_balls=row['total_balls'],
                dots=row['dots'] or 0,
                singles=row['singles'] or 0,
                twos=row['twos'] or 0,
                threes=row['threes'] or 0,
                fours=row['fours'] or 0,
                sixes=row['sixes'] or 0,
                wickets=row['wickets'] or 0
            )
            self.batter_distributions[row['batter_id']] = dist
        
        logger.info(f"Built distributions for {len(self.batter_distributions)} batters")
        
        # Build BOWLER distributions
        logger.info("Computing bowler distributions...")
        cursor.execute("""
            SELECT 
                d.bowler_id,
                COUNT(*) as total_balls,
                SUM(CASE WHEN d.runs_batter = 0 AND d.is_wicket = 0 THEN 1 ELSE 0 END) as dots,
                SUM(CASE WHEN d.runs_batter = 1 THEN 1 ELSE 0 END) as singles,
                SUM(CASE WHEN d.runs_batter = 2 THEN 1 ELSE 0 END) as twos,
                SUM(CASE WHEN d.runs_batter = 3 THEN 1 ELSE 0 END) as threes,
                SUM(CASE WHEN d.runs_batter = 4 THEN 1 ELSE 0 END) as fours,
                SUM(CASE WHEN d.runs_batter >= 6 THEN 1 ELSE 0 END) as sixes,
                SUM(CASE WHEN d.is_wicket = 1 THEN 1 ELSE 0 END) as wickets,
                SUM(CASE WHEN d.extras_wides > 0 THEN 1 ELSE 0 END) as wides,
                SUM(CASE WHEN d.extras_noballs > 0 THEN 1 ELSE 0 END) as noballs
            FROM deliveries d
            JOIN innings i ON d.innings_id = i.innings_id
            JOIN matches m ON i.match_id = m.match_id
            WHERE m.match_type = ? AND m.gender = ?
            GROUP BY d.bowler_id
            HAVING total_balls >= ?
        """, (self.format_type, self.gender, min_balls))
        
        for row in cursor.fetchall():
            dist = OutcomeDistribution(
                player_id=row['bowler_id'],
                total_balls=row['total_balls'],
                dots=row['dots'] or 0,
                singles=row['singles'] or 0,
                twos=row['twos'] or 0,
                threes=row['threes'] or 0,
                fours=row['fours'] or 0,
                sixes=row['sixes'] or 0,
                wickets=row['wickets'] or 0,
                wides=row['wides'] or 0,
                noballs=row['noballs'] or 0
            )
            self.bowler_distributions[row['bowler_id']] = dist
        
        logger.info(f"Built distributions for {len(self.bowler_distributions)} bowlers")
        
        conn.close()
        self._loaded = True
    
    def get_batter_distribution(self, player_id: int) -> OutcomeDistribution:
        """Get batter distribution, returning default if not found."""
        if player_id in self.batter_distributions:
            return self.batter_distributions[player_id]
        
        # Return default distribution for unknown batters
        return OutcomeDistribution(player_id=player_id, total_balls=0)
    
    def get_bowler_distribution(self, player_id: int) -> OutcomeDistribution:
        """Get bowler distribution, returning default if not found."""
        if player_id in self.bowler_distributions:
            return self.bowler_distributions[player_id]
        
        # Return default distribution for unknown bowlers
        return OutcomeDistribution(player_id=player_id, total_balls=0)
    
    def get_batter_vector(self, player_id: int) -> np.ndarray:
        """Get batter probability vector (8 elements)."""
        return self.get_batter_distribution(player_id).to_probability_vector()
    
    def get_bowler_vector(self, player_id: int) -> np.ndarray:
        """Get bowler probability vector (8 elements)."""
        return self.get_bowler_distribution(player_id).to_probability_vector()
    
    def save(self, filepath: str):
        """Save distributions to pickle file."""
        data = {
            'format_type': self.format_type,
            'gender': self.gender,
            'batter_distributions': {
                k: v.to_dict() for k, v in self.batter_distributions.items()
            },
            'bowler_distributions': {
                k: v.to_dict() for k, v in self.bowler_distributions.items()
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved distributions to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'PlayerDistributionBuilder':
        """Load distributions from pickle file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        builder = cls(data['format_type'], data['gender'])
        
        for player_id, d in data['batter_distributions'].items():
            builder.batter_distributions[int(player_id)] = OutcomeDistribution(
                player_id=d['player_id'],
                total_balls=d['total_balls'],
                dots=d['dots'],
                singles=d['singles'],
                twos=d['twos'],
                threes=d['threes'],
                fours=d['fours'],
                sixes=d['sixes'],
                wickets=d['wickets']
            )
        
        for player_id, d in data['bowler_distributions'].items():
            builder.bowler_distributions[int(player_id)] = OutcomeDistribution(
                player_id=d['player_id'],
                total_balls=d['total_balls'],
                dots=d['dots'],
                singles=d['singles'],
                twos=d['twos'],
                threes=d['threes'],
                fours=d['fours'],
                sixes=d['sixes'],
                wickets=d['wickets'],
                wides=d.get('wides', 0),
                noballs=d.get('noballs', 0)
            )
        
        builder._loaded = True
        logger.info(f"Loaded {len(builder.batter_distributions)} batter and "
                   f"{len(builder.bowler_distributions)} bowler distributions")
        
        return builder


def build_and_save_distributions(format_type: str = 'T20', gender: str = 'male', min_balls: int = 10):
    """Build and save distributions for specified format and gender."""
    builder = PlayerDistributionBuilder(format_type, gender)
    builder.build_from_database(min_balls=min_balls)
    
    output_path = Path(f'data/processed/player_distributions_{format_type.lower()}_{gender}.pkl')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    builder.save(str(output_path))
    
    logger.info(f"Saved {format_type} {gender} distributions: {len(builder.batter_distributions)} batters, {len(builder.bowler_distributions)} bowlers")
    
    return builder


def analyze_distributions(format_type: str = 'T20', gender: str = 'male'):
    """Analyze and print distribution statistics."""
    builder = PlayerDistributionBuilder(format_type, gender)
    builder.build_from_database(min_balls=50)
    
    print("=" * 70)
    print(f"PLAYER OUTCOME DISTRIBUTIONS ANALYSIS ({format_type} {gender.upper()})")
    print("=" * 70)
    
    print(f"\nBatters with ≥50 balls: {len(builder.batter_distributions)}")
    print(f"Bowlers with ≥50 balls: {len(builder.bowler_distributions)}")
    
    # Top batters by balls faced
    print("\n" + "=" * 70)
    print("TOP 10 BATTERS BY BALLS FACED")
    print("=" * 70)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    top_batters = sorted(
        builder.batter_distributions.values(),
        key=lambda x: x.total_balls,
        reverse=True
    )[:10]
    
    print(f"\n{'Name':<25} {'Balls':>7} {'SR':>7} {'Dot%':>7} {'Bdry%':>7} {'Dis%':>6}")
    print("-" * 70)
    
    for dist in top_batters:
        cursor.execute("SELECT name FROM players WHERE player_id = ?", (dist.player_id,))
        name = cursor.fetchone()['name']
        
        total = dist.total_balls
        runs = dist.singles + 2*dist.twos + 3*dist.threes + 4*dist.fours + 6*dist.sixes
        sr = runs * 100 / total if total > 0 else 0
        dot_pct = dist.dots * 100 / total if total > 0 else 0
        boundary_pct = (dist.fours + dist.sixes) * 100 / total if total > 0 else 0
        dismissal_pct = dist.wickets * 100 / total if total > 0 else 0
        
        print(f"{name:<25} {total:>7} {sr:>7.1f} {dot_pct:>7.1f} {boundary_pct:>7.1f} {dismissal_pct:>6.1f}")
    
    # Top bowlers by balls bowled
    print("\n" + "=" * 70)
    print("TOP 10 BOWLERS BY BALLS BOWLED")
    print("=" * 70)
    
    top_bowlers = sorted(
        builder.bowler_distributions.values(),
        key=lambda x: x.total_balls,
        reverse=True
    )[:10]
    
    print(f"\n{'Name':<25} {'Balls':>7} {'Econ':>7} {'Dot%':>7} {'Wkt%':>6}")
    print("-" * 70)
    
    for dist in top_bowlers:
        cursor.execute("SELECT name FROM players WHERE player_id = ?", (dist.player_id,))
        name = cursor.fetchone()['name']
        
        total = dist.total_balls
        runs_conceded = dist.singles + 2*dist.twos + 3*dist.threes + 4*dist.fours + 6*dist.sixes
        economy = runs_conceded * 6 / total if total > 0 else 0
        dot_pct = dist.dots * 100 / total if total > 0 else 0
        wicket_pct = dist.wickets * 100 / total if total > 0 else 0
        
        print(f"{name:<25} {total:>7} {economy:>7.2f} {dot_pct:>7.1f} {wicket_pct:>6.2f}")
    
    conn.close()
    
    # Show example probability vector
    if top_batters:
        print("\n" + "=" * 70)
        print("EXAMPLE: Probability Vector for Top Batter")
        print("=" * 70)
        vec = top_batters[0].to_probability_vector()
        labels = ['P(0)', 'P(1)', 'P(2)', 'P(3)', 'P(4)', 'P(6)', 'P(W)', 'P(ex)']
        for label, prob in zip(labels, vec):
            bar = "█" * int(prob * 50)
            print(f"  {label}: {prob:>6.1%} {bar}")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    parser = argparse.ArgumentParser(description='Build player outcome distributions')
    parser.add_argument('--format', default='T20', choices=['T20', 'ODI'])
    parser.add_argument('--gender', default='male', choices=['male', 'female'])
    parser.add_argument('--all', action='store_true', help='Build for all formats/genders')
    args = parser.parse_args()
    
    if args.all:
        # Build for all combinations
        for fmt in ['T20']:
            for gen in ['male', 'female']:
                print(f"\n{'='*70}")
                print(f"Building {fmt} {gen} distributions...")
                print("=" * 70)
                build_and_save_distributions(fmt, gen)
        print("\nDone! All distributions saved.")
    else:
        print(f"Building and analyzing {args.format} {args.gender} player distributions...\n")
        analyze_distributions(args.format, args.gender)
        
        print("\n" + "=" * 70)
        print("Saving distributions to file...")
        print("=" * 70)
        build_and_save_distributions(args.format, args.gender)
        print("Done!")

