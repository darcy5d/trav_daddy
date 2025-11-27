"""
Batter vs Bowler Head-to-Head Matchup System.

Builds and queries historical matchup data between specific batters and bowlers.
Uses H2H data when ≥25 balls faced, otherwise falls back to ELO-based distributions.
"""

import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import sqlite3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import DATABASE_PATH
from src.data.database import get_connection

logger = logging.getLogger(__name__)

# Minimum balls faced to use H2H data (roughly 4 overs)
MIN_BALLS_FOR_H2H = 25


@dataclass
class MatchupStats:
    """Statistics for a specific batter vs bowler matchup."""
    batter_id: int
    bowler_id: int
    balls_faced: int
    runs_scored: int
    dismissals: int
    dots: int
    singles: int
    twos: int
    threes: int
    fours: int
    sixes: int
    
    @property
    def strike_rate(self) -> float:
        """Runs per 100 balls."""
        if self.balls_faced == 0:
            return 0.0
        return self.runs_scored * 100 / self.balls_faced
    
    @property
    def dismissal_rate(self) -> float:
        """Probability of dismissal per ball."""
        if self.balls_faced == 0:
            return 0.0
        return self.dismissals / self.balls_faced
    
    @property
    def has_sufficient_data(self) -> bool:
        """Check if we have enough data to use H2H stats."""
        return self.balls_faced >= MIN_BALLS_FOR_H2H
    
    def get_outcome_distribution(self) -> Dict[str, float]:
        """
        Convert historical stats to probability distribution.
        
        Returns dict with probabilities for each outcome:
        '0', '1', '2', '3', '4', '6', 'W'
        """
        if self.balls_faced == 0:
            return None
        
        total = self.balls_faced
        
        return {
            '0': self.dots / total,
            '1': self.singles / total,
            '2': self.twos / total,
            '3': self.threes / total,
            '4': self.fours / total,
            '6': self.sixes / total,
            'W': self.dismissals / total
        }


class MatchupDatabase:
    """
    Manages batter vs bowler head-to-head matchup data.
    """
    
    def __init__(self, format_type: str = 'T20', gender: str = 'male'):
        self.format_type = format_type
        self.gender = gender
        self.matchups: Dict[Tuple[int, int], MatchupStats] = {}
        self._loaded = False
    
    def load_from_database(self):
        """Load all H2H matchup data from the deliveries table."""
        conn = get_connection()
        cursor = conn.cursor()
        
        logger.info(f"Loading H2H matchups for {self.format_type} {self.gender}...")
        
        # Query all batter vs bowler matchups
        cursor.execute("""
            SELECT 
                d.batter_id,
                d.bowler_id,
                COUNT(*) as balls_faced,
                SUM(d.runs_batter) as runs_scored,
                SUM(CASE WHEN d.is_wicket = 1 THEN 1 ELSE 0 END) as dismissals,
                SUM(CASE WHEN d.runs_batter = 0 AND d.is_wicket = 0 THEN 1 ELSE 0 END) as dots,
                SUM(CASE WHEN d.runs_batter = 1 THEN 1 ELSE 0 END) as singles,
                SUM(CASE WHEN d.runs_batter = 2 THEN 1 ELSE 0 END) as twos,
                SUM(CASE WHEN d.runs_batter = 3 THEN 1 ELSE 0 END) as threes,
                SUM(CASE WHEN d.runs_batter = 4 THEN 1 ELSE 0 END) as fours,
                SUM(CASE WHEN d.runs_batter >= 6 THEN 1 ELSE 0 END) as sixes
            FROM deliveries d
            JOIN innings i ON d.innings_id = i.innings_id
            JOIN matches m ON i.match_id = m.match_id
            WHERE m.match_type = ? AND m.gender = ?
            GROUP BY d.batter_id, d.bowler_id
        """, (self.format_type, self.gender))
        
        rows = cursor.fetchall()
        
        for row in rows:
            stats = MatchupStats(
                batter_id=row['batter_id'],
                bowler_id=row['bowler_id'],
                balls_faced=row['balls_faced'],
                runs_scored=row['runs_scored'] or 0,
                dismissals=row['dismissals'] or 0,
                dots=row['dots'] or 0,
                singles=row['singles'] or 0,
                twos=row['twos'] or 0,
                threes=row['threes'] or 0,
                fours=row['fours'] or 0,
                sixes=row['sixes'] or 0
            )
            
            self.matchups[(row['batter_id'], row['bowler_id'])] = stats
        
        conn.close()
        self._loaded = True
        
        # Count how many have sufficient data
        sufficient = sum(1 for s in self.matchups.values() if s.has_sufficient_data)
        logger.info(f"Loaded {len(self.matchups)} matchups, {sufficient} with ≥{MIN_BALLS_FOR_H2H} balls")
    
    def get_matchup(self, batter_id: int, bowler_id: int) -> Optional[MatchupStats]:
        """
        Get matchup stats for a specific batter vs bowler.
        
        Returns None if no data exists.
        """
        if not self._loaded:
            self.load_from_database()
        
        return self.matchups.get((batter_id, bowler_id))
    
    def get_h2h_distribution(
        self,
        batter_id: int,
        bowler_id: int
    ) -> Optional[Dict[str, float]]:
        """
        Get outcome distribution for batter vs bowler.
        
        Returns None if insufficient data (< MIN_BALLS_FOR_H2H).
        Falls back to None so caller can use ELO-based distribution.
        """
        stats = self.get_matchup(batter_id, bowler_id)
        
        if stats is None or not stats.has_sufficient_data:
            return None
        
        return stats.get_outcome_distribution()
    
    def get_h2h_distribution_array(
        self,
        batter_id: int,
        bowler_id: int
    ) -> Optional['np.ndarray']:
        """
        Get outcome distribution as numpy array (7 elements).
        
        Returns None if insufficient data.
        Order: [dot, single, two, three, four, six, wicket]
        """
        import numpy as np
        
        stats = self.get_matchup(batter_id, bowler_id)
        
        if stats is None or not stats.has_sufficient_data:
            return None
        
        total = stats.balls_faced
        return np.array([
            stats.dots / total,
            stats.singles / total,
            stats.twos / total,
            stats.threes / total,
            stats.fours / total,
            stats.sixes / total,
            stats.dismissals / total
        ], dtype=np.float32)
    
    def get_top_matchups(self, limit: int = 20) -> List[Tuple[MatchupStats, str, str]]:
        """Get matchups with most balls faced (for analysis)."""
        if not self._loaded:
            self.load_from_database()
        
        conn = get_connection()
        
        # Get player names
        sorted_matchups = sorted(
            self.matchups.values(),
            key=lambda x: x.balls_faced,
            reverse=True
        )[:limit]
        
        results = []
        for stats in sorted_matchups:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM players WHERE player_id = ?", (stats.batter_id,))
            batter_name = cursor.fetchone()['name']
            cursor.execute("SELECT name FROM players WHERE player_id = ?", (stats.bowler_id,))
            bowler_name = cursor.fetchone()['name']
            results.append((stats, batter_name, bowler_name))
        
        conn.close()
        return results
    
    def get_bowlers_faced_by_batter(self, batter_id: int) -> List[MatchupStats]:
        """Get all bowlers a batter has faced (with any data)."""
        if not self._loaded:
            self.load_from_database()
        
        return [
            stats for (bat_id, bowl_id), stats in self.matchups.items()
            if bat_id == batter_id
        ]
    
    def get_batters_faced_by_bowler(self, bowler_id: int) -> List[MatchupStats]:
        """Get all batters a bowler has faced (with any data)."""
        if not self._loaded:
            self.load_from_database()
        
        return [
            stats for (bat_id, bowl_id), stats in self.matchups.items()
            if bowl_id == bowler_id
        ]


# Global matchup databases (lazy loaded)
_matchup_dbs: Dict[str, MatchupDatabase] = {}


def get_matchup_db(format_type: str = 'T20', gender: str = 'male') -> MatchupDatabase:
    """Get or create matchup database for format/gender."""
    key = f"{format_type}_{gender}"
    
    if key not in _matchup_dbs:
        _matchup_dbs[key] = MatchupDatabase(format_type, gender)
    
    return _matchup_dbs[key]


def analyze_matchups():
    """Analyze and print matchup statistics."""
    db = get_matchup_db('T20', 'male')
    db.load_from_database()
    
    print("=" * 70)
    print("BATTER vs BOWLER HEAD-TO-HEAD ANALYSIS (T20 Men)")
    print("=" * 70)
    
    # Overall stats
    total = len(db.matchups)
    sufficient = sum(1 for s in db.matchups.values() if s.has_sufficient_data)
    
    print(f"\nTotal matchups: {total:,}")
    print(f"With ≥{MIN_BALLS_FOR_H2H} balls (usable H2H): {sufficient:,} ({sufficient/total*100:.1f}%)")
    
    # Top matchups by balls faced
    print(f"\n{'='*70}")
    print("TOP 20 MATCHUPS BY BALLS FACED")
    print("=" * 70)
    print(f"{'Batter':<25} {'Bowler':<25} {'Balls':>6} {'Runs':>6} {'SR':>7} {'Dis':>4}")
    print("-" * 70)
    
    top = db.get_top_matchups(20)
    for stats, batter, bowler in top:
        print(f"{batter:<25} {bowler:<25} {stats.balls_faced:>6} {stats.runs_scored:>6} "
              f"{stats.strike_rate:>7.1f} {stats.dismissals:>4}")
    
    # Example outcome distribution
    if top:
        stats, batter, bowler = top[0]
        print(f"\n{'='*70}")
        print(f"OUTCOME DISTRIBUTION: {batter} vs {bowler}")
        print("=" * 70)
        dist = stats.get_outcome_distribution()
        for outcome, prob in dist.items():
            bar = "█" * int(prob * 50)
            print(f"  {outcome}: {prob:>6.1%} {bar}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    analyze_matchups()

