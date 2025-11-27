"""
Venue Statistics Builder.

Builds venue-level statistics for training features:
- Average first innings score (scoring factor)
- Boundary rate (% of runs from 4s and 6s)
- Wicket rate (wickets per ball)
- Match count for reliability

These features help the neural network learn venue-specific patterns.
"""

import logging
from typing import Dict, Optional
import numpy as np
from pathlib import Path
import pickle

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.database import get_connection

logger = logging.getLogger(__name__)


class VenueStatsBuilder:
    """
    Build venue-level statistics for training features.
    """
    
    # Default values for unknown venues
    DEFAULT_AVG_SCORE_T20 = 155.0
    DEFAULT_AVG_SCORE_ODI = 250.0
    DEFAULT_BOUNDARY_RATE = 0.22  # ~22% of runs from boundaries
    DEFAULT_WICKET_RATE = 0.042   # ~4.2% wicket per ball
    MIN_MATCHES_RELIABLE = 10     # Minimum matches for reliable stats
    
    def __init__(self, format_type: str = 'T20', gender: str = 'male'):
        self.format_type = format_type
        self.gender = gender
        self.venue_stats: Dict[int, Dict] = {}
        self.global_avg_score: float = self.DEFAULT_AVG_SCORE_T20 if format_type == 'T20' else self.DEFAULT_AVG_SCORE_ODI
        
    def build_from_database(self):
        """
        Build venue statistics from historical match data.
        """
        conn = get_connection()
        cursor = conn.cursor()
        
        # First get global average 1st innings score
        cursor.execute("""
            SELECT AVG(i.total_runs) as global_avg
            FROM innings i
            JOIN matches m ON i.match_id = m.match_id
            WHERE m.match_type = ? AND m.gender = ?
            AND i.innings_number = 1
        """, (self.format_type, self.gender))
        
        row = cursor.fetchone()
        if row and row['global_avg']:
            self.global_avg_score = row['global_avg']
        
        logger.info(f"Global average 1st innings score: {self.global_avg_score:.1f}")
        
        # Get venue-level stats
        cursor.execute("""
            SELECT 
                v.venue_id,
                v.name as venue_name,
                v.city,
                COUNT(DISTINCT m.match_id) as matches,
                AVG(i.total_runs) as avg_first_innings
            FROM venues v
            JOIN matches m ON m.venue_id = v.venue_id
            JOIN innings i ON i.match_id = m.match_id AND i.innings_number = 1
            WHERE m.match_type = ? AND m.gender = ?
            GROUP BY v.venue_id
            HAVING matches >= 3
            ORDER BY matches DESC
        """, (self.format_type, self.gender))
        
        venue_rows = cursor.fetchall()
        logger.info(f"Found {len(venue_rows)} venues with 3+ matches")
        
        # Get delivery-level stats per venue
        for venue_row in venue_rows:
            venue_id = venue_row['venue_id']
            
            # Get boundary and wicket rates
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_balls,
                    SUM(CASE WHEN d.runs_batter = 4 THEN 1 ELSE 0 END) as fours,
                    SUM(CASE WHEN d.runs_batter = 6 THEN 1 ELSE 0 END) as sixes,
                    SUM(d.runs_batter) as total_runs,
                    SUM(CASE WHEN d.is_wicket = 1 THEN 1 ELSE 0 END) as wickets
                FROM deliveries d
                JOIN innings i ON d.innings_id = i.innings_id
                JOIN matches m ON i.match_id = m.match_id
                WHERE m.venue_id = ? AND m.match_type = ? AND m.gender = ?
            """, (venue_id, self.format_type, self.gender))
            
            delivery_row = cursor.fetchone()
            
            total_balls = delivery_row['total_balls'] if delivery_row else 0
            
            if total_balls > 0:
                fours = delivery_row['fours'] or 0
                sixes = delivery_row['sixes'] or 0
                total_runs = delivery_row['total_runs'] or 0
                wickets = delivery_row['wickets'] or 0
                
                # Boundary rate = % of runs from boundaries
                boundary_runs = (fours * 4) + (sixes * 6)
                boundary_rate = boundary_runs / total_runs if total_runs > 0 else self.DEFAULT_BOUNDARY_RATE
                
                # Wicket rate = wickets per ball
                wicket_rate = wickets / total_balls
                
                # Scoring factor = venue_avg / global_avg
                avg_score = venue_row['avg_first_innings'] or self.global_avg_score
                scoring_factor = avg_score / self.global_avg_score
            else:
                boundary_rate = self.DEFAULT_BOUNDARY_RATE
                wicket_rate = self.DEFAULT_WICKET_RATE
                scoring_factor = 1.0
                avg_score = self.global_avg_score
            
            self.venue_stats[venue_id] = {
                'venue_name': venue_row['venue_name'],
                'city': venue_row['city'],
                'matches': venue_row['matches'],
                'avg_score': avg_score,
                'scoring_factor': scoring_factor,
                'boundary_rate': boundary_rate,
                'wicket_rate': wicket_rate,
                'total_balls': total_balls
            }
        
        conn.close()
        logger.info(f"Built statistics for {len(self.venue_stats)} venues")
        
        # Log top venues
        top_scoring = sorted(
            self.venue_stats.items(), 
            key=lambda x: x[1]['scoring_factor'], 
            reverse=True
        )[:5]
        
        logger.info("Top 5 high-scoring venues:")
        for venue_id, stats in top_scoring:
            logger.info(f"  {stats['venue_name']}: {stats['avg_score']:.1f} avg ({stats['scoring_factor']:.2f}x)")
        
        low_scoring = sorted(
            self.venue_stats.items(), 
            key=lambda x: x[1]['scoring_factor']
        )[:5]
        
        logger.info("Top 5 low-scoring venues:")
        for venue_id, stats in low_scoring:
            logger.info(f"  {stats['venue_name']}: {stats['avg_score']:.1f} avg ({stats['scoring_factor']:.2f}x)")
    
    def get_venue_features(self, venue_id: Optional[int]) -> np.ndarray:
        """
        Get 4-element feature vector for a venue.
        
        Features:
            1. scoring_factor: venue_avg / global_avg (1.0 = neutral)
            2. boundary_rate: % of runs from boundaries
            3. wicket_rate: wickets per ball
            4. has_reliable_data: 1 if 10+ matches, else 0
        
        Returns:
            np.ndarray of shape (4,)
        """
        if venue_id is None or venue_id not in self.venue_stats:
            # Return default features for unknown venues
            return np.array([
                1.0,  # neutral scoring factor
                self.DEFAULT_BOUNDARY_RATE,
                self.DEFAULT_WICKET_RATE,
                0.0   # not reliable
            ], dtype=np.float32)
        
        stats = self.venue_stats[venue_id]
        
        return np.array([
            stats['scoring_factor'],
            stats['boundary_rate'],
            stats['wicket_rate'],
            1.0 if stats['matches'] >= self.MIN_MATCHES_RELIABLE else 0.0
        ], dtype=np.float32)
    
    def get_venue_info(self, venue_id: int) -> Dict:
        """Get full venue information."""
        if venue_id in self.venue_stats:
            return self.venue_stats[venue_id]
        return {
            'venue_name': 'Unknown',
            'city': None,
            'matches': 0,
            'avg_score': self.global_avg_score,
            'scoring_factor': 1.0,
            'boundary_rate': self.DEFAULT_BOUNDARY_RATE,
            'wicket_rate': self.DEFAULT_WICKET_RATE
        }
    
    def adjust_distribution_for_venue(
        self, 
        base_dist: np.ndarray, 
        venue_id: Optional[int]
    ) -> np.ndarray:
        """
        Adjust a ball outcome distribution based on venue characteristics.
        
        This shifts boundary and wicket probabilities based on venue stats.
        
        Args:
            base_dist: Base probability distribution [dot, 1, 2, 3, 4, 6, wicket]
            venue_id: Venue ID (or None for neutral)
        
        Returns:
            Adjusted distribution (normalized)
        """
        if venue_id is None or venue_id not in self.venue_stats:
            return base_dist
        
        stats = self.venue_stats[venue_id]
        scoring_factor = stats['scoring_factor']
        
        # Copy base distribution
        adjusted = base_dist.copy()
        
        # Adjust based on scoring factor
        # High scoring: more 4s and 6s, fewer dots
        # Low scoring: more dots, fewer boundaries
        
        if scoring_factor > 1.05:
            # High scoring venue: boost boundaries
            boundary_boost = (scoring_factor - 1.0) * 0.5  # 10% higher = 5% boost
            adjusted[4] *= (1 + boundary_boost)  # 4s
            adjusted[5] *= (1 + boundary_boost)  # 6s
            adjusted[0] *= (1 - boundary_boost * 0.3)  # fewer dots
        elif scoring_factor < 0.95:
            # Low scoring venue: reduce boundaries
            boundary_reduce = (1.0 - scoring_factor) * 0.5
            adjusted[4] *= (1 - boundary_reduce)  # 4s
            adjusted[5] *= (1 - boundary_reduce)  # 6s
            adjusted[0] *= (1 + boundary_reduce * 0.3)  # more dots
        
        # Normalize
        adjusted = adjusted / adjusted.sum()
        
        return adjusted
    
    def save(self, path: Optional[str] = None):
        """Save venue statistics to file."""
        if path is None:
            path = f'data/processed/venue_stats_{self.format_type.lower()}_{self.gender}.pkl'
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'format_type': self.format_type,
                'gender': self.gender,
                'global_avg_score': self.global_avg_score,
                'venue_stats': self.venue_stats
            }, f)
        logger.info(f"Saved venue statistics to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'VenueStatsBuilder':
        """Load venue statistics from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        builder = cls(data['format_type'], data['gender'])
        builder.global_avg_score = data['global_avg_score']
        builder.venue_stats = data['venue_stats']
        return builder


def build_venue_stats():
    """Build and save venue statistics for all formats."""
    print("=" * 60)
    print("VENUE STATISTICS BUILDER")
    print("=" * 60)
    
    for format_type in ['T20', 'ODI']:
        for gender in ['male', 'female']:
            print(f"\n--- {format_type} {gender.upper()} ---")
            
            builder = VenueStatsBuilder(format_type, gender)
            builder.build_from_database()
            builder.save()
            
            # Print summary
            if builder.venue_stats:
                avg_scoring_factor = np.mean([
                    v['scoring_factor'] for v in builder.venue_stats.values()
                ])
                avg_boundary_rate = np.mean([
                    v['boundary_rate'] for v in builder.venue_stats.values()
                ])
                avg_wicket_rate = np.mean([
                    v['wicket_rate'] for v in builder.venue_stats.values()
                ])
                
                print(f"  Venues: {len(builder.venue_stats)}")
                print(f"  Global avg score: {builder.global_avg_score:.1f}")
                print(f"  Avg boundary rate: {avg_boundary_rate:.1%}")
                print(f"  Avg wicket rate: {avg_wicket_rate:.2%}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    build_venue_stats()

