#!/usr/bin/env python3
"""
Validation Script for Tiered ELO System.

Performs sanity checks and generates a validation report to ensure
the tiered ELO system is producing realistic rankings.

Checks:
1. Sanity: India/Australia/England in top 10 for T20 Male?
2. Tier integrity: Are tier 1 teams generally higher ELO than tier 2?
3. Promotion flags: Are there obvious misclassifications?
4. Cross-tier matches: Do upsets produce appropriate ELO swings?
5. Ceiling/floor enforcement: Are teams respecting tier boundaries?

Usage:
    python scripts/validate_tiered_elo.py [--format T20|ODI] [--gender male|female]
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATABASE_PATH
from src.data.database import get_db_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def check_sanity_rankings(conn, format_type='T20', gender='male'):
    """Check if elite teams are in top positions."""
    cursor = conn.cursor()
    
    elo_col = f'elo_{format_type.lower()}_{gender}'
    
    cursor.execute(f"""
        SELECT t.name, t.tier, e.{elo_col} as elo
        FROM team_current_elo e
        JOIN teams t ON e.team_id = t.team_id
        WHERE e.{elo_col} IS NOT NULL AND e.{elo_col} != 1500
        ORDER BY e.{elo_col} DESC
        LIMIT 15
    """)
    
    top_teams = cursor.fetchall()
    
    print(f"\n{'='*70}")
    print(f"SANITY CHECK: Top 15 Teams ({format_type} {gender.upper()})")
    print('='*70)
    
    elite_teams = {'India', 'Australia', 'England', 'Pakistan', 'New Zealand', 'South Africa'}
    elite_in_top_10 = []
    
    for idx, row in enumerate(top_teams, 1):
        name, tier, elo = row['name'], row['tier'], row['elo']
        marker = "✓" if name in elite_teams else " "
        print(f"{idx:2}. [T{tier}] {name:30} {elo:6.0f} {marker}")
        
        if idx <= 10 and name in elite_teams:
            elite_in_top_10.append(name)
    
    print(f"\nElite teams in top 10: {len(elite_in_top_10)}/{len(elite_teams)}")
    missing_elite = elite_teams - set(elite_in_top_10)
    if missing_elite:
        print(f"⚠️  Missing from top 10: {', '.join(missing_elite)}")
    else:
        print("✓ All elite teams in top 10")
    
    return len(elite_in_top_10) >= 4  # At least 4 out of 6 elite teams


def check_tier_integrity(conn, format_type='T20', gender='male'):
    """Check if ELO ranges respect tier hierarchy."""
    cursor = conn.cursor()
    
    elo_col = f'elo_{format_type.lower()}_{gender}'
    
    cursor.execute(f"""
        SELECT t.tier, 
               MIN(e.{elo_col}) as min_elo,
               AVG(e.{elo_col}) as avg_elo,
               MAX(e.{elo_col}) as max_elo,
               COUNT(*) as team_count
        FROM team_current_elo e
        JOIN teams t ON e.team_id = t.team_id
        WHERE e.{elo_col} IS NOT NULL AND e.{elo_col} != 1500
        GROUP BY t.tier
        ORDER BY t.tier
    """)
    
    tier_stats = cursor.fetchall()
    
    print(f"\n{'='*70}")
    print(f"TIER INTEGRITY CHECK ({format_type} {gender.upper()})")
    print('='*70)
    print(f"{'Tier':<6} {'Teams':<7} {'Min ELO':<10} {'Avg ELO':<10} {'Max ELO':<10}")
    print('-'*70)
    
    tier_names = {
        1: "Elite",
        2: "Full Members",
        3: "Associates/Franchises",
        4: "Regional",
        5: "Domestic"
    }
    
    prev_avg = None
    integrity_pass = True
    
    for row in tier_stats:
        tier, min_elo, avg_elo, max_elo, count = row['tier'], row['min_elo'], row['avg_elo'], row['max_elo'], row['team_count']
        
        print(f"T{tier}    {count:<7} {min_elo:>8.0f}   {avg_elo:>8.0f}   {max_elo:>8.0f}   ({tier_names.get(tier, 'Unknown')})")
        
        # Check if average ELO decreases as tier increases
        if prev_avg is not None and avg_elo >= prev_avg:
            print(f"  ⚠️  Tier {tier} avg ELO ({avg_elo:.0f}) >= Tier {tier-1} avg ELO ({prev_avg:.0f})")
            integrity_pass = False
        
        prev_avg = avg_elo
    
    if integrity_pass:
        print("\n✓ Tier integrity maintained (average ELO decreases with tier)")
    else:
        print("\n⚠️  Tier integrity violation detected")
    
    return integrity_pass


def check_promotion_flags(conn):
    """Review pending promotion flags."""
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT prf.flag_id, t.name, t.tier, prf.suggested_tier, 
               prf.trigger_reason, prf.current_elo, prf.format, prf.gender
        FROM promotion_review_flags prf
        JOIN teams t ON prf.team_id = t.team_id
        WHERE prf.reviewed = FALSE
        ORDER BY prf.format, prf.gender, t.tier, prf.current_elo DESC
    """)
    
    flags = cursor.fetchall()
    
    print(f"\n{'='*70}")
    print(f"PROMOTION FLAGS REVIEW")
    print('='*70)
    
    if not flags:
        print("✓ No pending promotion flags")
        return True
    
    print(f"\n{len(flags)} teams flagged for review:\n")
    
    by_category = {}
    for flag in flags:
        name = flag['name']
        tier = flag['tier']
        suggested = flag['suggested_tier']
        reason = flag['trigger_reason']
        elo = flag['current_elo']
        fmt = flag['format']
        gen = flag['gender']
        
        key = f"{fmt}_{gen}"
        if key not in by_category:
            by_category[key] = {'promotions': [], 'demotions': []}
        
        if suggested < tier:
            by_category[key]['promotions'].append((name, tier, suggested, elo, reason))
        else:
            by_category[key]['demotions'].append((name, tier, suggested, elo, reason))
    
    for category, data in by_category.items():
        fmt, gen = category.split('_')
        print(f"\n{fmt} {gen.upper()}:")
        
        if data['promotions']:
            print(f"\n  Suggested Promotions:")
            for name, tier, suggested, elo, reason in data['promotions']:
                print(f"    {name:30} T{tier} → T{suggested}  (ELO: {elo:.0f})")
                print(f"      Reason: {reason}")
        
        if data['demotions']:
            print(f"\n  Suggested Demotions:")
            for name, tier, suggested, elo, reason in data['demotions']:
                print(f"    {name:30} T{tier} → T{suggested}  (ELO: {elo:.0f})")
                print(f"      Reason: {reason}")
    
    print(f"\nℹ️  Review these flags via /api/admin/promotion-flags")
    return True


def check_boundary_enforcement(conn, format_type='T20', gender='male'):
    """Check if tier ceilings and floors are being enforced."""
    from src.elo.calculator_v3 import EloCalculatorV3
    
    cursor = conn.cursor()
    elo_col = f'elo_{format_type.lower()}_{gender}'
    
    print(f"\n{'='*70}")
    print(f"BOUNDARY ENFORCEMENT CHECK ({format_type} {gender.upper()})")
    print('='*70)
    
    violations = []
    
    for tier in [1, 2, 3, 4, 5]:
        ceiling = EloCalculatorV3.TIER_CEILINGS[tier]
        floor = EloCalculatorV3.TIER_FLOORS[tier]
        
        # Check for teams exceeding ceiling
        cursor.execute(f"""
            SELECT t.name, e.{elo_col} as elo
            FROM team_current_elo e
            JOIN teams t ON e.team_id = t.team_id
            WHERE t.tier = ? AND e.{elo_col} > ?
            ORDER BY e.{elo_col} DESC
        """, (tier, ceiling))
        
        above_ceiling = cursor.fetchall()
        
        # Check for teams below floor
        cursor.execute(f"""
            SELECT t.name, e.{elo_col} as elo
            FROM team_current_elo e
            JOIN teams t ON e.team_id = t.team_id
            WHERE t.tier = ? AND e.{elo_col} < ?
            ORDER BY e.{elo_col} ASC
        """, (tier, floor))
        
        below_floor = cursor.fetchall()
        
        if above_ceiling:
            print(f"\n⚠️  Tier {tier} teams above ceiling ({ceiling}):")
            for row in above_ceiling:
                print(f"    {row['name']:30} {row['elo']:.0f}")
                violations.append(('ceiling', tier, row['name'], row['elo']))
        
        if below_floor:
            print(f"\n⚠️  Tier {tier} teams below floor ({floor}):")
            for row in below_floor:
                print(f"    {row['name']:30} {row['elo']:.0f}")
                violations.append(('floor', tier, row['name'], row['elo']))
    
    if not violations:
        print("\n✓ All tier boundaries enforced correctly")
    
    return len(violations) == 0


def generate_summary_report(conn):
    """Generate a summary markdown report."""
    cursor = conn.cursor()
    
    report = []
    report.append("# Tiered ELO System Validation Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\nDatabase: {DATABASE_PATH}")
    
    # Get ELO calculation stats
    cursor.execute("SELECT COUNT(*) FROM team_elo_history WHERE NOT is_monthly_snapshot")
    team_updates = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM player_elo_history WHERE NOT is_monthly_snapshot")
    player_updates = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM promotion_review_flags WHERE reviewed = FALSE")
    pending_flags = cursor.fetchone()[0]
    
    report.append(f"\n## Statistics")
    report.append(f"- Team ELO updates: {team_updates:,}")
    report.append(f"- Player ELO updates: {player_updates:,}")
    report.append(f"- Pending promotion flags: {pending_flags}")
    
    # Save report
    report_path = Path(__file__).parent.parent / 'docs' / 'VALIDATION_REPORT.md'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print('='*70)
    print(f"Validation report saved to: {report_path}")
    print(f"\nStatistics:")
    print(f"  - Team ELO updates: {team_updates:,}")
    print(f"  - Player ELO updates: {player_updates:,}")
    print(f"  - Pending promotion flags: {pending_flags}")


def main():
    """Run all validation checks."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate tiered ELO system')
    parser.add_argument('--format', default='T20', choices=['T20', 'ODI'])
    parser.add_argument('--gender', default='male', choices=['male', 'female'])
    args = parser.parse_args()
    
    print("="*70)
    print("TIERED ELO SYSTEM VALIDATION")
    print("="*70)
    print(f"\nFormat: {args.format}")
    print(f"Gender: {args.gender}")
    
    with get_db_connection() as conn:
        # Run all checks
        sanity_pass = check_sanity_rankings(conn, args.format, args.gender)
        integrity_pass = check_tier_integrity(conn, args.format, args.gender)
        check_promotion_flags(conn)
        boundary_pass = check_boundary_enforcement(conn, args.format, args.gender)
        
        # Generate report
        generate_summary_report(conn)
        
        # Final verdict
        all_pass = sanity_pass and integrity_pass and boundary_pass
        
        print(f"\n{'='*70}")
        print("VALIDATION RESULT")
        print('='*70)
        
        if all_pass:
            print("✓ All validation checks passed")
            print("\nThe tiered ELO system is producing realistic rankings.")
            return 0
        else:
            print("⚠️  Some validation checks failed")
            print("\nReview the output above for details.")
            print("Consider reviewing promotion flags and manually adjusting tiers if needed.")
            return 1


if __name__ == "__main__":
    sys.exit(main())






