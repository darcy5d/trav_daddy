#!/usr/bin/env python3
"""Analyze whether paper trades and live betting have diverged in betting patterns."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_connection
from datetime import datetime, timezone
import json

def analyze_divergence():
    """Compare paper vs live betting patterns."""
    with get_connection() as conn:
        cur = conn.cursor()
        
        # Find the timestamp of the first live bet
        cur.execute("""
            SELECT MIN(proposed_at) as first_live_bet
            FROM bet_ledger
            WHERE COALESCE(bet_kind, 'real') = 'real'
        """)
        first_live_row = cur.fetchone()
        first_live_bet_time = first_live_row[0] if first_live_row and first_live_row[0] else None
        
        if not first_live_bet_time:
            print("\n⚠️  No live bets found in the database. Cannot perform comparison.\n")
            return
        
        print(f"\n📅 Comparison Period: From first live bet at {first_live_bet_time}")
        print("    (Filtering both paper and live to this start time for apples-to-apples comparison)\n")
        
        # Get paper and live bets from the same time period
        cur.execute("""
            SELECT 
                bet_kind,
                strategy_label,
                COUNT(*) as n_bets,
                AVG(edge_pp) as avg_edge,
                AVG(model_prob) as avg_model_prob,
                AVG(market_price_at_proposal) as avg_market_price,
                AVG(size_usdc) as avg_stake,
                SUM(CASE WHEN status = 'settled' THEN 1 ELSE 0 END) as n_settled,
                SUM(CASE WHEN status = 'settled' AND settle_outcome = 1 THEN 1 ELSE 0 END) as n_wins,
                SUM(CASE WHEN status = 'settled' THEN pnl_realised_usdc ELSE 0 END) as total_pnl
            FROM bet_ledger
            WHERE bet_kind IN ('paper', 'real')
              AND proposed_at >= ?
            GROUP BY bet_kind, strategy_label
            ORDER BY bet_kind, strategy_label
        """, (first_live_bet_time,))
        
        results = cur.fetchall()
        
        paper_stats = {}
        live_stats = {}
        
        for row in results:
            kind = row[0]
            strat = row[1]
            stats = {
                'n_bets': row[2],
                'avg_edge': float(row[3]) if row[3] else 0,
                'avg_model_prob': float(row[4]) if row[4] else 0,
                'avg_market_price': float(row[5]) if row[5] else 0,
                'avg_stake': float(row[6]) if row[6] else 0,
                'n_settled': row[7],
                'n_wins': row[8],
                'total_pnl': float(row[9]) if row[9] else 0,
                'win_rate': float(row[8]) / float(row[7]) if row[7] > 0 else None,
            }
            
            if kind == 'paper':
                paper_stats[strat] = stats
            else:
                live_stats[strat] = stats
        
        # Get fixture-level comparison (same time filter)
        cur.execute("""
            SELECT 
                fixture_key,
                bet_kind,
                COUNT(*) as n_bets,
                GROUP_CONCAT(strategy_label, ', ') as strategies
            FROM bet_ledger
            WHERE bet_kind IN ('paper', 'real')
              AND proposed_at >= ?
            GROUP BY fixture_key, bet_kind
            ORDER BY fixture_key
        """, (first_live_bet_time,))
        
        fixture_results = cur.fetchall()
        fixtures_both = set()
        fixtures_paper_only = set()
        fixtures_live_only = set()
        
        fixture_map = {}
        for row in fixture_results:
            fk = row[0]
            kind = row[1]
            if fk not in fixture_map:
                fixture_map[fk] = {'paper': 0, 'real': 0}
            fixture_map[fk][kind] = row[2]
        
        for fk, counts in fixture_map.items():
            if counts['paper'] > 0 and counts['real'] > 0:
                fixtures_both.add(fk)
            elif counts['paper'] > 0:
                fixtures_paper_only.add(fk)
            else:
                fixtures_live_only.add(fk)
        
        # Print analysis
        print("\n" + "=" * 80)
        print("PAPER vs LIVE BETTING DIVERGENCE ANALYSIS")
        print("=" * 80)
        
        print("\n📊 STRATEGY-LEVEL COMPARISON")
        print("-" * 80)
        
        all_strategies = sorted(set(list(paper_stats.keys()) + list(live_stats.keys())))
        
        for strat in all_strategies:
            paper = paper_stats.get(strat)
            live = live_stats.get(strat)
            
            print(f"\n  Strategy: {strat}")
            print(f"  {'':40s} {'Paper':>15s} {'Live':>15s} {'Diff':>12s}")
            print(f"  {'-'*40} {'-'*15} {'-'*15} {'-'*12}")
            
            if paper:
                print(f"  {'Total bets':40s} {paper['n_bets']:>15d} {live['n_bets'] if live else 0:>15d} {(live['n_bets'] if live else 0) - paper['n_bets']:>12d}")
                print(f"  {'Avg edge (pp)':40s} {paper['avg_edge']:>15.2f} {live['avg_edge'] if live else 0:>15.2f} {(live['avg_edge'] if live else 0) - paper['avg_edge']:>12.2f}")
                print(f"  {'Avg model prob':40s} {paper['avg_model_prob']:>15.3f} {live['avg_model_prob'] if live else 0:>15.3f} {(live['avg_model_prob'] if live else 0) - paper['avg_model_prob']:>12.3f}")
                print(f"  {'Avg market price':40s} {paper['avg_market_price']:>15.3f} {live['avg_market_price'] if live else 0:>15.3f} {(live['avg_market_price'] if live else 0) - paper['avg_market_price']:>12.3f}")
                print(f"  {'Avg stake ($)':40s} {paper['avg_stake']:>15.2f} {live['avg_stake'] if live else 0:>15.2f} {(live['avg_stake'] if live else 0) - paper['avg_stake']:>12.2f}")
                print(f"  {'Settled bets':40s} {paper['n_settled']:>15d} {live['n_settled'] if live else 0:>15d} {(live['n_settled'] if live else 0) - paper['n_settled']:>12d}")
                if paper['win_rate'] is not None:
                    print(f"  {'Win rate':40s} {paper['win_rate']*100:>14.1f}% {(live['win_rate']*100) if (live and live['win_rate'] is not None) else 0:>14.1f}% {((live['win_rate'] if (live and live['win_rate']) else 0) - paper['win_rate'])*100:>11.1f}pp")
                print(f"  {'Total P&L ($)':40s} {paper['total_pnl']:>15.2f} {live['total_pnl'] if live else 0:>15.2f} {(live['total_pnl'] if live else 0) - paper['total_pnl']:>12.2f}")
            elif live:
                print(f"  ⚠️  Strategy exists in LIVE but not in PAPER")
                print(f"  {'Total bets':40s} {'N/A':>15s} {live['n_bets']:>15d}")
        
        print("\n\n📅 FIXTURE-LEVEL COVERAGE")
        print("-" * 80)
        print(f"  Fixtures with BOTH paper and live bets:     {len(fixtures_both):>5d}")
        print(f"  Fixtures with PAPER ONLY:                   {len(fixtures_paper_only):>5d}")
        print(f"  Fixtures with LIVE ONLY:                    {len(fixtures_live_only):>5d}")
        
        # Sample mismatches
        if fixtures_paper_only:
            print(f"\n  Sample paper-only fixtures (first 5):")
            for fk in sorted(fixtures_paper_only)[:5]:
                print(f"    • {fk} ({fixture_map[fk]['paper']} paper bets)")
        
        if fixtures_live_only:
            print(f"\n  Sample live-only fixtures (first 5):")
            for fk in sorted(fixtures_live_only)[:5]:
                print(f"    • {fk} ({fixture_map[fk]['real']} live bets)")
        
        # Check same fixtures, different counts
        print("\n\n🔍 FIXTURES WITH DIVERGENT BET COUNTS")
        print("-" * 80)
        divergent = []
        for fk in sorted(fixtures_both):
            paper_count = fixture_map[fk]['paper']
            live_count = fixture_map[fk]['real']
            if paper_count != live_count:
                divergent.append((fk, paper_count, live_count))
        
        if divergent:
            print(f"  Found {len(divergent)} fixtures where paper and live bet counts differ:\n")
            for fk, paper_count, live_count in divergent[:10]:
                diff = live_count - paper_count
                print(f"    • {fk:40s}  paper: {paper_count:3d}  live: {live_count:3d}  (Δ {diff:+3d})")
            if len(divergent) > 10:
                print(f"\n    ... and {len(divergent) - 10} more")
        else:
            print("  ✅ All shared fixtures have matching bet counts!")
        
        # Time-based analysis (same time filter)
        cur.execute("""
            SELECT 
                DATE(proposed_at) as bet_date,
                bet_kind,
                COUNT(*) as n_bets,
                AVG(edge_pp) as avg_edge
            FROM bet_ledger
            WHERE bet_kind IN ('paper', 'real')
              AND proposed_at >= ?
            GROUP BY bet_date, bet_kind
            ORDER BY bet_date DESC
            LIMIT 30
        """, (first_live_bet_time,))
        
        timeline = cur.fetchall()
        print("\n\n📈 RECENT DAILY BET COUNTS")
        print("-" * 80)
        print(f"  {'Date':12s} {'Paper':>10s} {'Live':>10s} {'Diff':>10s}")
        print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
        
        daily_map = {}
        for row in timeline:
            date = row[0]
            kind = row[1]
            count = row[2]
            if date not in daily_map:
                daily_map[date] = {'paper': 0, 'real': 0}
            daily_map[date][kind] = count
        
        for date in sorted(daily_map.keys(), reverse=True)[:14]:
            paper_count = daily_map[date]['paper']
            live_count = daily_map[date]['real']
            diff = live_count - paper_count
            print(f"  {date:12s} {paper_count:>10d} {live_count:>10d} {diff:>+10d}")
        
        print("\n\n💡 SUMMARY")
        print("-" * 80)
        
        total_paper = sum(s['n_bets'] for s in paper_stats.values())
        total_live = sum(s['n_bets'] for s in live_stats.values())
        
        if total_paper == 0 and total_live == 0:
            print("  ⚠️  No bets found in either paper or live systems.")
        elif total_live == 0:
            print(f"  ⚠️  Paper trading active ({total_paper} bets) but NO live bets yet.")
            print("      Live betting may be OFF or just getting started.")
        elif abs(total_paper - total_live) < 3:
            print(f"  ✅ Paper and live are tracking closely: {total_paper} paper vs {total_live} live")
        else:
            pct_diff = abs(total_paper - total_live) / max(total_paper, 1) * 100
            print(f"  ⚠️  Divergence detected: {total_paper} paper bets vs {total_live} live bets ({pct_diff:.1f}% difference)")
            if total_live < total_paper:
                print("      → Live is placing FEWER bets than paper (risk gate or filters may be stricter)")
            else:
                print("      → Live is placing MORE bets than paper (unusual; check config)")
        
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    analyze_divergence()
