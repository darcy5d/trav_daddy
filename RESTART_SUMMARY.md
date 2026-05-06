# System Restart Summary
**Date:** May 6, 2026

## Services Running

✅ **Flask Web App** - Running on http://127.0.0.1:5001
- Paper Trades page: http://127.0.0.1:5001/paper-trades
- Live Betting page: http://127.0.0.1:5001/live-betting
- All other routes active

## Paper vs Live Betting Divergence Analysis

### Summary Statistics
- **Paper Bets:** 184 total
- **Live Bets:** 49 total  
- **Divergence:** 73.4% (Live placing FEWER bets)

### Key Findings

#### 1. Timeline
- **Paper trading started:** April 26, 2026
- **Live betting started:** May 3, 2026
- **Gap:** 7 days head start for paper

#### 2. Stake Sizing (By Design)
- **Paper:** $100 average stake per bet
- **Live:** $10-15 average stake per bet
- **Ratio:** Intentional 1/10 scale for risk management

#### 3. Per-Strategy Comparison

| Strategy | Paper Bets | Live Bets | Paper Win Rate | Live Win Rate | Paper P&L | Live P&L |
|----------|------------|-----------|----------------|---------------|-----------|----------|
| consensus_5pp | 37 | 10 | 65.7% | 25.0% | +$2,056 | -$25 |
| v2_any_5pp | 40 | 12 | 57.9% | 16.7% | +$1,674 | -$49 |
| v2_diag_2pp | 50 | 12 | 62.2% | 20.0% | +$697 | -$27 |
| v2_odi_3pp | 4 | 1 | 25.0% | 0.0% | -$184 | -$12 |
| v3_marg_3pp | 53 | 14 | 62.0% | 50.0% | +$1,747 | -$33 |

⚠️ **Note:** Live sample sizes are very small (4-6 settled bets per strategy), so win rates are not yet statistically significant.

#### 4. Fixture Coverage
- **Fixtures with BOTH paper and live bets:** 13
- **Fixtures with PAPER ONLY:** 24  
- **Fixtures with LIVE ONLY:** 0

Paper-only fixtures occurred before live betting started (April 26 - May 2).

#### 5. Recent Daily Bet Counts

| Date | Paper | Live | Difference |
|------|-------|------|------------|
| 2026-05-05 | 21 | 15 | -6 |
| 2026-05-04 | 11 | 15 | +4 |
| 2026-05-03 | 18 | 19 | +1 |
| 2026-05-02 | 40 | 0 | -40 (live not started) |
| 2026-05-01 | 23 | 0 | -23 (live not started) |

### Reasons for Divergence

1. **Late Start:** Live betting began 7 days after paper trading
2. **Risk Gates:** Live has stricter caps:
   - Per-strategy bankroll cap: $100 (vs $1,000 paper)
   - Per-bet cap: $10 (vs $100 paper)
   - Daily stake cap: $500 total
   - Daily loss cap: $50
3. **Stake Scaling:** Intentional 1/10 scale means some paper opportunities fall below live's $1 Polymarket minimum
4. **Win Rate Divergence:** Early live results show lower win rates, but sample sizes are too small to draw conclusions (4-6 settled bets per strategy)

### Conclusion

✅ **By Design:** The divergence is mostly intentional. Live betting uses much stricter risk controls and smaller stakes by design.

⚠️ **Monitoring Needed:** The low win rates in live (16-50%) vs paper (57-66%) deserve attention once sample sizes grow. Current n=4-6 settled bets per strategy is too small to be meaningful.

## Changes Made

### 1. Added Bankroll Chart to Live Betting Page

**New API Endpoint:** `/api/betting/bankroll-history`
- Returns cumulative bankroll timeline per strategy
- Filters for `bet_kind='real'` (live bets only)
- Mirrors the paper trading endpoint structure

**UI Changes:** `app/templates/live_betting.html`
- Added canvas chart section (similar to paper trades)
- JavaScript fetches and renders bankroll over time
- Chart shows per-strategy cumulative P&L
- Auto-refreshes every 60 seconds

**Styling:** Chart matches paper trades design:
- Line graph with dots for settled bets
- Color-coded per strategy
- Reference line at starting bankroll ($100)
- Time-based X-axis with 12-hour buckets
- Legend showing current bankroll per strategy

### 2. Created Analysis Script

**New File:** `scripts/analyze_paper_vs_live.py`
- Comprehensive divergence analysis
- Strategy-level comparisons
- Fixture-level coverage analysis
- Daily bet count timeline
- Identifies mismatches and patterns

**Usage:**
```bash
venv311/bin/python scripts/analyze_paper_vs_live.py
```

## Next Steps (Recommendations)

1. **Monitor Win Rates:** Track live performance as sample sizes grow. If divergence persists at n>30, investigate:
   - Are live bets hitting worse market prices due to timing?
   - Is the risk gate rejecting the best edges?
   - Are paper settlements more favorable than reality?

2. **Daily Monitoring:** Run the analysis script daily to track convergence:
   ```bash
   venv311/bin/python scripts/analyze_paper_vs_live.py
   ```

3. **Consider Adjustments** (if live continues underperforming):
   - Review risk gate thresholds
   - Compare paper vs live market prices at placement time
   - Check if $1 Polymarket minimum is filtering out too many small +EV bets

4. **Wait for Statistical Significance:** Don't make changes until each strategy has at least 30-50 settled live bets.

## Quick Reference

### Start/Stop Services

**Start Flask App:**
```bash
cd /Users/darcy5d/Desktop/DD_AI_models/indias_dad
venv311/bin/python app/main.py
```

**Stop Flask App:**
```bash
pkill -f "venv311/bin/python app/main.py"
```

### Key URLs
- **Main App:** http://127.0.0.1:5001
- **Paper Trades:** http://127.0.0.1:5001/paper-trades
- **Live Betting:** http://127.0.0.1:5001/live-betting (now with bankroll chart!)
- **Bulk Predict:** http://127.0.0.1:5001/bulk-predict

### Environment
- Python: 3.11 (via venv311)
- Database: `/Users/darcy5d/Desktop/DD_AI_models/indias_dad/cricket.db`
- Config: `.env` file in project root
- Mode: BETTING_MODE=AUTO (in .env)
