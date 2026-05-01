# Paper Trading - Forward-Test Logging (Wave 5.7)

Logs model predictions vs Polymarket prices on upcoming cricket fixtures
with a fake $1000 bankroll per strategy. No real funds at risk.

## What it does

For each upcoming Polymarket cricket fixture (next ~96h), the scanner:
1. Maps the team labels to DB team_ids
2. Runs V2 + V3 Monte Carlo sims using each team's recent XI
3. For each enabled strategy, checks the moneyline edge vs market price
4. If the strategy's filters pass, logs a paper bet with status='filled' (no
   CLOB call) at the observed market price

Settlements are reconciled by querying Polymarket's `/prices-history` for
each open paper bet and detecting when the outcome token resolves to 0.0
or 1.0. P&L is computed using a 2% taker-fee assumption and added back to
the strategy's bankroll.

## Strategies (each starts at $1000 paper bankroll)

| name              | model    | tournaments        | edge   | sizing       |
|-------------------|----------|--------------------|--------|--------------|
| `v2_odi_3pp`      | V2       | ODI men only       | 3pp+   | half-Kelly   |
| `v2_any_5pp`      | V2       | any                | 5pp+   | half-Kelly   |
| `v3_marg_3pp`     | V3 marg. | any                | 3pp+   | half-Kelly   |
| `consensus_5pp`   | V2 ∧ V3  | any (must agree)   | 5pp+   | half-Kelly   |
| `v2_diag_2pp`     | V2       | any (DIAGNOSTIC)   | 2pp+   | quarter-Kelly|

The `v2_odi_3pp` cell is the historical winner from Wave 5.6 (88% strike,
n=9). The others are forward-test hypotheses. See
`src/integrations/polymarket/paper_strategies.py` to add or tweak.

## Running

### Manual

One-shot scan + reconcile + report (most common):

```bash
venv311/bin/python scripts/paper_bet_daily.py --hours-ahead 96
```

Just scan (no reconcile):

```bash
venv311/bin/python scripts/paper_bet_scan.py --hours-ahead 96
```

Just reconcile settled markets:

```bash
venv311/bin/python scripts/paper_bet_reconcile.py
```

Dry-run (sim+log decisions, no DB writes):

```bash
venv311/bin/python scripts/paper_bet_scan.py --hours-ahead 96 --dry-run
```

Daily reports persist to `data/paper_trading/daily_reports/*.json`.

### Automatic (cron)

Recommended hourly cron entry (`crontab -e`):

```cron
# Paper trading: scan + reconcile every hour
0 * * * * cd /Users/darcy5d/Desktop/DD_AI_models/indias_dad && \
    venv311/bin/python scripts/paper_bet_daily.py --hours-ahead 96 \
    >> logs/paper_daily.log 2>&1
```

The scanner is **idempotent**: running every hour will not double-bet —
each (strategy, fixture, market, side) combination only places a bet once.

### UI

Browse to <http://127.0.0.1:5050/paper-trades> with the Flask app running:

```bash
venv311/bin/python -m flask --app app.main run --port 5050
```

The page auto-refreshes every 60s and has a "Run Scan + Reconcile Now"
button so you don't strictly need cron — leaving the Flask app open works
too, especially when watching tonight's matches develop.

## What the page shows

- **Totals row** — combined bankroll, P&L, ROI, open stakes across all 5 strategies
- **Strategy table** — per-strategy: bankroll, P&L, ROI, win rate, # bets, open stakes
- **Bankroll-over-time chart** — one line per strategy, drawn after each settled bet
- **Open Paper Bets** — bets still pending Polymarket settlement
- **Recent Settled** — last 50 settled bets with realised P&L and outcome

## Schema

Paper bets live in the existing `bet_ledger` table with:
- `bet_kind = 'paper'` (vs `'real'`)
- `strategy_label = 'v2_odi_3pp'` etc.
- `bankroll_at_proposal` — strategy bankroll at bet placement time
- `bankroll_after_settle` — populated on reconciliation

Real bets (mode='manual'/'auto', bet_kind='real') and paper bets coexist
in the same table; the dashboard filters on `bet_kind='paper'`.
