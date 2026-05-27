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

Recommended hourly cron entry (`crontab -e`). Cron does not expand shell
variables — replace `<REPO_ROOT>` with the absolute path of your repo clone
before installing.

```cron
# Paper trading: scan + reconcile every hour
0 * * * * cd <REPO_ROOT> && \
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

## In-game cashout (Wave 5.10)

Paper bets participate in the same tiered cashout system as real bets. When
the cron scanner fires, it checks all `status='filled'` rows (paper and real)
and simulates a SELL for paper bets — no actual CLOB call is made, but the
`bet_ledger` row is updated with:

- `cashout_triggered_at` — timestamp the threshold was met
- `cashout_price` — the CLOB midpoint at the time
- `cashout_pnl_usdc` — realised P&L (net of 2% taker fee)
- `cashout_threshold_used` — the ratio that fired (e.g. 1.25x)
- `status = 'settled'` — marked immediately so reconciler skips it

**Thresholds are tiered by entry price** (see README for full breakdown):

| Entry bucket | Threshold |
|---|---|
| Heavy underdog 5–20¢ | 1.30x |
| Underdog 20–35¢ | 1.20x |
| Slight underdog 35–50¢ | 1.25x |
| Coin flip + above 50¢ | hold |

The threshold logic lives in `src/integrations/polymarket/cashout.py` →
`tiered_cashout_threshold(fill_price)`. Strategy-level overrides
(`PaperStrategy.cashout_return_threshold`) are supported but default to
`None`, meaning all strategies use the tiered lookup.

**Querying cashed-out paper bets:**

```sql
SELECT bet_id, side_label, fill_price, cashout_price,
       cashout_threshold_used, cashout_pnl_usdc
FROM bet_ledger
WHERE bet_kind = 'paper'
  AND cashout_triggered_at IS NOT NULL
ORDER BY cashout_triggered_at DESC;
```

## Schema

Paper bets live in the existing `bet_ledger` table with:
- `bet_kind = 'paper'` (vs `'real'`)
- `strategy_label = 'v2_odi_3pp'` etc.
- `bankroll_at_proposal` — strategy bankroll at bet placement time
- `bankroll_after_settle` — populated on reconciliation

Real bets (mode='manual'/'auto', bet_kind='real') and paper bets coexist
in the same table; the dashboard filters on `bet_kind='paper'`.
