-- Wave 5.7: paper-trade extensions for bet_ledger.
--
-- Adds four new columns:
--   bet_kind                   - 'real' (default) or 'paper'. Drives whether
--                                placement actually calls Polymarket CLOB.
--   strategy_label             - human-readable strategy name. Multiple paper
--                                strategies can run in parallel each tracking
--                                their own bankroll via this label.
--   bankroll_at_proposal       - the strategy bankroll BEFORE this bet was placed.
--                                Used for compounding stake calculations and
--                                bankroll-over-time charts.
--   bankroll_after_settle      - bankroll AFTER pnl_realised_usdc is added in.
--                                Populated at reconciliation time.
--
-- All four columns are added via ALTER TABLE ADD COLUMN at app startup
-- (see init_betting_tables in src/data/database.py). The original CHECK
-- constraint on `mode` is left untouched - paper bets continue to use
-- mode='manual' or 'auto' for real bets (or the same for paper bets;
-- bet_kind is the differentiator).

-- Indexes for the new columns - idempotent CREATE IF NOT EXISTS.
CREATE INDEX IF NOT EXISTS idx_bet_ledger_kind ON bet_ledger(bet_kind);
CREATE INDEX IF NOT EXISTS idx_bet_ledger_strategy ON bet_ledger(strategy_label);
