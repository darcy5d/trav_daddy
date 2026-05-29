-- Wave 5.10: in-game cashout tracking columns on bet_ledger.
--
-- A "cashout" is a SELL order placed while the match is still live,
-- locking in an unrealised gain (or limiting a loss) rather than
-- holding the position to settlement.
--
-- Four columns are added idempotently via ALTER TABLE ADD COLUMN in
-- src/data/database.py:init_cashout_columns(). The SQL here is for
-- documentation and for any external tools that read schema files directly.
--
-- Column semantics:
--
--   cashout_triggered_at   ISO8601 UTC when return_ratio >= threshold was
--                          first detected by the scanner. NULL = no cashout.
--
--   cashout_price          The CLOB midpoint / fill price at which the SELL
--                          order was executed (or simulated for paper bets).
--
--   cashout_pnl_usdc       Net signed P&L from the cashout:
--                              (shares * cashout_price) - fill_size_usdc - fee
--                          Equals pnl_realised_usdc for cashed-out rows.
--
--   cashout_threshold_used The return_ratio (cashout_price / fill_price) that
--                          triggered the cashout. Stored for audit + tuning.
--
--   cashout_reason         'profit' for a tiered profit-take SELL, 'stop' for a
--                          Wave 5.11 guarded stop-loss SELL. NULL on legacy rows.
--                          Lets us split recovered-loss vs profit-take when
--                          tuning. (return_ratio < 1 for a stop.)
--
-- Settled rows cashed out can be distinguished from naturally settled rows by:
--   WHERE cashout_triggered_at IS NOT NULL
--
-- Naturally settled rows have settle_outcome IN (0, 1); cashed-out rows
-- have settle_outcome = NULL (no binary resolution — sold before close).

-- Partial index: fast lookup of all cashed-out rows.
-- (SQLite supports WHERE clauses on indexes from version 3.8.0+.)
CREATE INDEX IF NOT EXISTS idx_bet_ledger_cashout
    ON bet_ledger(cashout_triggered_at)
    WHERE cashout_triggered_at IS NOT NULL;

-- Wave 5.12: direct link from bet_ledger row to the CLOB SELL order placed
-- during in-game cashout. Enables exact SELL attribution in reconcile.
CREATE INDEX IF NOT EXISTS idx_bet_ledger_cashout_order
    ON bet_ledger(cashout_order_id)
    WHERE cashout_order_id IS NOT NULL;
