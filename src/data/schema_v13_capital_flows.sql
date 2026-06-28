-- Wave 6 follow-up: live wallet capital-flow ledger (deposits / withdrawals).
--
-- The bankroll math (src/integrations/polymarket/live_bankroll.py) already reads
-- the live portfolio value (wallet cash + open MTM + redeemable tokens). What it
-- CANNOT do without this table is tell apart "portfolio grew because I funded
-- the wallet" from "portfolio grew because we won bets" - so any ROI computed
-- from portfolio movement alone is wrong the moment money is added or removed.
--
-- This append-only ledger records every external capital movement. With it:
--   net_contributed = SUM(deposits) - SUM(withdrawals)
--   net_pnl         = current_portfolio_value - net_contributed
--   roi_on_capital  = net_pnl / net_contributed
-- i.e. true cash-on-cash return on the money actually put into the wallet,
-- invariant to deposits/withdrawals.

CREATE TABLE IF NOT EXISTS capital_flows (
    flow_id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                      TEXT    NOT NULL,            -- ISO8601 UTC of the flow
    flow_type               TEXT    NOT NULL CHECK (flow_type IN ('deposit','withdrawal')),
    amount_usdc             REAL    NOT NULL CHECK (amount_usdc > 0),
    -- Wallet/portfolio snapshot captured at record time (audit + reconciliation)
    wallet_cash_before      REAL,
    wallet_cash_after       REAL,
    portfolio_value_at_flow REAL,
    tx_hash                 TEXT,                          -- on-chain tx (optional)
    source                  TEXT    NOT NULL DEFAULT 'manual',  -- manual / onchain
    note                    TEXT,
    created_at              TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_capital_flows_ts ON capital_flows(ts);
CREATE INDEX IF NOT EXISTS idx_capital_flows_type ON capital_flows(flow_type);
