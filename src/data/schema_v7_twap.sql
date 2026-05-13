-- Wave 5.9: TWAP order execution tables.
--
-- order_plans: one row per TWAP execution plan (maps 1:1 to a bet_ledger row).
-- order_chunks: individual limit-order chunks within a plan.

CREATE TABLE IF NOT EXISTS order_plans (
    plan_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    bet_ledger_id        INTEGER REFERENCES bet_ledger(bet_id),
    fixture_key          TEXT NOT NULL,
    strategy_label       TEXT NOT NULL,
    token_id             TEXT NOT NULL,
    side                 TEXT NOT NULL,           -- BUY or SELL
    total_size_usdc      REAL NOT NULL,
    chunk_size_usdc      REAL NOT NULL,
    chunks_total         INTEGER NOT NULL,
    chunks_placed        INTEGER DEFAULT 0,
    chunks_filled        INTEGER DEFAULT 0,
    filled_size_usdc     REAL DEFAULT 0,
    avg_fill_price       REAL,
    max_acceptable_price REAL NOT NULL,           -- model_prob - min_edge; never pay above this
    base_price           REAL NOT NULL,           -- best_bid at plan creation (escalation start)
    price_step_pp        REAL DEFAULT 2.0,
    kickoff_at           TEXT,
    model_prob           REAL,
    market_price_at_plan REAL,
    status               TEXT DEFAULT 'pending',  -- pending | executing | completed | cancelled
    created_at           TEXT NOT NULL,
    updated_at           TEXT
);

CREATE TABLE IF NOT EXISTS order_chunks (
    chunk_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id              INTEGER NOT NULL REFERENCES order_plans(plan_id),
    chunk_index          INTEGER NOT NULL,
    limit_price          REAL NOT NULL,
    size_usdc            REAL NOT NULL,
    size_shares          REAL,                    -- USDC / limit_price (shares to buy)
    polymarket_order_id  TEXT,
    status               TEXT DEFAULT 'pending',  -- pending | placed | filled | partially_filled | cancelled
    placed_at            TEXT,
    filled_at            TEXT,
    fill_price           REAL,
    fill_size_usdc       REAL
);

CREATE INDEX IF NOT EXISTS idx_order_plans_status ON order_plans(status);
CREATE INDEX IF NOT EXISTS idx_order_plans_fixture ON order_plans(fixture_key);
CREATE INDEX IF NOT EXISTS idx_order_chunks_plan ON order_chunks(plan_id);
CREATE INDEX IF NOT EXISTS idx_order_chunks_status ON order_chunks(status);
CREATE INDEX IF NOT EXISTS idx_order_chunks_order_id ON order_chunks(polymarket_order_id);
