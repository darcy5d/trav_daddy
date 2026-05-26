-- Wave 5.13: order_history — permanent audit table for every Polymarket order
-- id we have ever posted. Linked to bet_ledger / order_chunks / order_plans.
--
-- Why it exists
--   _reprice_placed_chunks and _resize_plan_to_bankroll overwrite or NULL the
--   live order_chunks.polymarket_order_id. When a CLOB fill arrives on the
--   old order id we have no way to attribute it back to the original bet
--   unless we keep an append-only history of every id we ever placed.
--
--   The reconcile loop uses order_history as the source-of-truth set of
--   "known" order ids so stray maker fills get matched to the right bet
--   instead of inserted as RECONCILE_GHOST rows.
--
-- Columns
--   final_status: placed | filled | cancelled | reprice_replaced | error
--   final_reason: free-text tag describing why the order ended in that state
--                 (kickoff_cancel | reprice | resize | never_posted |
--                  order_version_mismatch | unfillable | ...)
--   replaced_by_order_id: when reprice posts a new order, the old row is
--                         marked reprice_replaced and this field points at
--                         the new id. Reconcile follows the chain to find
--                         the still-live chunk.

CREATE TABLE IF NOT EXISTS order_history (
    history_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    polymarket_order_id  TEXT NOT NULL UNIQUE,
    bet_id               INTEGER REFERENCES bet_ledger(bet_id),
    chunk_id             INTEGER REFERENCES order_chunks(chunk_id),
    plan_id              INTEGER REFERENCES order_plans(plan_id),
    token_id             TEXT NOT NULL,
    side                 TEXT NOT NULL,            -- BUY / SELL
    order_kind           TEXT NOT NULL,            -- fok | twap_chunk | cashout
    limit_price          REAL,
    size_usdc            REAL,
    size_shares          REAL,
    posted_at            TEXT NOT NULL,
    final_status         TEXT NOT NULL,
    final_reason         TEXT,
    replaced_by_order_id TEXT,
    fill_usdc            REAL DEFAULT 0,
    fill_price           REAL,
    filled_at            TEXT,
    last_seen_at         TEXT,
    created_at           TEXT NOT NULL,
    updated_at           TEXT
);

CREATE INDEX IF NOT EXISTS idx_order_history_oid ON order_history(polymarket_order_id);
CREATE INDEX IF NOT EXISTS idx_order_history_bet ON order_history(bet_id);
CREATE INDEX IF NOT EXISTS idx_order_history_chunk ON order_history(chunk_id);
CREATE INDEX IF NOT EXISTS idx_order_history_status ON order_history(final_status);
CREATE INDEX IF NOT EXISTS idx_order_history_replaced_by
    ON order_history(replaced_by_order_id)
    WHERE replaced_by_order_id IS NOT NULL;
