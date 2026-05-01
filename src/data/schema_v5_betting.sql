-- Wave 5 Phase 6b: Live betting bet ledger schema.
--
-- Records every proposed/placed/filled/settled bet on Polymarket with full
-- state transitions for audit. The schema is intentionally append-mostly:
-- each state change updates timestamps + status but keeps the original
-- proposed values immutable so we can reconstruct what the model thought
-- at decision time.

CREATE TABLE IF NOT EXISTS bet_ledger (
    bet_id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Lifecycle timestamps (ISO8601 UTC). NULL until that state is reached.
    proposed_at TEXT NOT NULL,           -- model + risk gate said "go"
    placed_at TEXT,                      -- order successfully POSTed to CLOB
    filled_at TEXT,                      -- exchange confirmed fill (full or partial)
    cancelled_at TEXT,                   -- if cancelled before fill
    settled_at TEXT,                     -- after match completion + reconciliation

    -- Match + market reference
    match_id INTEGER,                    -- our internal cricket.db match_id (nullable for non-matched)
    fixture_key TEXT NOT NULL,           -- "team1_team2_yyyy-mm-ddTHH:MM:SS" deterministic
    market_type TEXT NOT NULL,           -- moneyline / top_batter / most_sixes / toss_match_double

    -- Polymarket pointers
    polymarket_market_id TEXT,           -- Gamma marketId / conditionId
    polymarket_token_id TEXT,            -- ERC1155 outcome token
    polymarket_order_id TEXT,            -- CLOB orderID returned on placement

    -- Decision context (frozen at proposal time; do NOT update later)
    side_label TEXT,                     -- e.g. "Mumbai Indians" or "Virat Kohli top batter"
    model_prob REAL NOT NULL,            -- model's P(this outcome happens) at proposal
    market_price_at_proposal REAL NOT NULL,  -- Polymarket implied price [0,1] at proposal
    edge_pp REAL NOT NULL,               -- (model_prob - market_price_at_proposal) * 100

    -- Order details
    side TEXT NOT NULL,                  -- BUY / SELL
    size_usdc REAL NOT NULL,             -- USD-equivalent stake at proposal
    fees_estimated_usdc REAL,            -- 2% taker fee estimate

    -- Fill details (populated on placement / fill)
    fill_price REAL,                     -- realised average price across partials
    fill_size_usdc REAL,                 -- USD equivalent actually filled

    -- Settlement (populated on reconciliation)
    settle_outcome INTEGER,              -- 0 or 1 (binary outcome resolution)
    pnl_realised_usdc REAL,              -- net of fees, signed (negative = loss)

    -- State machine
    status TEXT NOT NULL,                -- proposed / placed / filled / cancelled / settled / errored
    mode TEXT NOT NULL,                  -- manual / auto

    error_message TEXT,                  -- human-readable error if status=errored

    CHECK (mode IN ('manual', 'auto')),
    CHECK (status IN ('proposed', 'placed', 'filled', 'cancelled', 'settled', 'errored'))
);

CREATE INDEX IF NOT EXISTS idx_bet_ledger_status ON bet_ledger(status);
CREATE INDEX IF NOT EXISTS idx_bet_ledger_match ON bet_ledger(match_id);
CREATE INDEX IF NOT EXISTS idx_bet_ledger_settled_at ON bet_ledger(settled_at);
CREATE INDEX IF NOT EXISTS idx_bet_ledger_proposed_at ON bet_ledger(proposed_at);
CREATE INDEX IF NOT EXISTS idx_bet_ledger_market_type ON bet_ledger(market_type);
