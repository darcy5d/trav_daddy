-- XI-aware re-evaluation: rolling per-scan model view for live fixtures.
--
-- Every live scan upserts the latest re-simulated model probability, edge,
-- chosen side, market price and XI signature for each (strategy, fixture).
-- This lets the rebalancer and the UI compare the CURRENT model view against
-- the values stored on the bet that was actually placed (bet_ledger), so we
-- can detect when an updated CREX lineup has moved our edge.
--
-- Primary key: (strategy_label, fixture_key) - the chosen side can flip
-- between scans, so side_label is a column, not part of the key.

CREATE TABLE IF NOT EXISTS live_model_snapshots (
    strategy_label  TEXT    NOT NULL,
    fixture_key     TEXT    NOT NULL,
    side_label      TEXT,              -- model's currently-preferred BACK side
    model_prob      REAL,              -- P(side wins) from the latest sim
    market_price    REAL,              -- side's market price at the scan
    edge_pp         REAL,              -- (model_prob - market_price) * 100
    xi_signature    TEXT,              -- lineup hash used for the latest sim
    model_version   TEXT,              -- v2 / v3 / consensus
    kickoff_at      TEXT,              -- ISO8601 UTC, for context
    last_resim_at   TEXT    NOT NULL,  -- ISO8601 UTC of this snapshot
    PRIMARY KEY (strategy_label, fixture_key)
);

CREATE INDEX IF NOT EXISTS idx_live_model_snapshots_fixture
    ON live_model_snapshots(fixture_key);
