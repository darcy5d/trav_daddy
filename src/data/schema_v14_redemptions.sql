-- Gasless Polymarket winner redemption audit.
-- One condition can be claimed once; attempts are retained for operations
-- visibility and to make repeated cron runs idempotent.

CREATE TABLE IF NOT EXISTS redemption_attempts (
    redemption_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_id        TEXT NOT NULL,
    token_id            TEXT,
    market_id           TEXT,
    title               TEXT,
    neg_risk            INTEGER NOT NULL DEFAULT 0,
    estimated_value_usdc REAL NOT NULL DEFAULT 0,
    status              TEXT NOT NULL CHECK (status IN ('planned', 'submitted', 'confirmed', 'failed', 'skipped')),
    transaction_hash    TEXT,
    error_message       TEXT,
    created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_redemption_attempts_condition_active
ON redemption_attempts(condition_id)
WHERE status IN ('planned', 'submitted', 'confirmed');

CREATE INDEX IF NOT EXISTS idx_redemption_attempts_status
ON redemption_attempts(status, created_at DESC);
