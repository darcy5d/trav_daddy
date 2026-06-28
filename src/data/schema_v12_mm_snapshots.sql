-- Wave 6 pre-work (W2): order-book spread / depth / reward reconnaissance.
--
-- We currently store ZERO historical order-book data: spreads and reward
-- params are only ever read ephemerally at decision time. Before committing to
-- a market-making build we need an empirical time-series of:
--   * how wide cricket spreads actually are (per market type, over the
--     pre-match -> toss -> in-play window),
--   * where any non-toxic flow / depth exists,
--   * whether cricket is in Polymarket's liquidity-reward set at all, and at
--     what rebate / band if so.
--
-- `scripts/scan_mm_markets.py` polls live fixtures and appends one row per
-- (token, poll). This is an append-only log; analysis groups by token + time.

CREATE TABLE IF NOT EXISTS mm_market_snapshots (
    snapshot_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                 TEXT    NOT NULL,   -- ISO8601 UTC of this poll
    fixture_key        TEXT,               -- "{prefix}-{t1}-{t2}-{date}"
    tournament_prefix  TEXT,
    format             TEXT,               -- T20 / ODI / ...
    gender             TEXT,               -- male / female
    market_type        TEXT,               -- moneyline / top_batter / most_sixes / toss_match_double / ...
    market_id          TEXT,
    token_id           TEXT,
    outcome_label      TEXT,
    best_bid           REAL,
    best_ask           REAL,
    spread_pp          REAL,               -- (ask - bid) * 100
    best_bid_size      REAL,               -- shares at best bid
    best_ask_size      REAL,               -- shares at best ask
    midpoint           REAL,
    last_price         REAL,               -- Gamma outcomePrice at poll time
    volume_num         REAL,               -- Gamma market volume (if present)
    kickoff_at         TEXT,               -- ISO8601 UTC scheduled start estimate
    hours_to_kickoff   REAL,
    -- Reward / fee reconnaissance (best-effort; NULL when API doesn't expose it)
    in_reward_set      INTEGER,            -- 1 / 0 / NULL (unknown)
    reward_min_size    REAL,               -- min_incentive_size if found
    reward_max_spread  REAL,               -- max_incentive_spread if found
    reward_json        TEXT,               -- raw reward params blob
    fee_schedule_json  TEXT,               -- raw Gamma feeSchedule blob
    created_at         TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_mm_snapshots_fixture_ts
    ON mm_market_snapshots(fixture_key, ts);
CREATE INDEX IF NOT EXISTS idx_mm_snapshots_token_ts
    ON mm_market_snapshots(token_id, ts);
CREATE INDEX IF NOT EXISTS idx_mm_snapshots_market_type
    ON mm_market_snapshots(market_type);
