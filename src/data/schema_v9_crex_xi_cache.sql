-- Wave 5.11: CREX XI cache for pre-match lineup data.
--
-- Stores the resolved player_ids (as a JSON array) fetched from the CREX
-- Playwright scraper for each upcoming fixture+team pair. Populated by
-- scripts/refresh_crex_xi.py (cron */2h). Read by get_cached_xi() in
-- paper_inputs.py; live_bet_scan.py and paper_bet_scan.py use this before
-- falling back to get_recent_xi() (last 3 historical matches).
--
-- Primary key: (fixture_key, team_id) - one row per team per fixture.
-- fetched_at: ISO8601 UTC timestamp of the scrape.
-- n_matched / n_input: how many of the 11 CREX names resolved to DB ids.
-- source: 'crex' always for now; reserved for future data sources.

CREATE TABLE IF NOT EXISTS crex_xi_cache (
    fixture_key   TEXT    NOT NULL,
    team_id       INTEGER NOT NULL,
    players_json  TEXT    NOT NULL,  -- JSON array of player_ids, e.g. [42,17,...]
    n_matched     INTEGER NOT NULL DEFAULT 0,
    n_input       INTEGER NOT NULL DEFAULT 0,
    source        TEXT    NOT NULL DEFAULT 'crex',
    fetched_at    TEXT    NOT NULL,
    PRIMARY KEY (fixture_key, team_id)
);

CREATE INDEX IF NOT EXISTS idx_crex_xi_cache_fixture
    ON crex_xi_cache(fixture_key);
