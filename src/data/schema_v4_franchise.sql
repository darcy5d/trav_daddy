-- ============================================================================
-- FRANCHISE / TEAM-GROUP UNIFICATION - Schema V4
-- ============================================================================
-- Adds a soft grouping layer above the teams table so that:
--   * Multiple teams.team_id rows can belong to one franchise / national identity
--     (e.g. "Royal Challengers Bangalore" id 111 and "Royal Challengers Bengaluru"
--      id 283 are the same franchise; "Australia" and "Australia A" share an
--      identity for some workflows).
--   * One team_id is designated the canonical owner of the rating series for
--     the group (canonical_team_id on the teams table) so all ELO reads/writes
--     can be funnelled through it without rewriting historical match rows.
--   * External identifiers from other data sources (Cricinfo, CREX, Cricsheet,
--     etc.) can be attached at the group level so we have a stable join key
--     across feeds (paired with item 17 Cricsheet Register on the player side).
--   * Merge proposals raised in the new Team Explorer tab live in their own
--     audited table with an explicit pending -> approved -> applied flow.
--
-- Notes on idempotency:
--   * Everything here uses CREATE TABLE / INDEX IF NOT EXISTS so it can run on
--     every app startup safely.
--   * Columns added to the teams table are handled by Python because SQLite
--     does not support IF NOT EXISTS for ALTER TABLE ADD COLUMN.
--   * UNIQUE constraints on team_elo_history / player_elo_history are created
--     by the Phase C dedupe script after duplicates are removed (the index
--     creation would fail on a dirty DB).

-- ============================================================================
-- team_groups
-- ============================================================================
CREATE TABLE IF NOT EXISTS team_groups (
    group_id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name TEXT NOT NULL UNIQUE,
    -- One row per real-world identity. Most existing teams self-group 1:1;
    -- rebranded franchises and national-variant clusters group N:1.
    group_type TEXT NOT NULL DEFAULT 'domestic'
        CHECK(group_type IN (
            'franchise',
            'franchise_women',
            'national',
            'national_a',
            'national_u19',
            'national_u19_women',
            'national_women',
            'domestic',
            'special'
        )),
    country_code TEXT,
    -- Free-form display preferences (preferred display name, short code, colours,
    -- alternative spellings...) stored as JSON for flexibility without a schema
    -- change every time the UI grows a new field.
    display_preferences TEXT,  -- JSON
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_team_groups_type ON team_groups(group_type);
CREATE INDEX IF NOT EXISTS idx_team_groups_country ON team_groups(country_code);

-- ============================================================================
-- team_external_ids
-- ============================================================================
-- Cross-source identifier registry, paired conceptually with the Cricsheet
-- Register-style player work tracked as item 17 in NEXT_OVERHAUL.md. Schema
-- lands here; population from individual sources happens later.
CREATE TABLE IF NOT EXISTS team_external_ids (
    group_id INTEGER NOT NULL REFERENCES team_groups(group_id) ON DELETE CASCADE,
    source TEXT NOT NULL
        CHECK(source IN (
            'cricinfo',
            'crex',
            'cricsheet',
            'espn',
            'bcci',
            'polymarket',
            'betfair',
            'opta',
            'pulse'
        )),
    external_id TEXT NOT NULL,
    -- Optional extras useful for debugging mismatches
    external_name TEXT,
    external_url TEXT,
    confidence REAL,            -- 0..1, NULL if hand-mapped
    last_validated TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (group_id, source, external_id)
);

CREATE INDEX IF NOT EXISTS idx_team_external_ids_source ON team_external_ids(source, external_id);

-- ============================================================================
-- team_merge_proposals
-- ============================================================================
-- Backs the Team Explorer's staged-approval workflow. A proposal moves a
-- single source teams row into a target group. Approval is required before
-- "apply" actually rewrites any rows; apply is transactional and triggers an
-- ELO recalc (see scripts/dedupe_elo_history.py).
CREATE TABLE IF NOT EXISTS team_merge_proposals (
    proposal_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_team_id INTEGER NOT NULL REFERENCES teams(team_id),
    target_group_id INTEGER NOT NULL REFERENCES team_groups(group_id),
    -- The team_id within the target group that should own the rating series
    -- after the merge. Allows the proposer to flip the canonical owner to a
    -- different teams row in one step (e.g. when promoting the modern rebrand
    -- to canonical for an existing group).
    target_canonical_team_id INTEGER REFERENCES teams(team_id),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK(status IN ('pending', 'approved', 'rejected', 'applied')),
    proposer TEXT,
    rationale TEXT,
    fuzzy_score REAL,           -- 0..1 from the Explorer's candidate finder
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    decided_at TIMESTAMP,       -- when approve/reject happened
    applied_at TIMESTAMP,       -- when the merge was actually applied
    decided_by TEXT,
    apply_notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_team_merge_proposals_status
    ON team_merge_proposals(status, created_at);
CREATE INDEX IF NOT EXISTS idx_team_merge_proposals_source
    ON team_merge_proposals(source_team_id);
CREATE INDEX IF NOT EXISTS idx_team_merge_proposals_target
    ON team_merge_proposals(target_group_id);
