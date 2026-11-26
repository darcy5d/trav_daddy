-- V15: explicit cohort tag on bet_ledger rows.
--
-- Why it exists
--   The paper book up to 2026-11-08 is the calibration holdout that the model
--   work was validated against. From 2026-11-08 the edge/model pipeline changes,
--   so mixing old and new paper results in one headline P&L number silently
--   compares two different models. This column tags each row with the cohort it
--   belongs to so the default P&L view can show only the current cohort while
--   the legacy cohort stays fully readable and unmodified for calibration.
--
-- Values
--   NULL                     - real (bet_kind='real') rows, and any paper row
--                              written before this column existed that the
--                              backfill has not reached. Real rows are never
--                              tagged: this column is paper-only.
--   legacy_pre_edge_20261108 - the pre-existing paper book (calibration holdout).
--   post_edge_20261108       - paper rows written from 2026-11-08 onward
--                              (stamped at insert time by the paper scanners).
--
-- The legacy backfill lives in scripts/backfill_paper_cohort.py (idempotent,
-- writes ONLY this column, and only for bet_kind='paper').

CREATE INDEX IF NOT EXISTS idx_bet_ledger_paper_cohort
    ON bet_ledger(paper_cohort)
    WHERE bet_kind = 'paper';
