"""Paper-bet cohort tagging (V15).

The paper book is a continuous experiment, but the model/edge pipeline that
produces it changes. A single headline "paper P&L" across such a change compares
two different systems, so every paper row carries an explicit `paper_cohort`
tag and the default reporting view shows only the CURRENT cohort.

Cohorts
    LEGACY_PAPER_COHORT   the pre-existing book, frozen. This is the calibration
                          holdout: rows must stay byte-for-byte readable and are
                          never rewritten by reporting code.
    CURRENT_PAPER_COHORT  rows written from the cohort cutover onward.

Real rows (`bet_kind='real'`) are never tagged — the column is paper-only, and
real-money P&L figures are unaffected by any of this.

Retrieving each cohort (also the canonical calibration query)::

    -- legacy / calibration holdout
    SELECT * FROM bet_ledger
    WHERE bet_kind = 'paper' AND paper_cohort = 'legacy_pre_edge_20261108';

    -- current cohort (what the dashboard reports by default)
    SELECT * FROM bet_ledger
    WHERE bet_kind = 'paper' AND paper_cohort = 'post_edge_20261108';

The same is available over HTTP via ``?cohort=legacy_pre_edge_20261108`` on the
paper endpoints, and ``?cohort=all`` spans both.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

LEGACY_PAPER_COHORT = "legacy_pre_edge_20261108"
CURRENT_PAPER_COHORT = "post_edge_20261108"

#: Cohort applied to paper rows inserted from now on.
DEFAULT_PAPER_COHORT = CURRENT_PAPER_COHORT

ALL_COHORTS = "all"


def resolve_cohort_arg(raw: Optional[str]) -> str:
    """Map a user-supplied ``cohort`` query value onto a filter value.

    Returns one of the two concrete cohort names, or ``ALL_COHORTS``. Unknown
    values raise ValueError so a typo surfaces instead of silently reporting the
    wrong book. An absent/empty value means "the default view" == current cohort.
    """
    if raw is None or str(raw).strip() == "":
        return DEFAULT_PAPER_COHORT
    val = str(raw).strip()
    if val.lower() == ALL_COHORTS:
        return ALL_COHORTS
    if val in (LEGACY_PAPER_COHORT, CURRENT_PAPER_COHORT):
        return val
    raise ValueError(
        f"Unknown cohort '{val}'. Use '{CURRENT_PAPER_COHORT}' (default), "
        f"'{LEGACY_PAPER_COHORT}', or '{ALL_COHORTS}'."
    )


def cohort_filter_sql(cohort: str, column: str = "paper_cohort") -> Tuple[str, list]:
    """Return a ``(sql_fragment, params)`` pair to append to a WHERE clause.

    Filters at the QUERY level (never post-filtering rows in Python) so nothing
    is loaded that the caller did not ask for.

    ``cohort == ALL_COHORTS`` deliberately matches *every* paper row including
    untagged ones, so a partial backfill can never hide history.
    """
    if cohort == ALL_COHORTS:
        return "1 = 1", []
    return f"COALESCE({column}, ?) = ?", [LEGACY_PAPER_COHORT, cohort]


def new_paper_cohort_value() -> str:
    """Cohort stamp for a row being inserted right now."""
    return DEFAULT_PAPER_COHORT
