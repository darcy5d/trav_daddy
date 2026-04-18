"""
Franchise resolver: map any teams.team_id to its canonical_team_id.

Background: V4 introduced soft franchise grouping so multiple teams.team_id
rows (e.g. "Royal Challengers Bangalore" 111 and "Royal Challengers Bengaluru"
283) can share one rating series. The ELO calculator writes every rating
under the canonical_team_id of the franchise, leaving match rows themselves
labelled with their original team1_id/team2_id.

This module is the runtime side of that contract. Anywhere outside the
calculator that needs to look up a current rating, fetch a batter/bowler
distribution scoped to the team, render franchise-aware UI, or compare
ratings across feeds, should call `FranchiseResolver.canonical(team_id)`
first.

Usage:
    from src.data.franchise_resolver import get_resolver
    canonical_id = get_resolver().canonical(team_id)

The resolver is a process-wide singleton with a small in-process cache,
loaded once from `teams` on first use. Call `invalidate()` after a Team
Explorer apply or any other mutation to teams.canonical_team_id.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class FranchiseResolver:
    """Resolve team_id -> canonical_team_id (and franchise group_id).

    Designed for hot paths: the entire teams.canonical_team_id mapping is a
    few hundred rows, so we load it once and serve from a dict.
    """

    def __init__(self) -> None:
        self._canonical: Dict[int, int] = {}
        self._franchise: Dict[int, Optional[int]] = {}
        self._loaded = False
        self._lock = threading.Lock()

    def _load(self) -> None:
        """Load mappings from `teams` if not already loaded."""
        if self._loaded:
            return
        # Import inside the method so importing this module never opens a
        # connection (test isolation, side-effect-free import).
        from src.data.database import get_connection

        with self._lock:
            if self._loaded:  # double-checked locking
                return
            try:
                conn = get_connection()
                cur = conn.cursor()
                cur.execute(
                    "SELECT team_id, canonical_team_id, franchise_id FROM teams"
                )
                for row in cur.fetchall():
                    tid = row["team_id"]
                    canonical = row["canonical_team_id"] if row["canonical_team_id"] is not None else tid
                    self._canonical[tid] = canonical
                    self._franchise[tid] = row["franchise_id"]
                conn.close()
                self._loaded = True
                logger.info(
                    f"FranchiseResolver loaded {len(self._canonical)} team mappings"
                )
            except Exception as exc:
                # If the franchise schema isn't installed yet (older DB), the
                # query fails and we degrade gracefully: every team is its own
                # canonical, with no franchise group.
                logger.warning(
                    f"FranchiseResolver could not load teams "
                    f"({exc.__class__.__name__}: {exc}); falling back to identity"
                )
                self._loaded = True

    def canonical(self, team_id: Optional[int]) -> Optional[int]:
        """Return the canonical owner of `team_id`'s rating series.

        Identity for unknown ids (so callers don't blow up on stub teams or
        teams that haven't been migrated yet). Returns None for None input.
        """
        if team_id is None:
            return None
        self._load()
        return self._canonical.get(team_id, team_id)

    def franchise(self, team_id: Optional[int]) -> Optional[int]:
        """Return the franchise group_id for `team_id`, or None if unknown."""
        if team_id is None:
            return None
        self._load()
        return self._franchise.get(team_id)

    def invalidate(self) -> None:
        """Drop the cache. Call after a Team Explorer apply or backfill run."""
        with self._lock:
            self._canonical.clear()
            self._franchise.clear()
            self._loaded = False
            logger.info("FranchiseResolver cache invalidated")


_RESOLVER: Optional[FranchiseResolver] = None
_RESOLVER_LOCK = threading.Lock()


def get_resolver() -> FranchiseResolver:
    """Return the process-wide FranchiseResolver singleton."""
    global _RESOLVER
    if _RESOLVER is None:
        with _RESOLVER_LOCK:
            if _RESOLVER is None:
                _RESOLVER = FranchiseResolver()
    return _RESOLVER
