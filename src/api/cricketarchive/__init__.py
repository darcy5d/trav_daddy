"""CricketArchive enrichment package.

Polite, cached, authenticated reader for cricketarchive.com used to enrich the
local cricket database (canonical cross-source player IDs + player biographical
data + coverage gap-fill). PERSONAL USE ONLY — never redistribute scraped data.

Public surface:
    from src.api.cricketarchive import PoliteFetcher, ensure_session
    from src.api.cricketarchive import models, parsers
"""

from .auth import ensure_session, login_manual_assist
from .fetcher import PoliteFetcher, NotAuthenticatedError, RobotsDisallowedError

__all__ = [
    "ensure_session",
    "login_manual_assist",
    "PoliteFetcher",
    "NotAuthenticatedError",
    "RobotsDisallowedError",
]
