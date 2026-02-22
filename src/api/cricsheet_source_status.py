"""
Cricsheet source status fetcher.

Fetches the Cricsheet homepage and parses the paragraph that describes
the most recent matches added to the site, extracting the latest date
for "latest data from X days ago" in the Database Status UI.
"""

import logging
import re
from datetime import datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Default URL: the "most recent matches added" paragraph is on /matches/, not homepage
CRICSHEET_MATCHES_URL = "https://cricsheet.org/matches/"
REQUEST_TIMEOUT = 10

# Paragraph content we look for (present on cricsheet.org/matches/)
PHRASE_MOST_RECENT = "most recent matches added"

# Date pattern: "11th of February, 2026" or "11th of February" (year optional)
MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
)
MONTH_PATTERN = "|".join(re.escape(m) for m in MONTHS)
# Ordinal: 1st, 2nd, 3rd, 4th, 11th, 21st, 22nd, 23rd, 31st
DATE_REGEX = re.compile(
    r"(\d{1,2})(?:st|nd|rd|th)\s+of\s+(" + MONTH_PATTERN + r")(?:\s*,\s*(\d{4}))?",
    re.IGNORECASE
)


def _parse_dates_from_text(text: str) -> list[datetime]:
    """Extract all dates from paragraph text. Returns list of date objects."""
    if not text:
        return []
    now = datetime.now()
    dates = []
    for m in DATE_REGEX.finditer(text):
        day = int(m.group(1))
        month_str = m.group(2).strip()
        year_str = m.group(3)
        year = int(year_str) if year_str else now.year
        try:
            month = next(
                i for i, name in enumerate(MONTHS, 1)
                if name.lower() == month_str.lower()
            )
            dt = datetime(year, month, day)
            if dt.date() <= now.date():
                dates.append(dt)
        except (StopIteration, ValueError):
            continue
    return dates


def _get_paragraph_text(soup: BeautifulSoup) -> Optional[str]:
    """
    Get the "most recent matches added" paragraph text.
    The paragraph lives on cricsheet.org/matches/; find any <p> containing the phrase.
    """
    for tag in soup.find_all("p"):
        text = tag.get_text(separator=" ", strip=True)
        if PHRASE_MOST_RECENT.lower() in text.lower():
            return text
    logger.warning("Cricsheet page: no paragraph containing '%s' found", PHRASE_MOST_RECENT)
    return None


def get_source_status(matches_url: Optional[str] = None) -> Optional[dict]:
    """
    Fetch Cricsheet matches page and parse the latest match date from the
    "most recent matches added" paragraph.

    Returns:
        Dict with latest_date (YYYY-MM-DD), days_ago (int), snippet (str),
        or None on fetch/parse failure.
    """
    url = matches_url or CRICSHEET_MATCHES_URL
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.warning("Failed to fetch Cricsheet matches page: %s", e)
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    text = _get_paragraph_text(soup)
    if not text:
        return None

    dates = _parse_dates_from_text(text)
    if not dates:
        logger.warning("Cricsheet paragraph had no parseable dates: %s", text[:200])
        return None

    latest = max(dates)
    today = datetime.now().date()
    days_ago = (today - latest.date()).days
    if days_ago < 0:
        days_ago = 0

    return {
        "latest_date": latest.strftime("%Y-%m-%d"),
        "days_ago": days_ago,
        "snippet": text[:300] + ("..." if len(text) > 300 else ""),
    }
