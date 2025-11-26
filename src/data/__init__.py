"""
Data ingestion and processing module.
"""

from .downloader import download_cricsheet_data
from .ingest import ingest_matches

__all__ = ["download_cricsheet_data", "ingest_matches"]

