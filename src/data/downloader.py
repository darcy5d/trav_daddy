"""
Cricsheet data downloader.

Downloads T20 and ODI match data from Cricsheet.org in JSON format.
"""

import os
import shutil
import sys
import tempfile
import time as _time_mod
import zipfile
import logging
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import DATA_SOURCES, RAW_DATA_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def download_file(url: str, destination: Path, chunk_size: int = 8192) -> bool:
    """
    Download a file from URL with progress bar.
    
    Args:
        url: URL to download from
        destination: Path to save the file
        chunk_size: Size of chunks to download
        
    Returns:
        True if download successful, False otherwise
    """
    try:
        logger.info(f"Downloading from {url}")
        
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get("content-length", 0))
        
        # Create parent directories if needed
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        with open(destination, "wb") as f:
            with tqdm(
                total=total_size,
                unit="iB",
                unit_scale=True,
                desc=destination.name
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        logger.info(f"Downloaded to {destination}")
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path, max_retries: int = 3) -> bool:
    """
    Extract a ZIP file to the specified directory.

    Uses a temp directory outside iCloud/Desktop scope for extraction then
    moves the result into place.  This avoids [Errno 11] Resource deadlock
    errors that macOS raises when iCloud Drive monitors files being written
    under ~/Desktop while zipfile.extract() streams them.

    Args:
        zip_path: Path to the ZIP file
        extract_to: Directory to extract to (final destination)
        max_retries: Number of attempts before giving up (default 3)

    Returns:
        True if extraction successful, False otherwise
    """
    for attempt in range(1, max_retries + 1):
        tmp_dir = None
        try:
            logger.info(
                f"Extracting {zip_path.name} via temp dir (iCloud-safe)"
                + (f" [attempt {attempt}/{max_retries}]" if attempt > 1 else "")
            )

            # Extract to /private/var/folders/… (outside iCloud scope)
            tmp_dir = Path(tempfile.mkdtemp(prefix="cricsheet_"))

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                file_list = zip_ref.namelist()
                for file in tqdm(file_list, desc="Extracting", leave=False):
                    zip_ref.extract(file, tmp_dir)

            # Atomically move into place: remove old dir first
            if extract_to.exists():
                shutil.rmtree(extract_to)
            shutil.move(str(tmp_dir), str(extract_to))
            tmp_dir = None  # ownership transferred

            logger.info(f"Extracted {len(file_list)} files to {extract_to}")
            return True

        except zipfile.BadZipFile as e:
            logger.error(f"Invalid ZIP file: {e}")
            return False
        except Exception as e:
            logger.error(f"Extraction failed (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                logger.info(f"Retrying in 5s…")
                _time_mod.sleep(5)
        finally:
            if tmp_dir is not None and tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

    return False


def download_cricsheet_data(
    formats: Optional[list] = None,
    force_download: bool = False,
    age_threshold_hours: Optional[float] = None,
) -> dict:
    """
    Download cricket match data from Cricsheet.org.

    Args:
        formats: List of formats to download (e.g., ["t20i", "odi"]).
                If None, downloads all available formats.
        force_download: If True, re-download even if files exist.
        age_threshold_hours: If set, re-download when the local zip is older
                than this many hours (mtime-based). Lets daily-refresh callers
                avoid pulling 50MB on every cron tick while still picking up
                new data overnight. Ignored if `force_download=True`.

    Returns:
        Dictionary with format keys and paths to extracted data directories.
    """
    if formats is None:
        formats = list(DATA_SOURCES.keys())
    
    # Ensure raw data directory exists
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    for format_name in formats:
        if format_name not in DATA_SOURCES:
            logger.warning(f"Unknown format: {format_name}. Skipping.")
            continue

        url = DATA_SOURCES[format_name]
        zip_filename = f"{format_name}_json.zip"
        zip_path = RAW_DATA_DIR / zip_filename
        extract_dir = RAW_DATA_DIR / format_name

        # Decide whether to re-download. Priority:
        #   force_download=True              -> always redownload
        #   age_threshold_hours set + stale  -> redownload
        #   extract_dir exists with JSONs    -> skip (cached)
        #   else                             -> download
        zip_is_stale = False
        if age_threshold_hours is not None and zip_path.exists():
            zip_age_h = (_time_mod.time() - zip_path.stat().st_mtime) / 3600.0
            if zip_age_h > age_threshold_hours:
                zip_is_stale = True
                logger.info(
                    f"{format_name}: local zip is {zip_age_h:.1f}h old "
                    f"(threshold {age_threshold_hours}h) - redownloading"
                )

        if not force_download and not zip_is_stale and extract_dir.exists():
            json_files = list(extract_dir.glob("**/*.json"))
            if json_files:
                logger.info(
                    f"{format_name}: Already have {len(json_files)} JSON files. "
                    "Pass force_download=True or age_threshold_hours to re-download."
                )
                results[format_name] = extract_dir
                continue

        # Download the ZIP file
        if not zip_path.exists() or force_download or zip_is_stale:
            success = download_file(url, zip_path)
            if not success:
                logger.error(f"Failed to download {format_name} data")
                continue
        
        # Extract the ZIP file
        success = extract_zip(zip_path, extract_dir)
        if success:
            results[format_name] = extract_dir
            
            # Count extracted files
            json_files = list(extract_dir.glob("**/*.json"))
            logger.info(f"{format_name}: Extracted {len(json_files)} JSON files")
            
            # Optionally remove the ZIP file to save space
            # zip_path.unlink()
        else:
            logger.error(f"Failed to extract {format_name} data")
    
    return results


def get_json_files(format_name: str) -> list:
    """
    Get list of JSON match files for a specific format.
    
    Args:
        format_name: Format name (e.g., "t20i", "odi")
        
    Returns:
        List of Path objects for JSON files
    """
    format_dir = RAW_DATA_DIR / format_name
    
    if not format_dir.exists():
        logger.warning(f"No data found for {format_name}. Run download first.")
        return []
    
    # Find all JSON files (excluding any metadata files)
    json_files = sorted(format_dir.glob("**/*.json"))
    
    # Filter out any non-match files (like readme or info files)
    match_files = [f for f in json_files if f.stem.isdigit() or "_" in f.stem]
    
    return match_files


def print_download_summary():
    """Print a summary of downloaded data."""
    logger.info("\n" + "=" * 50)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 50)
    
    for format_name in DATA_SOURCES.keys():
        format_dir = RAW_DATA_DIR / format_name
        
        if format_dir.exists():
            json_files = list(format_dir.glob("**/*.json"))
            total_size = sum(f.stat().st_size for f in json_files)
            size_mb = total_size / (1024 * 1024)
            
            logger.info(f"{format_name.upper()}: {len(json_files)} files ({size_mb:.1f} MB)")
        else:
            logger.info(f"{format_name.upper()}: Not downloaded")
    
    logger.info("=" * 50)


def main():
    """Main function to download all cricket data."""
    logger.info("Starting Cricsheet data download...")
    
    # Download both T20I and ODI data
    results = download_cricsheet_data(formats=["t20i", "odi"])
    
    # Print summary
    print_download_summary()
    
    if results:
        logger.info("Download completed successfully!")
        return 0
    else:
        logger.error("Download failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

