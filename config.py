"""
Configuration settings for the Cricket Match Predictor project.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Database
DATABASE_PATH = BASE_DIR / "cricket.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# Data sources - Cricsheet URLs
CRICSHEET_BASE_URL = "https://cricsheet.org/downloads/"
DATA_SOURCES = {
    "t20i": f"{CRICSHEET_BASE_URL}t20s_json.zip",           # All T20s (men + women)
    "t20s_female": f"{CRICSHEET_BASE_URL}t20s_female_json.zip",  # Women's T20 internationals
    "all_female": f"{CRICSHEET_BASE_URL}all_female_json.zip",    # ALL women's cricket (T20+ODI+franchise)
    "wpl": f"{CRICSHEET_BASE_URL}wpl_json.zip",             # Women's Premier League (India)
    "odi": f"{CRICSHEET_BASE_URL}odis_json.zip",
}

# Data filtering
MIN_MATCH_DATE = "2019-01-01"  # Only use matches from this date onwards
SUPPORTED_FORMATS = ["T20", "ODI"]

# Cricket Data API (cricketdata.org)
CRICKET_DATA_API_KEY = os.getenv("CRICKET_DATA_API_KEY")
CRICKET_DATA_BASE_URL = "https://api.cricapi.com/v1"

# ELO Configuration
ELO_CONFIG = {
    "initial_rating": 1500,
    "k_factor_team": 32,  # K-factor for team ELO updates
    "k_factor_player_batting": 20,  # K-factor for player batting ELO
    "k_factor_player_bowling": 20,  # K-factor for player bowling ELO
    "rating_floor": 1000,  # Minimum possible rating
    "rating_ceiling": 2500,  # Maximum possible rating
}

# Model Configuration
MODEL_CONFIG = {
    "test_size": 0.2,
    "validation_size": 0.1,
    "random_state": 42,
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
}

# Monte Carlo Simulation
SIMULATION_CONFIG = {
    "num_simulations": 10000,  # Number of match simulations to run
    "confidence_level": 0.95,  # For confidence intervals
}

# Flask Configuration
class FlaskConfig:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    DEBUG = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    HOST = os.getenv("FLASK_HOST", "127.0.0.1")
    PORT = int(os.getenv("FLASK_PORT", 5000))


# Logging Configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": BASE_DIR / "app.log",
            "formatter": "standard",
            "level": "DEBUG",
        },
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "INFO",
    },
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    "recent_form_matches": 10,  # Number of recent matches for form calculation
    "min_matches_for_stats": 5,  # Minimum matches for reliable statistics
    "innings_phases": {
        "powerplay": (0, 6),
        "middle": (6, 15),  # For T20, adjusted for ODI
        "death": (15, 20),
    },
}

# Ball outcome categories
BALL_OUTCOMES = {
    "runs": [0, 1, 2, 3, 4, 6],
    "extras": ["wide", "noball", "bye", "legbye"],
    "wickets": ["bowled", "caught", "lbw", "run out", "stumped", "hit wicket"],
}

