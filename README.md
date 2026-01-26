# Cricket Match Predictor

A neural network-powered Monte Carlo simulation system for predicting T20 cricket match outcomes. Supports both Men's and Women's cricket including franchise leagues (IPL, BBL, WBBL, WPL, etc.) and international T20s.

## Features

### Core Prediction Engine
- **Neural Network Ball-by-Ball Simulation**: Predicts individual ball outcomes (dot, single, boundary, wicket) using a 34-feature deep learning model
- **ELO-Enhanced Features**: 5 ELO features integrated into predictions (batting/bowling team ELO, ELO differential, batter/bowler individual ELO)
- **Temporal ELO Consistency**: Training uses historical ELOs at match date; simulations use current ELOs
- **Monte Carlo Simulation**: Run 1,000-10,000 match simulations to generate win probabilities with real-time progress streaming
- **Tiered ELO Rating System (V3)**: 5-tier classification with tier-adjusted K-factors (stronger opponents = larger ELO gains)
- **Men's & Women's Cricket**: Separate models trained on men's and women's T20 data
- **Toss Simulation**: Per-match toss outcomes simulated based on historical venue/team data
- **Sample Scorecards**: View detailed ball-by-ball scorecards from representative simulations

### Match Discovery & Data Integration
- **CREX Integration**: Primary source for upcoming T20 matches with comprehensive squad data
- **Playwright-Powered Scraping**: Dynamic JavaScript rendering for reliable squad extraction from tabbed interfaces
- **Player Affiliation Verification**: Authoritative team identity using database player history (fixes squad mislabeling)
- **Unified Match View**: See all upcoming matches (Men's [M] and Women's [W]) in one place with countdown timers
- **Smart Team Matching**: Tab abbreviation matching (IRE, ITA) with fallback to player affiliations
- **Intelligent Venue Matching**: Fuzzy matching with canonical aliases (e.g., "Optus Stadium" -> "Perth Stadium")
- **Automatic Squad Loading**: Pre-fetch match squads with player-to-database matching and role detection
- **ESPN Cricinfo Fallback**: Secondary source when CREX data unavailable

### User Interface
- **Global Timezone Selector**: View all match times in your preferred timezone (defaults to Melbourne)
- **Dynamic Time Conversion**: All timestamps (match times, training dates, model versions) respect timezone selection
- **Data & Training Management**: Comprehensive GUI for managing models, data downloads, and retraining
- **Tiered Team Rankings**: View ELO rankings with International/Regional/Domestic tabs, tier badges, and 30-day ELO change indicators
- **Admin Promotion Review**: API endpoints for reviewing and approving tier adjustments for teams
- **Responsive Design**: Modern, clean interface with real-time updates

### Data Management & Training
- **Automated Data Downloads**: GUI-driven downloads from Cricsheet.org (male/female datasets)
- **Model Versioning**: Track all trained models with metadata (training date, samples, accuracy, data range)
- **Active Model Management**: Switch between different model versions on-the-fly
- **Database Status Dashboard**: Real-time view of data freshness, match counts, and coverage gaps
- **Reset & Rebuild (Day 0)**: Complete database reset with backup, re-ingestion, and full retraining pipeline
- **Real-time Progress Tracking**: Live progress bars and logs for downloads and training jobs

### Venue & Team Data
- **600+ Venues**: Comprehensive venue database with country flags and hierarchical grouping (including West Indies regions)
- **Venue Statistics**: Scoring rates, boundary percentages, and wicket rates per venue
- **Hierarchical Team Selection**: Segmented controls for easy team filtering
- **Gender-Aware Filtering**: Automatic filtering of teams, players, and venues by match gender

## Quick Start

### Apple Silicon (M1/M2/M3) - Recommended

For best performance with GPU acceleration, use the automated setup script:

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/cricket-match-predictor.git
cd cricket-match-predictor

# 2. Run the Apple Silicon setup script (installs Python 3.11 + Metal GPU)
./scripts/setup_apple_silicon.sh

# 3. Activate the environment
source venv311/bin/activate

# 4. Install Playwright browsers (for CREX squad scraping)
playwright install chromium

# 5. Run the web app
python app/main.py
```

This gives you **~400-600 simulations/second** with Metal GPU acceleration vs ~100 sims/sec on CPU.

### Standard Setup (Linux/Windows/Intel Mac)

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/cricket-match-predictor.git
cd cricket-match-predictor
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Install Playwright browsers (for CREX squad scraping)
playwright install chromium

# 3. Download data and build database (via GUI or CLI)
# Option A: Use the web GUI (recommended)
python app/main.py
# Visit http://localhost:5001/training and click "Download Latest Data"

# Option B: CLI method
python -m src.data.downloader
python -m src.data.ingest

# 4. Calculate ELO ratings (tiered system)
python scripts/recalculate_tiered_elo.py  # One-time full recalculation

# 5. Train neural network (optional - or use GUI)
python scripts/full_retrain.py  # Uses tiered ELO as features (34 features)

# 6. Run the web app
python app/main.py
```

Visit `http://localhost:5001` in your browser.

## How It Works

1. **Data**: Ball-by-ball JSON data from [Cricsheet.org](https://cricsheet.org/)
   - 16,700+ men's matches (IPL, BBL, T20I, PSL, CPL, etc.)
   - 3,900+ women's matches (WBBL, WPL, WT20I, etc.)
   - 600+ venues across 75+ countries

2. **Training**: Neural network learns ball outcome probabilities based on 34 features:
   - **Match State (6)**: Innings, over, ball, runs, wickets, target
   - **Phase (3)**: Powerplay, middle overs, death overs (one-hot)
   - **Batter Stats (8)**: Historical outcome distribution (dot%, 1s%, 2s%, 3s%, 4s%, 6s%, wicket%, balls faced)
   - **Bowler Stats (8)**: Historical outcome distribution (same breakdown)
   - **Venue (4)**: Scoring factor, boundary rate, wicket rate, data reliability flag
   - **Team ELO (3)**: Batting team ELO, bowling team ELO, ELO differential
   - **Player ELO (2)**: Batter batting ELO, bowler bowling ELO

   **Current Model Stats:**
   | Model | Training Samples | Features | Validation Accuracy | Data Range |
   |-------|------------------|----------|---------------------|------------|
   | Men's T20 | 1,477,440 | 34 | 44.2% | 2019-2025 |
   | Women's T20 | 571,659 | 34 | 51.1% | 2019-2025 |

3. **Simulation**: Monte Carlo engine simulates full matches ball-by-ball

4. **Prediction**: Aggregate simulation results into win probabilities

## Tech Stack

| Component | Technology |
|-----------|------------|
| ML Framework | TensorFlow/Keras (with Metal GPU on Apple Silicon) |
| Python Version | 3.11 recommended (required for Metal GPU) |
| Backend | Python, Flask |
| Database | SQLite |
| Historical Data | Cricsheet.org (16,700+ men's, 3,900+ women's matches) |
| Live Fixtures | CREX (primary), ESPNcricinfo (fallback) |
| Frontend | HTML, CSS, JavaScript |
| Web Scraping | Playwright (dynamic content), BeautifulSoup (static HTML) |

## Project Structure

```
├── app/                  # Flask web application
│   ├── main.py          # API endpoints & server
│   └── templates/       # HTML templates
│       ├── base.html    # Base template with global navbar & timezone selector
│       ├── index.html   # Home page
│       ├── predict.html # Match prediction interface
│       ├── rankings.html # ELO rankings
│       └── training.html # Data & training management
├── src/
│   ├── api/             # External API clients
│   │   ├── crex_scraper.py         # CREX scraper (primary) with Playwright support
│   │   ├── espn_scraper.py         # ESPNcricinfo scraper (fallback)
│   │   └── cricket_data_client.py  # Cricket Data API (legacy)
│   ├── data/            # Data ingestion & management
│   │   ├── downloader.py      # Cricsheet data downloader
│   │   ├── ingest.py          # JSON to SQLite ingestion
│   │   ├── database.py        # Database utilities & model versioning
│   │   ├── venue_normalizer.py # Venue deduplication & cleanup
│   │   └── schema_*.sql       # Database schemas
│   ├── models/          # Neural network & simulators
│   │   ├── ball_prediction_nn.py  # Main NN training (34 features)
│   │   ├── vectorized_nn_sim.py   # Optimized Monte Carlo simulator with ELO
│   │   ├── fast_lookup_sim.py     # Turbo mode simulator (distribution-based)
│   │   └── nn_simulator.py        # Legacy simulator
│   ├── features/        # Feature engineering
│   │   ├── ball_training_data.py   # Training data generation (34 features with ELO)
│   │   ├── player_distributions.py # Player stats & distributions
│   │   ├── venue_stats.py          # Venue statistics
│   │   ├── name_matcher.py         # Fuzzy player name matching
│   │   └── lineup_service.py       # Recent lineup fetching
│   ├── elo/             # ELO rating system
│   │   ├── calculator_v2.py       # Legacy ELO calculator
│   │   └── calculator_v3.py       # Tiered ELO calculator (V3)
│   └── utils/
│       └── job_manager.py         # Background job management
├── scripts/
│   ├── setup_apple_silicon.sh     # Apple Silicon setup (Python 3.11 + Metal GPU)
│   ├── full_retrain.py            # Complete retraining pipeline (uses V3 ELO)
│   ├── recalculate_tiered_elo.py  # Full tiered ELO recalculation
│   ├── validate_tiered_elo.py     # Validation & sanity checks
│   ├── capture_baseline.py        # Database metrics snapshot
│   └── migrate_venues.py          # Venue schema migration
├── data/
│   ├── raw/             # Cricsheet JSON files
│   ├── processed/       # Training data, models, distributions
│   ├── backups/         # Database backups
│   └── model_versions.json  # Model version metadata
└── docs/
    └── DATABASE_SCHEMA.md    # Database documentation
```

## API Endpoints

### Match & Team Data
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/teams` | GET | List teams with ELO ratings (filtered by gender) |
| `/api/players/{team}` | GET | Team players with ELO ratings |
| `/api/players/details` | POST | Get player details by IDs |
| `/api/team/<team_id>/recent-lineup` | GET | Fetch most recent playing XI for a team |
| `/api/venues` | GET | Venues with hierarchical country grouping |

### Match Discovery (CREX - Primary)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/crex/upcoming` | GET | Upcoming T20 matches with squad data |
| `/api/crex/match` | GET | Match details with squads, venue stats, and DB mappings (`?url=...`) |
| `/api/crex/live` | GET | Live match with toss result and playing XI (`?url=...`) |

### Match Discovery (ESPN - Fallback)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/espn/upcoming` | GET | Upcoming T20 matches (3h ago to 24h ahead) |
| `/api/espn/match/<match_id>` | GET | Match details with squads, venue, and DB mappings |

### Simulation
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/simulate-stream` | POST | Run simulation with real-time progress stream |

### Rankings (Tiered ELO System V3)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/rankings/teams` | GET | Team ELO rankings with tier filtering (`?tier=1&format=T20&gender=male`) |
| `/api/rankings/batting` | GET | Player batting ELO rankings |
| `/api/rankings/bowling` | GET | Player bowling ELO rankings |
| `/api/rankings/tier-stats` | GET | Team counts by tier and pending promotion flags |
| `/api/rankings/months` | GET | Available historical months for rankings |

### Admin (Tier Management)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/admin/promotion-flags` | GET | List pending tier promotion/demotion reviews |
| `/api/admin/promotion-flags/<id>/approve` | POST | Approve tier change for a team |
| `/api/admin/promotion-flags/<id>/reject` | POST | Reject tier change request |

### Data & Training Management
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/training/db-status` | GET | Database statistics (matches, players, data freshness) |
| `/api/training/models` | GET | List all model versions and active models |
| `/api/training/models/<id>/activate` | POST | Set a model version as active |
| `/api/training/download` | POST | Trigger background data download job |
| `/api/training/retrain` | POST | Trigger background model retraining job |
| `/api/training/reset-rebuild` | POST | Reset database and rebuild from scratch (Day 0) |
| `/api/training/job/<job_id>` | GET | Get status, progress, and logs of a background job |

## Performance

### Apple Silicon with Metal GPU (Recommended)

| Setup | Sims/Second | 10k Simulations | 50k Simulations |
|-------|-------------|-----------------|-----------------|
| M2 Pro + Metal GPU | ~400-600 | ~20 seconds | ~1.5 minutes |
| M2 Pro CPU Only | ~100-150 | ~80 seconds | ~7 minutes |
| Intel Mac | ~60-80 | ~2 minutes | ~10 minutes |

**Requirements for Metal GPU acceleration:**
- Apple Silicon Mac (M1/M2/M3)
- Python 3.11 (not 3.12 or 3.13)
- tensorflow-macos + tensorflow-metal

Run `./scripts/setup_apple_silicon.sh` for automated setup.

### Verify GPU is Active

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
# Should show: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### Other Optimizations
- TensorFlow threading configured for multi-core
- NumPy BLAS parallelization (vecLib on macOS)
- Pre-allocated feature buffers for reduced memory churn
- Background job management for long-running tasks
- Efficient caching for API responses (30-minute TTL)

## Key Features by Branch

### Main Features (Merged)
- **CREX + ESPN Integration**: Dual scraper system with CREX as primary and ESPN as fallback
- **Playwright Squad Extraction**: Dynamic JavaScript rendering for reliable squad data from tabbed interfaces
- **Data & Training GUI**: Comprehensive management interface for downloads, retraining, and model versioning
- **Timezone Support**: Global timezone selector with dynamic time conversion across all pages
- **Model Versioning**: Track training history with metadata (samples, accuracy, data range, file size)
- **Reset & Rebuild**: Day 0 functionality to start fresh with complete re-ingestion and training
- **Real-time Progress**: Live progress bars and logs for background jobs
- **Venue Cleanup**: Normalized and deduplicated 600+ venues with hierarchical grouping and aliases
- **Unified Match List**: All upcoming T20 matches (M/W) in one view with countdown timers
- **Fallback Lineups**: Auto-load recent lineups when API doesn't provide squads
- **Smart Venue Matching**: Fuzzy matching with canonical aliases and gender-aware filtering

### Recent Improvements
- **CREX Integration**: Primary match discovery with Playwright-powered squad extraction from dynamic tabs
- **Player Affiliation Verification**: Authoritative team identity using database player history (fixes squad mislabeling)
- **Team Abbreviation Matching**: Smart matching of tab labels (IRE, ITA, PRS) to team names with fallback resolution
- **ELO Feature Integration**: 34-feature model with 5 ELO features (team ELO, player ELO, differentials)
- **Temporal ELO Consistency**: Training uses historical ELOs at match date; simulations use current ELOs
- **Tier-Adjusted K-Factors**: Stronger opponents yield larger ELO gains (Ireland vs Australia > Ireland vs Bhutan)
- **Tiered ELO System (V3)**: Complete 5-tier classification with cross-pool normalization and automatic promotion review
- **Realistic Rankings**: India (1929) >> Somerset (1407) now reflects actual team strength hierarchy
- **Venue Alias System**: Canonical venue matching (e.g., "Optus Stadium" -> "Perth Stadium")
- **Conflict Resolution**: Handles edge cases where CREX mislabels both squads as the same team
- **Validation Framework**: Automated sanity checks and validation reports for ELO system integrity

## Data Sources

- [Cricsheet.org](https://cricsheet.org/) - Historical ball-by-ball match data (16,700+ men's, 3,900+ women's matches)
- [CREX](https://crex.com/) - Primary source for upcoming fixtures, squads, and venue statistics
- [ESPNcricinfo](https://www.espncricinfo.com/) - Fallback for match schedules and squads

## Development

### Running Tests
```bash
pytest tests/
```

### Database Migrations
```bash
# Capture baseline before changes
python scripts/capture_baseline.py

# Run venue migration (example)
python scripts/migrate_venues.py
```

### Full Retraining Pipeline
```bash
# CLI method
python scripts/full_retrain.py

# Or use the GUI at http://localhost:5001/training
```

## Inspiration

This project was inspired by Andrew Kuo's pioneering work on [ball-by-ball T20 cricket prediction using Monte Carlo simulation](https://towardsdatascience.com/predicting-t20-cricket-matches-with-a-ball-simulation-model-1e9cae5dea22/). In his 2021 article, Kuo demonstrated a bottom-up approach to cricket match prediction by training a neural network to predict individual ball outcomes (dot, single, boundary, wicket, etc.) and then simulating entire matches thousands of times to generate win probabilities. Working with 677,000+ balls from 3,651 T20 matches across 7 major leagues, he achieved 55.6% match prediction accuracy—outperforming bookmaker odds and revealing the inherent unpredictability of T20 cricket. His work highlighted the power of probabilistic simulation for capturing the nuanced, moment-to-moment dynamics that determine cricket match outcomes, providing the conceptual foundation for this implementation with expanded data coverage, model versioning, and real-time match integration.

## License

MIT License

## Acknowledgments

- Cricsheet.org for comprehensive cricket data
- ESPNcricinfo for live match schedules and squads
- Andrew Kuo for the original ball-by-ball simulation methodology
