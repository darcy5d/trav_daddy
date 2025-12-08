# Cricket Match Predictor

A neural network-powered Monte Carlo simulation system for predicting T20 cricket match outcomes. Supports both Men's and Women's cricket including franchise leagues (IPL, BBL, WBBL, WPL, etc.) and international T20s.

## Features

- **Neural Network Ball-by-Ball Simulation**: Predicts individual ball outcomes (dot, single, boundary, wicket) using a trained deep learning model
- **Monte Carlo Simulation**: Run 1,000-10,000 match simulations to generate win probabilities
- **ELO Rating System**: Tracks team and player ratings with historical snapshots
- **Men's & Women's Cricket**: Separate models trained on men's and women's T20 data
- **Live Match Integration**: Auto-load upcoming T20 matches from all competitions via Cricket Data API
- **Unified Match View**: See all upcoming matches (Men's [M] and Women's [W]) in one place with countdown timers
- **ESPN-Style Scorecards**: View detailed sample scorecards from simulations
- **Venue Effects**: 600+ venues with country flags and hierarchical grouping (including West Indies regions)
- **Toss Simulation**: Per-match toss outcomes simulated based on historical data
- **Smart Venue Matching**: Fuzzy matching for API venue names to database entries

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/cricket-match-predictor.git
cd cricket-match-predictor
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Download data and build database
python -m src.data.downloader
python -m src.data.ingest

# 3. Calculate ELO ratings
python -m src.elo.calculator_v2

# 4. Train neural network (optional - pre-trained models included)
python -m src.models.ball_prediction_nn

# 5. Run the web app
python app/main.py
```

Visit `http://localhost:5001` in your browser.

## How It Works

1. **Data**: Ball-by-ball JSON data from [Cricsheet.org](https://cricsheet.org/)
   - 16,700+ men's matches (IPL, BBL, T20I, PSL, CPL, etc.)
   - 3,900+ women's matches (WBBL, WPL, WT20I, etc.)
   - 600+ venues across 75+ countries

2. **Training**: Neural network learns ball outcome probabilities based on:
   - Batter/bowler ELO ratings and historical distributions
   - Match situation (score, wickets, required rate)
   - Innings phase (powerplay, middle, death)
   - Venue characteristics (scoring rate, boundary %, wicket rate)

   **Current Model Stats:**
   | Model | Training Samples | Features | Validation Accuracy |
   |-------|------------------|----------|---------------------|
   | Men's T20 | 1,477,440 | 29 | 44.2% |
   | Women's T20 | 571,659 | 29 | 51.1% |

3. **Simulation**: Monte Carlo engine simulates full matches ball-by-ball

4. **Prediction**: Aggregate simulation results into win probabilities

## Tech Stack

| Component | Technology |
|-----------|------------|
| ML Framework | TensorFlow/Keras |
| Backend | Python, Flask |
| Database | SQLite |
| Data | Cricsheet.org, Cricket Data API |

## Project Structure

```
├── app/                  # Flask web application
│   ├── main.py          # API endpoints
│   └── templates/       # HTML templates
├── src/
│   ├── api/             # Cricket Data API client
│   ├── data/            # Data ingestion & venue mapping
│   ├── models/          # Neural network & simulators
│   ├── features/        # Feature engineering
│   └── elo/             # ELO rating system
├── scripts/
│   ├── capture_baseline.py   # Database metrics snapshot
│   ├── migrate_venues.py     # Venue schema migration
│   └── full_retrain.py       # Complete retraining pipeline
├── data/
│   ├── raw/             # Cricsheet JSON files
│   └── processed/       # Training data, distributions
└── docs/                # Database schema documentation
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/teams` | GET | List teams with ELO ratings |
| `/api/players/{team}` | GET | Team players with ELO |
| `/api/venues` | GET | Venues with hierarchical country grouping |
| `/api/t20/upcoming` | GET | All upcoming T20 matches (next 24 hours) |
| `/api/t20/match/{id}` | GET | Load specific match with squads |
| `/api/simulate-stream` | POST | Run simulation with progress stream |
| `/api/rankings/teams` | GET | Team ELO rankings |
| `/api/rankings/players` | GET | Player ELO rankings |

## Performance

Optimized for Apple Silicon (M2 Pro):
- TensorFlow threading configured for multi-core
- NumPy BLAS parallelization
- ~60 simulations/second with Neural Network engine

## Data Sources

- [Cricsheet.org](https://cricsheet.org/) - Ball-by-ball match data
- [Cricket Data API](https://cricketdata.org/) - Live fixtures and squads

## License

MIT License

## Acknowledgments

- Cricsheet.org for comprehensive cricket data
- Inspired by [Towards Data Science cricket simulation article](https://towardsdatascience.com/)
