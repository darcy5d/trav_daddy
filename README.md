# Cricket Match Predictor

A machine learning system for predicting T20 and ODI cricket match outcomes using ball-by-ball simulation and ELO rating systems.

## Overview

This project builds a comprehensive cricket match prediction engine that:

- **Ingests ball-by-ball match data** from [Cricsheet.org](https://cricsheet.org/)
- **Maintains ELO ratings** for teams and players (batting, bowling, overall)
- **Trains deep learning models** to predict match outcomes and simulate ball-by-ball progression
- **Provides a Flask web interface** for making predictions on upcoming matches

## Features

- **ELO Rating System**: Track team and player ratings over time with monthly snapshots
- **Ball-by-Ball Simulation**: Monte Carlo simulation of matches using trained models
- **Multiple Model Support**: Baseline ML models (XGBoost, Random Forest) and deep learning (TensorFlow/PyTorch)
- **Web Interface**: User-friendly Flask application for match predictions

## Project Structure

```
cricket-match-predictor/
├── data/
│   ├── raw/               # Downloaded JSON files from Cricsheet
│   └── processed/         # Cleaned/processed data
├── src/
│   ├── data/              # Data ingestion and processing
│   ├── models/            # ML/DL model implementations
│   ├── features/          # Feature engineering
│   ├── elo/               # ELO rating system
│   └── utils/             # Helper functions
├── app/                   # Flask web application
│   ├── templates/
│   ├── static/
│   └── routes/
├── notebooks/             # Jupyter notebooks for EDA
├── tests/
├── requirements.txt
├── config.py
└── README.md
```

## Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cricket-match-predictor.git
cd cricket-match-predictor
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the cricket data:
```bash
python -m src.data.downloader
```

5. Initialize the database and ingest data:
```bash
python -m src.data.ingest
```

6. Calculate ELO ratings:
```bash
python -m src.elo.calculator
```

## Usage

### Running the Web Application

```bash
python -m app.main
```

Visit `http://localhost:5000` in your browser.

### Training Models

```bash
# Train baseline models
python -m src.models.baseline

# Train deep learning models
python -m src.models.deep_learning
```

### Running Predictions

```python
from src.models.predictor import MatchPredictor

predictor = MatchPredictor()
result = predictor.predict(
    team1="India",
    team2="Australia",
    venue="Melbourne Cricket Ground",
    match_type="T20"
)
print(f"Win probability: {result}")
```

## Data Sources

- **Match Data**: [Cricsheet.org](https://cricsheet.org/) - Ball-by-ball JSON data
- **Formats Supported**: T20 Internationals, One Day Internationals (ODI)

## Technology Stack

| Component | Technology |
|-----------|------------|
| Database | SQLite |
| Backend | Python 3.11+, Flask |
| ML/DL | TensorFlow, PyTorch, scikit-learn |
| Data Processing | pandas, numpy |
| Visualization | Plotly, Matplotlib |

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/teams` | List available teams with current ELOs |
| `GET /api/players/{team}` | Players for a team with ELO ratings |
| `GET /api/elo/team/{team_id}` | Team ELO history |
| `GET /api/elo/player/{player_id}` | Player ELO history |
| `POST /api/predict` | Run prediction for match setup |
| `POST /api/simulate` | Run Monte Carlo simulation |

## Model Architecture

### ELO System

- **Team ELO**: Updated after each match based on result and opponent strength
- **Player Batting ELO**: Based on runs scored vs expected, considering bowler quality
- **Player Bowling ELO**: Based on wickets taken and economy vs expected
- **Monthly Snapshots**: Historical ratings preserved for time-travel queries

### Prediction Models

1. **Baseline Models**: Logistic Regression, Random Forest, XGBoost
2. **Deep Learning**: MLP, LSTM/GRU for sequential modeling
3. **Ball Simulation**: Predicts outcome distribution per delivery for Monte Carlo simulation

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Cricsheet.org](https://cricsheet.org/) for providing comprehensive cricket data
- The cricket analytics community for research and inspiration

