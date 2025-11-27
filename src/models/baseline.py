"""
Baseline Machine Learning Models for Cricket Match Prediction.

Models:
- Logistic Regression (interpretable baseline)
- Random Forest (feature importance)
- XGBoost (strong gradient boosting baseline)

All models predict P(team1 wins) from match features.
"""

import logging
from typing import Dict, Tuple, Optional, List, Union
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score, brier_score_loss,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import MODEL_CONFIG

logger = logging.getLogger(__name__)

# Try to import XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not installed. Install with: pip install xgboost")


# Feature columns to use for prediction
FEATURE_COLUMNS = [
    # Team ELO features
    'team1_elo', 'team2_elo', 'elo_diff',
    'team1_momentum', 'team2_momentum', 'momentum_diff',
    
    # Head to head and form
    'h2h_team1_win_rate', 'h2h_total',
    'team1_recent_form', 'team2_recent_form', 'form_diff',
    
    # Team venue performance
    'team1_venue_win_rate', 'team2_venue_win_rate',
    
    # Team composition
    'team1_batting_elo_avg', 'team2_batting_elo_avg',
    'team1_bowling_elo_avg', 'team2_bowling_elo_avg',
    
    # Venue characteristics (NEW)
    'venue_avg_score', 'venue_avg_wickets', 'venue_chase_win_rate',
    'venue_is_high_scoring', 'venue_is_low_scoring',
    
    # Home advantage (NEW)
    'is_team1_home', 'is_team2_home',
    'home_advantage_team1', 'home_advantage_team2',
    'is_neutral_venue',
    
    # Toss
    'toss_winner_is_team1', 'chose_to_bat'
]


class BaselinePredictor:
    """Baseline ML models for match outcome prediction."""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = FEATURE_COLUMNS
        self.is_fitted = False
    
    def prepare_features(self, df: pd.DataFrame, include_target: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Extract feature matrix and optionally target from dataframe."""
        # Select only feature columns that exist
        available_features = [c for c in self.feature_columns if c in df.columns]
        
        X = df[available_features].copy()
        
        # Fill missing values
        X = X.fillna(X.mean())
        
        if include_target and 'team1_won' in df.columns:
            y = df['team1_won'].values
        else:
            y = None
        
        return X.values, y
    
    def train_test_split_chronological(
        self,
        df: pd.DataFrame,
        test_ratio: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data chronologically (most recent for test)."""
        df = df.sort_values('date')
        split_idx = int(len(df) * (1 - test_ratio))
        
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        logger.info(f"Train set: {len(train_df)} matches ({train_df['date'].min()} to {train_df['date'].max()})")
        logger.info(f"Test set: {len(test_df)} matches ({test_df['date'].min()} to {test_df['date'].max()})")
        
        return train_df, test_df
    
    def fit(self, df: pd.DataFrame):
        """Train all baseline models."""
        X, y = self.prepare_features(df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # 1. Logistic Regression
        logger.info("Training Logistic Regression...")
        self.models['logistic'] = LogisticRegression(
            random_state=MODEL_CONFIG['random_state'],
            max_iter=1000
        )
        self.models['logistic'].fit(X_scaled, y)
        
        # 2. Random Forest
        logger.info("Training Random Forest...")
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=MODEL_CONFIG['random_state'],
            n_jobs=-1
        )
        self.models['random_forest'].fit(X, y)  # RF doesn't need scaling
        
        # 3. XGBoost
        if HAS_XGBOOST:
            logger.info("Training XGBoost...")
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=MODEL_CONFIG['random_state'],
                use_label_encoder=False,
                eval_metric='logloss'
            )
            self.models['xgboost'].fit(X, y)
        
        self.is_fitted = True
        logger.info("All models trained successfully!")
    
    def predict_proba(self, df: pd.DataFrame, model_name: str = 'xgboost') -> np.ndarray:
        """Get probability predictions from specified model."""
        if not self.is_fitted:
            raise ValueError("Models not fitted. Call fit() first.")
        
        X, _ = self.prepare_features(df, include_target=False)
        
        if model_name == 'logistic':
            X = self.scaler.transform(X)
        
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not available")
        
        return model.predict_proba(X)[:, 1]
    
    def predict(self, df: pd.DataFrame, model_name: str = 'xgboost') -> np.ndarray:
        """Get binary predictions from specified model."""
        probs = self.predict_proba(df, model_name)
        return (probs >= 0.5).astype(int)
    
    def evaluate(self, df: pd.DataFrame) -> Dict:
        """Evaluate all models on dataset."""
        X, y = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        results = {}
        
        for name, model in self.models.items():
            X_eval = X_scaled if name == 'logistic' else X
            
            y_pred = model.predict(X_eval)
            y_proba = model.predict_proba(X_eval)[:, 1]
            
            results[name] = {
                'accuracy': accuracy_score(y, y_pred),
                'log_loss': log_loss(y, y_proba),
                'roc_auc': roc_auc_score(y, y_proba),
                'brier_score': brier_score_loss(y, y_proba)
            }
        
        return results
    
    def get_feature_importance(self, model_name: str = 'random_forest') -> pd.DataFrame:
        """Get feature importance from tree-based model."""
        if model_name not in ['random_forest', 'xgboost']:
            raise ValueError("Feature importance only available for tree models")
        
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not available")
        
        importance = model.feature_importances_
        
        # Get available features
        available_features = self.feature_columns[:len(importance)]
        
        df = pd.DataFrame({
            'feature': available_features,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df
    
    def cross_validate(self, df: pd.DataFrame, n_splits: int = 5) -> Dict:
        """Perform time-series cross-validation."""
        df = df.sort_values('date')
        X, y = self.prepare_features(df)
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        results = {}
        
        for name in ['logistic', 'random_forest']:
            if name == 'logistic':
                X_cv = self.scaler.fit_transform(X)
                model = LogisticRegression(random_state=42, max_iter=1000)
            else:
                X_cv = X
                model = RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
                )
            
            scores = cross_val_score(model, X_cv, y, cv=tscv, scoring='accuracy')
            results[name] = {
                'mean_accuracy': scores.mean(),
                'std_accuracy': scores.std(),
                'scores': scores.tolist()
            }
        
        return results
    
    def save(self, path: str):
        """Save trained models to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }, f)
        logger.info(f"Models saved to {path}")
    
    def load(self, path: str):
        """Load trained models from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.models = data['models']
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        self.is_fitted = True
        logger.info(f"Models loaded from {path}")


def train_and_evaluate(
    feature_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> Tuple[BaselinePredictor, Dict]:
    """
    Train baseline models and evaluate performance.
    
    Returns trained predictor and evaluation metrics.
    """
    predictor = BaselinePredictor()
    
    # Split chronologically
    train_df, test_df = predictor.train_test_split_chronological(feature_df)
    
    # Train
    predictor.fit(train_df)
    
    # Evaluate on test set
    results = predictor.evaluate(test_df)
    
    # Print results
    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS (Test Set)")
    print("=" * 60)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}")
        print("-" * 40)
        print(f"  Accuracy:    {metrics['accuracy']:.3f}")
        print(f"  Log Loss:    {metrics['log_loss']:.3f}")
        print(f"  ROC AUC:     {metrics['roc_auc']:.3f}")
        print(f"  Brier Score: {metrics['brier_score']:.3f}")
    
    # Feature importance
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (Random Forest)")
    print("=" * 60)
    importance_df = predictor.get_feature_importance('random_forest')
    print(importance_df.head(10).to_string(index=False))
    
    # Save if path provided
    if save_path:
        predictor.save(save_path)
    
    return predictor, results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    from src.features.match_features import build_training_dataset
    
    # Build feature dataset
    print("Building feature dataset...")
    df = build_training_dataset(format_type='T20', gender='male')
    
    # Train and evaluate
    predictor, results = train_and_evaluate(df)
    
    # Cross-validation
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 60)
    cv_results = predictor.cross_validate(df)
    for name, cv in cv_results.items():
        print(f"{name}: {cv['mean_accuracy']:.3f} (+/- {cv['std_accuracy']:.3f})")
