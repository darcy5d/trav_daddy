"""
Baseline Machine Learning Models for Cricket Match Prediction.

Implements traditional ML models:
- Logistic Regression
- Random Forest
- XGBoost

These serve as baselines before moving to deep learning approaches.
"""

import logging
import sys
import pickle
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not installed. Install with: pip install xgboost")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import MODEL_CONFIG, BASE_DIR
from src.features.engineer import create_training_dataset

logger = logging.getLogger(__name__)

# Model save directory
MODEL_DIR = BASE_DIR / "models" / "saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class BaselineModels:
    """
    Collection of baseline ML models for match prediction.
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
    
    def _create_models(self) -> Dict[str, Any]:
        """Create model instances."""
        models = {
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                random_state=MODEL_CONFIG['random_state'],
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                random_state=MODEL_CONFIG['random_state'],
                class_weight='balanced',
                n_jobs=-1
            )
        }
        
        if HAS_XGBOOST:
            models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=MODEL_CONFIG['random_state'],
                use_label_encoder=False,
                eval_metric='logloss'
            )
        
        return models
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        test_size: float = 0.2,
        use_time_split: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Train all baseline models.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: List of feature names
            test_size: Fraction of data for testing
            use_time_split: If True, use chronological split (recommended)
            
        Returns:
            Dictionary of metrics for each model
        """
        self.feature_names = feature_names
        self.models = self._create_models()
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Split data (chronological for time series)
        if use_time_split:
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=MODEL_CONFIG['random_state']
            )
        
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            # Train
            if name == 'logistic_regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
            
            # Evaluate
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_prob)
            }
            
            results[name] = metrics
            
            logger.info(f"{name} - Accuracy: {metrics['accuracy']:.4f}, "
                       f"ROC-AUC: {metrics['roc_auc']:.4f}")
        
        self.is_fitted = True
        
        # Print comparison
        self._print_comparison(results)
        
        # Feature importance for tree-based models
        self._print_feature_importance()
        
        return results
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform time-series cross-validation.
        
        Args:
            X: Feature matrix
            y: Target labels
            n_splits: Number of CV splits
            
        Returns:
            Dictionary of CV scores for each model
        """
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Cross-validating {name}...")
            
            if name == 'logistic_regression':
                X_scaled = self.scaler.fit_transform(X)
                scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='roc_auc')
            else:
                scores = cross_val_score(model, X, y, cv=tscv, scoring='roc_auc')
            
            results[name] = {
                'mean_roc_auc': scores.mean(),
                'std_roc_auc': scores.std(),
                'scores': scores.tolist()
            }
            
            logger.info(f"{name} - Mean ROC-AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return results
    
    def predict(
        self,
        X: np.ndarray,
        model_name: str = 'xgboost'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using a specific model.
        
        Args:
            X: Feature matrix
            model_name: Which model to use
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self.is_fitted:
            raise ValueError("Models not trained. Call train() first.")
        
        if model_name not in self.models:
            available = list(self.models.keys())
            raise ValueError(f"Unknown model: {model_name}. Available: {available}")
        
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        model = self.models[model_name]
        
        if model_name == 'logistic_regression':
            X = self.scaler.transform(X)
        
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        
        return predictions, probabilities
    
    def predict_match(
        self,
        features: Dict[str, float],
        model_name: str = 'xgboost'
    ) -> Dict[str, Any]:
        """
        Predict outcome for a single match.
        
        Args:
            features: Dictionary of feature values
            model_name: Which model to use
            
        Returns:
            Dictionary with prediction and probabilities
        """
        if self.feature_names is None:
            raise ValueError("Model not trained - feature names unknown")
        
        # Create feature array in correct order
        X = np.array([[features.get(name, 0) for name in self.feature_names]])
        
        pred, prob = self.predict(X, model_name)
        
        return {
            'prediction': 'team1' if pred[0] == 1 else 'team2',
            'team1_win_probability': float(prob[0]),
            'team2_win_probability': float(1 - prob[0]),
            'confidence': float(max(prob[0], 1 - prob[0]))
        }
    
    def _print_comparison(self, results: Dict[str, Dict[str, float]]):
        """Print comparison of model results."""
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        print(f"{'Model':<25} {'Accuracy':>10} {'F1':>10} {'ROC-AUC':>10}")
        print("-" * 60)
        
        for name, metrics in results.items():
            print(f"{name:<25} {metrics['accuracy']:>10.4f} "
                  f"{metrics['f1']:>10.4f} {metrics['roc_auc']:>10.4f}")
        
        print("=" * 60)
    
    def _print_feature_importance(self, top_n: int = 15):
        """Print feature importance for tree-based models."""
        if 'random_forest' not in self.models or self.feature_names is None:
            return
        
        rf = self.models['random_forest']
        importances = rf.feature_importances_
        
        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        print("\n" + "=" * 60)
        print(f"TOP {top_n} FEATURE IMPORTANCES (Random Forest)")
        print("=" * 60)
        
        for i, idx in enumerate(indices, 1):
            print(f"{i:2}. {self.feature_names[idx]:<35} {importances[idx]:.4f}")
        
        print("=" * 60)
    
    def save(self, filepath: Optional[Path] = None):
        """Save trained models to disk."""
        if filepath is None:
            filepath = MODEL_DIR / "baseline_models.pkl"
        
        save_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Models saved to {filepath}")
    
    def load(self, filepath: Optional[Path] = None):
        """Load trained models from disk."""
        if filepath is None:
            filepath = MODEL_DIR / "baseline_models.pkl"
        
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.models = save_data['models']
        self.scaler = save_data['scaler']
        self.feature_names = save_data['feature_names']
        self.is_fitted = save_data['is_fitted']
        
        logger.info(f"Models loaded from {filepath}")


def train_baseline_models(
    match_format: str = 'T20',
    min_date: Optional[str] = None,
    save_models: bool = True
) -> Dict[str, Any]:
    """
    Train baseline models on cricket match data.
    
    Args:
        match_format: 'T20' or 'ODI'
        min_date: Minimum date for training data
        save_models: Whether to save trained models
        
    Returns:
        Dictionary with training results
    """
    logger.info(f"Creating training dataset for {match_format}...")
    
    X, y, feature_names = create_training_dataset(match_format, min_date)
    
    if len(X) < 100:
        logger.error(f"Insufficient data: only {len(X)} samples")
        return {'error': 'Insufficient data'}
    
    logger.info(f"Dataset: {len(X)} samples, {len(feature_names)} features")
    
    # Train models
    baseline = BaselineModels()
    results = baseline.train(X, y, feature_names)
    
    # Cross-validation
    cv_results = baseline.cross_validate(X, y)
    
    if save_models:
        baseline.save()
    
    return {
        'test_results': results,
        'cv_results': cv_results,
        'n_samples': len(X),
        'n_features': len(feature_names),
        'feature_names': feature_names
    }


def main():
    """Train and evaluate baseline models."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    import argparse
    parser = argparse.ArgumentParser(description='Train baseline models')
    parser.add_argument('--format', default='T20', choices=['T20', 'ODI'])
    parser.add_argument('--min-date', default=None, help='Min date (YYYY-MM-DD)')
    parser.add_argument('--no-save', action='store_true', help='Do not save models')
    args = parser.parse_args()
    
    results = train_baseline_models(
        match_format=args.format,
        min_date=args.min_date,
        save_models=not args.no_save
    )
    
    if 'error' not in results:
        print("\nTraining completed successfully!")
        print(f"Samples: {results['n_samples']}")
        print(f"Features: {results['n_features']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

