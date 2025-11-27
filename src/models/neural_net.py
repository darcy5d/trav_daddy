"""
Neural Network Models for Cricket Match Prediction.

Models:
1. MatchOutcomeMLP - Predicts match winner probability
2. BallOutcomeNN - Predicts ball-by-ball outcome distributions from ELO matchups
3. DeepMatchPredictor - Combines both for full match simulation
"""

import logging
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import MODEL_CONFIG

logger = logging.getLogger(__name__)

# Try to import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    logger.warning("TensorFlow not installed. Install with: pip install tensorflow")

# Try to import PyTorch as alternative
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False


# Feature columns for match prediction
MATCH_FEATURE_COLUMNS = [
    'team1_elo', 'team2_elo', 'elo_diff',
    'team1_momentum', 'team2_momentum', 'momentum_diff',
    'h2h_team1_win_rate', 'h2h_total',
    'team1_recent_form', 'team2_recent_form', 'form_diff',
    'team1_venue_win_rate', 'team2_venue_win_rate',
    'team1_batting_elo_avg', 'team2_batting_elo_avg',
    'team1_bowling_elo_avg', 'team2_bowling_elo_avg',
    'venue_avg_score', 'venue_avg_wickets', 'venue_chase_win_rate',
    'venue_is_high_scoring', 'venue_is_low_scoring',
    'is_team1_home', 'is_team2_home', 'is_neutral_venue',
    'toss_winner_is_team1', 'chose_to_bat'
]


class MatchOutcomeMLP:
    """
    Multi-Layer Perceptron for match outcome prediction.
    
    Takes match features and predicts P(team1 wins).
    """
    
    def __init__(
        self,
        hidden_layers: List[int] = [64, 32, 16],
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001
    ):
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow required. Install with: pip install tensorflow")
        
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.feature_columns = MATCH_FEATURE_COLUMNS
        self.scaler_mean = None
        self.scaler_std = None
    
    def build_model(self, input_dim: int):
        """Build the neural network architecture."""
        inputs = keras.Input(shape=(input_dim,), name='match_features')
        
        x = inputs
        for i, units in enumerate(self.hidden_layers):
            x = layers.Dense(units, activation='relu', name=f'hidden_{i}')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout_rate)(x)
        
        # Output: probability of team1 winning
        outputs = layers.Dense(1, activation='sigmoid', name='win_probability')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name='MatchOutcomeMLP')
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        return self.model
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and normalize features."""
        available = [c for c in self.feature_columns if c in df.columns]
        X = df[available].fillna(0).values.astype(np.float32)
        
        if self.scaler_mean is not None:
            X = (X - self.scaler_mean) / (self.scaler_std + 1e-8)
        
        return X
    
    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1
    ):
        """Train the model."""
        # Prepare training data
        X_train = self.prepare_features(train_df)
        y_train = train_df['team1_won'].values.astype(np.float32)
        
        # Fit scaler on training data
        self.scaler_mean = X_train.mean(axis=0)
        self.scaler_std = X_train.std(axis=0)
        X_train = (X_train - self.scaler_mean) / (self.scaler_std + 1e-8)
        
        # Build model
        self.build_model(X_train.shape[1])
        
        # Prepare validation data
        validation_data = None
        if val_df is not None:
            X_val = self.prepare_features(val_df)
            X_val = (X_val - self.scaler_mean) / (self.scaler_std + 1e-8)
            y_val = val_df['team1_won'].values.astype(np.float32)
            validation_data = (X_val, y_val)
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss' if val_df is not None else 'loss'),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict win probability for team1."""
        X = self.prepare_features(df)
        X = (X - self.scaler_mean) / (self.scaler_std + 1e-8)
        return self.model.predict(X, verbose=0).flatten()
    
    def evaluate(self, df: pd.DataFrame) -> Dict:
        """Evaluate model on dataset."""
        from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss
        
        y_true = df['team1_won'].values
        y_proba = self.predict_proba(df)
        y_pred = (y_proba >= 0.5).astype(int)
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'log_loss': log_loss(y_true, y_proba),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'brier_score': brier_score_loss(y_true, y_proba)
        }
    
    def save(self, path: str):
        """Save model and scaler."""
        self.model.save(f"{path}_model.keras")
        with open(f"{path}_scaler.pkl", 'wb') as f:
            pickle.dump({
                'mean': self.scaler_mean,
                'std': self.scaler_std,
                'feature_columns': self.feature_columns
            }, f)
    
    def load(self, path: str):
        """Load model and scaler."""
        self.model = keras.models.load_model(f"{path}_model.keras")
        with open(f"{path}_scaler.pkl", 'rb') as f:
            data = pickle.load(f)
            self.scaler_mean = data['mean']
            self.scaler_std = data['std']
            self.feature_columns = data['feature_columns']


class BallOutcomeNN:
    """
    Neural Network to predict ball-by-ball outcome distributions.
    
    Input: Batter ELO, Bowler ELO, match situation
    Output: Probability distribution over outcomes [0, 1, 2, 3, 4, 6, W]
    """
    
    OUTCOMES = ['0', '1', '2', '3', '4', '6', 'W']
    
    def __init__(
        self,
        hidden_layers: List[int] = [128, 64, 32],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ):
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow required")
        
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
    
    def build_model(self, input_dim: int):
        """Build the neural network."""
        inputs = keras.Input(shape=(input_dim,), name='ball_features')
        
        x = inputs
        for i, units in enumerate(self.hidden_layers):
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout_rate)(x)
        
        # Output: 7 outcomes (0, 1, 2, 3, 4, 6, W)
        outputs = layers.Dense(7, activation='softmax', name='outcome_probs')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name='BallOutcomeNN')
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare ball-level features."""
        feature_cols = [
            'batter_elo', 'bowler_elo', 'elo_diff',
            'over_number', 'wickets_fallen', 'current_score',
            'is_chasing', 'pressure_index'
        ]
        
        available = [c for c in feature_cols if c in df.columns]
        X = df[available].fillna(0).values.astype(np.float32)
        
        # Add phase encoding
        if 'phase' in df.columns:
            phase_map = {'powerplay': 0, 'middle': 1, 'death': 2}
            phase_encoded = df['phase'].map(phase_map).fillna(1).values.reshape(-1, 1)
            X = np.hstack([X, phase_encoded])
        
        return X
    
    def prepare_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Convert outcomes to one-hot encoded labels."""
        outcome_map = {o: i for i, o in enumerate(self.OUTCOMES)}
        labels = df['outcome'].map(outcome_map).values
        return keras.utils.to_categorical(labels, num_classes=7)
    
    def fit(self, df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None, 
            epochs: int = 50, batch_size: int = 256, verbose: int = 1):
        """Train the model."""
        X_train = self.prepare_features(df)
        y_train = self.prepare_labels(df)
        
        # Normalize
        self.feature_mean = X_train.mean(axis=0)
        self.feature_std = X_train.std(axis=0) + 1e-8
        X_train = (X_train - self.feature_mean) / self.feature_std
        
        self.build_model(X_train.shape[1])
        
        validation_data = None
        if val_df is not None:
            X_val = self.prepare_features(val_df)
            X_val = (X_val - self.feature_mean) / self.feature_std
            y_val = self.prepare_labels(val_df)
            validation_data = (X_val, y_val)
        
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            class_weight=self._compute_class_weights(y_train)
        )
        
        return history
    
    def _compute_class_weights(self, y: np.ndarray) -> Dict:
        """Compute class weights for imbalanced outcomes."""
        class_counts = y.sum(axis=0)
        total = class_counts.sum()
        weights = {i: total / (7 * count) for i, count in enumerate(class_counts) if count > 0}
        return weights
    
    def predict_distribution(
        self,
        batter_elo: float,
        bowler_elo: float,
        over_number: int,
        wickets_fallen: int,
        current_score: int,
        is_chasing: bool,
        pressure: float = 0.0
    ) -> Dict[str, float]:
        """Predict outcome distribution for a single delivery."""
        features = np.array([[
            batter_elo, bowler_elo, batter_elo - bowler_elo,
            over_number, wickets_fallen, current_score,
            1 if is_chasing else 0, pressure,
            0 if over_number < 6 else (1 if over_number < 15 else 2)  # phase
        ]], dtype=np.float32)
        
        features = (features - self.feature_mean) / self.feature_std
        probs = self.model.predict(features, verbose=0)[0]
        
        return {outcome: float(prob) for outcome, prob in zip(self.OUTCOMES, probs)}


def train_match_neural_net(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    epochs: int = 100,
    save_path: Optional[str] = None
) -> Tuple[MatchOutcomeMLP, Dict]:
    """
    Train neural network for match outcome prediction.
    
    Returns trained model and evaluation metrics.
    """
    # Chronological split
    df = df.sort_values('date')
    split_idx = int(len(df) * (1 - test_ratio))
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # Further split train for validation
    val_split = int(len(train_df) * 0.9)
    val_df = train_df.iloc[val_split:]
    train_df = train_df.iloc[:val_split]
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Train model
    model = MatchOutcomeMLP(
        hidden_layers=[128, 64, 32],
        dropout_rate=0.3,
        learning_rate=0.001
    )
    
    history = model.fit(train_df, val_df, epochs=epochs, verbose=1)
    
    # Evaluate
    results = model.evaluate(test_df)
    
    print("\n" + "=" * 60)
    print("NEURAL NETWORK EVALUATION (Test Set)")
    print("=" * 60)
    print(f"  Accuracy:    {results['accuracy']:.3f}")
    print(f"  Log Loss:    {results['log_loss']:.3f}")
    print(f"  ROC AUC:     {results['roc_auc']:.3f}")
    print(f"  Brier Score: {results['brier_score']:.3f}")
    
    if save_path:
        model.save(save_path)
        logger.info(f"Model saved to {save_path}")
    
    return model, results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # Load feature dataset
    df = pd.read_csv('data/processed/t20_men_features_v2.csv')
    print(f"Loaded {len(df)} matches")
    
    # Train neural network
    model, results = train_match_neural_net(
        df, 
        epochs=100,
        save_path='data/processed/neural_net'
    )


