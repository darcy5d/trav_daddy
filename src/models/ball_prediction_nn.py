"""
Ball Prediction Neural Network.

A neural network that predicts the outcome of individual cricket deliveries.

Input features (29):
- Match state: innings, over, balls, runs, wickets, required_rate (6)
- Phase: powerplay/middle/death one-hot (3)
- Batter historical distribution (8)
- Bowler historical distribution (8)
- Venue features: scoring_factor, boundary_rate, wicket_rate, has_reliable_data (4)

Output: Probability distribution over 7 outcomes
- 0: Dot ball
- 1: Single
- 2: Two
- 3: Three  
- 4: Four
- 5: Six
- 6: Wicket

Based on the approach from:
https://towardsdatascience.com/predicting-t20-cricket-matches-with-a-ball-simulation-model-1e9cae5dea22/
"""

import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import TensorFlow/Keras
# TensorFlow 2.16+ uses keras 3.x which is a standalone package
import tensorflow as tf
try:
    # Try TensorFlow 2.16+ style (keras as standalone)
    import keras
    from keras import layers, regularizers
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
except ImportError:
    # Fall back to older TensorFlow style
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Sklearn for evaluation
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Outcome class names
OUTCOME_NAMES = ['Dot', 'Single', 'Two', 'Three', 'Four', 'Six', 'Wicket']
NUM_CLASSES = 7


def load_training_data(
    format_type: str = 'T20',
    gender: str = 'male'
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load training data from saved files."""
    data_dir = Path('data/processed')
    
    X = np.load(data_dir / f'ball_X_{format_type.lower()}_{gender}.npy')
    y = np.load(data_dir / f'ball_y_{format_type.lower()}_{gender}.npy')
    meta = pd.read_csv(data_dir / f'ball_meta_{format_type.lower()}_{gender}.csv')
    
    logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features")
    return X, y, meta


def create_ball_prediction_model(
    input_dim: int = 25,
    num_classes: int = NUM_CLASSES,
    hidden_units: List[int] = [64, 64],
    dropout_rate: float = 0.3,
    l2_reg: float = 0.001
) -> keras.Model:
    """
    Create the ball prediction neural network.
    
    Architecture follows the article:
    - 2 dense layers with ReLU activation
    - Batch normalization
    - Dropout for regularization
    - Softmax output for 7-class classification
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(input_dim,)),
        
        # First hidden layer
        layers.Dense(
            hidden_units[0],
            kernel_regularizer=regularizers.l2(l2_reg)
        ),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(dropout_rate),
        
        # Second hidden layer
        layers.Dense(
            hidden_units[1],
            kernel_regularizer=regularizers.l2(l2_reg)
        ),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(dropout_rate),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_ball_prediction_model(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    epochs: int = 50,
    batch_size: int = 256,
    validation_split: float = 0.2,
    model_path: str = 'data/processed/ball_prediction_model.keras'
) -> Tuple[keras.Model, dict]:
    """
    Train the ball prediction neural network.
    
    Uses chronological split for validation (train on older matches, validate on newer).
    """
    # Chronological split
    # Sort by date and split
    meta = meta.copy()
    meta['idx'] = range(len(meta))
    meta_sorted = meta.sort_values('date')
    
    split_idx = int(len(meta_sorted) * (1 - validation_split))
    train_indices = meta_sorted['idx'].iloc[:split_idx].values
    val_indices = meta_sorted['idx'].iloc[split_idx:].values
    
    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]
    
    logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Normalize features
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    
    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std
    
    # Create model
    model = create_ball_prediction_model(input_dim=X.shape[1])
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    history = model.fit(
        X_train_norm, y_train,
        validation_data=(X_val_norm, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save normalizer
    normalizer_path = model_path.replace('.keras', '_normalizer.pkl')
    with open(normalizer_path, 'wb') as f:
        pickle.dump({'mean': mean, 'std': std}, f)
    
    logger.info(f"Saved model to {model_path}")
    logger.info(f"Saved normalizer to {normalizer_path}")
    
    # Evaluate
    results = evaluate_model(model, X_val_norm, y_val)
    
    return model, results


def evaluate_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> dict:
    """Evaluate the model and return metrics."""
    # Predictions
    y_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)
    
    # Metrics
    accuracy = np.mean(y_pred == y_test)
    logloss = log_loss(y_test, y_proba)
    
    # Per-class metrics
    report = classification_report(y_test, y_pred, target_names=OUTCOME_NAMES, output_dict=True)
    
    results = {
        'accuracy': accuracy,
        'log_loss': logloss,
        'classification_report': report
    }
    
    return results


def print_evaluation_results(results: dict):
    """Print evaluation results nicely."""
    print("\n" + "=" * 70)
    print("BALL PREDICTION MODEL EVALUATION")
    print("=" * 70)
    
    print(f"\nOverall Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.1f}%)")
    print(f"Log Loss: {results['log_loss']:.4f}")
    
    print("\n" + "-" * 70)
    print("PER-CLASS PERFORMANCE")
    print("-" * 70)
    print(f"{'Outcome':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 70)
    
    for outcome in OUTCOME_NAMES:
        if outcome in results['classification_report']:
            m = results['classification_report'][outcome]
            print(f"{outcome:<10} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1-score']:>10.3f} {int(m['support']):>10}")


class BallPredictionModel:
    """
    Wrapper class for the ball prediction model.
    
    Provides methods for loading, predicting, and sampling outcomes.
    """
    
    def __init__(self, model_path: str = 'data/processed/ball_prediction_model.keras'):
        self.model_path = model_path
        self.model = None
        self.normalizer = None
        self._loaded = False
    
    def load(self):
        """Load the trained model and normalizer."""
        self.model = keras.models.load_model(self.model_path)
        
        normalizer_path = self.model_path.replace('.keras', '_normalizer.pkl')
        with open(normalizer_path, 'rb') as f:
            self.normalizer = pickle.load(f)
        
        self._loaded = True
        logger.info(f"Loaded ball prediction model from {self.model_path}")
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict outcome probabilities for given features.
        
        Args:
            features: Feature vector(s) of shape (n_samples, 25) or (25,)
        
        Returns:
            Probability distribution(s) over 7 outcomes
        """
        if not self._loaded:
            self.load()
        
        # Handle single sample
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Normalize
        X_norm = (features - self.normalizer['mean']) / self.normalizer['std']
        
        # Predict
        return self.model.predict(X_norm, verbose=0)
    
    def sample_outcome(self, features: np.ndarray) -> int:
        """
        Sample a single outcome from the predicted distribution.
        
        Args:
            features: Feature vector of shape (25,)
        
        Returns:
            Outcome class (0-6)
        """
        proba = self.predict_proba(features)[0]
        return np.random.choice(NUM_CLASSES, p=proba)
    
    def sample_outcome_batch(self, features: np.ndarray) -> np.ndarray:
        """
        Sample outcomes for a batch of features.
        
        Args:
            features: Feature matrix of shape (n_samples, 25)
        
        Returns:
            Array of outcome classes
        """
        proba = self.predict_proba(features)
        
        outcomes = []
        for p in proba:
            outcomes.append(np.random.choice(NUM_CLASSES, p=p))
        
        return np.array(outcomes)
    
    def get_outcome_name(self, class_idx: int) -> str:
        """Get human-readable outcome name."""
        return OUTCOME_NAMES[class_idx]
    
    def class_to_runs_wicket(self, class_idx: int) -> Tuple[int, bool]:
        """
        Convert class index to (runs, is_wicket).
        
        Returns:
            (runs_scored, is_wicket)
        """
        if class_idx == 6:  # Wicket
            return 0, True
        else:
            runs = [0, 1, 2, 3, 4, 6][class_idx]
            return runs, False


def main(format_type: str = 'T20', gender: str = 'male'):
    """Train and evaluate the ball prediction model."""
    print("=" * 70)
    print(f"BALL PREDICTION NEURAL NETWORK TRAINING ({format_type} {gender.upper()})")
    print("=" * 70)
    
    # Load data
    X, y, meta = load_training_data(format_type, gender)
    
    # Gender-specific model path
    model_path = f'data/processed/ball_prediction_model_{format_type.lower()}_{gender}.keras'
    
    # Train model
    model, results = train_ball_prediction_model(
        X, y, meta,
        epochs=50,
        batch_size=256,
        model_path=model_path
    )
    
    # Print results
    print_evaluation_results(results)
    
    # Test the wrapper class
    print("\n" + "=" * 70)
    print("TESTING MODEL WRAPPER")
    print("=" * 70)
    
    predictor = BallPredictionModel(model_path)
    predictor.load()
    
    # Test prediction on first sample
    test_features = X[0]
    proba = predictor.predict_proba(test_features)[0]
    
    print("\nSample prediction probabilities:")
    for i, (name, p) in enumerate(zip(OUTCOME_NAMES, proba)):
        bar = "█" * int(p * 50)
        print(f"  {name:<8}: {p:>6.1%} {bar}")
    
    # Sample multiple outcomes
    print("\nSampling 10 outcomes from this distribution:")
    for _ in range(10):
        outcome = predictor.sample_outcome(test_features)
        runs, is_wicket = predictor.class_to_runs_wicket(outcome)
        print(f"  → {OUTCOME_NAMES[outcome]} (runs={runs}, wicket={is_wicket})")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    parser = argparse.ArgumentParser(description='Train ball prediction neural network')
    parser.add_argument('--format', default='T20', choices=['T20', 'ODI'])
    parser.add_argument('--gender', default='male', choices=['male', 'female'])
    args = parser.parse_args()
    
    main(args.format, args.gender)


