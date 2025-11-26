"""
Deep Learning Models for Cricket Match Prediction.

Implements neural network models using TensorFlow/Keras and PyTorch:
- Multi-Layer Perceptron (MLP) for match outcome prediction
- LSTM for sequential modeling (innings progression)
- Architecture for ball-by-ball simulation

Supports both TensorFlow and PyTorch backends.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import MODEL_CONFIG, BASE_DIR

logger = logging.getLogger(__name__)

# Model save directory
MODEL_DIR = BASE_DIR / "models" / "saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Check available backends
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    logger.warning("TensorFlow not available")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    logger.warning("PyTorch not available")


# ============================================================================
# TensorFlow/Keras Models
# ============================================================================

if HAS_TENSORFLOW:
    
    class MatchPredictorMLP(keras.Model):
        """
        Multi-Layer Perceptron for match outcome prediction.
        
        Takes match features as input and outputs win probability.
        """
        
        def __init__(
            self,
            input_dim: int,
            hidden_dims: List[int] = [256, 128, 64],
            dropout_rate: float = 0.3
        ):
            super().__init__()
            
            self.hidden_layers = []
            self.dropout_layers = []
            self.bn_layers = []
            
            prev_dim = input_dim
            for dim in hidden_dims:
                self.hidden_layers.append(layers.Dense(dim, activation='relu'))
                self.bn_layers.append(layers.BatchNormalization())
                self.dropout_layers.append(layers.Dropout(dropout_rate))
                prev_dim = dim
            
            self.output_layer = layers.Dense(1, activation='sigmoid')
        
        def call(self, x, training=False):
            for hidden, bn, dropout in zip(self.hidden_layers, self.bn_layers, self.dropout_layers):
                x = hidden(x)
                x = bn(x, training=training)
                x = dropout(x, training=training)
            
            return self.output_layer(x)
    
    
    class MatchPredictorLSTM(keras.Model):
        """
        LSTM model for sequential match/innings prediction.
        
        Can be used for modeling innings progression or
        ball-by-ball simulation.
        """
        
        def __init__(
            self,
            input_dim: int,
            lstm_units: List[int] = [128, 64],
            dense_units: List[int] = [64, 32],
            dropout_rate: float = 0.3
        ):
            super().__init__()
            
            self.lstm_layers = []
            for i, units in enumerate(lstm_units):
                return_sequences = (i < len(lstm_units) - 1)
                self.lstm_layers.append(
                    layers.LSTM(units, return_sequences=return_sequences, dropout=dropout_rate)
                )
            
            self.dense_layers = []
            for units in dense_units:
                self.dense_layers.append(layers.Dense(units, activation='relu'))
                self.dense_layers.append(layers.Dropout(dropout_rate))
            
            self.output_layer = layers.Dense(1, activation='sigmoid')
        
        def call(self, x, training=False):
            for lstm in self.lstm_layers:
                x = lstm(x, training=training)
            
            for layer in self.dense_layers:
                x = layer(x, training=training)
            
            return self.output_layer(x)
    
    
    class BallOutcomePredictor(keras.Model):
        """
        Model for predicting ball-by-ball outcomes.
        
        Outputs probability distribution over possible outcomes:
        - Runs: 0, 1, 2, 3, 4, 6
        - Wicket probability
        - Extras probability
        """
        
        def __init__(
            self,
            input_dim: int,
            hidden_dims: List[int] = [128, 64],
            dropout_rate: float = 0.2
        ):
            super().__init__()
            
            self.hidden_layers = []
            self.dropout_layers = []
            
            prev_dim = input_dim
            for dim in hidden_dims:
                self.hidden_layers.append(layers.Dense(dim, activation='relu'))
                self.dropout_layers.append(layers.Dropout(dropout_rate))
                prev_dim = dim
            
            # Output heads
            self.runs_output = layers.Dense(6, activation='softmax', name='runs')  # 0,1,2,3,4,6
            self.wicket_output = layers.Dense(1, activation='sigmoid', name='wicket')
            self.extras_output = layers.Dense(1, activation='sigmoid', name='extras')
        
        def call(self, x, training=False):
            for hidden, dropout in zip(self.hidden_layers, self.dropout_layers):
                x = hidden(x)
                x = dropout(x, training=training)
            
            return {
                'runs': self.runs_output(x),
                'wicket': self.wicket_output(x),
                'extras': self.extras_output(x)
            }


# ============================================================================
# PyTorch Models
# ============================================================================

if HAS_PYTORCH:
    
    class MatchPredictorMLPTorch(nn.Module):
        """
        PyTorch MLP for match outcome prediction.
        """
        
        def __init__(
            self,
            input_dim: int,
            hidden_dims: List[int] = [256, 128, 64],
            dropout_rate: float = 0.3
        ):
            super().__init__()
            
            layers_list = []
            prev_dim = input_dim
            
            for dim in hidden_dims:
                layers_list.extend([
                    nn.Linear(prev_dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ])
                prev_dim = dim
            
            layers_list.append(nn.Linear(prev_dim, 1))
            layers_list.append(nn.Sigmoid())
            
            self.network = nn.Sequential(*layers_list)
        
        def forward(self, x):
            return self.network(x)
    
    
    class MatchPredictorLSTMTorch(nn.Module):
        """
        PyTorch LSTM for sequential modeling.
        """
        
        def __init__(
            self,
            input_dim: int,
            lstm_hidden: int = 128,
            lstm_layers: int = 2,
            fc_dim: int = 64,
            dropout_rate: float = 0.3
        ):
            super().__init__()
            
            self.lstm = nn.LSTM(
                input_dim, lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=dropout_rate if lstm_layers > 1 else 0
            )
            
            self.fc = nn.Sequential(
                nn.Linear(lstm_hidden, fc_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(fc_dim, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            # Take last timestep
            last_out = lstm_out[:, -1, :]
            return self.fc(last_out)


# ============================================================================
# Training and Evaluation Functions
# ============================================================================

class DeepLearningTrainer:
    """
    Trainer class for deep learning models.
    
    Supports both TensorFlow and PyTorch backends.
    """
    
    def __init__(self, backend: str = 'tensorflow'):
        """
        Initialize trainer.
        
        Args:
            backend: 'tensorflow' or 'pytorch'
        """
        if backend == 'tensorflow' and not HAS_TENSORFLOW:
            raise ImportError("TensorFlow not available")
        if backend == 'pytorch' and not HAS_PYTORCH:
            raise ImportError("PyTorch not available")
        
        self.backend = backend
        self.model = None
        self.history = None
        self.feature_names = None
    
    def build_model(
        self,
        input_dim: int,
        model_type: str = 'mlp',
        **kwargs
    ):
        """
        Build a model.
        
        Args:
            input_dim: Number of input features
            model_type: 'mlp', 'lstm', or 'ball_predictor'
            **kwargs: Additional model arguments
        """
        if self.backend == 'tensorflow':
            if model_type == 'mlp':
                self.model = MatchPredictorMLP(input_dim, **kwargs)
            elif model_type == 'lstm':
                self.model = MatchPredictorLSTM(input_dim, **kwargs)
            elif model_type == 'ball_predictor':
                self.model = BallOutcomePredictor(input_dim, **kwargs)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Compile model
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=MODEL_CONFIG['learning_rate']),
                loss='binary_crossentropy',
                metrics=['accuracy', keras.metrics.AUC(name='auc')]
            )
        
        elif self.backend == 'pytorch':
            if model_type == 'mlp':
                self.model = MatchPredictorMLPTorch(input_dim, **kwargs)
            elif model_type == 'lstm':
                self.model = MatchPredictorLSTMTorch(input_dim, **kwargs)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
    
    def train_tensorflow(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """Train using TensorFlow."""
        callbacks = [
            EarlyStopping(
                monitor='val_auc',
                patience=15,
                restore_best_weights=True,
                mode='max'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            ModelCheckpoint(
                str(MODEL_DIR / 'best_tf_model.keras'),
                monitor='val_auc',
                save_best_only=True,
                mode='max'
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history.history
        
        # Evaluate
        results = self.model.evaluate(X_val, y_val, verbose=0)
        
        return {
            'val_loss': results[0],
            'val_accuracy': results[1],
            'val_auc': results[2],
            'epochs_trained': len(history.history['loss'])
        }
    
    def train_pytorch(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """Train using PyTorch."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train.reshape(-1, 1))
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_X = torch.FloatTensor(X_val).to(device)
        val_y = torch.FloatTensor(y_val.reshape(-1, 1)).to(device)
        
        # Training setup
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=MODEL_CONFIG['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'loss': [], 'val_loss': [], 'val_accuracy': []}
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(val_X)
                val_loss = criterion(val_outputs, val_y).item()
                val_preds = (val_outputs > 0.5).float()
                val_acc = (val_preds == val_y).float().mean().item()
            
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), MODEL_DIR / 'best_pytorch_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}: loss={train_loss:.4f}, "
                           f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        # Load best model
        self.model.load_state_dict(torch.load(MODEL_DIR / 'best_pytorch_model.pt'))
        self.history = history
        
        return {
            'val_loss': best_val_loss,
            'val_accuracy': history['val_accuracy'][-1],
            'epochs_trained': len(history['loss'])
        }
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        test_size: float = 0.2,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Feature names
            test_size: Validation set size
            **kwargs: Additional training arguments
        """
        self.feature_names = feature_names
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize features
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8
        X = (X - self.mean) / self.std
        
        # Split data chronologically
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        logger.info(f"Training: {len(X_train)} samples, Validation: {len(X_val)} samples")
        
        # Build model if not already built
        if self.model is None:
            self.build_model(X.shape[1])
        
        # Train
        if self.backend == 'tensorflow':
            return self.train_tensorflow(X_train, y_train, X_val, y_val, **kwargs)
        else:
            return self.train_pytorch(X_train, y_train, X_val, y_val, **kwargs)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = (X - self.mean) / self.std
        
        if self.backend == 'tensorflow':
            return self.model.predict(X, verbose=0).flatten()
        else:
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                return self.model(X_tensor).numpy().flatten()
    
    def save(self, filepath: Optional[Path] = None):
        """Save the model."""
        if filepath is None:
            filepath = MODEL_DIR / f'deep_model_{self.backend}'
        
        if self.backend == 'tensorflow':
            self.model.save(str(filepath) + '.keras')
        else:
            torch.save({
                'model_state': self.model.state_dict(),
                'mean': self.mean,
                'std': self.std,
                'feature_names': self.feature_names
            }, str(filepath) + '.pt')
        
        logger.info(f"Model saved to {filepath}")


def train_deep_learning_models(
    match_format: str = 'T20',
    backend: str = 'tensorflow',
    min_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Train deep learning models.
    
    Args:
        match_format: 'T20' or 'ODI'
        backend: 'tensorflow' or 'pytorch'
        min_date: Minimum date for training data
        
    Returns:
        Training results
    """
    from src.features.engineer import create_training_dataset
    
    logger.info(f"Creating training dataset for {match_format}...")
    X, y, feature_names = create_training_dataset(match_format, min_date)
    
    if len(X) < 100:
        return {'error': 'Insufficient data'}
    
    logger.info(f"Dataset: {len(X)} samples, {len(feature_names)} features")
    
    trainer = DeepLearningTrainer(backend=backend)
    trainer.build_model(len(feature_names), model_type='mlp')
    
    results = trainer.train(
        X, y, feature_names,
        epochs=MODEL_CONFIG['epochs'],
        batch_size=MODEL_CONFIG['batch_size']
    )
    
    trainer.save()
    
    return results


def main():
    """Train deep learning models."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    import argparse
    parser = argparse.ArgumentParser(description='Train deep learning models')
    parser.add_argument('--format', default='T20', choices=['T20', 'ODI'])
    parser.add_argument('--backend', default='tensorflow', choices=['tensorflow', 'pytorch'])
    parser.add_argument('--min-date', default=None)
    args = parser.parse_args()
    
    # Check backend availability
    if args.backend == 'tensorflow' and not HAS_TENSORFLOW:
        logger.error("TensorFlow not available. Install with: pip install tensorflow")
        return 1
    if args.backend == 'pytorch' and not HAS_PYTORCH:
        logger.error("PyTorch not available. Install with: pip install torch")
        return 1
    
    results = train_deep_learning_models(
        match_format=args.format,
        backend=args.backend,
        min_date=args.min_date
    )
    
    print("\nTraining Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

