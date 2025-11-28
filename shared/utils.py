"""
Utility functions for federated learning:
- Data generation
- Training loop
- Parameter management
"""

import numpy as np
from typing import Tuple, Dict, Any
from . numba_kernel import compute_gradients


class SyntheticDataGenerator:
    """Generate synthetic linear regression data."""
    
    def __init__(self, w_true: float = 3.5, b_true: float = 2.0, 
                 noise_std: float = 0.5, random_seed: int = None):
        """
        Initialize data generator.
        
        Args:
            w_true: True weight parameter
            b_true: True bias parameter
            noise_std: Standard deviation of Gaussian noise
            random_seed: Random seed for reproducibility
        """
        self.w_true = w_true
        self.b_true = b_true
        self.noise_std = noise_std
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def generate(self, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic data.
        
        Args:
            n_samples: Number of samples to generate
        
        Returns:
            (X, y): Input features and target values
        """
        # Generate random features in range [-10, 10]
        X = np.random.uniform(-10, 10, n_samples). astype(np.float32)
        
        # Generate targets with true model + noise
        y = self. w_true * X + self. b_true + np.random.normal(0, self.noise_std, n_samples)
        y = y.astype(np.float32)
        
        return X, y


class LocalTrainer:
    """Local model trainer for federated learning."""
    
    def __init__(self, learning_rate: float = 0.01, iterations: int = 100,batch_size: int = 32, use_gpu: bool = True):
                        

        self.learning_rate = learning_rate
        self.iterations = iterations
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        
        # Initialize model parameters
        self.w = np. random.randn(). astype(np.float32) * 0.1
        self.b = np.random.randn().astype(np.float32) * 0.1
        
        self.training_history = {
            'losses': [],
            'w_values': [],
            'b_values': []
        }
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train the model using SGD with GPU-accelerated gradient computation.
        
        Args:
            X: Input features (N,)
            y: Target values (N,)
        
        Returns:
            Dictionary with final parameters and training history
        """
        n_samples = X.shape[0]
        
        for iteration in range(self.iterations):
            # Random sampling for SGD
            indices = np.random.choice(n_samples, self.batch_size, replace=False)
            X_batch = X[indices]
            y_batch = y[indices]
            
            # Compute gradients using GPU/CPU
            dw, db = compute_gradients(X_batch, y_batch, self.w, self.b, use_gpu=self.use_gpu)
            
            # Update parameters (SGD)
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            
            # Log loss and parameters
            predictions = self.w * X + self.b
            mse_loss = np.mean((predictions - y) ** 2)
            
            self.training_history['losses']. append(float(mse_loss))
            self.training_history['w_values'].append(float(self. w))
            self.training_history['b_values'].append(float(self.b))
        
        return {
            'w': float(self.w),
            'b': float(self.b),
            'history': self.training_history
        }
    
    def get_parameters(self) -> Dict[str, float]:
        """Get current model parameters."""
        return {
            'w': float(self.w),
            'b': float(self.b)
        }
    
    def set_parameters(self, w: float, b: float):
        """Set model parameters."""
        self. w = np.float32(w)
        self. b = np.float32(b)


def aggregate_models(client_updates: list) -> Dict[str, float]:
    """
    Aggregate model parameters from multiple clients.
    
    Args:
        client_updates: List of dicts with 'w' and 'b' keys
    
    Returns:
        Aggregated parameters {w, b}
    """
    w_values = [update['w'] for update in client_updates]
    b_values = [update['b'] for update in client_updates]
    
    global_w = np.mean(w_values)
    global_b = np.mean(b_values)
    
    return {
        'w': float(global_w),
        'b': float(global_b)
    }