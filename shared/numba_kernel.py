"""
Numba CUDA kernel for GPU-accelerated gradient computation
in federated learning linear regression.
"""

import numpy as np
from numba import cuda
import math

# Try to use GPU, fall back to CPU if not available
try:
    # Check if CUDA is available
    cuda.get_current_device()
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False
    print(" CUDA not available.  Using CPU fallback.")


@cuda.jit
def compute_gradients_kernel(X, y, w, b, gradients):
    idx = cuda.grid(1)  # Thread index in 1D grid
    
    if idx < X.shape[0]:
        # Forward pass: prediction
        x_i = X[idx]
        y_i = y[idx]
        pred = w * x_i + b
        
        # Compute error
        error = pred - y_i
        
        # Compute gradients for this data point
        dw_i = error * x_i
        db_i = error
        
        # Atomic operations to safely accumulate gradients
        cuda.atomic.add(gradients, 0, dw_i)  # Accumulate dw
        cuda.atomic.add(gradients, 1, db_i)  # Accumulate db


def compute_gradients_gpu(X, y, w, b): 
    if not GPU_AVAILABLE:
        return compute_gradients_cpu(X, y, w, b)
    
    # Prepare GPU memory
    X_gpu = cuda.to_device(X. astype(np.float32))
    y_gpu = cuda.to_device(y.astype(np.float32))
    gradients_gpu = cuda.to_device(np.array([0.0, 0. 0], dtype=np.float32))
    
    # Launch kernel
    n_samples = X.shape[0]
    threads_per_block = 256
    blocks = (n_samples + threads_per_block - 1) // threads_per_block
    
    compute_gradients_kernel[blocks, threads_per_block](
        X_gpu, y_gpu, np.float32(w), np. float32(b), gradients_gpu
    )
    
    # Copy results back to CPU
    gradients = gradients_gpu.copy_to_host()
    
    # Normalize by number of samples (Mean Squared Error)
    dw = gradients[0] / n_samples
    db = gradients[1] / n_samples
    
    return dw, db


def compute_gradients_cpu(X, y, w, b):

    # Forward pass
    predictions = w * X + b
    
    # Error
    errors = predictions - y
    
    # Gradients (MSE)
    dw = np.sum(errors * X) / X.shape[0]
    db = np.sum(errors) / X.shape[0]
    
    return dw, db


def compute_gradients(X, y, w, b, use_gpu=True):

    if use_gpu and GPU_AVAILABLE:
        return compute_gradients_gpu(X, y, w, b)
    else:
        return compute_gradients_cpu(X, y, w, b)