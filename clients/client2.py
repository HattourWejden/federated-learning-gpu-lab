"""
Client 2: Local training service for federated learning. 
Trains a linear regression model on local data and returns model parameters.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys. path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np
from shared.utils import SyntheticDataGenerator, LocalTrainer

# Initialize FastAPI app
app = FastAPI(
    title="Federated Learning Client 2",
    description="Local training service for federated learning",
    version="1.0.0"
)

# Global trainer instance
trainer = None
X_train = None
y_train = None

# Response model
class TrainResponse(BaseModel):
    """Response from training endpoint."""
    w: float
    b: float
    client_id: str
    status: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    client_id: str


@app.on_event("startup")
async def startup_event():
    """Initialize trainer and generate data on startup."""
    global trainer, X_train, y_train
    
    # Generate synthetic data (different seed for client 2)
    data_gen = SyntheticDataGenerator(
        w_true=3.5,
        b_true=2.0,
        noise_std=0.5,
        random_seed=123  # Different seed for each client
    )
    X_train, y_train = data_gen.generate(n_samples=200)
    
    # Initialize trainer
    trainer = LocalTrainer(
        learning_rate=0.01,
        iterations=100,
        batch_size=32,
        use_gpu=True
    )
    
    print("✅ Client 2 initialized")
    print(f"   Data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"   Training config: lr=0.01, iterations=100, batch_size=32")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status
    """
    return {
        "status": "healthy",
        "client_id": "client2"
    }


@app.post("/train", response_model=TrainResponse)
async def train():
    """
    Run local training on client's private data.
    
    Returns:
        Trained model parameters {w, b}
    """
    if trainer is None or X_train is None or y_train is None:
        raise HTTPException(status_code=500, detail="Trainer not initialized")
    
    try:
        print("\n🚀 Client 2 starting training...")
        
        # Reset model parameters for this round
        trainer.w = np.random. randn().astype(np. float32) * 0.1
        trainer.b = np. random.randn().astype(np.float32) * 0.1
        
        # Train the model
        result = trainer. train(X_train, y_train)
        
        print(f"✅ Client 2 training complete")
        print(f"   Final w={result['w']:.4f}, b={result['b']:.4f}")
        
        return {
            "w": result['w'],
            "b": result['b'],
            "client_id": "client2",
            "status": "success"
        }
    
    except Exception as e:
        print(f"❌ Client 2 training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )