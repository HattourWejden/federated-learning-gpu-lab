"""
Aggregator Service: Central hub for federated learning. 
Receives local model parameters from clients and computes global average.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__). parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import numpy as np
from shared.utils import aggregate_models

# Initialize FastAPI app
app = FastAPI(
    title="Federated Learning Aggregator",
    description="Central aggregator for federated learning",
    version="1.0.0"
)


# Request/Response models
class ClientUpdate(BaseModel):
    """Model update from a single client."""
    client_id: str
    w: float
    b: float


class AggregateRequest(BaseModel):
    """Request to aggregate client updates."""
    updates: List[ClientUpdate]


class AggregateResponse(BaseModel):
    """Global model after aggregation."""
    global_w: float
    global_b: float
    num_clients: int
    status: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status
    """
    return {
        "status": "healthy",
        "service": "aggregator"
    }


@app.post("/aggregate", response_model=AggregateResponse)
async def aggregate(request: AggregateRequest):
    """
    Aggregate model parameters from multiple clients.
    
    Computes:
    - Global w = average of all client w values
    - Global b = average of all client b values
    
    Args:
        request: AggregateRequest with list of client updates
    
    Returns:
        AggregateResponse with global parameters
    """
    try:
        if not request.updates:
            raise HTTPException(status_code=400, detail="No client updates provided")
        
        print("\n🔄 Aggregator starting aggregation...")
        print(f"   Received {len(request.updates)} client updates")
        
        # Log individual client updates
        for update in request.updates:
            print(f"   {update.client_id}: w={update.w:.4f}, b={update.b:.4f}")
        
        # Convert to dict format for aggregation function
        client_updates = [
            {"w": update.w, "b": update.b}
            for update in request.updates
        ]
        
        # Aggregate models
        global_model = aggregate_models(client_updates)
        
        print(f"\n✅ Aggregation complete")
        print(f"   Global w = {global_model['w']:.4f}")
        print(f"   Global b = {global_model['b']:.4f}")
        print(f"   Target: w=3.5, b=2. 0")
        
        return {
            "global_w": global_model['w'],
            "global_b": global_model['b'],
            "num_clients": len(request.updates),
            "status": "success"
        }
    
    except Exception as e:
        print(f"❌ Aggregation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9000,
        log_level="info"
    )