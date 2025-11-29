from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Federated Learning Aggregator")

class ClientWeights(BaseModel):
    w: float
    b: float

class AggregateRequest(BaseModel):
    clients: List[ClientWeights]

class AggregateResponse(BaseModel):
    global_w: float
    global_b: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/aggregate", response_model=AggregateResponse)
def aggregate(req: AggregateRequest):
    """
    Calcule la moyenne des paramètres (w, b) reçus des clients.
    """
    num_clients = len(req.clients)
    if num_clients == 0:
        return AggregateResponse(global_w=0.0, global_b=0.0)

    sum_w = 0.0
    sum_b = 0.0
    for cw in req.clients:
        sum_w += cw.w
        sum_b += cw.b

    global_w = sum_w / num_clients
    global_b = sum_b / num_clients

    return AggregateResponse(global_w=global_w, global_b=global_b)