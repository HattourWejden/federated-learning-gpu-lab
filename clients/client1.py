import numpy as np
from fastapi import FastAPI
from numba import cuda

app = FastAPI(title="Federated Learning Client 1 (Logistic Regression)")


def generate_logistic_data(num_samples: int):
    w_true = 3.5
    b_true = -1.0

    xs = np.random.uniform(-5, 5, size=num_samples).astype(np.float32)

    z = w_true * xs + b_true
    p = 1.0 / (1.0 + np.exp(-z))

    ys = (np.random.rand(num_samples).astype(np.float32) < p).astype(np.float32)
    return xs, ys, w_true, b_true


def local_train_cpu_logistic(num_samples=2000, num_epochs=2000, lr=0.05):
    xs, ys, w_true, b_true = generate_logistic_data(num_samples)

    w = 0.0
    b = 0.0
    N = float(num_samples)

    for epoch in range(num_epochs):
        z = w * xs + b
        y_pred = 1.0 / (1.0 + np.exp(-z))
        e = y_pred - ys

        grad_w = (1.0 / N) * np.sum(e * xs)
        grad_b = (1.0 / N) * np.sum(e)

        w -= lr * grad_w
        b -= lr * grad_b

    return float(w), float(b), float(w_true), float(b_true)


@app.get("/health")
def health():
    return {"status": "ok", "cuda_available": cuda.is_available()}


@app.post("/train")
def train():
    try:
        # For now, force CPU logistic regression
        w, b, w_true, b_true = local_train_cpu_logistic()
        backend = "cpu"

        return {
            "w": w,
            "b": b,
            "backend": backend,
            "true_w": w_true,
            "true_b": b_true,
        }
    except Exception as e:
        return {"error": "training_failed", "details": str(e)}