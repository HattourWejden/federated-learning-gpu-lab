import numpy as np
from fastapi import FastAPI
from numba import cuda

app = FastAPI(title="Federated Learning Client 1")

# -----------------------------
# 1. CUDA kernel pour gradients
# -----------------------------
@cuda.jit
def compute_gradients_kernel(x, y, w, b, grad_w_partial, grad_b_partial):
    """
    Chaque thread traite un échantillon et ajoute sa contribution aux gradients.
    x, y : tableaux de données (device)
    w, b : scalaires (valeurs actuelles des paramètres)
    grad_w_partial, grad_b_partial : tableaux 1D de taille 1, utilisés comme accumulateurs
    """
    i = cuda.grid(1)
    if i < x.size:
        xi = x[i]
        yi = y[i]

        # Prédiction et erreur
        y_pred = w * xi + b
        e = y_pred - yi

        # Contributions aux gradients (avant normalisation par N)
        gw = e * xi
        gb = e

        # Accumulation atomique des gradients partiels
        cuda.atomic.add(grad_w_partial, 0, gw)
        cuda.atomic.add(grad_b_partial, 0, gb)


def local_train_gpu(num_samples=1024, num_epochs=50, lr=0.01, noise_std=0.1):
    """
    Version GPU de l'entraînement local.
    """
    true_w = 3.5
    true_b = 2.0

    xs = np.random.uniform(-5, 5, size=num_samples).astype(np.float32)
    noise = np.random.normal(0, noise_std, size=num_samples).astype(np.float32)
    ys = true_w * xs + true_b + noise

    w = np.float32(0.0)
    b = np.float32(0.0)
    N = np.float32(num_samples)

    d_x = cuda.to_device(xs)
    d_y = cuda.to_device(ys)

    threads_per_block = 128
    blocks_per_grid = (num_samples + threads_per_block - 1) // threads_per_block

    for epoch in range(num_epochs):
        d_grad_w = cuda.device_array(shape=1, dtype=np.float32)
        d_grad_b = cuda.device_array(shape=1, dtype=np.float32)

        d_grad_w[0] = 0.0
        d_grad_b[0] = 0.0

        compute_gradients_kernel[blocks_per_grid, threads_per_block](d_x, d_y, w, b, d_grad_w, d_grad_b)
        cuda.synchronize()

        grad_w_sum = d_grad_w.copy_to_host()[0]
        grad_b_sum = d_grad_b.copy_to_host()[0]

        grad_w = (2.0 / N) * grad_w_sum
        grad_b = (2.0 / N) * grad_b_sum

        w = w - lr * grad_w
        b = b - lr * grad_b

    return float(w), float(b)


def local_train_cpu(num_samples=1024, num_epochs=50, lr=0.01, noise_std=0.1):
    """
    Version CPU de l'entraînement local (sans Numba, pure NumPy).
    Utilise les mêmes équations de gradient pour reproduire le comportement.
    """
    true_w = 3.5
    true_b = 2.0

    xs = np.random.uniform(-5, 5, size=num_samples).astype(np.float32)
    noise = np.random.normal(0, noise_std, size=num_samples).astype(np.float32)
    ys = true_w * xs + true_b + noise

    w = 0.0
    b = 0.0
    N = float(num_samples)

    for epoch in range(num_epochs):
        y_pred = w * xs + b
        e = y_pred - ys  # (N,)

        grad_w = (2.0 / N) * np.sum(e * xs)
        grad_b = (2.0 / N) * np.sum(e)

        w = w - lr * grad_w
        b = b - lr * grad_b

    return float(w), float(b)


@app.get("/health")
def health():
    return {"status": "ok", "cuda_available": cuda.is_available()}


@app.post("/train")
def train():
    """
    Lance un entraînement local et retourne les paramètres appris.
    Si CUDA est dispo, utilise le GPU, sinon fallback CPU.
    """
    try:
        if cuda.is_available():
            w, b = local_train_gpu()
            backend = "gpu"
        else:
            w, b = local_train_cpu()
            backend = "cpu"

        return {"w": w, "b": b, "backend": backend}
    except Exception as e:
        # Pour ne pas planter l'API, on renvoie l'erreur dans le JSON
        return {"error": "training_failed", "details": str(e)}