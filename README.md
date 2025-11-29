# Federated Learning GPU Lab

This project implements a simple **Federated Learning** setup using **FastAPI**, **NumPy**, and **Numba (CUDA)**, with deployment via **Docker Compose**.

It simulates:

- **3 client services** performing local training (with optional GPU acceleration via Numba/CUDA).
- **1 aggregator service** computing the global model by averaging client parameters.

The ML model is a **linear regression**:

\[
y = w \cdot x + b
\]

Synthetic data on each client is generated from a “true” model:

- \( w_{\text{true}} = 3.5 \)
- \( b_{\text{true}} = 2.0 \)
- plus Gaussian noise.

---

## 1. Repository Structure

```text
federated-learning-gpu-lab/
├─ agregator/
│  ├─ aggregator.py        # Aggregator FastAPI service
│  └─ Dockerfile           # Dockerfile for the aggregator service
├─ clients/
│  ├─ client1.py           # Client 1 FastAPI service
│  ├─ client2.py           # Client 2 FastAPI service
│  ├─ client3.py           # Client 3 FastAPI service
│  └─ Dockerfile           # Dockerfile for client services
├─ requirements.txt        # Common Python dependencies
├─ docker-compose.yml      # Orchestration of 3 clients + 1 aggregator
└─ test_fl_round.py        # (Optional) Script to test one FL round locally
```

---

## 2. Client Services

Each client service (`client1.py`, `client2.py`, `client3.py`) is a **FastAPI** app exposing:

- `GET /health`
- `POST /train`

### 2.1 Endpoints

#### `GET /health`

Returns basic health information and CUDA availability:

```json
{
  "status": "ok",
  "cuda_available": false
}
```

On a GPU-enabled environment with CUDA properly installed, `cuda_available` would be `true`.

#### `POST /train`

- Generates a synthetic dataset based on the true model `y = 3.5 * x + 2.0 + noise`.
- Trains a local linear regression model `y = w * x + b`.
- Computes gradients **either on GPU or CPU**:
  - If `cuda.is_available()` → GPU path via Numba CUDA kernel.
  - Otherwise → CPU path via pure NumPy as a fallback.
- Performs several epochs of SGD.
- Returns the learned parameters and backend used.

Example response:

```json
{
  "w": 3.49,
  "b": 1.27,
  "backend": "cpu"
}
```

On a CUDA-enabled system, `backend` would be `"gpu"` and computation would be GPU-accelerated.

### 2.2 Training Logic

#### Model

- Prediction: `y_pred = w * x + b`
- Error: `e = y_pred - y`
- MSE gradients:

\[
\frac{\partial L}{\partial w} = \frac{2}{N} \sum e \cdot x, \quad
\frac{\partial L}{\partial b} = \frac{2}{N} \sum e
\]

#### GPU Path (Numba CUDA)

A CUDA kernel (one per client file) does:

```python
@cuda.jit
def compute_gradients_kernel(x, y, w, b, grad_w_partial, grad_b_partial):
    i = cuda.grid(1)
    if i < x.size:
        xi = x[i]
        yi = y[i]
        y_pred = w * xi + b
        e = y_pred - yi
        gw = e * xi
        gb = e
        cuda.atomic.add(grad_w_partial, 0, gw)
        cuda.atomic.add(grad_b_partial, 0, gb)
```

- Each thread processes one data point.
- Per-thread contributions are accumulated using `cuda.atomic.add`.
- After kernel execution, accumulated gradients are copied back to host and scaled by `(2 / N)`.

#### CPU Path (NumPy Fallback)

When CUDA is not available, the client uses:

```python
y_pred = w * xs + b
e = y_pred - ys
grad_w = (2.0 / N) * np.sum(e * xs)
grad_b = (2.0 / N) * np.sum(e)
w = w - lr * grad_w
b = b - lr * grad_b
```

This ensures the entire lab still works even without a local GPU, while keeping the CUDA code in place for GPU-enabled deployments.

---

## 3. Aggregator Service

The aggregator lives in `agregator/aggregator.py` and is a separate FastAPI service exposing:

- `GET /health`
- `POST /aggregate`

### 3.1 Endpoints

#### `GET /health`

```json
{
  "status": "ok"
}
```

#### `POST /aggregate`

Input JSON (list of client weights):

```json
{
  "clients": [
    { "w": 3.50, "b": 1.27 },
    { "w": 3.49, "b": 1.26 },
    { "w": 3.49, "b": 1.27 }
  ]
}
```

The aggregator computes simple averages:

\[
w_{\text{global}} = \frac{1}{K} \sum_{k=1}^K w_k,\quad
b_{\text{global}} = \frac{1}{K} \sum_{k=1}^K b_k
\]

and returns:

```json
{
  "global_w": 3.50,
  "global_b": 1.27
}
```

This implements a basic **FedAvg** (federated averaging) step.

---

## 4. Running Locally (Without Docker)

In four separate terminals, from the project root:

```bash
# Terminal 1
uvicorn clients.client1:app --host 0.0.0.0 --port 8001 --reload

# Terminal 2
uvicorn clients.client2:app --host 0.0.0.0 --port 8002 --reload

# Terminal 3
uvicorn clients.client3:app --host 0.0.0.0 --port 8003 --reload

# Terminal 4
uvicorn agregator.aggregator:app --host 0.0.0.0 --port 9000 --reload
```

### 4.1 Example: One Federated Round (PowerShell)

```powershell
# Call local training on each client
$w1 = Invoke-WebRequest -Uri http://localhost:8001/train -Method POST | ConvertFrom-Json
$w2 = Invoke-WebRequest -Uri http://localhost:8002/train -Method POST | ConvertFrom-Json
$w3 = Invoke-WebRequest -Uri http://localhost:8003/train -Method POST | ConvertFrom-Json

$w1
$w2
$w3

# Build aggregation request
$body = @{
    clients = @(
        @{ w = $w1.w; b = $w1.b },
        @{ w = $w2.w; b = $w2.b },
        @{ w = $w3.w; b = $w3.b }
    )
} | ConvertTo-Json

# Call the aggregator
$response = Invoke-WebRequest -Uri http://localhost:9000/aggregate -Method POST -Body $body -ContentType "application/json"
$response.Content
```

Typical outputs:

- Clients:
  - `client1`: `w ≈ 3.50`, `b ≈ 1.27`, `"backend": "cpu"`
  - `client2`: `w ≈ 3.49`, `b ≈ 1.27`, `"backend": "cpu"`
  - `client3`: `w ≈ 3.49`, `b ≈ 1.27`, `"backend": "cpu"`
- Aggregated:
  - `global_w ≈ 3.50`
  - `global_b ≈ 1.27`

---

## 5. Running with Docker Compose

### 5.1 Dockerfiles

#### `clients/Dockerfile`

```dockerfile
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY clients/client1.py ./client1.py
COPY clients/client2.py ./client2.py
COPY clients/client3.py ./client3.py
```

#### `agregator/Dockerfile`

```dockerfile
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY agregator/aggregator.py ./aggregator.py
```

### 5.2 `docker-compose.yml`

```yaml
services:
  client1:
    build:
      context: .
      dockerfile: clients/Dockerfile
    command: uvicorn client1:app --host 0.0.0.0 --port 8001
    ports:
      - "8001:8001"

  client2:
    build:
      context: .
      dockerfile: clients/Dockerfile
    command: uvicorn client2:app --host 0.0.0.0 --port 8002
    ports:
      - "8002:8002"

  client3:
    build:
      context: .
      dockerfile: clients/Dockerfile
    command: uvicorn client3:app --host 0.0.0.0 --port 8003
    ports:
      - "8003:8003"

  aggregator:
    build:
      context: .
      dockerfile: agregator/Dockerfile
    command: uvicorn aggregator:app --host 0.0.0.0 --port 9000
    ports:
      - "9000:9000"
```

### 5.3 Start the Full Stack

From the project root:

```bash
docker compose up --build
```

Docker Desktop will show 4 running containers:

- `client1` on `8001:8001`
- `client2` on `8002:8002`
- `client3` on `8003:8003`
- `aggregator` on `9000:9000`

You can then call the same endpoints as in the local (non-Docker) case.

---

## 6. Task 5 – Observations and Discussion

### 6.1 Comparing Client and Global Models

Example client results:

- `client1`: `w ≈ 3.50`, `b ≈ 1.27`
- `client2`: `w ≈ 3.49`, `b ≈ 1.27`
- `client3`: `w ≈ 3.49`, `b ≈ 1.27`

Aggregated model:

- `global_w ≈ 3.50`
- `global_b ≈ 1.27`

Observations:

- All clients converge to **similar local models**.
- The aggregator’s model is essentially the **mean** of these local models.
- This illustrates **Federated Averaging (FedAvg)**: central server averages client parameters without seeing their raw data.

### 6.2 Proximity to Theoretical Values

- True values: `w_true = 3.5`, `b_true = 2.0`.
- Global model:
  - `global_w` is **very close** to `3.5`.
  - `global_b` is around `1.27`, clearly below `2.0`.

Reasons for the bias in `b`:

- Added Gaussian noise in the synthetic data.
- Limited dataset size and training epochs.
- Independent datasets on each client: local minima averaged are not exactly the global optimum.

Increasing `num_epochs` and `num_samples` would typically bring `b` closer to `2.0`.

### 6.3 GPU Usage

- On the current development machine:
  - `cuda_available` is `false` in `/health`.
  - `/train` returns `"backend": "cpu"`.
  - Training runs entirely on **CPU** (NumPy), even though GPU kernels are implemented.
- On a server with a proper NVIDIA GPU + CUDA + Numba setup:
  - `cuda.is_available()` becomes `True`.
  - `/train` would return `"backend": "gpu"`.
  - The Numba CUDA kernel would compute gradients on the GPU.

To monitor GPU usage on such a server (Linux):

```bash
watch nvidia-smi
```

You should see a Python/uvicorn process appear using some GPU memory and utilization when `/train` is called.

---

## 7. Possible Extensions (Bonus)

- **Multiple federated rounds**:
  - Implement a small orchestrator script that:
    - calls `/train` on all clients,
    - calls `/aggregate`,
    - (optionally) sends updated global parameters back to clients,
    - repeats for several rounds and logs convergence.

- **Different models**:
  - Replace linear regression with logistic regression on synthetic classification data.

- **Monitoring & Metrics**:
  - Expose Prometheus metrics (e.g., training time, loss, backend used).
  - Visualize convergence curves and system metrics with Grafana.

---

## 8. Technologies Used

- **Python**
- **FastAPI** – REST microservices for clients and aggregator
- **NumPy** – CPU-side numerical computations
- **Numba (CUDA)** – GPU kernels & acceleration (when CUDA is available)
- **Uvicorn** – ASGI server
- **Docker & Docker Compose** – containerized deployment
