# Federated Learning GPU Lab

A federated learning setup built with **FastAPI**, **NumPy**, **Numba (CUDA-ready)** and **Docker Compose**.

The project simulates:

- **3 client services** that train a local model on synthetic data.
- **1 aggregator service** that computes a global model by averaging client parameters.
- A **baseline with linear regression**, and a **bonus variant with logistic regression**.

> Note: On the local machine used to develop this lab, CUDA is not available, so training runs on CPU (NumPy). The code is written to support GPU acceleration via Numba CUDA when deployed on a proper GPU server.

---

## 📚 Table of Contents

1. [Overview](#-overview)
2. [Architecture](#-architecture)
3. [Implemented Models](#-implemented-models)
4. [Project Structure](#-project-structure)
5. [Prerequisites](#-prerequisites)
6. [Local Development (without Docker)](#-local-development-without-docker)
7. [Docker Deployment](#-docker-deployment)
8. [Federated Round Example](#-federated-round-example)
9. [Logistic Regression](#-logistic-regression)
10. [Monitoring GPU Usage](#-monitoring-gpu-usage)
11. [Troubleshooting](#-troubleshooting)


---

## 🎯 Overview

This lab implements a **federated learning (FL)** system where multiple clients independently train a model on their own synthetic data and share only model parameters with a central **aggregator**.

The goals are to:

- Understand **federated learning concepts and architecture**.
- Implement GPU-ready training using **Numba CUDA** (with CPU fallback).
- Build microservices with **FastAPI**.
- Deploy a distributed system with **Docker Compose**.
- Prepare the code to run on a remote **GPU server**.

### Core Idea

1. Each client:
   - Generates its own synthetic dataset.
   - Trains a local model (linear or logistic regression).
   - Exposes a REST API to trigger training (`/train`) and to check health (`/health`).

2. The aggregator:
   - Collects trained parameters from the clients.
   - Computes the **global model** by averaging parameters (`FedAvg`).
   - Exposes `/aggregate` and `/health` endpoints.

---

## 🏗 Architecture

High-level architecture:

```text
+---------------------+
|   Aggregator API    |
|   (Port 9000)       |
+----------+----------+
           |
           |  FedAvg on (w, b)
           |
   +-------+-------+
   |       |       |     
+--v--+ +--v--+ +--v--+
|cl1  | |cl2  | | cl3    |
|8001 | |8002 | | 8003   |
+----+  +----+  +-------+
   CPU/GPU-ready (Numba)
```

### Components

#### Client Services (×3)

- `clients/client1.py`
- `clients/client2.py`
- `clients/client3.py`

Each client:

- Generates **synthetic data**.
- Trains a model (linear or logistic, depending on branch/version).
- Uses:
  - **GPU path**: Numba CUDA kernel (if `cuda.is_available()` is `True`).
  - **CPU path**: NumPy implementation as fallback.
- Exposes:
  - `GET /health`
  - `POST /train`

#### Aggregator Service

- `agregator/aggregator.py`

Responsibilities:

- Receives client parameters `(w, b)`.
- Computes the **global model** via simple averaging.
- Exposes:
  - `GET /health`
  - `POST /aggregate`

---

## 🧠 Implemented Models

### 1. Linear Regression (Base Lab)

- Model:  
  `y = w * x + b`
- Synthetic data:
  - `x` uniformly sampled.
  - `y = 3.5 * x + 2.0 + noise`.
- Clients train with gradient descent.
- Aggregator averages `w` and `b`.

### 2. Logistic Regression (Bonus)

- Model (1D binary classification):

  ```text
  z = w * x + b
  y_pred = sigmoid(z) = 1 / (1 + exp(-z))
  ```

- Synthetic data:
  - `x` drawn uniformly.
  - True model uses `w_true = 3.5`, `b_true = -1.0`.
  - Probabilities: `p = sigmoid(w_true * x + b_true)`.
  - Labels: `y ~ Bernoulli(p)`.
- Loss: **binary cross-entropy**.
- Gradients for each sample:
  - `e = y_pred - y`
  - `dL/dw = (1/N) * Σ(e * x)`
  - `dL/db = (1/N) * Σ(e)`

Each client learns a local logistic model; the aggregator still averages `(w, b)`.

---

## 📁 Project Structure

```text
federated-learning-gpu-lab/
├─ agregator/
│  ├─ aggregator.py         # Aggregator FastAPI service (/health, /aggregate)
│  └─ Dockerfile            # Aggregator Docker image
├─ clients/
│  ├─ client1.py            # Client 1 FastAPI service
│  ├─ client2.py            # Client 2 FastAPI service
│  ├─ client3.py            # Client 3 FastAPI service
│  └─ Dockerfile            # Client Docker image (shared by the 3 clients)
├─ requirements.txt         # Python dependencies (FastAPI, NumPy, Numba, etc.)
├─ docker-compose.yml       # Orchestration of 3 clients + 1 aggregator
├─ test_fl_round.py         # Script to run multiple FL rounds (CPU calls to APIs)
└─ debug_logistic.py        # Standalone script to debug / verify logistic regression
```

---

## ✅ Prerequisites

### Local Development

- **Python** 3.10+ (3.11 used)
- Python packages (from `requirements.txt`):
  - `fastapi`
  - `uvicorn[standard]`
  - `numpy`
  - `numba`
  - `pydantic`
  - `requests` (for `test_fl_round.py` and debug scripts)

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### For GPU Acceleration (on remote server)

- NVIDIA GPU with CUDA support
- NVIDIA drivers
- CUDA Toolkit compatible with Numba
- NVIDIA Container Toolkit (for Docker GPU support)

> On the development machine used for this lab, `cuda.is_available()` is `False`, so all training runs on CPU. The CUDA kernels are implemented and ready to use on a suitable GPU environment.

---

## 🧪 Local Development (without Docker)

You can run all services directly with **uvicorn**.

From the project root:

```bash
# Terminal 1: Client 1
uvicorn clients.client1:app --host 0.0.0.0 --port 8001 --reload

# Terminal 2: Client 2
uvicorn clients.client2:app --host 0.0.0.0 --port 8002 --reload

# Terminal 3: Client 3
uvicorn clients.client3:app --host 0.0.0.0 --port 8003 --reload

# Terminal 4: Aggregator
uvicorn agregator.aggregator:app --host 0.0.0.0 --port 9000 --reload
```

Check health endpoints:

```bash
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health
curl http://localhost:9000/health
```

Trigger training on a single client:

```bash
curl -X POST http://localhost:8001/train
```

---

## 🐳 Docker Deployment

All services are containerized using **Docker Compose**.

### 1. Build and Start

From the project root:

```bash
docker compose up --build
```

This will:

- Build a **client image** from `clients/Dockerfile` and spin up:
  - `client1` (port `8001:8001`)
  - `client2` (port `8002:8002`)
  - `client3` (port `8003:8003`)
- Build an **aggregator image** from `agregator/Dockerfile` and spin up:
  - `aggregator` (port `9000:9000`)

Check running services:

```bash
docker compose ps
```

View logs:

```bash
docker compose logs -f client1
docker compose logs -f aggregator
```

Stop everything:

```bash
docker compose down
```

---

## 🔁 Federated Round Example

A simple PowerShell example (works from your host while containers are running):

```powershell
# 1. Train all 3 clients
$w1 = Invoke-WebRequest -Uri http://localhost:8001/train -Method POST | ConvertFrom-Json
$w2 = Invoke-WebRequest -Uri http://localhost:8002/train -Method POST | ConvertFrom-Json
$w3 = Invoke-WebRequest -Uri http://localhost:8003/train -Method POST | ConvertFrom-Json

$w1
$w2
$w3

# 2. Build aggregation payload
$body = @{
    clients = @(
        @{ w = $w1.w; b = $w1.b },
        @{ w = $w2.w; b = $w2.b },
        @{ w = $w3.w; b = $w3.b }
    )
} | ConvertTo-Json

# 3. Call aggregator
$response = Invoke-WebRequest -Uri http://localhost:9000/aggregate -Method POST -Body $body -ContentType "application/json"
$response.Content
```

Typical behavior:

- Each client returns a slightly different `(w, b)` because it uses its **own synthetic dataset**.
- The aggregator computes `(global_w, global_b)` as the **average** of the 3 clients.
- In the linear regression version, `global_w` is close to `3.5`, `global_b` close to `2.0`.  
- In the logistic regression version, the global parameters are in the same region as the generating values (e.g. `w ≈ 3`, `b ≈ -0.6` vs `w_true = 3.5`, `b_true = -1.0`), due to sampling and finite training steps.

---

## Logistic Regression

As a bonus, the client training was modified to use **logistic regression** instead of linear regression.

### Logistic Client Behavior

For each `/train` call:

1. Generate binary labels with a sigmoid-based probabilistic model.
2. Train a logistic regression model with gradient descent.
3. Return:

```json
{
  "w": 2.9,
  "b": -0.6,
  "backend": "cpu",
  "true_w": 3.5,
  "true_b": -1.0
}
```

Observations:

- The learned parameters converge around `w ≈ 3.0` and `b ≈ -0.6`.
- The true generating parameters are `(3.5, -1.0)`.
- Because data are randomly sampled and finite, the maximum-likelihood solution is in the same parameter region but not exactly equal to the true values.
- Aggregating across three logistic clients with FedAvg still gives a **meaningful global classifier**.

`test_fl_round.py` can be used to run multiple federated rounds programmatically and log the evolution of the global parameters.

---

## 📈 Monitoring GPU Usage

On a GPU-enabled Linux server:

```bash
watch -n 1 nvidia-smi
```

During calls to `/train` (when `cuda.is_available() == True`), you should see:

- A Python / uvicorn process using GPU memory.
- Non-zero GPU utilization.

From inside a GPU-enabled container:

```bash
nvidia-smi
```

> On the development machine used here, `cuda_available` was `false`, so training ran on CPU (`backend: "cpu"`). The same code can be executed on a remote NVIDIA GPU server once access is provided by the instructor.

---

## 🛠 Troubleshooting

### CUDA not available

- Check `/health`:

  ```bash
  curl http://localhost:8001/health
  ```

  If `cuda_available` is `false`:

  - On local dev: this is expected if you don't have CUDA / GPU configured.
  - On remote GPU server: verify driver and CUDA installation.

### Ports already in use

If `8001`, `8002`, `8003`, or `9000` are busy:

- Find the process:

  ```bash
  # Linux
  lsof -i :8001

  # Windows (PowerShell)
  netstat -ano | findstr 8001
  ```

- Kill it or change ports in `docker-compose.yml`.

### Service not responding

- Check container status:

  ```bash
  docker compose ps
  ```

- View logs:

  ```bash
  docker compose logs -f client1
  docker compose logs -f aggregator
  ```
##  Author

Lab completed by: **Wejden Hattour**  
**Federated Learning GPU Lab (3 clients + 1 aggregator, Docker, logistic regression)**
