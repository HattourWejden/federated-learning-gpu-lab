import requests

CLIENT_URLS = [
    "http://localhost:8001",
    "http://localhost:8002",
    "http://localhost:8003",
]

AGGREGATOR_URL = "http://localhost:9000"


def run_one_round(round_idx: int):
    print(f"\n=== Federated Round {round_idx} ===")

    client_weights = []

    # 1. Call /train on each client
    for i, base_url in enumerate(CLIENT_URLS, start=1):
        resp = requests.post(f"{base_url}/train")
        resp.raise_for_status()
        data = resp.json()
        w = data["w"]
        b = data["b"]
        backend = data.get("backend", "unknown")
        print(f"Client {i}: w={w:.4f}, b={b:.4f}, backend={backend}")
        client_weights.append({"w": w, "b": b})

    # 2. Aggregate
    agg_payload = {"clients": client_weights}
    resp = requests.post(f"{AGGREGATOR_URL}/aggregate", json=agg_payload)
    resp.raise_for_status()
    agg = resp.json()

    global_w = agg["global_w"]
    global_b = agg["global_b"]
    print(f"Global model: w={global_w:.4f}, b={global_b:.4f}")

    return global_w, global_b


def main(num_rounds: int = 5):
    print("Starting Federated Learning experiment...")
    for r in range(1, num_rounds + 1):
        run_one_round(r)


if __name__ == "__main__":
    main(num_rounds=5)