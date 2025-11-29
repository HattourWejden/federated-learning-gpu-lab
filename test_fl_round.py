"""
Test script for federated learning round. 
Calls all client services and then aggregates results.
"""

import requests
import json
import time
from typing import Dict, List

# Service URLs
CLIENT_1_URL = "http://localhost:8001"
CLIENT_2_URL = "http://localhost:8002"
CLIENT_3_URL = "http://localhost:8003"
AGGREGATOR_URL = "http://localhost:9000"

CLIENT_URLS = [CLIENT_1_URL, CLIENT_2_URL, CLIENT_3_URL]


def health_check(url: str, service_name: str) -> bool:
    """Check if a service is healthy."""
    try:
        response = requests.get(f"{url}/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ {service_name} is healthy")
            return True
        else:
            print(f"❌ {service_name} returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ {service_name} connection failed: {str(e)}")
        return False


def train_client(url: str, client_id: str) -> Dict:
    """Train a single client."""
    try:
        print(f"\n🚀 Training {client_id}...")
        response = requests.post(f"{url}/train", timeout=60)

        if response.status_code == 200:
            result = response.json()
            print(f"✅ {client_id} training complete")
            print(f"   w={result['w']:.4f}, b={result['b']:. 4f}")
            return result
        else:
            print(f"❌ {client_id} training failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
    except Exception as e:
        print(f"❌ {client_id} training error: {str(e)}")
        return None


def aggregate_models(client_results: List[Dict]) -> Dict:
    """Send results to aggregator."""
    try:
        print(f"\n🔄 Sending results to aggregator...")

        # Prepare aggregation request
        updates = [
            {
                "client_id": result['client_id'],
                "w": result['w'],
                "b": result['b']
            }
            for result in client_results
        ]

        payload = {"updates": updates}

        response = requests.post(
            f"{AGGREGATOR_URL}/aggregate",
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Aggregation complete")
            print(f"   Global w = {result['global_w']:.4f}")
            print(f"   Global b = {result['global_b']:. 4f}")
            print(f"   Clients aggregated: {result['num_clients']}")
            return result
        else:
            print(f"❌ Aggregation failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Aggregation error: {str(e)}")
        return None


def main():
    """Run full federated learning round."""
    print("=" * 70)
    print("🎯 FEDERATED LEARNING ROUND - TEST")
    print("=" * 70)

    # Step 1: Health checks
    print("\n📋 Step 1: Health Checks")
    print("-" * 70)

    all_healthy = True
    for i, url in enumerate(CLIENT_URLS, 1):
        if not health_check(url, f"Client {i}"):
            all_healthy = False

    if not health_check(AGGREGATOR_URL, "Aggregator"):
        all_healthy = False

    if not all_healthy:
        print("\n❌ Not all services are healthy. Exiting.")
        return

    # Step 2: Train clients
    print("\n📋 Step 2: Local Training on Clients")
    print("-" * 70)

    client_results = []
    for i, url in enumerate(CLIENT_URLS, 1):
        result = train_client(url, f"Client {i}")
        if result:
            client_results. append(result)
        time.sleep(1)  # Small delay between requests

    if len(client_results) < 3:
        print(f"\n❌ Not all clients trained. Got {len(client_results)}/3")
        return

    # Step 3: Aggregate
    print("\n📋 Step 3: Federated Aggregation")
    print("-" * 70)

    aggregation_result = aggregate_models(client_results)

    # Step 4: Summary
    print("\n📋 Step 4: Results Summary")
    print("-" * 70)
    print("\nClient Results:")
    for result in client_results:
        print(f"  {result['client_id']}: w={result['w']:.4f}, b={result['b']:. 4f}")

    if aggregation_result:
        print(f"\nGlobal Model:")
        print(f"  Global w = {aggregation_result['global_w']:.4f}")
        print(f"  Global b = {aggregation_result['global_b']:.4f}")
        print(f"\nTheoretical Target:")
        print(f"  w_true = 3.5000")
        print(f"  b_true = 2.0000")
        print(f"\nError:")
        w_error = abs(aggregation_result['global_w'] - 3.5)
        b_error = abs(aggregation_result['global_b'] - 2.0)
        print(f"  w_error = {w_error:.4f}")
        print(f"  b_error = {b_error:.4f}")

    print("\n" + "=" * 70)
    print("✅ FEDERATED LEARNING ROUND COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()