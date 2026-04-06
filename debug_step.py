#!/usr/bin/env python3
"""Debug script to understand the step endpoint error."""

import json
import requests

SERVER_URL = "http://localhost:7860"
TASK_ID = "easy-arbitrage"

session = requests.Session()

# Reset
print("Resetting...")
resp = session.post(f"{SERVER_URL}/reset", params={"task_id": TASK_ID}, timeout=10)
print(f"Reset status: {resp.status_code}")
print(f"Reset response: {resp.json()}")

obs = resp.json()

# Try step with proper Pydantic model format
action_data = {
    "global_charge_rate": 0.0,
    "min_reserve_pct": 0.2,
    "defer_ev_charging": 0.0,
    "accept_dr_bid": False,
    "p2p_export_rate": 0.0,
}

print("\nSending step...")
print(f"Action: {json.dumps(action_data)}")

resp = session.post(
    f"{SERVER_URL}/step",
    json=action_data,
    headers={"Content-Type": "application/json"},
    timeout=10
)

print(f"Step status: {resp.status_code}")
print(f"Step response: {resp.text[:500]}")

if resp.status_code != 200:
    print(f"\nERROR: {resp.text}")
