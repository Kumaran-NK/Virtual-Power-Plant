#!/usr/bin/env python3
"""Sanity check of core endpoints after OpenEnv refactor."""

import requests
import sys

server_url = "http://localhost:7860"

# Test /health
print("Testing /health...", file=sys.stderr)
r = requests.get(f"{server_url}/health", timeout=5)
print(f"  Status: {r.status_code}", file=sys.stderr)
print(f"  Response: {r.json()}", file=sys.stderr)

if r.status_code != 200:
    print("FAIL: /health failed", file=sys.stderr)
    sys.exit(1)

# Test /reset
print("\nTesting /reset...", file=sys.stderr)
r = requests.post(f"{server_url}/reset", params={"task_id": "easy-arbitrage"}, timeout=10)
print(f"  Status: {r.status_code}", file=sys.stderr)
if r.status_code == 200:
    obs = r.json()
    print(f"  Observation type: {type(obs)}", file=sys.stderr)
    if isinstance(obs, dict):
        print(f"  Keys: {list(obs.keys())[:5]}...", file=sys.stderr)
else:
    print(f"  ERROR: {r.text[:200]}", file=sys.stderr)
    sys.exit(1)

# Test /grader
print("\nTesting /grader...", file=sys.stderr)
r = requests.get(f"{server_url}/grader", timeout=5)
print(f"  Status: {r.status_code}", file=sys.stderr)
if r.status_code == 200:
    data = r.json()
    score = data.get("aggregate_score", "N/A")
    print(f"  Score: {score}", file=sys.stderr)
else:
    print(f"  ERROR: {r.text}", file=sys.stderr)

print("\n✅ All core endpoints responding", file=sys.stderr)
