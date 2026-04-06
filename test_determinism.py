#!/usr/bin/env python3
"""
Quick determinism test for grader.
Runs the same task twice with identical actions and verifies scores match.
"""

import json
import requests
import sys

SERVER_URL = "http://localhost:7860"
TASK_ID = "easy-arbitrage"

def test_grader_determinism():
    """Test that running the same episode twice produces identical scores."""
    session = requests.Session()
    
    # Baseline action (rule agent would choose)
    baseline_action = {
        "accept_dr_bid": False,
        "defer_ev_charging": 0.0,
        "global_charge_rate": 0.0,
        "min_reserve_pct": 0.2,
        "p2p_export_rate": 0.0,
    }
    
    scores = []
    
    for run in range(2):
        print(f"\n--- Run {run + 1} ---", file=sys.stderr)
        
        # Reset environment
        resp = session.post(f"{SERVER_URL}/reset", params={"task_id": TASK_ID}, timeout=10)
        resp.raise_for_status()
        obs = resp.json()
        
        total_reward = 0.0
        step_count = 0
        done = False
        
        # Run 5 steps with baseline action
        for step in range(5):
            resp = session.post(f"{SERVER_URL}/step", json=baseline_action, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            obs = data["observation"]
            reward = float(data["reward"])
            done = bool(data["done"])
            total_reward += reward
            step_count += 1
            
            print(f"  Step {step_count}: reward={reward:.2f}, done={done}", file=sys.stderr)
            
            if done:
                break
        
        # Get grader score
        resp = session.get(f"{SERVER_URL}/grader", timeout=10)
        resp.raise_for_status()
        grader_data = resp.json()
        
        score = float(grader_data.get("aggregate_score", 0.0))
        scores.append(score)
        
        print(f"  Aggregate Score: {score:.6f}", file=sys.stderr)
        print(f"  Total Reward: {total_reward:.2f}", file=sys.stderr)
    
    session.close()
    
    # Check determinism
    score1, score2 = scores[0], scores[1]
    delta = abs(score1 - score2)
    
    print(f"\n=== DETERMINISM CHECK ===", file=sys.stderr)
    print(f"Run 1 score: {score1:.6f}", file=sys.stderr)
    print(f"Run 2 score: {score2:.6f}", file=sys.stderr)
    print(f"Delta:       {delta:.9f}", file=sys.stderr)
    
    if delta < 1e-6:
        print("\n✅ PASS: Scores are deterministic (delta < 1e-6)", file=sys.stderr)
        return 0
    else:
        print(f"\n❌ FAIL: Scores differ by {delta:.9f}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(test_grader_determinism())
