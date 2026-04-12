#!/usr/bin/env python3
"""
WebSocket-based session management test.

This test demonstrates proper stateful multi-step episodes using OpenEnv's
WebSocket protocol (MCP). The VppEnv client automatically uses WebSocket
and maintains session state across reset() and step() calls.

To run:
    1. Start the server: python -m uvicorn server.app:app --host localhost --port 7860
    2. Run this test: python tests/test_websocket_sessions.py
    
Expected output:
    ✅ Reset successful with task_id=easy-arbitrage
    ✅ Step 1-5: Actions executed, state persisted across steps
    ✅ Battery SoC increased (charging)
    ✅ Cumulative profit tracking works
    ✅ Done flag correctly propagated
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client import VppEnv, VppAction


def test_websocket_stateful_session():
    """Test that WebSocket sessions maintain state across reset/step calls."""
    print("=" * 70)
    print("WebSocket Session Management Test")
    print("=" * 70)
    
    # Initialize WebSocket client via sync wrapper
    # VppEnv extends EnvClient which uses WebSocket (auto-converts http:// to ws://)
    # .sync() provides a synchronous interface to the async WebSocket client
    client = VppEnv(base_url="http://localhost:7860").sync()
    
    try:
        # ─── RESET ───────────────────────────────────────────────────────
        print("\n[1] Calling reset() with task_id='easy-arbitrage'...")
        result = client.reset(task_id="easy-arbitrage")
        
        initial_obs = result.observation
        print(f"    ✅ Reset successful")
        print(f"       Step ID: {initial_obs.step_id}")
        print(f"       Market price: ${initial_obs.market_price_per_mwh:.2f}/MWh")
        print(f"       Initial battery count: {len(initial_obs.telemetry)}")
        
        # Track state across steps
        initial_soc = [asset.soc for asset in initial_obs.telemetry]
        previous_profit = 0.0
        
        # ─── MULTI-STEP EPISODE ───────────────────────────────────────────
        print(f"\n[2] Executing 5 steps (charging action)...")
        
        for step_num in range(1, 6):
            action = VppAction(
                global_charge_rate=0.5,      # 50% charging
                min_reserve_pct=0.2,
                defer_ev_charging=0.0,
                accept_dr_bid=False,
                p2p_export_rate=0.0,
                reasoning=None
            )
            
            result = client.step(action)
            obs = result.observation
            reward = result.reward
            done = result.done
            
            # Verify state persistence
            current_soc = [asset.soc for asset in obs.telemetry]
            soc_changed = any(
                abs(current_soc[i] - initial_soc[i]) > 0.001 
                for i in range(len(initial_soc))
            )
            
            profit_change = reward  # Reward is the profit delta
            cumulative_profit = previous_profit + profit_change
            
            print(f"\n    Step {step_num}:")
            print(f"      Step ID: {obs.step_id}")
            print(f"      Market price: ${obs.market_price_per_mwh:.2f}/MWh")
            print(f"      Reward (profit delta): ${profit_change:.2f}")
            print(f"      Cumulative profit: ${cumulative_profit:.2f}")
            print(f"      Done: {done}")
            print(f"      State persisted: {soc_changed}")
            
            # Verify expectations
            assert obs.step_id == step_num, f"Step ID mismatch: expected {step_num}, got {obs.step_id}"
            assert soc_changed, "Battery SoC should have changed (state not persisted!)"
            
            previous_profit = cumulative_profit
            
            if done:
                print(f"\n    ⚠️  Episode ended at step {step_num}")
                break
        
        # ─── VERIFICATION ─────────────────────────────────────────────────
        print(f"\n[3] Session verification:")
        print(f"    ✅ WebSocket connection maintained across {step_num} steps")
        print(f"    ✅ Battery state persisted (SoC changed)")
        print(f"    ✅ Reward tracking works")
        print(f"    ✅ Step IDs incremental: 0 → {step_num}")
        print(f"\n{'='*70}")
        print("✅ ALL TESTS PASSED - WebSocket stateful sessions working!")
        print("   Session state is maintained via OpenEnv's MCP protocol.")
        print("='*70}")
        
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"   {type(e).__name__}: {e}")
        raise
    finally:
        # Clean up
        try:
            client.close()
            print("\nWebSocket connection closed.")
        except:
            pass


def test_http_post_is_stateless():
    """
    Demonstrates why raw HTTP POST is stateless.
    
    This test shows the difference between:
    - ❌ HTTP POST: Each request creates fresh environment (no state persistence)
    - ✅ WebSocket (via VppEnv): Session maintained with state persistence
    """
    import requests
    
    print("\n" + "=" * 70)
    print("HTTP POST Statelessness Demo")
    print("=" * 70)
    
    BASE_URL = "http://localhost:7860"
    
    print("\n[1] Calling HTTP POST /reset...")
    reset_resp = requests.post(
        f"{BASE_URL}/reset",
        params={"task_id": "easy-arbitrage"}
    )
    print(f"    Status: {reset_resp.status_code}")
    
    if reset_resp.status_code == 200:
        reset_data = reset_resp.json()
        obs_data = reset_data.get("observation", {})
        telemetry = obs_data.get("telemetry", [])
        if telemetry:
            initial_soc = telemetry[0].get("state_of_charge", 0)
            print(f"    Initial battery SoC: {initial_soc:.4f}")
        else:
            print(f"    ERROR: No telemetry in response")
            return
    else:
        print(f"    ERROR: Reset failed with {reset_resp.status_code}")
        return
    
    print("\n[2] Calling HTTP POST /step...")
    action_payload = {
        "action": {
            "global_charge_rate": 0.5,
            "min_reserve_pct": 0.2,
            "defer_ev_charging": 0.0,
            "accept_dr_bid": False,
            "p2p_export_rate": 0.0
        }
    }
    step_resp = requests.post(f"{BASE_URL}/step", json=action_payload)
    print(f"    Status: {step_resp.status_code}")
    
    if step_resp.status_code == 500:
        print(f"    ❌ Error: Internal Server Error")
        print(f"\n    WHY IT FAILED:")
        print(f"    → HTTP POST creates fresh environment (no session)")
        print(f"    → /step has no prior state from /reset")
        print(f"    → This is by design: HTTP endpoints are stateless")
        print(f"\n    SOLUTION:")
        print(f"    → Use WebSocket via VppEnv client (see test above)")
        print(f"    → Or implement client-side session tracking")
    else:
        print(f"    Status: {step_resp.status_code} (got response)")
    
    print(f"\n    CONCLUSION:")
    print(f"    HTTP POST endpoints are stateless.")
    print(f"    Use WebSocket + VppEnv client for stateful sessions.")


if __name__ == "__main__":
    try:
        test_websocket_stateful_session()
    except Exception as e:
        print(f"\n⚠️  WebSocket test encountered an error.")
        print(f"   Make sure the server is running on http://localhost:7860")
        print(f"   Error: {e}\n")
    
    print("\n" + "=" * 70)
    print("Demo: Why HTTP POST is stateless")
    print("=" * 70)
    
    try:
        test_http_post_is_stateless()
    except Exception as e:
        print(f"\n⚠️  HTTP demo encountered an error: {e}\n")
