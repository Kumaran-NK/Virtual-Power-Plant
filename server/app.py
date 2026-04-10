# server/app.py
"""
FastAPI application — VPP OpenEnv HTTP interface, Extended Edition.

Uses OpenEnv core create_app() for lifecycle and session management.

Custom OpenEnv-compatible endpoints:
  POST /trace          Submit reasoning trace (LLM-scored quality)
  GET  /grader         Returns ParetoScore (multi-objective)
  GET  /tasks          Lists all 5 tasks + schemas
  GET  /traces         Returns stored reasoning traces
  GET  /baseline       Returns pre-computed baseline scores
"""

import json
import os
import subprocess
import sys
import threading
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from openenv.core import create_app as create_openenv_app

from models import VppAction, VppObservation, VppState, ParetoScore, VppReward
from server.vpp_environment import VppEnvironment
from server.task_curves import ALL_TASK_IDS, TASK_METADATA


# ---------------------------------------------------------------------------
# Create OpenEnv app with lifecycle and session management
# ---------------------------------------------------------------------------

app: FastAPI = create_openenv_app(
    env=lambda: VppEnvironment(),  # Factory function for environment instances
    action_cls=VppAction,
    observation_cls=VppObservation,
    env_name="vpp",
    max_concurrent_envs=16,
)

# Global state for baseline computation (not per-session)
_baseline_lock    = threading.Lock()
_baseline_running = False
_baseline_result  = None



# ---------------------------------------------------------------------------
# IMPORTANT: Custom endpoints usage
# ---------------------------------------------------------------------------
# The custom endpoints (/trace, /grader, /traces) require STATEFUL sessions.
# OpenEnv provides state via WebSocket (/ws endpoint with MCP protocol).
# 
# ✅ RECOMMENDED: Use the VppEnv WebSocket client:
#   from client import VppEnv
#   with VppEnv(base_url="http://localhost:7860") as client:
#       env.reset(task_id="easy-arbitrage")
#       env.step(action)
#
# ❌ NOT SUPPORTED: Custom endpoints via raw HTTP POST
#   Each HTTP request creates a fresh environment with no prior state.
#
# If you're seeing "Environment not initialized" errors:
#   → You're using HTTP POST instead of WebSocket
#   → Switch to VppEnv client which auto-detects and uses WebSocket
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Custom Endpoints (OpenEnv-compatible extensions)
# ---------------------------------------------------------------------------
# Note: create_app already provides /health, /reset, /step, /state, /tasks (basic)
# These custom endpoints enhance or extend the core functionality.


@app.get("/tasks")
async def get_tasks_enhanced():
    """
    List all 5 available tasks with detailed metadata and schemas.
    Extends the basic OpenEnv /tasks endpoint with additional fields.
    """
    tasks_out = []
    for tid, meta in TASK_METADATA.items():
        tasks_out.append({
            "id":                   tid,
            "description":         meta["description"],
            "difficulty":          meta["difficulty"],
            "profit_target_usd":   meta["profit_target_usd"],
            "carbon_target_credits": meta["carbon_target_credits"],
            "has_islanding":       meta["has_islanding"],
            "has_dr_auction":      meta["has_dr_auction"],
            "weather":             meta["weather"],
        })
    return {
        "tasks":              tasks_out,
        "action_schema":      VppAction.model_json_schema(),
        "observation_schema": VppObservation.model_json_schema(),
        "reward_schema":      VppReward.model_json_schema(),
        "pareto_score_schema": ParetoScore.model_json_schema(),
    }


@app.get("/grader")
async def get_grader_score():
    """
    Return the deterministic multi-objective Pareto score for the current episode.

    ⚠️  REQUIRES WebSocket connection (use VppEnv client, not HTTP POST).
    HTTP POST requests are stateless and won't have an active episode.

    Returns a ParetoScore with:
      profit_score, safety_score, carbon_score, degradation_score, dr_score
      + weighted aggregate_score in [0.0, 1.0]
    Weights: 0.50 profit | 0.20 safety | 0.15 carbon | 0.10 degradation | 0.05 DR
    """
    raise HTTPException(
        status_code=400,
        detail=(
            "No active HTTP episode state available for /grader. "
            "This OpenEnv server version uses stateless HTTP /reset and /step; "
            "use the WebSocket client (VppEnv) for session-scoped grading, "
            "or consume per-step pareto metadata emitted in observations."
        ),
    )


# ---------------------------------------------------------------------------
# POST /trace — reasoning quality scoring
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# POST /trace — reasoning quality scoring (WebSocket only)
# ---------------------------------------------------------------------------
# NOTE: This endpoint is designed for WebSocket connections where sessions 
# have persistent state. For HTTP POST, each request is stateless.
#
# Usage:
#   1. Via WebSocket (VppEnv client):
#      action = VppAction(..., reasoning="explanation...")
#      result = client.step(action)  # Reasoning stored automatically
#   
#   2. Get stored traces:
#      # Access via VppEnvironment.get_reasoning_traces() if exposed via WebSocket
#
# HTTP POST requests can submit reasoning, but won't have prior episode state.

@app.post("/trace")
async def submit_trace(action: VppAction, reasoning: str = Query(...)):
    """
    Submit a reasoning trace alongside an action for LLM quality scoring.

    ⚠️  BEST USED VIA WEBSOCKET (use VppEnv client with .sync() wrapper).
    HTTP POST is stateless and won't maintain session state across requests.

    Accepts:
      - action: as JSON body (VppAction object)
      - reasoning: as query parameter (e.g., ?reasoning=agent_explanation)

    The trace is stored server-side if a session exists (via WebSocket).
    Otherwise, returns error explaining stateless HTTP.
    """
    raise HTTPException(
        status_code=400, 
        detail="POST /trace requires WebSocket (stateful session). "
               "For stateful reasoning tracking: "
               "(1) Use VppEnv client with WebSocket: "
               "    client = VppEnv(base_url='http://localhost:7860').sync(); "
               "    action = VppAction(..., reasoning='explanation'); "
               "    client.step(action)  # Reasoning stored automatically "
               "(2) HTTP POST /step and POST /trace are stateless (each request creates fresh env)"
    )


@app.get("/traces")
async def get_traces():
    """Return all stored reasoning traces for the current episode.
    
    ⚠️  REQUIRES WebSocket connection (use VppEnv client, not HTTP POST).
    HTTP POST requests are stateless and won't have prior traces.
    
    Returns:
        {traces: List[dict]} - Stored reasoning traces with step numbers and scores
    """
    raise HTTPException(
        status_code=400,
        detail="GET /traces requires WebSocket (stateful session). "
               "Use VppEnv client with WebSocket to maintain reasoning traces. "
               "Traces are automatically stored when VppAction includes reasoning field."
    )


# ---------------------------------------------------------------------------
# /baseline — pre-computed and live baseline scoring
# ---------------------------------------------------------------------------

def _run_baseline_subprocess() -> dict:
    """Trigger baseline_inference.py as subprocess and store results."""
    global _baseline_result, _baseline_running

    baseline_script = os.path.join(os.path.dirname(__file__), "..", "baseline_inference.py")
    baseline_script = os.path.abspath(baseline_script)
    env_vars = {**os.environ, "VPP_SERVER_URL": "http://localhost:7860"}

    try:
        result = subprocess.run(
            [sys.executable, baseline_script, "--json-only"],
            capture_output=True, text=True, timeout=300, env=env_vars,
        )
        if result.returncode == 0 and result.stdout.strip():
            scores = json.loads(result.stdout.strip())
            out_path = os.path.join(os.path.dirname(__file__), "..", "baseline_scores.json")
            with open(out_path, "w") as f:
                json.dump(scores, f, indent=2)
            _baseline_result = scores
            return scores
        else:
            return {"error": "Baseline script returned non-zero", "details": (result.stderr or "")[:500]}
    except subprocess.TimeoutExpired:
        return {"error": "Baseline computation timed out (>300 s)."}
    except json.JSONDecodeError as e:
        return {"error": f"Could not parse baseline output: {e}"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        with _baseline_lock:
            _baseline_running = False


@app.get("/baseline")
async def get_baseline(
    refresh: bool = Query(False, description="Set to true to recompute live."),
):
    """
    Return pre-computed baseline scores or trigger live recomputation.

    When refresh=true, asynchronously runs baseline_inference.py and returns results.
    Otherwise, returns pre-stored baseline_scores.json if available.
    """
    global _baseline_running, _baseline_result

    if refresh:
        with _baseline_lock:
            if _baseline_running:
                return JSONResponse(
                    status_code=202,
                    content={"status": "Baseline already running. Check back shortly."},
                )
            _baseline_running = True
        return _run_baseline_subprocess()

    if _baseline_result is not None:
        return _baseline_result

    baseline_path = os.path.join(os.path.dirname(__file__), "..", "baseline_scores.json")
    try:
        with open(baseline_path, "r") as f:
            scores = json.load(f)
        _baseline_result = scores
        return scores
    except FileNotFoundError:
        empty = {tid: {"aggregate_score": 0.0, "note": "Run baseline_inference.py"} for tid in ALL_TASK_IDS}
        return empty


# ---------------------------------------------------------------------------
# Entry points for openenv validate and direct execution
# ---------------------------------------------------------------------------

def main():
    """
    Zero-argument entry point for openenv validate.
    Reads HOST and PORT from environment variables (defaults: 0.0.0.0, 7860).
    """
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host=host, port=port)


def run_server(host: str = "0.0.0.0", port: int = 7860):
    """Direct entry point for running the server with custom host/port."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "7860")))
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)
