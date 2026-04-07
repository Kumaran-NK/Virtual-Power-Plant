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

import contextvars
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
from server.vpp_environment import VppEnvironment, get_current_env_instance
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
# Helper: access current environment instance (request-local, thread-safe)
# ---------------------------------------------------------------------------

def get_current_env() -> VppEnvironment:
    """
    Retrieve the current environment instance for this request/session.
    Uses contextvars for thread-safe, async-compatible storage.
    Raises HTTPException if no environment is active.
    """
    env = get_current_env_instance()
    if env is None or env.state is None:
        raise HTTPException(status_code=400, detail="Environment not initialised — call /reset first.")
    return env


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

    Returns a ParetoScore with:
      profit_score, safety_score, carbon_score, degradation_score, dr_score
      + weighted aggregate_score in [0.0, 1.0]
    Weights: 0.50 profit | 0.20 safety | 0.15 carbon | 0.10 degradation | 0.05 DR
    """
    try:
        env = get_current_env()
        pareto = env.get_pareto_score()
        result = pareto.model_dump()
        result["score"] = pareto.aggregate_score
        return result
    except HTTPException:
        # No active episode
        return {
            "aggregate_score": 0.0,
            "score": 0.0,
            "profit_score": 0.0,
            "safety_score": 1.0,
            "carbon_score": 0.0,
            "degradation_score": 1.0,
            "dr_score": 0.0,
            "detail": "No episode in progress.",
        }


# ---------------------------------------------------------------------------
# POST /trace — reasoning quality scoring
# ---------------------------------------------------------------------------

@app.post("/trace")
async def submit_trace(action: VppAction, reasoning: str = Query(...)):
    """
    Submit a reasoning trace alongside an action for LLM quality scoring.

    Accepts:
      - action: as JSON body (VppAction object)
      - reasoning: as query parameter (e.g., ?reasoning=agent_explanation)

    The trace is stored server-side and evaluated at episode end (GET /grader
    returns reasoning_quality_score when traces are present).

    The scoring LLM checks:
      - Does the reasoning correctly identify the relevant market signals?
      - Is the chosen action consistent with the stated reasoning?
      - Is the reserve management justified?
    """
    env = get_current_env()
    if env.state is None:
        raise HTTPException(status_code=400, detail="Call /reset before /trace.")

    # Inject reasoning into action and step
    action.reasoning = reasoning
    obs, reward, done, info = env.step(action)

    traces = env.get_reasoning_traces()
    return {
        "observation": obs,
        "reward":      reward,
        "done":        done,
        "info":        info,
        "trace_count": len(traces),
        "reasoning_stored": True,
    }


@app.get("/traces")
async def get_traces():
    """Return all stored reasoning traces for the current episode."""
    env = get_current_env()
    return {"traces": env.get_reasoning_traces()}


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