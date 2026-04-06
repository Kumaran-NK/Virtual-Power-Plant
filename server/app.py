# server/app.py
"""
FastAPI application — VPP OpenEnv HTTP interface, Extended Edition.

New endpoints vs base:
  POST /trace          Submit reasoning trace (LLM-scored quality)
  GET  /grader         Now returns ParetoScore (multi-objective)
  GET  /tasks          Now lists all 5 tasks
"""

import json
import os
import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from models import VppAction, VppObservation, VppState, ParetoScore
from server.vpp_environment import VppEnvironment
from server.task_curves import ALL_TASK_IDS, TASK_METADATA


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global env
    env = VppEnvironment()
    yield


app = FastAPI(
    title="VPP Orchestrator — OpenEnv Extended",
    description=(
        "Virtual Power Plant: manage 100 home batteries to maximise grid profit "
        "while balancing safety, carbon credits, battery health, P2P trading, "
        "demand-response auctions, and grid islanding emergencies."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

env: VppEnvironment | None = None

_baseline_lock    = threading.Lock()
_baseline_running = False
_baseline_result  = None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _require_env() -> VppEnvironment:
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialised — call /reset first.")
    return env


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "env_ready": env is not None}


@app.post("/reset")
async def reset(
    task_id: str = Query(
        ...,
        description=" | ".join(ALL_TASK_IDS),
    )
):
    """Reset the environment and start a new episode. Returns the initial observation."""
    global env
    if env is None:
        env = VppEnvironment()

    if task_id not in ALL_TASK_IDS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown task_id '{task_id}'. Valid: {sorted(ALL_TASK_IDS)}",
        )

    obs = env.reset(task_id)
    return obs


@app.post("/step")
async def step(action: VppAction):
    """Take one action. Returns observation, reward, done flag, and diagnostic info."""
    e = _require_env()
    if e.state is None:
        raise HTTPException(status_code=400, detail="Call /reset before /step.")
    if e.state.done:
        raise HTTPException(status_code=400, detail="Episode finished — call /reset to start a new one.")

    obs, reward, done, info = e.step(action)
    return {"observation": obs, "reward": reward, "done": done, "info": info}


@app.get("/state")
async def get_state():
    """Return the current ground-truth state (hidden from agent in production)."""
    e = _require_env()
    return e.state


@app.get("/tasks")
async def get_tasks():
    """List all 5 available tasks and the action schema."""
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
    if env is None or env.state is None:
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

    pareto = env.get_pareto_score()
    result = pareto.dict()
    result["score"] = pareto.aggregate_score
    return result


# ---------------------------------------------------------------------------
# POST /trace — reasoning quality scoring
# ---------------------------------------------------------------------------

@app.post("/trace")
async def submit_trace(reasoning: str, action: VppAction):
    """
    Submit a reasoning trace alongside an action for LLM quality scoring.

    The trace is stored server-side and evaluated at episode end (GET /grader
    returns reasoning_quality_score when traces are present).

    The scoring LLM checks:
      - Does the reasoning correctly identify the relevant market signals?
      - Is the chosen action consistent with the stated reasoning?
      - Is the reserve management justified?
    """
    e = _require_env()
    if e.state is None:
        raise HTTPException(status_code=400, detail="Call /reset before /trace.")

    # Inject reasoning into action and step
    action.reasoning = reasoning
    obs, reward, done, info = e.step(action)

    traces = e.get_reasoning_traces()
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
    e = _require_env()
    return {"traces": e.get_reasoning_traces()}


# ---------------------------------------------------------------------------
# /baseline
# ---------------------------------------------------------------------------

def _run_baseline_subprocess() -> dict:
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
    """Zero-argument main for openenv validate."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for running the server directly with custom host/port."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_server(port=args.port)