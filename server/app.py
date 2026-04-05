# server/app.py
"""
FastAPI application exposing the VPP environment via the OpenEnv HTTP interface.

Endpoints
---------
POST /reset          Start a new episode
POST /step           Take one action
GET  /state          Return current ground-truth state
GET  /tasks          List available tasks + action schema
GET  /grader         Return episode score (0.0–1.0)
GET  /baseline       Return (and optionally recompute) baseline LLM scores
GET  /health         Liveness probe (used by Docker HEALTHCHECK)
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse

from models import VppAction, VppObservation, VppState
from server.vpp_environment import VppEnvironment


# ---------------------------------------------------------------------------
# Lifespan: create a warm environment instance on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global env
    env = VppEnvironment()
    yield


app = FastAPI(
    title="VPP Orchestrator — OpenEnv",
    description=(
        "Virtual Power Plant environment: manage 100 home batteries "
        "to maximise grid profit while maintaining safety constraints."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

env: VppEnvironment | None = None

# Baseline computation state
_baseline_lock       = threading.Lock()
_baseline_running    = False
_baseline_result     = None      # cached last result


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _require_env() -> VppEnvironment:
    if env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialised — call /reset first.",
        )
    return env


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Docker HEALTHCHECK liveness probe."""
    return {"status": "ok", "env_ready": env is not None}


@app.post("/reset")
async def reset(
    task_id: str = Query(
        ...,
        description="Task ID: easy-arbitrage | medium-forecast-error | hard-frequency-response",
    )
):
    """Reset the environment and start a new episode. Returns the initial observation."""
    global env
    if env is None:
        env = VppEnvironment()

    valid_tasks = {
        "easy-arbitrage",
        "medium-forecast-error",
        "hard-frequency-response",
    }
    if task_id not in valid_tasks:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown task_id '{task_id}'. Valid: {sorted(valid_tasks)}",
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
        raise HTTPException(
            status_code=400,
            detail="Episode finished — call /reset to start a new one.",
        )

    obs, reward, done, info = e.step(action)
    return {"observation": obs, "reward": reward, "done": done, "info": info}


@app.get("/state")
async def get_state():
    """Return the current ground-truth state (hidden from agent in production)."""
    e = _require_env()
    return e.state


@app.get("/tasks")
async def get_tasks():
    """List all available tasks and the action schema."""
    return {
        "tasks": [
            {
                "id": "easy-arbitrage",
                "description": (
                    "High solar production, low household demand, flat $50/MWh price. "
                    "Strategy: sell solar surplus. Profit target: $500."
                ),
                "difficulty": "easy",
                "profit_target_usd": 500.0,
            },
            {
                "id": "medium-forecast-error",
                "description": (
                    "Heatwave event: AC demand spikes 4× from 10:00–14:00. "
                    "Sinusoidal pricing rewards time-of-use arbitrage. "
                    "Agent must manage forecast uncertainty. Profit target: $200."
                ),
                "difficulty": "medium",
                "profit_target_usd": 200.0,
            },
            {
                "id": "hard-frequency-response",
                "description": (
                    "Grid stress: a single-step 10× price spike at 12:30 (step 26). "
                    "Grid frequency drops to 49.5 Hz — agent must discharge immediately. "
                    "If batteries are depleted before the spike, revenue is lost. "
                    "Requires look-ahead planning and reserve management. Profit target: $1000."
                ),
                "difficulty": "hard",
                "profit_target_usd": 1000.0,
            },
        ],
        "action_schema": VppAction.model_json_schema(),
        "observation_schema": VppObservation.model_json_schema(),
    }


@app.get("/grader")
async def get_grader_score():
    """
    Return the deterministic grader score for the completed (or in-progress) episode.

    Score is in [0.0, 1.0]:
      1.0 = profit target met, zero safety violations
      0.0 = no profit or extreme violation count
    """
    if env is None or env.state is None:
        return {"score": 0.0, "detail": "No episode in progress."}

    score = env.get_current_task_score()
    state = env.state
    return {
        "score":                       score,
        "cumulative_profit_usd":       state.cumulative_profit_usd,
        "safety_violations":           state.safety_violations_count,
        "grid_emergencies_ignored":    state.grid_emergencies_ignored,
        "steps_completed":             state.current_step,
        "done":                        state.done,
    }


# ---------------------------------------------------------------------------
# /baseline — dynamic or cached baseline scores
# ---------------------------------------------------------------------------

def _run_baseline_subprocess() -> dict:
    """
    Execute baseline_inference.py in a subprocess and return parsed scores.

    Passes --json-only so the script emits only a JSON blob to stdout.
    Falls back to cached file if the subprocess fails.
    """
    global _baseline_result, _baseline_running

    baseline_script = os.path.join(os.path.dirname(__file__), "..", "baseline_inference.py")
    baseline_script = os.path.abspath(baseline_script)

    env_vars = {**os.environ, "VPP_SERVER_URL": "http://localhost:7860"}

    try:
        result = subprocess.run(
            [sys.executable, baseline_script, "--json-only"],
            capture_output=True,
            text=True,
            timeout=180,          # 3-minute hard cap
            env=env_vars,
        )
        if result.returncode == 0 and result.stdout.strip():
            scores = json.loads(result.stdout.strip())
            # Persist for future requests
            out_path = os.path.join(os.path.dirname(__file__), "..", "baseline_scores.json")
            with open(out_path, "w") as f:
                json.dump(scores, f, indent=2)
            _baseline_result = scores
            return scores
        else:
            stderr_snippet = (result.stderr or "")[:500]
            return {"error": "Baseline script returned non-zero", "details": stderr_snippet}

    except subprocess.TimeoutExpired:
        return {"error": "Baseline computation timed out (>180 s). Try again."}
    except json.JSONDecodeError as e:
        return {"error": f"Could not parse baseline output as JSON: {e}"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        with _baseline_lock:
            _baseline_running = False


@app.get("/baseline")
async def get_baseline(
    refresh: bool = Query(
        False,
        description="Set to true to re-run the baseline inference script synchronously.",
    )
):
    """
    Return pre-computed baseline scores.

    By default returns the cached baseline_scores.json.
    Pass ?refresh=true to recompute live (takes ~2 minutes, requires API key).
    """
    global _baseline_running, _baseline_result

    # If live refresh requested, run the subprocess synchronously
    if refresh:
        with _baseline_lock:
            if _baseline_running:
                return JSONResponse(
                    status_code=202,
                    content={"status": "Baseline already running. Check back shortly."},
                )
            _baseline_running = True

        scores = _run_baseline_subprocess()
        return scores

    # Try in-memory cache first
    if _baseline_result is not None:
        return _baseline_result

    # Fall back to file
    baseline_path = os.path.join(
        os.path.dirname(__file__), "..", "baseline_scores.json"
    )
    try:
        with open(baseline_path, "r") as f:
            scores = json.load(f)
        _baseline_result = scores
        return scores
    except FileNotFoundError:
        return {
            "easy-arbitrage": {
                "score": 0.0,
                "note": "Run baseline_inference.py or call /baseline?refresh=true",
            },
            "medium-forecast-error": {
                "score": 0.0,
                "note": "Run baseline_inference.py or call /baseline?refresh=true",
            },
            "hard-frequency-response": {
                "score": 0.0,
                "note": "Run baseline_inference.py or call /baseline?refresh=true",
            },
        }