# server/app.py
"""
FastAPI application — VPP OpenEnv HTTP interface, Extended Edition.

Uses OpenEnv core create_app() for lifecycle and session management.

Custom OpenEnv-compatible endpoints:
  POST /trace          Submit reasoning trace (LLM-scored quality)
  GET  /grader         Explains how to consume Pareto scores
  GET  /tasks          Lists all 5 tasks + schemas
  GET  /traces         Returns stored reasoning traces
  GET  /baseline       Returns pre-computed baseline scores
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.routing import APIRoute
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

# Remove default OpenEnv GET /tasks so the enhanced handler below is authoritative.
app.router.routes = [
    route
    for route in app.router.routes
    if not (isinstance(route, APIRoute) and route.path == "/tasks" and "GET" in route.methods)
]

# Global state for baseline computation (not per-session)
_baseline_lock    = threading.Lock()
_baseline_running = False
_baseline_result  = None
_baseline_error: Optional[str] = None
_baseline_task: Optional[asyncio.Task] = None
_baseline_proc: Optional[subprocess.Popen] = None

_PARETO_WEIGHTS = {
    "profit": 0.50,
    "safety": 0.20,
    "carbon": 0.15,
    "degradation": 0.10,
    "dr": 0.05,
}

def _env_flag_true(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


_EXPOSE_GRADER_SNAPSHOT = _env_flag_true("VPP_EXPOSE_GRADER_SNAPSHOT", "0")

def _baseline_scores_path() -> str:
    """Primary path for persisted baseline scores."""
    configured = os.getenv("BASELINE_SCORES_PATH")
    if configured:
        return os.path.abspath(configured)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "baseline_scores.json"))

def _fallback_baseline_scores_path() -> str:
    """Fallback cache path for environments where source directories are read-only."""
    return os.path.join(tempfile.gettempdir(), "vpp_baseline_scores.json")


def _resolve_server_url() -> str:
    """Resolve server URL for subprocess clients without hardcoding localhost:7860."""
    configured = os.getenv("VPP_SERVER_URL")
    if configured:
        return configured

    host = os.getenv("HOST", "127.0.0.1")
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    elif ":" in host and not host.startswith("["):
        # Bracket IPv6 literals so URL parsing stays valid (e.g., http://[::1]:7860).
        host = f"[{host}]"
    port = os.getenv("PORT", "7860")
    return f"http://{host}:{port}"


def _write_json_atomic(path: str, payload: dict) -> None:
    """Atomically persist JSON so readers never observe truncated content."""
    tmp_path = f"{path}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise

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
# Note: create_app provides /health, /reset, /step, /state, /tasks (basic).
# We intentionally replace GET /tasks above so this enhanced handler owns /tasks.

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
async def get_grader_score(
    task_id: Optional[str] = Query(
        None,
        description="Optional task id to include cached baseline score for that task.",
    ),
):
    """
    Guidance endpoint for Pareto score retrieval.

    HTTP /reset and /step are stateless in this server setup, so there is no
    request-scoped episode state to grade here. Consume per-step Pareto scores
    from observation metadata via WebSocket sessions.
    """
    if task_id is not None and task_id not in ALL_TASK_IDS:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")

    with _baseline_lock:
        baseline_running = _baseline_running
        baseline_error = _baseline_error
        cached_scores = _baseline_result if isinstance(_baseline_result, dict) else None

    latest_snapshot = VppEnvironment.get_last_grader_snapshot() if _EXPOSE_GRADER_SNAPSHOT else None

    response = {
        "status": "stateless-http",
        "message": (
            "No active HTTP episode state is available for /grader. "
            "Use VppEnv WebSocket sessions and consume observation.pareto_score "
            "from each observation (fallback: metadata.pareto_score)."
        ),
        "weights": _PARETO_WEIGHTS,
        "baseline_refresh_running": baseline_running,
        "pareto_score_schema": ParetoScore.model_json_schema(),
        "latest_client_grader_status": (
            "available"
            if latest_snapshot is not None
            else ("not-available" if _EXPOSE_GRADER_SNAPSHOT else "disabled")
        ),
    }

    if _EXPOSE_GRADER_SNAPSHOT:
        response["latest_client_grader_snapshot"] = latest_snapshot

    if baseline_error:
        response["baseline_error"] = baseline_error

    if task_id is not None:
        task_score = cached_scores.get(task_id) if cached_scores else None
        response["task_id"] = task_id
        response["cached_baseline_task_score"] = task_score

    return response


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

def _run_baseline_subprocess() -> None:
    """Trigger baseline_inference.py and store results in shared state."""
    global _baseline_result, _baseline_running, _baseline_error, _baseline_proc

    baseline_script = os.path.join(os.path.dirname(__file__), "..", "baseline_inference.py")
    baseline_script = os.path.abspath(baseline_script)
    env_vars = {**os.environ, "VPP_SERVER_URL": _resolve_server_url()}
    proc: Optional[subprocess.Popen] = None

    try:
        proc = subprocess.Popen(
            [sys.executable, baseline_script, "--json-only"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env_vars,
        )
        with _baseline_lock:
            _baseline_proc = proc

        try:
            stdout, stderr = proc.communicate(timeout=300)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except OSError:
                pass
            proc.communicate()
            with _baseline_lock:
                _baseline_error = "Baseline computation timed out (>300 s)."
            return

        if proc.returncode == 0 and stdout.strip():
            scores = json.loads(stdout.strip())
            out_path = _baseline_scores_path()
            try:
                _write_json_atomic(out_path, scores)
            except OSError:
                fallback_path = _fallback_baseline_scores_path()
                _write_json_atomic(fallback_path, scores)
            with _baseline_lock:
                _baseline_result = scores
                _baseline_error = None
        else:
            with _baseline_lock:
                _baseline_error = (
                    f"Baseline script returned non-zero exit code. "
                    f"Details: {(stderr or '')[:500]}"
                )
    except json.JSONDecodeError as e:
        with _baseline_lock:
            _baseline_error = f"Could not parse baseline output: {e}"
    except Exception as e:
        with _baseline_lock:
            _baseline_error = str(e)
    finally:
        with _baseline_lock:
            if _baseline_proc is proc:
                _baseline_proc = None
            _baseline_running = False


async def _run_baseline_subprocess_async() -> None:
    global _baseline_task
    try:
        await asyncio.to_thread(_run_baseline_subprocess)
    finally:
        with _baseline_lock:
            _baseline_task = None


def _terminate_baseline_process(proc: Optional[subprocess.Popen]) -> None:
    """Terminate baseline subprocess handle best-effort."""
    if proc is None or proc.poll() is not None:
        return

    try:
        proc.terminate()
    except OSError:
        pass

    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            pass


async def _cancel_baseline_task_if_running() -> None:
    """Cancel and await any in-flight baseline background task."""
    global _baseline_task, _baseline_proc, _baseline_running

    task_to_cancel: Optional[asyncio.Task] = None
    proc_to_cancel: Optional[subprocess.Popen] = None
    with _baseline_lock:
        if _baseline_task is not None and not _baseline_task.done():
            task_to_cancel = _baseline_task
        proc_to_cancel = _baseline_proc

    if task_to_cancel is not None:
        task_to_cancel.cancel()

    _terminate_baseline_process(proc_to_cancel)

    if task_to_cancel is not None:
        try:
            await task_to_cancel
        except asyncio.CancelledError:
            pass

    # Re-check for a subprocess handle that may have been published after snapshot.
    late_proc_to_cancel: Optional[subprocess.Popen] = None
    with _baseline_lock:
        if _baseline_proc is not None and _baseline_proc is not proc_to_cancel:
            late_proc_to_cancel = _baseline_proc

    _terminate_baseline_process(late_proc_to_cancel)

    with _baseline_lock:
        if _baseline_task is task_to_cancel:
            _baseline_task = None
        if _baseline_proc in {proc_to_cancel, late_proc_to_cancel}:
            _baseline_proc = None
        elif _baseline_proc is not None and _baseline_proc.poll() is not None:
            _baseline_proc = None
        if _baseline_task is None and _baseline_proc is None:
            _baseline_running = False


_openenv_lifespan_context = app.router.lifespan_context


@asynccontextmanager
async def _vpp_lifespan(app_instance: FastAPI):
    """Compose OpenEnv lifespan with VPP shutdown cleanup."""
    async with _openenv_lifespan_context(app_instance):
        try:
            yield
        finally:
            await _cancel_baseline_task_if_running()


app.router.lifespan_context = _vpp_lifespan


@app.get("/baseline")
async def get_baseline(
    refresh: bool = Query(False, description="Set to true to recompute live."),
):
    """
    Return pre-computed baseline scores or trigger live recomputation.

    When refresh=true, asynchronously runs baseline_inference.py and returns results.
    Otherwise, returns pre-stored baseline_scores.json if available.
    """
    global _baseline_running, _baseline_result, _baseline_error, _baseline_task

    if refresh:
        with _baseline_lock:
            if _baseline_running:
                return JSONResponse(
                    status_code=202,
                    content={"status": "Baseline already running. Check back shortly."},
                )
            _baseline_running = True
            _baseline_error = None
            try:
                _baseline_task = asyncio.create_task(_run_baseline_subprocess_async())
            except Exception as e:
                _baseline_running = False
                _baseline_task = None
                _baseline_error = f"Could not start baseline refresh task: {e}"
                return JSONResponse(status_code=500, content={"error": _baseline_error})
        return JSONResponse(
            status_code=202,
            content={"status": "Baseline refresh started. Poll /baseline for results."},
        )

    with _baseline_lock:
        if _baseline_running:
            return JSONResponse(
                status_code=202,
                content={"status": "Baseline refresh in progress."},
            )
        if _baseline_result is not None:
            return _baseline_result
        baseline_error = _baseline_error

    baseline_path = _baseline_scores_path()
    fallback_path = _fallback_baseline_scores_path()
    primary_decode_error: Optional[json.JSONDecodeError] = None
    primary_load_error: Optional[OSError] = None

    try:
        with open(baseline_path, "r", encoding="utf-8") as f:
            scores = json.load(f)
        with _baseline_lock:
            _baseline_result = scores
        return scores
    except json.JSONDecodeError as e:
        primary_decode_error = e
    except (FileNotFoundError, OSError) as e:
        primary_load_error = e

    if fallback_path != baseline_path and os.path.exists(fallback_path):
        try:
            with open(fallback_path, "r", encoding="utf-8") as f:
                scores = json.load(f)
            with _baseline_lock:
                _baseline_result = scores
            return scores
        except json.JSONDecodeError as e:
            if primary_decode_error is not None:
                raise HTTPException(
                    status_code=500,
                    detail=f"Invalid baseline score files: primary={primary_decode_error}; fallback={e}",
                ) from e
            raise HTTPException(status_code=500, detail=f"Invalid fallback baseline score file: {e}") from e

    if baseline_error:
        return JSONResponse(status_code=500, content={"error": baseline_error})

    if primary_decode_error is not None:
        raise HTTPException(status_code=500, detail=f"Invalid baseline score file: {primary_decode_error}") from primary_decode_error

    if primary_load_error is not None and not isinstance(primary_load_error, FileNotFoundError):
        raise HTTPException(status_code=500, detail=f"Could not read baseline score file: {primary_load_error}") from primary_load_error

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
