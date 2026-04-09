#!/usr/bin/env python3
"""
Inference Script for VPP (Virtual Power Plant) Environment — Extended Edition.

STDOUT FORMAT (strictly enforced):
    [START] task=<task_name> env=vpp model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

This script automatically falls back to a rule‑based agent if:
- No API key is provided (HF_TOKEN)
- The LLM call fails (network, rate limits, authentication, etc.)
- Monthly credits are exhausted (HTTP 402)

It supports OpenAI only.
"""

import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

# Required by submission checklist
API_BASE_URL   = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME_ENV = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN       = os.getenv("HF_TOKEN") or os.getenv("API_KEY")  # Support both HF_TOKEN and OPENAI_API_KEY for flexibility
# Optional when using from_docker_image() workflows
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

VPP_SERVER_URL = os.getenv("VPP_SERVER_URL", "http://localhost:7860")

BENCHMARK = "vpp"
MAX_STEPS = 48

# ---------------------------------------------------------------------------
# LLM client setup (auto‑detect provider)
# ---------------------------------------------------------------------------

client: Optional[OpenAI] = None
DEFAULT_MODEL: str = ""
USE_LLM = False  # Will be set to True if a working API key is found

# OpenAI client configured through required checklist variables
if HF_TOKEN:
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL, max_retries=0)
    DEFAULT_MODEL = MODEL_NAME_ENV
    USE_LLM = True
    print(f"[INFO] Using OpenAI client with model: {DEFAULT_MODEL}", file=sys.stderr)
else:
    USE_LLM = False
    DEFAULT_MODEL = "rule-based-smart-agent-v2"
    print("[WARNING] HF_TOKEN not found. Falling back to rule-based agent.", file=sys.stderr)

# ---------------------------------------------------------------------------
# Prompts (identical to baseline_inference.py)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert energy trader managing a Virtual Power Plant (VPP).
You maximise a multi-objective Pareto score:
  50% profit | 20% safety | 15% carbon credits | 10% battery health | 5% DR participation

Rules:
- global_charge_rate < 0  → sell to grid (earns money, drains batteries)
- global_charge_rate > 0  → buy from grid (costs money, charges batteries)
- grid_frequency_hz < 49.8 → MUST discharge (emergency)
- grid_connected=false → DO NOT buy or sell; preserve battery for reconnection
- defer_ev_charging > 0 → defers Zone B EV load (must replay by step 40)
- accept_dr_bid=true → commits to deliver dr_committed_power_kw for dr_committed_steps
- p2p_export_rate > 0 → routes Zone B solar surplus to Zone A (earns midpoint price)
- Avoid grid purchases during steps 0-16 (high carbon emission hours)

Respond ONLY with a valid JSON object."""

ACTION_PROMPT = """Step: {step_id}/47 ({time_of_day})
Price: ${price:.2f}/MWh  |  Freq: {freq:.2f} Hz  |  Grid: {grid_status}
Mean SoC: {mean_soc:.1%}  |  Mean SoH: {mean_soh:.3f}
Solar: {solar:.2f} kW/home  |  Demand: {demand:.2f} kW/home
Carbon balance: {carbon:.2f} credits
DR bid: {dr_info}
P2P available (Zone B): {p2p:.2f} kW/home
Price forecast (4-step): {price_forecast} ± {price_uncertainty} USD/MWh
Solar forecast (4-step): {solar_forecast} ± {solar_uncertainty} kW

Task: {task_id}

Respond with JSON only:
{{"global_charge_rate": <-1.0 to 1.0>, "min_reserve_pct": <0.0 to 1.0>,
  "defer_ev_charging": <0.0 to 1.0>, "accept_dr_bid": <true|false>,
  "p2p_export_rate": <0.0 to 1.0>}}"""

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _summarise_obs(obs: dict) -> dict:
    telemetry = obs.get("telemetry", [])
    socs    = [t["soc"]                       for t in telemetry] if telemetry else [0.5]
    sohs    = [t.get("state_of_health", 1.0)  for t in telemetry] if telemetry else [1.0]
    solar   = [t["current_solar_gen_kw"]      for t in telemetry] if telemetry else [0.0]
    demand  = [t["current_house_load_kw"]     for t in telemetry] if telemetry else [0.0]

    step    = obs.get("step_id", 0)
    h, m    = (step * 15) // 60, (step * 15) % 60
    dr_bid  = obs.get("dr_bid", {})
    zone_b  = next((z for z in obs.get("zone_aggregates", []) if z.get("zone_id") == "zone-b"), {})

    dr_info = "none"
    if dr_bid.get("active"):
        dr_info = (
            f"premium={dr_bid.get('premium_multiplier', 1.0):.1f}×, "
            f"require {dr_bid.get('committed_power_kw', 0):.1f} kW for "
            f"{dr_bid.get('committed_steps', 0)} steps"
        )

    price_fc = obs.get("short_term_price_forecast") or obs.get("forecast_24h_price", [])[:4]
    solar_fc = obs.get("short_term_solar_forecast")  or obs.get("forecast_24h_solar",  [])[:4]
    price_u  = obs.get("forecast_price_uncertainty", [2.5, 3.5, 4.5, 5.5])
    solar_u  = obs.get("forecast_solar_uncertainty", [0.25, 0.35, 0.50, 0.70])

    return {
        "step_id":          step,
        "time_of_day":      f"{h:02d}:{m:02d}",
        "price":            obs.get("market_price_per_mwh", 50.0),
        "freq":             obs.get("grid_frequency_hz", 50.0),
        "grid_status":      "CONNECTED" if obs.get("grid_connected", True) else "ISLANDED",
        "mean_soc":         sum(socs) / len(socs),
        "mean_soh":         sum(sohs) / len(sohs),
        "solar":            sum(solar)  / max(len(solar), 1),
        "demand":           sum(demand) / max(len(demand), 1),
        "carbon":           obs.get("carbon_credits_balance", 0.0),
        "dr_info":          dr_info,
        "p2p":              zone_b.get("p2p_available_kw", 0.0),
        "price_forecast":   [round(p, 1) for p in price_fc[:4]],
        "solar_forecast":   [round(s, 2) for s in solar_fc[:4]],
        "price_uncertainty": [round(u, 1) for u in price_u[:4]],
        "solar_uncertainty": [round(u, 2) for u in solar_u[:4]],
    }


def _extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    return json.loads(text)


def _extract_response_text(response: Any) -> str:
    # OpenAI Chat Completions
    if hasattr(response, "choices"):
        return (response.choices[0].message.content or "").strip()

    return str(response).strip()


def _rule_agent(obs: dict, task_id: str = "") -> Dict[str, Any]:
    """Deterministic rule‑based agent (covers all extended mechanics)."""
    freq      = obs.get("grid_frequency_hz", 50.0)
    price     = obs.get("market_price_per_mwh", 50.0)
    grid_conn = obs.get("grid_connected", True)
    t         = obs.get("telemetry", [])
    step      = obs.get("step_id", 0)
    zone_b    = next((z for z in obs.get("zone_aggregates", []) if z.get("zone_id") == "zone-b"), {})
    dr        = obs.get("dr_bid", {})

    mean_soc   = sum(x["soc"] for x in t) / max(len(t), 1) if t else 0.5
    mean_soh   = sum(x.get("state_of_health", 1.0) for x in t) / max(len(t), 1) if t else 1.0
    mean_solar = sum(x["current_solar_gen_kw"] for x in t) / max(len(t), 1) if t else 0.0
    p2p_avail  = zone_b.get("p2p_available_kw", 0.0)
    p2p_rate   = 0.8 if p2p_avail > 1.0 else 0.0
    defer_ev   = 0.8 if (price > 60.0 and step >= 32 and step < 38) else 0.0
    accept_dr  = bool(dr.get("active") and not dr.get("steps_remaining") and
                      dr.get("premium_multiplier", 1.0) >= 2.0 and mean_soc > 0.50)

    # Import here to avoid circular import (ISLANDING_END is used)
    from server.task_curves import ISLANDING_END

    if not grid_conn:
        return {"global_charge_rate": 0.0, "min_reserve_pct": 0.40,
                "defer_ev_charging": 0.0, "accept_dr_bid": False, "p2p_export_rate": 0.0}
    if "islanding" in task_id and step == ISLANDING_END and mean_soc > 0.40:
        return {"global_charge_rate": -1.0, "min_reserve_pct": 0.20,
                "defer_ev_charging": defer_ev, "accept_dr_bid": False, "p2p_export_rate": p2p_rate}
    if freq < 49.8:
        return {"global_charge_rate": -1.0, "min_reserve_pct": 0.10,
                "defer_ev_charging": 0.0, "accept_dr_bid": False, "p2p_export_rate": p2p_rate}
    if mean_soc < 0.20:
        rate = 0.4 if price < 42.0 else 0.0
        return {"global_charge_rate": rate, "min_reserve_pct": 0.20,
                "defer_ev_charging": 0.0, "accept_dr_bid": False, "p2p_export_rate": 0.0}
    if dr.get("active") and not dr.get("steps_remaining") and dr.get("premium_multiplier", 1.0) >= 2.0 and mean_soc > 0.50:
        commit_fraction = min(1.0, dr.get("committed_power_kw", 0) / 5.0)
        return {"global_charge_rate": -commit_fraction, "min_reserve_pct": 0.20,
                "defer_ev_charging": defer_ev, "accept_dr_bid": True, "p2p_export_rate": p2p_rate}
    if dr.get("steps_remaining", 0) > 0:
        commit_fraction = min(1.0, dr.get("committed_power_kw", 0) / 5.0)
        return {"global_charge_rate": -commit_fraction, "min_reserve_pct": 0.20,
                "defer_ev_charging": defer_ev, "accept_dr_bid": False, "p2p_export_rate": p2p_rate}
    if price > 200.0:
        return {"global_charge_rate": -1.0, "min_reserve_pct": 0.20,
                "defer_ev_charging": defer_ev, "accept_dr_bid": False, "p2p_export_rate": p2p_rate}
    if price > 55.0 and mean_soc > 0.35:
        rate = -0.5 if mean_soh < 0.92 else -0.7
        return {"global_charge_rate": rate, "min_reserve_pct": 0.20,
                "defer_ev_charging": defer_ev, "accept_dr_bid": accept_dr, "p2p_export_rate": p2p_rate}
    if mean_solar > 2.0 and mean_soc > 0.70:
        return {"global_charge_rate": -0.5, "min_reserve_pct": 0.20,
                "defer_ev_charging": defer_ev, "accept_dr_bid": accept_dr, "p2p_export_rate": p2p_rate}
    if price < 38.0 and mean_soc < 0.60:
        charge_rate = 0.5
        if step < 17 and obs.get("carbon_credits_balance", 0.0) < -2.0:
            charge_rate = 0.0
        return {"global_charge_rate": charge_rate, "min_reserve_pct": 0.20,
                "defer_ev_charging": defer_ev, "accept_dr_bid": False, "p2p_export_rate": 0.0}
    return {"global_charge_rate": 0.0, "min_reserve_pct": 0.20,
            "defer_ev_charging": defer_ev, "accept_dr_bid": accept_dr, "p2p_export_rate": p2p_rate}


def get_llm_action(obs: dict, task_id: str) -> Dict[str, Any]:
    """Call LLM with retries and fallback to rule agent."""
    if not USE_LLM or client is None:
        return _rule_agent(obs, task_id)

    summary = _summarise_obs(obs)
    prompt  = ACTION_PROMPT.format(task_id=task_id, **summary)

    max_attempts = 4
    for attempt in range(max_attempts):
        try:
            time.sleep(1.0)   # Basic rate limit pacing
            response = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.1,
                max_tokens=120,
            )

            text = _extract_response_text(response)
            decision = _extract_json(text)
            return {
                "global_charge_rate": float(max(-1.0, min(1.0, decision.get("global_charge_rate", 0.0)))),
                "min_reserve_pct":    float(max(0.0, min(1.0, decision.get("min_reserve_pct", 0.2)))),
                "defer_ev_charging":  float(max(0.0, min(1.0, decision.get("defer_ev_charging", 0.0)))),
                "accept_dr_bid":      bool(decision.get("accept_dr_bid", False)),
                "p2p_export_rate":    float(max(0.0, min(1.0, decision.get("p2p_export_rate", 0.0)))),
            }
        except Exception as e:
            err_msg = str(e).lower()
            if "429" in err_msg or "rate limit" in err_msg or "too many requests" in err_msg:
                sleep_time = 4 ** attempt
                print(f"[INFO] Rate limit hit. Retrying in {sleep_time}s...", file=sys.stderr)
                time.sleep(sleep_time)
            else:
                # Non‑rate‑limit error → fallback immediately
                print(f"[WARNING] LLM error: {e}. Falling back to rule agent.", file=sys.stderr)
                return _rule_agent(obs, task_id)
    # Max retries exceeded
    print("[WARNING] Max retries reached. Falling back to rule agent.", file=sys.stderr)
    return _rule_agent(obs, task_id)


def _extract_observation(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise API responses that may wrap observations under an 'observation' key."""
    if isinstance(payload, dict) and isinstance(payload.get("observation"), dict):
        return payload["observation"]
    return payload if isinstance(payload, dict) else {}


def _post_step(session: requests.Session, action: Dict[str, Any], timeout: int = 15) -> requests.Response:
    """Post one step action with compatibility for wrapped and flat request schemas."""
    # OpenEnv create_app expects a wrapped payload: {"action": {...}}
    response = session.post(f"{VPP_SERVER_URL}/step", json={"action": action}, timeout=timeout)
    if response.status_code != 422:
        return response

    # Backward compatibility for servers that expect a flat action body
    try:
        detail = response.json().get("detail", [])
    except Exception:
        return response

    expects_flat_body = any(
        isinstance(item, dict) and item.get("loc") == ["body", "global_charge_rate"]
        for item in detail
    )
    if expects_flat_body:
        return session.post(f"{VPP_SERVER_URL}/step", json=action, timeout=timeout)

    return response


def _post_trace(
    session: requests.Session,
    action: Dict[str, Any],
    reasoning: str,
    timeout: int = 15,
) -> requests.Response:
    """Submit a reasoning trace using the custom /trace endpoint."""
    return session.post(
        f"{VPP_SERVER_URL}/trace",
        params={"reasoning": reasoning},
        json=action,
        timeout=timeout,
    )


def run_episode(task_id: str) -> float:
    """
    Run one full episode and emit strict [START]/[STEP]/[END] format to stdout.
    All diagnostics go to stderr.
    
    Format (strictly enforced):
      [START] task=<task_id> env=vpp model=<model_name>
      [STEP] step=<int> action=<json_compact> reward=<0.00> done=<true|false> error=<null|msg>
      [END] success=<true|false> steps=<int> score=<0.00> rewards=<0.00,0.00,...>
    """
    session = requests.Session()
    step = 0
    rewards: List[float] = []
    done = False
    success = False
    score = 0.0

    try:
        # Wait for server readiness before printing START
        resp = session.post(f"{VPP_SERVER_URL}/reset", params={"task_id": task_id}, timeout=15)
        resp.raise_for_status()
        obs = _extract_observation(resp.json())

        # [START] line — exactly one line to stdout, nothing else
        sys.stdout.write(f"[START] task={task_id} env={BENCHMARK} model={DEFAULT_MODEL}\n")
        sys.stdout.flush()

        while not done and step < MAX_STEPS:
            error_msg = None  # None means "null" in output; str means error message
            action = {
                "global_charge_rate": 0.0,
                "min_reserve_pct": 0.2,
                "defer_ev_charging": 0.0,
                "accept_dr_bid": False,
                "p2p_export_rate": 0.0,
            }
            
            try:
                action = get_llm_action(obs, task_id)
            except Exception as llm_err:
                action = _rule_agent(obs, task_id)
                error_msg = str(llm_err)[:100]  # Truncate error message

            try:
                trace_resp = None
                if action.get("reasoning"):
                    try:
                        trace_resp = _post_trace(
                            session,
                            action,
                            reasoning=action["reasoning"],
                            timeout=15,
                        )
                    except Exception:
                        trace_resp = None

                if trace_resp is not None and trace_resp.status_code < 400:
                    step_resp = trace_resp
                else:
                    # Fall back to /step on any error (404, 400, 422, etc.)
                    step_resp = _post_step(session, action, timeout=15)
                    if step_resp.status_code >= 400:
                        raise requests.HTTPError(
                            f"step={step_resp.status_code} {step_resp.reason}"
                        )
                
                data = step_resp.json()
                obs = _extract_observation(data)
                reward = float(data["reward"])
                done = bool(data["done"])
                rewards.append(reward)
                step += 1

                # Prefer raw environment last_action_error when present.
                last_action_error = None
                if isinstance(data, dict):
                    info_obj = data.get("info")
                    if isinstance(info_obj, dict):
                        raw_err = info_obj.get("last_action_error")
                        if raw_err not in (None, ""):
                            last_action_error = str(raw_err)
                
                # [STEP] line — exactly one line to stdout, compact JSON action
                action_json = json.dumps(action, separators=(",", ":"), sort_keys=True)
                effective_error = last_action_error if last_action_error is not None else error_msg
                error_str = "null" if effective_error is None else str(effective_error).replace("\n", " ")
                sys.stdout.write(
                    f"[STEP] step={step} action={action_json} "
                    f"reward={reward:.2f} done={str(done).lower()} error={error_str}\n"
                )
                sys.stdout.flush()
                
            except Exception as step_err:
                rewards.append(0.0)
                step += 1
                
                # [STEP] line on error
                action_json = json.dumps(action, separators=(",", ":"), sort_keys=True)
                error_str = str(step_err)[:100].replace("\n", " ")
                sys.stdout.write(
                    f"[STEP] step={step} action={action_json} "
                    f"reward=0.00 done=false error={error_str}\n"
                )
                sys.stdout.flush()
                break

        # Retrieve final grader score
        try:
            grader_resp = session.get(f"{VPP_SERVER_URL}/grader", timeout=10)
            grader_data = grader_resp.json()
            score = float(grader_data.get("aggregate_score", 0.0))
            success = done and score > 0.0
        except Exception as grader_err:
            print(f"[WARNING] Could not fetch grader: {grader_err}", file=sys.stderr)
            score = 0.0
            success = False

    except Exception as outer_err:
        # [END] line even on failure
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        sys.stdout.write(
            f"[END] success=false steps={step} score=0.00 rewards={rewards_str}\n"
        )
        sys.stdout.flush()
        print(f"[ERROR] Episode failed: {outer_err}", file=sys.stderr)
        session.close()
        return 0.0

    # [END] line — exactly one line to stdout, completed episode
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    sys.stdout.write(
        f"[END] success={str(success).lower()} steps={step} score={score:.2f} rewards={rewards_str}\n"
    )
    sys.stdout.flush()
    
    session.close()
    return score


def _wait_for_server(timeout: int = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{VPP_SERVER_URL}/health", timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def main():
    """
    Main entry point for inference.
    All environment variable warnings and summary output go to stderr.
    Only [START], [STEP], [END] lines go to stdout.
    """
    # Log required env vars to stderr, never to stdout
    required_env = ["HF_TOKEN"]
    missing = [k for k in required_env if not os.getenv(k)]
    if missing:
        print(f"[WARNING] Missing expected env vars: {', '.join(missing)}", file=sys.stderr)

    if not _wait_for_server(30):
        print(f"[ERROR] VPP server not reachable at {VPP_SERVER_URL}", file=sys.stderr)
        sys.exit(1)

    from server.task_curves import ALL_TASK_IDS
    
    scores = {}
    for task in ALL_TASK_IDS:
        scores[task] = run_episode(task)

    # Summary output to stderr only
    print("\n--- Task Scores ---", file=sys.stderr)
    for task, sc in scores.items():
        print(f"  {task:<35}  {sc:.4f}", file=sys.stderr)

    avg_score = sum(scores.values()) / max(len(scores), 1)
    print(f"[INFO] Average score: {avg_score:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()