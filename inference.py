#!/usr/bin/env python3
"""
Inference Script for VPP (Virtual Power Plant) Environment
===========================================================

MANDATORY environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT (strictly enforced):
    [START] task=<task_name> env=vpp model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Rules:
    - One [START] at episode begin.
    - One [STEP] per step, immediately after env.step() returns.
    - One [END] after env.close(), always emitted (even on exception).
    - reward and rewards formatted to 2 decimal places.
    - done and success are lowercase true/false.
    - error is the raw error string, or null if none.
    - All fields on a single line.
    - Each task returns score in [0, 1].
"""

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment variables (mandatory per OpenEnv spec)
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
VPP_SERVER_URL = os.getenv("VPP_SERVER_URL", "http://localhost:7860")

BENCHMARK = "vpp"
MAX_STEPS = 48   # 12-hour episode

if not HF_TOKEN:
    print("[ERROR] Missing HF_TOKEN or API_KEY environment variable.", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# OpenAI client (using provided base URL and token)
# ---------------------------------------------------------------------------

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert energy trader managing a Virtual Power Plant (VPP).
You must maximise profit by deciding when to buy energy from the grid (store in batteries)
and when to sell energy back to the grid (discharge batteries).

Rules:
- global_charge_rate > 0  → buy from grid (costs money, charges batteries)
- global_charge_rate < 0  → sell to grid (earns money, drains batteries)
- If grid_frequency_hz < 49.8, you MUST set a negative charge_rate (emergency discharge)
- Keep min_reserve_pct >= 0.15 to avoid blackouts at night

Respond ONLY with a valid JSON object — no explanation, no markdown fences."""

ACTION_PROMPT = """Current VPP observation:

Step: {step_id}/47  ({time_of_day})
Current price: ${price:.2f}/MWh
Grid frequency: {freq:.2f} Hz
Mean battery SoC: {mean_soc:.1%}
Min battery SoC:  {min_soc:.1%}
Mean solar output: {solar:.2f} kW/home
Mean demand:       {demand:.2f} kW/home
Next 4-step price forecast: {price_forecast}
Next 4-step solar forecast: {solar_forecast}

Task: {task_id}

Decide your action. Return JSON only:
{{"global_charge_rate": <float -1.0 to 1.0>, "min_reserve_pct": <float 0.0 to 1.0>}}"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summarise_obs(obs: dict) -> dict:
    """Extract key summary stats from a raw observation dict."""
    telemetry = obs.get("telemetry", [])
    socs   = [t["soc"]                   for t in telemetry] if telemetry else [0.5]
    solar  = [t["current_solar_gen_kw"]  for t in telemetry] if telemetry else [0.0]
    demand = [t["current_house_load_kw"] for t in telemetry] if telemetry else [0.0]

    step = obs.get("step_id", 0)
    hour, minute = (step * 15) // 60, (step * 15) % 60

    price_fc = obs.get("short_term_price_forecast") or obs.get("forecast_24h_price", [])[:4]
    solar_fc = obs.get("short_term_solar_forecast")  or obs.get("forecast_24h_solar",  [])[:4]

    return {
        "step_id":        step,
        "time_of_day":    f"{hour:02d}:{minute:02d}",
        "price":          obs.get("market_price_per_mwh", 50.0),
        "freq":           obs.get("grid_frequency_hz", 50.0),
        "mean_soc":       sum(socs) / len(socs),
        "min_soc":        min(socs),
        "solar":          sum(solar)  / max(len(solar),  1),
        "demand":         sum(demand) / max(len(demand), 1),
        "price_forecast": [round(p, 1) for p in price_fc[:4]],
        "solar_forecast": [round(s, 2) for s in solar_fc[:4]],
    }


def _extract_json(text: str) -> dict:
    """Parse JSON from LLM output, stripping any markdown fences."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    return json.loads(text)


def _get_llm_action(obs: dict, task_id: str) -> Dict[str, Any]:
    """Query the LLM for an action given the current observation."""
    summary = _summarise_obs(obs)
    prompt  = ACTION_PROMPT.format(task_id=task_id, **summary)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.1,
        max_tokens=80,
    )
    text     = response.choices[0].message.content.strip()
    decision = _extract_json(text)

    # Validate and clamp to legal range
    charge_rate  = float(max(-1.0, min(1.0,  decision.get("global_charge_rate", 0.0))))
    reserve_pct  = float(max(0.0,  min(1.0,  decision.get("min_reserve_pct",  0.2))))
    return {"global_charge_rate": charge_rate, "min_reserve_pct": reserve_pct}


# ---------------------------------------------------------------------------
# Rule-based fallback agent (used when LLM call fails)
# ---------------------------------------------------------------------------

def _rule_agent(obs: dict) -> Dict[str, Any]:
    """Conservative rule-based agent — never fails."""
    freq  = obs.get("grid_frequency_hz", 50.0)
    price = obs.get("market_price_per_mwh", 50.0)
    t     = obs.get("telemetry", [])
    mean_soc   = sum(x["soc"] for x in t) / max(len(t), 1) if t else 0.5
    mean_solar = sum(x["current_solar_gen_kw"] for x in t) / max(len(t), 1) if t else 0.0

    if freq < 49.8:
        return {"global_charge_rate": -1.0, "min_reserve_pct": 0.10}
    if mean_soc < 0.20:
        return {"global_charge_rate": 0.4 if price < 42.0 else 0.0, "min_reserve_pct": 0.20}
    if price > 200.0:
        return {"global_charge_rate": -1.0, "min_reserve_pct": 0.20}
    if price > 55.0 and mean_soc > 0.35:
        return {"global_charge_rate": -0.7, "min_reserve_pct": 0.20}
    if mean_solar > 2.0 and mean_soc > 0.70:
        return {"global_charge_rate": -0.5, "min_reserve_pct": 0.20}
    if price < 38.0 and mean_soc < 0.60:
        return {"global_charge_rate": 0.5, "min_reserve_pct": 0.20}
    return {"global_charge_rate": 0.0, "min_reserve_pct": 0.20}


# ---------------------------------------------------------------------------
# Episode runner — emits mandatory stdout log lines
# ---------------------------------------------------------------------------

def run_episode(task_id: str) -> float:
    """
    Run one full episode against the VPP server.

    Returns the final grader score (0.0–1.0).
    Emits [START], [STEP]×N, [END] to stdout.
    """
    session = requests.Session()
    step       = 0
    rewards: List[float] = []
    done       = False
    success    = False
    score      = 0.0

    try:
        # ── Reset ──────────────────────────────────────────────────────────
        resp = session.post(
            f"{VPP_SERVER_URL}/reset",
            params={"task_id": task_id},
            timeout=15,
        )
        resp.raise_for_status()
        obs = resp.json()

        # ── [START] ────────────────────────────────────────────────────────
        print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

        # ── Steps ──────────────────────────────────────────────────────────
        while not done and step < MAX_STEPS:
            error_str = "null"
            action    = {"global_charge_rate": 0.0, "min_reserve_pct": 0.2}

            try:
                action = _get_llm_action(obs, task_id)
            except Exception as llm_err:
                # LLM call failed → use rule-based fallback silently
                action    = _rule_agent(obs)
                error_str = str(llm_err).replace("\n", " ")[:120]

            try:
                step_resp = session.post(
                    f"{VPP_SERVER_URL}/step",
                    json=action,
                    timeout=15,
                )
                step_resp.raise_for_status()
                data   = step_resp.json()
                obs    = data["observation"]
                reward = float(data["reward"])
                done   = bool(data["done"])
                rewards.append(reward)
                step += 1

                action_str = json.dumps(action)
                print(
                    f"[STEP] step={step} action={action_str} "
                    f"reward={reward:.2f} done={str(done).lower()} error={error_str}",
                    flush=True,
                )
            except Exception as step_err:
                error_str = str(step_err).replace("\n", " ")[:120]
                rewards.append(0.0)
                step += 1
                print(
                    f"[STEP] step={step} action={json.dumps(action)} "
                    f"reward=0.00 done=false error={error_str}",
                    flush=True,
                )
                break

        # ── Grader ─────────────────────────────────────────────────────────
        try:
            grader_resp = session.get(f"{VPP_SERVER_URL}/grader", timeout=10)
            grader_data = grader_resp.json()
            score   = float(grader_data.get("score", 0.0))
            success = done and score > 0.0
        except Exception:
            score   = 0.0
            success = False

    except Exception as outer_err:
        # Ensure [END] is always emitted
        print(
            f"[END] success=false steps={step} score=0.00 "
            f"rewards={','.join(f'{r:.2f}' for r in rewards)}",
            flush=True,
        )
        print(f"[ERROR] Episode failed: {outer_err}", file=sys.stderr)
        session.close()
        return 0.0

    # ── [END] ──────────────────────────────────────────────────────────────
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={step} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )
    session.close()
    return score


# ---------------------------------------------------------------------------
# Health check helper
# ---------------------------------------------------------------------------

def _wait_for_server(timeout: int = 30) -> bool:
    """Poll /health until the server is ready or timeout expires."""
    import time
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not _wait_for_server(timeout=30):
        print(
            f"[ERROR] VPP server not reachable at {VPP_SERVER_URL}. "
            "Start it with: uvicorn server.app:app --host 0.0.0.0 --port 7860",
            file=sys.stderr,
        )
        sys.exit(1)

    tasks = [
        "easy-arbitrage",
        "medium-forecast-error",
        "hard-frequency-response",
    ]

    scores = {}
    for task in tasks:
        scores[task] = run_episode(task)

    # Summary to stderr (not stdout, to avoid polluting the log format)
    print("\n--- Score Summary ---", file=sys.stderr)
    for task, sc in scores.items():
        print(f"  {task:<32}  {sc:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()