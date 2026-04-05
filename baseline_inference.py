# baseline_inference.py
"""
Baseline inference script — required by the OpenEnv spec.

Runs a zero-shot LLM agent (or rule-based agent) against all 3 tasks
and writes baseline_scores.json.

API key priority:
  1. OPENAI_API_KEY  → uses api.openai.com (GPT-4o-mini)
  2. GROQ_API_KEY    → uses api.groq.com  (Llama-3 8B instant)
  3. HF_TOKEN        → uses router.huggingface.co (fallback)

Environment variables:
  VPP_SERVER_URL   URL of the running FastAPI server (default: http://localhost:7860)
  OPENAI_API_KEY   OpenAI API key (preferred)
  GROQ_API_KEY     Groq API key  (fallback)
  HF_TOKEN         HuggingFace   (fallback)
  MODEL_NAME       Override model name (optional)

Flags:
  --json-only      Print only the final JSON scores dict and exit (no progress logs)
  --agent llm      Use LLM agent (default)
  --agent rule     Use deterministic rule-based smart agent (no API key required)

Usage:
  export OPENAI_API_KEY=sk-...
  python baseline_inference.py
  python baseline_inference.py --json-only
  python baseline_inference.py --agent rule
"""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, Optional

import requests
from openai import OpenAI

from models import VppAction

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="VPP Baseline Inference")
parser.add_argument(
    "--json-only",
    action="store_true",
    help="Suppress all progress logs; emit only the final JSON scores to stdout.",
)
parser.add_argument(
    "--agent",
    choices=["llm", "rule"],
    default="llm",
    help="Agent strategy: 'llm' (default) or 'rule' (deterministic, no API key needed).",
)
args, _unknown = parser.parse_known_args()

JSON_ONLY = args.json_only
AGENT_TYPE = args.agent


def _log(*msg_parts):
    """Print only if not in --json-only mode."""
    if not JSON_ONLY:
        print(*msg_parts)


# ---------------------------------------------------------------------------
# API client setup
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY",   "")
HF_TOKEN       = os.getenv("HF_TOKEN",        "") or os.getenv("API_KEY", "")
API_BASE_URL   = os.getenv("API_BASE_URL",    "")
MODEL_NAME_ENV = os.getenv("MODEL_NAME",      "")

# Only set up LLM client when we actually need it
client: Optional[OpenAI] = None
DEFAULT_MODEL: str = ""

if AGENT_TYPE == "llm":
    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
        DEFAULT_MODEL = MODEL_NAME_ENV or "gpt-4o-mini"
        _log(f"Using OpenAI API with model: {DEFAULT_MODEL}")
    elif GROQ_API_KEY:
        client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
        DEFAULT_MODEL = MODEL_NAME_ENV or "llama-3.1-8b-instant"
        _log(f"Using Groq API with model: {DEFAULT_MODEL}")
    elif HF_TOKEN:
        base = API_BASE_URL or "https://router.huggingface.co/v1"
        client = OpenAI(api_key=HF_TOKEN, base_url=base)
        DEFAULT_MODEL = MODEL_NAME_ENV or "Qwen/Qwen2.5-72B-Instruct"
        _log(f"Using HuggingFace Router with model: {DEFAULT_MODEL}")
    else:
        if not JSON_ONLY:
            print(
                "ERROR: Set OPENAI_API_KEY, GROQ_API_KEY, or HF_TOKEN before running with --agent llm.",
                file=sys.stderr,
            )
        sys.exit(1)
else:
    DEFAULT_MODEL = "rule-based-smart-agent"
    _log("Using deterministic rule-based agent (no API key required).")

SERVER_URL = os.getenv("VPP_SERVER_URL", "http://localhost:7860")

# ---------------------------------------------------------------------------
# LLM action generation
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

Decide your action. Return JSON:
{{"global_charge_rate": <float -1.0 to 1.0>, "min_reserve_pct": <float 0.0 to 1.0>}}"""


def _summarise_obs(obs: dict) -> dict:
    """Extract key summary stats from a raw observation dict."""
    telemetry = obs.get("telemetry", [])
    socs   = [t["soc"]                   for t in telemetry] if telemetry else [0.5]
    solar  = [t["current_solar_gen_kw"]  for t in telemetry] if telemetry else [0.0]
    demand = [t["current_house_load_kw"] for t in telemetry] if telemetry else [0.0]

    step       = obs.get("step_id", 0)
    hour       = (step * 15) // 60
    minute     = (step * 15) % 60
    time_of_day = f"{hour:02d}:{minute:02d}"

    price_fc = obs.get("short_term_price_forecast") or obs.get("forecast_24h_price", [])[:4]
    solar_fc = obs.get("short_term_solar_forecast")  or obs.get("forecast_24h_solar",  [])[:4]

    return {
        "step_id":        step,
        "time_of_day":    time_of_day,
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
    """Parse JSON from LLM output, handling markdown fences."""
    text  = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    return json.loads(text)


# ---------------------------------------------------------------------------
# Rule-based smart agent (deterministic — no API key)
# ---------------------------------------------------------------------------

def rule_based_action(obs: dict) -> Dict[str, Any]:
    """
    Priority-ordered heuristic:
      1. Grid emergency (< 49.8 Hz)  → discharge immediately
      2. Battery critically low       → stop selling, possibly recharge
      3. Price spike (> 200 $/MWh)   → sell at full power
      4. High price (> 55 $/MWh)     → sell at 70 % rate
      5. Solar surplus + battery full → sell excess at 50 % rate
      6. Cheap grid power             → buy and store
      7. Default                      → idle
    """
    telemetry  = obs.get("telemetry", [])
    freq       = obs.get("grid_frequency_hz", 50.0)
    price      = obs.get("market_price_per_mwh", 50.0)
    mean_soc   = sum(t["soc"]                   for t in telemetry) / max(len(telemetry), 1)
    mean_solar = sum(t["current_solar_gen_kw"]  for t in telemetry) / max(len(telemetry), 1)

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
# LLM agent action
# ---------------------------------------------------------------------------

def get_llm_action(obs: dict, task_id: str) -> VppAction:
    """Query the LLM for an action given the current observation."""
    assert client is not None, "LLM client not initialised"
    summary = _summarise_obs(obs)
    prompt  = ACTION_PROMPT.format(task_id=task_id, **summary)

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.1,
            max_tokens=80,
        )
        text     = response.choices[0].message.content.strip()
        decision = _extract_json(text)
        action   = VppAction(**decision)
        _log(
            f"  step={summary['step_id']:02d} t={summary['time_of_day']}"
            f"  price=${summary['price']:.0f}"
            f"  soc={summary['mean_soc']:.0%}"
            f"  → rate={action.global_charge_rate:+.2f}  reserve={action.min_reserve_pct:.2f}"
        )
        return action
    except Exception as e:
        _log(f"  LLM error ({e}) — falling back to rule-based agent.")
        raw = rule_based_action(obs)
        return VppAction(**raw)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_task(task_id: str, session: requests.Session) -> dict:
    """Run one full episode and return score + metadata."""
    _log(f"\n{'─'*50}")
    _log(f"  Task: {task_id}  |  Agent: {AGENT_TYPE}")
    _log(f"{'─'*50}")

    resp = session.post(f"{SERVER_URL}/reset", params={"task_id": task_id})
    if resp.status_code != 200:
        _log(f"  Reset failed: {resp.text}")
        return {"score": 0.0, "total_reward": 0.0, "steps": 0, "error": resp.text}

    obs          = resp.json()
    total_reward = 0.0
    steps        = 0
    done         = False

    while not done:
        # Choose action based on agent type
        if AGENT_TYPE == "rule":
            raw    = rule_based_action(obs)
            action = VppAction(**raw)
            _log(
                f"  step={steps:02d}"
                f"  → rate={action.global_charge_rate:+.2f}"
                f"  reserve={action.min_reserve_pct:.2f}"
            )
        else:
            action = get_llm_action(obs, task_id)

        resp = session.post(
            f"{SERVER_URL}/step",
            json={
                "global_charge_rate": action.global_charge_rate,
                "min_reserve_pct":    action.min_reserve_pct,
            },
        )
        if resp.status_code != 200:
            _log(f"  Step failed: {resp.text}")
            break

        data          = resp.json()
        obs           = data["observation"]
        total_reward += float(data["reward"])
        done          = data["done"]
        steps        += 1

    grader_resp = session.get(f"{SERVER_URL}/grader")
    grader_data = grader_resp.json()
    score       = grader_data.get("score", 0.0)

    _log(f"\n  Steps completed   : {steps}")
    _log(f"  Total reward      : {total_reward:.2f}")
    _log(f"  Profit (USD)      : ${grader_data.get('cumulative_profit_usd', 0.0):.2f}")
    _log(f"  Safety violations : {grader_data.get('safety_violations', 0)}")
    _log(f"  Grader score      : {score:.4f}")

    return {
        "score":             score,
        "total_reward":      round(total_reward, 4),
        "steps":             steps,
        "profit_usd":        grader_data.get("cumulative_profit_usd", 0.0),
        "safety_violations": grader_data.get("safety_violations", 0),
        "model":             DEFAULT_MODEL,
        "agent_type":        AGENT_TYPE,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _log(f"VPP Baseline Inference")
    _log(f"Server : {SERVER_URL}")
    _log(f"Model  : {DEFAULT_MODEL}")
    _log(f"Agent  : {AGENT_TYPE}")

    try:
        health = requests.get(f"{SERVER_URL}/health", timeout=5)
        assert health.status_code == 200
    except Exception as e:
        if not JSON_ONLY:
            print(f"\nERROR: Cannot reach server at {SERVER_URL}: {e}", file=sys.stderr)
        sys.exit(1)

    session = requests.Session()
    tasks   = [
        "easy-arbitrage",
        "medium-forecast-error",
        "hard-frequency-response",
    ]
    results = {}

    for task in tasks:
        results[task] = run_task(task, session)

    if not JSON_ONLY:
        print(f"\n{'='*50}")
        print("  FINAL BASELINE SCORES")
        print(f"{'='*50}")
        for task, data in results.items():
            print(f"  {task:<32}  score={data['score']:.4f}")

    # Write baseline_scores.json
    with open("baseline_scores.json", "w") as f:
        json.dump(results, f, indent=2)

    if not JSON_ONLY:
        print("\nSaved → baseline_scores.json")
    else:
        # --json-only: emit ONLY the JSON to stdout (for subprocess parsing)
        print(json.dumps(results))


if __name__ == "__main__":
    main()