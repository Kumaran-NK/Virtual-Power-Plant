# baseline_inference.py
"""
Baseline inference script — Extended Edition.

Covers all 5 tasks with a smart rule-based agent that exploits:
  - Battery SoH (avoids unnecessary cycling)
  - Carbon credits (prefers solar sell, avoids grid buy in high-emission hours)
  - P2P trading (exports Zone B surplus when profitable)
  - DR bids (accepts when SoC is high enough to commit)
  - EV deferral (defers when price is high, replays when cheap)
  - Grid islanding (hoards battery during islanding, sells on reconnection)

Usage:
  python baseline_inference.py --agent rule            # no API key needed
  python baseline_inference.py --agent llm             # needs API key
  python baseline_inference.py --json-only --agent rule
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, Optional

import requests
from openai import OpenAI

from models import VppAction
from server.task_curves import ALL_TASK_IDS, ISLANDING_START, ISLANDING_END

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--json-only", action="store_true")
parser.add_argument("--agent", choices=["llm", "rule"], default="llm")
args, _ = parser.parse_known_args()

JSON_ONLY  = args.json_only
AGENT_TYPE = args.agent


def _log(*msg):
    if not JSON_ONLY:
        print(*msg)


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY",   "")
HF_TOKEN       = os.getenv("HF_TOKEN",        "") or os.getenv("API_KEY", "")
API_BASE_URL   = os.getenv("API_BASE_URL",    "")
MODEL_NAME_ENV = os.getenv("MODEL_NAME",      "")

client: Optional[OpenAI] = None
DEFAULT_MODEL: str = ""

if AGENT_TYPE == "llm":
    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY, max_retries=0)
        DEFAULT_MODEL = MODEL_NAME_ENV or "gpt-4o-mini"
    elif GROQ_API_KEY:
        client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1", max_retries=0)
        DEFAULT_MODEL = MODEL_NAME_ENV or "llama-3.1-8b-instant"
    elif HF_TOKEN:
        base = API_BASE_URL or "https://router.huggingface.co/v1"
        client = OpenAI(api_key=HF_TOKEN, base_url=base, max_retries=0)
        DEFAULT_MODEL = MODEL_NAME_ENV or "Qwen/Qwen2.5-72B-Instruct"
    else:
        if not JSON_ONLY:
            print("WARNING: No API key found. Falling back to rule-based agent.", file=sys.stderr)
        AGENT_TYPE = "rule"

if AGENT_TYPE == "rule":
    DEFAULT_MODEL = "rule-based-smart-agent-v2"
    _log("Using extended rule-based agent (no API key required).")

SERVER_URL = os.getenv("VPP_SERVER_URL", "http://localhost:7860")

# ---------------------------------------------------------------------------
# Extended rule-based smart agent
# ---------------------------------------------------------------------------

def rule_based_action(obs: dict, task_id: str) -> Dict[str, Any]:
    """
    Priority-ordered heuristic covering all extended mechanics:

      1. Grid islanding → hoard battery, no grid transactions
      2. Grid reconnection spike → sell immediately at max rate
      3. Grid frequency emergency (< 49.8 Hz) → discharge immediately
      4. Battery critically low (< 20 % SoC) → stop selling
      5. DR bid available + SoC high → accept and deliver
      6. Price spike (> 200 $/MWh) → sell at full power
      7. High price (> 55 $/MWh) + SoC > 35 % → sell at 70 %
      8. EV deferral: defer when price is high, replay when cheap
      9. P2P: enable Zone B export when surplus solar available
      10. Solar surplus + battery > 70 % → sell excess at 50 %
      11. Cheap power (< 38 $/MWh) + SoC < 60 % → buy and store
      12. Default: idle
    """
    telemetry   = obs.get("telemetry", [])
    freq        = obs.get("grid_frequency_hz", 50.0)
    price       = obs.get("market_price_per_mwh", 50.0)
    grid_conn   = obs.get("grid_connected", True)
    step        = obs.get("step_id", 0)
    carbon_bal  = obs.get("carbon_credits_balance", 0.0)
    dr_bid      = obs.get("dr_bid", {})
    zone_aggs   = obs.get("zone_aggregates", [])

    mean_soc    = sum(t["soc"] for t in telemetry) / max(len(telemetry), 1)
    mean_soh    = sum(t.get("state_of_health", 1.0) for t in telemetry) / max(len(telemetry), 1)
    mean_solar  = sum(t["current_solar_gen_kw"] for t in telemetry) / max(len(telemetry), 1)

    zone_b      = next((z for z in zone_aggs if z.get("zone_id") == "zone-b"), {})
    p2p_avail   = zone_b.get("p2p_available_kw", 0.0)

    # DR bid info
    dr_active   = dr_bid.get("active", False)
    dr_premium  = dr_bid.get("premium_multiplier", 1.0)
    dr_power    = dr_bid.get("committed_power_kw", 0.0)
    dr_steps_remaining = dr_bid.get("steps_remaining", 0)

    # EV deferral: defer when price > 60 and before step 38; replay if near deadline
    defer_ev    = 0.0
    if step >= 32:
        if price > 60.0 and step < 38:
            defer_ev = 0.8
        elif step >= 38:
            defer_ev = 0.0   # replay deferred EV charging before deadline

    # P2P: export when Zone B has surplus
    p2p_rate    = 0.8 if p2p_avail > 1.0 else 0.0

    # 1. Grid islanding — no grid; hoard battery for islanding window
    if not grid_conn:
        # During islanding: idle, preserve charge, no P2P/grid trading
        return {"global_charge_rate": 0.0, "min_reserve_pct": 0.40,
                "defer_ev_charging": 0.0, "accept_dr_bid": False,
                "p2p_export_rate": 0.0}

    # 2. Reconnection spike (task-specific: right after islanding)
    if "islanding" in task_id and step == ISLANDING_END and mean_soc > 0.40:
        return {"global_charge_rate": -1.0, "min_reserve_pct": 0.20,
                "defer_ev_charging": defer_ev, "accept_dr_bid": False,
                "p2p_export_rate": p2p_rate}

    # 3. Grid frequency emergency
    if freq < 49.8:
        return {"global_charge_rate": -1.0, "min_reserve_pct": 0.10,
                "defer_ev_charging": 0.0, "accept_dr_bid": False,
                "p2p_export_rate": p2p_rate}

    # 4. Battery critically low
    if mean_soc < 0.20:
        rate = 0.4 if price < 42.0 else 0.0
        return {"global_charge_rate": rate, "min_reserve_pct": 0.20,
                "defer_ev_charging": 0.0, "accept_dr_bid": False,
                "p2p_export_rate": 0.0}

    # 5. DR bid: accept if premium is high and SoC supports delivery
    if dr_active and not dr_steps_remaining and dr_premium >= 2.0 and mean_soc > 0.50:
        # Accepting means we need to discharge at dr_power — feasible if SoC is enough
        commit_fraction = min(1.0, dr_power / 5.0)
        return {"global_charge_rate": -commit_fraction, "min_reserve_pct": 0.20,
                "defer_ev_charging": defer_ev, "accept_dr_bid": True,
                "p2p_export_rate": p2p_rate}

    # If currently in DR commitment: keep delivering
    if dr_steps_remaining > 0:
        commit_fraction = min(1.0, dr_power / 5.0)
        return {"global_charge_rate": -commit_fraction, "min_reserve_pct": 0.20,
                "defer_ev_charging": defer_ev, "accept_dr_bid": False,
                "p2p_export_rate": p2p_rate}

    # 6. Price spike
    if price > 200.0:
        return {"global_charge_rate": -1.0, "min_reserve_pct": 0.20,
                "defer_ev_charging": defer_ev, "accept_dr_bid": False,
                "p2p_export_rate": p2p_rate}

    # 7. High price with sufficient SoC
    if price > 55.0 and mean_soc > 0.35:
        # Reduce cycling if SoH is getting low (battery degradation awareness)
        rate = -0.5 if mean_soh < 0.92 else -0.7
        return {"global_charge_rate": rate, "min_reserve_pct": 0.20,
                "defer_ev_charging": defer_ev, "accept_dr_bid": False,
                "p2p_export_rate": p2p_rate}

    # 8. Solar surplus and full battery → sell excess
    if mean_solar > 2.0 and mean_soc > 0.70:
        return {"global_charge_rate": -0.5, "min_reserve_pct": 0.20,
                "defer_ev_charging": defer_ev, "accept_dr_bid": False,
                "p2p_export_rate": p2p_rate}

    # 9. Cheap grid power → buy, but avoid high-emission hours if carbon balance is negative
    if price < 38.0 and mean_soc < 0.60:
        charge_rate = 0.5
        if step < 17 and carbon_bal < -2.0:   # avoid buying in high-emission hours if carbon hurts
            charge_rate = 0.0
        return {"global_charge_rate": charge_rate, "min_reserve_pct": 0.20,
                "defer_ev_charging": defer_ev, "accept_dr_bid": False,
                "p2p_export_rate": 0.0}

    # 10. Default: idle
    return {"global_charge_rate": 0.0, "min_reserve_pct": 0.20,
            "defer_ev_charging": defer_ev, "accept_dr_bid": False,
            "p2p_export_rate": p2p_rate}


# ---------------------------------------------------------------------------
# LLM agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert energy trader managing a Virtual Power Plant (VPP).
You maximise a multi-objective score: profit (50%), safety (20%), carbon credits (15%),
battery health (10%), and demand-response participation (5%).

Core rules:
- global_charge_rate < 0  → sell to grid (earns money, drains batteries)
- global_charge_rate > 0  → buy from grid (costs money, charges batteries)
- If grid_frequency_hz < 49.8 OR grid_connected=False: NEVER buy from grid
- If grid_connected=False (islanding): set charge_rate=0.0, preserve battery for when grid reconnects
- Accept DR bids (accept_dr_bid=true) only when SoC > 0.50 AND premium >= 2.0
- defer_ev_charging > 0 reduces immediate Zone B EV load (saves battery) but must be repaid by step 40
- p2p_export_rate > 0 earns revenue by routing Zone B solar surplus to Zone A (better than grid export)
- Avoid unnecessary charging during high-emission steps (0-16) to protect carbon credits
- Higher min_reserve_pct protects safety but reduces revenue opportunity

Respond ONLY with a valid JSON object."""

ACTION_PROMPT = """Step: {step_id}/47 ({time_of_day})
Price: ${price:.2f}/MWh  |  Freq: {freq:.2f} Hz  |  Grid: {grid_status}
Mean SoC: {mean_soc:.1%}  |  Mean SoH: {mean_soh:.3f}
Solar: {solar:.2f} kW/home  |  Demand: {demand:.2f} kW/home
Carbon balance: {carbon:.2f} credits
DR bid: {dr_info}
P2P available: {p2p:.2f} kW/home
Price forecast (4-step): {price_forecast}  ± {price_uncertainty}
Solar forecast (4-step): {solar_forecast}  ± {solar_uncertainty}
Task: {task_id}

Respond with JSON only:
{{"global_charge_rate": <-1.0 to 1.0>, "min_reserve_pct": <0.0 to 1.0>,
  "defer_ev_charging": <0.0 to 1.0>, "accept_dr_bid": <true|false>,
  "p2p_export_rate": <0.0 to 1.0>}}"""


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
        "grid_status":      "CONNECTED" if obs.get("grid_connected", True) else "⚠️ ISLANDED",
        "mean_soc":         sum(socs) / len(socs),
        "mean_soh":         sum(sohs) / len(sohs),
        "solar":            sum(solar) / max(len(solar), 1),
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
    import re
    text = re.sub(r"```(?:json)?", "", text).strip()
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end+1])
    return json.loads(text)


def get_llm_action(obs: dict, task_id: str) -> VppAction:
    assert client is not None
    summary = _summarise_obs(obs)
    prompt  = ACTION_PROMPT.format(task_id=task_id, **summary)
    
    max_attempts = 4
    for attempt in range(max_attempts):
        try:
            # Simple rate limit pacing to avoid hitting Groq's 30 RPM too hard
            time.sleep(1.0)
            
            response = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.1,
                max_tokens=120,
            )
            text     = response.choices[0].message.content.strip()
            decision = _extract_json(text)
            return VppAction(**decision)
        except Exception as e:
            err_msg = str(e).lower()
            if "429" in err_msg or "rate limit" in err_msg or "too many requests" in err_msg:
                sleep_time = 4 ** attempt
                _log(f"  Rate limit hit ({e}). Sleeping for {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                _log(f"  LLM error ({e}) — fallback to rule agent.")
                raw = rule_based_action(obs, task_id)
                return VppAction(**raw)
    
    _log(f"  Max retries reached — fallback to rule agent.")
    raw = rule_based_action(obs, task_id)
    return VppAction(**raw)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_task(task_id: str, session: requests.Session) -> dict:
    _log(f"\n{'─'*55}")
    _log(f"  Task: {task_id}  |  Agent: {AGENT_TYPE}")
    _log(f"{'─'*55}")

    resp = session.post(f"{SERVER_URL}/reset", params={"task_id": task_id})
    if resp.status_code != 200:
        _log(f"  Reset failed: {resp.text}")
        return {"aggregate_score": 0.0, "error": resp.text}

    obs          = resp.json()
    total_reward = 0.0
    steps        = 0
    done         = False

    while not done:
        if AGENT_TYPE == "rule":
            raw    = rule_based_action(obs, task_id)
            action = VppAction(**raw)
        else:
            action = get_llm_action(obs, task_id)

        step_resp = session.post(
            f"{SERVER_URL}/step",
            json=action.model_dump(exclude={"reasoning"}),
        )
        if step_resp.status_code != 200:
            _log(f"  Step failed: {step_resp.text}")
            break

        data          = step_resp.json()
        obs           = data["observation"]
        total_reward += float(data["reward"])
        done          = data["done"]
        steps        += 1

    grader_data = session.get(f"{SERVER_URL}/grader").json()
    agg_score   = grader_data.get("aggregate_score", 0.0)

    _log(f"\n  Steps         : {steps}")
    _log(f"  Total reward  : {total_reward:.2f}")
    _log(f"  Profit (USD)  : ${grader_data.get('cumulative_profit_usd', 0.0):.2f}")
    _log(f"  P2P (USD)     : ${grader_data.get('cumulative_p2p_usd', 0.0):.2f}")
    _log(f"  Carbon bal.   : {grader_data.get('carbon_credits_balance', 0.0):.3f}")
    _log(f"  Mean SoH      : {grader_data.get('mean_state_of_health', 1.0):.4f}")
    _log(f"  Safety viols  : {grader_data.get('safety_violations', 0)}")
    _log(f"  DR fulfilled  : {grader_data.get('dr_bids_fulfilled', 0)}")
    _log(f"  Pareto score  : {agg_score:.4f}")
    _log(f"    profit={grader_data.get('profit_score',0):.3f}  "
         f"safety={grader_data.get('safety_score',0):.3f}  "
         f"carbon={grader_data.get('carbon_score',0):.3f}  "
         f"degrad={grader_data.get('degradation_score',0):.3f}  "
         f"dr={grader_data.get('dr_score',0):.3f}")

    return {
        "aggregate_score":     agg_score,
        "profit_score":        grader_data.get("profit_score", 0.0),
        "safety_score":        grader_data.get("safety_score", 0.0),
        "carbon_score":        grader_data.get("carbon_score", 0.0),
        "degradation_score":   grader_data.get("degradation_score", 0.0),
        "dr_score":            grader_data.get("dr_score", 0.0),
        "total_reward":        round(total_reward, 4),
        "steps":               steps,
        "profit_usd":          grader_data.get("cumulative_profit_usd", 0.0),
        "p2p_usd":             grader_data.get("cumulative_p2p_usd", 0.0),
        "carbon_credits":      grader_data.get("carbon_credits_balance", 0.0),
        "mean_soh":            grader_data.get("mean_state_of_health", 1.0),
        "safety_violations":   grader_data.get("safety_violations", 0),
        "dr_fulfilled":        grader_data.get("dr_bids_fulfilled", 0),
        "dr_failed":           grader_data.get("dr_bids_failed", 0),
        "model":               DEFAULT_MODEL,
        "agent_type":          AGENT_TYPE,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _log(f"VPP Extended Baseline Inference")
    _log(f"Server : {SERVER_URL}  |  Agent : {AGENT_TYPE}  |  Model : {DEFAULT_MODEL}")

    try:
        h = requests.get(f"{SERVER_URL}/health", timeout=5)
        assert h.status_code == 200
    except Exception as e:
        if not JSON_ONLY:
            print(f"ERROR: Cannot reach server at {SERVER_URL}: {e}", file=sys.stderr)
        sys.exit(1)

    session = requests.Session()
    results = {}
    for task in ALL_TASK_IDS:
        results[task] = run_task(task, session)

    if not JSON_ONLY:
        print(f"\n{'='*60}")
        print("  FINAL BASELINE SCORES (Pareto)")
        print(f"{'='*60}")
        for task, data in results.items():
            bar = "█" * int(data["aggregate_score"] * 20)
            print(f"  {task:<35}  {data['aggregate_score']:.4f}  {bar}")

    with open("baseline_scores.json", "w") as f:
        json.dump(results, f, indent=2)

    if not JSON_ONLY:
        print("\nSaved → baseline_scores.json")
    else:
        print(json.dumps(results))


if __name__ == "__main__":
    main()