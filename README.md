---
title: Virtual Power Plant Orchestrator — Extended Edition
emoji: ⚡
colorFrom: green
colorTo: blue
sdk: docker
app_file: Dockerfile
app_port: 7860
python_version: "3.10"
tags:
  - fastapi
  - reinforcement-learning
  - virtual-power-plant
  - energy
  - simulation
license: mit
---

# Virtual Power Plant Orchestrator — Extended Edition

> **OpenEnv environment** — an AI agent manages 100 home batteries in a simulated neighbourhood to maximise a **multi-objective Pareto score** across five difficulty tiers, incorporating battery degradation, carbon credits, P2P trading, demand-response auctions, and grid islanding emergencies.

---

## What's New (Extended Edition)

| Feature | Description | Impact |
|---|---|---|
| **Battery Degradation (SoH)** | Each cycle degrades capacity by 0.001 per full cycle; SoH floors at 80% | New temporal objective: earn money without killing batteries |
| **Carbon Credits Subsystem** | Solar earns 0.05 credits/kWh; grid purchase in high-emission hours costs 0.08 credits/kWh | Multi-objective grader: profit + carbon together |
| **Forecast Confidence Bands** | `forecast_price_uncertainty` and `forecast_solar_uncertainty` grow with horizon | Enables risk-averse agent behaviour |
| **Pareto Multi-objective Grader** | 5-vector score: profit (50%) + safety (20%) + carbon (15%) + degradation (10%) + DR (5%) | Matches real VPP operator objectives |
| **P2P Energy Trading** | Zone B solar surplus routes to Zone A at midpoint price via `p2p_export_rate` action | Demand-side control without grid involvement |
| **Demand Response Auction** | Every 6 steps, grid posts a bid (1.5–3.0× premium); agent commits via `accept_dr_bid` | Commitment under uncertainty — frontiers LLMs struggle with |
| **Grid Islanding Emergency** | New `islanding-emergency` task: grid disconnects steps 20–29, reconnects with 8× spike | Tests mid-episode strategy switching |
| **Load Deferral** | `defer_ev_charging` action delays Zone B EV load (must repay by step 40) | Demand-side flexibility |
| **Adversarial Weather** | Expert task: forecast says clear sky, but cloud event at step 24 drops solar 80% | Tests forecast robustness |
| **Reasoning Trace Scorer** | `POST /trace` stores agent reasoning; evaluated for coherence vs action taken | Supports research into explainable agents |

---

## Quick Start Guide

### 1. Installation

Clone the repository and install the required dependencies:

### Install

```bash
# Clone the repo
git clone <your-repo>
cd Virtual-Power-Plant

# Install dependencies (requires Python 3.10+)
uv sync

# Or: pip install -e .
```

### Run the Server

```bash
# Start on port 8000
python -m server.app --host 0.0.0.0 --port 8000

# Or with uvicorn directly
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Health check:
```bash
# 1. Install
git clone https://huggingface.co/spaces/<your-username>/vpp-env
cd vpp-env
pip install -r requirements.txt

# 2. Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# 3. Run the rule-based baseline (no API key)
python baseline_inference.py --agent rule

# 4. Run the LLM agent
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=hf_xxx_or_api_key
python inference.py
```

## Required Environment Variables

The evaluator expects these variables to be configured in your runtime environment:

- `API_BASE_URL` — API endpoint used by the OpenAI-compatible client
- `MODEL_NAME` — model identifier used for inference
- `HF_TOKEN` — Hugging Face token (or compatible API key)
- `LOCAL_IMAGE_NAME` — optional, only if your workflow uses `from_docker_image()`

Defaults in `inference.py` are set only for:

- `API_BASE_URL`
- `MODEL_NAME`

`HF_TOKEN` has no default and must be provided in deployment/runtime config.

The submission script is `inference.py` at project root and uses the OpenAI Python client interface for all LLM calls.

## Inference Output Contract

`inference.py` emits structured stdout in strict evaluator format:

- `[START] task=<task_name> env=vpp model=<model_name>`
- `[STEP] step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>`
- `[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>`

Only those lines are printed to stdout. Diagnostics are written to stderr.

---

## Tasks (5 Total)

### Easy — Arbitrage (`easy-arbitrage`) ⭐☆☆☆☆
Clear sky, low demand, flat $50/MWh. Sell solar surplus. Profit target: $500.

### Medium — Forecast Error (`medium-forecast-error`) ⭐⭐☆☆☆
Heatwave: AC demand spikes 4× from 10:00–14:00. Sinusoidal $35–$65/MWh pricing. Profit target: $200.

### Hard — Frequency Response (`hard-frequency-response`) ⭐⭐⭐☆☆
10× price spike at 12:30 (step 26). Grid frequency drops to 49.5 Hz. Must discharge immediately. Profit target: $1000.

### Expert — Demand Response (`expert-demand-response`) ⭐⭐⭐⭐☆
DR bids every 6 steps (1.5–3.0× premium). **Adversarial cloud event at step 24** (forecast says clear sky, solar drops 80%). Demand spike at step 20. Agent must decide which bids to accept vs risk. Profit target: $800.

### Islanding Emergency (`islanding-emergency`) ⭐⭐⭐⭐☆
Grid disconnects at 11:00 (step 20) for 10 steps. Agent gets `grid_connected=False` warning. Must keep 100 homes powered from batteries alone. Grid reconnects at 13:30 (step 30) with **8× price spike** — agent must have charge ready. Profit target: $400.

---

## Action Space (Extended)

```json
{
  "global_charge_rate": -0.8,
  "min_reserve_pct": 0.2,
  "defer_ev_charging": 0.5,
  "accept_dr_bid": true,
  "p2p_export_rate": 0.3,
  "reasoning": "Price is high at $65/MWh and SoC is 72%; selling at -0.7 rate. Accepting DR bid with 2.5× premium since SoC supports delivery."
}
```

| Field | Type | Range | Description |
|---|---|---|---|
| `global_charge_rate` | float | [-1, +1] | Charge/discharge rate |
| `min_reserve_pct` | float | [0, 1] | Safety SoC floor |
| `defer_ev_charging` | float | [0, 1] | Fraction of Zone B EV load to defer (repaid by step 40) |
| `accept_dr_bid` | bool | — | Accept the current demand-response grid bid |
| `p2p_export_rate` | float | [0, 1] | Fraction of Zone B solar surplus to route to Zone A |
| `reasoning` | string | ≤500 chars | Optional reasoning trace for LLM quality scoring |

---

## Observation Space (Extended)

New fields vs base version:

```json
{
  "grid_connected": true,
  "carbon_credits_balance": 3.42,
  "forecast_price_uncertainty": [2.5, 3.5, 4.5, 5.5],
  "forecast_solar_uncertainty": [0.25, 0.35, 0.50, 0.70],
  "dr_bid": {
    "active": true,
    "premium_multiplier": 2.5,
    "committed_power_kw": 2.5,
    "committed_steps": 3,
    "steps_remaining": 0
  },
  "ev_defer_deadline_step": 40,
  "p2p_last_revenue_usd": 0.84,
  "zone_aggregates": [
    { "zone_id": "zone-a", ..., "p2p_available_kw": 0.0 },
    { "zone_id": "zone-b", ..., "mean_soh": 0.993, "p2p_available_kw": 1.82 }
  ],
  "telemetry": [
    { "asset_id": "home-000", "soc": 0.68, "state_of_health": 0.994, ... }
  ]
}
```

Key new signals:
- `grid_connected` — **False** during islanding (do not trade with grid)
- `carbon_credits_balance` — track your carbon footprint
- `forecast_*_uncertainty` — 1-σ bands; larger = less reliable forecast
- `dr_bid` — check `active`, `premium_multiplier`, `committed_steps` before accepting
- `state_of_health` — monitor battery degradation per home
- `p2p_available_kw` — Zone B surplus available for P2P export

---

## Pareto Grader

```
profit_score     = min(1, total_profit / profit_target)
safety_score     = 1 − violation_ratio × 0.60 − emergency_ratio × 0.40
carbon_score     = min(1, carbon_balance / carbon_target)
degradation_score = (mean_soh − 0.80) / 0.20
dr_score          = fulfilled_bids / accepted_bids   (1.0 if no bids)

aggregate = 0.50 × profit + 0.20 × safety + 0.15 × carbon
          + 0.10 × degradation + 0.05 × DR
```

`GET /grader` returns the full `ParetoScore`:
```json
{
  "profit_score": 0.82,
  "safety_score": 0.95,
  "carbon_score": 0.67,
  "degradation_score": 0.96,
  "dr_score": 0.75,
  "aggregate_score": 0.855,
  "cumulative_profit_usd": 412.5,
  "cumulative_p2p_usd": 18.3,
  "cumulative_dr_bonus_usd": 34.2,
  "carbon_credits_balance": 4.02,
  "mean_state_of_health": 0.992,
  "dr_bids_fulfilled": 3,
  "dr_bids_failed": 1
}
```

---

## Reasoning Trace Scoring

Submit reasoning alongside an action via `POST /trace`:

```bash
curl -X POST "http://localhost:7860/trace?reasoning=Price+is+high+at+%2465" \
  -H "Content-Type: application/json" \
  -d '{"global_charge_rate": -0.7, "min_reserve_pct": 0.2}'
```

The server stores all traces. At episode end, `GET /traces` returns them all. You can evaluate reasoning quality externally using any LLM.

---

## Battery Degradation Physics

```
cycle_increment     = |delta_kwh| / (2 × capacity_kwh)   per asset per step
cumulative_cycles  += cycle_increment
new_soh             = max(0.80, 1.0 − cumulative_cycles × 0.001)
effective_capacity  = capacity_kwh × soh
```

A battery cycled at full rate every step degrades ~0.12% over a 12-hour episode. Agents that unnecessarily cycle batteries (buying high, selling low) will see measurable SoH degradation within 10 episodes of RL training.

---

## Carbon Credits Physics

```
earned per step = solar_kw × 0.25 h × 0.05 credits/kWh × 100 homes
spent per step  = grid_charge_kw × η × 0.25 h × 0.08 credits/kWh × 100 homes
                  (only during steps 0–16, the high-emission morning window)
```

---

## P2P Trading Physics

```
zone_b_surplus_kw = max(0, solar_kw − demand_kw) per Zone B home
p2p_exported_kw   = zone_b_surplus_kw × p2p_export_rate
p2p_price         = market_price × 0.75   (midpoint benefit vs spot)
p2p_revenue       = p2p_exported_kw × 0.25 h × (p2p_price / 1000) per home
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness probe |
| GET | `/tasks` | List all 5 tasks + action/observation schemas |
| POST | `/reset?task_id=...` | Start new episode |
| POST | `/step` | Take one action (extended action schema) |
| POST | `/trace?reasoning=...` | Take one action + store reasoning trace |
| GET | `/traces` | Return all reasoning traces for current episode |
| GET | `/state` | Ground-truth state (debugging) |
| GET | `/grader` | Multi-objective Pareto score |
| GET | `/baseline` | Cached baseline scores |
| GET | `/baseline?refresh=true` | Recompute baseline scores live |

---

## RL Training (5-Phase Curriculum)

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
python train_rl.py
```

## OpenEnv and Pre-Submission Validation

```bash
# OpenEnv schema/runtime validation
openenv validate

# Local validator
python validate.py --url http://localhost:7860

# Baseline reproducibility check
python baseline_inference.py --agent rule --json-only
python baseline_inference.py --agent rule --json-only
```

## Docker and HF Spaces

```bash
docker build -t openenv-vpp .
docker run --rm -p 7860:7860 openenv-vpp
```

Health check:

```bash
curl http://localhost:7860/health
```

For Hugging Face Spaces, set runtime variables (`API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`) in Space settings and tag the Space with `openenv`.

Curriculum:
1. **Easy** (200k steps) — learn basic arbitrage
2. **Medium** (150k steps) — heatwave + demand spikes
3. **Hard** (150k steps) — spike planning + frequency response
4. **Expert** (150k steps) — DR auctions + P2P + adversarial weather
5. **Islanding** (100k steps) — strategy switching, islanding survival

---

## Project Structure

```
├── inference.py             # Submission entry point (extended action schema)
├── baseline_inference.py    # Extended rule-based + LLM agent
├── server/
│   ├── __init__.py
│   ├── app.py               # FastAPI (POST /trace, Pareto /grader)
│   ├── vpp_environment.py   # Core simulation (all 10 new mechanics)
│   └── task_curves.py       # All 5 task curves + DR schedule
├── models.py                # Extended Pydantic schemas
├── client.py                # OpenEnv EnvClient wrapper
├── gymwrapper.py            # Gymnasium wrapper (39-dim obs, 5-dim action)
├── train_rl.py              # 5-phase PPO curriculum
├── demo.py                  # Multi-agent demo
├── validate.py              # Pre-submission smoke test
├── tests/
│   ├── test_curves.py
│   ├── test_vpp_environment.py
│   └── test_grader.py
├── baseline_scores.json     # Pre-computed Pareto scores
├── Dockerfile
└── requirements.txt
```