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

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103.2-009688.svg)](https://fastapi.tiangolo.com/)

**OpenEnv Environment** — An advanced AI agent framework designed to manage 100 home batteries within a simulated neighborhood. The objective is to maximize a **multi-objective Pareto score** across five distinct difficulty tiers. The simulation incorporates complex dynamics including battery degradation, carbon credits, peer-to-peer (P2P) energy trading, demand-response auctions, and grid islanding emergencies.

---

## Key Features (Extended Edition)

| Feature | Description | Strategic Impact |
|:---|:---|:---|
| **Battery Degradation (SoH)** | Each full cycle degrades capacity by 0.001 (SoH floors at 80%). | Establishes a temporal objective: maximize profit without deteriorating battery asset health. |
| **Carbon Credits Subsystem** | Solar generation earns 0.05 credits/kWh. Grid purchases during high-emission periods cost 0.08 credits/kWh. | Introduces a multi-objective optimization goal balancing financial profit with environmental sustainability. |
| **Forecast Uncertainty** | `forecast_price_uncertainty` and `forecast_solar_uncertainty` expand over the forecasting horizon. | Necessitates the development of risk-averse, robust agent behavior. |
| **Multi-Objective Pareto Grader** | 5-vector evaluation score: Profit (50%), Safety (20%), Carbon (15%), Degradation (10%), and Demand Response (5%). | Accurately models real-world Virtual Power Plant (VPP) operational priorities. |
| **P2P Energy Trading** | Zone B's solar surplus can be routed to Zone A at a midpoint price via the `p2p_export_rate` action. | Facilitates demand-side management entirely independent of grid infrastructure. |
| **Demand Response Auctions** | The grid initiates a bid every 6 steps at a 1.5–3.0× premium. Agents commit using `accept_dr_bid`. | Tests an agent's capability to commit to load shedding under uncertainty. |
| **Islanding Emergencies** | Grid disconnection occurs between steps 20–29, followed by an 8× price surge upon reconnection. | Evaluates the model's capacity for mid-simulation strategy switching and resilience. |
| **Load Deferral** | The `defer_ev_charging` action temporarily delays Zone B Electric Vehicle load (must be completely fulfilled by step 40). | Introduces localized demand-side flexibility options. |
| **Adversarial Weather Events** | Introduces unforecasted cloud coverage (e.g., an 80% drop in solar generation) despite a clear-sky forecast. | Rigorously tests the agent's robustness against forecasting errors. |
| **Reasoning Trace Scoring** | Captures the agent's decision-making rationale via `POST /trace`. | Facilitates research into Explainable AI (XAI) and evaluating reasoning coherence. |

---

## Quick Start Guide

### 1. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://huggingface.co/spaces/<your-username>/vpp-env
cd vpp-env
pip install -r requirements.txt
```

### 2. Server Initialization

Launch the FastAPI orchestrator server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### 3. Inference Execution

**Running the Rule-Based Baseline (No API Key Required):**
```bash
python baseline_inference.py --agent rule
```

**Running the LLM Agent:**
```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=your_hf_api_token_here
python inference.py
```

### Required Environment Variables

For rigorous evaluation, the following environment variables must be configured:

- `API_BASE_URL`: The underlying API endpoint for the OpenAI-compatible client.
- `MODEL_NAME`: The model identifier for inference execution.
- `HF_TOKEN`: Your Hugging Face token or a compatible API key.
- `LOCAL_IMAGE_NAME` *(Optional)*: Required only if utilizing `from_docker_image()`.

*Note: Default values are provided for `API_BASE_URL` and `MODEL_NAME` within `inference.py`. However, `HF_TOKEN` must be explicitly provided in your environment configuration.*

### Inference Output Contract

`inference.py` ensures structured stdout in strict evaluator compliance:

- `[START] task=<task_name> env=vpp model=<model_name>`
- `[STEP] step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>`
- `[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>`

*Diagnostic information is reserved for stderr.*

---

## System Architecture & Physics Models

### Battery Degradation Dynamics
```text
cycle_increment    = |delta_kwh| / (2 × capacity_kwh)  [per asset/step]
cumulative_cycles += cycle_increment
new_soh            = max(0.80, 1.0 - cumulative_cycles × 0.001)
effective_capacity = capacity_kwh × soh
```
Excessive cycling (buying high, selling low unnecessarily) yields measurable degradation over time (~0.12% per 12-hour episode under full load), directly penalizing operational efficiency.

### Carbon Credits
```text
earnings/step = solar_kw × 0.25h × 0.05 credits/kWh × 100 homes
expenses/step = grid_charge_kw × η × 0.25h × 0.08 credits/kWh × 100 homes 
                (expenses are incurred strictly during morning emission spikes: steps 0-16)
```

### Peer-to-Peer (P2P) Trading
```text
zone_b_surplus_kw = max(0, solar_kw - demand_kw) [per Zone B home]
p2p_exported_kw   = zone_b_surplus_kw × p2p_export_rate
p2p_price         = market_price × 0.75 [midpoint benefit against spot pricing]
p2p_revenue       = p2p_exported_kw × 0.25h × (p2p_price / 1000) [per home]
```

---

## Evaluation Tasks

The environment supports five progressive difficulty tiers:

1. **Easy — Arbitrage** (`easy-arbitrage`) 
   Ideal conditions (clear sky, low demand, consistent $50/MWh). The objective is straightforward solar surplus arbitrage. *Profit target: $500.*
2. **Medium — Forecast Error** (`medium-forecast-error`) 
   Simulates heatwaves with 4× AC demand spikes between 10:00–14:00 alongside a sinusoidal $35–$65/MWh pricing model. *Profit target: $200.*
3. **Hard — Frequency Response** (`hard-frequency-response`) 
   Grid frequencies drop to 49.5 Hz, coupled with a 10× price spike at 12:30 (step 26). Immediate discharge is mandated. *Profit target: $1000.*
4. **Expert — Demand Response** (`expert-demand-response`) 
   Demand response bids appear every 6 steps. An adversarial cloud event severely limits solar yield at step 24, forcing agents to balance bid commitments against weather risks. *Profit target: $800.*
5. **Islanding Emergency** (`islanding-emergency`) 
   The grid fails at 11:00 (step 20) for 10 consecutive steps. The agent must sustain 100 homes independently until grid reconnection at 13:30 (step 30), which introduces an 8× price spike. *Profit target: $400.*

---

## State and Action Geometries

### Action Space Outline

The extended environment leverages an enriched multi-dimensional action space. Structured JSON format is expected for all action submissions.

```json
{
  "global_charge_rate": -0.8,
  "min_reserve_pct": 0.2,
  "defer_ev_charging": 0.5,
  "accept_dr_bid": true,
  "p2p_export_rate": 0.3,
  "reasoning": "Price is elevated at $65/MWh with SoC at 72%. Executing sell at -0.7 rate and fulfilling DR bid due to reserve adequacy."
}
```

| Field | Type | Range | Description |
|:---|:---|:---|:---|
| `global_charge_rate` | float | `[-1, +1]` | Unified charge/discharge rate. |
| `min_reserve_pct` | float | `[0, 1]` | Predefined safety State of Charge (SoC) floor. |
| `defer_ev_charging` | float | `[0, 1]` | Fraction of Zone B's EV demand to postpone (must be returned by step 40). |
| `accept_dr_bid` | boolean | — | Action to secure the current active demand-response bid. |
| `p2p_export_rate` | float | `[0, 1]` | Fraction of excess Zone B solar output directed to Zone A. |
| `reasoning` | string | `≤500 chars` | Explanatory rationale evaluated during Trace Scoring. |

### Expanded Observation Signals

Key additions to the state representations include:
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
    { "zone_id": "zone-a", "p2p_available_kw": 0.0 },
    { "zone_id": "zone-b", "mean_soh": 0.993, "p2p_available_kw": 1.82 }
  ],
  "telemetry": [
    { "asset_id": "home-000", "soc": 0.68, "state_of_health": 0.994 }
  ]
}
```

- `grid_connected`: A boolean flag critical during islanding simulations. Avoid generic grid trading when `False`.
- `carbon_credits_balance`: Total accumulated emission offsets.
- `forecast_price_uncertainty` / `forecast_solar_uncertainty`: 1-sigma distribution bands.
- `dr_bid`: Contains active bid states, premiums, and commitment windows.
- `state_of_health`: Granular track of battery degradation per asset.
- `p2p_available_kw`: Assess available localized export energy reserves.

---

## The Pareto Grader

Scoring relies on a multi-dimensional Pareto approach to balance competing utility functions:

```text
profit_score       = min(1, total_profit / profit_target)
safety_score       = 1 - (violation_ratio × 0.60) - (emergency_ratio × 0.40)
carbon_score       = min(1, carbon_balance / carbon_target)
degradation_score  = (mean_soh - 0.80) / 0.20
dr_score           = fulfilled_bids / accepted_bids (Defaults to 1.0 if unaffected)

Aggregate Score = (0.50 × profit) + (0.20 × safety) + (0.15 × carbon) + (0.10 × degradation) + (0.05 × DR)
```

The comprehensive response structure from `GET /grader` displays performance vectors including aggregated totals, cumulative bonus distributions, missing targets, and current degradation indicators.

---

## Reasoning Trace Implementation

Submit cognitive reasoning alongside an execution step specifically via `POST /trace`.

```bash
curl -X POST "http://localhost:7860/trace?reasoning=Price+is+high+at+%2465" \
  -H "Content-Type: application/json" \
  -d '{"global_charge_rate": -0.7, "min_reserve_pct": 0.2}'
```

All semantic rationales are permanently logged in the session memory. `GET /traces` retrieves operations logs post-episode, enabling quality evaluation using parallel Language Models.

---

## API Reference

| HTTP Method | Endpoint | Functional Description |
|:---:|:---|:---|
| `GET` | `/health` | Core subsystem liveness probe. |
| `GET` | `/tasks` | Retrieves action schemas, observation schema, and task listings. |
| `POST` | `/reset?task_id=...` | Initializes a fresh environment episode. |
| `POST` | `/step` | Executes a simulation step via the extended action schema. |
| `POST` | `/trace?reasoning=...` | Executes an action while recording explainable reasoning traces. |
| `GET` | `/traces` | Consolidates and returns recorded reasoning arrays for the active session. |
| `GET` | `/state` | Provides standard ground-truth state observations. |
| `GET` | `/grader` | Returns real-time aggregate Pareto scores. |
| `GET` | `/baseline` | Retrieves pre-computed, cached baseline performance metrics. |
| `GET` | `/baseline?refresh=true` | Forces a live recalculation of baseline scores. |

---

## Reinforcement Learning (RL) Curriculum

A standard setup to train a custom agent via Proximal Policy Optimization (PPO):

```bash
# Initiate Environment Backend
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Start 5-Phase Curriculum Execution
python train_rl.py
```

**Curriculum Trajectory:**
1. **Easy** *(200k steps)* — Arbitrage basics.
2. **Medium** *(150k steps)* — Weather patterns and demand turbulence.
3. **Hard** *(150k steps)* — Strategic spike maneuvering.
4. **Expert** *(150k steps)* — Integrating Auctions, P2P networks, and complex forecasting.
5. **Islanding** *(100k steps)* — Extreme contingency resilience and system autonomy.

---

## Deployment & Validations

**Docker Build & Verification:**
```bash
docker build -t openenv-vpp .
docker run --rm -p 7860:7860 openenv-vpp
```

*Sanity Check Status:*
```bash
curl http://localhost:7860/health
```

**Local Framework Validation Suite:**
```bash
# General Engine Diagnostics
openenv validate

# Internal Integration Status
python validate.py --url http://localhost:7860

# Rule-Based Baseline Check
python baseline_inference.py --agent rule --json-only
```

*For Hugging Face Spaces deployments, ensure variables (`API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`) are configured within the UI settings and the Space is logically tagged with `openenv`.*

---

## Project Structure

```text
├── inference.py             # Main submission utility block
├── baseline_inference.py    # Hybrid Baseline / LLM Reference framework
├── server/
│   ├── __init__.py
│   ├── app.py               # Main FastAPI orchestrator application
│   ├── vpp_environment.py   # Core physics simulations & logic definitions
│   └── task_curves.py       # Configuration presets and temporal schedules
├── models.py                # Systemic Pydantic typing definitions
├── client.py                # Wrapper mappings integrating with OpenEnv
├── gymwrapper.py            # Custom wrapper porting Gymnasium endpoints
├── train_rl.py              # PPO Reinforcement algorithm bootstrap
├── demo.py                  # Live multi-agent orchestration preview
├── validate.py              # Unit testing sanity-check execution utility
├── tests/                   # Independent testing suites
│   ├── test_curves.py
│   ├── test_vpp_environment.py
│   └── test_grader.py
├── baseline_scores.json     # Saved Pareto evaluations reference baseline
├── Dockerfile               # Automated compilation manifest
└── requirements.txt         # Required Python packages matrix
```