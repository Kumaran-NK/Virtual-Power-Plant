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
| **Demand Response Auction** | Every 6 steps, grid posts a bid (1.5–3.0× premium); accepted windows are scored by delivered-vs-committed energy with escalating failure penalties | Commitment quality under uncertainty |
| **Grid Islanding Emergency** | New `islanding-emergency` task: grid disconnects steps 20–29, critical load is prioritized, and reconnection uses a one-step soft-sync discharge cap before the 8× spike opportunity | Tests resilience-first strategy switching |
| **Hard Reserve Policy** | Home reserve floor defaults to 20%; emergency events can use reserve down to 10% | Encodes home-first safety behavior |
| **Load Deferral** | `defer_ev_charging` action delays Zone B EV load (must repay by step 40) | Demand-side flexibility |
| **Adversarial Weather** | Expert task: forecast says clear sky, but cloud event at step 24 drops solar 80% | Tests forecast robustness |
| **Reasoning Trace Capture** | Reasoning can be attached to `VppAction.reasoning` in WebSocket sessions; HTTP `/trace` and `/traces` are guidance-only in stateless mode | Supports explainability workflows in session-based runs |

---

## Quick Start

```bash
# 1. Install
git clone https://huggingface.co/spaces/<your-username>/vpp-env
cd vpp-env
pip install -r requirements.txt

# 2. Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# 3. (Optional) Run the rule-based baseline locally (no API key)
python baseline_inference.py --agent rule

# 4. Run inference agent (LLM if HF_TOKEN is set; otherwise rule fallback)
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=hf_xxx_or_api_key
python inference.py
```

## Inference Environment Variables

`inference.py` supports both LLM mode and rule-based fallback mode.

- `API_BASE_URL` — optional (default: `https://router.huggingface.co/v1`)
- `MODEL_NAME` — optional (default: `Qwen/Qwen2.5-72B-Instruct`)
- `HF_TOKEN` — optional for rule mode, required for LLM mode
- `VPP_SERVER_URL` — optional (default: `http://localhost:7860`)
- `LOCAL_IMAGE_NAME` — optional, used with `VppEnv.from_docker_image(...)`

If `HF_TOKEN` is missing, `inference.py` intentionally falls back to the built-in rule-based agent.

## Inference Output Contract

`inference.py` emits structured stdout in strict evaluator format:

- `[START] task=<task_name> env=vpp model=<model_name>`
- `[STEP] step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>`
- `[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>`

Only those lines are printed to stdout. Diagnostics are written to stderr.

## Judging Scope

The evaluator scores only `inference.py` output. Baseline artifacts/endpoints
(`baseline_inference.py`, `baseline_scores.json`, `/baseline`) are for local benchmarking
and debugging only; 

---

## Tasks (5 Total)

### Easy — Arbitrage (`easy-arbitrage`) ⭐☆☆☆☆
Clear sky, low demand, and simple TOU pricing (cheap daytime, expensive evening). Profit target: $500.

### Medium — Forecast Error (`medium-forecast-error`) ⭐⭐☆☆☆
Heatwave: AC demand spikes 4× from 10:00–14:00 and real solar drops suddenly around midday while forecasts remain optimistic. Profit target: $200.

### Hard — Frequency Response (`hard-frequency-response`) ⭐⭐⭐☆☆
10× price spike at 12:30 (step 26). Grid frequency drops to 49.5 Hz. Must discharge immediately. Profit target: $1000.

### Expert — Demand Response (`expert-demand-response`) ⭐⭐⭐⭐☆
DR bids every 6 steps (1.5–3.0× premium). Accepted windows are graded by delivered-vs-committed energy, and consecutive failed accepted windows escalate penalties (2×, 3×, then 4×). **Adversarial cloud event at step 24** (forecast says clear sky, solar drops 80%). Demand spike at step 20. Agent must decide which bids to accept vs risk. Profit target: $800.

### Islanding Emergency (`islanding-emergency`) ⭐⭐⭐☆☆
Grid disconnects at 11:00 (step 20) for 10 steps. Agent gets `grid_connected=False` warning and must prioritize critical household load while flexible load can be shed first. Only unserved critical load counts as blackout penalty. Grid reconnects at 13:30 (step 30) with **8× price spike** and a one-step soft-sync discharge cap (50% max discharge on the first reconnected step). Profit target: $400.

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
  "safety_margin_pct": 14.3,
  "emergency_active": false,
  "demand_shed_this_step_kwh": 0.0,
  "cumulative_demand_shed_kwh": 0.0,
  "carbon_earned_this_step": 3.08,
  "carbon_spent_this_step": 0.0,
  "grid_frequency_trend_hz": [50.0, 50.0, 49.5],
  "response_latency_steps_to_emergency": 0,
  "forecast_realtime_error_price_usd": 0.0,
  "forecast_realtime_error_solar_kw": -1.62,
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
- `emergency_active` — true during frequency emergencies or islanding periods
- `carbon_credits_balance` — track your carbon footprint
- `carbon_earned_this_step` / `carbon_spent_this_step` — immediate carbon trade-off signals
- `safety_margin_pct` — reserve headroom above active floor
- `demand_shed_this_step_kwh` — unmet demand signal when constrained by reserve limits
- `grid_frequency_trend_hz` — trailing 3-step frequency trend
- `response_latency_steps_to_emergency` — reaction delay when frequency event occurs
- `forecast_*_uncertainty` — 1-σ bands; larger = less reliable forecast
- `forecast_realtime_error_*` — actual vs forecast mismatch at current step
- `dr_bid` — check `active`, `premium_multiplier`, `committed_steps` before accepting
- `state_of_health` — monitor battery degradation per home
- `p2p_available_kw` — Zone B surplus available for P2P export

---

## Pareto Grader

```text
eps              = 1e-4
profit_score     = clip(total_profit / profit_target, eps, 1 - eps)

violation_ratio  = safety_violations / 48
emergency_ratio  = min(1, grid_emergencies_ignored / 48)
demand_shed_ratio = min(1, cumulative_demand_shed_kwh / 30)
latency_penalty  = min(0.15, response_latency_steps_to_emergency * 0.05)  # if latency >= 0

safety_score     = clip(
                    1
                    - 0.40 * violation_ratio
                    - 0.30 * emergency_ratio
                    - 0.30 * demand_shed_ratio
                    - latency_penalty,
                    eps,
                    0.999
                  )
if islanding_blackout_home_steps > 0:
  safety_score   = clip(safety_score - min(0.30, 0.01 * islanding_blackout_home_steps), eps, 0.999)

carbon_score      = clip(carbon_balance / carbon_target, eps, 1 - eps)
degradation_score = clip((mean_soh - 0.80) / 0.20, eps, 1 - eps)

if accepted_bids > 0:
  dr_score = clip(mean_fulfillment_ratio, eps, 1 - eps)
elif task_has_dr_auction:
  dr_score = eps
else:
  dr_score = 0.90

aggregate = clip(
  0.50 * profit + 0.20 * safety + 0.15 * carbon + 0.10 * degradation + 0.05 * dr,
  eps,
  1 - eps
)
```

Per-step observations expose the full `ParetoScore` as top-level `pareto_score` (and in metadata for compatibility):
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
  "cumulative_demand_shed_kwh": 6.5,
  "response_latency_steps_to_emergency": 1,
  "islanding_blackout_home_steps": 2,
  "islanding_blackout_unique_homes": 2,
  "dr_bids_fulfilled": 3,
  "dr_bids_failed": 1,
  "dr_consecutive_failures": 0,
  "dr_mean_fulfillment_ratio": 0.75
}
```

## Step Reward (Shaped Signal)

Per-step reward is now a shaped operational signal, not profit alone:

```text
reward = step_profit
  + carbon_step_bonus
  + safety_margin_bonus
  + emergency_response_bonus
  - safety_violation_penalty
  - emergency_miss_penalty
  - degradation_penalty
```

Interpretation:
- Keep homes safe first (reserve margin and emergency response)
- Profit remains primary, but unsafe dispatch is penalised
- Carbon and degradation are visible every step, not only at episode end

### Active Reserve Floors

- Normal operation: minimum reserve floor is 20%
- Emergency operation (`grid_frequency_hz < 49.8` or `grid_connected=false`): reserve may drop to 10%
- Agent-provided `min_reserve_pct` is respected if stricter than these floors

---

## Reasoning Traces

In the current server setup, custom trace endpoints are not stateful over raw HTTP:

- `POST /trace` returns guidance to use WebSocket sessions
- `GET /traces` returns guidance to use WebSocket sessions

To capture reasoning, send `reasoning` in `VppAction` while stepping through a persistent WebSocket session using `client.VppEnv`.

---

## Battery Degradation Physics

```
cycle_increment     = |delta_kwh| / (2 × capacity_kwh)   per asset per step
cumulative_cycles  += cycle_increment
new_soh             = max(0.80, 1.0 − cumulative_cycles × 0.001)
effective_capacity  = capacity_kwh × soh
```

A battery cycled at full rate every step degrades ~0.12% over a 12-hour episode. Agents that unnecessarily cycle batteries (buying high, selling low) will see measurable SoH degradation across repeated episodes.

---

## Carbon Credits Physics

```text
earned per step = solar_kw × 0.25 h × 0.05 credits/kWh × 100 homes
spent per step  = grid_charge_kw × 0.25 h × 0.08 credits/kWh × 100 homes
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
| GET | `/tasks` | List all 5 tasks + action/observation/reward schemas |
| POST | `/reset` | Start new episode |
| POST | `/step` | Take one action (extended action schema) |
| POST | `/trace?reasoning=...` | HTTP guidance endpoint (returns 400; use WebSocket sessions) |
| GET | `/traces` | HTTP guidance endpoint (returns 400; use WebSocket sessions) |
| GET | `/state` | Ground-truth state (debugging) |
| GET | `/grader` | Guidance for consuming `pareto_score` from step observations |
| GET | `/baseline` | Cached baseline scores (local benchmarking only) |
| GET | `/baseline?refresh=true` | Recompute baseline scores live (local benchmarking only) |

---

## OpenEnv and Pre-Submission Validation

```bash
# OpenEnv schema/runtime validation
openenv validate

# Submission validator (expects a reachable deployment URL)
python validate.py https://your-space.hf.space

# Optional local baseline reproducibility check (not part of competition judging)
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

---

## Project Structure

```
├── inference.py             # Submission entry point (LLM + rule fallback)
├── baseline_inference.py    # Deterministic rule-based baseline scorer
├── baseline_scores.json     # Cached baseline Pareto scores
├── server/
│   ├── __init__.py
│   ├── app.py               # FastAPI (POST /trace, GET /tasks, GET /baseline)
│   ├── vpp_environment.py   # Core simulation (all 10 new mechanics)
│   └── task_curves.py       # All 5 task curves + DR schedule
├── models.py                # Extended Pydantic schemas
├── client.py                # WebSocket client wrapper around OpenEnv EnvClient
├── validate.py              # Pre-submission smoke test
├── openenv.yaml             # OpenEnv manifest
├── pyproject.toml           # Python package metadata
├── requirements.txt         # Runtime dependencies
├── Dockerfile
├── tests/
│   ├── test_comprehensive.py
│   ├── test_inference.py
│   └── test_websocket_sessions.py
└── __init__.py
```
