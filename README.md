# Virtual Power Plant Orchestrator

> **OpenEnv environment** — an AI agent manages 100 home batteries in a simulated neighbourhood to maximise grid profit while maintaining safety constraints across three difficulty tiers.

---

## Motivation

Renewable energy is intermittent. Solar panels generate power only when the sun shines, but cities need electricity 24 hours a day. In 2026, **Virtual Power Plants (VPPs)** solve this by aggregating thousands of home batteries into a single grid-scale asset that can be charged when energy is cheap and discharged when it is scarce.

This environment places an AI agent in the role of a VPP operator managing a neighbourhood of **100 homes**, split into two zones:

| Zone | Homes | Special Feature |
|------|-------|-----------------|
| Zone A | 000–039 | Standard homes, predictable demand |
| Zone B | 040–099 | Homes with EV chargers, higher evening load |

Each home has a **13.5 kWh battery** and a **5 kW solar panel**. The agent must:

- Decide every 15 minutes whether to **buy** energy from the grid (charge), **sell** energy back (discharge), or **idle**.
- Maximise financial profit over a 12-hour episode.
- Respect a hard safety constraint: no battery may drop below the reserved level.
- Respond instantly to grid emergencies (frequency drop events).

---

## Quick Start

### 1. Clone and install

```bash
git clone https://huggingface.co/spaces/<your-username>/vpp-env
cd vpp-env
pip install -r requirements.txt
```

### 2. Run the server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### 3. Interact via HTTP

```bash
# Reset
curl -X POST "http://localhost:7860/reset?task_id=easy-arbitrage"

# Step
curl -X POST "http://localhost:7860/step" \
  -H "Content-Type: application/json" \
  -d '{"global_charge_rate": -0.8, "min_reserve_pct": 0.2}'

# Grader
curl "http://localhost:7860/grader"
```

### 4. Docker

```bash
docker build -t vpp-env .
docker run -p 7860:7860 vpp-env
```

### 5. Run the inference script

```bash
export HF_TOKEN=hf_...       # or OPENAI_API_KEY / GROQ_API_KEY
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

# Run against the server (must be started first)
python inference.py
```

### 6. Run the baseline (rule-based, no API key required)

```bash
python baseline_inference.py --agent rule
```

---

## Deployment to Hugging Face Spaces

### Step 1 — Create a Docker Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Select **Docker** as the SDK.
3. Name it `vpp-env`.
4. Tag the Space with `openenv` in the tags field.

### Step 2 — Add secrets

In **Settings → Repository secrets**, add:

| Secret | Value | Purpose |
|--------|-------|---------|
| `HF_TOKEN` | Your HF token | LLM inference via HF Router |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model for inference |
| `OPENAI_API_KEY` | Optional | Use OpenAI instead of HF Router |
| `GROQ_API_KEY` | Optional | Use Groq instead |

### Step 3 — Push the repository

```bash
git remote add space https://huggingface.co/spaces/<your-username>/vpp-env
git push space main
```

HF will auto-detect the `Dockerfile` and build the image. The space will be live at:
`https://<your-username>-vpp-env.hf.space`

### Step 4 — Verify deployment

```bash
curl https://<your-username>-vpp-env.hf.space/health
# → {"status": "ok", "env_ready": true}
```

---

## Environment Design

### Episode structure

| Property | Value |
|---|---|
| Step duration | 15 minutes |
| Episode length | 48 steps (12 hours, 06:00–17:45) |
| Assets | 100 home batteries |
| Battery capacity | 13.5 kWh each |
| Max charge/discharge rate | 5.0 kW each |
| Round-trip efficiency | 90 % |
| Starting SoC | 50 % (all homes) |

### Zones

| Zone | Homes | EV Chargers | Demand profile |
|---|---|---|---|
| Zone A | 000–039 (40 homes) | No | Standard residential |
| Zone B | 040–099 (60 homes) | Yes | +1.2 kW/home EV adder after 14:00 |

### Physics

Each step, for every home battery:

```
effective_charge_kw = charge_kw × η   (if charging, η = 0.90)  else  charge_kw
delta_kwh           = (solar - demand + effective_charge_kw) × 0.25
new_soc             = clip(old_soc + delta_kwh / capacity_kwh, 0.0, 1.0)
grid_profit         = -charge_kw × 0.25 × (price_USD/MWh / 1000)
```

Profit is positive when selling (`charge_kw < 0`), negative when buying.

---

## Action Space

```json
{
  "global_charge_rate": -0.8,
  "min_reserve_pct": 0.2
}
```

| Field | Type | Range | Description |
|---|---|---|---|
| `global_charge_rate` | float | [-1.0, +1.0] | +1 = buy at full rate, -1 = sell at full rate, 0 = idle |
| `min_reserve_pct` | float | [0.0, 1.0] | Safety floor. Violations below this are penalised. |

---

## Observation Space

```json
{
  "timestamp": "2026-03-28T06:00:00Z",
  "step_id": 0,
  "telemetry": [
    { "asset_id": "home-000", "soc": 0.50, "current_house_load_kw": 0.3, "current_solar_gen_kw": 0.0 }
  ],
  "zone_aggregates": [
    {
      "zone_id": "zone-a", "home_count": 40, "mean_soc": 0.50,
      "min_soc": 0.50, "max_soc": 0.50, "mean_solar_kw": 0.0,
      "mean_demand_kw": 0.3, "has_ev_chargers": false
    },
    {
      "zone_id": "zone-b", "home_count": 60, "mean_soc": 0.50,
      "min_soc": 0.50, "max_soc": 0.50, "mean_solar_kw": 0.0,
      "mean_demand_kw": 0.3, "has_ev_chargers": true
    }
  ],
  "grid_frequency_hz": 50.0,
  "grid_voltage_v": 230.0,
  "market_price_per_mwh": 50.0,
  "forecast_24h_price": [...],
  "forecast_24h_solar": [...],
  "short_term_price_forecast": [50.1, 50.3, 49.8, 50.2],
  "short_term_solar_forecast": [0.01, 0.12, 0.35, 0.72]
}
```

Key signals:

- `market_price_per_mwh` — sell when high, buy when low.
- `grid_frequency_hz` — if < 49.8 Hz, **discharge immediately** (grid emergency).
- `telemetry[*].soc` — track per-home battery state.
- `zone_aggregates` — fast zone-level summary (40 homes vs 60 EV homes).
- `forecast_24h_price` — full 48-step true price curve (useful for look-ahead).
- `short_term_*_forecast` — noisy 4-step / 60-minute look-ahead.

---

## Tasks

### Easy — Arbitrage (`easy-arbitrage`)

**Scenario:** Clear sky, low demand, flat $50/MWh price.

**Strategy:** Sell solar surplus whenever the battery is above the reserve level.

**Profit target:** $500 | **Difficulty:** ⭐☆☆

---

### Medium — Forecast Error (`medium-forecast-error`)

**Scenario:** Heatwave. AC demand spikes **4×** between 10:00–14:00 (steps 16–31). Sinusoidal $35–$65/MWh pricing.

**Strategy:** Reserve capacity for the demand spike while profiting from time-of-use arbitrage.

**Profit target:** $200 | **Difficulty:** ⭐⭐☆

---

### Hard — Frequency Response (`hard-frequency-response`)

**Scenario:** Grid stress at **12:30 (step 26)**. Price spikes to **10× normal** (~$500/MWh) and frequency drops to **49.5 Hz** for exactly one step. Reduced solar (0.7×) and high base demand (1.2×) drain batteries faster.

**Challenge:** Agent must have batteries charged and ready for the spike — greedy morning sell depletes reserves.

**Profit target:** $1000 | **Difficulty:** ⭐⭐⭐

---

## Reward Function

```
reward = step_profit
       − 2.0  (if any battery violated reserve floor this step)
       − 2.0  (if freq < 49.8 Hz and agent was not discharging)
```

Dense, gradient signal at every step — not a sparse binary outcome. Suitable for PPO/SAC RL training.

---

## Grader (0.0 → 1.0)

```
profit_ratio      = min(1.0, cumulative_profit / goal)
violation_penalty = min(0.40, (safety_violations / 48) × 0.40)
emergency_penalty = min(0.30, grid_emergencies_ignored × 0.10)

score = max(0.0, profit_ratio − violation_penalty − emergency_penalty)
```

Fully deterministic and programmatic — no LLM-as-judge. Identical agent + task = identical score.

---

## Baseline Scores

Pre-computed results (rule-based smart agent — no API key required):

| Task | Score | Profit (USD) | Violations |
|---|---|---|---|
| easy-arbitrage | 0.71 | $178 | 0 |
| medium-forecast-error | 0.43 | $47 | 0 |
| hard-frequency-response | 0.52 | $261 | 0 |

LLM baseline (zero-shot, `Qwen2.5-72B-Instruct`):

| Task | Score | Profit (USD) | Violations |
|---|---|---|---|
| easy-arbitrage | 0.68 | $170 | 1 |
| medium-forecast-error | 0.38 | $38 | 2 |
| hard-frequency-response | 0.11 | $55 | 4 |

> Run `python baseline_inference.py --agent rule` to regenerate rule-based scores.  
> Run `python baseline_inference.py --agent llm` (requires API key) for LLM scores.  
> Call `GET /baseline?refresh=true` to recompute via the API.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness probe |
| GET | `/tasks` | List tasks + action schema |
| POST | `/reset?task_id=...` | Start new episode |
| POST | `/step` | Take one action |
| GET | `/state` | Ground-truth state (debugging) |
| GET | `/grader` | Episode score 0.0–1.0 |
| GET | `/baseline` | Cached baseline scores |
| GET | `/baseline?refresh=true` | Recompute baseline scores live |

---

## Inference Script

The `inference.py` script (root directory) is the submission entry point. It:

1. Reads `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` from environment variables.
2. Runs all 3 tasks sequentially against the VPP server.
3. Emits strictly formatted stdout logs:

```
[START] task=easy-arbitrage env=vpp model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"global_charge_rate": -0.7, "min_reserve_pct": 0.2} reward=4.38 done=false error=null
...
[END] success=true steps=48 score=0.68 rewards=4.38,4.21,...
```

Runtime is under 20 minutes on 2 vCPU / 8 GB RAM.

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

Tests cover:

- `tests/test_curves.py` — solar/demand/price curve shapes and values.
- `tests/test_vpp_environment.py` — reset/step physics, SoC bounds, zone aggregates.
- `tests/test_grader.py` — scoring formula, penalty caps, determinism.

---

## RL Training

```bash
# Start server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Train PPO with 3-phase curriculum (easy → medium → hard)
python train_rl.py
```

Monitor: `tensorboard --logdir ./vpp_tensorboard/`

---

## Project Structure

```
├── inference.py             # Submission entry point (mandatory name)
├── baseline_inference.py    # Baseline script (LLM or rule-based)
├── server/
│   ├── __init__.py
│   ├── app.py               # FastAPI application
│   ├── vpp_environment.py   # Core simulation engine
│   └── task_curves.py       # Deterministic solar/demand/price curves
├── models.py                # Pydantic schemas (Action, Observation, State, Zone)
├── client.py                # OpenEnv EnvClient wrapper
├── gymwrapper.py            # Gymnasium wrapper for RL
├── train_rl.py              # PPO curriculum training
├── demo.py                  # Rule-based agent demo
├── validate.py              # Pre-submission smoke test
├── tests/
│   ├── test_curves.py
│   ├── test_vpp_environment.py
│   └── test_grader.py
├── baseline_scores.json     # Pre-computed scores
├── openenv.yaml             # OpenEnv manifest
├── Dockerfile
└── requirements.txt
```

---

## Requirements

```
openenv-core[core]>=0.2.1
fastapi>=0.115.0
uvicorn>=0.24.0
openai>=1.0.0
numpy>=1.24.0
requests>=2.31.0
pydantic>=2.0.0
gymnasium>=0.29.0
stable-baselines3>=2.0.0
pytest>=8.0.0
```