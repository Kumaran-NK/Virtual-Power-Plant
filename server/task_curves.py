# server/task_curves.py
"""
Deterministic 48-step energy curves for each task tier — Extended Edition.

New tasks:
  expert-demand-response   — DR auction every 6 steps, sinusoidal price
    islanding-emergency      — Grid disconnects steps 20–29, must run on batteries

Episode design
──────────────
  One step  = 15 minutes
  48 steps  = 12 hours  (06:00 → 18:00)
  Step 0    = 06:00     Step 47 = 17:45

Time index reference
  Step  0 → 06:00      Step 16 → 10:00
  Step  8 → 08:00      Step 20 → 11:00  ← islanding starts
  Step 12 → 09:00      Step 24 → 12:00
  Step 14 → 09:30      Step 26 → 12:30  ← grid stress spike (hard task)
  Step 16 → 10:00      Step 29 → 13:15  ← islanding ends
                        Step 32 → 14:00  ← heatwave ends (medium task)
                        Step 40 → 16:00  ← EV defer deadline
                        Step 47 → 17:45

All functions take a full task_id string and extract the tier internally.
"""

import numpy as np

EPISODE_STEPS    = 48          # 12-hour episode
GRID_STRESS_STEP = 26          # 12:30 — single-step 10× price spike (hard task)
HEATWAVE_START   = 16          # 10:00
HEATWAVE_END     = 32          # 14:00  (exclusive)
ISLANDING_START  = 20          # 11:00 — grid disconnects (islanding task)
ISLANDING_END    = 30          # 13:30 — grid reconnects (exclusive)
EV_DEFER_DEADLINE = 40         # 16:00 — all deferred EV charging must be repaid
DR_BID_INTERVAL  = 6           # DR bid posted every 6 steps

# High-emission hours (before solar peaks): grid purchases in this window cost carbon credits
HIGH_EMISSION_STEPS = range(0, 17)   # steps 0–16 inclusive


def _tier(task_id: str) -> str:
    """'easy-arbitrage' → 'easy', 'expert-demand-response' → 'expert', etc."""
    return task_id.split("-")[0].lower()


# ─────────────────────────────────────────────────────────────────────────────
# Solar
# ─────────────────────────────────────────────────────────────────────────────

def solar_curve(task_id: str) -> np.ndarray:
    """
    48-step solar generation curve (kW per home).

    Bell curve centred on solar noon (step 24 = 12:00).
    Easy   → abundant sun  (1.5×, peak ~6.0 kW)
    Medium → normal sun with sudden storm drop (steps 18–23)
    Hard   → reduced sun   (0.7×, peak ~2.8 kW)
    Expert → normal sun    (1.0×) with adversarial cloud event at steps 24–26
    Islanding → partly cloudy (0.8×) — forces careful reserve management
    """
    steps = np.arange(EPISODE_STEPS)
    base = np.maximum(0.0, 4.0 * np.sin(np.pi * steps / EPISODE_STEPS))

    tier = _tier(task_id)
    multipliers = {
        "easy":     1.5,
        "medium":   1.0,
        "hard":     0.7,
        "expert":   1.0,
        "islanding": 0.8,
    }
    base *= multipliers.get(tier, 1.0)

    # Medium task: forecast-error scenario with sudden storm drop around midday.
    if tier == "medium":
        base[18:24] *= 0.35

    # Adversarial cloud event for expert task: sudden 80% drop at steps 24–26
    if tier == "expert":
        base[24:27] *= 0.2   # forecast says clear sky, but clouds appear

    return base


# ─────────────────────────────────────────────────────────────────────────────
# Demand
# ─────────────────────────────────────────────────────────────────────────────

def demand_curve(task_id: str) -> np.ndarray:
    """
    48-step household demand curve (kW per home).

    Easy   → low demand    (0.5×)
    Medium → heatwave AC spike steps 16–31 (10:00–13:45), 4× demand
    Hard   → high demand   (1.2×)
    Expert → sinusoidal with 3× spike at steps 20–25 (overlaps islanding window)
    Islanding → moderate demand (1.0×) — stress-tests battery reserve
    """
    steps = np.arange(EPISODE_STEPS)

    # Gaussian morning peak at step 12 (09:00) + flat afternoon
    morning_peak = 0.5 * np.exp(-0.5 * ((steps - 12) / 6) ** 2)
    base = 0.25 + morning_peak
    base = np.clip(base, 0.15, 1.2)

    tier = _tier(task_id)
    if tier == "easy":
        base *= 0.5
    elif tier == "medium":
        heatwave = np.ones(EPISODE_STEPS)
        heatwave[HEATWAVE_START:HEATWAVE_END] = 4.0
        base *= heatwave
    elif tier == "hard":
        base *= 1.2
    elif tier == "expert":
        # Moderate base + spike during the adversarial cloud window
        base *= 1.1
        demand_spike = np.ones(EPISODE_STEPS)
        demand_spike[20:26] = 3.0
        base *= demand_spike
    elif tier == "islanding":
        base *= 1.0   # neutral — challenge is sustaining power without grid

    return base


# ─────────────────────────────────────────────────────────────────────────────
# Price
# ─────────────────────────────────────────────────────────────────────────────

def price_curve(task_id: str) -> np.ndarray:
    """
    48-step wholesale electricity price (USD/MWh).

    Easy      → simple day/night TOU profile (cheap daytime, expensive evening)
    Medium    → sinusoidal $35–$65/MWh
    Hard      → sinusoidal + 10× spike at step 26 (12:30)
    Expert    → sinusoidal + DR premium windows
    Islanding → sinusoidal (grid price irrelevant when disconnected, but
                reconnection premium at step 30 rewards quick arbitrage)
    """
    steps = np.arange(EPISODE_STEPS)
    tier  = _tier(task_id)

    if tier == "easy":
        base = np.full(EPISODE_STEPS, 50.0)
        base[:16] = 36.0       # cheap daytime charging window
        base[16:32] = 50.0     # shoulder period
        base[32:44] = 72.0     # expensive evening peak
        base[44:] = 58.0       # late-evening taper
    else:
        # Morning cheap, midday peak, afternoon moderate
        base = 50.0 + 15.0 * np.sin(2 * np.pi * (steps - 8) / EPISODE_STEPS)
        base = np.clip(base, 30.0, 70.0)

    if tier == "hard":
        base[GRID_STRESS_STEP] *= 10.0   # 10× spike at 12:30

    if tier == "islanding":
        # Reconnection bonus: spike at step 30 when grid comes back online
        base[ISLANDING_END] *= 8.0   # 8× spike — grid pays premium for fast response

    return base


def forecast_solar_curve(task_id: str) -> np.ndarray:
    """
    Agent-visible solar forecast curve.

    For medium task, the forecast remains optimistic and does not include the
    storm drop, creating a real forecast-vs-reality mismatch.
    """
    tier = _tier(task_id)
    if tier not in {"medium", "expert"}:
        return solar_curve(task_id)

    steps = np.arange(EPISODE_STEPS)
    base = np.maximum(0.0, 4.0 * np.sin(np.pi * steps / EPISODE_STEPS))
    return base


def forecast_price_curve(task_id: str) -> np.ndarray:
    """Agent-visible price forecast curve."""
    return price_curve(task_id)


# ─────────────────────────────────────────────────────────────────────────────
# DR Bid schedule  (expert task only)
# ─────────────────────────────────────────────────────────────────────────────

def dr_bid_schedule(task_id: str) -> dict:
    """
    Return a dict mapping step → DRBid parameters for the expert task.

    DR bids are posted at steps 0, 6, 12, 18, 24, 30, 36, 42.
    Each bid specifies a premium multiplier, required power, and duration.
    """
    if _tier(task_id) != "expert":
        return {}

    # (premium_multiplier, committed_power_kw_per_home, committed_steps)
    bids = {
        0:  (1.5, 1.5, 3),   # 06:00 — mild morning bid
        6:  (2.0, 2.0, 3),   # 07:30 — rising demand
        12: (2.5, 2.5, 3),   # 09:00 — high demand window opens
        18: (3.0, 3.0, 3),   # 10:30 — peak demand bid (hardest to fulfill with clouds)
        24: (2.0, 2.0, 3),   # 12:00 — cloud event, risky to accept
        30: (1.8, 1.5, 3),   # 13:30 — afternoon moderation
        36: (1.5, 1.0, 3),   # 15:00 — winding down
        42: (1.2, 0.5, 3),   # 16:30 — EV charging pressure, easy bid
    }
    return bids


# ─────────────────────────────────────────────────────────────────────────────
# Carbon emission intensity (gCO₂eq / kWh)  — used for credit calculation
# ─────────────────────────────────────────────────────────────────────────────

def emission_intensity_curve(task_id: str) -> np.ndarray:
    """
    Grid carbon intensity per step (gCO₂eq/kWh).

    Morning (steps 0–16): high-emission gas peakers dominate → 400 g/kWh
    Midday  (steps 17–31): solar penetration drops to → 150 g/kWh
    Afternoon (steps 32+): moderate → 250 g/kWh
    """
    intensity = np.full(EPISODE_STEPS, 250.0)
    intensity[:17]  = 400.0   # high emission in the morning
    intensity[17:32] = 150.0  # cleaner midday grid
    return intensity


# ─────────────────────────────────────────────────────────────────────────────
# Task metadata
# ─────────────────────────────────────────────────────────────────────────────

TASK_METADATA = {
    "easy-arbitrage": {
        "difficulty": "easy",
        "profit_target_usd": 500.0,
        "carbon_target_credits": 5.0,
        "description": (
            "High solar production, low household demand, and simple TOU pricing "
            "(cheap daytime, expensive evening). Strategy: charge by day, "
            "discharge in evening peak. Profit target: $500."
        ),
        "weather": "clear_sky",
        "has_islanding": False,
        "has_dr_auction": False,
    },
    "medium-forecast-error": {
        "difficulty": "medium",
        "profit_target_usd": 200.0,
        "carbon_target_credits": 3.0,
        "description": (
            "Heatwave event: AC demand spikes 4× from 10:00–14:00, and "
            "a sudden storm suppresses real solar around midday while forecasts "
            "remain optimistic. "
            "Sinusoidal pricing rewards time-of-use arbitrage. "
            "Agent must manage forecast uncertainty. Profit target: $200."
        ),
        "weather": "heatwave",
        "has_islanding": False,
        "has_dr_auction": False,
    },
    "hard-frequency-response": {
        "difficulty": "hard",
        "profit_target_usd": 1000.0,
        "carbon_target_credits": 8.0,
        "description": (
            "Grid stress: a single-step 10× price spike at 12:30 (step 26). "
            "Grid frequency drops to 49.5 Hz — agent must discharge immediately. "
            "If batteries are depleted before the spike, revenue is lost. "
            "Requires look-ahead planning and reserve management. Profit target: $1000."
        ),
        "weather": "partly_cloudy",
        "has_islanding": False,
        "has_dr_auction": False,
    },
    "expert-demand-response": {
        "difficulty": "expert",
        "profit_target_usd": 800.0,
        "carbon_target_credits": 6.0,
        "description": (
            "DR auction every 6 steps with premium multipliers 1.5–3.0×. "
            "Accepted windows are graded by delivered-vs-committed energy, "
            "and consecutive failed windows escalate penalties (2×, 3×, then 4×). "
            "Adversarial cloud event at 12:00 disrupts solar for 3 steps — forecast is wrong. "
            "Demand spike at 10:30 coincides with lowest solar. "
            "Agent must decide which bids to accept, balancing commitment vs reserve risk. "
            "Profit target: $800."
        ),
        "weather": "cloudy_disruption",
        "has_islanding": False,
        "has_dr_auction": True,
    },
    "islanding-emergency": {
        "difficulty": "hard",
        "profit_target_usd": 400.0,
        "carbon_target_credits": 4.0,
        "description": (
            "Grid disconnects at 11:00 (step 20) for 10 steps — neighbourhood runs on batteries alone. "
            "Critical household load is prioritized; flexible load may be shed first when supply is scarce. "
            "Only unserved critical load counts as blackout penalty. "
            "Grid reconnects at 13:30 (step 30) with an 8× price spike and one-step soft-sync discharge cap. "
            "Any blackout during islanding incurs massive penalty. Profit target: $400."
        ),
        "weather": "partly_cloudy",
        "has_islanding": True,
        "has_dr_auction": False,
    },
}

ALL_TASK_IDS = list(TASK_METADATA.keys())