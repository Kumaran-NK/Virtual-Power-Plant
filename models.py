# models.py
"""
Pydantic data models for the VPP Environment.

These form the typed "contract" between the environment server and any agent.
All models are serialisable to JSON and validated on every request.
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation, State


# ---------------------------------------------------------------------------
# Static registry
# ---------------------------------------------------------------------------

class BatteryAsset(BaseModel):
    """Physical specification of one home battery unit (read-only)."""

    asset_id:      str   = Field(..., description="Unique home identifier, e.g. 'home-042'.")
    capacity_kwh:  float = Field(..., description="Maximum energy storage in kWh (e.g. 13.5).")
    max_power_kw:  float = Field(..., description="Maximum charge/discharge rate in kW (e.g. 5.0).")
    efficiency_rt: float = Field(
        0.90,
        description="Round-trip efficiency. 0.90 → 10 % energy lost as heat on the charging leg.",
    )


# ---------------------------------------------------------------------------
# Dynamic telemetry
# ---------------------------------------------------------------------------

class BatteryTelemetry(BaseModel):
    """Real-time snapshot of one home battery asset."""

    asset_id:              str   = Field(..., description="Matches a BatteryAsset.asset_id.")
    soc:                   float = Field(..., ge=0.0, le=1.0, description="State of Charge: 0.0 = empty, 1.0 = full.")
    current_house_load_kw: float = Field(..., description="Instantaneous household power consumption in kW.")
    current_solar_gen_kw:  float = Field(..., description="Instantaneous solar panel output in kW.")


# ---------------------------------------------------------------------------
# Zone-level aggregates  (new — for improved observability)
# ---------------------------------------------------------------------------

class ZoneTelemetry(BaseModel):
    """
    Aggregate statistics for a logical zone of homes.

    Zone A (homes 000–039): Standard homes without EVs.
                            Predictable demand profile.
    Zone B (homes 040–099): Homes with electric vehicles.
                            Higher evening demand due to EV charging.
    """

    zone_id:          str   = Field(..., description="Zone identifier, e.g. 'zone-a' or 'zone-b'.")
    home_count:       int   = Field(..., description="Number of homes in this zone.")
    mean_soc:         float = Field(..., ge=0.0, le=1.0, description="Mean State of Charge across zone.")
    min_soc:          float = Field(..., ge=0.0, le=1.0, description="Minimum SoC in the zone (worst-case home).")
    max_soc:          float = Field(..., ge=0.0, le=1.0, description="Maximum SoC in the zone (best-case home).")
    mean_solar_kw:    float = Field(..., description="Mean solar generation per home in kW.")
    mean_demand_kw:   float = Field(..., description="Mean household demand per home in kW.")
    has_ev_chargers:  bool  = Field(..., description="True if zone homes have EV chargers (higher evening load).")


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class VppAction(Action):
    """
    Dispatch command sent by the agent to all 100 home batteries simultaneously.

    global_charge_rate : float  in [-1.0, 1.0]
        +1.0 → charge at full rate (buy from grid, fill battery)
        -1.0 → discharge at full rate (sell to grid, drain battery)
         0.0 → idle (no grid transaction)

    min_reserve_pct : float  in [0.0, 1.0]
        Minimum State-of-Charge the agent promises to maintain.
        Dropping below this triggers a safety violation penalty.
        Recommended: ≥ 0.15 to keep lights on at night.
    """

    global_charge_rate: float = Field(
        ...,
        ge=-1.0, le=1.0,
        description="Dispatch command: -1 = full sell, 0 = idle, +1 = full buy.",
    )
    min_reserve_pct: float = Field(
        0.2,
        ge=0.0, le=1.0,
        description="Safety floor for SoC. Agent is penalised if any battery drops below this.",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class VppObservation(Observation):
    """
    Everything the agent is allowed to observe at each step.

    short_term_*_forecast fields include Gaussian noise to simulate
    real-world forecast uncertainty.  The full 24-hour arrays are the
    true underlying curves (useful for look-ahead planning).

    zone_aggregates groups the 100 homes into two logical zones:
      Zone A (40 homes, no EVs) — predictable demand.
      Zone B (60 homes, with EVs) — higher evening demand.
    """

    timestamp:   datetime = Field(..., description="Wall-clock time of this step (UTC).")
    step_id:     int      = Field(..., description="Step index 0–47 (15-min intervals).")
    telemetry:   List[BatteryTelemetry] = Field(
        ..., description="Per-home real-time snapshot (100 entries)."
    )

    # ── Zone-level aggregates (NEW) ──────────────────────────────────────────
    zone_aggregates: List[ZoneTelemetry] = Field(
        default_factory=list,
        description=(
            "Aggregated stats per zone. "
            "Zone A = homes 000–039 (no EVs). "
            "Zone B = homes 040–099 (with EVs, higher evening load)."
        ),
    )

    # ── Grid state ────────────────────────────────────────────────────────────
    grid_frequency_hz: float = Field(
        50.0,
        description="Grid frequency. < 49.8 Hz = emergency requiring immediate discharge.",
    )
    grid_voltage_v: float = Field(
        230.0,
        description="Grid voltage. > 250 V requires charging to absorb excess power.",
    )

    # ── Market data ───────────────────────────────────────────────────────────
    market_price_per_mwh: float = Field(
        ..., description="Current wholesale electricity price in USD/MWh."
    )

    # ── Full-horizon forecasts (ground-truth) ────────────────────────────────
    forecast_24h_price: List[float] = Field(
        ..., description="True price curve for all 48 steps (USD/MWh)."
    )
    forecast_24h_solar: List[float] = Field(
        ..., description="True solar generation curve for all 48 steps (kW/home)."
    )

    # ── Short-term noisy forecasts (next 4 steps = next 60 minutes) ──────────
    short_term_price_forecast: List[float] = Field(
        default_factory=list,
        description="Noisy price forecast for the next 4 steps (Gaussian σ=2.5 USD/MWh).",
    )
    short_term_solar_forecast: List[float] = Field(
        default_factory=list,
        description="Noisy solar forecast for the next 4 steps (Gaussian σ=0.25 kW).",
    )


# ---------------------------------------------------------------------------
# State (ground truth — hidden from agent in competitive evaluation)
# ---------------------------------------------------------------------------

class VppState(State):
    """
    Full ground-truth state of the VPP episode.

    Returned by GET /state.  Agents should not use this during evaluation.
    """

    # ── Temporal ──────────────────────────────────────────────────────────────
    current_step: int = Field(..., description="Current 15-min interval index (0–47).")
    task_tier:    str = Field(
        ...,
        description="Active scenario ID: 'easy-arbitrage' | 'medium-forecast-error' | 'hard-frequency-response'.",
    )
    done: bool = Field(False, description="True when the episode has ended.")

    # ── Financial accumulators ────────────────────────────────────────────────
    cumulative_revenue_usd: float = Field(0.0, description="Total revenue from grid sales (USD).")
    cumulative_cost_usd:    float = Field(0.0, description="Total cost of grid purchases (USD).")
    cumulative_profit_usd:  float = Field(0.0, description="Revenue − Cost (USD).")

    # ── Safety & performance ──────────────────────────────────────────────────
    blackout_events_count:    int = Field(0, description="Steps where a battery hit 0 % while demand was non-zero.")
    safety_violations_count:  int = Field(0, description="Cumulative count of per-step reserve violations.")
    grid_emergencies_ignored: int = Field(0, description="Steps where freq < 49.8 Hz but agent was not discharging.")

    # ── Physical ground truth ─────────────────────────────────────────────────
    actual_weather_mode: str = Field(
        ..., description="'clear_sky' | 'heatwave' | 'partly_cloudy'."
    )
    battery_true_soc: Dict[str, float] = Field(
        ..., description="Precise SoC for every home (0.0–1.0)."
    )