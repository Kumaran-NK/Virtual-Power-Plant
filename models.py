# models.py
"""
Pydantic data models for the VPP Environment — Hackathon Extended Edition.

New in this version:
  - BatteryAsset: state_of_health (SoH) for degradation tracking
  - VppAction: defer_ev_charging, accept_dr_bid, p2p_export_rate, reasoning
  - VppObservation: carbon_credits_balance, forecast uncertainty bands,
                    grid_connected flag, dr_bid, zone P2P stats
  - VppState: full ground-truth including SoH, carbon, P2P, islanding state
  - ParetoScore: multi-objective grader output
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

    asset_id:        str   = Field(..., description="Unique home identifier, e.g. 'home-042'.")
    capacity_kwh:    float = Field(..., description="Maximum energy storage in kWh (e.g. 13.5).")
    max_power_kw:    float = Field(..., description="Maximum charge/discharge rate in kW (e.g. 5.0).")
    efficiency_rt:   float = Field(
        0.90,
        description="Round-trip efficiency. 0.90 → 10 % energy lost as heat on the charging leg.",
    )
    state_of_health: float = Field(
        1.0,
        ge=0.0, le=1.0,
        description=(
            "Battery health (0.8–1.0). Degrades ~0.001 per full charge/discharge cycle. "
            "Effective capacity = capacity_kwh × state_of_health."
        ),
    )


# ---------------------------------------------------------------------------
# Dynamic telemetry
# ---------------------------------------------------------------------------

class BatteryTelemetry(BaseModel):
    """Real-time snapshot of one home battery asset."""

    asset_id:              str   = Field(..., description="Matches a BatteryAsset.asset_id.")
    soc:                   float = Field(..., ge=0.0, le=1.0, description="State of Charge: 0.0 = empty, 1.0 = full.")
    state_of_health:       float = Field(1.0, ge=0.0, le=1.0, description="Current battery health (0.8–1.0).")
    current_house_load_kw: float = Field(..., description="Instantaneous household power consumption in kW.")
    current_solar_gen_kw:  float = Field(..., description="Instantaneous solar panel output in kW.")


# ---------------------------------------------------------------------------
# Zone-level aggregates
# ---------------------------------------------------------------------------

class ZoneTelemetry(BaseModel):
    """
    Aggregate statistics for a logical zone of homes.

    Zone A (homes 000–039): Standard homes without EVs.
    Zone B (homes 040–099): Homes with electric vehicles.
    """

    zone_id:          str   = Field(..., description="Zone identifier, e.g. 'zone-a' or 'zone-b'.")
    home_count:       int   = Field(..., description="Number of homes in this zone.")
    mean_soc:         float = Field(..., ge=0.0, le=1.0, description="Mean State of Charge across zone.")
    min_soc:          float = Field(..., ge=0.0, le=1.0, description="Minimum SoC in the zone (worst-case home).")
    max_soc:          float = Field(..., ge=0.0, le=1.0, description="Maximum SoC in the zone (best-case home).")
    mean_soh:         float = Field(1.0, ge=0.0, le=1.0, description="Mean State of Health across zone.")
    mean_solar_kw:    float = Field(..., description="Mean solar generation per home in kW.")
    mean_demand_kw:   float = Field(..., description="Mean household demand per home in kW (includes EV if applicable).")
    has_ev_chargers:  bool  = Field(..., description="True if zone homes have EV chargers (higher evening load).")
    p2p_available_kw: float = Field(
        0.0,
        description=(
            "kW/home available for P2P export to the other zone. "
            "Positive when the zone has a solar surplus above its own demand."
        ),
    )


# ---------------------------------------------------------------------------
# Demand Response Bid
# ---------------------------------------------------------------------------

class DRBid(BaseModel):
    """
    A Demand Response bid posted by the grid operator.

    Appears in the observation every 6 steps. The agent may accept it via
    VppAction.accept_dr_bid = True. If accepted, the agent must export at
    ≥ committed_power_kw for the next committed_steps steps; failure incurs
    a penalty equal to 2× the missed revenue.
    """

    active:              bool  = Field(False, description="True if a DR bid is currently open.")
    premium_multiplier:  float = Field(1.0,   description="Price multiplier for the committed period (1.5–3.0×).")
    committed_power_kw:  float = Field(0.0,   description="Minimum kW the fleet must export per home.")
    committed_steps:     int   = Field(0,     description="Number of steps the commitment lasts.")
    steps_remaining:     int   = Field(0,     description="Steps left in the current DR commitment (0 = none active).")


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
        Minimum SoC the agent promises to maintain (safety floor).

    defer_ev_charging : float  in [0.0, 1.0]
        Fraction of Zone B EV charging to defer. 0.0 = charge now,
        1.0 = fully defer until later steps (must be repaid by step 40).

    accept_dr_bid : bool
        If True and a DR bid is active, the agent commits to deliver
        the bid's required power for the next committed_steps steps.

    p2p_export_rate : float  in [0.0, 1.0]
        Fraction of Zone B solar surplus to route to Zone A via P2P market
        (bypasses grid, earns midpoint price instead of spot price).

    reasoning : Optional[str]
        Free-text explanation of the agent's decision. Submitted to
        POST /trace for LLM-scored reasoning quality evaluation.
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
    defer_ev_charging: float = Field(
        0.0,
        ge=0.0, le=1.0,
        description="Fraction of Zone B EV load to defer (0 = charge now, 1 = fully defer).",
    )
    accept_dr_bid: bool = Field(
        False,
        description="Accept the current demand-response grid bid.",
    )
    p2p_export_rate: float = Field(
        0.0,
        ge=0.0, le=1.0,
        description="Fraction of Zone B solar surplus to route to Zone A via P2P.",
    )
    reasoning: Optional[str] = Field(
        None,
        max_length=500,
        description="Optional free-text reasoning trace submitted for LLM scoring.",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class VppObservation(Observation):
    """
    Everything the agent is allowed to observe at each step — Extended Edition.

    New fields vs base version:
      carbon_credits_balance     — running carbon credit balance
      forecast_price_uncertainty — σ per step for price forecast
      forecast_solar_uncertainty — σ per step for solar forecast
      grid_connected             — False during islanding emergency (task 5)
      dr_bid                     — active demand-response bid details
      ev_defer_deadline_step     — step by which deferred EV charging must be repaid
      p2p_last_revenue_usd       — P2P revenue earned in the previous step
    """

    timestamp:   datetime = Field(..., description="Wall-clock time of this step (UTC).")
    step_id:     int      = Field(..., description="Step index 0–47 (15-min intervals).")
    telemetry:   List[BatteryTelemetry] = Field(
        ..., description="Per-home real-time snapshot (100 entries)."
    )

    # ── Zone-level aggregates ─────────────────────────────────────────────────
    zone_aggregates: List[ZoneTelemetry] = Field(
        default_factory=list,
        description="Aggregated stats per zone (Zone A = no EVs, Zone B = EVs).",
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
    grid_connected: bool = Field(
        True,
        description="False during islanding event — agent must sustain homes from batteries alone.",
    )

    # ── Market data ───────────────────────────────────────────────────────────
    market_price_per_mwh: float = Field(
        ..., description="Current wholesale electricity price in USD/MWh."
    )

    # ── Carbon credits ────────────────────────────────────────────────────────
    carbon_credits_balance: float = Field(
        0.0,
        description=(
            "Running carbon credit balance. "
            "Earned at 0.05 credits/kWh from solar generation. "
            "Spent at 0.08 credits/kWh for grid purchases during high-emission steps (0–16)."
        ),
    )

    # ── Full-horizon forecasts (ground-truth) ────────────────────────────────
    forecast_24h_price: List[float] = Field(
        ..., description="True price curve for all 48 steps (USD/MWh)."
    )
    forecast_24h_solar: List[float] = Field(
        ..., description="True solar generation curve for all 48 steps (kW/home)."
    )

    # ── Short-term noisy forecasts + uncertainty bands ────────────────────────
    short_term_price_forecast: List[float] = Field(
        default_factory=list,
        description="Noisy price forecast for the next 4 steps (Gaussian σ=2.5 USD/MWh).",
    )
    short_term_solar_forecast: List[float] = Field(
        default_factory=list,
        description="Noisy solar forecast for the next 4 steps (Gaussian σ=0.25 kW).",
    )
    forecast_price_uncertainty: List[float] = Field(
        default_factory=list,
        description=(
            "1-σ uncertainty band for each step of short_term_price_forecast. "
            "Grows with horizon: [2.5, 3.5, 4.5, 5.5] USD/MWh."
        ),
    )
    forecast_solar_uncertainty: List[float] = Field(
        default_factory=list,
        description=(
            "1-σ uncertainty band for each step of short_term_solar_forecast. "
            "Grows with horizon: [0.25, 0.35, 0.50, 0.70] kW."
        ),
    )

    # ── Demand Response ───────────────────────────────────────────────────────
    dr_bid: DRBid = Field(
        default_factory=DRBid,
        description="Current demand-response bid posted by the grid operator.",
    )
    ev_defer_deadline_step: int = Field(
        40,
        description="Step by which all deferred EV charging must be repaid (default 40 = 16:00).",
    )

    # ── P2P market ────────────────────────────────────────────────────────────
    p2p_last_revenue_usd: float = Field(
        0.0,
        description="USD earned by Zone B in the previous step via P2P sales to Zone A.",
    )


# ---------------------------------------------------------------------------
# State (ground truth — hidden from agent in competitive evaluation)
# ---------------------------------------------------------------------------

class VppState(State):
    """
    Full ground-truth state of the VPP episode — Extended Edition.
    """

    # ── Temporal ──────────────────────────────────────────────────────────────
    current_step: int = Field(..., description="Current 15-min interval index (0–47).")
    task_tier:    str = Field(..., description="Active scenario ID.")
    done:         bool = Field(False, description="True when the episode has ended.")

    # ── Financial accumulators ────────────────────────────────────────────────
    cumulative_revenue_usd:  float = Field(0.0, description="Total revenue from grid sales (USD).")
    cumulative_cost_usd:     float = Field(0.0, description="Total cost of grid purchases (USD).")
    cumulative_profit_usd:   float = Field(0.0, description="Revenue − Cost (USD).")
    cumulative_p2p_usd:      float = Field(0.0, description="Revenue earned via P2P trades (USD).")
    cumulative_dr_bonus_usd: float = Field(0.0, description="Bonus revenue earned from fulfilled DR bids (USD).")
    cumulative_dr_penalty_usd: float = Field(0.0, description="Penalties for failing DR commitments (USD).")

    # ── Carbon credits ────────────────────────────────────────────────────────
    carbon_credits_balance:  float = Field(0.0, description="Current carbon credit balance.")
    carbon_credits_earned:   float = Field(0.0, description="Total credits earned from solar generation.")
    carbon_credits_spent:    float = Field(0.0, description="Total credits spent on grid purchases.")

    # ── Safety & performance ──────────────────────────────────────────────────
    blackout_events_count:    int   = Field(0, description="Steps where a battery hit 0 % while demand was non-zero.")
    safety_violations_count:  int   = Field(0, description="Cumulative count of per-step reserve violations.")
    grid_emergencies_ignored: int   = Field(0, description="Steps where freq < 49.8 Hz but agent was not discharging.")
    islanding_blackouts:      int   = Field(0, description="Homes that blacked out during grid islanding.")
    dr_bids_accepted:         int   = Field(0, description="Number of DR bids accepted.")
    dr_bids_fulfilled:        int   = Field(0, description="Number of DR commitments successfully fulfilled.")
    dr_bids_failed:           int   = Field(0, description="Number of DR commitments that failed.")
    ev_defer_debt_kwh:        float = Field(0.0, description="kWh of EV charging still owed (deferred but not repaid).")

    # ── Battery health ────────────────────────────────────────────────────────
    mean_state_of_health:     float = Field(1.0, ge=0.0, le=1.0, description="Fleet-average battery health.")
    min_state_of_health:      float = Field(1.0, ge=0.0, le=1.0, description="Worst-home battery health.")
    total_cycle_count:        float = Field(0.0, description="Total half-cycles across all 100 batteries.")

    # ── Physical ground truth ─────────────────────────────────────────────────
    actual_weather_mode:  str             = Field(..., description="'clear_sky' | 'heatwave' | 'partly_cloudy' | 'cloudy_disruption'.")
    battery_true_soc:     Dict[str, float] = Field(..., description="Precise SoC for every home (0.0–1.0).")
    battery_true_soh:     Dict[str, float] = Field(default_factory=dict, description="Precise SoH for every home (0.8–1.0).")

    # ── Islanding ─────────────────────────────────────────────────────────────
    grid_connected:       bool = Field(True, description="False during islanding event.")
    islanding_start_step: Optional[int] = Field(None, description="Step when islanding began (None if no islanding).")
    islanding_end_step:   Optional[int] = Field(None, description="Step when grid reconnects (None if ongoing/none).")

    # ── Demand Response runtime ───────────────────────────────────────────────
    dr_active:            bool  = Field(False, description="True if a DR commitment is currently in force.")
    dr_committed_until:   int   = Field(0,     description="Step at which the current DR commitment ends.")
    dr_committed_power_kw: float = Field(0.0,  description="kW/home committed in the current DR bid.")
    dr_premium_multiplier: float = Field(1.0,  description="Price multiplier for the current DR commitment.")


# ---------------------------------------------------------------------------
# Pareto / Multi-objective score
# ---------------------------------------------------------------------------

class ParetoScore(BaseModel):
    """
    Multi-objective grader output.

    Returns three component scores and a weighted aggregate.
    Replaces the single scalar in /grader.
    """

    # Component scores (each 0.0–1.0)
    profit_score:    float = Field(..., ge=0.0, le=1.0, description="Financial performance vs profit target.")
    safety_score:    float = Field(..., ge=0.0, le=1.0, description="1.0 = zero violations; degrades with violations.")
    carbon_score:    float = Field(..., ge=0.0, le=1.0, description="Carbon credit balance normalised to target.")
    degradation_score: float = Field(..., ge=0.0, le=1.0, description="1.0 = zero degradation; degrades with SoH loss.")
    dr_score:        float = Field(0.0, ge=0.0, le=1.0, description="Demand-response participation quality.")

    # Weighted aggregate (replaces old scalar `score`)
    aggregate_score: float = Field(..., ge=0.0, le=1.0,
        description="Weighted sum: 0.50×profit + 0.20×safety + 0.15×carbon + 0.10×degradation + 0.05×dr")

    # Detail
    cumulative_profit_usd:    float = Field(0.0)
    cumulative_p2p_usd:       float = Field(0.0)
    cumulative_dr_bonus_usd:  float = Field(0.0)
    safety_violations:        int   = Field(0)
    grid_emergencies_ignored: int   = Field(0)
    islanding_blackouts:      int   = Field(0)
    carbon_credits_balance:   float = Field(0.0)
    mean_state_of_health:     float = Field(1.0)
    dr_bids_fulfilled:        int   = Field(0)
    dr_bids_failed:           int   = Field(0)
    steps_completed:          int   = Field(0)
    done:                     bool  = Field(False)


class VppReward(BaseModel):
    """Typed reward payload for documentation and schema introspection."""

    reward: float = Field(..., description="Per-step scalar reward returned by /step.")