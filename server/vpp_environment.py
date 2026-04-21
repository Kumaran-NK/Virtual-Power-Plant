# server/vpp_environment.py
"""
VPP Environment — Extended Hackathon Edition.

New mechanics vs base version:
  1. Battery degradation (State of Health — SoH)
  2. Carbon credits subsystem
  3. Forecast confidence bands
  4. Pareto multi-objective grader
  5. P2P energy trading between zones
  6. Demand Response auction (expert task)
  7. Grid islanding emergency (islanding task)
  8. Load deferral action (defer_ev_charging)
  9. Adversarial weather injection (expert task)
  10. Reasoning trace storage (evaluated via POST /trace)

Physics
───────
  charge_kw > 0  → buying from grid  (battery fills,  cost)
  charge_kw < 0  → selling to grid   (battery drains, revenue)
  effective_kw   = charge_kw × η      (η = 0.90, charging leg only)
  delta_kwh      = (solar − demand + effective_kw) × 0.25 h
  new_soc        = old_soc + delta_kwh / (capacity_kwh × SoH)
  grid_profit    = −charge_kw × 0.25 h × (price $/MWh ÷ 1000)

SoH degradation
───────────────
  Each step, |delta_kwh| / capacity_kwh contributes to cycle counting.
  degradation_per_cycle = 0.001 per full equivalent cycle (charge + discharge)
  effective_full_cycles = total_delta_kwh_abs / (2 × capacity_kwh)
  new_soh = 1.0 - effective_full_cycles × 0.001
  SoH is floored at 0.80 (real batteries rarely go below 80% usable health).

Carbon credits
──────────────
  Earned: 0.05 credits per kWh of solar generated (all steps)
  Spent:  0.08 credits per kWh purchased from grid during steps 0–16 (high emission)

P2P trading
───────────
  Zone B solar surplus = max(0, solar_kw − demand_kw) × p2p_export_rate
  P2P price = midpoint of spot price (better than grid sell for Zone B,
              cheaper than grid buy for Zone A — mutual benefit)
  Only activates if Zone A has demand to absorb the export.
"""

import random
import threading
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np

from openenv.core.env_server.interfaces import Environment
from models import (
    BatteryAsset, BatteryTelemetry, ZoneTelemetry,
    DRBid, VppAction, VppObservation, VppState, ParetoScore,
)
from server.task_curves import (
    solar_curve, demand_curve, price_curve,
    forecast_solar_curve, forecast_price_curve,
    emission_intensity_curve, dr_bid_schedule,
    EPISODE_STEPS, GRID_STRESS_STEP,
    ISLANDING_START, ISLANDING_END, EV_DEFER_DEADLINE,
    HIGH_EMISSION_STEPS, DR_BID_INTERVAL, TASK_METADATA, ALL_TASK_IDS,
)

# NOTE: contextvars no longer used for session management.
# OpenEnv's WebSocket endpoint (/ws) handles stateful session management via MCP protocol.
# HTTP endpoints (/reset, /step) are stateless - use WebSocket for multi-step episodes.
# Clients should use: from client import VppEnv
#   with VppEnv(base_url="http://localhost:7860") as client:
#       result = client.reset(task_id="easy-arbitrage")
#       result = client.step(action)  # Session persists via WebSocket


_ZONE_A_START, _ZONE_A_END = 0,  40
_ZONE_B_START, _ZONE_B_END = 40, 100

# SoH physics constants
_SOH_FLOOR           = 0.80    # batteries never degrade below 80 %
_DEGRADATION_PER_CYCLE = 0.001  # SoH loss per full equivalent cycle

# Carbon credit rates
_CARBON_EARN_RATE    = 0.05    # credits per kWh from solar generation
_CARBON_SPEND_RATE   = 0.08    # credits per kWh from grid during high-emission hours

# Forecast uncertainty bands (grow with horizon)
_PRICE_SIGMA = [2.5, 3.5, 4.5, 5.5]    # USD/MWh, 4-step horizon
_SOLAR_SIGMA = [0.25, 0.35, 0.50, 0.70] # kW/home, 4-step horizon

# P2P pricing: fraction of spot price as midpoint benefit
_P2P_PRICE_FRACTION  = 0.75   # Zone B earns 75% of spot (better than export tariff)

# DR failure penalty multipliers (progressive for consecutive failed windows)
_DR_FAIL_PENALTY_BASE_MULT = 2.0
_DR_FAIL_PENALTY_MAX_MULT = 4.0

# Islanding blackout penalty per home-step
_ISLANDING_BLACKOUT_PENALTY = 50.0   # USD per home blackout during islanding

# Score epsilon: minimum deviation from boundaries to ensure open interval (0, 1)
_SCORE_EPSILON = 0.0001

# Phase-2 score calibration constants (distribution stability)
_NON_DR_TASK_NEUTRAL_DR_SCORE = 0.90
_SAFETY_SCORE_SOFT_CEILING = 0.999
_DEMAND_SHED_NORMALISATION_KWH = 30.0
_SAFETY_VIOLATION_WEIGHT = 0.40
_SAFETY_EMERGENCY_WEIGHT = 0.30
_SAFETY_DEMAND_SHED_WEIGHT = 0.30
_SAFETY_LATENCY_MAX_PENALTY = 0.15

# Hard reserve policy
_HOME_RESERVE_FLOOR = 0.20
_EMERGENCY_RESERVE_FLOOR = 0.10

# Islanding load service semantics
_CRITICAL_LOAD_FRACTION = 0.40
_FLEXIBLE_LOAD_FRACTION = 1.0 - _CRITICAL_LOAD_FRACTION

# First step after reconnection is soft-synced to limit dispatch surge.
_RECONNECTION_DISCHARGE_CAP_RATIO = 0.50

# Step reward shaping
_STEP_CARBON_REWARD_MULT = 0.25
_STEP_DEGRADATION_PENALTY_MULT = 250.0
_STEP_SAFETY_MARGIN_BONUS = 0.5
_STEP_SAFETY_VIOLATION_PENALTY = 3.0
_STEP_EMERGENCY_MISS_PENALTY = 8.0
_STEP_EMERGENCY_WEAK_RESPONSE_PENALTY = 3.0
_STEP_EMERGENCY_STRONG_RESPONSE_BONUS = 2.0


class VppEnvironment(Environment):
    """Simulated Virtual Power Plant — 100 home batteries, 12-hour episodes, Extended Edition."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    # Process-wide snapshot of the most recent Pareto score emitted by any client/session.
    _last_grader_lock = threading.Lock()
    _last_grader_snapshot: Optional[Dict[str, object]] = None

    @classmethod
    def _set_last_grader_snapshot(cls, snapshot: Dict[str, object]) -> None:
        with cls._last_grader_lock:
            cls._last_grader_snapshot = deepcopy(snapshot)

    @classmethod
    def get_last_grader_snapshot(cls) -> Optional[Dict[str, object]]:
        with cls._last_grader_lock:
            if cls._last_grader_snapshot is None:
                return None
            return deepcopy(cls._last_grader_snapshot)

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(self):
        self.assets: List[BatteryAsset] = [
            BatteryAsset(
                asset_id=f"home-{i:03d}",
                capacity_kwh=13.5,
                max_power_kw=5.0,
                efficiency_rt=0.90,
                state_of_health=1.0,
            )
            for i in range(100)
        ]

        self._task_id:       str              = "easy-arbitrage"
        self._state:         VppState | None  = None
        self._battery_soc:   Dict[str, float] = {}
        self._battery_soh:   Dict[str, float] = {}   # SoH per asset
        self._battery_cycles: Dict[str, float] = {}  # cumulative equivalent cycles
        self._current_step:  int              = 0
        self._total_profit:  float            = 0.0
        self._episode_start: datetime         = datetime.now(timezone.utc)

        self._true_solar:     np.ndarray | None = None
        self._true_demand:    np.ndarray | None = None
        self._true_price:     np.ndarray | None = None
        self._forecast_solar: np.ndarray | None = None
        self._forecast_price: np.ndarray | None = None
        self._emission_curve: np.ndarray | None = None
        self._dr_schedule:    dict              = {}

        # DR runtime state
        self._dr_committed:   bool  = False
        self._dr_until_step:  int   = 0
        self._dr_power_kw:    float = 0.0
        self._dr_premium:     float = 1.0
        self._dr_missed_steps: int  = 0
        self._dr_failure_streak: int = 0
        self._dr_committed_kwh_total: float = 0.0
        self._dr_delivered_kwh_total: float = 0.0

        # EV deferral state
        self._ev_defer_debt_kwh: float = 0.0   # kWh still owed

        # P2P last-step revenue
        self._p2p_last_revenue: float = 0.0

        # Carbon credits
        self._carbon_balance:  float = 0.0
        self._carbon_earned:   float = 0.0
        self._carbon_spent:    float = 0.0
        self._last_step_carbon_earned: float = 0.0
        self._last_step_carbon_spent: float = 0.0
        self._last_step_demand_shed_kwh: float = 0.0
        self._cumulative_demand_shed_kwh: float = 0.0
        self._islanding_blackout_home_ids: set[str] = set()

        # Emergency response tracking
        self._emergency_start_step: Optional[int] = None
        self._response_latency_steps: Optional[int] = None
        self._frequency_history: List[float] = []

        # Reasoning trace storage
        self._reasoning_traces: List[dict] = []

        # Session management is handled by OpenEnv's /ws endpoint (MCP protocol).
        # HTTP /reset and /step endpoints are stateless and create fresh instances.
        # For stateful multi-step episodes, use WebSocket via VppEnv client.

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int | str] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> VppObservation:
        task_id_arg = kwargs.get("task_id")
        task_id = str(task_id_arg) if task_id_arg is not None else "easy-arbitrage"

        if isinstance(seed, str):
            if task_id_arg is None and not seed.lstrip("-").isdigit():
                task_id = seed
                seed = None
            else:
                seed = int(seed)

        self._task_id = task_id

        seed_value = int(seed) if seed is not None else abs(hash(task_id)) % (2 ** 31)
        random.seed(seed_value)
        np.random.seed(seed_value)

        self._true_solar  = solar_curve(task_id)
        self._true_demand = demand_curve(task_id)
        self._true_price  = price_curve(task_id)
        self._forecast_solar = forecast_solar_curve(task_id)
        self._forecast_price = forecast_price_curve(task_id)
        self._emission_curve = emission_intensity_curve(task_id)
        self._dr_schedule = dr_bid_schedule(task_id)

        self._current_step  = 0
        self._total_profit  = 0.0
        self._episode_start = datetime.now(timezone.utc)
        self._p2p_last_revenue = 0.0
        self._carbon_balance = 0.0
        self._carbon_earned  = 0.0
        self._carbon_spent   = 0.0
        self._last_step_carbon_earned = 0.0
        self._last_step_carbon_spent = 0.0
        self._last_step_demand_shed_kwh = 0.0
        self._cumulative_demand_shed_kwh = 0.0
        self._emergency_start_step = None
        self._response_latency_steps = None
        self._frequency_history = []
        self._dr_committed   = False
        self._dr_until_step  = 0
        self._dr_power_kw    = 0.0
        self._dr_premium     = 1.0
        self._dr_missed_steps = 0
        self._dr_failure_streak = 0
        self._dr_committed_kwh_total = 0.0
        self._dr_delivered_kwh_total = 0.0
        self._ev_defer_debt_kwh = 0.0
        self._reasoning_traces  = []
        self._islanding_blackout_home_ids = set()

        self._battery_soc    = {a.asset_id: 0.5     for a in self.assets}
        self._battery_soh    = {a.asset_id: 1.0     for a in self.assets}
        self._battery_cycles = {a.asset_id: 0.0     for a in self.assets}

        meta    = TASK_METADATA.get(task_id, TASK_METADATA["easy-arbitrage"])
        weather = meta["weather"]

        self._state = VppState(
            episode_id=episode_id or f"ep_{task_id}_{seed_value % 10_000:04d}",
            step_count=0,
            current_step=0,
            task_tier=task_id,
            cumulative_profit_usd=0.0,
            cumulative_revenue_usd=0.0,
            cumulative_cost_usd=0.0,
            cumulative_p2p_usd=0.0,
            cumulative_dr_bonus_usd=0.0,
            cumulative_dr_penalty_usd=0.0,
            blackout_events_count=0,
            safety_violations_count=0,
            grid_emergencies_ignored=0,
            islanding_blackouts=0,
            islanding_blackout_home_steps=0,
            islanding_blackout_unique_homes=0,
            cumulative_demand_shed_kwh=0.0,
            response_latency_steps_to_emergency=-1,
            dr_bids_accepted=0,
            dr_bids_fulfilled=0,
            dr_bids_failed=0,
            dr_consecutive_failures=0,
            dr_fulfillment_ratio_sum=0.0,
            ev_defer_debt_kwh=0.0,
            carbon_credits_balance=0.0,
            carbon_credits_earned=0.0,
            carbon_credits_spent=0.0,
            mean_state_of_health=1.0,
            min_state_of_health=1.0,
            total_cycle_count=0.0,
            actual_weather_mode=weather,
            battery_true_soc=self._battery_soc.copy(),
            battery_true_soh=self._battery_soh.copy(),
            grid_connected=True,
            islanding_start_step=ISLANDING_START if meta.get("has_islanding") else None,
            islanding_end_step=ISLANDING_END if meta.get("has_islanding") else None,
            dr_active=False,
            dr_committed_until=0,
            dr_committed_power_kw=0.0,
            dr_premium_multiplier=1.0,
            done=False,
        )

        return self._build_observation()

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, action: VppAction, timeout_s: Optional[float] = None, **kwargs) -> VppObservation:
        if self._state is None or self._state.done:
            raise RuntimeError("Call reset() before step().")

        if (
            self._true_solar is None
            or self._true_demand is None
            or self._true_price is None
            or self._emission_curve is None
        ):
            raise RuntimeError("Missing environment curves. Call reset() before step().")

        s             = self._current_step
        solar_kw      = float(self._true_solar[s])
        demand_kw     = float(self._true_demand[s])
        price_usd     = float(self._true_price[s])
        emission_g    = float(self._emission_curve[s])
        freq_hz       = self._grid_frequency(s)
        grid_connected = self._is_grid_connected(s)
        meta          = TASK_METADATA.get(self._task_id, {})
        step_emergency_active = (freq_hz < 49.8) or (not grid_connected)
        step_reserve_floor = _EMERGENCY_RESERVE_FLOOR if step_emergency_active else _HOME_RESERVE_FLOOR
        step_required_reserve = max(float(action.min_reserve_pct), step_reserve_floor)
        prev_mean_soh = self._state.mean_state_of_health

        if freq_hz < 49.8 and self._emergency_start_step is None:
            self._emergency_start_step = s

        # ── Clamp action ───────────────────────────────────────────────────
        max_kw    = 5.0
        requested_charge_kw = float(np.clip(action.global_charge_rate * max_kw, -max_kw, max_kw))
        charge_kw = requested_charge_kw
        reconnection_soft_sync_active = bool(meta.get("has_islanding", False) and s == ISLANDING_END)
        reconnection_soft_sync_applied = False
        if reconnection_soft_sync_active and charge_kw < 0.0:
            capped_discharge_kw = -max_kw * _RECONNECTION_DISCHARGE_CAP_RATIO
            charge_kw = max(charge_kw, capped_discharge_kw)
            reconnection_soft_sync_applied = charge_kw > requested_charge_kw

        # ── DR: accept or check commitment ───────────────────────────────────
        dr_bonus       = 0.0
        dr_penalty     = 0.0
        new_dr_bid_accepted = False
        dr_penalty_multiplier_active = min(
            _DR_FAIL_PENALTY_MAX_MULT,
            _DR_FAIL_PENALTY_BASE_MULT + float(self._dr_failure_streak),
        )
        dr_window_fulfillment_ratio = 1.0

        if action.accept_dr_bid and not self._dr_committed:
            bid_params = self._dr_schedule.get(s)
            if bid_params:
                self._dr_committed  = True
                self._dr_until_step = s + bid_params[2]
                self._dr_power_kw   = bid_params[1]
                self._dr_premium    = bid_params[0]
                self._dr_missed_steps = 0
                self._dr_committed_kwh_total = 0.0
                self._dr_delivered_kwh_total = 0.0
                self._state.dr_bids_accepted += 1
                self._state.dr_active = True
                self._state.dr_committed_until  = self._dr_until_step
                self._state.dr_committed_power_kw = self._dr_power_kw
                self._state.dr_premium_multiplier = self._dr_premium
                new_dr_bid_accepted = True

        # Check if DR commitment window ended
        if self._dr_committed and s >= self._dr_until_step:
            self._dr_committed = False
            self._state.dr_active = False
            self._state.dr_committed_until = 0
            self._state.dr_committed_power_kw = 0.0
            self._state.dr_premium_multiplier = 1.0

        # ── EV deferral ────────────────────────────────────────────────────
        # Only Zone B (homes 40-99) has EV chargers; load added after step 32 (14:00)
        ev_defer_frac = float(np.clip(action.defer_ev_charging, 0.0, 1.0))

        # ── Grid emergency tracking ────────────────────────────────────────
        # Evaluate emergency response after physical dispatch is computed.

        # ── Islanding: can't buy from / sell to grid ───────────────────────
        if not grid_connected:
            charge_kw = 0.0   # grid is disconnected; no grid transactions

        # ── Per-asset physics ──────────────────────────────────────────────
        step_profit        = 0.0
        step_revenue       = 0.0
        step_cost          = 0.0
        step_p2p_revenue   = 0.0
        step_carbon_earned = 0.0
        step_carbon_spent  = 0.0
        step_demand_shed_kwh = 0.0
        step_has_violation = False
        step_has_blackout  = False
        fleet_charge_kw_total = 0.0
        fleet_discharge_kw_total = 0.0
        islanding_blackout_home_steps = 0
        islanding_blackout_home_ids_step: set[str] = set()

        # P2P calculation (Zone B surplus → Zone A)
        zone_b_assets  = self.assets[_ZONE_B_START:_ZONE_B_END]
        zone_a_assets  = self.assets[_ZONE_A_START:_ZONE_A_END]
        p2p_export_rate = float(np.clip(action.p2p_export_rate, 0.0, 1.0))

        # EV adder for Zone B (after 14:00 = step 32)
        ev_adder_kw = 0.0
        zone_b_count = len(zone_b_assets)
        if s >= 32 and zone_b_count > 0:
            base_ev_kw = 1.2
            old_debt_kwh = self._ev_defer_debt_kwh
            deferred_kwh = base_ev_kw * ev_defer_frac * 0.25 * zone_b_count
            ev_adder_kw = base_ev_kw * (1.0 - ev_defer_frac)
            replay_kwh = 0.0

            # Replay deferred EV energy progressively; force full replay at/after deadline.
            if old_debt_kwh > 0:
                if s < EV_DEFER_DEADLINE:
                    remaining_steps = max(1, EV_DEFER_DEADLINE - s)
                    replay_kwh = old_debt_kwh / remaining_steps
                else:
                    replay_kwh = old_debt_kwh

                ev_adder_kw += replay_kwh / (0.25 * zone_b_count)
            self._ev_defer_debt_kwh = max(0.0, old_debt_kwh - replay_kwh) + deferred_kwh

        p2p_price_usd = price_usd * _P2P_PRICE_FRACTION
        # P2P exports from Zone B are capped by Zone A's absorbable deficit.
        zone_a_absorbable_kw_total = max(0.0, (demand_kw - solar_kw) * len(zone_a_assets))
        p2p_cap_per_zone_b_home = (
            zone_a_absorbable_kw_total / max(len(zone_b_assets), 1)
            if zone_b_assets
            else 0.0
        )

        for i, asset in enumerate(self.assets):
            is_zone_b = (i >= _ZONE_B_START)
            old_soc   = self._battery_soc[asset.asset_id]
            soh       = self._battery_soh[asset.asset_id]
            eff_capacity = asset.capacity_kwh * soh

            home_demand = demand_kw + (ev_adder_kw if is_zone_b else 0.0)
            flexible_demand_kw = home_demand * _FLEXIBLE_LOAD_FRACTION if not grid_connected else 0.0

            # P2P: Zone B can export to Zone A
            p2p_kw = 0.0
            if is_zone_b:
                surplus = max(0.0, solar_kw - home_demand)
                p2p_kw  = min(surplus * p2p_export_rate, p2p_cap_per_zone_b_home)

            # Battery dispatch constrained by available energy/headroom this step.
            base_net_kw = solar_kw - home_demand - p2p_kw
            if charge_kw >= 0:
                desired_effective_kw = charge_kw * asset.efficiency_rt
                max_effective_kw_from_soc = ((1.0 - old_soc) * eff_capacity / 0.25) - base_net_kw
                max_effective_kw_from_soc = max(0.0, max_effective_kw_from_soc)
                effective_kw = float(np.clip(desired_effective_kw, 0.0, max_effective_kw_from_soc))
                actual_charge_kw = effective_kw / max(asset.efficiency_rt, 1e-6)
            else:
                desired_effective_kw = charge_kw
                min_effective_kw_from_soc = ((step_required_reserve - old_soc) * eff_capacity / 0.25) - base_net_kw
                min_effective_kw_from_soc = min(0.0, min_effective_kw_from_soc)
                effective_kw = float(np.clip(desired_effective_kw, min_effective_kw_from_soc, 0.0))
                actual_charge_kw = effective_kw

            if actual_charge_kw < 0:
                fleet_discharge_kw_total += abs(actual_charge_kw)
            elif actual_charge_kw > 0:
                fleet_charge_kw_total += actual_charge_kw

            net_kw       = base_net_kw + effective_kw
            delta_kwh    = net_kw * 0.25
            new_soc      = old_soc + delta_kwh / max(eff_capacity, 0.01)
            new_soc      = float(np.clip(new_soc, 0.0, 1.0))

            # Violation check
            if new_soc < step_required_reserve:
                step_has_violation = True

            if (
                not grid_connected
                and net_kw < 0.0
                and new_soc <= step_required_reserve + _SCORE_EPSILON
            ):
                unmet_kw = abs(net_kw)
                critical_unserved_kw = max(0.0, unmet_kw - flexible_demand_kw)
                step_demand_shed_kwh += unmet_kw * 0.25
                if critical_unserved_kw > _SCORE_EPSILON:
                    step_has_blackout = True
                    islanding_blackout_home_steps += 1
                    islanding_blackout_home_ids_step.add(asset.asset_id)

            # Blackout: battery empty while demand exists (non-islanding semantics)
            if grid_connected and new_soc <= 0.0 and home_demand > 0:
                step_has_blackout = True

            # ── SoH degradation ──────────────────────────────────────────
            # Count battery wear from actual battery energy throughput.
            abs_delta_kwh = abs(delta_kwh)
            self._battery_cycles[asset.asset_id] += abs_delta_kwh / (2.0 * asset.capacity_kwh)
            cycles = self._battery_cycles[asset.asset_id]
            new_soh = max(_SOH_FLOOR, 1.0 - cycles * _DEGRADATION_PER_CYCLE)

            self._battery_soc[asset.asset_id] = new_soc
            self._battery_soh[asset.asset_id] = new_soh

            # ── Grid financial ───────────────────────────────────────────
            if grid_connected:
                exported_kwh    = -actual_charge_kw * 0.25
                effective_price = price_usd * (
                    self._dr_premium if (self._dr_committed and exported_kwh > 0) else 1.0
                )
                home_profit     = exported_kwh * (effective_price / 1000.0)
                step_profit    += home_profit
                if exported_kwh > 0:
                    step_revenue += home_profit
                    if self._dr_committed:
                        dr_bonus += exported_kwh * (price_usd / 1000.0) * (self._dr_premium - 1.0)
                else:
                    step_cost    += abs(home_profit)

            # ── P2P revenue ──────────────────────────────────────────────
            if is_zone_b and p2p_kw > 0:
                p2p_revenue_home  = p2p_kw * 0.25 * (p2p_price_usd / 1000.0)
                step_p2p_revenue += p2p_revenue_home
                step_revenue     += p2p_revenue_home
                step_profit      += p2p_revenue_home

            # ── Carbon credits ────────────────────────────────────────────
            # Earn from solar generation (kWh this step)
            solar_kwh = solar_kw * 0.25
            earned    = solar_kwh * _CARBON_EARN_RATE
            step_carbon_earned += earned

        # ── Global carbon spend (grid purchases during high-emission hours) ─
        if s in HIGH_EMISSION_STEPS and fleet_charge_kw_total > 0 and grid_connected:
            bought_kwh = fleet_charge_kw_total * 0.25
            spent      = bought_kwh * _CARBON_SPEND_RATE
            step_carbon_spent += spent

        fleet_discharge_kw = fleet_discharge_kw_total / max(len(self.assets), 1)
        if freq_hz < 49.8 and fleet_discharge_kw > _SCORE_EPSILON and self._response_latency_steps is None:
            start_step = self._emergency_start_step if self._emergency_start_step is not None else s
            self._response_latency_steps = max(0, s - start_step)
        if freq_hz < 49.8 and fleet_discharge_kw <= _SCORE_EPSILON:
            self._state.grid_emergencies_ignored += 1

        # ── DR commitment enforcement ──────────────────────────────────────
        if self._dr_committed:
            discharge_kw = max(0.0, fleet_discharge_kw)
            committed_kwh_step = self._dr_power_kw * 0.25
            delivered_kwh_step = min(discharge_kw, self._dr_power_kw) * 0.25
            shortfall_kwh = max(0.0, committed_kwh_step - delivered_kwh_step)

            self._dr_committed_kwh_total += committed_kwh_step
            self._dr_delivered_kwh_total += delivered_kwh_step

            if shortfall_kwh > _SCORE_EPSILON:
                dr_penalty_multiplier_active = min(
                    _DR_FAIL_PENALTY_MAX_MULT,
                    _DR_FAIL_PENALTY_BASE_MULT + float(self._dr_failure_streak),
                )
                dr_penalty_step = (
                    shortfall_kwh
                    * (price_usd * self._dr_premium / 1000.0)
                    * dr_penalty_multiplier_active
                )
                dr_penalty += dr_penalty_step
                step_profit -= dr_penalty_step
                step_cost += dr_penalty_step
                self._dr_missed_steps += 1

            if s == self._dr_until_step - 1:
                if self._dr_committed_kwh_total > _SCORE_EPSILON:
                    dr_window_fulfillment_ratio = float(
                        np.clip(
                            self._dr_delivered_kwh_total / self._dr_committed_kwh_total,
                            0.0,
                            1.0,
                        )
                    )
                else:
                    dr_window_fulfillment_ratio = 1.0

                self._state.dr_fulfillment_ratio_sum = round(
                    self._state.dr_fulfillment_ratio_sum + dr_window_fulfillment_ratio,
                    6,
                )
                if dr_window_fulfillment_ratio >= 1.0 - _SCORE_EPSILON:
                    self._state.dr_bids_fulfilled += 1
                    self._dr_failure_streak = 0
                else:
                    self._state.dr_bids_failed += 1
                    self._dr_failure_streak += 1
                self._state.dr_consecutive_failures = self._dr_failure_streak

        # ── Islanding blackout penalty ─────────────────────────────────────
        # Home-step blackout penalty tracks unserved critical load only.
        if not grid_connected and islanding_blackout_home_steps > 0:
            islanding_penalty = islanding_blackout_home_steps * _ISLANDING_BLACKOUT_PENALTY
            step_profit -= islanding_penalty
            step_cost   += islanding_penalty
            self._state.islanding_blackout_home_steps += islanding_blackout_home_steps
            self._state.islanding_blackouts = self._state.islanding_blackout_home_steps
            self._islanding_blackout_home_ids.update(islanding_blackout_home_ids_step)
            self._state.islanding_blackout_unique_homes = len(self._islanding_blackout_home_ids)

        # ── Accumulate state ───────────────────────────────────────────────
        if step_has_violation:
            self._state.safety_violations_count += 1
        if step_has_blackout:
            self._state.blackout_events_count   += 1

        self._carbon_balance  += step_carbon_earned - step_carbon_spent
        self._carbon_earned   += step_carbon_earned
        self._carbon_spent    += step_carbon_spent
        self._last_step_carbon_earned = step_carbon_earned
        self._last_step_carbon_spent = step_carbon_spent
        self._last_step_demand_shed_kwh = step_demand_shed_kwh
        self._cumulative_demand_shed_kwh += step_demand_shed_kwh

        self._total_profit += step_profit
        self._p2p_last_revenue = step_p2p_revenue

        self._state.cumulative_profit_usd    = round(self._total_profit, 4)
        self._state.cumulative_revenue_usd   = round(self._state.cumulative_revenue_usd + step_revenue, 4)
        self._state.cumulative_cost_usd      = round(self._state.cumulative_cost_usd + step_cost, 4)
        self._state.cumulative_p2p_usd       = round(self._state.cumulative_p2p_usd + step_p2p_revenue, 4)
        self._state.cumulative_dr_bonus_usd  = round(self._state.cumulative_dr_bonus_usd + dr_bonus, 4)
        self._state.cumulative_dr_penalty_usd = round(self._state.cumulative_dr_penalty_usd + dr_penalty, 4)
        self._state.carbon_credits_balance   = round(self._carbon_balance, 4)
        self._state.carbon_credits_earned    = round(self._carbon_earned, 4)
        self._state.carbon_credits_spent     = round(self._carbon_spent, 4)
        self._state.cumulative_demand_shed_kwh = round(self._cumulative_demand_shed_kwh, 4)
        self._state.response_latency_steps_to_emergency = (
            self._response_latency_steps if self._response_latency_steps is not None else -1
        )
        self._state.ev_defer_debt_kwh        = round(self._ev_defer_debt_kwh, 4)
        self._state.battery_true_soc         = self._battery_soc.copy()
        self._state.battery_true_soh         = self._battery_soh.copy()
        self._state.grid_connected           = grid_connected

        # SoH aggregate
        soh_vals = list(self._battery_soh.values())
        self._state.mean_state_of_health = round(float(np.mean(soh_vals)), 6)
        self._state.min_state_of_health  = round(float(np.min(soh_vals)), 6)
        self._state.total_cycle_count    = round(sum(self._battery_cycles.values()), 4)

        mean_soc = float(np.mean(list(self._battery_soc.values()))) if self._battery_soc else 0.0
        step_safety_margin_pct = (mean_soc - step_required_reserve) * 100.0
        carbon_step_bonus = (step_carbon_earned - step_carbon_spent) * _STEP_CARBON_REWARD_MULT
        soh_drop = max(0.0, prev_mean_soh - self._state.mean_state_of_health)
        degradation_penalty = soh_drop * _STEP_DEGRADATION_PENALTY_MULT
        safety_margin_bonus = _STEP_SAFETY_MARGIN_BONUS if step_safety_margin_pct >= 10.0 else 0.0

        # ── Reward ─────────────────────────────────────────────────────────
        safety_penalty = _STEP_SAFETY_VIOLATION_PENALTY if step_has_violation else 0.0
        emergency_penalty = 0.0
        emergency_bonus = 0.0
        if freq_hz < 49.8:
            if fleet_discharge_kw <= _SCORE_EPSILON:
                emergency_penalty = _STEP_EMERGENCY_MISS_PENALTY
            elif fleet_discharge_kw < 1.0:
                emergency_penalty = _STEP_EMERGENCY_WEAK_RESPONSE_PENALTY
            elif fleet_discharge_kw >= 2.0:
                emergency_bonus = _STEP_EMERGENCY_STRONG_RESPONSE_BONUS

        reward = (
            step_profit
            + carbon_step_bonus
            + safety_margin_bonus
            + emergency_bonus
            - safety_penalty
            - emergency_penalty
            - degradation_penalty
        )

        # ── Advance step ───────────────────────────────────────────────────
        self._current_step         += 1
        done                        = self._current_step >= EPISODE_STEPS
        self._state.step_count     += 1
        self._state.current_step    = self._current_step
        self._state.done            = done
        # Keep trend and emergency flags aligned with the observation index returned below.
        obs_step_index = min(self._current_step, EPISODE_STEPS - 1)
        obs_freq_hz = self._grid_frequency(obs_step_index)
        obs_grid_connected = self._is_grid_connected(obs_step_index)
        obs_emergency_active = (obs_freq_hz < 49.8) or (not obs_grid_connected)
        obs_reserve_floor = _EMERGENCY_RESERVE_FLOOR if obs_emergency_active else _HOME_RESERVE_FLOOR
        obs_required_reserve = max(float(action.min_reserve_pct), obs_reserve_floor)
        obs_safety_margin_pct = (mean_soc - obs_required_reserve) * 100.0

        self._frequency_history.append(round(obs_freq_hz, 3))
        self._frequency_history = self._frequency_history[-3:]

        # Reasoning trace
        if action.reasoning:
            self._reasoning_traces.append({
                "step": s,
                "action": {
                    "global_charge_rate": action.global_charge_rate,
                    "min_reserve_pct": action.min_reserve_pct,
                },
                "reasoning": action.reasoning,
                "step_profit": round(step_profit, 4),
            })

        info = {
            "step_profit_usd":            round(step_profit, 4),
            "step_p2p_revenue_usd":       round(step_p2p_revenue, 4),
            "step_dr_bonus_usd":          round(dr_bonus, 4),
            "step_dr_penalty_usd":        round(dr_penalty, 4),
            "step_carbon_earned":         round(step_carbon_earned, 4),
            "step_carbon_spent":          round(step_carbon_spent, 4),
            "step_demand_shed_kwh":       round(step_demand_shed_kwh, 4),
            "safety_violation_this_step": step_has_violation,
            "blackout_this_step":         step_has_blackout,
            "grid_frequency_hz":          round(freq_hz, 3),
            "grid_connected":             grid_connected,
            "safety_margin_pct":          round(step_safety_margin_pct, 3),
            "emergency_active":           step_emergency_active,
            "response_latency_steps_to_emergency": (
                self._response_latency_steps if self._response_latency_steps is not None else -1
            ),
            "mean_soh":                   round(self._state.mean_state_of_health, 4),
            "ev_defer_debt_kwh":          round(self._ev_defer_debt_kwh, 4),
            "islanding_blackout_homes":   islanding_blackout_home_steps,
            "islanding_blackout_home_steps": islanding_blackout_home_steps,
            "islanding_blackout_unique_homes": self._state.islanding_blackout_unique_homes,
            "new_dr_bid_accepted":        new_dr_bid_accepted,
            "dr_penalty_multiplier_active": round(dr_penalty_multiplier_active, 3),
            "dr_window_fulfillment_ratio": round(dr_window_fulfillment_ratio, 4),
            "dr_consecutive_failures":    self._state.dr_consecutive_failures,
            "requested_charge_kw_per_home": round(requested_charge_kw, 4),
            "applied_charge_kw_per_home": round(charge_kw, 4),
            "reconnection_soft_sync_active": reconnection_soft_sync_active,
            "reconnection_soft_sync_applied": reconnection_soft_sync_applied,
        }
        pareto_score = self._get_pareto_score().model_dump()
        self._set_last_grader_snapshot(
            {
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "task_id": self._task_id,
                "episode_id": self._state.episode_id,
                "step_id": self._state.current_step,
                "done": self._state.done,
                "environment_instance_id": id(self),
                "pareto_score": pareto_score,
            }
        )
        info["pareto_score"] = pareto_score
        obs = self._build_observation(
            reward=round(reward, 6),
            done=done,
            metadata=info,
            pareto_score=pareto_score,
            safety_margin_pct=round(obs_safety_margin_pct, 3),
            emergency_active=obs_emergency_active,
            demand_shed_this_step_kwh=round(step_demand_shed_kwh, 4),
            cumulative_demand_shed_kwh=round(self._cumulative_demand_shed_kwh, 4),
            carbon_earned_this_step=round(step_carbon_earned, 4),
            carbon_spent_this_step=round(step_carbon_spent, 4),
            grid_frequency_trend_hz=list(self._frequency_history),
            response_latency_steps_to_emergency=(
                self._response_latency_steps if self._response_latency_steps is not None else -1
            ),
        )
        return obs

    # ── Grader ────────────────────────────────────────────────────────────────

    def _get_pareto_score(self) -> ParetoScore:
        """
        Deterministic multi-objective Pareto score.

        profit_score     = min(1, profit / goal)
        safety_score     = 1 − min(1, violations / 48)
        carbon_score     = min(1, carbon_balance / carbon_target)
        degradation_score = (mean_soh − 0.80) / (1.0 − 0.80)   [normalised]
        dr_score          = fulfilled / accepted (task-aware fallback when no bids accepted)

        aggregate = 0.50×profit + 0.20×safety + 0.15×carbon + 0.10×degradation + 0.05×dr
        
        All scores are clamped to the open interval (0, 1) using epsilon = 0.0001.
        """
        if self._state is None:
            return ParetoScore(
                profit_score=0.5, safety_score=0.5, carbon_score=0.5,
                degradation_score=0.5, dr_score=0.5, aggregate_score=0.5,
                cumulative_profit_usd=0.0,
                cumulative_p2p_usd=0.0,
                cumulative_dr_bonus_usd=0.0,
                safety_violations=0,
                grid_emergencies_ignored=0,
                islanding_blackouts=0,
                islanding_blackout_home_steps=0,
                islanding_blackout_unique_homes=0,
                cumulative_demand_shed_kwh=0.0,
                response_latency_steps_to_emergency=-1,
                carbon_credits_balance=0.0,
                mean_state_of_health=1.0,
                dr_bids_fulfilled=0,
                dr_bids_failed=0,
                dr_consecutive_failures=0,
                dr_mean_fulfillment_ratio=0.0,
                steps_completed=0,
                done=False,
            )

        meta = TASK_METADATA.get(self._task_id, TASK_METADATA["easy-arbitrage"])
        goal = meta["profit_target_usd"]
        carbon_target = meta["carbon_target_credits"]

        # Profit (with epsilon clamping to stay strictly in (0, 1))
        total_profit = self._state.cumulative_profit_usd
        profit_score = float(np.clip(total_profit / max(goal, 1.0), _SCORE_EPSILON, 1.0 - _SCORE_EPSILON))

        # Safety (violations + emergencies) (with epsilon clamping)
        violation_ratio   = self._state.safety_violations_count / EPISODE_STEPS
        emergency_ratio   = min(1.0, self._state.grid_emergencies_ignored / max(EPISODE_STEPS, 1))
        demand_shed_ratio = min(1.0, self._state.cumulative_demand_shed_kwh / _DEMAND_SHED_NORMALISATION_KWH)
        latency_penalty = 0.0
        if self._state.response_latency_steps_to_emergency >= 0:
            latency_penalty = min(_SAFETY_LATENCY_MAX_PENALTY, self._state.response_latency_steps_to_emergency * 0.05)
        safety_score = float(
            np.clip(
                1.0
                - violation_ratio * _SAFETY_VIOLATION_WEIGHT
                - emergency_ratio * _SAFETY_EMERGENCY_WEIGHT
                - demand_shed_ratio * _SAFETY_DEMAND_SHED_WEIGHT
                - latency_penalty,
                _SCORE_EPSILON,
                _SAFETY_SCORE_SOFT_CEILING,
            )
        )

        # Carbon (with epsilon clamping)
        carbon_score = float(np.clip(self._carbon_balance / max(carbon_target, 1.0), _SCORE_EPSILON, 1.0 - _SCORE_EPSILON))

        # Battery degradation (1.0 = no degradation, 0.0 = all at SoH floor) (with epsilon clamping)
        normalised_soh = (self._state.mean_state_of_health - _SOH_FLOOR) / (1.0 - _SOH_FLOOR)
        degradation_score = float(np.clip(normalised_soh, _SCORE_EPSILON, 1.0 - _SCORE_EPSILON))

        # DR participation (with epsilon clamping and task-aware fallback)
        dr_mean_fulfillment_ratio = 0.0
        if self._state.dr_bids_accepted > 0:
            dr_mean_fulfillment_ratio = self._state.dr_fulfillment_ratio_sum / self._state.dr_bids_accepted
            dr_score = dr_mean_fulfillment_ratio
            dr_score = float(np.clip(dr_score, _SCORE_EPSILON, 1.0 - _SCORE_EPSILON))
        elif meta.get("has_dr_auction", False):
            # Expert task posts DR opportunities; declining all bids should not score near-perfect.
            dr_score = _SCORE_EPSILON
        else:
            # Non-DR tasks should not be penalized on the DR objective.
            dr_score = _NON_DR_TASK_NEUTRAL_DR_SCORE

        # Islanding blackout deduction to safety
        islanding_blackout_home_steps = self._state.islanding_blackout_home_steps
        if islanding_blackout_home_steps > 0:
            safety_score = float(
                np.clip(
                    safety_score - min(0.3, islanding_blackout_home_steps * 0.01),
                    _SCORE_EPSILON,
                    _SAFETY_SCORE_SOFT_CEILING,
                )
            )

        # Weighted aggregate (with epsilon clamping)
        aggregate = (
            0.50 * profit_score
            + 0.20 * safety_score
            + 0.15 * carbon_score
            + 0.10 * degradation_score
            + 0.05 * dr_score
        )
        aggregate = float(np.clip(aggregate, _SCORE_EPSILON, 1.0 - _SCORE_EPSILON))

        return ParetoScore(
            profit_score=round(profit_score, 4),
            safety_score=round(safety_score, 4),
            carbon_score=round(carbon_score, 4),
            degradation_score=round(degradation_score, 4),
            dr_score=round(dr_score, 4),
            aggregate_score=round(aggregate, 4),
            cumulative_profit_usd=self._state.cumulative_profit_usd,
            cumulative_p2p_usd=self._state.cumulative_p2p_usd,
            cumulative_dr_bonus_usd=self._state.cumulative_dr_bonus_usd,
            safety_violations=self._state.safety_violations_count,
            grid_emergencies_ignored=self._state.grid_emergencies_ignored,
            islanding_blackouts=self._state.islanding_blackouts,
            islanding_blackout_home_steps=self._state.islanding_blackout_home_steps,
            islanding_blackout_unique_homes=self._state.islanding_blackout_unique_homes,
            cumulative_demand_shed_kwh=self._state.cumulative_demand_shed_kwh,
            response_latency_steps_to_emergency=self._state.response_latency_steps_to_emergency,
            carbon_credits_balance=round(self._carbon_balance, 4),
            mean_state_of_health=self._state.mean_state_of_health,
            dr_bids_fulfilled=self._state.dr_bids_fulfilled,
            dr_bids_failed=self._state.dr_bids_failed,
            dr_consecutive_failures=self._state.dr_consecutive_failures,
            dr_mean_fulfillment_ratio=round(dr_mean_fulfillment_ratio, 4),
            steps_completed=self._state.current_step,
            done=self._state.done,
        )

    def get_current_task_score(self) -> float:
        return self._get_pareto_score().aggregate_score

    def get_pareto_score(self) -> ParetoScore:
        return self._get_pareto_score()

    # ── Reasoning traces ─────────────────────────────────────────────────────

    def get_reasoning_traces(self) -> List[dict]:
        return list(self._reasoning_traces)

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def state(self) -> VppState:
        if self._state is None:
            raise RuntimeError("State unavailable. Call reset() first.")
        return self._state

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _grid_frequency(self, step: int) -> float:
        """
        49.5 Hz emergency ONLY during hard-frequency-response task at step 26.
        Other tasks always see nominal 50.0 Hz grid frequency.
        """
        if self._task_id == "hard-frequency-response" and step == GRID_STRESS_STEP:
            return 49.5
        return 50.0

    def _is_grid_connected(self, step: int) -> bool:
        """Grid disconnects during islanding task between ISLANDING_START and ISLANDING_END."""
        meta = TASK_METADATA.get(self._task_id, {})
        if meta.get("has_islanding", False):
            return not (ISLANDING_START <= step < ISLANDING_END)
        return True

    def _get_current_dr_bid(self, step: int) -> DRBid:
        """Return the DR bid object for the current step (if any)."""
        bid_params = self._dr_schedule.get(step)
        if bid_params:
            return DRBid(
                active=True,
                premium_multiplier=bid_params[0],
                committed_power_kw=bid_params[1],
                committed_steps=bid_params[2],
                steps_remaining=0 if not self._dr_committed else max(0, self._dr_until_step - step),
            )
        if self._dr_committed:
            return DRBid(
                active=True,
                premium_multiplier=self._dr_premium,
                committed_power_kw=self._dr_power_kw,
                committed_steps=0,
                steps_remaining=max(0, self._dr_until_step - step),
            )
        return DRBid(
            active=False,
            premium_multiplier=1.0,
            committed_power_kw=0.0,
            committed_steps=0,
            steps_remaining=0,
        )

    def _build_zone_aggregates(self, idx: int) -> List[ZoneTelemetry]:
        if self._true_solar is None or self._true_demand is None:
            raise RuntimeError("Missing environment curves. Call reset() before building aggregates.")

        solar_kw  = float(self._true_solar[idx])
        demand_kw = float(self._true_demand[idx])

        zones = [
            ("zone-a", _ZONE_A_START, _ZONE_A_END, False),
            ("zone-b", _ZONE_B_START, _ZONE_B_END, True),
        ]
        result: List[ZoneTelemetry] = []

        for zone_id, start, end, has_ev in zones:
            zone_assets = self.assets[start:end]
            socs = [self._battery_soc[a.asset_id] for a in zone_assets]
            sohs = [self._battery_soh[a.asset_id] for a in zone_assets]

            ev_adder = 0.0
            if has_ev and idx >= 32:
                ev_adder = self._project_zone_b_ev_adder(idx)

            # P2P available power (Zone B surplus)
            surplus_kw = max(0.0, solar_kw - (demand_kw + ev_adder)) if has_ev else 0.0

            result.append(
                ZoneTelemetry(
                    zone_id=zone_id,
                    home_count=len(zone_assets),
                    mean_soc=float(np.mean(socs)),
                    min_soc=float(np.min(socs)),
                    max_soc=float(np.max(socs)),
                    mean_soh=float(np.mean(sohs)),
                    mean_solar_kw=solar_kw,
                    mean_demand_kw=demand_kw + ev_adder,
                    has_ev_chargers=has_ev,
                    p2p_available_kw=round(surplus_kw, 3),
                )
            )

        return result

    def _project_zone_b_ev_adder(self, step: int) -> float:
        """Projected Zone B EV load per home for observations/aggregates."""
        if step < 32:
            return 0.0

        base_ev_kw = 1.2
        zone_b_count = _ZONE_B_END - _ZONE_B_START
        if zone_b_count <= 0 or self._ev_defer_debt_kwh <= 0:
            return base_ev_kw

        if step < EV_DEFER_DEADLINE:
            remaining_steps = max(1, EV_DEFER_DEADLINE - step)
            replay_kwh = self._ev_defer_debt_kwh / remaining_steps
        else:
            replay_kwh = self._ev_defer_debt_kwh

        return base_ev_kw + replay_kwh / (0.25 * zone_b_count)

    def _build_observation(
        self,
        reward: float = 0.0,
        done: bool = False,
        metadata: Optional[Dict[str, object]] = None,
        pareto_score: Optional[Dict[str, object]] = None,
        safety_margin_pct: float = 0.0,
        emergency_active: bool = False,
        demand_shed_this_step_kwh: float = 0.0,
        cumulative_demand_shed_kwh: float = 0.0,
        carbon_earned_this_step: float = 0.0,
        carbon_spent_this_step: float = 0.0,
        grid_frequency_trend_hz: Optional[List[float]] = None,
        response_latency_steps_to_emergency: int = -1,
    ) -> VppObservation:
        if self._true_solar is None or self._true_demand is None or self._true_price is None:
            raise RuntimeError("Missing environment curves. Call reset() before observation.")

        idx = min(self._current_step, EPISODE_STEPS - 1)
        now = self._episode_start + timedelta(minutes=15 * idx)

        # Per-home telemetry (includes SoH)
        ev_zone_b_kw = self._project_zone_b_ev_adder(idx)
        telemetry = [
            BatteryTelemetry(
                asset_id=asset.asset_id,
                soc=self._battery_soc[asset.asset_id],
                state_of_health=self._battery_soh[asset.asset_id],
                current_house_load_kw=float(
                    self._true_demand[idx] + (ev_zone_b_kw if i >= _ZONE_B_START else 0.0)
                ),
                current_solar_gen_kw=float(self._true_solar[idx]),
            )
            for i, asset in enumerate(self.assets)
        ]

        zone_aggregates = self._build_zone_aggregates(idx)

        forecast_price_source = self._forecast_price if self._forecast_price is not None else self._true_price
        forecast_solar_source = self._forecast_solar if self._forecast_solar is not None else self._true_solar

        # Noisy 4-step look-ahead + uncertainty bands
        n = 4
        price_slice = list(forecast_price_source[idx: idx + n])
        solar_slice = list(forecast_solar_source[idx: idx + n])
        while len(price_slice) < n:
            price_slice.append(price_slice[-1])
        while len(solar_slice) < n:
            solar_slice.append(solar_slice[-1])

        noisy_price = [float(p + random.gauss(0, _PRICE_SIGMA[i])) for i, p in enumerate(price_slice)]
        noisy_solar = [float(s + random.gauss(0, _SOLAR_SIGMA[i])) for i, s in enumerate(solar_slice)]

        # DR bid for this step
        dr_bid = self._get_current_dr_bid(idx)
        forecast_realtime_error_price_usd = float(self._true_price[idx] - forecast_price_source[idx])
        forecast_realtime_error_solar_kw = float(self._true_solar[idx] - forecast_solar_source[idx])

        payload = {
            "timestamp": now,
            "step_id": idx,
            "telemetry": telemetry,
            "zone_aggregates": zone_aggregates,
            "grid_frequency_hz": self._grid_frequency(idx),
            "grid_voltage_v": 230.0,
            "grid_connected": self._is_grid_connected(idx),
            "market_price_per_mwh": float(self._true_price[idx]),
            "reward": reward,
            "done": done,
            "metadata": metadata or {},
            "carbon_credits_balance": round(self._carbon_balance, 4),
            "forecast_24h_price": forecast_price_source.tolist(),
            "forecast_24h_solar": forecast_solar_source.tolist(),
            "short_term_price_forecast": noisy_price,
            "short_term_solar_forecast": noisy_solar,
            "forecast_price_uncertainty": list(_PRICE_SIGMA),
            "forecast_solar_uncertainty": list(_SOLAR_SIGMA),
            "dr_bid": dr_bid,
            "ev_defer_deadline_step": EV_DEFER_DEADLINE,
            "p2p_last_revenue_usd": round(self._p2p_last_revenue, 4),
            "safety_margin_pct": safety_margin_pct,
            "emergency_active": emergency_active,
            "demand_shed_this_step_kwh": demand_shed_this_step_kwh,
            "cumulative_demand_shed_kwh": cumulative_demand_shed_kwh,
            "carbon_earned_this_step": carbon_earned_this_step,
            "carbon_spent_this_step": carbon_spent_this_step,
            "grid_frequency_trend_hz": grid_frequency_trend_hz or [],
            "response_latency_steps_to_emergency": response_latency_steps_to_emergency,
            "forecast_realtime_error_price_usd": forecast_realtime_error_price_usd,
            "forecast_realtime_error_solar_kw": forecast_realtime_error_solar_kw,
            "pareto_score": pareto_score,
        }
        return VppObservation.model_validate(payload)

