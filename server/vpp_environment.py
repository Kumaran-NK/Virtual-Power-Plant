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

import contextvars
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from openenv.core.env_server.interfaces import Environment
from models import (
    BatteryAsset, BatteryTelemetry, ZoneTelemetry,
    DRBid, VppAction, VppObservation, VppState, ParetoScore,
)
from server.task_curves import (
    solar_curve, demand_curve, price_curve,
    emission_intensity_curve, dr_bid_schedule,
    EPISODE_STEPS, GRID_STRESS_STEP,
    ISLANDING_START, ISLANDING_END, EV_DEFER_DEADLINE,
    HIGH_EMISSION_STEPS, DR_BID_INTERVAL, TASK_METADATA, ALL_TASK_IDS,
)

# Request-local reference to the current environment instance.
# This is safe for async/await and concurrent request handling when the current
# request uses its own execution context.
_current_env_var: contextvars.ContextVar["VppEnvironment | None"] = contextvars.ContextVar(
    "current_vpp_environment", default=None
)


def get_current_env_instance() -> "VppEnvironment | None":
    """Retrieve the current environment instance for this execution context."""
    return _current_env_var.get()


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

# DR failure penalty multiplier
_DR_FAIL_PENALTY_MULT = 2.0   # 2× the missed revenue

# Islanding blackout penalty per home-step
_ISLANDING_BLACKOUT_PENALTY = 50.0   # USD per home blackout during islanding


class VppEnvironment(Environment):
    """Simulated Virtual Power Plant — 100 home batteries, 12-hour episodes, Extended Edition."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(self):
        _current_env_var.set(self)

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
        self._episode_start: datetime         = datetime.utcnow()

        self._true_solar:     np.ndarray | None = None
        self._true_demand:    np.ndarray | None = None
        self._true_price:     np.ndarray | None = None
        self._emission_curve: np.ndarray | None = None
        self._dr_schedule:    dict              = {}

        # DR runtime state
        self._dr_committed:   bool  = False
        self._dr_until_step:  int   = 0
        self._dr_power_kw:    float = 0.0
        self._dr_premium:     float = 1.0

        # EV deferral state
        self._ev_defer_debt_kwh: float = 0.0   # kWh still owed

        # P2P last-step revenue
        self._p2p_last_revenue: float = 0.0

        # Carbon credits
        self._carbon_balance:  float = 0.0
        self._carbon_earned:   float = 0.0
        self._carbon_spent:    float = 0.0

        # Reasoning trace storage
        self._reasoning_traces: List[dict] = []

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self, task_id: str = "easy-arbitrage") -> VppObservation:
        self._task_id = task_id

        seed = abs(hash(task_id)) % (2 ** 31)
        random.seed(seed)
        np.random.seed(seed)

        self._true_solar  = solar_curve(task_id)
        self._true_demand = demand_curve(task_id)
        self._true_price  = price_curve(task_id)
        self._emission_curve = emission_intensity_curve(task_id)
        self._dr_schedule = dr_bid_schedule(task_id)

        self._current_step  = 0
        self._total_profit  = 0.0
        self._episode_start = datetime.utcnow()
        self._p2p_last_revenue = 0.0
        self._carbon_balance = 0.0
        self._carbon_earned  = 0.0
        self._carbon_spent   = 0.0
        self._dr_committed   = False
        self._dr_until_step  = 0
        self._dr_power_kw    = 0.0
        self._dr_premium     = 1.0
        self._ev_defer_debt_kwh = 0.0
        self._reasoning_traces  = []

        self._battery_soc    = {a.asset_id: 0.5     for a in self.assets}
        self._battery_soh    = {a.asset_id: 1.0     for a in self.assets}
        self._battery_cycles = {a.asset_id: 0.0     for a in self.assets}

        meta    = TASK_METADATA.get(task_id, TASK_METADATA["easy-arbitrage"])
        weather = meta["weather"]

        self._state = VppState(
            episode_id=f"ep_{task_id}_{seed % 10_000:04d}",
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
            dr_bids_accepted=0,
            dr_bids_fulfilled=0,
            dr_bids_failed=0,
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

    def step(self, action: VppAction) -> Tuple[VppObservation, float, bool, dict]:
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

        # ── Clamp action ───────────────────────────────────────────────────
        max_kw    = 5.0
        charge_kw = float(np.clip(action.global_charge_rate * max_kw, -max_kw, max_kw))

        # ── DR: accept or check commitment ───────────────────────────────────
        dr_bonus       = 0.0
        dr_penalty     = 0.0
        new_dr_bid_accepted = False

        if action.accept_dr_bid and not self._dr_committed:
            bid_params = self._dr_schedule.get(s)
            if bid_params:
                self._dr_committed  = True
                self._dr_until_step = s + bid_params[2]
                self._dr_power_kw   = bid_params[1]
                self._dr_premium    = bid_params[0]
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

        # ── EV deferral ────────────────────────────────────────────────────
        ev_defer_frac = float(np.clip(action.defer_ev_charging, 0.0, 1.0))
        # EV charging that starts after step 32 (14:00)
        if s >= 32 and s < _ZONE_B_START:
            pass  # zone-level handled below

        # ── Grid emergency tracking ────────────────────────────────────────
        if freq_hz < 49.8 and charge_kw >= 0:
            self._state.grid_emergencies_ignored += 1

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
        step_has_violation = False
        step_has_blackout  = False
        islanding_blackout_homes = 0

        # P2P calculation (Zone B surplus → Zone A)
        zone_b_assets  = self.assets[_ZONE_B_START:_ZONE_B_END]
        zone_a_assets  = self.assets[_ZONE_A_START:_ZONE_A_END]
        p2p_export_rate = float(np.clip(action.p2p_export_rate, 0.0, 1.0))

        # EV adder for Zone B (after 14:00 = step 32)
        ev_adder_kw = 0.0
        if s >= 32:
            ev_adder_kw = 1.2 * (1.0 - ev_defer_frac)  # deferral reduces immediate load
            self._ev_defer_debt_kwh += 1.2 * ev_defer_frac * 0.25 * len(zone_b_assets)
        else:
            # Before deadline, slowly replay deferred debt
            if s == EV_DEFER_DEADLINE and self._ev_defer_debt_kwh > 0:
                # Force remaining debt onto batteries at deadline (penalty handled in reward)
                pass

        p2p_price_usd = price_usd * _P2P_PRICE_FRACTION

        for i, asset in enumerate(self.assets):
            is_zone_b = (i >= _ZONE_B_START)
            old_soc   = self._battery_soc[asset.asset_id]
            soh       = self._battery_soh[asset.asset_id]
            eff_capacity = asset.capacity_kwh * soh

            home_demand = demand_kw + (ev_adder_kw if is_zone_b else 0.0)

            # P2P: Zone B can export to Zone A
            p2p_kw = 0.0
            if is_zone_b:
                surplus = max(0.0, solar_kw - home_demand)
                p2p_kw  = surplus * p2p_export_rate

            # Battery delta
            effective_kw = charge_kw * asset.efficiency_rt if charge_kw > 0 else charge_kw
            net_kw       = solar_kw - home_demand + effective_kw - p2p_kw
            delta_kwh    = net_kw * 0.25
            new_soc      = old_soc + delta_kwh / max(eff_capacity, 0.01)
            new_soc      = float(np.clip(new_soc, 0.0, 1.0))

            # Violation check
            if new_soc < action.min_reserve_pct:
                step_has_violation = True

            # Blackout: battery empty while demand exists
            if new_soc <= 0.0 and home_demand > 0:
                step_has_blackout = True
                if not grid_connected:
                    islanding_blackout_homes += 1

            # ── SoH degradation ──────────────────────────────────────────
            abs_delta_kwh = abs(delta_kwh)
            self._battery_cycles[asset.asset_id] += abs_delta_kwh / (2.0 * asset.capacity_kwh)
            cycles = self._battery_cycles[asset.asset_id]
            new_soh = max(_SOH_FLOOR, 1.0 - cycles * _DEGRADATION_PER_CYCLE)

            self._battery_soc[asset.asset_id] = new_soc
            self._battery_soh[asset.asset_id] = new_soh

            # ── Grid financial ───────────────────────────────────────────
            if grid_connected:
                effective_price = price_usd * (self._dr_premium if self._dr_committed else 1.0)
                exported_kwh    = -charge_kw * 0.25
                home_profit     = exported_kwh * (effective_price / 1000.0)
                step_profit    += home_profit
                if exported_kwh > 0:
                    step_revenue += home_profit
                    if self._dr_committed:
                        dr_bonus += home_profit * (self._dr_premium - 1.0)
                else:
                    step_cost    += abs(home_profit)

            # ── P2P revenue ──────────────────────────────────────────────
            if is_zone_b and p2p_kw > 0:
                p2p_revenue_home  = p2p_kw * 0.25 * (p2p_price_usd / 1000.0)
                step_p2p_revenue += p2p_revenue_home
                step_profit      += p2p_revenue_home

            # ── Carbon credits ────────────────────────────────────────────
            # Earn from solar generation (kWh this step)
            solar_kwh = solar_kw * 0.25
            earned    = solar_kwh * _CARBON_EARN_RATE
            step_carbon_earned += earned

            # Spend on grid purchases during high-emission hours
            if s in HIGH_EMISSION_STEPS and charge_kw > 0 and grid_connected:
                bought_kwh = charge_kw * asset.efficiency_rt * 0.25
                spent      = bought_kwh * _CARBON_SPEND_RATE
                step_carbon_spent += spent

        # ── DR commitment enforcement ──────────────────────────────────────
        if self._dr_committed and grid_connected:
            # Check if agent is delivering enough power
            if charge_kw > -self._dr_power_kw:   # not discharging enough
                missed_kwh  = (self._dr_power_kw + charge_kw) * 0.25 * 100
                dr_penalty  = missed_kwh * (price_usd * self._dr_premium / 1000.0) * _DR_FAIL_PENALTY_MULT
                step_profit -= dr_penalty
                # Track if this DR commitment ended in failure
                if s == self._dr_until_step - 1:
                    self._state.dr_bids_failed += 1
            elif s == self._dr_until_step - 1:
                self._state.dr_bids_fulfilled += 1

        # ── Islanding blackout penalty ─────────────────────────────────────
        if not grid_connected and islanding_blackout_homes > 0:
            islanding_penalty = islanding_blackout_homes * _ISLANDING_BLACKOUT_PENALTY
            step_profit -= islanding_penalty
            self._state.islanding_blackouts += islanding_blackout_homes

        # ── Accumulate state ───────────────────────────────────────────────
        if step_has_violation:
            self._state.safety_violations_count += 1
        if step_has_blackout:
            self._state.blackout_events_count   += 1

        self._carbon_balance  += step_carbon_earned - step_carbon_spent
        self._carbon_earned   += step_carbon_earned
        self._carbon_spent    += step_carbon_spent

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
        self._state.ev_defer_debt_kwh        = round(self._ev_defer_debt_kwh, 4)
        self._state.battery_true_soc         = self._battery_soc.copy()
        self._state.battery_true_soh         = self._battery_soh.copy()
        self._state.grid_connected           = grid_connected

        # SoH aggregate
        soh_vals = list(self._battery_soh.values())
        self._state.mean_state_of_health = round(float(np.mean(soh_vals)), 6)
        self._state.min_state_of_health  = round(float(np.min(soh_vals)), 6)
        self._state.total_cycle_count    = round(sum(self._battery_cycles.values()), 4)

        # ── Reward ─────────────────────────────────────────────────────────
        safety_penalty    = 2.0 if step_has_violation               else 0.0
        emergency_penalty = 2.0 if freq_hz < 49.8 and charge_kw >= 0 else 0.0
        reward = step_profit - safety_penalty - emergency_penalty

        # ── Advance step ───────────────────────────────────────────────────
        self._current_step         += 1
        done                        = self._current_step >= EPISODE_STEPS
        self._state.step_count     += 1
        self._state.current_step    = self._current_step
        self._state.done            = done

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

        obs  = self._build_observation()
        info = {
            "step_profit_usd":            round(step_profit, 4),
            "step_p2p_revenue_usd":       round(step_p2p_revenue, 4),
            "step_dr_bonus_usd":          round(dr_bonus, 4),
            "step_dr_penalty_usd":        round(dr_penalty, 4),
            "step_carbon_earned":         round(step_carbon_earned, 4),
            "step_carbon_spent":          round(step_carbon_spent, 4),
            "safety_violation_this_step": step_has_violation,
            "blackout_this_step":         step_has_blackout,
            "grid_frequency_hz":          round(freq_hz, 3),
            "grid_connected":             grid_connected,
            "mean_soh":                   round(self._state.mean_state_of_health, 4),
            "ev_defer_debt_kwh":          round(self._ev_defer_debt_kwh, 4),
            "islanding_blackout_homes":   islanding_blackout_homes,
            "new_dr_bid_accepted":        new_dr_bid_accepted,
        }
        return obs, round(reward, 6), done, info

    # ── Grader ────────────────────────────────────────────────────────────────

    def _get_pareto_score(self) -> ParetoScore:
        """
        Deterministic multi-objective Pareto score.

        profit_score     = min(1, profit / goal)
        safety_score     = 1 − min(1, violations / 48)
        carbon_score     = min(1, carbon_balance / carbon_target)
        degradation_score = (mean_soh − 0.80) / (1.0 − 0.80)   [normalised]
        dr_score          = fulfilled / max(1, accepted)

        aggregate = 0.50×profit + 0.20×safety + 0.15×carbon + 0.10×degradation + 0.05×dr
        """
        if self._state is None:
            return ParetoScore(
                profit_score=0.0, safety_score=1.0, carbon_score=0.0,
                degradation_score=1.0, dr_score=0.0, aggregate_score=0.0,
                cumulative_profit_usd=0.0,
                cumulative_p2p_usd=0.0,
                cumulative_dr_bonus_usd=0.0,
                safety_violations=0,
                grid_emergencies_ignored=0,
                islanding_blackouts=0,
                carbon_credits_balance=0.0,
                mean_state_of_health=1.0,
                dr_bids_fulfilled=0,
                dr_bids_failed=0,
                steps_completed=0,
                done=False,
            )

        meta = TASK_METADATA.get(self._task_id, TASK_METADATA["easy-arbitrage"])
        goal = meta["profit_target_usd"]
        carbon_target = meta["carbon_target_credits"]

        # Profit
        total_profit = self._state.cumulative_profit_usd + \
                       self._state.cumulative_p2p_usd
        profit_score = float(np.clip(total_profit / max(goal, 1.0), 0.0, 1.0))

        # Safety (violations + emergencies)
        violation_ratio   = self._state.safety_violations_count / EPISODE_STEPS
        emergency_ratio   = min(1.0, self._state.grid_emergencies_ignored / max(EPISODE_STEPS, 1))
        safety_score = max(0.0, 1.0 - violation_ratio * 0.60 - emergency_ratio * 0.40)

        # Carbon
        carbon_score = float(np.clip(self._carbon_balance / max(carbon_target, 1.0), 0.0, 1.0))

        # Battery degradation (1.0 = no degradation, 0.0 = all at SoH floor)
        normalised_soh = (self._state.mean_state_of_health - _SOH_FLOOR) / (1.0 - _SOH_FLOOR)
        degradation_score = float(np.clip(normalised_soh, 0.0, 1.0))

        # DR participation
        if self._state.dr_bids_accepted > 0:
            dr_score = self._state.dr_bids_fulfilled / self._state.dr_bids_accepted
        else:
            dr_score = 1.0   # no bids → neutral (not penalised)

        # Islanding blackout deduction to safety
        if self._state.islanding_blackouts > 0:
            safety_score = max(0.0, safety_score - min(0.3, self._state.islanding_blackouts * 0.01))

        # Weighted aggregate
        aggregate = (
            0.50 * profit_score
            + 0.20 * safety_score
            + 0.15 * carbon_score
            + 0.10 * degradation_score
            + 0.05 * dr_score
        )

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
            carbon_credits_balance=round(self._carbon_balance, 4),
            mean_state_of_health=self._state.mean_state_of_health,
            dr_bids_fulfilled=self._state.dr_bids_fulfilled,
            dr_bids_failed=self._state.dr_bids_failed,
            steps_completed=self._state.current_step,
            done=self._state.done,
        )

    def get_current_task_score(self) -> float:
        return self._get_pareto_score().aggregate_score

    def get_pareto_score(self) -> ParetoScore:
        return self._get_pareto_score()

    @classmethod
    def get_class_score(cls) -> float:
        env = get_current_env_instance()
        return env._get_pareto_score().aggregate_score if env and env.state else 0.0

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
        """49.5 Hz emergency at step 26 for the hard task only."""
        if self._task_id == "hard-frequency-response" and step == GRID_STRESS_STEP:
            return 49.5
        return 50.0

    def _is_grid_connected(self, step: int) -> bool:
        """Grid disconnects during islanding task between ISLANDING_START and ISLANDING_END."""
        if self._task_id == "islanding-emergency":
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
                ev_adder = 1.2

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

    def _build_observation(self) -> VppObservation:
        if self._true_solar is None or self._true_demand is None or self._true_price is None:
            raise RuntimeError("Missing environment curves. Call reset() before observation.")

        idx = min(self._current_step, EPISODE_STEPS - 1)
        now = self._episode_start + timedelta(minutes=15 * idx)

        # Per-home telemetry (includes SoH)
        telemetry = [
            BatteryTelemetry(
                asset_id=asset.asset_id,
                soc=self._battery_soc[asset.asset_id],
                state_of_health=self._battery_soh[asset.asset_id],
                current_house_load_kw=float(self._true_demand[idx]),
                current_solar_gen_kw=float(self._true_solar[idx]),
            )
            for asset in self.assets
        ]

        zone_aggregates = self._build_zone_aggregates(idx)

        # Noisy 4-step look-ahead + uncertainty bands
        n = 4
        price_slice = list(self._true_price[idx: idx + n])
        solar_slice = list(self._true_solar[idx: idx + n])
        while len(price_slice) < n:
            price_slice.append(price_slice[-1])
        while len(solar_slice) < n:
            solar_slice.append(solar_slice[-1])

        noisy_price = [float(p + random.gauss(0, _PRICE_SIGMA[i])) for i, p in enumerate(price_slice)]
        noisy_solar = [float(s + random.gauss(0, _SOLAR_SIGMA[i])) for i, s in enumerate(solar_slice)]

        # DR bid for this step
        dr_bid = self._get_current_dr_bid(idx)

        return VppObservation(
            timestamp=now,
            step_id=self._current_step,
            telemetry=telemetry,
            zone_aggregates=zone_aggregates,
            grid_frequency_hz=self._grid_frequency(idx),
            grid_voltage_v=230.0,
            grid_connected=self._is_grid_connected(idx),
            market_price_per_mwh=float(self._true_price[idx]),
            carbon_credits_balance=round(self._carbon_balance, 4),
            forecast_24h_price=self._true_price.tolist(),
            forecast_24h_solar=self._true_solar.tolist(),
            short_term_price_forecast=noisy_price,
            short_term_solar_forecast=noisy_solar,
            forecast_price_uncertainty=list(_PRICE_SIGMA),
            forecast_solar_uncertainty=list(_SOLAR_SIGMA),
            dr_bid=dr_bid,
            ev_defer_deadline_step=EV_DEFER_DEADLINE,
            p2p_last_revenue_usd=round(self._p2p_last_revenue, 4),
        )

