# server/vpp_environment.py
"""
VPP Environment — core simulation.

48-step (12-hour) episodes: 06:00 → 17:45, one step = 15 minutes.

Key design decisions
────────────────────
1. Violation counting: one flag per STEP, not per home.
   → max 48 violations per episode, grader penalty stays meaningful.

2. _build_observation idx cap: min(step, 47), not min(step, 95).

3. Grid stress step: 26 (12:30 in 12-hour window).

4. Grader profit targets scaled to 12-hour achievable range.

5. No early termination on violation — full 48 steps always run.

6. Zone aggregates: homes split into Zone A (no EV, idx 0–39) and
   Zone B (with EV, idx 40–99) — exposed in observation.
"""

import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np

from openenv.core.env_server.interfaces import Environment
from models import (
    BatteryAsset, BatteryTelemetry, ZoneTelemetry,
    VppAction, VppObservation, VppState,
)
from server.task_curves import (
    solar_curve, demand_curve, price_curve,
    EPISODE_STEPS, GRID_STRESS_STEP,
)

_current_instance: "VppEnvironment | None" = None

# Zone boundaries (index into the 100-home list)
_ZONE_A_START, _ZONE_A_END = 0,  40   # homes 000–039, no EV
_ZONE_B_START, _ZONE_B_END = 40, 100  # homes 040–099, with EV chargers


class VppEnvironment(Environment):
    """Simulated Virtual Power Plant — 100 home batteries, 12-hour episodes."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(self):
        global _current_instance
        _current_instance = self

        self.assets: List[BatteryAsset] = [
            BatteryAsset(
                asset_id=f"home-{i:03d}",
                capacity_kwh=13.5,
                max_power_kw=5.0,
                efficiency_rt=0.90,
            )
            for i in range(100)
        ]

        self._task_id:      str              = "easy-arbitrage"
        self._state:        VppState | None  = None
        self._battery_soc:  Dict[str, float] = {}
        self._current_step: int              = 0
        self._total_profit: float            = 0.0
        self._episode_start: datetime        = datetime.utcnow()

        self._true_solar:  np.ndarray | None = None
        self._true_demand: np.ndarray | None = None
        self._true_price:  np.ndarray | None = None

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self, task_id: str = "easy-arbitrage") -> VppObservation:
        """
        Start a new 48-step (12-hour) episode.

        Both Python and NumPy RNGs are seeded from task_id so forecast noise
        is identical across runs of the same task.
        Ground-truth curves are purely mathematical — no randomness.
        """
        self._task_id = task_id

        seed = abs(hash(task_id)) % (2 ** 31)
        random.seed(seed)
        np.random.seed(seed)

        self._true_solar  = solar_curve(task_id)
        self._true_demand = demand_curve(task_id)
        self._true_price  = price_curve(task_id)

        self._current_step  = 0
        self._total_profit  = 0.0
        self._episode_start = datetime.utcnow()

        self._battery_soc = {a.asset_id: 0.5 for a in self.assets}

        tier    = task_id.split("-")[0]
        weather = {"easy": "clear_sky", "medium": "heatwave", "hard": "partly_cloudy"}.get(
            tier, "clear_sky"
        )

        self._state = VppState(
            episode_id=f"ep_{task_id}_{seed % 10_000:04d}",
            step_count=0,
            current_step=0,
            task_tier=task_id,
            cumulative_profit_usd=0.0,
            cumulative_revenue_usd=0.0,
            cumulative_cost_usd=0.0,
            blackout_events_count=0,
            safety_violations_count=0,
            grid_emergencies_ignored=0,
            actual_weather_mode=weather,
            battery_true_soc=self._battery_soc.copy(),
            done=False,
        )

        return self._build_observation()

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, action: VppAction) -> Tuple[VppObservation, float, bool, dict]:
        """
        Advance by one 15-minute interval.

        Physics
        ───────
          charge_kw > 0  → buying from grid  (battery fills,  cost)
          charge_kw < 0  → selling to grid   (battery drains, revenue)

          effective_kw = charge_kw × η      (η = 0.90, charging leg only)
          delta_kwh    = (solar − demand + effective_kw) × 0.25 h
          new_soc      = old_soc + delta_kwh / capacity_kwh

          grid_profit  = −charge_kw × 0.25 h × (price $/MWh ÷ 1000)
                         positive when selling
        """
        if self._state is None or self._state.done:
            raise RuntimeError("Call reset() before step().")

        s         = self._current_step
        solar_kw  = float(self._true_solar[s])
        demand_kw = float(self._true_demand[s])
        price_usd = float(self._true_price[s])
        freq_hz   = self._grid_frequency(s)

        max_kw    = 5.0
        charge_kw = float(np.clip(action.global_charge_rate * max_kw, -max_kw, max_kw))

        # Grid emergency tracking
        if freq_hz < 49.8 and charge_kw >= 0:
            self._state.grid_emergencies_ignored += 1

        step_profit        = 0.0
        step_revenue       = 0.0
        step_cost          = 0.0
        step_has_violation = False
        step_has_blackout  = False

        for asset in self.assets:
            old_soc      = self._battery_soc[asset.asset_id]
            effective_kw = charge_kw * asset.efficiency_rt if charge_kw > 0 else charge_kw
            delta_kwh    = (solar_kw - demand_kw + effective_kw) * 0.25
            new_soc      = old_soc + delta_kwh / asset.capacity_kwh
            new_soc      = float(np.clip(new_soc, 0.0, 1.0))

            if new_soc < action.min_reserve_pct:
                step_has_violation = True
            if new_soc <= 0.0 and demand_kw > 0:
                step_has_blackout = True

            self._battery_soc[asset.asset_id] = new_soc

            exported_kwh = -charge_kw * 0.25
            home_profit  = exported_kwh * (price_usd / 1000.0)
            step_profit += home_profit
            if exported_kwh > 0:
                step_revenue += home_profit
            else:
                step_cost    += abs(home_profit)

        if step_has_violation:
            self._state.safety_violations_count += 1
        if step_has_blackout:
            self._state.blackout_events_count   += 1

        self._total_profit                    += step_profit
        self._state.cumulative_profit_usd      = round(self._total_profit, 4)
        self._state.cumulative_revenue_usd     = round(
            self._state.cumulative_revenue_usd + step_revenue, 4
        )
        self._state.cumulative_cost_usd        = round(
            self._state.cumulative_cost_usd + step_cost, 4
        )

        safety_penalty    = 2.0 if step_has_violation               else 0.0
        emergency_penalty = 2.0 if freq_hz < 49.8 and charge_kw >= 0 else 0.0
        reward            = step_profit - safety_penalty - emergency_penalty

        self._current_step         += 1
        done                        = self._current_step >= EPISODE_STEPS
        self._state.step_count     += 1
        self._state.current_step    = self._current_step
        self._state.battery_true_soc = self._battery_soc.copy()
        self._state.done            = done

        obs  = self._build_observation()
        info = {
            "step_profit_usd":            round(step_profit, 4),
            "safety_violation_this_step": step_has_violation,
            "blackout_this_step":         step_has_blackout,
            "grid_frequency_hz":          round(freq_hz, 3),
        }
        return obs, round(reward, 6), done, info

    # ── Grader ────────────────────────────────────────────────────────────────

    def _get_grader_score(self) -> float:
        """
        Deterministic 0.0–1.0 score.

        profit_ratio      = cumulative_profit / goal   (capped at 1.0)
        violation_penalty = (violations / 48) × 0.40  (capped at 0.40)
        emergency_penalty = ignored_emergencies × 0.10 (capped at 0.30)
        score             = max(0, profit_ratio − violation_penalty − emergency_penalty)
        """
        if self._state is None:
            return 0.0

        goals = {
            "easy-arbitrage":          250.0,
            "medium-forecast-error":   100.0,
            "hard-frequency-response": 500.0,
        }
        goal         = goals.get(self._task_id, 250.0)
        profit_ratio = float(np.clip(self._total_profit / goal, 0.0, 1.0))

        violation_ratio   = self._state.safety_violations_count / EPISODE_STEPS
        violation_penalty = min(0.40, violation_ratio * 0.40)

        emergency_penalty = min(0.30, self._state.grid_emergencies_ignored * 0.10)

        return round(max(0.0, profit_ratio - violation_penalty - emergency_penalty), 4)

    def get_current_task_score(self) -> float:
        return self._get_grader_score()

    @classmethod
    def get_class_score(cls) -> float:
        return _current_instance._get_grader_score() if _current_instance else 0.0

    # ── Property ──────────────────────────────────────────────────────────────

    @property
    def state(self) -> VppState:
        return self._state

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _grid_frequency(self, step: int) -> float:
        """49.5 Hz emergency at step 26 (12:30) for the hard task only."""
        if self._task_id == "hard-frequency-response" and step == GRID_STRESS_STEP:
            return 49.5
        return 50.0

    def _build_zone_aggregates(self, idx: int) -> List[ZoneTelemetry]:
        """
        Compute per-zone aggregate statistics for the current step.

        Zone A: homes 000–039 (no EV chargers).
        Zone B: homes 040–099 (EV chargers → higher evening demand).
        """
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

            # Zone B has an EV demand adder in the afternoon (steps 32+)
            ev_adder = 0.0
            if has_ev and idx >= 32:   # after 14:00, EVs start plugging in
                ev_adder = 1.2          # +1.2 kW per home for EV charging

            result.append(
                ZoneTelemetry(
                    zone_id=zone_id,
                    home_count=len(zone_assets),
                    mean_soc=float(np.mean(socs)),
                    min_soc=float(np.min(socs)),
                    max_soc=float(np.max(socs)),
                    mean_solar_kw=solar_kw,
                    mean_demand_kw=demand_kw + ev_adder,
                    has_ev_chargers=has_ev,
                )
            )

        return result

    def _build_observation(self) -> VppObservation:
        """Construct the typed observation for the current step."""
        # Cap at 47 — the last valid index in a 48-step episode
        idx = min(self._current_step, EPISODE_STEPS - 1)
        now = self._episode_start + timedelta(minutes=15 * idx)

        # Per-home telemetry
        telemetry = [
            BatteryTelemetry(
                asset_id=asset.asset_id,
                soc=self._battery_soc[asset.asset_id],
                current_house_load_kw=float(self._true_demand[idx]),
                current_solar_gen_kw=float(self._true_solar[idx]),
            )
            for asset in self.assets
        ]

        # Zone aggregates
        zone_aggregates = self._build_zone_aggregates(idx)

        # Noisy 4-step (60-minute) look-ahead forecast
        n           = 4
        price_slice = list(self._true_price[idx: idx + n])
        solar_slice = list(self._true_solar[idx: idx + n])
        while len(price_slice) < n:
            price_slice.append(price_slice[-1])
        while len(solar_slice) < n:
            solar_slice.append(solar_slice[-1])

        noisy_price = [float(p + random.gauss(0, 2.5))  for p in price_slice]
        noisy_solar = [float(s + random.gauss(0, 0.25)) for s in solar_slice]

        return VppObservation(
            timestamp=now,
            step_id=self._current_step,
            telemetry=telemetry,
            zone_aggregates=zone_aggregates,
            grid_frequency_hz=self._grid_frequency(idx),
            grid_voltage_v=230.0,
            market_price_per_mwh=float(self._true_price[idx]),
            forecast_24h_price=self._true_price.tolist(),
            forecast_24h_solar=self._true_solar.tolist(),
            short_term_price_forecast=noisy_price,
            short_term_solar_forecast=noisy_solar,
        )