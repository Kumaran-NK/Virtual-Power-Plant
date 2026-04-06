# gymwrapper.py
"""
Gymnasium wrapper for the VPP environment — Extended Edition.

Observation vector (34 floats, all normalised ≈ [−1, 1])
──────────────────────────────────────────────────────────
  mean_soc, min_soc, max_soc, std_soc          battery aggregate   4
  mean_soh, min_soh                            SoH (degradation)   2
  mean_solar_norm, mean_demand_norm            generation          2
  step_norm                                    temporal            1
  freq_norm, volt_norm                         grid state          2
  grid_connected                               islanding signal    1
  price_norm                                   market price        1
  carbon_balance_norm                          carbon credits      1
  price_forecast[0:4] norm                     short-term fc       4
  solar_forecast[0:4] norm                     short-term fc       4
  price_uncertainty[0:4] norm                  uncertainty bands   4
  solar_uncertainty[0:4] norm                  uncertainty bands   4
  zone_a_mean_soc, zone_a_min_soc              Zone A aggregate    2
  zone_b_mean_soc, zone_b_min_soc, zone_b_p2p Zone B + P2P        3
  dr_active, dr_premium_norm, dr_remain_norm   DR bid state        3
  ev_defer_debt_norm                           EV deferral debt    1
                                                           total  39

Action space (5 floats):
  [charge_rate, reserve_pct, defer_ev, p2p_rate, accept_dr_continuous]
  accept_dr_continuous > 0.5 → accept_dr_bid=True in the real env
"""

from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import requests
from gymnasium import spaces

# ── Normalisation constants ────────────────────────────────────────────────────
_PRICE_MAX    = 600.0
_SOLAR_MAX    = 8.0
_DEMAND_MAX   = 6.0
_FREQ_RANGE   = 0.5
_VOLT_RANGE   = 20.0
_EV_DEBT_MAX  = 200.0   # kWh (100 homes × 1.2 kW × 0.25h × 16 steps, worst case)
_CARBON_MAX   = 30.0    # credits (generous upper bound for normalisation)
_P2P_KW_MAX   = 5.0
_PRICE_U_MAX  = 10.0    # max price uncertainty sigma
_SOLAR_U_MAX  = 1.0     # max solar uncertainty sigma
_MAX_STEP     = 47

OBS_DIM = 39
ACT_DIM = 5


class VppGymEnv(gym.Env):
    """
    Single-agent Gymnasium env wrapping the VPP HTTP server — Extended Edition.

    Action space:
        [0] global_charge_rate  ∈ [-1, 1]
        [1] min_reserve_pct     ∈ [ 0, 1]
        [2] defer_ev_charging   ∈ [ 0, 1]
        [3] p2p_export_rate     ∈ [ 0, 1]
        [4] accept_dr_bid_cont  ∈ [ 0, 1]  (> 0.5 → accept)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        base_url: str = "http://localhost:7860",
        task_id:  str = "easy-arbitrage",
    ):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.task_id  = task_id
        self._session = requests.Session()

        self.action_space = spaces.Box(
            low  = np.array([-1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high = np.array([ 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        resp = self._session.post(
            f"{self.base_url}/reset",
            params={"task_id": self.task_id},
            timeout=15,
        )
        resp.raise_for_status()
        return self._flatten(resp.json()), {}

    def step(self, action: np.ndarray):
        charge_rate  = float(np.clip(action[0], -1.0, 1.0))
        reserve_pct  = float(np.clip(action[1],  0.0, 1.0))
        defer_ev     = float(np.clip(action[2],  0.0, 1.0))
        p2p_rate     = float(np.clip(action[3],  0.0, 1.0))
        accept_dr    = bool(action[4] > 0.5)

        resp = self._session.post(
            f"{self.base_url}/step",
            json={
                "global_charge_rate": charge_rate,
                "min_reserve_pct":    reserve_pct,
                "defer_ev_charging":  defer_ev,
                "p2p_export_rate":    p2p_rate,
                "accept_dr_bid":      accept_dr,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return (
            self._flatten(data["observation"]),
            float(data["reward"]),
            bool(data["done"]),
            False,
            data.get("info", {}),
        )

    def close(self):
        self._session.close()

    # ── Feature engineering ───────────────────────────────────────────────────

    def _flatten(self, obs: dict) -> np.ndarray:
        telemetry = obs.get("telemetry", [])

        if telemetry:
            soc_arr    = np.array([t["soc"]                                for t in telemetry], dtype=np.float32)
            soh_arr    = np.array([t.get("state_of_health", 1.0)           for t in telemetry], dtype=np.float32)
            solar_arr  = np.array([t["current_solar_gen_kw"]               for t in telemetry], dtype=np.float32)
            demand_arr = np.array([t["current_house_load_kw"]              for t in telemetry], dtype=np.float32)
        else:
            soc_arr = np.array([0.5], dtype=np.float32)
            soh_arr = np.array([1.0], dtype=np.float32)
            solar_arr = demand_arr = np.array([0.0], dtype=np.float32)

        # Battery aggregate (4)
        mean_soc = float(np.mean(soc_arr))
        min_soc  = float(np.min(soc_arr))
        max_soc  = float(np.max(soc_arr))
        std_soc  = float(np.std(soc_arr))

        # SoH (2)
        mean_soh = float(np.mean(soh_arr))
        min_soh  = float(np.min(soh_arr))

        # Generation / consumption (2)
        mean_solar  = float(np.mean(solar_arr))  / _SOLAR_MAX
        mean_demand = float(np.mean(demand_arr)) / _DEMAND_MAX

        # Temporal (1)
        step_norm = obs.get("step_id", 0) / _MAX_STEP

        # Grid state (3: freq, volt, connected)
        freq_norm    = (obs.get("grid_frequency_hz", 50.0) - 50.0) / _FREQ_RANGE
        volt_norm    = (obs.get("grid_voltage_v", 230.0) - 230.0) / _VOLT_RANGE
        grid_conn    = 1.0 if obs.get("grid_connected", True) else 0.0

        # Market price (1)
        price_norm = obs.get("market_price_per_mwh", 50.0) / _PRICE_MAX

        # Carbon balance (1)
        carbon_norm = obs.get("carbon_credits_balance", 0.0) / _CARBON_MAX

        # Short-term forecasts + uncertainty (4+4+4+4 = 16)
        raw_pfc = obs.get("short_term_price_forecast") or obs.get("forecast_24h_price", [50.0]*4)
        raw_sfc = obs.get("short_term_solar_forecast")  or obs.get("forecast_24h_solar", [0.0]*4)
        raw_pu  = obs.get("forecast_price_uncertainty", [2.5, 3.5, 4.5, 5.5])
        raw_su  = obs.get("forecast_solar_uncertainty",  [0.25, 0.35, 0.50, 0.70])

        pfc = (list(raw_pfc) + [raw_pfc[-1]] * 4)[:4]
        sfc = (list(raw_sfc) + [raw_sfc[-1]] * 4)[:4]
        pu  = (list(raw_pu)  + [raw_pu[-1]]  * 4)[:4]
        su  = (list(raw_su)  + [raw_su[-1]]  * 4)[:4]

        price_fc = [p / _PRICE_MAX  for p in pfc]
        solar_fc = [s / _SOLAR_MAX  for s in sfc]
        price_u  = [u / _PRICE_U_MAX for u in pu]
        solar_u  = [u / _SOLAR_U_MAX for u in su]

        # Zone aggregates (5)
        zone_aggs   = obs.get("zone_aggregates", [])
        zone_a_dict = next((z for z in zone_aggs if z.get("zone_id") == "zone-a"), {})
        zone_b_dict = next((z for z in zone_aggs if z.get("zone_id") == "zone-b"), {})

        zone_a_mean_soc = float(zone_a_dict.get("mean_soc", mean_soc))
        zone_a_min_soc  = float(zone_a_dict.get("min_soc",  min_soc))
        zone_b_mean_soc = float(zone_b_dict.get("mean_soc", mean_soc))
        zone_b_min_soc  = float(zone_b_dict.get("min_soc",  min_soc))
        zone_b_p2p_norm = float(zone_b_dict.get("p2p_available_kw", 0.0)) / _P2P_KW_MAX

        # DR bid state (3)
        dr_bid      = obs.get("dr_bid", {})
        dr_active   = 1.0 if dr_bid.get("active", False) else 0.0
        dr_premium  = float(dr_bid.get("premium_multiplier", 1.0)) / 3.0   # max premium = 3×
        dr_remain   = float(dr_bid.get("steps_remaining", 0)) / _MAX_STEP

        # EV deferral debt (1)
        ev_debt_norm = obs.get("ev_defer_deadline_step", 40) / _MAX_STEP   # proxy; actual debt in state

        features = [
            mean_soc, min_soc, max_soc, std_soc,    # 4
            mean_soh, min_soh,                       # 2
            mean_solar, mean_demand,                 # 2
            step_norm,                               # 1
            freq_norm, volt_norm, grid_conn,         # 3
            price_norm,                              # 1
            carbon_norm,                             # 1
            *price_fc,                               # 4
            *solar_fc,                               # 4
            *price_u,                                # 4
            *solar_u,                                # 4
            zone_a_mean_soc, zone_a_min_soc,         # 2
            zone_b_mean_soc, zone_b_min_soc, zone_b_p2p_norm,  # 3
            dr_active, dr_premium, dr_remain,        # 3
            ev_debt_norm,                            # 1
        ]                                            # = 39

        assert len(features) == OBS_DIM, \
            f"OBS_DIM mismatch: got {len(features)}, expected {OBS_DIM}"
        return np.array(features, dtype=np.float32)