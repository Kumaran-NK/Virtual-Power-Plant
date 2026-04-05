# gymwrapper.py
"""
Gymnasium wrapper for the VPP environment.

Design decisions
────────────────
• Pure HTTP (requests.Session) — no WebSocket / OpenEnv EnvClient.
  Direct HTTP to /reset and /step is simpler and avoids auth issues.

• OBS_DIM = 24 includes zone-level aggregate features.

Observation vector (24 floats, all normalised ≈ [−1, 1])
──────────────────────────────────────────────────────────
  mean_soc, min_soc, max_soc, std_soc       battery aggregate   4
  mean_solar_norm, mean_demand_norm          generation          2
  step_norm  (0 → 1 over 48 steps)          temporal            1
  freq_norm, volt_norm                       grid state          2
  price_norm                                 market price        1
  price_forecast[0:4] norm                  short-term fc        4
  solar_forecast[0:4] norm                  short-term fc        4
  zone_a_mean_soc, zone_a_min_soc           Zone A aggregate    2
  zone_b_mean_soc, zone_b_min_soc           Zone B aggregate    2
  zone_b_ev_demand_norm                     EV demand signal    1
  zone_b_demand_delta_norm                  EV demand diff      1
                                                         total  24
"""

from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import requests
from gymnasium import spaces

# ── Normalisation constants ───────────────────────────────────────────────────
_PRICE_MAX  = 600.0    # hard-task spike ~500 $/MWh; leave headroom
_SOLAR_MAX  = 8.0      # easy-task peak  ~6.0 kW;   leave headroom
_DEMAND_MAX = 6.0
_FREQ_RANGE = 0.5      # ±0.5 Hz from 50.0
_VOLT_RANGE = 20.0     # ±20 V  from 230.0
_EV_DEMAND_MAX = 2.5   # max EV adder kW/home

# 48-step episode (step 0–47)
_MAX_STEP   = 47

OBS_DIM = 24    # must match feature list in _flatten()


class VppGymEnv(gym.Env):
    """
    Single-agent Gymnasium env wrapping the VPP HTTP server.

    Args:
        base_url: Running FastAPI server URL.
        task_id:  'easy-arbitrage' | 'medium-forecast-error' |
                  'hard-frequency-response'
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        base_url: str = "http://localhost:7860",
        task_id: str = "easy-arbitrage",
    ):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.task_id  = task_id
        self._session = requests.Session()

        # action: [charge_rate ∈ [-1,1], reserve_pct ∈ [0,1]]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([ 1.0, 1.0], dtype=np.float32),
        )

        # Fixed-size normalised observation
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        super().reset(seed=seed)
        resp = self._session.post(
            f"{self.base_url}/reset",
            params={"task_id": self.task_id},
            timeout=15,
        )
        resp.raise_for_status()
        return self._flatten(resp.json()), {}

    def step(self, action: np.ndarray):
        resp = self._session.post(
            f"{self.base_url}/step",
            json={
                "global_charge_rate": float(np.clip(action[0], -1.0,  1.0)),
                "min_reserve_pct":    float(np.clip(action[1],  0.0,  1.0)),
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
        """
        Convert a raw JSON observation dict to a float32 feature vector.
        All values are normalised so the neural network sees similar scales.
        """
        telemetry = obs.get("telemetry", [])

        if telemetry:
            soc_arr    = np.array([t["soc"]                   for t in telemetry], dtype=np.float32)
            solar_arr  = np.array([t["current_solar_gen_kw"]  for t in telemetry], dtype=np.float32)
            demand_arr = np.array([t["current_house_load_kw"] for t in telemetry], dtype=np.float32)
        else:
            soc_arr = solar_arr = demand_arr = np.array([0.5], dtype=np.float32)

        # ── Battery aggregate (4) ─────────────────────────────────────────
        mean_soc = float(np.mean(soc_arr))
        min_soc  = float(np.min(soc_arr))
        max_soc  = float(np.max(soc_arr))
        std_soc  = float(np.std(soc_arr))

        # ── Generation / consumption (2) ──────────────────────────────────
        mean_solar  = float(np.mean(solar_arr))  / _SOLAR_MAX
        mean_demand = float(np.mean(demand_arr)) / _DEMAND_MAX

        # ── Temporal progress (1) ─────────────────────────────────────────
        step_norm = obs.get("step_id", 0) / _MAX_STEP

        # ── Grid state (2) ────────────────────────────────────────────────
        freq_norm = (obs.get("grid_frequency_hz", 50.0)  - 50.0)  / _FREQ_RANGE
        volt_norm = (obs.get("grid_voltage_v",   230.0) - 230.0) / _VOLT_RANGE

        # ── Market price (1) ──────────────────────────────────────────────
        price_norm = obs.get("market_price_per_mwh", 50.0) / _PRICE_MAX

        # ── Short-term forecasts (4 + 4) ──────────────────────────────────
        raw_pfc = (
            obs.get("short_term_price_forecast")
            or obs.get("forecast_24h_price", [50.0] * 4)
        )
        raw_sfc = (
            obs.get("short_term_solar_forecast")
            or obs.get("forecast_24h_solar", [0.0] * 4)
        )
        pfc = (list(raw_pfc) + [raw_pfc[-1]] * 4)[:4]
        sfc = (list(raw_sfc) + [raw_sfc[-1]] * 4)[:4]

        price_fc = [p / _PRICE_MAX for p in pfc]
        solar_fc = [s / _SOLAR_MAX for s in sfc]

        # ── Zone aggregates (6) ───────────────────────────────────────────
        zone_aggs   = obs.get("zone_aggregates", [])
        zone_a_dict = next((z for z in zone_aggs if z.get("zone_id") == "zone-a"), {})
        zone_b_dict = next((z for z in zone_aggs if z.get("zone_id") == "zone-b"), {})

        zone_a_mean_soc = float(zone_a_dict.get("mean_soc", mean_soc))
        zone_a_min_soc  = float(zone_a_dict.get("min_soc",  min_soc))
        zone_b_mean_soc = float(zone_b_dict.get("mean_soc", mean_soc))
        zone_b_min_soc  = float(zone_b_dict.get("min_soc",  min_soc))

        # EV demand signal: zone_b demand − zone_a demand (extra EV load)
        zone_a_demand = float(zone_a_dict.get("mean_demand_kw", 0.0))
        zone_b_demand = float(zone_b_dict.get("mean_demand_kw", 0.0))
        ev_demand_norm    = zone_b_demand / max(_EV_DEMAND_MAX, 1.0)
        demand_delta_norm = (zone_b_demand - zone_a_demand) / max(_EV_DEMAND_MAX, 1.0)

        features = [
            mean_soc, min_soc, max_soc, std_soc,   # 4
            mean_solar, mean_demand,               # 2
            step_norm,                             # 1
            freq_norm, volt_norm,                  # 2
            price_norm,                            # 1
            *price_fc,                             # 4
            *solar_fc,                             # 4
            zone_a_mean_soc, zone_a_min_soc,       # 2
            zone_b_mean_soc, zone_b_min_soc,       # 2
            ev_demand_norm,                        # 1
            demand_delta_norm,                     # 1
        ]                                          # = 24

        assert len(features) == OBS_DIM, \
            f"OBS_DIM mismatch: got {len(features)}, expected {OBS_DIM}"
        return np.array(features, dtype=np.float32)