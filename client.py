# client.py
"""VPP Environment HTTP client (OpenEnv EnvClient wrapper)."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import VppAction, VppObservation, BatteryTelemetry


class VppEnv(EnvClient[VppAction, VppObservation, State]):
    """
    HTTP client for the VPP Environment server.

    Maintains a persistent connection and provides typed step()/reset().

    Example:
        with VppEnv(base_url="http://localhost:7860") as client:
            result = client.reset(task_id="easy-arbitrage")
            obs = result.observation
            print(obs.market_price_per_mwh)

            result = client.step(VppAction(global_charge_rate=-0.5, min_reserve_pct=0.2))
            print(result.reward, result.done)
    """

    def _step_payload(self, action: VppAction) -> Dict:
        return {
            "global_charge_rate": action.global_charge_rate,
            "min_reserve_pct": action.min_reserve_pct,
        }

    def _parse_result(self, payload: Dict) -> StepResult[VppObservation]:
        obs_data = payload.get("observation", {})
        telemetry = [BatteryTelemetry(**t) for t in obs_data.get("telemetry", [])]

        observation = VppObservation(
            timestamp=obs_data.get("timestamp"),
            step_id=obs_data.get("step_id", 0),
            telemetry=telemetry,
            grid_frequency_hz=obs_data.get("grid_frequency_hz", 50.0),
            grid_voltage_v=obs_data.get("grid_voltage_v", 230.0),
            market_price_per_mwh=obs_data.get("market_price_per_mwh", 0.0),
            forecast_24h_price=obs_data.get("forecast_24h_price", []),
            forecast_24h_solar=obs_data.get("forecast_24h_solar", []),
            short_term_price_forecast=obs_data.get("short_term_price_forecast", []),
            short_term_solar_forecast=obs_data.get("short_term_solar_forecast", []),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )