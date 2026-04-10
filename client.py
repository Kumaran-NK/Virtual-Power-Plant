# client.py
"""VPP Environment HTTP client (OpenEnv EnvClient wrapper)."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from models import (
    BatteryTelemetry,
    DRBid,
    VppAction,
    VppObservation,
    VppState,
    ZoneTelemetry,
)


class VppEnv(EnvClient[VppAction, VppObservation, VppState]):
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
        payload = {
            "global_charge_rate": action.global_charge_rate,
            "min_reserve_pct": action.min_reserve_pct,
            "defer_ev_charging": action.defer_ev_charging,
            "accept_dr_bid": action.accept_dr_bid,
            "p2p_export_rate": action.p2p_export_rate,
        }
        if action.reasoning is not None:
            payload["reasoning"] = action.reasoning
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[VppObservation]:
        obs_data = payload.get("observation", {})
        metadata = obs_data.get("metadata") or payload.get("metadata") or payload.get("info") or {}
        telemetry = [BatteryTelemetry(**t) for t in obs_data.get("telemetry", [])]
        zone_aggregates = [ZoneTelemetry(**z) for z in obs_data.get("zone_aggregates", [])]
        dr_bid_data = obs_data.get("dr_bid") or {}
        dr_bid = DRBid(**dr_bid_data)

        observation = VppObservation(
            timestamp=obs_data.get("timestamp"),
            step_id=obs_data.get("step_id", 0),
            telemetry=telemetry,
            zone_aggregates=zone_aggregates,
            grid_frequency_hz=obs_data.get("grid_frequency_hz", 50.0),
            grid_voltage_v=obs_data.get("grid_voltage_v", 230.0),
            grid_connected=obs_data.get("grid_connected", True),
            market_price_per_mwh=obs_data.get("market_price_per_mwh", 0.0),
            carbon_credits_balance=obs_data.get("carbon_credits_balance", 0.0),
            forecast_24h_price=obs_data.get("forecast_24h_price", []),
            forecast_24h_solar=obs_data.get("forecast_24h_solar", []),
            short_term_price_forecast=obs_data.get("short_term_price_forecast", []),
            short_term_solar_forecast=obs_data.get("short_term_solar_forecast", []),
            forecast_price_uncertainty=obs_data.get("forecast_price_uncertainty", []),
            forecast_solar_uncertainty=obs_data.get("forecast_solar_uncertainty", []),
            dr_bid=dr_bid,
            ev_defer_deadline_step=obs_data.get("ev_defer_deadline_step", 40),
            p2p_last_revenue_usd=obs_data.get("p2p_last_revenue_usd", 0.0),
            done=payload.get("done", obs_data.get("done", False)),
            reward=payload.get("reward", obs_data.get("reward")),
            metadata=metadata,
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> VppState:
        # Prefer full typed parsing when the server provides extended ground-truth state.
        required = {"current_step", "task_tier", "actual_weather_mode", "battery_true_soc"}
        if required.issubset(payload.keys()):
            return VppState(**payload)

        # Backward-compatible fallback for partial state payloads.
        return VppState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            current_step=payload.get("current_step", payload.get("step_count", 0)),
            task_tier=payload.get("task_tier", "unknown"),
            done=payload.get("done", False),
            cumulative_revenue_usd=payload.get("cumulative_revenue_usd", 0.0),
            cumulative_cost_usd=payload.get("cumulative_cost_usd", 0.0),
            cumulative_profit_usd=payload.get("cumulative_profit_usd", 0.0),
            cumulative_p2p_usd=payload.get("cumulative_p2p_usd", 0.0),
            cumulative_dr_bonus_usd=payload.get("cumulative_dr_bonus_usd", 0.0),
            cumulative_dr_penalty_usd=payload.get("cumulative_dr_penalty_usd", 0.0),
            carbon_credits_balance=payload.get("carbon_credits_balance", 0.0),
            carbon_credits_earned=payload.get("carbon_credits_earned", 0.0),
            carbon_credits_spent=payload.get("carbon_credits_spent", 0.0),
            blackout_events_count=payload.get("blackout_events_count", 0),
            safety_violations_count=payload.get("safety_violations_count", 0),
            grid_emergencies_ignored=payload.get("grid_emergencies_ignored", 0),
            islanding_blackouts=payload.get("islanding_blackouts", 0),
            dr_bids_accepted=payload.get("dr_bids_accepted", 0),
            dr_bids_fulfilled=payload.get("dr_bids_fulfilled", 0),
            dr_bids_failed=payload.get("dr_bids_failed", 0),
            ev_defer_debt_kwh=payload.get("ev_defer_debt_kwh", 0.0),
            mean_state_of_health=payload.get("mean_state_of_health", 1.0),
            min_state_of_health=payload.get("min_state_of_health", 1.0),
            total_cycle_count=payload.get("total_cycle_count", 0.0),
            actual_weather_mode=payload.get("actual_weather_mode", "unknown"),
            battery_true_soc=payload.get("battery_true_soc", {}),
            battery_true_soh=payload.get("battery_true_soh", {}),
            grid_connected=payload.get("grid_connected", True),
            islanding_start_step=payload.get("islanding_start_step"),
            islanding_end_step=payload.get("islanding_end_step"),
            dr_active=payload.get("dr_active", False),
            dr_committed_until=payload.get("dr_committed_until", 0),
            dr_committed_power_kw=payload.get("dr_committed_power_kw", 0.0),
            dr_premium_multiplier=payload.get("dr_premium_multiplier", 1.0),
        )