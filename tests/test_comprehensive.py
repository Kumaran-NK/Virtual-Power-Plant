"""Comprehensive pytest tests for core VPP environment behavior.

These tests run directly against VppEnvironment and do not require
an external HTTP/WebSocket server.
"""

import numpy as np
import pytest

from models import VppAction
from server.task_curves import ISLANDING_END, price_curve
from server.vpp_environment import VppEnvironment


def _action(
    global_charge_rate: float,
    min_reserve_pct: float,
    accept_dr_bid: bool = False,
) -> VppAction:
    return VppAction(
        global_charge_rate=global_charge_rate,
        min_reserve_pct=min_reserve_pct,
        defer_ev_charging=0.0,
        accept_dr_bid=accept_dr_bid,
        p2p_export_rate=0.0,
        reasoning=None,
    )


def test_full_episode_progresses_steps():
    env = VppEnvironment()
    obs = env.reset(seed=123, task_id="easy-arbitrage")

    assert obs.step_id == 0
    assert obs.done is False

    for step_num in range(1, 6):
        obs = env.step(_action(global_charge_rate=-0.5 if step_num % 2 == 0 else 0.5, min_reserve_pct=0.2))

    assert env.state.current_step == 5
    assert env.state.step_count == 5
    assert isinstance(obs.reward, float)


def test_islanding_disconnect_window_activates():
    env = VppEnvironment()
    obs = env.reset(seed=123, task_id="islanding-emergency")

    while obs.step_id < 20:
        obs = env.step(_action(global_charge_rate=0.0, min_reserve_pct=0.2))

    assert obs.step_id == 20
    assert obs.grid_connected is False


def test_carbon_spend_scaled_to_fleet():
    env = VppEnvironment()
    env.reset(seed=123, task_id="easy-arbitrage")
    env.step(_action(global_charge_rate=1.0, min_reserve_pct=0.2))

    # 5 kW/home * 0.25 h * 100 homes * 0.08 credits/kWh = 10.0 credits
    assert env.state.carbon_credits_spent == pytest.approx(10.0, rel=1e-6)


def test_degradation_increases_without_grid_dispatch_when_soc_changes():
    env = VppEnvironment()
    env.reset(seed=123, task_id="medium-forecast-error")
    start_soc = env.state.battery_true_soc["home-000"]

    for _ in range(10):
        env.step(_action(global_charge_rate=0.0, min_reserve_pct=0.2))

    end_soc = env.state.battery_true_soc["home-000"]
    assert end_soc != pytest.approx(start_soc)
    assert env.state.total_cycle_count > 0.0
    assert env.state.mean_state_of_health < 1.0


def test_dr_commitment_stops_penalty_after_window_closes():
    env = VppEnvironment()
    env.reset(seed=123, task_id="expert-demand-response")

    penalty_values = []
    for i in range(5):
        env.step(_action(global_charge_rate=0.0, min_reserve_pct=0.2, accept_dr_bid=(i == 0)))
        penalty_values.append(env.state.cumulative_dr_penalty_usd)

    assert env.state.dr_bids_failed >= 1
    assert env.state.dr_active is False
    # Once window closes, penalty should stop increasing.
    assert penalty_values[3] == pytest.approx(penalty_values[2], rel=1e-9)
    assert penalty_values[4] == pytest.approx(penalty_values[3], rel=1e-9)


def test_step_observation_exposes_pareto_score_top_level():
    env = VppEnvironment()
    env.reset(seed=123, task_id="easy-arbitrage")
    obs = env.step(_action(global_charge_rate=0.0, min_reserve_pct=0.2))

    assert isinstance(obs.pareto_score, dict)
    assert "aggregate_score" in obs.pareto_score
    aggregate_score = obs.pareto_score.get("aggregate_score")
    assert isinstance(aggregate_score, (int, float))
    assert 0.0 < float(aggregate_score) < 1.0


def test_empty_battery_cannot_create_discharge_revenue():
    env = VppEnvironment()
    env.reset(seed=123, task_id="hard-frequency-response")
    env._battery_soc = {asset.asset_id: 0.0 for asset in env.assets}

    obs = env.step(_action(global_charge_rate=-1.0, min_reserve_pct=0.0))
    assert obs.reward is not None
    assert float(obs.reward) <= 0.0
    assert env.state.cumulative_profit_usd <= 0.0


def test_easy_price_curve_has_day_night_spread():
    curve = price_curve("easy-arbitrage")
    assert min(curve) < max(curve)
    assert curve[4] < curve[36]  # daytime cheaper than evening peak


def test_medium_task_exposes_forecast_realtime_solar_error():
    env = VppEnvironment()
    obs = env.reset(seed=123, task_id="medium-forecast-error")

    for _ in range(20):
        obs = env.step(_action(global_charge_rate=0.0, min_reserve_pct=0.2))

    assert abs(obs.forecast_realtime_error_solar_kw) > 0.01


def test_hard_task_emergency_latency_is_observable():
    env = VppEnvironment()
    obs = env.reset(seed=123, task_id="hard-frequency-response")

    for _ in range(26):
        obs = env.step(_action(global_charge_rate=0.0, min_reserve_pct=0.2))

    obs = env.step(_action(global_charge_rate=-1.0, min_reserve_pct=0.2))

    assert obs.emergency_active is True
    assert obs.response_latency_steps_to_emergency >= 0


def test_dr_penalty_escalates_across_consecutive_failed_windows():
    env = VppEnvironment()
    env.reset(seed=123, task_id="expert-demand-response")

    env._true_price = np.full_like(env._true_price, 50.0)
    env._dr_schedule = {
        0: (2.0, 3.0, 1),
        2: (2.0, 3.0, 1),
    }

    env.step(_action(global_charge_rate=0.0, min_reserve_pct=0.2, accept_dr_bid=True))
    first_window_penalty = env.state.cumulative_dr_penalty_usd
    assert env.state.dr_consecutive_failures == 1

    env.step(_action(global_charge_rate=0.0, min_reserve_pct=0.2, accept_dr_bid=False))
    penalty_before_second = env.state.cumulative_dr_penalty_usd

    env.step(_action(global_charge_rate=0.0, min_reserve_pct=0.2, accept_dr_bid=True))
    second_window_penalty = env.state.cumulative_dr_penalty_usd - penalty_before_second

    assert second_window_penalty > first_window_penalty
    assert env.state.dr_consecutive_failures == 2


def test_dr_failure_streak_resets_after_fulfilled_window():
    env = VppEnvironment()
    env.reset(seed=123, task_id="expert-demand-response")

    env._true_price = np.full_like(env._true_price, 50.0)
    env._dr_schedule = {
        0: (2.0, 3.0, 1),
        2: (2.0, 1.0, 1),
    }

    env.step(_action(global_charge_rate=0.0, min_reserve_pct=0.2, accept_dr_bid=True))
    assert env.state.dr_consecutive_failures == 1

    env.step(_action(global_charge_rate=0.0, min_reserve_pct=0.2, accept_dr_bid=False))

    env.step(_action(global_charge_rate=-1.0, min_reserve_pct=0.2, accept_dr_bid=True))

    assert env.state.dr_bids_failed == 1
    assert env.state.dr_bids_fulfilled == 1
    assert env.state.dr_consecutive_failures == 0
    assert env.state.dr_fulfillment_ratio_sum == pytest.approx(1.0, rel=1e-6)


def test_islanding_critical_load_metrics_and_reconnection_soft_sync():
    env = VppEnvironment()
    obs = env.reset(seed=123, task_id="islanding-emergency")

    env._true_solar = np.full_like(env._true_solar, 0.6)
    env._true_demand = np.full_like(env._true_demand, 1.0)
    env._true_price = np.full_like(env._true_price, 50.0)

    while obs.step_id < 20:
        obs = env.step(_action(global_charge_rate=0.0, min_reserve_pct=0.2, accept_dr_bid=False))

    env._battery_soc = {asset.asset_id: 0.1 for asset in env.assets}

    obs = env.step(_action(global_charge_rate=0.0, min_reserve_pct=0.1, accept_dr_bid=False))
    assert obs.grid_connected is False
    assert obs.demand_shed_this_step_kwh > 0.0
    assert env.state.islanding_blackout_home_steps == 0
    assert env.state.islanding_blackout_unique_homes == 0

    env._true_solar = np.zeros_like(env._true_solar)

    env.step(_action(global_charge_rate=0.0, min_reserve_pct=0.1, accept_dr_bid=False))
    assert env.state.islanding_blackout_home_steps == 100
    assert env.state.islanding_blackouts == 100
    assert env.state.islanding_blackout_unique_homes == 100

    env.step(_action(global_charge_rate=0.0, min_reserve_pct=0.1, accept_dr_bid=False))
    assert env.state.islanding_blackout_home_steps == 200
    assert env.state.islanding_blackout_unique_homes == 100

    while env.state.current_step < ISLANDING_END:
        env.step(_action(global_charge_rate=0.0, min_reserve_pct=0.1, accept_dr_bid=False))

    env._battery_soc = {asset.asset_id: 0.9 for asset in env.assets}
    obs = env.step(_action(global_charge_rate=-1.0, min_reserve_pct=0.1, accept_dr_bid=False))
    metadata = obs.metadata or {}

    assert metadata.get("reconnection_soft_sync_active") is True
    assert metadata.get("reconnection_soft_sync_applied") is True
    applied_charge_kw = metadata.get("applied_charge_kw_per_home")
    assert isinstance(applied_charge_kw, (int, float))
    assert float(applied_charge_kw) == pytest.approx(-2.5, rel=1e-6)
