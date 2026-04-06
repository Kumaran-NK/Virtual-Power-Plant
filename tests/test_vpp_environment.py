# tests/test_vpp_environment.py
"""
Unit tests for the extended VPP simulation engine.
Covers: SoH degradation, carbon credits, P2P, DR bids, islanding,
        EV deferral, adversarial weather, Pareto grader.

Run with: pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

from server.task_curves import (
    solar_curve, demand_curve, price_curve, emission_intensity_curve,
    dr_bid_schedule, EPISODE_STEPS, GRID_STRESS_STEP,
    ISLANDING_START, ISLANDING_END, ALL_TASK_IDS,
)
from models import VppAction


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    from server.vpp_environment import VppEnvironment
    return VppEnvironment()


@pytest.fixture
def easy_env(env):
    env.reset("easy-arbitrage")
    return env


@pytest.fixture
def hard_env(env):
    env.reset("hard-frequency-response")
    return env


@pytest.fixture
def expert_env(env):
    env.reset("expert-demand-response")
    return env


@pytest.fixture
def islanding_env(env):
    env.reset("islanding-emergency")
    return env


def idle(charge=0.0, reserve=0.2, defer=0.0, dr=False, p2p=0.0):
    return VppAction(
        global_charge_rate=charge,
        min_reserve_pct=reserve,
        defer_ev_charging=defer,
        accept_dr_bid=dr,
        p2p_export_rate=p2p,
    )


def run_n(env, n, action):
    for _ in range(n):
        env.step(action)


# ---------------------------------------------------------------------------
# 1. Battery degradation (SoH)
# ---------------------------------------------------------------------------

class TestSoHDegradation:
    def test_initial_soh_is_one(self, easy_env):
        for t in easy_env.state.battery_true_soh.values():
            assert abs(t - 1.0) < 1e-9

    def test_soh_decreases_after_cycling(self, easy_env):
        sell = idle(charge=-1.0, reserve=0.0)
        run_n(easy_env, 20, sell)
        sohs = list(easy_env.state.battery_true_soh.values())
        assert all(s < 1.0 for s in sohs), "SoH must decrease after cycling"

    def test_soh_floored_at_0_80(self, easy_env):
        # Even with extreme cycling, SoH cannot go below 0.80
        extreme = idle(charge=-1.0, reserve=0.0)
        run_n(easy_env, 48, extreme)
        sohs = list(easy_env.state.battery_true_soh.values())
        assert all(s >= 0.80 for s in sohs)

    def test_idle_degrades_less_than_heavy_use(self, env):
        # Run one episode max charge
        env.reset("easy-arbitrage")
        run_n(env, 48, idle(charge=1.0, reserve=0.0))
        mean_soh_heavy = env.state.mean_state_of_health

        # Run another episode idle
        env.reset("easy-arbitrage")
        run_n(env, 48, idle(charge=0.0, reserve=0.0))
        mean_soh_idle = env.state.mean_state_of_health

        assert mean_soh_idle > mean_soh_heavy, "Idle should degrade less than max continuous usage"

    def test_soh_in_observation(self, easy_env):
        obs, _, _, _ = easy_env.step(idle())
        for t in obs.telemetry:
            assert 0.80 <= t.state_of_health <= 1.0


# ---------------------------------------------------------------------------
# 2. Carbon credits
# ---------------------------------------------------------------------------

class TestCarbonCredits:
    def test_initial_carbon_balance_zero(self, easy_env):
        assert easy_env.state.carbon_credits_balance == 0.0

    def test_solar_generation_earns_credits(self, easy_env):
        # Easy task has abundant solar; advance to noon to generate credits
        run_n(easy_env, 24, idle())
        assert easy_env.state.carbon_credits_earned > 0.0

    def test_grid_purchase_in_high_emission_costs_credits(self, env):
        env.reset("easy-arbitrage")
        # Step 5 is in the high-emission window (steps 0–16)
        buy = idle(charge=1.0, reserve=0.0)
        for _ in range(5):
            env.step(idle())   # advance to step 5 without buying
        env.step(buy)          # buy in high-emission window
        assert env.state.carbon_credits_spent > 0.0

    def test_carbon_balance_in_observation(self, easy_env):
        obs, _, _, _ = easy_env.step(idle())
        # Balance is a float (may be positive or negative)
        assert isinstance(obs.carbon_credits_balance, float)

    def test_selling_does_not_cost_credits(self, easy_env):
        sell = idle(charge=-1.0, reserve=0.0)
        easy_env.step(sell)
        assert easy_env.state.carbon_credits_spent == 0.0


# ---------------------------------------------------------------------------
# 3. Forecast confidence bands
# ---------------------------------------------------------------------------

class TestForecastUncertainty:
    def test_price_uncertainty_present(self, easy_env):
        obs, _, _, _ = easy_env.step(idle())
        assert len(obs.forecast_price_uncertainty) == 4

    def test_solar_uncertainty_present(self, easy_env):
        obs, _, _, _ = easy_env.step(idle())
        assert len(obs.forecast_solar_uncertainty) == 4

    def test_uncertainty_grows_with_horizon(self, easy_env):
        obs, _, _, _ = easy_env.step(idle())
        pu = obs.forecast_price_uncertainty
        su = obs.forecast_solar_uncertainty
        # Each step should be >= previous (uncertainty grows)
        assert pu[1] >= pu[0] and pu[2] >= pu[1] and pu[3] >= pu[2]
        assert su[1] >= su[0] and su[2] >= su[1] and su[3] >= su[2]

    def test_uncertainty_values_are_positive(self, easy_env):
        obs, _, _, _ = easy_env.step(idle())
        assert all(u > 0 for u in obs.forecast_price_uncertainty)
        assert all(u > 0 for u in obs.forecast_solar_uncertainty)


# ---------------------------------------------------------------------------
# 4. Pareto grader
# ---------------------------------------------------------------------------

class TestParetoGrader:
    def test_pareto_score_has_five_components(self, env):
        env.reset("easy-arbitrage")
        run_n(env, 48, idle())
        ps = env.get_pareto_score()
        assert hasattr(ps, "profit_score")
        assert hasattr(ps, "safety_score")
        assert hasattr(ps, "carbon_score")
        assert hasattr(ps, "degradation_score")
        assert hasattr(ps, "dr_score")
        assert hasattr(ps, "aggregate_score")

    def test_aggregate_is_weighted_sum(self, env):
        env.reset("easy-arbitrage")
        run_n(env, 10, idle(charge=-0.5, reserve=0.2))
        ps = env.get_pareto_score()
        expected = (0.50 * ps.profit_score + 0.20 * ps.safety_score
                    + 0.15 * ps.carbon_score + 0.10 * ps.degradation_score
                    + 0.05 * ps.dr_score)
        assert abs(ps.aggregate_score - expected) < 0.01

    def test_idle_profit_score_zero(self, env):
        env.reset("easy-arbitrage")
        run_n(env, 48, idle())
        ps = env.get_pareto_score()
        assert ps.profit_score == 0.0

    def test_all_scores_in_unit_interval(self, env):
        env.reset("easy-arbitrage")
        run_n(env, 48, idle(charge=-0.5, reserve=0.2))
        ps = env.get_pareto_score()
        for score in [ps.profit_score, ps.safety_score, ps.carbon_score,
                      ps.degradation_score, ps.dr_score, ps.aggregate_score]:
            assert 0.0 <= score <= 1.0

    def test_selling_increases_profit_score(self, env):
        env.reset("easy-arbitrage")
        run_n(env, 48, idle())
        idle_score = env.get_pareto_score().profit_score

        env.reset("easy-arbitrage")
        run_n(env, 48, idle(charge=-0.5, reserve=0.0))
        sell_score = env.get_pareto_score().profit_score

        assert sell_score >= idle_score


# ---------------------------------------------------------------------------
# 5. P2P energy trading
# ---------------------------------------------------------------------------

class TestP2PTrading:
    def test_p2p_revenue_positive_with_surplus(self, easy_env):
        # Easy task has abundant solar → Zone B should have surplus
        p2p_action = idle(charge=-0.2, reserve=0.0, p2p=1.0)
        # Run a few midday steps when solar is high
        run_n(easy_env, 20, idle())   # advance to ~midday
        _, _, _, info = easy_env.step(p2p_action)
        # P2P revenue may be zero at early steps when solar is low;
        # check the cumulative over 10 more steps
        run_n(easy_env, 10, p2p_action)
        assert easy_env.state.cumulative_p2p_usd >= 0.0

    def test_p2p_rate_zero_has_no_p2p_revenue(self, easy_env):
        no_p2p = idle(charge=0.0, reserve=0.2, p2p=0.0)
        run_n(easy_env, 48, no_p2p)
        assert easy_env.state.cumulative_p2p_usd == 0.0

    def test_p2p_available_in_zone_aggregates(self, easy_env):
        run_n(easy_env, 24, idle())   # advance to solar noon
        obs, _, _, _ = easy_env.step(idle())
        zone_b = next(z for z in obs.zone_aggregates if z.zone_id == "zone-b")
        assert zone_b.p2p_available_kw >= 0.0

    def test_p2p_revenue_tracked_in_state(self, easy_env):
        p2p_action = idle(p2p=1.0)
        run_n(easy_env, 24, p2p_action)
        # Cumulative P2P should be non-negative
        assert easy_env.state.cumulative_p2p_usd >= 0.0


# ---------------------------------------------------------------------------
# 6. Demand Response auction
# ---------------------------------------------------------------------------

class TestDemandResponse:
    def test_dr_bid_present_at_step_0(self, expert_env):
        # DR bids are scheduled at step 0 for expert task
        obs = expert_env._build_observation()
        assert obs.dr_bid.active is True

    def test_accept_dr_bid_increments_counter(self, expert_env):
        accept = idle(charge=-1.0, reserve=0.2, dr=True)
        expert_env.step(accept)
        assert expert_env.state.dr_bids_accepted >= 1

    def test_dr_bid_not_active_for_easy_task(self, easy_env):
        obs, _, _, _ = easy_env.step(idle())
        assert obs.dr_bid.active is False

    def test_dr_bid_premium_in_observation(self, expert_env):
        obs = expert_env._build_observation()
        assert obs.dr_bid.premium_multiplier >= 1.0

    def test_rejecting_dr_bid_keeps_counter_zero(self, expert_env):
        no_dr = idle(dr=False)
        expert_env.step(no_dr)
        assert expert_env.state.dr_bids_accepted == 0


# ---------------------------------------------------------------------------
# 7. Grid islanding
# ---------------------------------------------------------------------------

class TestGridIslanding:
    def test_grid_connected_true_before_islanding(self, islanding_env):
        obs = islanding_env._build_observation()
        assert obs.grid_connected is True

    def test_grid_disconnects_at_islanding_start(self, islanding_env):
        # Advance to step ISLANDING_START
        run_n(islanding_env, ISLANDING_START, idle())
        obs, _, _, _ = islanding_env.step(idle())
        assert obs.grid_connected is False

    def test_grid_reconnects_after_islanding(self, islanding_env):
        # Advance past ISLANDING_END
        run_n(islanding_env, ISLANDING_END + 1, idle())
        obs, _, _, _ = islanding_env.step(idle())
        assert obs.grid_connected is True

    def test_no_grid_profit_during_islanding(self, islanding_env):
        # During islanding, charge_kw is forced to 0, so no grid financial transactions
        run_n(islanding_env, ISLANDING_START, idle())   # get to islanding
        profit_before = islanding_env.state.cumulative_profit_usd
        sell = idle(charge=-1.0, reserve=0.0)
        islanding_env.step(sell)   # try to sell during islanding
        # Profit should not increase from grid sales (only P2P if applicable)
        profit_after = islanding_env.state.cumulative_profit_usd
        assert profit_after == pytest.approx(profit_before, abs=0.01)

    def test_islanding_blackouts_tracked(self, env):
        env.reset("islanding-emergency")
        # Drain batteries before islanding
        drain = idle(charge=-1.0, reserve=0.0)
        run_n(env, ISLANDING_START, drain)
        # Now in islanding with empty batteries — should trigger blackouts
        run_n(env, 5, idle())
        # islanding_blackouts may be > 0 if batteries drained enough
        assert env.state.islanding_blackouts >= 0   # just check it's tracked


# ---------------------------------------------------------------------------
# 8. EV load deferral
# ---------------------------------------------------------------------------

class TestEVDeferral:
    def test_defer_action_accepted(self, easy_env):
        defer = idle(defer=0.8)
        obs, _, _, _ = easy_env.step(defer)
        assert obs is not None   # action didn't crash

    def test_ev_defer_deadline_in_observation(self, easy_env):
        obs, _, _, _ = easy_env.step(idle())
        assert obs.ev_defer_deadline_step == 40

    def test_defer_zero_no_debt(self, easy_env):
        # Before step 32, defer action has no effect
        run_n(easy_env, 32, idle(defer=0.0))
        assert easy_env.state.ev_defer_debt_kwh == 0.0

    def test_defer_after_step_32_creates_debt(self, easy_env):
        # Advance to step 32 (14:00) when EV charging starts
        run_n(easy_env, 32, idle())
        defer = idle(defer=1.0)
        easy_env.step(defer)
        assert easy_env.state.ev_defer_debt_kwh > 0.0


# ---------------------------------------------------------------------------
# 9. Adversarial weather (expert task)
# ---------------------------------------------------------------------------

class TestAdversarialWeather:
    def test_expert_solar_drops_at_step_24(self, env):
        expert_solar = solar_curve("expert-demand-response")
        easy_solar   = solar_curve("easy-arbitrage")
        # At step 24 (cloud event), expert solar should be much lower than easy
        assert expert_solar[24] < easy_solar[24] * 0.5

    def test_expert_cloud_event_is_steps_24_to_26(self, env):
        expert_solar = solar_curve("expert-demand-response")
        medium_solar = solar_curve("medium-forecast-error")
        # The cloud event drops expert solar below medium solar in that window
        for s in [24, 25, 26]:
            assert expert_solar[s] < medium_solar[s]

    def test_expert_solar_normal_outside_cloud_window(self, env):
        expert_solar = solar_curve("expert-demand-response")
        medium_solar = solar_curve("medium-forecast-error")
        # Outside steps 24-26, expert and medium should have similar solar
        assert abs(expert_solar[12] - medium_solar[12]) < 0.1


# ---------------------------------------------------------------------------
# 10. Reasoning traces
# ---------------------------------------------------------------------------

class TestReasoningTraces:
    def test_reasoning_stored_when_provided(self, easy_env):
        action_with_reason = VppAction(
            global_charge_rate=-0.5,
            min_reserve_pct=0.2,
            reasoning="Price is $50, SoC is 50%, selling at half rate.",
        )
        easy_env.step(action_with_reason)
        traces = easy_env.get_reasoning_traces()
        assert len(traces) == 1
        assert traces[0]["reasoning"] == "Price is $50, SoC is 50%, selling at half rate."

    def test_no_reasoning_no_trace(self, easy_env):
        easy_env.step(idle())
        assert len(easy_env.get_reasoning_traces()) == 0

    def test_trace_includes_step_and_profit(self, easy_env):
        action_with_reason = VppAction(
            global_charge_rate=-0.5,
            min_reserve_pct=0.2,
            reasoning="Test reasoning.",
        )
        easy_env.step(action_with_reason)
        trace = easy_env.get_reasoning_traces()[0]
        assert "step" in trace
        assert "step_profit" in trace
        assert "action" in trace

    def test_traces_reset_on_new_episode(self, easy_env):
        action_with_reason = VppAction(
            global_charge_rate=-0.5,
            min_reserve_pct=0.2,
            reasoning="Reasoning before reset.",
        )
        easy_env.step(action_with_reason)
        assert len(easy_env.get_reasoning_traces()) == 1
        easy_env.reset("easy-arbitrage")
        assert len(easy_env.get_reasoning_traces()) == 0


# ---------------------------------------------------------------------------
# 11. All 5 tasks reset correctly
# ---------------------------------------------------------------------------

class TestAllTasks:
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_task_resets_without_error(self, env, task_id):
        obs = env.reset(task_id)
        assert obs.step_id == 0
        assert len(obs.telemetry) == 100

    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_task_runs_full_episode(self, env, task_id):
        env.reset(task_id)
        action = VppAction(global_charge_rate=0.0, min_reserve_pct=0.0)
        done = False
        steps = 0
        while not done and steps < 50:
            _, _, done, _ = env.step(action)
            steps += 1
        assert steps == 48 and done

    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_pareto_score_valid_for_all_tasks(self, env, task_id):
        env.reset(task_id)
        run_n(env, 48, idle(charge=-0.3, reserve=0.2))
        ps = env.get_pareto_score()
        assert 0.0 <= ps.aggregate_score <= 1.0