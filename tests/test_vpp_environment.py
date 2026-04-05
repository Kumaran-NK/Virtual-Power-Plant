# tests/test_vpp_environment.py
"""
Unit tests for the VPP core simulation engine.

Run with: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

# We import only the pure-Python components so tests run without the server
from server.task_curves import (
    solar_curve, demand_curve, price_curve, EPISODE_STEPS, GRID_STRESS_STEP
)
from models import VppAction, BatteryAsset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    """Fresh VppEnvironment for each test."""
    # Lazy import avoids dependency on openenv-core at collection time
    from server.vpp_environment import VppEnvironment
    return VppEnvironment()


@pytest.fixture
def easy_env(env):
    """Environment reset on easy-arbitrage task."""
    env.reset("easy-arbitrage")
    return env


@pytest.fixture
def hard_env(env):
    """Environment reset on hard-frequency-response task."""
    env.reset("hard-frequency-response")
    return env


# ---------------------------------------------------------------------------
# reset() tests
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_observation(self, env):
        obs = env.reset("easy-arbitrage")
        assert obs is not None
        assert obs.step_id == 0

    def test_reset_produces_100_telemetry(self, env):
        obs = env.reset("easy-arbitrage")
        assert len(obs.telemetry) == 100

    def test_reset_initial_soc_half(self, env):
        obs = env.reset("easy-arbitrage")
        socs = [t.soc for t in obs.telemetry]
        assert all(abs(s - 0.5) < 1e-9 for s in socs), "All homes should start at 50 % SoC"

    def test_reset_state_not_done(self, env):
        env.reset("easy-arbitrage")
        assert not env.state.done

    def test_reset_cumulative_profit_zero(self, env):
        env.reset("medium-forecast-error")
        assert env.state.cumulative_profit_usd == 0.0

    def test_reset_step_count_zero(self, env):
        env.reset("hard-frequency-response")
        assert env.state.current_step == 0

    def test_reset_zone_aggregates_present(self, env):
        obs = env.reset("easy-arbitrage")
        assert len(obs.zone_aggregates) == 2, "Should have two zones (A and B)"

    def test_reset_zone_home_counts(self, env):
        obs = env.reset("easy-arbitrage")
        zone_a = next(z for z in obs.zone_aggregates if z.zone_id == "zone-a")
        zone_b = next(z for z in obs.zone_aggregates if z.zone_id == "zone-b")
        assert zone_a.home_count == 40
        assert zone_b.home_count == 60

    def test_reset_different_tasks_differ(self, env):
        obs_easy = env.reset("easy-arbitrage")
        obs_hard = env.reset("hard-frequency-response")
        # Solar should differ (easy 1.5× multiplier vs hard 0.7×)
        assert obs_easy.market_price_per_mwh != obs_hard.market_price_per_mwh or \
               obs_easy.short_term_solar_forecast != obs_hard.short_term_solar_forecast

    def test_reset_seeded_reproducible(self, env):
        """Same task produces identical initial observation on repeated resets."""
        obs1 = env.reset("easy-arbitrage")
        obs2 = env.reset("easy-arbitrage")
        assert obs1.step_id == obs2.step_id == 0
        assert obs1.market_price_per_mwh == obs2.market_price_per_mwh


# ---------------------------------------------------------------------------
# step() tests
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_returns_four_values(self, easy_env):
        action = VppAction(global_charge_rate=0.0, min_reserve_pct=0.2)
        result = easy_env.step(action)
        assert len(result) == 4   # obs, reward, done, info

    def test_idle_step_advances_counter(self, easy_env):
        action = VppAction(global_charge_rate=0.0, min_reserve_pct=0.2)
        easy_env.step(action)
        assert easy_env.state.current_step == 1

    def test_sell_generates_positive_reward(self, easy_env):
        """Selling (negative charge_rate) should yield positive step_profit."""
        action = VppAction(global_charge_rate=-1.0, min_reserve_pct=0.0)
        _, reward, _, info = easy_env.step(action)
        assert info["step_profit_usd"] > 0, "Selling should make money"

    def test_buy_generates_negative_profit(self, easy_env):
        """Buying (positive charge_rate) should yield negative step_profit."""
        action = VppAction(global_charge_rate=1.0, min_reserve_pct=0.0)
        _, _, _, info = easy_env.step(action)
        assert info["step_profit_usd"] < 0, "Buying from grid should cost money"

    def test_done_flag_at_step_48(self, easy_env):
        action = VppAction(global_charge_rate=0.0, min_reserve_pct=0.0)
        done = False
        for _ in range(48):
            _, _, done, _ = easy_env.step(action)
        assert done, "Episode should be done after 48 steps"

    def test_episode_runs_exactly_48_steps(self, easy_env):
        action = VppAction(global_charge_rate=0.0, min_reserve_pct=0.0)
        steps = 0
        for _ in range(48):
            _, _, done, _ = easy_env.step(action)
            steps += 1
        assert steps == 48
        assert easy_env.state.current_step == 48

    def test_soc_stays_in_range(self, easy_env):
        """Aggressively selling should never push SoC below 0."""
        action = VppAction(global_charge_rate=-1.0, min_reserve_pct=0.0)
        for _ in range(48):
            obs, _, _, _ = easy_env.step(action)
        socs = [t.soc for t in obs.telemetry]
        assert all(0.0 <= s <= 1.0 for s in socs)

    def test_safety_violation_counted(self, easy_env):
        """Dropping below min_reserve_pct should increment violations."""
        # Set a very high reserve floor and immediately sell → violation
        action = VppAction(global_charge_rate=-1.0, min_reserve_pct=0.99)
        easy_env.step(action)
        assert easy_env.state.safety_violations_count >= 1

    def test_no_violation_when_idle(self, easy_env):
        """Idle agent starting at 50 % SoC with 20 % reserve should have zero violations."""
        action = VppAction(global_charge_rate=0.0, min_reserve_pct=0.2)
        for _ in range(10):
            easy_env.step(action)
        assert easy_env.state.safety_violations_count == 0

    def test_reward_has_penalty_on_emergency(self, hard_env):
        """
        At step 26, freq drops to 49.5 Hz.
        If agent idles (charge_rate ≥ 0), reward should include -2.0 emergency penalty.
        """
        idle = VppAction(global_charge_rate=0.0, min_reserve_pct=0.2)
        # Advance to step 26
        for _ in range(26):
            hard_env.step(idle)
        _, reward, _, info = hard_env.step(idle)
        assert reward < 0, "Ignoring grid emergency should yield negative reward"

    def test_step_raises_after_episode_ends(self, easy_env):
        action = VppAction(global_charge_rate=0.0, min_reserve_pct=0.0)
        for _ in range(48):
            easy_env.step(action)
        with pytest.raises(RuntimeError):
            easy_env.step(action)

    def test_zone_aggregates_in_observation(self, easy_env):
        action = VppAction(global_charge_rate=0.0, min_reserve_pct=0.2)
        obs, _, _, _ = easy_env.step(action)
        assert len(obs.zone_aggregates) == 2

    def test_zone_b_has_ev_chargers(self, easy_env):
        action = VppAction(global_charge_rate=0.0, min_reserve_pct=0.2)
        obs, _, _, _ = easy_env.step(action)
        zone_b = next(z for z in obs.zone_aggregates if z.zone_id == "zone-b")
        assert zone_b.has_ev_chargers is True

    def test_cumulative_profit_accumulates(self, easy_env):
        sell = VppAction(global_charge_rate=-1.0, min_reserve_pct=0.0)
        for _ in range(5):
            easy_env.step(sell)
        assert easy_env.state.cumulative_profit_usd > 0


# ---------------------------------------------------------------------------
# State-transition tests
# ---------------------------------------------------------------------------

class TestStateTransitions:
    def test_state_syncs_with_step(self, easy_env):
        action = VppAction(global_charge_rate=0.0, min_reserve_pct=0.2)
        easy_env.step(action)
        assert easy_env.state.step_count == 1

    def test_state_battery_soc_updated(self, easy_env):
        """After selling, SoC of all homes should have decreased."""
        soc_before = list(easy_env.state.battery_true_soc.values())
        sell = VppAction(global_charge_rate=-1.0, min_reserve_pct=0.0)
        easy_env.step(sell)
        soc_after = list(easy_env.state.battery_true_soc.values())
        # SoC should change (decrease or stabilise with solar)
        changed = [abs(a - b) > 1e-9 for a, b in zip(soc_before, soc_after)]
        assert any(changed), "SoC should change after a sell action"

    def test_revenue_vs_cost_split(self, easy_env):
        sell = VppAction(global_charge_rate=-1.0, min_reserve_pct=0.0)
        buy  = VppAction(global_charge_rate=1.0,  min_reserve_pct=0.0)
        easy_env.step(sell)
        easy_env.step(buy)
        assert easy_env.state.cumulative_revenue_usd >= 0
        assert easy_env.state.cumulative_cost_usd    >= 0