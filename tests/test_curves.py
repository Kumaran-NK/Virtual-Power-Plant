# tests/test_curves.py
"""
Unit tests for the deterministic task curves (solar / demand / price).

Run with: pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from server.task_curves import (
    solar_curve, demand_curve, price_curve,
    EPISODE_STEPS, GRID_STRESS_STEP,
    HEATWAVE_START, HEATWAVE_END,
)

TASKS = ["easy-arbitrage", "medium-forecast-error", "hard-frequency-response"]


# ---------------------------------------------------------------------------
# Solar
# ---------------------------------------------------------------------------

class TestSolarCurve:
    @pytest.mark.parametrize("task_id", TASKS)
    def test_length_48(self, task_id):
        assert len(solar_curve(task_id)) == EPISODE_STEPS

    @pytest.mark.parametrize("task_id", TASKS)
    def test_all_non_negative(self, task_id):
        arr = solar_curve(task_id)
        assert np.all(arr >= 0.0), "Solar generation can never be negative"

    @pytest.mark.parametrize("task_id", TASKS)
    def test_dtype_float(self, task_id):
        assert solar_curve(task_id).dtype in (np.float32, np.float64)

    def test_easy_peak_higher_than_hard(self):
        easy = solar_curve("easy-arbitrage")
        hard = solar_curve("hard-frequency-response")
        assert np.max(easy) > np.max(hard), \
            "Easy task should have more solar than hard"

    def test_medium_between_easy_and_hard(self):
        easy   = solar_curve("easy-arbitrage")
        medium = solar_curve("medium-forecast-error")
        hard   = solar_curve("hard-frequency-response")
        assert np.max(hard) < np.max(medium) < np.max(easy)

    def test_peak_near_solar_noon(self):
        """Easy task peak should occur around step 24 (12:00)."""
        easy = solar_curve("easy-arbitrage")
        peak_step = int(np.argmax(easy))
        assert 20 <= peak_step <= 28, \
            f"Solar peak expected near step 24 (noon), got step {peak_step}"

    def test_zero_at_step_0(self):
        """Solar at dawn (step 0) should be ≈ 0."""
        for task in TASKS:
            arr = solar_curve(task)
            assert arr[0] < 0.1, f"Solar at step 0 should be near zero for {task}"

    def test_multipliers_applied_correctly(self):
        easy   = solar_curve("easy-arbitrage")
        medium = solar_curve("medium-forecast-error")
        hard   = solar_curve("hard-frequency-response")
        # Multipliers: easy=1.5, medium=1.0, hard=0.7
        peak_e = np.max(easy)
        peak_m = np.max(medium)
        peak_h = np.max(hard)
        assert abs(peak_e / peak_m - 1.5) < 0.05, "Easy/medium peak ratio should be ~1.5"
        assert abs(peak_h / peak_m - 0.7) < 0.05, "Hard/medium peak ratio should be ~0.7"


# ---------------------------------------------------------------------------
# Demand
# ---------------------------------------------------------------------------

class TestDemandCurve:
    @pytest.mark.parametrize("task_id", TASKS)
    def test_length_48(self, task_id):
        assert len(demand_curve(task_id)) == EPISODE_STEPS

    @pytest.mark.parametrize("task_id", TASKS)
    def test_all_positive(self, task_id):
        arr = demand_curve(task_id)
        assert np.all(arr > 0.0), "Demand is always positive (someone is always consuming)"

    def test_heatwave_spike_exists(self):
        """Medium task heatwave should produce significantly elevated demand."""
        medium = demand_curve("medium-forecast-error")
        easy   = demand_curve("easy-arbitrage")
        heatwave_medium = medium[HEATWAVE_START:HEATWAVE_END]
        heatwave_easy   = easy[HEATWAVE_START:HEATWAVE_END]
        assert np.mean(heatwave_medium) > np.mean(heatwave_easy) * 2, \
            "Heatwave demand should be much higher than easy demand in same window"

    def test_heatwave_window_indices(self):
        """Heatwave steps (16–31) for medium should differ from baseline."""
        medium = demand_curve("medium-forecast-error")
        non_hw = np.concatenate([medium[:HEATWAVE_START], medium[HEATWAVE_END:]])
        hw     = medium[HEATWAVE_START:HEATWAVE_END]
        assert np.mean(hw) > np.mean(non_hw), "Heatwave window should have higher mean demand"

    def test_easy_lower_than_hard(self):
        easy = demand_curve("easy-arbitrage")
        hard = demand_curve("hard-frequency-response")
        assert np.mean(easy) < np.mean(hard), \
            "Easy task should have lower demand than hard"

    @pytest.mark.parametrize("task_id", TASKS)
    def test_reasonable_range(self, task_id):
        """Demand should be plausible residential kW values."""
        arr = demand_curve(task_id)
        assert np.all(arr < 10.0), f"Demand > 10 kW/home is unrealistic for {task_id}"
        assert np.all(arr > 0.0)


# ---------------------------------------------------------------------------
# Price
# ---------------------------------------------------------------------------

class TestPriceCurve:
    @pytest.mark.parametrize("task_id", TASKS)
    def test_length_48(self, task_id):
        assert len(price_curve(task_id)) == EPISODE_STEPS

    @pytest.mark.parametrize("task_id", TASKS)
    def test_all_positive(self, task_id):
        arr = price_curve(task_id)
        assert np.all(arr > 0.0), "Price should always be positive"

    def test_easy_flat(self):
        """Easy task price should be nearly constant at $50/MWh."""
        easy = price_curve("easy-arbitrage")
        assert np.all(easy == 50.0), "Easy task should have flat $50/MWh price"

    def test_medium_sinusoidal(self):
        """Medium price should vary (not flat)."""
        medium = price_curve("medium-forecast-error")
        assert np.std(medium) > 1.0, "Medium price should have noticeable variation"

    def test_hard_has_spike(self):
        """Hard task price should contain a single 10× spike at GRID_STRESS_STEP."""
        hard = price_curve("hard-frequency-response")
        spike_value = hard[GRID_STRESS_STEP]
        other_values = np.delete(hard, GRID_STRESS_STEP)
        assert spike_value > np.max(other_values) * 5, \
            f"Spike at step {GRID_STRESS_STEP} should be >> other prices"

    def test_spike_is_at_correct_step(self):
        hard = price_curve("hard-frequency-response")
        peak_step = int(np.argmax(hard))
        assert peak_step == GRID_STRESS_STEP, \
            f"Price spike should be at step {GRID_STRESS_STEP}, found at {peak_step}"

    def test_medium_range(self):
        """Medium price should span roughly $35–$65/MWh."""
        medium = price_curve("medium-forecast-error")
        non_spike = medium   # no spike for medium
        assert np.min(non_spike) >= 28.0
        assert np.max(non_spike) <= 72.0

    def test_hard_non_spike_similar_to_medium(self):
        """Hard task (excluding spike) should have similar range to medium."""
        hard   = price_curve("hard-frequency-response")
        hard_no_spike = np.delete(hard, GRID_STRESS_STEP)
        medium = price_curve("medium-forecast-error")
        # Both use the same sinusoidal base
        assert abs(np.mean(hard_no_spike) - np.mean(medium)) < 5.0


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_episode_steps_is_48(self):
        assert EPISODE_STEPS == 48

    def test_grid_stress_step_is_26(self):
        assert GRID_STRESS_STEP == 26

    def test_heatwave_window_is_correct(self):
        assert HEATWAVE_START == 16
        assert HEATWAVE_END   == 32