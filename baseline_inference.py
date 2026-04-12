#!/usr/bin/env python3
"""
Deterministic baseline scorer for the VPP environment.

This script runs one episode per task and emits baseline Pareto scores.
It is intentionally rule-based (no external LLM/API dependencies).
"""

import argparse
import json
import sys
from typing import Dict

from models import VppAction, VppObservation
from server.task_curves import ALL_TASK_IDS
from server.vpp_environment import VppEnvironment


def _mean_soc(obs: VppObservation) -> float:
    if not obs.telemetry:
        return 0.5
    return sum(t.soc for t in obs.telemetry) / len(obs.telemetry)


def _zone_b_p2p_available(obs: VppObservation) -> float:
    for zone in obs.zone_aggregates:
        if zone.zone_id == "zone-b":
            return zone.p2p_available_kw
    return 0.0


def _rule_action(obs: VppObservation, task_id: str) -> VppAction:
    """Simple deterministic policy used for baseline generation."""
    step = obs.step_id
    price = obs.market_price_per_mwh
    mean_soc = _mean_soc(obs)
    p2p_avail = _zone_b_p2p_available(obs)
    dr_bid = obs.dr_bid

    defer_ev = 0.6 if (32 <= step < 40 and price > 55.0) else 0.0
    p2p_rate = 0.8 if p2p_avail > 0.75 else 0.0

    if not obs.grid_connected:
        return VppAction(
            global_charge_rate=0.0,
            min_reserve_pct=0.40,
            defer_ev_charging=0.0,
            accept_dr_bid=False,
            p2p_export_rate=0.0,
            reasoning=None,
        )

    if obs.grid_frequency_hz < 49.8:
        return VppAction(
            global_charge_rate=-1.0,
            min_reserve_pct=0.10,
            defer_ev_charging=0.0,
            accept_dr_bid=False,
            p2p_export_rate=p2p_rate,
            reasoning=None,
        )

    if dr_bid.steps_remaining > 0:
        commit_fraction = min(1.0, dr_bid.committed_power_kw / 5.0)
        return VppAction(
            global_charge_rate=-commit_fraction,
            min_reserve_pct=0.20,
            defer_ev_charging=defer_ev,
            accept_dr_bid=False,
            p2p_export_rate=p2p_rate,
            reasoning=None,
        )

    if dr_bid.active and dr_bid.premium_multiplier >= 2.0 and mean_soc > 0.55:
        commit_fraction = min(1.0, dr_bid.committed_power_kw / 5.0)
        return VppAction(
            global_charge_rate=-commit_fraction,
            min_reserve_pct=0.20,
            defer_ev_charging=defer_ev,
            accept_dr_bid=True,
            p2p_export_rate=p2p_rate,
            reasoning=None,
        )

    if price > 70.0 and mean_soc > 0.30:
        charge_rate = -0.8
    elif price > 55.0 and mean_soc > 0.45:
        charge_rate = -0.5
    elif price < 38.0 and mean_soc < 0.65 and step >= 17:
        charge_rate = 0.6
    elif mean_soc < 0.20 and price < 45.0:
        charge_rate = 0.4
    else:
        charge_rate = 0.0

    # During the high-emission morning window, avoid unnecessary grid charging.
    if step < 17 and charge_rate > 0 and task_id != "hard-frequency-response":
        charge_rate = 0.0

    return VppAction(
        global_charge_rate=charge_rate,
        min_reserve_pct=0.20,
        defer_ev_charging=defer_ev,
        accept_dr_bid=False,
        p2p_export_rate=p2p_rate,
        reasoning=None,
    )


def _run_episode(task_id: str, seed: int) -> Dict[str, float]:
    env = VppEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)

    steps = 0
    while not obs.done and steps < 48:
        action = _rule_action(obs, task_id)
        obs = env.step(action)
        steps += 1

    pareto = env.get_pareto_score()
    return {
        "aggregate_score": round(pareto.aggregate_score, 4),
        "profit_score": round(pareto.profit_score, 4),
        "safety_score": round(pareto.safety_score, 4),
        "carbon_score": round(pareto.carbon_score, 4),
        "degradation_score": round(pareto.degradation_score, 4),
        "dr_score": round(pareto.dr_score, 4),
    }


def compute_baseline_scores(seed: int = 123) -> Dict[str, Dict[str, float]]:
    scores: Dict[str, Dict[str, float]] = {}
    for index, task_id in enumerate(ALL_TASK_IDS):
        scores[task_id] = _run_episode(task_id, seed + index)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute deterministic VPP baseline scores.")
    parser.add_argument("--agent", default="rule", help="Agent type (only 'rule' is supported).")
    parser.add_argument("--json-only", action="store_true", help="Emit JSON only to stdout.")
    parser.add_argument("--seed", type=int, default=123, help="Base random seed.")
    args = parser.parse_args()

    if args.agent.lower() != "rule":
        print("[WARN] Unsupported agent type requested; using rule baseline.", file=sys.stderr)

    scores = compute_baseline_scores(seed=args.seed)

    if args.json_only:
        print(json.dumps(scores))
        return

    print("Baseline scores:")
    for task_id in ALL_TASK_IDS:
        row = scores[task_id]
        print(f"  {task_id:<25} aggregate={row['aggregate_score']:.4f}")


if __name__ == "__main__":
    main()
