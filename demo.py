#!/usr/bin/env python3
"""
demo.py — Live demo of the VPP environment.

Four rule-based agents compete across all three tasks.
Demonstrates the scoring gradient clearly:

  Idle        → $0 profit,   score ≈ 0
  Naive Sell  → high profit, many violations  → score hurt by penalty
  Conservative→ modest profit, zero violations → reliable middle score
  Smart Rules → best profit + low violations  → highest rule-based score

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860   # terminal 1
  python demo.py                                        # terminal 2
"""

import os, sys
import requests

BASE         = os.getenv("VPP_SERVER_URL", "http://localhost:7860")
EPISODE_STEPS = 48    # 12-hour episodes


# ── Agents ────────────────────────────────────────────────────────────────────

def agent_idle(obs: dict) -> dict:
    """Never trades — establishes the zero-profit baseline."""
    return {"global_charge_rate": 0.0, "min_reserve_pct": 0.2}


def agent_naive_sell(obs: dict) -> dict:
    """Full discharge every step — high revenue, drains batteries fast."""
    return {"global_charge_rate": -1.0, "min_reserve_pct": 0.05}


def agent_conservative(obs: dict) -> dict:
    """
    Sell only when solar is actively generating (daytime).
    Never drops reserve below 25 %. Protects batteries at cost of profit.
    """
    telemetry  = obs.get("telemetry", [])
    mean_solar = sum(t["current_solar_gen_kw"] for t in telemetry) / max(len(telemetry), 1)
    mean_soc   = sum(t["soc"]                  for t in telemetry) / max(len(telemetry), 1)

    if mean_solar > 1.0 and mean_soc > 0.30:
        return {"global_charge_rate": -0.4, "min_reserve_pct": 0.25}
    return {"global_charge_rate": 0.0, "min_reserve_pct": 0.25}


def agent_smart(obs: dict) -> dict:
    """
    Rule-based heuristic (priority order):

      1. Grid frequency emergency (< 49.8 Hz) → discharge hard immediately.
      2. Battery critically low (< 20 % SoC)  → stop selling, charge if cheap.
      3. Price spike (> 200 $/MWh)            → sell at full power.
      4. High price (> 55 $/MWh) + SoC > 35% → sell at 70 % rate.
      5. Solar surplus + battery > 70 %       → sell surplus at 50 % rate.
      6. Cheap price (< 38 $/MWh) + SoC < 60%→ buy and store.
      7. Otherwise                            → idle.

    Always maintains 20 % reserve except during a grid emergency.
    """
    telemetry  = obs.get("telemetry", [])
    freq       = obs.get("grid_frequency_hz", 50.0)
    price      = obs.get("market_price_per_mwh", 50.0)
    mean_soc   = sum(t["soc"]                  for t in telemetry) / max(len(telemetry), 1)
    mean_solar = sum(t["current_solar_gen_kw"] for t in telemetry) / max(len(telemetry), 1)

    # 1. Grid emergency — discharge regardless of SoC
    if freq < 49.8:
        return {"global_charge_rate": -1.0, "min_reserve_pct": 0.10}

    # 2. Battery critically low — protect and possibly recharge
    if mean_soc < 0.20:
        rate = 0.4 if price < 42.0 else 0.0
        return {"global_charge_rate": rate, "min_reserve_pct": 0.20}

    # 3. Massive price spike (hard task: 10× = ~500 $/MWh)
    if price > 200.0:
        return {"global_charge_rate": -1.0, "min_reserve_pct": 0.20}

    # 4. High price — sell moderately
    if price > 55.0 and mean_soc > 0.35:
        return {"global_charge_rate": -0.7, "min_reserve_pct": 0.20}

    # 5. Surplus solar, battery nearly full — sell the excess
    if mean_solar > 2.0 and mean_soc > 0.70:
        return {"global_charge_rate": -0.5, "min_reserve_pct": 0.20}

    # 6. Cheap grid power — buy and store for later high-price window
    if price < 38.0 and mean_soc < 0.60:
        return {"global_charge_rate": 0.5, "min_reserve_pct": 0.20}

    # 7. Default: idle
    return {"global_charge_rate": 0.0, "min_reserve_pct": 0.20}


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(task_id: str, agent_fn, agent_name: str, session: requests.Session) -> dict:
    resp = session.post(
        f"{BASE}/reset",
        params={"task_id": task_id},
        timeout=15,
    )
    resp.raise_for_status()
    obs          = resp.json()
    total_reward = 0.0
    steps        = 0

    for _ in range(EPISODE_STEPS + 2):   # +2 safety cap
        action = agent_fn(obs)
        resp   = session.post(f"{BASE}/step", json=action, timeout=15)
        resp.raise_for_status()
        data          = resp.json()
        obs           = data["observation"]
        total_reward += float(data["reward"])
        steps        += 1
        if data["done"]:
            break

    grader = session.get(f"{BASE}/grader", timeout=10).json()
    return {
        "agent":      agent_name,
        "task":       task_id,
        "steps":      steps,
        "reward":     round(total_reward, 2),
        "profit_usd": round(grader.get("cumulative_profit_usd", 0.0), 2),
        "violations": grader.get("safety_violations", 0),
        "score":      grader.get("score", 0.0),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    session = requests.Session()

    try:
        assert session.get(f"{BASE}/health", timeout=5).status_code == 200
    except Exception:
        print(f"ERROR: Server not reachable at {BASE}")
        print("Run:  uvicorn server.app:app --host 0.0.0.0 --port 7860")
        sys.exit(1)

    agents = [
        ("Idle",         agent_idle),
        ("Naive Sell",   agent_naive_sell),
        ("Conservative", agent_conservative),
        ("Smart Rules",  agent_smart),
    ]
    tasks = [
        "easy-arbitrage",
        "medium-forecast-error",
        "hard-frequency-response",
    ]

    results = []
    total   = len(agents) * len(tasks)
    n       = 0

    print(f"\nVPP Demo — {total} episodes ({len(agents)} agents × {len(tasks)} tasks)\n")

    for task in tasks:
        print(f"  ── {task} ──")
        for name, fn in agents:
            n += 1
            print(f"    [{n:02d}/{total}] {name:<14} ...", end=" ", flush=True)
            r = run_episode(task, fn, name, session)
            results.append(r)
            bar = "█" * int(r["score"] * 20)
            print(
                f"score={r['score']:.4f}  "
                f"profit=${r['profit_usd']:>6.0f}  "
                f"steps={r['steps']}  "
                f"viols={r['violations']}  "
                f"{bar}"
            )
        print()

    # ── Summary table ─────────────────────────────────────────────────────────
    W = 82
    print("=" * W)
    print(f"  {'Agent':<14}  {'Task':<32}  {'Score':>6}  {'Profit':>8}  {'Viols':>5}")
    print("=" * W)

    prev_task = None
    for r in results:
        if prev_task and r["task"] != prev_task:
            print("─" * W)
        prev_task = r["task"]
        bar = "█" * int(r["score"] * 20)
        print(
            f"  {r['agent']:<14}  {r['task']:<32}  "
            f"{r['score']:>6.4f}  ${r['profit_usd']:>7.0f}  "
            f"{r['violations']:>5}  {bar}"
        )
    print("=" * W)

    best = max(results, key=lambda x: x["score"])
    print(f"\n🏆  Best: {best['agent']} on {best['task']} — score {best['score']:.4f}")
    print("\nNext: python baseline_inference.py  (LLM agent scores)\n")


if __name__ == "__main__":
    main()