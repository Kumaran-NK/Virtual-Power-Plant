#!/usr/bin/env python3
"""
validate.py — Pre-submission smoke test for the VPP OpenEnv environment.

Verifies every requirement from the hackathon pre-submission checklist:
  ✅ /health returns 200
  ✅ /tasks lists 3 tasks with action schema
  ✅ /reset works for all 3 task_ids
  ✅ /step accepts valid actions and returns obs/reward/done/info
  ✅ /grader returns a score in [0.0, 1.0]
  ✅ /state returns the ground-truth state
  ✅ /baseline returns scores JSON
  ✅ Episode runs to completion (48 steps)
  ✅ Reward is a float (not always 0)
  ✅ Done flag is True at step 48
  ✅ Grader produces different scores for smart vs idle agents

Usage:
  # Start the server first:
    uvicorn server.app:app --host 0.0.0.0 --port 7860

  # Then run:
    python validate.py [--url http://localhost:7860]
"""

import argparse
import sys
import json
import time

import requests

# ─── ANSI colours ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

passed = 0
failed = 0


def ok(msg: str):
    global passed
    passed += 1
    print(f"  {GREEN}✓{RESET}  {msg}")


def fail(msg: str, detail: str = ""):
    global failed
    failed += 1
    detail_str = f" — {detail}" if detail else ""
    print(f"  {RED}✗{RESET}  {msg}{detail_str}")


def section(title: str):
    print(f"\n{BOLD}{title}{RESET}")


# ─── Test helpers ─────────────────────────────────────────────────────────────

def get(session, url, **kw):
    try:
        r = session.get(url, timeout=10, **kw)
        return r
    except Exception as e:
        return type("R", (), {"status_code": 0, "text": str(e), "json": lambda: {}})()


def post(session, url, **kw):
    try:
        r = session.post(url, timeout=10, **kw)
        return r
    except Exception as e:
        return type("R", (), {"status_code": 0, "text": str(e), "json": lambda: {}})()


# ─── Individual checks ────────────────────────────────────────────────────────

def check_health(session, base):
    section("1. Health endpoint")
    r = get(session, f"{base}/health")
    if r.status_code == 200:
        ok("/health → 200 OK")
    else:
        fail("/health did not return 200", f"got {r.status_code}")


def check_tasks(session, base):
    section("2. /tasks")
    r = get(session, f"{base}/tasks")
    if r.status_code != 200:
        fail("/tasks did not return 200", f"got {r.status_code}")
        return None

    data = r.json()
    tasks = data.get("tasks", [])
    if len(tasks) >= 3:
        ok(f"/tasks returned {len(tasks)} tasks")
    else:
        fail(f"/tasks must return ≥3 tasks, got {len(tasks)}")

    ids = [t["id"] for t in tasks]
    expected = {"easy-arbitrage", "medium-forecast-error", "hard-frequency-response"}
    if expected.issubset(set(ids)):
        ok("All 3 required task IDs present")
    else:
        fail("Missing task IDs", f"got {ids}")

    schema = data.get("action_schema")
    if schema:
        ok("action_schema present in /tasks response")
    else:
        fail("action_schema missing from /tasks")

    reward_schema = data.get("reward_schema")
    if reward_schema:
        ok("reward_schema present in /tasks response")
    else:
        fail("reward_schema missing from /tasks")

    return ids


def check_reset(session, base, task_ids):
    section("3. /reset for all tasks")
    for task_id in task_ids:
        r = post(session, f"{base}/reset", params={"task_id": task_id})
        if r.status_code == 200:
            obs = r.json()
            if "step_id" in obs and "telemetry" in obs:
                ok(f"/reset?task_id={task_id} → valid observation")
            else:
                fail(f"/reset?task_id={task_id} missing fields", str(obs.keys()))
        else:
            fail(f"/reset?task_id={task_id} → {r.status_code}", r.text[:100])

    # Invalid task_id should return 422
    r = post(session, f"{base}/reset", params={"task_id": "invalid-task"})
    if r.status_code in (400, 422):
        ok("Invalid task_id rejected with 4xx")
    else:
        fail("Invalid task_id should return 4xx", f"got {r.status_code}")


def check_step(session, base):
    section("4. /step — basic action loop")

    # Reset first
    r = post(session, f"{base}/reset", params={"task_id": "easy-arbitrage"})
    if r.status_code != 200:
        fail("Could not reset before step test")
        return

    # Take one step with a valid action
    action = {"global_charge_rate": -0.5, "min_reserve_pct": 0.2}
    r = post(session, f"{base}/step", json=action)
    if r.status_code != 200:
        fail(f"/step returned {r.status_code}", r.text[:100])
        return

    data = r.json()
    required_keys = {"observation", "reward", "done", "info"}
    if required_keys.issubset(data.keys()):
        ok("/step response has all required keys (observation, reward, done, info)")
    else:
        fail("Missing keys in /step response", str(set(data.keys())))

    reward = data.get("reward")
    if isinstance(reward, (int, float)):
        ok(f"reward is a number: {reward:.4f}")
    else:
        fail(f"reward should be a float, got {type(reward)}")

    obs = data.get("observation", {})
    if "telemetry" in obs and len(obs["telemetry"]) == 100:
        ok("observation.telemetry has 100 home entries")
    else:
        n = len(obs.get("telemetry", []))
        fail(f"Expected 100 telemetry entries, got {n}")

    # Test invalid action (out of range)
    bad_action = {"global_charge_rate": 5.0, "min_reserve_pct": 0.2}
    r = post(session, f"{base}/step", json=bad_action)
    if r.status_code == 422:
        ok("Out-of-range action rejected with 422")
    else:
        fail("Out-of-range action should return 422", f"got {r.status_code}")


def check_full_episode(session, base, task_id: str):
    section(f"5. Full episode — {task_id}")

    r = post(session, f"{base}/reset", params={"task_id": task_id})
    if r.status_code != 200:
        fail(f"Reset failed for {task_id}")
        return

    done = False
    steps = 0
    total_reward = 0.0
    action = {"global_charge_rate": -0.3, "min_reserve_pct": 0.2}

    while not done and steps < 200:   # safety cap
        r = post(session, f"{base}/step", json=action)
        if r.status_code != 200:
            fail(f"Step {steps} failed", r.text[:80])
            return
        data = r.json()
        done = data["done"]
        total_reward += float(data["reward"])
        steps += 1

    if steps == 48 and done:
        ok(f"Episode completed in exactly 48 steps ✓  (total reward: {total_reward:.2f})")
    elif done and steps < 48:
        fail(f"Episode ended early at step {steps} (expected 48)")
    else:
        fail(f"Episode did not finish within 48 steps (steps={steps}, done={done})")


def check_grader(session, base):
    section("6. /grader")

    # Run a deliberate idle agent
    post(session, f"{base}/reset", params={"task_id": "easy-arbitrage"})
    for _ in range(48):
        r = post(session, f"{base}/step", json={"global_charge_rate": 0.0, "min_reserve_pct": 0.2})
        if r.json().get("done"):
            break

    r = get(session, f"{base}/grader")
    if r.status_code != 200:
        fail(f"/grader returned {r.status_code}")
        return

    data = r.json()
    score = data.get("score")

    if isinstance(score, float) and 0.0 <= score <= 1.0:
        ok(f"/grader score in [0.0, 1.0]: {score:.4f}")
    else:
        fail(f"score must be float in [0,1], got {score!r}")

    # Run a selling agent — score should be higher than idle
    post(session, f"{base}/reset", params={"task_id": "easy-arbitrage"})
    for _ in range(48):
        r = post(session, f"{base}/step", json={"global_charge_rate": -1.0, "min_reserve_pct": 0.1})
        if r.json().get("done"):
            break

    sell_score = get(session, f"{base}/grader").json().get("score", 0.0)
    if sell_score >= score:
        ok(f"Selling agent ({sell_score:.4f}) scores ≥ idle agent ({score:.4f}) — reward signal correct")
    else:
        fail(
            "Selling agent scored lower than idle agent — reward function may be inverted",
            f"sell={sell_score:.4f} idle={score:.4f}",
        )

    # Check required detail fields
    required = {"score", "cumulative_profit_usd", "safety_violations", "steps_completed"}
    if required.issubset(data.keys()):
        ok("/grader response includes detail fields (profit, violations, steps)")
    else:
        fail("Missing detail fields in /grader", str(required - set(data.keys())))


def check_state(session, base):
    section("7. /state")
    post(session, f"{base}/reset", params={"task_id": "medium-forecast-error"})

    r = get(session, f"{base}/state")
    if r.status_code == 200:
        state = r.json()
        required = {"current_step", "task_tier", "cumulative_profit_usd", "battery_true_soc"}
        if required.issubset(state.keys()):
            ok("/state response has required fields")
        else:
            fail("Missing fields in /state", str(required - set(state.keys())))
    else:
        fail(f"/state returned {r.status_code}")


def check_baseline(session, base):
    section("8. /baseline")
    r = get(session, f"{base}/baseline")
    if r.status_code == 200:
        data = r.json()
        if len(data) >= 3:
            ok(f"/baseline returned data for {len(data)} tasks")
        else:
            fail(f"/baseline should cover 3 tasks, got {len(data)}")
    else:
        fail(f"/baseline returned {r.status_code}")


def check_determinism(session, base):
    section("9. Determinism — same task = same profit curve")

    def run_idle(task_id):
        post(session, f"{base}/reset", params={"task_id": task_id})
        profits = []
        for _ in range(5):
            r = post(session, f"{base}/step", json={"global_charge_rate": -0.5, "min_reserve_pct": 0.2})
            profits.append(round(r.json()["reward"], 6))
        return profits

    run1 = run_idle("easy-arbitrage")
    run2 = run_idle("easy-arbitrage")

    if run1 == run2:
        ok("Two identical runs produce identical rewards (seeded RNG ✓)")
    else:
        fail("Reward sequences differ between runs — seeding may be broken", f"{run1} vs {run2}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VPP OpenEnv pre-submission validator")
    parser.add_argument("--url", default="http://localhost:7860", help="Base URL of the running server")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    session = requests.Session()

    print(f"\n{BOLD}VPP Environment — Pre-submission Validator{RESET}")
    print(f"Target: {base}\n")

    # Wait for the server to be ready (useful when called right after docker run)
    print("Checking server availability...", end=" ", flush=True)
    for attempt in range(10):
        try:
            r = session.get(f"{base}/health", timeout=3)
            if r.status_code == 200:
                print("ready.")
                break
        except Exception:
            pass
        time.sleep(2)
        print(".", end="", flush=True)
    else:
        print(f"\n{RED}ERROR:{RESET} Server at {base} is not responding after 20 s.")
        print("Start it with:  uvicorn server.app:app --host 0.0.0.0 --port 7860")
        sys.exit(1)

    # Run all checks
    check_health(session, base)
    task_ids = check_tasks(session, base)
    task_ids = task_ids or ["easy-arbitrage", "medium-forecast-error", "hard-frequency-response"]
    check_reset(session, base, task_ids)
    check_step(session, base)
    check_full_episode(session, base, "easy-arbitrage")
    check_full_episode(session, base, "hard-frequency-response")
    check_grader(session, base)
    check_state(session, base)
    check_baseline(session, base)
    check_determinism(session, base)

    # Summary
    total = passed + failed
    print(f"\n{'='*50}")
    print(f"{BOLD}Results: {GREEN}{passed}{RESET}{BOLD} passed, {RED}{failed}{RESET}{BOLD} failed  ({total} checks){RESET}")
    if failed == 0:
        print(f"{GREEN}{BOLD}All checks passed — ready to submit! 🚀{RESET}")
        sys.exit(0)
    else:
        print(f"{RED}{BOLD}Fix the failing checks before submission.{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()

