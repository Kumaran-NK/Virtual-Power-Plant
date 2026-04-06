# train_rl.py
"""
Curriculum RL training for the VPP environment — Extended Edition.

5-phase curriculum:
  Phase 1 (easy-arbitrage,          200k steps) — basic buy-low-sell-high
  Phase 2 (medium-forecast-error,   150k steps) — heatwave + adversarial demand
  Phase 3 (hard-frequency-response, 150k steps) — spike planning + reserve mgmt
  Phase 4 (expert-demand-response,  150k steps) — DR auctions + P2P + carbon
  Phase 5 (islanding-emergency,     100k steps) — grid islanding stress test

Total: 750k environment steps.

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860   # terminal 1
  python train_rl.py                                    # terminal 2
"""

import os
import time
import requests

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from gymwrapper import VppGymEnv

SERVER_URL      = os.getenv("VPP_SERVER_URL", "http://localhost:7860")
TENSORBOARD_LOG = "./vpp_tensorboard/"
CHECKPOINT_DIR  = "./checkpoints/"


def make_env(task_id: str):
    def _init():
        e = VppGymEnv(base_url=SERVER_URL, task_id=task_id)
        return Monitor(e)
    return _init


def evaluate(model: PPO, task_id: str, n_episodes: int = 3) -> dict:
    rewards       = []
    pareto_scores = []

    for _ in range(n_episodes):
        env = VppGymEnv(base_url=SERVER_URL, task_id=task_id)
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            ep_reward += float(reward)
            if truncated:
                break
        env.close()

        try:
            resp = requests.get(f"{SERVER_URL}/grader", timeout=5)
            pareto = resp.json()
            pareto_scores.append(pareto.get("aggregate_score", 0.0))
        except Exception:
            pareto_scores.append(0.0)

        rewards.append(ep_reward)

    return {
        "task_id":              task_id,
        "mean_reward":          round(float(np.mean(rewards)), 2),
        "std_reward":           round(float(np.std(rewards)), 2),
        "mean_pareto_score":    round(float(np.mean(pareto_scores)), 4),
    }


def train_phase(model: PPO, task_id: str, total_timesteps: int, phase_name: str) -> PPO:
    print(f"\n{'='*65}")
    print(f"  Phase: {phase_name}  |  Task: {task_id}  |  Steps: {total_timesteps:,}")
    print(f"{'='*65}")

    train_env = DummyVecEnv([make_env(task_id)])
    train_env = VecMonitor(train_env)
    model.set_env(train_env)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, total_timesteps // 5),
        save_path=os.path.join(CHECKPOINT_DIR, task_id),
        name_prefix=f"vpp_ppo_{task_id}",
        verbose=0,
    )

    t0 = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_cb,
        reset_num_timesteps=False,
        progress_bar=True,
    )
    elapsed = time.time() - t0
    print(f"  Phase complete in {elapsed/60:.1f} min.")
    train_env.close()
    return model


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_LOG, exist_ok=True)

    try:
        resp = requests.get(f"{SERVER_URL}/health", timeout=5)
        assert resp.status_code == 200
        print(f"Server reachable at {SERVER_URL}.")
    except Exception as e:
        print(f"ERROR: Cannot reach VPP server.\nStart with: uvicorn server.app:app --host 0.0.0.0 --port 7860\nDetails: {e}")
        return

    init_env = DummyVecEnv([make_env("easy-arbitrage")])
    init_env = VecMonitor(init_env)

    model = PPO(
        "MlpPolicy",
        init_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=TENSORBOARD_LOG,
        policy_kwargs={"net_arch": [256, 256, 128]},   # deeper net for complex multi-obj task
    )
    init_env.close()

    # 5-phase curriculum
    model = train_phase(model, "easy-arbitrage",          200_000, "Phase 1 — Easy")
    model = train_phase(model, "medium-forecast-error",   150_000, "Phase 2 — Medium")
    model = train_phase(model, "hard-frequency-response", 150_000, "Phase 3 — Hard")
    model = train_phase(model, "expert-demand-response",  150_000, "Phase 4 — Expert (DR+P2P)")
    model = train_phase(model, "islanding-emergency",     100_000, "Phase 5 — Islanding")

    model.save("vpp_ppo_final")
    print("\nFinal model saved → vpp_ppo_final.zip")

    print("\n" + "="*65)
    print("  FINAL MULTI-OBJECTIVE EVALUATION")
    print("="*65)
    tasks = [
        "easy-arbitrage",
        "medium-forecast-error",
        "hard-frequency-response",
        "expert-demand-response",
        "islanding-emergency",
    ]
    for task in tasks:
        r = evaluate(model, task, n_episodes=3)
        bar = "█" * int(r["mean_pareto_score"] * 20)
        print(
            f"  {r['task_id']:<35}"
            f"  reward={r['mean_reward']:>8.2f} ± {r['std_reward']:<7.2f}"
            f"  pareto={r['mean_pareto_score']:.4f}  {bar}"
        )

    print("\nTensorBoard: tensorboard --logdir ./vpp_tensorboard/")
    print("Checkpoints: ./checkpoints/")


if __name__ == "__main__":
    main()