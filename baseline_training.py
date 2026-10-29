"""Project PPO/SAC entry-point helpers using Stable-Baselines3 algorithms."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from env import cfg
from gym_env import RobotPushingEnv


ROOT = Path(__file__).resolve().parent


def make_env():
    """Expose normalized actions; the existing environment keeps physical units."""
    return gym.wrappers.RescaleAction(
        RobotPushingEnv(), np.float32(-1.0), np.float32(1.0)
    )


def make_ppo_env(num_envs, monitor_file=None):
    """Create independent environments with batched policy inference.

    DummyVecEnv steps the lightweight environments sequentially in one process;
    it does not launch one operating-system process per environment.
    """
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    vector = DummyVecEnv([make_env for _ in range(num_envs)])
    try:
        return VecMonitor(vector, filename=monitor_file, info_keywords=("success",))
    except Exception:
        vector.close()
        raise


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.policy.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class ProgressCallback(BaseCallback):
    def _on_step(self):
        if not np.isfinite(self.locals["new_obs"]).all():
            raise FloatingPointError("Non-finite training observation.")
        if not np.isfinite(self.locals["rewards"]).all():
            raise FloatingPointError("Non-finite training reward.")
        if self.num_timesteps % 4096 == 0:
            print(f"Training progress: {self.num_timesteps} steps", flush=True)
        return True


def evaluate(model, episodes, first_seed, output_file):
    """Evaluate the final checkpoint once; no selection or tuning uses this set."""
    env = make_env()
    rows = []
    try:
        for index in range(episodes):
            seed = first_seed + index
            observation, _ = env.reset(seed=seed)
            initial = observation.copy()
            episode_return = 0.0
            for step in range(1, cfg.episode_steps + 1):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                if not np.isfinite(observation).all() or not np.isfinite(reward):
                    raise FloatingPointError("Non-finite evaluation transition.")
                episode_return += reward
                if terminated or truncated:
                    break
            rows.append({
                "episode": index, "seed": seed,
                "initial_gripper_x": float(initial[0]),
                "initial_gripper_y": float(initial[1]),
                "initial_object_x": float(initial[2]),
                "initial_object_y": float(initial[3]),
                "initial_target_x": float(initial[4]),
                "initial_target_y": float(initial[5]),
                "success": bool(terminated), "truncated": bool(truncated),
                "episode_return": float(episode_return), "episode_length": step,
                "final_distance": info["object_target_distance"],
            })
    finally:
        env.close()
    with output_file.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "episodes": episodes, "first_seed": first_seed,
        "distribution": "Current Gymnasium reset mixture: 50% targeted branch, 50% uniform branch.",
        "deterministic_actions": True,
        "success_count": sum(row["success"] for row in rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "mean_final_distance": float(np.mean([row["final_distance"] for row in rows])),
        "median_final_distance": float(np.median([row["final_distance"] for row in rows])),
        "mean_episode_return": float(np.mean([row["episode_return"] for row in rows])),
        "mean_episode_length": float(np.mean([row["episode_length"] for row in rows])),
    }


def main(algorithm):
    parser = argparse.ArgumentParser(description=f"Train one SB3 {algorithm.upper()} run.")
    parser.add_argument(
        "--timesteps", type=int,
        help="Total environment transitions across all environments. PPO defaults to the smallest complete-rollout budget >= 20480; SAC defaults to 20480.",
    )
    if algorithm == "ppo":
        parser.add_argument("--num-envs", type=int, default=cfg.num_envs)
        parser.add_argument("--rollout-steps", type=int, default=256,
                            help="Steps collected per environment before a PPO update.")
    else:
        parser.set_defaults(num_envs=1, rollout_steps=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=10000)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.num_envs < 1 or args.eval_episodes < 1:
        parser.error("--num-envs and --eval-episodes must be positive")
    rollout_size = None
    if algorithm == "ppo":
        if args.rollout_steps < 1:
            parser.error("--rollout-steps must be positive")
        rollout_size = args.num_envs * args.rollout_steps
        if rollout_size < 2:
            parser.error("PPO requires at least two samples per rollout.")
        if args.timesteps is None:
            args.timesteps = ((20480 + rollout_size - 1) // rollout_size) * rollout_size
        if args.timesteps < 1 or args.timesteps % rollout_size:
            parser.error(
                f"With {args.num_envs} environments and {args.rollout_steps} steps per environment, "
                f"--timesteps must be a positive multiple of {rollout_size}."
            )
    elif args.timesteps is None:
        args.timesteps = 20480
    if args.timesteps < 1:
        parser.error("--timesteps must be positive")
    torch.set_num_threads(1)
    cfg.reward_debug_enabled = False
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = args.output_dir or ROOT / "training_runs/v0.4" / algorithm / f"{stamp}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=False)

    hyperparameters = {
        "learning_rate": 3e-4, "gamma": cfg.gamma,
        "policy_kwargs": {"net_arch": [64, 64]},
        "seed": args.seed, "device": "cpu", "verbose": 0,
    }
    if algorithm == "ppo":
        algorithm_class = PPO
        hyperparameters.update(n_steps=args.rollout_steps, batch_size=64, n_epochs=10,
                               gae_lambda=0.95, clip_range=0.2, ent_coef=0.0)
        activation = torch.nn.Tanh
    elif algorithm == "sac":
        algorithm_class = SAC
        hyperparameters.update(buffer_size=100000, learning_starts=1000,
                               batch_size=128, tau=0.005, train_freq=1,
                               gradient_steps=1, ent_coef="auto", target_entropy="auto")
        activation = torch.nn.ReLU
    else:
        raise ValueError(algorithm)
    serializable = {**hyperparameters, "policy_kwargs": {
        "net_arch": [64, 64], "activation_fn": activation.__name__
    }}
    config = {
        "algorithm": algorithm.upper(), "implementation": "stable-baselines3",
        "requested_timesteps": args.timesteps,
        "num_envs": args.num_envs,
        "rollout_steps_per_env": args.rollout_steps,
        "rollout_size": rollout_size,
        "rollout_batches": args.timesteps // rollout_size if rollout_size else None,
        "environment_backend": "DummyVecEnv (single process)" if algorithm == "ppo" else "Single Gymnasium environment",
        "hyperparameters": serializable, "environment_config": asdict(cfg),
        "normalized_action_range": [-1, 1],
        "physical_action_range": [-cfg.action_limit, cfg.action_limit],
        "torch_threads": 1,
        "evaluation": {"episodes": args.eval_episodes, "first_seed": args.eval_seed},
        "versions": {name: version(name) for name in (
            "torch", "numpy", "gymnasium", "stable-baselines3"
        )},
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("env.py", "gym_env.py", "baseline_training.py", f"train_{algorithm}.py")},
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    hyperparameters["policy_kwargs"]["activation_fn"] = activation
    if algorithm == "ppo":
        training_env = make_ppo_env(args.num_envs, str(run_dir / "episodes.monitor.csv"))
    else:
        training_env = Monitor(make_env(), filename=str(run_dir / "episodes.monitor.csv"),
                               info_keywords=("success",))
    try:
        model = algorithm_class("MlpPolicy", training_env, **hyperparameters)
        model.set_logger(configure(str(run_dir), ["csv"]))
        before = parameter_hash(model)
        if model.n_envs != args.num_envs:
            raise RuntimeError("The model environment count differs from the requested count.")
        print(f"{algorithm.upper()} training started: {run_dir}", flush=True)
        print(f"Environments: {model.n_envs}; total transitions: {args.timesteps}; rollout size: {rollout_size}", flush=True)
        started = time.perf_counter()
        model.learn(total_timesteps=args.timesteps, callback=ProgressCallback(), log_interval=1)
        elapsed = time.perf_counter() - started
        if model.num_timesteps != args.timesteps:
            raise RuntimeError("Actual training budget differs from the requested budget.")
        if any(not torch.isfinite(p).all() for p in model.policy.parameters()):
            raise FloatingPointError("Non-finite model parameters.")
        after = parameter_hash(model)
        if before == after:
            raise RuntimeError("Training did not change model parameters.")
        model.save(run_dir / "model.zip")
        # Use the reloaded file so the reported result also checks serialization.
        reloaded = algorithm_class.load(run_dir / "model.zip", device="cpu")
        if parameter_hash(reloaded) != after:
            raise RuntimeError("Saved model does not reproduce trained parameters.")
        evaluation = evaluate(reloaded, args.eval_episodes, args.eval_seed,
                              run_dir / "evaluation.csv")
        summary = {
            "algorithm": algorithm.upper(), "implementation": "stable-baselines3",
            "training_seed": args.seed, "actual_timesteps": model.num_timesteps,
            "num_envs": model.n_envs,
            "rollout_size": rollout_size,
            "training_seconds": elapsed,
            "sb3_update_counter": model._n_updates,
            "optimizer_steps": model._n_updates * ((rollout_size + 63) // 64) if rollout_size else model._n_updates,
            "parameters_changed": True, "saved_model_reload_verified": True,
            "evaluation": evaluation,
            "interpretation": "One run with fixed hyperparameters; no multi-seed comparison or tuning. PPO's SB3 update counter counts epochs; optimizer_steps counts minibatch Adam steps.",
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        training_env.close()
    print("Saved run:", run_dir, flush=True)

