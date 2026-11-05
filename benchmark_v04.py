"""Run the fixed-budget, multi-seed PPO/SAC experiment for v0.4.

This is the experiment entry point, not an additional environment test suite.
Historical training helpers and task dynamics remain unchanged.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from baseline_training import make_env, make_ppo_env, parameter_hash
from env import cfg


ROOT = Path(__file__).resolve().parent
INITIAL_FIELDS = (
    "initial_gripper_x", "initial_gripper_y", "initial_object_x",
    "initial_object_y", "initial_target_x", "initial_target_y",
)
BUCKETS = ("abs_y <= 0.05", "0.05 < abs_y <= 0.10",
           "0.10 < abs_y <= 0.15", "abs_y > 0.15")
SOURCE_NAMES = ("env.py", "gym_env.py", "baseline_training.py",
                "train_ppo.py", "train_sac.py", "benchmark_v04.py")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def bucket(y):
    y = abs(float(y))
    return BUCKETS[0 if y <= .05 else 1 if y <= .10 else 2 if y <= .15 else 3]


def save_initial_states(path, first_seed, count):
    env = make_env()
    rows = []
    try:
        for seed in range(first_seed, first_seed + count):
            obs, _ = env.reset(seed=seed)
            rows.append({"seed": seed, **dict(zip(INITIAL_FIELDS, map(float, obs))),
                         "y_offset_bucket": bucket(obs[3])})
    finally:
        env.close()
    write_csv(path, rows)


def initialize(directory, budget):
    if budget < 65536 or budget % 65536:
        raise ValueError("The budget must be a positive multiple of 65,536.")
    directory.mkdir(parents=True, exist_ok=False)
    cfg.reward_debug_enabled = False
    save_initial_states(directory / "curve_initial_states.csv", 10000, 100)
    save_initial_states(directory / "holdout_initial_states.csv", 20000, 300)
    for seed in (7, 17, 27):
        save_initial_states(directory / f"training_initial_states_seed{seed}.csv", seed, 256)
    source_dir = directory / "sources"
    source_dir.mkdir()
    for name in SOURCE_NAMES:
        (source_dir / name).write_bytes((ROOT / name).read_bytes())
    protocol = {
        "training_seeds": [7, 17, 27], "total_transitions_per_run": budget,
        "algorithms": ["ppo", "sac"],
        "common_evaluation_steps": list(range(0, budget + 1, 65536)),
        "shared_network": {"hidden_layers": [64, 64], "activation": "Tanh"},
        "shared_minibatch_size": 64, "shared_learning_rate": 3e-4,
        "shared_optimizer": {"class": "Adam", "eps": 1e-5, "betas": [.9, .999]},
        "shared_new_transitions_per_collection": 65536,
        "shared_actor_critic_updates_per_collection": 10240,
        "ppo_environments": 256, "ppo_steps_per_environment": 256,
        "sac_environments": 256, "sac_steps_per_environment": 256,
        "torch_threads_per_process": 1,
        "curve_scenarios": 100, "holdout_scenarios": 300,
        "deterministic_evaluation": True,
        "checkpoint_selection": "Final prescribed-budget checkpoint only; no tuning or early stopping.",
        "comparison_scope": "Matched environment count, observations, actions, task, initial RNG streams, interactions, collection schedule, hidden layers, activation, minibatch size, Adam settings, and actor/critic update counts. Algorithm-specific objectives, critic structure, replay, target networks, and entropy mechanisms differ.",
        "evaluation_timing": "At common rollout boundaries, evaluate after optimizer updates; final capture follows the last update.",
        "environment_config": asdict(cfg),
        "versions": {name: version(name) for name in (
            "torch", "numpy", "gymnasium", "stable-baselines3")},
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in SOURCE_NAMES},
        "state_files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(directory.glob("*initial_states*.csv"))},
    }
    write_json(directory / "protocol.json", protocol)
    print("Prepared experiment:", directory, flush=True)


@contextmanager
def preserve_training_randomness(model):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    training_mode = model.policy.training
    try:
        model.policy.set_training_mode(False)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        model.policy.set_training_mode(training_mode)


@torch.no_grad()
def evaluate_states(model, states, output):
    env = make_env()
    rows = []
    try:
        with preserve_training_randomness(model):
            for initial in states:
                obs, _ = env.reset(seed=int(initial["seed"]))
                expected = np.array([float(initial[k]) for k in INITIAL_FIELDS], dtype=np.float32)
                np.testing.assert_array_equal(obs, expected)
                episode_return = 0.0
                for step in range(1, cfg.episode_steps + 1):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    if not np.isfinite(obs).all() or not np.isfinite(reward):
                        raise FloatingPointError("Non-finite evaluation transition")
                    episode_return += reward
                    if terminated or truncated:
                        break
                rows.append({**initial, "success": bool(terminated),
                             "truncated": bool(truncated), "episode_length": step,
                             "episode_return": episode_return,
                             "final_distance": info["object_target_distance"]})
    finally:
        env.close()
    write_csv(output, rows)
    return {
        "episodes": len(rows), "success_count": sum(r["success"] for r in rows),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "mean_final_distance": float(np.mean([r["final_distance"] for r in rows])),
        "mean_episode_return": float(np.mean([r["episode_return"] for r in rows])),
        "mean_episode_length": float(np.mean([r["episode_length"] for r in rows])),
    }


class BenchmarkCallback(BaseCallback):
    def __init__(self, directory, states, evaluation_steps, initial_training_states):
        super().__init__()
        self.directory = directory
        self.states = states
        self.evaluation_steps = set(evaluation_steps)
        self.initial_training_states = initial_training_states
        self.rows = []
        self.captured = set()
        self.evaluation_seconds = 0.0
        self.checkpoint_seconds = 0.0
        (directory / "checkpoints").mkdir()
        (directory / "evaluations").mkdir()

    def capture(self):
        step = self.model.num_timesteps
        if step not in self.evaluation_steps or step in self.captured:
            return
        started = time.perf_counter()
        metrics = evaluate_states(self.model, self.states,
                                  self.directory / "evaluations" / f"step_{step:07d}.csv")
        self.evaluation_seconds += time.perf_counter() - started
        started = time.perf_counter()
        self.model.save(self.directory / "checkpoints" / f"step_{step:07d}.zip")
        self.checkpoint_seconds += time.perf_counter() - started
        self.rows.append({"transitions": step, **metrics})
        self.captured.add(step)
        write_csv(self.directory / "learning_curve.csv", self.rows)
        self.model.logger.record("time/total_timesteps", step)
        for name, value in metrics.items():
            self.model.logger.record("eval/" + name, value)
        if self.model.ep_info_buffer:
            self.model.logger.record("rollout/ep_rew_mean", float(np.mean([r["r"] for r in self.model.ep_info_buffer])))
            self.model.logger.record("rollout/ep_len_mean", float(np.mean([r["l"] for r in self.model.ep_info_buffer])))
        self.model.logger.dump(step=step)
        print(f"{self.directory.name}: evaluated {step:,} transitions; "
              f"success {metrics['success_count']}/{metrics['episodes']}; "
              f"distance {metrics['mean_final_distance']:.6f}", flush=True)

    def _on_training_start(self):
        np.testing.assert_array_equal(self.model._last_obs, self.initial_training_states)
        self.capture()

    def _on_rollout_start(self):
        # SB3 calls this after the previous rollout's optimizer work.
        self.capture()

    def _on_step(self):
        if not np.isfinite(self.locals["new_obs"]).all() or not np.isfinite(self.locals["rewards"]).all():
            raise FloatingPointError("Non-finite training transition")
        if self.num_timesteps % 32768 == 0:
            print(f"{self.directory.name}: collected {self.num_timesteps:,} transitions", flush=True)
        return True


def train(directory, algorithm, seed):
    protocol = json.loads((directory / "protocol.json").read_text())
    if seed not in protocol["training_seeds"]:
        raise ValueError("Seed is not in the saved protocol")
    for name, expected in protocol["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Source changed after protocol initialization: {name}")
    for name, expected in protocol["state_files_sha256"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Saved evaluation states changed: {name}")
    torch.set_num_threads(1)
    cfg.reward_debug_enabled = False
    if asdict(cfg) != protocol["environment_config"]:
        raise RuntimeError("Environment configuration differs from the protocol")
    run = directory / f"{algorithm}_seed{seed}"
    run.mkdir(exist_ok=False)
    parameters = dict(learning_rate=3e-4, gamma=cfg.gamma,
                      policy_kwargs={"net_arch": [64, 64], "activation_fn": torch.nn.Tanh,
                                     "optimizer_class": torch.optim.Adam,
                                     "optimizer_kwargs": {"eps": 1e-5, "betas": (.9, .999)}},
                      seed=seed, device="cpu", verbose=0)
    training_env = make_ppo_env(256, str(run / "episodes.monitor.csv"))
    if algorithm == "ppo":
        constructor = PPO
        parameters.update(n_steps=256, batch_size=64, n_epochs=10,
                          gae_lambda=.95, clip_range=.2, ent_coef=0.0)
    else:
        constructor = SAC
        parameters.update(buffer_size=protocol["total_transitions_per_run"],
                          learning_starts=0, batch_size=64,
                          tau=.005, train_freq=256, gradient_steps=10240,
                          ent_coef="auto", target_entropy="auto")
    saved_parameters = {**parameters, "policy_kwargs": {
        "net_arch": [64, 64],
        "activation_fn": "Tanh", "optimizer_class": "Adam",
        "optimizer_kwargs": {"eps": 1e-5, "betas": [.9, .999]}}}
    write_json(run / "config.json", {"algorithm": algorithm.upper(), "seed": seed,
               "parameters": saved_parameters, "protocol": "../protocol.json"})
    points = list(protocol["common_evaluation_steps"])
    initial_training_states = np.array([
        [float(r[k]) for k in INITIAL_FIELDS]
        for r in read_csv(directory / f"training_initial_states_seed{seed}.csv")], dtype=np.float32)
    callback = BenchmarkCallback(run, read_csv(directory / "curve_initial_states.csv"),
                                 points, initial_training_states)
    try:
        model = constructor("MlpPolicy", training_env, **parameters)
        model.set_logger(configure(str(run), ["csv"]))
        initial_hash = parameter_hash(model)
        started = time.perf_counter()
        model.learn(total_timesteps=protocol["total_transitions_per_run"],
                    callback=callback, log_interval=None)
        callback.capture()
        elapsed = time.perf_counter() - started
        if model.num_timesteps != protocol["total_transitions_per_run"]:
            raise RuntimeError("Training budget mismatch")
        if callback.captured != set(points):
            raise RuntimeError("Some scheduled evaluations were missed")
        if any(not torch.isfinite(p).all() for p in model.policy.parameters()):
            raise FloatingPointError("Non-finite trained parameters")
        trained_hash = parameter_hash(model)
        if trained_hash == initial_hash:
            raise RuntimeError("Policy parameters did not change")
        updates = model._n_updates * 1024 if algorithm == "ppo" else model._n_updates
        if updates != protocol["total_transitions_per_run"] // 65536 * 10240:
            raise RuntimeError("Actor/critic update budget mismatch")
        model.save(run / "model.zip")
        reloaded = constructor.load(run / "model.zip", device="cpu")
        if parameter_hash(reloaded) != trained_hash:
            raise RuntimeError("Model reload mismatch")
        final_evaluation = evaluate_states(reloaded, read_csv(directory / "holdout_initial_states.csv"),
                                           run / "holdout_evaluation.csv")
        summary = {
            "algorithm": algorithm.upper(), "training_seed": seed,
            "actual_timesteps": model.num_timesteps, "num_envs": model.n_envs,
            "training_seconds_excluding_evaluation_and_checkpoints": elapsed - callback.evaluation_seconds - callback.checkpoint_seconds,
            "learn_elapsed_seconds": elapsed,
            "evaluation_seconds_during_learning": callback.evaluation_seconds,
            "checkpoint_seconds_during_learning": callback.checkpoint_seconds,
            "sb3_update_counter": model._n_updates,
            "ppo_minibatch_optimizer_steps": model._n_updates * 1024 if algorithm == "ppo" else None,
            "sac_gradient_updates": model._n_updates if algorithm == "sac" else None,
            "parameters_changed": True, "saved_model_reload_verified": True,
            "initial_parameter_sha256": initial_hash, "final_parameter_sha256": trained_hash,
            "curve_evaluation": callback.rows[-1], "holdout_evaluation": final_evaluation,
            "interpretation": "One preregistered seed; final prescribed-budget model only. Evaluation preserves training RNG and mode.",
        }
        write_json(run / "summary.json", summary)
        print("COMPLETED", run.name, json.dumps(final_evaluation), flush=True)
    finally:
        training_env.close()


def statistics(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)),
            "min": float(values.min()), "max": float(values.max())}


def summarize(directory):
    protocol = json.loads((directory / "protocol.json").read_text())
    per_run, curves, buckets, result = [], [], [], {}
    common_steps = protocol["common_evaluation_steps"]
    for algorithm in protocol["algorithms"]:
        summaries = []
        for seed in protocol["training_seeds"]:
            run = directory / f"{algorithm}_seed{seed}"
            summary = json.loads((run / "summary.json").read_text())
            assert summary["actual_timesteps"] == protocol["total_transitions_per_run"]
            summaries.append(summary)
            curve = read_csv(run / "learning_curve.csv")
            common = [r for r in curve if int(r["transitions"]) in common_steps]
            assert [int(r["transitions"]) for r in common] == common_steps
            successful = [int(r["transitions"]) for r in curve if float(r["success_rate"]) >= .9]
            common_auc = float(np.trapezoid([float(r["success_rate"]) for r in common], common_steps)
                               / protocol["total_transitions_per_run"])
            per_run.append({"algorithm": algorithm.upper(), "seed": seed,
                "transitions": summary["actual_timesteps"],
                "curve_success_rate": summary["curve_evaluation"]["success_rate"],
                "holdout_success_rate": summary["holdout_evaluation"]["success_rate"],
                "holdout_mean_final_distance": summary["holdout_evaluation"]["mean_final_distance"],
                "holdout_mean_return": summary["holdout_evaluation"]["mean_episode_return"],
                "first_observed_90pct_transitions": min(successful) if successful else None,
                "success_auc_common_grid": common_auc})
            curves.extend({"algorithm": algorithm.upper(), "seed": seed, **r} for r in curve)
            holdout = read_csv(run / "holdout_evaluation.csv")
            for name in BUCKETS:
                selected = [r for r in holdout if r["y_offset_bucket"] == name]
                buckets.append({"algorithm": algorithm.upper(), "seed": seed, "bucket": name,
                    "episodes": len(selected), "success_count": sum(r["success"] == "True" for r in selected),
                    "success_rate": sum(r["success"] == "True" for r in selected) / len(selected),
                    "mean_final_distance": float(np.mean([float(r["final_distance"]) for r in selected]))})
        result[algorithm.upper()] = {
            "training_runs": len(summaries),
            "curve_success_rate": statistics([s["curve_evaluation"]["success_rate"] for s in summaries]),
            "holdout_success_rate": statistics([s["holdout_evaluation"]["success_rate"] for s in summaries]),
            "holdout_mean_final_distance": statistics([s["holdout_evaluation"]["mean_final_distance"] for s in summaries]),
            "holdout_mean_return": statistics([s["holdout_evaluation"]["mean_episode_return"] for s in summaries]),
            "success_auc_common_grid": statistics([r["success_auc_common_grid"] for r in per_run if r["algorithm"] == algorithm.upper()]),
        }
    write_csv(directory / "per_run.csv", per_run)
    write_csv(directory / "learning_curves.csv", curves)
    write_csv(directory / "bucket_results.csv", buckets)
    write_json(directory / "aggregate.json", result)
    plot(directory, protocol, curves, buckets)
    print(json.dumps(result, indent=2), flush=True)


def plot(directory, protocol, curves, buckets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 11, "figure.facecolor": "white"})
    colors = {"PPO": "#416C8F", "SAC": "#AE7551"}
    steps = protocol["common_evaluation_steps"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for algorithm, color in colors.items():
        for ax, metric in zip(axes, ("success_rate", "mean_final_distance")):
            values = np.array([[float(next(r[metric] for r in curves if r["algorithm"] == algorithm
                                         and r["seed"] == seed and int(r["transitions"]) == step))
                                for step in steps] for seed in protocol["training_seeds"]])
            mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
            for row in values:
                ax.plot(np.array(steps)/1000, row, color=color, alpha=.16, lw=.8)
            ax.plot(np.array(steps)/1000, mean, color=color, lw=2, marker="o", ms=3, label=algorithm)
            lo, hi = mean-std, mean+std
            if metric == "success_rate":
                lo, hi = np.clip(lo, 0, 1), np.clip(hi, 0, 1)
            ax.fill_between(np.array(steps)/1000, lo, hi, color=color, alpha=.15)
    for ax in axes:
        ax.set_xlabel("Training interactions (thousands)"); ax.grid(alpha=.15); ax.legend(frameon=False)
    axes[0].set(ylabel="Success rate", ylim=(-.02, 1.03), title="Fixed 100-scenario learning curves")
    axes[1].set(ylabel="Mean final target distance", title="Goal convergence")
    fig.suptitle("PPO vs SAC: matched shared training conditions, 3 seeds\nMean ± sample SD; final checkpoints, no selection")
    fig.savefig(directory / "learning_curves.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    x = np.arange(len(BUCKETS))
    for offset, (algorithm, color) in zip((-.18, .18), colors.items()):
        values = np.array([[float(next(r["success_rate"] for r in buckets if r["algorithm"] == algorithm
                                     and r["seed"] == seed and r["bucket"] == name))
                            for name in BUCKETS] for seed in protocol["training_seeds"]])
        ax.bar(x+offset, values.mean(axis=0), width=.34, color=color, label=algorithm)
        mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
        ax.errorbar(x+offset, mean, yerr=np.vstack([np.minimum(std,mean), np.minimum(std,1-mean)]),
                    fmt="none", color="#42494F", capsize=4, lw=1)
    counts = [next(r["episodes"] for r in buckets if r["bucket"] == name) for name in BUCKETS]
    ax.set_xticks(x, [name+f"\n(n={n} per seed)" for name,n in zip(BUCKETS,counts)])
    ax.set(ylabel="Success rate", ylim=(0,1.12), title="Final 300-scenario holdout: initial object y-offset buckets")
    ax.grid(axis="y",alpha=.15); ax.legend(frameon=False)
    fig.savefig(directory / "holdout_buckets.png", dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("init", "train", "summarize"))
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--timesteps", type=int, default=655360)
    parser.add_argument("--algorithm", choices=("ppo", "sac"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    directory = args.experiment_dir.resolve()
    if args.mode == "init":
        initialize(directory, args.timesteps)
    elif args.mode == "train":
        if args.algorithm is None or args.seed is None:
            parser.error("train requires --algorithm and --seed")
        train(directory, args.algorithm, args.seed)
    else:
        summarize(directory)


if __name__ == "__main__":
    main()
