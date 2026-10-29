"""Replay a saved PPO evaluation and diagnose failures without retraining.

Reward components are observed through the existing logging hook; transition
and reward formulas are not modified. All original evaluation rows must match.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO

import env as task
from baseline_training import make_env, parameter_hash


ROOT = Path(task.__file__).resolve().parent
INITIAL_FIELDS = ("initial_gripper_x", "initial_gripper_y", "initial_object_x",
                  "initial_object_y", "initial_target_x", "initial_target_y")
BUCKETS = ("abs_y <= 0.05", "0.05 < abs_y <= 0.10",
           "0.10 < abs_y <= 0.15", "abs_y > 0.15")


def bucket(y):
    value = abs(y)
    return BUCKETS[0 if value <= 0.05 else 1 if value <= 0.10 else 2 if value <= 0.15 else 3]


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def describe(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "min": float(values.min()), "max": float(values.max())}


@torch.no_grad()
def replay(model, saved_rows):
    all_steps, episodes, paths = [], [], []
    for episode_index, saved in enumerate(saved_rows):
        env = make_env()
        base = env.unwrapped
        observation, _ = env.reset(seed=int(saved["seed"]))
        initial = observation.copy()
        expected = np.array([float(saved[key]) for key in INITIAL_FIELDS], dtype=np.float32)
        np.testing.assert_array_equal(initial, expected)
        states = [initial.copy()]
        steps, components, episode_return = [], [], 0.0

        def record(parts):
            components.append({name: float(value.item()) for name, value in parts.items()})

        try:
            with patch("env._record_reward_components", side_effect=record):
                for step in range(1, task.cfg.episode_steps + 1):
                    before = observation.copy()
                    old_object = base.object_pos.clone()
                    obs_tensor, _ = model.policy.obs_to_tensor(before)
                    normal = model.policy.get_distribution(obs_tensor).distribution
                    mean = normal.mean.cpu().numpy()[0]
                    std = normal.stddev.cpu().numpy()[0]
                    lower_tail = normal.cdf(torch.full_like(normal.mean, -1.0))
                    upper_tail = 1.0 - normal.cdf(torch.ones_like(normal.mean))
                    noise_clip_probability = (lower_tail + upper_tail).cpu().numpy()[0]
                    value = float(model.policy.predict_values(obs_tensor).item())
                    action, _ = model.predict(before, deterministic=True)
                    np.testing.assert_array_equal(action, np.clip(mean, -1.0, 1.0))
                    physical_action = np.clip(env.action(action), -task.cfg.action_limit, task.cfg.action_limit)
                    observation, reward, terminated, truncated, info = env.step(action)
                    assert len(components) == step
                    parts = components[-1]
                    assert abs(parts["total_reward"] - reward) < 1e-6
                    assert abs(sum(v for k, v in parts.items() if k != "total_reward") - reward) < 2e-5
                    # The engine tests contact AFTER moving the gripper but
                    # BEFORE moving the object. End-of-step distance differs.
                    contact = bool((torch.linalg.norm(base.gripper - old_object, dim=-1)
                                    < task.cfg.contact_radius).item())
                    object_delta = observation[2:4] - before[2:4]
                    gripper_delta = observation[:2] - before[:2]
                    motion = float(np.linalg.norm(object_delta))
                    if not contact:
                        assert motion == 0.0
                    old_goal_distance = float(np.linalg.norm(before[2:4] - before[4:6]))
                    goal_progress = old_goal_distance - info["object_target_distance"]
                    target_direction = before[4:6] - before[2:4]
                    alignment = None
                    if motion > 1e-8:
                        alignment = float(np.dot(object_delta, target_direction)
                                          / (motion * max(np.linalg.norm(target_direction), 1e-8)))
                    row = {
                        "episode": episode_index, "seed": int(saved["seed"]), "step": step,
                        "before_gripper_x": float(before[0]), "before_gripper_y": float(before[1]),
                        "before_object_x": float(before[2]), "before_object_y": float(before[3]),
                        "gripper_x": float(observation[0]), "gripper_y": float(observation[1]),
                        "object_x": float(observation[2]), "object_y": float(observation[3]),
                        "target_x": float(observation[4]), "target_y": float(observation[5]),
                        "raw_mean_x": float(mean[0]), "raw_mean_y": float(mean[1]),
                        "policy_std_x": float(std[0]), "policy_std_y": float(std[1]),
                        "normalized_action_x": float(action[0]), "normalized_action_y": float(action[1]),
                        "physical_action_x": float(physical_action[0]), "physical_action_y": float(physical_action[1]),
                        "mean_action_clipped": bool(np.any(np.abs(mean) > 1.0)),
                        "gaussian_clip_probability_x": float(noise_clip_probability[0]),
                        "gaussian_clip_probability_y": float(noise_clip_probability[1]),
                        "value_estimate": value, "active_contact": contact,
                        "close_after_object_update": info["gripper_object_distance"] < task.cfg.contact_radius,
                        "gripper_object_distance": info["gripper_object_distance"],
                        "object_target_distance": info["object_target_distance"],
                        "goal_progress": goal_progress,
                        "object_motion": motion, "object_direction_alignment": alignment,
                        "gripper_motion": float(np.linalg.norm(gripper_delta)),
                        "object_x_progress": float(object_delta[0]),
                        "y_error_before": abs(float(before[3] - before[5])),
                        "y_error_after": abs(float(observation[3] - observation[5])),
                        "reward": reward, "terminated": terminated, "truncated": truncated,
                    }
                    row.update({f"component_{key}": val for key, val in parts.items() if key != "total_reward"})
                    steps.append(row)
                    states.append(observation.copy())
                    episode_return += reward
                    if terminated or truncated:
                        break
        finally:
            env.close()

        assert bool(terminated) == (saved["success"] == "True")
        assert bool(truncated) == (saved["truncated"] == "True")
        assert len(steps) == int(saved["episode_length"])
        assert abs(episode_return - float(saved["episode_return"])) < 1e-5
        assert abs(info["object_target_distance"] - float(saved["final_distance"])) < 1e-6
        contacts = [row for row in steps if row["active_contact"]]
        moving = [row for row in steps if row["object_motion"] > 1e-8]
        initial_distance = float(np.linalg.norm(initial[2:4] - initial[4:6]))
        minimum_distance = min([initial_distance] + [row["object_target_distance"] for row in steps])
        final_distance = steps[-1]["object_target_distance"]
        total_motion = sum(row["object_motion"] for row in steps)
        longest_contact, current_contact, losses = 0, 0, 0
        for index, row in enumerate(steps):
            current_contact = current_contact + 1 if row["active_contact"] else 0
            longest_contact = max(longest_contact, current_contact)
            if index and steps[index-1]["active_contact"] and not row["active_contact"]:
                losses += 1
        if terminated:
            category = "success"
        elif not contacts:
            category = "no_contact"
        elif total_motion < 1e-5:
            category = "contact_without_object_motion"
        elif final_distance > initial_distance + 0.01:
            category = "net_motion_away_from_goal"
        elif final_distance - minimum_distance > 0.05:
            category = "progress_then_regression"
        else:
            category = "partial_progress_without_success"
        summary = {
            "episode": episode_index, "seed": int(saved["seed"]),
            **{key: float(initial[index]) for index, key in enumerate(INITIAL_FIELDS)},
            "y_offset_bucket": bucket(initial[3]), "category": category,
            "success": bool(terminated), "truncated": bool(truncated),
            "steps": len(steps), "episode_return": episode_return,
            "first_contact_step": contacts[0]["step"] if contacts else None,
            "contact_steps": len(contacts), "contact_fraction": len(contacts)/len(steps),
            "longest_contact_streak": longest_contact, "contact_loss_events": losses,
            "moving_object_steps": len(moving), "object_path_length": total_motion,
            "initial_distance": initial_distance, "final_distance": final_distance,
            "minimum_distance": minimum_distance, "goal_distance_reduction": initial_distance-final_distance,
            "regression_from_best": final_distance-minimum_distance,
            "final_abs_y_error": abs(float(observation[3]-observation[5])),
            "initial_abs_y_error": abs(float(initial[3]-initial[5])),
            "last_five_goal_progress": sum(row["goal_progress"] for row in steps[-5:]),
            "last_five_object_motion": sum(row["object_motion"] for row in steps[-5:]),
            "mean_clipped_steps": sum(row["mean_action_clipped"] for row in steps),
            "mean_normalized_action_x": float(np.mean([row["normalized_action_x"] for row in steps])),
            "mean_normalized_action_y": float(np.mean([row["normalized_action_y"] for row in steps])),
            "mean_moving_alignment": float(np.mean([row["object_direction_alignment"] for row in moving])) if moving else None,
        }
        summary.update({f"sum_{key}": sum(part[key] for part in components)
                        for key in components[0] if key != "total_reward"})
        episodes.append(summary)
        all_steps.extend(steps)
        paths.append(np.stack(states))
    return episodes, all_steps, paths


def summarize(episodes, steps):
    contact_episodes = [row for row in episodes if row["contact_steps"]]
    moving_steps = [row for row in steps if row["object_motion"] > 1e-8]
    components = {key.removeprefix("sum_"): float(np.mean([row[key] for row in episodes]))
                  for key in episodes[0] if key.startswith("sum_")}
    groups = {}
    for name in BUCKETS:
        selected = [row for row in episodes if row["y_offset_bucket"] == name]
        groups[name] = {"count": len(selected), "success_count": sum(row["success"] for row in selected),
                        "contact_count": sum(row["contact_steps"] > 0 for row in selected),
                        "mean_final_distance": float(np.mean([row["final_distance"] for row in selected])) if selected else None}
    final_steps = [row for row in steps if row["step"] == task.cfg.episode_steps]
    signed_groups = {}
    for sign, name in ((1, "positive_initial_y"), (-1, "negative_initial_y")):
        selected = [row for row in episodes if sign * row["initial_object_y"] > 0]
        ids = {row["episode"] for row in selected}
        final_selected = [row for row in final_steps if row["episode"] in ids]
        signed_groups[name] = {
            "count": len(selected),
            "mean_initial_y": float(np.mean([row["initial_object_y"] for row in selected])) if selected else None,
            "mean_final_y": float(np.mean([row["object_y"] for row in final_selected])) if final_selected else None,
            "episodes_with_worse_abs_y": sum(row["final_abs_y_error"] > row["initial_abs_y_error"] for row in selected),
        }
    timeline = []
    for index in range(1, task.cfg.episode_steps + 1):
        selected = [row for row in steps if row["step"] == index]
        timeline.append({"step": index, **{
            f"mean_{key}": float(np.mean([row[key] for row in selected]))
            for key in ("object_target_distance", "normalized_action_x", "normalized_action_y", "object_y", "object_motion")
        }})
    return {
        "episodes": len(episodes), "total_steps": len(steps),
        "original_evaluation_reproduced": True,
        "success_count": sum(row["success"] for row in episodes),
        "timeout_count": sum(row["truncated"] for row in episodes),
        "episodes_with_contact": len(contact_episodes),
        "episodes_with_object_motion": sum(row["object_path_length"] > 1e-5 for row in episodes),
        "failure_categories": dict(Counter(row["category"] for row in episodes)),
        "initial_distance": describe([row["initial_distance"] for row in episodes]),
        "final_distance": describe([row["final_distance"] for row in episodes]),
        "episode_return": describe([row["episode_return"] for row in episodes]),
        "first_contact_step": describe([row["first_contact_step"] for row in contact_episodes]) if contact_episodes else None,
        "contact_steps": describe([row["contact_steps"] for row in episodes]),
        "contact_loss_events": describe([row["contact_loss_events"] for row in episodes]),
        "object_path_length": describe([row["object_path_length"] for row in episodes]),
        "goal_distance_reduction": describe([row["goal_distance_reduction"] for row in episodes]),
        "initial_abs_y_error": describe([row["initial_abs_y_error"] for row in episodes]),
        "final_abs_y_error": describe([row["final_abs_y_error"] for row in episodes]),
        "episodes_with_worse_abs_y": sum(row["final_abs_y_error"] > row["initial_abs_y_error"] for row in episodes),
        "episodes_with_negative_final_y": sum(row["object_y"] < 0 for row in final_steps),
        "episodes_with_last_five_regression": sum(row["last_five_goal_progress"] < 0 for row in episodes),
        "signed_y_groups": signed_groups,
        "last_five_goal_progress": describe([row["last_five_goal_progress"] for row in episodes]),
        "last_five_object_motion": describe([row["last_five_object_motion"] for row in episodes]),
        "mean_action_clipped_steps": sum(row["mean_action_clipped"] for row in steps),
        "mean_action_clipped_fraction": sum(row["mean_action_clipped"] for row in steps) / len(steps),
        "mean_gaussian_sample_clip_probability_x": float(np.mean([row["gaussian_clip_probability_x"] for row in steps])),
        "mean_gaussian_sample_clip_probability_y": float(np.mean([row["gaussian_clip_probability_y"] for row in steps])),
        "sample_clip_probability_scope": "Analytic Gaussian tail probabilities at states on deterministic evaluation paths, not measured training action frequencies.",
        "mean_moving_direction_alignment": float(np.mean([row["object_direction_alignment"] for row in moving_steps])) if moving_steps else None,
        "reward_components_mean_episode_sum": components,
        "y_offset_buckets": groups,
        "per_step_means": timeline,
        "classification_rules": {
            "no_contact": "No active contact at the engine's contact-test instant.",
            "contact_without_object_motion": "Contact occurred but summed object displacement was below 1e-5.",
            "net_motion_away_from_goal": "Final distance exceeds initial distance by over 0.01.",
            "progress_then_regression": "After earlier rules, final distance exceeds best distance by over 0.05.",
            "partial_progress_without_success": "Other failures with contact and object motion; inspect continuous metrics rather than treating this heuristic label as a cause.",
        },
        "training_performed": False,
    }


def choose_examples(episodes):
    chosen = []
    for group in BUCKETS:
        candidates = [row for row in episodes if row["y_offset_bucket"] == group]
        if candidates:
            median = np.median([row["final_distance"] for row in candidates])
            chosen.append(min(candidates, key=lambda row: abs(row["final_distance"]-median))["episode"])
    for row in sorted(episodes, key=lambda row: row["final_distance"]):
        if row["episode"] not in chosen:
            chosen.append(row["episode"])
            break
    for row in sorted(episodes, key=lambda row: row["final_distance"], reverse=True):
        if row["episode"] not in chosen:
            chosen.append(row["episode"])
            break
    return chosen


def figures(out, episodes, steps, paths):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    blue, orange, green = "#3F6687", "#BB7844", "#54816B"
    by_episode = [[row for row in steps if row["episode"] == index] for index in range(len(episodes))]
    ticks = np.arange(1, task.cfg.episode_steps + 1)
    def matrix(key):
        return np.array([[rows[index][key] if index < len(rows) else np.nan
                          for index in range(task.cfg.episode_steps)] for rows in by_episode], dtype=float)
    def band(ax, key, color, label):
        values = matrix(key)
        ax.plot(ticks, np.nanmedian(values, axis=0), color=color, label=label)
        ax.fill_between(ticks, np.nanquantile(values, .25, axis=0), np.nanquantile(values, .75, axis=0), color=color, alpha=.18)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8), constrained_layout=True)
    band(axes[0,0], "object_target_distance", blue, "Median; shaded IQR")
    axes[0,0].axhline(.07, color=green, linestyle="--", label="Success threshold")
    axes[0,0].set(title="Object-to-target distance", ylabel="Distance")
    band(axes[0,1], "gripper_object_distance", orange, "End-of-step distance")
    axes[0,1].axhline(task.cfg.contact_radius, color=green, linestyle="--", label="Contact radius")
    axes[0,1].set(title="Gripper-to-object distance", ylabel="Distance")
    axes[0,2].plot(ticks, matrix("active_contact").mean(axis=0), color=blue, label="Active contact")
    axes[0,2].set(title="Contact at the engine test instant", ylabel="Fraction of episodes", ylim=(-.02,1.05))
    band(axes[1,0], "normalized_action_x", blue, "Action x")
    band(axes[1,0], "normalized_action_y", orange, "Action y")
    axes[1,0].axhline(0, color="0.65", linewidth=.8)
    axes[1,0].set(title="Deterministic normalized actions", ylabel="Action")
    means = summarize(episodes, steps)["reward_components_mean_episode_sum"]
    ordered = sorted(means, key=means.get)
    axes[1,1].barh([name.removesuffix('_reward').replace('_', ' ') for name in ordered],
                  [means[name] for name in ordered], color=[orange if means[name] < 0 else blue for name in ordered])
    axes[1,1].axvline(0, color="0.6", linewidth=.8)
    axes[1,1].set(title="Mean episode reward contributions", xlabel="Sum across the episode")
    axes[1,1].tick_params(axis="y", labelsize=8)
    axes[1,2].scatter([abs(row["initial_object_y"]) for row in episodes],
                      [row["final_distance"] for row in episodes], s=24, color=blue, alpha=.75)
    axes[1,2].axhline(.07, color=green, linestyle="--", label="Success threshold")
    axes[1,2].set(title="Failure distance vs initial y-offset", xlabel="Initial absolute object y", ylabel="Final distance")
    for ax in (axes[0,0],axes[0,1],axes[0,2],axes[1,0]):
        ax.set_xlabel("Environment step")
        ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.16)
    fig.suptitle("Saved PPO policy: 100 original evaluation scenarios, no retraining", fontsize=14)
    fig.savefig(out / "overview.png", dpi=160)
    plt.close(fig)

    selected = choose_examples(episodes)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), constrained_layout=True)
    for ax, index in zip(axes.flat, selected):
        states, episode, rows = paths[index], episodes[index], by_episode[index]
        ax.plot(states[:,0], states[:,1], color=blue, marker=".", markersize=4, label="Gripper")
        ax.plot(states[:,2], states[:,3], color=orange, marker=".", markersize=4, label="Object")
        ax.scatter(states[0,0], states[0,1], marker="o", s=45, facecolors="none", edgecolors=blue)
        ax.scatter(states[0,2], states[0,3], marker="s", s=40, facecolors="none", edgecolors=orange)
        ax.scatter(states[-1,0], states[-1,1], marker="x", s=45, color=blue)
        ax.scatter(states[-1,2], states[-1,3], marker="x", s=45, color=orange)
        ax.scatter(states[0,4], states[0,5], marker="*", s=110, color=green, label="Target")
        ax.add_patch(plt.Circle(states[0,4:6], .07, fill=False, color=green, alpha=.55))
        if episode["first_contact_step"] is not None:
            contact_step = episode["first_contact_step"]
            ax.annotate(f"contact @ {contact_step}", xy=states[contact_step,:2],
                        xytext=(4,14), textcoords="offset points", fontsize=8)
        ax.set_title(f"Seed {episode['seed']} | initial y={episode['initial_object_y']:+.3f}\n"
                     f"final distance={episode['final_distance']:.3f}; contact={episode['contact_steps']}/35", fontsize=10)
        ax.set(xlim=(-1,1), ylim=(-.65,.65), xlabel="x", ylabel="y")
        ax.set_aspect("equal", adjustable="box")
        # Expand the y range if any selected trajectory would otherwise be cut.
        low, high = float(states[:,:4][:, [1,3]].min()), float(states[:,:4][:, [1,3]].max())
        if low < -.65 or high > .65:
            ax.set_ylim(min(-.65, low-.05), max(.65, high+.05))
        ax.grid(alpha=.18)
    axes.flat[0].legend(loc="lower right", fontsize=8)
    for ax in list(axes.flat)[len(selected):]:
        ax.set_visible(False)
    fig.suptitle("Representative failures: one median case per y bucket, plus best and worst distances", fontsize=12)
    fig.savefig(out / "failure_trajectories.png", dpi=170)
    plt.close(fig)
    return [episodes[index]["seed"] for index in selected]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.run_dir is None:
        runs = list((ROOT / "training_runs/v0.4/ppo").glob("*/model.zip"))
        if len(runs) != 1:
            parser.error("Specify --run-dir when there is not exactly one saved PPO run.")
        run = runs[0].parent
    else:
        run = args.run_dir.resolve()
    config = json.loads((run / "config.json").read_text())
    for name, digest in config["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, f"Training source changed: {name}"
    for key, value in config["environment_config"].items():
        if key != "reward_debug_enabled":
            assert asdict(task.cfg)[key] == value, f"Environment setting changed: {key}"
    with (run / "evaluation.csv").open(newline="") as handle:
        saved = list(csv.DictReader(handle))
    torch.set_num_threads(1)
    model = PPO.load(run / "model.zip", device="cpu")
    assert not model.policy.squash_output, "This action-clipping diagnostic expects the original Gaussian PPO policy."
    model.policy.set_training_mode(False)
    original_hash = parameter_hash(model)
    checkpoint_hash = hashlib.sha256((run / "model.zip").read_bytes()).hexdigest()
    original_debug = task.cfg.reward_debug_enabled
    task.cfg.reward_debug_enabled = False
    try:
        episodes, steps, paths = replay(model, saved)
    finally:
        task.cfg.reward_debug_enabled = original_debug
    assert parameter_hash(model) == original_hash
    assert hashlib.sha256((run / "model.zip").read_bytes()).hexdigest() == checkpoint_hash
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = args.output_dir or ROOT / "results/ppo_failure_diagnosis" / stamp
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "episodes.csv", episodes)
    write_csv(out / "steps.csv", steps)
    report = summarize(episodes, steps)
    report.update({"original_run": str(run), "model_sha256": checkpoint_hash,
                   "model_parameters_unchanged": True,
                   "diagnostic_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "source_sha256": config["source_sha256"]})
    report["plotted_seeds"] = figures(out, episodes, steps, paths)
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    print("Saved diagnosis:", out, flush=True)


if __name__ == "__main__":
    main()
