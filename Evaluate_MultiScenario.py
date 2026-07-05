import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import glob

from env import cfg, make_obs, env_step, reset_env
from Policy import Policy


EVAL_SEED = 0
LARGE_Y_OFFSET_THRESHOLD = 0.15

Y_OFFSET_BUCKETS = [
    ("abs_y <= 0.05", 0.0, 0.05),
    ("0.05 < abs_y <= 0.10", 0.05, 0.10),
    ("0.10 < abs_y <= 0.15", 0.10, 0.15),
    ("abs_y > 0.15", 0.15, float("inf")),
]


def get_y_offset_bucket(abs_y):
    for bucket_name, lower_bound, upper_bound in Y_OFFSET_BUCKETS:
        if lower_bound == 0.0:
            if abs_y <= upper_bound:
                return bucket_name
        elif lower_bound < abs_y <= upper_bound:
            return bucket_name

    return "unknown"


def save_failure_trajectory_plot(scenario_id, gripper_path, object_path, target):
    os.makedirs("results/failure_cases", exist_ok=True)

    gripper_path = torch.stack(gripper_path).numpy()
    object_path = torch.stack(object_path).numpy()
    target = target.numpy()

    plot_path = f"results/failure_cases/scenario_{scenario_id:03d}.png"

    plt.figure(figsize=(6, 6))
    plt.plot(gripper_path[:, 0], gripper_path[:, 1], marker="o", label="gripper path")
    plt.plot(object_path[:, 0], object_path[:, 1], marker="x", label="object path")
    plt.scatter(target[0], target[1], marker="*", s=150, label="target")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Failed Scenario {scenario_id}")
    plt.xlim(-1.0, 1.0)
    plt.ylim(-0.5, 0.5)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return plot_path


def evaluate_one_scenario(policy, scenario_id):
    gripper, object_pos, target = reset_env(1)
    initial_gripper = gripper[0].detach().cpu().clone()
    initial_object = object_pos[0].detach().cpu().clone()
    initial_target = target[0].detach().cpu().clone()
    done = torch.zeros(1, dtype=torch.bool, device=cfg.device)

    gripper_path = [initial_gripper]
    object_path = [initial_object]

    for step in range(cfg.episode_steps):
        obs = make_obs(gripper, object_pos, target)
        action = policy(obs)

        already_done = done
        action = torch.where(already_done.unsqueeze(-1), torch.zeros_like(action), action)

        gripper, object_pos, reward, success = env_step(gripper, object_pos, target, action)

        done = done | success

        gripper_path.append(gripper[0].detach().cpu().clone())
        object_path.append(object_pos[0].detach().cpu().clone())

    final_distance_object_target = torch.linalg.norm(object_pos - target, dim=-1).item()
    final_distance_gripper_object = torch.linalg.norm(gripper - object_pos, dim=-1).item()

    saved_failure_plot = ""
    abs_initial_object_y = abs(initial_object[1].item())
    y_offset_bucket = get_y_offset_bucket(abs_initial_object_y)
    is_large_y_offset = abs_initial_object_y > LARGE_Y_OFFSET_THRESHOLD

    if not done.item() and is_large_y_offset:
        saved_failure_plot = save_failure_trajectory_plot(
            scenario_id=scenario_id,
            gripper_path=gripper_path,
            object_path=object_path,
            target=initial_target,
        )

    return {
        "scenario_id": scenario_id,
        "success": int(done.item()),
        "initial_gripper_x": initial_gripper[0].item(),
        "initial_gripper_y": initial_gripper[1].item(),
        "initial_object_x": initial_object[0].item(),
        "initial_object_y": initial_object[1].item(),
        "abs_initial_object_y": abs_initial_object_y,
        "y_offset_bucket": y_offset_bucket,
        "large_y_offset": int(is_large_y_offset),
        "initial_target_x": initial_target[0].item(),
        "initial_target_y": initial_target[1].item(),
        "final_distance_object_target": final_distance_object_target,
        "final_distance_gripper_object": final_distance_gripper_object,
        "saved_failure_plot": saved_failure_plot,
    }


def main():
    random.seed(EVAL_SEED)
    np.random.seed(EVAL_SEED)
    torch.manual_seed(EVAL_SEED)

    os.makedirs("results/failure_cases", exist_ok=True)
    for old_plot in glob.glob("results/failure_cases/*.png"):
        os.remove(old_plot)

    policy = Policy().to(cfg.device)
    policy.load_state_dict(torch.load("checkpoints/policy.pt", map_location=cfg.device))
    policy.eval()

    num_scenarios = 100
    results = []

    with torch.no_grad():
        for scenario_id in range(num_scenarios):
            result = evaluate_one_scenario(policy, scenario_id)
            results.append(result)

    successes = [r["success"] for r in results]
    final_distances = [r["final_distance_object_target"] for r in results]

    success_rate = np.mean(successes)
    mean_final_distance = np.mean(final_distances)
    median_final_distance = np.median(final_distances)
    worst_final_distance = np.max(final_distances)
    saved_failure_plots = [r["saved_failure_plot"] for r in results if r["saved_failure_plot"]]

    total_failures = sum(1 for r in results if r["success"] == 0)
    large_y_failures = sum(
        1 for r in results if r["success"] == 0 and r["large_y_offset"] == 1
    )
    small_y_failures = total_failures - large_y_failures

    print(f"Number of scenarios: {len(results)}")
    print(f"Evaluation seed: {EVAL_SEED}")
    print(f"Success rate: {success_rate * 100:.2f}%")
    print(f"Mean final distance: {mean_final_distance:.3f}")
    print(f"Median final distance: {median_final_distance:.3f}")
    print(f"Worst final distance: {worst_final_distance:.3f}")
    print(f"Total failures: {total_failures}")
    print(f"Large-y-offset failures: {large_y_failures}")
    print(f"Small-y-offset failures: {small_y_failures}")
    print(f"Saved failure trajectory plots: {len(saved_failure_plots)}")

    print("\nY-offset bucket evaluation:")
    for bucket_name, _, _ in Y_OFFSET_BUCKETS:
        bucket_results = [r for r in results if r["y_offset_bucket"] == bucket_name]

        if not bucket_results:
            print(f"  {bucket_name}:")
            print("    count: 0")
            continue

        bucket_successes = [r["success"] for r in bucket_results]
        bucket_final_distances = [r["final_distance_object_target"] for r in bucket_results]
        bucket_failures = sum(1 for r in bucket_results if r["success"] == 0)

        print(f"  {bucket_name}:")
        print(f"    count: {len(bucket_results)}")
        print(f"    success rate: {np.mean(bucket_successes) * 100:.2f}%")
        print(f"    failures: {bucket_failures}")
        print(f"    mean final distance: {np.mean(bucket_final_distances):.3f}")
        print(f"    median final distance: {np.median(bucket_final_distances):.3f}")

    os.makedirs("results", exist_ok=True)

    with open("results/evaluation_report.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario_id",
                "success",
                "initial_gripper_x",
                "initial_gripper_y",
                "initial_object_x",
                "initial_object_y",
                "abs_initial_object_y",
                "y_offset_bucket",
                "large_y_offset",
                "initial_target_x",
                "initial_target_y",
                "final_distance_object_target",
                "final_distance_gripper_object",
                "saved_failure_plot",
            ],
        )
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
