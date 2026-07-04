import os
import csv
import torch
import numpy as np

from env import cfg, make_obs, env_step, reset_env
from Policy import Policy


def evaluate_one_scenario(policy, scenario_id):
    gripper, object_pos, target = reset_env(1)
    done = torch.zeros(1, dtype=torch.bool, device=cfg.device)

    for step in range(cfg.episode_steps):
        obs = make_obs(gripper, object_pos, target)
        action = policy(obs)

        already_done = done
        action = torch.where(already_done.unsqueeze(-1), torch.zeros_like(action), action)

        gripper, object_pos, reward, success = env_step(gripper, object_pos, target, action)

        done = done | success

    final_distance_object_target = torch.linalg.norm(object_pos - target, dim=-1).item()
    final_distance_gripper_object = torch.linalg.norm(gripper - object_pos, dim=-1).item()

    return {
        "scenario_id": scenario_id,
        "success": int(done.item()),
        "final_distance_object_target": final_distance_object_target,
        "final_distance_gripper_object": final_distance_gripper_object,
    }


def main():
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

    print(f"Number of scenarios: {len(results)}")
    print(f"Success rate: {success_rate * 100:.2f}%")
    print(f"Mean final distance: {mean_final_distance:.3f}")
    print(f"Median final distance: {median_final_distance:.3f}")
    print(f"Worst final distance: {worst_final_distance:.3f}")

    os.makedirs("results", exist_ok=True)

    with open("results/evaluation_report.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario_id",
                "success",
                "final_distance_object_target",
                "final_distance_gripper_object",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
if __name__ == "__main__":
    main()
