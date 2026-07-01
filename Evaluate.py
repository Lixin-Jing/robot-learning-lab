import os

import torch

from env import cfg, make_obs, env_step
from Policy import Policy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    os.makedirs("results", exist_ok=True)

    policy = Policy().to(cfg.device)
    policy.load_state_dict(torch.load("checkpoints/policy.pt", map_location=cfg.device))
    policy.eval()

    done= torch.zeros(1,dtype=bool,device=cfg.device)
    with torch.no_grad():
        gripper = torch.tensor([[-0.75, 0.0]], device=cfg.device)
        object_pos = torch.tensor([[-0.25, 0.0]], device=cfg.device)
        target = torch.tensor([[0.75, 0.0]], device=cfg.device)

        gripper_path = [gripper[0].detach().cpu().clone()]
        object_path = [object_pos[0].detach().cpu().clone()]

        for _ in range(cfg.episode_steps):
            obs = make_obs(gripper, object_pos, target)
            action = policy(obs)
            already_done= done
            action =torch.where(already_done.unsqueeze(-1),torch.zeros_like(action),action)
            gripper, object_pos, reward, success = env_step(gripper, object_pos, target, action)
            done=done|success

            gripper_path.append(gripper[0].detach().cpu().clone())
            object_path.append(object_pos[0].detach().cpu().clone())

        final_distance = torch.linalg.norm(object_pos - target, dim=-1).item()
        reached_goal = done.item()
        final_gripper_object_distance = torch.linalg.norm(gripper - object_pos, dim=-1).item()

    print("===== Evaluation Result =====")
    print(f"Final object position: x={object_pos[0, 0].item():.3f}, y={object_pos[0, 1].item():.3f}")
    print(f"Target position:       x={target[0, 0].item():.3f}, y={target[0, 1].item():.3f}")
    print(f"Final distance to target: {final_distance:.3f}")
    print(f"Final gripper-object distance: {final_gripper_object_distance:.3f}")
    print(f"Reached goal: {reached_goal}")



    image_path = "results/trajectory_after_training.png"

    gripper_xy = torch.stack(gripper_path).numpy()
    object_xy = torch.stack(object_path).numpy()
    target_xy = target[0].detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(gripper_xy[:, 0], gripper_xy[:, 1], marker="o", markersize=3, label="gripper path")
    ax.plot(object_xy[:, 0], object_xy[:, 1], marker="s", markersize=3, label="object path")
    ax.scatter([target_xy[0]], [target_xy[1]], s=200, marker="*", label="target")
    ax.scatter([gripper_xy[0, 0]], [gripper_xy[0, 1]], s=100, marker="o", label="gripper start")
    ax.scatter([object_xy[0, 0]], [object_xy[0, 1]], s=100, marker="s", label="object start")

    ax.set_title("RL Manipulation: Learned Push-to-Target Behavior")
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.7, 0.7)
    ax.grid(True)
    ax.legend(loc="upper left")
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(image_path, dpi=200)
    plt.close(fig)

    print(f"Saved trajectory image: {image_path}")


if __name__ == "__main__":
    main()
