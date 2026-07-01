import csv
import os

import torch
from torch.distributions import Normal

from env import cfg, discounted_returns, env_step, make_obs, reset_env
from Policy import Policy


def main():
    torch.manual_seed(7)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Stage 1 is designed to run locally without CUDA.
    # The actual device is controlled by cfg.device in env.py.
    policy = Policy().to(cfg.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)

    print("===== Device Check =====")
    print("torch:", torch.__version__)
    print("device used:", cfg.device)
    print("cuda available:", torch.cuda.is_available())
    print("Note: Stage 1 can run locally on CPU. GPU/CUDA is not required.")
    print()

    training_history = []

    for iteration in range(1, cfg.train_iterations + 1):
        gripper, object_pos, target = reset_env(cfg.num_envs)

        log_probs = []
        rewards = []
        successes = []
        done= torch.zeros(cfg.num_envs,dtype=torch.bool,device=cfg.device)

        for _ in range(cfg.episode_steps):
            obs = make_obs(gripper, object_pos, target)
            mean_action = policy(obs)

            dist = Normal(mean_action, cfg.action_std)
            raw_action = dist.sample()
            action = torch.clamp(raw_action, -cfg.action_limit, cfg.action_limit)

            already_done=done
            action =torch.where(already_done.unsqueeze(-1), torch.zeros_like(action),action)

            log_prob = dist.log_prob(raw_action).sum(dim=-1)

            gripper, object_pos, reward, success = env_step(gripper, object_pos, target, action)

            reward =torch.where(already_done,torch.zeros_like(reward),reward)
            done = done|success

            log_probs.append(log_prob)
            rewards.append(reward)
            successes.append(done.float())

        log_probs = torch.stack(log_probs)
        rewards = torch.stack(rewards)
        successes = torch.stack(successes)

        returns = discounted_returns(rewards)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = -(log_probs * returns.detach()).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_episode_reward = rewards.sum(dim=0).mean().item()
        final_success_rate = successes[-1].mean().item()
        training_history.append((iteration, mean_episode_reward, final_success_rate))

        if iteration == 1 or iteration % 50 == 0:
            print(
                f"iter {iteration:04d} | "
                f"mean episode reward: {mean_episode_reward:7.3f} | "
                f"final success rate: {final_success_rate * 100:6.2f}%"
            )

    checkpoint_path = "checkpoints/policy.pt"
    log_path = "results/training_log.csv"

    torch.save(policy.state_dict(), checkpoint_path)

    with open(log_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["iteration", "mean_episode_reward", "final_success_rate"])
        writer.writerows(training_history)

    print()
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved training log: {log_path}")


if __name__ == "__main__":
    main()