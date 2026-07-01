from dataclasses import dataclass

import torch


@dataclass
class Config:
    """Configuration for the stage-1 local RL robot control environment.

    This version is intentionally CPU-friendly so it can run locally on a Mac.
    Later, when moving back to the cloud server, only `device`, `num_envs`, and
    `train_iterations` need to be increased again.
    """

    num_envs: int = 256
    episode_steps: int = 35
    train_iterations: int = 1000
    learning_rate: float = 3e-4
    gamma: float = 0.97
    action_std: float = 0.05
    action_limit: float = 0.08
    contact_radius: float = 0.20
    device: str = "cpu"


cfg = Config()


def reset_env(num_envs: int):
    """Create a batch of independent 2D pushing environments.

    Each environment contains:
        - a gripper position
        - an object position
        - a fixed target position

    Tensor shapes:
        gripper:    [num_envs, 2]
        object_pos: [num_envs, 2]
        target:     [num_envs, 2]
    """
    gripper = torch.empty(num_envs, 2, device=cfg.device)
    gripper[:, 0] = torch.empty(num_envs, device=cfg.device).uniform_(-0.9, -0.6)
    gripper[:, 1] = torch.empty(num_envs, device=cfg.device).uniform_(-0.25, 0.25)

    object_pos = torch.empty(num_envs, 2, device=cfg.device)
    object_pos[:, 0] = torch.empty(num_envs, device=cfg.device).uniform_(-0.35, -0.15)
    object_pos[:, 1] = torch.empty(num_envs, device=cfg.device).uniform_(-0.25, 0.25)

    target = torch.zeros(num_envs, 2, device=cfg.device)
    target[:, 0] = 0.75
    target[:, 1] = 0.0

    return gripper, object_pos, target


def make_obs(gripper: torch.Tensor, object_pos: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Build the low-dimensional observation used by the policy.

    Observation:
        [gripper_x, gripper_y, object_x, object_y, target_x, target_y]
    """
    return torch.cat([gripper, object_pos, target], dim=-1)


def env_step(
    gripper: torch.Tensor,
    object_pos: torch.Tensor,
    target: torch.Tensor,
    action: torch.Tensor,
):
    """Advance the simplified robot environment by one step.

    The gripper moves according to the action. If the gripper is close enough to
    the object, the object is pushed along with the gripper.

    Returns:
        gripper: updated gripper position
        object_pos: updated object position
        reward: reward for this step, shape [num_envs]
        success: whether the object reached the target, shape [num_envs]
    """
    old_dist_to_target = torch.linalg.norm(object_pos - target, dim=-1)
    old_dist_object_gripper = torch.linalg.norm(gripper-object_pos,dim=-1)

    action = torch.clamp(action, -cfg.action_limit, cfg.action_limit)
    gripper = torch.clamp(gripper + action, -1.0, 1.0)

    distance_gripper_object = torch.linalg.norm(gripper - object_pos, dim=-1)
    contact = distance_gripper_object < cfg.contact_radius

    object_pos = object_pos + contact.float().unsqueeze(-1) * action * 0.9
    object_pos = torch.clamp(object_pos, -1.0, 1.0)

    new_dist_object_to_target = torch.linalg.norm(object_pos - target, dim=-1)
    new_dist_gripper_object = torch.linalg.norm(gripper - object_pos, dim=-1)

    pushing_reward = (old_dist_to_target - new_dist_object_to_target) * 8.0
    reaching_reward = (old_dist_object_gripper-new_dist_gripper_object)* 8.0
    object_goal_reward = -new_dist_object_to_target
    reach_object_reward = -0.15 * new_dist_gripper_object
    contact_reward = 0.2 * contact.float()
    success_reaching = new_dist_gripper_object < cfg.contact_radius
    success_pushing = new_dist_object_to_target <0.07
    success_pushing_reward = 2.0 * success_pushing.float()
    success_reaching_reward = 3.0 * success_reaching.float()
    action_penalty = -0.02 * torch.sum(action * action, dim=-1)

    reward = (
        reaching_reward
        + pushing_reward
        + object_goal_reward
        +success_pushing_reward
        + reach_object_reward
        + contact_reward
        + success_reaching_reward
        + action_penalty
    )

    return gripper, object_pos, reward, success_pushing


def discounted_returns(rewards: torch.Tensor) -> torch.Tensor:
    """Compute discounted returns for REINFORCE.

    Input shape:
        rewards: [episode_steps, num_envs]

    Output shape:
        returns: [episode_steps, num_envs]
    """
    returns = torch.zeros_like(rewards)
    running_return = torch.zeros(rewards.shape[1], device=cfg.device)

    for t in reversed(range(rewards.shape[0])):
        running_return = rewards[t] + cfg.gamma * running_return
        returns[t] = running_return

    return returns