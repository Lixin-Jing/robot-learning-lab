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

    reward_debug_enabled: bool = True
    reward_debug_interval_iterations: int = 100

    targeted_y_offset_sampling: bool = True
    targeted_y_offset_fraction: float = 0.5
    targeted_y_offset_min: float = 0.10
    targeted_y_offset_max: float = 0.25


cfg = Config()

_reward_debug_step_count = 0
_reward_debug_sums = {}


def _record_reward_components(components: dict[str, torch.Tensor]) -> None:
    """Accumulate and periodically print mean reward components for debugging.

    The logging interval is defined in training iterations, then converted to
    environment steps using `cfg.episode_steps`. This keeps the debug output
    aligned with the training loop without changing `Train.py`.
    """
    global _reward_debug_step_count, _reward_debug_sums

    if not cfg.reward_debug_enabled:
        return

    _reward_debug_step_count += 1

    for name, value in components.items():
        component_mean = value.detach().mean().item()
        _reward_debug_sums[name] = _reward_debug_sums.get(name, 0.0) + component_mean

    log_interval_steps = cfg.episode_steps * cfg.reward_debug_interval_iterations

    if _reward_debug_step_count % log_interval_steps == 0:
        print("\nReward component means over last "
              f"{cfg.reward_debug_interval_iterations} training iterations:")

        for name in sorted(_reward_debug_sums):
            mean_value = _reward_debug_sums[name] / log_interval_steps
            print(f"  {name}: {mean_value:.6f}")

        print()
        _reward_debug_sums = {}


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

    if cfg.targeted_y_offset_sampling:
        targeted_count = int(num_envs * cfg.targeted_y_offset_fraction)

        if targeted_count > 0:
            targeted_abs_y = torch.empty(targeted_count, device=cfg.device).uniform_(
                cfg.targeted_y_offset_min,
                cfg.targeted_y_offset_max,
            )
            targeted_sign = torch.where(
                torch.rand(targeted_count, device=cfg.device) < 0.5,
                -torch.ones(targeted_count, device=cfg.device),
                torch.ones(targeted_count, device=cfg.device),
            )
            object_pos[:targeted_count, 1] = targeted_abs_y * targeted_sign

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
    old_gripper = gripper.clone()
    old_object_pos = object_pos.clone()
    old_dist_to_target = torch.linalg.norm(object_pos - target, dim=-1)
    old_dist_object_gripper = torch.linalg.norm(gripper - object_pos, dim=-1)

    action = torch.clamp(action, -cfg.action_limit, cfg.action_limit)
    gripper = torch.clamp(gripper + action, -1.0, 1.0)

    distance_gripper_object = torch.linalg.norm(gripper - object_pos, dim=-1)
    contact = distance_gripper_object < cfg.contact_radius

    object_pos = object_pos + contact.float().unsqueeze(-1) * action * 0.9
    object_pos = torch.clamp(object_pos, -1.0, 1.0)

    new_dist_object_to_target = torch.linalg.norm(object_pos - target, dim=-1)
    new_dist_gripper_object = torch.linalg.norm(gripper - object_pos, dim=-1)

    desired_push_offset = cfg.contact_radius * 0.8

    old_target_direction = target - old_object_pos
    old_target_direction_norm = torch.linalg.norm(old_target_direction, dim=-1, keepdim=True).clamp_min(1e-6)
    old_target_direction_unit = old_target_direction / old_target_direction_norm
    old_desired_push_position = old_object_pos - old_target_direction_unit * desired_push_offset

    new_target_direction = target - object_pos
    new_target_direction_norm = torch.linalg.norm(new_target_direction, dim=-1, keepdim=True).clamp_min(1e-6)
    new_target_direction_unit = new_target_direction / new_target_direction_norm
    new_desired_push_position = object_pos - new_target_direction_unit * desired_push_offset

    old_dist_to_push_position = torch.linalg.norm(old_gripper - old_desired_push_position, dim=-1)
    new_dist_to_push_position = torch.linalg.norm(gripper - new_desired_push_position, dim=-1)
    pre_contact = (old_dist_object_gripper > cfg.contact_radius).float()
    pre_push_positioning_reward = (old_dist_to_push_position - new_dist_to_push_position) * pre_contact * 2.0

    object_movement = object_pos - old_object_pos
    object_movement_distance = torch.linalg.norm(object_movement, dim=-1)

    target_direction = target - old_object_pos
    target_direction_norm = torch.linalg.norm(target_direction, dim=-1, keepdim=True).clamp_min(1e-6)
    object_movement_norm = torch.linalg.norm(object_movement, dim=-1, keepdim=True).clamp_min(1e-6)

    target_direction_unit = target_direction / target_direction_norm
    object_movement_unit = object_movement / object_movement_norm

    direction_alignment = torch.sum(object_movement_unit * target_direction_unit, dim=-1)
    directional_pushing_reward = torch.clamp(direction_alignment, min=0.0) * object_movement_distance * 8.0

    old_y_error = torch.abs(old_object_pos[:, 1] - target[:, 1])
    new_y_error = torch.abs(object_pos[:, 1] - target[:, 1])
    y_correction_progress = old_y_error - new_y_error
    y_correction_reward = y_correction_progress * contact.float() * 4.0

    pushing_reward = (old_dist_to_target - new_dist_object_to_target) * contact.float() * 12.0
    reaching_reward = (old_dist_object_gripper - new_dist_gripper_object) * pre_contact * 4.0
    object_goal_reward = -0.3 * new_dist_object_to_target
    reach_object_reward = -0.05 * new_dist_gripper_object * pre_contact
    contact_reward = 0.05 * contact.float() * pre_contact
    success_reaching = new_dist_gripper_object < cfg.contact_radius
    success_pushing = new_dist_object_to_target < 0.07
    success_pushing_reward = 8.0 * success_pushing.float()
    success_reaching_reward = 0.5 * success_reaching.float() * pre_contact
    action_penalty = -0.02 * torch.sum(action * action, dim=-1)

    reward = (
        reaching_reward
        + pre_push_positioning_reward
        + pushing_reward
        + directional_pushing_reward
        + y_correction_reward
        + object_goal_reward
        + success_pushing_reward
        + reach_object_reward
        + contact_reward
        + success_reaching_reward
        + action_penalty
    )

    _record_reward_components(
        {
            "total_reward": reward,
            "reaching_reward": reaching_reward,
            "pre_push_positioning_reward": pre_push_positioning_reward,
            "pushing_reward": pushing_reward,
            "directional_pushing_reward": directional_pushing_reward,
            "y_correction_reward": y_correction_reward,
            "object_goal_reward": object_goal_reward,
            "success_pushing_reward": success_pushing_reward,
            "reach_object_reward": reach_object_reward,
            "contact_reward": contact_reward,
            "success_reaching_reward": success_reaching_reward,
            "action_penalty": action_penalty,
        }
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