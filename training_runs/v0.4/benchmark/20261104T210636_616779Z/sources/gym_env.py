"""Gymnasium-style wrapper for the 2D robot pushing environment.

This file is part of v0.3 of the project. It wraps the existing script-style
functions from `env.py` into a standard Gymnasium environment interface so that
future PPO / SAC baselines can interact with the task through `reset()` and
`step()`.
"""

from __future__ import annotations

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from env import cfg, env_step, make_obs


class RobotPushingEnv(gym.Env):
    """Single-environment Gymnasium wrapper for the 2D pushing task.

    Observation:
        A 6-dimensional vector:
        [gripper_x, gripper_y, object_x, object_y, target_x, target_y]

    Action:
        A 2-dimensional continuous vector:
        [delta_x, delta_y]

    Termination:
        The episode terminates when the object reaches the target region.

    Truncation:
        The episode truncates after `cfg.episode_steps` environment steps.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-cfg.action_limit,
            high=cfg.action_limit,
            shape=(2,),
            dtype=np.float32,
        )

        self.gripper: torch.Tensor | None = None
        self.object_pos: torch.Tensor | None = None
        self.target: torch.Tensor | None = None
        self.step_count = 0

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset using this environment's independent random-number generator."""
        super().reset(seed=seed)

        self.step_count = 0

        self.gripper = torch.tensor(
            [[self.np_random.uniform(-0.9, -0.6), self.np_random.uniform(-0.25, 0.25)]],
            dtype=torch.float32,
            device=cfg.device,
        )
        self.object_pos = torch.tensor(
            [[self.np_random.uniform(-0.35, -0.15), self._sample_object_y()]],
            dtype=torch.float32,
            device=cfg.device,
        )
        self.target = torch.tensor(
            [[0.75, 0.0]], dtype=torch.float32, device=cfg.device
        )

        observation = self._get_obs()
        info = self._get_info(success=False, reward=0.0)

        return observation, info

    def step(self, action):
        """Apply one action and return the Gymnasium step tuple."""
        if self.gripper is None or self.object_pos is None or self.target is None:
            raise RuntimeError("Environment must be reset before calling step().")

        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape != self.action_space.shape:
            raise ValueError(
                f"Action must have shape {self.action_space.shape}, got {action_array.shape}."
            )
        if not np.isfinite(action_array).all():
            raise ValueError("Action must contain only finite values.")

        action_tensor = torch.as_tensor(
            action_array, dtype=torch.float32, device=cfg.device
        ).unsqueeze(0)
        action_tensor = torch.clamp(action_tensor, -cfg.action_limit, cfg.action_limit)

        self.gripper, self.object_pos, reward, success = env_step(
            self.gripper,
            self.object_pos,
            self.target,
            action_tensor,
        )

        self.step_count += 1

        terminated = bool(success.item())
        truncated = self.step_count >= cfg.episode_steps
        reward_value = float(reward.item())

        observation = self._get_obs()
        info = self._get_info(success=terminated, reward=reward_value)

        return observation, reward_value, terminated, truncated, info

    def _sample_object_y(self) -> float:
        """Sample object y-position using this environment's random generator.

        The original vectorized `reset_env()` uses `int(num_envs * fraction)` for
        targeted y-offset sampling. That does not work for a single Gymnasium
        environment because `num_envs` is 1. Here we apply the same idea
        probabilistically at the single-environment level.
        """
        if (
            cfg.targeted_y_offset_sampling
            and self.np_random.random() < cfg.targeted_y_offset_fraction
        ):
            abs_y = self.np_random.uniform(
                cfg.targeted_y_offset_min,
                cfg.targeted_y_offset_max,
            )
            sign = -1.0 if self.np_random.random() < 0.5 else 1.0
            return float(abs_y * sign)

        return float(self.np_random.uniform(-0.25, 0.25))

    def _get_obs(self) -> np.ndarray:
        """Return the current observation as a NumPy float32 array."""
        if self.gripper is None or self.object_pos is None or self.target is None:
            raise RuntimeError("Environment state is not initialized. Call reset() first.")

        obs = make_obs(self.gripper, self.object_pos, self.target)
        return obs.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def _get_info(self, success: bool, reward: float) -> dict:
        """Return diagnostic information for debugging and evaluation."""
        if self.gripper is None or self.object_pos is None or self.target is None:
            return {}

        object_target_distance = torch.linalg.norm(self.object_pos - self.target, dim=-1).item()
        gripper_object_distance = torch.linalg.norm(self.gripper - self.object_pos, dim=-1).item()
        abs_initial_or_current_y = abs(float(self.object_pos[0, 1].detach().cpu().item()))

        return {
            "success": success,
            "step_count": self.step_count,
            "reward": reward,
            "object_target_distance": object_target_distance,
            "gripper_object_distance": gripper_object_distance,
            "object_y_abs": abs_initial_or_current_y,
        }
