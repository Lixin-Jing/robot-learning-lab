

import torch
import torch.nn as nn

from env import cfg


class Policy(nn.Module):
    """Small MLP policy for the stage-1 local robot control task.

    Input observation shape:
        [batch_size, 6]
        [gripper_x, gripper_y, object_x, object_y, target_x, target_y]

    Output action shape:
        [batch_size, 2]
        [gripper_dx, gripper_dy]
    """

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Convert an observation tensor into a bounded continuous action."""
        raw_action = self.net(obs)
        bounded_action = torch.tanh(raw_action) * cfg.action_limit
        return bounded_action