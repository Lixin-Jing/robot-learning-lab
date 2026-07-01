# Development Log

This document records the development process, debugging steps, experiment results, and design decisions of this robot learning project.

---

## 2026-07-01 — Initial Push-to-Target Experiment

### Goal

Train a policy network to control a simplified 2D gripper and push an object toward a target.

### Initial Result

The training and evaluation scripts ran successfully, but the learned policy did not complete the push-to-target task.

Observed behavior:

- The gripper moved diagonally upward.
- The object stayed at its initial position.
- The object did not move toward the target.
- The final object-target distance remained high.

### Problem

The policy did not learn to contact the object. Since the object only moves when the gripper is in contact with it, the pushing task was too difficult at the beginning.

### Diagnosis

The original task required several behaviors at once:

1. Move the gripper toward the object.
2. Make contact with the object.
3. Push the object.
4. Move the object toward the target.

The policy failed at the reaching/contact stage, so pushing never occurred.

### First Fix

The task was simplified toward a reaching objective.

Changes:

- Added reward for reducing the gripper-object distance.
- Added reward for being close to the object.
- Added contact reward.
- Changed the success condition to gripper-object contact.

### Result

The training success rate increased significantly after switching to reaching success.

Example:

```text
iter 1000 | mean episode reward: 68.855 | final success rate: 94.53%
```

However, the evaluation trajectory still showed the gripper moving diagonally upward instead of moving directly toward the object.

### Further Diagnosis

The training and evaluation initial conditions were inconsistent.

In training, the gripper was initialized with both x and y sampled from `[-0.9, -0.6]`, meaning it often started in the lower-left region.

In evaluation, the gripper started at `[-0.75, 0.0]`, horizontally aligned with the object.

This caused a train-evaluation mismatch.

### Next Step

Modify `reset_env()` so that:

- `gripper_x` starts on the left side.
- `gripper_y` is sampled from a range similar to the object y-position.
- The training distribution better matches the evaluation scenario.

Planned change:

```python
gripper = torch.empty(num_envs, 2, device=cfg.device)
gripper[:, 0] = torch.empty(num_envs, device=cfg.device).uniform_(-0.9, -0.6)
gripper[:, 1] = torch.empty(num_envs, device=cfg.device).uniform_(-0.25, 0.25)
```

### Lesson Learned


For reinforcement learning tasks, reward design alone is not enough. The training distribution and evaluation scenario must also match. A high training success rate may not imply successful evaluation if the policy is tested outside its training distribution.

---

## 2026-07-02 — Early Stopping for Push-to-Target

### Goal

Prevent the object from being pushed past the target after it has already reached the success region.

### Change

Added a `done` mask to both training and evaluation.

Once an environment reaches the pushing success condition, the action is set to zero in later steps:

```python
action = torch.where(already_done.unsqueeze(-1), torch.zeros_like(action), action)
```

The success state is accumulated with:

```python
done = done | success
```

### Result

The evaluation trajectory improved significantly.

The gripper first moved toward the object, contacted it, and pushed it toward the target. The object stopped near the target instead of being pushed past it.

Example evaluation result:

```text
Final object position: x=0.682, y=-0.015
Target position:       x=0.750, y=0.000
Final distance to target: 0.069
Final gripper-object distance: 0.157
Reached goal: True
```

Example training output:

```text
iter 1000 | mean episode reward: 60.934 | final success rate: 13.67%
```

### Diagnosis

The previous policy could push the object, but without early stopping the object continued moving after passing the target. The `done` mask fixed this by stopping the action after success.

### Remaining Issue

The final training success rate is still relatively low, around 10%–16%. This means the policy can solve the fixed evaluation scenario, but it is not yet robust across all randomized training environments.

### Next Step

Improve robustness by tuning reward weights, increasing training iterations, and evaluating across multiple randomized test environments instead of only one fixed scenario.

### Lesson Learned

For long-horizon reinforcement learning tasks, reaching the goal once and staying at the goal are different problems. Early stopping or terminal-state handling is necessary when the environment should stop changing after success.

