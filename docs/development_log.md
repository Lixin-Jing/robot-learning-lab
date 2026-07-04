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

---

## 2026-07-02 — Multi-Scenario Evaluation

### Goal

Evaluate whether the trained policy can generalize beyond one fixed evaluation scenario.

The previous `Evaluate.py` tested only one fixed initial condition:

```text
gripper = [-0.75, 0.0]
object  = [-0.25, 0.0]
target  = [ 0.75, 0.0]
```

This was useful for visualizing one trajectory, but it could not show whether the policy was robust across different initial states.

### Change

Added `Evaluate_MultiScenario.py`.

The new evaluation script:

- Loads the trained policy from `checkpoints/policy.pt`.
- Runs the policy on 100 randomized evaluation scenarios.
- Uses `reset_env(1)` to generate a new initial gripper, object, and target state for each scenario.
- Keeps the same `done` mask logic as the single-scenario evaluation, so actions become zero after success.
- Records the result of each scenario.
- Saves the evaluation results to `results/evaluation_report.csv`.

Recorded metrics for each scenario:

- `scenario_id`
- `success`
- `final_distance_object_target`
- `final_distance_gripper_object`

### Result

The multi-scenario evaluation ran successfully over 100 randomized scenarios.

Summary:

```text
Number of scenarios: 100
Successful scenarios: 19
Success rate: 19%
Mean final object-target distance: 0.333
Median final object-target distance: 0.295
Worst final object-target distance: 1.031
Best final object-target distance: 0.015
```

### Diagnosis

The fixed evaluation scenario can be solved, but the policy does not yet generalize well to randomized initial states.

The low multi-scenario success rate shows that the current policy is not robust enough.

From the evaluation results, many failed scenarios still had a small final gripper-object distance. This suggests that the policy often reaches or stays close to the object, but does not reliably push the object to the target.

Therefore, the main current failure mode is not reaching, but unstable pushing behavior.

### Next Step

Improve the pushing behavior and diagnose failure cases more clearly.

Planned improvements:

- Record the initial gripper, object, and target positions in `evaluation_report.csv`.
- Save trajectory plots for selected failed scenarios.
- Analyze whether failures are caused by pushing in the wrong direction, weak pushing, overshooting, or difficult initial states.
- Tune the pushing reward and object-target reward weights.
- Retrain the policy and rerun multi-scenario evaluation.

### Lesson Learned

A policy that succeeds in one fixed evaluation scenario may still fail under randomized initial conditions.

Multi-scenario evaluation is necessary to measure generalization and expose hidden failure modes. In this experiment, it showed that the policy had learned basic reaching/contact behavior, but had not yet learned robust pushing.
