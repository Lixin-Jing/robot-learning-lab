# Development Log

This document records the development process, debugging steps, experiment results, and design decisions of this robot learning project.

---

## 2026-07-01

### Initial Push-to-Target Experiment

#### Goal

Train a policy network to control a simplified 2D gripper and push an object toward a target.

#### Initial Result

The training and evaluation scripts ran successfully, but the learned policy did not complete the push-to-target task.

Observed behavior:

- The gripper moved diagonally upward.
- The object stayed at its initial position.
- The object did not move toward the target.
- The final object-target distance remained high.

#### Problem

The policy did not learn to contact the object. Since the object only moves when the gripper is in contact with it, the pushing task was too difficult at the beginning.

#### Diagnosis

The original task required several behaviors at once:

1. Move the gripper toward the object.
2. Make contact with the object.
3. Push the object.
4. Move the object toward the target.

The policy failed at the reaching/contact stage, so pushing never occurred.

#### First Fix

The task was simplified toward a reaching objective.

Changes:

- Added reward for reducing the gripper-object distance.
- Added reward for being close to the object.
- Added contact reward.
- Changed the success condition to gripper-object contact.

#### Result

The training success rate increased significantly after switching to reaching success.

Example:

```text
iter 1000 | mean episode reward: 68.855 | final success rate: 94.53%
```

However, the evaluation trajectory still showed the gripper moving diagonally upward instead of moving directly toward the object.

#### Further Diagnosis

The training and evaluation initial conditions were inconsistent.

In training, the gripper was initialized with both x and y sampled from `[-0.9, -0.6]`, meaning it often started in the lower-left region.

In evaluation, the gripper started at `[-0.75, 0.0]`, horizontally aligned with the object.

This caused a train-evaluation mismatch.

#### Next Step

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

#### Lesson Learned

For reinforcement learning tasks, reward design alone is not enough. The training distribution and evaluation scenario must also match. A high training success rate may not imply successful evaluation if the policy is tested outside its training distribution.

---

## 2026-07-02

### Early Stopping for Push-to-Target

#### Goal

Prevent the object from being pushed past the target after it has already reached the success region.

#### Change

Added a `done` mask to both training and evaluation.

Once an environment reaches the pushing success condition, the action is set to zero in later steps:

```python
action = torch.where(already_done.unsqueeze(-1), torch.zeros_like(action), action)
```

The success state is accumulated with:

```python
done = done | success
```

#### Result

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

#### Diagnosis

The previous policy could push the object, but without early stopping the object continued moving after passing the target. The `done` mask fixed this by stopping the action after success.

#### Remaining Issue

The final training success rate is still relatively low, around 10%–16%. This means the policy can solve the fixed evaluation scenario, but it is not yet robust across all randomized training environments.

#### Next Step

Improve robustness by tuning reward weights, increasing training iterations, and evaluating across multiple randomized test environments instead of only one fixed scenario.

#### Lesson Learned

For long-horizon reinforcement learning tasks, reaching the goal once and staying at the goal are different problems. Early stopping or terminal-state handling is necessary when the environment should stop changing after success.

---

### Multi-Scenario Evaluation

#### Goal

Evaluate whether the trained policy can generalize beyond one fixed evaluation scenario.

The previous `Evaluate.py` tested only one fixed initial condition:

```text
gripper = [-0.75, 0.0]
object  = [-0.25, 0.0]
target  = [ 0.75, 0.0]
```

This was useful for visualizing one trajectory, but it could not show whether the policy was robust across different initial states.

#### Change

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

#### Result

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

#### Diagnosis

The fixed evaluation scenario can be solved, but the policy does not yet generalize well to randomized initial states.

The low multi-scenario success rate shows that the current policy is not robust enough.

From the evaluation results, many failed scenarios still had a small final gripper-object distance. This suggests that the policy often reaches or stays close to the object, but does not reliably push the object to the target.

Therefore, the main current failure mode is not reaching, but unstable pushing behavior.

#### Next Step

Improve the pushing behavior and diagnose failure cases more clearly.

Planned improvements:

- Record the initial gripper, object, and target positions in `evaluation_report.csv`.
- Save trajectory plots for selected failed scenarios.
- Analyze whether failures are caused by pushing in the wrong direction, weak pushing, overshooting, or difficult initial states.
- Tune the pushing reward and object-target reward weights.
- Retrain the policy and rerun multi-scenario evaluation.

#### Lesson Learned

A policy that succeeds in one fixed evaluation scenario may still fail under randomized initial conditions.

Multi-scenario evaluation is necessary to measure generalization and expose hidden failure modes. In this experiment, it showed that the policy had learned basic reaching/contact behavior, but had not yet learned robust pushing.

---

## 2026-07-03

### Initial-State Logging for Multi-Scenario Evaluation

#### Goal

Diagnose why the policy performs poorly in multi-scenario evaluation by recording the initial state of each scenario.

The previous evaluation report only recorded final results, such as success and final distance. This showed that the policy often failed, but it did not explain which initial conditions caused the failures.

#### Change

Updated `Evaluate_MultiScenario.py` to record the initial gripper, object, and target positions for every evaluation scenario.

Added the following columns to `results/evaluation_report.csv`:

- `initial_gripper_x`
- `initial_gripper_y`
- `initial_object_x`
- `initial_object_y`
- `initial_target_x`
- `initial_target_y`

#### Result

After adding initial-state logging, the evaluation results showed a clear relationship between the initial object y-position and task success.

The policy performed much better when the object started close to the target's y-axis. When the object started far above or below `target_y = 0`, the success rate dropped sharply.

#### Diagnosis

The policy had learned a mostly horizontal pushing behavior.

It could often push the object when the object and target were nearly horizontally aligned, but it did not reliably correct the object's y-position when the object started above or below the target.

This suggests that the policy had not learned robust 2D target-directed pushing.

#### Lesson Learned

Final success/failure metrics are not enough for debugging reinforcement learning policies. Logging initial states makes it possible to identify which parts of the state distribution are difficult for the policy.

---

### Failure Trajectory Logging

#### Goal

Visualize failed scenarios to understand how the policy fails, instead of relying only on CSV statistics.

#### Change

Updated `Evaluate_MultiScenario.py` to save trajectory plots for selected failed scenarios.

A failure trajectory plot is saved when:

```python
is_failure = not done.item()
is_large_y_offset = abs(initial_object_y) > 0.15
```

Saved plots are written to:

```text
results/failure_cases/
```

Each plot shows:

- gripper path
- object path
- target position

The evaluation report also records the saved image path in:

- `saved_failure_plot`

#### Result

The failure plots showed that the gripper could usually reach or contact the object, and the object could often be moved. However, the object trajectory frequently stayed close to its original y-level or moved in the wrong diagonal direction.

Typical failure patterns:

- The object was pushed mostly horizontally while the target was at `y = 0`.
- The object moved diagonally, but not toward the target.
- The gripper stayed close to the object, but the object did not converge to the target.

#### Diagnosis

The main failure mode is not reaching. The policy has learned basic reaching/contact behavior.

The main failure mode is weak target-directed pushing. The policy can move the object, but after contact it does not reliably choose an action that aligns with the object-to-target direction.

#### Lesson Learned

Visual trajectory debugging is necessary for robot learning tasks. A scalar success rate can say that a policy failed, but trajectory plots show how it failed.

---

### Direction-Aware Pushing Reward

#### Goal

Encourage the object to move toward the target direction, especially when the object starts above or below the target's y-axis.

The failure trajectories showed that the object was often pushed horizontally or in the wrong diagonal direction. The existing distance-based pushing reward was not strong enough to teach reliable y-direction correction.

#### Change

Added `directional_pushing_reward` in `env.py`.

The new reward compares:

- the actual object movement direction in the current step
- the desired direction from the object to the target

Core idea:

```text
If the object movement direction aligns with the object-to-target direction, give extra reward.
If the object moves sideways or away from the target, give little or no directional reward.
```

Implementation idea:

```python
object_movement = object_pos - old_object_pos
target_direction = target - old_object_pos

object_movement_unit = object_movement / ||object_movement||
target_direction_unit = target_direction / ||target_direction||

direction_alignment = dot(object_movement_unit, target_direction_unit)
directional_pushing_reward = clamp(direction_alignment, min=0.0) * object_movement_distance * 6.0
```

The reward was added to the total reward:

```python
reward = (
    reaching_reward
    + pushing_reward
    + directional_pushing_reward
    + object_goal_reward
    + success_pushing_reward
    + reach_object_reward
    + contact_reward
    + success_reaching_reward
    + action_penalty
)
```

#### Result

After retraining, the multi-scenario success rate improved in one run from about 19% to about 29%.

However, visual inspection of failed trajectories showed that the improvement was limited. Many failed trajectories still showed weak y-direction correction or movement in the wrong diagonal direction.

#### Diagnosis

Direction-aware reward helped, but it was not sufficient by itself.

The reward only evaluates the direction of object movement after the object has already moved. It does not directly teach the gripper where it should contact the object before pushing.

#### Lesson Learned

Distance-based reward theoretically contains direction information, but in practice it may be too weak or indirect. Direction-aware reward provides a more explicit signal, but pushing also depends on contact position and post-contact action direction.

---

### Pre-Push Positioning Reward

#### Goal

Encourage the gripper to approach the object from a better pushing position before contact.

The previous failure analysis suggested that the policy often contacted the object from a suboptimal side. If the gripper contacts the object from the wrong side, the object is likely to move horizontally or in the wrong diagonal direction.

#### Change

Added `pre_push_positioning_reward` in `env.py`.

The idea is to define an ideal gripper position behind the object, opposite to the object-to-target direction:

```python
desired_push_position = object_pos - object_to_target_unit * desired_push_offset
```

This means:

```text
If the target is to the right and below the object, the gripper should approach from the left and above the object.
```

The reward compares whether the gripper moved closer to this desired push position:

```python
old_dist_to_push_position = ||old_gripper - old_desired_push_position||
new_dist_to_push_position = ||gripper - new_desired_push_position||

pre_push_positioning_reward = (
    old_dist_to_push_position - new_dist_to_push_position
) * pre_contact * 4.0
```

The `pre_contact` mask activates this reward only before contact:

```python
pre_contact = (old_dist_object_gripper > cfg.contact_radius).float()
```

The reward was added to the total reward:

```python
reward = (
    reaching_reward
    + pre_push_positioning_reward
    + pushing_reward
    + directional_pushing_reward
    + object_goal_reward
    + success_pushing_reward
    + reach_object_reward
    + contact_reward
    + success_reaching_reward
    + action_penalty
)
```

#### Result

After retraining and evaluating, the success rate did not clearly improve. With fixed-seed evaluation, the result was:

```text
Number of scenarios: 100
Evaluation seed: 0
Success rate: 15.00%
Mean final distance: 0.376
Median final distance: 0.310
Worst final distance: 1.031
Total failures: 85
Large-y-offset failures: 46
Small-y-offset failures: 39
Saved failure trajectory plots: 46
```

#### Diagnosis

The pre-push positioning reward did not significantly improve the final success rate.

Visual inspection showed that the policy still often failed after contact. The object was moved, but the post-contact action direction was not reliably aligned with the object-to-target direction.

A likely reason is that this simplified environment does not model realistic contact geometry. In the current environment, once the gripper is in contact with the object, the object movement is mainly determined by the action direction. Therefore, where the gripper stands before contact may be less important than the action direction after contact.

#### Lesson Learned

A reward term should match the environment dynamics. Pre-push positioning is important in real physical pushing, but in this simplified environment the post-contact action direction is more directly responsible for object movement.

---

### Fixed-Seed Evaluation and Failure Breakdown

#### Goal

Make evaluation results comparable across different reward versions and training runs.

Previous evaluations used randomly generated scenarios without a fixed seed, so success rates such as 19%, 24%, and 29% were not fully comparable.

#### Change

Updated `Evaluate_MultiScenario.py` to use a fixed evaluation seed:

```python
EVAL_SEED = 0

random.seed(EVAL_SEED)
np.random.seed(EVAL_SEED)
torch.manual_seed(EVAL_SEED)
```

The script now evaluates the policy on the same 100 scenarios every time.

The script also clears old failure plots before each run:

```python
for old_plot in glob.glob("results/failure_cases/*.png"):
    os.remove(old_plot)
```

Added failure breakdown metrics:

- `total_failures`
- `large_y_failures`
- `small_y_failures`
- `saved_failure_plots`

Added CSV columns:

- `abs_initial_object_y`
- `large_y_offset`

#### Result

The evaluation output now clearly separates total failures from saved failure plots.

Example fixed-seed result:

```text
Number of scenarios: 100
Evaluation seed: 0
Success rate: 15.00%
Mean final distance: 0.376
Median final distance: 0.310
Worst final distance: 1.031
Total failures: 85
Large-y-offset failures: 46
Small-y-offset failures: 39
Saved failure trajectory plots: 46
```

The number of saved failure plots equals the number of large-y-offset failures because the script saves plots only for failed scenarios where:

```python
abs(initial_object_y) > 0.15
```

#### Diagnosis

Fixed-seed evaluation is now the baseline for comparing future reward changes.

The current baseline under `EVAL_SEED = 0` is:

```text
Success rate: 15%
Total failures: 85
Large-y-offset failures: 46
Small-y-offset failures: 39
Mean final distance: 0.376
```

#### Next Step

The next likely improvement is to directly reward post-contact action alignment.

The current simplified environment makes object movement depend strongly on the action direction after contact. Therefore, a future `contact_action_alignment_reward` may be more effective than adding more pre-contact shaping.

#### Lesson Learned


Evaluation must be reproducible before reward changes can be compared. Fixed-seed multi-scenario evaluation provides a stable benchmark for measuring whether a new reward term actually improves policy performance.

---

## 2026-07-05

### Reward Reweighting Based on Potential-Based Reward Shaping

#### Source

This change was motivated by the reward shaping principle from:

```text
Ng, A. Y., Harada, D., & Russell, S. (1999).
Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping.
International Conference on Machine Learning (ICML).
```

The key idea from this paper is that reward shaping should preferably be based on a potential difference between states, rather than continuously rewarding a static intermediate state.

In this project, this motivated the use of progress-based rewards such as:

```python
old_error - new_error
```

instead of relying too heavily on state-based rewards such as contact bonuses or static distance penalties.

#### Goal

Reduce reward hacking around intermediate objectives and make the reward structure more aligned with the final pushing objective.

The previous reward design encouraged several useful behaviors, but some intermediate rewards could dominate training:

- staying close to the object
- staying in contact with the object
- receiving large reaching/contact rewards even after contact had already been achieved

The intended final objective, however, is not simply to reach or contact the object. The real objective is to push the object to the target.

#### Change

The reward structure was adjusted to make the task more stage-aware:

- Reaching-related rewards mainly apply before contact.
- Pushing-related rewards mainly apply after contact.
- Final pushing success receives a larger reward than reaching success.
- Intermediate contact/reaching rewards were reduced.
- Directional pushing reward was slightly increased.
- A y-correction progress reward was added.

The y-correction reward was first implemented as a conservative positive-only progress reward:

```python
old_y_error = torch.abs(old_object_pos[:, 1] - target[:, 1])
new_y_error = torch.abs(object_pos[:, 1] - target[:, 1])
y_correction_progress = old_y_error - new_y_error
y_correction_reward = torch.clamp(y_correction_progress, min=0.0) * contact.float() * 4.0
```

After evaluation showed limited improvement, it was changed to a signed progress reward:

```python
y_correction_reward = y_correction_progress * contact.float() * 4.0
```

This means:

```text
object_y moves closer to target_y    -> positive reward
object_y moves farther from target_y -> negative reward
object_y does not change             -> zero reward
```

#### Result

The first reward reweighting improved overall pushing performance:

```text
Success rate: 15% -> 20%
Mean final distance: 0.376 -> 0.167
Median final distance: 0.310 -> 0.132
Worst final distance: 1.031 -> 0.794
```

However, large-y-offset failures did not improve:

```text
Large-y-offset failures: 46 -> 46
```

Adding y-correction progress reward only slightly improved the overall success rate, but still did not reduce large-y-offset failures.

#### Diagnosis

The policy improved at general pushing and small-y-offset scenarios, but it still failed to solve large-y-offset scenarios.

This suggests that the main bottleneck was not only reward weighting. The policy needed more training exposure to scenarios where object_y was significantly different from target_y.

#### Lesson Learned

Progress-based reward shaping is useful, but reward shaping alone cannot fix poor coverage of difficult initial states. If the policy rarely receives useful learning signals in difficult regions of the state distribution, changing reward weights may not be enough.

---

### Reward Component Logging

#### Goal

Avoid blind reward tuning by measuring the actual scale of each reward component during training.

#### Change

Added reward component debugging in `env.py`.

The environment now periodically logs the mean value of each reward component over the last 100 training iterations:

```text
action_penalty
contact_reward
directional_pushing_reward
object_goal_reward
pre_push_positioning_reward
pushing_reward
reach_object_reward
reaching_reward
success_pushing_reward
success_reaching_reward
total_reward
y_correction_reward
```

#### Result

The logging showed that:

- `success_pushing_reward` and `pushing_reward` were major positive contributors.
- `object_goal_reward` was a large negative state-based component before being reduced.
- `y_correction_reward` had a much smaller magnitude than the main pushing rewards.

Example reward component output:

```text
pushing_reward: 0.288639
directional_pushing_reward: 0.199457
success_pushing_reward: 1.213714
y_correction_reward: -0.001808
object_goal_reward: -0.117791
total_reward: 1.654928
```

#### Diagnosis

The y-correction reward was too small to dominate learning by itself. This explained why adding y-correction shaping did not significantly improve large-y-offset scenarios.

#### Lesson Learned

Reward terms should not only be logically correct. Their relative scales also matter. Logging reward components makes reward tuning more evidence-based.

---

### Y-Offset Bucket Evaluation

#### Goal

Diagnose whether the policy fails uniformly across all scenarios or whether failures are concentrated in specific initial-state regions.

The previous evaluation reported only aggregate metrics such as overall success rate and total failures. This made it difficult to understand whether the policy failed randomly or systematically.

#### Change

Updated `Evaluate_MultiScenario.py` to classify each scenario by the initial absolute object y-offset:

```text
abs_y <= 0.05
0.05 < abs_y <= 0.10
0.10 < abs_y <= 0.15
abs_y > 0.15
```

The evaluation now prints, for each bucket:

- count
- success rate
- failures
- mean final distance
- median final distance

A new CSV column was also added:

```text
y_offset_bucket
```

#### Result

The first bucket evaluation showed a clear difficulty cliff:

```text
abs_y <= 0.05:
  success rate: 75.00%

0.05 < abs_y <= 0.10:
  success rate: 25.00%

0.10 < abs_y <= 0.15:
  success rate: 5.56%

abs_y > 0.15:
  success rate: 0.00%
```

#### Diagnosis

The policy had learned near-horizontal pushing, but had not learned robust 2D target-directed pushing.

The buckets represent different task difficulties:

```text
small-y-offset  -> mostly 1D pushing
medium-y-offset -> weak 2D correction
large-y-offset  -> true 2D target-directed pushing
```

#### Lesson Learned

Aggregate success rate can hide structured failure modes. Bucketed evaluation provides a clearer picture of what the policy has actually learned.

---

### Targeted Y-Offset Initial-State Sampling

#### Goal

Improve policy learning in medium and large y-offset scenarios by increasing their presence during training.

The bucket evaluation showed that the policy failed mostly when the object started far above or below the target y-position. Instead of continuing to tune reward weights blindly, the training initial-state distribution was adjusted.

#### Change

Updated `reset_env()` in `env.py` to include targeted y-offset sampling.

Added configuration values:

```python
targeted_y_offset_sampling: bool = True
targeted_y_offset_fraction: float = 0.5
targeted_y_offset_min: float = 0.10
targeted_y_offset_max: float = 0.25
```

The new sampling rule is:

```text
50% of environments use normal object_y sampling.
50% of environments force abs(object_y) to be between 0.10 and 0.25.
```

#### Result

Targeted sampling significantly improved medium-y-offset performance:

```text
Overall success rate: 18% -> 29%

0.05 < abs_y <= 0.10:
  success rate: 25.00% -> 50.00%

0.10 < abs_y <= 0.15:
  success rate: 5.56% -> 38.89%
```

However, the hardest bucket remained unsolved:

```text
abs_y > 0.15:
  success rate: 0.00%
```

The final bucket evaluation after targeted sampling was:

```text
abs_y <= 0.05:
  count: 16
  success rate: 75.00%
  failures: 4
  mean final distance: 0.068
  median final distance: 0.049

0.05 < abs_y <= 0.10:
  count: 20
  success rate: 50.00%
  failures: 10
  mean final distance: 0.128
  median final distance: 0.086

0.10 < abs_y <= 0.15:
  count: 18
  success rate: 38.89%
  failures: 11
  mean final distance: 0.166
  median final distance: 0.114

abs_y > 0.15:
  count: 46
  success rate: 0.00%
  failures: 46
  mean final distance: 0.228
  median final distance: 0.159
```

#### Diagnosis

Targeted initial-state sampling was effective. It moved the policy's capability boundary from around `abs_y = 0.10` to around `abs_y = 0.15`.

However, `abs_y > 0.15` remained too difficult under the current setup. This suggests that future training may need a more gradual curriculum, such as first focusing on `0.10 <= abs_y <= 0.18` before expanding to larger offsets.

#### Lesson Learned

When the policy fails in a specific part of the state distribution, changing the reward alone may not be enough. Adjusting the initial-state distribution can provide more useful learning signals and improve robustness in targeted regions.
