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

#### Note

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

#### Note

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

#### Note

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

#### Note

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

#### Note

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

#### Note

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

#### Note

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

#### Note


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

#### Note

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

#### Note

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

#### Note

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

#### Note


---

## 2026-07-08

### Targeted Y-Offset Sampling Range Ablation

#### Source

This ablation was motivated by curriculum learning, prioritized sampling, and domain randomization ideas in reinforcement learning.

Relevant references:

```text
Florensa, C., Held, D., Wulfmeier, M., Zhang, M., & Abbeel, P. (2017).
Reverse Curriculum Generation for Reinforcement Learning.
Conference on Robot Learning (CoRL).
```

```text
Jiang, M., Grefenstette, E., & Rocktäschel, T. (2021).
Prioritized Level Replay.
International Conference on Machine Learning (ICML).
```

```text
Mozifian, M., Gamboa Higuera, J. C., Meger, D., & Dudek, G. (2019).
Learning Domain Randomization Distributions for Training Robust Locomotion Policies.
arXiv:1906.00410.
```

The common idea behind these references is that the training distribution matters. A useful curriculum should not only make the task easier; it should expose the policy to states that have learning potential and that are relevant to the target evaluation distribution.

#### Goal

Test whether a narrower and milder targeted y-offset sampling range would improve learning near the current capability boundary.

The previous targeted sampling configuration was:

```python
targeted_y_offset_min: float = 0.10
targeted_y_offset_max: float = 0.25
```

This improved medium-y-offset performance, but the hardest bucket remained unsolved:

```text
abs_y > 0.15:
  success rate: 0.00%
```

A reasonable hypothesis was that `[0.10, 0.25]` might be too broad and too difficult. Therefore, a narrower curriculum range was tested.

#### Change

Changed the targeted y-offset sampling range from:

```python
targeted_y_offset_min: float = 0.10
targeted_y_offset_max: float = 0.25
```

to:

```python
targeted_y_offset_min: float = 0.10
targeted_y_offset_max: float = 0.18
```

The goal was to focus training on the current medium-hard region instead of exposing the policy to the full large-y-offset range.

#### Result

The narrower range did not improve performance.

Previous result with `[0.10, 0.25]`:

```text
abs_y <= 0.05:
  success rate: 75.00%

0.05 < abs_y <= 0.10:
  success rate: 50.00%

0.10 < abs_y <= 0.15:
  success rate: 38.89%

abs_y > 0.15:
  success rate: 0.00%
```

New result with `[0.10, 0.18]`:

```text
abs_y <= 0.05:
  count: 16
  success rate: 81.25%
  failures: 3
  mean final distance: 0.067
  median final distance: 0.052

0.05 < abs_y <= 0.10:
  count: 20
  success rate: 50.00%
  failures: 10
  mean final distance: 0.128
  median final distance: 0.089

0.10 < abs_y <= 0.15:
  count: 18
  success rate: 11.11%
  failures: 16
  mean final distance: 0.172
  median final distance: 0.131

abs_y > 0.15:
  count: 46
  success rate: 0.00%
  failures: 46
  mean final distance: 0.239
  median final distance: 0.175
```

The most important regression was:

```text
0.10 < abs_y <= 0.15:
  38.89% -> 11.11%
```

#### Diagnosis

The hypothesis that a milder range would improve curriculum learning was not supported.

The broader range `[0.10, 0.25]` appears to provide more useful y-direction correction experience. Although the highest-offset cases are difficult and still not solved, they may force the policy to learn stronger corrective y-actions. Narrowing the range to `[0.10, 0.18]` reduced exposure to more extreme y-offset states and weakened the learned y-correction behavior.

This suggests that the targeted sampling range is not simply "easier is better". The policy needs enough exposure to challenging y-offset states to learn clear directional correction.

#### Conclusion

The best current v0.2 targeted sampling setting is still:

```python
targeted_y_offset_min: float = 0.10
targeted_y_offset_max: float = 0.25
```

The `[0.10, 0.18]` experiment is useful as an ablation result, but it should not replace the broader range.

#### Note

A curriculum should not only reduce difficulty. It should also preserve useful diversity and expose the agent to states with learning potential. In this project, broader y-offset exposure produced better medium-y generalization than a narrower and milder curriculum range.


---

## 2026-07-15

### Gymnasium Interface Added and Fully Tested

#### Goal

Convert the existing script-style pushing environment into a standard Gymnasium environment so that PPO and SAC can interact with the task through the same interface.

#### Change

Added `gym_env.py` to wrap the existing environment logic.

The wrapper provides:

- A 6-dimensional observation containing the gripper, object, and target positions.
- A 2-dimensional continuous action representing the gripper displacement.
- Standard `reset()` and `step()` methods.
- Separate signals for task success and the episode time limit.

The observation bounds were changed to `[-1, 1]` after testing showed that valid vertical actions could move the gripper beyond the originally declared y-range of `[-0.5, 0.5]`.

Initial-state sampling was also changed to use each environment's own random-number generator. Resetting one environment therefore no longer changes another environment's sampling sequence.

Motion, contact, rewards, success conditions, time limits, and random-state independence were checked during environment development. Invalid action shapes and non-finite actions are rejected before updating the environment.

#### Result

All 30 behavioral checks passed. The environment also passed the Gymnasium and Stable-Baselines3 interface checks, including operation with two environments and automatic resets after the time limit.

The existing policy was evaluated through both the original environment functions and the Gymnasium wrapper using identical initial coordinates.

```text
Aligned scenarios: 201
Maximum action difference: 0
Maximum observation difference: 0
Maximum per-step reward difference: 0
Success decisions: identical
```

The scenarios included one fixed horizontal case, 100 ordinary uniform-sampling cases, and 100 mixed targeted-sampling cases. Their success counts were 1, 29, and 23, respectively.

#### Diagnosis

The interface conversion preserved the existing task behavior. Matching actions, trajectories, and rewards showed that the wrapper did not change how the policy interacted with the environment in the tested scenarios.

The sampling distributions required separate attention. The original single-environment reset used `int(1 * 0.5)=0` targeted samples, whereas the wrapper selects the targeted branch with a probability of 0.5. The two reset methods therefore do not produce the same evaluation distribution. Explicit initial coordinates were used for the migration comparison.

Task success is treated as termination, while reaching the 35-step sampling limit is treated as truncation. This distinction allows later value-based updates to bootstrap at a time limit without bootstrapping after task completion.

#### Note

Passing an interface checker alone does not establish that an environment behaves correctly. Controlled action checks and comparisons against the original implementation are needed to check the transition logic, reward behavior, and episode endings.

---

## 2026-08-08

### PPO Added with Initial Training Results

#### Goal

Introduce a standard PPO baseline and observe whether it can learn the existing reaching-and-pushing task under an initial fixed training budget.

#### Change

Added PPO training through Stable-Baselines3 using the Gymnasium environment.

The policy outputs normalized actions in `[-1, 1]`, which are converted to the gripper displacement range `[-0.08, 0.08]`. The existing transition rules and reward function were retained.

The policy uses two hidden layers with 64 units each, a learning rate of `3e-4`, and a discount factor of `0.97`.

The initial PPO settings were:

- 256-step rollouts.
- Batch size 64.
- 10 update epochs.
- GAE lambda 0.95.
- Clipping range 0.2.

The initial PPO run used one environment, seed 7, and 20,480 environment interactions. The final policy was then evaluated deterministically on 100 initial states.

#### Result

```text
successful scenarios: 0 / 100
success rate: 0%
mean final object-target distance: 0.554076
mean episode return: 4.844446
mean episode length: 35.00
```

The model changed during training. Evaluation used the saved and reloaded final policy.

#### Diagnosis

Under this initial configuration and budget, PPO did not complete any of the evaluated scenarios.

The mean episode length remained at the 35-step limit, showing that evaluation episodes ended by timeout. The positive mean return also shows that receiving intermediate rewards did not necessarily lead to completing the pushing task.

These aggregate metrics do not identify PPO's specific failure behavior. Determining whether the policy fails during reaching, contact, or target-directed pushing requires inspecting its trajectories.

#### Note

Reward and task success should be evaluated separately. A policy can obtain positive shaped rewards without reaching the goal. This result comes from one training seed and does not establish that PPO consistently fails across repeated runs or other configurations.

---

## 2026-08-27

### SAC Added with Initial Training Results

#### Goal

Introduce a standard SAC baseline and observe whether it can learn the existing reaching-and-pushing task under the same initial training budget used for PPO.

#### Change

Added SAC training through Stable-Baselines3 using the Gymnasium environment.

The policy outputs normalized actions in `[-1, 1]`, which are converted to the gripper displacement range `[-0.08, 0.08]`. The existing transition rules and reward function were retained.

The policy uses two hidden layers with 64 units each, a learning rate of `3e-4`, and a discount factor of `0.97`.

The initial SAC settings were:

- 1,000 initial random-action steps.
- Batch size 128.
- One gradient update per subsequent interaction.
- Automatic entropy-coefficient adjustment.

SAC was trained once using seed 7 and 20,480 environment interactions. The final policy was then evaluated deterministically on the same 100 initial states used for PPO.

#### Result

```text
successful scenarios: 100 / 100
success rate: 100%
mean final object-target distance: 0.045325
mean episode return: 27.411743
mean episode length: 17.99
```

The model changed during training. Evaluation used the saved and reloaded final policy.

#### Diagnosis

Under this initial configuration and budget, SAC learned a policy that completed all evaluated scenarios.

The policy reached the target in about 18 steps on average. Its smaller final object-target distance and successful episode endings indicate that it learned task-completing behavior on this evaluation set.

Using the same initial states as PPO makes these observed outcomes easier to compare. However, one training seed does not establish that SAC is consistently better across repeated runs or explain the cause of the difference between the algorithms.

#### Note

A successful initial run is useful evidence that the algorithm can learn the current task, but repeated training runs are still needed before drawing conclusions about stability or general performance.


---

## 2026-10-28

### PPO Failure Trajectory Diagnosis

#### Goal

Identify where the initial PPO policy fails: reaching the object, maintaining contact, or pushing toward the target.

The previous evaluation showed 0% success, but its aggregate metrics did not explain the policy's behavior.

#### Change

Replayed the saved PPO policy on the same 100 initial states used in its original evaluation, without retraining or changing the task.

Recorded the gripper and object trajectories, policy actions, first contact, contact duration, object-target distance, y-error, and each reward component at every step.

Contact was measured at the same instant as the environment's transition rule: after moving the gripper and before moving the object. The replayed returns, final distances, episode lengths, and success decisions matched the original evaluation.

#### Result

```text
Evaluated scenarios: 100
Successful scenarios: 0
Episodes ending at the time limit: 100
Scenarios with contact: 100
Scenarios with object movement: 100
Mean first contact step: 4.52
Mean contact steps per episode: 31.05 / 35

Mean object-target distance:
  initial: 1.009478
  final: 0.554076

Mean absolute object y-error:
  initial: 0.143470
  final: 0.351042

Scenarios with increased absolute y-error: 83 / 100
Scenarios moving farther from the target over the last five steps: 85 / 100
```

All scenarios made contact within steps 2–7. Even the 9 scenarios with initial absolute object y-offset at most 0.05 failed.

A near-horizontal example started with object y=0.008918, made contact at step 7, and ended with object y=-0.331746 and object-target distance 0.527752.

A negative-y example started at object y=-0.167644, made contact at step 3, and ended at y=-0.888843 with final distance 0.984388.

#### Diagnosis

The main bottleneck is not reaching. Every scenario reached the object and moved it, with contact present for most of the episode.

The failure occurs during target-directed pushing. The policy initially pushes rightward and reduces the target distance, but then tends to push downward and slow down before reaching the goal. Of the 100 scenarios, 91 ended below the target's y-coordinate. All 45 scenarios that started below the target increased their absolute y-error.

The late trajectory is consistent with stagnation away from the goal. By step 35, the mean normalized x-action was about 0.00095. Mean object movement over the last five steps totaled only 0.004586, while the object remained about 0.554 from the target. This differs from a policy that is still making substantial goal-directed progress when its time limit expires.

Positive return did not come only from reaching or contact rewards. Mean episode contributions included:

```text
pushing_reward: 5.464823
directional_pushing_reward: 4.054170
reaching_reward: 1.417756
y_correction_reward: -0.830290
object_goal_reward: -6.602733
success_pushing_reward: 0.000000
```

The policy receives shaping rewards for its earlier progress but never earns the final success bonus. The negative y-correction contribution confirms that moving away from the target y-coordinate is already penalized; these observations alone do not establish that changing its weight would solve the problem.

The deterministic mean action was clipped in 24.37% of evaluation steps, mainly during the early rightward motion. This is a recorded characteristic of the current policy, not proof that clipping caused the failure.

#### Note

Trajectory diagnosis localizes the observed failure to directional control and late-stage goal convergence. It does not yet identify whether training budget, exploration, optimization settings, or reward balance caused that behavior. Those hypotheses require separate controlled interventions. The saved policy and its parameters were preserved throughout diagnosis.


---

## 2026-09-16

### PPO Sampling Extended to 256 Environments

#### Goal

Collect PPO training experience from 256 independent environments, matching the environment count used by the original policy-gradient training loop.

#### Change

Changed PPO from one environment to 256 environments. Each environment maintains its own state and reset random-number generator, while the same policy produces a batch of actions for all environments.

The default collection length remains 256 steps per environment. One rollout therefore contains:

```text
256 environments × 256 steps = 65,536 transitions
```

The total training budget counts transitions across all environments. Explicit budgets must contain a whole number of rollouts; incompatible budgets are rejected rather than silently exceeded. The environment count and per-environment rollout length can be set independently.

The sampling backend batches policy computation and steps the lightweight environments sequentially within one process. SAC retains its previous single-environment configuration.

#### Result

The 256 environments produced distinct initial states, reproduced them under repeated seeding, and correctly reset after the episode time limit.

A short training run used 35 collection steps per environment:

```text
Environments: 256
Steps per environment: 35
Total transitions: 8,960
Rollout batches: 1
Optimization epochs: 10
Minibatch optimizer steps: 1,400
```

The model parameters changed, and saving and reloading reproduced the trained parameters. The short run evaluated 5 scenarios and completed none; this was an implementation check after one rollout, not a replacement for the earlier 100-scenario experiment.

#### Diagnosis

The environment count controls how many trajectories are collected together. It does not, by itself, determine the total amount of training. Increasing it also increases the samples in a rollout when the per-environment collection length is held fixed.

The previous 20,480-step single-environment result remains a historical baseline. No improvement in PPO performance is claimed from this short run.

#### Note

Environment count, rollout length, total interactions, and optimizer steps describe different quantities. They should be stated separately when comparing training runs.


### PPO Training with 256 Environments

#### Goal

Determine whether PPO can learn successful pushing with the expanded training configuration after the earlier policy reached and moved the object but failed to converge to the target. Compare the results before and after increasing the interaction budget, and explain what the observed improvement does and does not establish.

#### Change

Trained a new PPO policy from initialization with seed 7. The environment count increased from 1 to 256, while collection length remained 256 steps per environment. The total interaction budget increased from 20,480 to 655,360 transitions.

One interaction means one environment executing an action and returning its reward and next observation. A batched step across 256 environments therefore contributes 256 transitions. Reusing collected experience for another optimization epoch does not add interactions.

```text
Earlier PPO:
  1 environment × 256 steps × 80 rollouts = 20,480 transitions

Current PPO:
  256 environments × 256 steps × 10 rollouts = 655,360 transitions
```

The reward function, environment dynamics, policy architecture, learning rate, minibatch size, and optimization epochs per rollout were unchanged. With minibatch size 64 and 10 epochs per rollout, the larger dataset also increased the total number of minibatch optimizer steps.

Evaluated the final saved policy with deterministic actions on the same 100 scenarios as the earlier PPO evaluation. All six stored initial coordinates and the evaluation seeds were checked and matched exactly. No checkpoint selection used these scenarios.

#### Result

```text
                                   Earlier PPO       Current PPO
Training environments:                       1               256
Collection steps per environment:          256               256
Transitions per rollout:                   256            65,536
Rollout batches:                            80                10
Total transitions:                      20,480           655,360
Minibatch size:                              64                64
Optimization epochs per rollout:            10                10
Minibatch optimizer steps:               3,200           102,400
Training time (seconds):                 14.59            279.94

Evaluation scenarios:                      100               100
Successful scenarios:                  0 / 100          99 / 100
Success rate:                             0.0%             99.0%
Mean final distance:                  0.554076          0.043853
Median final distance:                0.515480          0.039828
Mean episode return:                  4.844446         27.485413
Mean episode length:                    35.00             17.72
```

The total interaction budget and minibatch optimizer steps both increased by a factor of 32. On the identical evaluation scenarios, success rate increased by 99 percentage points, mean final distance decreased by 92.09%, and mean episode return increased by 22.640968. The current configuration therefore performed substantially better on this evaluation set.

All 45 scenarios starting below the target y-coordinate succeeded, as did all 46 scenarios with an initial absolute object y-offset above 0.15. One of the 55 scenarios starting at or above the target y-coordinate failed.

Replaying all 100 scenarios reproduced the saved evaluation results. Two scenarios used to illustrate the earlier policy's failure now succeeded:

```text
Scenario      Earlier final distance      Current final distance      Current steps
10005                      0.527752                    0.033783                 19
10081                      0.984388                    0.044092                 19
```

The remaining failure, scenario 10069, first made contact at step 2 and reached its minimum target distance of 0.075941 at step 17. It then passed the target and moved to the right boundary at x=1. The episode ended at the 35-step limit with final distance 0.250233.

Training performance also improved during the run:

```text
Collected transitions      Mean return of the latest 100 training episodes
65,536                                                           -11.249796
393,216                                                            7.932620
655,360                                                           21.234739
```

These training returns include exploratory action sampling. They are distinct from the final deterministic evaluation metrics above.

The trained parameters changed, and reloading the saved model reproduced them. The earlier PPO checkpoint and original policy-gradient checkpoint were preserved.

#### Diagnosis

The earlier 0% success rate did not mean that the policy had learned nothing. Every earlier evaluation scenario reached and moved the object, reducing mean target distance from about 1.009478 initially to 0.554076 finally. The missing behavior was reliable directional control and final convergence to the goal.

Additional interactions provide more opportunities to encounter different states, try actions, and observe their subsequent rewards. More near-goal and successful experience can help PPO learn which actions improve on its current expectations and refine the final part of the task. As the policy improves, its later training data may also contain more useful near-goal trajectories. These are plausible learning mechanisms, not separately measured causal contributions in this experiment.

The larger rollout collects experience from more independently reset environments before optimization. This may reduce reliance on a few trajectories when forming an update. However, environment count, rollout size, total interactions, and total optimizer steps all changed together; this experiment does not isolate their individual effects. The result supports insufficient learning under the earlier configuration as a plausible explanation, rather than proving that interaction count alone caused the improvement.

Success rate also has a threshold: an episode succeeds only when object-target distance is below 0.07. Continuous improvements in control can move many scenarios across that threshold and produce a large increase in measured success rate. Scenario 10069 illustrates the distinction: its minimum distance of 0.075941 was close to the threshold but still counted as failure. The 92.09% reduction in mean final distance confirms a substantial improvement in physical task performance as well as in the binary success metric.

The remaining failure is consistent with overshooting near the goal and failing to recover before the time limit. Its underlying cause has not been isolated.

#### Note

The original policy-gradient loop also has an interaction budget. Its current configuration implies:

```text
256 environments × 35 steps × 1,000 iterations = 8,960,000 nominal transitions
```

This nominal count is about 13.67 times the current PPO budget. It includes padded steps after early success, during which the old loop applies zero actions and masks rewards. The saved original checkpoint contains weights rather than an actual interaction counter, so its historical completed budget is not confirmed by this calculation. In the PPO environment, completed episodes reset and collection continues; 256 collection steps can span several episodes, each still limited to 35 steps.

The 32-fold budget increase in the before/after comparison refers to the two PPO runs, not to the original policy-gradient algorithm. A matched-budget experiment would be needed to separate environment-count effects from additional training. One training seed does not establish reliability across seeds, and the earlier SAC result used a different training budget. The completed PPO experiment demonstrates 99% success on this fixed evaluation set without establishing a general ranking of the algorithms.
