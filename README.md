# Robot Learning Lab

A progressive robot learning project starting from simplified PyTorch-based 2D control tasks and gradually extending toward reinforcement learning, robot control, Isaac Lab, and manipulation tasks.

## Current Stage

The current implementation is a simplified 2D reinforcement learning environment.

A policy network controls a gripper in a low-dimensional environment. The current task is a push-to-target experiment, where the gripper should push an object toward a target position.

## Project Structure

- `env.py`: Defines the simplified robot environment, transition logic, reward function, and success condition.
- `Policy.py`: Defines the policy neural network.
- `Train.py`: Trains the policy using a REINFORCE-style policy gradient method.
- `Evaluate.py`: Loads the trained policy and evaluates it in a fixed test scenario.

