# Lab 10: Navigation and Sim2Real Transfer - Step-by-Step Instructions

## Overview

In this lab, you will train a navigation policy using reinforcement learning in Isaac Sim, then deploy it to a real or simulated robot using ROS 2. You'll learn domain randomization techniques for robust sim-to-real transfer.

**Estimated Time**: 3-4 hours

## Prerequisites

- Completed Lab 8 and Lab 9
- Isaac Sim installed and working
- ROS 2 Humble with Nav2 stack
- PyTorch installed
- Understanding of reinforcement learning basics

## Learning Outcomes

By completing this lab, you will:

1. Set up an RL environment in Isaac Sim
2. Train a navigation policy with PPO
3. Apply domain randomization for robustness
4. Deploy trained policy to ROS 2 robot
5. Handle sim-to-real discrepancies

---

## Part 1: RL Environment Setup (30 minutes)

### Step 1.1: Understand the Environment

Review the NavigationEnvironment class in `train_nav.py`:

**Observation Space (12 dimensions)**:
- Robot position (x, y): 2
- Robot orientation (yaw): 1
- Goal relative to robot: 2
- Robot velocities (linear, angular): 2
- Distance sensor readings: 5

**Action Space (2 dimensions)**:
- Linear velocity: [-1, 1] → scaled to m/s
- Angular velocity: [-1, 1] → scaled to rad/s

**Rewards**:
- Goal reached: +100
- Collision: -10
- Per step: -0.1 (encourages efficiency)
- Distance shaping: -0.1 * distance_to_goal

### Step 1.2: Test Environment

```bash
cd /isaac-sim
./python.sh /path/to/labs/module-3/ch10-sim2real/train_nav.py
```

Watch the environment setup and verify:
- [ ] Robot spawns correctly
- [ ] Goal marker visible
- [ ] Obstacles appear

### Step 1.3: Configure Domain Randomization

Edit `train_nav.py` to adjust randomization:

```python
env.dr_config.update({
    "randomize_goal": True,
    "randomize_obstacles": True,
    "randomize_friction": True,
    "randomize_mass": True,
    "num_obstacles": (3, 8),        # Range of obstacles
    "friction_range": (0.3, 1.0),   # Floor friction
    "mass_scale_range": (0.8, 1.2), # Robot mass variation
})
```

**Expected Output**: Environment runs with varying conditions each episode.

---

## Part 2: Policy Training (60 minutes)

### Step 2.1: Understand PPO Algorithm

PPO (Proximal Policy Optimization) key concepts:
- **Actor-Critic**: Policy (actor) and value function (critic)
- **Clipped objective**: Prevents large policy updates
- **GAE**: Generalized Advantage Estimation for variance reduction

### Step 2.2: Configure Training

In `train_nav.py`, adjust hyperparameters:

```python
trainer = PPOTrainer(
    env=env,
    hidden_dim=256,      # Network size
    lr=3e-4,             # Learning rate
    gamma=0.99,          # Discount factor
    gae_lambda=0.95,     # GAE parameter
    clip_epsilon=0.2,    # PPO clipping
    n_epochs=10,         # Updates per rollout
    batch_size=64,       # Mini-batch size
)
```

### Step 2.3: Start Training

```bash
./python.sh train_nav.py
```

Monitor training progress:
```
Iter 10 | Steps: 20480 | Reward: -45.32 | Length: 203.4 | Success: 5.0%
Iter 20 | Steps: 40960 | Reward: -12.45 | Length: 156.2 | Success: 23.0%
Iter 30 | Steps: 61440 | Reward: 25.67 | Length: 98.3 | Success: 58.0%
...
```

### Step 2.4: Training Tips

**If reward doesn't improve**:
1. Increase `hidden_dim` for more capacity
2. Reduce `lr` for stability
3. Add more domain randomization
4. Check reward shaping

**If training is unstable**:
1. Reduce `clip_epsilon` (0.1 instead of 0.2)
2. Increase rollout length (4096 instead of 2048)
3. Add entropy bonus (already included)

### Step 2.5: Save Checkpoint

Training automatically saves to `models/nav_policy.pt`. To save manually:

```python
trainer.save("models/nav_policy_checkpoint.pt")
```

**Expected Output**: Training converges to 60%+ success rate after ~100k steps.

---

## Part 3: Policy Evaluation (30 minutes)

### Step 3.1: Visualize Trained Policy

Modify `train_nav.py` to run evaluation:

```python
# After training
env = NavigationEnvironment(max_episode_steps=500)

# Disable randomization for consistent evaluation
env.dr_config["randomize_obstacles"] = False

# Load policy
trainer.load("models/nav_policy.pt")

# Run evaluation
for episode in range(10):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        action = trainer.policy.get_action(obs_tensor, deterministic=True)
        action_np = action.cpu().numpy().squeeze()
        obs, reward, done, info = env.step(action_np)
        total_reward += reward

    print(f"Episode {episode}: Reward={total_reward:.2f}, Success={info.get('success', False)}")
```

### Step 3.2: Test with Different Scenarios

Test robustness:

```python
# Test with more obstacles
env.dr_config["num_obstacles"] = (5, 10)

# Test with farther goals
env.arena_size = 8.0

# Test with different starting positions
env.robot.set_world_pose(position=np.array([1.0, 1.0, 0.1]))
```

### Step 3.3: Analyze Failures

Common failure modes:
1. **Oscillation near goal**: Reduce action scale or add terminal velocity constraint
2. **Stuck on obstacles**: Improve distance sensor coverage
3. **Overshooting**: Reduce max velocity or add velocity penalty

**Expected Output**: Policy achieves 70%+ success on evaluation.

---

## Part 4: ROS 2 Deployment (45 minutes)

### Step 4.1: Set Up ROS 2 Workspace

```bash
mkdir -p ~/nav_deploy_ws/src
cd ~/nav_deploy_ws/src

# Create package
ros2 pkg create --build-type ament_python nav_policy_deploy

# Copy deployment script
cp /path/to/labs/module-3/ch10-sim2real/deploy.py \
   nav_policy_deploy/nav_policy_deploy/

# Copy policy model
mkdir -p nav_policy_deploy/models
cp models/nav_policy.pt nav_policy_deploy/models/
```

### Step 4.2: Update Package Configuration

Edit `setup.py`:

```python
entry_points={
    'console_scripts': [
        'deploy_node = nav_policy_deploy.deploy:main',
    ],
},
```

### Step 4.3: Build Package

```bash
cd ~/nav_deploy_ws
colcon build --packages-select nav_policy_deploy
source install/setup.bash
```

### Step 4.4: Launch Robot Simulation (for testing)

Terminal 1 - Launch TurtleBot3 simulation:
```bash
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

Terminal 2 - Launch navigation deployment:
```bash
ros2 run nav_policy_deploy deploy_node \
    --ros-args \
    -p model_path:=/path/to/nav_policy.pt \
    -p goal_x:=2.0 \
    -p goal_y:=0.0 \
    -p action_scale:=0.5
```

### Step 4.5: Send Goals

```bash
ros2 topic pub /goal_pose geometry_msgs/msg/PoseStamped "{
  header: {frame_id: 'map'},
  pose: {position: {x: 3.0, y: 1.0, z: 0.0}}
}" --once
```

**Expected Output**: Robot navigates toward goal using learned policy.

---

## Part 5: Sim2Real Adaptation (30 minutes)

### Step 5.1: Understand Sim2Real Challenges

Common discrepancies:
| Aspect | Simulation | Real World |
|--------|------------|------------|
| Dynamics | Perfect physics | Friction, slip |
| Sensors | No noise | Noise, dropout |
| Timing | Fixed dt | Variable delays |
| Visual | Synthetic | Real lighting |

### Step 5.2: Observation Adaptation

The `Sim2RealAdapter` class handles:

```python
# Online normalization
def adapt_observation(self, obs):
    # Update running statistics
    self.obs_count += 1
    delta = obs - self.running_mean
    self.running_mean += delta / self.obs_count

    # Normalize using running stats
    std = np.sqrt(self.running_var / self.obs_count)
    return (obs - self.running_mean) / std
```

### Step 5.3: Action Scaling

Tune `action_scale` for real robot:

```python
# Conservative start
action_scale = 0.3  # 30% of training velocities

# Gradually increase if stable
action_scale = 0.5

# Full speed (if robot handles it)
action_scale = 1.0
```

### Step 5.4: Safety Limits

Ensure safety bounds in `adapt_action`:

```python
def adapt_action(self, action):
    # Hard velocity limits
    action[0] = np.clip(action[0], -0.3, 0.3)   # Linear: 0.3 m/s max
    action[1] = np.clip(action[1], -0.5, 0.5)   # Angular: 0.5 rad/s max

    # Smooth acceleration (optional)
    action = 0.5 * action + 0.5 * self.last_action
    self.last_action = action

    return action
```

### Step 5.5: Test on Real Robot

If you have access to a real robot:

1. Ensure emergency stop is accessible
2. Start with very low `action_scale` (0.1)
3. Test in open area first
4. Gradually increase complexity

**Expected Output**: Policy transfers to real robot with >50% success rate.

---

## Part 6: Fine-Tuning (Optional, 30 minutes)

### Step 6.1: Collect Real-World Data

Record trajectories from real robot:

```bash
ros2 bag record /odom /scan /cmd_vel -o real_world_data
```

### Step 6.2: Domain Adaptation

Use real data to fine-tune:

1. Update observation normalization statistics
2. Fine-tune policy with behavior cloning
3. Continue RL training with real data

### Step 6.3: Residual Policy Learning

Train a residual policy on real robot that corrects the sim policy:

```python
# Real action = Sim action + Residual correction
real_action = sim_policy(obs) + residual_policy(obs)
```

---

## Troubleshooting

### Policy Not Converging

**Problem**: Reward stays flat or decreases
**Solutions**:
1. Check environment rewards are computed correctly
2. Reduce learning rate
3. Increase network capacity
4. Simplify task (fewer obstacles, shorter goals)

### Robot Oscillates

**Problem**: Robot vibrates or oscillates near goal
**Solutions**:
1. Add terminal velocity constraint
2. Reduce action frequency
3. Add action smoothing in deployment

### Poor Sim2Real Transfer

**Problem**: Good in sim, fails on real
**Solutions**:
1. Increase domain randomization
2. Add sensor noise in training
3. Tune action scaling
4. Collect real data for normalization

### ROS Node Crashes

**Problem**: Deployment node exits
**Solutions**:
1. Check model path is correct
2. Verify PyTorch/CUDA compatibility
3. Check topic names match robot

---

## Acceptance Criteria

Your lab is complete when:

- [ ] RL environment runs in Isaac Sim
- [ ] Domain randomization configured
- [ ] Policy trains to 60%+ success rate
- [ ] Policy saved and loadable
- [ ] ROS 2 deployment node runs
- [ ] Robot responds to goal commands
- [ ] Sim2Real adapter applies transformations
- [ ] Safety limits enforced
- [ ] Documentation of training curves

---

## Challenge Extensions

### Extension 1: Multi-Goal Navigation

Train policy to navigate through waypoints:
- Modify observation to include waypoint list
- Train with sequence of goals
- Deploy with waypoint following

### Extension 2: Dynamic Obstacle Avoidance

Add moving obstacles:
- Random velocity obstacles in training
- Predict obstacle motion in policy
- Test reactive avoidance

### Extension 3: Visual Navigation

Replace LIDAR with camera:
- Add image observation
- Use CNN feature extractor
- Train with visual domain randomization

---

## Files in This Lab

| File | Description |
|------|-------------|
| `train_nav.py` | RL training script with PPO |
| `deploy.py` | ROS 2 deployment node |
| `INSTRUCTIONS.md` | This file |
| `README.md` | Lab overview |

---

## Next Steps

After completing this lab:
1. Move to Module 4: VLA and Humanoid Robotics
2. Apply RL techniques to locomotion control
3. Integrate vision-language models for task understanding
