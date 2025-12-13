# Lab 11: Humanoid Walking - Step-by-Step Instructions

## Overview

In this lab, you will implement the fundamental components of humanoid bipedal locomotion: ZMP-based balance control and gait pattern generation. These are essential for stable walking.

**Estimated Time**: 3-4 hours

## Prerequisites

- Completed Module 3 (Isaac Sim and RL)
- Understanding of rigid body dynamics
- Linear algebra basics
- Python with NumPy

## Learning Outcomes

By completing this lab, you will:

1. Understand ZMP (Zero Moment Point) theory
2. Implement balance control for humanoids
3. Generate walking gait patterns
4. Create smooth foot trajectories
5. Integrate balance and gait systems

---

## Part 1: ZMP Theory (30 minutes)

### Step 1.1: Understanding ZMP

The **Zero Moment Point (ZMP)** is the point on the ground where the sum of horizontal inertial and gravity forces equals zero.

**Key concept**: For stable walking, ZMP must stay within the support polygon (convex hull of foot contacts).

```
ZMP equations:
x_zmp = x_com - (z_com / g) * ẍ_com
y_zmp = y_com - (z_com / g) * ÿ_com

Where:
- x_com, y_com, z_com: Center of mass position
- ẍ_com, ÿ_com: CoM acceleration
- g: Gravity (9.81 m/s²)
```

### Step 1.2: Support Polygon

The support polygon is:
- **Double support**: Convex hull of both feet
- **Single support**: Only the stance foot

```
Double Support:     Single Support (Right):
┌───────┐
│ Left  │           ┌───────┐
│       │           │ Right │
└───────┘           │       │
    ╲   ╱           └───────┘
     ╲ ╱                 │
      │                  │
┌───────┐                │
│ Right │
│       │
└───────┘
```

### Step 1.3: Study the Balance Controller

Review `balance_controller.py`:

```python
# Key components:
class ZMPController:
    def compute_zmp(self, state)      # Compute actual ZMP
    def get_support_polygon(self, state)  # Get polygon vertices
    def is_zmp_stable(self, zmp, state)   # Check if within polygon
    def compute_ankle_torques(...)        # Correct ZMP with ankles
```

**Expected Output**: Understanding of ZMP theory and controller structure.

---

## Part 2: Balance Controller Implementation (45 minutes)

### Step 2.1: Test Basic ZMP Computation

Run the balance controller test:

```bash
python balance_controller.py
```

Verify output:
```
ZMP Reference: [x, y]
ZMP Actual: [x, y]
CoM Adjustment: [dx, dy, dz]
ZMP Stable: True
```

### Step 2.2: Experiment with Controller Gains

Modify `ZMPController` gains:

```python
# In balance_controller.py
self.kp_zmp = np.array([50.0, 50.0])    # ZMP tracking gain
self.kp_torso = np.array([200.0, 200.0, 50.0])  # Torso orientation

# Try different values:
# - Higher kp_zmp: Faster ZMP correction, may oscillate
# - Lower kp_zmp: Slower response, smoother
```

### Step 2.3: Test Different States

Create test scenarios:

```python
# Leaning forward
state.torso_orientation = np.array([0.0, 0.1, 0.0])  # 0.1 rad pitch

# Single support (left foot only)
state.right_foot_contact = False

# Moving CoM
state.com_velocity = np.array([0.2, 0.0, 0.0])  # Walking forward
```

### Step 2.4: Visualize ZMP and Support Polygon

Create a simple visualization:

```python
import matplotlib.pyplot as plt

def plot_balance_state(state, command):
    fig, ax = plt.subplots()

    # Draw support polygon
    polygon = controller.get_support_polygon(state)
    polygon = np.vstack([polygon, polygon[0]])  # Close polygon
    ax.plot(polygon[:, 0], polygon[:, 1], 'b-', label='Support Polygon')

    # Draw ZMP
    ax.plot(command.zmp_actual[0], command.zmp_actual[1], 'ro', label='Actual ZMP')
    ax.plot(command.zmp_reference[0], command.zmp_reference[1], 'g^', label='Reference ZMP')

    # Draw CoM projection
    ax.plot(state.com_position[0], state.com_position[1], 'kx', markersize=10, label='CoM')

    ax.legend()
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('ZMP Balance State')
    plt.show()
```

**Expected Output**: Balance controller responds correctly to various states.

---

## Part 3: Gait Generation (45 minutes)

### Step 3.1: Understand Gait Phases

Walking phases:
1. **Double Support**: Both feet on ground
2. **Left Swing**: Right foot stance, left foot moving
3. **Right Swing**: Left foot stance, right foot moving

```
Gait cycle:
DS → LS → DS → RS → DS → LS → ...

Time:
|--DS--|--LS--|--DS--|--RS--|
   20%   30%    20%    30%
```

### Step 3.2: Test Footstep Planning

Run the gait generator test:

```bash
python gait_generator.py
```

Expected output:
```
Planned 6 footsteps
  Step 0: Left -> (0.25, 0.10)
  Step 1: Right -> (0.50, -0.10)
  Step 2: Left -> (0.75, 0.10)
  ...
```

### Step 3.3: Modify Gait Parameters

Experiment with different gaits:

```python
# Fast walking
params = GaitParameters(
    step_length=0.35,
    step_duration=0.4,
    walking_speed=0.5
)

# Careful walking
params = GaitParameters(
    step_length=0.15,
    step_duration=0.8,
    step_height=0.08  # Higher foot lift
)

# Wide stance
params = GaitParameters(
    step_width=0.3,  # Wider feet separation
)
```

### Step 3.4: Test Swing Trajectories

Visualize foot trajectories:

```python
import matplotlib.pyplot as plt
import numpy as np

swing_gen = SwingTrajectoryGenerator(step_height=0.05)

start = np.array([0.0, 0.1, 0.0])
end = np.array([0.3, 0.1, 0.0])
duration = 0.5

times = np.linspace(0, duration, 50)
positions = []

for t in times:
    traj = swing_gen.generate_trajectory(start, end, duration, t)
    positions.append(traj.position)

positions = np.array(positions)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# XZ trajectory
ax1.plot(positions[:, 0], positions[:, 2], 'b-')
ax1.plot([start[0], end[0]], [start[2], end[2]], 'ko')
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Z (m)')
ax1.set_title('Swing Foot Trajectory (Side View)')

# Time series
ax2.plot(times, positions[:, 0], label='X')
ax2.plot(times, positions[:, 2], label='Z')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Position (m)')
ax2.set_title('Position vs Time')
ax2.legend()

plt.tight_layout()
plt.show()
```

**Expected Output**: Smooth parabolic foot trajectories with proper timing.

---

## Part 4: Integration (45 minutes)

### Step 4.1: Create Integrated Controller

Combine balance and gait:

```python
# integrated_controller.py
from balance_controller import ZMPController, RobotState
from gait_generator import GaitController, GaitParameters

class WalkingController:
    def __init__(self):
        self.balance = ZMPController()
        self.gait = GaitController()

    def command_walk(self, direction, steps):
        self.gait.initialize_walk(direction, steps)

    def update(self, state: RobotState, dt: float):
        # Get gait targets
        gait_out = self.gait.update(dt)

        # Update balance controller with gait reference
        desired_velocity = np.zeros(3)
        if gait_out['phase'] != GaitPhase.DOUBLE_SUPPORT:
            # During swing, CoM should move toward stance foot
            com_target = gait_out['com_reference']
            desired_velocity[:2] = (com_target - state.com_position[:2]) * 2.0

        balance_cmd = self.balance.update(state, desired_velocity)

        return {
            'gait': gait_out,
            'balance': balance_cmd
        }
```

### Step 4.2: Simulate Walking

Create simulation loop:

```python
def simulate_walk():
    controller = WalkingController()
    controller.command_walk(direction=0.0, steps=6)

    # Initial state
    state = RobotState(
        com_position=np.array([0.0, 0.0, 0.9]),
        com_velocity=np.array([0.0, 0.0, 0.0]),
        torso_orientation=np.array([0.0, 0.0, 0.0]),
        torso_angular_velocity=np.array([0.0, 0.0, 0.0]),
        left_foot_position=np.array([0.0, 0.1, 0.0]),
        right_foot_position=np.array([0.0, -0.1, 0.0]),
        left_foot_orientation=np.array([0.0, 0.0, 0.0]),
        right_foot_orientation=np.array([0.0, 0.0, 0.0]),
        left_foot_contact=True,
        right_foot_contact=True
    )

    dt = 0.01
    history = []

    while controller.gait.get_progress() < 1.0:
        result = controller.update(state, dt)

        # Simple state update (in real system, physics engine does this)
        state.com_position[:2] += result['balance'].com_adjustment[:2] * dt * 0.1
        state.left_foot_position = result['gait']['left_foot'].position.copy()
        state.right_foot_position = result['gait']['right_foot'].position.copy()

        # Update contacts based on phase
        phase = result['gait']['phase']
        state.left_foot_contact = phase != GaitPhase.LEFT_SWING
        state.right_foot_contact = phase != GaitPhase.RIGHT_SWING

        history.append({
            'com': state.com_position.copy(),
            'zmp': result['balance'].zmp_actual.copy(),
            'left_foot': state.left_foot_position.copy(),
            'right_foot': state.right_foot_position.copy(),
            'phase': phase
        })

    return history
```

### Step 4.3: Visualize Walking

Plot the walking simulation:

```python
def plot_walking(history):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Extract data
    com = np.array([h['com'] for h in history])
    zmp = np.array([h['zmp'] for h in history])
    left = np.array([h['left_foot'] for h in history])
    right = np.array([h['right_foot'] for h in history])

    # Top-down view
    ax = axes[0, 0]
    ax.plot(com[:, 0], com[:, 1], 'b-', label='CoM')
    ax.plot(zmp[:, 0], zmp[:, 1], 'r--', label='ZMP')
    ax.plot(left[:, 0], left[:, 1], 'g.-', label='Left Foot')
    ax.plot(right[:, 0], right[:, 1], 'm.-', label='Right Foot')
    ax.legend()
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Walking Path (Top View)')
    ax.set_aspect('equal')

    # X position over time
    ax = axes[0, 1]
    t = np.arange(len(history)) * 0.01
    ax.plot(t, com[:, 0], 'b-', label='CoM X')
    ax.plot(t, left[:, 0], 'g--', label='Left X')
    ax.plot(t, right[:, 0], 'm--', label='Right X')
    ax.legend()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('X Position (m)')
    ax.set_title('Forward Progress')

    # Y position (lateral sway)
    ax = axes[1, 0]
    ax.plot(t, com[:, 1], 'b-', label='CoM Y')
    ax.plot(t, left[:, 1], 'g--', label='Left Y')
    ax.plot(t, right[:, 1], 'm--', label='Right Y')
    ax.legend()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Lateral Motion')

    # Foot heights
    ax = axes[1, 1]
    ax.plot(t, left[:, 2], 'g-', label='Left Z')
    ax.plot(t, right[:, 2], 'm-', label='Right Z')
    ax.legend()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Z Position (m)')
    ax.set_title('Foot Heights')

    plt.tight_layout()
    plt.show()
```

**Expected Output**: Visualization showing coordinated walking motion.

---

## Part 5: ROS 2 Integration (30 minutes)

### Step 5.1: Create Walking Node

```python
# walking_node.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import String

class WalkingNode(Node):
    def __init__(self):
        super().__init__('walking_controller')

        self.controller = WalkingController()

        # Subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)

        # Publishers
        self.state_pub = self.create_publisher(String, '/walking/state', 10)

        # Timer
        self.timer = self.create_timer(0.01, self.control_loop)

    def cmd_vel_callback(self, msg):
        # Convert velocity command to walking direction
        if msg.linear.x > 0.1:
            self.controller.command_walk(0.0, 6)  # Walk forward

    def control_loop(self):
        # Update controller
        # Publish state
        pass
```

### Step 5.2: Test with Simulation

Connect to Gazebo or Isaac Sim:
1. Subscribe to joint states
2. Publish joint commands
3. Verify walking behavior

---

## Troubleshooting

### ZMP Outside Support Polygon

**Problem**: ZMP computation shows instability
**Solutions**:
1. Reduce walking speed
2. Increase double support duration
3. Lower CoM height (wider stance)
4. Check for incorrect sensor data

### Foot Trajectory Discontinuities

**Problem**: Jerky foot motion
**Solutions**:
1. Increase trajectory samples
2. Use higher-order polynomial
3. Add velocity continuity constraints

### Lateral Oscillation

**Problem**: Robot sways side to side
**Solutions**:
1. Increase lateral stiffness gains
2. Widen stance width
3. Slow down step transitions

---

## Acceptance Criteria

Your lab is complete when:

- [ ] ZMP computed correctly from robot state
- [ ] Support polygon calculated for all phases
- [ ] Balance controller stabilizes perturbations
- [ ] Footstep planner generates valid steps
- [ ] Swing trajectories are smooth
- [ ] Gait phases transition correctly
- [ ] Integrated controller coordinates balance and gait
- [ ] Walking simulation produces forward motion

---

## Challenge Extensions

### Extension 1: Stair Climbing

Modify gait generator for stairs:
- Adjust step height per step
- Plan footsteps on stair geometry
- Handle variable terrain height

### Extension 2: Push Recovery

Implement push recovery:
- Detect external disturbance
- Step to recover balance
- Adjust CoM trajectory

### Extension 3: Variable Speed Walking

Add speed modulation:
- Adjust step length based on command
- Smooth transitions between speeds
- Emergency stop capability

---

## Files in This Lab

| File | Description |
|------|-------------|
| `balance_controller.py` | ZMP-based balance control |
| `gait_generator.py` | Walking pattern generation |
| `INSTRUCTIONS.md` | This file |
| `README.md` | Lab overview |

---

## Next Steps

After completing this lab:
1. Move to Chapter 12: Dexterous Manipulation
2. Apply balance concepts to whole-body control
3. Integrate with perception for reactive walking
