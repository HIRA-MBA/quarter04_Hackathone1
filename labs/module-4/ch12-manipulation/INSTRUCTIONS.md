# Lab 12: Dexterous Manipulation

## Overview

In this lab, you will implement and experiment with grasp planning and force control algorithms for multi-fingered robotic hands. You'll learn how to plan stable grasps, implement force control for compliant manipulation, and detect slip during grasping.

## Learning Objectives

By completing this lab, you will:

1. Understand grasp quality metrics and force closure
2. Implement grasp planning for various object geometries
3. Apply impedance and hybrid force/position control
4. Detect and prevent slip during grasping
5. Coordinate multi-finger force control

## Prerequisites

- Completed Chapter 11 (Humanoid Locomotion)
- Python 3.8+ with NumPy
- Understanding of kinematics and control theory
- Basic linear algebra (matrix operations, SVD)

## Lab Files

| File | Description |
|------|-------------|
| `grasp_planner.py` | Grasp planning and quality metrics |
| `force_controller.py` | Force and impedance control |
| `README.md` | Lab overview |

## Setup

1. Navigate to the lab directory:
   ```bash
   cd labs/module-4/ch12-manipulation
   ```

2. Ensure dependencies are installed:
   ```bash
   pip install numpy
   ```

3. Run the test scripts to verify setup:
   ```bash
   python grasp_planner.py
   python force_controller.py
   ```

---

## Part 1: Understanding Grasp Planning (30 minutes)

### Background

A **grasp** is defined by contact points between the robot's fingers and the object. For stable manipulation, the grasp must have **force closure** - the ability to resist arbitrary external forces and torques.

### Key Concepts

1. **Grasp Matrix (G)**: Maps contact forces to object wrench
   ```
   w = G @ f
   ```
   Where w is the 6D wrench (force + torque) and f are contact forces.

2. **Force Closure**: Exists when the origin is inside the convex hull of the wrench space.

3. **Friction Cone**: Set of forces a contact can apply without slipping.

### Exercise 1.1: Explore Grasp Quality Metrics

Open `grasp_planner.py` and examine the `ForceClosureMetric` class.

**Task**: Modify the main function to test different object geometries:

```python
# Add to main() function
test_objects = [
    ObjectGeometry(
        center=np.array([0.3, 0.0, 0.05]),
        dimensions=np.array([0.03, 0.03, 0.08]),  # Tall thin box
        shape='box',
        surface_friction=0.4  # Lower friction
    ),
    ObjectGeometry(
        center=np.array([0.3, 0.0, 0.05]),
        dimensions=np.array([0.08, 0.08, 0.02]),  # Flat wide box
        shape='box',
        surface_friction=0.8  # Higher friction
    ),
]
```

**Questions to Answer**:
1. How does object shape affect grasp quality?
2. What is the minimum friction coefficient needed for stable grasps?
3. How many contact points are needed for force closure?

### Exercise 1.2: Implement Custom Grasp Sampling

**Task**: Add a new method to sample antipodal grasps (contacts on opposite sides):

```python
def sample_antipodal_grasps(self, obj: ObjectGeometry,
                             n_samples: int = 20) -> List[Tuple[ContactPoint, ContactPoint]]:
    """
    Sample pairs of antipodal contact points.

    Antipodal grasps have contacts with opposing normals,
    ideal for two-finger precision grasps.
    """
    # Your implementation here
    # Hint: Sample points, then find pairs where normals are nearly opposite
    pass
```

**Expected Output**:
```
Antipodal grasp pair found:
  Contact 1: pos=[0.03, 0.0, 0.02], normal=[-1, 0, 0]
  Contact 2: pos=[-0.03, 0.0, 0.02], normal=[1, 0, 0]
  Angle between normals: 180.0 degrees
```

---

## Part 2: Grasp Planning Algorithm (45 minutes)

### Exercise 2.1: Plan Grasps for Different Objects

**Task**: Use the `GraspPlanner` to generate grasps for a set of household objects:

```python
# Household objects
objects = {
    "mug": ObjectGeometry(
        center=np.array([0.0, 0.0, 0.05]),
        dimensions=np.array([0.04, 0.08]),  # radius, height
        shape='cylinder',
        mass=0.3
    ),
    "apple": ObjectGeometry(
        center=np.array([0.0, 0.0, 0.04]),
        dimensions=np.array([0.04]),  # radius
        shape='sphere',
        mass=0.15
    ),
    "book": ObjectGeometry(
        center=np.array([0.0, 0.0, 0.01]),
        dimensions=np.array([0.15, 0.10, 0.02]),
        shape='box',
        mass=0.5
    ),
}

planner = GraspPlanner(n_fingers=5)

for name, obj in objects.items():
    grasps = planner.plan_multiple_grasps(obj, n_grasps=3)
    print(f"\n{name}:")
    for i, g in enumerate(grasps):
        print(f"  Grasp {i+1}: type={g.grasp_type.value}, quality={g.quality_score:.3f}")
```

### Exercise 2.2: Visualize Grasp Configuration

**Task**: Create a simple visualization of the grasp:

```python
def visualize_grasp(grasp: GraspConfiguration, obj: ObjectGeometry):
    """Print ASCII visualization of grasp."""
    print("\nGrasp Visualization:")
    print(f"Object: {obj.shape} at {obj.center}")
    print(f"Grasp Type: {grasp.grasp_type.value}")
    print(f"\nContacts:")
    for i, c in enumerate(grasp.contacts):
        print(f"  Finger {c.finger_id}: pos={c.position.round(3)}")
        print(f"           normal={c.normal.round(3)}")
    print(f"\nWrist Pose:")
    print(f"  Position: {grasp.wrist_pose[:3, 3].round(3)}")
    print(f"  Approach: {grasp.approach_direction.round(3)}")
```

### Exercise 2.3: Grasp Ranking

**Task**: Implement a multi-criteria grasp ranking function:

```python
def rank_grasps(grasps: List[GraspConfiguration],
                obj: ObjectGeometry,
                task: str = "pick") -> List[GraspConfiguration]:
    """
    Rank grasps based on task requirements.

    Tasks:
    - "pick": Prioritize top grasps for lifting
    - "handover": Prioritize side grasps
    - "place": Prioritize stable release poses
    """
    # Your implementation here
    pass
```

---

## Part 3: Force Control Implementation (45 minutes)

### Background

**Impedance Control** makes the robot behave like a virtual mass-spring-damper:
```
M*ẍ + D*ẋ + K*(x - x_d) = F_ext
```

**Hybrid Control** partitions task space into position-controlled and force-controlled directions.

### Exercise 3.1: Tune Impedance Parameters

Open `force_controller.py` and experiment with the `ImpedanceController`.

**Task**: Find optimal impedance parameters for:

1. **Soft Contact** (assembly insertion):
   ```python
   soft_impedance = ImpedanceController(
       stiffness=np.array([100, 100, 50, 10, 10, 10]),  # Low stiffness
       damping=np.array([20, 20, 10, 2, 2, 2])
   )
   ```

2. **Stiff Tracking** (precise positioning):
   ```python
   stiff_impedance = ImpedanceController(
       stiffness=np.array([2000, 2000, 2000, 200, 200, 200]),
       damping=np.array([100, 100, 100, 10, 10, 10])
   )
   ```

**Experiment**: Simulate a peg-in-hole insertion:

```python
def simulate_insertion(controller, hole_position, peg_position):
    """Simulate insertion with force feedback."""
    dt = 0.001
    pose = peg_position.copy()
    velocity = np.zeros(6)

    for step in range(1000):
        # Simulate contact force when near hole
        distance = np.linalg.norm(pose[:3] - hole_position[:3])
        if distance < 0.01:
            # Contact force
            contact_force = np.zeros(6)
            contact_force[:3] = (hole_position[:3] - pose[:3]) * 1000
        else:
            contact_force = np.zeros(6)

        # Get control command
        command = controller.compute_command(
            pose, velocity, contact_force,
            hole_position, np.zeros(6)
        )

        # Simple dynamics
        acceleration = command / 10.0  # Virtual mass
        velocity += acceleration * dt
        pose += velocity * dt

        if step % 100 == 0:
            print(f"Step {step}: pos={pose[:3].round(4)}, force={contact_force[:3].round(2)}")

    return pose
```

### Exercise 3.2: Hybrid Force/Position Control

**Task**: Set up hybrid control for surface wiping:

```python
# Surface wiping: control force normal to surface, position tangent
hybrid = HybridForcePositionController()

# Surface normal is Z-axis
hybrid.set_control_directions(
    position_axes=[0, 1, 3, 4, 5],  # X, Y position, all rotations
    force_axes=[2]                   # Z force (pushing into surface)
)

# Desired: 5N into surface, move in XY plane
desired_force = np.array([0, 0, -5.0, 0, 0, 0])

# Simulate wiping motion
for waypoint in wiping_path:
    desired_pose = waypoint
    command = hybrid.compute_command(
        current_pose, current_velocity, measured_force,
        desired_pose, desired_force
    )
```

---

## Part 4: Slip Detection and Prevention (30 minutes)

### Background

Slip occurs when tangential force exceeds the friction limit:
```
|f_t| > μ * f_n
```

The **slip ratio** is: `|f_t| / (μ * f_n)`. Values > 1 indicate slipping.

### Exercise 4.1: Implement Slip Prevention

**Task**: Modify the `GraspForceController` to proactively prevent slip:

```python
def predictive_slip_prevention(self, contact_states, current_forces,
                                predicted_acceleration):
    """
    Adjust forces based on predicted motion to prevent slip.

    When the object accelerates, tangential forces increase.
    Preemptively increase normal forces.
    """
    adjusted_forces = current_forces.copy()

    for i, state in enumerate(contact_states):
        # Predict tangential force increase
        predicted_tangent = np.linalg.norm(predicted_acceleration[:3]) * 0.1

        # Calculate required normal force
        required_normal = (state.tangent_force + predicted_tangent) / self.slip_detector.mu

        # Apply with safety margin
        adjusted_forces[i] = max(
            current_forces[i],
            required_normal * 1.2  # 20% safety margin
        )

    return adjusted_forces
```

### Exercise 4.2: Vibration-Based Slip Detection

**Task**: Enhance slip detection using force vibration analysis:

```python
def detect_incipient_slip(self, force_history: np.ndarray) -> bool:
    """
    Detect incipient (beginning) slip from force vibrations.

    Incipient slip causes high-frequency oscillations before
    gross slip occurs.
    """
    if len(force_history) < 50:
        return False

    # Compute FFT
    fft = np.fft.fft(force_history[-50:, 2])  # Z-axis force
    frequencies = np.fft.fftfreq(50, d=0.001)

    # Look for high-frequency content (> 50 Hz)
    high_freq_power = np.sum(np.abs(fft[np.abs(frequencies) > 50]))
    total_power = np.sum(np.abs(fft))

    # If high frequencies dominate, slip is starting
    return (high_freq_power / total_power) > 0.3
```

---

## Part 5: Integrated Manipulation Task (30 minutes)

### Exercise 5.1: Pick and Place Pipeline

**Task**: Combine grasp planning and force control for a complete pick-and-place task:

```python
class PickAndPlaceController:
    """Integrated controller for pick and place."""

    def __init__(self):
        self.grasp_planner = GraspPlanner()
        self.force_controller = GraspForceController(n_fingers=3)
        self.state = "idle"

    def execute_pick(self, object_geometry, pick_pose):
        """Execute pick operation."""
        # 1. Plan grasp
        grasp = self.grasp_planner.plan_grasp(object_geometry)
        if grasp is None:
            return False, "No valid grasp found"

        # 2. Approach
        self.state = "approaching"
        approach_pose = grasp.wrist_pose.copy()
        approach_pose[:3, 3] += grasp.approach_direction * 0.1

        # 3. Move to grasp
        self.state = "grasping"
        # Use impedance for compliant contact

        # 4. Close fingers with force control
        self.state = "closing"

        # 5. Lift with slip monitoring
        self.state = "lifting"

        return True, grasp

    def execute_place(self, place_pose, release_force=2.0):
        """Execute place operation."""
        # 1. Move to place location
        self.state = "moving"

        # 2. Lower with force sensing
        self.state = "lowering"

        # 3. Release grasp
        self.state = "releasing"

        # 4. Retreat
        self.state = "retreating"

        return True
```

### Exercise 5.2: Robust Grasp Execution

**Task**: Add error recovery to handle failures:

```python
def execute_with_recovery(self, task, max_retries=3):
    """Execute task with automatic recovery."""
    for attempt in range(max_retries):
        try:
            success, result = task()
            if success:
                return True, result

            # Analyze failure
            if "slip" in str(result):
                # Increase grasp force
                self.force_controller.internal_force *= 1.5
            elif "collision" in str(result):
                # Try different approach
                pass

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")

    return False, "Max retries exceeded"
```

---

## Validation Checklist

Before completing this lab, verify:

- [ ] Grasp planner successfully plans grasps for box, cylinder, and sphere
- [ ] Force closure check correctly identifies stable vs unstable grasps
- [ ] Impedance controller produces smooth compliant behavior
- [ ] Hybrid controller correctly partitions force/position control
- [ ] Slip detector identifies slipping contacts
- [ ] Grasp force controller adjusts forces to prevent slip
- [ ] Pick and place pipeline executes successfully

## Expected Outputs

### Grasp Planner Test
```
Testing Grasp Planner
==================================================

Object 1: box
  Center: [0.3 0.  0.05]
  Dimensions: [0.06 0.04 0.1]
  Grasp Type: power
  Quality Score: 0.7234
  Contacts: 4
  Force Closure: True
```

### Force Controller Test
```
Testing Force Controllers
==================================================

1. Impedance Controller
  Command wrench: [50.0, 0.0, 40.0] N, [0.0, 0.0, 0.0] Nm

2. Hybrid Force/Position Controller
  Command wrench: [150.0, 0.0, -10.0] N, [0.0, 0.0, 0.0] Nm

3. Slip Detector
  Contact 0: slip_ratio=0.00, slipping=False
  Contact 1: slip_ratio=1.60, slipping=True
```

## Troubleshooting

### Common Issues

1. **No valid grasp found**
   - Increase `n_candidates` in `plan_grasp()`
   - Check object dimensions are reasonable
   - Verify friction coefficient is > 0.3

2. **Unstable force control**
   - Reduce stiffness gains
   - Increase damping (critical damping = 2*sqrt(K*M))
   - Check control loop frequency

3. **False slip detection**
   - Adjust `slip_threshold` (default 0.9)
   - Increase force history size for vibration detection
   - Verify friction coefficient estimate

## Further Exploration

1. **Advanced Grasp Quality**: Implement epsilon-quality metric using convex hulls
2. **Learning from Demonstration**: Record human grasps and learn grasp preferences
3. **Tactile Sensing**: Integrate tactile sensor data for contact estimation
4. **Deformable Objects**: Extend force control for soft/deformable manipulation

## References

- Murray, Li, Sastry: "A Mathematical Introduction to Robotic Manipulation"
- Siciliano et al.: "Robotics: Modelling, Planning and Control" (Chapter 9)
- MoveIt 2 Grasp Pipeline: https://moveit.ros.org/
