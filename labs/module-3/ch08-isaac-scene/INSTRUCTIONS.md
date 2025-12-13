# Lab 8: Isaac Sim Scene Creation - Step-by-Step Instructions

## Overview

In this lab, you will learn the fundamentals of NVIDIA Isaac Sim by creating a warehouse environment, importing a robot, adding sensors, and generating synthetic training data with domain randomization.

**Estimated Time**: 2-3 hours

## Prerequisites

- NVIDIA RTX GPU (RTX 2070+ recommended, RTX 3080+ ideal)
- Isaac Sim 2023.1.1 or later installed via Omniverse Launcher
- Completed Module 2 (Digital Twin)
- Basic Python knowledge
- Ubuntu 22.04 (recommended) or Windows 10/11

## Learning Outcomes

By completing this lab, you will:

1. Navigate the Isaac Sim interface and understand USD
2. Create robotic scenes with physics
3. Import robots from URDF
4. Add and configure sensors (camera, IMU, LIDAR)
5. Generate synthetic data with Replicator

---

## Part 1: Isaac Sim Setup (20 minutes)

### Step 1.1: Launch Isaac Sim

1. Open **Omniverse Launcher**
2. Go to the **Library** tab
3. Click **Launch** on Isaac Sim
4. Wait for Isaac Sim to fully load (may take 2-3 minutes first time)

### Step 1.2: Interface Overview

Familiarize yourself with the main panels:

| Panel | Purpose |
|-------|---------|
| **Viewport** | 3D scene view |
| **Stage** | Scene hierarchy (USD prims) |
| **Property** | Selected object properties |
| **Content** | Asset browser |
| **Console** | Python output and errors |

### Step 1.3: Verify Python Environment

Open the Script Editor (Window → Script Editor) and run:

```python
from omni.isaac.core import World
print("Isaac Sim Python OK!")
```

You should see the output in the Console panel.

**Expected Output**: "Isaac Sim Python OK!" message in console.

---

## Part 2: Create Warehouse Scene (30 minutes)

### Step 2.1: New Stage

1. File → New Stage
2. Save as `warehouse_scene.usd`

### Step 2.2: Add Ground Plane

1. Create → Physics → Ground Plane
2. Or use Python:

```python
from omni.isaac.core import World
from omni.isaac.core.objects import GroundPlane

world = World()
world.scene.add_default_ground_plane()
world.reset()
```

### Step 2.3: Add Walls and Obstacles

Using the Stage panel:
1. Create → Shapes → Cube
2. Scale to create wall dimensions
3. Position using Transform properties
4. Repeat for multiple walls

Or run the script:

```python
from omni.isaac.core.objects import VisualCuboid
import numpy as np

# North wall
wall = VisualCuboid(
    prim_path="/World/Walls/wall_north",
    name="wall_north",
    position=np.array([0, 5, 1.5]),
    scale=np.array([10, 0.2, 3]),
    color=np.array([0.7, 0.7, 0.7])
)
world.scene.add(wall)
```

### Step 2.4: Configure Lighting

1. Create → Light → Distant Light (sun)
2. Set Intensity to 1000
3. Set Color Temperature to 6500K
4. Create → Light → Dome Light (ambient)
5. Set Intensity to 300

### Step 2.5: Save Scene

1. File → Save
2. Verify `warehouse_scene.usd` is saved

**Expected Output**: Warehouse environment with walls, floor, and lighting visible.

---

## Part 3: Import Robot (30 minutes)

### Step 3.1: URDF Import

If you have a custom URDF from Module 1:

1. Isaac Utils → Workflows → URDF Importer
2. Select your URDF file
3. Configure import settings:
   - Fix Base: Unchecked (for mobile robots)
   - Self Collision: Unchecked
   - Merge Fixed Joints: Checked
4. Click Import

### Step 3.2: Import via Python

```python
from omni.isaac.urdf import _urdf
import omni.kit.commands

# Configure import
urdf_config = _urdf.ImportConfig()
urdf_config.merge_fixed_joints = False
urdf_config.convex_decomp = True
urdf_config.fix_base = False
urdf_config.default_drive_strength = 1000.0
urdf_config.default_position_drive_damping = 100.0

# Import
result, prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path="/path/to/your/robot.urdf",
    import_config=urdf_config,
    dest_path="/World/Robot"
)

print(f"Robot imported at: {prim_path}")
```

### Step 3.3: Use Built-in Robot (Alternative)

If you don't have a URDF, use Isaac's built-in robots:

```python
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path

assets_root = get_assets_root_path()
robot_usd = f"{assets_root}/Isaac/Robots/Franka/franka_alt_fingers.usd"

add_reference_to_stage(robot_usd, "/World/Robot")
```

### Step 3.4: Verify Robot Import

In the Stage panel:
1. Expand `/World/Robot`
2. Verify joints and links are present
3. Select a joint and check Articulation properties

**Expected Output**: Robot visible in scene with articulated joints.

---

## Part 4: Joint Control (30 minutes)

### Step 4.1: Create Articulation Wrapper

```python
from omni.isaac.core.articulations import Articulation
import numpy as np

# Wrap robot as articulation
robot = Articulation(
    prim_path="/World/Robot",
    name="robot"
)
world.scene.add(robot)
world.reset()

# Get joint info
print(f"DOFs: {robot.num_dof}")
print(f"Joint names: {robot.dof_names}")
```

### Step 4.2: Position Control

```python
# Set joint targets
num_joints = robot.num_dof
target_positions = np.zeros(num_joints)
target_positions[0] = 0.5  # Move joint 0 to 0.5 rad

robot.set_joint_position_targets(target_positions)

# Step simulation
for _ in range(100):
    world.step(render=True)
```

### Step 4.3: Sinusoidal Motion

Run the provided `spawn_robot.py` script:

```bash
cd /isaac-sim
./python.sh /path/to/labs/module-3/ch08-isaac-scene/spawn_robot.py
```

Watch the robot joints move in a sinusoidal pattern.

### Step 4.4: Read Joint States

```python
# Get current states
positions = robot.get_joint_positions()
velocities = robot.get_joint_velocities()
efforts = robot.get_joint_efforts()

print(f"Positions: {positions}")
print(f"Velocities: {velocities}")
```

**Expected Output**: Robot joints moving and state values printed to console.

---

## Part 5: Sensor Setup (30 minutes)

### Step 5.1: Add Camera

```python
from omni.isaac.sensor import Camera
import numpy as np

camera = Camera(
    prim_path="/World/Camera",
    name="main_camera",
    position=np.array([3.0, 0.0, 2.0]),
    frequency=30,
    resolution=(640, 480),
    orientation=np.array([0.0, 0.383, 0.0, 0.924])
)
world.scene.add(camera)
world.reset()
camera.initialize()
```

### Step 5.2: Get Camera Data

```python
# After world.step()
rgb = camera.get_rgba()[:, :, :3]
depth = camera.get_depth()

print(f"RGB shape: {rgb.shape}")
print(f"Depth range: [{depth.min():.2f}, {depth.max():.2f}]")
```

### Step 5.3: Add IMU

```python
from omni.isaac.sensor import IMUSensor

imu = IMUSensor(
    prim_path="/World/Robot/base_link/IMU",
    name="robot_imu",
    frequency=200,
    translation=np.array([0, 0, 0])
)
world.scene.add(imu)
```

### Step 5.4: Get IMU Data

```python
# After world.step()
imu_data = imu.get_current_frame()
print(f"Linear acc: {imu_data['lin_acc']}")
print(f"Angular vel: {imu_data['ang_vel']}")
```

### Step 5.5: Attach Camera to Robot (Optional)

To mount camera on robot:

```python
camera = Camera(
    prim_path="/World/Robot/base_link/camera",
    name="robot_camera",
    position=np.array([0.1, 0.0, 0.2]),  # Local offset
    frequency=30,
    resolution=(640, 480)
)
```

**Expected Output**: Camera and IMU providing data each simulation step.

---

## Part 6: Synthetic Data Generation (30 minutes)

### Step 6.1: Understand Replicator

Replicator is Isaac Sim's synthetic data generation framework:
- Domain randomization (lighting, materials, poses)
- Multi-format output (RGB, depth, segmentation, bounding boxes)
- Scalable generation (headless mode)

### Step 6.2: Run Data Generation Script

```bash
cd /isaac-sim
./python.sh /path/to/labs/module-3/ch08-isaac-scene/synthetic_data_generator.py
```

### Step 6.3: Configure Randomization

Edit the script to customize:

```python
# Object randomization
generator.setup_object_randomization(
    position_range=((-2, -2, 0.1), (2, 2, 1.0)),  # Larger area
    scale_range=((0.05, 0.05, 0.05), (0.3, 0.3, 0.3))  # Larger objects
)

# More frames
generator.generate(num_frames=500)
```

### Step 6.4: Inspect Generated Data

Check the output directory:

```bash
ls /tmp/isaac_synthetic_data/
# Should see:
# - rgb/
# - distance_to_camera/
# - semantic_segmentation/
# - instance_segmentation/
# - bounding_box_2d_tight/
```

View images:
```bash
eog /tmp/isaac_synthetic_data/rgb/rgb_0001.png
```

### Step 6.5: Use Generated Data

The COCO-format annotations can be used directly with:
- YOLO training
- Detectron2
- PyTorch detection models

**Expected Output**: Directory with synthetic images and annotations.

---

## Part 7: Integration Test (20 minutes)

### Step 7.1: Full Pipeline Test

Run the complete spawn_robot.py script:

```bash
./python.sh /path/to/labs/module-3/ch08-isaac-scene/spawn_robot.py
```

Verify:
- [ ] Scene loads with warehouse environment
- [ ] Robot is visible and articulated
- [ ] Camera provides RGB and depth data
- [ ] Joints move with sinusoidal control
- [ ] State values print correctly

### Step 7.2: Performance Check

Monitor GPU usage:
```bash
nvidia-smi -l 1
```

Target metrics:
- GPU utilization: 50-80%
- VRAM usage: < 80% of available
- Frame rate: 30+ FPS

---

## Troubleshooting

### Isaac Sim Won't Start

**Problem**: Crash on launch
**Solutions**:
1. Check GPU driver version (525+)
2. Verify GPU has enough VRAM
3. Try launching with `--/app/window/width=800 --/app/window/height=600`

### URDF Import Fails

**Problem**: Robot looks wrong after import
**Solutions**:
1. Check mesh file paths in URDF
2. Ensure meshes are STL or DAE format
3. Try with `convex_decomp = False`

### Camera Returns Empty Data

**Problem**: `get_rgba()` returns None
**Solutions**:
1. Call `camera.initialize()` after `world.reset()`
2. Wait one simulation step before reading
3. Check camera is in viewport frustum

### Low Frame Rate

**Problem**: Simulation runs slowly
**Solutions**:
1. Reduce physics timestep
2. Disable ray tracing (use Path Tracing only for renders)
3. Reduce render resolution
4. Check GPU is not thermal throttling

---

## Acceptance Criteria

Your lab is complete when:

- [ ] Isaac Sim launches successfully
- [ ] Warehouse scene with walls and obstacles created
- [ ] Robot imported and visible in scene
- [ ] Robot joints respond to position commands
- [ ] Camera provides RGB and depth images
- [ ] IMU provides acceleration and orientation data
- [ ] Synthetic data generation produces valid images
- [ ] Output includes segmentation masks and bounding boxes
- [ ] Performance maintains 30+ FPS

---

## Challenge Extensions

### Extension 1: Custom Materials

Create PBR materials with textures:

```python
import omni.kit.commands

omni.kit.commands.execute(
    "CreateMdlMaterialPrim",
    mtl_url="OmniPBR.mdl",
    mtl_name="OmniPBR",
    mtl_path="/World/Materials/CustomMaterial"
)
```

### Extension 2: Robot on Conveyor

Add animated conveyor belt that moves the robot.

### Extension 3: Multi-Camera Setup

Configure 4 cameras (front, back, left, right) and generate multi-view data.

---

## Files in This Lab

| File | Description |
|------|-------------|
| `spawn_robot.py` | Scene creation and robot control |
| `synthetic_data_generator.py` | Replicator-based data generation |
| `INSTRUCTIONS.md` | This file |
| `README.md` | Lab overview |

---

## Next Steps

After completing this lab:
1. Move to Chapter 9: Isaac ROS and GPU Perception
2. Learn to integrate Isaac Sim with ROS 2
3. Implement GPU-accelerated perception pipelines
