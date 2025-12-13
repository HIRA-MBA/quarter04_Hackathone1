# Lab 7: Unity Digital Twin - Step-by-Step Instructions

## Overview

In this lab, you will build a complete digital twin system using Unity and ROS 2. The digital twin will synchronize with a simulated robot in real-time, provide visualization dashboards, and support domain randomization for synthetic data generation.

**Estimated Time**: 3-4 hours

## Prerequisites

- Completed Lab 6 (Gazebo World)
- Unity 2022 LTS or later installed
- Basic C# programming knowledge
- ROS 2 Humble workspace set up
- Understanding of URDF and robot models

## Learning Outcomes

By completing this lab, you will:

1. Set up Unity for robotics development with ROS-TCP-Connector
2. Import and configure robot models from URDF
3. Implement bidirectional ROS 2 communication
4. Create real-time state synchronization
5. Build a monitoring dashboard
6. Implement domain randomization for training data

---

## Part 1: Unity Project Setup (30 minutes)

### Step 1.1: Create New Unity Project

1. Open Unity Hub
2. Click "New project"
3. Select **3D (URP)** template (or HDRP for higher fidelity)
4. Name it `PhysicalAI_DigitalTwin`
5. Click "Create project"

### Step 1.2: Install Required Packages

Open the Package Manager (Window → Package Manager) and add these packages by git URL (+ button → Add package from git URL):

```
https://github.com/Unity-Technologies/ROS-TCP-Connector.git?path=/com.unity.robotics.ros-tcp-connector

https://github.com/Unity-Technologies/URDF-Importer.git?path=/com.unity.robotics.urdf-importer

https://github.com/Unity-Technologies/ROS-TCP-Connector.git?path=/com.unity.robotics.visualizations
```

Wait for each package to install before adding the next.

### Step 1.3: Configure ROS Settings

1. Go to **Robotics → ROS Settings**
2. Set Protocol to **ROS2**
3. Set ROS IP Address to `127.0.0.1` (or your robot's IP)
4. Set ROS Port to `10000`
5. Click "Connect"

### Step 1.4: Create Project Structure

Create the following folder structure in your Assets:

```
Assets/
├── Robots/
│   └── Humanoid/
├── Environments/
├── Scripts/
│   ├── ROS/
│   └── UI/
├── Prefabs/
├── Materials/
└── Scenes/
```

**Expected Output**: Unity project with ROS packages installed and folder structure created.

---

## Part 2: Import Robot Model (30 minutes)

### Step 2.1: Prepare URDF Files

Copy your humanoid robot URDF from Lab 4 to `Assets/Robots/Humanoid/`:

```bash
# From your ROS workspace
cp -r ~/ros2_ws/src/humanoid_description/urdf/* /path/to/unity/Assets/Robots/Humanoid/
```

### Step 2.2: Import URDF

1. In Unity, right-click on your `.urdf` file
2. Select **Import Robot from URDF**
3. Configure import settings:
   - Axis Type: **Z-Axis**
   - Mesh Decomposer: **VHACD**
   - Convex Decomposition Resolution: **100000**

### Step 2.3: Configure Joints

After import, select the robot root and configure each joint:

1. Select a joint GameObject
2. In the Articulation Body component:
   - Set **X Drive Stiffness**: 10000
   - Set **X Drive Damping**: 100
   - Set **X Drive Force Limit**: Max value
3. Repeat for all joints

### Step 2.4: Position Robot

1. Position the robot at origin (0, 0, 0)
2. Ensure the robot is upright and grounded
3. Save the scene as `DigitalTwin.unity`

**Expected Output**: Robot model visible in scene with properly configured joints.

---

## Part 3: ROS 2 Communication Setup (45 minutes)

### Step 3.1: Install ROS-TCP-Endpoint

On your ROS 2 machine:

```bash
cd ~/ros2_ws/src
git clone https://github.com/Unity-Technologies/ROS-TCP-Endpoint.git -b ROS2v0.7.0
cd ~/ros2_ws
colcon build --packages-select ros_tcp_endpoint
source install/setup.bash
```

### Step 3.2: Add RosBridge Script

1. Copy `RosBridge.cs` from this lab to `Assets/Scripts/ROS/`
2. Create an empty GameObject named `RosBridge`
3. Add the `RosBridge` component to it

### Step 3.3: Configure RosBridge

In the Inspector, configure the RosBridge component:

1. **ROS IP**: `127.0.0.1` (or your ROS machine IP)
2. **ROS Port**: `10000`
3. **Robot Base**: Drag the robot's root transform
4. **Joints**: Drag all ArticulationBody components
5. **Topic Names**: Use defaults or customize

### Step 3.4: Test Connection

Terminal 1 - Start ROS TCP Endpoint:
```bash
source ~/ros2_ws/install/setup.bash
ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_IP:=0.0.0.0
```

Terminal 2 - Start Gazebo simulation (from Lab 6):
```bash
ros2 launch gazebo_world robot_spawn.launch.py
```

Terminal 3 - Verify topics:
```bash
ros2 topic list
# Should see /joint_states, /odom, etc.
```

In Unity:
1. Press Play
2. Check Console for connection messages
3. Verify robot moves when Gazebo robot moves

**Expected Output**: Unity robot mirrors Gazebo robot movements in real-time.

---

## Part 4: State Synchronization (30 minutes)

### Step 4.1: Verify Joint Synchronization

1. In Gazebo, command the robot to move
2. Observe Unity robot following

Send test joint commands from terminal:
```bash
ros2 topic pub /joint_states sensor_msgs/msg/JointState "{
  header: {stamp: {sec: 0, nanosec: 0}, frame_id: ''},
  name: ['joint_1'],
  position: [0.5],
  velocity: [],
  effort: []
}" --once
```

### Step 4.2: Verify Odometry Synchronization

Send test odometry:
```bash
ros2 topic pub /odom nav_msgs/msg/Odometry "{
  header: {frame_id: 'odom'},
  child_frame_id: 'base_link',
  pose: {pose: {position: {x: 1.0, y: 0.0, z: 0.0}}}
}" --once
```

### Step 4.3: Tune Interpolation

Adjust RosBridge settings for smooth motion:

1. **Position Lerp Speed**: Higher = faster response, lower = smoother
2. **Rotation Lerp Speed**: Same as above
3. **Joint Lerp Speed**: Higher for responsive joints

Recommended starting values:
- Position Lerp: 10
- Rotation Lerp: 10
- Joint Lerp: 20

**Expected Output**: Smooth, responsive synchronization between Gazebo and Unity.

---

## Part 5: Dashboard UI (45 minutes)

### Step 5.1: Create Dashboard Canvas

1. Right-click in Hierarchy → UI → Canvas
2. Name it `DashboardCanvas`
3. Set Canvas Scaler to **Scale With Screen Size**
4. Reference Resolution: 1920 x 1080

### Step 5.2: Create Status Panel

1. Create a Panel under Canvas
2. Position in top-left corner
3. Add components:
   - Connection indicator (Image)
   - Status text (TextMeshPro)
   - Latency text (TextMeshPro)

### Step 5.3: Create Robot State Panel

1. Create another Panel
2. Position below status panel
3. Add:
   - Position text
   - Rotation text
   - Velocity text

### Step 5.4: Add TwinDashboard Script

1. Copy `TwinDashboard.cs` to `Assets/Scripts/UI/`
2. Create empty GameObject named `Dashboard`
3. Add `TwinDashboard` component
4. Connect UI references in Inspector

### Step 5.5: Create Joint Display

1. Create a prefab for joint display:
   - Panel with name text, value text, and slider
2. Assign to `TwinDashboard.jointDisplayPrefab`
3. Create parent container for joint displays

**Expected Output**: Real-time dashboard showing robot state, connection status, and joint positions.

---

## Part 6: Domain Randomization (30 minutes)

### Step 6.1: Add DomainRandomizer

1. Copy `DomainRandomizer.cs` to `Assets/Scripts/`
2. Create empty GameObject named `DomainRandomizer`
3. Add the component

### Step 6.2: Configure Lighting Randomization

1. Assign the scene's Directional Light to **Sun Light**
2. Set time range (e.g., 8-18 for daytime)
3. Set intensity range (0.5-2.0)
4. Set color temperature range (4000K-7000K)

### Step 6.3: Configure Material Randomization

1. Create alternative materials for floor, walls, objects
2. Add Material Targets in Inspector
3. For each target:
   - Assign renderer
   - Add possible materials
   - Enable color randomization

### Step 6.4: Configure Camera Randomization

1. Add camera settings
2. Set position noise range (0.1m)
3. Set rotation noise range (2 degrees)
4. Optionally randomize FOV

### Step 6.5: Test Randomization

1. Press Play
2. Call `RandomizeAll()` from code or button
3. Observe lighting/material changes

Create a test button:
```csharp
// Add to DomainRandomizer or separate script
void OnGUI()
{
    if (GUILayout.Button("Randomize"))
    {
        RandomizeAll();
    }
}
```

**Expected Output**: Pressing randomize changes lighting, materials, and camera parameters.

---

## Part 7: Integration and Testing (30 minutes)

### Step 7.1: Full System Test

1. Start all components:
   - ROS TCP Endpoint
   - Gazebo simulation
   - Unity in Play mode

2. Verify:
   - [ ] Connection indicator shows green
   - [ ] Robot synchronizes with Gazebo
   - [ ] Dashboard updates in real-time
   - [ ] Domain randomization works

### Step 7.2: Performance Check

Monitor in Unity Profiler (Window → Analysis → Profiler):

- Target: 60+ FPS
- Check for GC spikes
- Monitor network latency

### Step 7.3: Record Demo

1. Use Unity Recorder or OBS
2. Record a 2-3 minute demo showing:
   - Real-time synchronization
   - Dashboard functionality
   - Domain randomization

---

## Troubleshooting

### Connection Issues

**Problem**: Unity won't connect to ROS
**Solutions**:
1. Check firewall settings
2. Verify ROS TCP Endpoint is running
3. Confirm IP addresses match
4. Check port 10000 is not blocked

```bash
# Test port connectivity
nc -zv <ROS_IP> 10000
```

### Synchronization Lag

**Problem**: Unity robot lags behind Gazebo
**Solutions**:
1. Increase lerp speeds
2. Check network latency
3. Reduce publish rate if bandwidth limited

### Joint Jittering

**Problem**: Joints shake or oscillate
**Solutions**:
1. Increase joint damping
2. Lower lerp speed
3. Check for conflicting position updates

### URDF Import Fails

**Problem**: Robot imports incorrectly
**Solutions**:
1. Check mesh file paths are relative
2. Ensure meshes are in compatible format (STL, DAE)
3. Try manual joint configuration

---

## Acceptance Criteria

Your lab is complete when:

- [ ] Unity project runs without errors
- [ ] ROS-TCP-Connector connects successfully
- [ ] Robot URDF imported and configured
- [ ] Joint states synchronize at 30+ Hz
- [ ] Odometry synchronizes correctly
- [ ] Dashboard displays connection status
- [ ] Dashboard shows robot position/rotation
- [ ] Dashboard shows joint positions
- [ ] Domain randomization changes lighting
- [ ] Domain randomization changes materials
- [ ] Performance maintains 60+ FPS
- [ ] Demo video recorded

---

## Challenge Extensions

### Extension 1: Camera Feed Display

Implement camera image subscription and display:
1. Subscribe to `/camera/image_raw` topic
2. Convert ROS Image to Unity Texture2D
3. Display in dashboard RawImage

### Extension 2: Point Cloud Visualization

Visualize LIDAR data:
1. Subscribe to `/scan` or `/points` topic
2. Render points using particle system or VFX Graph
3. Add color based on intensity or distance

### Extension 3: Teleop Control

Add keyboard/gamepad control:
1. Read Unity Input
2. Publish Twist messages to `/cmd_vel`
3. Add on-screen joystick for mobile

---

## Files in This Lab

| File | Description |
|------|-------------|
| `RosBridge.cs` | Main ROS 2 communication handler |
| `TwinDashboard.cs` | Dashboard UI controller |
| `DomainRandomizer.cs` | Domain randomization for training data |
| `INSTRUCTIONS.md` | This file |
| `README.md` | Lab overview |

---

## Next Steps

After completing this lab:
1. Review the capstone requirements in Chapter 7
2. Plan your complete digital twin project
3. Proceed to Module 3: NVIDIA Isaac
