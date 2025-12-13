# Lab 4: URDF & Humanoid Modeling

## Objectives

By the end of this lab, you will:
1. Understand URDF structure (links, joints, properties)
2. Create a humanoid robot description from scratch
3. Use Xacro for modular robot descriptions
4. Visualize robots in RViz

## Prerequisites

- Completed Labs 1-3
- ROS 2 Humble with visualization tools
- Basic understanding of 3D geometry

## URDF Concepts

### Links
Links are rigid bodies with:
- **Visual**: What you see (mesh or primitive)
- **Collision**: What collides (often simplified)
- **Inertial**: Mass and inertia for physics

### Joints
Joints connect links:
- **fixed**: No movement
- **revolute**: Rotation with limits
- **continuous**: Rotation without limits
- **prismatic**: Linear sliding

## Part 1: Examine the Basic URDF

### Exercise 1.1: Open and read humanoid.urdf

```bash
cat urdf/humanoid.urdf
```

Identify:
- [ ] How many links are there?
- [ ] How many joints are there?
- [ ] What joint types are used?

### Exercise 1.2: Visualize in RViz

```bash
# Install if needed
sudo apt install ros-humble-urdf-tutorial ros-humble-joint-state-publisher-gui

# View the URDF
ros2 launch urdf_tutorial display.launch.py model:=/workspace/module-1/ch04-urdf-humanoid/urdf/humanoid.urdf
```

### Exercise 1.3: Move joints with GUI

In RViz:
1. Enable the Joint State Publisher GUI
2. Move sliders to see joints rotate
3. Observe how child links move with parents

## Part 2: Understanding Xacro

### Exercise 2.1: Compare URDF and Xacro

Open both files and note the differences:
- Properties (variables)
- Macros (reusable components)
- Math expressions

### Exercise 2.2: Generate URDF from Xacro

```bash
# Install xacro if needed
sudo apt install ros-humble-xacro

# Generate URDF
xacro urdf/humanoid.xacro > urdf/humanoid_from_xacro.urdf

# Compare file sizes
ls -la urdf/
```

### Exercise 2.3: Understand macros

In `humanoid.xacro`, find:
- The `arm` macro
- The `leg` macro
- How `reflect` parameter creates left/right symmetry

## Part 3: Modify the Robot

### Challenge 1: Add a sensor

Add a camera link to the head:

```xml
<link name="camera_link">
  <visual>
    <geometry>
      <box size="0.02 0.05 0.02"/>
    </geometry>
    <material name="black">
      <color rgba="0.1 0.1 0.1 1.0"/>
    </material>
  </visual>
</link>

<joint name="camera_joint" type="fixed">
  <parent link="head"/>
  <child link="camera_link"/>
  <origin xyz="0.1 0 0" rpy="0 0 0"/>
</joint>
```

### Challenge 2: Add joint limits

Modify joint limits to be more realistic:
- Elbow: can only bend one way (0 to 150 degrees)
- Knee: can only bend one way (0 to 140 degrees)

### Challenge 3: Create a Xacro property

Add a `scale` property that scales the entire robot:

```xml
<xacro:property name="scale" value="1.0"/>
<!-- Use ${scale * original_value} for dimensions -->
```

### Challenge 4: Add a gripper

Create a simple two-finger gripper attached to each hand.

## Part 4: Validation

### Exercise 4.1: Check URDF validity

```bash
# Install check tool
sudo apt install liburdfdom-tools

# Check URDF
check_urdf urdf/humanoid.urdf
```

### Exercise 4.2: View TF tree

```bash
# In one terminal, publish robot state
ros2 launch urdf_tutorial display.launch.py model:=urdf/humanoid.urdf

# In another terminal
ros2 run tf2_tools view_frames
```

This generates `frames.pdf` showing the transform tree.

## Part 5: ROS 2 Integration

### Exercise 5.1: Create a launch file

Create `launch/display.launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    urdf_file = os.path.join(
        get_package_share_directory('humanoid_description'),
        'urdf', 'humanoid.urdf'
    )

    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description}]
        ),
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', 'config/humanoid.rviz']
        ),
    ])
```

### Exercise 5.2: View joint limits

```bash
ros2 topic echo /joint_states
```

## Verification Checklist

- [ ] Successfully visualized the humanoid in RViz
- [ ] Moved joints using the GUI
- [ ] Generated URDF from Xacro
- [ ] Added at least one modification (sensor, gripper, etc.)
- [ ] Validated URDF with check_urdf
- [ ] Viewed the TF tree

## Common Issues

### "Package not found" error
Make sure to build and source your workspace.

### Robot not visible in RViz
- Check "Fixed Frame" is set to "base_link"
- Add RobotModel display
- Verify robot_description parameter is published

### Joint not moving
- Check joint type (fixed joints don't move)
- Verify axis is correct
- Check joint limits

## Key Concepts Summary

| Element | Purpose | Example |
|---------|---------|---------|
| `<link>` | Rigid body | Torso, arm segment |
| `<joint>` | Connection | Shoulder, elbow |
| `<visual>` | Appearance | Mesh, cylinder |
| `<collision>` | Physics | Simplified geometry |
| `<inertial>` | Dynamics | Mass, inertia tensor |

## Next Steps

Proceed to Lab 5: Edge Deployment, where you'll learn to deploy ROS 2 on embedded systems.

## Resources

- [URDF Specification](http://wiki.ros.org/urdf/XML)
- [Xacro Guide](http://wiki.ros.org/xacro)
- [RViz User Guide](https://docs.ros.org/en/humble/Tutorials/Intermediate/RViz/RViz-User-Guide.html)
