# Lab 6: Gazebo Physics Simulation

## Objectives

By the end of this lab, you will:
1. Create and customize Gazebo simulation worlds
2. Spawn robots into the simulation
3. Understand SDF world file format
4. Bridge Gazebo and ROS 2 topics

## Prerequisites

- Completed Module 1 labs
- ROS 2 Humble with Gazebo (gz-sim)
- Understanding of robot models (URDF from Lab 4)

## Gazebo Overview

Gazebo is a physics simulator for robotics:
- Accurate physics (ODE, Bullet, DART engines)
- Sensor simulation (cameras, LIDAR, IMU)
- Plugin system for custom behaviors
- Integration with ROS 2 via ros_gz bridge

## Part 1: Launch the World

### Exercise 1.1: Start Gazebo with our world

```bash
# Navigate to lab directory
cd /workspace/module-2/ch06-gazebo-world

# Launch Gazebo with our world
gz sim worlds/robot_world.sdf
```

### Exercise 1.2: Explore the world

In Gazebo:
1. Use mouse to orbit (right-click drag)
2. Zoom with scroll wheel
3. Click on objects to select them
4. View object properties in the panel

### Exercise 1.3: Understand the SDF structure

Open `worlds/robot_world.sdf` and identify:
- [ ] Physics configuration
- [ ] Lighting setup
- [ ] Ground plane
- [ ] Wall models
- [ ] Obstacle models
- [ ] Dynamic vs static objects

## Part 2: Spawn a Robot

### Exercise 2.1: Build the package

```bash
cd ~/ros2_ws/src
cp -r /workspace/module-2/ch06-gazebo-world .
cd ~/ros2_ws
colcon build --packages-select gazebo_world
source install/setup.bash
```

### Exercise 2.2: Launch Gazebo and spawn robot

Terminal 1 - Start Gazebo:
```bash
gz sim /path/to/worlds/robot_world.sdf
```

Terminal 2 - Spawn robot:
```bash
ros2 run gazebo_world spawn_robot --ros-args \
  -p robot_name:=my_robot \
  -p x:=0.0 \
  -p y:=0.0 \
  -p z:=0.2
```

### Exercise 2.3: View robot topics

```bash
# List topics
ros2 topic list

# Should see:
# /cmd_vel
# /odom
```

## Part 3: Control the Robot

### Exercise 3.1: Drive the robot

```bash
# Send velocity command
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.5}, angular: {z: 0.0}}" -r 10
```

### Exercise 3.2: Read odometry

```bash
ros2 topic echo /odom
```

### Exercise 3.3: Keyboard control

Install and use teleop:
```bash
sudo apt install ros-humble-teleop-twist-keyboard
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel
```

## Part 4: ROS-Gazebo Bridge

### Exercise 4.1: Understand the bridge

The `ros_gz_bridge` connects Gazebo and ROS 2:
```bash
ros2 run ros_gz_bridge parameter_bridge \
  /cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist \
  /odom@nav_msgs/msg/Odometry@gz.msgs.Odometry
```

### Exercise 4.2: Bridge more topics

Add a LIDAR bridge:
```bash
ros2 run ros_gz_bridge parameter_bridge \
  /scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan
```

## Part 5: Customize the World

### Challenge 1: Add more obstacles

Edit `robot_world.sdf` to add:
- A sphere obstacle
- A ramp
- A narrow corridor

### Challenge 2: Add a sensor

Add a LIDAR sensor to the robot SDF:
```xml
<sensor name="lidar" type="gpu_lidar">
  <pose>0 0 0.1 0 0 0</pose>
  <topic>scan</topic>
  <update_rate>10</update_rate>
  <lidar>
    <scan>
      <horizontal>
        <samples>360</samples>
        <resolution>1</resolution>
        <min_angle>-3.14159</min_angle>
        <max_angle>3.14159</max_angle>
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>10.0</max>
    </range>
  </lidar>
</sensor>
```

### Challenge 3: Dynamic objects

Create obstacles that move:
```xml
<plugin filename="gz-sim-triggered-publisher-system"
        name="gz::sim::systems::TriggeredPublisher">
  <!-- Move obstacle on trigger -->
</plugin>
```

### Challenge 4: Multiple robots

Spawn multiple robots with different namespaces:
```bash
ros2 run gazebo_world spawn_robot --ros-args \
  -p robot_name:=robot1 -p x:=-2.0 -p robot_namespace:=robot1

ros2 run gazebo_world spawn_robot --ros-args \
  -p robot_name:=robot2 -p x:=2.0 -p robot_namespace:=robot2
```

## Part 6: Recording and Playback

### Exercise 6.1: Record simulation

```bash
gz log record --path /tmp/sim_recording.gz
```

### Exercise 6.2: Playback

```bash
gz log playback --path /tmp/sim_recording.gz
```

## Verification Checklist

- [ ] Launched Gazebo with robot_world.sdf
- [ ] Spawned a robot into the world
- [ ] Drove the robot with cmd_vel
- [ ] Read odometry data
- [ ] Used keyboard teleop
- [ ] Made at least one modification to the world

## SDF Reference

### Common Elements

| Element | Description |
|---------|-------------|
| `<world>` | Top-level container |
| `<model>` | A robot or object |
| `<link>` | Rigid body |
| `<joint>` | Connection between links |
| `<collision>` | Physics geometry |
| `<visual>` | Render geometry |
| `<sensor>` | Simulated sensor |
| `<plugin>` | Custom behavior |

### Static vs Dynamic

```xml
<!-- Static: doesn't move, efficient -->
<model name="wall">
  <static>true</static>
  ...
</model>

<!-- Dynamic: responds to physics -->
<model name="box">
  <static>false</static>
  <link>
    <inertial>...</inertial>
  </link>
</model>
```

## Common Issues

### Gazebo won't start
- Check GPU drivers: `glxinfo | grep OpenGL`
- Try software rendering: `export LIBGL_ALWAYS_SOFTWARE=1`

### Robot falls through floor
- Ensure collision geometry is defined
- Check spawn z-position is above ground

### Topics not appearing
- Verify ros_gz_bridge is running
- Check topic names match

## Next Steps

Proceed to Lab 7: Unity Digital Twin, where you'll create a Unity-based visualization.

## Resources

- [Gazebo Documentation](https://gazebosim.org/docs)
- [SDF Specification](http://sdformat.org/spec)
- [ros_gz_bridge](https://github.com/gazebosim/ros_gz)
