# Lab 2: Sensor Fusion - Step-by-Step Instructions

## Objectives

By the end of this lab, you will:
1. Understand common robot sensor types (camera, LIDAR, IMU)
2. Subscribe to multiple sensor topics simultaneously
3. Process different sensor data formats
4. Implement basic sensor fusion logic

## Prerequisites

- Completed Lab 1 (Hello Robot)
- Understanding of ROS 2 publishers/subscribers
- Basic Python knowledge

## Sensor Overview

### Camera (sensor_msgs/Image)
- Provides visual information (RGB images)
- Used for: object detection, visual SLAM, lane following
- Typical rate: 30-60 FPS

### LIDAR (sensor_msgs/LaserScan)
- Provides distance measurements in a plane
- Used for: obstacle detection, mapping, localization
- Typical rate: 10-40 Hz

### IMU (sensor_msgs/Imu)
- Provides orientation, angular velocity, linear acceleration
- Used for: pose estimation, motion detection, balance
- Typical rate: 100-1000 Hz

## Part 1: Build and Setup

### Step 1: Copy package to workspace

```bash
cd ~/ros2_ws/src
cp -r /workspace/module-1/ch02-sensor-fusion .
```

### Step 2: Build

```bash
cd ~/ros2_ws
colcon build --packages-select sensor_fusion
source install/setup.bash
```

## Part 2: Run the Sensor Simulator

### Exercise 2.1: Start the simulator

```bash
ros2 run sensor_fusion sensor_simulator
```

### Exercise 2.2: Inspect the topics

In another terminal:
```bash
ros2 topic list
```

**Expected Topics:**
- `/camera/image_raw`
- `/scan`
- `/imu/data`

### Exercise 2.3: Echo sensor data

```bash
# View LIDAR data
ros2 topic echo /scan --once

# View IMU data
ros2 topic echo /imu/data --once

# View camera info (raw data is too large)
ros2 topic info /camera/image_raw
```

## Part 3: Run the Fusion Node

### Exercise 3.1: Start fusion

In a new terminal:
```bash
ros2 run sensor_fusion fusion_node
```

### Exercise 3.2: View fusion output

```bash
ros2 topic echo /fusion/status
```

**Expected Output:**
```
data: 'Fusion Status | Sensors: CAM:OK LDR:OK IMU:OK | Obstacle: 2.50m @ 45° | Yaw: 15° | Moving: False'
```

### Exercise 3.3: View suggested commands

```bash
ros2 topic echo /fusion/suggested_cmd
```

## Part 4: Use Launch File

### Exercise 4.1: Run everything with launch

```bash
ros2 launch sensor_fusion sensor_fusion.launch.py
```

This starts both the simulator and fusion node together.

## Part 5: Understanding the Code

### Exercise 5.1: Study the sensor simulator

Open `sensor_fusion/sensor_simulator.py` and answer:

1. How does the LIDAR simulate walls?
2. What determines the IMU orientation?
3. How is noise added to the sensor data?

### Exercise 5.2: Study the fusion node

Open `sensor_fusion/fusion_node.py` and answer:

1. How does `_check_sensor_health` work?
2. What triggers the obstacle avoidance behavior?
3. How is yaw extracted from the quaternion?

## Part 6: Challenges

### Challenge 1: Add sensor timeout warning

Modify `fusion_callback` to log a warning when a sensor hasn't sent data for more than 2 seconds.

### Challenge 2: Improve obstacle avoidance

Modify the obstacle avoidance logic to:
- Consider obstacles in a wider angle range
- Adjust turn speed based on obstacle proximity

### Challenge 3: Add a fourth sensor

Create a `gps_node.py` that publishes `sensor_msgs/NavSatFix` and integrate it into the fusion node.

### Challenge 4: Synchronized callbacks

Use `message_filters` to create a callback that only fires when all three sensors have new data:

```python
from message_filters import ApproximateTimeSynchronizer, Subscriber

# In __init__:
camera_sub = Subscriber(self, Image, '/camera/image_raw')
lidar_sub = Subscriber(self, LaserScan, '/scan')
imu_sub = Subscriber(self, Imu, '/imu/data')

sync = ApproximateTimeSynchronizer(
    [camera_sub, lidar_sub, imu_sub],
    queue_size=10,
    slop=0.1
)
sync.registerCallback(self.synchronized_callback)
```

## Verification Checklist

- [ ] Built the sensor_fusion package
- [ ] Ran the sensor simulator and saw data on topics
- [ ] Ran the fusion node and saw status output
- [ ] Used the launch file to start everything
- [ ] Understood how each sensor type is processed
- [ ] Completed at least one challenge

## Common Issues

### "Message type not found" error
Make sure you have sensor_msgs installed:
```bash
sudo apt install ros-humble-sensor-msgs
```

### Fusion shows "CAM:--" even though simulator is running
Check that topics match: `/camera/image_raw` not `/camera/rgb/image_raw`

### High CPU usage
Reduce the sensor rates in the simulator parameters.

## Next Steps

Proceed to Lab 3: ROS 2 Architecture, where you'll learn about lifecycle nodes, services, and actions.

## Resources

- [sensor_msgs documentation](https://docs.ros.org/en/humble/p/sensor_msgs/)
- [Message Filters](https://docs.ros.org/en/humble/Tutorials/Intermediate/ROS2-Message-Filter.html)
- [QoS Settings](https://docs.ros.org/en/humble/Concepts/About-Quality-of-Service-Settings.html)
