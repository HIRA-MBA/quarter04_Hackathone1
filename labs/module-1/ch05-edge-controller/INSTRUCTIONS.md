# Lab 5: Edge Deployment Capstone

## Objectives

By the end of this lab, you will:
1. Understand edge computing constraints for robotics
2. Implement resource-efficient ROS 2 nodes
3. Monitor and adapt to system resource limitations
4. Deploy ROS 2 on embedded platforms

## Prerequisites

- Completed Labs 1-4
- Understanding of ROS 2 architecture
- Optional: Access to Raspberry Pi, Jetson, or similar

## Edge Computing Challenges

### Resource Constraints
| Constraint | Desktop | Raspberry Pi 4 | Jetson Nano |
|------------|---------|----------------|-------------|
| CPU | 8+ cores | 4 cores | 4 cores |
| RAM | 16+ GB | 4-8 GB | 4 GB |
| Power | Unlimited | 15W | 10-20W |

### Optimization Strategies
1. **Memory**: Reuse objects, avoid allocations in loops
2. **CPU**: Adaptive rates, efficient algorithms
3. **Power**: Sleep when idle, reduce sensor rates
4. **Network**: Compress data, batch messages

## Part 1: Edge-Optimized Node

### Exercise 1.1: Build and run

```bash
cd ~/ros2_ws/src
cp -r /workspace/module-1/ch05-edge-controller .
cd ~/ros2_ws
colcon build --packages-select edge_controller
source install/setup.bash
```

### Exercise 1.2: Run edge node

```bash
ros2 run edge_controller edge_node
```

### Exercise 1.3: Monitor resource topics

```bash
ros2 topic echo /edge/cpu_usage
ros2 topic echo /edge/memory_usage
ros2 topic echo /edge/status
```

### Exercise 1.4: Stress test

In another terminal, create CPU load:
```bash
# Linux
stress --cpu 4 --timeout 30

# Or Python
python3 -c "import time; [sum(range(10000000)) for _ in range(100)]"
```

Watch the edge node adapt its rate.

## Part 2: Optimized Controller

### Exercise 2.1: Run controller

```bash
ros2 run edge_controller optimized_controller
```

### Exercise 2.2: Change modes

```bash
# Set to autonomous
ros2 topic pub /controller/mode std_msgs/msg/String "data: 'auto'" --once

# Check status
ros2 topic echo /controller/status

# Emergency stop
ros2 topic pub /controller/mode std_msgs/msg/String "data: 'stop'" --once
```

### Exercise 2.3: Simulate LIDAR

```bash
# In another terminal, run sensor simulator from Lab 2
ros2 run sensor_fusion sensor_simulator
```

Watch the controller react to obstacles.

## Part 3: Resource Monitor

### Exercise 3.1: Run monitor

```bash
ros2 run edge_controller resource_monitor
```

### Exercise 3.2: View diagnostics

```bash
ros2 topic echo /edge/diagnostics
```

### Exercise 3.3: Set warning thresholds

```bash
ros2 run edge_controller resource_monitor --ros-args \
  -p warn_cpu_threshold:=50.0 \
  -p warn_memory_threshold:=60.0
```

## Part 4: Code Analysis

### Exercise 4.1: Memory optimization patterns

Open `edge_node.py` and identify:

1. **Preallocated messages**:
```python
self._cmd_msg = Twist()  # Created once, reused
```

2. **Object reuse in callbacks**:
```python
def main_loop(self):
    self._status_msg.data = f'...'  # Reuse existing object
```

### Exercise 4.2: QoS for edge

Find the edge-optimized QoS settings:
```python
edge_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,  # Don't retry
    durability=DurabilityPolicy.VOLATILE,       # No persistence
    history=HistoryPolicy.KEEP_LAST,
    depth=1  # Minimal queue
)
```

### Exercise 4.3: Adaptive rate control

Study the rate adaptation logic in `monitor_resources()`:
- When does rate decrease?
- When does rate increase?
- What are the minimum and maximum rates?

## Part 5: Deployment Simulation

### Exercise 5.1: Resource-limited Docker

Run with limited resources:
```bash
docker run -it --rm \
  --cpus="1.0" \
  --memory="512m" \
  -v $(pwd):/workspace \
  osrf/ros:humble-desktop \
  bash
```

Inside container:
```bash
cd /workspace
colcon build --packages-select edge_controller
source install/setup.bash
ros2 run edge_controller edge_node
```

### Exercise 5.2: Measure baseline

```bash
# Before running nodes
free -h
top -bn1 | head -5

# After running nodes
ros2 run edge_controller resource_monitor &
ros2 run edge_controller edge_node &
ros2 run edge_controller optimized_controller &

# Check resource usage
top
```

## Capstone Challenge

### Build a Complete Edge System

Create a launch file that runs all three nodes together with appropriate parameters for a Raspberry Pi:

Requirements:
1. Resource monitor with tight thresholds
2. Edge node with low target rate (5 Hz)
3. Optimized controller
4. All using edge-optimized QoS

### Grading Rubric

| Criterion | Points |
|-----------|--------|
| All three nodes run concurrently | 20 |
| Total CPU usage under 50% | 20 |
| Memory usage under 200MB | 20 |
| Adaptive rate responds to load | 20 |
| Clean shutdown on Ctrl+C | 10 |
| Documentation of approach | 10 |

## Verification Checklist

- [ ] Built and ran edge_node
- [ ] Observed adaptive rate changes under load
- [ ] Ran optimized_controller with mode changes
- [ ] Monitored resources with resource_monitor
- [ ] Identified memory optimization patterns in code
- [ ] Tested with resource-limited Docker (optional)

## Common Issues

### "psutil not found"
```bash
pip install psutil
```

### High CPU even when idle
- Check timer rates (lower them)
- Ensure best-effort QoS
- Profile with `py-spy` or similar

### Messages dropping
- Expected with BEST_EFFORT QoS
- Reduce publisher rates if needed

## Best Practices Summary

1. **Preallocate**: Create messages once, reuse always
2. **Minimal QoS**: BEST_EFFORT, depth=1 for edge
3. **Adaptive rates**: Monitor resources, adjust accordingly
4. **State machines**: Efficient mode switching
5. **Batch processing**: Don't process every message

## Next Steps

Proceed to Module 2: Digital Twin, where you'll learn simulation with Gazebo.

## Resources

- [ROS 2 QoS Tuning](https://docs.ros.org/en/humble/Concepts/About-Quality-of-Service-Settings.html)
- [Raspberry Pi ROS 2](https://docs.ros.org/en/humble/How-To-Guides/Installing-on-Raspberry-Pi.html)
- [NVIDIA Jetson ROS 2](https://nvidia-isaac-ros.github.io/)
- [psutil Documentation](https://psutil.readthedocs.io/)
