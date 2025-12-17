---
sidebar_position: 4
title: "Appendix D: Troubleshooting"
description: Common issues and their solutions
---

# Appendix D: Troubleshooting

This appendix addresses common issues you may encounter while working through the labs.

## ROS 2 Communication Problems

### Nodes Can't Discover Each Other

**Symptom:** `ros2 topic list` shows topics but `ros2 topic echo` receives no messages.

**Solutions:**

1. **Check ROS_DOMAIN_ID:**
```bash
# Ensure both machines use same domain
echo $ROS_DOMAIN_ID
export ROS_DOMAIN_ID=42
```

2. **Firewall blocking DDS:**
```bash
# Ubuntu - allow DDS ports
sudo ufw allow 7400:7500/udp
sudo ufw allow 7400:7500/tcp
```

3. **Multicast disabled:**
```bash
# Check multicast
cat /proc/sys/net/ipv4/icmp_echo_ignore_broadcasts
# Should be 0, if 1:
sudo sysctl -w net.ipv4.icmp_echo_ignore_broadcasts=0
```

### "No executable found" Error

**Symptom:** `ros2 run package_name node_name` fails.

**Solutions:**

```bash
# 1. Source workspace
source ~/ros2_ws/install/setup.bash

# 2. Rebuild package
cd ~/ros2_ws
colcon build --packages-select package_name

# 3. Check entry point in setup.py
# entry_points should list: 'node_name = package.module:main'
```

### Topic Type Mismatch

**Symptom:** Subscriber receives no messages despite publisher running.

**Solution:**
```bash
# Check topic types
ros2 topic info /topic_name -v

# Verify publisher and subscriber use same message type
# std_msgs/msg/String != std_msgs/String (common typo)
```

## Gazebo Issues

### Gazebo Won't Start

**Symptom:** Black screen or immediate crash.

**Solutions:**

1. **Check OpenGL:**
```bash
glxinfo | grep "OpenGL version"
# Need OpenGL 3.3+
```

2. **Software rendering fallback:**
```bash
export LIBGL_ALWAYS_SOFTWARE=1
gazebo
```

3. **Reset Gazebo config:**
```bash
rm -rf ~/.gazebo
```

### Models Not Loading

**Symptom:** Empty world or missing robot.

**Solutions:**

```bash
# 1. Check model paths
echo $GAZEBO_MODEL_PATH

# 2. Add custom model path
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/my_models

# 3. Download missing models
cd ~/.gazebo/models
wget -r -np -nH --cut-dirs=4 http://models.gazebosim.org/
```

### Simulation Running Slow

**Solutions:**

1. **Reduce physics update rate:**
```xml
<!-- In world file -->
<physics type="ode">
  <real_time_update_rate>500</real_time_update_rate>
</physics>
```

2. **Disable shadows:**
```xml
<scene>
  <shadows>false</shadows>
</scene>
```

3. **Simplify collision meshes**

## Isaac Sim Issues

### Isaac Sim Won't Launch

**Symptom:** Crash on startup or NVIDIA driver error.

**Requirements Check:**
```bash
# Check NVIDIA driver (need 525+)
nvidia-smi

# Check CUDA
nvcc --version

# Check GPU memory (need 8GB+)
nvidia-smi --query-gpu=memory.total --format=csv
```

### "CUDA out of memory"

**Solutions:**

```bash
# 1. Close other GPU applications
nvidia-smi  # Check what's using GPU

# 2. Reduce simulation complexity
# - Lower resolution
# - Fewer physics substeps
# - Smaller batch sizes for RL

# 3. Use mixed precision
```

### Isaac ROS Bridge Not Working

**Solutions:**

```bash
# 1. Check ROS 2 is sourced
source /opt/ros/humble/setup.bash

# 2. Launch bridge explicitly
ros2 launch isaac_ros_bridge bridge.launch.py

# 3. Verify topics
ros2 topic list | grep isaac
```

## GPU and CUDA Issues

### CUDA Version Mismatch

**Symptom:** `CUDA driver version is insufficient`

**Solution:**
```bash
# Check installed CUDA
ls /usr/local/ | grep cuda

# Check required version
python3 -c "import torch; print(torch.version.cuda)"

# Install matching version
sudo apt install cuda-toolkit-12-2
```

### GPU Not Detected

**Solutions:**

```bash
# 1. Check driver
nvidia-smi

# 2. Reinstall driver
sudo apt purge nvidia-*
sudo apt install nvidia-driver-535

# 3. For Docker
docker run --gpus all nvidia/cuda:12.2-base nvidia-smi
```

### cuDNN Errors

**Solution:**
```bash
# Install matching cuDNN
sudo apt install libcudnn8 libcudnn8-dev

# Verify
cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

## Network Configuration

### ROS 2 Multi-Machine Setup

```bash
# On all machines, set same domain
export ROS_DOMAIN_ID=42

# If using different subnets, configure DDS
# Create ~/cyclonedds.xml:
```

```xml
<?xml version="1.0" encoding="UTF-8"?>
<CycloneDDS>
  <Domain>
    <General>
      <NetworkInterfaceAddress>192.168.1.0/24</NetworkInterfaceAddress>
    </General>
  </Domain>
</CycloneDDS>
```

```bash
export CYCLONEDDS_URI=file://$HOME/cyclonedds.xml
```

### Docker Networking

```bash
# Use host network for ROS 2
docker run --network=host ros:humble

# Or create bridge network
docker network create ros-net
docker run --network=ros-net ros:humble
```

## Permission Errors

### "Permission denied" for Serial Ports

```bash
# Add user to dialout group
sudo usermod -a -G dialout $USER

# Or set permissions
sudo chmod 666 /dev/ttyUSB0

# Reboot or re-login for group changes
```

### Docker Permission Denied

```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Restart Docker
sudo systemctl restart docker

# Re-login
```

### Cannot Write to Workspace

```bash
# Fix ownership
sudo chown -R $USER:$USER ~/ros2_ws

# Fix permissions
chmod -R u+rwX ~/ros2_ws
```

## Dependency Errors

### Missing ROS 2 Package

```bash
# Update rosdep
rosdep update

# Install dependencies
rosdep install --from-paths src --ignore-src -r -y

# If specific package missing
sudo apt install ros-humble-<package-name>
```

### Python Import Error

```bash
# Install in ROS 2 environment
pip3 install package_name

# Or use rosdep
rosdep install python3-package-name
```

### Colcon Build Fails

```bash
# Clean build
rm -rf build/ install/ log/

# Build with verbose output
colcon build --event-handlers console_direct+

# Build specific package
colcon build --packages-select package_name
```

## Quick Diagnostic Commands

```bash
# System info
uname -a
lsb_release -a

# ROS 2 info
ros2 doctor --report

# GPU info
nvidia-smi
nvcc --version

# Network info
ip addr
ros2 multicast receive

# Process info
htop
nvidia-smi -l 1
```

## Getting Help

1. **Check ROS Answers:** https://answers.ros.org
2. **GitHub Issues:** Search the relevant repository
3. **ROS Discourse:** https://discourse.ros.org
4. **Stack Overflow:** Tag with `ros2`

When asking for help, include:
- OS and version
- ROS 2 distribution
- Full error message
- Steps to reproduce
- Output of `ros2 doctor --report`
