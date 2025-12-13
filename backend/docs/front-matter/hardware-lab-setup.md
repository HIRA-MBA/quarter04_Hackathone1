---
sidebar_position: 4
title: Hardware & Lab Setup
description: Required hardware and software setup for hands-on exercises
---

# Hardware & Lab Setup

## Hardware Requirements

### Development Machine
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 8 cores | 12+ cores |
| RAM | 16 GB | 32 GB |
| GPU | RTX 3060 | RTX 4080+ |
| Storage | 256 GB SSD | 1 TB NVMe |
| OS | Ubuntu 22.04 | Ubuntu 22.04 |

### Optional Robot Kit
- Jetson Orin Nano (edge deployment)
- Intel RealSense D435i (depth camera)
- RPLidar A1 (2D LIDAR)
- Servo motors for manipulation

## Software Setup

### ROS 2 Humble
```bash
sudo apt update
sudo apt install -y curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | \
  sudo tee /etc/apt/sources.list.d/ros2.list
sudo apt update
sudo apt install -y ros-humble-desktop ros-humble-gazebo-ros-pkgs
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

### Python Dependencies
```bash
pip install numpy scipy matplotlib opencv-python torch transformers
```

### NVIDIA Isaac (Module 3)
1. Install Omniverse Launcher
2. Download Isaac Sim 2023.1.1+
3. Install Isaac ROS packages

### Docker
```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
docker pull osrf/ros:humble-desktop
```

## Workspace Setup
```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws
colcon build
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```

## Verification
```bash
ros2 run demo_nodes_cpp talker &
ros2 run demo_nodes_cpp listener
gazebo --verbose
nvidia-smi
```

## Cloud Alternative
No local GPU? See [Appendix B: Cloud Setup](../back-matter/appendix-b-cloud-setup).

---

*Proceed to [Chapter 1](../module-1-ros2/ch01-welcome-first-node).*
