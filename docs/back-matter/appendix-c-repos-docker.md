---
sidebar_position: 3
title: "Appendix C: Repositories & Docker"
description: Code repositories and containerization guides
---

# Appendix C: Repositories & Docker

This appendix provides reference links to official repositories and Docker containerization guides for robotics development.

## Official ROS 2 Repositories

### Core ROS 2

| Repository | Description |
|------------|-------------|
| [ros2/ros2](https://github.com/ros2/ros2) | Main ROS 2 repository |
| [ros2/rclpy](https://github.com/ros2/rclpy) | Python client library |
| [ros2/rclcpp](https://github.com/ros2/rclcpp) | C++ client library |
| [ros2/examples](https://github.com/ros2/examples) | Example packages |
| [ros2/demos](https://github.com/ros2/demos) | Demo applications |

### Navigation & SLAM

| Repository | Description |
|------------|-------------|
| [ros-navigation/navigation2](https://github.com/ros-navigation/navigation2) | Nav2 stack |
| [SteveMacenski/slam_toolbox](https://github.com/SteveMacenski/slam_toolbox) | SLAM Toolbox |
| [ros-perception/vision_opencv](https://github.com/ros-perception/vision_opencv) | OpenCV bridge |

### Simulation

| Repository | Description |
|------------|-------------|
| [gazebosim/gz-sim](https://github.com/gazebosim/gz-sim) | Gazebo Sim |
| [Unity-Technologies/Unity-Robotics-Hub](https://github.com/Unity-Technologies/Unity-Robotics-Hub) | Unity robotics |
| [NVIDIA-ISAAC-ROS](https://github.com/NVIDIA-ISAAC-ROS) | Isaac ROS packages |

## Docker Images

### Official ROS 2 Images

```bash
# ROS 2 Humble (recommended)
docker pull ros:humble

# ROS 2 with desktop tools (RViz, Gazebo)
docker pull ros:humble-desktop

# ROS 2 perception variant
docker pull ros:humble-perception
```

### NVIDIA Isaac Images

```bash
# Isaac ROS base
docker pull nvcr.io/nvidia/isaac/ros:humble-aarch64

# Isaac Sim
docker pull nvcr.io/nvidia/isaac-sim:2023.1.1

# Isaac Lab (RL)
docker pull nvcr.io/nvidia/isaac-lab:1.0
```

### Custom Development Image

```dockerfile
# Dockerfile.ros2-dev
FROM ros:humble

# Install development tools
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    ros-humble-rviz2 \
    ros-humble-gazebo-ros-pkgs \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install \
    numpy \
    opencv-python \
    torch \
    transformers

# Set up workspace
WORKDIR /workspace
ENV ROS_DOMAIN_ID=42

# Source ROS 2 on shell start
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

CMD ["/bin/bash"]
```

Build and run:

```bash
docker build -t ros2-dev -f Dockerfile.ros2-dev .
docker run -it --rm ros2-dev
```

## Docker Compose Configurations

### Basic ROS 2 Development

```yaml
# docker-compose.yml
version: '3.8'

services:
  ros2-dev:
    image: ros:humble
    container_name: ros2-dev
    network_mode: host
    environment:
      - ROS_DOMAIN_ID=42
      - DISPLAY=${DISPLAY}
    volumes:
      - ./workspace:/workspace
      - /tmp/.X11-unix:/tmp/.X11-unix
    working_dir: /workspace
    command: /bin/bash
    stdin_open: true
    tty: true
```

### Full Simulation Stack

```yaml
# docker-compose.sim.yml
version: '3.8'

services:
  gazebo:
    image: ros:humble-desktop
    container_name: gazebo-sim
    network_mode: host
    environment:
      - DISPLAY=${DISPLAY}
      - ROS_DOMAIN_ID=42
    volumes:
      - ./workspace:/workspace
      - /tmp/.X11-unix:/tmp/.X11-unix
    command: ros2 launch gazebo_ros gazebo.launch.py

  robot:
    image: ros:humble
    container_name: robot-controller
    network_mode: host
    environment:
      - ROS_DOMAIN_ID=42
    volumes:
      - ./workspace:/workspace
    depends_on:
      - gazebo
    command: ros2 launch my_robot robot.launch.py

  rviz:
    image: ros:humble-desktop
    container_name: rviz-viz
    network_mode: host
    environment:
      - DISPLAY=${DISPLAY}
      - ROS_DOMAIN_ID=42
    volumes:
      - ./workspace:/workspace
      - /tmp/.X11-unix:/tmp/.X11-unix
    depends_on:
      - robot
    command: ros2 run rviz2 rviz2
```

### GPU-Enabled Stack

```yaml
# docker-compose.gpu.yml
version: '3.8'

services:
  isaac-ros:
    image: nvcr.io/nvidia/isaac/ros:humble
    container_name: isaac-ros
    runtime: nvidia
    network_mode: host
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - ROS_DOMAIN_ID=42
    volumes:
      - ./workspace:/workspace
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Development Containers (VS Code)

### devcontainer.json

```json
{
  "name": "ROS 2 Humble",
  "image": "ros:humble",
  "runArgs": [
    "--network=host",
    "-e", "DISPLAY=${env:DISPLAY}",
    "-v", "/tmp/.X11-unix:/tmp/.X11-unix"
  ],
  "workspaceMount": "source=${localWorkspaceFolder},target=/workspace,type=bind",
  "workspaceFolder": "/workspace",
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-vscode.cpptools",
        "ms-iot.vscode-ros"
      ],
      "settings": {
        "python.defaultInterpreterPath": "/usr/bin/python3"
      }
    }
  },
  "postCreateCommand": "apt-get update && apt-get install -y python3-pip"
}
```

## CI/CD Pipeline Examples

### GitHub Actions for ROS 2

```yaml
# .github/workflows/ros2-ci.yml
name: ROS 2 CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build-and-test:
    runs-on: ubuntu-22.04
    container:
      image: ros:humble

    steps:
      - uses: actions/checkout@v4

      - name: Install dependencies
        run: |
          apt-get update
          rosdep update
          rosdep install --from-paths src --ignore-src -r -y

      - name: Build
        run: |
          source /opt/ros/humble/setup.bash
          colcon build --symlink-install

      - name: Test
        run: |
          source /opt/ros/humble/setup.bash
          source install/setup.bash
          colcon test
          colcon test-result --verbose
```

## Quick Reference Commands

### Docker Basics

```bash
# Build image
docker build -t my-robot .

# Run with GUI support
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  my-robot

# Run with GPU
docker run --gpus all -it my-robot

# Enter running container
docker exec -it <container-id> /bin/bash

# Clean up
docker system prune -a
```

### Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and start
docker-compose up --build
```

## Useful Links

- [ROS 2 Docker Hub](https://hub.docker.com/_/ros)
- [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/)
- [Docker Docs](https://docs.docker.com/)
- [VS Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)
