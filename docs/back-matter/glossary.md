---
sidebar_position: 6
title: "Glossary"
description: Definitions of key terms used throughout the textbook
---

# Glossary

Definitions of key terms used throughout this textbook, organized alphabetically.

## A

**Action (ROS 2):** A communication mechanism for long-running tasks with feedback, combining request-response and publish-subscribe patterns.

**Actuator:** A mechanical device that converts energy into motion, such as motors, servos, and pneumatic cylinders.

**AMCL:** Adaptive Monte Carlo Localization - a probabilistic localization algorithm using particle filters.

## B

**Behavior Tree:** A hierarchical model for task execution that organizes actions into trees of selectors, sequences, and conditions.

**BNO055:** A 9-axis absolute orientation sensor combining accelerometer, gyroscope, and magnetometer.

## C

**CAN Bus:** Controller Area Network - a robust serial communication protocol commonly used in robotics and automotive systems.

**CUDA:** Compute Unified Device Architecture - NVIDIA's parallel computing platform for GPU acceleration.

**Colcon:** The build tool for ROS 2 workspaces, replacing catkin from ROS 1.

## D

**DDS:** Data Distribution Service - the middleware standard underlying ROS 2 communication, providing real-time publish-subscribe messaging.

**Depth Camera:** A sensor that captures distance information per pixel, enabling 3D perception (e.g., Intel RealSense, OAK-D).

**Digital Twin:** A virtual replica of a physical system that mirrors its state and behavior in real-time.

**Domain Randomization:** A sim-to-real technique that varies simulation parameters to improve policy robustness.

**DOF:** Degrees of Freedom - the number of independent parameters defining a system's configuration.

## E

**EKF:** Extended Kalman Filter - a nonlinear state estimation algorithm used for sensor fusion and localization.

**End Effector:** The device at the end of a robotic arm that interacts with the environment (e.g., gripper, tool).

**Executor (ROS 2):** The component that manages callback execution for nodes, handling subscriptions and timers.

## F

**FK:** Forward Kinematics - computing end effector position given joint angles.

**FoV:** Field of View - the angular extent visible to a sensor or camera.

## G

**Gazebo:** An open-source robotics simulator providing physics, rendering, and sensor simulation.

**GPU:** Graphics Processing Unit - hardware accelerator used for parallel computing in perception and learning.

**Grasp Planning:** The process of determining hand/gripper configurations to securely hold an object.

## H

**Humanoid Robot:** A robot with a body structure resembling the human form, typically bipedal with arms and a head.

**Hz:** Hertz - unit of frequency, commonly used to describe sensor update rates and control loop frequencies.

## I

**IK:** Inverse Kinematics - computing joint angles required to achieve a desired end effector position.

**IMU:** Inertial Measurement Unit - a sensor combining accelerometer, gyroscope, and optionally magnetometer for motion tracking.

**Isaac Sim:** NVIDIA's robotics simulation platform built on Omniverse, supporting physics simulation and synthetic data generation.

**Isaac ROS:** NVIDIA's hardware-accelerated ROS 2 packages for perception and navigation.

## J

**Jacobian:** A matrix relating joint velocities to end effector velocities, fundamental to robot control.

**Joint:** A connection between two links allowing relative motion (revolute, prismatic, fixed, etc.).

## K

**Kinematic Chain:** A series of links connected by joints, forming the structure of a robot arm or leg.

## L

**LIDAR:** Light Detection and Ranging - a sensor using laser pulses to measure distances, producing point cloud data.

**Lifecycle Node:** A ROS 2 node with managed states (unconfigured, inactive, active, finalized) for controlled startup/shutdown.

**Link:** A rigid body segment of a robot, connected to other links via joints.

## M

**MPC:** Model Predictive Control - an optimization-based control strategy that predicts future states.

**Manipulation:** The robotic task of grasping, moving, and placing objects.

## N

**Nav2:** Navigation2 - the ROS 2 navigation stack providing path planning, localization, and obstacle avoidance.

**Node:** The fundamental processing unit in ROS 2 that performs computation and communicates via topics, services, and actions.

## O

**Odometry:** Estimation of position change based on sensor data (wheel encoders, IMU, visual).

**Omniverse:** NVIDIA's platform for 3D design collaboration and simulation, underlying Isaac Sim.

## P

**PID:** Proportional-Integral-Derivative - a feedback control algorithm widely used in robotics.

**Point Cloud:** A set of 3D points representing spatial data, typically from LIDAR or depth cameras.

**Pose:** Position and orientation of an object in space, typically represented as (x, y, z, roll, pitch, yaw) or a transformation matrix.

**Publisher:** A ROS 2 entity that sends messages to a topic.

## Q

**QoS:** Quality of Service - ROS 2 settings controlling message delivery reliability, durability, and history.

**Quaternion:** A four-element representation of 3D rotation, avoiding gimbal lock issues of Euler angles.

## R

**ROS 2:** Robot Operating System 2 - the second generation of the open-source robotics middleware framework.

**RVIZ:** ROS Visualization - a 3D visualization tool for displaying sensor data, robot models, and navigation information.

**RL:** Reinforcement Learning - a machine learning paradigm where agents learn through trial and error with rewards.

## S

**SLAM:** Simultaneous Localization and Mapping - the problem of building a map while tracking position within it.

**Service:** A synchronous request-response communication pattern in ROS 2.

**Sim-to-Real (Sim2Real):** The transfer of policies or models trained in simulation to real-world robots.

**Subscriber:** A ROS 2 entity that receives messages from a topic.

## T

**TF2:** The ROS 2 transform library for tracking coordinate frame relationships over time.

**Topic:** A named bus for publish-subscribe communication in ROS 2.

**Torque:** Rotational force applied by motors, measured in Newton-meters (Nm).

## U

**URDF:** Unified Robot Description Format - an XML format describing robot kinematics, dynamics, and visual properties.

**USD:** Universal Scene Description - a file format for 3D scenes used by NVIDIA Omniverse and Isaac Sim.

## V

**VLA:** Vision-Language-Action - a class of models that combine visual perception, natural language understanding, and robotic action generation.

**VSLAM:** Visual SLAM - SLAM using camera images rather than LIDAR.

## W

**Workspace:** In ROS 2, a directory containing source packages, build artifacts, and installation files managed by colcon.

## X

**Xacro:** XML Macros - a templating language for creating parameterized URDF files.

## Z

**ZMP:** Zero Moment Point - the point on the ground where the sum of horizontal inertia and gravity forces equals zero, critical for bipedal balance.

---

## Quick Reference Tables

### Common ROS 2 Message Types

| Type | Description |
|------|-------------|
| `std_msgs/String` | Simple string message |
| `geometry_msgs/Twist` | Linear and angular velocity |
| `sensor_msgs/Image` | Camera image data |
| `sensor_msgs/LaserScan` | 2D LIDAR scan |
| `nav_msgs/Odometry` | Position and velocity estimate |

### Common Units

| Quantity | SI Unit | Symbol |
|----------|---------|--------|
| Distance | meter | m |
| Angle | radian | rad |
| Time | second | s |
| Frequency | hertz | Hz |
| Force | newton | N |
| Torque | newton-meter | Nm |
