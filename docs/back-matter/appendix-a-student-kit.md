---
sidebar_position: 1
title: "Appendix A: Student Hardware Kit"
description: Recommended hardware components for hands-on learning
---

# Appendix A: Student Hardware Kit

This appendix details the recommended hardware components for completing the hands-on labs in this textbook.

## Recommended Computing Platforms

### Primary: NVIDIA Jetson Orin Nano

| Specification | Details |
|---------------|---------|
| GPU | 1024-core NVIDIA Ampere |
| CPU | 6-core Arm Cortex-A78AE |
| Memory | 8GB LPDDR5 |
| Storage | microSD (64GB+ recommended) |
| Power | 7-15W |

**Why Jetson?** Native support for Isaac ROS, CUDA acceleration, and optimal for edge AI deployment covered in Modules 3-4.

### Alternative: Raspberry Pi 5

| Specification | Details |
|---------------|---------|
| CPU | Quad-core Arm Cortex-A76 @ 2.4GHz |
| Memory | 8GB LPDDR4X |
| Storage | microSD (32GB+ recommended) |
| Power | 5V/5A USB-C |

**Use case:** Suitable for Modules 1-2 (ROS 2 fundamentals, basic simulation). Limited for GPU-accelerated perception.

## Sensors

### Camera Options

| Sensor | Resolution | Interface | Use Case |
|--------|------------|-----------|----------|
| Intel RealSense D435i | 1280x720 depth | USB 3.0 | Depth perception, SLAM |
| Raspberry Pi Camera v3 | 12MP | CSI | Basic vision, low cost |
| OAK-D Lite | 4K + stereo depth | USB 3.0 | On-device AI inference |

### LIDAR Options

| Sensor | Range | Points/sec | Use Case |
|--------|-------|------------|----------|
| RPLiDAR A1 | 12m | 8K | Indoor navigation |
| YDLIDAR X4 | 10m | 5K | Budget-friendly |
| Livox Mid-360 | 40m | 200K | Outdoor, high precision |

### IMU (Inertial Measurement Unit)

| Sensor | DOF | Interface |
|--------|-----|-----------|
| BNO055 | 9-axis | I2C |
| MPU6050 | 6-axis | I2C |
| ICM-20948 | 9-axis | SPI/I2C |

## Actuators and Motor Controllers

### Servo Motors

| Type | Torque | Use Case |
|------|--------|----------|
| Dynamixel XL330 | 0.6 Nm | Robot arm joints |
| MG996R | 0.1 Nm | Hobby projects |
| Dynamixel XM430 | 1.4 Nm | Humanoid joints |

### Motor Controllers

- **ODrive v3.6**: Brushless DC motor control, ideal for wheeled robots
- **Roboclaw 2x15A**: Dual brushed DC motor controller
- **Arduino Motor Shield**: Simple DC/stepper control

## Power Systems

### Battery Options

| Type | Voltage | Capacity | Runtime |
|------|---------|----------|---------|
| LiPo 3S | 11.1V | 5000mAh | ~2 hours |
| LiPo 4S | 14.8V | 6000mAh | ~3 hours |
| Li-ion 18650 pack | 12V | 10000mAh | ~4 hours |

### Power Distribution

- Voltage regulators (5V, 12V outputs)
- Power monitoring (INA219 current sensor)
- Emergency stop switch (required for safety)

## Budget Tiers

### Tier 1: Essentials (~$200)

- Raspberry Pi 5 (8GB)
- Pi Camera v3
- BNO055 IMU
- Basic servo kit
- Power supply

### Tier 2: Intermediate (~$600)

- Jetson Orin Nano
- Intel RealSense D435i
- RPLiDAR A1
- Dynamixel XL330 servos
- LiPo battery system

### Tier 3: Advanced (~$1500+)

- Jetson AGX Orin
- OAK-D Pro
- Livox Mid-360
- Dynamixel XM430 servos
- Full humanoid frame kit

## Assembly Guides

### Basic Wheeled Robot

1. Mount computing platform to chassis
2. Connect motor controller to GPIO/USB
3. Wire motors with proper polarity
4. Mount camera at 30° downward angle
5. Install LIDAR at highest point
6. Connect power with emergency stop inline

### Manipulator Arm

1. Assemble servo chain base-to-gripper
2. Wire Dynamixel bus (daisy chain)
3. Mount camera for eye-in-hand configuration
4. Calibrate joint zero positions
5. Test range of motion before loading

## Sourcing Components

| Vendor | Region | Specialty |
|--------|--------|-----------|
| NVIDIA | Global | Jetson platforms |
| Robotis | Global | Dynamixel servos |
| SparkFun | US | Sensors, breakouts |
| Pimoroni | UK/EU | Pi accessories |
| Waveshare | Global | Displays, HATs |
| Amazon/AliExpress | Global | Generic components |

## Safety Requirements

- LiPo-safe charging bag
- Fire extinguisher nearby
- Safety glasses during assembly
- Anti-static wrist strap for electronics
- First aid kit

See [Safety & Ethics](./safety-ethics.md) for complete safety guidelines.
