#!/usr/bin/env python3
"""
Sensor Simulator Node

Simulates camera, LIDAR, and IMU sensors for testing the fusion pipeline
without real hardware. Publishes synthetic data that mimics real sensor output.

This is useful for:
- Testing fusion algorithms
- Development without hardware
- Understanding sensor data formats
"""

import math
import random

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu
from geometry_msgs.msg import Vector3
from std_msgs.msg import Header
import numpy as np


class SensorSimulator(Node):
    """Simulates multiple sensors for testing."""

    def __init__(self):
        super().__init__('sensor_simulator')

        # Declare parameters
        self.declare_parameter('camera_fps', 30.0)
        self.declare_parameter('lidar_hz', 10.0)
        self.declare_parameter('imu_hz', 100.0)

        # Get parameters
        camera_fps = self.get_parameter('camera_fps').value
        lidar_hz = self.get_parameter('lidar_hz').value
        imu_hz = self.get_parameter('imu_hz').value

        # Publishers
        self.camera_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.lidar_pub = self.create_publisher(LaserScan, '/scan', 10)
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)

        # Timers
        self.camera_timer = self.create_timer(1.0 / camera_fps, self.publish_camera)
        self.lidar_timer = self.create_timer(1.0 / lidar_hz, self.publish_lidar)
        self.imu_timer = self.create_timer(1.0 / imu_hz, self.publish_imu)

        # State for simulation
        self.time_offset = 0.0
        self.robot_angle = 0.0

        self.get_logger().info('Sensor Simulator started')
        self.get_logger().info(f'  Camera: {camera_fps} FPS')
        self.get_logger().info(f'  LIDAR: {lidar_hz} Hz')
        self.get_logger().info(f'  IMU: {imu_hz} Hz')

    def publish_camera(self):
        """Publish simulated camera image."""
        msg = Image()
        msg.header = self._make_header('camera_link')
        msg.height = 480
        msg.width = 640
        msg.encoding = 'rgb8'
        msg.is_bigendian = False
        msg.step = msg.width * 3

        # Create a simple gradient image with some noise
        # In real code, you'd use cv_bridge to convert OpenCV images
        data = []
        for y in range(msg.height):
            for x in range(msg.width):
                # Simple gradient with time-varying component
                r = int((x / msg.width) * 255)
                g = int((y / msg.height) * 255)
                b = int((math.sin(self.time_offset) + 1) * 127)
                # Add noise
                noise = random.randint(-10, 10)
                r = max(0, min(255, r + noise))
                g = max(0, min(255, g + noise))
                data.extend([r, g, b])

        msg.data = bytes(data[:100])  # Truncated for efficiency in simulation
        self.camera_pub.publish(msg)
        self.time_offset += 0.1

    def publish_lidar(self):
        """Publish simulated LIDAR scan."""
        msg = LaserScan()
        msg.header = self._make_header('laser_link')

        # LIDAR parameters (typical 2D LIDAR)
        msg.angle_min = -math.pi
        msg.angle_max = math.pi
        msg.angle_increment = math.pi / 180  # 1 degree resolution
        msg.time_increment = 0.0
        msg.scan_time = 0.1
        msg.range_min = 0.1
        msg.range_max = 10.0

        # Generate ranges - simulate a room with walls
        num_readings = int((msg.angle_max - msg.angle_min) / msg.angle_increment)
        ranges = []

        for i in range(num_readings):
            angle = msg.angle_min + i * msg.angle_increment + self.robot_angle

            # Simulate rectangular room (5m x 5m)
            # Calculate distance to walls
            if abs(math.cos(angle)) > 0.01:
                dist_x = abs(2.5 / math.cos(angle))
            else:
                dist_x = float('inf')

            if abs(math.sin(angle)) > 0.01:
                dist_y = abs(2.5 / math.sin(angle))
            else:
                dist_y = float('inf')

            distance = min(dist_x, dist_y, msg.range_max)

            # Add noise
            distance += random.gauss(0, 0.02)
            distance = max(msg.range_min, min(msg.range_max, distance))

            ranges.append(distance)

        msg.ranges = ranges
        msg.intensities = [100.0] * len(ranges)

        self.lidar_pub.publish(msg)

    def publish_imu(self):
        """Publish simulated IMU data."""
        msg = Imu()
        msg.header = self._make_header('imu_link')

        # Simulate slight movement
        t = self.time_offset

        # Angular velocity (simulating slight rotation)
        msg.angular_velocity.x = 0.0
        msg.angular_velocity.y = 0.0
        msg.angular_velocity.z = 0.1 * math.sin(t * 0.5)  # Gentle yaw rotation

        # Linear acceleration (gravity + small movements)
        msg.linear_acceleration.x = random.gauss(0, 0.1)
        msg.linear_acceleration.y = random.gauss(0, 0.1)
        msg.linear_acceleration.z = 9.81 + random.gauss(0, 0.1)  # Gravity

        # Orientation (quaternion) - simplified, just rotating around z
        self.robot_angle += msg.angular_velocity.z * 0.01
        msg.orientation.x = 0.0
        msg.orientation.y = 0.0
        msg.orientation.z = math.sin(self.robot_angle / 2)
        msg.orientation.w = math.cos(self.robot_angle / 2)

        # Covariance matrices (diagonal, typical values)
        msg.orientation_covariance = [
            0.01, 0.0, 0.0,
            0.0, 0.01, 0.0,
            0.0, 0.0, 0.01
        ]
        msg.angular_velocity_covariance = [
            0.001, 0.0, 0.0,
            0.0, 0.001, 0.0,
            0.0, 0.0, 0.001
        ]
        msg.linear_acceleration_covariance = [
            0.01, 0.0, 0.0,
            0.0, 0.01, 0.0,
            0.0, 0.0, 0.01
        ]

        self.imu_pub.publish(msg)

    def _make_header(self, frame_id: str) -> Header:
        """Create a header with current timestamp."""
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id
        return header


def main(args=None):
    rclpy.init(args=args)
    node = SensorSimulator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
