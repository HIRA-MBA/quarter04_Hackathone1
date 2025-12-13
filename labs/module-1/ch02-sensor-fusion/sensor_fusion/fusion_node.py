#!/usr/bin/env python3
"""
Sensor Fusion Node

Demonstrates basic sensor fusion concepts by combining data from
multiple sensors (camera, LIDAR, IMU) into a unified perception output.

Key concepts:
- Message synchronization
- Sensor data alignment
- Basic fusion algorithms
"""

import math
from dataclasses import dataclass
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, LaserScan, Imu
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import String
import message_filters


@dataclass
class SensorState:
    """Holds the latest data from each sensor."""
    camera_timestamp: Optional[float] = None
    lidar_timestamp: Optional[float] = None
    imu_timestamp: Optional[float] = None

    # Processed data
    nearest_obstacle_distance: float = float('inf')
    nearest_obstacle_angle: float = 0.0
    robot_orientation_yaw: float = 0.0
    is_moving: bool = False


class SensorFusionNode(Node):
    """
    Fuses data from camera, LIDAR, and IMU sensors.

    This node demonstrates:
    1. Subscribing to multiple sensor topics
    2. Processing different sensor data types
    3. Combining information for decision making
    """

    def __init__(self):
        super().__init__('sensor_fusion')

        # State
        self.state = SensorState()

        # QoS profile for sensor data (best effort for real-time)
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        # Individual subscribers (for asynchronous processing)
        self.camera_sub = self.create_subscription(
            Image, '/camera/image_raw', self.camera_callback, sensor_qos
        )
        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, sensor_qos
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, sensor_qos
        )

        # Publishers for fusion output
        self.status_pub = self.create_publisher(String, '/fusion/status', 10)
        self.cmd_pub = self.create_publisher(Twist, '/fusion/suggested_cmd', 10)

        # Timer for periodic fusion and status updates
        self.fusion_timer = self.create_timer(0.1, self.fusion_callback)

        self.get_logger().info('Sensor Fusion Node started')

    def camera_callback(self, msg: Image):
        """Process camera data."""
        self.state.camera_timestamp = self._get_timestamp(msg.header.stamp)

        # In a real application, you would:
        # 1. Convert to OpenCV format using cv_bridge
        # 2. Run object detection/segmentation
        # 3. Extract relevant features

        # For this lab, we just log that we received data
        self.get_logger().debug(
            f'Camera frame received: {msg.width}x{msg.height}'
        )

    def lidar_callback(self, msg: LaserScan):
        """Process LIDAR data to find nearest obstacle."""
        self.state.lidar_timestamp = self._get_timestamp(msg.header.stamp)

        # Find the nearest obstacle
        min_distance = float('inf')
        min_angle = 0.0

        for i, distance in enumerate(msg.ranges):
            if msg.range_min < distance < msg.range_max:
                if distance < min_distance:
                    min_distance = distance
                    min_angle = msg.angle_min + i * msg.angle_increment

        self.state.nearest_obstacle_distance = min_distance
        self.state.nearest_obstacle_angle = min_angle

        self.get_logger().debug(
            f'LIDAR: nearest obstacle at {min_distance:.2f}m, '
            f'angle {math.degrees(min_angle):.1f}°'
        )

    def imu_callback(self, msg: Imu):
        """Process IMU data for orientation and motion detection."""
        self.state.imu_timestamp = self._get_timestamp(msg.header.stamp)

        # Extract yaw from quaternion
        # Simplified: assumes small roll/pitch
        q = msg.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.state.robot_orientation_yaw = math.atan2(siny_cosp, cosy_cosp)

        # Detect if robot is moving (based on angular velocity)
        angular_speed = math.sqrt(
            msg.angular_velocity.x ** 2 +
            msg.angular_velocity.y ** 2 +
            msg.angular_velocity.z ** 2
        )
        self.state.is_moving = angular_speed > 0.05

        self.get_logger().debug(
            f'IMU: yaw={math.degrees(self.state.robot_orientation_yaw):.1f}°, '
            f'moving={self.state.is_moving}'
        )

    def fusion_callback(self):
        """
        Periodic fusion of all sensor data.

        This is where we combine information from all sensors
        to make decisions or generate outputs.
        """
        # Check sensor health (are we receiving data?)
        current_time = self.get_clock().now().nanoseconds / 1e9
        sensor_status = self._check_sensor_health(current_time)

        # Generate status message
        status_msg = String()
        status_msg.data = (
            f'Fusion Status | '
            f'Sensors: {sensor_status} | '
            f'Obstacle: {self.state.nearest_obstacle_distance:.2f}m @ '
            f'{math.degrees(self.state.nearest_obstacle_angle):.0f}° | '
            f'Yaw: {math.degrees(self.state.robot_orientation_yaw):.0f}° | '
            f'Moving: {self.state.is_moving}'
        )
        self.status_pub.publish(status_msg)

        # Generate suggested command based on fused data
        cmd = Twist()

        # Simple obstacle avoidance logic
        if self.state.nearest_obstacle_distance < 0.5:
            # Too close! Stop and turn away
            cmd.linear.x = 0.0
            if self.state.nearest_obstacle_angle > 0:
                cmd.angular.z = -0.5  # Turn right
            else:
                cmd.angular.z = 0.5   # Turn left
            self.get_logger().warn('Obstacle too close! Suggesting evasive action.')
        elif self.state.nearest_obstacle_distance < 1.0:
            # Getting close, slow down
            cmd.linear.x = 0.2
            cmd.angular.z = 0.0
        else:
            # Clear path, can move forward
            cmd.linear.x = 0.5
            cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)

    def _check_sensor_health(self, current_time: float) -> str:
        """Check if sensors are providing recent data."""
        timeout = 1.0  # seconds

        statuses = []

        if self.state.camera_timestamp and \
           (current_time - self.state.camera_timestamp) < timeout:
            statuses.append('CAM:OK')
        else:
            statuses.append('CAM:--')

        if self.state.lidar_timestamp and \
           (current_time - self.state.lidar_timestamp) < timeout:
            statuses.append('LDR:OK')
        else:
            statuses.append('LDR:--')

        if self.state.imu_timestamp and \
           (current_time - self.state.imu_timestamp) < timeout:
            statuses.append('IMU:OK')
        else:
            statuses.append('IMU:--')

        return ' '.join(statuses)

    def _get_timestamp(self, stamp) -> float:
        """Convert ROS timestamp to float seconds."""
        return stamp.sec + stamp.nanosec / 1e9


def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
