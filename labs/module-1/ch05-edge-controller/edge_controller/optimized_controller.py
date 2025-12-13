#!/usr/bin/env python3
"""
Optimized Robot Controller for Edge Deployment

Demonstrates advanced optimization techniques:
- Zero-copy message passing where possible
- Efficient data structures
- Computation batching
- State machine for mode switching
"""

import math
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


class ControlMode(Enum):
    """Robot control modes for edge deployment."""
    IDLE = auto()
    MANUAL = auto()
    AUTONOMOUS = auto()
    EMERGENCY_STOP = auto()


@dataclass
class RobotState:
    """Efficient state storage - uses slots for memory optimization."""
    __slots__ = ['x', 'y', 'theta', 'linear_vel', 'angular_vel', 'mode']

    x: float
    y: float
    theta: float
    linear_vel: float
    angular_vel: float
    mode: ControlMode

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.mode = ControlMode.IDLE


class OptimizedController(Node):
    """
    Memory and CPU optimized robot controller.

    Optimization techniques demonstrated:
    1. Preallocated message objects
    2. Efficient state machine
    3. Batched sensor processing
    4. Minimal allocations in callbacks
    """

    # Class-level constants (shared across instances)
    MAX_LINEAR_VEL = 0.5  # m/s
    MAX_ANGULAR_VEL = 1.0  # rad/s
    OBSTACLE_THRESHOLD = 0.5  # meters
    SAFE_DISTANCE = 1.0  # meters

    def __init__(self):
        super().__init__('optimized_controller')

        # Lightweight QoS for edge
        edge_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # State (single allocation)
        self.state = RobotState()

        # Preallocated messages (reused every cycle)
        self._cmd_vel = Twist()
        self._status = String()

        # Sensor data buffer (fixed size, reused)
        self._scan_ranges: list[float] = []
        self._front_distance = float('inf')
        self._left_distance = float('inf')
        self._right_distance = float('inf')

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', edge_qos)
        self.status_pub = self.create_publisher(String, 'controller/status', 10)

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, edge_qos
        )
        self.mode_sub = self.create_subscription(
            String, 'controller/mode', self.mode_callback, 10
        )

        # Control loop timer (20 Hz - reasonable for edge)
        self.control_timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info('Optimized controller started')

    def scan_callback(self, msg: LaserScan):
        """
        Process LIDAR scan efficiently.

        - Extract only needed values
        - No unnecessary copies
        """
        num_ranges = len(msg.ranges)
        if num_ranges == 0:
            return

        # Calculate sector indices (front, left, right)
        # Assuming 360-degree scan
        front_idx = num_ranges // 2
        left_idx = num_ranges // 4
        right_idx = 3 * num_ranges // 4

        # Get minimum in each sector (with bounds checking)
        sector_size = max(1, num_ranges // 8)

        def min_in_range(start: int, end: int) -> float:
            """Get minimum valid range in sector."""
            valid = [r for r in msg.ranges[start:end]
                    if msg.range_min < r < msg.range_max]
            return min(valid) if valid else float('inf')

        self._front_distance = min_in_range(
            front_idx - sector_size, front_idx + sector_size
        )
        self._left_distance = min_in_range(
            left_idx - sector_size, left_idx + sector_size
        )
        self._right_distance = min_in_range(
            right_idx - sector_size, right_idx + sector_size
        )

    def mode_callback(self, msg: String):
        """Handle mode change requests."""
        mode_map = {
            'idle': ControlMode.IDLE,
            'manual': ControlMode.MANUAL,
            'auto': ControlMode.AUTONOMOUS,
            'autonomous': ControlMode.AUTONOMOUS,
            'stop': ControlMode.EMERGENCY_STOP,
            'emergency': ControlMode.EMERGENCY_STOP,
        }

        new_mode = mode_map.get(msg.data.lower())
        if new_mode:
            self.state.mode = new_mode
            self.get_logger().info(f'Mode changed to: {new_mode.name}')

    def control_loop(self):
        """
        Main control loop - optimized for edge execution.

        State machine approach is efficient:
        - Single branch per cycle
        - No redundant computations
        """
        # State machine dispatch
        if self.state.mode == ControlMode.IDLE:
            self._handle_idle()
        elif self.state.mode == ControlMode.AUTONOMOUS:
            self._handle_autonomous()
        elif self.state.mode == ControlMode.EMERGENCY_STOP:
            self._handle_emergency()
        # MANUAL mode: do nothing, external commands control robot

        # Publish command (reusing preallocated message)
        self.cmd_pub.publish(self._cmd_vel)

        # Publish status periodically (every 10th cycle)
        if hasattr(self, '_cycle_count'):
            self._cycle_count += 1
        else:
            self._cycle_count = 0

        if self._cycle_count % 10 == 0:
            self._publish_status()

    def _handle_idle(self):
        """Idle mode - stop all motion."""
        self._cmd_vel.linear.x = 0.0
        self._cmd_vel.angular.z = 0.0

    def _handle_autonomous(self):
        """
        Autonomous navigation with obstacle avoidance.

        Simple but efficient algorithm:
        - Move forward if clear
        - Turn away from obstacles
        """
        # Check for obstacles
        if self._front_distance < self.OBSTACLE_THRESHOLD:
            # Emergency: obstacle too close
            self._cmd_vel.linear.x = -0.1  # Back up
            # Turn toward clearer side
            if self._left_distance > self._right_distance:
                self._cmd_vel.angular.z = self.MAX_ANGULAR_VEL
            else:
                self._cmd_vel.angular.z = -self.MAX_ANGULAR_VEL

        elif self._front_distance < self.SAFE_DISTANCE:
            # Caution: slow down and start turning
            self._cmd_vel.linear.x = self.MAX_LINEAR_VEL * 0.3

            # Proportional steering
            if self._left_distance > self._right_distance:
                self._cmd_vel.angular.z = 0.5
            else:
                self._cmd_vel.angular.z = -0.5

        else:
            # Clear path - full speed ahead
            self._cmd_vel.linear.x = self.MAX_LINEAR_VEL
            self._cmd_vel.angular.z = 0.0

    def _handle_emergency(self):
        """Emergency stop - all motion halted."""
        self._cmd_vel.linear.x = 0.0
        self._cmd_vel.angular.z = 0.0

    def _publish_status(self):
        """Publish status efficiently."""
        self._status.data = (
            f'{self.state.mode.name}|'
            f'F:{self._front_distance:.2f}|'
            f'L:{self._left_distance:.2f}|'
            f'R:{self._right_distance:.2f}'
        )
        self.status_pub.publish(self._status)


def main(args=None):
    rclpy.init(args=args)
    node = OptimizedController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
