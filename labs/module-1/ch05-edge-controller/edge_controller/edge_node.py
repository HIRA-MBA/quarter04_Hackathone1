#!/usr/bin/env python3
"""
Edge-Optimized ROS 2 Node

Demonstrates techniques for running ROS 2 on resource-constrained devices:
- Memory-efficient message handling
- CPU-conscious processing
- Adaptive rate control
- Graceful degradation

Target platforms: Raspberry Pi, NVIDIA Jetson, BeagleBone
"""

import time
import psutil
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Twist


class EdgeOptimizedNode(Node):
    """
    A ROS 2 node optimized for edge/embedded deployment.

    Key optimizations:
    1. Minimal memory footprint
    2. Adaptive processing rate
    3. Resource monitoring
    4. Graceful degradation under load
    """

    def __init__(self):
        super().__init__('edge_node')

        # Declare parameters for runtime tuning
        self.declare_parameter('target_rate', 10.0)  # Hz
        self.declare_parameter('min_rate', 1.0)      # Hz
        self.declare_parameter('max_cpu_percent', 80.0)
        self.declare_parameter('max_memory_percent', 70.0)

        # Get parameters
        self.target_rate = self.get_parameter('target_rate').value
        self.min_rate = self.get_parameter('min_rate').value
        self.max_cpu = self.get_parameter('max_cpu_percent').value
        self.max_memory = self.get_parameter('max_memory_percent').value

        # Current adaptive rate
        self.current_rate = self.target_rate

        # Edge-optimized QoS: Best effort, volatile, small queue
        edge_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1  # Minimal queue to save memory
        )

        # Publishers
        self.status_pub = self.create_publisher(String, 'edge/status', edge_qos)
        self.cmd_pub = self.create_publisher(Twist, 'edge/cmd_vel', edge_qos)
        self.cpu_pub = self.create_publisher(Float32, 'edge/cpu_usage', edge_qos)
        self.mem_pub = self.create_publisher(Float32, 'edge/memory_usage', edge_qos)

        # Subscriber with edge QoS
        self.sensor_sub = self.create_subscription(
            String, 'edge/sensor_input', self.sensor_callback, edge_qos
        )

        # Adaptive timer
        self.main_timer = self.create_timer(1.0 / self.current_rate, self.main_loop)
        self.monitor_timer = self.create_timer(1.0, self.monitor_resources)

        # Processing state (reuse objects to avoid allocation)
        self._cmd_msg = Twist()
        self._status_msg = String()
        self._float_msg = Float32()
        self._last_sensor_data = None
        self._process_count = 0

        self.get_logger().info(f'Edge node started at {self.current_rate} Hz')
        self.get_logger().info(f'CPU limit: {self.max_cpu}%, Memory limit: {self.max_memory}%')

    def sensor_callback(self, msg: String):
        """
        Handle incoming sensor data efficiently.

        - Store reference instead of copying when possible
        - Minimal processing in callback
        """
        self._last_sensor_data = msg.data

    def main_loop(self):
        """
        Main processing loop - runs at adaptive rate.

        Demonstrates:
        - Object reuse (no new allocations)
        - Efficient string formatting
        - Minimal computation
        """
        self._process_count += 1

        # Process sensor data if available
        if self._last_sensor_data:
            # Simple processing - would be replaced with actual control logic
            self._cmd_msg.linear.x = 0.1
            self._cmd_msg.angular.z = 0.0
            self.cmd_pub.publish(self._cmd_msg)

        # Publish status (reuse message object)
        self._status_msg.data = f'cycle:{self._process_count},rate:{self.current_rate:.1f}'
        self.status_pub.publish(self._status_msg)

    def monitor_resources(self):
        """
        Monitor system resources and adapt processing rate.

        This is the key to running reliably on edge devices:
        - Reduce rate when resources are constrained
        - Increase rate when resources are available
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory_percent = psutil.virtual_memory().percent

            # Publish metrics
            self._float_msg.data = cpu_percent
            self.cpu_pub.publish(self._float_msg)

            self._float_msg.data = memory_percent
            self.mem_pub.publish(self._float_msg)

            # Adaptive rate control
            old_rate = self.current_rate

            if cpu_percent > self.max_cpu or memory_percent > self.max_memory:
                # System overloaded - reduce rate
                self.current_rate = max(self.min_rate, self.current_rate * 0.8)
            elif cpu_percent < self.max_cpu * 0.6 and memory_percent < self.max_memory * 0.6:
                # System has headroom - increase rate
                self.current_rate = min(self.target_rate, self.current_rate * 1.1)

            # Update timer if rate changed significantly
            if abs(self.current_rate - old_rate) > 0.5:
                self.main_timer.cancel()
                self.main_timer = self.create_timer(1.0 / self.current_rate, self.main_loop)
                self.get_logger().info(
                    f'Rate adjusted: {old_rate:.1f} -> {self.current_rate:.1f} Hz '
                    f'(CPU: {cpu_percent:.0f}%, MEM: {memory_percent:.0f}%)'
                )

        except Exception as e:
            self.get_logger().warn(f'Resource monitoring error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = EdgeOptimizedNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
