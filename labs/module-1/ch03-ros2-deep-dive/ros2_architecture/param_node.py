#!/usr/bin/env python3
"""
ROS 2 Parameters Example

Parameters allow runtime configuration of nodes:
- Declared at startup with default values
- Can be changed at runtime via CLI or services
- Support various types (bool, int, float, string, arrays)

Use parameters for:
- Tuning values (PID gains, thresholds)
- Feature flags
- File paths, device names
"""

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult


class ParameterDemoNode(Node):
    """Demonstrates ROS 2 parameter usage."""

    def __init__(self):
        super().__init__('param_demo')

        # Declare parameters with descriptions
        self.declare_parameter(
            'robot_name',
            'DefaultBot',
            ParameterDescriptor(description='Name of the robot')
        )

        self.declare_parameter(
            'max_speed',
            1.0,
            ParameterDescriptor(
                description='Maximum speed in m/s',
                additional_constraints='Must be positive'
            )
        )

        self.declare_parameter(
            'enable_logging',
            True,
            ParameterDescriptor(description='Enable verbose logging')
        )

        self.declare_parameter(
            'sensor_topics',
            ['camera', 'lidar', 'imu'],
            ParameterDescriptor(description='List of sensor topics to subscribe')
        )

        # Register callback for parameter changes
        self.add_on_set_parameters_callback(self.parameter_callback)

        # Timer to demonstrate parameter usage
        self.timer = self.create_timer(2.0, self.timer_callback)

        self.get_logger().info('Parameter Demo Node started')
        self._log_parameters()

    def parameter_callback(self, params):
        """Called when parameters are changed."""
        for param in params:
            self.get_logger().info(
                f'Parameter changed: {param.name} = {param.value}'
            )

            # Validate max_speed
            if param.name == 'max_speed' and param.value <= 0:
                self.get_logger().warn('max_speed must be positive, rejecting')
                return SetParametersResult(
                    successful=False,
                    reason='max_speed must be positive'
                )

        return SetParametersResult(successful=True)

    def timer_callback(self):
        """Periodically log current parameter values."""
        if self.get_parameter('enable_logging').value:
            self._log_parameters()

    def _log_parameters(self):
        """Log all parameter values."""
        robot_name = self.get_parameter('robot_name').value
        max_speed = self.get_parameter('max_speed').value
        logging = self.get_parameter('enable_logging').value
        sensors = self.get_parameter('sensor_topics').value

        self.get_logger().info('Current parameters:')
        self.get_logger().info(f'  robot_name: {robot_name}')
        self.get_logger().info(f'  max_speed: {max_speed} m/s')
        self.get_logger().info(f'  enable_logging: {logging}')
        self.get_logger().info(f'  sensor_topics: {sensors}')


def main(args=None):
    rclpy.init(args=args)
    node = ParameterDemoNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
