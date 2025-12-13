#!/usr/bin/env python3
"""
ROS 2 Services Example

Services provide request/response communication:
- Client sends a request
- Server processes the request
- Server sends back a response

Use services for:
- One-time operations (spawn robot, take photo)
- Configuration queries
- State changes that need confirmation
"""

import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts
from std_srvs.srv import SetBool, Trigger


class RobotServiceServer(Node):
    """A service server that provides robot control services."""

    def __init__(self):
        super().__init__('robot_service_server')

        # Robot state
        self._enabled = False
        self._position = 0.0

        # Create services
        self.enable_srv = self.create_service(
            SetBool, 'robot/enable', self.enable_callback
        )
        self.status_srv = self.create_service(
            Trigger, 'robot/status', self.status_callback
        )
        self.calc_srv = self.create_service(
            AddTwoInts, 'robot/calculate', self.calculate_callback
        )

        self.get_logger().info('Robot Service Server ready')
        self.get_logger().info('Services:')
        self.get_logger().info('  /robot/enable (std_srvs/SetBool)')
        self.get_logger().info('  /robot/status (std_srvs/Trigger)')
        self.get_logger().info('  /robot/calculate (example_interfaces/AddTwoInts)')

    def enable_callback(self, request, response):
        """Handle enable/disable requests."""
        self._enabled = request.data

        if self._enabled:
            response.success = True
            response.message = 'Robot enabled successfully'
            self.get_logger().info('Robot ENABLED')
        else:
            response.success = True
            response.message = 'Robot disabled successfully'
            self.get_logger().info('Robot DISABLED')

        return response

    def status_callback(self, request, response):
        """Return current robot status."""
        status = 'ENABLED' if self._enabled else 'DISABLED'
        response.success = True
        response.message = f'Robot status: {status}, Position: {self._position:.2f}'

        self.get_logger().info(f'Status requested: {response.message}')
        return response

    def calculate_callback(self, request, response):
        """Perform calculation (simulating robot computation)."""
        response.sum = request.a + request.b

        self.get_logger().info(f'Calculate: {request.a} + {request.b} = {response.sum}')
        return response


class RobotServiceClient(Node):
    """A service client that calls robot control services."""

    def __init__(self):
        super().__init__('robot_service_client')

        # Create service clients
        self.enable_client = self.create_client(SetBool, 'robot/enable')
        self.status_client = self.create_client(Trigger, 'robot/status')
        self.calc_client = self.create_client(AddTwoInts, 'robot/calculate')

        self.get_logger().info('Robot Service Client ready')

    def wait_for_services(self, timeout_sec=5.0):
        """Wait for all services to be available."""
        services = [
            (self.enable_client, 'robot/enable'),
            (self.status_client, 'robot/status'),
            (self.calc_client, 'robot/calculate'),
        ]

        for client, name in services:
            self.get_logger().info(f'Waiting for service {name}...')
            if not client.wait_for_service(timeout_sec=timeout_sec):
                self.get_logger().error(f'Service {name} not available')
                return False

        self.get_logger().info('All services available')
        return True

    def enable_robot(self, enable: bool):
        """Call the enable service."""
        request = SetBool.Request()
        request.data = enable

        future = self.enable_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            response = future.result()
            self.get_logger().info(f'Enable response: {response.message}')
            return response.success
        else:
            self.get_logger().error('Enable service call failed')
            return False

    def get_status(self):
        """Call the status service."""
        request = Trigger.Request()

        future = self.status_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            response = future.result()
            self.get_logger().info(f'Status: {response.message}')
            return response.message
        else:
            self.get_logger().error('Status service call failed')
            return None

    def calculate(self, a: int, b: int):
        """Call the calculate service."""
        request = AddTwoInts.Request()
        request.a = a
        request.b = b

        future = self.calc_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            response = future.result()
            self.get_logger().info(f'Calculation result: {a} + {b} = {response.sum}')
            return response.sum
        else:
            self.get_logger().error('Calculate service call failed')
            return None


def main_server(args=None):
    """Entry point for service server."""
    rclpy.init(args=args)
    node = RobotServiceServer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main_client(args=None):
    """Entry point for service client (demo)."""
    rclpy.init(args=args)
    node = RobotServiceClient()

    try:
        if node.wait_for_services():
            # Demo sequence
            node.get_logger().info('=== Starting service demo ===')

            # Check initial status
            node.get_status()

            # Enable robot
            node.enable_robot(True)
            node.get_status()

            # Do calculation
            node.calculate(10, 32)

            # Disable robot
            node.enable_robot(False)
            node.get_status()

            node.get_logger().info('=== Demo complete ===')

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main_server()
