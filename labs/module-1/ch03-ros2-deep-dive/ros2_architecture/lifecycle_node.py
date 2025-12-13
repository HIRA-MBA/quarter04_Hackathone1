#!/usr/bin/env python3
"""
Lifecycle Node Example

Demonstrates ROS 2 managed (lifecycle) nodes that have a defined state machine:
- Unconfigured -> Inactive -> Active -> Finalized

Lifecycle nodes are important for:
- Deterministic startup sequences
- Graceful error handling
- System-wide coordination
"""

import rclpy
from rclpy.lifecycle import Node as LifecycleNode
from rclpy.lifecycle import State, TransitionCallbackReturn
from std_msgs.msg import String


class ManagedSensorNode(LifecycleNode):
    """
    A lifecycle-managed sensor node.

    States:
    - Unconfigured: Initial state, no resources allocated
    - Inactive: Configured but not processing
    - Active: Fully operational, publishing data
    - Finalized: Shutting down
    """

    def __init__(self):
        super().__init__('managed_sensor')

        # These will be initialized during configuration
        self._publisher = None
        self._timer = None
        self._count = 0

        self.get_logger().info('Lifecycle node created (Unconfigured)')

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        """
        Configure callback - allocate resources.

        Called when transitioning from Unconfigured to Inactive.
        """
        self.get_logger().info('Configuring...')

        try:
            # Create publisher (but don't start publishing yet)
            self._publisher = self.create_publisher(
                String, 'sensor_data', 10
            )
            self._count = 0

            self.get_logger().info('Configuration successful')
            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f'Configuration failed: {e}')
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        """
        Activate callback - start processing.

        Called when transitioning from Inactive to Active.
        """
        self.get_logger().info('Activating...')

        try:
            # Start the timer to publish data
            self._timer = self.create_timer(1.0, self._publish_data)

            self.get_logger().info('Activation successful - now publishing')
            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f'Activation failed: {e}')
            return TransitionCallbackReturn.FAILURE

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        """
        Deactivate callback - stop processing but keep resources.

        Called when transitioning from Active to Inactive.
        """
        self.get_logger().info('Deactivating...')

        # Stop the timer but keep publisher
        if self._timer:
            self._timer.cancel()
            self._timer = None

        self.get_logger().info('Deactivation successful - stopped publishing')
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        """
        Cleanup callback - release resources.

        Called when transitioning from Inactive to Unconfigured.
        """
        self.get_logger().info('Cleaning up...')

        # Destroy publisher
        if self._publisher:
            self.destroy_publisher(self._publisher)
            self._publisher = None

        self._count = 0

        self.get_logger().info('Cleanup successful')
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        """
        Shutdown callback - final cleanup before destruction.

        Called when transitioning to Finalized from any state.
        """
        self.get_logger().info('Shutting down...')

        # Clean up everything
        if self._timer:
            self._timer.cancel()
        if self._publisher:
            self.destroy_publisher(self._publisher)

        self.get_logger().info('Shutdown complete')
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state: State) -> TransitionCallbackReturn:
        """
        Error callback - handle errors gracefully.

        Called when an error occurs during a transition.
        """
        self.get_logger().error(f'Error occurred in state: {state.label}')

        # Attempt to clean up
        if self._timer:
            self._timer.cancel()
            self._timer = None

        return TransitionCallbackReturn.SUCCESS

    def _publish_data(self):
        """Publish sensor data (only called when Active)."""
        if self._publisher is None:
            return

        msg = String()
        msg.data = f'Sensor reading #{self._count}: value={self._count * 1.5:.2f}'
        self._publisher.publish(msg)

        self.get_logger().info(f'Published: {msg.data}')
        self._count += 1


def main(args=None):
    rclpy.init(args=args)

    node = ManagedSensorNode()

    # Use the lifecycle executor
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
