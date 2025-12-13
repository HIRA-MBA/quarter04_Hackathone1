#!/usr/bin/env python3
"""
ROS 2 Actions Example

Actions are for long-running tasks with feedback:
- Client sends a goal
- Server accepts/rejects the goal
- Server sends periodic feedback
- Server sends final result
- Client can cancel mid-execution

Use actions for:
- Navigation (move to position)
- Manipulation (pick object)
- Any task that takes time and needs progress updates
"""

import time
import rclpy
from rclpy.action import ActionServer, ActionClient, CancelResponse, GoalResponse
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from example_interfaces.action import Fibonacci


class RobotMoveServer(Node):
    """
    Action server for robot movement.

    Uses Fibonacci action as a stand-in for a custom MoveToPosition action.
    In a real robot, you would define a custom action message.
    """

    def __init__(self):
        super().__init__('robot_move_server')

        self._action_server = ActionServer(
            self,
            Fibonacci,
            'robot/move',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup()
        )

        self.get_logger().info('Robot Move Action Server ready')
        self.get_logger().info('Action: /robot/move (Fibonacci)')

    def goal_callback(self, goal_request):
        """Accept or reject incoming goals."""
        self.get_logger().info(f'Received goal request: order={goal_request.order}')

        # Validate the goal
        if goal_request.order < 0:
            self.get_logger().warn('Rejecting goal: negative order')
            return GoalResponse.REJECT

        if goal_request.order > 50:
            self.get_logger().warn('Rejecting goal: order too large')
            return GoalResponse.REJECT

        self.get_logger().info('Accepting goal')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Handle cancellation requests."""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """Execute the goal - this is where the work happens."""
        self.get_logger().info('Executing goal...')

        # Get the goal
        order = goal_handle.request.order

        # Initialize Fibonacci sequence
        feedback_msg = Fibonacci.Feedback()
        feedback_msg.partial_sequence = [0, 1]

        # Execute with feedback
        for i in range(1, order):
            # Check for cancellation
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                result = Fibonacci.Result()
                result.sequence = feedback_msg.partial_sequence
                return result

            # Compute next Fibonacci number
            next_num = feedback_msg.partial_sequence[-1] + feedback_msg.partial_sequence[-2]
            feedback_msg.partial_sequence.append(next_num)

            # Send feedback
            self.get_logger().info(f'Feedback: {feedback_msg.partial_sequence}')
            goal_handle.publish_feedback(feedback_msg)

            # Simulate work (in real robot, this would be actual movement)
            time.sleep(0.5)

        # Goal succeeded
        goal_handle.succeed()

        result = Fibonacci.Result()
        result.sequence = feedback_msg.partial_sequence

        self.get_logger().info(f'Goal succeeded: {result.sequence}')
        return result


class RobotMoveClient(Node):
    """Action client for robot movement."""

    def __init__(self):
        super().__init__('robot_move_client')

        self._action_client = ActionClient(
            self, Fibonacci, 'robot/move'
        )

        self.get_logger().info('Robot Move Action Client ready')

    def send_goal(self, order: int):
        """Send a goal to the action server."""
        self.get_logger().info(f'Waiting for action server...')
        self._action_client.wait_for_server()

        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self.get_logger().info(f'Sending goal: order={order}')

        # Send goal with feedback callback
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Handle goal acceptance/rejection."""
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().warn('Goal rejected')
            return

        self.get_logger().info('Goal accepted')

        # Get result
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        """Handle feedback from the server."""
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Feedback received: {feedback.partial_sequence}')

    def get_result_callback(self, future):
        """Handle the final result."""
        result = future.result().result
        self.get_logger().info(f'Result: {result.sequence}')
        self.get_logger().info('Action complete!')

    def cancel_goal(self):
        """Cancel the current goal."""
        self.get_logger().info('Canceling goal...')
        if hasattr(self, '_goal_handle'):
            self._goal_handle.cancel_goal_async()


def main_server(args=None):
    """Entry point for action server."""
    rclpy.init(args=args)
    node = RobotMoveServer()

    # Use multi-threaded executor for actions
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main_client(args=None):
    """Entry point for action client (demo)."""
    rclpy.init(args=args)
    node = RobotMoveClient()

    # Send a goal
    node.send_goal(10)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main_server()
