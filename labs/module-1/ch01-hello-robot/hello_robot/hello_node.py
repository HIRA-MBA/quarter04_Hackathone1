#!/usr/bin/env python3
"""
Lab 1: Hello Robot - Your First ROS 2 Node

This module demonstrates the fundamental concepts of ROS 2:
- Creating a node
- Publishing messages to a topic
- Subscribing to messages from a topic
- Using timers for periodic callbacks

Run the publisher:
    ros2 run hello_robot talker

Run the subscriber:
    ros2 run hello_robot listener

Run both in one node:
    ros2 run hello_robot hello_robot
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class HelloPublisher(Node):
    """A simple ROS 2 publisher node that sends greeting messages."""

    def __init__(self):
        super().__init__('hello_publisher')

        # Create a publisher on the 'hello_topic' topic
        # Queue size of 10 means up to 10 messages can be buffered
        self.publisher_ = self.create_publisher(String, 'hello_topic', 10)

        # Create a timer that fires every 1 second
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Counter to track number of messages sent
        self.count = 0

        self.get_logger().info('Hello Publisher node started!')

    def timer_callback(self):
        """Called every timer_period seconds to publish a message."""
        msg = String()
        msg.data = f'Hello, Robot World! Message #{self.count}'

        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')

        self.count += 1


class HelloSubscriber(Node):
    """A simple ROS 2 subscriber node that receives greeting messages."""

    def __init__(self):
        super().__init__('hello_subscriber')

        # Create a subscription to the 'hello_topic' topic
        self.subscription = self.create_subscription(
            String,
            'hello_topic',
            self.listener_callback,
            10  # Queue size
        )
        # Prevent unused variable warning
        self.subscription

        self.get_logger().info('Hello Subscriber node started!')

    def listener_callback(self, msg: String):
        """Called whenever a message is received on the subscribed topic."""
        self.get_logger().info(f'Received: "{msg.data}"')


class HelloRobot(Node):
    """
    Combined publisher and subscriber node.

    Demonstrates how a single node can both publish and subscribe to topics.
    This pattern is common in robotics for nodes that process data and
    output transformed results.
    """

    def __init__(self):
        super().__init__('hello_robot')

        # Publisher
        self.publisher_ = self.create_publisher(String, 'robot_says', 10)

        # Subscriber
        self.subscription = self.create_subscription(
            String,
            'human_says',
            self.human_callback,
            10
        )

        # Timer for autonomous messages
        self.timer = self.create_timer(2.0, self.timer_callback)
        self.count = 0

        self.get_logger().info('Hello Robot node started!')
        self.get_logger().info('Listening on "human_says" topic')
        self.get_logger().info('Publishing to "robot_says" topic')

    def timer_callback(self):
        """Publish periodic status messages."""
        msg = String()
        msg.data = f'Robot status: Active (heartbeat #{self.count})'
        self.publisher_.publish(msg)
        self.count += 1

    def human_callback(self, msg: String):
        """Respond to messages from humans."""
        self.get_logger().info(f'Human says: "{msg.data}"')

        # Echo back with a robot response
        response = String()
        response.data = f'Robot received: "{msg.data}" - Processing...'
        self.publisher_.publish(response)


def main_publisher(args=None):
    """Entry point for the publisher node."""
    rclpy.init(args=args)

    node = HelloPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main_subscriber(args=None):
    """Entry point for the subscriber node."""
    rclpy.init(args=args)

    node = HelloSubscriber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main(args=None):
    """Entry point for the combined hello_robot node."""
    rclpy.init(args=args)

    node = HelloRobot()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
