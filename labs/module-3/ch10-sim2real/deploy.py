#!/usr/bin/env python3
"""
Sim-to-Real Deployment Script

This script deploys a trained navigation policy to a real or simulated robot
via ROS 2, handling the necessary transformations for sim2real transfer.

Lab 10: Navigation and Sim2Real Transfer
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray

import numpy as np
import torch
import torch.nn as nn
from typing import Optional
import os


class PolicyNetwork(nn.Module):
    """
    Policy network architecture (must match training).
    """

    def __init__(self, obs_dim: int = 12, act_dim: int = 2, hidden_dim: int = 256):
        super().__init__()

        self.features = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.actor_mean = nn.Linear(hidden_dim, act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        features = self.features(obs)
        return features

    def get_action(self, obs, deterministic=True):
        features = self.forward(obs)
        action_mean = self.actor_mean(features)

        if deterministic:
            return action_mean

        action_std = torch.exp(self.actor_log_std)
        noise = torch.randn_like(action_mean) * action_std
        return action_mean + noise


class Sim2RealAdapter:
    """
    Adapter for handling sim-to-real discrepancies.

    Applies:
    - Observation normalization
    - Action scaling
    - Sensor noise filtering
    - Domain adaptation
    """

    def __init__(self,
                 obs_mean: Optional[np.ndarray] = None,
                 obs_std: Optional[np.ndarray] = None,
                 action_scale: float = 1.0,
                 sensor_noise_std: float = 0.02):
        """
        Initialize sim2real adapter.

        Args:
            obs_mean: Running mean for observation normalization
            obs_std: Running std for observation normalization
            action_scale: Scale factor for actions (sim vs real differences)
            sensor_noise_std: Expected sensor noise standard deviation
        """
        self.obs_mean = obs_mean if obs_mean is not None else np.zeros(12)
        self.obs_std = obs_std if obs_std is not None else np.ones(12)
        self.action_scale = action_scale
        self.sensor_noise_std = sensor_noise_std

        # Running statistics for online normalization
        self.obs_count = 0
        self.running_mean = np.zeros(12)
        self.running_var = np.ones(12)

    def adapt_observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Adapt real-world observation to match simulation distribution.

        Args:
            obs: Raw observation from real sensors

        Returns:
            Adapted observation for policy input
        """
        # Update running statistics
        self.obs_count += 1
        delta = obs - self.running_mean
        self.running_mean += delta / self.obs_count
        self.running_var += delta * (obs - self.running_mean)

        # Normalize observation
        if self.obs_count > 100:
            std = np.sqrt(self.running_var / self.obs_count) + 1e-8
            obs_norm = (obs - self.running_mean) / std
        else:
            obs_norm = (obs - self.obs_mean) / (self.obs_std + 1e-8)

        return obs_norm.astype(np.float32)

    def adapt_action(self, action: np.ndarray) -> np.ndarray:
        """
        Adapt policy action for real robot.

        Args:
            action: Raw action from policy

        Returns:
            Scaled action for real robot
        """
        # Scale actions
        action = action * self.action_scale

        # Apply safety limits
        action[0] = np.clip(action[0], -0.5, 0.5)  # Linear velocity (m/s)
        action[1] = np.clip(action[1], -1.0, 1.0)  # Angular velocity (rad/s)

        return action

    def filter_lidar(self, ranges: np.ndarray) -> np.ndarray:
        """
        Filter LIDAR readings to match simulation behavior.

        Args:
            ranges: Raw LIDAR range readings

        Returns:
            Filtered and subsampled ranges
        """
        # Replace inf/nan values
        ranges = np.nan_to_num(ranges, nan=10.0, posinf=10.0, neginf=10.0)

        # Clip to valid range
        ranges = np.clip(ranges, 0.1, 10.0)

        # Subsample to 5 readings (matching simulation)
        n_readings = 5
        indices = np.linspace(0, len(ranges) - 1, n_readings, dtype=int)
        subsampled = ranges[indices]

        return subsampled


class NavigationDeploymentNode(Node):
    """
    ROS 2 node for deploying trained navigation policy.
    """

    def __init__(self):
        super().__init__('navigation_deployment')

        # Parameters
        self.declare_parameter('model_path', 'models/nav_policy.pt')
        self.declare_parameter('goal_x', 2.0)
        self.declare_parameter('goal_y', 0.0)
        self.declare_parameter('action_scale', 0.5)
        self.declare_parameter('control_frequency', 20.0)
        self.declare_parameter('goal_threshold', 0.3)

        model_path = self.get_parameter('model_path').value
        self.goal = np.array([
            self.get_parameter('goal_x').value,
            self.get_parameter('goal_y').value
        ])
        action_scale = self.get_parameter('action_scale').value
        control_freq = self.get_parameter('control_frequency').value
        self.goal_threshold = self.get_parameter('goal_threshold').value

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        # Load policy
        self.policy = self._load_policy(model_path)

        # Sim2Real adapter
        self.adapter = Sim2RealAdapter(action_scale=action_scale)

        # State
        self.robot_pose = None
        self.robot_velocity = None
        self.lidar_ranges = None
        self.is_active = False

        # QoS
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=1
        )

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            sensor_qos
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            sensor_qos
        )

        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.debug_pub = self.create_publisher(
            Float32MultiArray,
            '/nav_policy/debug',
            10
        )

        # Control timer
        period = 1.0 / control_freq
        self.control_timer = self.create_timer(period, self.control_loop)

        self.get_logger().info("Navigation deployment node initialized")
        self.get_logger().info(f"Initial goal: ({self.goal[0]:.2f}, {self.goal[1]:.2f})")

    def _load_policy(self, model_path: str) -> PolicyNetwork:
        """Load trained policy from checkpoint."""
        policy = PolicyNetwork().to(self.device)

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            policy.load_state_dict(checkpoint["policy_state_dict"])
            policy.eval()
            self.get_logger().info(f"Loaded policy from {model_path}")
        else:
            self.get_logger().warn(f"Policy file not found: {model_path}")
            self.get_logger().warn("Using random policy!")

        return policy

    def odom_callback(self, msg: Odometry):
        """Process odometry data."""
        # Extract position
        pos = msg.pose.pose.position
        self.robot_pose = np.array([pos.x, pos.y])

        # Extract orientation (yaw from quaternion)
        q = msg.pose.pose.orientation
        yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y * q.y + q.z * q.z))
        self.robot_yaw = yaw

        # Extract velocities
        self.robot_velocity = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.angular.z
        ])

    def scan_callback(self, msg: LaserScan):
        """Process laser scan data."""
        ranges = np.array(msg.ranges)
        self.lidar_ranges = self.adapter.filter_lidar(ranges)

    def goal_callback(self, msg: PoseStamped):
        """Update navigation goal."""
        self.goal = np.array([
            msg.pose.position.x,
            msg.pose.position.y
        ])
        self.is_active = True
        self.get_logger().info(f"New goal: ({self.goal[0]:.2f}, {self.goal[1]:.2f})")

    def _build_observation(self) -> Optional[np.ndarray]:
        """Construct observation vector from sensor data."""
        if self.robot_pose is None or self.lidar_ranges is None:
            return None

        # Goal relative to robot
        goal_rel = self.goal - self.robot_pose

        # Build observation (must match training format)
        obs = np.concatenate([
            self.robot_pose,                    # 2: robot position
            [self.robot_yaw],                   # 1: robot orientation
            goal_rel,                           # 2: goal relative
            self.robot_velocity,                # 2: velocities
            self.lidar_ranges,                  # 5: distance sensors
        ])

        return obs

    def control_loop(self):
        """Main control loop - runs at control_frequency."""
        # Check if we have all sensor data
        obs = self._build_observation()
        if obs is None:
            return

        # Check if goal reached
        dist_to_goal = np.linalg.norm(self.goal - self.robot_pose)

        if dist_to_goal < self.goal_threshold:
            self._stop_robot()
            if self.is_active:
                self.get_logger().info("Goal reached!")
                self.is_active = False
            return

        # Adapt observation for policy
        obs_adapted = self.adapter.adapt_observation(obs)

        # Get action from policy
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs_adapted).unsqueeze(0).to(self.device)
            action = self.policy.get_action(obs_tensor, deterministic=True)
            action_np = action.cpu().numpy().squeeze()

        # Adapt action for real robot
        action_adapted = self.adapter.adapt_action(action_np)

        # Publish velocity command
        cmd = Twist()
        cmd.linear.x = float(action_adapted[0])
        cmd.angular.z = float(action_adapted[1])
        self.cmd_vel_pub.publish(cmd)

        # Publish debug info
        debug_msg = Float32MultiArray()
        debug_msg.data = [
            float(dist_to_goal),
            float(action_adapted[0]),
            float(action_adapted[1]),
            float(np.min(self.lidar_ranges)),
        ]
        self.debug_pub.publish(debug_msg)

    def _stop_robot(self):
        """Send zero velocity command."""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)

    def shutdown(self):
        """Clean shutdown."""
        self._stop_robot()
        self.get_logger().info("Navigation deployment node shutting down")


def main(args=None):
    """Main entry point."""
    print("=" * 60)
    print("Lab 10: Sim2Real Navigation Deployment")
    print("=" * 60)

    rclpy.init(args=args)

    node = NavigationDeploymentNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
