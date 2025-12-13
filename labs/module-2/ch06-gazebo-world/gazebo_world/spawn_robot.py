#!/usr/bin/env python3
"""
Robot Spawner for Gazebo Simulation

Spawns a robot model into a running Gazebo simulation using the
/spawn_entity service provided by gazebo_ros.

Features:
- URDF/SDF model loading
- Position and orientation specification
- Multiple robot spawning
- Namespace support
"""

import os
import sys
from typing import Optional

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from geometry_msgs.msg import Pose, Point, Quaternion
from ament_index_python.packages import get_package_share_directory


class RobotSpawner(Node):
    """Spawns robot models into Gazebo simulation."""

    def __init__(self):
        super().__init__('robot_spawner')

        # Declare parameters
        self.declare_parameter('robot_name', 'robot')
        self.declare_parameter('robot_namespace', '')
        self.declare_parameter('x', 0.0)
        self.declare_parameter('y', 0.0)
        self.declare_parameter('z', 0.1)
        self.declare_parameter('roll', 0.0)
        self.declare_parameter('pitch', 0.0)
        self.declare_parameter('yaw', 0.0)
        self.declare_parameter('urdf_file', '')
        self.declare_parameter('sdf_file', '')

        # Create service clients
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_client = self.create_client(DeleteEntity, '/delete_entity')

        self.get_logger().info('Robot Spawner initialized')

    def wait_for_service(self, timeout: float = 10.0) -> bool:
        """Wait for spawn service to be available."""
        self.get_logger().info('Waiting for /spawn_entity service...')
        return self.spawn_client.wait_for_service(timeout_sec=timeout)

    def load_urdf(self, urdf_path: str) -> Optional[str]:
        """Load URDF file content."""
        try:
            with open(urdf_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            self.get_logger().error(f'URDF file not found: {urdf_path}')
            return None
        except Exception as e:
            self.get_logger().error(f'Error loading URDF: {e}')
            return None

    def load_sdf(self, sdf_path: str) -> Optional[str]:
        """Load SDF file content."""
        try:
            with open(sdf_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            self.get_logger().error(f'SDF file not found: {sdf_path}')
            return None
        except Exception as e:
            self.get_logger().error(f'Error loading SDF: {e}')
            return None

    def euler_to_quaternion(self, roll: float, pitch: float, yaw: float) -> Quaternion:
        """Convert Euler angles to quaternion."""
        import math

        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        q = Quaternion()
        q.w = cr * cp * cy + sr * sp * sy
        q.x = sr * cp * cy - cr * sp * sy
        q.y = cr * sp * cy + sr * cp * sy
        q.z = cr * cp * sy - sr * sp * cy

        return q

    def spawn_robot(
        self,
        name: str,
        xml: str,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.1,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        namespace: str = '',
        reference_frame: str = 'world',
    ) -> bool:
        """
        Spawn a robot into Gazebo.

        Args:
            name: Model name in Gazebo
            xml: URDF or SDF content
            x, y, z: Position
            roll, pitch, yaw: Orientation (radians)
            namespace: ROS namespace for the robot
            reference_frame: Frame to spawn relative to

        Returns:
            True if spawn succeeded
        """
        if not self.spawn_client.service_is_ready():
            self.get_logger().error('Spawn service not available')
            return False

        request = SpawnEntity.Request()
        request.name = name
        request.xml = xml
        request.robot_namespace = namespace
        request.reference_frame = reference_frame

        # Set initial pose
        request.initial_pose = Pose()
        request.initial_pose.position = Point(x=x, y=y, z=z)
        request.initial_pose.orientation = self.euler_to_quaternion(roll, pitch, yaw)

        self.get_logger().info(f'Spawning {name} at ({x}, {y}, {z})...')

        future = self.spawn_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Successfully spawned {name}')
                return True
            else:
                self.get_logger().error(f'Failed to spawn: {response.status_message}')
                return False
        else:
            self.get_logger().error('Spawn service call failed')
            return False

    def delete_robot(self, name: str) -> bool:
        """Delete a robot from Gazebo."""
        if not self.delete_client.service_is_ready():
            self.get_logger().error('Delete service not available')
            return False

        request = DeleteEntity.Request()
        request.name = name

        self.get_logger().info(f'Deleting {name}...')

        future = self.delete_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Successfully deleted {name}')
                return True
            else:
                self.get_logger().error(f'Failed to delete: {response.status_message}')
                return False
        return False

    def spawn_from_parameters(self) -> bool:
        """Spawn robot using node parameters."""
        name = self.get_parameter('robot_name').value
        namespace = self.get_parameter('robot_namespace').value
        x = self.get_parameter('x').value
        y = self.get_parameter('y').value
        z = self.get_parameter('z').value
        roll = self.get_parameter('roll').value
        pitch = self.get_parameter('pitch').value
        yaw = self.get_parameter('yaw').value
        urdf_file = self.get_parameter('urdf_file').value
        sdf_file = self.get_parameter('sdf_file').value

        # Load model
        xml = None
        if urdf_file:
            xml = self.load_urdf(urdf_file)
        elif sdf_file:
            xml = self.load_sdf(sdf_file)

        if not xml:
            self.get_logger().error('No valid model file specified')
            return False

        return self.spawn_robot(
            name=name,
            xml=xml,
            x=x, y=y, z=z,
            roll=roll, pitch=pitch, yaw=yaw,
            namespace=namespace,
        )


# Simple differential drive robot SDF for testing
SIMPLE_ROBOT_SDF = """<?xml version="1.0"?>
<sdf version="1.8">
  <model name="simple_robot">
    <link name="base_link">
      <inertial>
        <mass>5.0</mass>
        <inertia>
          <ixx>0.1</ixx>
          <iyy>0.1</iyy>
          <izz>0.1</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <box>
            <size>0.4 0.3 0.15</size>
          </box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>0.4 0.3 0.15</size>
          </box>
        </geometry>
        <material>
          <ambient>0.2 0.4 0.8 1</ambient>
          <diffuse>0.2 0.4 0.8 1</diffuse>
        </material>
      </visual>
    </link>

    <link name="left_wheel">
      <pose>0 0.17 -0.05 -1.5708 0 0</pose>
      <inertial>
        <mass>0.5</mass>
        <inertia>
          <ixx>0.001</ixx>
          <iyy>0.001</iyy>
          <izz>0.001</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>0.08</radius>
            <length>0.04</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.08</radius>
            <length>0.04</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.1 0.1 0.1 1</ambient>
          <diffuse>0.1 0.1 0.1 1</diffuse>
        </material>
      </visual>
    </link>

    <link name="right_wheel">
      <pose>0 -0.17 -0.05 -1.5708 0 0</pose>
      <inertial>
        <mass>0.5</mass>
        <inertia>
          <ixx>0.001</ixx>
          <iyy>0.001</iyy>
          <izz>0.001</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>0.08</radius>
            <length>0.04</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.08</radius>
            <length>0.04</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.1 0.1 0.1 1</ambient>
          <diffuse>0.1 0.1 0.1 1</diffuse>
        </material>
      </visual>
    </link>

    <link name="caster">
      <pose>-0.15 0 -0.1 0 0 0</pose>
      <inertial>
        <mass>0.2</mass>
        <inertia>
          <ixx>0.0001</ixx>
          <iyy>0.0001</iyy>
          <izz>0.0001</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <sphere>
            <radius>0.03</radius>
          </sphere>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>0.0</mu>
              <mu2>0.0</mu2>
            </ode>
          </friction>
        </surface>
      </collision>
      <visual name="visual">
        <geometry>
          <sphere>
            <radius>0.03</radius>
          </sphere>
        </geometry>
        <material>
          <ambient>0.3 0.3 0.3 1</ambient>
          <diffuse>0.3 0.3 0.3 1</diffuse>
        </material>
      </visual>
    </link>

    <joint name="left_wheel_joint" type="revolute">
      <parent>base_link</parent>
      <child>left_wheel</child>
      <axis>
        <xyz>0 0 1</xyz>
      </axis>
    </joint>

    <joint name="right_wheel_joint" type="revolute">
      <parent>base_link</parent>
      <child>right_wheel</child>
      <axis>
        <xyz>0 0 1</xyz>
      </axis>
    </joint>

    <joint name="caster_joint" type="ball">
      <parent>base_link</parent>
      <child>caster</child>
    </joint>

    <!-- Differential drive plugin -->
    <plugin filename="gz-sim-diff-drive-system" name="gz::sim::systems::DiffDrive">
      <left_joint>left_wheel_joint</left_joint>
      <right_joint>right_wheel_joint</right_joint>
      <wheel_separation>0.34</wheel_separation>
      <wheel_radius>0.08</wheel_radius>
      <topic>cmd_vel</topic>
      <odom_topic>odom</odom_topic>
      <frame_id>odom</frame_id>
      <child_frame_id>base_link</child_frame_id>
    </plugin>

  </model>
</sdf>
"""


def main(args=None):
    rclpy.init(args=args)
    spawner = RobotSpawner()

    if not spawner.wait_for_service(timeout=30.0):
        spawner.get_logger().error('Spawn service not available after 30s')
        spawner.destroy_node()
        rclpy.shutdown()
        return

    # Check if model file specified via parameters
    urdf_file = spawner.get_parameter('urdf_file').value
    sdf_file = spawner.get_parameter('sdf_file').value

    if urdf_file or sdf_file:
        # Spawn from file
        success = spawner.spawn_from_parameters()
    else:
        # Spawn default simple robot
        x = spawner.get_parameter('x').value
        y = spawner.get_parameter('y').value
        z = spawner.get_parameter('z').value
        name = spawner.get_parameter('robot_name').value

        spawner.get_logger().info('No model file specified, spawning simple robot')
        success = spawner.spawn_robot(
            name=name,
            xml=SIMPLE_ROBOT_SDF,
            x=x, y=y, z=z,
        )

    spawner.destroy_node()
    rclpy.shutdown()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
