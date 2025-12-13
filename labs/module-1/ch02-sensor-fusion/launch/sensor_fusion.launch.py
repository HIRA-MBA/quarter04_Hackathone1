"""Launch file for sensor fusion lab."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for sensor fusion demo."""
    return LaunchDescription([
        # Sensor simulator (generates fake sensor data)
        Node(
            package='sensor_fusion',
            executable='sensor_simulator',
            name='sensor_simulator',
            output='screen',
            parameters=[{
                'camera_fps': 10.0,  # Reduced for simulation
                'lidar_hz': 10.0,
                'imu_hz': 50.0,
            }],
        ),

        # Fusion node (processes and combines sensor data)
        Node(
            package='sensor_fusion',
            executable='fusion_node',
            name='sensor_fusion',
            output='screen',
        ),
    ])
