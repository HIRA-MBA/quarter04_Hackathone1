import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'sensor_fusion'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Student',
    maintainer_email='student@example.com',
    description='Lab 2: Sensor fusion with camera, LIDAR, and IMU',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_node = sensor_fusion.camera_node:main',
            'lidar_node = sensor_fusion.lidar_node:main',
            'imu_node = sensor_fusion.imu_node:main',
            'fusion_node = sensor_fusion.fusion_node:main',
            'sensor_simulator = sensor_fusion.sensor_simulator:main',
        ],
    },
)
