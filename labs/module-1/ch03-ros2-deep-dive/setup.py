from setuptools import find_packages, setup

package_name = 'ros2_architecture'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Student',
    maintainer_email='student@example.com',
    description='Lab 3: ROS 2 Architecture - Lifecycle, Services, Actions',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lifecycle_node = ros2_architecture.lifecycle_node:main',
            'service_server = ros2_architecture.service_node:main_server',
            'service_client = ros2_architecture.service_node:main_client',
            'action_server = ros2_architecture.action_node:main_server',
            'action_client = ros2_architecture.action_node:main_client',
            'param_node = ros2_architecture.param_node:main',
        ],
    },
)
