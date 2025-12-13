from setuptools import find_packages, setup

package_name = 'hello_robot'

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
    description='Lab 1: Your first ROS 2 node',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = hello_robot.hello_node:main_publisher',
            'listener = hello_robot.hello_node:main_subscriber',
            'hello_robot = hello_robot.hello_node:main',
        ],
    },
)
