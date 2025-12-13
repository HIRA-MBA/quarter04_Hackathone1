from setuptools import find_packages, setup

package_name = 'edge_controller'

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
    description='Lab 5: Edge deployment for embedded robotics',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'edge_node = edge_controller.edge_node:main',
            'optimized_controller = edge_controller.optimized_controller:main',
            'resource_monitor = edge_controller.resource_monitor:main',
        ],
    },
)
