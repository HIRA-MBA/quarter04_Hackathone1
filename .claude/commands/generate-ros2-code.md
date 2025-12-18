# Generate ROS 2 Code

Generate ROS 2 Python code following best practices with automatic test generation.

## Arguments
- `$ARGUMENTS` - `<type> <name> [--with-tests]`

## Usage
```
/generate-ros2-code node sensor_processor
/generate-ros2-code publisher camera_feed --with-tests
/generate-ros2-code lifecycle navigation_manager
/generate-ros2-code action move_arm --with-tests
```

## Types
| Type | Description | Output Files |
|------|-------------|--------------|
| `node` | Basic ROS 2 node | `{name}_node.py` |
| `publisher` | Publisher node | `{name}_publisher.py` |
| `subscriber` | Subscriber node | `{name}_subscriber.py` |
| `service` | Service server/client | `{name}_service.py`, `{name}_client.py` |
| `action` | Action server/client | `{name}_action_server.py`, `{name}_action_client.py` |
| `lifecycle` | Managed lifecycle node | `{name}_lifecycle.py` |
| `launch` | Launch file | `{name}_launch.py` |
| `package` | Full ROS 2 package | `package.xml`, `setup.py`, `{name}/` |

---

## Templates

### Basic Node Template
```python
#!/usr/bin/env python3
"""
{{node_name}}_node.py - Brief description

This node handles [purpose].

Subscriptions:
    - /input_topic (sensor_msgs/msg/Image): Input data

Publishers:
    - /output_topic (std_msgs/msg/String): Processed output

Parameters:
    - rate (float): Processing rate in Hz (default: 10.0)
"""
from typing import Optional
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


class {{ClassName}}(Node):
    """
    {{ClassName}} processes [description].

    Attributes:
        _publisher: Output data publisher
        _timer: Processing timer
        _rate: Update rate in Hz
    """

    def __init__(self) -> None:
        """Initialize the {{node_name}} node."""
        super().__init__('{{node_name}}')

        # Declare parameters
        self.declare_parameter('rate', 10.0)
        self._rate = self.get_parameter('rate').value

        # Create publisher with QoS
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._publisher = self.create_publisher(String, 'output_topic', qos)

        # Create timer
        self._timer = self.create_timer(1.0 / self._rate, self._timer_callback)

        self.get_logger().info(f'{{ClassName}} initialized at {self._rate} Hz')

    def _timer_callback(self) -> None:
        """Process and publish data on each timer tick."""
        try:
            msg = String()
            msg.data = 'Hello from {{node_name}}'
            self._publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f'Error in timer callback: {e}')


def main(args: Optional[list] = None) -> None:
    """Entry point for the {{node_name}} node."""
    rclpy.init(args=args)
    node = {{ClassName}}()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Publisher Node Template
```python
#!/usr/bin/env python3
"""
{{node_name}}_publisher.py - Publishes {{topic_type}} messages.
"""
from typing import Optional
import rclpy
from rclpy.node import Node
from {{msg_package}}.msg import {{MsgType}}


class {{ClassName}}Publisher(Node):
    """Publisher node for {{MsgType}} messages."""

    def __init__(self) -> None:
        super().__init__('{{node_name}}_publisher')
        self.declare_parameter('rate', 10.0)
        self.declare_parameter('topic', '{{topic_name}}')

        rate = self.get_parameter('rate').value
        topic = self.get_parameter('topic').value

        self._publisher = self.create_publisher({{MsgType}}, topic, 10)
        self._timer = self.create_timer(1.0 / rate, self._publish)
        self._count = 0

        self.get_logger().info(f'Publishing to {topic} at {rate} Hz')

    def _publish(self) -> None:
        """Publish message."""
        msg = {{MsgType}}()
        # TODO: Populate message fields
        self._publisher.publish(msg)
        self._count += 1


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = {{ClassName}}Publisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Subscriber Node Template
```python
#!/usr/bin/env python3
"""
{{node_name}}_subscriber.py - Subscribes to {{topic_type}} messages.
"""
from typing import Optional
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from {{msg_package}}.msg import {{MsgType}}


class {{ClassName}}Subscriber(Node):
    """Subscriber node for {{MsgType}} messages."""

    def __init__(self) -> None:
        super().__init__('{{node_name}}_subscriber')
        self.declare_parameter('topic', '{{topic_name}}')

        topic = self.get_parameter('topic').value

        # Sensor QoS for real-time data
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self._subscription = self.create_subscription(
            {{MsgType}},
            topic,
            self._callback,
            qos
        )
        self.get_logger().info(f'Subscribed to {topic}')

    def _callback(self, msg: {{MsgType}}) -> None:
        """Process incoming message."""
        # TODO: Process message
        self.get_logger().debug(f'Received message')


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = {{ClassName}}Subscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Lifecycle Node Template
```python
#!/usr/bin/env python3
"""
{{node_name}}_lifecycle.py - Managed lifecycle node.

States: unconfigured -> inactive -> active -> finalized
"""
from typing import Optional
import rclpy
from rclpy.lifecycle import Node as LifecycleNode
from rclpy.lifecycle import State, TransitionCallbackReturn
from std_msgs.msg import String


class {{ClassName}}Lifecycle(LifecycleNode):
    """Lifecycle-managed node with state transitions."""

    def __init__(self) -> None:
        super().__init__('{{node_name}}_lifecycle')
        self._publisher = None
        self._timer = None
        self.get_logger().info('Lifecycle node created (unconfigured)')

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        """Configure the node (allocate resources)."""
        self.get_logger().info(f'Configuring from {state.label}')
        self._publisher = self.create_publisher(String, 'lifecycle_output', 10)
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        """Activate the node (start processing)."""
        self.get_logger().info(f'Activating from {state.label}')
        self._timer = self.create_timer(1.0, self._timer_callback)
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        """Deactivate the node (pause processing)."""
        self.get_logger().info(f'Deactivating from {state.label}')
        self.destroy_timer(self._timer)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        """Cleanup resources."""
        self.get_logger().info(f'Cleaning up from {state.label}')
        self.destroy_publisher(self._publisher)
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        """Shutdown the node."""
        self.get_logger().info(f'Shutting down from {state.label}')
        return TransitionCallbackReturn.SUCCESS

    def _timer_callback(self) -> None:
        """Timer callback (only active when node is active)."""
        msg = String()
        msg.data = 'Lifecycle node active'
        self._publisher.publish(msg)


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = {{ClassName}}Lifecycle()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Service Server Template
```python
#!/usr/bin/env python3
"""
{{node_name}}_service.py - Service server for {{service_type}}.
"""
from typing import Optional
import rclpy
from rclpy.node import Node
from {{srv_package}}.srv import {{SrvType}}


class {{ClassName}}Service(Node):
    """Service server providing {{service_description}}."""

    def __init__(self) -> None:
        super().__init__('{{node_name}}_service')
        self._service = self.create_service(
            {{SrvType}},
            '{{service_name}}',
            self._handle_request
        )
        self.get_logger().info('Service ready: {{service_name}}')

    def _handle_request(
        self,
        request: {{SrvType}}.Request,
        response: {{SrvType}}.Response
    ) -> {{SrvType}}.Response:
        """Handle incoming service request."""
        self.get_logger().info(f'Received request: {request}')
        # TODO: Process request and populate response
        return response


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = {{ClassName}}Service()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Action Server Template
```python
#!/usr/bin/env python3
"""
{{node_name}}_action_server.py - Action server for {{action_type}}.
"""
from typing import Optional
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from {{action_package}}.action import {{ActionType}}


class {{ClassName}}ActionServer(Node):
    """Action server for long-running {{action_description}}."""

    def __init__(self) -> None:
        super().__init__('{{node_name}}_action_server')
        self._action_server = ActionServer(
            self,
            {{ActionType}},
            '{{action_name}}',
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=ReentrantCallbackGroup()
        )
        self.get_logger().info('Action server ready: {{action_name}}')

    def _goal_callback(self, goal_request) -> GoalResponse:
        """Accept or reject goal request."""
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle) -> CancelResponse:
        """Accept or reject cancel request."""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def _execute_callback(self, goal_handle):
        """Execute the goal."""
        self.get_logger().info('Executing goal...')
        feedback = {{ActionType}}.Feedback()
        result = {{ActionType}}.Result()

        # TODO: Implement action execution with feedback
        for i in range(10):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return result

            # Update feedback
            feedback.progress = float(i + 1) / 10.0
            goal_handle.publish_feedback(feedback)
            await asyncio.sleep(0.5)

        goal_handle.succeed()
        result.success = True
        return result


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = {{ClassName}}ActionServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Launch File Template
```python
#!/usr/bin/env python3
"""
{{node_name}}_launch.py - Launch file for {{description}}.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Generate launch description."""
    # Declare arguments
    rate_arg = DeclareLaunchArgument(
        'rate',
        default_value='10.0',
        description='Processing rate in Hz'
    )

    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Node namespace'
    )

    # Create node
    {{node_name}}_node = Node(
        package='{{package_name}}',
        executable='{{node_name}}_node',
        name='{{node_name}}',
        namespace=LaunchConfiguration('namespace'),
        parameters=[{
            'rate': LaunchConfiguration('rate'),
        }],
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([
        rate_arg,
        namespace_arg,
        LogInfo(msg='Launching {{node_name}}...'),
        {{node_name}}_node,
    ])
```

---

## Test Generation (`--with-tests`)

When `--with-tests` flag is provided, generate corresponding test file:

### Test Template
```python
#!/usr/bin/env python3
"""
test_{{node_name}}.py - Unit tests for {{ClassName}}.
"""
import pytest
import rclpy
from rclpy.node import Node
from {{package_name}}.{{node_name}}_node import {{ClassName}}


@pytest.fixture(scope='module')
def ros_context():
    """Initialize ROS 2 context for tests."""
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node(ros_context):
    """Create node instance for testing."""
    node = {{ClassName}}()
    yield node
    node.destroy_node()


class Test{{ClassName}}:
    """Test suite for {{ClassName}}."""

    def test_node_initialization(self, node):
        """Test that node initializes correctly."""
        assert node.get_name() == '{{node_name}}'

    def test_parameter_defaults(self, node):
        """Test default parameter values."""
        rate = node.get_parameter('rate').value
        assert rate == 10.0

    def test_publisher_created(self, node):
        """Test that publisher is created."""
        publishers = node.publishers
        topic_names = [p.topic_name for p in publishers]
        assert '/output_topic' in topic_names or 'output_topic' in topic_names

    def test_timer_callback(self, node):
        """Test timer callback execution."""
        # Manually invoke callback
        node._timer_callback()
        # Verify no exceptions raised


class Test{{ClassName}}Integration:
    """Integration tests requiring full ROS 2 graph."""

    @pytest.mark.integration
    def test_publish_receive(self, node, ros_context):
        """Test message publishing and receiving."""
        from std_msgs.msg import String

        received_msgs = []

        def callback(msg):
            received_msgs.append(msg)

        test_node = Node('test_receiver')
        subscription = test_node.create_subscription(
            String,
            'output_topic',
            callback,
            10
        )

        # Spin briefly to allow message exchange
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(node)
        executor.add_node(test_node)

        for _ in range(10):
            executor.spin_once(timeout_sec=0.1)

        test_node.destroy_node()
        assert len(received_msgs) > 0
```

---

## Package Generation

When generating a full package, create:

```
{{package_name}}/
├── package.xml
├── setup.py
├── setup.cfg
├── resource/{{package_name}}
├── {{package_name}}/
│   ├── __init__.py
│   └── {{node_name}}_node.py
├── launch/
│   └── {{node_name}}_launch.py
├── config/
│   └── params.yaml
└── test/
    ├── test_copyright.py
    ├── test_flake8.py
    ├── test_pep257.py
    └── test_{{node_name}}.py
```

### package.xml Template
```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>{{package_name}}</name>
  <version>0.1.0</version>
  <description>{{description}}</description>
  <maintainer email="{{email}}">{{author}}</maintainer>
  <license>MIT</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

### setup.py Template
```python
from setuptools import find_packages, setup

package_name = '{{package_name}}'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/{{node_name}}_launch.py']),
        ('share/' + package_name + '/config', ['config/params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='{{author}}',
    maintainer_email='{{email}}',
    description='{{description}}',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            '{{node_name}}_node = {{package_name}}.{{node_name}}_node:main',
        ],
    },
)
```

---

## Guidelines

### Code Quality
- ✅ Use type hints for all functions
- ✅ Add comprehensive docstrings (Google style)
- ✅ Handle exceptions with proper logging
- ✅ Use parameters for all configuration
- ✅ Follow PEP 8 formatting
- ✅ Use QoS profiles appropriately

### ROS 2 Best Practices
- ✅ Use lifecycle nodes for managed components
- ✅ Implement proper cleanup in destructors
- ✅ Use callback groups for concurrent execution
- ✅ Prefer composition over inheritance
- ✅ Use async/await for action servers
