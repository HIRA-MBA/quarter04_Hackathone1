# Generate ROS 2 Code

Generate ROS 2 Python code following best practices.

## Usage
```
/generate-ros2-code <type> <name>
```

## Types
- `node` - Basic ROS 2 node
- `publisher` - Publisher node
- `subscriber` - Subscriber node
- `service` - Service server/client
- `action` - Action server/client
- `lifecycle` - Lifecycle node
- `launch` - Launch file

## Node Template
```python
import rclpy
from rclpy.node import Node

class {{ClassName}}(Node):
    def __init__(self):
        super().__init__('{{node_name}}')
        self.get_logger().info('Node started')

def main(args=None):
    rclpy.init(args=args)
    node = {{ClassName}}()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Guidelines
- Use type hints
- Add docstrings
- Handle exceptions
- Use parameters for configuration
- Follow PEP 8
