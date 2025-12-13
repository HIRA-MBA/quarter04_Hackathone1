# Lab 3: ROS 2 Architecture Deep Dive

## Objectives

By the end of this lab, you will:
1. Understand and implement lifecycle (managed) nodes
2. Create and use ROS 2 services for request/response communication
3. Implement action servers and clients for long-running tasks
4. Work with ROS 2 parameters for runtime configuration

## Prerequisites

- Completed Labs 1 and 2
- Understanding of ROS 2 nodes, topics, and messages

## Part 1: Lifecycle Nodes

### Concept

Lifecycle nodes have a defined state machine:
```
[Unconfigured] --configure--> [Inactive] --activate--> [Active]
     ^                            |                        |
     |                            v                        v
     +--------cleanup-------------+         deactivate-----+
```

### Exercise 1.1: Run the lifecycle node

```bash
ros2 run ros2_architecture lifecycle_node
```

### Exercise 1.2: View lifecycle state

```bash
ros2 lifecycle list /managed_sensor
ros2 lifecycle get /managed_sensor
```

### Exercise 1.3: Transition through states

```bash
# Configure (allocate resources)
ros2 lifecycle set /managed_sensor configure

# Activate (start publishing)
ros2 lifecycle set /managed_sensor activate

# Watch the data
ros2 topic echo /sensor_data

# Deactivate (stop publishing, keep resources)
ros2 lifecycle set /managed_sensor deactivate

# Cleanup (release resources)
ros2 lifecycle set /managed_sensor cleanup
```

### Key Takeaways

- Lifecycle nodes enable deterministic startup
- Resources are only allocated when configured
- Processing only happens when active
- Enables graceful error handling

## Part 2: Services

### Concept

Services provide synchronous request/response:
```
Client  ---[Request]-->  Server
Client  <--[Response]--  Server
```

### Exercise 2.1: Start the service server

```bash
ros2 run ros2_architecture service_server
```

### Exercise 2.2: List available services

```bash
ros2 service list
ros2 service type /robot/enable
```

### Exercise 2.3: Call services from CLI

```bash
# Enable robot
ros2 service call /robot/enable std_srvs/srv/SetBool "{data: true}"

# Get status
ros2 service call /robot/status std_srvs/srv/Trigger

# Calculate
ros2 service call /robot/calculate example_interfaces/srv/AddTwoInts "{a: 10, b: 32}"

# Disable robot
ros2 service call /robot/enable std_srvs/srv/SetBool "{data: false}"
```

### Exercise 2.4: Run the client demo

```bash
# In another terminal
ros2 run ros2_architecture service_client
```

### Key Takeaways

- Services are for one-time operations
- Client blocks until response received
- Good for: configuration, state queries, commands

## Part 3: Actions

### Concept

Actions are for long-running tasks with feedback:
```
Client  ---[Goal]------>  Server
Client  <--[Accepted]---  Server
Client  <--[Feedback]---  Server (repeated)
Client  <--[Result]-----  Server
```

### Exercise 3.1: Start the action server

```bash
ros2 run ros2_architecture action_server
```

### Exercise 3.2: List actions

```bash
ros2 action list
ros2 action info /robot/move
```

### Exercise 3.3: Send goal from CLI

```bash
ros2 action send_goal /robot/move example_interfaces/action/Fibonacci "{order: 10}" --feedback
```

### Exercise 3.4: Run the client

```bash
ros2 run ros2_architecture action_client
```

### Exercise 3.5: Cancel an action

While an action is running:
```bash
ros2 action send_goal /robot/move example_interfaces/action/Fibonacci "{order: 20}" --feedback
# Press Ctrl+C to cancel
```

### Key Takeaways

- Actions are for tasks that take time
- Provide continuous feedback
- Can be canceled mid-execution
- Good for: navigation, manipulation, any robot motion

## Part 4: Parameters

### Concept

Parameters allow runtime configuration of nodes.

### Exercise 4.1: Run the parameter node

```bash
ros2 run ros2_architecture param_node
```

### Exercise 4.2: List and get parameters

```bash
ros2 param list /param_demo
ros2 param get /param_demo robot_name
ros2 param get /param_demo max_speed
```

### Exercise 4.3: Set parameters at runtime

```bash
ros2 param set /param_demo robot_name "SuperBot"
ros2 param set /param_demo max_speed 2.5
ros2 param set /param_demo enable_logging false
```

### Exercise 4.4: Launch with parameters

```bash
ros2 run ros2_architecture param_node --ros-args \
  -p robot_name:="LaunchBot" \
  -p max_speed:=3.0
```

### Exercise 4.5: Parameter validation

Try setting an invalid value:
```bash
ros2 param set /param_demo max_speed -1.0
```

### Key Takeaways

- Parameters enable runtime configuration
- Can be validated with callbacks
- Support multiple types including arrays
- Can be set at launch or runtime

## Challenges

### Challenge 1: Lifecycle Service Client

Create a node that automatically transitions a lifecycle node through all states with delays between each transition.

### Challenge 2: Custom Service

Define a custom service message and implement a server that performs a robot-specific operation.

### Challenge 3: Action with Progress

Modify the action server to report percentage completion in the feedback.

### Challenge 4: Parameter YAML File

Create a YAML file with all parameters and launch the node loading from it:
```yaml
param_demo:
  ros__parameters:
    robot_name: "ConfigBot"
    max_speed: 2.0
```

## Verification Checklist

- [ ] Transitioned lifecycle node through all states
- [ ] Called services from CLI and programmatically
- [ ] Sent action goals and received feedback
- [ ] Modified parameters at runtime
- [ ] Understood when to use each communication pattern

## Communication Pattern Selection Guide

| Pattern | Use When | Example |
|---------|----------|---------|
| Topics | Continuous data stream | Sensor readings |
| Services | Quick request/response | Enable motor |
| Actions | Long task with feedback | Navigate to goal |
| Parameters | Runtime configuration | Speed limits |

## Next Steps

Proceed to Lab 4: URDF & Humanoid Modeling, where you'll create robot descriptions.

## Resources

- [Lifecycle Nodes](https://design.ros2.org/articles/node_lifecycle.html)
- [Services Tutorial](https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Understanding-ROS2-Services.html)
- [Actions Tutorial](https://docs.ros.org/en/humble/Tutorials/Intermediate/Creating-an-Action.html)
- [Parameters Tutorial](https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Understanding-ROS2-Parameters.html)
