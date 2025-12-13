# Lab 1: Hello Robot - Step-by-Step Instructions

## Objectives

By the end of this lab, you will:
1. Understand the ROS 2 workspace structure
2. Create and build a ROS 2 Python package
3. Run publisher and subscriber nodes
4. Use ROS 2 CLI tools for debugging

## Prerequisites

- ROS 2 Humble installed (or use the provided Docker container)
- Basic Python knowledge
- Terminal/command line familiarity

## Setup

### Option 1: Using Docker (Recommended)

```bash
cd labs
docker-compose up -d ros2-dev
docker exec -it physical-ai-ros2-dev bash
```

### Option 2: Native ROS 2 Installation

```bash
source /opt/ros/humble/setup.bash
```

## Part 1: Understanding the Package Structure

Examine the files in this lab directory:

```
ch01-hello-robot/
├── package.xml          # Package manifest (dependencies, metadata)
├── setup.py             # Python package configuration
├── resource/
│   └── hello_robot      # Ament resource marker
├── hello_robot/
│   ├── __init__.py      # Python package init
│   └── hello_node.py    # Our ROS 2 node code
├── README.md
└── INSTRUCTIONS.md      # This file
```

### Exercise 1.1: Read the package.xml

Open `package.xml` and identify:
- [ ] Package name
- [ ] Dependencies (what packages does this depend on?)
- [ ] Build type

## Part 2: Building the Package

### Step 1: Create a workspace

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
```

### Step 2: Copy or symlink the lab package

```bash
# Option A: Copy
cp -r /workspace/module-1/ch01-hello-robot ~/ros2_ws/src/

# Option B: Symlink (better for development)
ln -s /workspace/module-1/ch01-hello-robot ~/ros2_ws/src/hello_robot
```

### Step 3: Build with colcon

```bash
cd ~/ros2_ws
colcon build --packages-select hello_robot
```

### Step 4: Source the workspace

```bash
source ~/ros2_ws/install/setup.bash
```

## Part 3: Running the Nodes

### Exercise 3.1: Run the Publisher

In Terminal 1:
```bash
ros2 run hello_robot talker
```

**Expected Output:**
```
[INFO] [hello_publisher]: Hello Publisher node started!
[INFO] [hello_publisher]: Publishing: "Hello, Robot World! Message #0"
[INFO] [hello_publisher]: Publishing: "Hello, Robot World! Message #1"
...
```

### Exercise 3.2: Run the Subscriber

In Terminal 2 (keep Terminal 1 running):
```bash
source ~/ros2_ws/install/setup.bash
ros2 run hello_robot listener
```

**Expected Output:**
```
[INFO] [hello_subscriber]: Hello Subscriber node started!
[INFO] [hello_subscriber]: Received: "Hello, Robot World! Message #5"
[INFO] [hello_subscriber]: Received: "Hello, Robot World! Message #6"
...
```

### Exercise 3.3: Run the Combined Node

Stop the previous nodes (Ctrl+C) and run:
```bash
ros2 run hello_robot hello_robot
```

In another terminal, send a message:
```bash
ros2 topic pub /human_says std_msgs/msg/String "data: 'Hello from human!'" --once
```

## Part 4: ROS 2 CLI Tools

### Exercise 4.1: List Topics

```bash
ros2 topic list
```

**Expected Output:**
```
/hello_topic
/parameter_events
/rosout
```

### Exercise 4.2: Echo a Topic

```bash
ros2 topic echo /hello_topic
```

### Exercise 4.3: Get Topic Info

```bash
ros2 topic info /hello_topic
```

### Exercise 4.4: List Nodes

```bash
ros2 node list
```

### Exercise 4.5: Get Node Info

```bash
ros2 node info /hello_publisher
```

## Part 5: Modify and Experiment

### Challenge 1: Change the Message

Edit `hello_node.py` and change the message content. Rebuild and test:
```bash
cd ~/ros2_ws
colcon build --packages-select hello_robot
source install/setup.bash
ros2 run hello_robot talker
```

### Challenge 2: Change the Timer Period

Modify the `timer_period` variable to publish faster or slower.

### Challenge 3: Add a Counter to Subscriber

Modify `HelloSubscriber` to count how many messages it has received.

### Challenge 4: Create a New Message Type

Create a subscriber that responds differently based on the message content.

## Verification Checklist

- [ ] Successfully built the package
- [ ] Ran the publisher and saw messages
- [ ] Ran the subscriber and received messages
- [ ] Used `ros2 topic list` to see active topics
- [ ] Used `ros2 topic echo` to inspect message contents
- [ ] Made at least one modification to the code

## Common Issues

### "Package not found" error
- Make sure you sourced the workspace: `source ~/ros2_ws/install/setup.bash`

### "No module named rclpy" error
- ROS 2 is not sourced: `source /opt/ros/humble/setup.bash`

### Messages not being received
- Check that both nodes are using the same topic name
- Verify with `ros2 topic list` that the topic exists

## Next Steps

Proceed to Lab 2: Sensor Fusion, where you'll work with camera and LIDAR data.

## Resources

- [ROS 2 Humble Documentation](https://docs.ros.org/en/humble/)
- [ROS 2 Tutorials](https://docs.ros.org/en/humble/Tutorials.html)
- [rclpy API Reference](https://docs.ros2.org/latest/api/rclpy/)
