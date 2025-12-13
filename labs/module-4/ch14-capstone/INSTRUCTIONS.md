# Lab 14: Final Capstone - Complete Humanoid System

## Overview

This is the final capstone lab where you integrate all the components from the course into a complete humanoid robot system. You will combine perception, locomotion, manipulation, and language understanding into a unified system using behavior trees for task orchestration.

## Learning Objectives

By completing this lab, you will be able to:

1. Integrate multiple robot subsystems into a cohesive system
2. Design and implement behavior trees for complex task execution
3. Handle inter-subsystem communication via shared state
4. Implement safety systems and fault handling
5. Deploy and operate a complete humanoid robot system

## Prerequisites

- Completed all previous labs (1-13)
- Understanding of:
  - ROS 2 concepts (nodes, topics, services)
  - Balance control and locomotion (Lab 11)
  - Manipulation and grasping (Lab 12)
  - Vision-Language-Action models (Lab 13)
- Python 3.10+
- Access to simulation environment (Gazebo/Isaac Sim) or hardware

## Lab Duration

Estimated time: 8-12 hours (project)

---

## Part 1: System Architecture (1 hour)

### 1.1 Understanding the Full System

The complete humanoid system consists of four main subsystems:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Humanoid Robot System                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐│
│  │ Perception  │  │ Locomotion  │  │Manipulation │  │ Safety  ││
│  │  Subsystem  │  │  Subsystem  │  │  Subsystem  │  │Subsystem││
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └────┬────┘│
│         │                │                │               │     │
│         └────────────────┴────────────────┴───────────────┘     │
│                              │                                   │
│                    ┌─────────┴─────────┐                        │
│                    │    Blackboard     │                        │
│                    │  (Shared State)   │                        │
│                    └─────────┬─────────┘                        │
│                              │                                   │
│                    ┌─────────┴─────────┐                        │
│                    │  Behavior Tree    │                        │
│                    │   (Task Exec)     │                        │
│                    └───────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Review the Code

```bash
# Navigate to lab directory
cd labs/module-4/ch14-capstone

# Review the files
cat full_system.py
cat behavior_tree.py
```

**Exercise 1.1**: Identify the following in `full_system.py`:
- [ ] The four main subsystem classes
- [ ] How sensor data flows through the system
- [ ] How tasks are queued and processed
- [ ] The safety monitoring implementation

---

## Part 2: Running the Full System (1 hour)

### 2.1 Basic System Test

```python
from full_system import HumanoidRobotSystem, RobotConfiguration
import numpy as np

# Create custom configuration
config = RobotConfiguration(
    name="MyHumanoid",
    height=1.7,
    mass=65.0,
    arm_reach=0.8
)

# Initialize system
system = HumanoidRobotSystem(config)
system.initialize()

# Check status
status = system.get_status()
print(f"System state: {status['state']}")
for name, sub in status['subsystems'].items():
    print(f"  {name}: {sub['status']}")
```

### 2.2 Executing Tasks

```python
import time

# Start the control loop
system.start()

# Queue a navigation task
task_id = system.navigate_to(np.array([2.0, 0.0, 0.0]))
print(f"Navigation task queued: {task_id}")

# Queue a pick task
task_id = system.pick_up("cup")
print(f"Pick task queued: {task_id}")

# Let it run
time.sleep(5)

# Check completed tasks
print(f"Completed tasks: {len(system.task_history)}")

# Stop and cleanup
system.stop()
system.shutdown()
```

**Exercise 2.1**: Add a callback to monitor task completion:

```python
def on_task_done(result):
    print(f"Task {result.task_id} completed!")
    print(f"  Success: {result.success}")
    print(f"  Duration: {result.duration:.2f}s")
    print(f"  Message: {result.message}")

system.on_task_complete = on_task_done
# Now run tasks and observe callbacks
```

---

## Part 3: Behavior Trees (2 hours)

### 3.1 Understanding Behavior Tree Nodes

| Node Type | Symbol | Behavior |
|-----------|--------|----------|
| Sequence | [→] | Execute children in order, fail on first failure |
| Selector | [?] | Try children until one succeeds |
| Parallel | [⇒] | Execute all children simultaneously |
| Decorator | [D] | Modify child behavior (invert, repeat, etc.) |
| Condition | [C] | Check a condition, never returns RUNNING |
| Action | [A] | Execute an action, can return RUNNING |

### 3.2 Running a Behavior Tree

```python
from behavior_tree import (
    BehaviorTree, Sequence, Selector, Inverter,
    NavigateToAction, PickObjectAction, PlaceObjectAction,
    IsHoldingObject, visualize_tree
)

# Build a simple tree
tree = BehaviorTree(name="TestTree")

# Create the root sequence
root = Sequence("Main Task")
root.add_child(NavigateToAction("Go to Object"))
root.add_child(PickObjectAction("Pick Up"))
root.add_child(NavigateToAction("Go to Destination"))
root.add_child(PlaceObjectAction("Put Down"))

tree.set_root(root)

# Visualize
print(visualize_tree(tree.root))

# Set up blackboard data
tree.blackboard.set("navigation_target", [1.0, 0.0, 0.0])
tree.blackboard.set("target_object", "cup")
tree.blackboard.set("place_destination", [2.0, 0.0, 0.0])

# Execute
while True:
    status = tree.tick()
    print(f"Tree status: {status.value}")
    if status.value != "running":
        break
```

### 3.3 Building Complex Trees

**Exercise 3.1**: Create a behavior tree for this scenario:

> "The robot should patrol between three waypoints. At each waypoint,
> it should look for objects. If an object is found, pick it up and
> bring it to the home position."

Suggested structure:
```
Root (Sequence)
├── Patrol (Repeater, infinite)
│   └── Sequence
│       ├── Navigate to Waypoint 1
│       ├── Look for Objects (Selector)
│       │   ├── No Object Found → Continue
│       │   └── Object Found → Pick and Return Subtree
│       ├── Navigate to Waypoint 2
│       ├── Look for Objects
│       ├── Navigate to Waypoint 3
│       └── Look for Objects
```

### 3.4 Adding Robustness

**Exercise 3.2**: Add these robustness features to your tree:

1. **Retry on failure**: Wrap navigation actions with `Retry(3)`
2. **Timeout**: Add `Timeout(30s)` for long actions
3. **Fallback behavior**: If pick fails 3 times, skip the object

```python
from behavior_tree import Retry, Timeout, Selector

# Robust navigation
nav_with_retry = Retry("RetryNav", NavigateToAction("Navigate"), max_attempts=3)

# Navigation with timeout
nav_with_timeout = Timeout("TimeoutNav", NavigateToAction("Navigate"), timeout_seconds=30)

# Fallback if action fails
fallback = Selector("TryPickOrSkip")
fallback.add_child(Retry("TryPick", PickObjectAction("Pick"), max_attempts=3))
fallback.add_child(ActionNode("Skip", lambda: NodeStatus.SUCCESS))
```

---

## Part 4: System Integration (2 hours)

### 4.1 Connecting Behavior Trees to System

Create a behavior tree that uses the full system:

```python
from full_system import HumanoidRobotSystem
from behavior_tree import BehaviorTree, Sequence, ActionNode, NodeStatus

class SystemIntegration:
    def __init__(self):
        self.system = HumanoidRobotSystem()
        self.tree = BehaviorTree(name="IntegratedTree")

    def initialize(self):
        self.system.initialize()
        self._build_tree()

    def _build_tree(self):
        # Create action nodes that call system methods
        def navigate_action():
            target = self.tree.blackboard.get("target")
            if target is None:
                return NodeStatus.FAILURE
            self.system.navigate_to(target)
            # Check if navigation complete
            if self.system.locomotion.is_walking:
                return NodeStatus.RUNNING
            return NodeStatus.SUCCESS

        def pick_action():
            obj = self.tree.blackboard.get("object")
            if obj is None:
                return NodeStatus.FAILURE
            self.system.pick_up(obj)
            if self.system.manipulation.holding_object:
                return NodeStatus.SUCCESS
            return NodeStatus.RUNNING

        root = Sequence("Main")
        root.add_child(ActionNode("Navigate", navigate_action))
        root.add_child(ActionNode("Pick", pick_action))

        self.tree.set_root(root)

    def run(self):
        self.system.start()
        while True:
            status = self.tree.tick()
            if status != NodeStatus.RUNNING:
                break
        self.system.stop()
```

### 4.2 Multi-Step Task Execution

**Exercise 4.1**: Implement a "Fetch Object" behavior:

1. Receive command: "Fetch the red cup"
2. Parse command to extract object
3. Use perception to locate object
4. Navigate to object location
5. Pick up object
6. Navigate back to start
7. Hand over object

```python
def build_fetch_tree(system):
    """Build a complete fetch behavior tree."""
    tree = BehaviorTree(name="FetchTree")

    # Your implementation here
    # ...

    return tree
```

---

## Part 5: Safety Implementation (1 hour)

### 5.1 Understanding Safety Checks

Review the safety subsystem:

```python
from full_system import SafetySubsystem, RobotConfiguration

safety = SafetySubsystem(RobotConfiguration())
safety.initialize()

# Check thresholds
print(f"Max joint velocity: {safety.max_joint_velocity} rad/s")
print(f"Max tilt: {safety.max_tilt} degrees")
```

### 5.2 Implementing Safety Nodes

**Exercise 5.1**: Create safety-aware behavior tree nodes:

```python
class SafeCheckNode(BTNode):
    """Check if system is safe before proceeding."""

    def __init__(self, safety_subsystem):
        super().__init__("SafetyCheck")
        self.safety = safety_subsystem

    def tick(self):
        if self.safety.is_safe():
            return NodeStatus.SUCCESS
        return NodeStatus.FAILURE


class EmergencyStopAction(BTNode):
    """Trigger emergency stop."""

    def __init__(self, system):
        super().__init__("EmergencyStop")
        self.system = system

    def tick(self):
        self.system.safety.trigger_emergency_stop("Manual trigger")
        return NodeStatus.SUCCESS
```

### 5.3 Safety-First Tree Structure

Build trees that always check safety first:

```python
# Safety-wrapped action
safe_action = Sequence("SafeAction")
safe_action.add_child(SafeCheckNode(system.safety))
safe_action.add_child(your_action_node)

# Tree with emergency handling
root = Selector("RootWithSafety")
root.add_child(Sequence("NormalOperation")
    .add_child(SafeCheckNode(system.safety))
    .add_child(main_behavior_tree)
)
root.add_child(EmergencyStopAction(system))
```

---

## Part 6: Capstone Project (4-6 hours)

### 6.1 Project Requirements

Build a complete demonstration that showcases:

1. **Perception**: Detect and identify objects in the environment
2. **Locomotion**: Navigate to multiple locations
3. **Manipulation**: Pick up and place objects
4. **Language Interface**: Accept natural language commands
5. **Safety**: Handle errors and emergency situations

### 6.2 Suggested Demo Scenarios

Choose one or create your own:

**Scenario A: Table Setting**
> Robot sets a table by fetching plates, cups, and utensils from a cabinet
> and placing them in correct positions.

**Scenario B: Object Sorting**
> Robot sorts objects by color or type, moving them to designated areas.

**Scenario C: Assistant Robot**
> Robot responds to voice commands like "bring me the water bottle"
> or "take this to the kitchen."

### 6.3 Implementation Steps

1. **Design your behavior tree** (30 min)
   - Draw the tree structure on paper
   - Identify all required nodes

2. **Implement custom nodes** (2 hours)
   - Create action nodes for your specific tasks
   - Add condition nodes for state checks

3. **Integrate with full system** (1 hour)
   - Connect behavior tree to system
   - Set up blackboard communication

4. **Add robustness** (1 hour)
   - Retry mechanisms
   - Error recovery
   - Timeout handling

5. **Testing and debugging** (1-2 hours)
   - Run in simulation
   - Fix issues
   - Optimize performance

### 6.4 Deliverables

Your capstone should include:

- [ ] `capstone_demo.py` - Main demonstration script
- [ ] Custom behavior tree implementation
- [ ] Documentation of tree structure
- [ ] Video/GIF of demo running (optional but recommended)
- [ ] Brief report (1-2 pages) describing:
  - Design decisions
  - Challenges encountered
  - Future improvements

---

## Grading Rubric

| Criteria | Points | Description |
|----------|--------|-------------|
| System Integration | 25 | All subsystems work together correctly |
| Task Completion | 25 | Demo achieves stated objectives |
| Safety & Robustness | 20 | Proper error handling and recovery |
| Code Quality | 15 | Clean, documented, well-structured code |
| Documentation & Demo | 15 | Clear explanation and working demo |

**Total: 100 points**

### Grading Breakdown

**A (90-100)**: All requirements met, robust error handling, creative solution
**B (80-89)**: Core requirements met, basic error handling
**C (70-79)**: Most requirements met, some issues
**D (60-69)**: Partial implementation, significant issues
**F (<60)**: Incomplete or non-functional

---

## Troubleshooting

### Common Issues

1. **System doesn't initialize**
   - Check that all dependencies are installed
   - Verify Python version (3.10+)
   - Check for import errors in subsystems

2. **Tasks never complete**
   - Verify blackboard is set up correctly
   - Check that progress is updating in action nodes
   - Look for infinite loops in behavior tree

3. **Safety triggers unexpectedly**
   - Review safety thresholds
   - Check sensor data values
   - Ensure IMU simulation is reasonable

4. **Behavior tree returns FAILURE immediately**
   - Check that blackboard has required keys
   - Verify condition nodes are configured
   - Add debug prints to trace execution

### Debug Tips

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Add tick counter to track execution
class DebugSequence(Sequence):
    def tick(self):
        print(f"Ticking {self.name}, child {self.current_child_idx}")
        return super().tick()

# Visualize tree state after each tick
def debug_tick(tree):
    status = tree.tick()
    print(f"Tick {tree.tick_count}: {status.value}")
    print(f"Blackboard: {tree.blackboard.data}")
    return status
```

---

## Next Steps

After completing this capstone:

1. **ROS 2 Integration**: Connect your system to actual ROS 2 nodes
2. **Real Hardware**: Deploy on physical robot hardware
3. **Advanced Perception**: Integrate real vision models (YOLO, CLIP)
4. **Multi-Robot**: Extend to coordinate multiple robots
5. **Learning**: Add reinforcement learning for skill acquisition

## References

- [Behavior Trees in Robotics and AI](https://arxiv.org/abs/1709.00084)
- [BehaviorTree.CPP Library](https://www.behaviortree.dev/)
- [ROS 2 BT Navigator](https://navigation.ros.org/behavior_trees/index.html)
- [Safety in Human-Robot Interaction](https://link.springer.com/book/10.1007/978-3-319-24862-5)

---

**Congratulations on reaching the capstone!** This lab represents the culmination of your learning journey through Physical AI and Humanoid Robotics. Good luck with your project!
