#!/usr/bin/env python3
"""
Behavior Tree Implementation for Humanoid Robot Control

Implements a behavior tree framework for organizing complex robot
behaviors into modular, reusable components. Supports standard BT
nodes: Sequence, Selector, Parallel, Decorator, Action, and Condition.

Lab 14: Final Capstone - Complete Humanoid System
"""

from enum import Enum
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time


class NodeStatus(Enum):
    """Status returned by behavior tree nodes."""
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"


@dataclass
class Blackboard:
    """
    Shared data storage for behavior tree.

    The blackboard pattern allows nodes to share data without
    direct coupling between them.
    """
    data: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from blackboard."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set value on blackboard."""
        self.data[key] = value

    def has(self, key: str) -> bool:
        """Check if key exists on blackboard."""
        return key in self.data

    def delete(self, key: str) -> None:
        """Delete key from blackboard."""
        if key in self.data:
            del self.data[key]

    def clear(self) -> None:
        """Clear all data from blackboard."""
        self.data.clear()


class BTNode(ABC):
    """
    Abstract base class for all behavior tree nodes.

    Each node has a tick() method that is called during tree traversal.
    The tick returns one of: SUCCESS, FAILURE, or RUNNING.
    """

    def __init__(self, name: str):
        """
        Initialize behavior tree node.

        Args:
            name: Human-readable name for the node
        """
        self.name = name
        self.status = NodeStatus.FAILURE
        self.blackboard: Optional[Blackboard] = None

    @abstractmethod
    def tick(self) -> NodeStatus:
        """
        Execute this node's behavior.

        Returns:
            NodeStatus indicating result
        """
        pass

    def reset(self) -> None:
        """Reset node to initial state."""
        self.status = NodeStatus.FAILURE

    def set_blackboard(self, blackboard: Blackboard) -> None:
        """Set the blackboard for data sharing."""
        self.blackboard = blackboard


class CompositeNode(BTNode):
    """Base class for nodes that have children."""

    def __init__(self, name: str, children: Optional[List[BTNode]] = None):
        """
        Initialize composite node.

        Args:
            name: Node name
            children: List of child nodes
        """
        super().__init__(name)
        self.children: List[BTNode] = children or []

    def add_child(self, child: BTNode) -> 'CompositeNode':
        """Add a child node. Returns self for chaining."""
        self.children.append(child)
        return self

    def set_blackboard(self, blackboard: Blackboard) -> None:
        """Set blackboard for this node and all children."""
        super().set_blackboard(blackboard)
        for child in self.children:
            child.set_blackboard(blackboard)

    def reset(self) -> None:
        """Reset this node and all children."""
        super().reset()
        for child in self.children:
            child.reset()


class Sequence(CompositeNode):
    """
    Sequence Node (AND logic)

    Executes children in order. Returns:
    - SUCCESS if all children succeed
    - FAILURE if any child fails
    - RUNNING if a child is running
    """

    def __init__(self, name: str = "Sequence", children: Optional[List[BTNode]] = None):
        super().__init__(name, children)
        self.current_child_idx = 0

    def tick(self) -> NodeStatus:
        """Tick children in sequence."""
        while self.current_child_idx < len(self.children):
            child = self.children[self.current_child_idx]
            status = child.tick()

            if status == NodeStatus.RUNNING:
                self.status = NodeStatus.RUNNING
                return self.status

            if status == NodeStatus.FAILURE:
                self.current_child_idx = 0  # Reset for next tick
                self.status = NodeStatus.FAILURE
                return self.status

            # Child succeeded, move to next
            self.current_child_idx += 1

        # All children succeeded
        self.current_child_idx = 0
        self.status = NodeStatus.SUCCESS
        return self.status

    def reset(self) -> None:
        super().reset()
        self.current_child_idx = 0


class Selector(CompositeNode):
    """
    Selector Node (OR logic / Fallback)

    Executes children in order until one succeeds. Returns:
    - SUCCESS if any child succeeds
    - FAILURE if all children fail
    - RUNNING if a child is running
    """

    def __init__(self, name: str = "Selector", children: Optional[List[BTNode]] = None):
        super().__init__(name, children)
        self.current_child_idx = 0

    def tick(self) -> NodeStatus:
        """Tick children until one succeeds."""
        while self.current_child_idx < len(self.children):
            child = self.children[self.current_child_idx]
            status = child.tick()

            if status == NodeStatus.RUNNING:
                self.status = NodeStatus.RUNNING
                return self.status

            if status == NodeStatus.SUCCESS:
                self.current_child_idx = 0
                self.status = NodeStatus.SUCCESS
                return self.status

            # Child failed, try next
            self.current_child_idx += 1

        # All children failed
        self.current_child_idx = 0
        self.status = NodeStatus.FAILURE
        return self.status

    def reset(self) -> None:
        super().reset()
        self.current_child_idx = 0


class Parallel(CompositeNode):
    """
    Parallel Node

    Executes all children simultaneously. Configurable success/failure thresholds.
    """

    def __init__(self, name: str = "Parallel",
                 children: Optional[List[BTNode]] = None,
                 success_threshold: int = -1,
                 failure_threshold: int = 1):
        """
        Initialize parallel node.

        Args:
            name: Node name
            children: Child nodes
            success_threshold: Number of successes needed (-1 = all)
            failure_threshold: Number of failures to fail (1 = any)
        """
        super().__init__(name, children)
        self.success_threshold = success_threshold
        self.failure_threshold = failure_threshold

    def tick(self) -> NodeStatus:
        """Tick all children and evaluate thresholds."""
        success_count = 0
        failure_count = 0
        running_count = 0

        for child in self.children:
            status = child.tick()

            if status == NodeStatus.SUCCESS:
                success_count += 1
            elif status == NodeStatus.FAILURE:
                failure_count += 1
            else:
                running_count += 1

        # Check failure threshold
        if failure_count >= self.failure_threshold:
            self.status = NodeStatus.FAILURE
            return self.status

        # Check success threshold
        threshold = self.success_threshold if self.success_threshold > 0 else len(self.children)
        if success_count >= threshold:
            self.status = NodeStatus.SUCCESS
            return self.status

        # Still running
        if running_count > 0:
            self.status = NodeStatus.RUNNING
        else:
            # All done, but didn't meet thresholds
            self.status = NodeStatus.FAILURE

        return self.status


class DecoratorNode(BTNode):
    """Base class for decorator nodes (single child)."""

    def __init__(self, name: str, child: Optional[BTNode] = None):
        super().__init__(name)
        self.child = child

    def set_child(self, child: BTNode) -> 'DecoratorNode':
        """Set the child node. Returns self for chaining."""
        self.child = child
        return self

    def set_blackboard(self, blackboard: Blackboard) -> None:
        super().set_blackboard(blackboard)
        if self.child:
            self.child.set_blackboard(blackboard)

    def reset(self) -> None:
        super().reset()
        if self.child:
            self.child.reset()


class Inverter(DecoratorNode):
    """
    Inverter Decorator

    Inverts the result of its child:
    - SUCCESS → FAILURE
    - FAILURE → SUCCESS
    - RUNNING → RUNNING
    """

    def __init__(self, name: str = "Inverter", child: Optional[BTNode] = None):
        super().__init__(name, child)

    def tick(self) -> NodeStatus:
        if not self.child:
            return NodeStatus.FAILURE

        status = self.child.tick()

        if status == NodeStatus.SUCCESS:
            self.status = NodeStatus.FAILURE
        elif status == NodeStatus.FAILURE:
            self.status = NodeStatus.SUCCESS
        else:
            self.status = NodeStatus.RUNNING

        return self.status


class Repeater(DecoratorNode):
    """
    Repeater Decorator

    Repeats its child a specified number of times.
    """

    def __init__(self, name: str = "Repeater",
                 child: Optional[BTNode] = None,
                 repeat_count: int = 3):
        super().__init__(name, child)
        self.repeat_count = repeat_count
        self.current_count = 0

    def tick(self) -> NodeStatus:
        if not self.child:
            return NodeStatus.FAILURE

        while self.current_count < self.repeat_count:
            status = self.child.tick()

            if status == NodeStatus.RUNNING:
                self.status = NodeStatus.RUNNING
                return self.status

            if status == NodeStatus.FAILURE:
                self.current_count = 0
                self.status = NodeStatus.FAILURE
                return self.status

            # Child succeeded
            self.current_count += 1
            self.child.reset()

        # All repetitions complete
        self.current_count = 0
        self.status = NodeStatus.SUCCESS
        return self.status

    def reset(self) -> None:
        super().reset()
        self.current_count = 0


class Retry(DecoratorNode):
    """
    Retry Decorator

    Retries its child on failure up to a maximum number of attempts.
    """

    def __init__(self, name: str = "Retry",
                 child: Optional[BTNode] = None,
                 max_attempts: int = 3):
        super().__init__(name, child)
        self.max_attempts = max_attempts
        self.current_attempt = 0

    def tick(self) -> NodeStatus:
        if not self.child:
            return NodeStatus.FAILURE

        while self.current_attempt < self.max_attempts:
            status = self.child.tick()

            if status == NodeStatus.RUNNING:
                self.status = NodeStatus.RUNNING
                return self.status

            if status == NodeStatus.SUCCESS:
                self.current_attempt = 0
                self.status = NodeStatus.SUCCESS
                return self.status

            # Child failed, retry
            self.current_attempt += 1
            self.child.reset()

        # All attempts failed
        self.current_attempt = 0
        self.status = NodeStatus.FAILURE
        return self.status

    def reset(self) -> None:
        super().reset()
        self.current_attempt = 0


class Timeout(DecoratorNode):
    """
    Timeout Decorator

    Fails if child takes longer than specified duration.
    """

    def __init__(self, name: str = "Timeout",
                 child: Optional[BTNode] = None,
                 timeout_seconds: float = 5.0):
        super().__init__(name, child)
        self.timeout_seconds = timeout_seconds
        self.start_time: Optional[float] = None

    def tick(self) -> NodeStatus:
        if not self.child:
            return NodeStatus.FAILURE

        if self.start_time is None:
            self.start_time = time.time()

        # Check timeout
        elapsed = time.time() - self.start_time
        if elapsed > self.timeout_seconds:
            self.start_time = None
            self.status = NodeStatus.FAILURE
            return self.status

        status = self.child.tick()

        if status != NodeStatus.RUNNING:
            self.start_time = None

        self.status = status
        return self.status

    def reset(self) -> None:
        super().reset()
        self.start_time = None


class ConditionNode(BTNode):
    """
    Condition Node

    Evaluates a condition and returns SUCCESS or FAILURE.
    Never returns RUNNING.
    """

    def __init__(self, name: str, condition_func: Callable[[], bool]):
        """
        Initialize condition node.

        Args:
            name: Node name
            condition_func: Function that returns True/False
        """
        super().__init__(name)
        self.condition_func = condition_func

    def tick(self) -> NodeStatus:
        """Evaluate the condition."""
        try:
            result = self.condition_func()
            self.status = NodeStatus.SUCCESS if result else NodeStatus.FAILURE
        except Exception:
            self.status = NodeStatus.FAILURE

        return self.status


class ActionNode(BTNode):
    """
    Action Node

    Executes an action. Can return SUCCESS, FAILURE, or RUNNING.
    """

    def __init__(self, name: str,
                 action_func: Optional[Callable[[], NodeStatus]] = None):
        """
        Initialize action node.

        Args:
            name: Node name
            action_func: Function that returns NodeStatus
        """
        super().__init__(name)
        self.action_func = action_func

    def tick(self) -> NodeStatus:
        """Execute the action."""
        if self.action_func is None:
            self.status = NodeStatus.FAILURE
            return self.status

        try:
            self.status = self.action_func()
        except Exception:
            self.status = NodeStatus.FAILURE

        return self.status


class BlackboardCondition(BTNode):
    """
    Blackboard Condition Node

    Checks a condition on the blackboard.
    """

    def __init__(self, name: str, key: str, expected_value: Any = True):
        """
        Initialize blackboard condition.

        Args:
            name: Node name
            key: Blackboard key to check
            expected_value: Expected value (default True)
        """
        super().__init__(name)
        self.key = key
        self.expected_value = expected_value

    def tick(self) -> NodeStatus:
        """Check blackboard condition."""
        if self.blackboard is None:
            self.status = NodeStatus.FAILURE
            return self.status

        value = self.blackboard.get(self.key)
        if value == self.expected_value:
            self.status = NodeStatus.SUCCESS
        else:
            self.status = NodeStatus.FAILURE

        return self.status


class BlackboardSet(BTNode):
    """
    Blackboard Set Node

    Sets a value on the blackboard and returns SUCCESS.
    """

    def __init__(self, name: str, key: str, value: Any):
        super().__init__(name)
        self.key = key
        self.value = value

    def tick(self) -> NodeStatus:
        """Set blackboard value."""
        if self.blackboard is not None:
            self.blackboard.set(self.key, self.value)
            self.status = NodeStatus.SUCCESS
        else:
            self.status = NodeStatus.FAILURE

        return self.status


class BehaviorTree:
    """
    Behavior Tree Manager

    Manages the root node and provides utilities for tree execution.
    """

    def __init__(self, root: Optional[BTNode] = None, name: str = "BehaviorTree"):
        """
        Initialize behavior tree.

        Args:
            root: Root node of the tree
            name: Name for the tree
        """
        self.name = name
        self.root = root
        self.blackboard = Blackboard()
        self.tick_count = 0

        if root:
            root.set_blackboard(self.blackboard)

    def set_root(self, root: BTNode) -> None:
        """Set the root node."""
        self.root = root
        root.set_blackboard(self.blackboard)

    def tick(self) -> NodeStatus:
        """
        Tick the behavior tree.

        Returns:
            Status of root node after tick
        """
        if self.root is None:
            return NodeStatus.FAILURE

        self.tick_count += 1
        return self.root.tick()

    def reset(self) -> None:
        """Reset the entire tree."""
        if self.root:
            self.root.reset()
        self.tick_count = 0


# =============================================================================
# Robot-Specific Behavior Tree Nodes
# =============================================================================

class NavigateToAction(ActionNode):
    """Navigate robot to a target position."""

    def __init__(self, name: str = "NavigateTo"):
        super().__init__(name)
        self.progress = 0.0
        self.target_key = "navigation_target"

    def tick(self) -> NodeStatus:
        if self.blackboard is None:
            return NodeStatus.FAILURE

        target = self.blackboard.get(self.target_key)
        if target is None:
            return NodeStatus.FAILURE

        # Simulate navigation progress
        self.progress += 0.2
        if self.progress >= 1.0:
            self.progress = 0.0
            self.blackboard.set("robot_position", target)
            return NodeStatus.SUCCESS

        return NodeStatus.RUNNING

    def reset(self) -> None:
        super().reset()
        self.progress = 0.0


class PickObjectAction(ActionNode):
    """Pick up an object."""

    def __init__(self, name: str = "PickObject"):
        super().__init__(name)
        self.progress = 0.0
        self.object_key = "target_object"

    def tick(self) -> NodeStatus:
        if self.blackboard is None:
            return NodeStatus.FAILURE

        target_obj = self.blackboard.get(self.object_key)
        if target_obj is None:
            return NodeStatus.FAILURE

        # Simulate pick progress
        self.progress += 0.25
        if self.progress >= 1.0:
            self.progress = 0.0
            self.blackboard.set("holding_object", target_obj)
            return NodeStatus.SUCCESS

        return NodeStatus.RUNNING

    def reset(self) -> None:
        super().reset()
        self.progress = 0.0


class PlaceObjectAction(ActionNode):
    """Place held object."""

    def __init__(self, name: str = "PlaceObject"):
        super().__init__(name)
        self.progress = 0.0
        self.destination_key = "place_destination"

    def tick(self) -> NodeStatus:
        if self.blackboard is None:
            return NodeStatus.FAILURE

        holding = self.blackboard.get("holding_object")
        if holding is None:
            return NodeStatus.FAILURE

        # Simulate place progress
        self.progress += 0.25
        if self.progress >= 1.0:
            self.progress = 0.0
            self.blackboard.delete("holding_object")
            return NodeStatus.SUCCESS

        return NodeStatus.RUNNING

    def reset(self) -> None:
        super().reset()
        self.progress = 0.0


class IsHoldingObject(BTNode):
    """Check if robot is holding an object."""

    def __init__(self, name: str = "IsHoldingObject"):
        super().__init__(name)

    def tick(self) -> NodeStatus:
        if self.blackboard is None:
            return NodeStatus.FAILURE

        holding = self.blackboard.get("holding_object")
        if holding is not None:
            return NodeStatus.SUCCESS
        return NodeStatus.FAILURE


class IsAtPosition(BTNode):
    """Check if robot is at target position."""

    def __init__(self, name: str = "IsAtPosition", tolerance: float = 0.1):
        super().__init__(name)
        self.tolerance = tolerance

    def tick(self) -> NodeStatus:
        if self.blackboard is None:
            return NodeStatus.FAILURE

        current = self.blackboard.get("robot_position")
        target = self.blackboard.get("navigation_target")

        if current is None or target is None:
            return NodeStatus.FAILURE

        import numpy as np
        distance = np.linalg.norm(np.array(current) - np.array(target))

        if distance < self.tolerance:
            return NodeStatus.SUCCESS
        return NodeStatus.FAILURE


def build_pick_and_place_tree() -> BehaviorTree:
    """
    Build a behavior tree for pick-and-place task.

    Tree structure:
    Root (Sequence)
    ├── Navigate to object
    ├── Pick object
    ├── Navigate to destination
    └── Place object
    """
    # Create nodes
    nav_to_object = NavigateToAction("Navigate to Object")
    pick = PickObjectAction("Pick Object")
    nav_to_dest = NavigateToAction("Navigate to Destination")
    place = PlaceObjectAction("Place Object")

    # Build tree
    root = Sequence("PickAndPlace").add_child(
        nav_to_object
    ).add_child(
        pick
    ).add_child(
        nav_to_dest
    ).add_child(
        place
    )

    tree = BehaviorTree(root, "PickAndPlaceTree")
    return tree


def build_robust_pick_and_place_tree() -> BehaviorTree:
    """
    Build a more robust pick-and-place tree with fallbacks and retries.

    Tree structure:
    Root (Sequence)
    ├── Selector (Find and approach object)
    │   ├── IsAtPosition
    │   └── Retry(3) -> Navigate to object
    ├── Selector (Pick object)
    │   ├── IsHoldingObject
    │   └── Retry(3) -> Pick
    ├── Timeout(30s) -> Navigate to destination
    └── Place object
    """
    # Condition nodes
    at_object = IsAtPosition("At Object Position")
    holding = IsHoldingObject("Holding Object")

    # Action nodes with retries
    nav_to_object = Retry("Retry Navigate", NavigateToAction("Navigate to Object"), max_attempts=3)
    pick = Retry("Retry Pick", PickObjectAction("Pick Object"), max_attempts=3)
    nav_to_dest = Timeout("Timeout Navigate", NavigateToAction("Navigate to Dest"), timeout_seconds=30)
    place = PlaceObjectAction("Place Object")

    # Build tree with selectors for robustness
    approach_selector = Selector("Approach Object").add_child(at_object).add_child(nav_to_object)
    pick_selector = Selector("Acquire Object").add_child(holding).add_child(pick)

    root = Sequence("Robust PickAndPlace").add_child(
        approach_selector
    ).add_child(
        pick_selector
    ).add_child(
        nav_to_dest
    ).add_child(
        place
    )

    tree = BehaviorTree(root, "RobustPickAndPlaceTree")
    return tree


def visualize_tree(node: BTNode, indent: int = 0) -> str:
    """Generate text visualization of behavior tree."""
    lines = []
    prefix = "  " * indent

    # Node type indicator
    if isinstance(node, Sequence):
        type_str = "[→]"  # Sequence
    elif isinstance(node, Selector):
        type_str = "[?]"  # Selector
    elif isinstance(node, Parallel):
        type_str = "[⇒]"  # Parallel
    elif isinstance(node, DecoratorNode):
        type_str = "[D]"  # Decorator
    elif isinstance(node, ConditionNode) or isinstance(node, BlackboardCondition):
        type_str = "[C]"  # Condition
    else:
        type_str = "[A]"  # Action

    lines.append(f"{prefix}{type_str} {node.name}")

    # Add children
    if isinstance(node, CompositeNode):
        for child in node.children:
            lines.append(visualize_tree(child, indent + 1))
    elif isinstance(node, DecoratorNode) and node.child:
        lines.append(visualize_tree(node.child, indent + 1))

    return "\n".join(lines)


def main():
    """Test behavior tree implementation."""
    print("Testing Behavior Tree Implementation")
    print("=" * 60)

    # Test basic tree
    print("\n1. Basic Pick and Place Tree")
    tree = build_pick_and_place_tree()

    # Set up blackboard
    tree.blackboard.set("navigation_target", [1.0, 0.0, 0.0])
    tree.blackboard.set("target_object", "cup")
    tree.blackboard.set("place_destination", [2.0, 0.0, 0.0])

    print("Tree structure:")
    print(visualize_tree(tree.root))

    # Execute tree
    print("\nExecuting tree...")
    max_ticks = 20
    for i in range(max_ticks):
        status = tree.tick()
        print(f"  Tick {i+1}: {status.value}")

        if status != NodeStatus.RUNNING:
            break

    print(f"\nFinal status: {status.value}")
    print(f"Holding object: {tree.blackboard.get('holding_object')}")

    # Test robust tree
    print("\n" + "=" * 60)
    print("\n2. Robust Pick and Place Tree")
    robust_tree = build_robust_pick_and_place_tree()

    # Set up blackboard
    robust_tree.blackboard.set("navigation_target", [1.0, 0.0, 0.0])
    robust_tree.blackboard.set("target_object", "bottle")
    robust_tree.blackboard.set("place_destination", [3.0, 0.0, 0.0])

    print("Tree structure:")
    print(visualize_tree(robust_tree.root))

    # Execute
    print("\nExecuting robust tree...")
    for i in range(max_ticks):
        status = robust_tree.tick()
        print(f"  Tick {i+1}: {status.value}")

        if status != NodeStatus.RUNNING:
            break

    print(f"\nFinal status: {status.value}")

    # Test individual nodes
    print("\n" + "=" * 60)
    print("\n3. Testing Individual Nodes")

    # Test Inverter
    success_action = ActionNode("AlwaysSucceed", lambda: NodeStatus.SUCCESS)
    inverter = Inverter("InvertSuccess", success_action)
    result = inverter.tick()
    print(f"  Inverter(SUCCESS) = {result.value}")

    # Test Repeater
    counter = {"count": 0}
    def count_action():
        counter["count"] += 1
        return NodeStatus.SUCCESS

    repeat_action = ActionNode("Count", count_action)
    repeater = Repeater("Repeat3", repeat_action, repeat_count=3)
    while repeater.tick() == NodeStatus.RUNNING:
        pass
    print(f"  Repeater executed action {counter['count']} times")

    print("\n" + "=" * 60)
    print("Behavior Tree tests completed!")


if __name__ == "__main__":
    main()
