#!/usr/bin/env python3
"""
Full Humanoid Robot System Integration

Integrates perception, locomotion, manipulation, and language understanding
into a complete humanoid robot control system. This is the capstone module
that ties together all components from the course.

Lab 14: Final Capstone - Complete Humanoid System
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import time
import threading
from queue import Queue, Empty


class SystemState(Enum):
    """Overall system state."""
    IDLE = "idle"
    EXECUTING = "executing"
    PAUSED = "paused"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"


class SubsystemStatus(Enum):
    """Status of individual subsystems."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class RobotConfiguration:
    """Robot hardware configuration."""
    name: str = "Humanoid-01"
    # Physical parameters
    height: float = 1.7  # meters
    mass: float = 70.0   # kg
    arm_reach: float = 0.8  # meters
    # Joint limits
    num_dof: int = 32
    joint_limits: Optional[np.ndarray] = None  # [num_dof, 2] min/max
    # Sensor config
    has_stereo_camera: bool = True
    has_depth_sensor: bool = True
    has_lidar: bool = False
    has_force_sensors: bool = True
    # Capabilities
    can_walk: bool = True
    can_manipulate: bool = True
    can_speak: bool = True


@dataclass
class SensorData:
    """Aggregated sensor data from robot."""
    timestamp: float
    # Vision
    rgb_image: Optional[np.ndarray] = None
    depth_image: Optional[np.ndarray] = None
    # Proprioception
    joint_positions: Optional[np.ndarray] = None
    joint_velocities: Optional[np.ndarray] = None
    joint_torques: Optional[np.ndarray] = None
    # IMU
    orientation: Optional[np.ndarray] = None  # [roll, pitch, yaw]
    angular_velocity: Optional[np.ndarray] = None
    linear_acceleration: Optional[np.ndarray] = None
    # Force/Torque
    left_foot_force: Optional[np.ndarray] = None
    right_foot_force: Optional[np.ndarray] = None
    left_hand_force: Optional[np.ndarray] = None
    right_hand_force: Optional[np.ndarray] = None
    # Localization
    position: Optional[np.ndarray] = None  # [x, y, z]
    velocity: Optional[np.ndarray] = None


@dataclass
class MotorCommand:
    """Command to motor controllers."""
    joint_positions: Optional[np.ndarray] = None
    joint_velocities: Optional[np.ndarray] = None
    joint_torques: Optional[np.ndarray] = None
    control_mode: str = "position"  # position, velocity, torque


@dataclass
class TaskResult:
    """Result of task execution."""
    success: bool
    task_id: str
    duration: float
    message: str
    data: Dict[str, Any] = field(default_factory=dict)


class Subsystem(ABC):
    """Abstract base class for robot subsystems."""

    def __init__(self, name: str):
        self.name = name
        self.status = SubsystemStatus.UNINITIALIZED
        self.last_update: float = 0.0
        self.error_message: str = ""

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the subsystem."""
        pass

    @abstractmethod
    def update(self, sensor_data: SensorData, dt: float) -> None:
        """Update subsystem state."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the subsystem."""
        pass

    def get_status(self) -> Dict[str, Any]:
        """Get subsystem status."""
        return {
            "name": self.name,
            "status": self.status.value,
            "last_update": self.last_update,
            "error": self.error_message
        }


class PerceptionSubsystem(Subsystem):
    """
    Perception Subsystem

    Handles visual perception, object detection, scene understanding,
    and localization.
    """

    def __init__(self):
        super().__init__("perception")
        self.detected_objects: List[Dict] = []
        self.scene_description: str = ""
        self.robot_pose: np.ndarray = np.zeros(6)  # [x, y, z, roll, pitch, yaw]

    def initialize(self) -> bool:
        """Initialize perception pipeline."""
        self.status = SubsystemStatus.INITIALIZING
        try:
            # Initialize vision models (simulated)
            self.status = SubsystemStatus.READY
            return True
        except Exception as e:
            self.error_message = str(e)
            self.status = SubsystemStatus.ERROR
            return False

    def update(self, sensor_data: SensorData, dt: float) -> None:
        """Process sensor data and update perception state."""
        if self.status != SubsystemStatus.READY:
            return

        self.status = SubsystemStatus.BUSY
        self.last_update = sensor_data.timestamp

        # Update robot pose from IMU and localization
        if sensor_data.position is not None:
            self.robot_pose[:3] = sensor_data.position
        if sensor_data.orientation is not None:
            self.robot_pose[3:] = sensor_data.orientation

        # Detect objects (simulated)
        if sensor_data.rgb_image is not None:
            self._detect_objects(sensor_data.rgb_image, sensor_data.depth_image)

        self.status = SubsystemStatus.READY

    def _detect_objects(self, rgb: np.ndarray, depth: Optional[np.ndarray]) -> None:
        """Run object detection on images."""
        # Simulated detection
        np.random.seed(int(time.time() * 1000) % 1000)

        self.detected_objects = []
        n_objects = np.random.randint(2, 5)

        objects = ["cup", "bottle", "book", "phone", "ball"]
        for i in range(n_objects):
            obj = {
                "label": np.random.choice(objects),
                "confidence": np.random.uniform(0.7, 0.99),
                "position": np.random.uniform([-1, -1, 0.5], [1, 1, 2]),
                "bbox": np.random.randint(0, 400, 4)
            }
            self.detected_objects.append(obj)

        self.scene_description = f"Detected {n_objects} objects in scene"

    def get_object_position(self, label: str) -> Optional[np.ndarray]:
        """Get position of object by label."""
        for obj in self.detected_objects:
            if obj["label"] == label:
                return obj["position"]
        return None

    def shutdown(self) -> None:
        """Shutdown perception."""
        self.status = SubsystemStatus.DISABLED


class LocomotionSubsystem(Subsystem):
    """
    Locomotion Subsystem

    Handles walking, balance control, and navigation.
    """

    def __init__(self, config: RobotConfiguration):
        super().__init__("locomotion")
        self.config = config
        self.is_walking = False
        self.target_position: Optional[np.ndarray] = None
        self.current_velocity: np.ndarray = np.zeros(3)
        self.balance_stable: bool = True

    def initialize(self) -> bool:
        """Initialize locomotion controllers."""
        self.status = SubsystemStatus.INITIALIZING
        try:
            # Initialize gait generator and balance controller
            self.status = SubsystemStatus.READY
            return True
        except Exception as e:
            self.error_message = str(e)
            self.status = SubsystemStatus.ERROR
            return False

    def update(self, sensor_data: SensorData, dt: float) -> None:
        """Update locomotion state and control."""
        if self.status != SubsystemStatus.READY:
            return

        self.last_update = sensor_data.timestamp

        # Update balance state from foot forces
        if sensor_data.left_foot_force is not None and sensor_data.right_foot_force is not None:
            total_force = np.sum(sensor_data.left_foot_force) + np.sum(sensor_data.right_foot_force)
            self.balance_stable = total_force > 100  # N threshold

        # Update velocity from sensor data
        if sensor_data.velocity is not None:
            self.current_velocity = sensor_data.velocity

    def walk_to(self, target: np.ndarray) -> bool:
        """Start walking to target position."""
        if not self.config.can_walk:
            return False

        self.target_position = target.copy()
        self.is_walking = True
        self.status = SubsystemStatus.BUSY
        return True

    def stop(self) -> None:
        """Stop walking."""
        self.is_walking = False
        self.target_position = None
        self.status = SubsystemStatus.READY

    def get_walking_progress(self) -> float:
        """Get progress towards target (0-1)."""
        if self.target_position is None:
            return 1.0
        # Simulated progress
        return np.random.uniform(0.5, 1.0)

    def shutdown(self) -> None:
        """Shutdown locomotion."""
        self.stop()
        self.status = SubsystemStatus.DISABLED


class ManipulationSubsystem(Subsystem):
    """
    Manipulation Subsystem

    Handles arm control, grasping, and object manipulation.
    """

    def __init__(self, config: RobotConfiguration):
        super().__init__("manipulation")
        self.config = config
        self.left_gripper_open: bool = True
        self.right_gripper_open: bool = True
        self.holding_object: Optional[str] = None
        self.arm_positions: np.ndarray = np.zeros(14)  # 7 DOF per arm

    def initialize(self) -> bool:
        """Initialize manipulation controllers."""
        self.status = SubsystemStatus.INITIALIZING
        try:
            # Home position for arms
            self.arm_positions = np.zeros(14)
            self.status = SubsystemStatus.READY
            return True
        except Exception as e:
            self.error_message = str(e)
            self.status = SubsystemStatus.ERROR
            return False

    def update(self, sensor_data: SensorData, dt: float) -> None:
        """Update manipulation state."""
        if self.status != SubsystemStatus.READY:
            return

        self.last_update = sensor_data.timestamp

        # Update arm positions from joint state
        if sensor_data.joint_positions is not None and len(sensor_data.joint_positions) >= 14:
            self.arm_positions = sensor_data.joint_positions[:14]

    def reach_to(self, position: np.ndarray, arm: str = "right") -> bool:
        """Move arm end-effector to position."""
        if not self.config.can_manipulate:
            return False

        # Check reachability
        distance = np.linalg.norm(position)
        if distance > self.config.arm_reach:
            return False

        self.status = SubsystemStatus.BUSY
        # IK would be computed here
        return True

    def grasp(self, arm: str = "right") -> bool:
        """Close gripper to grasp object."""
        self.status = SubsystemStatus.BUSY
        if arm == "right":
            self.right_gripper_open = False
        else:
            self.left_gripper_open = False
        self.status = SubsystemStatus.READY
        return True

    def release(self, arm: str = "right") -> bool:
        """Open gripper to release object."""
        if arm == "right":
            self.right_gripper_open = True
        else:
            self.left_gripper_open = True
        self.holding_object = None
        return True

    def pick_object(self, label: str, position: np.ndarray) -> bool:
        """Pick up an object at given position."""
        if self.reach_to(position):
            if self.grasp():
                self.holding_object = label
                self.status = SubsystemStatus.READY
                return True
        return False

    def place_object(self, position: np.ndarray) -> bool:
        """Place held object at position."""
        if self.holding_object is None:
            return False

        if self.reach_to(position):
            if self.release():
                self.status = SubsystemStatus.READY
                return True
        return False

    def shutdown(self) -> None:
        """Shutdown manipulation."""
        self.release("left")
        self.release("right")
        self.status = SubsystemStatus.DISABLED


class SafetySubsystem(Subsystem):
    """
    Safety Subsystem

    Monitors robot state for safety violations and handles
    emergency situations.
    """

    def __init__(self, config: RobotConfiguration):
        super().__init__("safety")
        self.config = config
        self.emergency_stop_active: bool = False
        self.safety_violations: List[str] = []
        # Safety thresholds
        self.max_joint_velocity: float = 2.0  # rad/s
        self.max_joint_torque: float = 100.0  # Nm
        self.min_battery: float = 10.0  # percent
        self.max_tilt: float = 30.0  # degrees

    def initialize(self) -> bool:
        """Initialize safety monitoring."""
        self.status = SubsystemStatus.INITIALIZING
        self.emergency_stop_active = False
        self.safety_violations = []
        self.status = SubsystemStatus.READY
        return True

    def update(self, sensor_data: SensorData, dt: float) -> None:
        """Check for safety violations."""
        if self.status != SubsystemStatus.READY:
            return

        self.last_update = sensor_data.timestamp
        self.safety_violations = []

        # Check joint velocities
        if sensor_data.joint_velocities is not None:
            if np.any(np.abs(sensor_data.joint_velocities) > self.max_joint_velocity):
                self.safety_violations.append("joint_velocity_exceeded")

        # Check joint torques
        if sensor_data.joint_torques is not None:
            if np.any(np.abs(sensor_data.joint_torques) > self.max_joint_torque):
                self.safety_violations.append("joint_torque_exceeded")

        # Check orientation (tilt)
        if sensor_data.orientation is not None:
            roll_deg = np.degrees(sensor_data.orientation[0])
            pitch_deg = np.degrees(sensor_data.orientation[1])
            if abs(roll_deg) > self.max_tilt or abs(pitch_deg) > self.max_tilt:
                self.safety_violations.append("excessive_tilt")

        # Trigger emergency stop if critical violation
        if "excessive_tilt" in self.safety_violations:
            self.trigger_emergency_stop("Robot tilt exceeded safe limits")

    def trigger_emergency_stop(self, reason: str) -> None:
        """Trigger emergency stop."""
        self.emergency_stop_active = True
        self.error_message = f"EMERGENCY STOP: {reason}"
        self.status = SubsystemStatus.ERROR

    def reset_emergency_stop(self) -> bool:
        """Reset emergency stop if safe."""
        if not self.safety_violations:
            self.emergency_stop_active = False
            self.error_message = ""
            self.status = SubsystemStatus.READY
            return True
        return False

    def is_safe(self) -> bool:
        """Check if robot is in safe state."""
        return not self.emergency_stop_active and len(self.safety_violations) == 0

    def shutdown(self) -> None:
        """Shutdown safety system."""
        self.status = SubsystemStatus.DISABLED


class HumanoidRobotSystem:
    """
    Complete Humanoid Robot System

    Main system class that orchestrates all subsystems and provides
    high-level robot control interface.
    """

    def __init__(self, config: Optional[RobotConfiguration] = None):
        """
        Initialize humanoid robot system.

        Args:
            config: Robot configuration (uses defaults if None)
        """
        self.config = config or RobotConfiguration()
        self.state = SystemState.IDLE

        # Initialize subsystems
        self.perception = PerceptionSubsystem()
        self.locomotion = LocomotionSubsystem(self.config)
        self.manipulation = ManipulationSubsystem(self.config)
        self.safety = SafetySubsystem(self.config)

        self.subsystems: List[Subsystem] = [
            self.perception,
            self.locomotion,
            self.manipulation,
            self.safety
        ]

        # Task queue and execution
        self.task_queue: Queue = Queue()
        self.current_task: Optional[Dict] = None
        self.task_history: List[TaskResult] = []

        # Control loop
        self.control_frequency: float = 100.0  # Hz
        self.running: bool = False
        self.control_thread: Optional[threading.Thread] = None

        # Sensor data
        self.latest_sensor_data: Optional[SensorData] = None

        # Callbacks
        self.on_task_complete: Optional[Callable[[TaskResult], None]] = None

    def initialize(self) -> bool:
        """
        Initialize all subsystems.

        Returns:
            True if all subsystems initialized successfully
        """
        print(f"Initializing {self.config.name}...")

        all_success = True
        for subsystem in self.subsystems:
            print(f"  Initializing {subsystem.name}...", end=" ")
            if subsystem.initialize():
                print("OK")
            else:
                print(f"FAILED: {subsystem.error_message}")
                all_success = False

        if all_success:
            self.state = SystemState.IDLE
            print("System initialization complete")
        else:
            self.state = SystemState.ERROR
            print("System initialization failed")

        return all_success

    def start(self) -> None:
        """Start the control loop."""
        if self.state == SystemState.ERROR:
            print("Cannot start: system in error state")
            return

        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.start()
        print("Control loop started")

    def stop(self) -> None:
        """Stop the control loop."""
        self.running = False
        if self.control_thread:
            self.control_thread.join(timeout=2.0)
        print("Control loop stopped")

    def _control_loop(self) -> None:
        """Main control loop running at control_frequency."""
        dt = 1.0 / self.control_frequency
        last_time = time.time()

        while self.running:
            current_time = time.time()
            elapsed = current_time - last_time

            if elapsed >= dt:
                # Get sensor data (simulated)
                sensor_data = self._get_sensor_data(current_time)
                self.latest_sensor_data = sensor_data

                # Update all subsystems
                for subsystem in self.subsystems:
                    subsystem.update(sensor_data, elapsed)

                # Check safety
                if not self.safety.is_safe():
                    self.state = SystemState.EMERGENCY_STOP
                    self._handle_emergency()
                    continue

                # Process tasks
                if self.state == SystemState.EXECUTING:
                    self._process_current_task()
                elif self.state == SystemState.IDLE:
                    self._check_task_queue()

                last_time = current_time

            # Sleep to maintain frequency
            sleep_time = dt - (time.time() - current_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _get_sensor_data(self, timestamp: float) -> SensorData:
        """Get current sensor data (simulated)."""
        return SensorData(
            timestamp=timestamp,
            rgb_image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            depth_image=np.random.uniform(0.5, 5.0, (480, 640)),
            joint_positions=np.random.uniform(-1, 1, self.config.num_dof),
            joint_velocities=np.random.uniform(-0.5, 0.5, self.config.num_dof),
            joint_torques=np.random.uniform(-10, 10, self.config.num_dof),
            orientation=np.random.uniform(-0.1, 0.1, 3),
            angular_velocity=np.random.uniform(-0.1, 0.1, 3),
            linear_acceleration=np.array([0, 0, 9.81]) + np.random.uniform(-0.1, 0.1, 3),
            left_foot_force=np.array([0, 0, 350]) + np.random.uniform(-10, 10, 3),
            right_foot_force=np.array([0, 0, 350]) + np.random.uniform(-10, 10, 3),
            position=np.random.uniform(-0.1, 0.1, 3),
            velocity=np.random.uniform(-0.1, 0.1, 3)
        )

    def _handle_emergency(self) -> None:
        """Handle emergency stop condition."""
        print(f"EMERGENCY: {self.safety.error_message}")
        self.locomotion.stop()
        self.manipulation.release("left")
        self.manipulation.release("right")

    def _check_task_queue(self) -> None:
        """Check for new tasks in queue."""
        try:
            task = self.task_queue.get_nowait()
            self.current_task = task
            self.state = SystemState.EXECUTING
        except Empty:
            pass

    def _process_current_task(self) -> None:
        """Process the current task."""
        if self.current_task is None:
            self.state = SystemState.IDLE
            return

        task_type = self.current_task.get("type")
        task_id = self.current_task.get("id", "unknown")

        # Simulated task completion
        progress = self.current_task.get("progress", 0.0)
        progress += 0.1  # Increment progress
        self.current_task["progress"] = progress

        if progress >= 1.0:
            # Task complete
            result = TaskResult(
                success=True,
                task_id=task_id,
                duration=time.time() - self.current_task.get("start_time", time.time()),
                message=f"Task {task_type} completed successfully"
            )
            self.task_history.append(result)

            if self.on_task_complete:
                self.on_task_complete(result)

            self.current_task = None
            self.state = SystemState.IDLE

    # High-level API methods

    def navigate_to(self, position: np.ndarray) -> str:
        """
        Navigate robot to position.

        Args:
            position: Target [x, y, z] position

        Returns:
            Task ID
        """
        task_id = f"nav_{int(time.time()*1000)}"
        task = {
            "id": task_id,
            "type": "navigate",
            "target": position.copy(),
            "start_time": time.time(),
            "progress": 0.0
        }
        self.task_queue.put(task)
        return task_id

    def pick_up(self, object_label: str) -> str:
        """
        Pick up an object.

        Args:
            object_label: Label of object to pick

        Returns:
            Task ID
        """
        task_id = f"pick_{int(time.time()*1000)}"
        task = {
            "id": task_id,
            "type": "pick",
            "object": object_label,
            "start_time": time.time(),
            "progress": 0.0
        }
        self.task_queue.put(task)
        return task_id

    def place_at(self, position: np.ndarray) -> str:
        """
        Place held object at position.

        Args:
            position: Target position

        Returns:
            Task ID
        """
        task_id = f"place_{int(time.time()*1000)}"
        task = {
            "id": task_id,
            "type": "place",
            "target": position.copy(),
            "start_time": time.time(),
            "progress": 0.0
        }
        self.task_queue.put(task)
        return task_id

    def execute_command(self, command: str) -> str:
        """
        Execute natural language command.

        Args:
            command: Natural language command string

        Returns:
            Task ID
        """
        task_id = f"cmd_{int(time.time()*1000)}"
        task = {
            "id": task_id,
            "type": "language_command",
            "command": command,
            "start_time": time.time(),
            "progress": 0.0
        }
        self.task_queue.put(task)
        return task_id

    def get_status(self) -> Dict[str, Any]:
        """Get complete system status."""
        return {
            "state": self.state.value,
            "config": {
                "name": self.config.name,
                "height": self.config.height,
                "mass": self.config.mass
            },
            "subsystems": {
                s.name: s.get_status() for s in self.subsystems
            },
            "current_task": self.current_task,
            "queue_size": self.task_queue.qsize(),
            "task_history_count": len(self.task_history)
        }

    def shutdown(self) -> None:
        """Shutdown the entire system."""
        print("Shutting down system...")
        self.stop()

        for subsystem in self.subsystems:
            print(f"  Shutting down {subsystem.name}...")
            subsystem.shutdown()

        self.state = SystemState.IDLE
        print("System shutdown complete")


def main():
    """Test full humanoid system."""
    print("Testing Full Humanoid Robot System")
    print("=" * 60)

    # Create system with custom config
    config = RobotConfiguration(
        name="TestBot-01",
        height=1.6,
        mass=60.0
    )

    system = HumanoidRobotSystem(config)

    # Initialize
    if not system.initialize():
        print("Initialization failed!")
        return

    # Get status
    print("\nSystem Status:")
    status = system.get_status()
    print(f"  State: {status['state']}")
    for name, sub_status in status['subsystems'].items():
        print(f"  {name}: {sub_status['status']}")

    # Queue some tasks
    print("\nQueueing tasks...")
    task1 = system.navigate_to(np.array([1.0, 0.0, 0.0]))
    print(f"  Navigate task: {task1}")

    task2 = system.pick_up("cup")
    print(f"  Pick task: {task2}")

    task3 = system.execute_command("Put the cup on the table")
    print(f"  Command task: {task3}")

    # Start control loop briefly
    print("\nStarting control loop...")
    system.start()
    time.sleep(2.0)  # Run for 2 seconds
    system.stop()

    # Check results
    print(f"\nCompleted {len(system.task_history)} tasks")
    for result in system.task_history:
        print(f"  {result.task_id}: {result.message}")

    # Shutdown
    system.shutdown()

    print("\n" + "=" * 60)
    print("Full System tests completed!")


if __name__ == "__main__":
    main()
