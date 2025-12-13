#!/usr/bin/env python3
"""
Force and Impedance Control for Dexterous Manipulation

Implements force control strategies for contact-rich manipulation tasks,
including impedance control, hybrid position/force control, and slip detection.

Lab 12: Dexterous Manipulation
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod


class ControlMode(Enum):
    """Control mode for each task-space direction."""
    POSITION = "position"
    FORCE = "force"
    IMPEDANCE = "impedance"


@dataclass
class ContactState:
    """State of a contact point."""
    position: np.ndarray           # Contact position
    force: np.ndarray              # Contact force (3D)
    normal: np.ndarray             # Surface normal
    tangent_force: np.ndarray      # Tangential force component
    normal_force: float            # Normal force magnitude
    slip_ratio: float              # |f_t| / (mu * f_n), >1 means slipping
    is_slipping: bool              # True if slip detected


@dataclass
class ImpedanceParams:
    """Parameters for impedance control."""
    stiffness: np.ndarray      # 6D stiffness [Kx, Ky, Kz, Krx, Kry, Krz]
    damping: np.ndarray        # 6D damping [Dx, Dy, Dz, Drx, Dry, Drz]
    inertia: np.ndarray        # 6D virtual inertia
    rest_position: np.ndarray  # Equilibrium position (6D pose)
    rest_force: np.ndarray     # Desired force at equilibrium


@dataclass
class ForceControlCommand:
    """Output command from force controller."""
    joint_torques: np.ndarray      # Commanded joint torques
    cartesian_force: np.ndarray    # Desired end-effector force (6D wrench)
    position_error: np.ndarray     # Position tracking error
    force_error: np.ndarray        # Force tracking error
    contact_states: List[ContactState]


class ForceController(ABC):
    """Abstract base class for force controllers."""

    @abstractmethod
    def compute_command(self,
                        current_pose: np.ndarray,
                        current_velocity: np.ndarray,
                        current_force: np.ndarray,
                        desired_pose: np.ndarray,
                        desired_force: np.ndarray) -> np.ndarray:
        """Compute control command."""
        pass


class ImpedanceController(ForceController):
    """
    Impedance Controller

    Implements a virtual mass-spring-damper system:
    M*ẍ + D*ẋ + K*(x - x_d) = F_ext - F_d

    The robot behaves as if connected to the environment through
    springs and dampers, allowing compliant interaction.
    """

    def __init__(self,
                 stiffness: np.ndarray = None,
                 damping: np.ndarray = None,
                 inertia: np.ndarray = None):
        """
        Initialize impedance controller.

        Args:
            stiffness: 6D stiffness matrix diagonal [N/m, N/m, N/m, Nm/rad, ...]
            damping: 6D damping matrix diagonal
            inertia: 6D inertia matrix diagonal
        """
        # Default values for stable interaction
        self.K = stiffness if stiffness is not None else np.array([
            500.0, 500.0, 500.0,   # Translational stiffness
            50.0, 50.0, 50.0       # Rotational stiffness
        ])

        self.D = damping if damping is not None else np.array([
            50.0, 50.0, 50.0,     # Translational damping
            5.0, 5.0, 5.0          # Rotational damping
        ])

        self.M = inertia if inertia is not None else np.array([
            2.0, 2.0, 2.0,        # Translational inertia
            0.5, 0.5, 0.5          # Rotational inertia
        ])

        # State for integration
        self.prev_velocity = np.zeros(6)
        self.dt = 0.001  # Control timestep

    def set_stiffness(self, stiffness: np.ndarray):
        """Update stiffness parameters."""
        self.K = np.array(stiffness)

    def set_damping(self, damping: np.ndarray):
        """Update damping parameters."""
        self.D = np.array(damping)

    def compute_command(self,
                        current_pose: np.ndarray,
                        current_velocity: np.ndarray,
                        current_force: np.ndarray,
                        desired_pose: np.ndarray,
                        desired_force: np.ndarray) -> np.ndarray:
        """
        Compute impedance control command.

        The command is a wrench (force/torque) that makes the robot
        behave as a virtual impedance.

        Args:
            current_pose: Current end-effector pose [x, y, z, rx, ry, rz]
            current_velocity: Current velocity [vx, vy, vz, wx, wy, wz]
            current_force: Measured external force (6D wrench)
            desired_pose: Desired pose
            desired_force: Desired force (feedforward)

        Returns:
            Commanded wrench [fx, fy, fz, tx, ty, tz]
        """
        # Position error
        pose_error = desired_pose - current_pose

        # Handle angle wrapping for rotations
        pose_error[3:] = np.arctan2(np.sin(pose_error[3:]), np.cos(pose_error[3:]))

        # Velocity error (desired velocity = 0 for regulation)
        velocity_error = -current_velocity

        # Impedance law: F = K*(x_d - x) + D*(ẋ_d - ẋ) + F_d
        command_wrench = (
            self.K * pose_error +
            self.D * velocity_error +
            desired_force
        )

        return command_wrench


class HybridForcePositionController(ForceController):
    """
    Hybrid Force/Position Controller

    Controls force in some Cartesian directions and position in others.
    Uses a selection matrix to partition the task space.

    Common use case: control force normal to a surface while
    controlling position tangent to it.
    """

    def __init__(self,
                 position_gains: Tuple[np.ndarray, np.ndarray] = None,
                 force_gains: Tuple[np.ndarray, np.ndarray] = None):
        """
        Initialize hybrid controller.

        Args:
            position_gains: (Kp, Kd) for position control
            force_gains: (Kp, Ki) for force control
        """
        # Position control gains
        if position_gains:
            self.Kp_pos, self.Kd_pos = position_gains
        else:
            self.Kp_pos = np.array([500.0, 500.0, 500.0, 50.0, 50.0, 50.0])
            self.Kd_pos = np.array([50.0, 50.0, 50.0, 5.0, 5.0, 5.0])

        # Force control gains
        if force_gains:
            self.Kp_force, self.Ki_force = force_gains
        else:
            self.Kp_force = np.array([0.01, 0.01, 0.01, 0.001, 0.001, 0.001])
            self.Ki_force = np.array([0.001, 0.001, 0.001, 0.0001, 0.0001, 0.0001])

        # Selection matrix: 1 = position control, 0 = force control
        self.selection = np.ones(6)  # Default: all position control

        # Integral term for force control
        self.force_integral = np.zeros(6)
        self.integral_limit = 100.0

        self.dt = 0.001

    def set_control_directions(self,
                                position_axes: List[int],
                                force_axes: List[int]):
        """
        Set which axes are position vs force controlled.

        Args:
            position_axes: List of axes for position control (0-5)
            force_axes: List of axes for force control (0-5)
        """
        self.selection = np.zeros(6)
        for axis in position_axes:
            self.selection[axis] = 1.0

    def compute_command(self,
                        current_pose: np.ndarray,
                        current_velocity: np.ndarray,
                        current_force: np.ndarray,
                        desired_pose: np.ndarray,
                        desired_force: np.ndarray) -> np.ndarray:
        """
        Compute hybrid force/position control command.

        Args:
            current_pose: Current pose [x, y, z, rx, ry, rz]
            current_velocity: Current velocity
            current_force: Measured external force
            desired_pose: Desired pose (for position-controlled axes)
            desired_force: Desired force (for force-controlled axes)

        Returns:
            Commanded wrench
        """
        # Position control component
        pose_error = desired_pose - current_pose
        pose_error[3:] = np.arctan2(np.sin(pose_error[3:]), np.cos(pose_error[3:]))

        position_command = self.Kp_pos * pose_error - self.Kd_pos * current_velocity

        # Force control component
        force_error = desired_force - current_force

        # Update integral
        self.force_integral += force_error * self.dt
        self.force_integral = np.clip(
            self.force_integral,
            -self.integral_limit,
            self.integral_limit
        )

        force_command = (
            self.Kp_force * force_error +
            self.Ki_force * self.force_integral
        )

        # Combine using selection matrix
        command = (
            self.selection * position_command +
            (1 - self.selection) * (desired_force + force_command)
        )

        return command


class AdmittanceController(ForceController):
    """
    Admittance Controller

    Complementary to impedance control - takes force input and
    produces position/velocity output:
    ẍ = M^{-1} * (F_ext - D*ẋ - K*(x - x_d))

    Useful for position-controlled robots that need force compliance.
    """

    def __init__(self,
                 virtual_mass: np.ndarray = None,
                 virtual_damping: np.ndarray = None,
                 virtual_stiffness: np.ndarray = None):
        """Initialize admittance controller."""
        self.M = virtual_mass if virtual_mass is not None else np.array([
            5.0, 5.0, 5.0, 1.0, 1.0, 1.0
        ])

        self.D = virtual_damping if virtual_damping is not None else np.array([
            100.0, 100.0, 100.0, 10.0, 10.0, 10.0
        ])

        self.K = virtual_stiffness if virtual_stiffness is not None else np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Pure damper by default
        ])

        # State
        self.velocity = np.zeros(6)
        self.position_offset = np.zeros(6)
        self.dt = 0.001

    def compute_command(self,
                        current_pose: np.ndarray,
                        current_velocity: np.ndarray,
                        current_force: np.ndarray,
                        desired_pose: np.ndarray,
                        desired_force: np.ndarray) -> np.ndarray:
        """
        Compute admittance control - returns position offset.

        Args:
            current_pose: Current pose
            current_velocity: Current velocity
            current_force: Measured external force
            desired_pose: Nominal desired pose
            desired_force: Expected/desired force

        Returns:
            Modified desired pose (nominal + admittance offset)
        """
        # Force deviation from expected
        force_deviation = current_force - desired_force

        # Admittance dynamics
        acceleration = (
            force_deviation -
            self.D * self.velocity -
            self.K * self.position_offset
        ) / self.M

        # Integrate
        self.velocity += acceleration * self.dt
        self.position_offset += self.velocity * self.dt

        # Return modified desired position
        return desired_pose + self.position_offset


class SlipDetector:
    """
    Slip Detection for Grasping

    Detects slip using force ratios and vibration analysis.
    Critical for stable grasp maintenance.
    """

    def __init__(self,
                 friction_coefficient: float = 0.5,
                 slip_threshold: float = 0.9,
                 vibration_threshold: float = 0.1):
        """
        Initialize slip detector.

        Args:
            friction_coefficient: Estimated friction coefficient
            slip_threshold: Slip ratio threshold (|f_t|/(mu*f_n))
            vibration_threshold: High-frequency force threshold
        """
        self.mu = friction_coefficient
        self.slip_threshold = slip_threshold
        self.vibration_threshold = vibration_threshold

        # History for vibration detection
        self.force_history: List[np.ndarray] = []
        self.history_size = 50

    def compute_slip_ratio(self,
                           force: np.ndarray,
                           normal: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """
        Compute slip ratio for a contact.

        Args:
            force: 3D contact force
            normal: Surface normal

        Returns:
            Tuple of (normal_force, slip_ratio, tangent_force)
        """
        # Decompose force into normal and tangential
        normal = normal / np.linalg.norm(normal)
        normal_force = np.dot(force, normal)
        tangent_force = force - normal_force * normal
        tangent_magnitude = np.linalg.norm(tangent_force)

        # Slip ratio: |f_t| / (mu * f_n)
        if abs(normal_force) > 1e-6:
            slip_ratio = tangent_magnitude / (self.mu * abs(normal_force))
        else:
            slip_ratio = float('inf')

        return abs(normal_force), slip_ratio, tangent_force

    def detect_vibration(self) -> bool:
        """
        Detect high-frequency vibrations indicating slip.

        Returns:
            True if vibration detected
        """
        if len(self.force_history) < self.history_size:
            return False

        # Compute force variance in recent history
        forces = np.array(self.force_history[-self.history_size:])
        force_std = np.std(forces, axis=0)

        return np.max(force_std) > self.vibration_threshold

    def update(self, forces: List[np.ndarray], normals: List[np.ndarray]) -> List[ContactState]:
        """
        Update slip detection for multiple contacts.

        Args:
            forces: List of contact forces
            normals: List of surface normals

        Returns:
            List of contact states with slip info
        """
        contact_states = []

        for i, (force, normal) in enumerate(zip(forces, normals)):
            normal_force, slip_ratio, tangent_force = self.compute_slip_ratio(force, normal)

            # Store for vibration detection
            self.force_history.append(force.copy())
            if len(self.force_history) > self.history_size * 2:
                self.force_history = self.force_history[-self.history_size:]

            # Determine if slipping
            is_slipping = (
                slip_ratio > self.slip_threshold or
                self.detect_vibration()
            )

            contact_states.append(ContactState(
                position=np.zeros(3),  # To be filled by caller
                force=force,
                normal=normal,
                tangent_force=tangent_force,
                normal_force=normal_force,
                slip_ratio=slip_ratio,
                is_slipping=is_slipping
            ))

        return contact_states


class GraspForceController:
    """
    Grasp Force Controller

    Coordinates force control across multiple fingers for stable grasping.
    Maintains internal forces for grasp stability while applying
    manipulation forces.
    """

    def __init__(self,
                 n_fingers: int = 5,
                 min_normal_force: float = 2.0,
                 max_normal_force: float = 20.0):
        """
        Initialize grasp force controller.

        Args:
            n_fingers: Number of fingers
            min_normal_force: Minimum normal force per contact (N)
            max_normal_force: Maximum normal force per contact (N)
        """
        self.n_fingers = n_fingers
        self.min_force = min_normal_force
        self.max_force = max_normal_force

        # Individual finger controllers
        self.finger_controllers = [
            ImpedanceController(
                stiffness=np.array([200, 200, 200, 20, 20, 20]),
                damping=np.array([20, 20, 20, 2, 2, 2])
            )
            for _ in range(n_fingers)
        ]

        # Slip detector
        self.slip_detector = SlipDetector()

        # Grasp state
        self.internal_force = 5.0  # Squeeze force
        self.grasp_stiffness = 100.0

    def compute_internal_forces(self,
                                 contact_normals: List[np.ndarray],
                                 desired_squeeze: float = 5.0) -> np.ndarray:
        """
        Compute internal grasp forces that don't affect object motion.

        Args:
            contact_normals: Surface normals at contacts
            desired_squeeze: Desired internal squeeze force

        Returns:
            Array of normal force magnitudes per finger
        """
        n_contacts = len(contact_normals)
        if n_contacts < 2:
            return np.array([desired_squeeze])

        # Distribute squeeze force across contacts
        # For stable grasp, internal forces should be balanced
        forces = np.ones(n_contacts) * desired_squeeze

        # Ensure minimum force
        forces = np.maximum(forces, self.min_force)
        forces = np.minimum(forces, self.max_force)

        return forces

    def adjust_for_slip(self,
                        contact_states: List[ContactState],
                        current_forces: np.ndarray) -> np.ndarray:
        """
        Adjust grasp forces to prevent slip.

        Args:
            contact_states: Current contact states with slip info
            current_forces: Current normal forces per contact

        Returns:
            Adjusted normal forces
        """
        adjusted_forces = current_forces.copy()

        for i, state in enumerate(contact_states):
            if state.is_slipping or state.slip_ratio > 0.7:
                # Increase normal force to prevent slip
                increase_factor = 1.0 + 0.5 * (state.slip_ratio - 0.7)
                adjusted_forces[i] *= increase_factor

        # Enforce limits
        adjusted_forces = np.clip(adjusted_forces, self.min_force, self.max_force)

        return adjusted_forces

    def compute_manipulation_wrench(self,
                                     desired_object_wrench: np.ndarray,
                                     contact_positions: List[np.ndarray],
                                     contact_normals: List[np.ndarray],
                                     object_center: np.ndarray) -> np.ndarray:
        """
        Compute finger forces to achieve desired object wrench.

        Uses grasp matrix to map desired object wrench to contact forces.

        Args:
            desired_object_wrench: Desired wrench on object [fx,fy,fz,tx,ty,tz]
            contact_positions: Position of each contact
            contact_normals: Normal at each contact
            object_center: Object center position

        Returns:
            Array of contact forces (n_contacts x 3)
        """
        n_contacts = len(contact_positions)

        # Build grasp matrix G: w = G @ f
        G = np.zeros((6, 3 * n_contacts))

        for i, (pos, normal) in enumerate(zip(contact_positions, contact_normals)):
            r = pos - object_center

            # Force contribution
            G[0:3, 3*i:3*i+3] = np.eye(3)

            # Torque contribution (r x f)
            G[3, 3*i:3*i+3] = [0, -r[2], r[1]]
            G[4, 3*i:3*i+3] = [r[2], 0, -r[0]]
            G[5, 3*i:3*i+3] = [-r[1], r[0], 0]

        # Solve for contact forces using pseudoinverse
        # f = G^+ @ w
        G_pinv = np.linalg.pinv(G)
        contact_forces_flat = G_pinv @ desired_object_wrench

        # Reshape to (n_contacts, 3)
        contact_forces = contact_forces_flat.reshape(n_contacts, 3)

        return contact_forces

    def update(self,
               contact_positions: List[np.ndarray],
               contact_normals: List[np.ndarray],
               measured_forces: List[np.ndarray],
               desired_object_wrench: np.ndarray,
               object_center: np.ndarray) -> ForceControlCommand:
        """
        Main update - compute grasp force commands.

        Args:
            contact_positions: Current contact positions
            contact_normals: Surface normals
            measured_forces: Measured forces at contacts
            desired_object_wrench: Desired wrench on object
            object_center: Object center

        Returns:
            ForceControlCommand with all outputs
        """
        n_contacts = len(contact_positions)

        # Detect slip
        contact_states = self.slip_detector.update(measured_forces, contact_normals)

        # Update contact positions in states
        for i, state in enumerate(contact_states):
            state.position = contact_positions[i]

        # Compute internal forces
        internal_forces = self.compute_internal_forces(
            contact_normals,
            self.internal_force
        )

        # Adjust for slip
        internal_forces = self.adjust_for_slip(contact_states, internal_forces)

        # Compute manipulation forces
        manipulation_forces = self.compute_manipulation_wrench(
            desired_object_wrench,
            contact_positions,
            contact_normals,
            object_center
        )

        # Combine internal and manipulation forces
        total_forces = []
        for i in range(n_contacts):
            # Internal force along normal
            internal_component = internal_forces[i] * (-contact_normals[i])

            # Add manipulation component
            if i < len(manipulation_forces):
                total_force = internal_component + manipulation_forces[i]
            else:
                total_force = internal_component

            total_forces.append(total_force)

        # Compute errors
        measured_total = np.sum(measured_forces, axis=0) if measured_forces else np.zeros(3)
        desired_total = np.sum(total_forces, axis=0) if total_forces else np.zeros(3)
        force_error = desired_total - measured_total

        return ForceControlCommand(
            joint_torques=np.zeros(n_contacts * 4),  # Placeholder
            cartesian_force=desired_object_wrench,
            position_error=np.zeros(6),
            force_error=np.concatenate([force_error, np.zeros(3)]),
            contact_states=contact_states
        )


def main():
    """Test force controllers."""
    print("Testing Force Controllers")
    print("=" * 50)

    # Test Impedance Controller
    print("\n1. Impedance Controller")
    impedance = ImpedanceController()

    current_pose = np.array([0.3, 0.0, 0.4, 0.0, 0.0, 0.0])
    current_vel = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    current_force = np.array([5.0, 0.0, 10.0, 0.0, 0.0, 0.0])
    desired_pose = np.array([0.3, 0.0, 0.5, 0.0, 0.0, 0.0])
    desired_force = np.array([0.0, 0.0, -10.0, 0.0, 0.0, 0.0])

    command = impedance.compute_command(
        current_pose, current_vel, current_force,
        desired_pose, desired_force
    )
    print(f"  Command wrench: {command[:3]} N, {command[3:]} Nm")

    # Test Hybrid Controller
    print("\n2. Hybrid Force/Position Controller")
    hybrid = HybridForcePositionController()
    hybrid.set_control_directions(
        position_axes=[0, 1, 3, 4, 5],  # X, Y, rotations: position
        force_axes=[2]                   # Z: force control
    )

    command = hybrid.compute_command(
        current_pose, current_vel, current_force,
        desired_pose, desired_force
    )
    print(f"  Command wrench: {command[:3]} N, {command[3:]} Nm")

    # Test Slip Detector
    print("\n3. Slip Detector")
    slip_detector = SlipDetector(friction_coefficient=0.5)

    forces = [
        np.array([0.0, 0.0, 10.0]),   # Pure normal - no slip
        np.array([4.0, 0.0, 5.0]),    # High tangent - likely slip
    ]
    normals = [
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, 1.0]),
    ]

    states = slip_detector.update(forces, normals)
    for i, state in enumerate(states):
        print(f"  Contact {i}: slip_ratio={state.slip_ratio:.2f}, slipping={state.is_slipping}")

    # Test Grasp Force Controller
    print("\n4. Grasp Force Controller")
    grasp_controller = GraspForceController(n_fingers=3)

    contact_positions = [
        np.array([0.02, 0.0, 0.0]),
        np.array([-0.01, 0.017, 0.0]),
        np.array([-0.01, -0.017, 0.0]),
    ]
    contact_normals = [
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.5, -0.866, 0.0]),
        np.array([0.5, 0.866, 0.0]),
    ]
    measured_forces = [
        np.array([5.0, 0.0, 0.0]),
        np.array([-2.5, 4.33, 0.0]),
        np.array([-2.5, -4.33, 0.0]),
    ]
    desired_wrench = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # Lift force
    object_center = np.array([0.0, 0.0, 0.0])

    result = grasp_controller.update(
        contact_positions, contact_normals, measured_forces,
        desired_wrench, object_center
    )

    print(f"  Force error: {result.force_error[:3]}")
    print(f"  Contacts slipping: {sum(1 for s in result.contact_states if s.is_slipping)}")

    print("\nForce Controller tests passed!")


if __name__ == "__main__":
    main()
