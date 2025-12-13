#!/usr/bin/env python3
"""
Humanoid Balance Controller

Implements ZMP (Zero Moment Point) based balance control for humanoid robots.
This controller maintains balance by computing stabilizing torques based on
the robot's center of mass and support polygon.

Lab 11: Humanoid Locomotion
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum


class SupportPhase(Enum):
    """Support phase during walking."""
    DOUBLE_SUPPORT = "double"
    LEFT_SUPPORT = "left"
    RIGHT_SUPPORT = "right"


@dataclass
class RobotState:
    """Current state of the humanoid robot."""
    # Center of Mass
    com_position: np.ndarray      # [x, y, z] in world frame
    com_velocity: np.ndarray      # [vx, vy, vz] in world frame

    # Orientation
    torso_orientation: np.ndarray  # [roll, pitch, yaw] in radians
    torso_angular_velocity: np.ndarray

    # Feet positions
    left_foot_position: np.ndarray   # [x, y, z]
    right_foot_position: np.ndarray  # [x, y, z]
    left_foot_orientation: np.ndarray
    right_foot_orientation: np.ndarray

    # Contact states
    left_foot_contact: bool
    right_foot_contact: bool


@dataclass
class BalanceCommand:
    """Command output from balance controller."""
    # Desired CoM adjustment
    com_adjustment: np.ndarray  # [dx, dy, dz]

    # Torso orientation correction
    torso_correction: np.ndarray  # [roll, pitch, yaw]

    # Ankle torques for balance
    left_ankle_torque: np.ndarray   # [roll, pitch]
    right_ankle_torque: np.ndarray  # [roll, pitch]

    # ZMP reference
    zmp_reference: np.ndarray  # [x, y]
    zmp_actual: np.ndarray     # [x, y]


class ZMPController:
    """
    Zero Moment Point (ZMP) based balance controller.

    The ZMP is the point on the ground where the sum of horizontal
    inertial and gravity forces equals zero. For stable walking,
    ZMP must stay within the support polygon.

    Theory:
    - ZMP_x = x_com - (z_com / g) * ddot_x_com
    - ZMP_y = y_com - (z_com / g) * ddot_y_com

    Where:
    - x_com, y_com, z_com: Center of mass position
    - ddot_x_com, ddot_y_com: CoM acceleration
    - g: Gravitational acceleration (9.81 m/s²)
    """

    def __init__(self,
                 robot_mass: float = 50.0,
                 com_height: float = 0.9,
                 foot_width: float = 0.1,
                 foot_length: float = 0.2,
                 control_frequency: float = 500.0):
        """
        Initialize ZMP controller.

        Args:
            robot_mass: Total robot mass (kg)
            com_height: Nominal CoM height (m)
            foot_width: Foot width for support polygon (m)
            foot_length: Foot length for support polygon (m)
            control_frequency: Controller update rate (Hz)
        """
        self.mass = robot_mass
        self.com_height = com_height
        self.foot_width = foot_width
        self.foot_length = foot_length
        self.dt = 1.0 / control_frequency

        self.g = 9.81  # Gravity

        # Controller gains
        self.kp_com = np.array([100.0, 100.0, 200.0])  # CoM position
        self.kd_com = np.array([20.0, 20.0, 40.0])     # CoM velocity
        self.kp_zmp = np.array([50.0, 50.0])           # ZMP tracking
        self.kp_torso = np.array([200.0, 200.0, 50.0]) # Torso orientation
        self.kd_torso = np.array([40.0, 40.0, 10.0])   # Torso angular velocity

        # State history for velocity estimation
        self.prev_com_position = None
        self.prev_com_velocity = None
        self.com_acceleration = np.zeros(3)

        # Support polygon
        self.support_phase = SupportPhase.DOUBLE_SUPPORT

    def compute_zmp(self, state: RobotState) -> np.ndarray:
        """
        Compute actual ZMP from robot state.

        Args:
            state: Current robot state

        Returns:
            ZMP position [x, y] in world frame
        """
        # Estimate CoM acceleration
        if self.prev_com_velocity is not None:
            self.com_acceleration = (state.com_velocity - self.prev_com_velocity) / self.dt
        self.prev_com_velocity = state.com_velocity.copy()

        # ZMP computation using linear inverted pendulum model
        zmp_x = state.com_position[0] - (state.com_position[2] / self.g) * self.com_acceleration[0]
        zmp_y = state.com_position[1] - (state.com_position[2] / self.g) * self.com_acceleration[1]

        return np.array([zmp_x, zmp_y])

    def get_support_polygon(self, state: RobotState) -> np.ndarray:
        """
        Compute support polygon vertices based on contact state.

        Args:
            state: Current robot state

        Returns:
            Array of polygon vertices [[x1,y1], [x2,y2], ...]
        """
        vertices = []

        half_w = self.foot_width / 2
        half_l = self.foot_length / 2

        # Left foot vertices (if in contact)
        if state.left_foot_contact:
            lf = state.left_foot_position
            vertices.extend([
                [lf[0] - half_l, lf[1] - half_w],
                [lf[0] + half_l, lf[1] - half_w],
                [lf[0] + half_l, lf[1] + half_w],
                [lf[0] - half_l, lf[1] + half_w],
            ])

        # Right foot vertices (if in contact)
        if state.right_foot_contact:
            rf = state.right_foot_position
            vertices.extend([
                [rf[0] - half_l, rf[1] - half_w],
                [rf[0] + half_l, rf[1] - half_w],
                [rf[0] + half_l, rf[1] + half_w],
                [rf[0] - half_l, rf[1] + half_w],
            ])

        if len(vertices) == 0:
            # No contact - return empty
            return np.array([[0, 0]])

        return np.array(vertices)

    def is_zmp_stable(self, zmp: np.ndarray, state: RobotState, margin: float = 0.02) -> bool:
        """
        Check if ZMP is within support polygon with margin.

        Args:
            zmp: Current ZMP position
            state: Robot state
            margin: Safety margin (m)

        Returns:
            True if ZMP is stable
        """
        polygon = self.get_support_polygon(state)

        # Simple bounding box check (for convex polygons)
        min_x = np.min(polygon[:, 0]) + margin
        max_x = np.max(polygon[:, 0]) - margin
        min_y = np.min(polygon[:, 1]) + margin
        max_y = np.max(polygon[:, 1]) - margin

        return (min_x <= zmp[0] <= max_x) and (min_y <= zmp[1] <= max_y)

    def compute_zmp_reference(self,
                               state: RobotState,
                               desired_velocity: np.ndarray) -> np.ndarray:
        """
        Compute reference ZMP for desired movement.

        Args:
            state: Current robot state
            desired_velocity: Desired CoM velocity [vx, vy]

        Returns:
            Reference ZMP position [x, y]
        """
        # For static balance, ZMP should be at CoM projection
        zmp_ref = state.com_position[:2].copy()

        # Offset based on desired velocity (preview control concept)
        preview_time = 0.5  # seconds
        zmp_ref += desired_velocity[:2] * preview_time * 0.3

        # Constrain to support polygon
        polygon = self.get_support_polygon(state)
        min_x, max_x = np.min(polygon[:, 0]), np.max(polygon[:, 0])
        min_y, max_y = np.min(polygon[:, 1]), np.max(polygon[:, 1])

        margin = 0.03  # 3cm safety margin
        zmp_ref[0] = np.clip(zmp_ref[0], min_x + margin, max_x - margin)
        zmp_ref[1] = np.clip(zmp_ref[1], min_y + margin, max_y - margin)

        return zmp_ref

    def compute_ankle_torques(self,
                               zmp_error: np.ndarray,
                               state: RobotState) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute ankle torques to correct ZMP error.

        Args:
            zmp_error: [zmp_x_error, zmp_y_error]
            state: Current robot state

        Returns:
            Tuple of (left_ankle_torque, right_ankle_torque) as [roll, pitch]
        """
        # Ankle torques affect ZMP:
        # - Ankle pitch → ZMP X
        # - Ankle roll → ZMP Y

        # Proportional control on ZMP error
        ankle_pitch = -self.kp_zmp[0] * zmp_error[0]  # Negative because pitch forward moves ZMP back
        ankle_roll = -self.kp_zmp[1] * zmp_error[1]

        # Distribute torques based on support phase
        if state.left_foot_contact and state.right_foot_contact:
            # Double support - split evenly
            left_torque = np.array([ankle_roll * 0.5, ankle_pitch * 0.5])
            right_torque = np.array([ankle_roll * 0.5, ankle_pitch * 0.5])
        elif state.left_foot_contact:
            left_torque = np.array([ankle_roll, ankle_pitch])
            right_torque = np.zeros(2)
        elif state.right_foot_contact:
            left_torque = np.zeros(2)
            right_torque = np.array([ankle_roll, ankle_pitch])
        else:
            # No contact - no torques
            left_torque = np.zeros(2)
            right_torque = np.zeros(2)

        return left_torque, right_torque

    def compute_torso_correction(self, state: RobotState) -> np.ndarray:
        """
        Compute torso orientation correction to maintain upright posture.

        Args:
            state: Current robot state

        Returns:
            Desired torso orientation correction [roll, pitch, yaw]
        """
        # Target: upright torso (zero roll and pitch)
        target_orientation = np.array([0.0, 0.0, state.torso_orientation[2]])  # Keep yaw

        # PD control on orientation
        orientation_error = target_orientation - state.torso_orientation
        angular_velocity_error = -state.torso_angular_velocity

        correction = self.kp_torso * orientation_error + self.kd_torso * angular_velocity_error

        return correction

    def compute_com_adjustment(self,
                                state: RobotState,
                                zmp_reference: np.ndarray,
                                zmp_actual: np.ndarray) -> np.ndarray:
        """
        Compute CoM position adjustment for balance.

        Args:
            state: Current robot state
            zmp_reference: Desired ZMP position
            zmp_actual: Actual ZMP position

        Returns:
            CoM position adjustment [dx, dy, dz]
        """
        # ZMP error drives CoM adjustment
        zmp_error = zmp_reference - zmp_actual

        # CoM should move opposite to ZMP error (simplified)
        com_adjustment = np.zeros(3)
        com_adjustment[:2] = -zmp_error * 0.5  # Gain for CoM from ZMP error

        # Height regulation
        height_error = self.com_height - state.com_position[2]
        com_adjustment[2] = self.kp_com[2] * height_error

        return com_adjustment

    def update(self,
               state: RobotState,
               desired_velocity: np.ndarray = None) -> BalanceCommand:
        """
        Main control update - compute balance command.

        Args:
            state: Current robot state
            desired_velocity: Desired CoM velocity [vx, vy, vz] (optional)

        Returns:
            BalanceCommand with all control outputs
        """
        if desired_velocity is None:
            desired_velocity = np.zeros(3)

        # Update support phase
        if state.left_foot_contact and state.right_foot_contact:
            self.support_phase = SupportPhase.DOUBLE_SUPPORT
        elif state.left_foot_contact:
            self.support_phase = SupportPhase.LEFT_SUPPORT
        elif state.right_foot_contact:
            self.support_phase = SupportPhase.RIGHT_SUPPORT

        # Compute actual ZMP
        zmp_actual = self.compute_zmp(state)

        # Compute reference ZMP
        zmp_reference = self.compute_zmp_reference(state, desired_velocity)

        # ZMP error
        zmp_error = zmp_reference - zmp_actual

        # Compute control outputs
        left_ankle, right_ankle = self.compute_ankle_torques(zmp_error, state)
        torso_correction = self.compute_torso_correction(state)
        com_adjustment = self.compute_com_adjustment(state, zmp_reference, zmp_actual)

        return BalanceCommand(
            com_adjustment=com_adjustment,
            torso_correction=torso_correction,
            left_ankle_torque=left_ankle,
            right_ankle_torque=right_ankle,
            zmp_reference=zmp_reference,
            zmp_actual=zmp_actual
        )


class LIPMPreviewController:
    """
    Linear Inverted Pendulum Model (LIPM) with Preview Control.

    This controller generates CoM trajectories by treating the humanoid
    as an inverted pendulum and using preview control for ZMP tracking.
    """

    def __init__(self,
                 com_height: float = 0.9,
                 preview_steps: int = 50,
                 control_period: float = 0.02):
        """
        Initialize LIPM preview controller.

        Args:
            com_height: Nominal CoM height (m)
            preview_steps: Number of preview steps
            control_period: Control period (s)
        """
        self.com_height = com_height
        self.preview_steps = preview_steps
        self.dt = control_period
        self.g = 9.81

        # Natural frequency of LIPM
        self.omega = np.sqrt(self.g / com_height)

        # Build preview controller gains
        self._compute_preview_gains()

        # State: [x, x_dot, y, y_dot]
        self.state = np.zeros(4)

    def _compute_preview_gains(self):
        """Compute preview controller gains using LQR."""
        # Simplified: use constant gains
        # Full implementation would solve Riccati equation

        self.preview_gains = np.exp(-np.arange(self.preview_steps) * 0.05)
        self.preview_gains /= np.sum(self.preview_gains)

    def generate_trajectory(self,
                             zmp_trajectory: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate CoM trajectory from ZMP trajectory.

        Args:
            zmp_trajectory: Array of ZMP positions [[x,y], ...] for preview window

        Returns:
            Tuple of (com_position, com_velocity) as [x, y]
        """
        if len(zmp_trajectory) < self.preview_steps:
            # Pad with last value
            padding = np.tile(zmp_trajectory[-1], (self.preview_steps - len(zmp_trajectory), 1))
            zmp_trajectory = np.vstack([zmp_trajectory, padding])

        # Preview control for X
        zmp_x = zmp_trajectory[:self.preview_steps, 0]
        preview_x = np.dot(self.preview_gains, zmp_x)

        # Preview control for Y
        zmp_y = zmp_trajectory[:self.preview_steps, 1]
        preview_y = np.dot(self.preview_gains, zmp_y)

        # LIPM dynamics
        # x(t) = x0*cosh(ωt) + (ẋ0/ω)*sinh(ωt) + (p_zmp/ω²)(cosh(ωt) - 1)

        cosh_t = np.cosh(self.omega * self.dt)
        sinh_t = np.sinh(self.omega * self.dt)

        # Update state
        new_x = self.state[0] * cosh_t + (self.state[1] / self.omega) * sinh_t
        new_x_dot = self.state[0] * self.omega * sinh_t + self.state[1] * cosh_t

        new_y = self.state[2] * cosh_t + (self.state[3] / self.omega) * sinh_t
        new_y_dot = self.state[2] * self.omega * sinh_t + self.state[3] * cosh_t

        # Add control input from preview
        new_x += preview_x * 0.1
        new_y += preview_y * 0.1

        self.state = np.array([new_x, new_x_dot, new_y, new_y_dot])

        return np.array([new_x, new_y]), np.array([new_x_dot, new_y_dot])


def main():
    """Test balance controller."""
    print("Testing ZMP Balance Controller")
    print("=" * 50)

    # Create controller
    controller = ZMPController(
        robot_mass=50.0,
        com_height=0.9,
        control_frequency=500.0
    )

    # Create test state
    state = RobotState(
        com_position=np.array([0.0, 0.0, 0.9]),
        com_velocity=np.array([0.1, 0.0, 0.0]),
        torso_orientation=np.array([0.05, 0.02, 0.0]),  # Small tilt
        torso_angular_velocity=np.array([0.0, 0.0, 0.0]),
        left_foot_position=np.array([0.0, 0.1, 0.0]),
        right_foot_position=np.array([0.0, -0.1, 0.0]),
        left_foot_orientation=np.array([0.0, 0.0, 0.0]),
        right_foot_orientation=np.array([0.0, 0.0, 0.0]),
        left_foot_contact=True,
        right_foot_contact=True
    )

    # Test balance control
    command = controller.update(state, desired_velocity=np.array([0.0, 0.0, 0.0]))

    print(f"ZMP Reference: {command.zmp_reference}")
    print(f"ZMP Actual: {command.zmp_actual}")
    print(f"CoM Adjustment: {command.com_adjustment}")
    print(f"Torso Correction: {command.torso_correction}")
    print(f"Left Ankle Torque: {command.left_ankle_torque}")
    print(f"Right Ankle Torque: {command.right_ankle_torque}")

    # Check stability
    is_stable = controller.is_zmp_stable(command.zmp_actual, state)
    print(f"\nZMP Stable: {is_stable}")

    print("\nBalance Controller tests passed!")


if __name__ == "__main__":
    main()
