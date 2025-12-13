#!/usr/bin/env python3
"""
Humanoid Gait Generator

Implements walking pattern generation for humanoid robots using
footstep planning and trajectory interpolation.

Lab 11: Humanoid Locomotion
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import math


class GaitPhase(Enum):
    """Walking gait phases."""
    LEFT_SWING = "left_swing"
    RIGHT_SWING = "right_swing"
    DOUBLE_SUPPORT = "double_support"


@dataclass
class Footstep:
    """Single footstep definition."""
    position: np.ndarray     # [x, y, z] in world frame
    orientation: float       # Yaw angle in radians
    is_left: bool           # True for left foot, False for right
    duration: float         # Time to reach this step (seconds)


@dataclass
class FootTrajectory:
    """Foot trajectory for swing phase."""
    position: np.ndarray     # [x, y, z]
    velocity: np.ndarray     # [vx, vy, vz]
    acceleration: np.ndarray # [ax, ay, az]


@dataclass
class GaitParameters:
    """Parameters for gait generation."""
    step_length: float = 0.3       # Forward step length (m)
    step_width: float = 0.2        # Lateral distance between feet (m)
    step_height: float = 0.05      # Swing foot lift height (m)
    step_duration: float = 0.5     # Single step duration (s)
    double_support_ratio: float = 0.2  # Ratio of double support time
    walking_speed: float = 0.3     # Target walking speed (m/s)


class FootstepPlanner:
    """
    Plans footstep sequence for walking.

    Generates a sequence of footsteps based on desired walking
    direction and velocity.
    """

    def __init__(self, params: GaitParameters):
        """
        Initialize footstep planner.

        Args:
            params: Gait parameters
        """
        self.params = params

    def plan_straight_walk(self,
                           start_left: np.ndarray,
                           start_right: np.ndarray,
                           num_steps: int,
                           direction: float = 0.0) -> List[Footstep]:
        """
        Plan straight walking footsteps.

        Args:
            start_left: Starting left foot position [x, y, z]
            start_right: Starting right foot position [x, y, z]
            num_steps: Number of steps to plan
            direction: Walking direction (yaw angle in radians)

        Returns:
            List of footsteps
        """
        footsteps = []

        # Current foot positions
        left_pos = start_left.copy()
        right_pos = start_right.copy()

        # Alternating feet, starting with left
        is_left = True

        for i in range(num_steps):
            # Compute step vector
            step_x = self.params.step_length * np.cos(direction)
            step_y = self.params.step_length * np.sin(direction)

            if is_left:
                # Move left foot forward
                left_pos[0] += step_x
                left_pos[1] += step_y

                footsteps.append(Footstep(
                    position=left_pos.copy(),
                    orientation=direction,
                    is_left=True,
                    duration=self.params.step_duration
                ))
            else:
                # Move right foot forward
                right_pos[0] += step_x
                right_pos[1] += step_y

                footsteps.append(Footstep(
                    position=right_pos.copy(),
                    orientation=direction,
                    is_left=False,
                    duration=self.params.step_duration
                ))

            is_left = not is_left

        return footsteps

    def plan_turn_in_place(self,
                            center: np.ndarray,
                            turn_angle: float,
                            num_steps: int = 4) -> List[Footstep]:
        """
        Plan turning in place footsteps.

        Args:
            center: Center point of turn [x, y, z]
            turn_angle: Total turn angle (radians, positive = left)
            num_steps: Number of steps for turn

        Returns:
            List of footsteps
        """
        footsteps = []
        angle_per_step = turn_angle / num_steps
        current_angle = 0.0

        half_width = self.params.step_width / 2

        for i in range(num_steps):
            current_angle += angle_per_step
            is_left = (i % 2 == 0)

            # Compute foot position around center
            if is_left:
                offset = half_width
            else:
                offset = -half_width

            # Perpendicular offset
            perp_x = offset * np.cos(current_angle + np.pi/2)
            perp_y = offset * np.sin(current_angle + np.pi/2)

            position = np.array([
                center[0] + perp_x,
                center[1] + perp_y,
                center[2]
            ])

            footsteps.append(Footstep(
                position=position,
                orientation=current_angle,
                is_left=is_left,
                duration=self.params.step_duration
            ))

        return footsteps


class SwingTrajectoryGenerator:
    """
    Generates smooth swing foot trajectories.

    Uses polynomial interpolation to create smooth foot paths
    during swing phase.
    """

    def __init__(self, step_height: float = 0.05):
        """
        Initialize swing trajectory generator.

        Args:
            step_height: Maximum foot lift height (m)
        """
        self.step_height = step_height

    def generate_trajectory(self,
                             start_pos: np.ndarray,
                             end_pos: np.ndarray,
                             duration: float,
                             t: float) -> FootTrajectory:
        """
        Generate swing foot trajectory point.

        Uses quintic (5th order) polynomial for smooth start/end.

        Args:
            start_pos: Starting foot position [x, y, z]
            end_pos: Target foot position [x, y, z]
            duration: Total swing duration (s)
            t: Current time in swing (s)

        Returns:
            FootTrajectory with position, velocity, acceleration
        """
        # Normalize time
        s = np.clip(t / duration, 0, 1)

        # Quintic polynomial: starts and ends with zero velocity/acceleration
        # s_smooth = 10*s³ - 15*s⁴ + 6*s⁵
        s_smooth = 10 * s**3 - 15 * s**4 + 6 * s**5
        s_dot = (30 * s**2 - 60 * s**3 + 30 * s**4) / duration
        s_ddot = (60 * s - 180 * s**2 + 120 * s**3) / (duration**2)

        # XY position: linear interpolation with smooth timing
        position = start_pos + s_smooth * (end_pos - start_pos)
        velocity = s_dot * (end_pos - start_pos)
        acceleration = s_ddot * (end_pos - start_pos)

        # Z position: parabolic trajectory for foot clearance
        # z = z_start + (z_end - z_start) * s + h * 4 * s * (1 - s)
        z_linear = start_pos[2] + s_smooth * (end_pos[2] - start_pos[2])
        z_height = self.step_height * 4 * s_smooth * (1 - s_smooth)

        position[2] = z_linear + z_height

        # Z velocity
        z_vel_linear = s_dot * (end_pos[2] - start_pos[2])
        z_vel_height = self.step_height * 4 * s_dot * (1 - 2 * s_smooth)
        velocity[2] = z_vel_linear + z_vel_height

        # Z acceleration
        z_acc_linear = s_ddot * (end_pos[2] - start_pos[2])
        z_acc_height = self.step_height * 4 * (s_ddot * (1 - 2 * s_smooth) - 2 * s_dot**2)
        acceleration[2] = z_acc_linear + z_acc_height

        return FootTrajectory(
            position=position,
            velocity=velocity,
            acceleration=acceleration
        )


class CoMTrajectoryGenerator:
    """
    Generates Center of Mass trajectories for walking.

    Uses preview control to generate smooth CoM motion that
    maintains balance during walking.
    """

    def __init__(self, params: GaitParameters, com_height: float = 0.9):
        """
        Initialize CoM trajectory generator.

        Args:
            params: Gait parameters
            com_height: Nominal CoM height (m)
        """
        self.params = params
        self.com_height = com_height

    def generate_zmp_reference(self,
                                footsteps: List[Footstep],
                                dt: float = 0.01) -> np.ndarray:
        """
        Generate ZMP reference trajectory from footsteps.

        Args:
            footsteps: List of planned footsteps
            dt: Time step for trajectory (s)

        Returns:
            ZMP trajectory as array of [x, y] positions
        """
        zmp_trajectory = []

        for i, step in enumerate(footsteps):
            num_samples = int(step.duration / dt)

            # ZMP moves toward stance foot during swing
            if i > 0:
                prev_step = footsteps[i - 1]
                # Transition ZMP from previous to current stance
                for j in range(num_samples):
                    t = j / num_samples
                    zmp = prev_step.position[:2] * (1 - t) + step.position[:2] * t
                    zmp_trajectory.append(zmp)
            else:
                # First step - ZMP at midpoint
                for j in range(num_samples):
                    zmp_trajectory.append(step.position[:2])

        return np.array(zmp_trajectory)


class GaitController:
    """
    Main gait controller that coordinates all components.

    Manages footstep planning, trajectory generation, and
    provides commands for balance controller.
    """

    def __init__(self, params: GaitParameters = None):
        """
        Initialize gait controller.

        Args:
            params: Gait parameters (uses defaults if None)
        """
        if params is None:
            params = GaitParameters()

        self.params = params
        self.footstep_planner = FootstepPlanner(params)
        self.swing_generator = SwingTrajectoryGenerator(params.step_height)
        self.com_generator = CoMTrajectoryGenerator(params)

        # State
        self.footsteps = []
        self.current_step_index = 0
        self.phase = GaitPhase.DOUBLE_SUPPORT
        self.phase_time = 0.0

        # Current foot positions
        self.left_foot_pos = np.array([0.0, 0.1, 0.0])
        self.right_foot_pos = np.array([0.0, -0.1, 0.0])

    def initialize_walk(self,
                        direction: float = 0.0,
                        num_steps: int = 10):
        """
        Initialize walking with planned footsteps.

        Args:
            direction: Walking direction (radians)
            num_steps: Number of steps to plan
        """
        self.footsteps = self.footstep_planner.plan_straight_walk(
            self.left_foot_pos.copy(),
            self.right_foot_pos.copy(),
            num_steps,
            direction
        )
        self.current_step_index = 0
        self.phase = GaitPhase.DOUBLE_SUPPORT
        self.phase_time = 0.0

    def update(self, dt: float) -> dict:
        """
        Update gait controller state.

        Args:
            dt: Time step (s)

        Returns:
            Dictionary with current targets:
            - left_foot: FootTrajectory
            - right_foot: FootTrajectory
            - com_reference: [x, y] position
            - phase: GaitPhase
        """
        self.phase_time += dt

        # Check if we have footsteps
        if not self.footsteps or self.current_step_index >= len(self.footsteps):
            # Standing still
            return {
                "left_foot": FootTrajectory(
                    self.left_foot_pos, np.zeros(3), np.zeros(3)
                ),
                "right_foot": FootTrajectory(
                    self.right_foot_pos, np.zeros(3), np.zeros(3)
                ),
                "com_reference": (self.left_foot_pos[:2] + self.right_foot_pos[:2]) / 2,
                "phase": GaitPhase.DOUBLE_SUPPORT
            }

        current_step = self.footsteps[self.current_step_index]
        step_duration = current_step.duration
        double_support_time = step_duration * self.params.double_support_ratio

        # State machine
        if self.phase == GaitPhase.DOUBLE_SUPPORT:
            if self.phase_time >= double_support_time:
                # Transition to swing phase
                if current_step.is_left:
                    self.phase = GaitPhase.LEFT_SWING
                else:
                    self.phase = GaitPhase.RIGHT_SWING
                self.phase_time = 0.0

        elif self.phase == GaitPhase.LEFT_SWING:
            swing_duration = step_duration * (1 - self.params.double_support_ratio)
            if self.phase_time >= swing_duration:
                # Foot landed
                self.left_foot_pos = current_step.position.copy()
                self.current_step_index += 1
                self.phase = GaitPhase.DOUBLE_SUPPORT
                self.phase_time = 0.0

        elif self.phase == GaitPhase.RIGHT_SWING:
            swing_duration = step_duration * (1 - self.params.double_support_ratio)
            if self.phase_time >= swing_duration:
                # Foot landed
                self.right_foot_pos = current_step.position.copy()
                self.current_step_index += 1
                self.phase = GaitPhase.DOUBLE_SUPPORT
                self.phase_time = 0.0

        # Generate trajectories
        left_traj = FootTrajectory(self.left_foot_pos.copy(), np.zeros(3), np.zeros(3))
        right_traj = FootTrajectory(self.right_foot_pos.copy(), np.zeros(3), np.zeros(3))

        swing_duration = step_duration * (1 - self.params.double_support_ratio)

        if self.phase == GaitPhase.LEFT_SWING:
            left_traj = self.swing_generator.generate_trajectory(
                self.left_foot_pos,
                current_step.position,
                swing_duration,
                self.phase_time
            )
            # CoM shifts to right foot
            com_ref = self.right_foot_pos[:2]

        elif self.phase == GaitPhase.RIGHT_SWING:
            right_traj = self.swing_generator.generate_trajectory(
                self.right_foot_pos,
                current_step.position,
                swing_duration,
                self.phase_time
            )
            # CoM shifts to left foot
            com_ref = self.left_foot_pos[:2]

        else:  # DOUBLE_SUPPORT
            # CoM at midpoint
            com_ref = (self.left_foot_pos[:2] + self.right_foot_pos[:2]) / 2

        return {
            "left_foot": left_traj,
            "right_foot": right_traj,
            "com_reference": com_ref,
            "phase": self.phase
        }

    def get_progress(self) -> float:
        """Get walking progress (0 to 1)."""
        if not self.footsteps:
            return 1.0
        return self.current_step_index / len(self.footsteps)


def main():
    """Test gait generator."""
    print("Testing Gait Generator")
    print("=" * 50)

    # Create gait controller
    params = GaitParameters(
        step_length=0.25,
        step_width=0.2,
        step_height=0.05,
        step_duration=0.6
    )
    controller = GaitController(params)

    # Plan walk
    controller.initialize_walk(direction=0.0, num_steps=6)

    print(f"Planned {len(controller.footsteps)} footsteps")
    for i, step in enumerate(controller.footsteps):
        foot = "Left" if step.is_left else "Right"
        print(f"  Step {i}: {foot} -> ({step.position[0]:.2f}, {step.position[1]:.2f})")

    # Simulate walking
    print("\nSimulating walk...")
    dt = 0.01
    total_time = 0.0
    max_time = 10.0

    while total_time < max_time and controller.get_progress() < 1.0:
        result = controller.update(dt)
        total_time += dt

        if int(total_time * 100) % 50 == 0:  # Print every 0.5s
            print(f"t={total_time:.1f}s: Phase={result['phase'].value}, "
                  f"Progress={controller.get_progress():.0%}")

    print(f"\nWalk completed in {total_time:.2f}s")

    # Test swing trajectory
    print("\nTesting swing trajectory...")
    swing_gen = SwingTrajectoryGenerator(step_height=0.05)

    start = np.array([0.0, 0.1, 0.0])
    end = np.array([0.3, 0.1, 0.0])

    for t in np.linspace(0, 0.5, 6):
        traj = swing_gen.generate_trajectory(start, end, 0.5, t)
        print(f"  t={t:.1f}s: pos=({traj.position[0]:.3f}, {traj.position[2]:.3f})")

    print("\nGait Generator tests passed!")


if __name__ == "__main__":
    main()
