#!/usr/bin/env python3
"""
Isaac Sim Robot Spawning Script

This script demonstrates how to create an Isaac Sim scene, spawn a robot,
configure sensors, and implement basic joint control.

Lab 8: Isaac Sim Fundamentals
"""

from omni.isaac.kit import SimulationApp

# Configuration for Isaac Sim
CONFIG = {
    "headless": False,  # Set to True for training/data generation
    "width": 1280,
    "height": 720,
    "sync_loads": True,
    "renderer": "RayTracedLighting",  # or "PathTracing" for higher quality
}

# Initialize simulation app (must be done before other imports)
simulation_app = SimulationApp(CONFIG)

# Now import Isaac Sim modules
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, VisualCuboid, GroundPlane
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.prims import XFormPrim, RigidPrim
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.sensor import Camera, IMUSensor
from omni.isaac.urdf import _urdf
import omni.kit.commands
from pxr import UsdPhysics, PhysxSchema, Gf
import numpy as np
import os


class IsaacSceneBuilder:
    """
    Builder class for creating Isaac Sim scenes with robots and sensors.

    This class encapsulates the common patterns for:
    - Creating and configuring worlds
    - Importing robots from URDF
    - Adding sensors (cameras, LIDAR, IMU)
    - Setting up physics
    """

    def __init__(self, stage_units: float = 1.0):
        """
        Initialize the scene builder.

        Args:
            stage_units: Scale factor for the stage (1.0 = meters)
        """
        self.world = World(stage_units_in_meters=stage_units)
        self.stage = self.world.stage
        self.robot = None
        self.sensors = {}

    def setup_physics(self,
                      timesteps_per_second: int = 500,
                      enable_gpu_dynamics: bool = True,
                      enable_ccd: bool = True):
        """
        Configure physics simulation parameters.

        Args:
            timesteps_per_second: Physics update rate (Hz)
            enable_gpu_dynamics: Use GPU for physics (requires RTX)
            enable_ccd: Enable continuous collision detection
        """
        # Create physics scene if not exists
        physics_scene_path = "/physicsScene"

        if not self.stage.GetPrimAtPath(physics_scene_path):
            physics_scene = UsdPhysics.Scene.Define(self.stage, physics_scene_path)
            physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
            physics_scene.CreateGravityMagnitudeAttr().Set(9.81)

        # Apply PhysX settings
        physx_scene = PhysxSchema.PhysxSceneAPI.Apply(
            self.stage.GetPrimAtPath(physics_scene_path)
        )
        physx_scene.CreateTimeStepsPerSecondAttr().Set(timesteps_per_second)
        physx_scene.CreateEnableCCDAttr().Set(enable_ccd)
        physx_scene.CreateEnableGPUDynamicsAttr().Set(enable_gpu_dynamics)

        if enable_gpu_dynamics:
            physx_scene.CreateBroadphaseTypeAttr().Set("GPU")
            physx_scene.CreateSolverTypeAttr().Set("TGS")

        print(f"Physics configured: {timesteps_per_second} Hz, GPU={enable_gpu_dynamics}")

    def add_ground_plane(self, size: float = 100.0):
        """Add a ground plane to the scene."""
        self.world.scene.add_default_ground_plane(
            z_position=0,
            name="ground_plane",
            prim_path="/World/GroundPlane",
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.1
        )

    def add_warehouse_environment(self):
        """
        Add a simple warehouse-like environment with walls and obstacles.
        """
        # Create walls
        wall_height = 3.0
        wall_thickness = 0.2
        room_size = 10.0

        walls = [
            ("wall_north", [0, room_size/2, wall_height/2], [room_size, wall_thickness, wall_height]),
            ("wall_south", [0, -room_size/2, wall_height/2], [room_size, wall_thickness, wall_height]),
            ("wall_east", [room_size/2, 0, wall_height/2], [wall_thickness, room_size, wall_height]),
            ("wall_west", [-room_size/2, 0, wall_height/2], [wall_thickness, room_size, wall_height]),
        ]

        for name, position, scale in walls:
            self.world.scene.add(
                VisualCuboid(
                    prim_path=f"/World/Walls/{name}",
                    name=name,
                    position=np.array(position),
                    scale=np.array(scale),
                    color=np.array([0.7, 0.7, 0.7])  # Gray walls
                )
            )

        # Add some obstacles (shelves/boxes)
        obstacles = [
            ("shelf_1", [3, 2, 0.5], [0.5, 2, 1]),
            ("shelf_2", [3, -2, 0.5], [0.5, 2, 1]),
            ("shelf_3", [-3, 2, 0.5], [0.5, 2, 1]),
            ("box_1", [0, 3, 0.25], [0.5, 0.5, 0.5]),
            ("box_2", [-2, -2, 0.25], [0.5, 0.5, 0.5]),
        ]

        for name, position, scale in obstacles:
            self.world.scene.add(
                VisualCuboid(
                    prim_path=f"/World/Obstacles/{name}",
                    name=name,
                    position=np.array(position),
                    scale=np.array(scale),
                    color=np.array([0.5, 0.3, 0.1])  # Brown
                )
            )

        print("Warehouse environment created")

    def import_robot_urdf(self,
                          urdf_path: str,
                          robot_name: str = "robot",
                          position: tuple = (0, 0, 0),
                          fix_base: bool = False):
        """
        Import a robot from URDF file.

        Args:
            urdf_path: Path to the URDF file
            robot_name: Name for the robot in the scene
            position: Initial position (x, y, z)
            fix_base: Whether to fix the robot base to world

        Returns:
            Articulation object representing the robot
        """
        # Configure URDF import
        urdf_config = _urdf.ImportConfig()
        urdf_config.merge_fixed_joints = False
        urdf_config.convex_decomp = True
        urdf_config.import_inertia_tensor = True
        urdf_config.fix_base = fix_base
        urdf_config.make_default_prim = True
        urdf_config.self_collision = False
        urdf_config.create_physics_scene = True
        urdf_config.default_drive_strength = 1000.0
        urdf_config.default_position_drive_damping = 100.0
        urdf_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION

        # Import URDF
        prim_path = f"/World/{robot_name}"
        result, imported_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=urdf_path,
            import_config=urdf_config,
            dest_path=prim_path
        )

        if not result:
            raise RuntimeError(f"Failed to import URDF from {urdf_path}")

        print(f"Robot imported at: {imported_path}")

        # Wrap as Articulation
        self.robot = self.world.scene.add(
            Articulation(
                prim_path=prim_path,
                name=robot_name,
                position=np.array(position)
            )
        )

        return self.robot

    def import_isaac_robot(self,
                           robot_type: str = "franka",
                           robot_name: str = "robot",
                           position: tuple = (0, 0, 0)):
        """
        Import a robot from Isaac Sim's built-in assets.

        Args:
            robot_type: Type of robot ("franka", "ur10", "carter", "jetbot")
            robot_name: Name for the robot in the scene
            position: Initial position (x, y, z)

        Returns:
            Articulation object representing the robot
        """
        assets_root = get_assets_root_path()

        robot_paths = {
            "franka": f"{assets_root}/Isaac/Robots/Franka/franka_alt_fingers.usd",
            "ur10": f"{assets_root}/Isaac/Robots/UniversalRobots/ur10/ur10.usd",
            "carter": f"{assets_root}/Isaac/Robots/Carter/carter_v1.usd",
            "jetbot": f"{assets_root}/Isaac/Robots/Jetbot/jetbot.usd",
        }

        if robot_type not in robot_paths:
            raise ValueError(f"Unknown robot type: {robot_type}. Available: {list(robot_paths.keys())}")

        prim_path = f"/World/{robot_name}"
        add_reference_to_stage(robot_paths[robot_type], prim_path)

        self.robot = self.world.scene.add(
            Articulation(
                prim_path=prim_path,
                name=robot_name,
                position=np.array(position)
            )
        )

        print(f"Loaded {robot_type} robot at {prim_path}")
        return self.robot

    def add_camera(self,
                   name: str,
                   position: tuple,
                   orientation: tuple = (0, 0, 0, 1),
                   parent_prim: str = None,
                   resolution: tuple = (640, 480),
                   frequency: int = 30):
        """
        Add a camera sensor to the scene.

        Args:
            name: Name for the camera
            position: Camera position (x, y, z)
            orientation: Camera orientation as quaternion (x, y, z, w)
            parent_prim: Optional parent prim path to attach camera to
            resolution: Image resolution (width, height)
            frequency: Capture frequency in Hz

        Returns:
            Camera sensor object
        """
        if parent_prim:
            prim_path = f"{parent_prim}/{name}"
        else:
            prim_path = f"/World/{name}"

        camera = Camera(
            prim_path=prim_path,
            name=name,
            position=np.array(position),
            orientation=np.array(orientation),
            frequency=frequency,
            resolution=resolution
        )

        self.world.scene.add(camera)
        self.sensors[name] = camera

        print(f"Camera '{name}' added at {prim_path}")
        return camera

    def add_imu(self,
                name: str,
                parent_prim: str,
                position: tuple = (0, 0, 0),
                frequency: int = 200):
        """
        Add an IMU sensor to the robot.

        Args:
            name: Name for the IMU
            parent_prim: Prim path to attach IMU to (e.g., robot base link)
            position: Local position offset
            frequency: Update frequency in Hz

        Returns:
            IMUSensor object
        """
        imu = IMUSensor(
            prim_path=f"{parent_prim}/{name}",
            name=name,
            translation=np.array(position),
            frequency=frequency
        )

        self.world.scene.add(imu)
        self.sensors[name] = imu

        print(f"IMU '{name}' added to {parent_prim}")
        return imu

    def reset(self):
        """Reset the world and all objects."""
        self.world.reset()

        # Initialize sensors
        for name, sensor in self.sensors.items():
            if hasattr(sensor, 'initialize'):
                sensor.initialize()

    def step(self, render: bool = True):
        """
        Step the simulation forward.

        Args:
            render: Whether to render the frame
        """
        self.world.step(render=render)

    def get_robot_state(self):
        """
        Get the current state of the robot.

        Returns:
            Dictionary with joint positions, velocities, and base pose
        """
        if self.robot is None:
            return None

        return {
            "joint_positions": self.robot.get_joint_positions(),
            "joint_velocities": self.robot.get_joint_velocities(),
            "base_position": self.robot.get_world_pose()[0],
            "base_orientation": self.robot.get_world_pose()[1],
        }

    def set_joint_positions(self, positions: np.ndarray):
        """Set target joint positions for the robot."""
        if self.robot is not None:
            self.robot.set_joint_position_targets(positions)

    def get_camera_data(self, camera_name: str):
        """
        Get data from a camera sensor.

        Args:
            camera_name: Name of the camera

        Returns:
            Dictionary with RGB, depth, and other data
        """
        if camera_name not in self.sensors:
            return None

        camera = self.sensors[camera_name]

        return {
            "rgb": camera.get_rgba()[:, :, :3],
            "depth": camera.get_depth(),
        }

    def get_imu_data(self, imu_name: str):
        """
        Get data from an IMU sensor.

        Args:
            imu_name: Name of the IMU

        Returns:
            Dictionary with acceleration, angular velocity, and orientation
        """
        if imu_name not in self.sensors:
            return None

        imu = self.sensors[imu_name]
        frame = imu.get_current_frame()

        return {
            "linear_acceleration": frame["lin_acc"],
            "angular_velocity": frame["ang_vel"],
            "orientation": frame["orientation"],
        }


def main():
    """Main function demonstrating Isaac Sim scene creation."""

    print("=" * 60)
    print("Lab 8: Isaac Sim Scene Creation")
    print("=" * 60)

    # Create scene builder
    builder = IsaacSceneBuilder(stage_units=1.0)

    # Setup physics
    builder.setup_physics(
        timesteps_per_second=500,
        enable_gpu_dynamics=True
    )

    # Add ground and environment
    builder.add_ground_plane()
    builder.add_warehouse_environment()

    # Import robot (using built-in Franka for demonstration)
    # In your lab, replace this with your custom URDF:
    # builder.import_robot_urdf("/path/to/humanoid.urdf", "humanoid")
    try:
        robot = builder.import_isaac_robot("franka", "robot", position=(0, 0, 0))
    except Exception as e:
        print(f"Could not load Isaac robot (may not have Nucleus access): {e}")
        print("Creating a simple test scene instead...")

        # Add test objects
        builder.world.scene.add(
            DynamicCuboid(
                prim_path="/World/TestCube",
                name="test_cube",
                position=np.array([0.5, 0, 0.5]),
                scale=np.array([0.1, 0.1, 0.1]),
                color=np.array([1, 0, 0])
            )
        )

    # Add camera
    camera = builder.add_camera(
        name="overhead_camera",
        position=(3, 0, 3),
        orientation=(0, 0.383, 0, 0.924),  # Looking down at 45 degrees
        resolution=(640, 480),
        frequency=30
    )

    # Reset world
    builder.reset()

    # Print robot info
    if builder.robot is not None:
        print(f"\nRobot DOFs: {builder.robot.num_dof}")
        print(f"Joint names: {builder.robot.dof_names}")

    # Control loop
    print("\nStarting simulation loop...")
    print("Press Ctrl+C to exit")

    step = 0
    try:
        while simulation_app.is_running():
            # Generate simple sinusoidal motion
            if builder.robot is not None:
                num_joints = builder.robot.num_dof
                targets = np.zeros(num_joints)
                targets[0] = 0.5 * np.sin(step * 0.01)  # Move first joint
                if num_joints > 3:
                    targets[3] = -1.5 + 0.3 * np.sin(step * 0.02)
                builder.set_joint_positions(targets)

            # Step simulation
            builder.step(render=True)

            # Print state periodically
            if step % 100 == 0:
                state = builder.get_robot_state()
                if state:
                    pos = state["joint_positions"]
                    print(f"Step {step}: Joint 0 position = {pos[0]:.3f}")

                # Get camera data
                cam_data = builder.get_camera_data("overhead_camera")
                if cam_data and cam_data["rgb"] is not None:
                    print(f"  Camera: {cam_data['rgb'].shape}")

            step += 1

    except KeyboardInterrupt:
        print("\nSimulation stopped by user")

    # Cleanup
    simulation_app.close()
    print("Done!")


if __name__ == "__main__":
    main()
