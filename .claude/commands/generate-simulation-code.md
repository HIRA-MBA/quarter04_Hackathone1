# Generate Simulation Code

Generate simulation code for Gazebo, Unity, or Isaac Sim with test scaffolding.

## Arguments
- `$ARGUMENTS` - `<platform> <component> [--with-tests]`

## Usage
```
/generate-simulation-code gazebo world warehouse_environment
/generate-simulation-code gazebo plugin sensor_plugin --with-tests
/generate-simulation-code unity ros-bridge camera_publisher
/generate-simulation-code isaac scene robot_manipulation
/generate-simulation-code isaac rl-env navigation --with-tests
```

## Platforms & Components

| Platform | Components | Description |
|----------|------------|-------------|
| `gazebo` | `world`, `model`, `plugin`, `launch` | Gazebo Sim (Ignition) |
| `unity` | `ros-bridge`, `sensor`, `controller`, `scene` | Unity with ROS-TCP |
| `isaac` | `scene`, `robot`, `rl-env`, `extension` | NVIDIA Isaac Sim |

---

## Gazebo Templates

### World File (SDF)
```xml
<?xml version="1.0"?>
<sdf version="1.8">
  <world name="{{world_name}}">
    <!-- Physics configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Lighting -->
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <!-- Ground plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>100</mu>
                <mu2>50</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Plugins -->
    <plugin filename="ignition-gazebo-physics-system" name="ignition::gazebo::systems::Physics"/>
    <plugin filename="ignition-gazebo-user-commands-system" name="ignition::gazebo::systems::UserCommands"/>
    <plugin filename="ignition-gazebo-scene-broadcaster-system" name="ignition::gazebo::systems::SceneBroadcaster"/>
    <plugin filename="ignition-gazebo-sensors-system" name="ignition::gazebo::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>

    <!-- Include robot model -->
    <include>
      <uri>model://{{robot_model}}</uri>
      <pose>0 0 0.1 0 0 0</pose>
    </include>

  </world>
</sdf>
```

### Robot Model (SDF)
```xml
<?xml version="1.0"?>
<sdf version="1.8">
  <model name="{{robot_name}}">
    <pose>0 0 0 0 0 0</pose>

    <!-- Base link -->
    <link name="base_link">
      <inertial>
        <mass>5.0</mass>
        <inertia>
          <ixx>0.1</ixx>
          <iyy>0.1</iyy>
          <izz>0.1</izz>
        </inertia>
      </inertial>
      <collision name="base_collision">
        <geometry>
          <box>
            <size>0.4 0.3 0.15</size>
          </box>
        </geometry>
      </collision>
      <visual name="base_visual">
        <geometry>
          <box>
            <size>0.4 0.3 0.15</size>
          </box>
        </geometry>
        <material>
          <ambient>0.2 0.2 0.8 1</ambient>
          <diffuse>0.2 0.2 0.8 1</diffuse>
        </material>
      </visual>

      <!-- Camera sensor -->
      <sensor name="camera" type="camera">
        <pose>0.2 0 0.1 0 0 0</pose>
        <camera>
          <horizontal_fov>1.047</horizontal_fov>
          <image>
            <width>640</width>
            <height>480</height>
            <format>R8G8B8</format>
          </image>
          <clip>
            <near>0.1</near>
            <far>100</far>
          </clip>
        </camera>
        <always_on>true</always_on>
        <update_rate>30</update_rate>
        <topic>/{{robot_name}}/camera</topic>
      </sensor>

      <!-- LiDAR sensor -->
      <sensor name="lidar" type="gpu_lidar">
        <pose>0 0 0.2 0 0 0</pose>
        <lidar>
          <scan>
            <horizontal>
              <samples>360</samples>
              <resolution>1</resolution>
              <min_angle>-3.14159</min_angle>
              <max_angle>3.14159</max_angle>
            </horizontal>
          </scan>
          <range>
            <min>0.1</min>
            <max>30</max>
            <resolution>0.01</resolution>
          </range>
        </lidar>
        <always_on>true</always_on>
        <update_rate>10</update_rate>
        <topic>/{{robot_name}}/scan</topic>
      </sensor>
    </link>

    <!-- Differential drive plugin -->
    <plugin filename="ignition-gazebo-diff-drive-system" name="ignition::gazebo::systems::DiffDrive">
      <left_joint>left_wheel_joint</left_joint>
      <right_joint>right_wheel_joint</right_joint>
      <wheel_separation>0.35</wheel_separation>
      <wheel_radius>0.05</wheel_radius>
      <odom_publish_frequency>50</odom_publish_frequency>
      <topic>/cmd_vel</topic>
      <odom_topic>/odom</odom_topic>
      <frame_id>odom</frame_id>
      <child_frame_id>base_link</child_frame_id>
    </plugin>

  </model>
</sdf>
```

### Gazebo Plugin (C++)
```cpp
/**
 * @file {{plugin_name}}.cpp
 * @brief Gazebo plugin for {{description}}
 */
#include <ignition/gazebo/System.hh>
#include <ignition/gazebo/Model.hh>
#include <ignition/gazebo/components/JointPosition.hh>
#include <ignition/plugin/Register.hh>
#include <ignition/transport/Node.hh>

namespace {{namespace}}
{

class {{PluginClass}} : public ignition::gazebo::System,
                        public ignition::gazebo::ISystemConfigure,
                        public ignition::gazebo::ISystemPreUpdate
{
public:
    {{PluginClass}}() = default;
    ~{{PluginClass}}() override = default;

    void Configure(
        const ignition::gazebo::Entity &_entity,
        const std::shared_ptr<const sdf::Element> &_sdf,
        ignition::gazebo::EntityComponentManager &_ecm,
        ignition::gazebo::EventManager &_eventMgr) override
    {
        // Get model
        this->model = ignition::gazebo::Model(_entity);

        // Read parameters from SDF
        if (_sdf->HasElement("topic"))
        {
            this->topicName = _sdf->Get<std::string>("topic");
        }

        // Set up transport
        this->node.Subscribe(this->topicName, &{{PluginClass}}::OnMessage, this);

        ignmsg << "{{PluginClass}} configured for " << this->model.Name(_ecm) << std::endl;
    }

    void PreUpdate(
        const ignition::gazebo::UpdateInfo &_info,
        ignition::gazebo::EntityComponentManager &_ecm) override
    {
        if (_info.paused)
            return;

        // Update logic here
    }

private:
    void OnMessage(const ignition::msgs::Twist &_msg)
    {
        // Handle incoming messages
    }

    ignition::gazebo::Model model;
    ignition::transport::Node node;
    std::string topicName{"/cmd_vel"};
};

}  // namespace {{namespace}}

IGNITION_ADD_PLUGIN(
    {{namespace}}::{{PluginClass}},
    ignition::gazebo::System,
    {{namespace}}::{{PluginClass}}::ISystemConfigure,
    {{namespace}}::{{PluginClass}}::ISystemPreUpdate)
```

### Gazebo Launch File (Python)
```python
#!/usr/bin/env python3
"""
{{launch_name}}_launch.py - Launch Gazebo simulation with ROS 2
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for Gazebo simulation."""

    # Package directories
    pkg_share = get_package_share_directory('{{package_name}}')
    gazebo_ros_pkg = get_package_share_directory('ros_gz_sim')

    # Paths
    world_path = os.path.join(pkg_share, 'worlds', '{{world_name}}.sdf')
    urdf_path = os.path.join(pkg_share, 'urdf', '{{robot_name}}.urdf')

    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world = LaunchConfiguration('world', default=world_path)

    # Declare arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation clock'
    )

    declare_world = DeclareLaunchArgument(
        'world',
        default_value=world_path,
        description='World file path'
    )

    # Gazebo server
    gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(gazebo_ros_pkg, 'launch', 'gz_sim.launch.py')
        ]),
        launch_arguments={
            'gz_args': ['-r -v 4 ', world],
        }.items()
    )

    # Spawn robot
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', '{{robot_name}}',
            '-file', urdf_path,
            '-x', '0', '-y', '0', '-z', '0.1'
        ],
        output='screen'
    )

    # ROS-Gazebo bridge
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry@ignition.msgs.Odometry',
            '/scan@sensor_msgs/msg/LaserScan@ignition.msgs.LaserScan',
            '/camera@sensor_msgs/msg/Image@ignition.msgs.Image',
        ],
        output='screen'
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_world,
        gazebo_server,
        spawn_robot,
        bridge,
    ])
```

---

## Unity Templates

### ROS Bridge (C#)
```csharp
/**
 * {{ClassName}}.cs - ROS 2 bridge for {{description}}
 *
 * Publishes: {{topic}} ({{msg_type}})
 * Subscribes: {{sub_topic}} ({{sub_msg_type}})
 */
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;
using RosMessageTypes.Geometry;
using RosMessageTypes.Sensor;

namespace {{Namespace}}
{
    public class {{ClassName}} : MonoBehaviour
    {
        [Header("ROS Connection")]
        [SerializeField] private string publishTopic = "{{topic}}";
        [SerializeField] private string subscribeTopic = "{{sub_topic}}";
        [SerializeField] private float publishRate = 30f;

        [Header("References")]
        [SerializeField] private Camera robotCamera;
        [SerializeField] private Transform robotBase;

        private ROSConnection ros;
        private float publishInterval;
        private float timeSincePublish;

        void Start()
        {
            // Initialize ROS connection
            ros = ROSConnection.GetOrCreateInstance();
            ros.RegisterPublisher<{{MsgType}}>(publishTopic);
            ros.Subscribe<{{SubMsgType}}>(subscribeTopic, OnMessageReceived);

            publishInterval = 1f / publishRate;
            timeSincePublish = 0f;

            Debug.Log($"{{ClassName}} initialized: pub={publishTopic}, sub={subscribeTopic}");
        }

        void Update()
        {
            timeSincePublish += Time.deltaTime;

            if (timeSincePublish >= publishInterval)
            {
                PublishMessage();
                timeSincePublish = 0f;
            }
        }

        private void PublishMessage()
        {
            var msg = new {{MsgType}}();
            // TODO: Populate message fields
            ros.Publish(publishTopic, msg);
        }

        private void OnMessageReceived({{SubMsgType}} msg)
        {
            // TODO: Handle incoming message
            Debug.Log($"Received message on {subscribeTopic}");
        }

        void OnDestroy()
        {
            // Cleanup
        }
    }
}
```

### Camera Publisher (C#)
```csharp
/**
 * CameraPublisher.cs - Publishes Unity camera images to ROS 2
 */
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Std;

namespace RobotSimulation
{
    [RequireComponent(typeof(Camera))]
    public class CameraPublisher : MonoBehaviour
    {
        [Header("ROS Settings")]
        [SerializeField] private string topicName = "/camera/image_raw";
        [SerializeField] private string frameId = "camera_link";
        [SerializeField] private float publishRate = 30f;

        [Header("Image Settings")]
        [SerializeField] private int imageWidth = 640;
        [SerializeField] private int imageHeight = 480;

        private ROSConnection ros;
        private Camera cam;
        private RenderTexture renderTexture;
        private Texture2D texture2D;
        private float publishInterval;
        private float timeSincePublish;
        private uint sequenceId;

        void Start()
        {
            ros = ROSConnection.GetOrCreateInstance();
            ros.RegisterPublisher<ImageMsg>(topicName);

            cam = GetComponent<Camera>();
            renderTexture = new RenderTexture(imageWidth, imageHeight, 24);
            texture2D = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
            cam.targetTexture = renderTexture;

            publishInterval = 1f / publishRate;
        }

        void Update()
        {
            timeSincePublish += Time.deltaTime;

            if (timeSincePublish >= publishInterval)
            {
                PublishImage();
                timeSincePublish = 0f;
            }
        }

        private void PublishImage()
        {
            RenderTexture.active = renderTexture;
            texture2D.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
            texture2D.Apply();
            RenderTexture.active = null;

            byte[] imageBytes = texture2D.GetRawTextureData();

            var msg = new ImageMsg
            {
                header = new HeaderMsg
                {
                    seq = sequenceId++,
                    stamp = new TimeMsg
                    {
                        sec = (int)Time.time,
                        nanosec = (uint)((Time.time % 1) * 1e9)
                    },
                    frame_id = frameId
                },
                height = (uint)imageHeight,
                width = (uint)imageWidth,
                encoding = "rgb8",
                is_bigendian = 0,
                step = (uint)(imageWidth * 3),
                data = imageBytes
            };

            ros.Publish(topicName, msg);
        }

        void OnDestroy()
        {
            if (renderTexture != null)
                renderTexture.Release();
        }
    }
}
```

### Robot Controller (C#)
```csharp
/**
 * DifferentialDriveController.cs - Controls robot via cmd_vel
 */
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;

namespace RobotSimulation
{
    public class DifferentialDriveController : MonoBehaviour
    {
        [Header("ROS Settings")]
        [SerializeField] private string cmdVelTopic = "/cmd_vel";
        [SerializeField] private string odomTopic = "/odom";

        [Header("Robot Parameters")]
        [SerializeField] private float wheelRadius = 0.05f;
        [SerializeField] private float wheelSeparation = 0.35f;
        [SerializeField] private float maxLinearVelocity = 1.0f;
        [SerializeField] private float maxAngularVelocity = 2.0f;

        [Header("Wheel References")]
        [SerializeField] private WheelCollider leftWheel;
        [SerializeField] private WheelCollider rightWheel;

        private ROSConnection ros;
        private float linearVelocity;
        private float angularVelocity;

        void Start()
        {
            ros = ROSConnection.GetOrCreateInstance();
            ros.Subscribe<TwistMsg>(cmdVelTopic, OnCmdVelReceived);

            Debug.Log($"DifferentialDriveController subscribed to {cmdVelTopic}");
        }

        void FixedUpdate()
        {
            ApplyVelocities();
        }

        private void OnCmdVelReceived(TwistMsg msg)
        {
            linearVelocity = Mathf.Clamp((float)msg.linear.x, -maxLinearVelocity, maxLinearVelocity);
            angularVelocity = Mathf.Clamp((float)msg.angular.z, -maxAngularVelocity, maxAngularVelocity);
        }

        private void ApplyVelocities()
        {
            // Differential drive kinematics
            float leftWheelVelocity = (linearVelocity - angularVelocity * wheelSeparation / 2) / wheelRadius;
            float rightWheelVelocity = (linearVelocity + angularVelocity * wheelSeparation / 2) / wheelRadius;

            // Convert to RPM for wheel colliders
            leftWheel.motorTorque = leftWheelVelocity * 10f;
            rightWheel.motorTorque = rightWheelVelocity * 10f;
        }
    }
}
```

---

## Isaac Sim Templates

### Scene Setup Script
```python
#!/usr/bin/env python3
"""
{{scene_name}}_scene.py - Isaac Sim scene setup for {{description}}

Usage:
    ./isaac_sim.sh --/ext/isaacsim.python.kit={{scene_name}}_scene.py
"""
from omni.isaac.kit import SimulationApp

# Initialize simulation
config = {
    "headless": False,
    "width": 1280,
    "height": 720,
}
simulation_app = SimulationApp(config)

from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
import numpy as np


def setup_scene():
    """Set up the simulation scene."""
    # Create world
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1/60.0,
        rendering_dt=1/60.0
    )

    # Add ground plane
    world.scene.add_default_ground_plane()

    # Add lighting
    from pxr import UsdLux
    stage = world.stage
    light = UsdLux.DistantLight.Define(stage, "/World/Sun")
    light.CreateIntensityAttr(3000)
    light.CreateAngleAttr(0.53)

    # Add robot
    assets_root = get_assets_root_path()
    robot_usd = f"{assets_root}/Isaac/Robots/{{robot_path}}/{{robot_name}}.usd"

    robot = world.scene.add(
        Robot(
            prim_path="/World/{{robot_name}}",
            name="{{robot_name}}",
            usd_path=robot_usd,
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0])
        )
    )

    # Add obstacles
    world.scene.add(
        DynamicCuboid(
            prim_path="/World/Obstacle1",
            name="obstacle_1",
            position=np.array([2.0, 0.0, 0.25]),
            size=np.array([0.5, 0.5, 0.5]),
            color=np.array([1.0, 0.0, 0.0])
        )
    )

    return world, robot


def main():
    """Main simulation loop."""
    world, robot = setup_scene()
    world.reset()

    print("Isaac Sim scene ready. Press Ctrl+C to exit.")

    while simulation_app.is_running():
        world.step(render=True)

        # Get robot state
        position, orientation = robot.get_world_pose()

        # TODO: Add control logic


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        simulation_app.close()
```

### RL Environment (OmniIsaacGymEnvs style)
```python
#!/usr/bin/env python3
"""
{{env_name}}_env.py - RL Environment for {{description}}

Compatible with OmniIsaacGymEnvs and stable-baselines3.
"""
from typing import Dict, Tuple, Optional
import numpy as np
import torch
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.articulations import ArticulationView


class {{EnvName}}Task(BaseTask):
    """
    RL Task for {{description}}.

    Observation space: {{obs_description}}
    Action space: {{action_description}}
    """

    def __init__(
        self,
        name: str = "{{env_name}}",
        offset: Optional[np.ndarray] = None
    ) -> None:
        """Initialize the task."""
        self._num_envs = 1
        self._env_spacing = 2.0

        # Reward scales
        self._success_reward = 10.0
        self._distance_reward_scale = 1.0
        self._action_penalty_scale = 0.01

        # Episode settings
        self._max_episode_length = 500

        super().__init__(name=name, offset=offset)

    def set_up_scene(self, scene) -> None:
        """Set up the simulation scene."""
        super().set_up_scene(scene)

        # Add ground plane
        scene.add_default_ground_plane()

        # Add robot
        # TODO: Add robot USD reference

        # Create articulation view for vectorized control
        self._robots = ArticulationView(
            prim_paths_expr="/World/envs/.*/{{robot_name}}",
            name="robot_view"
        )
        scene.add(self._robots)

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """
        Get current observations.

        Returns:
            Dictionary of observation tensors
        """
        # Get robot joint positions and velocities
        joint_positions = self._robots.get_joint_positions()
        joint_velocities = self._robots.get_joint_velocities()

        # Get end-effector pose
        ee_positions, ee_orientations = self._robots.get_ee_pose()

        # Stack observations
        obs = torch.cat([
            joint_positions,
            joint_velocities,
            ee_positions,
        ], dim=-1)

        return {"obs": obs}

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Apply actions before physics step.

        Args:
            actions: Action tensor from policy
        """
        # Clip actions
        actions = torch.clamp(actions, -1.0, 1.0)

        # Scale to joint velocity limits
        scaled_actions = actions * self._action_scale

        # Apply joint velocity targets
        self._robots.set_joint_velocity_targets(scaled_actions)

    def calculate_metrics(self) -> Dict[str, torch.Tensor]:
        """Calculate reward and done signals."""
        # Get current state
        ee_pos, _ = self._robots.get_ee_pose()
        target_pos = self._target_positions

        # Calculate distance to target
        distance = torch.norm(ee_pos - target_pos, dim=-1)

        # Reward components
        distance_reward = -self._distance_reward_scale * distance
        success_reward = torch.where(
            distance < 0.05,
            self._success_reward,
            torch.zeros_like(distance)
        )

        # Total reward
        reward = distance_reward + success_reward

        # Done signal
        done = (distance < 0.05) | (self._step_count >= self._max_episode_length)

        return {
            "reward": reward,
            "done": done,
            "distance": distance,
        }

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset specific environments."""
        num_resets = len(env_ids)

        # Reset robot to initial positions
        initial_positions = torch.zeros((num_resets, self._num_joints))
        self._robots.set_joint_positions(initial_positions, indices=env_ids)
        self._robots.set_joint_velocities(
            torch.zeros_like(initial_positions),
            indices=env_ids
        )

        # Randomize target positions
        self._target_positions[env_ids] = self._sample_target_positions(num_resets)

        # Reset step counter
        self._step_count[env_ids] = 0


class {{EnvName}}Env:
    """
    Gym-compatible wrapper for {{EnvName}}Task.

    Example:
        env = {{EnvName}}Env()
        obs = env.reset()
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
    """

    def __init__(self, headless: bool = False):
        """Initialize environment."""
        from omni.isaac.kit import SimulationApp
        self._simulation_app = SimulationApp({"headless": headless})

        from omni.isaac.core import World
        self._world = World()
        self._task = {{EnvName}}Task()
        self._world.add_task(self._task)
        self._world.reset()

    @property
    def observation_space(self):
        """Return observation space."""
        import gymnasium as gym
        return gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._obs_dim,), dtype=np.float32
        )

    @property
    def action_space(self):
        """Return action space."""
        import gymnasium as gym
        return gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self._action_dim,), dtype=np.float32
        )

    def reset(self) -> np.ndarray:
        """Reset environment."""
        self._world.reset()
        obs = self._task.get_observations()
        return obs["obs"].cpu().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Take environment step."""
        action_tensor = torch.from_numpy(action).unsqueeze(0)
        self._task.pre_physics_step(action_tensor)
        self._world.step(render=True)

        obs = self._task.get_observations()
        metrics = self._task.calculate_metrics()

        return (
            obs["obs"].cpu().numpy(),
            metrics["reward"].item(),
            metrics["done"].item(),
            {"distance": metrics["distance"].item()}
        )

    def close(self):
        """Close environment."""
        self._simulation_app.close()
```

### Isaac Sim Extension
```python
#!/usr/bin/env python3
"""
{{extension_name}}_extension.py - Isaac Sim extension for {{description}}
"""
import omni.ext
import omni.ui as ui
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage


class {{ExtensionName}}Extension(omni.ext.IExt):
    """
    Extension for {{description}}.
    """

    def on_startup(self, ext_id: str) -> None:
        """Called when extension starts."""
        print(f"[{{extension_name}}] Extension starting up")

        self._window = ui.Window("{{ExtensionName}}", width=300, height=200)
        with self._window.frame:
            with ui.VStack():
                ui.Label("{{ExtensionName}} Controls")
                ui.Spacer(height=10)

                ui.Button("Load Robot", clicked_fn=self._on_load_robot)
                ui.Button("Reset Scene", clicked_fn=self._on_reset_scene)
                ui.Button("Start Simulation", clicked_fn=self._on_start_sim)

    def on_shutdown(self) -> None:
        """Called when extension shuts down."""
        print(f"[{{extension_name}}] Extension shutting down")
        self._window = None

    def _on_load_robot(self) -> None:
        """Load robot into scene."""
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        assets_root = get_assets_root_path()
        robot_usd = f"{assets_root}/Isaac/Robots/{{robot_path}}/{{robot_name}}.usd"
        add_reference_to_stage(robot_usd, "/World/Robot")

    def _on_reset_scene(self) -> None:
        """Reset simulation scene."""
        world = World.instance()
        if world:
            world.reset()

    def _on_start_sim(self) -> None:
        """Start simulation."""
        world = World.instance()
        if world:
            world.play()
```

---

## Test Generation (`--with-tests`)

### Gazebo Test (pytest)
```python
#!/usr/bin/env python3
"""
test_{{world_name}}.py - Tests for Gazebo simulation
"""
import pytest
import subprocess
import time
import rclpy
from rclpy.node import Node


@pytest.fixture(scope="module")
def gazebo_process():
    """Start Gazebo simulation."""
    proc = subprocess.Popen([
        "ros2", "launch", "{{package_name}}", "{{launch_name}}_launch.py"
    ])
    time.sleep(10)  # Wait for Gazebo to start
    yield proc
    proc.terminate()
    proc.wait()


@pytest.fixture(scope="module")
def ros_context():
    """Initialize ROS context."""
    rclpy.init()
    yield
    rclpy.shutdown()


class TestGazeboSimulation:
    """Test Gazebo simulation."""

    def test_gazebo_running(self, gazebo_process):
        """Test that Gazebo process is running."""
        assert gazebo_process.poll() is None

    def test_topics_available(self, gazebo_process, ros_context):
        """Test that expected topics are available."""
        node = Node("test_node")
        topics = node.get_topic_names_and_types()
        topic_names = [t[0] for t in topics]

        assert "/cmd_vel" in topic_names
        assert "/odom" in topic_names
        node.destroy_node()
```

### Unity Test (NUnit)
```csharp
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using System.Collections;

namespace {{Namespace}}.Tests
{
    public class {{ClassName}}Tests
    {
        private GameObject robotObject;
        private {{ClassName}} component;

        [SetUp]
        public void Setup()
        {
            robotObject = new GameObject("TestRobot");
            component = robotObject.AddComponent<{{ClassName}}>();
        }

        [TearDown]
        public void Teardown()
        {
            Object.DestroyImmediate(robotObject);
        }

        [Test]
        public void Component_InitializesCorrectly()
        {
            Assert.IsNotNull(component);
        }

        [UnityTest]
        public IEnumerator Component_PublishesAtCorrectRate()
        {
            // Wait for a few publish cycles
            yield return new WaitForSeconds(0.5f);
            // Assert publish count
        }
    }
}
```

---

## Validation Checklist

- [ ] Physics configured (timestep, solver)
- [ ] Sensors publishing at correct rates
- [ ] ROS 2 bridge connected
- [ ] Collision meshes simplified
- [ ] Materials/textures optimized
- [ ] Tests pass
