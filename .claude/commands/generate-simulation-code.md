# Generate Simulation Code Skill

Generate simulation code for Gazebo, Unity, or Isaac Sim.

## Arguments
- `$ARGUMENTS` - Platform: gazebo|unity|isaac

## Templates

### Gazebo World (SDF)
```xml
<?xml version="1.0"?>
<sdf version="1.8">
  <world name="robot_world">
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <light type="directional" name="sun">
      <pose>0 0 10 0 0 0</pose>
    </light>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

### Unity ROS Bridge (C#)
```csharp
using UnityEngine;
using RosMessageTypes.Std;
using Unity.Robotics.ROSTCPConnector;

public class RosBridge : MonoBehaviour
{
    ROSConnection ros;
    public string topicName = "unity_topic";

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<StringMsg>(topicName);
    }

    void Update()
    {
        var msg = new StringMsg { data = "Hello from Unity" };
        ros.Publish(topicName, msg);
    }
}
```

### Isaac Sim Script (Python)
```python
from omni.isaac.core import World
from omni.isaac.core.robots import Robot

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

robot = world.scene.add(Robot(
    prim_path="/World/Robot",
    name="my_robot"
))

world.reset()
while True:
    world.step(render=True)
```

## Validation
- [ ] Syntax correct for platform
- [ ] Physics configured
- [ ] ROS integration included where applicable
