# Generate Diagram

Generate Mermaid diagrams for the Physical AI Textbook with robotics-specific templates.

## Arguments
- `$ARGUMENTS` - `<type> <description>`

## Usage
```
/generate-diagram architecture ROS 2 node communication
/generate-diagram flowchart sensor fusion pipeline
/generate-diagram sequence robot arm motion planning
/generate-diagram state navigation state machine
/generate-diagram ros2-graph camera processing pipeline
```

## Available Diagram Types

| Type | Use Case | Best For |
|------|----------|----------|
| `architecture` | System overview | High-level design, component relationships |
| `flowchart` | Process flow | Algorithms, decision trees, data pipelines |
| `sequence` | Interactions | Message passing, API calls, protocols |
| `state` | State machines | Lifecycle, behavior states, modes |
| `class` | OOP design | Class hierarchies, interfaces |
| `ros2-graph` | ROS 2 topology | Nodes, topics, services |
| `er` | Data models | Database schemas, relationships |
| `gantt` | Timelines | Project phases, timing diagrams |

---

## Templates

### Architecture Diagram
```mermaid
graph TB
    subgraph "Robot System Architecture"
        direction TB

        subgraph "Perception Layer"
            CAM[Camera Node]
            LIDAR[LiDAR Node]
            IMU[IMU Node]
        end

        subgraph "Processing Layer"
            FUSION[Sensor Fusion]
            SLAM[SLAM Node]
            DET[Object Detection]
        end

        subgraph "Planning Layer"
            NAV[Navigation]
            PATH[Path Planner]
            CTRL[Controller]
        end

        subgraph "Actuation Layer"
            MOTOR[Motor Driver]
            ARM[Arm Controller]
        end
    end

    CAM --> FUSION
    LIDAR --> FUSION
    IMU --> FUSION
    FUSION --> SLAM
    FUSION --> DET
    SLAM --> NAV
    DET --> NAV
    NAV --> PATH
    PATH --> CTRL
    CTRL --> MOTOR
    CTRL --> ARM

    style CAM fill:#e1f5fe
    style LIDAR fill:#e1f5fe
    style IMU fill:#e1f5fe
    style FUSION fill:#fff3e0
    style NAV fill:#e8f5e9
    style CTRL fill:#fce4ec
```

### ROS 2 Node Graph
```mermaid
graph LR
    subgraph "ROS 2 Computation Graph"
        direction LR

        %% Nodes
        CAM_NODE((camera_node))
        PROC_NODE((image_processor))
        DET_NODE((detector_node))
        NAV_NODE((nav2_controller))

        %% Topics
        RAW[/camera/raw]
        PROC[/camera/processed]
        DET[/detections]
        CMD[/cmd_vel]

        %% Connections
        CAM_NODE -->|pub| RAW
        RAW -->|sub| PROC_NODE
        PROC_NODE -->|pub| PROC
        PROC -->|sub| DET_NODE
        DET_NODE -->|pub| DET
        DET -->|sub| NAV_NODE
        NAV_NODE -->|pub| CMD
    end

    style CAM_NODE fill:#4fc3f7
    style PROC_NODE fill:#4fc3f7
    style DET_NODE fill:#4fc3f7
    style NAV_NODE fill:#4fc3f7
    style RAW fill:#ffcc80
    style PROC fill:#ffcc80
    style DET fill:#ffcc80
    style CMD fill:#ffcc80
```

### Flowchart - Algorithm
```mermaid
flowchart TD
    START([Start]) --> INIT[Initialize Sensors]
    INIT --> READ[Read Sensor Data]
    READ --> VALID{Data Valid?}

    VALID -->|No| ERROR[Log Error]
    ERROR --> READ

    VALID -->|Yes| PROCESS[Process Data]
    PROCESS --> FUSE[Sensor Fusion]
    FUSE --> DETECT{Object Detected?}

    DETECT -->|Yes| PLAN[Plan Path]
    DETECT -->|No| READ

    PLAN --> EXECUTE[Execute Motion]
    EXECUTE --> GOAL{Goal Reached?}

    GOAL -->|No| READ
    GOAL -->|Yes| FINISH([End])

    style START fill:#c8e6c9
    style FINISH fill:#c8e6c9
    style VALID fill:#fff9c4
    style DETECT fill:#fff9c4
    style GOAL fill:#fff9c4
    style ERROR fill:#ffcdd2
```

### Sequence Diagram - Message Passing
```mermaid
sequenceDiagram
    autonumber
    participant C as Camera Node
    participant P as Perception Node
    participant N as Navigation Node
    participant M as Motor Controller

    Note over C,M: Robot Navigation Sequence

    C->>P: Image (30 Hz)
    activate P
    P->>P: Object Detection
    P->>N: DetectionArray
    deactivate P

    activate N
    N->>N: Path Planning
    N->>M: Twist (cmd_vel)
    deactivate N

    activate M
    M->>M: PID Control
    M-->>N: Odometry Feedback
    deactivate M

    Note over C,M: Loop continues at 30 Hz
```

### State Machine - Lifecycle
```mermaid
stateDiagram-v2
    [*] --> Unconfigured: create

    Unconfigured --> Inactive: configure
    Unconfigured --> [*]: destroy

    Inactive --> Active: activate
    Inactive --> Unconfigured: cleanup
    Inactive --> [*]: destroy

    Active --> Inactive: deactivate
    Active --> [*]: destroy

    state Active {
        [*] --> Idle
        Idle --> Processing: data_received
        Processing --> Idle: complete
        Processing --> Error: fault
        Error --> Idle: recover
    }

    note right of Active
        Active state contains
        internal processing states
    end note
```

### State Machine - Robot Behavior
```mermaid
stateDiagram-v2
    [*] --> Idle

    Idle --> Searching: start_mission
    Searching --> Approaching: target_found
    Searching --> Idle: timeout

    Approaching --> Grasping: in_range
    Approaching --> Searching: target_lost

    Grasping --> Lifting: grasp_secure
    Grasping --> Approaching: grasp_failed

    Lifting --> Transporting: lifted
    Transporting --> Placing: at_destination

    Placing --> Idle: placed
    Placing --> Lifting: place_failed

    state Searching {
        [*] --> Scanning
        Scanning --> Rotating: scan_complete
        Rotating --> Scanning: rotation_complete
    }
```

### Class Diagram - ROS 2 Node
```mermaid
classDiagram
    class Node {
        <<abstract>>
        +get_name() str
        +get_logger() Logger
        +create_publisher()
        +create_subscription()
        +create_timer()
    }

    class LifecycleNode {
        +on_configure() CallbackReturn
        +on_activate() CallbackReturn
        +on_deactivate() CallbackReturn
        +on_cleanup() CallbackReturn
    }

    class SensorNode {
        -_publisher: Publisher
        -_timer: Timer
        -_rate: float
        +__init__()
        +read_sensor() SensorData
        +publish_data()
    }

    class PerceptionNode {
        -_subscription: Subscription
        -_publisher: Publisher
        -_model: MLModel
        +__init__()
        +process_image(Image) DetectionArray
    }

    class ControlNode {
        -_subscription: Subscription
        -_publisher: Publisher
        -_pid: PIDController
        +__init__()
        +compute_control(State) Command
    }

    Node <|-- LifecycleNode
    Node <|-- SensorNode
    Node <|-- PerceptionNode
    Node <|-- ControlNode
    LifecycleNode <|-- SensorNode
```

### Entity-Relationship Diagram
```mermaid
erDiagram
    USER ||--o{ SESSION : has
    USER ||--o{ PROGRESS : tracks
    USER ||--o{ PREFERENCE : configures

    SESSION ||--o{ CHAT_MESSAGE : contains

    CHAPTER ||--o{ PROGRESS : tracked_in
    CHAPTER ||--o{ LAB : has
    CHAPTER ||--o{ QUIZ : contains

    CHAT_MESSAGE ||--o{ RAG_CONTEXT : uses

    USER {
        uuid id PK
        string email
        string name
        timestamp created_at
    }

    CHAPTER {
        int id PK
        string title
        int module_id FK
        text content
    }

    PROGRESS {
        uuid id PK
        uuid user_id FK
        int chapter_id FK
        float completion
        timestamp updated_at
    }
```

### Gantt Chart - Project Timeline
```mermaid
gantt
    title Robot Arm Motion Planning
    dateFormat X
    axisFormat %L ms

    section Path Planning
    Inverse Kinematics    :a1, 0, 50
    Collision Check       :a2, after a1, 30
    Trajectory Generation :a3, after a2, 100

    section Execution
    Joint Interpolation   :b1, after a3, 200
    Motor Commands        :b2, after b1, 50

    section Feedback
    Position Feedback     :c1, after b2, 50
    Error Correction      :c2, after c1, 30
```

---

## Robotics-Specific Patterns

### Sensor Fusion Pipeline
```mermaid
graph LR
    subgraph "Sensor Fusion Architecture"
        CAM[Camera<br/>30 Hz] --> SYNC
        LIDAR[LiDAR<br/>10 Hz] --> SYNC
        IMU[IMU<br/>100 Hz] --> SYNC

        SYNC[Time<br/>Synchronizer] --> FUSE[Extended<br/>Kalman Filter]
        FUSE --> STATE[Robot State<br/>Estimate]
    end

    style SYNC fill:#ffeb3b
    style FUSE fill:#4caf50
```

### Control Loop
```mermaid
graph LR
    subgraph "PID Control Loop"
        REF[Reference] --> SUM((+))
        SUM --> PID[PID Controller]
        PID --> PLANT[Robot/Plant]
        PLANT --> SENSOR[Sensor]
        SENSOR --> FB[Feedback]
        FB --> SUM
    end

    style SUM fill:#ff9800
    style PID fill:#2196f3
```

### Sim2Real Pipeline
```mermaid
graph TB
    subgraph "Simulation"
        SIM[Isaac Sim] --> TRAIN[RL Training]
        TRAIN --> POLICY[Trained Policy]
    end

    subgraph "Domain Randomization"
        POLICY --> DR[Domain<br/>Randomization]
        DR --> ROBUST[Robust Policy]
    end

    subgraph "Real World"
        ROBUST --> DEPLOY[Deploy to Robot]
        DEPLOY --> TEST[Real-world Testing]
    end

    style SIM fill:#9c27b0
    style DEPLOY fill:#4caf50
```

---

## Guidelines

### General Rules
- ✅ Keep diagrams focused (max 15-20 nodes)
- ✅ Use clear, descriptive labels
- ✅ Group related components with subgraphs
- ✅ Use consistent color coding
- ✅ Add comments for complex diagrams

### Color Conventions
| Category | Color | Hex |
|----------|-------|-----|
| Input/Sensor | Light Blue | `#e1f5fe` |
| Processing | Light Orange | `#fff3e0` |
| Output/Actuation | Light Pink | `#fce4ec` |
| Decision | Light Yellow | `#fff9c4` |
| Error/Warning | Light Red | `#ffcdd2` |
| Success/Complete | Light Green | `#c8e6c9` |

### Accessibility
- Use high-contrast colors
- Include text labels (not just colors)
- Keep line crossings minimal
- Use directional arrows clearly
