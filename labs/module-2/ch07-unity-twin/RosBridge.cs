// RosBridge.cs - Unity ROS 2 Connector for Digital Twin
// This script provides the core ROS 2 communication functionality for Unity

using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Nav;
using RosMessageTypes.Geometry;
using RosMessageTypes.Std;
using System.Collections.Generic;

namespace PhysicalAI.DigitalTwin
{
    /// <summary>
    /// Main ROS 2 bridge component for synchronizing Unity digital twin with ROS 2 robot.
    /// Handles bidirectional communication for state sync, sensor data, and commands.
    /// </summary>
    public class RosBridge : MonoBehaviour
    {
        #region Configuration

        [Header("ROS Connection Settings")]
        [Tooltip("IP address of the ROS 2 machine running ros_tcp_endpoint")]
        public string rosIP = "127.0.0.1";

        [Tooltip("Port for ROS TCP endpoint (default: 10000)")]
        public int rosPort = 10000;

        [Tooltip("Automatically connect on Start")]
        public bool autoConnect = true;

        [Header("Robot Configuration")]
        [Tooltip("Root transform of the robot model")]
        public Transform robotBase;

        [Tooltip("All articulation bodies representing robot joints")]
        public ArticulationBody[] joints;

        [Header("Topic Names")]
        public string jointStateTopic = "/joint_states";
        public string odometryTopic = "/odom";
        public string cmdVelTopic = "/cmd_vel";
        public string cameraTopic = "/unity/camera/image_raw";

        [Header("Synchronization Settings")]
        [Tooltip("Interpolation speed for position updates")]
        public float positionLerpSpeed = 10f;

        [Tooltip("Interpolation speed for rotation updates")]
        public float rotationLerpSpeed = 10f;

        [Tooltip("Interpolation speed for joint position updates")]
        public float jointLerpSpeed = 20f;

        [Header("Camera Publishing")]
        [Tooltip("Camera to publish images from")]
        public Camera publishCamera;

        [Tooltip("Image width for published camera feed")]
        public int imageWidth = 640;

        [Tooltip("Image height for published camera feed")]
        public int imageHeight = 480;

        [Tooltip("Camera publish rate in Hz")]
        public float cameraPublishRate = 30f;

        #endregion

        #region State

        private ROSConnection ros;
        private Dictionary<string, ArticulationBody> jointDict;

        // Target states from ROS
        private Vector3 targetPosition;
        private Quaternion targetRotation;
        private Dictionary<string, float> targetJointPositions;

        // Connection status
        private bool isConnected = false;
        private float lastMessageTime;

        // Camera publishing
        private RenderTexture renderTexture;
        private Texture2D texture2D;
        private float cameraPublishInterval;
        private float timeSinceLastPublish;

        #endregion

        #region Events

        public System.Action<bool> OnConnectionStatusChanged;
        public System.Action<Vector3, Quaternion> OnOdometryReceived;
        public System.Action<Dictionary<string, float>> OnJointStatesReceived;

        #endregion

        #region Unity Lifecycle

        void Awake()
        {
            // Initialize dictionaries
            jointDict = new Dictionary<string, ArticulationBody>();
            targetJointPositions = new Dictionary<string, float>();

            // Build joint dictionary
            if (joints != null)
            {
                foreach (var joint in joints)
                {
                    if (joint != null)
                    {
                        jointDict[joint.name] = joint;
                    }
                }
            }

            // Initialize target states
            if (robotBase != null)
            {
                targetPosition = robotBase.position;
                targetRotation = robotBase.rotation;
            }
        }

        void Start()
        {
            if (autoConnect)
            {
                Connect();
            }

            // Setup camera publishing
            if (publishCamera != null)
            {
                SetupCameraPublishing();
            }
        }

        void Update()
        {
            // Check connection status
            bool currentlyConnected = ros != null && ros.HasConnectionThread;
            if (currentlyConnected != isConnected)
            {
                isConnected = currentlyConnected;
                OnConnectionStatusChanged?.Invoke(isConnected);
            }

            // Check for stale data (no messages in 1 second)
            if (Time.time - lastMessageTime > 1f)
            {
                // Connection may be stale
            }

            // Interpolate robot state
            UpdateRobotState();

            // Publish camera if enabled
            if (publishCamera != null && isConnected)
            {
                UpdateCameraPublishing();
            }
        }

        void OnDestroy()
        {
            Cleanup();
        }

        #endregion

        #region Connection Management

        /// <summary>
        /// Connect to the ROS 2 TCP endpoint
        /// </summary>
        public void Connect()
        {
            // Get or create ROS connection
            ros = ROSConnection.GetOrCreateInstance();
            ros.RosIPAddress = rosIP;
            ros.RosPort = rosPort;

            // Subscribe to topics
            ros.Subscribe<JointStateMsg>(jointStateTopic, OnJointStateReceived);
            ros.Subscribe<OdometryMsg>(odometryTopic, OnOdometryReceived);

            // Register publishers
            ros.RegisterPublisher<TwistMsg>(cmdVelTopic);

            if (publishCamera != null)
            {
                ros.RegisterPublisher<ImageMsg>(cameraTopic);
            }

            Debug.Log($"[RosBridge] Connecting to ROS at {rosIP}:{rosPort}");
        }

        /// <summary>
        /// Disconnect from ROS
        /// </summary>
        public void Disconnect()
        {
            if (ros != null)
            {
                ros.Disconnect();
            }
        }

        private void Cleanup()
        {
            if (renderTexture != null)
            {
                Destroy(renderTexture);
            }
            if (texture2D != null)
            {
                Destroy(texture2D);
            }
        }

        #endregion

        #region Message Handlers

        private void OnJointStateReceived(JointStateMsg msg)
        {
            lastMessageTime = Time.time;

            for (int i = 0; i < msg.name.Length; i++)
            {
                string jointName = msg.name[i];
                float position = (float)msg.position[i];
                targetJointPositions[jointName] = position;
            }

            OnJointStatesReceived?.Invoke(targetJointPositions);
        }

        private void OnOdometryReceived(OdometryMsg msg)
        {
            lastMessageTime = Time.time;

            // Convert from ROS coordinate frame (FLU) to Unity coordinate frame (LUF)
            // ROS: X-forward, Y-left, Z-up
            // Unity: X-right, Y-up, Z-forward
            targetPosition = new Vector3(
                -(float)msg.pose.pose.position.y,  // ROS Y -> Unity -X
                (float)msg.pose.pose.position.z,   // ROS Z -> Unity Y
                (float)msg.pose.pose.position.x    // ROS X -> Unity Z
            );

            // Convert quaternion from ROS to Unity
            targetRotation = new Quaternion(
                (float)msg.pose.pose.orientation.y,   // ROS Y -> Unity X
                -(float)msg.pose.pose.orientation.z,  // ROS Z -> Unity -Y
                -(float)msg.pose.pose.orientation.x,  // ROS X -> Unity -Z
                (float)msg.pose.pose.orientation.w
            );

            OnOdometryReceived?.Invoke(targetPosition, targetRotation);
        }

        #endregion

        #region State Updates

        private void UpdateRobotState()
        {
            if (robotBase == null) return;

            // Interpolate base position and rotation
            robotBase.position = Vector3.Lerp(
                robotBase.position,
                targetPosition,
                Time.deltaTime * positionLerpSpeed
            );

            robotBase.rotation = Quaternion.Slerp(
                robotBase.rotation,
                targetRotation,
                Time.deltaTime * rotationLerpSpeed
            );

            // Interpolate joint positions
            foreach (var kvp in targetJointPositions)
            {
                if (jointDict.TryGetValue(kvp.Key, out var joint))
                {
                    var drive = joint.xDrive;
                    float currentTarget = drive.target;
                    float newTarget = kvp.Value * Mathf.Rad2Deg;

                    drive.target = Mathf.Lerp(
                        currentTarget,
                        newTarget,
                        Time.deltaTime * jointLerpSpeed
                    );
                    joint.xDrive = drive;
                }
            }
        }

        #endregion

        #region Publishing

        private void SetupCameraPublishing()
        {
            renderTexture = new RenderTexture(imageWidth, imageHeight, 24);
            texture2D = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
            cameraPublishInterval = 1f / cameraPublishRate;
        }

        private void UpdateCameraPublishing()
        {
            timeSinceLastPublish += Time.deltaTime;

            if (timeSinceLastPublish >= cameraPublishInterval)
            {
                PublishCameraImage();
                timeSinceLastPublish = 0f;
            }
        }

        private void PublishCameraImage()
        {
            // Render camera to texture
            publishCamera.targetTexture = renderTexture;
            publishCamera.Render();

            // Read pixels
            RenderTexture.active = renderTexture;
            texture2D.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
            texture2D.Apply();

            publishCamera.targetTexture = null;
            RenderTexture.active = null;

            // Create and publish ROS message
            byte[] imageData = texture2D.GetRawTextureData();

            var msg = new ImageMsg
            {
                header = new HeaderMsg
                {
                    stamp = new RosMessageTypes.BuiltinInterfaces.TimeMsg
                    {
                        sec = (int)Time.time,
                        nanosec = (uint)((Time.time % 1) * 1e9)
                    },
                    frame_id = "unity_camera"
                },
                height = (uint)imageHeight,
                width = (uint)imageWidth,
                encoding = "rgb8",
                is_bigendian = 0,
                step = (uint)(imageWidth * 3),
                data = imageData
            };

            ros.Publish(cameraTopic, msg);
        }

        /// <summary>
        /// Send velocity command to robot
        /// </summary>
        /// <param name="linear">Linear velocity (m/s)</param>
        /// <param name="angular">Angular velocity (rad/s)</param>
        public void SendVelocityCommand(Vector3 linear, Vector3 angular)
        {
            if (!isConnected) return;

            // Convert from Unity to ROS coordinate frame
            var msg = new TwistMsg
            {
                linear = new Vector3Msg
                {
                    x = linear.z,   // Unity Z -> ROS X
                    y = -linear.x,  // Unity X -> ROS -Y
                    z = linear.y    // Unity Y -> ROS Z
                },
                angular = new Vector3Msg
                {
                    x = angular.z,
                    y = -angular.x,
                    z = angular.y
                }
            };

            ros.Publish(cmdVelTopic, msg);
        }

        /// <summary>
        /// Send a single joint position command
        /// </summary>
        public void SendJointCommand(string jointName, float position)
        {
            // Implementation depends on robot interface
            // Could publish to a custom joint command topic
            Debug.Log($"[RosBridge] Joint command: {jointName} -> {position}");
        }

        #endregion

        #region Public Properties

        public bool IsConnected => isConnected;
        public float TimeSinceLastMessage => Time.time - lastMessageTime;
        public Vector3 CurrentTargetPosition => targetPosition;
        public Quaternion CurrentTargetRotation => targetRotation;

        #endregion
    }
}
