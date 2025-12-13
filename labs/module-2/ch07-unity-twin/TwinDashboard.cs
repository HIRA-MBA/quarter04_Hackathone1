// TwinDashboard.cs - Digital Twin Monitoring Dashboard
// Provides real-time visualization of robot state and sensor data

using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.Collections.Generic;

namespace PhysicalAI.DigitalTwin
{
    /// <summary>
    /// Dashboard UI controller for monitoring digital twin state.
    /// Displays connection status, robot position, joint states, and sensor data.
    /// </summary>
    public class TwinDashboard : MonoBehaviour
    {
        #region UI References

        [Header("Connection Status")]
        [Tooltip("Image indicator for connection status")]
        public Image connectionIndicator;

        [Tooltip("Text showing connection status")]
        public TMP_Text connectionStatusText;

        [Tooltip("Text showing latency/message rate")]
        public TMP_Text latencyText;

        [Header("Robot State")]
        [Tooltip("Text showing robot position")]
        public TMP_Text positionText;

        [Tooltip("Text showing robot rotation")]
        public TMP_Text rotationText;

        [Tooltip("Text showing robot velocity")]
        public TMP_Text velocityText;

        [Header("Joint Display")]
        [Tooltip("Parent transform for joint display items")]
        public Transform jointPanelParent;

        [Tooltip("Prefab for individual joint display")]
        public GameObject jointDisplayPrefab;

        [Header("Sensor Displays")]
        [Tooltip("RawImage for camera feed")]
        public RawImage cameraFeedImage;

        [Tooltip("Text for IMU data")]
        public TMP_Text imuDataText;

        [Header("Colors")]
        public Color connectedColor = new Color(0.2f, 0.8f, 0.2f);
        public Color disconnectedColor = new Color(0.8f, 0.2f, 0.2f);
        public Color warningColor = new Color(0.8f, 0.8f, 0.2f);

        #endregion

        #region References

        [Header("References")]
        [Tooltip("RosBridge component for connection status")]
        public RosBridge rosBridge;

        [Tooltip("Robot transform to monitor")]
        public Transform robotTransform;

        #endregion

        #region State

        private Dictionary<string, JointDisplayItem> jointDisplays = new Dictionary<string, JointDisplayItem>();
        private Vector3 lastPosition;
        private float lastUpdateTime;
        private int messageCount;
        private float messageRateUpdateInterval = 1f;
        private float timeSinceRateUpdate;

        #endregion

        #region Unity Lifecycle

        void Start()
        {
            // Subscribe to RosBridge events
            if (rosBridge != null)
            {
                rosBridge.OnConnectionStatusChanged += OnConnectionStatusChanged;
                rosBridge.OnOdometryReceived += OnOdometryReceived;
                rosBridge.OnJointStatesReceived += OnJointStatesReceived;
            }

            // Initialize displays
            if (robotTransform != null)
            {
                lastPosition = robotTransform.position;
            }

            lastUpdateTime = Time.time;
        }

        void Update()
        {
            UpdateConnectionDisplay();
            UpdateRobotStateDisplay();
            UpdateMessageRate();
        }

        void OnDestroy()
        {
            if (rosBridge != null)
            {
                rosBridge.OnConnectionStatusChanged -= OnConnectionStatusChanged;
                rosBridge.OnOdometryReceived -= OnOdometryReceived;
                rosBridge.OnJointStatesReceived -= OnJointStatesReceived;
            }
        }

        #endregion

        #region Display Updates

        private void UpdateConnectionDisplay()
        {
            if (rosBridge == null) return;

            bool connected = rosBridge.IsConnected;
            float timeSinceMessage = rosBridge.TimeSinceLastMessage;

            // Update indicator color
            if (connectionIndicator != null)
            {
                if (!connected)
                {
                    connectionIndicator.color = disconnectedColor;
                }
                else if (timeSinceMessage > 0.5f)
                {
                    connectionIndicator.color = warningColor;
                }
                else
                {
                    connectionIndicator.color = connectedColor;
                }
            }

            // Update status text
            if (connectionStatusText != null)
            {
                if (!connected)
                {
                    connectionStatusText.text = "Disconnected";
                }
                else if (timeSinceMessage > 0.5f)
                {
                    connectionStatusText.text = $"Connected (Stale: {timeSinceMessage:F1}s)";
                }
                else
                {
                    connectionStatusText.text = "Connected";
                }
            }
        }

        private void UpdateRobotStateDisplay()
        {
            if (robotTransform == null) return;

            // Update position display
            if (positionText != null)
            {
                Vector3 pos = robotTransform.position;
                positionText.text = $"Position:\n  X: {pos.x:F3}\n  Y: {pos.y:F3}\n  Z: {pos.z:F3}";
            }

            // Update rotation display
            if (rotationText != null)
            {
                Vector3 euler = robotTransform.eulerAngles;
                rotationText.text = $"Rotation:\n  Roll: {euler.x:F1}°\n  Pitch: {euler.y:F1}°\n  Yaw: {euler.z:F1}°";
            }

            // Calculate and display velocity
            if (velocityText != null)
            {
                float deltaTime = Time.time - lastUpdateTime;
                if (deltaTime > 0)
                {
                    Vector3 velocity = (robotTransform.position - lastPosition) / deltaTime;
                    velocityText.text = $"Velocity:\n  {velocity.magnitude:F2} m/s";
                }
                lastPosition = robotTransform.position;
                lastUpdateTime = Time.time;
            }
        }

        private void UpdateMessageRate()
        {
            timeSinceRateUpdate += Time.deltaTime;

            if (timeSinceRateUpdate >= messageRateUpdateInterval)
            {
                if (latencyText != null)
                {
                    float rate = messageCount / timeSinceRateUpdate;
                    latencyText.text = $"Message Rate: {rate:F1} Hz";
                }

                messageCount = 0;
                timeSinceRateUpdate = 0f;
            }
        }

        #endregion

        #region Event Handlers

        private void OnConnectionStatusChanged(bool connected)
        {
            Debug.Log($"[Dashboard] Connection status changed: {connected}");
        }

        private void OnOdometryReceived(Vector3 position, Quaternion rotation)
        {
            messageCount++;
        }

        private void OnJointStatesReceived(Dictionary<string, float> jointPositions)
        {
            messageCount++;

            // Update or create joint displays
            foreach (var kvp in jointPositions)
            {
                UpdateJointDisplay(kvp.Key, kvp.Value);
            }
        }

        #endregion

        #region Joint Display Management

        private void UpdateJointDisplay(string jointName, float position)
        {
            if (jointPanelParent == null || jointDisplayPrefab == null) return;

            // Create display if it doesn't exist
            if (!jointDisplays.TryGetValue(jointName, out var display))
            {
                GameObject displayObj = Instantiate(jointDisplayPrefab, jointPanelParent);
                display = displayObj.GetComponent<JointDisplayItem>();

                if (display == null)
                {
                    display = displayObj.AddComponent<JointDisplayItem>();
                }

                display.Initialize(jointName);
                jointDisplays[jointName] = display;
            }

            // Update display value
            display.UpdateValue(position);
        }

        /// <summary>
        /// Clear all joint displays
        /// </summary>
        public void ClearJointDisplays()
        {
            foreach (var display in jointDisplays.Values)
            {
                if (display != null)
                {
                    Destroy(display.gameObject);
                }
            }
            jointDisplays.Clear();
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// Update IMU data display
        /// </summary>
        public void UpdateIMUDisplay(Vector3 orientation, Vector3 angularVelocity, Vector3 linearAcceleration)
        {
            if (imuDataText != null)
            {
                imuDataText.text = $"IMU Data:\n" +
                    $"  Orientation: ({orientation.x:F1}, {orientation.y:F1}, {orientation.z:F1})\n" +
                    $"  Angular Vel: ({angularVelocity.x:F2}, {angularVelocity.y:F2}, {angularVelocity.z:F2})\n" +
                    $"  Linear Acc: ({linearAcceleration.x:F2}, {linearAcceleration.y:F2}, {linearAcceleration.z:F2})";
            }
        }

        /// <summary>
        /// Set the camera feed texture
        /// </summary>
        public void SetCameraFeed(Texture texture)
        {
            if (cameraFeedImage != null)
            {
                cameraFeedImage.texture = texture;
            }
        }

        #endregion
    }

    /// <summary>
    /// Individual joint display item component
    /// </summary>
    public class JointDisplayItem : MonoBehaviour
    {
        [Header("UI Elements")]
        public TMP_Text nameText;
        public TMP_Text valueText;
        public Slider positionSlider;

        private string jointName;
        private float minRange = -3.14159f;
        private float maxRange = 3.14159f;

        public void Initialize(string name)
        {
            jointName = name;

            if (nameText != null)
            {
                nameText.text = name;
            }

            if (positionSlider != null)
            {
                positionSlider.minValue = minRange;
                positionSlider.maxValue = maxRange;
            }
        }

        public void UpdateValue(float position)
        {
            if (valueText != null)
            {
                // Display in degrees
                float degrees = position * Mathf.Rad2Deg;
                valueText.text = $"{degrees:F1}°";
            }

            if (positionSlider != null)
            {
                positionSlider.value = position;
            }
        }

        public void SetRange(float min, float max)
        {
            minRange = min;
            maxRange = max;

            if (positionSlider != null)
            {
                positionSlider.minValue = minRange;
                positionSlider.maxValue = maxRange;
            }
        }
    }
}
