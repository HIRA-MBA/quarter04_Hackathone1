// DomainRandomizer.cs - Domain Randomization for Synthetic Data Generation
// Randomizes lighting, materials, and camera parameters for training robust perception models

using UnityEngine;
using System.Collections.Generic;

namespace PhysicalAI.DigitalTwin
{
    /// <summary>
    /// Domain randomization controller for generating diverse training data.
    /// Randomizes lighting, materials, camera parameters, and object placements.
    /// </summary>
    public class DomainRandomizer : MonoBehaviour
    {
        #region Lighting Randomization

        [System.Serializable]
        public class LightingSettings
        {
            [Header("Sun Light")]
            public Light sunLight;

            [Range(0f, 24f)]
            public float timeOfDayMin = 8f;

            [Range(0f, 24f)]
            public float timeOfDayMax = 18f;

            [Range(0f, 5f)]
            public float intensityMin = 0.5f;

            [Range(0f, 5f)]
            public float intensityMax = 2f;

            [Header("Color Temperature")]
            [Range(1000f, 20000f)]
            public float colorTempMin = 4000f;

            [Range(1000f, 20000f)]
            public float colorTempMax = 7000f;

            [Header("Additional Lights")]
            public Light[] additionalLights;
            public bool randomizeAdditionalLights = true;
        }

        #endregion

        #region Material Randomization

        [System.Serializable]
        public class MaterialTarget
        {
            public Renderer targetRenderer;
            public Material[] possibleMaterials;
            public bool randomizeColor = true;
            public Color colorMin = new Color(0.3f, 0.3f, 0.3f);
            public Color colorMax = Color.white;
            public bool randomizeMetallic = false;
            public bool randomizeSmoothness = false;
        }

        #endregion

        #region Camera Randomization

        [System.Serializable]
        public class CameraSettings
        {
            public Camera targetCamera;

            [Header("Position Noise")]
            public bool randomizePosition = true;
            public Vector3 positionNoiseRange = new Vector3(0.1f, 0.1f, 0.1f);

            [Header("Rotation Noise")]
            public bool randomizeRotation = true;
            public Vector3 rotationNoiseRange = new Vector3(2f, 2f, 2f);

            [Header("Field of View")]
            public bool randomizeFOV = false;
            public float fovMin = 50f;
            public float fovMax = 70f;

            [Header("Post-Processing")]
            public bool randomizeExposure = false;
            public float exposureMin = -1f;
            public float exposureMax = 1f;
        }

        #endregion

        #region Configuration

        [Header("Lighting")]
        public LightingSettings lighting = new LightingSettings();

        [Header("Materials")]
        public List<MaterialTarget> materialTargets = new List<MaterialTarget>();

        [Header("Cameras")]
        public List<CameraSettings> cameraSettings = new List<CameraSettings>();

        [Header("Distractor Objects")]
        [Tooltip("Prefabs to spawn as distractors")]
        public GameObject[] distractorPrefabs;

        [Tooltip("Area where distractors can spawn")]
        public Bounds distractorSpawnArea = new Bounds(Vector3.zero, new Vector3(5, 0, 5));

        [Range(0, 20)]
        public int maxDistractors = 5;

        [Header("Randomization Settings")]
        [Tooltip("Randomize on Start")]
        public bool randomizeOnStart = true;

        [Tooltip("Interval for periodic randomization (0 = disabled)")]
        public float randomizationInterval = 0f;

        [Tooltip("Random seed (-1 for time-based)")]
        public int randomSeed = -1;

        #endregion

        #region State

        private List<GameObject> spawnedDistractors = new List<GameObject>();
        private Dictionary<Camera, Vector3> originalCameraPositions = new Dictionary<Camera, Vector3>();
        private Dictionary<Camera, Quaternion> originalCameraRotations = new Dictionary<Camera, Quaternion>();

        #endregion

        #region Unity Lifecycle

        void Awake()
        {
            // Store original camera transforms
            foreach (var cam in cameraSettings)
            {
                if (cam.targetCamera != null)
                {
                    originalCameraPositions[cam.targetCamera] = cam.targetCamera.transform.position;
                    originalCameraRotations[cam.targetCamera] = cam.targetCamera.transform.rotation;
                }
            }
        }

        void Start()
        {
            // Initialize random seed
            if (randomSeed >= 0)
            {
                Random.InitState(randomSeed);
            }

            if (randomizeOnStart)
            {
                RandomizeAll();
            }

            if (randomizationInterval > 0)
            {
                InvokeRepeating(nameof(RandomizeAll), randomizationInterval, randomizationInterval);
            }
        }

        #endregion

        #region Randomization Methods

        /// <summary>
        /// Randomize all configured elements
        /// </summary>
        public void RandomizeAll()
        {
            RandomizeLighting();
            RandomizeMaterials();
            RandomizeCameras();
            RandomizeDistractors();
        }

        /// <summary>
        /// Randomize lighting conditions
        /// </summary>
        public void RandomizeLighting()
        {
            if (lighting.sunLight == null) return;

            // Random time of day affects sun angle
            float timeOfDay = Random.Range(lighting.timeOfDayMin, lighting.timeOfDayMax);
            float sunAngle = (timeOfDay / 24f) * 360f - 90f;
            lighting.sunLight.transform.rotation = Quaternion.Euler(sunAngle, Random.Range(0f, 360f), 0f);

            // Random intensity
            lighting.sunLight.intensity = Random.Range(lighting.intensityMin, lighting.intensityMax);

            // Random color temperature
            lighting.sunLight.colorTemperature = Random.Range(lighting.colorTempMin, lighting.colorTempMax);

            // Randomize additional lights
            if (lighting.randomizeAdditionalLights && lighting.additionalLights != null)
            {
                foreach (var light in lighting.additionalLights)
                {
                    if (light != null)
                    {
                        light.intensity = Random.Range(0.5f, 1.5f) * light.intensity;
                        light.colorTemperature = Random.Range(3000f, 8000f);
                    }
                }
            }
        }

        /// <summary>
        /// Randomize material properties
        /// </summary>
        public void RandomizeMaterials()
        {
            foreach (var target in materialTargets)
            {
                if (target.targetRenderer == null) continue;

                // Random material selection
                if (target.possibleMaterials != null && target.possibleMaterials.Length > 0)
                {
                    int index = Random.Range(0, target.possibleMaterials.Length);
                    target.targetRenderer.material = new Material(target.possibleMaterials[index]);
                }

                Material mat = target.targetRenderer.material;

                // Random color
                if (target.randomizeColor)
                {
                    Color randomColor = new Color(
                        Random.Range(target.colorMin.r, target.colorMax.r),
                        Random.Range(target.colorMin.g, target.colorMax.g),
                        Random.Range(target.colorMin.b, target.colorMax.b)
                    );
                    mat.color = randomColor;
                }

                // Random metallic
                if (target.randomizeMetallic && mat.HasProperty("_Metallic"))
                {
                    mat.SetFloat("_Metallic", Random.Range(0f, 1f));
                }

                // Random smoothness
                if (target.randomizeSmoothness && mat.HasProperty("_Smoothness"))
                {
                    mat.SetFloat("_Smoothness", Random.Range(0f, 1f));
                }
            }
        }

        /// <summary>
        /// Randomize camera parameters
        /// </summary>
        public void RandomizeCameras()
        {
            foreach (var cam in cameraSettings)
            {
                if (cam.targetCamera == null) continue;

                // Get original transform
                Vector3 originalPos = originalCameraPositions.GetValueOrDefault(cam.targetCamera, cam.targetCamera.transform.position);
                Quaternion originalRot = originalCameraRotations.GetValueOrDefault(cam.targetCamera, cam.targetCamera.transform.rotation);

                // Random position noise
                if (cam.randomizePosition)
                {
                    Vector3 noise = new Vector3(
                        Random.Range(-cam.positionNoiseRange.x, cam.positionNoiseRange.x),
                        Random.Range(-cam.positionNoiseRange.y, cam.positionNoiseRange.y),
                        Random.Range(-cam.positionNoiseRange.z, cam.positionNoiseRange.z)
                    );
                    cam.targetCamera.transform.position = originalPos + noise;
                }

                // Random rotation noise
                if (cam.randomizeRotation)
                {
                    Vector3 rotNoise = new Vector3(
                        Random.Range(-cam.rotationNoiseRange.x, cam.rotationNoiseRange.x),
                        Random.Range(-cam.rotationNoiseRange.y, cam.rotationNoiseRange.y),
                        Random.Range(-cam.rotationNoiseRange.z, cam.rotationNoiseRange.z)
                    );
                    cam.targetCamera.transform.rotation = originalRot * Quaternion.Euler(rotNoise);
                }

                // Random FOV
                if (cam.randomizeFOV)
                {
                    cam.targetCamera.fieldOfView = Random.Range(cam.fovMin, cam.fovMax);
                }
            }
        }

        /// <summary>
        /// Spawn and randomize distractor objects
        /// </summary>
        public void RandomizeDistractors()
        {
            // Clear existing distractors
            ClearDistractors();

            if (distractorPrefabs == null || distractorPrefabs.Length == 0) return;

            // Spawn random number of distractors
            int numDistractors = Random.Range(0, maxDistractors + 1);

            for (int i = 0; i < numDistractors; i++)
            {
                // Random prefab
                GameObject prefab = distractorPrefabs[Random.Range(0, distractorPrefabs.Length)];
                if (prefab == null) continue;

                // Random position within spawn area
                Vector3 position = new Vector3(
                    Random.Range(distractorSpawnArea.min.x, distractorSpawnArea.max.x),
                    distractorSpawnArea.center.y,
                    Random.Range(distractorSpawnArea.min.z, distractorSpawnArea.max.z)
                );

                // Random rotation
                Quaternion rotation = Quaternion.Euler(0, Random.Range(0f, 360f), 0);

                // Spawn distractor
                GameObject distractor = Instantiate(prefab, position, rotation);

                // Random scale
                float scale = Random.Range(0.8f, 1.2f);
                distractor.transform.localScale *= scale;

                spawnedDistractors.Add(distractor);
            }
        }

        /// <summary>
        /// Clear all spawned distractor objects
        /// </summary>
        public void ClearDistractors()
        {
            foreach (var distractor in spawnedDistractors)
            {
                if (distractor != null)
                {
                    Destroy(distractor);
                }
            }
            spawnedDistractors.Clear();
        }

        #endregion

        #region Utility Methods

        /// <summary>
        /// Reset all randomized elements to default
        /// </summary>
        public void ResetToDefault()
        {
            // Reset camera transforms
            foreach (var cam in cameraSettings)
            {
                if (cam.targetCamera != null)
                {
                    if (originalCameraPositions.TryGetValue(cam.targetCamera, out var pos))
                    {
                        cam.targetCamera.transform.position = pos;
                    }
                    if (originalCameraRotations.TryGetValue(cam.targetCamera, out var rot))
                    {
                        cam.targetCamera.transform.rotation = rot;
                    }
                }
            }

            // Clear distractors
            ClearDistractors();
        }

        /// <summary>
        /// Set random seed for reproducible randomization
        /// </summary>
        public void SetSeed(int seed)
        {
            randomSeed = seed;
            Random.InitState(seed);
        }

        #endregion

        #region Editor Visualization

        void OnDrawGizmosSelected()
        {
            // Draw distractor spawn area
            Gizmos.color = new Color(0, 1, 0, 0.3f);
            Gizmos.DrawCube(distractorSpawnArea.center, distractorSpawnArea.size);
            Gizmos.color = Color.green;
            Gizmos.DrawWireCube(distractorSpawnArea.center, distractorSpawnArea.size);
        }

        #endregion
    }
}
