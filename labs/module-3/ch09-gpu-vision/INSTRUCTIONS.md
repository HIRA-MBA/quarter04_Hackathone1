# Lab 9: GPU Vision Pipeline - Step-by-Step Instructions

## Overview

In this lab, you will build a GPU-accelerated perception pipeline using Isaac ROS. You'll learn to process camera and depth data on the GPU, run object detection inference, and generate point clouds efficiently.

**Estimated Time**: 2-3 hours

## Prerequisites

- Completed Lab 8 (Isaac Sim Scene)
- NVIDIA RTX GPU with CUDA support
- ROS 2 Humble installed
- Isaac ROS packages installed
- Basic understanding of computer vision

## Learning Outcomes

By completing this lab, you will:

1. Set up Isaac ROS GPU perception stack
2. Process RGB-D data using NITROS (zero-copy GPU)
3. Implement object detection with DetectNet
4. Generate point clouds from depth images
5. Profile GPU performance

---

## Part 1: Isaac ROS Setup (30 minutes)

### Step 1.1: Install Isaac ROS

Isaac ROS is distributed via Docker containers:

```bash
# Clone Isaac ROS workspace
mkdir -p ~/isaac_ros_ws/src
cd ~/isaac_ros_ws/src

# Clone common packages
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_pipeline
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_object_detection
```

### Step 1.2: Launch Isaac ROS Container

```bash
cd ~/isaac_ros_ws/src/isaac_ros_common
./scripts/run_dev.sh
```

This starts a Docker container with all Isaac ROS dependencies.

### Step 1.3: Build Packages

Inside the container:

```bash
cd /workspaces/isaac_ros-dev
colcon build --symlink-install
source install/setup.bash
```

### Step 1.4: Verify Installation

```bash
ros2 pkg list | grep isaac
# Should see: isaac_ros_image_proc, isaac_ros_dnn_inference, etc.
```

**Expected Output**: Isaac ROS packages listed and built successfully.

---

## Part 2: Camera Pipeline Setup (30 minutes)

### Step 2.1: Launch Camera Node

For a RealSense camera:

```bash
ros2 launch realsense2_camera rs_launch.py \
    depth_module.profile:=640x480x30 \
    rgb_camera.profile:=640x480x30 \
    enable_sync:=true \
    align_depth.enable:=true
```

Or use Isaac Sim's camera publisher (from Lab 8).

### Step 2.2: Verify Camera Topics

```bash
ros2 topic list | grep camera
# Should see:
# /camera/color/image_raw
# /camera/depth/image_rect_raw
# /camera/color/camera_info
```

### Step 2.3: Launch Isaac ROS Image Pipeline

```bash
ros2 launch isaac_ros_image_proc isaac_ros_image_pipeline.launch.py \
    image_input:=/camera/color/image_raw \
    camera_info_input:=/camera/color/camera_info
```

This provides GPU-accelerated:
- Image rectification
- Color space conversion
- Resize operations

### Step 2.4: Verify GPU Processing

Check GPU usage:

```bash
nvidia-smi -l 1
# Should see GPU utilization when processing images
```

**Expected Output**: Camera images being processed on GPU with low latency.

---

## Part 3: Object Detection (45 minutes)

### Step 3.1: Download DetectNet Model

```bash
cd ~/isaac_ros_ws
mkdir -p models/detectnet

# Download PeopleNet model (example)
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/pruned_quantized_v2.3.2/files/resnet34_peoplenet_pruned_int8.etlt \
    -O models/detectnet/peoplenet.etlt
```

### Step 3.2: Convert Model to TensorRT

```bash
ros2 run isaac_ros_dnn_inference trt_converter.py \
    --model_path models/detectnet/peoplenet.etlt \
    --output_path models/detectnet/peoplenet.plan \
    --input_dims 3,544,960 \
    --data_type int8
```

### Step 3.3: Launch Detection Node

```bash
ros2 launch isaac_ros_detectnet isaac_ros_detectnet.launch.py \
    model_file_path:=/workspaces/isaac_ros-dev/models/detectnet/peoplenet.plan \
    input_topic:=/image_rect \
    detection_topic:=/detectnet/detections
```

### Step 3.4: Visualize Detections

```bash
ros2 run rviz2 rviz2
# Add Detection2DArray display
# Set topic to /detectnet/detections
```

Or use rqt_image_view:

```bash
ros2 run rqt_image_view rqt_image_view /detectnet/output_image
```

**Expected Output**: Object detections appearing in real-time with bounding boxes.

---

## Part 4: Depth Processing (30 minutes)

### Step 4.1: Launch Depth Pipeline

```bash
ros2 launch isaac_ros_image_proc isaac_ros_image_proc_depth.launch.py \
    depth_input:=/camera/depth/image_rect_raw \
    camera_info_input:=/camera/depth/camera_info
```

### Step 4.2: Run Perception Pipeline Node

Copy the lab files to your workspace:

```bash
cp /path/to/labs/module-3/ch09-gpu-vision/*.py ~/isaac_ros_ws/src/my_perception/
```

Build and run:

```bash
cd ~/isaac_ros_ws
colcon build --packages-select my_perception
source install/setup.bash

ros2 run my_perception perception_pipeline
```

### Step 4.3: Test Depth Processor

Run the depth processor test:

```bash
python3 /path/to/labs/module-3/ch09-gpu-vision/depth_processor.py
```

### Step 4.4: Verify Point Cloud

```bash
ros2 topic echo /perception/point_cloud --no-arr
# Should see point cloud messages being published
```

Visualize in RViz:
1. Add PointCloud2 display
2. Set topic to /perception/point_cloud
3. Set color transformer to RGB

**Expected Output**: Colored point cloud visible in RViz, updating in real-time.

---

## Part 5: Performance Profiling (30 minutes)

### Step 5.1: Enable NITROS Profiling

```bash
export ISAAC_ROS_ENABLE_PROFILING=1
ros2 launch isaac_ros_detectnet isaac_ros_detectnet.launch.py
```

### Step 5.2: Monitor GPU Memory

```bash
watch -n 0.5 nvidia-smi
```

Key metrics:
- GPU Memory Used: Should stay under 80%
- GPU Utilization: Target 50-70% for real-time
- Power Draw: Monitor for thermal issues

### Step 5.3: Measure Latency

The perception pipeline node prints timing info:

```
Processing: 15.2ms (65.8 FPS)
```

Target metrics:
- Image processing: < 20ms
- Detection inference: < 30ms
- Point cloud: < 10ms
- Total pipeline: < 50ms (20 FPS)

### Step 5.4: CPU vs GPU Comparison

Run depth processor without GPU:

```python
processor = DepthProcessor(use_gpu=False)
```

Compare timing with GPU enabled:

```python
processor = DepthProcessor(use_gpu=True)
```

**Expected Output**: 3-10x speedup with GPU for depth processing.

---

## Part 6: Integration with Robot (20 minutes)

### Step 6.1: Launch Full Stack

Terminal 1 - Camera:
```bash
ros2 launch realsense2_camera rs_launch.py
```

Terminal 2 - Isaac ROS Pipeline:
```bash
ros2 launch isaac_ros_detectnet isaac_ros_detectnet.launch.py
```

Terminal 3 - Perception Node:
```bash
ros2 run my_perception perception_pipeline
```

Terminal 4 - RViz:
```bash
ros2 run rviz2 rviz2 -d perception.rviz
```

### Step 6.2: Test Detection Accuracy

Place known objects in camera view and verify:
- [ ] Objects are detected
- [ ] Bounding boxes are accurate
- [ ] Confidence scores are reasonable
- [ ] False positive rate is low

### Step 6.3: Record Data

```bash
ros2 bag record \
    /camera/color/image_raw \
    /camera/depth/image_rect_raw \
    /detectnet/detections \
    /perception/point_cloud \
    -o perception_test
```

**Expected Output**: Full perception stack running with all outputs.

---

## Troubleshooting

### Isaac ROS Container Issues

**Problem**: Container won't start
**Solutions**:
1. Check Docker installation: `docker --version`
2. Verify NVIDIA container toolkit: `nvidia-docker --version`
3. Check GPU access: `docker run --gpus all nvidia/cuda:11.0-base nvidia-smi`

### Model Conversion Fails

**Problem**: TensorRT conversion error
**Solutions**:
1. Check input dimensions match model
2. Verify CUDA version compatibility
3. Ensure enough GPU memory for conversion

### High Latency

**Problem**: Processing takes > 100ms
**Solutions**:
1. Reduce image resolution
2. Use INT8 quantized models
3. Enable NITROS zero-copy transfers
4. Check for CPU bottlenecks in pipeline

### No GPU Acceleration

**Problem**: Processing uses CPU
**Solutions**:
1. Verify CuPy is installed: `python3 -c "import cupy"`
2. Check CUDA toolkit: `nvcc --version`
3. Verify GPU is visible: `nvidia-smi`

---

## Acceptance Criteria

Your lab is complete when:

- [ ] Isaac ROS packages built successfully
- [ ] Camera pipeline produces rectified images
- [ ] Object detection runs on GPU
- [ ] Detections appear in RViz/rqt
- [ ] Depth processor filters depth images
- [ ] Point cloud generated from RGB-D
- [ ] Pipeline achieves 20+ FPS
- [ ] GPU utilization visible in nvidia-smi
- [ ] CPU vs GPU timing comparison documented

---

## Challenge Extensions

### Extension 1: Multi-Object Tracking

Integrate Isaac ROS Object Tracking:
- Track detected objects across frames
- Assign unique IDs to each object
- Handle occlusion and reappearance

### Extension 2: Semantic Segmentation

Replace DetectNet with semantic segmentation:
- Use ESPNet or BiSeNet model
- Generate per-pixel class labels
- Apply to point cloud coloring

### Extension 3: 6-DoF Pose Estimation

Implement object pose estimation:
- Use DOPE (Deep Object Pose Estimation)
- Estimate object position and orientation
- Visualize pose axes in RViz

---

## Files in This Lab

| File | Description |
|------|-------------|
| `perception_pipeline.py` | Main ROS 2 perception node |
| `depth_processor.py` | GPU-accelerated depth processing |
| `INSTRUCTIONS.md` | This file |
| `README.md` | Lab overview |

---

## Next Steps

After completing this lab:
1. Move to Chapter 10: Navigation and Sim2Real
2. Learn to transfer trained models to real hardware
3. Implement navigation using perception outputs
