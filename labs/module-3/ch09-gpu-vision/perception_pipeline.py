#!/usr/bin/env python3
"""
Isaac ROS GPU Perception Pipeline

This script demonstrates GPU-accelerated perception using Isaac ROS,
including object detection, depth processing, and point cloud generation.

Lab 9: Isaac ROS GPU Perception
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import cv2
from typing import Tuple, List, Optional
import time


class GPUPerceptionPipeline(Node):
    """
    GPU-accelerated perception pipeline for robotics.

    This node demonstrates:
    - Image preprocessing on GPU
    - Object detection integration
    - Depth processing and filtering
    - Point cloud generation from RGB-D
    - Performance monitoring

    In production, you would use Isaac ROS nodes for GPU operations.
    This implementation shows the concepts and can be extended.
    """

    def __init__(self):
        super().__init__('gpu_perception_pipeline')

        # Declare parameters
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_rect_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('detection_confidence_threshold', 0.5)
        self.declare_parameter('depth_min', 0.1)
        self.declare_parameter('depth_max', 10.0)
        self.declare_parameter('enable_visualization', True)

        # Get parameters
        self.image_topic = self.get_parameter('image_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.confidence_threshold = self.get_parameter('detection_confidence_threshold').value
        self.depth_min = self.get_parameter('depth_min').value
        self.depth_max = self.get_parameter('depth_max').value
        self.enable_viz = self.get_parameter('enable_visualization').value

        # QoS for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # CV Bridge for image conversion
        self.bridge = CvBridge()

        # Camera intrinsics
        self.camera_info = None
        self.camera_matrix = None

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            sensor_qos
        )

        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            sensor_qos
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            sensor_qos
        )

        # Publishers
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/perception/detections',
            10
        )

        self.processed_image_pub = self.create_publisher(
            Image,
            '/perception/processed_image',
            10
        )

        self.point_cloud_pub = self.create_publisher(
            PointCloud2,
            '/perception/point_cloud',
            10
        )

        # State
        self.latest_rgb = None
        self.latest_depth = None
        self.frame_count = 0
        self.processing_times = []

        # Initialize detector (placeholder - use Isaac ROS in production)
        self.detector = SimpleObjectDetector(confidence_threshold=self.confidence_threshold)

        self.get_logger().info('GPU Perception Pipeline initialized')
        self.get_logger().info(f'  Image topic: {self.image_topic}')
        self.get_logger().info(f'  Depth topic: {self.depth_topic}')

    def camera_info_callback(self, msg: CameraInfo):
        """Store camera intrinsics."""
        if self.camera_info is None:
            self.camera_info = msg
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.get_logger().info(f'Camera info received: {msg.width}x{msg.height}')

    def image_callback(self, msg: Image):
        """Process incoming RGB image."""
        start_time = time.time()

        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_rgb = cv_image

            # Preprocess image
            processed = self.preprocess_image(cv_image)

            # Run detection
            detections = self.detector.detect(processed)

            # Publish detections
            self.publish_detections(detections, msg.header)

            # Visualization
            if self.enable_viz:
                viz_image = self.visualize_detections(cv_image, detections)
                viz_msg = self.bridge.cv2_to_imgmsg(viz_image, encoding='bgr8')
                viz_msg.header = msg.header
                self.processed_image_pub.publish(viz_msg)

            # Performance tracking
            elapsed = time.time() - start_time
            self.processing_times.append(elapsed)
            self.frame_count += 1

            if self.frame_count % 30 == 0:
                avg_time = np.mean(self.processing_times[-30:])
                fps = 1.0 / avg_time if avg_time > 0 else 0
                self.get_logger().info(f'Processing: {avg_time*1000:.1f}ms ({fps:.1f} FPS)')

        except Exception as e:
            self.get_logger().error(f'Image processing error: {e}')

    def depth_callback(self, msg: Image):
        """Process incoming depth image."""
        try:
            # Convert to numpy array
            if msg.encoding == '32FC1':
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            elif msg.encoding == '16UC1':
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
                depth = depth.astype(np.float32) / 1000.0  # Convert mm to m
            else:
                self.get_logger().warn(f'Unknown depth encoding: {msg.encoding}')
                return

            # Filter depth
            depth = self.filter_depth(depth)
            self.latest_depth = depth

            # Generate point cloud if we have RGB and camera info
            if self.latest_rgb is not None and self.camera_matrix is not None:
                self.generate_point_cloud(self.latest_rgb, depth, msg.header)

        except Exception as e:
            self.get_logger().error(f'Depth processing error: {e}')

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for detection.

        In production, use Isaac ROS image_proc or NITROS for GPU acceleration.
        """
        # Resize if needed
        target_size = (640, 480)
        if image.shape[:2] != target_size[::-1]:
            image = cv2.resize(image, target_size)

        # Normalize (example - actual preprocessing depends on model)
        # image = image.astype(np.float32) / 255.0

        return image

    def filter_depth(self, depth: np.ndarray) -> np.ndarray:
        """
        Filter and clean depth image.

        In production, use Isaac ROS depth processing for GPU filtering.
        """
        # Clip to valid range
        depth = np.clip(depth, self.depth_min, self.depth_max)

        # Replace invalid values
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        # Optional: median filter for noise reduction
        # depth = cv2.medianBlur(depth.astype(np.float32), 5)

        return depth

    def publish_detections(self, detections: List[dict], header: Header):
        """Publish detections as ROS messages."""
        msg = Detection2DArray()
        msg.header = header

        for det in detections:
            detection = Detection2D()
            detection.header = header

            # Bounding box center
            detection.bbox.center.position.x = float(det['cx'])
            detection.bbox.center.position.y = float(det['cy'])
            detection.bbox.size_x = float(det['width'])
            detection.bbox.size_y = float(det['height'])

            # Hypothesis
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = det['class_name']
            hypothesis.hypothesis.score = float(det['confidence'])
            detection.results.append(hypothesis)

            msg.detections.append(detection)

        self.detection_pub.publish(msg)

    def visualize_detections(self, image: np.ndarray, detections: List[dict]) -> np.ndarray:
        """Draw detection results on image."""
        viz = image.copy()

        for det in detections:
            x1 = int(det['cx'] - det['width'] / 2)
            y1 = int(det['cy'] - det['height'] / 2)
            x2 = int(det['cx'] + det['width'] / 2)
            y2 = int(det['cy'] + det['height'] / 2)

            # Draw box
            cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            cv2.putText(viz, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return viz

    def generate_point_cloud(self, rgb: np.ndarray, depth: np.ndarray, header: Header):
        """
        Generate colored point cloud from RGB-D.

        In production, use Isaac ROS depth_image_proc for GPU point cloud generation.
        """
        if self.camera_matrix is None:
            return

        # Resize depth to match RGB if needed
        if rgb.shape[:2] != depth.shape[:2]:
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]))

        # Get camera parameters
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        # Create pixel coordinate grids
        rows, cols = depth.shape
        u, v = np.meshgrid(np.arange(cols), np.arange(rows))

        # Convert to 3D points
        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # Filter valid points
        valid = (z > self.depth_min) & (z < self.depth_max)

        # Stack into point cloud
        points = np.stack([x[valid], y[valid], z[valid]], axis=-1)
        colors = rgb[valid]

        # Create PointCloud2 message (simplified - use sensor_msgs_py in production)
        # This is a placeholder - full implementation would use proper PointCloud2 structure
        self.get_logger().debug(f'Generated {len(points)} points')


class SimpleObjectDetector:
    """
    Simple object detector placeholder.

    In production, replace with:
    - Isaac ROS DOPE (Deep Object Pose Estimation)
    - Isaac ROS DetectNet
    - Isaac ROS Triton inference
    """

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.classes = ['cube', 'sphere', 'cylinder', 'robot', 'person']

    def detect(self, image: np.ndarray) -> List[dict]:
        """
        Run object detection on image.

        Returns list of detections with format:
        {
            'class_name': str,
            'confidence': float,
            'cx': float,  # center x
            'cy': float,  # center y
            'width': float,
            'height': float
        }
        """
        # Placeholder: return dummy detection
        # In production, run actual model inference here

        # Simple color-based detection as example
        detections = []

        # Convert to HSV for color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Detect red objects (example)
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                detections.append({
                    'class_name': 'red_object',
                    'confidence': 0.8,
                    'cx': x + w / 2,
                    'cy': y + h / 2,
                    'width': w,
                    'height': h
                })

        return detections


def main(args=None):
    """Main entry point."""
    print("=" * 60)
    print("Lab 9: GPU Perception Pipeline")
    print("=" * 60)

    rclpy.init(args=args)

    node = GPUPerceptionPipeline()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
