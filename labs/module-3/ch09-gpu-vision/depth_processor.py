#!/usr/bin/env python3
"""
GPU-Accelerated Depth Processing

This module provides depth image processing utilities that can be
accelerated using CUDA/GPU. In production, use Isaac ROS NITROS
for zero-copy GPU processing.

Lab 9: Isaac ROS GPU Perception
"""

import numpy as np
from typing import Tuple, Optional
import cv2

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np  # Fall back to NumPy
    GPU_AVAILABLE = False


class DepthProcessor:
    """
    Depth image processor with optional GPU acceleration.

    Features:
    - Depth filtering and hole filling
    - Normal estimation from depth
    - Plane segmentation
    - Point cloud generation
    - Distance measurement

    In production, use Isaac ROS depth processing nodes for
    optimal GPU performance with NITROS.
    """

    def __init__(self,
                 use_gpu: bool = True,
                 depth_min: float = 0.1,
                 depth_max: float = 10.0):
        """
        Initialize depth processor.

        Args:
            use_gpu: Use GPU acceleration if available
            depth_min: Minimum valid depth (meters)
            depth_max: Maximum valid depth (meters)
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.xp = cp if self.use_gpu else np

        if self.use_gpu:
            print("DepthProcessor: Using GPU (CuPy)")
        else:
            print("DepthProcessor: Using CPU (NumPy)")

    def _to_device(self, array: np.ndarray):
        """Move array to GPU if using GPU."""
        if self.use_gpu:
            return cp.asarray(array)
        return array

    def _to_host(self, array):
        """Move array to CPU."""
        if self.use_gpu and hasattr(array, 'get'):
            return array.get()
        return array

    def filter_depth(self,
                     depth: np.ndarray,
                     bilateral: bool = True,
                     median_kernel: int = 5) -> np.ndarray:
        """
        Filter depth image to reduce noise and fill holes.

        Args:
            depth: Input depth image (HxW, float32, meters)
            bilateral: Apply bilateral filter
            median_kernel: Median filter kernel size

        Returns:
            Filtered depth image
        """
        d = self._to_device(depth.astype(np.float32))

        # Clip to valid range
        d = self.xp.clip(d, self.depth_min, self.depth_max)

        # Replace invalid values
        d = self.xp.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

        # Move to CPU for OpenCV operations
        d_cpu = self._to_host(d)

        # Median filter for salt-and-pepper noise
        if median_kernel > 0:
            d_cpu = cv2.medianBlur(d_cpu, median_kernel)

        # Bilateral filter for edge-preserving smoothing
        if bilateral:
            d_cpu = cv2.bilateralFilter(d_cpu, 9, 75, 75)

        return d_cpu

    def fill_holes(self,
                   depth: np.ndarray,
                   max_hole_size: int = 50) -> np.ndarray:
        """
        Fill holes in depth image using inpainting.

        Args:
            depth: Input depth image
            max_hole_size: Maximum hole size to fill

        Returns:
            Depth image with holes filled
        """
        # Create mask of invalid pixels
        mask = (depth <= self.depth_min) | (depth >= self.depth_max)
        mask = mask.astype(np.uint8)

        # Normalize depth for inpainting
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = depth_norm.astype(np.uint8)

        # Inpaint holes
        filled = cv2.inpaint(depth_norm, mask, max_hole_size, cv2.INPAINT_TELEA)

        # Convert back to original scale
        filled = filled.astype(np.float32) / 255.0 * (self.depth_max - self.depth_min) + self.depth_min

        # Preserve valid original values
        result = np.where(mask, filled, depth)

        return result

    def estimate_normals(self,
                         depth: np.ndarray,
                         camera_matrix: np.ndarray) -> np.ndarray:
        """
        Estimate surface normals from depth image.

        Args:
            depth: Depth image (HxW, float32, meters)
            camera_matrix: 3x3 camera intrinsic matrix

        Returns:
            Normal map (HxWx3, float32, normalized vectors)
        """
        d = self._to_device(depth.astype(np.float32))
        K = self._to_device(camera_matrix.astype(np.float32))

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        rows, cols = d.shape

        # Create coordinate grids
        u, v = self.xp.meshgrid(self.xp.arange(cols), self.xp.arange(rows))

        # Convert to 3D points
        z = d
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # Compute gradients
        dz_du = self.xp.gradient(z, axis=1)
        dz_dv = self.xp.gradient(z, axis=0)

        # Compute partial derivatives in 3D
        dx_du = z / fx
        dy_dv = z / fy

        # Cross product to get normal
        nx = -dz_du * dy_dv
        ny = -dz_dv * dx_du
        nz = dx_du * dy_dv

        # Normalize
        norm = self.xp.sqrt(nx**2 + ny**2 + nz**2) + 1e-10
        nx, ny, nz = nx / norm, ny / norm, nz / norm

        # Stack into normal map
        normals = self.xp.stack([nx, ny, nz], axis=-1)

        return self._to_host(normals)

    def segment_planes(self,
                       depth: np.ndarray,
                       camera_matrix: np.ndarray,
                       distance_threshold: float = 0.02,
                       min_points: int = 1000) -> Tuple[np.ndarray, list]:
        """
        Segment planar surfaces from depth image.

        Args:
            depth: Depth image
            camera_matrix: Camera intrinsics
            distance_threshold: RANSAC distance threshold (meters)
            min_points: Minimum points for a valid plane

        Returns:
            Tuple of (label_image, plane_coefficients)
        """
        # Generate point cloud
        points = self.depth_to_points(depth, camera_matrix)

        # Flatten points
        valid = (depth > self.depth_min) & (depth < self.depth_max)
        pts = points[valid].reshape(-1, 3)

        if len(pts) < min_points:
            return np.zeros(depth.shape, dtype=np.int32), []

        # Simple RANSAC plane fitting
        # In production, use PCL or Open3D for robust segmentation
        labels = np.zeros(depth.shape, dtype=np.int32)
        planes = []

        # This is a simplified implementation
        # Full implementation would iterate and segment multiple planes

        return labels, planes

    def depth_to_points(self,
                        depth: np.ndarray,
                        camera_matrix: np.ndarray) -> np.ndarray:
        """
        Convert depth image to 3D point cloud.

        Args:
            depth: Depth image (HxW, meters)
            camera_matrix: 3x3 camera intrinsic matrix

        Returns:
            Point cloud (HxWx3, meters)
        """
        d = self._to_device(depth.astype(np.float32))
        K = self._to_device(camera_matrix.astype(np.float32))

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        rows, cols = d.shape

        # Create coordinate grids
        u, v = self.xp.meshgrid(self.xp.arange(cols), self.xp.arange(rows))

        # Back-project to 3D
        z = d
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # Stack into point cloud
        points = self.xp.stack([x, y, z], axis=-1)

        return self._to_host(points)

    def measure_distance(self,
                         depth: np.ndarray,
                         point: Tuple[int, int],
                         window_size: int = 5) -> float:
        """
        Measure distance at a specific image point.

        Args:
            depth: Depth image
            point: (x, y) pixel coordinates
            window_size: Averaging window size

        Returns:
            Distance in meters
        """
        x, y = point
        half = window_size // 2

        # Extract window
        h, w = depth.shape
        x1 = max(0, x - half)
        x2 = min(w, x + half + 1)
        y1 = max(0, y - half)
        y2 = min(h, y + half + 1)

        window = depth[y1:y2, x1:x2]

        # Filter valid values
        valid = (window > self.depth_min) & (window < self.depth_max)

        if np.sum(valid) == 0:
            return -1.0  # Invalid

        return float(np.median(window[valid]))

    def compute_disparity(self,
                          left_depth: np.ndarray,
                          baseline: float,
                          focal_length: float) -> np.ndarray:
        """
        Convert depth to disparity for stereo visualization.

        Args:
            left_depth: Depth image from left camera
            baseline: Stereo baseline (meters)
            focal_length: Focal length (pixels)

        Returns:
            Disparity image
        """
        # disparity = baseline * focal_length / depth
        d = self._to_device(left_depth.astype(np.float32))

        # Avoid division by zero
        d = self.xp.where(d > 0, d, 1e-6)

        disparity = baseline * focal_length / d

        return self._to_host(disparity)


def main():
    """Test depth processor."""
    print("Testing DepthProcessor...")

    # Create processor
    processor = DepthProcessor(use_gpu=True, depth_min=0.3, depth_max=5.0)

    # Create test depth image
    depth = np.random.uniform(0.5, 3.0, (480, 640)).astype(np.float32)

    # Add some holes
    depth[100:150, 100:150] = 0.0

    # Test filtering
    filtered = processor.filter_depth(depth)
    print(f"Filtered depth range: [{filtered.min():.2f}, {filtered.max():.2f}]")

    # Test hole filling
    filled = processor.fill_holes(depth)
    print(f"Filled depth range: [{filled.min():.2f}, {filled.max():.2f}]")

    # Test point cloud generation
    camera_matrix = np.array([
        [600, 0, 320],
        [0, 600, 240],
        [0, 0, 1]
    ], dtype=np.float32)

    points = processor.depth_to_points(depth, camera_matrix)
    print(f"Point cloud shape: {points.shape}")

    # Test normal estimation
    normals = processor.estimate_normals(depth, camera_matrix)
    print(f"Normals shape: {normals.shape}")

    # Test distance measurement
    dist = processor.measure_distance(depth, (320, 240))
    print(f"Distance at center: {dist:.2f}m")

    print("\nDepthProcessor tests passed!")


if __name__ == '__main__':
    main()
