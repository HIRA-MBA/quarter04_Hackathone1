#!/usr/bin/env python3
"""
Isaac Sim Synthetic Data Generation Script

This script demonstrates how to generate synthetic training data
using NVIDIA Replicator with domain randomization.

Lab 8: Isaac Sim Fundamentals
"""

from omni.isaac.kit import SimulationApp

# Configure for headless data generation
CONFIG = {
    "headless": True,
    "width": 1024,
    "height": 1024,
}

simulation_app = SimulationApp(CONFIG)

import omni.replicator.core as rep
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, VisualCuboid
from omni.isaac.core.utils.nucleus import get_assets_root_path
import numpy as np
import os
from datetime import datetime


class SyntheticDataGenerator:
    """
    Generator for creating synthetic training data with domain randomization.

    Features:
    - Random object placement and materials
    - Lighting randomization
    - Camera pose randomization
    - Multi-format output (RGB, depth, segmentation, bounding boxes)
    """

    def __init__(self, output_dir: str = None):
        """
        Initialize the synthetic data generator.

        Args:
            output_dir: Directory to save generated data
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"/tmp/synthetic_data_{timestamp}"

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.world = World(stage_units_in_meters=1.0)
        self.objects = []
        self.cameras = []
        self.render_products = []

        print(f"Synthetic data will be saved to: {output_dir}")

    def setup_scene(self):
        """Create the base scene with ground plane."""
        self.world.scene.add_default_ground_plane()

    def add_target_objects(self,
                           num_objects: int = 10,
                           object_types: list = ["cube", "sphere", "cylinder"]):
        """
        Add target objects for detection/segmentation.

        Args:
            num_objects: Number of objects to create
            object_types: Types of objects to randomly choose from
        """
        with rep.new_layer():
            # Create varied objects
            for i in range(num_objects):
                obj_type = np.random.choice(object_types)

                if obj_type == "cube":
                    obj = rep.create.cube(
                        semantics=[("class", "cube")],
                        count=1,
                    )
                elif obj_type == "sphere":
                    obj = rep.create.sphere(
                        semantics=[("class", "sphere")],
                        count=1,
                    )
                elif obj_type == "cylinder":
                    obj = rep.create.cylinder(
                        semantics=[("class", "cylinder")],
                        count=1,
                    )

                self.objects.append(obj)

        print(f"Added {num_objects} target objects")

    def setup_lighting_randomization(self):
        """Configure lighting randomization."""
        with rep.new_layer():
            # Main directional light (sun)
            self.sun = rep.create.light(
                light_type="distant",
                temperature=rep.distribution.uniform(4000, 8000),
                intensity=rep.distribution.uniform(500, 2000),
            )

            # Dome light for ambient
            self.dome = rep.create.light(
                light_type="dome",
                intensity=rep.distribution.uniform(100, 500),
            )

        print("Lighting randomization configured")

    def setup_camera(self,
                     position_range: tuple = ((2, -2, 1), (4, 2, 3)),
                     resolution: tuple = (1024, 1024)):
        """
        Setup camera with position randomization.

        Args:
            position_range: ((min_x, min_y, min_z), (max_x, max_y, max_z))
            resolution: Image resolution
        """
        with rep.new_layer():
            camera = rep.create.camera(
                position=rep.distribution.uniform(position_range[0], position_range[1]),
                look_at=(0, 0, 0.5),  # Look at scene center
            )
            self.cameras.append(camera)

            # Create render product
            render_product = rep.create.render_product(camera, resolution)
            self.render_products.append(render_product)

        print(f"Camera configured with resolution {resolution}")
        return render_product

    def setup_object_randomization(self,
                                    position_range: tuple = ((-1, -1, 0.1), (1, 1, 0.5)),
                                    scale_range: tuple = ((0.05, 0.05, 0.05), (0.2, 0.2, 0.2))):
        """
        Configure object position, scale, and material randomization.

        Args:
            position_range: ((min), (max)) for position
            scale_range: ((min), (max)) for scale
        """
        with rep.trigger.on_frame():
            for obj in self.objects:
                with obj:
                    # Random position
                    rep.modify.pose(
                        position=rep.distribution.uniform(position_range[0], position_range[1]),
                        rotation=rep.distribution.uniform((0, 0, 0), (360, 360, 360)),
                        scale=rep.distribution.uniform(scale_range[0], scale_range[1]),
                    )

            # Randomize lighting each frame
            with self.sun:
                rep.modify.pose(
                    rotation=rep.distribution.uniform((30, 0, 0), (60, 360, 0))
                )

        print("Object randomization configured")

    def setup_material_randomization(self):
        """Configure material/texture randomization for objects."""
        with rep.trigger.on_frame():
            for obj in self.objects:
                rep.randomizer.materials(
                    obj,
                    materials=rep.create.material_omnipbr(
                        diffuse=rep.distribution.uniform((0.1, 0.1, 0.1), (1.0, 1.0, 1.0)),
                        roughness=rep.distribution.uniform(0.1, 0.9),
                        metallic=rep.distribution.choice([0.0, 0.5, 1.0]),
                    )
                )

        print("Material randomization configured")

    def add_distractor_objects(self,
                                num_distractors: int = 5,
                                spawn_area: tuple = ((-2, -2, 0), (2, 2, 0.5))):
        """
        Add background distractor objects.

        Args:
            num_distractors: Number of distractors
            spawn_area: Area to spawn distractors
        """
        with rep.new_layer():
            # Create random distractor shapes
            for i in range(num_distractors):
                shape = np.random.choice(["cube", "sphere", "cone"])

                if shape == "cube":
                    rep.create.cube(
                        semantics=[("class", "distractor")],
                        position=rep.distribution.uniform(spawn_area[0], spawn_area[1]),
                        scale=rep.distribution.uniform((0.1, 0.1, 0.1), (0.3, 0.3, 0.3)),
                    )
                elif shape == "sphere":
                    rep.create.sphere(
                        semantics=[("class", "distractor")],
                        position=rep.distribution.uniform(spawn_area[0], spawn_area[1]),
                        scale=rep.distribution.uniform((0.1, 0.1, 0.1), (0.3, 0.3, 0.3)),
                    )
                elif shape == "cone":
                    rep.create.cone(
                        semantics=[("class", "distractor")],
                        position=rep.distribution.uniform(spawn_area[0], spawn_area[1]),
                        scale=rep.distribution.uniform((0.1, 0.1, 0.1), (0.3, 0.3, 0.3)),
                    )

        print(f"Added {num_distractors} distractor objects")

    def setup_writer(self,
                     rgb: bool = True,
                     depth: bool = True,
                     semantic_segmentation: bool = True,
                     instance_segmentation: bool = True,
                     bounding_box_2d: bool = True,
                     bounding_box_3d: bool = True):
        """
        Configure the data writer.

        Args:
            rgb: Output RGB images
            depth: Output depth maps
            semantic_segmentation: Output semantic segmentation masks
            instance_segmentation: Output instance segmentation masks
            bounding_box_2d: Output 2D bounding boxes
            bounding_box_3d: Output 3D bounding boxes
        """
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(
            output_dir=self.output_dir,
            rgb=rgb,
            distance_to_camera=depth,
            semantic_segmentation=semantic_segmentation,
            instance_segmentation=instance_segmentation,
            bounding_box_2d_tight=bounding_box_2d,
            bounding_box_3d=bounding_box_3d,
        )

        # Attach to all render products
        writer.attach(self.render_products)

        print(f"Writer configured with outputs: RGB={rgb}, Depth={depth}, " +
              f"Semantic={semantic_segmentation}, Instance={instance_segmentation}, " +
              f"BBox2D={bounding_box_2d}, BBox3D={bounding_box_3d}")

        return writer

    def generate(self, num_frames: int = 100):
        """
        Generate synthetic data.

        Args:
            num_frames: Number of frames to generate
        """
        print(f"\nGenerating {num_frames} frames of synthetic data...")

        with rep.trigger.on_frame(num_frames=num_frames):
            pass  # Randomization already configured

        rep.orchestrator.run()

        print(f"\nData generation complete!")
        print(f"Output saved to: {self.output_dir}")


def main():
    """Main function for synthetic data generation."""

    print("=" * 60)
    print("Lab 8: Synthetic Data Generation with Domain Randomization")
    print("=" * 60)

    # Create generator
    generator = SyntheticDataGenerator(
        output_dir="/tmp/isaac_synthetic_data"
    )

    # Setup scene
    generator.setup_scene()

    # Add target objects
    generator.add_target_objects(
        num_objects=10,
        object_types=["cube", "sphere", "cylinder"]
    )

    # Add distractors
    generator.add_distractor_objects(num_distractors=5)

    # Setup camera
    generator.setup_camera(
        position_range=((2, -2, 1.5), (4, 2, 3)),
        resolution=(1024, 1024)
    )

    # Setup randomizations
    generator.setup_lighting_randomization()
    generator.setup_object_randomization()
    generator.setup_material_randomization()

    # Configure writer
    generator.setup_writer(
        rgb=True,
        depth=True,
        semantic_segmentation=True,
        instance_segmentation=True,
        bounding_box_2d=True,
        bounding_box_3d=True
    )

    # Generate data
    generator.generate(num_frames=100)

    # Cleanup
    simulation_app.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
