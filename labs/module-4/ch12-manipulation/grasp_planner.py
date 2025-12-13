#!/usr/bin/env python3
"""
Grasp Planning for Dexterous Manipulation

Implements grasp planning algorithms for multi-fingered hands,
including grasp quality metrics, force closure analysis, and
grasp synthesis for various object geometries.

Lab 12: Dexterous Manipulation
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod


class GraspType(Enum):
    """Types of grasps for manipulation."""
    PRECISION = "precision"      # Fingertip contact
    POWER = "power"              # Full hand wrap
    LATERAL = "lateral"          # Side pinch
    TRIPOD = "tripod"            # Thumb + 2 fingers
    SPHERICAL = "spherical"      # Curved hand around object
    CYLINDRICAL = "cylindrical"  # Wrap around cylinder


class ContactType(Enum):
    """Contact models for grasp analysis."""
    POINT_CONTACT_WITHOUT_FRICTION = "pcwf"
    POINT_CONTACT_WITH_FRICTION = "pcwf"
    SOFT_FINGER = "soft"


@dataclass
class ContactPoint:
    """A contact point between finger and object."""
    position: np.ndarray          # 3D position on object surface
    normal: np.ndarray            # Surface normal (pointing outward)
    finger_id: int                # Which finger (0-4 for 5-finger hand)
    friction_coefficient: float   # Coulomb friction coefficient
    contact_type: ContactType = ContactType.POINT_CONTACT_WITH_FRICTION


@dataclass
class GraspConfiguration:
    """Complete grasp configuration."""
    contacts: List[ContactPoint]
    grasp_type: GraspType
    approach_direction: np.ndarray  # Direction to approach object
    wrist_pose: np.ndarray          # 4x4 transform for wrist
    finger_joint_angles: np.ndarray # Joint angles for all fingers
    quality_score: float = 0.0


@dataclass
class ObjectGeometry:
    """Object geometric representation."""
    center: np.ndarray           # Object center
    dimensions: np.ndarray       # [length, width, height] or [radius]
    shape: str                   # 'box', 'cylinder', 'sphere'
    mass: float = 0.1            # kg
    surface_friction: float = 0.5


class GraspQualityMetric(ABC):
    """Abstract base for grasp quality metrics."""

    @abstractmethod
    def evaluate(self, grasp: GraspConfiguration, obj: ObjectGeometry) -> float:
        """Evaluate grasp quality (higher is better)."""
        pass


class ForceClosureMetric(GraspQualityMetric):
    """
    Force Closure Quality Metric

    A grasp has force closure if it can resist arbitrary external
    wrenches (forces and torques). This metric computes how well
    the grasp can resist disturbances.
    """

    def __init__(self, friction_cone_sides: int = 8):
        self.friction_cone_sides = friction_cone_sides

    def compute_grasp_matrix(self, contacts: List[ContactPoint],
                              obj_center: np.ndarray) -> np.ndarray:
        """
        Compute grasp matrix G that maps contact forces to object wrench.

        The grasp matrix relates contact forces f to the wrench w on the object:
        w = G @ f

        For point contact with friction, each contact contributes 3 force components.

        Args:
            contacts: List of contact points
            obj_center: Object center position

        Returns:
            6 x (3*n_contacts) grasp matrix
        """
        n_contacts = len(contacts)
        G = np.zeros((6, 3 * n_contacts))

        for i, contact in enumerate(contacts):
            # Position vector from object center to contact
            r = contact.position - obj_center

            # Force contribution (direct)
            G[0:3, 3*i:3*i+3] = np.eye(3)

            # Torque contribution (r x f)
            # Using skew-symmetric matrix for cross product
            G[3:6, 3*i:3*i+3] = self._skew_symmetric(r)

        return G

    def _skew_symmetric(self, v: np.ndarray) -> np.ndarray:
        """Create skew-symmetric matrix for cross product."""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    def compute_friction_cone(self, contact: ContactPoint) -> np.ndarray:
        """
        Approximate friction cone with linear constraints.

        Args:
            contact: Contact point with friction coefficient

        Returns:
            Matrix of friction cone edge directions (3 x n_sides)
        """
        mu = contact.friction_coefficient
        n = contact.normal / np.linalg.norm(contact.normal)

        # Find two tangent vectors
        if abs(n[0]) < 0.9:
            t1 = np.cross(n, np.array([1, 0, 0]))
        else:
            t1 = np.cross(n, np.array([0, 1, 0]))
        t1 = t1 / np.linalg.norm(t1)
        t2 = np.cross(n, t1)

        # Generate friction cone edges
        edges = []
        for i in range(self.friction_cone_sides):
            angle = 2 * np.pi * i / self.friction_cone_sides
            tangent = np.cos(angle) * t1 + np.sin(angle) * t2
            edge = n + mu * tangent
            edge = edge / np.linalg.norm(edge)
            edges.append(edge)

        return np.array(edges).T

    def check_force_closure(self, grasp: GraspConfiguration,
                            obj: ObjectGeometry) -> bool:
        """
        Check if grasp has force closure property.

        Force closure exists if the origin is strictly inside the
        convex hull of the wrench space.

        Args:
            grasp: Grasp configuration to check
            obj: Object geometry

        Returns:
            True if force closure exists
        """
        G = self.compute_grasp_matrix(grasp.contacts, obj.center)

        # Build wrench set from friction cones
        wrenches = []
        for i, contact in enumerate(grasp.contacts):
            cone = self.compute_friction_cone(contact)
            for j in range(cone.shape[1]):
                # Map cone edge through grasp matrix
                f = np.zeros(3 * len(grasp.contacts))
                f[3*i:3*i+3] = cone[:, j]
                w = G @ f
                wrenches.append(w)

        wrenches = np.array(wrenches)

        # Simple check: can we generate forces in all directions?
        # Full implementation would use convex hull or LP
        min_singular = np.linalg.svd(wrenches.T, compute_uv=False)[-1]

        return min_singular > 1e-6

    def evaluate(self, grasp: GraspConfiguration, obj: ObjectGeometry) -> float:
        """
        Evaluate grasp quality using smallest singular value of G.

        Higher value means the grasp can better resist disturbances.
        """
        G = self.compute_grasp_matrix(grasp.contacts, obj.center)

        # Quality = smallest singular value (isotropy measure)
        singular_values = np.linalg.svd(G, compute_uv=False)

        if len(singular_values) > 0:
            # Normalize by largest singular value for isotropy
            quality = singular_values[-1] / singular_values[0]
        else:
            quality = 0.0

        return quality


class GraspWrenchSpaceMetric(GraspQualityMetric):
    """
    Grasp Wrench Space (GWS) Quality Metric

    Measures the radius of the largest ball centered at origin
    that fits inside the grasp wrench space.
    """

    def __init__(self, force_limit: float = 10.0):
        self.force_limit = force_limit

    def evaluate(self, grasp: GraspConfiguration, obj: ObjectGeometry) -> float:
        """
        Compute GWS quality metric (epsilon quality).

        This is the minimum wrench that can be resisted in any direction.
        """
        # Simplified: use singular value based approximation
        fc_metric = ForceClosureMetric()
        G = fc_metric.compute_grasp_matrix(grasp.contacts, obj.center)

        # Quality approximation based on grasp matrix
        s = np.linalg.svd(G, compute_uv=False)

        if len(s) > 0 and s[-1] > 1e-10:
            return s[-1] * self.force_limit
        return 0.0


class GraspPlanner:
    """
    Grasp Planner for Multi-Fingered Hands

    Plans stable grasps for objects using geometric analysis
    and quality metrics.
    """

    def __init__(self,
                 n_fingers: int = 5,
                 finger_length: float = 0.1,
                 palm_width: float = 0.08):
        """
        Initialize grasp planner.

        Args:
            n_fingers: Number of fingers (default 5)
            finger_length: Length of each finger (m)
            palm_width: Width of palm (m)
        """
        self.n_fingers = n_fingers
        self.finger_length = finger_length
        self.palm_width = palm_width

        # Quality metrics
        self.force_closure = ForceClosureMetric()
        self.gws_metric = GraspWrenchSpaceMetric()

        # Finger workspace (simplified as reachable volume)
        self.finger_reach = finger_length * 0.9

    def sample_contact_points_box(self, obj: ObjectGeometry,
                                   n_samples: int = 100) -> List[ContactPoint]:
        """
        Sample contact points on a box surface.

        Args:
            obj: Box object geometry
            n_samples: Number of points to sample

        Returns:
            List of sampled contact points
        """
        contacts = []
        dims = obj.dimensions  # [length, width, height]

        # Sample on each face
        for _ in range(n_samples):
            # Random face (0-5)
            face = np.random.randint(6)

            # Generate point on face
            if face == 0:  # +X face
                pos = np.array([dims[0]/2,
                               np.random.uniform(-dims[1]/2, dims[1]/2),
                               np.random.uniform(-dims[2]/2, dims[2]/2)])
                normal = np.array([-1, 0, 0])  # Inward normal for grasping
            elif face == 1:  # -X face
                pos = np.array([-dims[0]/2,
                               np.random.uniform(-dims[1]/2, dims[1]/2),
                               np.random.uniform(-dims[2]/2, dims[2]/2)])
                normal = np.array([1, 0, 0])
            elif face == 2:  # +Y face
                pos = np.array([np.random.uniform(-dims[0]/2, dims[0]/2),
                               dims[1]/2,
                               np.random.uniform(-dims[2]/2, dims[2]/2)])
                normal = np.array([0, -1, 0])
            elif face == 3:  # -Y face
                pos = np.array([np.random.uniform(-dims[0]/2, dims[0]/2),
                               -dims[1]/2,
                               np.random.uniform(-dims[2]/2, dims[2]/2)])
                normal = np.array([0, 1, 0])
            elif face == 4:  # +Z face
                pos = np.array([np.random.uniform(-dims[0]/2, dims[0]/2),
                               np.random.uniform(-dims[1]/2, dims[1]/2),
                               dims[2]/2])
                normal = np.array([0, 0, -1])
            else:  # -Z face
                pos = np.array([np.random.uniform(-dims[0]/2, dims[0]/2),
                               np.random.uniform(-dims[1]/2, dims[1]/2),
                               -dims[2]/2])
                normal = np.array([0, 0, 1])

            pos += obj.center

            contacts.append(ContactPoint(
                position=pos,
                normal=normal,
                finger_id=-1,  # Unassigned
                friction_coefficient=obj.surface_friction
            ))

        return contacts

    def sample_contact_points_cylinder(self, obj: ObjectGeometry,
                                        n_samples: int = 100) -> List[ContactPoint]:
        """Sample contact points on a cylinder surface."""
        contacts = []
        radius = obj.dimensions[0]
        height = obj.dimensions[1] if len(obj.dimensions) > 1 else radius * 2

        for _ in range(n_samples):
            # Mostly sample on curved surface
            if np.random.random() < 0.8:
                angle = np.random.uniform(0, 2 * np.pi)
                z = np.random.uniform(-height/2, height/2)

                pos = np.array([
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    z
                ])
                normal = -np.array([np.cos(angle), np.sin(angle), 0])
            else:
                # Top or bottom
                r = np.random.uniform(0, radius)
                angle = np.random.uniform(0, 2 * np.pi)
                z = height/2 if np.random.random() < 0.5 else -height/2

                pos = np.array([r * np.cos(angle), r * np.sin(angle), z])
                normal = np.array([0, 0, -np.sign(z)])

            pos += obj.center

            contacts.append(ContactPoint(
                position=pos,
                normal=normal,
                finger_id=-1,
                friction_coefficient=obj.surface_friction
            ))

        return contacts

    def sample_contact_points_sphere(self, obj: ObjectGeometry,
                                      n_samples: int = 100) -> List[ContactPoint]:
        """Sample contact points on a sphere surface."""
        contacts = []
        radius = obj.dimensions[0]

        for _ in range(n_samples):
            # Uniform sampling on sphere
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.arccos(np.random.uniform(-1, 1))

            direction = np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ])

            pos = obj.center + radius * direction
            normal = -direction  # Inward for grasping

            contacts.append(ContactPoint(
                position=pos,
                normal=normal,
                finger_id=-1,
                friction_coefficient=obj.surface_friction
            ))

        return contacts

    def select_grasp_contacts(self, candidates: List[ContactPoint],
                               n_contacts: int = 3) -> List[ContactPoint]:
        """
        Select optimal contact points from candidates.

        Uses heuristics to select well-distributed contacts.
        """
        if len(candidates) <= n_contacts:
            return candidates

        selected = []
        remaining = list(candidates)

        # Start with a random contact
        idx = np.random.randint(len(remaining))
        selected.append(remaining.pop(idx))

        # Greedily add contacts that maximize spread
        while len(selected) < n_contacts and remaining:
            best_idx = 0
            best_min_dist = 0

            for i, candidate in enumerate(remaining):
                # Compute minimum distance to selected contacts
                min_dist = min(
                    np.linalg.norm(candidate.position - s.position)
                    for s in selected
                )

                # Also consider normal diversity
                normal_diversity = min(
                    1 - abs(np.dot(candidate.normal, s.normal))
                    for s in selected
                )

                score = min_dist + 0.5 * normal_diversity

                if score > best_min_dist:
                    best_min_dist = score
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        # Assign finger IDs
        for i, contact in enumerate(selected):
            contact.finger_id = i % self.n_fingers

        return selected

    def compute_wrist_pose(self, contacts: List[ContactPoint],
                            obj: ObjectGeometry,
                            approach: np.ndarray) -> np.ndarray:
        """
        Compute wrist pose for a grasp.

        Args:
            contacts: Selected contact points
            obj: Object geometry
            approach: Approach direction

        Returns:
            4x4 homogeneous transformation matrix
        """
        # Wrist position: offset from object center along approach
        contact_center = np.mean([c.position for c in contacts], axis=0)
        wrist_offset = self.finger_length * 0.7
        wrist_position = contact_center + approach * wrist_offset

        # Orientation: Z-axis along approach (toward object)
        z_axis = -approach / np.linalg.norm(approach)

        # X-axis: perpendicular to Z, roughly horizontal
        if abs(z_axis[2]) < 0.9:
            x_axis = np.cross(np.array([0, 0, 1]), z_axis)
        else:
            x_axis = np.cross(np.array([0, 1, 0]), z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        # Y-axis: complete right-handed frame
        y_axis = np.cross(z_axis, x_axis)

        # Build transformation matrix
        T = np.eye(4)
        T[:3, 0] = x_axis
        T[:3, 1] = y_axis
        T[:3, 2] = z_axis
        T[:3, 3] = wrist_position

        return T

    def compute_finger_ik(self, wrist_pose: np.ndarray,
                          contacts: List[ContactPoint]) -> np.ndarray:
        """
        Compute finger joint angles for contacts (simplified IK).

        Args:
            wrist_pose: Wrist transformation matrix
            contacts: Contact points

        Returns:
            Joint angles array (n_fingers x 4 joints)
        """
        # Simplified: assume 4 joints per finger
        joint_angles = np.zeros((self.n_fingers, 4))

        # Inverse of wrist pose
        wrist_inv = np.linalg.inv(wrist_pose)

        for contact in contacts:
            if contact.finger_id < 0 or contact.finger_id >= self.n_fingers:
                continue

            # Transform contact to wrist frame
            contact_wrist = wrist_inv[:3, :3] @ (contact.position - wrist_pose[:3, 3])

            # Simplified IK: distribute bend across joints
            reach = np.linalg.norm(contact_wrist)
            bend_total = np.arccos(np.clip(reach / self.finger_length, -1, 1))

            # Distribute: more bend at fingertip
            joint_angles[contact.finger_id, 0] = bend_total * 0.1  # Base
            joint_angles[contact.finger_id, 1] = bend_total * 0.3  # Proximal
            joint_angles[contact.finger_id, 2] = bend_total * 0.4  # Middle
            joint_angles[contact.finger_id, 3] = bend_total * 0.2  # Distal

        return joint_angles

    def determine_grasp_type(self, obj: ObjectGeometry,
                              contacts: List[ContactPoint]) -> GraspType:
        """Determine appropriate grasp type based on object and contacts."""
        max_dim = max(obj.dimensions)
        n_contacts = len(contacts)

        if obj.shape == 'sphere':
            if max_dim < 0.05:  # Small sphere
                return GraspType.PRECISION
            else:
                return GraspType.SPHERICAL

        elif obj.shape == 'cylinder':
            if max_dim < 0.04:
                return GraspType.PRECISION
            else:
                return GraspType.CYLINDRICAL

        else:  # Box or other
            if max_dim < 0.04:
                return GraspType.PRECISION
            elif n_contacts <= 3:
                return GraspType.TRIPOD
            else:
                return GraspType.POWER

    def plan_grasp(self, obj: ObjectGeometry,
                   n_candidates: int = 100,
                   n_contacts: int = 4) -> Optional[GraspConfiguration]:
        """
        Plan a grasp for the given object.

        Args:
            obj: Object to grasp
            n_candidates: Number of contact candidates to sample
            n_contacts: Desired number of contact points

        Returns:
            Best grasp configuration or None if no valid grasp found
        """
        # Sample contact candidates based on shape
        if obj.shape == 'box':
            candidates = self.sample_contact_points_box(obj, n_candidates)
        elif obj.shape == 'cylinder':
            candidates = self.sample_contact_points_cylinder(obj, n_candidates)
        elif obj.shape == 'sphere':
            candidates = self.sample_contact_points_sphere(obj, n_candidates)
        else:
            raise ValueError(f"Unsupported shape: {obj.shape}")

        # Try multiple grasp configurations
        best_grasp = None
        best_quality = 0.0

        for _ in range(10):  # Try 10 different contact selections
            contacts = self.select_grasp_contacts(candidates, n_contacts)

            if len(contacts) < 3:
                continue

            # Compute approach direction (from contact centroid outward)
            centroid = np.mean([c.position for c in contacts], axis=0)
            approach = centroid - obj.center
            if np.linalg.norm(approach) > 1e-6:
                approach = approach / np.linalg.norm(approach)
            else:
                approach = np.array([1, 0, 0])

            # Compute wrist pose
            wrist_pose = self.compute_wrist_pose(contacts, obj, approach)

            # Compute finger IK
            joint_angles = self.compute_finger_ik(wrist_pose, contacts)

            # Determine grasp type
            grasp_type = self.determine_grasp_type(obj, contacts)

            # Create grasp configuration
            grasp = GraspConfiguration(
                contacts=contacts,
                grasp_type=grasp_type,
                approach_direction=approach,
                wrist_pose=wrist_pose,
                finger_joint_angles=joint_angles
            )

            # Evaluate quality
            fc_quality = self.force_closure.evaluate(grasp, obj)
            gws_quality = self.gws_metric.evaluate(grasp, obj)

            # Combined quality score
            quality = 0.5 * fc_quality + 0.5 * (gws_quality / 10.0)
            grasp.quality_score = quality

            # Check force closure
            if self.force_closure.check_force_closure(grasp, obj):
                if quality > best_quality:
                    best_quality = quality
                    best_grasp = grasp

        return best_grasp

    def plan_multiple_grasps(self, obj: ObjectGeometry,
                              n_grasps: int = 5) -> List[GraspConfiguration]:
        """
        Plan multiple diverse grasps for an object.

        Args:
            obj: Object to grasp
            n_grasps: Number of grasps to generate

        Returns:
            List of grasp configurations sorted by quality
        """
        grasps = []

        for _ in range(n_grasps * 3):  # Over-sample then select best
            grasp = self.plan_grasp(obj)
            if grasp is not None:
                grasps.append(grasp)

        # Sort by quality and return top N
        grasps.sort(key=lambda g: g.quality_score, reverse=True)

        return grasps[:n_grasps]


def main():
    """Test grasp planner."""
    print("Testing Grasp Planner")
    print("=" * 50)

    # Create planner
    planner = GraspPlanner(n_fingers=5)

    # Test objects
    objects = [
        ObjectGeometry(
            center=np.array([0.3, 0.0, 0.05]),
            dimensions=np.array([0.06, 0.04, 0.1]),
            shape='box',
            mass=0.2,
            surface_friction=0.6
        ),
        ObjectGeometry(
            center=np.array([0.3, 0.2, 0.05]),
            dimensions=np.array([0.03, 0.12]),
            shape='cylinder',
            mass=0.15,
            surface_friction=0.5
        ),
        ObjectGeometry(
            center=np.array([0.3, -0.2, 0.04]),
            dimensions=np.array([0.04]),
            shape='sphere',
            mass=0.1,
            surface_friction=0.7
        )
    ]

    for i, obj in enumerate(objects):
        print(f"\nObject {i+1}: {obj.shape}")
        print(f"  Center: {obj.center}")
        print(f"  Dimensions: {obj.dimensions}")

        grasp = planner.plan_grasp(obj)

        if grasp:
            print(f"  Grasp Type: {grasp.grasp_type.value}")
            print(f"  Quality Score: {grasp.quality_score:.4f}")
            print(f"  Contacts: {len(grasp.contacts)}")
            print(f"  Approach: {grasp.approach_direction}")

            # Verify force closure
            fc = planner.force_closure.check_force_closure(grasp, obj)
            print(f"  Force Closure: {fc}")
        else:
            print("  No valid grasp found")

    # Test multiple grasp generation
    print("\n" + "=" * 50)
    print("Generating multiple grasps for box...")

    grasps = planner.plan_multiple_grasps(objects[0], n_grasps=3)
    for i, grasp in enumerate(grasps):
        print(f"  Grasp {i+1}: {grasp.grasp_type.value}, quality={grasp.quality_score:.4f}")

    print("\nGrasp Planner tests passed!")


if __name__ == "__main__":
    main()
