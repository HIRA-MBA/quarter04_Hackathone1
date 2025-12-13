#!/usr/bin/env python3
"""
Vision-Language-Action Agent

Integrates vision encoders, language models, and action decoders
for natural language robot control. This module demonstrates VLA
model architecture and inference for humanoid robots.

Lab 13: Vision-Language-Action Models
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json


class ActionType(Enum):
    """Types of robot actions."""
    NAVIGATE = "navigate"
    PICK = "pick"
    PLACE = "place"
    PUSH = "push"
    PULL = "pull"
    OPEN = "open"
    CLOSE = "close"
    WAVE = "wave"
    POINT = "point"
    LOOK = "look"
    SPEAK = "speak"
    WAIT = "wait"
    STOP = "stop"


@dataclass
class VisualObservation:
    """Visual observation from robot cameras."""
    rgb_image: np.ndarray           # [H, W, 3] RGB image
    depth_image: Optional[np.ndarray] = None  # [H, W] depth map
    camera_intrinsics: Optional[np.ndarray] = None  # 3x3 intrinsic matrix
    camera_pose: Optional[np.ndarray] = None  # 4x4 world pose
    timestamp: float = 0.0


@dataclass
class DetectedObject:
    """Detected object in the scene."""
    label: str
    confidence: float
    bbox_2d: np.ndarray  # [x1, y1, x2, y2]
    position_3d: Optional[np.ndarray] = None  # [x, y, z] in world frame
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneUnderstanding:
    """Scene understanding from vision system."""
    objects: List[DetectedObject]
    scene_description: str
    spatial_relations: Dict[str, List[str]]  # e.g., {"on": ["cup", "table"]}
    affordances: Dict[str, List[str]]  # e.g., {"graspable": ["cup", "pen"]}


@dataclass
class ActionPrimitive:
    """A single action primitive."""
    action_type: ActionType
    target_object: Optional[str] = None
    target_position: Optional[np.ndarray] = None
    target_orientation: Optional[np.ndarray] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    duration: float = 1.0
    confidence: float = 1.0


@dataclass
class ActionSequence:
    """Sequence of action primitives."""
    actions: List[ActionPrimitive]
    language_command: str
    reasoning: str
    total_duration: float = 0.0
    success_probability: float = 0.0


class VisionEncoder(ABC):
    """Abstract base class for vision encoders."""

    @abstractmethod
    def encode(self, observation: VisualObservation) -> np.ndarray:
        """Encode visual observation to feature vector."""
        pass

    @abstractmethod
    def detect_objects(self, observation: VisualObservation) -> List[DetectedObject]:
        """Detect objects in the observation."""
        pass


class SimulatedVisionEncoder(VisionEncoder):
    """
    Simulated Vision Encoder

    Simulates a vision transformer (ViT) based encoder for demonstration.
    In production, this would wrap a model like CLIP, DINOv2, or RT-2's vision encoder.
    """

    def __init__(self, embedding_dim: int = 512):
        """
        Initialize vision encoder.

        Args:
            embedding_dim: Dimension of visual embeddings
        """
        self.embedding_dim = embedding_dim

        # Simulated object detection vocabulary
        self.object_vocabulary = [
            "cup", "bottle", "plate", "bowl", "fork", "knife", "spoon",
            "table", "chair", "door", "drawer", "cabinet", "shelf",
            "book", "pen", "phone", "laptop", "box", "ball", "toy",
            "person", "hand", "face"
        ]

        # Simulated attribute vocabulary
        self.attribute_vocabulary = {
            "color": ["red", "blue", "green", "yellow", "white", "black", "brown"],
            "size": ["small", "medium", "large"],
            "material": ["plastic", "metal", "glass", "wood", "ceramic"],
            "state": ["open", "closed", "full", "empty"]
        }

    def encode(self, observation: VisualObservation) -> np.ndarray:
        """
        Encode visual observation to feature vector.

        In production, this would run the image through a ViT or CNN backbone.
        Here we simulate the encoding process.

        Args:
            observation: Visual observation with RGB image

        Returns:
            Feature vector of shape [embedding_dim]
        """
        # Simulate encoding by extracting simple image statistics
        if observation.rgb_image is not None:
            # Use image statistics as pseudo-features
            mean_rgb = np.mean(observation.rgb_image, axis=(0, 1))
            std_rgb = np.std(observation.rgb_image, axis=(0, 1))

            # Expand to embedding dimension with random projection
            np.random.seed(int(mean_rgb.sum()) % 1000)  # Deterministic for same image
            projection = np.random.randn(6, self.embedding_dim) * 0.1

            features = np.concatenate([mean_rgb / 255.0, std_rgb / 255.0])
            embedding = features @ projection

            # Normalize
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

            return embedding

        return np.zeros(self.embedding_dim)

    def detect_objects(self, observation: VisualObservation) -> List[DetectedObject]:
        """
        Detect objects in the visual observation.

        Simulates object detection. In production, would use YOLO, DETR, etc.

        Args:
            observation: Visual observation

        Returns:
            List of detected objects
        """
        detected = []

        # Simulate detection based on image content
        np.random.seed(42)  # For reproducibility

        # Generate some plausible detections
        n_objects = np.random.randint(2, 6)

        h, w = observation.rgb_image.shape[:2] if observation.rgb_image is not None else (480, 640)

        for i in range(n_objects):
            label = np.random.choice(self.object_vocabulary[:15])  # Common objects
            confidence = np.random.uniform(0.7, 0.99)

            # Random bounding box
            x1 = np.random.randint(0, w - 100)
            y1 = np.random.randint(0, h - 100)
            x2 = x1 + np.random.randint(50, 150)
            y2 = y1 + np.random.randint(50, 150)

            # Random 3D position if depth available
            position_3d = None
            if observation.depth_image is not None:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                depth = observation.depth_image[cy, cx] if cy < h and cx < w else 1.0
                position_3d = np.array([
                    (cx - w/2) * depth / 500,  # Simplified projection
                    (cy - h/2) * depth / 500,
                    depth
                ])

            # Random attributes
            attributes = {
                "color": np.random.choice(self.attribute_vocabulary["color"]),
                "size": np.random.choice(self.attribute_vocabulary["size"])
            }

            detected.append(DetectedObject(
                label=label,
                confidence=confidence,
                bbox_2d=np.array([x1, y1, x2, y2]),
                position_3d=position_3d,
                attributes=attributes
            ))

        return detected


class LanguageEncoder(ABC):
    """Abstract base class for language encoders."""

    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """Encode text to feature vector."""
        pass

    @abstractmethod
    def parse_command(self, text: str) -> Dict[str, Any]:
        """Parse command into structured form."""
        pass


class SimulatedLanguageEncoder(LanguageEncoder):
    """
    Simulated Language Encoder

    Simulates a language model encoder for demonstration.
    In production, would wrap models like BERT, T5, or PaLM.
    """

    def __init__(self, embedding_dim: int = 512):
        """
        Initialize language encoder.

        Args:
            embedding_dim: Dimension of text embeddings
        """
        self.embedding_dim = embedding_dim

        # Action keywords mapping
        self.action_keywords = {
            ActionType.NAVIGATE: ["go", "walk", "move", "navigate", "come", "approach"],
            ActionType.PICK: ["pick", "grab", "grasp", "take", "get", "fetch"],
            ActionType.PLACE: ["place", "put", "set", "drop", "leave"],
            ActionType.PUSH: ["push", "shove"],
            ActionType.PULL: ["pull", "drag"],
            ActionType.OPEN: ["open"],
            ActionType.CLOSE: ["close", "shut"],
            ActionType.WAVE: ["wave", "greet"],
            ActionType.POINT: ["point", "indicate", "show"],
            ActionType.LOOK: ["look", "see", "find", "locate", "search"],
            ActionType.SPEAK: ["say", "speak", "tell"],
            ActionType.WAIT: ["wait", "pause", "hold"],
            ActionType.STOP: ["stop", "halt", "freeze"]
        }

        # Preposition mappings for spatial understanding
        self.spatial_prepositions = {
            "on": "on_top_of",
            "under": "below",
            "beneath": "below",
            "above": "above",
            "next to": "beside",
            "beside": "beside",
            "near": "near",
            "in front of": "in_front",
            "behind": "behind",
            "inside": "inside",
            "into": "inside",
            "to": "towards",
            "towards": "towards",
            "from": "from"
        }

    def encode(self, text: str) -> np.ndarray:
        """
        Encode text to feature vector.

        Simulates text encoding. In production, would use transformer encoder.

        Args:
            text: Input text string

        Returns:
            Feature vector of shape [embedding_dim]
        """
        # Simple bag-of-words style encoding
        words = text.lower().split()

        # Create deterministic embedding based on words
        embedding = np.zeros(self.embedding_dim)

        for i, word in enumerate(words):
            # Hash word to embedding dimension
            word_hash = hash(word) % self.embedding_dim
            embedding[word_hash] += 1.0 / (i + 1)  # Position-weighted

        # Normalize
        if np.linalg.norm(embedding) > 0:
            embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def parse_command(self, text: str) -> Dict[str, Any]:
        """
        Parse natural language command into structured form.

        Args:
            text: Natural language command

        Returns:
            Structured command dictionary
        """
        text_lower = text.lower()
        words = text_lower.split()

        result = {
            "raw_text": text,
            "action": None,
            "target_object": None,
            "destination": None,
            "spatial_relation": None,
            "attributes": {},
            "modifiers": []
        }

        # Detect action type
        for action_type, keywords in self.action_keywords.items():
            if any(kw in text_lower for kw in keywords):
                result["action"] = action_type
                break

        # Detect spatial prepositions
        for prep, relation in self.spatial_prepositions.items():
            if prep in text_lower:
                result["spatial_relation"] = relation
                # Extract object after preposition
                idx = text_lower.find(prep)
                after_prep = text_lower[idx + len(prep):].strip()
                first_word = after_prep.split()[0] if after_prep else None
                if first_word and first_word not in ["the", "a", "an"]:
                    result["destination"] = first_word
                elif len(after_prep.split()) > 1:
                    result["destination"] = after_prep.split()[1]
                break

        # Extract target object (typically after action verb)
        common_objects = ["cup", "bottle", "ball", "box", "book", "pen",
                         "phone", "plate", "bowl", "door", "drawer"]
        for obj in common_objects:
            if obj in text_lower:
                if result["target_object"] is None:
                    result["target_object"] = obj
                elif result["destination"] is None:
                    result["destination"] = obj

        # Detect color attributes
        colors = ["red", "blue", "green", "yellow", "white", "black", "brown"]
        for color in colors:
            if color in text_lower:
                result["attributes"]["color"] = color
                break

        # Detect size attributes
        sizes = ["small", "big", "large", "tiny"]
        for size in sizes:
            if size in text_lower:
                result["attributes"]["size"] = size
                break

        # Detect modifiers
        if "carefully" in text_lower:
            result["modifiers"].append("careful")
        if "quickly" in text_lower or "fast" in text_lower:
            result["modifiers"].append("fast")
        if "slowly" in text_lower:
            result["modifiers"].append("slow")

        return result


class ActionDecoder(ABC):
    """Abstract base class for action decoders."""

    @abstractmethod
    def decode(self,
               visual_features: np.ndarray,
               language_features: np.ndarray,
               scene: SceneUnderstanding,
               parsed_command: Dict[str, Any]) -> ActionSequence:
        """Decode features into action sequence."""
        pass


class SimulatedActionDecoder(ActionDecoder):
    """
    Simulated Action Decoder

    Converts multimodal features into executable action sequences.
    In production, would use a transformer decoder with action tokens.
    """

    def __init__(self):
        """Initialize action decoder."""
        # Default action parameters
        self.default_durations = {
            ActionType.NAVIGATE: 5.0,
            ActionType.PICK: 3.0,
            ActionType.PLACE: 2.0,
            ActionType.PUSH: 2.0,
            ActionType.PULL: 2.0,
            ActionType.OPEN: 2.0,
            ActionType.CLOSE: 1.5,
            ActionType.WAVE: 2.0,
            ActionType.POINT: 1.5,
            ActionType.LOOK: 1.0,
            ActionType.SPEAK: 2.0,
            ActionType.WAIT: 1.0,
            ActionType.STOP: 0.5
        }

        # Action preconditions
        self.preconditions = {
            ActionType.PICK: ["object_visible", "hand_empty", "object_reachable"],
            ActionType.PLACE: ["holding_object", "destination_clear"],
            ActionType.OPEN: ["object_is_openable", "object_reachable"],
            ActionType.CLOSE: ["object_is_closeable", "object_reachable"]
        }

    def find_object_in_scene(self,
                              scene: SceneUnderstanding,
                              target_name: str,
                              attributes: Dict[str, Any] = None) -> Optional[DetectedObject]:
        """
        Find object in scene matching criteria.

        Args:
            scene: Scene understanding with detected objects
            target_name: Name of object to find
            attributes: Optional attributes to match

        Returns:
            Matching detected object or None
        """
        for obj in scene.objects:
            if target_name.lower() in obj.label.lower():
                # Check attributes if provided
                if attributes:
                    match = True
                    for key, value in attributes.items():
                        if key in obj.attributes and obj.attributes[key] != value:
                            match = False
                            break
                    if match:
                        return obj
                else:
                    return obj
        return None

    def generate_navigation_action(self,
                                    target_position: np.ndarray,
                                    modifiers: List[str]) -> ActionPrimitive:
        """Generate navigation action primitive."""
        duration = self.default_durations[ActionType.NAVIGATE]

        if "fast" in modifiers:
            duration *= 0.7
        elif "slow" in modifiers:
            duration *= 1.5

        return ActionPrimitive(
            action_type=ActionType.NAVIGATE,
            target_position=target_position,
            duration=duration,
            parameters={"speed": "fast" if "fast" in modifiers else "normal"}
        )

    def generate_manipulation_sequence(self,
                                        action_type: ActionType,
                                        target_object: DetectedObject,
                                        destination: Optional[np.ndarray],
                                        modifiers: List[str]) -> List[ActionPrimitive]:
        """Generate manipulation action sequence."""
        actions = []

        # Pre-approach: look at target
        actions.append(ActionPrimitive(
            action_type=ActionType.LOOK,
            target_object=target_object.label,
            target_position=target_object.position_3d,
            duration=0.5
        ))

        # Main action
        duration = self.default_durations.get(action_type, 2.0)
        if "careful" in modifiers:
            duration *= 1.5
        elif "fast" in modifiers:
            duration *= 0.7

        main_action = ActionPrimitive(
            action_type=action_type,
            target_object=target_object.label,
            target_position=target_object.position_3d,
            duration=duration,
            parameters={
                "careful": "careful" in modifiers,
                "object_attributes": target_object.attributes
            }
        )
        actions.append(main_action)

        # If placing, add place action
        if action_type == ActionType.PICK and destination is not None:
            actions.append(ActionPrimitive(
                action_type=ActionType.PLACE,
                target_position=destination,
                duration=self.default_durations[ActionType.PLACE]
            ))

        return actions

    def decode(self,
               visual_features: np.ndarray,
               language_features: np.ndarray,
               scene: SceneUnderstanding,
               parsed_command: Dict[str, Any]) -> ActionSequence:
        """
        Decode multimodal features into action sequence.

        Args:
            visual_features: Encoded visual features
            language_features: Encoded language features
            scene: Scene understanding with objects
            parsed_command: Parsed language command

        Returns:
            ActionSequence with planned actions
        """
        actions = []
        reasoning_steps = []

        action_type = parsed_command.get("action")
        target_name = parsed_command.get("target_object")
        destination_name = parsed_command.get("destination")
        attributes = parsed_command.get("attributes", {})
        modifiers = parsed_command.get("modifiers", [])

        # Find target object in scene
        target_object = None
        if target_name:
            target_object = self.find_object_in_scene(scene, target_name, attributes)
            if target_object:
                reasoning_steps.append(f"Found {target_name} at position {target_object.position_3d}")
            else:
                reasoning_steps.append(f"Could not find {target_name} in scene")

        # Find destination
        destination_position = None
        if destination_name:
            dest_obj = self.find_object_in_scene(scene, destination_name)
            if dest_obj and dest_obj.position_3d is not None:
                destination_position = dest_obj.position_3d.copy()
                destination_position[2] += 0.1  # Slightly above
                reasoning_steps.append(f"Destination: {destination_name} at {destination_position}")

        # Generate action sequence based on type
        if action_type == ActionType.NAVIGATE:
            if destination_position is not None:
                actions.append(self.generate_navigation_action(destination_position, modifiers))
                reasoning_steps.append("Generated navigation action")
            elif target_object and target_object.position_3d is not None:
                # Navigate near the object
                nav_pos = target_object.position_3d.copy()
                nav_pos[0] -= 0.5  # Stand 0.5m in front
                actions.append(self.generate_navigation_action(nav_pos, modifiers))
                reasoning_steps.append("Generated navigation to object vicinity")

        elif action_type in [ActionType.PICK, ActionType.PUSH, ActionType.PULL]:
            if target_object:
                manipulation_actions = self.generate_manipulation_sequence(
                    action_type, target_object, destination_position, modifiers
                )
                actions.extend(manipulation_actions)
                reasoning_steps.append(f"Generated {action_type.value} sequence with {len(manipulation_actions)} steps")
            else:
                reasoning_steps.append("Cannot execute manipulation without target object")

        elif action_type == ActionType.PLACE:
            if destination_position is not None:
                actions.append(ActionPrimitive(
                    action_type=ActionType.PLACE,
                    target_position=destination_position,
                    duration=self.default_durations[ActionType.PLACE]
                ))
                reasoning_steps.append("Generated place action")

        elif action_type in [ActionType.OPEN, ActionType.CLOSE]:
            if target_object:
                actions.append(ActionPrimitive(
                    action_type=action_type,
                    target_object=target_object.label,
                    target_position=target_object.position_3d,
                    duration=self.default_durations[action_type]
                ))
                reasoning_steps.append(f"Generated {action_type.value} action")

        elif action_type == ActionType.LOOK:
            if target_object:
                actions.append(ActionPrimitive(
                    action_type=ActionType.LOOK,
                    target_object=target_object.label,
                    target_position=target_object.position_3d,
                    duration=self.default_durations[ActionType.LOOK]
                ))
            reasoning_steps.append("Generated look action")

        elif action_type == ActionType.WAVE:
            actions.append(ActionPrimitive(
                action_type=ActionType.WAVE,
                duration=self.default_durations[ActionType.WAVE]
            ))
            reasoning_steps.append("Generated wave gesture")

        elif action_type == ActionType.STOP:
            actions.append(ActionPrimitive(
                action_type=ActionType.STOP,
                duration=self.default_durations[ActionType.STOP]
            ))
            reasoning_steps.append("Generated stop action")

        else:
            # Default: wait and observe
            actions.append(ActionPrimitive(
                action_type=ActionType.WAIT,
                duration=1.0
            ))
            reasoning_steps.append("No clear action identified, waiting")

        # Calculate totals
        total_duration = sum(a.duration for a in actions)
        success_prob = 0.9 if target_object else 0.5

        return ActionSequence(
            actions=actions,
            language_command=parsed_command["raw_text"],
            reasoning="; ".join(reasoning_steps),
            total_duration=total_duration,
            success_probability=success_prob
        )


class VLAAgent:
    """
    Vision-Language-Action Agent

    Main agent class that integrates vision encoder, language encoder,
    and action decoder for natural language robot control.
    """

    def __init__(self, embedding_dim: int = 512):
        """
        Initialize VLA agent.

        Args:
            embedding_dim: Dimension for multimodal embeddings
        """
        self.embedding_dim = embedding_dim

        # Initialize components
        self.vision_encoder = SimulatedVisionEncoder(embedding_dim)
        self.language_encoder = SimulatedLanguageEncoder(embedding_dim)
        self.action_decoder = SimulatedActionDecoder()

        # State tracking
        self.current_observation: Optional[VisualObservation] = None
        self.current_scene: Optional[SceneUnderstanding] = None
        self.action_history: List[ActionSequence] = []

    def perceive(self, observation: VisualObservation) -> SceneUnderstanding:
        """
        Process visual observation and build scene understanding.

        Args:
            observation: Visual observation from robot cameras

        Returns:
            Scene understanding with detected objects and relations
        """
        self.current_observation = observation

        # Detect objects
        objects = self.vision_encoder.detect_objects(observation)

        # Build spatial relations (simplified)
        spatial_relations = {}
        for i, obj1 in enumerate(objects):
            if obj1.position_3d is not None:
                for obj2 in objects[i+1:]:
                    if obj2.position_3d is not None:
                        # Check vertical relation
                        if obj1.position_3d[2] > obj2.position_3d[2] + 0.05:
                            if "above" not in spatial_relations:
                                spatial_relations["above"] = []
                            spatial_relations["above"].append(f"{obj1.label} above {obj2.label}")

        # Build affordances (simplified)
        affordances = {
            "graspable": [o.label for o in objects if o.label in
                         ["cup", "bottle", "ball", "book", "pen", "phone"]],
            "openable": [o.label for o in objects if o.label in
                        ["door", "drawer", "cabinet", "box"]],
            "surface": [o.label for o in objects if o.label in
                       ["table", "shelf", "chair"]]
        }

        # Generate scene description
        obj_names = [o.label for o in objects]
        scene_description = f"Scene contains {len(objects)} objects: {', '.join(obj_names)}"

        self.current_scene = SceneUnderstanding(
            objects=objects,
            scene_description=scene_description,
            spatial_relations=spatial_relations,
            affordances=affordances
        )

        return self.current_scene

    def process_command(self, command: str) -> ActionSequence:
        """
        Process natural language command and generate action sequence.

        Args:
            command: Natural language command string

        Returns:
            ActionSequence with planned robot actions
        """
        if self.current_scene is None:
            # Create empty scene if no perception yet
            self.current_scene = SceneUnderstanding(
                objects=[],
                scene_description="No visual observation available",
                spatial_relations={},
                affordances={}
            )

        # Encode visual observation
        visual_features = np.zeros(self.embedding_dim)
        if self.current_observation is not None:
            visual_features = self.vision_encoder.encode(self.current_observation)

        # Encode language command
        language_features = self.language_encoder.encode(command)

        # Parse command structure
        parsed_command = self.language_encoder.parse_command(command)

        # Decode to actions
        action_sequence = self.action_decoder.decode(
            visual_features,
            language_features,
            self.current_scene,
            parsed_command
        )

        # Store in history
        self.action_history.append(action_sequence)

        return action_sequence

    def get_multimodal_embedding(self,
                                  observation: VisualObservation,
                                  command: str) -> np.ndarray:
        """
        Get fused multimodal embedding for observation and command.

        Args:
            observation: Visual observation
            command: Language command

        Returns:
            Fused embedding vector
        """
        visual_features = self.vision_encoder.encode(observation)
        language_features = self.language_encoder.encode(command)

        # Simple fusion: concatenate and project
        combined = np.concatenate([visual_features, language_features])

        # Project to original dimension (simulated)
        np.random.seed(42)
        projection = np.random.randn(2 * self.embedding_dim, self.embedding_dim) * 0.1
        fused = combined @ projection

        # Normalize
        fused = fused / (np.linalg.norm(fused) + 1e-8)

        return fused

    def explain_action(self, action_sequence: ActionSequence) -> str:
        """
        Generate human-readable explanation of action sequence.

        Args:
            action_sequence: Action sequence to explain

        Returns:
            Explanation string
        """
        lines = [
            f"Command: \"{action_sequence.language_command}\"",
            f"Reasoning: {action_sequence.reasoning}",
            f"Planned {len(action_sequence.actions)} action(s):",
        ]

        for i, action in enumerate(action_sequence.actions, 1):
            desc = f"  {i}. {action.action_type.value.upper()}"
            if action.target_object:
                desc += f" {action.target_object}"
            if action.target_position is not None:
                pos = action.target_position
                desc += f" at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
            desc += f" [{action.duration:.1f}s]"
            lines.append(desc)

        lines.append(f"Total duration: {action_sequence.total_duration:.1f}s")
        lines.append(f"Success probability: {action_sequence.success_probability:.0%}")

        return "\n".join(lines)


def main():
    """Test VLA Agent."""
    print("Testing Vision-Language-Action Agent")
    print("=" * 60)

    # Create agent
    agent = VLAAgent(embedding_dim=512)

    # Create simulated observation
    rgb_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    depth_image = np.random.uniform(0.5, 3.0, (480, 640))

    observation = VisualObservation(
        rgb_image=rgb_image,
        depth_image=depth_image,
        timestamp=0.0
    )

    # Perceive scene
    print("\n1. Perceiving scene...")
    scene = agent.perceive(observation)
    print(f"   {scene.scene_description}")
    print(f"   Graspable: {scene.affordances.get('graspable', [])}")

    # Test commands
    test_commands = [
        "Pick up the red cup",
        "Go to the table",
        "Put the bottle on the shelf",
        "Open the door carefully",
        "Wave hello",
        "Look for the book"
    ]

    print("\n2. Processing commands...")
    for command in test_commands:
        print(f"\n{'='*60}")
        action_seq = agent.process_command(command)
        explanation = agent.explain_action(action_seq)
        print(explanation)

    # Test multimodal embedding
    print(f"\n{'='*60}")
    print("\n3. Testing multimodal embedding...")
    embedding = agent.get_multimodal_embedding(observation, "pick up the cup")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   Embedding norm: {np.linalg.norm(embedding):.4f}")

    print("\n" + "=" * 60)
    print("VLA Agent tests completed!")


if __name__ == "__main__":
    main()
