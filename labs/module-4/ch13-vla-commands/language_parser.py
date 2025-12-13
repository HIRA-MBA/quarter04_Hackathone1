#!/usr/bin/env python3
"""
Natural Language Parser for Robot Commands

Implements parsing of natural language commands into structured
representations for robot control. Includes intent classification,
entity extraction, and semantic parsing.

Lab 13: Vision-Language-Action Models
"""

import re
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


class Intent(Enum):
    """High-level intent categories for robot commands."""
    MANIPULATION = "manipulation"  # Pick, place, push, pull
    NAVIGATION = "navigation"      # Go, move, approach
    OBSERVATION = "observation"    # Look, find, search
    INTERACTION = "interaction"    # Open, close, turn on/off
    COMMUNICATION = "communication"  # Say, wave, gesture
    CONTROL = "control"            # Stop, wait, pause, continue
    QUERY = "query"                # What, where, how many
    UNKNOWN = "unknown"


@dataclass
class Entity:
    """Extracted entity from natural language."""
    text: str
    entity_type: str  # object, location, person, attribute, quantity
    normalized: str   # Normalized/canonical form
    start_idx: int    # Start position in original text
    end_idx: int      # End position in original text
    confidence: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpatialRelation:
    """Spatial relation between entities."""
    relation_type: str  # on, under, next_to, inside, etc.
    subject: str        # Entity being described
    reference: str      # Reference entity
    preposition: str    # Original preposition


@dataclass
class ParsedCommand:
    """Fully parsed natural language command."""
    raw_text: str
    intent: Intent
    action_verb: Optional[str]
    entities: List[Entity]
    target_entity: Optional[Entity]
    destination_entity: Optional[Entity]
    spatial_relations: List[SpatialRelation]
    modifiers: Dict[str, Any]  # speed, care level, etc.
    negation: bool
    question_type: Optional[str]  # what, where, how_many
    confidence: float


class Tokenizer:
    """Simple tokenizer for natural language."""

    def __init__(self):
        """Initialize tokenizer."""
        # Common contractions
        self.contractions = {
            "don't": "do not",
            "doesn't": "does not",
            "won't": "will not",
            "can't": "cannot",
            "couldn't": "could not",
            "wouldn't": "would not",
            "shouldn't": "should not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "i'm": "i am",
            "you're": "you are",
            "he's": "he is",
            "she's": "she is",
            "it's": "it is",
            "we're": "we are",
            "they're": "they are",
            "i've": "i have",
            "you've": "you have",
            "we've": "we have",
            "they've": "they have",
            "i'll": "i will",
            "you'll": "you will",
            "he'll": "he will",
            "she'll": "she will",
            "we'll": "we will",
            "they'll": "they will",
            "i'd": "i would",
            "you'd": "you would",
            "he'd": "he would",
            "she'd": "she would",
            "we'd": "we would",
            "they'd": "they would",
            "let's": "let us"
        }

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Input text string

        Returns:
            List of tokens
        """
        # Lowercase
        text = text.lower()

        # Expand contractions
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)

        # Remove punctuation except hyphens in compound words
        text = re.sub(r'[^\w\s-]', ' ', text)

        # Split on whitespace
        tokens = text.split()

        return tokens


class IntentClassifier:
    """Classifies intent of natural language commands."""

    def __init__(self):
        """Initialize intent classifier."""
        # Intent keyword mappings
        self.intent_keywords = {
            Intent.MANIPULATION: {
                "pick", "grab", "grasp", "take", "get", "fetch",
                "put", "place", "set", "drop", "leave", "release",
                "push", "pull", "drag", "slide", "move", "lift",
                "hold", "carry", "throw", "catch", "hand"
            },
            Intent.NAVIGATION: {
                "go", "walk", "move", "navigate", "come", "approach",
                "follow", "lead", "turn", "step", "run", "head",
                "return", "back", "forward", "left", "right"
            },
            Intent.OBSERVATION: {
                "look", "see", "find", "locate", "search", "scan",
                "observe", "watch", "check", "inspect", "examine",
                "detect", "identify", "recognize"
            },
            Intent.INTERACTION: {
                "open", "close", "shut", "turn", "switch", "press",
                "push", "pull", "rotate", "twist", "flip", "toggle",
                "activate", "deactivate", "start", "begin"
            },
            Intent.COMMUNICATION: {
                "say", "speak", "tell", "wave", "greet", "gesture",
                "point", "indicate", "show", "nod", "shake", "signal"
            },
            Intent.CONTROL: {
                "stop", "halt", "freeze", "pause", "wait", "hold",
                "continue", "resume", "cancel", "abort", "reset"
            },
            Intent.QUERY: {
                "what", "where", "which", "who", "how", "when",
                "count", "many", "much", "is", "are", "can", "could"
            }
        }

        # Priority for disambiguation
        self.intent_priority = [
            Intent.MANIPULATION,
            Intent.NAVIGATION,
            Intent.INTERACTION,
            Intent.OBSERVATION,
            Intent.COMMUNICATION,
            Intent.CONTROL,
            Intent.QUERY
        ]

    def classify(self, tokens: List[str]) -> Tuple[Intent, float]:
        """
        Classify intent from tokens.

        Args:
            tokens: List of tokens from command

        Returns:
            Tuple of (Intent, confidence)
        """
        token_set = set(tokens)

        # Score each intent
        scores = {}
        for intent, keywords in self.intent_keywords.items():
            matches = token_set & keywords
            scores[intent] = len(matches)

        # Find highest scoring intent
        max_score = max(scores.values()) if scores else 0

        if max_score == 0:
            return Intent.UNKNOWN, 0.0

        # Handle ties using priority
        candidates = [i for i, s in scores.items() if s == max_score]

        for intent in self.intent_priority:
            if intent in candidates:
                confidence = min(0.9, 0.5 + max_score * 0.2)
                return intent, confidence

        return candidates[0], 0.5


class EntityExtractor:
    """Extracts entities from natural language."""

    def __init__(self):
        """Initialize entity extractor."""
        # Object vocabulary
        self.objects = {
            # Kitchen items
            "cup", "mug", "glass", "bottle", "can", "jar",
            "plate", "bowl", "dish", "pot", "pan", "kettle",
            "fork", "knife", "spoon", "chopsticks",
            # Furniture
            "table", "chair", "desk", "couch", "sofa", "bed",
            "shelf", "cabinet", "drawer", "door", "window",
            # Electronics
            "phone", "laptop", "computer", "tablet", "remote",
            "tv", "television", "monitor", "keyboard", "mouse",
            # Office items
            "book", "pen", "pencil", "paper", "folder", "box",
            "bag", "backpack", "notebook", "magazine",
            # Toys and misc
            "ball", "toy", "game", "puzzle", "block", "cube",
            # People/body parts
            "person", "hand", "face", "arm", "head"
        }

        # Location vocabulary
        self.locations = {
            "kitchen", "living room", "bedroom", "bathroom",
            "office", "garage", "hallway", "entrance", "exit",
            "corner", "center", "middle", "side", "edge",
            "floor", "ground", "wall", "ceiling",
            "left", "right", "front", "back", "top", "bottom"
        }

        # Color attributes
        self.colors = {
            "red", "blue", "green", "yellow", "orange", "purple",
            "pink", "brown", "black", "white", "gray", "grey",
            "gold", "silver", "beige", "tan"
        }

        # Size attributes
        self.sizes = {
            "small", "tiny", "little", "mini", "miniature",
            "big", "large", "huge", "giant", "massive",
            "medium", "regular", "normal", "standard"
        }

        # Material attributes
        self.materials = {
            "plastic", "metal", "glass", "wood", "wooden",
            "paper", "cardboard", "fabric", "cloth", "leather",
            "ceramic", "porcelain", "rubber", "foam"
        }

        # State attributes
        self.states = {
            "open", "closed", "empty", "full", "half",
            "on", "off", "broken", "working", "new", "old",
            "clean", "dirty", "wet", "dry", "hot", "cold"
        }

    def extract(self, tokens: List[str], raw_text: str) -> List[Entity]:
        """
        Extract entities from tokens.

        Args:
            tokens: List of tokens
            raw_text: Original raw text

        Returns:
            List of extracted entities
        """
        entities = []
        text_lower = raw_text.lower()

        # Track found positions to avoid duplicates
        found_spans: Set[Tuple[int, int]] = set()

        # Extract objects
        for obj in self.objects:
            if obj in tokens or obj in text_lower:
                idx = text_lower.find(obj)
                if idx >= 0:
                    span = (idx, idx + len(obj))
                    if span not in found_spans:
                        found_spans.add(span)

                        # Look for preceding attributes
                        attrs = self._find_preceding_attributes(text_lower, idx)

                        entities.append(Entity(
                            text=obj,
                            entity_type="object",
                            normalized=obj,
                            start_idx=idx,
                            end_idx=idx + len(obj),
                            attributes=attrs
                        ))

        # Extract locations
        for loc in self.locations:
            if loc in text_lower:
                idx = text_lower.find(loc)
                if idx >= 0:
                    span = (idx, idx + len(loc))
                    if span not in found_spans:
                        found_spans.add(span)
                        entities.append(Entity(
                            text=loc,
                            entity_type="location",
                            normalized=loc,
                            start_idx=idx,
                            end_idx=idx + len(loc)
                        ))

        # Extract standalone attributes (for queries like "the red one")
        for color in self.colors:
            if color in tokens:
                # Check if not already associated with an object
                idx = text_lower.find(color)
                if idx >= 0:
                    # Check if "one" or "thing" follows
                    after = text_lower[idx + len(color):].strip()
                    if after.startswith("one") or after.startswith("thing"):
                        entities.append(Entity(
                            text=f"{color} one",
                            entity_type="object",
                            normalized="object",
                            start_idx=idx,
                            end_idx=idx + len(color) + 4,
                            attributes={"color": color}
                        ))

        # Sort by position
        entities.sort(key=lambda e: e.start_idx)

        return entities

    def _find_preceding_attributes(self, text: str, obj_idx: int) -> Dict[str, Any]:
        """Find attributes preceding an object mention."""
        attrs = {}

        # Look at text before the object
        prefix = text[:obj_idx].split()[-3:] if obj_idx > 0 else []

        for word in prefix:
            if word in self.colors:
                attrs["color"] = word
            elif word in self.sizes:
                attrs["size"] = word
            elif word in self.materials:
                attrs["material"] = word
            elif word in self.states:
                attrs["state"] = word

        return attrs


class SpatialRelationExtractor:
    """Extracts spatial relations from text."""

    def __init__(self):
        """Initialize spatial relation extractor."""
        # Preposition to relation mappings
        self.preposition_map = {
            "on": "on_top_of",
            "onto": "on_top_of",
            "on top of": "on_top_of",
            "under": "below",
            "underneath": "below",
            "beneath": "below",
            "below": "below",
            "above": "above",
            "over": "above",
            "next to": "beside",
            "beside": "beside",
            "by": "beside",
            "near": "near",
            "close to": "near",
            "in front of": "in_front",
            "before": "in_front",
            "behind": "behind",
            "in back of": "behind",
            "inside": "inside",
            "in": "inside",
            "into": "inside",
            "within": "inside",
            "outside": "outside",
            "out of": "outside",
            "to": "towards",
            "toward": "towards",
            "towards": "towards",
            "from": "from",
            "away from": "away_from",
            "between": "between",
            "among": "among",
            "against": "against",
            "around": "around",
            "through": "through",
            "across": "across",
            "along": "along"
        }

    def extract(self, text: str, entities: List[Entity]) -> List[SpatialRelation]:
        """
        Extract spatial relations from text.

        Args:
            text: Original text
            entities: Extracted entities

        Returns:
            List of spatial relations
        """
        relations = []
        text_lower = text.lower()

        # Sort prepositions by length (longer first) to match "on top of" before "on"
        sorted_preps = sorted(self.preposition_map.keys(), key=len, reverse=True)

        for prep in sorted_preps:
            if prep in text_lower:
                idx = text_lower.find(prep)
                prep_end = idx + len(prep)

                # Find subject (entity before preposition)
                subject = None
                for entity in entities:
                    if entity.end_idx <= idx:
                        subject = entity.text
                    else:
                        break

                # Find reference (entity after preposition)
                reference = None
                for entity in entities:
                    if entity.start_idx >= prep_end:
                        reference = entity.text
                        break

                if subject and reference:
                    relations.append(SpatialRelation(
                        relation_type=self.preposition_map[prep],
                        subject=subject,
                        reference=reference,
                        preposition=prep
                    ))

        return relations


class ModifierExtractor:
    """Extracts modifiers (adverbs, manner) from text."""

    def __init__(self):
        """Initialize modifier extractor."""
        # Speed modifiers
        self.speed_modifiers = {
            "quickly": "fast",
            "fast": "fast",
            "rapidly": "fast",
            "hurry": "fast",
            "slowly": "slow",
            "slow": "slow",
            "carefully": "slow",
            "gently": "slow"
        }

        # Manner modifiers
        self.manner_modifiers = {
            "carefully": "careful",
            "gently": "gentle",
            "softly": "gentle",
            "firmly": "firm",
            "tightly": "tight",
            "loosely": "loose",
            "quietly": "quiet",
            "silently": "quiet"
        }

        # Degree modifiers
        self.degree_modifiers = {
            "very": 1.5,
            "really": 1.5,
            "extremely": 2.0,
            "slightly": 0.5,
            "a little": 0.5,
            "a bit": 0.5,
            "somewhat": 0.7
        }

        # Negation words
        self.negation_words = {
            "not", "don't", "do not", "doesn't", "does not",
            "no", "never", "none", "nothing", "neither"
        }

    def extract(self, tokens: List[str], text: str) -> Dict[str, Any]:
        """
        Extract modifiers from text.

        Args:
            tokens: List of tokens
            text: Original text

        Returns:
            Dictionary of modifiers
        """
        modifiers = {}
        text_lower = text.lower()

        # Check speed
        for word, speed in self.speed_modifiers.items():
            if word in tokens or word in text_lower:
                modifiers["speed"] = speed
                break

        # Check manner
        for word, manner in self.manner_modifiers.items():
            if word in tokens or word in text_lower:
                modifiers["manner"] = manner
                break

        # Check degree
        for word, degree in self.degree_modifiers.items():
            if word in text_lower:
                modifiers["degree"] = degree
                break

        # Check negation
        modifiers["negation"] = any(neg in text_lower for neg in self.negation_words)

        return modifiers


class VerbExtractor:
    """Extracts action verbs from commands."""

    def __init__(self):
        """Initialize verb extractor."""
        # Action verbs by category
        self.action_verbs = {
            # Manipulation verbs
            "pick", "grab", "grasp", "take", "get", "fetch",
            "put", "place", "set", "drop", "leave", "release",
            "push", "pull", "drag", "slide", "lift", "lower",
            "hold", "carry", "throw", "catch", "hand", "give",
            # Navigation verbs
            "go", "walk", "move", "navigate", "come", "approach",
            "follow", "lead", "turn", "step", "run", "head",
            "return", "enter", "exit", "climb", "descend",
            # Observation verbs
            "look", "see", "find", "locate", "search", "scan",
            "observe", "watch", "check", "inspect", "examine",
            # Interaction verbs
            "open", "close", "shut", "turn", "switch", "press",
            "rotate", "twist", "flip", "toggle", "activate",
            # Communication verbs
            "say", "speak", "tell", "wave", "greet", "point",
            "show", "nod", "shake", "signal",
            # Control verbs
            "stop", "halt", "freeze", "pause", "wait", "hold",
            "continue", "resume", "cancel", "start", "begin"
        }

    def extract(self, tokens: List[str]) -> Optional[str]:
        """
        Extract primary action verb from tokens.

        Args:
            tokens: List of tokens

        Returns:
            Action verb or None
        """
        for token in tokens:
            if token in self.action_verbs:
                return token
        return None


class NaturalLanguageParser:
    """
    Complete Natural Language Parser for Robot Commands

    Combines tokenization, intent classification, entity extraction,
    spatial relation extraction, and modifier extraction.
    """

    def __init__(self):
        """Initialize parser components."""
        self.tokenizer = Tokenizer()
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.spatial_extractor = SpatialRelationExtractor()
        self.modifier_extractor = ModifierExtractor()
        self.verb_extractor = VerbExtractor()

    def parse(self, text: str) -> ParsedCommand:
        """
        Parse natural language command into structured form.

        Args:
            text: Natural language command string

        Returns:
            ParsedCommand with all extracted information
        """
        # Tokenize
        tokens = self.tokenizer.tokenize(text)

        # Classify intent
        intent, intent_confidence = self.intent_classifier.classify(tokens)

        # Extract action verb
        action_verb = self.verb_extractor.extract(tokens)

        # Extract entities
        entities = self.entity_extractor.extract(tokens, text)

        # Extract spatial relations
        spatial_relations = self.spatial_extractor.extract(text, entities)

        # Extract modifiers
        modifiers = self.modifier_extractor.extract(tokens, text)

        # Determine target and destination entities
        target_entity = None
        destination_entity = None

        object_entities = [e for e in entities if e.entity_type == "object"]
        location_entities = [e for e in entities if e.entity_type == "location"]

        if object_entities:
            target_entity = object_entities[0]
            if len(object_entities) > 1:
                destination_entity = object_entities[1]

        if location_entities and destination_entity is None:
            destination_entity = location_entities[0]

        # Check for question type
        question_type = None
        text_lower = text.lower()
        if text_lower.startswith("what") or "what is" in text_lower:
            question_type = "what"
        elif text_lower.startswith("where") or "where is" in text_lower:
            question_type = "where"
        elif "how many" in text_lower or "how much" in text_lower:
            question_type = "how_many"
        elif text_lower.startswith("which"):
            question_type = "which"
        elif text_lower.startswith("is ") or text_lower.startswith("are "):
            question_type = "yes_no"

        # Calculate overall confidence
        confidence = intent_confidence
        if action_verb:
            confidence = min(1.0, confidence + 0.1)
        if target_entity:
            confidence = min(1.0, confidence + 0.1)

        return ParsedCommand(
            raw_text=text,
            intent=intent,
            action_verb=action_verb,
            entities=entities,
            target_entity=target_entity,
            destination_entity=destination_entity,
            spatial_relations=spatial_relations,
            modifiers=modifiers,
            negation=modifiers.get("negation", False),
            question_type=question_type,
            confidence=confidence
        )

    def parse_batch(self, texts: List[str]) -> List[ParsedCommand]:
        """
        Parse multiple commands.

        Args:
            texts: List of command strings

        Returns:
            List of ParsedCommand objects
        """
        return [self.parse(text) for text in texts]


def format_parsed_command(parsed: ParsedCommand) -> str:
    """Format parsed command for display."""
    lines = [
        f"Raw: \"{parsed.raw_text}\"",
        f"Intent: {parsed.intent.value} (confidence: {parsed.confidence:.2f})",
        f"Action Verb: {parsed.action_verb or 'None'}"
    ]

    if parsed.entities:
        lines.append("Entities:")
        for entity in parsed.entities:
            attr_str = f" {entity.attributes}" if entity.attributes else ""
            lines.append(f"  - {entity.text} ({entity.entity_type}){attr_str}")

    if parsed.target_entity:
        lines.append(f"Target: {parsed.target_entity.text}")

    if parsed.destination_entity:
        lines.append(f"Destination: {parsed.destination_entity.text}")

    if parsed.spatial_relations:
        lines.append("Spatial Relations:")
        for rel in parsed.spatial_relations:
            lines.append(f"  - {rel.subject} {rel.relation_type} {rel.reference}")

    if parsed.modifiers:
        mod_str = ", ".join(f"{k}={v}" for k, v in parsed.modifiers.items())
        lines.append(f"Modifiers: {mod_str}")

    if parsed.question_type:
        lines.append(f"Question Type: {parsed.question_type}")

    return "\n".join(lines)


def main():
    """Test natural language parser."""
    print("Testing Natural Language Parser")
    print("=" * 60)

    parser = NaturalLanguageParser()

    test_commands = [
        "Pick up the red cup from the table",
        "Go to the kitchen quickly",
        "Put the bottle on the shelf carefully",
        "Open the door",
        "Find the blue ball near the couch",
        "Where is the phone?",
        "Don't touch the hot plate",
        "Wave to the person",
        "How many cups are on the table?",
        "Move the large box to the corner slowly",
        "Stop immediately",
        "Look for my keys"
    ]

    for command in test_commands:
        print(f"\n{'='*60}")
        parsed = parser.parse(command)
        print(format_parsed_command(parsed))

    print(f"\n{'='*60}")
    print("Natural Language Parser tests completed!")


if __name__ == "__main__":
    main()
