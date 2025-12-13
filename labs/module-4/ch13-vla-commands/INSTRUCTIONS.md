# Lab 13: Vision-Language-Action Models

## Overview

In this lab, you will implement and experiment with Vision-Language-Action (VLA) models for natural language robot control. You will learn how to parse natural language commands, integrate visual perception, and generate executable robot actions.

## Learning Objectives

By completing this lab, you will be able to:

1. Understand VLA model architecture and components
2. Parse natural language commands into structured representations
3. Extract entities, intents, and spatial relations from text
4. Map visual observations to action spaces
5. Generate action sequences from multimodal inputs

## Prerequisites

- Completed Labs 11-12 (locomotion and manipulation)
- Python 3.10+
- Basic understanding of transformers and language models
- PyTorch experience recommended

## Lab Duration

Estimated time: 3-4 hours

---

## Part 1: Understanding the VLA Architecture (30 minutes)

### 1.1 VLA Model Components

A Vision-Language-Action model consists of three main components:

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────┐
│  Vision Encoder │     │ Language Encoder │     │ Action Decoder │
│    (Images)     │     │    (Commands)    │     │   (Actions)    │
└────────┬────────┘     └────────┬─────────┘     └───────▲────────┘
         │                       │                       │
         ▼                       ▼                       │
    ┌────────────────────────────────────────────────────┤
    │              Multimodal Fusion Layer               │
    └────────────────────────────────────────────────────┘
```

### 1.2 Explore the Code Structure

Open and review the lab files:

```bash
# Navigate to lab directory
cd labs/module-4/ch13-vla-commands

# Review the VLA agent implementation
cat vla_agent.py

# Review the language parser
cat language_parser.py
```

**Exercise 1.1**: Read through `vla_agent.py` and identify:
- [ ] The three main encoder/decoder classes
- [ ] How visual features are extracted
- [ ] How language commands are parsed
- [ ] How actions are generated

---

## Part 2: Natural Language Parsing (45 minutes)

### 2.1 Running the Language Parser

```python
from language_parser import NaturalLanguageParser, format_parsed_command

# Create parser
parser = NaturalLanguageParser()

# Test a command
command = "Pick up the red cup from the table"
parsed = parser.parse(command)
print(format_parsed_command(parsed))
```

### 2.2 Understanding Parse Results

The parser extracts:

| Component | Description | Example |
|-----------|-------------|---------|
| **Intent** | High-level goal | `manipulation`, `navigation` |
| **Action Verb** | Primary action | `pick`, `go`, `open` |
| **Entities** | Objects/locations | `cup`, `table`, `kitchen` |
| **Spatial Relations** | Object relationships | `on`, `next_to`, `inside` |
| **Modifiers** | Manner/speed | `carefully`, `quickly` |

**Exercise 2.1**: Parse these commands and analyze the output:

```python
test_commands = [
    "Go to the kitchen",
    "Put the bottle on the shelf carefully",
    "Find the blue ball near the couch",
    "Don't touch the hot plate",
    "Where is my phone?"
]

for cmd in test_commands:
    parsed = parser.parse(cmd)
    print(f"\nCommand: {cmd}")
    print(f"Intent: {parsed.intent.value}")
    print(f"Verb: {parsed.action_verb}")
    print(f"Target: {parsed.target_entity.text if parsed.target_entity else 'None'}")
```

### 2.3 Entity Extraction Deep Dive

**Exercise 2.2**: Extend the entity extractor with new objects:

```python
# In language_parser.py, add to EntityExtractor.objects:
# Add 5 new objects relevant to your use case
# Example: "screwdriver", "hammer", "wrench", etc.

# Test your additions:
parser = NaturalLanguageParser()
result = parser.parse("Hand me the screwdriver from the toolbox")
print(result.entities)
```

---

## Part 3: Vision-Language Integration (60 minutes)

### 3.1 Creating Visual Observations

```python
import numpy as np
from vla_agent import VLAAgent, VisualObservation

# Create agent
agent = VLAAgent(embedding_dim=512)

# Create a simulated camera observation
rgb_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
depth_image = np.random.uniform(0.5, 3.0, (480, 640))

observation = VisualObservation(
    rgb_image=rgb_image,
    depth_image=depth_image,
    timestamp=0.0
)
```

### 3.2 Scene Perception

```python
# Process the observation
scene = agent.perceive(observation)

# Examine detected objects
print(f"Scene: {scene.scene_description}")
print(f"\nDetected objects:")
for obj in scene.objects:
    print(f"  - {obj.label}: {obj.attributes}")

print(f"\nAffordances:")
for affordance, objects in scene.affordances.items():
    print(f"  {affordance}: {objects}")
```

**Exercise 3.1**: Modify the `SimulatedVisionEncoder.detect_objects()` method to:
- [ ] Return objects based on image brightness (darker = fewer objects)
- [ ] Add position variance based on depth values
- [ ] Include at least one graspable object when depth < 1.0m

### 3.3 Multimodal Embedding

```python
# Get fused embedding
embedding = agent.get_multimodal_embedding(observation, "pick up the cup")

print(f"Embedding shape: {embedding.shape}")
print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")

# Compare embeddings for similar commands
emb1 = agent.get_multimodal_embedding(observation, "pick up the cup")
emb2 = agent.get_multimodal_embedding(observation, "grab the cup")
emb3 = agent.get_multimodal_embedding(observation, "go to the door")

# Compute similarity
similarity_12 = np.dot(emb1, emb2)
similarity_13 = np.dot(emb1, emb3)

print(f"Similarity (pick/grab): {similarity_12:.4f}")
print(f"Similarity (pick/go): {similarity_13:.4f}")
```

**Exercise 3.2**: The similar commands "pick up the cup" and "grab the cup" should have higher similarity. Verify this is true with your embeddings.

---

## Part 4: Action Generation (60 minutes)

### 4.1 Processing Commands

```python
# First perceive the scene
scene = agent.perceive(observation)

# Process a command
action_seq = agent.process_command("Pick up the red cup")

# View the result
explanation = agent.explain_action(action_seq)
print(explanation)
```

### 4.2 Action Sequence Structure

Each `ActionSequence` contains:

```python
@dataclass
class ActionSequence:
    actions: List[ActionPrimitive]  # Individual actions
    language_command: str           # Original command
    reasoning: str                  # Explanation of decisions
    total_duration: float           # Estimated time
    success_probability: float      # Confidence
```

Each `ActionPrimitive` has:

```python
@dataclass
class ActionPrimitive:
    action_type: ActionType         # PICK, PLACE, NAVIGATE, etc.
    target_object: Optional[str]    # Object name
    target_position: Optional[np.ndarray]  # 3D position
    duration: float                 # Estimated duration
    parameters: Dict[str, Any]      # Additional params
```

### 4.3 Testing Multiple Commands

**Exercise 4.1**: Process these commands and analyze the generated actions:

```python
commands = [
    "Pick up the red cup",
    "Go to the table",
    "Put the bottle on the shelf",
    "Open the door carefully",
    "Wave hello",
    "Find the book"
]

for cmd in commands:
    print(f"\n{'='*50}")
    action_seq = agent.process_command(cmd)
    print(agent.explain_action(action_seq))
```

### 4.4 Implementing Custom Actions

**Exercise 4.2**: Add support for a new action type:

```python
# In vla_agent.py, add to ActionType enum:
# STACK = "stack"  # Stack objects

# In SimulatedActionDecoder, add handling for STACK:
# - Should generate: LOOK → PICK → NAVIGATE → PLACE sequence
# - Target should be first object, destination should be second object

# Test:
action_seq = agent.process_command("Stack the red block on the blue block")
print(agent.explain_action(action_seq))
```

---

## Part 5: Integration Exercise (45 minutes)

### 5.1 Complete Pipeline Test

Create a complete test scenario:

```python
import numpy as np
from vla_agent import VLAAgent, VisualObservation

def run_vla_pipeline(commands: list):
    """Run complete VLA pipeline with multiple commands."""
    agent = VLAAgent(embedding_dim=512)

    # Simulate camera feed
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    depth = np.random.uniform(0.5, 3.0, (480, 640))

    obs = VisualObservation(rgb_image=rgb, depth_image=depth)

    # Perceive scene
    scene = agent.perceive(obs)
    print(f"Scene: {scene.scene_description}\n")

    # Process commands sequentially
    total_time = 0.0
    for cmd in commands:
        print(f"Command: \"{cmd}\"")
        action_seq = agent.process_command(cmd)

        print(f"  Actions: {[a.action_type.value for a in action_seq.actions]}")
        print(f"  Duration: {action_seq.total_duration:.1f}s")
        print(f"  Success prob: {action_seq.success_probability:.0%}\n")

        total_time += action_seq.total_duration

    print(f"Total estimated time: {total_time:.1f}s")

# Test scenario
commands = [
    "Look for the cup",
    "Pick up the cup",
    "Go to the table",
    "Put the cup on the table",
    "Wave goodbye"
]

run_vla_pipeline(commands)
```

### 5.2 Error Handling

**Exercise 5.1**: Test how the system handles:

```python
# Ambiguous commands
agent.process_command("Get that thing")

# Unknown objects
agent.process_command("Pick up the zorblax")

# Conflicting instructions
agent.process_command("Quickly and carefully grab the cup")

# Questions (not commands)
agent.process_command("Where is the cup?")
```

---

## Part 6: Advanced Exercises (Optional)

### 6.1 Implement Confidence-Based Clarification

When confidence is low, the agent should ask for clarification:

```python
def process_with_clarification(agent, command):
    action_seq = agent.process_command(command)

    if action_seq.success_probability < 0.5:
        # Generate clarification request
        if action_seq.actions[0].target_object is None:
            return "I didn't understand which object. Can you specify?"
        # Add more clarification logic...

    return action_seq
```

### 6.2 Implement Command Chaining

Parse compound commands like:
- "Pick up the cup and put it on the table"
- "Go to the kitchen, then find the bottle"

```python
def parse_compound_command(text):
    """Split compound commands into individual commands."""
    # Split on "and", "then", ","
    connectors = [" and ", " then ", ", "]
    # Implement splitting logic...
    pass
```

### 6.3 Context-Aware Parsing

Implement pronoun resolution:
- User: "Pick up the red cup"
- User: "Now put it on the table"  # "it" refers to the cup

---

## Verification Checklist

Before completing this lab, verify:

- [ ] Language parser correctly identifies intents for all test commands
- [ ] Entity extraction finds objects and locations
- [ ] Spatial relations are extracted correctly
- [ ] VLA agent generates appropriate action sequences
- [ ] Multimodal embeddings show expected similarity patterns
- [ ] Custom action type implemented and tested
- [ ] Error handling works for edge cases

## Expected Output

Running the complete pipeline should produce output similar to:

```
Scene: Scene contains 4 objects: cup, table, bottle, chair

Command: "Pick up the red cup"
  Actions: ['look', 'pick']
  Duration: 3.5s
  Success prob: 90%

Command: "Put the cup on the table"
  Actions: ['place']
  Duration: 2.0s
  Success prob: 85%

Total estimated time: 5.5s
```

## Troubleshooting

### Common Issues

1. **No objects detected**: Check that the image array has the correct shape and dtype
2. **Intent always UNKNOWN**: Verify the command contains recognized action verbs
3. **Empty action sequences**: Ensure scene perception ran before processing commands
4. **Low confidence scores**: Check entity extraction is finding target objects

### Debug Tips

```python
# Enable verbose output
import logging
logging.basicConfig(level=logging.DEBUG)

# Print intermediate results
parsed = parser.parse(command)
print(f"Tokens: {parser.tokenizer.tokenize(command)}")
print(f"Raw entities: {parsed.entities}")
```

## Next Steps

After completing this lab:
1. Proceed to Lab 14 (Capstone) to integrate VLA with the complete humanoid system
2. Experiment with real vision models (CLIP, DINOv2) instead of simulated encoders
3. Try connecting to ROS 2 for real robot control

## References

- [RT-2: Vision-Language-Action Models](https://robotics-transformer2.github.io/)
- [PaLM-E: An Embodied Multimodal Language Model](https://palm-e.github.io/)
- [CLIP: Learning Transferable Visual Models](https://openai.com/research/clip)
