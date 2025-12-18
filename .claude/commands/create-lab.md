# Create Lab

Generate comprehensive lab exercise files for the Physical AI Textbook chapters.

## Arguments
- `$ARGUMENTS` - `<module> <chapter> <topic> [--difficulty basic|intermediate|advanced]`

## Usage
```
/create-lab 1 ch01 hello-robot
/create-lab 2 ch06 gazebo-world --difficulty intermediate
/create-lab 4 ch14 capstone-humanoid --difficulty advanced
```

## Output Structure
```
labs/module-{{MODULE}}/ch{{CHAPTER}}-{{topic}}/
├── README.md              # Lab overview and quick start
├── INSTRUCTIONS.md        # Detailed step-by-step guide
├── SOLUTIONS.md           # Instructor solutions (hidden by default)
├── package.xml            # ROS 2 package manifest
├── setup.py               # Python package setup
├── setup.cfg              # Package configuration
├── resource/              # Package marker
│   └── {{package_name}}
├── config/
│   └── params.yaml        # Default parameters
├── {{package_name}}/
│   ├── __init__.py
│   └── {{main_file}}.py   # Starter code with TODOs
├── launch/
│   └── {{lab_name}}_launch.py
├── test/
│   ├── test_{{main_file}}.py
│   └── test_integration.py
├── docker/
│   └── Dockerfile         # Lab-specific container
└── .devcontainer/
    └── devcontainer.json  # VS Code dev container
```

---

## README.md Template
```markdown
# Lab {{CHAPTER}}: {{Title}}

[![Difficulty](https://img.shields.io/badge/difficulty-{{difficulty}}-{{color}}.svg)]()
[![Time](https://img.shields.io/badge/time-{{time_estimate}}-blue.svg)]()
[![ROS 2](https://img.shields.io/badge/ROS%202-Jazzy-brightgreen.svg)]()

## Overview

{{brief_description}}

**What you'll learn:**
- Learning outcome 1
- Learning outcome 2
- Learning outcome 3

**What you'll build:**
{{deliverable_description}}

## Quick Start

### Option 1: Docker (Recommended)
\`\`\`bash
# Pull and run the lab container
docker compose up -d
docker exec -it {{container_name}} bash

# Inside container
ros2 launch {{package_name}} {{lab_name}}_launch.py
\`\`\`

### Option 2: Local Setup
\`\`\`bash
# Clone and build
cd ~/ros2_ws/src
cp -r /path/to/labs/module-{{MODULE}}/ch{{CHAPTER}}-{{topic}} .
cd ~/ros2_ws
colcon build --packages-select {{package_name}}
source install/setup.bash
\`\`\`

## Prerequisites

| Requirement | Version | Check Command |
|-------------|---------|---------------|
| ROS 2 | Jazzy | `ros2 --version` |
| Python | 3.10+ | `python3 --version` |
| Chapter {{PREV}} | Complete | - |

## Files to Modify

| File | Purpose | Difficulty |
|------|---------|------------|
| `{{package_name}}/{{main_file}}.py` | Main implementation | ⭐⭐ |
| `config/params.yaml` | Configuration | ⭐ |
| `launch/{{lab_name}}_launch.py` | Launch setup | ⭐ |

## Validation

Run the test suite to verify your implementation:

\`\`\`bash
# Unit tests
pytest test/test_{{main_file}}.py -v

# Integration tests (requires running simulation)
pytest test/test_integration.py -v --timeout=60
\`\`\`

## Submission

1. Ensure all tests pass
2. Record a demo video (optional for bonus points)
3. Submit via: [Submission Portal Link]

## Need Help?

- 📖 Review [Chapter {{CHAPTER}}](/docs/module-{{MODULE}}-*/ch{{CHAPTER}}-*.md)
- 💬 Ask the [RAG Chatbot](/chat) about this lab
- 🐛 Check [Troubleshooting](#troubleshooting) below
```

---

## INSTRUCTIONS.md Template
```markdown
# Lab Instructions: {{Title}}

**Estimated Time:** {{time_estimate}}
**Difficulty:** {{difficulty}} {{difficulty_stars}}
**Chapter Reference:** [Chapter {{CHAPTER}}](/docs/module-{{MODULE}}-*/ch{{CHAPTER}}-*.md)

---

## Learning Objectives

By completing this lab, you will be able to:

- [ ] **LO1:** {{objective_1}} *({{points_1}} points)*
- [ ] **LO2:** {{objective_2}} *({{points_2}} points)*
- [ ] **LO3:** {{objective_3}} *({{points_3}} points)*
- [ ] **LO4:** {{objective_4}} *({{points_4}} points)*

**Total Points:** {{total_points}}

---

## Prerequisites

### Knowledge Requirements
- [ ] Understand {{concept_1}} (covered in Chapter {{ref_1}})
- [ ] Familiar with {{concept_2}} (covered in Chapter {{ref_2}})

### Software Requirements

| Software | Required Version | Installation Guide |
|----------|-----------------|-------------------|
| ROS 2 | Jazzy | [Install ROS 2](https://docs.ros.org/en/jazzy/Installation.html) |
| Python | 3.10+ | Pre-installed with ROS 2 |
| {{extra_dep}} | {{version}} | See [Appendix A](/docs/back-matter/appendix-a-student-kit.md) |

### Hardware Requirements (if applicable)
- [ ] {{hardware_1}}
- [ ] {{hardware_2}}

---

## Environment Setup

### Step 0: Verify Prerequisites

\`\`\`bash
# Verify ROS 2 installation
source /opt/ros/jazzy/setup.bash
ros2 --version
# Expected: ros2 version 0.x.x

# Verify Python packages
python3 -c "import rclpy; print('rclpy OK')"
python3 -c "import numpy; print('numpy OK')"
\`\`\`

### Step 1: Create Workspace (if not exists)

\`\`\`bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
colcon build
source install/setup.bash
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
\`\`\`

### Step 2: Copy Lab Files

\`\`\`bash
# Copy lab package to workspace
cp -r labs/module-{{MODULE}}/ch{{CHAPTER}}-{{topic}} ~/ros2_ws/src/{{package_name}}

# Build the package
cd ~/ros2_ws
colcon build --packages-select {{package_name}}
source install/setup.bash
\`\`\`

### Step 3: Verify Setup

\`\`\`bash
# List package executables
ros2 pkg executables {{package_name}}
# Expected output:
# {{package_name}} {{main_file}}
\`\`\`

---

## Part 1: {{Part1_Title}} ({{part1_points}} points)

**Goal:** {{part1_goal}}

### Background

{{part1_background_explanation}}

### Task 1.1: {{Task1_1_Title}}

**File to edit:** `{{package_name}}/{{main_file}}.py`

Open the file and locate the following TODO:

\`\`\`python
# TODO: Task 1.1 - {{task_description}}
# Your code here
pass
\`\`\`

**Instructions:**

1. {{instruction_1}}
2. {{instruction_2}}
3. {{instruction_3}}

**Hints:**
<details>
<summary>Click for Hint 1</summary>

{{hint_1}}

</details>

<details>
<summary>Click for Hint 2</summary>

{{hint_2}}

</details>

### Expected Output

After completing Task 1.1, run:

\`\`\`bash
ros2 run {{package_name}} {{main_file}}
\`\`\`

You should see:

\`\`\`
[INFO] [{{timestamp}}] [{{node_name}}]: {{expected_log_message}}
{{expected_output}}
\`\`\`

### Checkpoint 1.1 ✓

- [ ] Node starts without errors
- [ ] Log message appears as expected
- [ ] Test passes: `pytest test/test_{{main_file}}.py::test_task_1_1 -v`

---

### Task 1.2: {{Task1_2_Title}}

**File to edit:** `{{package_name}}/{{main_file}}.py`

{{task_1_2_instructions}}

### Expected Output

\`\`\`
{{expected_output_1_2}}
\`\`\`

### Checkpoint 1.2 ✓

- [ ] {{checkpoint_1_2_item_1}}
- [ ] {{checkpoint_1_2_item_2}}
- [ ] Test passes: `pytest test/test_{{main_file}}.py::test_task_1_2 -v`

---

## Part 2: {{Part2_Title}} ({{part2_points}} points)

**Goal:** {{part2_goal}}

### Task 2.1: {{Task2_1_Title}}

{{task_2_1_instructions}}

**Code Template:**

\`\`\`python
# Add this to your implementation
{{code_template}}
\`\`\`

### Expected Output

\`\`\`
{{expected_output_2_1}}
\`\`\`

### Checkpoint 2.1 ✓

- [ ] {{checkpoint_2_1}}
- [ ] Test passes: `pytest test/test_{{main_file}}.py::test_task_2_1 -v`

---

## Part 3: Integration & Testing ({{part3_points}} points)

**Goal:** Integrate all components and verify end-to-end functionality

### Task 3.1: Run Full System

\`\`\`bash
# Terminal 1: Launch the system
ros2 launch {{package_name}} {{lab_name}}_launch.py

# Terminal 2: Verify topics
ros2 topic list
ros2 topic echo /{{expected_topic}}

# Terminal 3: Send test command (if applicable)
ros2 topic pub /{{input_topic}} {{msg_type}} "{{test_data}}"
\`\`\`

### Task 3.2: Run Integration Tests

\`\`\`bash
# Run all tests
pytest test/ -v

# Run with coverage
pytest test/ --cov={{package_name}} --cov-report=html
\`\`\`

### Final Checkpoint ✓

- [ ] All unit tests pass ({{unit_test_count}} tests)
- [ ] Integration test passes
- [ ] System runs for 60 seconds without errors
- [ ] Expected behavior matches specification

---

## Grading Rubric

| Criterion | Points | Description |
|-----------|--------|-------------|
| Part 1: {{Part1_Title}} | {{part1_points}} | {{part1_criteria}} |
| Part 2: {{Part2_Title}} | {{part2_points}} | {{part2_criteria}} |
| Part 3: Integration | {{part3_points}} | All tests pass, system stable |
| Code Quality | {{quality_points}} | Type hints, docstrings, PEP 8 |
| **Total** | **{{total_points}}** | |

### Grade Scale
- **A (90-100%):** Excellent - all objectives met, clean code
- **B (80-89%):** Good - most objectives met, minor issues
- **C (70-79%):** Satisfactory - core objectives met
- **D (60-69%):** Needs improvement - partial completion
- **F (<60%):** Incomplete - significant work required

---

## Troubleshooting

### Common Issues

<details>
<summary><strong>Issue: "Package not found"</strong></summary>

**Symptoms:** `ros2 run {{package_name}} {{main_file}}` fails with "Package not found"

**Cause:** Package not built or workspace not sourced

**Solution:**
\`\`\`bash
cd ~/ros2_ws
colcon build --packages-select {{package_name}}
source install/setup.bash
\`\`\`
</details>

<details>
<summary><strong>Issue: "Import error"</strong></summary>

**Symptoms:** `ModuleNotFoundError: No module named '{{module}}'`

**Cause:** Missing dependency

**Solution:**
\`\`\`bash
pip install {{module}}
# Or for ROS packages:
sudo apt install ros-jazzy-{{package}}
\`\`\`
</details>

<details>
<summary><strong>Issue: "Topic not publishing"</strong></summary>

**Symptoms:** `ros2 topic echo /{{topic}}` shows no messages

**Cause:** Publisher not initialized or callback not running

**Debugging steps:**
\`\`\`bash
# Check if node is running
ros2 node list

# Check node info
ros2 node info /{{node_name}}

# Check topic info
ros2 topic info /{{topic}} -v
\`\`\`
</details>

<details>
<summary><strong>Issue: "Test failures"</strong></summary>

**Symptoms:** `pytest` shows test failures

**Debugging:**
\`\`\`bash
# Run single test with verbose output
pytest test/test_{{main_file}}.py::test_name -v -s

# Check for syntax errors
python3 -m py_compile {{package_name}}/{{main_file}}.py
\`\`\`
</details>

### Error Reference Table

| Error Message | Likely Cause | Quick Fix |
|--------------|--------------|-----------|
| `RCLError: context is already initialized` | Multiple rclpy.init() | Remove duplicate init |
| `Cannot transform from X to Y` | TF not published | Check tf_broadcaster |
| `Timeout waiting for service` | Service not running | Start service node |
| `QoS incompatibility` | Mismatched QoS settings | Use compatible QoS |

---

## Bonus Challenges (Optional)

### Bonus 1: Performance Optimization (+{{bonus1_points}} points)

{{bonus_1_description}}

**Acceptance Criteria:**
- [ ] {{bonus_1_criterion_1}}
- [ ] {{bonus_1_criterion_2}}

### Bonus 2: Extended Functionality (+{{bonus2_points}} points)

{{bonus_2_description}}

**Acceptance Criteria:**
- [ ] {{bonus_2_criterion_1}}
- [ ] {{bonus_2_criterion_2}}

---

## Resources

### Documentation
- 📖 [ROS 2 {{topic}} Documentation](https://docs.ros.org/)
- 📖 [Chapter {{CHAPTER}} Reference](/docs/module-{{MODULE}}-*/ch{{CHAPTER}}-*.md)

### Related Labs
- ⬅️ Previous: [Lab {{PREV_CHAPTER}}](/labs/module-{{MODULE}}/ch{{PREV_CHAPTER}}-*)
- ➡️ Next: [Lab {{NEXT_CHAPTER}}](/labs/module-{{MODULE}}/ch{{NEXT_CHAPTER}}-*)

### Video Tutorials
- 🎥 [Lab Walkthrough]({{video_url}})
- 🎥 [Debugging Tips]({{debug_video_url}})
```

---

## SOLUTIONS.md Template (Instructor Only)

```markdown
# Solutions: Lab {{CHAPTER}} - {{Title}}

> ⚠️ **INSTRUCTOR USE ONLY** - Do not distribute to students

## Part 1 Solutions

### Task 1.1 Solution

\`\`\`python
# Complete solution for Task 1.1
{{solution_code_1_1}}
\`\`\`

**Explanation:** {{solution_explanation_1_1}}

### Task 1.2 Solution

\`\`\`python
{{solution_code_1_2}}
\`\`\`

## Part 2 Solutions

### Task 2.1 Solution

\`\`\`python
{{solution_code_2_1}}
\`\`\`

## Common Student Mistakes

1. **Mistake:** {{common_mistake_1}}
   **Fix:** {{fix_1}}

2. **Mistake:** {{common_mistake_2}}
   **Fix:** {{fix_2}}

## Grading Notes

- {{grading_note_1}}
- {{grading_note_2}}
```

---

## Test File Template

```python
#!/usr/bin/env python3
"""
test_{{main_file}}.py - Unit tests for Lab {{CHAPTER}}

Run: pytest test/test_{{main_file}}.py -v
"""
import pytest
import rclpy
from rclpy.node import Node
from {{package_name}}.{{main_file}} import {{ClassName}}


@pytest.fixture(scope="module")
def ros_context():
    """Initialize ROS 2 context."""
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node(ros_context):
    """Create node instance."""
    node = {{ClassName}}()
    yield node
    node.destroy_node()


class TestTask1:
    """Tests for Part 1 tasks."""

    def test_task_1_1_node_initialization(self, node):
        """Test that node initializes correctly."""
        assert node.get_name() == '{{node_name}}'
        # TODO: Add specific assertions

    def test_task_1_2_{{test_name}}(self, node):
        """Test {{test_description}}."""
        # TODO: Add test implementation
        pass


class TestTask2:
    """Tests for Part 2 tasks."""

    def test_task_2_1_{{test_name}}(self, node):
        """Test {{test_description}}."""
        pass


class TestIntegration:
    """Integration tests."""

    @pytest.mark.integration
    @pytest.mark.timeout(30)
    def test_full_system(self, node, ros_context):
        """Test complete system integration."""
        pass
```

---

## Difficulty Levels

| Level | Time | Points | Instructor Support |
|-------|------|--------|-------------------|
| **Basic** ⭐ | 45-60 min | 100 | Full scaffolding, detailed hints |
| **Intermediate** ⭐⭐ | 60-90 min | 150 | Partial scaffolding, some hints |
| **Advanced** ⭐⭐⭐ | 90-120 min | 200 | Minimal scaffolding, research required |

### Difficulty Guidelines

**Basic Labs:**
- Starter code with TODOs clearly marked
- Extensive hints available
- Step-by-step verification
- Single concept focus

**Intermediate Labs:**
- Partial starter code
- Limited hints
- Multi-concept integration
- Some independent problem-solving

**Advanced Labs:**
- Minimal starter code
- No hints (research required)
- Complex integration
- Open-ended bonus challenges

---

## Acceptance Criteria Checklist

Before marking a lab complete, verify:

- [ ] All TODO comments addressed
- [ ] All tests pass (`pytest test/ -v`)
- [ ] Code follows PEP 8 style
- [ ] Type hints on all functions
- [ ] Docstrings on classes and public methods
- [ ] No hardcoded values (use parameters)
- [ ] Proper error handling
- [ ] Clean git history (if version controlled)
