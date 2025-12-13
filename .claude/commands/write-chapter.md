# Write Chapter

Generate textbook chapter content following the Physical AI Textbook structure.

## Usage
```
/write-chapter <chapter-id> [word-count]
```

## Template Structure

```markdown
---
sidebar_position: {{POSITION}}
title: {{TITLE}}
description: {{DESCRIPTION}}
---

# {{TITLE}}

[Introduction paragraph - 100-150 words]

## Learning Objectives
- Objective 1
- Objective 2
- Objective 3

## Prerequisites
- Prerequisite 1
- Prerequisite 2

## Section 1: [Topic]
[Content with code examples]

## Section 2: [Topic]
[Content with diagrams using Mermaid]

## Hands-On Exercise
[Step-by-step instructions]

## Summary
[Key takeaways]

## Lab Exercise
Complete the hands-on lab in [`labs/module-X/chXX-name/`](https://github.com/physical-ai-textbook/physical-ai-textbook/tree/main/labs/module-X/chXX-name)

## Further Reading
- Resource 1
- Resource 2
```

## Word Count Guidelines
- Regular chapters: 1700-2500 words
- Capstone chapters: 3000-3500 words

## Code Example Format
```python
# Always include comments
# Use type hints
def example_function(param: str) -> bool:
    """Docstring with description."""
    pass
```
