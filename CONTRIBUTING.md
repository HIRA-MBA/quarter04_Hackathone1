# Contributing to Physical AI & Robotics Textbook

Thank you for your interest in contributing! This guide will help you get started.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Content Guidelines](#content-guidelines)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)

---

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:

- Be respectful and considerate
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Avoid discriminatory language or behavior

---

## Getting Started

### Prerequisites

- **Node.js 18+** - For Docusaurus frontend
- **Python 3.11+** - For FastAPI backend
- **Docker** - For local services (Postgres, Qdrant)
- **Git** - For version control

### Local Setup

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/physical-ai-textbook.git
cd physical-ai-textbook

# Install frontend dependencies
npm install

# Set up backend
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cd ..

# Start local services
docker-compose up -d

# Start development servers
npm run start          # Frontend at http://localhost:3000
cd backend && uvicorn app.main:app --reload  # Backend at http://localhost:8000
```

### Environment Setup

Create `.env` files from examples:

```bash
cp .env.example .env
cp backend/.env.example backend/.env
```

---

## Development Workflow

### Branch Strategy

- `main` - Production-ready code
- `develop` - Integration branch for features
- `feature/*` - New features
- `fix/*` - Bug fixes
- `docs/*` - Documentation updates

### Workflow Steps

1. **Fork** the repository
2. **Clone** your fork locally
3. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make changes** following the code standards
5. **Test** your changes:
   ```bash
   npm run typecheck
   npm run lint
   npm run build
   cd backend && pytest
   ```
6. **Commit** with descriptive messages:
   ```bash
   git commit -m "feat: add new chapter on sensor fusion"
   ```
7. **Push** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
8. **Open a Pull Request** against `main`

### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation only
- `style` - Code style (formatting, etc.)
- `refactor` - Code refactoring
- `test` - Adding tests
- `chore` - Maintenance tasks

**Examples:**
```
feat(chapter): add chapter 15 on multi-robot coordination
fix(chatbot): resolve memory leak in RAG context
docs(readme): update installation instructions
```

---

## Code Standards

### Frontend (TypeScript/React)

```typescript
// ✅ Good: Type annotations, JSDoc, functional component
/**
 * ChatMessage displays a single message in the chat interface.
 * @param message - The message content and metadata
 * @param isUser - Whether this message is from the user
 */
export function ChatMessage({ message, isUser }: ChatMessageProps): JSX.Element {
  const [isExpanded, setIsExpanded] = useState(false);
  // ...
}

// ❌ Bad: No types, no documentation
export function ChatMessage(props) {
  var expanded = false;
  // ...
}
```

**Rules:**
- Use TypeScript strict mode (`strict: true`)
- Follow ESLint configuration
- Use functional components with hooks
- Add JSDoc comments for public APIs
- Prefer named exports over default exports
- Use CSS Modules for styling

### Backend (Python/FastAPI)

```python
# ✅ Good: Type hints, docstring, error handling
async def get_chat_response(
    query: str,
    context: list[str],
    *,
    max_tokens: int = 500,
) -> ChatResponse:
    """
    Generate a chat response using RAG.

    Args:
        query: The user's question
        context: Retrieved context passages
        max_tokens: Maximum response length

    Returns:
        ChatResponse with generated answer and sources

    Raises:
        OpenAIError: If the API request fails
    """
    if not query.strip():
        raise ValueError("Query cannot be empty")
    # ...


# ❌ Bad: No types, no docstring
def get_chat_response(query, context, max_tokens=500):
    return openai.chat(query)
```

**Rules:**
- Follow PEP 8 style guide
- Use type hints for all functions
- Format with Ruff: `ruff format app/`
- Lint with Ruff: `ruff check app/`
- Add docstrings (Google style)
- Use async/await for I/O operations

### Documentation (Markdown)

```markdown
<!-- ✅ Good: Structured chapter with all required sections -->
---
sidebar_position: 1
title: "Chapter 1: Welcome to ROS 2"
description: Introduction to ROS 2 fundamentals
---

# Welcome to ROS 2

[Introduction paragraph...]

## Learning Objectives

By the end of this chapter, you will be able to:
- [ ] Create a ROS 2 node
- [ ] Understand publisher/subscriber patterns

## Prerequisites

- Completed Chapter 0: Setup
- Basic Python knowledge

## Section 1: Your First Node

[Content with code examples...]

## Summary

Key takeaways:
- Point 1
- Point 2

## Lab Exercise

Complete Lab 1 in `/labs/module-1/ch01-hello-robot/`
```

**Rules:**
- Follow the chapter template structure
- Include learning objectives (measurable, action-oriented)
- Add code examples with expected output
- Use Mermaid for diagrams
- Keep within word count limits

---

## Content Guidelines

### Chapter Word Counts

| Type | Word Count | Sections |
|------|------------|----------|
| Regular chapter | 1700-2500 | 4-5 |
| Capstone chapter | 3000-3500 | 6-8 |
| Front matter | 500-2000 | Varies |
| Appendix | 600-2000 | Varies |

### Required Chapter Sections

1. **Frontmatter** - Title, description, sidebar position
2. **Learning Objectives** - 3-5 measurable objectives
3. **Prerequisites** - Required knowledge/chapters
4. **Main Content** - 2-4 sections with examples
5. **Summary** - Key takeaways
6. **Lab Exercise** - Link to hands-on lab

### Code Examples

Always include:
- Language identifier in code blocks
- Comments explaining key parts
- Expected output where applicable

```python
# Example: Creating a ROS 2 publisher
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        # Timer fires every 0.5 seconds
        self.timer = self.create_timer(0.5, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World'
        self.publisher_.publish(msg)
```

### Lab Exercise Structure

```
labs/module-X/chXX-name/
├── README.md           # Quick overview
├── INSTRUCTIONS.md     # Detailed steps
├── package.xml         # ROS 2 package manifest
├── setup.py            # Python package setup
└── src/                # Starter code
```

**INSTRUCTIONS.md must include:**
- Learning objectives with point values
- Software/hardware prerequisites
- Step-by-step instructions with verification
- Expected output for each step
- Troubleshooting table
- Grading rubric

---

## Testing

### Frontend Tests

```bash
# Type checking
npm run typecheck

# Linting
npm run lint

# Build test
npm run build
```

### Backend Tests

```bash
cd backend

# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_chat.py -v
```

### Content Validation

```bash
# Validate chapter content
node scripts/validate-content.js

# Validate internal links
node scripts/validate-links.js
```

---

## Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated if needed
- [ ] Commit messages follow convention
- [ ] No merge conflicts with `main`

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring

## Testing
How were the changes tested?

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No breaking changes
```

### Review Process

1. Automated CI checks must pass
2. At least one maintainer approval required
3. All conversations must be resolved
4. Branch must be up-to-date with `main`

---

## Issue Guidelines

### Bug Reports

Include:
- Clear, descriptive title
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Node version, etc.)
- Screenshots/logs if applicable

### Feature Requests

Include:
- Clear description of the feature
- Use case / motivation
- Proposed implementation (optional)
- Alternatives considered

### Questions

- Check existing issues and documentation first
- Use the Discussions tab for general questions
- Tag with `question` label

---

## Recognition

Contributors will be recognized in:
- The Contributors section of README.md
- Release notes for significant contributions
- The project's contributor graph

Thank you for contributing to the Physical AI & Robotics Textbook! 🤖
