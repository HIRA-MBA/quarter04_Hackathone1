# Physical AI & Humanoid Robotics: Embodied Intelligence

An interactive, open-source textbook teaching physical AI and humanoid robotics from ROS 2 fundamentals through advanced Vision-Language-Action models.

## Overview

This project delivers a comprehensive learning experience combining:

- **14 Chapters** across 4 modules covering ROS 2, Digital Twins, NVIDIA Isaac, and VLA models
- **Hands-on Labs** with reproducible exercises for each chapter
- **RAG-Powered Chatbot** providing contextual help from book content
- **Multi-language Support** (English and Urdu)

## Quick Start

See the [Quickstart Guide](specs/001-physical-ai-textbook/quickstart.md) for detailed setup instructions.

### Prerequisites

- Node.js 18+ (for Docusaurus frontend)
- Python 3.11+ (for FastAPI backend)
- Docker & Docker Compose (for local services)
- ROS 2 Humble (for lab exercises)

### Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd claude_hackathone

# Frontend (Docusaurus)
npm install
npm run start

# Backend (FastAPI)
cd backend
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
uvicorn app.main:app --reload

# Local services (Postgres, Qdrant)
docker-compose up -d
```

## Project Structure

```
├── docs/                    # Book content (Markdown/MDX)
│   ├── front-matter/        # Foreword, Preface, Introduction
│   ├── module-1-ros2/       # Chapters 1-5: ROS 2 Fundamentals
│   ├── module-2-digital-twin/ # Chapters 6-7: Simulation
│   ├── module-3-isaac/      # Chapters 8-10: NVIDIA Isaac
│   ├── module-4-vla/        # Chapters 11-14: VLA & Capstone
│   └── back-matter/         # Appendices, Glossary, Index
├── labs/                    # Hands-on lab exercises
├── backend/                 # FastAPI backend for RAG chatbot
├── src/                     # Docusaurus customizations
├── specs/                   # Feature specifications
└── history/                 # Prompt history records
```

## Modules

| Module | Chapters | Focus |
|--------|----------|-------|
| 1. ROS 2 Fundamentals | 1-5 | Nodes, sensors, URDF, edge deployment |
| 2. Digital Twin | 6-7 | Gazebo, Unity integration |
| 3. NVIDIA Isaac | 8-10 | Isaac Sim, GPU perception, Sim2Real |
| 4. VLA & Capstone | 11-14 | Locomotion, manipulation, VLA models |

## Tech Stack

| Component | Technology |
|-----------|------------|
| Documentation | Docusaurus 3.x |
| Backend API | FastAPI |
| Vector Database | Qdrant |
| Relational Database | Neon Postgres |
| AI Integration | OpenAI Agents SDK |
| Frontend Deployment | GitHub Pages (manual upload) |
| Backend Deployment | Railway/Fly.io (separate) |

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
