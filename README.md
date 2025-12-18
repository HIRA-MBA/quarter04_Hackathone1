# Physical AI & Humanoid Robotics: Embodied Intelligence

[![CI](https://github.com/[org]/physical-ai-textbook/actions/workflows/ci.yml/badge.svg)](https://github.com/[org]/physical-ai-textbook/actions/workflows/ci.yml)
[![Security](https://github.com/[org]/physical-ai-textbook/actions/workflows/security.yml/badge.svg)](https://github.com/[org]/physical-ai-textbook/actions/workflows/security.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Node.js](https://img.shields.io/badge/Node.js-18+-green.svg)](https://nodejs.org/)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)

An interactive, open-source textbook teaching physical AI and humanoid robotics from ROS 2 fundamentals through advanced Vision-Language-Action models.

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-modules">Modules</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

## Overview

This project delivers a comprehensive learning experience combining:

- **14 Chapters** across 4 modules covering ROS 2, Digital Twins, NVIDIA Isaac, and VLA models
- **Hands-on Labs** with reproducible exercises for each chapter
- **RAG-Powered Chatbot** providing contextual help from book content
- **Multi-language Support** (English and Urdu with RTL layout)
- **Personalized Learning** with progress tracking and recommendations

## Quick Start

### Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Node.js | 18+ | Docusaurus frontend |
| Python | 3.11+ | FastAPI backend |
| Docker | Latest | Local services |
| ROS 2 | Jazzy | Lab exercises |

### Installation

```bash
# Clone the repository
git clone https://github.com/[org]/physical-ai-textbook.git
cd physical-ai-textbook

# Install frontend dependencies
npm install

# Set up backend
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cd ..

# Copy environment files
cp .env.example .env
cp backend/.env.example backend/.env
# Edit .env files with your API keys
```

### Running Locally

```bash
# Option 1: Frontend only (for reading content)
npm run start
# Open http://localhost:3000

# Option 2: Full stack (with chatbot)
# Terminal 1: Start local services
docker-compose up -d

# Terminal 2: Start backend
cd backend
source venv/bin/activate
uvicorn app.main:app --reload

# Terminal 3: Start frontend
npm run start
```

### Quick Commands

```bash
# Development
npm run start          # Start dev server (hot reload)
npm run build          # Production build
npm run typecheck      # TypeScript validation
npm run lint           # ESLint check
npm run serve          # Serve production build locally

# Backend
cd backend
uvicorn app.main:app --reload    # Start API server
pytest                           # Run tests
ruff check app/                  # Lint Python code

# Content Validation
node scripts/validate-content.js  # Check word counts, structure
node scripts/validate-links.js    # Check internal links
```

---

## Features

### 📚 Sequential Learning

Progress through chapters with clear prerequisites and measurable learning objectives.

```
Module 1 → Module 2 → Module 3 → Module 4
   ↓          ↓          ↓          ↓
 Ch 1-5    Ch 6-7     Ch 8-10   Ch 11-14
   ↓          ↓          ↓          ↓
Capstone  Capstone     Labs    Final Capstone
```

### 🤖 RAG-Powered Chatbot

Ask questions about the book content and get contextually relevant answers with source citations.

- Semantic search across all chapters
- Chapter-aware context injection
- Streaming responses
- Conversation history

### 🔬 Hands-on Labs

Each chapter includes a corresponding lab exercise with:
- Step-by-step instructions
- Starter code with TODOs
- Verification commands
- Grading rubrics
- Troubleshooting guides

### 🌐 Bilingual Support

- Full English content
- Urdu translation with RTL layout support
- Real-time translation toggle
- Translation caching for performance

### 👤 Personalization

- User authentication (email/OAuth)
- Background questionnaire
- Progress tracking
- Personalized recommendations
- Bookmarks and notes

---

## Modules

| Module | Chapters | Topics | Capstone |
|--------|----------|--------|----------|
| **1. ROS 2 Fundamentals** | 1-5 | Nodes, Publishers/Subscribers, Sensors, URDF, Edge Deployment | Edge Robot Controller |
| **2. Digital Twin** | 6-7 | Gazebo Physics, Unity Integration, Sensor Simulation | Unity Digital Twin |
| **3. NVIDIA Isaac** | 8-10 | Isaac Sim, GPU Perception, Navigation, Sim2Real | RL Navigation |
| **4. VLA & Capstone** | 11-14 | Humanoid Locomotion, Manipulation, Vision-Language-Action | Full Humanoid System |

### Chapter Overview

<details>
<summary><b>Module 1: ROS 2 Fundamentals</b></summary>

- **Ch 1**: Welcome to ROS 2 - First node, publisher/subscriber
- **Ch 2**: Sensors & Perception - Camera, LiDAR, IMU integration
- **Ch 3**: ROS 2 Architecture - Lifecycle, services, actions
- **Ch 4**: URDF & Humanoid - Robot modeling, Xacro macros
- **Ch 5**: Edge Capstone - Cross-compilation, deployment
</details>

<details>
<summary><b>Module 2: Digital Twin</b></summary>

- **Ch 6**: Gazebo Physics - World creation, sensor plugins
- **Ch 7**: Unity Capstone - ROS 2 bridge, visualization
</details>

<details>
<summary><b>Module 3: NVIDIA Isaac</b></summary>

- **Ch 8**: Isaac Sim - Scene creation, asset import
- **Ch 9**: Isaac ROS GPU - GPU perception pipelines
- **Ch 10**: Navigation & RL - Nav2, RL training, domain randomization
</details>

<details>
<summary><b>Module 4: VLA & Capstone</b></summary>

- **Ch 11**: Humanoid Locomotion - ZMP, balance control, gait
- **Ch 12**: Dexterous Manipulation - Hand kinematics, grasping
- **Ch 13**: Vision-Language-Action - VLA models, language parsing
- **Ch 14**: Final Capstone - Full system integration
</details>

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Browser                            │
└─────────────────────────────┬───────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌─────────────────┐   ┌─────────────┐   ┌─────────────┐
│   Docusaurus    │   │  Chat UI    │   │  Auth UI    │
│  (Static Docs)  │   │ (React)     │   │  (React)    │
└─────────────────┘   └──────┬──────┘   └──────┬──────┘
                             │                  │
                             └────────┬─────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Chat API │  │ Auth API │  │ User API │  │ RAG Svc  │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
└───────┼─────────────┼─────────────┼─────────────┼──────────┘
        │             │             │             │
        ▼             ▼             ▼             ▼
┌──────────────┐ ┌──────────┐ ┌──────────────┐ ┌──────────┐
│   OpenAI     │ │  JWT     │ │   Neon       │ │  Qdrant  │
│   GPT-4      │ │  Auth    │ │  Postgres    │ │  Vectors │
└──────────────┘ └──────────┘ └──────────────┘ └──────────┘
```

### Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Frontend** | Docusaurus 3.x | Documentation site |
| | React 18 | UI components |
| | TypeScript | Type safety |
| **Backend** | FastAPI | REST API |
| | Python 3.11 | Backend logic |
| | Pydantic | Data validation |
| **Database** | Neon Postgres | User data, sessions |
| | Qdrant Cloud | Vector embeddings |
| **AI** | OpenAI GPT-4 | Chat responses |
| | text-embedding-3-small | Semantic search |
| **Deployment** | Vercel / GitHub Pages | Frontend hosting |
| | Railway | Backend hosting |
| | GitHub Actions | CI/CD |

---

## Project Structure

```
physical-ai-textbook/
├── docs/                      # 📖 Book content (Markdown)
│   ├── front-matter/          #    Foreword, Preface, Introduction
│   ├── module-1-ros2/         #    Chapters 1-5
│   ├── module-2-digital-twin/ #    Chapters 6-7
│   ├── module-3-isaac/        #    Chapters 8-10
│   ├── module-4-vla/          #    Chapters 11-14
│   └── back-matter/           #    Appendices, Glossary
├── labs/                      # 🔬 Hands-on exercises
│   ├── module-1/              #    Labs for chapters 1-5
│   ├── module-2/              #    Labs for chapters 6-7
│   ├── module-3/              #    Labs for chapters 8-10
│   └── module-4/              #    Labs for chapters 11-14
├── backend/                   # ⚙️ FastAPI backend
│   ├── app/
│   │   ├── api/               #    REST endpoints
│   │   ├── services/          #    Business logic
│   │   ├── models/            #    Database models
│   │   └── db/                #    Database connections
│   ├── migrations/            #    Alembic migrations
│   └── tests/                 #    pytest tests
├── src/                       # 🎨 Docusaurus customizations
│   ├── components/            #    React components
│   ├── pages/                 #    Custom pages
│   ├── services/              #    API clients
│   └── css/                   #    Stylesheets
├── specs/                     # 📋 Feature specifications
├── history/                   # 📝 Prompt history records
├── scripts/                   # 🛠️ Utility scripts
├── .github/workflows/         # 🔄 CI/CD pipelines
└── .claude/commands/          # 🤖 Claude Code skills
```

---

## Environment Variables

### Frontend (.env)

```bash
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000
```

### Backend (backend/.env)

```bash
# Database
DATABASE_URL=postgresql://user:pass@host:5432/db

# Vector Database
QDRANT_URL=https://xxx.qdrant.io:6333
QDRANT_API_KEY=your-qdrant-api-key

# OpenAI
OPENAI_API_KEY=sk-your-openai-api-key

# Authentication
JWT_SECRET=your-super-secret-jwt-key

# CORS (optional)
CORS_ORIGINS=http://localhost:3000
```

---

## Deployment

### Frontend (Vercel)

```bash
# Deploy to Vercel
vercel --prod
```

### Frontend (GitHub Pages)

```bash
# Build and deploy
npm run build
npm run deploy
```

### Backend (Railway)

```bash
# Deploy to Railway
cd backend
railway up
```

See [docs/RUNBOOK.md](docs/RUNBOOK.md) for detailed deployment and operations guide.

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Development workflow
- Code standards
- Content guidelines
- Pull request process

### Quick Contribution Guide

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes following code standards
4. Run tests: `npm run typecheck && npm run lint`
5. Commit with conventional commits: `git commit -m "feat: add new feature"`
6. Push and open a Pull Request

---

## Documentation

- [Quickstart Guide](specs/001-physical-ai-textbook/quickstart.md)
- [Feature Specification](specs/001-physical-ai-textbook/spec.md)
- [Implementation Plan](specs/001-physical-ai-textbook/plan.md)
- [Task List](specs/001-physical-ai-textbook/tasks.md)
- [Production Runbook](docs/RUNBOOK.md)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- ROS 2 Community
- NVIDIA Isaac Sim Team
- OpenAI
- Docusaurus Team
- All contributors

---

<p align="center">
  Built with ❤️ for the robotics education community
</p>
