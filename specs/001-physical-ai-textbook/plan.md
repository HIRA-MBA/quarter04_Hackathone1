# Implementation Plan: Physical AI & Robotics Textbook

**Branch**: `001-physical-ai-textbook` | **Date**: 2025-12-10 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-physical-ai-textbook/spec.md`

## Summary

Multi-module Docusaurus textbook on Physical AI & Robotics (ROS 2 → VLA) with integrated RAG chatbot, personalization, Urdu translation, and Better-Auth authentication. Delivers 14 chapters across 4 modules with code labs, plus front/back matter.

## Technical Context

**Language/Version**: TypeScript 5.x (Docusaurus), Python 3.11+ (FastAPI)
**Primary Dependencies**: Docusaurus 3.x, FastAPI 0.109+, OpenAI Agents SDK, Qdrant Client, Neon Postgres (psycopg), Better-Auth
**Storage**: Neon Postgres (users, sessions, preferences), Qdrant (vector embeddings)
**Testing**: Vitest (frontend), pytest (backend), Playwright (E2E)
**Target Platform**: Vercel (Docusaurus SSG), Railway/Fly.io (FastAPI)
**Project Type**: web (frontend + backend)
**Performance Goals**: <500ms chatbot response, <2s page load, 99.5% uptime
**Constraints**: Serverless-friendly, GDPR-compliant user data, offline-capable book content
**Scale/Scope**: ~30k words book content, 14 chapters, 50+ code examples, 14+ lab exercises

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| I. Content-First Authoring | ✅ PASS | Each chapter has objectives, code examples, labs, diagrams |
| II. Markdown-Only Format | ✅ PASS | GFM + MDX for interactive features only |
| III. Modular Architecture | ✅ PASS | 4 modules, independent chapters with prerequisites |
| IV. RAG-Integrated Learning | ✅ PASS | OpenAI Agents + FastAPI + Neon + Qdrant stack |
| V. Multi-Language Support | ✅ PASS | English primary, Urdu secondary |
| VI. Reproducible Lab Exercises | ✅ PASS | Pinned versions, step-by-step, troubleshooting |

**Excluded Content Check**: ✅ No vendor comparisons, no ethics in chapters, no hardcoded secrets

## Project Structure

### Documentation (this feature)

```text
specs/001-physical-ai-textbook/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (OpenAPI specs)
└── tasks.md             # Phase 2 output (/sp.tasks)
```

### Source Code (repository root)

```text
# Docusaurus Book (frontend)
docs/
├── front-matter/
│   ├── foreword.md
│   ├── preface.md
│   ├── introduction.md
│   └── hardware-lab-setup.md
├── module-1-ros2/
│   ├── ch01-welcome-first-node.md
│   ├── ch02-sensors-perception.md
│   ├── ch03-ros2-architecture.md
│   ├── ch04-urdf-humanoid.md
│   └── ch05-edge-capstone.md
├── module-2-digital-twin/
│   ├── ch06-gazebo-physics.md
│   └── ch07-unity-capstone.md
├── module-3-isaac/
│   ├── ch08-isaac-sim.md
│   ├── ch09-isaac-ros-gpu.md
│   └── ch10-nav-rl-sim2real.md
├── module-4-vla/
│   ├── ch11-humanoid-locomotion.md
│   ├── ch12-dexterous-manipulation.md
│   ├── ch13-vision-language-action.md
│   └── ch14-capstone-humanoid.md
└── back-matter/
    ├── appendix-a-student-kit.md
    ├── appendix-b-cloud-setup.md
    ├── appendix-c-repos-docker.md
    ├── appendix-d-troubleshooting.md
    ├── safety-ethics.md
    ├── glossary.md
    └── index.md

src/
├── components/
│   ├── ChatBot/
│   │   ├── ChatBot.tsx
│   │   ├── ChatMessage.tsx
│   │   └── ChatInput.tsx
│   ├── Personalization/
│   │   ├── BackgroundQuestionnaire.tsx
│   │   └── ChapterRenderer.tsx
│   └── Translation/
│       └── UrduToggle.tsx
├── pages/
│   └── auth/
│       ├── signin.tsx
│       └── signup.tsx
├── services/
│   └── api.ts
└── theme/
    └── DocItem/
        └── index.tsx  # RAG integration wrapper

# FastAPI Backend
backend/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── models/
│   │   ├── user.py
│   │   ├── session.py
│   │   └── preference.py
│   ├── services/
│   │   ├── rag/
│   │   │   ├── ingestion.py
│   │   │   ├── embeddings.py
│   │   │   ├── retrieval.py
│   │   │   └── chat.py
│   │   ├── auth.py
│   │   └── translation.py
│   ├── api/
│   │   ├── routes/
│   │   │   ├── chat.py
│   │   │   ├── auth.py
│   │   │   ├── user.py
│   │   │   └── translation.py
│   │   └── deps.py
│   └── db/
│       ├── postgres.py
│       └── qdrant.py
└── tests/
    ├── unit/
    ├── integration/
    └── contract/

# Code Examples & Labs
labs/
├── module-1/
│   ├── ch01-hello-robot/
│   ├── ch02-sensor-fusion/
│   ├── ch03-ros2-deep-dive/
│   ├── ch04-urdf-humanoid/
│   └── ch05-edge-controller/
├── module-2/
│   ├── ch06-gazebo-world/
│   └── ch07-unity-twin/
├── module-3/
│   ├── ch08-isaac-scene/
│   ├── ch09-gpu-vision/
│   └── ch10-sim2real/
└── module-4/
    ├── ch11-walking/
    ├── ch12-manipulation/
    ├── ch13-vla-commands/
    └── ch14-capstone/

# CI/CD
.github/
└── workflows/
    ├── deploy-docs.yml
    ├── deploy-backend.yml
    └── test.yml

# Configuration
docusaurus.config.ts
sidebars.ts
package.json
```

**Structure Decision**: Web application with Docusaurus frontend (SSG to Vercel) + FastAPI backend (to Railway/Fly.io). Labs stored separately for easy version control and Docker mounting.

## Milestone Plan (JSON)

```json
{
  "project": "physical-ai-textbook",
  "version": "1.0.0",
  "milestones": [
    {
      "id": "M1",
      "title": "Project Bootstrap",
      "dependencies": [],
      "sub_milestones": [
        {
          "id": "M1.1",
          "title": "Repository Setup",
          "tasks": [
            "Initialize git repo with .gitignore",
            "Create branch structure (main, develop, feature/*)",
            "Add LICENSE and README.md"
          ],
          "acceptance": ["Git repo initialized", "Branches created", "README exists"]
        },
        {
          "id": "M1.2",
          "title": "Docusaurus Initialization",
          "tasks": [
            "npx create-docusaurus@latest",
            "Configure TypeScript",
            "Set up custom theme structure"
          ],
          "acceptance": ["Docusaurus dev server runs", "TypeScript compiles", "Custom theme loads"]
        },
        {
          "id": "M1.3",
          "title": "Vercel Configuration",
          "tasks": [
            "Connect GitHub repo to Vercel",
            "Configure build settings",
            "Set up preview deployments"
          ],
          "acceptance": ["Vercel builds on push", "Preview URLs work", "Production deploy succeeds"]
        },
        {
          "id": "M1.4",
          "title": "Backend Scaffolding",
          "tasks": [
            "Initialize FastAPI project",
            "Configure pyproject.toml with dependencies",
            "Set up Docker development environment"
          ],
          "acceptance": ["FastAPI server starts", "Health endpoint responds", "Docker container runs"]
        }
      ]
    },
    {
      "id": "M2",
      "title": "Book Skeleton",
      "dependencies": ["M1"],
      "sub_milestones": [
        {
          "id": "M2.1",
          "title": "Front Matter Structure",
          "tasks": [
            "Create docs/front-matter/ directory",
            "Add placeholder files: foreword.md, preface.md, introduction.md, hardware-lab-setup.md",
            "Configure sidebar routing"
          ],
          "acceptance": ["All front matter files exist", "Sidebar shows front matter section", "Navigation works"]
        },
        {
          "id": "M2.2",
          "title": "Module Directories",
          "tasks": [
            "Create module-1-ros2/ through module-4-vla/ directories",
            "Add chapter placeholder files (ch01-ch14)",
            "Configure category metadata (_category_.json)"
          ],
          "acceptance": ["All 14 chapter files exist", "Module grouping correct", "Chapter order correct"]
        },
        {
          "id": "M2.3",
          "title": "Back Matter Structure",
          "tasks": [
            "Create docs/back-matter/ directory",
            "Add appendix files (A-D), safety-ethics.md, glossary.md, index.md",
            "Configure sidebar routing"
          ],
          "acceptance": ["All back matter files exist", "Sidebar shows back matter section", "Appendix ordering correct"]
        },
        {
          "id": "M2.4",
          "title": "Labs Directory Structure",
          "tasks": [
            "Create labs/ directory with module subdirectories",
            "Add README.md for each lab folder",
            "Set up Docker compose for lab environments"
          ],
          "acceptance": ["Labs structure matches chapters", "Each lab has README", "Docker compose valid"]
        }
      ]
    },
    {
      "id": "M3",
      "title": "Module Writing",
      "dependencies": ["M2"],
      "sub_milestones": [
        {
          "id": "M3.1",
          "title": "Module 1: ROS 2 (Chapters 1-5)",
          "tasks": [
            "Write ch01-welcome-first-node.md (1700-2500 words)",
            "Write ch02-sensors-perception.md (1700-2500 words)",
            "Write ch03-ros2-architecture.md (1700-2500 words)",
            "Write ch04-urdf-humanoid.md (1700-2500 words)",
            "Write ch05-edge-capstone.md (3000-3500 words)"
          ],
          "acceptance": [
            "Each chapter has learning objectives",
            "Word counts within range",
            "Code examples present",
            "Lab exercises defined",
            "RAG integration points marked"
          ]
        },
        {
          "id": "M3.2",
          "title": "Module 2: Digital Twin (Chapters 6-7)",
          "tasks": [
            "Write ch06-gazebo-physics.md (1700-2500 words)",
            "Write ch07-unity-capstone.md (3000-3500 words)"
          ],
          "acceptance": [
            "Word counts within range",
            "Gazebo integration examples",
            "Unity-ROS bridge documented",
            "Digital twin capstone complete"
          ]
        },
        {
          "id": "M3.3",
          "title": "Module 3: NVIDIA Isaac (Chapters 8-10)",
          "tasks": [
            "Write ch08-isaac-sim.md (1700-2500 words)",
            "Write ch09-isaac-ros-gpu.md (1700-2500 words)",
            "Write ch10-nav-rl-sim2real.md (1700-2500 words)"
          ],
          "acceptance": [
            "Isaac Sim setup documented",
            "GPU perception pipeline examples",
            "Sim-to-real transfer process documented"
          ]
        },
        {
          "id": "M3.4",
          "title": "Module 4: VLA + Capstone (Chapters 11-14)",
          "tasks": [
            "Write ch11-humanoid-locomotion.md (1700-2500 words)",
            "Write ch12-dexterous-manipulation.md (1700-2500 words)",
            "Write ch13-vision-language-action.md (1700-2500 words)",
            "Write ch14-capstone-humanoid.md (3000-3500 words)"
          ],
          "acceptance": [
            "Locomotion algorithms documented",
            "Manipulation examples complete",
            "VLA integration working",
            "Final capstone integrates all modules"
          ]
        },
        {
          "id": "M3.5",
          "title": "Front & Back Matter Content",
          "tasks": [
            "Write foreword.md (500-800 words)",
            "Write preface.md (800-1200 words)",
            "Write introduction.md (1000-1500 words)",
            "Write hardware-lab-setup.md (1500-2000 words)",
            "Write all appendices (A-D)",
            "Write safety-ethics.md (1000-1500 words)",
            "Compile glossary.md",
            "Generate index.md"
          ],
          "acceptance": [
            "All front matter complete",
            "All appendices complete",
            "Glossary has 100+ terms",
            "Index cross-references valid"
          ]
        }
      ]
    },
    {
      "id": "M4",
      "title": "Code & Labs",
      "dependencies": ["M3.1"],
      "sub_milestones": [
        {
          "id": "M4.1",
          "title": "ROS 2 Labs (Chapters 1-5)",
          "tasks": [
            "Create ch01-hello-robot lab package",
            "Create ch02-sensor-fusion lab package",
            "Create ch03-ros2-deep-dive lab package",
            "Create ch04-urdf-humanoid lab package",
            "Create ch05-edge-controller lab package"
          ],
          "acceptance": [
            "Each lab has working ROS 2 package",
            "Instructions match chapter content",
            "Expected outputs documented"
          ]
        },
        {
          "id": "M4.2",
          "title": "Digital Twin Labs (Chapters 6-7)",
          "tasks": [
            "Create ch06-gazebo-world lab with world files",
            "Create ch07-unity-twin lab with Unity project"
          ],
          "acceptance": [
            "Gazebo world loads correctly",
            "Unity project compiles",
            "ROS 2 bridge functional"
          ]
        },
        {
          "id": "M4.3",
          "title": "Isaac Labs (Chapters 8-10)",
          "tasks": [
            "Create ch08-isaac-scene lab with Isaac assets",
            "Create ch09-gpu-vision lab with perception pipeline",
            "Create ch10-sim2real lab with RL training scripts"
          ],
          "acceptance": [
            "Isaac Sim scenes load",
            "GPU pipelines execute",
            "RL training runs"
          ]
        },
        {
          "id": "M4.4",
          "title": "VLA Labs (Chapters 11-14)",
          "tasks": [
            "Create ch11-walking lab with locomotion controllers",
            "Create ch12-manipulation lab with grasp planning",
            "Create ch13-vla-commands lab with language integration",
            "Create ch14-capstone lab with full integration"
          ],
          "acceptance": [
            "Walking simulation works",
            "Manipulation demos functional",
            "VLA responds to commands",
            "Capstone integrates all systems"
          ]
        }
      ]
    },
    {
      "id": "M5",
      "title": "RAG Chatbot",
      "dependencies": ["M1.4", "M3.1"],
      "sub_milestones": [
        {
          "id": "M5.1",
          "title": "Text Ingestion Pipeline",
          "tasks": [
            "Create Markdown parser for book content",
            "Implement chunking strategy (semantic sections)",
            "Build ingestion CLI tool",
            "Add metadata extraction (chapter, section, keywords)"
          ],
          "acceptance": [
            "All chapters parsed successfully",
            "Chunks are semantically coherent",
            "Metadata correctly extracted"
          ]
        },
        {
          "id": "M5.2",
          "title": "Neon Postgres Schema",
          "tasks": [
            "Design user table schema",
            "Design session table schema",
            "Design chat_history table schema",
            "Design user_preferences table schema",
            "Create migrations with Alembic"
          ],
          "acceptance": [
            "Schema supports all user flows",
            "Migrations run cleanly",
            "Indexes on frequent queries"
          ]
        },
        {
          "id": "M5.3",
          "title": "Qdrant Embeddings",
          "tasks": [
            "Set up Qdrant Cloud or self-hosted",
            "Implement embedding generation (OpenAI ada-002)",
            "Create collection with proper indexing",
            "Build vector upsert pipeline"
          ],
          "acceptance": [
            "All book content embedded",
            "Similarity search returns relevant results",
            "Query latency <200ms"
          ]
        },
        {
          "id": "M5.4",
          "title": "API Endpoints",
          "tasks": [
            "POST /api/chat - send message, get response",
            "GET /api/chat/history - retrieve conversation",
            "POST /api/chat/feedback - rate response",
            "WebSocket /ws/chat - streaming responses"
          ],
          "acceptance": [
            "All endpoints return correct responses",
            "Error handling complete",
            "Rate limiting configured"
          ]
        },
        {
          "id": "M5.5",
          "title": "UI Integration",
          "tasks": [
            "Create ChatBot React component",
            "Implement floating chat widget",
            "Add chapter-aware context injection",
            "Style with Docusaurus theme"
          ],
          "acceptance": [
            "Chat widget appears on all pages",
            "Context matches current chapter",
            "Responsive on mobile"
          ]
        }
      ]
    },
    {
      "id": "M6",
      "title": "Personalization",
      "dependencies": ["M5.2"],
      "sub_milestones": [
        {
          "id": "M6.1",
          "title": "Better-Auth Signup/Signin",
          "tasks": [
            "Integrate Better-Auth library",
            "Create signup page with email/OAuth",
            "Create signin page",
            "Implement session management",
            "Add password reset flow"
          ],
          "acceptance": [
            "Users can sign up",
            "Users can sign in",
            "Sessions persist correctly",
            "OAuth providers work"
          ]
        },
        {
          "id": "M6.2",
          "title": "User Background Questionnaire",
          "tasks": [
            "Design questionnaire schema",
            "Create multi-step form component",
            "Store responses in Postgres",
            "Build recommendation engine"
          ],
          "acceptance": [
            "Questionnaire captures user background",
            "Data stored securely",
            "Recommendations generated"
          ]
        },
        {
          "id": "M6.3",
          "title": "Personalized Chapter Rendering",
          "tasks": [
            "Create conditional content blocks",
            "Implement difficulty adjustment",
            "Add progress tracking",
            "Show personalized recommendations"
          ],
          "acceptance": [
            "Content adapts to user profile",
            "Progress persists across sessions",
            "Recommendations are relevant"
          ]
        }
      ]
    },
    {
      "id": "M7",
      "title": "Urdu Translation System",
      "dependencies": ["M3", "M5.4"],
      "sub_milestones": [
        {
          "id": "M7.1",
          "title": "Chapter-Level Translation Trigger",
          "tasks": [
            "Add language toggle component",
            "Implement translation API endpoint",
            "Create translation queue system",
            "Handle RTL layout for Urdu"
          ],
          "acceptance": [
            "Toggle switches language",
            "Translation API responds",
            "RTL renders correctly"
          ]
        },
        {
          "id": "M7.2",
          "title": "Inline Translation Cache",
          "tasks": [
            "Design translation cache schema",
            "Implement cache-first retrieval",
            "Add background translation jobs",
            "Create translation quality review queue"
          ],
          "acceptance": [
            "Cached translations load fast",
            "New translations queue correctly",
            "Cache invalidation works"
          ]
        }
      ]
    },
    {
      "id": "M8",
      "title": "Claude Code Subagents & Skills",
      "dependencies": ["M2"],
      "sub_milestones": [
        {
          "id": "M8.1",
          "title": "Writing Skill",
          "tasks": [
            "Create chapter-writing skill template",
            "Define word count validation",
            "Add structure enforcement",
            "Implement RAG point injection"
          ],
          "acceptance": [
            "Skill generates compliant chapters",
            "Word counts validated",
            "Structure matches spec"
          ]
        },
        {
          "id": "M8.2",
          "title": "Code Generation Skill",
          "tasks": [
            "Create ROS 2 code generation skill",
            "Create simulation code skill",
            "Add test generation capability",
            "Implement docstring generation"
          ],
          "acceptance": [
            "Generated code runs",
            "Tests pass",
            "Documentation complete"
          ]
        },
        {
          "id": "M8.3",
          "title": "Diagram Generation Skill",
          "tasks": [
            "Create Mermaid diagram skill",
            "Add architecture diagram templates",
            "Implement flowchart generation",
            "Create sequence diagram capability"
          ],
          "acceptance": [
            "Diagrams render correctly",
            "Templates cover common patterns",
            "Integration with chapters"
          ]
        },
        {
          "id": "M8.4",
          "title": "Lab Exercise Skill",
          "tasks": [
            "Create lab template skill",
            "Add step-by-step instruction generator",
            "Implement acceptance criteria generation",
            "Create troubleshooting section generator"
          ],
          "acceptance": [
            "Labs follow consistent format",
            "Instructions are clear",
            "Troubleshooting is comprehensive"
          ]
        }
      ]
    },
    {
      "id": "M9",
      "title": "Deployment",
      "dependencies": ["M5", "M6", "M7"],
      "sub_milestones": [
        {
          "id": "M9.1",
          "title": "GitHub Actions CI",
          "tasks": [
            "Create test workflow (frontend + backend)",
            "Add lint workflow",
            "Create build verification workflow",
            "Add security scanning"
          ],
          "acceptance": [
            "All tests pass in CI",
            "Lint errors caught",
            "Builds succeed",
            "No critical vulnerabilities"
          ]
        },
        {
          "id": "M9.2",
          "title": "Vercel Production Build",
          "tasks": [
            "Configure production environment variables",
            "Set up custom domain",
            "Enable edge caching",
            "Configure redirects and headers"
          ],
          "acceptance": [
            "Production build succeeds",
            "Custom domain works",
            "Caching headers correct",
            "Performance metrics green"
          ]
        },
        {
          "id": "M9.3",
          "title": "Backend Deployment",
          "tasks": [
            "Deploy FastAPI to Railway/Fly.io",
            "Configure Neon Postgres connection",
            "Set up Qdrant Cloud",
            "Configure secrets management"
          ],
          "acceptance": [
            "API responds in production",
            "Database connections stable",
            "Vector search works",
            "Secrets not exposed"
          ]
        },
        {
          "id": "M9.4",
          "title": "Testing & Acceptance",
          "tasks": [
            "Run full E2E test suite",
            "Perform load testing",
            "Validate all acceptance criteria",
            "Complete security audit"
          ],
          "acceptance": [
            "E2E tests pass",
            "Load tests meet SLAs",
            "All SC-001 through SC-010 verified",
            "Security audit passed"
          ]
        }
      ]
    }
  ],
  "dependency_graph": {
    "M1": [],
    "M2": ["M1"],
    "M3": ["M2"],
    "M4": ["M3.1"],
    "M5": ["M1.4", "M3.1"],
    "M6": ["M5.2"],
    "M7": ["M3", "M5.4"],
    "M8": ["M2"],
    "M9": ["M5", "M6", "M7"]
  },
  "critical_path": ["M1", "M2", "M3", "M5", "M6", "M9"]
}
```

## Complexity Tracking

*No constitution violations requiring justification.*

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| OpenAI API rate limits | Chat degradation | Implement caching, fallback responses |
| Isaac Sim GPU requirements | Lab inaccessibility | Provide cloud GPU alternatives in Appendix B |
| Translation quality | User confusion | Human review queue, user feedback loop |
| Large content volume | Delayed delivery | Parallelize writing with skills, prioritize core chapters |

## Next Steps

1. Run `/sp.tasks` to generate detailed task breakdown
2. Execute M1 (Project Bootstrap) first
3. Parallelize M3 (Writing) with M5 (RAG) after M2 complete
4. Use Claude Code skills (M8) to accelerate content generation
