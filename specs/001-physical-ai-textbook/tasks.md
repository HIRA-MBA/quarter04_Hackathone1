# Tasks: Physical AI & Robotics Textbook

**Input**: Design documents from `/specs/001-physical-ai-textbook/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/openapi.yaml, quickstart.md
**Branch**: `001-physical-ai-textbook`
**Generated**: 2025-12-10
**Updated**: 2025-12-20 (clarification sync: OAuth-only auth, translation deferred)

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1-US5)
- Include exact file paths in descriptions

## User Stories Mapping

| ID | Title | Priority | Milestone |
|----|-------|----------|-----------|
| US1 | Sequential Chapter Learning | P1 | M2, M3 |
| US2 | Interactive RAG Chatbot Assistance | P2 | M5 |
| US3 | Lab Exercise Completion | P2 | M4 |
| US4 | Capstone Project Completion | P3 | M3.4, M4.4 |
| US5 | Personalized Learning Experience | P3 | M6 (M7 deferred) |

---

## Phase 1: Setup (M1 - Project Bootstrap)

**Purpose**: Initialize repository, Docusaurus, backend scaffolding, and CI/CD

### M1.1 - Repository Setup

- [x] T001 [P] Create .gitignore with Node, Python, IDE, and env exclusions in .gitignore
- [x] T002 [P] Create README.md with project overview and quickstart link in README.md
- [x] T003 [P] Add MIT LICENSE file in LICENSE
- [x] T004 Create branch structure (main, develop) via git commands

### M1.2 - Docusaurus Initialization

- [x] T005 Initialize Docusaurus 3.x project with TypeScript via npx create-docusaurus@latest
- [x] T006 Configure docusaurus.config.ts with project metadata and theme settings
- [x] T007 [P] Create sidebars.ts with initial structure for modules
- [x] T008 [P] Configure package.json with scripts (start, build, serve, typecheck, lint)
- [x] T009 Set up custom theme structure in src/theme/DocItem/index.tsx

### M1.3 - Vercel Configuration

- [x] T010 Create vercel.json with build settings and environment variables
- [x] T011 [P] Configure .env.example with required frontend environment variables

### M1.4 - Backend Scaffolding

- [x] T012 Create backend/ directory structure per plan.md
- [x] T013 Initialize FastAPI project in backend/pyproject.toml with dependencies
- [x] T014 [P] Create backend/.env.example with required environment variables
- [x] T015 Create backend/app/main.py with FastAPI app and health endpoint
- [x] T016 Create backend/app/config.py with settings using pydantic-settings
- [x] T017 [P] Create docker-compose.yml with postgres, qdrant services
- [x] T018 [P] Create backend/Dockerfile for FastAPI containerization

**Checkpoint**: Project structure ready, dev servers start (`npm run start`, `uvicorn app.main:app`)

---

## Phase 2: Foundational (M2 - Book Skeleton + DB Setup)

**Purpose**: Create book structure, initialize databases - BLOCKS all user stories

### M2.1 - Front Matter Structure

- [x] T019 Create docs/front-matter/ directory with _category_.json
- [x] T020 [P] Create placeholder docs/front-matter/foreword.md with frontmatter
- [x] T021 [P] Create placeholder docs/front-matter/preface.md with frontmatter
- [x] T022 [P] Create placeholder docs/front-matter/introduction.md with frontmatter
- [x] T023 [P] Create placeholder docs/front-matter/hardware-lab-setup.md with frontmatter
- [x] T024 Update sidebars.ts to include front-matter section

### M2.2 - Module Directories

- [x] T025 Create docs/module-1-ros2/ directory with _category_.json
- [x] T026 [P] Create placeholder docs/module-1-ros2/ch01-welcome-first-node.md
- [x] T027 [P] Create placeholder docs/module-1-ros2/ch02-sensors-perception.md
- [x] T028 [P] Create placeholder docs/module-1-ros2/ch03-ros2-architecture.md
- [x] T029 [P] Create placeholder docs/module-1-ros2/ch04-urdf-humanoid.md
- [x] T030 [P] Create placeholder docs/module-1-ros2/ch05-edge-capstone.md
- [x] T031 Create docs/module-2-digital-twin/ directory with _category_.json
- [x] T032 [P] Create placeholder docs/module-2-digital-twin/ch06-gazebo-physics.md
- [x] T033 [P] Create placeholder docs/module-2-digital-twin/ch07-unity-capstone.md
- [x] T034 Create docs/module-3-isaac/ directory with _category_.json
- [x] T035 [P] Create placeholder docs/module-3-isaac/ch08-isaac-sim.md
- [x] T036 [P] Create placeholder docs/module-3-isaac/ch09-isaac-ros-gpu.md
- [x] T037 [P] Create placeholder docs/module-3-isaac/ch10-nav-rl-sim2real.md
- [x] T038 Create docs/module-4-vla/ directory with _category_.json
- [x] T039 [P] Create placeholder docs/module-4-vla/ch11-humanoid-locomotion.md
- [x] T040 [P] Create placeholder docs/module-4-vla/ch12-dexterous-manipulation.md
- [x] T041 [P] Create placeholder docs/module-4-vla/ch13-vision-language-action.md
- [x] T042 [P] Create placeholder docs/module-4-vla/ch14-capstone-humanoid.md
- [x] T043 Update sidebars.ts to include all module sections

### M2.3 - Back Matter Structure

- [x] T044 Create docs/back-matter/ directory with _category_.json
- [x] T045 [P] Create placeholder docs/back-matter/appendix-a-student-kit.md
- [x] T046 [P] Create placeholder docs/back-matter/appendix-b-cloud-setup.md
- [x] T047 [P] Create placeholder docs/back-matter/appendix-c-repos-docker.md
- [x] T048 [P] Create placeholder docs/back-matter/appendix-d-troubleshooting.md
- [x] T049 [P] Create placeholder docs/back-matter/safety-ethics.md
- [x] T050 [P] Create placeholder docs/back-matter/glossary.md
- [x] T051 [P] Create placeholder docs/back-matter/index.md
- [x] T052 Update sidebars.ts to include back-matter section

### M2.4 - Labs Directory Structure

- [x] T053 Create labs/ directory structure matching modules
- [x] T054 [P] Create labs/module-1/ch01-hello-robot/README.md with lab template
- [x] T055 [P] Create labs/module-1/ch02-sensor-fusion/README.md with lab template
- [x] T056 [P] Create labs/module-1/ch03-ros2-deep-dive/README.md with lab template
- [x] T057 [P] Create labs/module-1/ch04-urdf-humanoid/README.md with lab template
- [x] T058 [P] Create labs/module-1/ch05-edge-controller/README.md with lab template
- [x] T059 [P] Create labs/module-2/ch06-gazebo-world/README.md with lab template
- [x] T060 [P] Create labs/module-2/ch07-unity-twin/README.md with lab template
- [x] T061 [P] Create labs/module-3/ch08-isaac-scene/README.md with lab template
- [x] T062 [P] Create labs/module-3/ch09-gpu-vision/README.md with lab template
- [x] T063 [P] Create labs/module-3/ch10-sim2real/README.md with lab template
- [x] T064 [P] Create labs/module-4/ch11-walking/README.md with lab template
- [x] T065 [P] Create labs/module-4/ch12-manipulation/README.md with lab template
- [x] T066 [P] Create labs/module-4/ch13-vla-commands/README.md with lab template
- [x] T067 [P] Create labs/module-4/ch14-capstone/README.md with lab template
- [x] T068 Create labs/docker-compose.yml for ROS 2 development environment

### Database Foundation (M5.2)

- [x] T069 Create backend/app/db/postgres.py with Neon connection pool
- [x] T070 Create backend/app/models/user.py with User SQLAlchemy model
- [x] T071 [P] Create backend/app/models/session.py with Session model
- [x] T072 [P] Create backend/app/models/preference.py with UserPreference model
- [x] T073 Create backend/alembic.ini and migrations/ directory
- [x] T074 Create backend/migrations/versions/001_create_users.py (combined schema)
- [x] T075 Create backend/migrations/versions/002_create_sessions.py (combined in 001)
- [x] T076 Create backend/migrations/versions/003_create_user_preferences.py (combined in 001)
- [x] T077 Create backend/migrations/versions/004_create_chat_history.py (combined in 001)
- [x] T078 Create backend/migrations/versions/005_create_translation_cache.py (combined in 001)
- [x] T079 Create backend/app/db/qdrant.py with Qdrant client initialization

**Checkpoint**: Book skeleton visible at localhost:3000, database migrations run cleanly

---

## Phase 3: User Story 1 - Sequential Chapter Learning (P1) 🎯 MVP

**Goal**: Student can read chapters sequentially with learning objectives, code examples, and navigation

**Independent Test**: Open Chapter 1, verify objectives visible, code examples render, navigation to Chapter 2 works

### M3.1 - Module 1: ROS 2 Content (Chapters 1-5)

- [x] T080 [US1] Write docs/module-1-ros2/ch01-welcome-first-node.md with full content (1700-2500 words)
- [x] T081 [US1] Add code examples to ch01: ROS 2 node, publisher/subscriber in docs/module-1-ros2/ch01-welcome-first-node.md
- [x] T082 [US1] Add diagram placeholders to ch01 using Mermaid syntax
- [x] T083 [US1] Write docs/module-1-ros2/ch02-sensors-perception.md with full content (1700-2500 words)
- [x] T084 [US1] Add code examples to ch02: camera, LIDAR, IMU processing
- [x] T085 [US1] Add diagram placeholders to ch02 using Mermaid syntax
- [x] T086 [US1] Write docs/module-1-ros2/ch03-ros2-architecture.md with full content (1700-2500 words)
- [x] T087 [US1] Add code examples to ch03: lifecycle, services, actions, parameters
- [x] T088 [US1] Add diagram placeholders to ch03 using Mermaid syntax
- [x] T089 [US1] Write docs/module-1-ros2/ch04-urdf-humanoid.md with full content (1700-2500 words)
- [x] T090 [US1] Add code examples to ch04: URDF, Xacro, mesh integration
- [x] T091 [US1] Add diagram placeholders to ch04 using Mermaid syntax
- [x] T092 [US1] Write docs/module-1-ros2/ch05-edge-capstone.md with full content (3000-3500 words)
- [x] T093 [US1] Add code examples to ch05: cross-compilation, edge deployment
- [x] T094 [US1] Add diagram placeholders to ch05 using Mermaid syntax
- [x] T095 [US1] Add capstone grading rubric to ch05

### M3.2 - Module 2: Digital Twin Content (Chapters 6-7)

- [x] T096 [US1] Write docs/module-2-digital-twin/ch06-gazebo-physics.md with full content (1700-2500 words)
- [x] T097 [US1] Add code examples to ch06: world file, model spawning, sensor plugins
- [x] T098 [US1] Add diagram placeholders to ch06 using Mermaid syntax
- [x] T099 [US1] Write docs/module-2-digital-twin/ch07-unity-capstone.md with full content (3000-3500 words)
- [x] T100 [US1] Add code examples to ch07: Unity-ROS 2 bridge, visualization, sync
- [x] T101 [US1] Add diagram placeholders to ch07 using Mermaid syntax
- [x] T102 [US1] Add capstone grading rubric to ch07

### M3.3 - Module 3: NVIDIA Isaac Content (Chapters 8-10)

- [x] T103 [US1] Write docs/module-3-isaac/ch08-isaac-sim.md with full content (1700-2500 words)
- [x] T104 [US1] Add code examples to ch08: scene creation, asset import, scripting
- [x] T105 [US1] Add diagram placeholders to ch08 using Mermaid syntax
- [x] T106 [US1] Write docs/module-3-isaac/ch09-isaac-ros-gpu.md with full content (1700-2500 words)
- [x] T107 [US1] Add code examples to ch09: GPU perception, object detection, depth
- [x] T108 [US1] Add diagram placeholders to ch09 using Mermaid syntax
- [x] T109 [US1] Write docs/module-3-isaac/ch10-nav-rl-sim2real.md with full content (1700-2500 words)
- [x] T110 [US1] Add code examples to ch10: Nav2, RL training, domain randomization
- [x] T111 [US1] Add diagram placeholders to ch10 using Mermaid syntax

### M3.4 - Module 4: VLA + Capstone Content (Chapters 11-14)

- [x] T112 [US1] Write docs/module-4-vla/ch11-humanoid-locomotion.md with full content (1700-2500 words)
- [x] T113 [US1] Add code examples to ch11: ZMP, balance controller, gait generation
- [x] T114 [US1] Add diagram placeholders to ch11 using Mermaid syntax
- [x] T115 [US1] Write docs/module-4-vla/ch12-dexterous-manipulation.md with full content (1700-2500 words)
- [x] T116 [US1] Add code examples to ch12: hand kinematics, grasp planning, force control
- [x] T117 [US1] Add diagram placeholders to ch12 using Mermaid syntax
- [x] T118 [US1] Write docs/module-4-vla/ch13-vision-language-action.md with full content (1700-2500 words)
- [x] T119 [US1] Add code examples to ch13: VLA model, language parsing, action generation
- [x] T120 [US1] Add diagram placeholders to ch13 using Mermaid syntax
- [x] T121 [US1] Write docs/module-4-vla/ch14-capstone-humanoid.md with full content (3000-3500 words)
- [x] T122 [US1] Add code examples to ch14: full integration, behavior trees, safety
- [x] T123 [US1] Add diagram placeholders to ch14 using Mermaid syntax
- [x] T124 [US1] Add final capstone grading rubric to ch14

### M3.5 - Front & Back Matter Content

- [x] T125 [US1] Write docs/front-matter/foreword.md with full content (500-800 words)
- [x] T126 [US1] Write docs/front-matter/preface.md with full content (800-1200 words)
- [x] T127 [US1] Write docs/front-matter/introduction.md with full content (1000-1500 words)
- [x] T128 [US1] Write docs/front-matter/hardware-lab-setup.md with full content (1500-2000 words)
- [x] T129 [US1] Write docs/back-matter/appendix-a-student-kit.md with full content (1000-1500 words)
- [x] T130 [US1] Write docs/back-matter/appendix-b-cloud-setup.md with full content (800-1200 words)
- [x] T131 [US1] Write docs/back-matter/appendix-c-repos-docker.md with full content (600-1000 words)
- [x] T132 [US1] Write docs/back-matter/appendix-d-troubleshooting.md with full content (1500-2000 words)
- [x] T133 [US1] Write docs/back-matter/safety-ethics.md with full content (1000-1500 words)
- [x] T134 [US1] Compile docs/back-matter/glossary.md with 100+ terms
- [x] T135 [US1] Generate docs/back-matter/index.md with cross-references

**Checkpoint**: All 14 chapters readable, navigation works, word counts verified

---

## Phase 4: User Story 2 - Interactive RAG Chatbot Assistance (P2)

**Goal**: Student can ask the chatbot questions and receive contextually relevant answers from book content

**Independent Test**: Open any chapter, click chat widget, ask "What is ROS 2?", verify response is accurate

### M5.1 - Text Ingestion Pipeline

- [x] T136 [US2] Create backend/app/services/rag/ingestion.py with Markdown parser
- [x] T137 [US2] Implement semantic chunking strategy in ingestion.py (section-based)
- [x] T138 [US2] Add metadata extraction (chapter, section, keywords) to ingestion.py
- [x] T139 [US2] Create backend/app/cli/ingest.py CLI tool for book ingestion

### M5.3 - Qdrant Embeddings with OpenAI

- [x] T140 [US2] Create backend/app/services/rag/embeddings.py with OpenAI text-embedding-3-small
- [x] T141 [US2] Implement collection creation in backend/app/db/qdrant.py with proper indexing
- [x] T142 [US2] Build vector upsert pipeline in embeddings.py
- [x] T143 [US2] Add similarity search function with metadata filtering in embeddings.py

### M5.4 - OpenAI Agent RAG Implementation

- [x] T144 [US2] Install OpenAI Agents SDK in backend/pyproject.toml
- [x] T145 [US2] Create backend/app/services/rag/chat.py with OpenAI Agent initialization
- [x] T146 [US2] Configure Agent with book content retrieval tool in chat.py
- [x] T147 [US2] Implement context injection from Qdrant retrieval in chat.py
- [x] T148 [US2] Add response formatting with source citations in chat.py
- [x] T149 [US2] Implement streaming response support in chat.py

### M5.4 - API Endpoints

- [x] T150 [US2] Create backend/app/api/routes/chat.py with POST /api/chat endpoint
- [x] T151 [US2] Add GET /api/chat/history endpoint in chat.py
- [x] T152 [US2] Add POST /api/chat/feedback endpoint in chat.py
- [x] T153 [US2] Create backend/app/models/chat_history.py with ChatHistory model
- [x] T154 [US2] Implement WebSocket /ws/chat endpoint for streaming in chat.py
- [x] T155 [US2] Add rate limiting middleware in backend/app/api/deps.py
- [x] T156 [US2] Register chat routes in backend/app/main.py

### M5.5 - UI Integration in Docusaurus

- [x] T157 [US2] Create src/components/ChatBot/ChatBot.tsx with floating widget
- [x] T158 [US2] Create src/components/ChatBot/ChatMessage.tsx with message rendering
- [x] T159 [US2] Create src/components/ChatBot/ChatInput.tsx with input field and send button
- [x] T160 [US2] Create src/services/api.ts with chat API client
- [x] T161 [US2] Implement WebSocket connection in api.ts for streaming
- [x] T162 [US2] Add chapter-aware context injection in src/theme/DocItem/index.tsx
- [x] T163 [US2] Style ChatBot components with Docusaurus theme variables
- [x] T164 [US2] Add mobile-responsive styles for chat widget

**Checkpoint**: Chat widget visible on all pages, asks "What is ROS 2?" returns relevant content with sources

---

## Phase 5: User Story 3 - Lab Exercise Completion (P2)

**Goal**: Student can complete hands-on labs with step-by-step instructions and verify results

**Independent Test**: Follow ch01 lab instructions, run code, verify expected output matches documentation

### M4.1 - ROS 2 Labs (Chapters 1-5)

- [x] T165 [US3] Create labs/module-1/ch01-hello-robot/package.xml with ROS 2 package
- [x] T166 [US3] Create labs/module-1/ch01-hello-robot/setup.py with package config
- [x] T167 [US3] Create labs/module-1/ch01-hello-robot/hello_node.py with publisher
- [x] T168 [US3] Create labs/module-1/ch01-hello-robot/INSTRUCTIONS.md with step-by-step
- [x] T169 [US3] Create labs/module-1/ch02-sensor-fusion/sensor_node.py with camera+LIDAR
- [x] T170 [US3] Create labs/module-1/ch02-sensor-fusion/INSTRUCTIONS.md with step-by-step
- [x] T171 [US3] Create labs/module-1/ch03-ros2-deep-dive/lifecycle_node.py with lifecycle
- [x] T172 [US3] Create labs/module-1/ch03-ros2-deep-dive/service_node.py with service
- [x] T173 [US3] Create labs/module-1/ch03-ros2-deep-dive/action_node.py with action
- [x] T174 [US3] Create labs/module-1/ch03-ros2-deep-dive/INSTRUCTIONS.md with step-by-step
- [x] T175 [US3] Create labs/module-1/ch04-urdf-humanoid/humanoid.urdf with robot model
- [x] T176 [US3] Create labs/module-1/ch04-urdf-humanoid/humanoid.xacro with macros
- [x] T177 [US3] Create labs/module-1/ch04-urdf-humanoid/INSTRUCTIONS.md with step-by-step
- [x] T178 [US3] Create labs/module-1/ch05-edge-controller/edge_node.py with optimized code
- [x] T179 [US3] Create labs/module-1/ch05-edge-controller/INSTRUCTIONS.md with deployment steps

### M4.2 - Digital Twin Labs (Chapters 6-7)

- [x] T180 [US3] Create labs/module-2/ch06-gazebo-world/robot_world.sdf with Gazebo world
- [x] T181 [US3] Create labs/module-2/ch06-gazebo-world/spawn_robot.py with spawning script
- [x] T182 [US3] Create labs/module-2/ch06-gazebo-world/INSTRUCTIONS.md with step-by-step
- [x] T183 [US3] Create labs/module-2/ch07-unity-twin/README.md with Unity project setup
- [x] T184 [US3] Create labs/module-2/ch07-unity-twin/RosBridge.cs with ROS 2 connector
- [x] T185 [US3] Create labs/module-2/ch07-unity-twin/INSTRUCTIONS.md with step-by-step

### M4.3 - Isaac Labs (Chapters 8-10)

- [x] T186 [US3] Create labs/module-3/ch08-isaac-scene/scene.usd with Isaac Sim scene
- [x] T187 [US3] Create labs/module-3/ch08-isaac-scene/spawn_robot.py with Isaac script
- [x] T188 [US3] Create labs/module-3/ch08-isaac-scene/INSTRUCTIONS.md with step-by-step
- [x] T189 [US3] Create labs/module-3/ch09-gpu-vision/perception_pipeline.py with GPU code
- [x] T190 [US3] Create labs/module-3/ch09-gpu-vision/INSTRUCTIONS.md with step-by-step
- [x] T191 [US3] Create labs/module-3/ch10-sim2real/train_nav.py with RL training
- [x] T192 [US3] Create labs/module-3/ch10-sim2real/deploy.py with transfer script
- [x] T193 [US3] Create labs/module-3/ch10-sim2real/INSTRUCTIONS.md with step-by-step

### M4.4 - VLA Labs (Chapters 11-14)

- [x] T194 [US3] Create labs/module-4/ch11-walking/balance_controller.py with ZMP
- [x] T195 [US3] Create labs/module-4/ch11-walking/gait_generator.py with walking
- [x] T196 [US3] Create labs/module-4/ch11-walking/INSTRUCTIONS.md with step-by-step
- [x] T197 [US3] Create labs/module-4/ch12-manipulation/grasp_planner.py with grasping
- [x] T198 [US3] Create labs/module-4/ch12-manipulation/force_controller.py with force control
- [x] T199 [US3] Create labs/module-4/ch12-manipulation/INSTRUCTIONS.md with step-by-step
- [x] T200 [US3] Create labs/module-4/ch13-vla-commands/vla_agent.py with VLA integration
- [x] T201 [US3] Create labs/module-4/ch13-vla-commands/language_parser.py with NL parsing
- [x] T202 [US3] Create labs/module-4/ch13-vla-commands/INSTRUCTIONS.md with step-by-step
- [x] T203 [US3] Create labs/module-4/ch14-capstone/full_system.py with integration
- [x] T204 [US3] Create labs/module-4/ch14-capstone/behavior_tree.py with behavior tree
- [x] T205 [US3] Create labs/module-4/ch14-capstone/INSTRUCTIONS.md with step-by-step

**Checkpoint**: All 14 labs executable, instructions match chapter content, expected outputs documented

---

## Phase 6: User Story 4 - Capstone Project Completion (P3)

**Goal**: Advanced student completes final capstone integrating all modules

**Independent Test**: Complete ch14 capstone following rubric, verify all integration requirements met

### Capstone Support Tasks

- [x] T206 [US4] Create docs/module-1-ros2/ch05-edge-capstone.md capstone section with detailed rubric
- [x] T207 [US4] Create docs/module-2-digital-twin/ch07-unity-capstone.md capstone section with detailed rubric
- [x] T208 [US4] Create docs/module-4-vla/ch14-capstone-humanoid.md capstone section with comprehensive rubric
- [x] T209 [US4] Add prerequisite checklist to ch14 referencing modules 1-3
- [x] T210 [US4] Create labs/module-4/ch14-capstone/grading_rubric.md with evaluation criteria
- [x] T211 [US4] Add demo configuration options to ch14 lab

**Checkpoint**: Capstone rubrics clear, prerequisites documented, demo options available

---

## Phase 7: User Story 5 - Personalized Learning Experience (P3)

**Goal**: User can sign in via OAuth, track progress, and get personalized content

**Independent Test**: Sign in via Google/GitHub OAuth, complete questionnaire, verify progress saves

**Clarification (2025-12-20)**: OAuth-only authentication, translation deferred to future phase

### M6.1 - OAuth-Only Authentication (Updated 2025-12-20)

- [x] T212 [US5] Install better-auth in package.json
- [x] T213 [US5] Create src/pages/auth/signin.tsx with Google/GitHub OAuth buttons only
- [x] T214 [US5] Create backend/app/services/auth.py with Better-Auth OAuth integration
- [x] T215 [US5] Create backend/app/api/routes/auth.py with OAuth endpoints
- [x] T216 [US5] Implement session management in auth.py with refresh tokens
- [x] T217 [US5] Add OAuth provider configuration (Google, GitHub) in auth.py
- [x] T218 [US5] Configure OAuth callback handling in auth.py

### M6.2 - User Background Questionnaire

- [x] T219 [US5] Create src/components/Personalization/BackgroundQuestionnaire.tsx
- [x] T220 [US5] Implement multi-step form in BackgroundQuestionnaire.tsx
- [x] T221 [US5] Create backend/app/api/routes/user.py with questionnaire endpoint
- [x] T222 [US5] Store questionnaire responses in user_preferences table
- [x] T223 [US5] Build recommendation engine in backend/app/services/personalization.py

### M6.3 - Personalized Chapter Rendering

- [x] T224 [US5] Create src/components/Personalization/ChapterRenderer.tsx with conditional content
- [x] T225 [US5] Implement difficulty adjustment based on user profile
- [x] T226 [US5] Add progress tracking API in backend/app/api/routes/user.py
- [x] T227 [US5] Create progress tracking UI component
- [x] T228 [US5] Show personalized recommendations in chapter sidebar

**Checkpoint**: Users can sign in via OAuth, complete questionnaire, progress tracked

---

### M7 - Translation System

**Status**: Core translation complete (sub-agent + RTL + fonts). Only background jobs deferred.

#### M7.1 - Urdu Translation Trigger

- [x] T229 Create src/components/Translation/UrduToggle.tsx with language switch
- [x] T230 Create backend/app/api/routes/translation.py with translation endpoint
- [x] T231 Create backend/app/services/translation.py with OpenAI translation
- [x] T232 Implement RTL layout support in Docusaurus theme (src/css/custom.css)
- [x] T233 Add Urdu font loading in docusaurus.config.ts (Noto Nastaliq Urdu)

#### M7.2 - Translation Cache (DEFERRED - Sub-agent handles translation)

- [x] T234 Create backend/app/models/translation_cache.py with cache model
- [ ] T235 [DEFERRED] Create backend/migrations/versions/006_create_translation_jobs.py
- [ ] T236 [DEFERRED] Implement cache-first retrieval in translation.py
- [ ] T237 [DEFERRED] Add background translation jobs with Celery/ARQ
- [ ] T238 [DEFERRED] Create translation quality review queue

---

## Phase 8: Claude Code Subagents & Skills (M8)

**Goal**: Reusable skills for content generation acceleration

### M8.1 - Writing Skill

- [x] T239 Create .claude/commands/write-chapter.md with chapter template skill
- [x] T240 Add word count validation logic to write-chapter.md
- [x] T241 Add chapter structure enforcement to write-chapter.md
- [x] T242 Implement RAG integration point injection in write-chapter.md

### M8.2 - Code Generation Skill

- [x] T243 Create .claude/commands/generate-ros2-code.md with ROS 2 templates
- [x] T244 Create .claude/commands/generate-simulation-code.md with sim templates
- [x] T245 Add test generation capability to code skills
- [x] T246 Add docstring generation to code skills

### M8.3 - Diagram Generation Skill

- [x] T247 Create .claude/commands/generate-diagram.md with Mermaid templates
- [x] T248 Add architecture diagram templates to generate-diagram.md
- [x] T249 Add flowchart generation to generate-diagram.md
- [x] T250 Add sequence diagram capability to generate-diagram.md

### M8.4 - Lab Exercise Skill

- [x] T251 Create .claude/commands/create-lab.md with lab template
- [x] T252 Add step-by-step instruction generator to create-lab.md
- [x] T253 Add acceptance criteria generation to create-lab.md
- [x] T254 Add troubleshooting section generator to create-lab.md

**Checkpoint**: Skills available via /write-chapter, /generate-ros2-code, etc.

---

## Phase 9: Deployment (M9)

**Goal**: Production deployment with CI/CD

### M9.1 - GitHub Actions CI

- [x] T255 Create .github/workflows/test.yml with frontend + backend tests
- [x] T256 [P] Create .github/workflows/lint.yml with ESLint + Ruff
- [x] T257 [P] Create .github/workflows/build.yml with build verification
- [x] T258 [P] Create .github/workflows/security.yml with CodeQL + Dependabot

### M9.2 - Vercel Production Build

- [x] T259 Configure production environment variables in Vercel dashboard
- [x] T260 Set up custom domain in vercel.json
- [x] T261 Enable edge caching with proper cache headers
- [x] T262 Configure redirects and security headers

### M9.3 - Backend Deployment

- [x] T263 Create railway.toml or fly.toml for backend deployment
- [x] T264 Configure Neon Postgres connection string in production
- [x] T265 Set up Qdrant Cloud and configure connection
- [x] T266 Configure secrets management (vault/env vars)

### M9.4 - Testing & Acceptance

- [x] T267 Run full E2E test suite with Playwright (playwright.config.ts, e2e/*.spec.ts)
- [ ] T268 Perform load testing with k6 or artillery
- [x] T269 Validate all SC-001 through SC-010 acceptance criteria (docs/ACCEPTANCE_CRITERIA.md)
- [x] T270 Complete security audit checklist (docs/SECURITY_AUDIT.md)
- [x] T271 Create production runbook documentation (docs/RUNBOOK.md)

**Checkpoint**: Production deployment live, all tests pass, performance metrics green

---

## Phase 10: Polish & Cross-Cutting Concerns

**Purpose**: Final validation and documentation

- [x] T272 [P] Validate all chapter word counts are within spec tolerance
- [x] T273 [P] Verify all code examples execute without errors
- [x] T274 [P] Check all internal links and cross-references
- [x] T275 [P] Validate glossary terms match chapter usage
- [x] T276 [P] Run accessibility audit on Docusaurus site (e2e/navigation.spec.ts)
- [ ] T277 [P] Optimize images and assets for performance
- [x] T278 Create CONTRIBUTING.md with development guidelines
- [x] T279 Update README.md with full project documentation

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1 (Setup) → Phase 2 (Foundational) → [User Stories can start]
                                         ↓
                          ┌──────────────┼──────────────┐
                          ↓              ↓              ↓
                     Phase 3         Phase 4        Phase 5
                      (US1)          (US2)          (US3)
                          ↓              ↓              ↓
                          └──────────────┼──────────────┘
                                         ↓
                                    Phase 6 (US4)
                                         ↓
                                    Phase 7 (US5)
                                         ↓
                                    Phase 8 (Skills)
                                         ↓
                                    Phase 9 (Deploy)
                                         ↓
                                    Phase 10 (Polish)
```

### User Story Dependencies

| Story | Depends On | Can Start After |
|-------|------------|-----------------|
| US1 | Phase 2 | Foundational complete |
| US2 | Phase 2, partial US1 (M3.1) | Backend + some content |
| US3 | US1 (chapter content exists) | Chapter writing |
| US4 | US1, US3 | All chapters + labs |
| US5 | Phase 2 | Foundational complete |

### Critical Path

```
T001-T018 → T019-T079 → T080-T135 → T136-T164 → T256-T272
  Setup    Foundational   Content      RAG        Deploy
```

---

## Parallel Execution Examples

### Setup Phase (all [P] tasks)

```bash
# Launch in parallel:
Task: T001 "Create .gitignore"
Task: T002 "Create README.md"
Task: T003 "Add LICENSE"
```

### Chapter Placeholders (M2.2)

```bash
# Launch all chapter placeholders in parallel:
Task: T026-T042 "Create placeholder ch01-ch14"
```

### Content Writing (US1 - can parallelize by module)

```bash
# Module 1 writer:
Task: T080-T095 "Write chapters 1-5"

# Module 2 writer (parallel):
Task: T096-T102 "Write chapters 6-7"

# Module 3 writer (parallel):
Task: T103-T111 "Write chapters 8-10"

# Module 4 writer (parallel):
Task: T112-T124 "Write chapters 11-14"
```

### RAG Implementation (US2)

```bash
# Backend tasks (sequential):
T136 → T137 → T138 → T139 (ingestion)
T140 → T141 → T142 → T143 (embeddings)
T144 → T145 → T146 → T147 → T148 → T149 (OpenAI Agent)

# Frontend tasks (parallel with backend after API):
T157 → T158 → T159 → T160 → T161 → T162 → T163 → T164
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T018)
2. Complete Phase 2: Foundational (T019-T079)
3. Complete Phase 3: US1 Content (T080-T135)
4. **STOP and VALIDATE**: Book readable end-to-end
5. Deploy to Vercel preview

### Incremental Delivery

1. MVP: Book content only (US1) → Deploy
2. Add RAG Chatbot (US2) → Deploy
3. Add Labs (US3) → Deploy
4. Add Capstones (US4) → Deploy
5. Add Personalization (US5) → Deploy
6. Add Skills (M8) → Final polish

---

## Summary

| Category | Count | Completed | Remaining | Status |
|----------|-------|-----------|-----------|--------|
| **Total Tasks (Active)** | 269 | 267 | 2 | Nearly Complete |
| **Phase 1 (Setup)** | 18 | 18 | 0 | ✅ COMPLETE |
| **Phase 2 (Foundational)** | 61 | 61 | 0 | ✅ COMPLETE |
| **Phase 3 (US1 Content)** | 56 | 56 | 0 | ✅ COMPLETE |
| **Phase 4 (US2 RAG)** | 29 | 29 | 0 | ✅ COMPLETE |
| **Phase 5 (US3 Labs)** | 41 | 41 | 0 | ✅ COMPLETE |
| **Phase 6 (US4 Capstone)** | 6 | 6 | 0 | ✅ COMPLETE |
| **Phase 7 (US5 Auth + Personalization)** | 17 | 17 | 0 | ✅ COMPLETE |
| **Phase 7 (M7 Translation)** | 10 | 6 | 4 | ✅ Core Complete |
| **Phase 8 (Skills)** | 16 | 16 | 0 | ✅ COMPLETE |
| **Phase 9 (Deployment)** | 17 | 16 | 1 | ✅ Nearly Complete |
| **Phase 10 (Polish)** | 8 | 7 | 1 | ✅ Nearly Complete |

**Last Updated**: 2025-12-20

**New Files Created**:
- `playwright.config.ts` - E2E test configuration
- `e2e/navigation.spec.ts` - Navigation & accessibility tests
- `e2e/chat.spec.ts` - RAG chatbot tests
- `e2e/auth.spec.ts` - Authentication tests
- `docs/SECURITY_AUDIT.md` - Security checklist
- `docs/ACCEPTANCE_CRITERIA.md` - SC-001 to SC-010 validation

**Translation System**:
- ✅ Sub-agent `content-translator` for English↔Urdu translation
- ✅ RTL layout support (src/css/custom.css)
- ✅ Urdu fonts (Noto Nastaliq Urdu)
- ✅ Language toggle component
- Deferred: Background translation jobs (T235-T238)

**Remaining Tasks (2 total)**:
- T268: Load testing with k6 or artillery
- T277: Optimize images and assets

**All MVP Features Complete** - Ready for production
