# Tasks: Physical AI & Robotics Textbook

**Input**: Design documents from `/specs/001-physical-ai-textbook/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/openapi.yaml, quickstart.md
**Branch**: `001-physical-ai-textbook`
**Generated**: 2025-12-10

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
| US5 | Personalized Learning Experience | P3 | M6, M7 |

---

## Phase 1: Setup (M1 - Project Bootstrap)

**Purpose**: Initialize repository, Docusaurus, backend scaffolding, and CI/CD

### M1.1 - Repository Setup

- [ ] T001 [P] Create .gitignore with Node, Python, IDE, and env exclusions in .gitignore
- [ ] T002 [P] Create README.md with project overview and quickstart link in README.md
- [ ] T003 [P] Add MIT LICENSE file in LICENSE
- [ ] T004 Create branch structure (main, develop) via git commands

### M1.2 - Docusaurus Initialization

- [ ] T005 Initialize Docusaurus 3.x project with TypeScript via npx create-docusaurus@latest
- [ ] T006 Configure docusaurus.config.ts with project metadata and theme settings
- [ ] T007 [P] Create sidebars.ts with initial structure for modules
- [ ] T008 [P] Configure package.json with scripts (start, build, serve, typecheck, lint)
- [ ] T009 Set up custom theme structure in src/theme/DocItem/index.tsx

### M1.3 - Vercel Configuration

- [ ] T010 Create vercel.json with build settings and environment variables
- [ ] T011 [P] Configure .env.example with required frontend environment variables

### M1.4 - Backend Scaffolding

- [ ] T012 Create backend/ directory structure per plan.md
- [ ] T013 Initialize FastAPI project in backend/pyproject.toml with dependencies
- [ ] T014 [P] Create backend/.env.example with required environment variables
- [ ] T015 Create backend/app/main.py with FastAPI app and health endpoint
- [ ] T016 Create backend/app/config.py with settings using pydantic-settings
- [ ] T017 [P] Create docker-compose.yml with postgres, qdrant services
- [ ] T018 [P] Create backend/Dockerfile for FastAPI containerization

**Checkpoint**: Project structure ready, dev servers start (`npm run start`, `uvicorn app.main:app`)

---

## Phase 2: Foundational (M2 - Book Skeleton + DB Setup)

**Purpose**: Create book structure, initialize databases - BLOCKS all user stories

### M2.1 - Front Matter Structure

- [ ] T019 Create docs/front-matter/ directory with _category_.json
- [ ] T020 [P] Create placeholder docs/front-matter/foreword.md with frontmatter
- [ ] T021 [P] Create placeholder docs/front-matter/preface.md with frontmatter
- [ ] T022 [P] Create placeholder docs/front-matter/introduction.md with frontmatter
- [ ] T023 [P] Create placeholder docs/front-matter/hardware-lab-setup.md with frontmatter
- [ ] T024 Update sidebars.ts to include front-matter section

### M2.2 - Module Directories

- [ ] T025 Create docs/module-1-ros2/ directory with _category_.json
- [ ] T026 [P] Create placeholder docs/module-1-ros2/ch01-welcome-first-node.md
- [ ] T027 [P] Create placeholder docs/module-1-ros2/ch02-sensors-perception.md
- [ ] T028 [P] Create placeholder docs/module-1-ros2/ch03-ros2-architecture.md
- [ ] T029 [P] Create placeholder docs/module-1-ros2/ch04-urdf-humanoid.md
- [ ] T030 [P] Create placeholder docs/module-1-ros2/ch05-edge-capstone.md
- [ ] T031 Create docs/module-2-digital-twin/ directory with _category_.json
- [ ] T032 [P] Create placeholder docs/module-2-digital-twin/ch06-gazebo-physics.md
- [ ] T033 [P] Create placeholder docs/module-2-digital-twin/ch07-unity-capstone.md
- [ ] T034 Create docs/module-3-isaac/ directory with _category_.json
- [ ] T035 [P] Create placeholder docs/module-3-isaac/ch08-isaac-sim.md
- [ ] T036 [P] Create placeholder docs/module-3-isaac/ch09-isaac-ros-gpu.md
- [ ] T037 [P] Create placeholder docs/module-3-isaac/ch10-nav-rl-sim2real.md
- [ ] T038 Create docs/module-4-vla/ directory with _category_.json
- [ ] T039 [P] Create placeholder docs/module-4-vla/ch11-humanoid-locomotion.md
- [ ] T040 [P] Create placeholder docs/module-4-vla/ch12-dexterous-manipulation.md
- [ ] T041 [P] Create placeholder docs/module-4-vla/ch13-vision-language-action.md
- [ ] T042 [P] Create placeholder docs/module-4-vla/ch14-capstone-humanoid.md
- [ ] T043 Update sidebars.ts to include all module sections

### M2.3 - Back Matter Structure

- [ ] T044 Create docs/back-matter/ directory with _category_.json
- [ ] T045 [P] Create placeholder docs/back-matter/appendix-a-student-kit.md
- [ ] T046 [P] Create placeholder docs/back-matter/appendix-b-cloud-setup.md
- [ ] T047 [P] Create placeholder docs/back-matter/appendix-c-repos-docker.md
- [ ] T048 [P] Create placeholder docs/back-matter/appendix-d-troubleshooting.md
- [ ] T049 [P] Create placeholder docs/back-matter/safety-ethics.md
- [ ] T050 [P] Create placeholder docs/back-matter/glossary.md
- [ ] T051 [P] Create placeholder docs/back-matter/index.md
- [ ] T052 Update sidebars.ts to include back-matter section

### M2.4 - Labs Directory Structure

- [ ] T053 Create labs/ directory structure matching modules
- [ ] T054 [P] Create labs/module-1/ch01-hello-robot/README.md with lab template
- [ ] T055 [P] Create labs/module-1/ch02-sensor-fusion/README.md with lab template
- [ ] T056 [P] Create labs/module-1/ch03-ros2-deep-dive/README.md with lab template
- [ ] T057 [P] Create labs/module-1/ch04-urdf-humanoid/README.md with lab template
- [ ] T058 [P] Create labs/module-1/ch05-edge-controller/README.md with lab template
- [ ] T059 [P] Create labs/module-2/ch06-gazebo-world/README.md with lab template
- [ ] T060 [P] Create labs/module-2/ch07-unity-twin/README.md with lab template
- [ ] T061 [P] Create labs/module-3/ch08-isaac-scene/README.md with lab template
- [ ] T062 [P] Create labs/module-3/ch09-gpu-vision/README.md with lab template
- [ ] T063 [P] Create labs/module-3/ch10-sim2real/README.md with lab template
- [ ] T064 [P] Create labs/module-4/ch11-walking/README.md with lab template
- [ ] T065 [P] Create labs/module-4/ch12-manipulation/README.md with lab template
- [ ] T066 [P] Create labs/module-4/ch13-vla-commands/README.md with lab template
- [ ] T067 [P] Create labs/module-4/ch14-capstone/README.md with lab template
- [ ] T068 Create labs/docker-compose.yml for ROS 2 development environment

### Database Foundation (M5.2)

- [ ] T069 Create backend/app/db/postgres.py with Neon connection pool
- [ ] T070 Create backend/app/models/user.py with User SQLAlchemy model
- [ ] T071 [P] Create backend/app/models/session.py with Session model
- [ ] T072 [P] Create backend/app/models/preference.py with UserPreference model
- [ ] T073 Create backend/alembic.ini and migrations/ directory
- [ ] T074 Create backend/migrations/versions/001_create_users.py
- [ ] T075 Create backend/migrations/versions/002_create_sessions.py
- [ ] T076 Create backend/migrations/versions/003_create_user_preferences.py
- [ ] T077 Create backend/migrations/versions/004_create_chat_history.py
- [ ] T078 Create backend/migrations/versions/005_create_translation_cache.py
- [ ] T079 Create backend/app/db/qdrant.py with Qdrant client initialization

**Checkpoint**: Book skeleton visible at localhost:3000, database migrations run cleanly

---

## Phase 3: User Story 1 - Sequential Chapter Learning (P1) 🎯 MVP

**Goal**: Student can read chapters sequentially with learning objectives, code examples, and navigation

**Independent Test**: Open Chapter 1, verify objectives visible, code examples render, navigation to Chapter 2 works

### M3.1 - Module 1: ROS 2 Content (Chapters 1-5)

- [ ] T080 [US1] Write docs/module-1-ros2/ch01-welcome-first-node.md with full content (1700-2500 words)
- [ ] T081 [US1] Add code examples to ch01: ROS 2 node, publisher/subscriber in docs/module-1-ros2/ch01-welcome-first-node.md
- [ ] T082 [US1] Add diagram placeholders to ch01 using Mermaid syntax
- [ ] T083 [US1] Write docs/module-1-ros2/ch02-sensors-perception.md with full content (1700-2500 words)
- [ ] T084 [US1] Add code examples to ch02: camera, LIDAR, IMU processing
- [ ] T085 [US1] Add diagram placeholders to ch02 using Mermaid syntax
- [ ] T086 [US1] Write docs/module-1-ros2/ch03-ros2-architecture.md with full content (1700-2500 words)
- [ ] T087 [US1] Add code examples to ch03: lifecycle, services, actions, parameters
- [ ] T088 [US1] Add diagram placeholders to ch03 using Mermaid syntax
- [ ] T089 [US1] Write docs/module-1-ros2/ch04-urdf-humanoid.md with full content (1700-2500 words)
- [ ] T090 [US1] Add code examples to ch04: URDF, Xacro, mesh integration
- [ ] T091 [US1] Add diagram placeholders to ch04 using Mermaid syntax
- [ ] T092 [US1] Write docs/module-1-ros2/ch05-edge-capstone.md with full content (3000-3500 words)
- [ ] T093 [US1] Add code examples to ch05: cross-compilation, edge deployment
- [ ] T094 [US1] Add diagram placeholders to ch05 using Mermaid syntax
- [ ] T095 [US1] Add capstone grading rubric to ch05

### M3.2 - Module 2: Digital Twin Content (Chapters 6-7)

- [ ] T096 [US1] Write docs/module-2-digital-twin/ch06-gazebo-physics.md with full content (1700-2500 words)
- [ ] T097 [US1] Add code examples to ch06: world file, model spawning, sensor plugins
- [ ] T098 [US1] Add diagram placeholders to ch06 using Mermaid syntax
- [ ] T099 [US1] Write docs/module-2-digital-twin/ch07-unity-capstone.md with full content (3000-3500 words)
- [ ] T100 [US1] Add code examples to ch07: Unity-ROS 2 bridge, visualization, sync
- [ ] T101 [US1] Add diagram placeholders to ch07 using Mermaid syntax
- [ ] T102 [US1] Add capstone grading rubric to ch07

### M3.3 - Module 3: NVIDIA Isaac Content (Chapters 8-10)

- [ ] T103 [US1] Write docs/module-3-isaac/ch08-isaac-sim.md with full content (1700-2500 words)
- [ ] T104 [US1] Add code examples to ch08: scene creation, asset import, scripting
- [ ] T105 [US1] Add diagram placeholders to ch08 using Mermaid syntax
- [ ] T106 [US1] Write docs/module-3-isaac/ch09-isaac-ros-gpu.md with full content (1700-2500 words)
- [ ] T107 [US1] Add code examples to ch09: GPU perception, object detection, depth
- [ ] T108 [US1] Add diagram placeholders to ch09 using Mermaid syntax
- [ ] T109 [US1] Write docs/module-3-isaac/ch10-nav-rl-sim2real.md with full content (1700-2500 words)
- [ ] T110 [US1] Add code examples to ch10: Nav2, RL training, domain randomization
- [ ] T111 [US1] Add diagram placeholders to ch10 using Mermaid syntax

### M3.4 - Module 4: VLA + Capstone Content (Chapters 11-14)

- [ ] T112 [US1] Write docs/module-4-vla/ch11-humanoid-locomotion.md with full content (1700-2500 words)
- [ ] T113 [US1] Add code examples to ch11: ZMP, balance controller, gait generation
- [ ] T114 [US1] Add diagram placeholders to ch11 using Mermaid syntax
- [ ] T115 [US1] Write docs/module-4-vla/ch12-dexterous-manipulation.md with full content (1700-2500 words)
- [ ] T116 [US1] Add code examples to ch12: hand kinematics, grasp planning, force control
- [ ] T117 [US1] Add diagram placeholders to ch12 using Mermaid syntax
- [ ] T118 [US1] Write docs/module-4-vla/ch13-vision-language-action.md with full content (1700-2500 words)
- [ ] T119 [US1] Add code examples to ch13: VLA model, language parsing, action generation
- [ ] T120 [US1] Add diagram placeholders to ch13 using Mermaid syntax
- [ ] T121 [US1] Write docs/module-4-vla/ch14-capstone-humanoid.md with full content (3000-3500 words)
- [ ] T122 [US1] Add code examples to ch14: full integration, behavior trees, safety
- [ ] T123 [US1] Add diagram placeholders to ch14 using Mermaid syntax
- [ ] T124 [US1] Add final capstone grading rubric to ch14

### M3.5 - Front & Back Matter Content

- [ ] T125 [US1] Write docs/front-matter/foreword.md with full content (500-800 words)
- [ ] T126 [US1] Write docs/front-matter/preface.md with full content (800-1200 words)
- [ ] T127 [US1] Write docs/front-matter/introduction.md with full content (1000-1500 words)
- [ ] T128 [US1] Write docs/front-matter/hardware-lab-setup.md with full content (1500-2000 words)
- [ ] T129 [US1] Write docs/back-matter/appendix-a-student-kit.md with full content (1000-1500 words)
- [ ] T130 [US1] Write docs/back-matter/appendix-b-cloud-setup.md with full content (800-1200 words)
- [ ] T131 [US1] Write docs/back-matter/appendix-c-repos-docker.md with full content (600-1000 words)
- [ ] T132 [US1] Write docs/back-matter/appendix-d-troubleshooting.md with full content (1500-2000 words)
- [ ] T133 [US1] Write docs/back-matter/safety-ethics.md with full content (1000-1500 words)
- [ ] T134 [US1] Compile docs/back-matter/glossary.md with 100+ terms
- [ ] T135 [US1] Generate docs/back-matter/index.md with cross-references

**Checkpoint**: All 14 chapters readable, navigation works, word counts verified

---

## Phase 4: User Story 2 - Interactive RAG Chatbot Assistance (P2)

**Goal**: Student can ask the chatbot questions and receive contextually relevant answers from book content

**Independent Test**: Open any chapter, click chat widget, ask "What is ROS 2?", verify response is accurate

### M5.1 - Text Ingestion Pipeline

- [ ] T136 [US2] Create backend/app/services/rag/ingestion.py with Markdown parser
- [ ] T137 [US2] Implement semantic chunking strategy in ingestion.py (section-based)
- [ ] T138 [US2] Add metadata extraction (chapter, section, keywords) to ingestion.py
- [ ] T139 [US2] Create backend/app/cli/ingest.py CLI tool for book ingestion

### M5.3 - Qdrant Embeddings with OpenAI

- [ ] T140 [US2] Create backend/app/services/rag/embeddings.py with OpenAI text-embedding-3-small
- [ ] T141 [US2] Implement collection creation in backend/app/db/qdrant.py with proper indexing
- [ ] T142 [US2] Build vector upsert pipeline in embeddings.py
- [ ] T143 [US2] Add similarity search function with metadata filtering in embeddings.py

### M5.4 - OpenAI Agent RAG Implementation

- [ ] T144 [US2] Install OpenAI Agents SDK in backend/pyproject.toml
- [ ] T145 [US2] Create backend/app/services/rag/chat.py with OpenAI Agent initialization
- [ ] T146 [US2] Configure Agent with book content retrieval tool in chat.py
- [ ] T147 [US2] Implement context injection from Qdrant retrieval in chat.py
- [ ] T148 [US2] Add response formatting with source citations in chat.py
- [ ] T149 [US2] Implement streaming response support in chat.py

### M5.4 - API Endpoints

- [ ] T150 [US2] Create backend/app/api/routes/chat.py with POST /api/chat endpoint
- [ ] T151 [US2] Add GET /api/chat/history endpoint in chat.py
- [ ] T152 [US2] Add POST /api/chat/feedback endpoint in chat.py
- [ ] T153 [US2] Create backend/app/models/chat_history.py with ChatHistory model
- [ ] T154 [US2] Implement WebSocket /ws/chat endpoint for streaming in chat.py
- [ ] T155 [US2] Add rate limiting middleware in backend/app/api/deps.py
- [ ] T156 [US2] Register chat routes in backend/app/main.py

### M5.5 - UI Integration in Docusaurus

- [ ] T157 [US2] Create src/components/ChatBot/ChatBot.tsx with floating widget
- [ ] T158 [US2] Create src/components/ChatBot/ChatMessage.tsx with message rendering
- [ ] T159 [US2] Create src/components/ChatBot/ChatInput.tsx with input field and send button
- [ ] T160 [US2] Create src/services/api.ts with chat API client
- [ ] T161 [US2] Implement WebSocket connection in api.ts for streaming
- [ ] T162 [US2] Add chapter-aware context injection in src/theme/DocItem/index.tsx
- [ ] T163 [US2] Style ChatBot components with Docusaurus theme variables
- [ ] T164 [US2] Add mobile-responsive styles for chat widget

**Checkpoint**: Chat widget visible on all pages, asks "What is ROS 2?" returns relevant content with sources

---

## Phase 5: User Story 3 - Lab Exercise Completion (P2)

**Goal**: Student can complete hands-on labs with step-by-step instructions and verify results

**Independent Test**: Follow ch01 lab instructions, run code, verify expected output matches documentation

### M4.1 - ROS 2 Labs (Chapters 1-5)

- [ ] T165 [US3] Create labs/module-1/ch01-hello-robot/package.xml with ROS 2 package
- [ ] T166 [US3] Create labs/module-1/ch01-hello-robot/setup.py with package config
- [ ] T167 [US3] Create labs/module-1/ch01-hello-robot/hello_node.py with publisher
- [ ] T168 [US3] Create labs/module-1/ch01-hello-robot/INSTRUCTIONS.md with step-by-step
- [ ] T169 [US3] Create labs/module-1/ch02-sensor-fusion/sensor_node.py with camera+LIDAR
- [ ] T170 [US3] Create labs/module-1/ch02-sensor-fusion/INSTRUCTIONS.md with step-by-step
- [ ] T171 [US3] Create labs/module-1/ch03-ros2-deep-dive/lifecycle_node.py with lifecycle
- [ ] T172 [US3] Create labs/module-1/ch03-ros2-deep-dive/service_node.py with service
- [ ] T173 [US3] Create labs/module-1/ch03-ros2-deep-dive/action_node.py with action
- [ ] T174 [US3] Create labs/module-1/ch03-ros2-deep-dive/INSTRUCTIONS.md with step-by-step
- [ ] T175 [US3] Create labs/module-1/ch04-urdf-humanoid/humanoid.urdf with robot model
- [ ] T176 [US3] Create labs/module-1/ch04-urdf-humanoid/humanoid.xacro with macros
- [ ] T177 [US3] Create labs/module-1/ch04-urdf-humanoid/INSTRUCTIONS.md with step-by-step
- [ ] T178 [US3] Create labs/module-1/ch05-edge-controller/edge_node.py with optimized code
- [ ] T179 [US3] Create labs/module-1/ch05-edge-controller/INSTRUCTIONS.md with deployment steps

### M4.2 - Digital Twin Labs (Chapters 6-7)

- [ ] T180 [US3] Create labs/module-2/ch06-gazebo-world/robot_world.sdf with Gazebo world
- [ ] T181 [US3] Create labs/module-2/ch06-gazebo-world/spawn_robot.py with spawning script
- [ ] T182 [US3] Create labs/module-2/ch06-gazebo-world/INSTRUCTIONS.md with step-by-step
- [ ] T183 [US3] Create labs/module-2/ch07-unity-twin/README.md with Unity project setup
- [ ] T184 [US3] Create labs/module-2/ch07-unity-twin/RosBridge.cs with ROS 2 connector
- [ ] T185 [US3] Create labs/module-2/ch07-unity-twin/INSTRUCTIONS.md with step-by-step

### M4.3 - Isaac Labs (Chapters 8-10)

- [ ] T186 [US3] Create labs/module-3/ch08-isaac-scene/scene.usd with Isaac Sim scene
- [ ] T187 [US3] Create labs/module-3/ch08-isaac-scene/spawn_robot.py with Isaac script
- [ ] T188 [US3] Create labs/module-3/ch08-isaac-scene/INSTRUCTIONS.md with step-by-step
- [ ] T189 [US3] Create labs/module-3/ch09-gpu-vision/perception_pipeline.py with GPU code
- [ ] T190 [US3] Create labs/module-3/ch09-gpu-vision/INSTRUCTIONS.md with step-by-step
- [ ] T191 [US3] Create labs/module-3/ch10-sim2real/train_nav.py with RL training
- [ ] T192 [US3] Create labs/module-3/ch10-sim2real/deploy.py with transfer script
- [ ] T193 [US3] Create labs/module-3/ch10-sim2real/INSTRUCTIONS.md with step-by-step

### M4.4 - VLA Labs (Chapters 11-14)

- [ ] T194 [US3] Create labs/module-4/ch11-walking/balance_controller.py with ZMP
- [ ] T195 [US3] Create labs/module-4/ch11-walking/gait_generator.py with walking
- [ ] T196 [US3] Create labs/module-4/ch11-walking/INSTRUCTIONS.md with step-by-step
- [ ] T197 [US3] Create labs/module-4/ch12-manipulation/grasp_planner.py with grasping
- [ ] T198 [US3] Create labs/module-4/ch12-manipulation/force_controller.py with force control
- [ ] T199 [US3] Create labs/module-4/ch12-manipulation/INSTRUCTIONS.md with step-by-step
- [ ] T200 [US3] Create labs/module-4/ch13-vla-commands/vla_agent.py with VLA integration
- [ ] T201 [US3] Create labs/module-4/ch13-vla-commands/language_parser.py with NL parsing
- [ ] T202 [US3] Create labs/module-4/ch13-vla-commands/INSTRUCTIONS.md with step-by-step
- [ ] T203 [US3] Create labs/module-4/ch14-capstone/full_system.py with integration
- [ ] T204 [US3] Create labs/module-4/ch14-capstone/behavior_tree.py with behavior tree
- [ ] T205 [US3] Create labs/module-4/ch14-capstone/INSTRUCTIONS.md with step-by-step

**Checkpoint**: All 14 labs executable, instructions match chapter content, expected outputs documented

---

## Phase 6: User Story 4 - Capstone Project Completion (P3)

**Goal**: Advanced student completes final capstone integrating all modules

**Independent Test**: Complete ch14 capstone following rubric, verify all integration requirements met

### Capstone Support Tasks

- [ ] T206 [US4] Create docs/module-1-ros2/ch05-edge-capstone.md capstone section with detailed rubric
- [ ] T207 [US4] Create docs/module-2-digital-twin/ch07-unity-capstone.md capstone section with detailed rubric
- [ ] T208 [US4] Create docs/module-4-vla/ch14-capstone-humanoid.md capstone section with comprehensive rubric
- [ ] T209 [US4] Add prerequisite checklist to ch14 referencing modules 1-3
- [ ] T210 [US4] Create labs/module-4/ch14-capstone/grading_rubric.md with evaluation criteria
- [ ] T211 [US4] Add demo configuration options to ch14 lab

**Checkpoint**: Capstone rubrics clear, prerequisites documented, demo options available

---

## Phase 7: User Story 5 - Personalized Learning Experience (P3)

**Goal**: User can sign up, track progress, get personalized content, and translate to Urdu

**Independent Test**: Sign up, complete questionnaire, verify progress saves, toggle Urdu translation

### M6.1 - Better-Auth Signup/Signin

- [ ] T212 [US5] Install better-auth in package.json
- [ ] T213 [US5] Create src/pages/auth/signup.tsx with email/OAuth signup form
- [ ] T214 [US5] Create src/pages/auth/signin.tsx with signin form
- [ ] T215 [US5] Create backend/app/services/auth.py with Better-Auth integration
- [ ] T216 [US5] Create backend/app/api/routes/auth.py with auth endpoints
- [ ] T217 [US5] Implement session management in auth.py with refresh tokens
- [ ] T218 [US5] Add OAuth provider configuration (Google, GitHub) in auth.py
- [ ] T219 [US5] Add password reset flow in auth.py

### M6.2 - User Background Questionnaire

- [ ] T220 [US5] Create src/components/Personalization/BackgroundQuestionnaire.tsx
- [ ] T221 [US5] Implement multi-step form in BackgroundQuestionnaire.tsx
- [ ] T222 [US5] Create backend/app/api/routes/user.py with questionnaire endpoint
- [ ] T223 [US5] Store questionnaire responses in user_preferences table
- [ ] T224 [US5] Build recommendation engine in backend/app/services/personalization.py

### M6.3 - Personalized Chapter Rendering

- [ ] T225 [US5] Create src/components/Personalization/ChapterRenderer.tsx with conditional content
- [ ] T226 [US5] Implement difficulty adjustment based on user profile
- [ ] T227 [US5] Add progress tracking API in backend/app/api/routes/user.py
- [ ] T228 [US5] Create progress tracking UI component
- [ ] T229 [US5] Show personalized recommendations in chapter sidebar

### M7.1 - Urdu Translation Trigger

- [ ] T230 [US5] Create src/components/Translation/UrduToggle.tsx with language switch
- [ ] T231 [US5] Create backend/app/api/routes/translation.py with translation endpoint
- [ ] T232 [US5] Create backend/app/services/translation.py with OpenAI translation
- [ ] T233 [US5] Implement RTL layout support in Docusaurus theme
- [ ] T234 [US5] Add Urdu font loading in docusaurus.config.ts

### M7.2 - Translation Cache

- [ ] T235 [US5] Create backend/app/models/translation_cache.py with cache model
- [ ] T236 [US5] Create backend/migrations/versions/006_create_translation_jobs.py
- [ ] T237 [US5] Implement cache-first retrieval in translation.py
- [ ] T238 [US5] Add background translation jobs with Celery/ARQ
- [ ] T239 [US5] Create translation quality review queue

**Checkpoint**: Users can sign up, complete questionnaire, progress tracked, Urdu translation works

---

## Phase 8: Claude Code Subagents & Skills (M8)

**Goal**: Reusable skills for content generation acceleration

### M8.1 - Writing Skill

- [ ] T240 Create .claude/commands/write-chapter.md with chapter template skill
- [ ] T241 Add word count validation logic to write-chapter.md
- [ ] T242 Add chapter structure enforcement to write-chapter.md
- [ ] T243 Implement RAG integration point injection in write-chapter.md

### M8.2 - Code Generation Skill

- [ ] T244 Create .claude/commands/generate-ros2-code.md with ROS 2 templates
- [ ] T245 Create .claude/commands/generate-simulation-code.md with sim templates
- [ ] T246 Add test generation capability to code skills
- [ ] T247 Add docstring generation to code skills

### M8.3 - Diagram Generation Skill

- [ ] T248 Create .claude/commands/generate-diagram.md with Mermaid templates
- [ ] T249 Add architecture diagram templates to generate-diagram.md
- [ ] T250 Add flowchart generation to generate-diagram.md
- [ ] T251 Add sequence diagram capability to generate-diagram.md

### M8.4 - Lab Exercise Skill

- [ ] T252 Create .claude/commands/create-lab.md with lab template
- [ ] T253 Add step-by-step instruction generator to create-lab.md
- [ ] T254 Add acceptance criteria generation to create-lab.md
- [ ] T255 Add troubleshooting section generator to create-lab.md

**Checkpoint**: Skills available via /write-chapter, /generate-ros2-code, etc.

---

## Phase 9: Deployment (M9)

**Goal**: Production deployment with CI/CD

### M9.1 - GitHub Actions CI

- [ ] T256 Create .github/workflows/test.yml with frontend + backend tests
- [ ] T257 [P] Create .github/workflows/lint.yml with ESLint + Ruff
- [ ] T258 [P] Create .github/workflows/build.yml with build verification
- [ ] T259 [P] Create .github/workflows/security.yml with CodeQL + Dependabot

### M9.2 - Vercel Production Build

- [ ] T260 Configure production environment variables in Vercel dashboard
- [ ] T261 Set up custom domain in vercel.json
- [ ] T262 Enable edge caching with proper cache headers
- [ ] T263 Configure redirects and security headers

### M9.3 - Backend Deployment

- [ ] T264 Create railway.toml or fly.toml for backend deployment
- [ ] T265 Configure Neon Postgres connection string in production
- [ ] T266 Set up Qdrant Cloud and configure connection
- [ ] T267 Configure secrets management (vault/env vars)

### M9.4 - Testing & Acceptance

- [ ] T268 Run full E2E test suite with Playwright
- [ ] T269 Perform load testing with k6 or artillery
- [ ] T270 Validate all SC-001 through SC-010 acceptance criteria
- [ ] T271 Complete security audit checklist
- [ ] T272 Create production runbook documentation

**Checkpoint**: Production deployment live, all tests pass, performance metrics green

---

## Phase 10: Polish & Cross-Cutting Concerns

**Purpose**: Final validation and documentation

- [ ] T273 [P] Validate all chapter word counts are within spec tolerance
- [ ] T274 [P] Verify all code examples execute without errors
- [ ] T275 [P] Check all internal links and cross-references
- [ ] T276 [P] Validate glossary terms match chapter usage
- [ ] T277 [P] Run accessibility audit on Docusaurus site
- [ ] T278 [P] Optimize images and assets for performance
- [ ] T279 Create CONTRIBUTING.md with development guidelines
- [ ] T280 Update README.md with full project documentation

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

| Category | Count |
|----------|-------|
| **Total Tasks** | 280 |
| **Phase 1 (Setup)** | 18 |
| **Phase 2 (Foundational)** | 61 |
| **Phase 3 (US1 Content)** | 56 |
| **Phase 4 (US2 RAG)** | 29 |
| **Phase 5 (US3 Labs)** | 41 |
| **Phase 6 (US4 Capstone)** | 6 |
| **Phase 7 (US5 Personalization)** | 28 |
| **Phase 8 (Skills)** | 16 |
| **Phase 9 (Deployment)** | 17 |
| **Phase 10 (Polish)** | 8 |

**Parallel Opportunities**: 89 tasks marked [P]
**MVP Scope**: Phases 1-3 (US1) = 135 tasks
