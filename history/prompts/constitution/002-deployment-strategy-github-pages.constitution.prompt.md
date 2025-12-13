---
id: 002
title: Deployment Strategy GitHub Pages
stage: constitution
date: 2025-12-12
surface: agent
model: claude-opus-4-5-20251101
feature: none
branch: 001-physical-ai-textbook
user: user
command: /sp.constitution
labels: ["deployment", "github-pages", "constitution", "infrastructure"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
  - README.md
tests:
  - null
---

## Prompt

Deployment will be done manually via GitHub upload.
- Automated token-based deployment is not required.
- After upload, the project will be accessible via GitHub Pages.

## Response snapshot

Updated the project constitution from v1.0.0 to v1.0.1, adding a new "Deployment Strategy" section under Technical Constraints. The amendment specifies:
- Frontend hosting via GitHub Pages
- Manual upload deployment (no CI/CD tokens required)
- Local build process using `npm run build`
- Separate backend deployment for FastAPI
- Repository access via GitHub Pages URL

Also updated README.md Tech Stack section to reflect GitHub Pages deployment instead of Vercel.

## Outcome

- Impact: Constitution amended with deployment strategy clarification (PATCH version bump)
- Tests: N/A (documentation change)
- Files: 2 files modified (.specify/memory/constitution.md, README.md)
- Next prompts: Update tasks.md Phase 9 to align with GitHub Pages deployment
- Reflection: Clean amendment following semantic versioning; deployment strategy now documented

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): N/A (constitution update)
- Prompt variant (if applicable): Standard /sp.constitution invocation
- Next experiment: None needed - straightforward amendment
