---
id: 012
title: Plan Update Post Clarification
stage: plan
date: 2025-12-20
surface: agent
model: claude-opus-4-5-20251101
feature: 001-physical-ai-textbook
branch: main
user: user
command: /sp.plan
labels: ["plan", "clarification-sync", "nfr-update"]
links:
  spec: specs/001-physical-ai-textbook/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/001-physical-ai-textbook/plan.md
  - specs/001-physical-ai-textbook/research.md
  - specs/001-physical-ai-textbook/data-model.md
  - specs/001-physical-ai-textbook/contracts/openapi.yaml
tests:
  - null
---

## Prompt

/sp.plan

User executed the planning workflow to update existing plan artifacts based on clarifications from `/sp.clarify` session on 2025-12-20.

## Response snapshot

Updated all plan artifacts to reflect clarification decisions:

### plan.md Updates:
- Performance goal: Updated RAG latency to <3s (p95)
- Constraints: Added OAuth-only auth (Google/GitHub), English-only MVP
- Constitution check: Multi-language support updated to "English only for MVP"
- M6.1: Renamed to "OAuth-Only Authentication", removed password flows
- M7: Entire Translation System milestone marked as DEFERRED
- Dependency graph: Removed M7 from M9 dependencies
- Added clarification_updates tracking section

### research.md Updates:
- Section 5 (Authentication): Added clarification note for OAuth-only
- Section 8 (Translation): Marked as DEFERRED TO FUTURE PHASE

### data-model.md Updates:
- User table: Removed password_hash, made oauth_provider/oauth_id required
- user_preferences: Updated preferred_language to English-only
- ChatHistory: Added RAG knowledge scope documentation (current + completed chapters)

### contracts/openapi.yaml Updates:
- Removed /auth/signup and /auth/signin endpoints
- Removed SignupRequest and SigninRequest schemas
- Updated preferred_language enum to [en] only
- Added clarification notes throughout

## Outcome

- Impact: All plan artifacts synchronized with clarification decisions
- Tests: N/A (planning phase)
- Files: 4 modified
- Next prompts: /sp.tasks (to regenerate task breakdown with updated milestones)
- Reflection: Efficient update of existing comprehensive plan with targeted changes

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A
