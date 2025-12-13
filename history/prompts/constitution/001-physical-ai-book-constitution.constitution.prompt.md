---
id: 001
title: Physical AI Book Constitution Creation
stage: constitution
date: 2025-12-10
surface: agent
model: claude-opus-4-5-20251101
feature: none
branch: master
user: user
command: /sp.constitution
labels: ["constitution", "physical-ai", "robotics", "docusaurus", "rag-chatbot"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
tests:
  - none
---

## Prompt

Book Title: "Physical AI & Humanoid Robotics – Embodied Intelligence (2025–2026)"
Project Type: Docusaurus Technical Book
Writing Style: Clear, technical, step-by-step, practical with moderate code, Markdown only.

Scope:
- AI/Spec-Driven book creation using Spec-Kit Plus and Claude Code
- Integrated RAG chatbot with OpenAI Agents/ChatKit SDK, FastAPI, Neon Postgres, and Qdrant
- Personalized content per user
- Multi-language support (English + Urdu)
- Front matter: Foreword, Preface, Introduction, Hardware & Lab Setup Guide
- Back matter: Appendices, Safety & Ethics, Glossary, Index

Objectives:
1. Write each chapter with step-by-step instructions and code examples.
2. Integrate a RAG chatbot answering only from book content or user-selected text.
3. Implement optional personalization, translation, signup/signin.
4. Ensure content is modular, versioned, and reproducible.

Guidelines Regarding Book Structure:
- Book modules, chapters, and detailed content will be specified in the next SPEC step.
- Constitution should remain high-level, focusing on rules, constraints, and objectives, not chapter-by-chapter content.
- All tasks, specifications, and plans will follow the Constitution in subsequent steps.

Constraints:
- Exclude vendor comparisons and ethics commentary
- Use Markdown only
- Include code examples, lab instructions, and diagram placeholders
- Validate chapter structure and consistency

Output Format:
- Constitution in Markdown
- Module-wise, chapter-wise objectives NOT included (they go in SPEC)
- Key decisions and constraints highlighted
- Keep output concise to fit token limits

## Response snapshot

Created constitution v1.0.0 with 6 core principles:
1. Content-First Authoring - practical, step-by-step learning
2. Markdown-Only Format - GFM compatibility
3. Modular Architecture - independent chapter consumption
4. RAG-Integrated Learning - chatbot with book-only answers
5. Multi-Language Support - English + Urdu
6. Reproducible Lab Exercises - versioned, testable labs

Technical stack defined: Docusaurus, FastAPI, Qdrant, Neon Postgres, OpenAI Agents/ChatKit SDK.
Excluded content: vendor comparisons, ethics commentary, proprietary code.
Quality gates and chapter structure standardized.

## Outcome

- Impact: Established foundational governance for Physical AI book project
- Tests: N/A (constitution document)
- Files: .specify/memory/constitution.md created
- Next prompts: /sp.specify to define book modules and chapters
- Reflection: Constitution successfully captures high-level constraints without premature chapter details

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Proceed to SPEC phase
