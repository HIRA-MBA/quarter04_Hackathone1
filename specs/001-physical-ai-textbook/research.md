# Research: Physical AI & Robotics Textbook

**Branch**: `001-physical-ai-textbook` | **Date**: 2025-12-10

## Overview

This document captures technology decisions, alternatives considered, and rationale for the Physical AI Textbook implementation.

---

## 1. Documentation Platform

### Decision: Docusaurus 3.x

**Rationale**:
- Native MDX support for interactive components (chatbot, code playgrounds)
- Built-in versioning for future textbook editions
- Algolia DocSearch integration for content search
- React-based theming compatible with ChatBot components
- Static site generation for Vercel deployment
- Active community and Meta backing

**Alternatives Considered**:

| Option | Pros | Cons | Rejected Because |
|--------|------|------|------------------|
| GitBook | Easy setup, nice UI | Limited customization, paid for advanced features | Cannot embed custom React components |
| MkDocs | Python ecosystem, Material theme | No native React support | Would require separate frontend for chatbot |
| Nextra | Next.js native, modern | Smaller community, less mature | Docusaurus has better documentation features |
| VitePress | Fast, Vue-based | Vue ecosystem mismatch with React chatbot | Team expertise in React |

---

## 2. Backend Framework

### Decision: FastAPI 0.109+

**Rationale**:
- Native async support for RAG streaming responses
- Automatic OpenAPI documentation
- Pydantic validation for type safety
- Excellent Python ML/AI library compatibility
- WebSocket support for real-time chat

**Alternatives Considered**:

| Option | Pros | Cons | Rejected Because |
|--------|------|------|------------------|
| Flask | Simple, widely known | Sync by default, manual async | RAG needs efficient streaming |
| Django | Batteries included | Heavy for API-only service | Over-engineered for our use case |
| Express.js | JavaScript ecosystem | Different language from ML libraries | Python better for embeddings/NLP |
| Hono | Edge-native, fast | Young ecosystem | Less mature for production |

---

## 3. Vector Database

### Decision: Qdrant Cloud

**Rationale**:
- Native filtering by chapter/section metadata
- Efficient for ~30k word corpus (small-medium scale)
- Free tier sufficient for development
- gRPC support for low-latency queries
- Payload storage for chunk text (no separate lookup needed)

**Alternatives Considered**:

| Option | Pros | Cons | Rejected Because |
|--------|------|------|------------------|
| Pinecone | Managed, scalable | Expensive for small projects, vendor lock-in | Cost for hobby/edu project |
| Weaviate | GraphQL API, hybrid search | More complex setup | Overkill for our corpus size |
| ChromaDB | Simple, local-first | Limited production features | Scaling concerns for cloud deploy |
| pgvector | Single DB, Postgres native | Slower for pure vector search | Neon better for relational; Qdrant better for vectors |

---

## 4. Relational Database

### Decision: Neon Postgres (Serverless)

**Rationale**:
- Serverless scales to zero (cost-efficient)
- Postgres compatibility (Better-Auth support)
- Branching for dev/staging environments
- Built-in connection pooling
- Free tier generous for edu projects

**Alternatives Considered**:

| Option | Pros | Cons | Rejected Because |
|--------|------|------|------------------|
| Supabase | Full backend suite | More than needed, higher cost | Just need DB, not full BaaS |
| PlanetScale | MySQL, good DX | MySQL vs Postgres ecosystem | Better-Auth Postgres-optimized |
| Railway Postgres | Simple deploy | No serverless scaling | Would pay for idle time |
| SQLite (Turso) | Edge-native, fast | Limited for multi-user writes | Concurrent user sessions |

---

## 5. Authentication

### Decision: Better-Auth (OAuth-Only)

**Rationale**:
- TypeScript-native, modern auth library
- Self-hosted (no vendor lock-in)
- Postgres adapter built-in
- OAuth providers (Google, GitHub) supported
- Session management with refresh tokens
- **Clarification (2025-12-20)**: OAuth-only implementation - no email/password storage

**Alternatives Considered**:

| Option | Pros | Cons | Rejected Because |
|--------|------|------|------------------|
| NextAuth | Popular, mature | Next.js-centric, Docusaurus incompatible | Not designed for non-Next apps |
| Clerk | Great DX, managed | Paid, vendor lock-in | Budget constraints |
| Auth0 | Enterprise-grade | Complex, expensive at scale | Overkill for edu project |
| Lucia Auth | Simple, flexible | Less feature-complete | Better-Auth more batteries-included |

---

## 6. AI/Chat Integration

### Decision: OpenAI Agents SDK + GPT-4o-mini

**Rationale**:
- Agents SDK provides conversation management
- GPT-4o-mini cost-effective for high-volume edu use
- Function calling for structured responses
- Streaming support for responsive UX
- Wide context window for chapter-aware responses

**Alternatives Considered**:

| Option | Pros | Cons | Rejected Because |
|--------|------|------|------------------|
| Claude API | Longer context, reasoning | Higher latency for streaming | OpenAI streaming more mature |
| LangChain | Flexible, many integrations | Abstraction overhead | Direct SDK simpler for our use case |
| LlamaIndex | Great for RAG | Python-heavy, less streaming | OpenAI SDK better DX |
| Local LLM (Ollama) | Free, private | Requires GPU, inconsistent quality | Can't guarantee hardware availability |

---

## 7. Embedding Model

### Decision: OpenAI text-embedding-3-small

**Rationale**:
- 1536 dimensions (good balance of quality/cost)
- Lower cost than ada-002 with better performance
- Consistent with chat model provider
- Handles technical content well

**Alternatives Considered**:

| Option | Pros | Cons | Rejected Because |
|--------|------|------|------------------|
| text-embedding-3-large | Higher quality | More expensive, 3072 dims | Overkill for 30k word corpus |
| Cohere embed | Multilingual | Additional API key, complexity | Prefer single provider |
| sentence-transformers | Free, local | Requires GPU for speed | Hosting complexity |

---

## 8. Translation Approach

### Decision: DEFERRED TO FUTURE PHASE

**Clarification (2025-12-20)**: English-only for MVP. Translation system deferred to post-MVP phase.

**Original Decision (when implemented)**: OpenAI GPT-4o-mini with Caching

**Original Rationale** (preserved for future reference):
- Quality Urdu translation
- Same provider as chat (simpler ops)
- Cache translations in Postgres
- On-demand + background pre-translation

**Alternatives Considered** (preserved for future reference):

| Option | Pros | Cons | Rejected Because |
|--------|------|------|------------------|
| Google Translate API | Fast, cheap | Lower quality for technical content | Urdu technical terms poor |
| DeepL | High quality | No Urdu support | Language not available |
| Human translation | Best quality | Expensive, slow | Budget/time constraints |
| Azure Translator | Good Urdu support | Additional vendor | Prefer consolidated providers |

---

## 9. Deployment Platform

### Decision: Vercel (Docs) + Railway (API)

**Rationale**:
- Vercel: Best-in-class for static sites, free tier, preview deployments
- Railway: Simple Python deployment, Postgres-friendly, affordable
- Both have GitHub integration for CI/CD

**Alternatives Considered**:

| Option | Pros | Cons | Rejected Because |
|--------|------|------|------------------|
| Vercel (both) | Single platform | Python support limited | FastAPI better on Railway |
| AWS (all) | Full control | Complex setup, overkill | Team velocity over control |
| Render | Simple, affordable | Slower cold starts | Railway faster spin-up |
| Fly.io | Edge deployment | More complex config | Railway simpler for MVP |

---

## 10. CI/CD Strategy

### Decision: GitHub Actions

**Rationale**:
- Native GitHub integration
- Free for public repos
- Reusable workflows
- Matrix builds for testing
- Secrets management built-in

**Configuration**:
```yaml
# Workflows planned:
- test.yml: Run Vitest + pytest on PR
- deploy-docs.yml: Build and deploy to Vercel on main
- deploy-backend.yml: Deploy to Railway on main
- security.yml: Dependabot + CodeQL scanning
```

---

## 11. Code Quality Tools

### Decision: ESLint + Prettier (TS) / Ruff (Python)

**Rationale**:
- ESLint: Industry standard for TypeScript
- Prettier: Consistent formatting
- Ruff: Fast Python linting (replaces flake8, isort, black)
- Pre-commit hooks for enforcement

---

## Unresolved Items

All technical decisions resolved. No NEEDS CLARIFICATION items remaining.

---

## References

- [Docusaurus Documentation](https://docusaurus.io/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Neon Documentation](https://neon.tech/docs)
- [Better-Auth Documentation](https://www.better-auth.com/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
