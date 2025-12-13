<!--
  ============================================================================
  SYNC IMPACT REPORT
  ============================================================================
  Version change: 1.0.0 → 1.0.1

  Modified principles:
  - None (no principle changes)

  Added sections:
  - Deployment Strategy (under Technical Constraints)

  Removed sections:
  - None

  Templates requiring updates:
  - ✅ plan-template.md (no changes needed - generic)
  - ✅ spec-template.md (no changes needed - generic)
  - ✅ tasks-template.md (no changes needed - generic)
  - ⚠️ README.md (deployment section updated to reflect GitHub Pages)

  Follow-up TODOs:
  - Update tasks.md Phase 9 (Deployment) to reflect GitHub Pages strategy
  ============================================================================
-->

# Physical AI & Humanoid Robotics – Embodied Intelligence Constitution

## Core Principles

### I. Content-First Authoring

Every chapter MUST deliver practical, step-by-step learning value. Content requirements:
- Clear learning objectives stated at chapter start
- Progressive complexity from fundamentals to advanced topics
- Code examples that are complete, tested, and copy-paste ready
- Diagram placeholders with descriptive alt-text for visual concepts
- Lab exercises with expected outcomes documented

### II. Markdown-Only Format

All book content MUST use standard Markdown:
- GitHub-Flavored Markdown (GFM) for compatibility
- No proprietary markup or custom syntax
- Docusaurus-compatible MDX only where interactive features require it
- Code blocks with language identifiers for syntax highlighting
- Consistent heading hierarchy (H1 for chapter, H2 for sections, H3 for subsections)

### III. Modular Architecture

Book structure MUST support independent consumption:
- Each module is self-contained with clear prerequisites listed
- Chapters can be read independently where feasible
- Cross-references use relative links, not absolute paths
- Front matter (Foreword, Preface, Introduction, Hardware & Lab Setup) introduces context
- Back matter (Appendices, Safety & Ethics, Glossary, Index) provides reference material
- Version control at chapter level for targeted updates

### IV. RAG-Integrated Learning

The integrated chatbot MUST enhance learning:
- Answers sourced ONLY from book content or user-selected text
- OpenAI Agents/ChatKit SDK for conversational interface
- FastAPI backend for API operations
- Neon Postgres for user data and preferences
- Qdrant for vector embeddings and semantic search
- No hallucinated content; graceful "not found" responses when content unavailable

### V. Multi-Language Support

Content MUST support internationalization:
- Primary language: English
- Secondary language: Urdu
- Translation-ready structure with language keys
- Cultural context preserved in examples where relevant
- UI strings externalized for localization

### VI. Reproducible Lab Exercises

All practical exercises MUST be reproducible:
- Hardware requirements explicitly listed in setup guide
- Software dependencies pinned to specific versions
- Step-by-step instructions with expected outputs
- Troubleshooting sections for common issues
- Safety warnings prominently displayed for hardware interactions

## Technical Constraints

### Stack Requirements

| Component | Technology | Purpose |
|-----------|------------|---------|
| Documentation | Docusaurus | Book rendering and navigation |
| Backend API | FastAPI | RAG chatbot and user services |
| Vector DB | Qdrant | Semantic search and embeddings |
| Relational DB | Neon Postgres | User data, preferences, sessions |
| AI Integration | OpenAI Agents/ChatKit SDK | Conversational interface |
| Authentication | Optional signup/signin | Personalization features |

### Deployment Strategy

Deployment MUST follow the manual GitHub workflow:
- **Frontend Hosting**: GitHub Pages for static Docusaurus site
- **Deployment Method**: Manual upload via GitHub (no automated CI/CD tokens required)
- **Build Process**: Local `npm run build` generates static files for upload
- **Backend Hosting**: Separate deployment for FastAPI (Railway, Fly.io, or similar)
- **No Token-Based Automation**: Deployment does not require GitHub Actions secrets or
  automated deployment tokens
- **Repository Access**: Project accessible via GitHub Pages URL after upload

### Excluded Content

The following MUST NOT appear in the book:
- Vendor comparisons or product recommendations
- Ethics commentary or philosophical debates
- Proprietary code without explicit licensing
- Unverified claims or speculative content
- Hardcoded secrets or credentials

### Format Requirements

- All content in Markdown (`.md` or `.mdx`)
- Code examples with syntax highlighting
- Diagrams as placeholders with clear descriptions
- Tables for structured data
- Consistent terminology throughout

## Content Standards

### Chapter Structure

Every chapter MUST follow this structure:
1. **Learning Objectives**: 3-5 measurable outcomes
2. **Prerequisites**: Required knowledge and setup
3. **Core Content**: Concepts with examples
4. **Code Examples**: Tested, annotated code
5. **Lab Exercise**: Hands-on practice
6. **Summary**: Key takeaways
7. **Further Reading**: Optional deep-dive resources

### Quality Gates

Before any chapter is marked complete:
- [ ] Learning objectives are measurable
- [ ] Code examples execute without errors
- [ ] Lab instructions produce expected results
- [ ] Cross-references resolve correctly
- [ ] Terminology matches glossary definitions
- [ ] Safety warnings present for hardware content

### Consistency Validation

Automated checks MUST verify:
- Heading hierarchy consistency
- Code block language identifiers present
- Internal links resolve
- No placeholder text remains
- Front matter metadata complete

## Governance

### Amendment Process

1. Propose changes via spec with rationale
2. Review impact on existing chapters
3. Update affected content
4. Increment version per semantic versioning
5. Document in changelog

### Versioning Policy

- **MAJOR**: Structural reorganization or principle changes
- **MINOR**: New chapters, sections, or features added
- **PATCH**: Typo fixes, clarifications, minor updates

### Compliance Review

All PRs MUST verify:
- Constitution principles followed
- Quality gates passed
- No excluded content introduced
- Consistent formatting maintained

**Version**: 1.0.1 | **Ratified**: 2025-12-10 | **Last Amended**: 2025-12-12
