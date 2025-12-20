# Data Model: Physical AI & Robotics Textbook

**Branch**: `001-physical-ai-textbook` | **Date**: 2025-12-10

## Overview

This document defines the data entities, relationships, and schemas for the Physical AI Textbook system.

---

## Entity Relationship Diagram

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│      User       │────<│    Session      │     │   Preference    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                                               │
        │                                               │
        └──────────────────────┬────────────────────────┘
                               │
                               ▼
                    ┌─────────────────┐
                    │  ChatHistory    │
                    └─────────────────┘
                               │
                               │ references
                               ▼
                    ┌─────────────────┐
                    │    Chapter      │
                    └─────────────────┘
                               │
                               │ embedded in
                               ▼
                    ┌─────────────────┐
                    │  ChunkVector    │
                    │   (Qdrant)      │
                    └─────────────────┘

┌─────────────────┐     ┌─────────────────┐
│  Translation    │────<│ TranslationJob  │
│    Cache        │     │    (Queue)      │
└─────────────────┘     └─────────────────┘
```

---

## Postgres Entities (Neon)

### 1. User

Stores authenticated user accounts.

```sql
-- NOTE: OAuth-only authentication per clarification 2025-12-20
-- No password_hash field - users authenticate via Google/GitHub only
CREATE TABLE users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email           VARCHAR(255) UNIQUE NOT NULL,
    email_verified  BOOLEAN DEFAULT TRUE,  -- OAuth emails are pre-verified
    name            VARCHAR(255),
    avatar_url      TEXT,

    -- OAuth fields (required - no password auth)
    oauth_provider  VARCHAR(50) NOT NULL,  -- 'google' or 'github'
    oauth_id        VARCHAR(255) NOT NULL,

    -- Metadata
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login_at   TIMESTAMP WITH TIME ZONE,

    -- Constraints
    CONSTRAINT valid_email CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'),
    CONSTRAINT valid_oauth_provider CHECK (oauth_provider IN ('google', 'github')),
    UNIQUE(oauth_provider, oauth_id)
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_oauth ON users(oauth_provider, oauth_id);
```

**Validation Rules**:
- Email must be valid format
- OAuth provider required (google or github)
- OAuth ID unique per provider

---

### 2. Session

Manages user authentication sessions.

```sql
CREATE TABLE sessions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token           VARCHAR(255) UNIQUE NOT NULL,
    refresh_token   VARCHAR(255) UNIQUE,

    -- Metadata
    user_agent      TEXT,
    ip_address      INET,

    -- Lifecycle
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at      TIMESTAMP WITH TIME ZONE NOT NULL,
    last_active_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_sessions_user ON sessions(user_id);
CREATE INDEX idx_sessions_token ON sessions(token);
CREATE INDEX idx_sessions_expires ON sessions(expires_at);
```

**State Transitions**:
- `active` → `expired` (automatic on expires_at)
- `active` → `revoked` (manual logout)

---

### 3. UserPreference

Stores user personalization settings and questionnaire responses.

```sql
CREATE TABLE user_preferences (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             UUID UNIQUE NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    -- Background Questionnaire
    experience_level    VARCHAR(20) CHECK (experience_level IN ('beginner', 'intermediate', 'advanced')),
    programming_langs   TEXT[],      -- ['python', 'cpp', 'rust']
    robotics_experience BOOLEAN DEFAULT FALSE,
    hardware_available  TEXT[],      -- ['ros2', 'gazebo', 'isaac', 'jetson']
    learning_goals      TEXT[],      -- ['career', 'research', 'hobby']

    -- Display Preferences (English-only for MVP per clarification 2025-12-20)
    preferred_language  VARCHAR(10) DEFAULT 'en' CHECK (preferred_language = 'en'),  -- Urdu deferred
    theme               VARCHAR(10) DEFAULT 'light' CHECK (theme IN ('light', 'dark', 'system')),
    font_size           VARCHAR(10) DEFAULT 'medium' CHECK (font_size IN ('small', 'medium', 'large')),

    -- Progress Tracking
    current_chapter     INTEGER DEFAULT 1,
    completed_chapters  INTEGER[] DEFAULT '{}',
    bookmarks           JSONB DEFAULT '[]',  -- [{chapter: 1, section: "intro", note: "..."}]

    -- Timestamps
    questionnaire_completed_at TIMESTAMP WITH TIME ZONE,
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_preferences_user ON user_preferences(user_id);
```

**Validation Rules**:
- experience_level must be one of: beginner, intermediate, advanced
- preferred_language: en (English) only for MVP
- completed_chapters must be integers 1-14

---

### 4. ChatHistory

Stores conversation history for RAG chatbot.

**RAG Knowledge Scope** (per clarification 2025-12-20):
- Chatbot queries are scoped to: current chapter + all completed chapters
- Progressive access based on student progress
- Implemented via filter on user_preferences.completed_chapters during vector search

```sql
CREATE TABLE chat_history (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID REFERENCES users(id) ON DELETE SET NULL,  -- NULL for anonymous
    session_id      UUID,  -- Group messages in conversation

    -- Message Content
    role            VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content         TEXT NOT NULL,

    -- Context
    chapter_number  INTEGER CHECK (chapter_number BETWEEN 1 AND 14),
    section_id      VARCHAR(100),  -- e.g., "ch01-ros2-basics"

    -- RAG Metadata
    retrieved_chunks JSONB,  -- [{chunk_id, score, text_preview}]
    model_used      VARCHAR(50),   -- 'gpt-4o-mini'
    tokens_used     INTEGER,

    -- Feedback
    feedback_rating INTEGER CHECK (feedback_rating BETWEEN 1 AND 5),
    feedback_text   TEXT,

    -- Timestamps
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_chat_user ON chat_history(user_id);
CREATE INDEX idx_chat_session ON chat_history(session_id);
CREATE INDEX idx_chat_chapter ON chat_history(chapter_number);
CREATE INDEX idx_chat_created ON chat_history(created_at DESC);
```

**Validation Rules**:
- role must be: user, assistant, or system
- chapter_number between 1-14 if provided
- feedback_rating 1-5 scale

---

### 5. TranslationCache

Caches translated content to avoid repeated API calls.

```sql
CREATE TABLE translation_cache (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Source Content
    source_language VARCHAR(10) NOT NULL DEFAULT 'en',
    target_language VARCHAR(10) NOT NULL,
    content_hash    VARCHAR(64) NOT NULL,  -- SHA-256 of source content

    -- Content
    source_text     TEXT NOT NULL,
    translated_text TEXT NOT NULL,

    -- Metadata
    content_type    VARCHAR(20) NOT NULL CHECK (content_type IN ('chapter', 'section', 'ui', 'code_comment')),
    chapter_number  INTEGER,
    section_id      VARCHAR(100),

    -- Quality
    quality_score   FLOAT,  -- 0.0 - 1.0, from review
    reviewed_at     TIMESTAMP WITH TIME ZONE,
    reviewer_id     UUID REFERENCES users(id),

    -- Timestamps
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(content_hash, target_language)
);

CREATE INDEX idx_translation_hash ON translation_cache(content_hash, target_language);
CREATE INDEX idx_translation_chapter ON translation_cache(chapter_number, target_language);
```

---

### 6. TranslationJob (Queue)

Manages background translation jobs.

```sql
CREATE TABLE translation_jobs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Job Details
    source_text     TEXT NOT NULL,
    target_language VARCHAR(10) NOT NULL,
    content_type    VARCHAR(20) NOT NULL,
    chapter_number  INTEGER,

    -- Status
    status          VARCHAR(20) NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    priority        INTEGER DEFAULT 0,  -- Higher = more urgent

    -- Result
    result_cache_id UUID REFERENCES translation_cache(id),
    error_message   TEXT,
    retry_count     INTEGER DEFAULT 0,

    -- Timestamps
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at      TIMESTAMP WITH TIME ZONE,
    completed_at    TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_jobs_status ON translation_jobs(status, priority DESC);
CREATE INDEX idx_jobs_chapter ON translation_jobs(chapter_number);
```

**State Transitions**:
```
pending → processing → completed
                    ↘ failed (retry_count < 3) → pending
                    ↘ failed (retry_count >= 3) → [terminal]
```

---

## Qdrant Entities (Vector DB)

### ChunkVector Collection

Stores embedded book content chunks for semantic search.

```json
{
  "collection_name": "book_chunks",
  "vectors": {
    "size": 1536,
    "distance": "Cosine"
  },
  "payload_schema": {
    "chunk_id": "keyword",
    "chapter_number": "integer",
    "module_number": "integer",
    "section_title": "keyword",
    "content_type": "keyword",
    "text": "text",
    "word_count": "integer",
    "heading_path": "keyword[]",
    "code_language": "keyword",
    "has_code": "bool",
    "created_at": "datetime"
  }
}
```

**Example Payload**:
```json
{
  "chunk_id": "ch03-ros2-arch-services-01",
  "chapter_number": 3,
  "module_number": 1,
  "section_title": "ROS 2 Services",
  "content_type": "prose",
  "text": "Services in ROS 2 provide request-response communication between nodes...",
  "word_count": 245,
  "heading_path": ["Module 1", "Chapter 3", "ROS 2 Services"],
  "code_language": null,
  "has_code": false,
  "created_at": "2025-12-10T00:00:00Z"
}
```

**Indexing Strategy**:
- Primary: Vector similarity (cosine distance)
- Filters: chapter_number, module_number, content_type
- Full-text: section_title for hybrid search

---

## Content Entities (Markdown/JSON)

### Chapter Structure

Each chapter follows this JSON metadata structure:

```json
{
  "chapter_number": 3,
  "module_number": 1,
  "title": "ROS 2 Architecture & Core Concepts",
  "slug": "ch03-ros2-architecture",
  "word_count_target": {
    "min": 1700,
    "max": 2500
  },
  "objectives": [
    "Master ROS 2 node lifecycle",
    "Understand DDS communication",
    "Implement services and actions"
  ],
  "prerequisites": ["ch01", "ch02"],
  "code_examples": [
    {
      "id": "ce-03-01",
      "title": "Node Lifecycle Management",
      "language": "python",
      "path": "labs/module-1/ch03-ros2-deep-dive/lifecycle_node.py"
    }
  ],
  "lab_exercise": {
    "id": "lab-03",
    "title": "ROS 2 Deep Dive",
    "estimated_duration": "90 minutes",
    "difficulty": "intermediate"
  },
  "rag_integration_points": [
    {
      "id": "rag-03-01",
      "trigger": "after_services_section",
      "suggested_queries": ["What is a ROS 2 service?", "When should I use services vs topics?"]
    }
  ],
  "personalization_triggers": [
    {
      "type": "complexity_adjustment",
      "context": "advanced_users_skip_basics"
    }
  ]
}
```

---

## Database Migrations

### Migration Order

1. `001_create_users.sql`
2. `002_create_sessions.sql`
3. `003_create_user_preferences.sql`
4. `004_create_chat_history.sql`
5. `005_create_translation_cache.sql`
6. `006_create_translation_jobs.sql`
7. `007_add_indexes.sql`

### Rollback Strategy

Each migration includes corresponding `down` migration for rollback.

---

## Data Retention Policy

| Entity | Retention | Rationale |
|--------|-----------|-----------|
| User | Indefinite | Account persistence |
| Session | 30 days after expiry | Security compliance |
| ChatHistory | 1 year | Learning analytics |
| TranslationCache | Indefinite | Cumulative asset |
| TranslationJob | 90 days after completion | Audit trail |

---

## Privacy Considerations

- User email encrypted at rest
- Password hashes use bcrypt (cost 12)
- Chat history anonymizable on account deletion
- GDPR export includes: users, preferences, chat_history
- No PII in Qdrant vectors
