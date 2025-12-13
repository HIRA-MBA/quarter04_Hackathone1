# Quickstart: Physical AI & Robotics Textbook

**Branch**: `001-physical-ai-textbook` | **Date**: 2025-12-10

## Prerequisites

- Node.js 18+ (for Docusaurus)
- Python 3.11+ (for FastAPI backend)
- Docker (for local services)
- Git

## 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/your-org/physical-ai-textbook.git
cd physical-ai-textbook

# Checkout feature branch
git checkout 001-physical-ai-textbook
```

## 2. Environment Variables

Create `.env` files:

```bash
# Root .env (Docusaurus)
cp .env.example .env

# Backend .env
cp backend/.env.example backend/.env
```

Required variables:

```bash
# .env (Docusaurus)
VITE_API_URL=http://localhost:8000/v1
VITE_WS_URL=ws://localhost:8000/ws

# backend/.env
DATABASE_URL=postgresql://user:pass@localhost:5432/textbook
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
OPENAI_API_KEY=sk-...
BETTER_AUTH_SECRET=your-secret-key
BETTER_AUTH_URL=http://localhost:8000
```

## 3. Start Local Services

```bash
# Start Postgres and Qdrant with Docker
docker compose up -d postgres qdrant

# Verify services
docker compose ps
```

## 4. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Run database migrations
alembic upgrade head

# Start FastAPI server
uvicorn app.main:app --reload --port 8000
```

Verify: http://localhost:8000/docs (OpenAPI UI)

## 5. Frontend Setup

```bash
# From project root
npm install

# Start Docusaurus dev server
npm run start
```

Verify: http://localhost:3000

## 6. Seed Book Content

```bash
# Ingest markdown files into Qdrant
cd backend
python -m app.cli.ingest --source ../docs --collection book_chunks

# Verify embedding count
python -m app.cli.ingest --status
```

## 7. Verify Full Stack

1. Open http://localhost:3000
2. Navigate to Chapter 1
3. Click the chat widget (bottom-right)
4. Ask: "What is ROS 2?"
5. Verify response includes source citations

## Development Commands

```bash
# Frontend
npm run start        # Dev server with HMR
npm run build        # Production build
npm run serve        # Serve production build
npm run typecheck    # TypeScript check
npm run lint         # ESLint

# Backend
uvicorn app.main:app --reload  # Dev server
pytest                          # Run tests
pytest --cov                    # Tests with coverage
ruff check .                    # Lint Python
ruff format .                   # Format Python

# Docker
docker compose up -d            # Start all services
docker compose down             # Stop all services
docker compose logs -f backend  # Tail backend logs
```

## Project Structure Overview

```
physical-ai-textbook/
├── docs/                    # Book content (Markdown)
│   ├── front-matter/
│   ├── module-1-ros2/
│   ├── module-2-digital-twin/
│   ├── module-3-isaac/
│   ├── module-4-vla/
│   └── back-matter/
├── src/                     # Docusaurus customizations
│   ├── components/          # React components
│   ├── pages/               # Custom pages
│   └── theme/               # Theme overrides
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/             # API routes
│   │   ├── models/          # SQLAlchemy models
│   │   ├── services/        # Business logic
│   │   └── db/              # Database connections
│   └── tests/
├── labs/                    # Lab exercise code
├── specs/                   # Specifications (this feature)
└── docker-compose.yml
```

## Common Issues

### Port Already in Use

```bash
# Find process on port 8000
lsof -i :8000
# Kill it
kill -9 <PID>
```

### Database Connection Failed

```bash
# Check Postgres is running
docker compose ps postgres

# Reset database
docker compose down -v
docker compose up -d postgres
cd backend && alembic upgrade head
```

### Qdrant Connection Failed

```bash
# Check Qdrant is running
curl http://localhost:6333/health

# Restart Qdrant
docker compose restart qdrant
```

### OpenAI API Errors

- Verify `OPENAI_API_KEY` is set
- Check API key has credits
- Verify model access (gpt-4o-mini)

## Next Steps

1. Run `/sp.tasks` to generate implementation tasks
2. Start with M1 (Project Bootstrap)
3. Follow dependency order in plan.md

## Resources

- [Docusaurus Documentation](https://docusaurus.io/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Qdrant Quickstart](https://qdrant.tech/documentation/quick-start/)
- [Better-Auth Docs](https://www.better-auth.com/)
