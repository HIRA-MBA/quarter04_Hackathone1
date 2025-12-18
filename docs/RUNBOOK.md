# Production Runbook

Physical AI & Robotics Textbook - Operations Guide

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Service Dependencies](#service-dependencies)
3. [Deployment Procedures](#deployment-procedures)
4. [Monitoring & Alerting](#monitoring--alerting)
5. [Incident Response](#incident-response)
6. [Common Issues & Fixes](#common-issues--fixes)
7. [Rollback Procedures](#rollback-procedures)
8. [Maintenance Tasks](#maintenance-tasks)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Traffic                                 │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Vercel / GitHub Pages                             │
│                    (Frontend - Docusaurus)                           │
│                    https://physical-ai-textbook.vercel.app           │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ /api/* proxy
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Railway                                      │
│                    (Backend - FastAPI)                               │
│              https://api.physical-ai-textbook.railway.app            │
└───────────┬─────────────────┴─────────────────┬─────────────────────┘
            │                                   │
            ▼                                   ▼
┌───────────────────────┐           ┌───────────────────────┐
│     Neon Postgres     │           │    Qdrant Cloud       │
│    (User Data, Auth)  │           │   (Vector Embeddings) │
└───────────────────────┘           └───────────────────────┘
            │
            └───────────── OpenAI API ─────────────┘
                        (GPT-4, Embeddings)
```

### Service URLs

| Service | Environment | URL |
|---------|-------------|-----|
| Frontend | Production | https://physical-ai-textbook.vercel.app |
| Frontend | GitHub Pages | https://[org].github.io/physical-ai-textbook |
| Backend | Production | https://api.physical-ai-textbook.railway.app |
| Database | Production | Neon Dashboard |
| Vectors | Production | Qdrant Cloud Dashboard |

---

## Service Dependencies

### External Services

| Service | Purpose | Status Page | SLA |
|---------|---------|-------------|-----|
| Vercel | Frontend hosting | https://www.vercel-status.com | 99.99% |
| Railway | Backend hosting | https://status.railway.app | 99.9% |
| Neon | PostgreSQL database | https://neon.tech/status | 99.95% |
| Qdrant Cloud | Vector database | https://status.qdrant.io | 99.9% |
| OpenAI | LLM & Embeddings | https://status.openai.com | 99.9% |

### Environment Variables

#### Backend (Railway)

```bash
# Required
DATABASE_URL=postgresql://user:pass@host:5432/db?sslmode=require
QDRANT_URL=https://xxx.qdrant.io:6333
QDRANT_API_KEY=xxx
OPENAI_API_KEY=sk-xxx
JWT_SECRET=xxx

# Optional
CORS_ORIGINS=https://physical-ai-textbook.vercel.app
LOG_LEVEL=INFO
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
```

#### Frontend (Vercel)

```bash
REACT_APP_API_URL=https://api.physical-ai-textbook.railway.app
REACT_APP_WS_URL=wss://api.physical-ai-textbook.railway.app
```

---

## Deployment Procedures

### Automatic Deployment (CI/CD)

Deployments are triggered automatically on push to `main`:

1. CI workflow runs (lint, typecheck, tests)
2. If CI passes, Deploy workflow triggers
3. Frontend deploys to Vercel/GitHub Pages
4. Backend deploys to Railway
5. Database migrations run
6. Health checks verify deployment

### Manual Deployment

#### Frontend

```bash
# Build locally
npm ci
npm run build

# Deploy to Vercel
vercel --prod

# Or deploy to GitHub Pages
npm run deploy
```

#### Backend

```bash
# Deploy to Railway
cd backend
railway up --service backend

# Or using CLI
railway login
railway link
railway up
```

### Database Migrations

```bash
# Run migrations
cd backend
export DATABASE_URL="postgresql://..."
alembic upgrade head

# Rollback one migration
alembic downgrade -1

# Show current revision
alembic current

# Show migration history
alembic history
```

### Re-ingest Book Content (RAG)

```bash
# SSH into Railway or run locally with DATABASE_URL set
cd backend
python -m app.cli.ingest --source ../docs --force
```

---

## Monitoring & Alerting

### Health Check Endpoints

| Endpoint | Purpose | Expected Response |
|----------|---------|-------------------|
| `GET /health` | Basic health | `{"status": "healthy"}` |
| `GET /health/ready` | Readiness probe | `{"status": "ready", "db": "ok", "qdrant": "ok"}` |
| `GET /health/live` | Liveness probe | `{"status": "alive"}` |

### Key Metrics to Monitor

1. **Response Time** - p50, p95, p99 latencies
2. **Error Rate** - 5xx errors per minute
3. **Request Rate** - Requests per second
4. **Database Connections** - Active connections
5. **Memory Usage** - Container memory
6. **CPU Usage** - Container CPU

### Log Locations

| Service | Log Access |
|---------|------------|
| Vercel | Vercel Dashboard → Project → Logs |
| Railway | Railway Dashboard → Service → Logs |
| Neon | Neon Console → Project → Logs |

### Alerting Rules

| Condition | Severity | Action |
|-----------|----------|--------|
| Error rate > 5% for 5 min | Critical | Page on-call |
| p95 latency > 2s for 10 min | Warning | Notify team |
| Health check fails 3x | Critical | Auto-restart, page on-call |
| Database connections > 80% | Warning | Notify team |

---

## Incident Response

### Severity Levels

| Level | Description | Response Time | Example |
|-------|-------------|---------------|---------|
| P1 | Service down | 15 min | Complete outage |
| P2 | Major degradation | 1 hour | Chat not working |
| P3 | Minor issue | 4 hours | Slow page loads |
| P4 | Low impact | 24 hours | UI bug |

### Incident Checklist

#### 1. Assess

```bash
# Check service health
curl https://api.physical-ai-textbook.railway.app/health

# Check frontend
curl -I https://physical-ai-textbook.vercel.app

# Check recent deployments
gh run list --workflow=deploy.yml --limit=5
```

#### 2. Communicate

- Update status page (if applicable)
- Notify stakeholders in #incidents channel
- Create incident ticket

#### 3. Mitigate

- Roll back if deployment-related
- Scale up if load-related
- Enable maintenance mode if needed

#### 4. Resolve

- Implement fix
- Verify in staging
- Deploy to production
- Confirm resolution

#### 5. Post-mortem

- Document timeline
- Identify root cause
- List action items
- Share learnings

---

## Common Issues & Fixes

### Issue: "502 Bad Gateway"

**Symptoms:** Backend returns 502 errors

**Causes:**
1. Backend crashed
2. Database connection timeout
3. Memory exceeded

**Resolution:**
```bash
# Check Railway logs
railway logs --service backend

# Restart service
railway restart --service backend

# Check database
psql $DATABASE_URL -c "SELECT 1"
```

### Issue: "Chat not responding"

**Symptoms:** RAG chatbot returns errors or hangs

**Causes:**
1. OpenAI API issues
2. Qdrant connection failed
3. Rate limiting

**Resolution:**
```bash
# Check OpenAI status
curl https://status.openai.com/api/v2/status.json

# Test Qdrant
curl -H "api-key: $QDRANT_API_KEY" $QDRANT_URL/collections

# Check rate limits in logs
railway logs --service backend | grep "rate"
```

### Issue: "Database connection errors"

**Symptoms:** `connection refused` or `too many connections`

**Causes:**
1. Database at connection limit
2. Network issues
3. Credentials expired

**Resolution:**
```bash
# Check active connections
psql $DATABASE_URL -c "SELECT count(*) FROM pg_stat_activity"

# Kill idle connections (be careful!)
psql $DATABASE_URL -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle' AND query_start < now() - interval '5 minutes'"
```

### Issue: "Build failing"

**Symptoms:** CI/CD pipeline fails

**Resolution:**
```bash
# Check workflow runs
gh run list --workflow=ci.yml --limit=5
gh run view <run-id> --log

# Run locally
npm ci && npm run build
cd backend && pip install -r requirements.txt && pytest
```

### Issue: "Slow page loads"

**Symptoms:** Frontend takes > 3s to load

**Causes:**
1. Large bundle size
2. Unoptimized images
3. Too many API calls

**Resolution:**
```bash
# Analyze bundle
npm run build -- --analyze

# Check Lighthouse
npx lighthouse https://physical-ai-textbook.vercel.app --output=html
```

---

## Rollback Procedures

### Frontend Rollback (Vercel)

```bash
# List recent deployments
vercel ls

# Rollback to previous deployment
vercel rollback <deployment-url>

# Or via dashboard: Vercel → Project → Deployments → ⋮ → Promote
```

### Frontend Rollback (GitHub Pages)

```bash
# Find last good commit
git log --oneline -10

# Revert and push
git revert HEAD
git push origin main

# Or force push to previous commit (destructive)
git reset --hard <commit>
git push --force origin main
```

### Backend Rollback (Railway)

```bash
# List deployments
railway deployments

# Rollback
railway rollback

# Or redeploy specific commit
git checkout <commit>
railway up
```

### Database Rollback

```bash
# CAUTION: Test in staging first!

# Rollback one migration
alembic downgrade -1

# Rollback to specific revision
alembic downgrade <revision>

# Restore from backup (Neon)
# Use Neon dashboard → Backups → Restore
```

---

## Maintenance Tasks

### Weekly

- [ ] Review error logs
- [ ] Check disk usage
- [ ] Review security alerts
- [ ] Update dependencies (if needed)

### Monthly

- [ ] Run database vacuum
- [ ] Review and rotate API keys
- [ ] Check backup integrity
- [ ] Review access permissions

### Quarterly

- [ ] Load testing
- [ ] Security audit
- [ ] Dependency updates (major)
- [ ] Disaster recovery drill

### Database Maintenance

```bash
# Vacuum analyze (Neon handles automatically, but can force)
psql $DATABASE_URL -c "VACUUM ANALYZE"

# Check table sizes
psql $DATABASE_URL -c "
SELECT
  schemaname || '.' || relname AS table,
  pg_size_pretty(pg_total_relation_size(relid)) AS size
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC
LIMIT 10;
"

# Check index usage
psql $DATABASE_URL -c "
SELECT
  indexrelname AS index,
  idx_scan AS scans,
  pg_size_pretty(pg_relation_size(indexrelid)) AS size
FROM pg_stat_user_indexes
ORDER BY idx_scan ASC
LIMIT 10;
"
```

### RAG Index Maintenance

```bash
# Check collection stats
curl -H "api-key: $QDRANT_API_KEY" \
  "$QDRANT_URL/collections/textbook_content"

# Re-index if needed
python -m app.cli.ingest --source ../docs --force
```

---

## Emergency Contacts

| Role | Contact | Escalation |
|------|---------|------------|
| Primary On-Call | Check schedule | Slack #on-call |
| Secondary On-Call | Check schedule | Slack #on-call |
| Engineering Lead | - | Slack DM |
| Infrastructure | - | Slack #infra |

## Useful Links

- [GitHub Repository](https://github.com/[org]/physical-ai-textbook)
- [Vercel Dashboard](https://vercel.com/[org]/physical-ai-textbook)
- [Railway Dashboard](https://railway.app/project/[id])
- [Neon Console](https://console.neon.tech/app/projects/[id])
- [Qdrant Cloud](https://cloud.qdrant.io/)
- [OpenAI Usage](https://platform.openai.com/usage)

---

*Last updated: 2025-12-18*
