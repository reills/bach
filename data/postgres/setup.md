# Postgres Setup

## 1. Create the database

```bash
createdb bach_gen
```

Or from within `psql`:

```sql
CREATE DATABASE bach_gen;
```

## 2. Apply the schema

```bash
psql bach_gen < data/postgres/schema.sql
```

## 3. Set the connection string

Export `DATABASE_URL` before starting the backend. The standard `psycopg2` DSN format:

```bash
export DATABASE_URL="postgresql://localhost/bach_gen"
```

With a specific user and password:

```bash
export DATABASE_URL="postgresql://myuser:mypassword@localhost/bach_gen"
```

## 4. Start the backend

```bash
CONDA_NO_PLUGINS=true conda run -n bach \
  DATABASE_URL="postgresql://localhost/bach_gen" \
  python -m uvicorn src.api.app:app --reload --port 8001
```

If `DATABASE_URL` is not set the backend falls back to the in-memory store automatically (useful for tests and quick runs).

## Schema overview

| Table    | Purpose |
|----------|---------|
| `scores` | Committed score revisions with name, creation, and last-save timestamps |
| `drafts` | Pending inpaint/fingering edits; automatically purged after 24 hours |

## Draft cleanup

Stale drafts (older than 24 hours) are deleted automatically each time `create_draft` is called. No cron job or manual cleanup is needed during normal use.

To change the TTL, pass `draft_ttl_hours` when constructing `PostgresScoreRepository` in `src/api/app.py`.

## Useful queries

List all scores:
```sql
SELECT score_id, name, revision, created_at, updated_at FROM scores ORDER BY updated_at DESC;
```

List open drafts:
```sql
SELECT draft_id, score_id, base_revision, created_at FROM drafts ORDER BY created_at DESC;
```

Manually purge drafts older than 24 hours:
```sql
DELETE FROM drafts WHERE created_at < NOW() - INTERVAL '24 hours';
```
