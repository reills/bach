-- bach-gen persistent store schema
-- Run once against the bach_gen database after creating it.

CREATE TABLE IF NOT EXISTS scores (
    score_id    TEXT        PRIMARY KEY,
    name        TEXT        NOT NULL DEFAULT 'Untitled',
    revision    INTEGER     NOT NULL DEFAULT 1,
    score_json  JSONB       NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS drafts (
    draft_id       TEXT        PRIMARY KEY,
    score_id       TEXT        NOT NULL REFERENCES scores(score_id) ON DELETE CASCADE,
    base_revision  INTEGER     NOT NULL,
    score_json     JSONB       NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS drafts_score_id_idx ON drafts(score_id);
CREATE INDEX IF NOT EXISTS drafts_created_at_idx ON drafts(created_at);
