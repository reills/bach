"""PostgreSQL-backed score/draft repository with draft TTL cleanup."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import psycopg2
import psycopg2.extras

from src.api.canonical.types import CanonicalScore
from src.api.store import (
    DraftNotFoundError,
    ScoreNotFoundError,
    StaleRevisionError,
    StoredDraft,
    StoredScore,
)
from src.api.store_serde import score_from_json, score_to_json

_DEFAULT_DRAFT_TTL_HOURS = 24


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class PostgresScoreRepository:
    """Persistent score/draft store backed by PostgreSQL.

    Drafts older than ``draft_ttl_hours`` are purged automatically on
    ``create_draft`` so stale sessions don't accumulate indefinitely.
    """

    def __init__(self, dsn: str, *, draft_ttl_hours: int = _DEFAULT_DRAFT_TTL_HOURS) -> None:
        self._dsn = dsn
        self._draft_ttl = timedelta(hours=draft_ttl_hours)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _conn(self):
        return psycopg2.connect(self._dsn, cursor_factory=psycopg2.extras.RealDictCursor)

    def _purge_stale_drafts(self, cur) -> None:
        cutoff = datetime.now(timezone.utc) - self._draft_ttl
        cur.execute("DELETE FROM drafts WHERE created_at < %s", (cutoff,))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_score(
        self, score: CanonicalScore, *, name: str = "Untitled"
    ) -> StoredScore[CanonicalScore]:
        score_id = _new_id("score")
        now = datetime.now(timezone.utc)
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO scores (score_id, name, revision, score_json, created_at, updated_at)
                    VALUES (%s, %s, 1, %s, %s, %s)
                    """,
                    (score_id, name, score_to_json(score), now, now),
                )
        return StoredScore(
            score_id=score_id,
            revision=1,
            score=score,
            name=name,
            created_at=now,
            updated_at=now,
        )

    def get_score(self, score_id: str) -> StoredScore[CanonicalScore]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT name, revision, score_json, created_at, updated_at "
                    "FROM scores WHERE score_id = %s",
                    (score_id,),
                )
                row = cur.fetchone()
        if row is None:
            raise ScoreNotFoundError(f"unknown score_id: {score_id}")
        return StoredScore(
            score_id=score_id,
            revision=row["revision"],
            score=score_from_json(row["score_json"]),
            name=row["name"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def create_draft(
        self,
        score_id: str,
        *,
        base_revision: int | None = None,
    ) -> StoredDraft[CanonicalScore]:
        draft_id = _new_id("draft")
        now = datetime.now(timezone.utc)

        with self._conn() as conn:
            with conn.cursor() as cur:
                # Purge stale drafts on each create to avoid accumulation.
                self._purge_stale_drafts(cur)

                cur.execute(
                    "SELECT revision, score_json FROM scores WHERE score_id = %s FOR UPDATE",
                    (score_id,),
                )
                row = cur.fetchone()
                if row is None:
                    raise ScoreNotFoundError(f"unknown score_id: {score_id}")

                current_revision = row["revision"]
                expected_revision = current_revision if base_revision is None else base_revision
                if expected_revision != current_revision:
                    raise StaleRevisionError(
                        f"score {score_id!r} is at revision {current_revision}, "
                        f"not {expected_revision}"
                    )

                score = score_from_json(row["score_json"])
                cur.execute(
                    """
                    INSERT INTO drafts (draft_id, score_id, base_revision, score_json, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (draft_id, score_id, current_revision, score_to_json(score), now),
                )

        return StoredDraft(
            draft_id=draft_id,
            score_id=score_id,
            base_revision=current_revision,
            score=score,
            created_at=now,
        )

    def get_draft(self, draft_id: str) -> StoredDraft[CanonicalScore]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT score_id, base_revision, score_json, created_at "
                    "FROM drafts WHERE draft_id = %s",
                    (draft_id,),
                )
                row = cur.fetchone()
        if row is None:
            raise DraftNotFoundError(f"unknown draft_id: {draft_id}")
        return StoredDraft(
            draft_id=draft_id,
            score_id=row["score_id"],
            base_revision=row["base_revision"],
            score=score_from_json(row["score_json"]),
            created_at=row["created_at"],
        )

    def save_draft(self, draft_id: str, score: CanonicalScore) -> StoredDraft[CanonicalScore]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE drafts SET score_json = %s WHERE draft_id = %s RETURNING score_id, base_revision, created_at",
                    (score_to_json(score), draft_id),
                )
                row = cur.fetchone()
        if row is None:
            raise DraftNotFoundError(f"unknown draft_id: {draft_id}")
        return StoredDraft(
            draft_id=draft_id,
            score_id=row["score_id"],
            base_revision=row["base_revision"],
            score=score,
            created_at=row["created_at"],
        )

    def commit_draft(self, draft_id: str) -> StoredScore[CanonicalScore]:
        now = datetime.now(timezone.utc)
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT score_id, base_revision, score_json FROM drafts WHERE draft_id = %s",
                    (draft_id,),
                )
                draft_row = cur.fetchone()
                if draft_row is None:
                    raise DraftNotFoundError(f"unknown draft_id: {draft_id}")

                cur.execute(
                    "SELECT revision, name, created_at FROM scores WHERE score_id = %s FOR UPDATE",
                    (draft_row["score_id"],),
                )
                score_row = cur.fetchone()
                if score_row is None:
                    raise ScoreNotFoundError(f"unknown score_id: {draft_row['score_id']}")

                if score_row["revision"] != draft_row["base_revision"]:
                    raise StaleRevisionError(
                        f"draft {draft_id!r} is based on revision {draft_row['base_revision']}, "
                        f"but score {draft_row['score_id']!r} is now at revision {score_row['revision']}"
                    )

                new_revision = score_row["revision"] + 1
                cur.execute(
                    "UPDATE scores SET revision = %s, score_json = %s, updated_at = %s "
                    "WHERE score_id = %s",
                    (new_revision, draft_row["score_json"], now, draft_row["score_id"]),
                )
                cur.execute("DELETE FROM drafts WHERE draft_id = %s", (draft_id,))

        return StoredScore(
            score_id=draft_row["score_id"],
            revision=new_revision,
            score=score_from_json(draft_row["score_json"]),
            name=score_row["name"],
            created_at=score_row["created_at"],
            updated_at=now,
        )

    def discard_draft(self, draft_id: str) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM drafts WHERE draft_id = %s RETURNING draft_id",
                    (draft_id,),
                )
                if cur.fetchone() is None:
                    raise DraftNotFoundError(f"unknown draft_id: {draft_id}")
