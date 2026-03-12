from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import count
from typing import Generic, Protocol, TypeVar

ScoreT = TypeVar("ScoreT")


@dataclass(frozen=True)
class StoredScore(Generic[ScoreT]):
    score_id: str
    revision: int
    score: ScoreT


@dataclass(frozen=True)
class StoredDraft(Generic[ScoreT]):
    draft_id: str
    score_id: str
    base_revision: int
    score: ScoreT


class ScoreNotFoundError(KeyError):
    pass


class DraftNotFoundError(KeyError):
    pass


class StaleRevisionError(RuntimeError):
    pass


class ScoreDraftRepository(Protocol[ScoreT]):
    def create_score(self, score: ScoreT) -> StoredScore[ScoreT]:
        ...

    def get_score(self, score_id: str) -> StoredScore[ScoreT]:
        ...

    def create_draft(
        self,
        score_id: str,
        *,
        base_revision: int | None = None,
    ) -> StoredDraft[ScoreT]:
        ...

    def get_draft(self, draft_id: str) -> StoredDraft[ScoreT]:
        ...

    def save_draft(self, draft_id: str, score: ScoreT) -> StoredDraft[ScoreT]:
        ...

    def commit_draft(self, draft_id: str) -> StoredScore[ScoreT]:
        ...

    def discard_draft(self, draft_id: str) -> None:
        ...


@dataclass
class _ScoreState(Generic[ScoreT]):
    revision: int
    score: ScoreT


@dataclass
class _DraftState(Generic[ScoreT]):
    score_id: str
    base_revision: int
    score: ScoreT


class InMemoryScoreRepository(Generic[ScoreT]):
    def __init__(self) -> None:
        self._scores: dict[str, _ScoreState[ScoreT]] = {}
        self._drafts: dict[str, _DraftState[ScoreT]] = {}
        self._score_ids = count(1)
        self._draft_ids = count(1)

    def create_score(self, score: ScoreT) -> StoredScore[ScoreT]:
        score_id = self._next_id("score", self._score_ids)
        self._scores[score_id] = _ScoreState(
            revision=1,
            score=deepcopy(score),
        )
        return self.get_score(score_id)

    def get_score(self, score_id: str) -> StoredScore[ScoreT]:
        state = self._get_score_state(score_id)
        return StoredScore(
            score_id=score_id,
            revision=state.revision,
            score=deepcopy(state.score),
        )

    def create_draft(
        self,
        score_id: str,
        *,
        base_revision: int | None = None,
    ) -> StoredDraft[ScoreT]:
        score_state = self._get_score_state(score_id)
        expected_revision = score_state.revision if base_revision is None else base_revision
        if expected_revision != score_state.revision:
            raise StaleRevisionError(
                f"score {score_id!r} is at revision {score_state.revision}, "
                f"not {expected_revision}"
            )

        draft_id = self._next_id("draft", self._draft_ids)
        self._drafts[draft_id] = _DraftState(
            score_id=score_id,
            base_revision=score_state.revision,
            score=deepcopy(score_state.score),
        )
        return self.get_draft(draft_id)

    def get_draft(self, draft_id: str) -> StoredDraft[ScoreT]:
        state = self._get_draft_state(draft_id)
        return StoredDraft(
            draft_id=draft_id,
            score_id=state.score_id,
            base_revision=state.base_revision,
            score=deepcopy(state.score),
        )

    def save_draft(self, draft_id: str, score: ScoreT) -> StoredDraft[ScoreT]:
        state = self._get_draft_state(draft_id)
        state.score = deepcopy(score)
        return self.get_draft(draft_id)

    def commit_draft(self, draft_id: str) -> StoredScore[ScoreT]:
        draft_state = self._get_draft_state(draft_id)
        score_state = self._get_score_state(draft_state.score_id)
        if score_state.revision != draft_state.base_revision:
            raise StaleRevisionError(
                f"draft {draft_id!r} is based on revision {draft_state.base_revision}, "
                f"but score {draft_state.score_id!r} is now at revision {score_state.revision}"
            )

        score_state.revision += 1
        score_state.score = deepcopy(draft_state.score)
        del self._drafts[draft_id]
        return self.get_score(draft_state.score_id)

    def discard_draft(self, draft_id: str) -> None:
        self._get_draft_state(draft_id)
        del self._drafts[draft_id]

    def _get_score_state(self, score_id: str) -> _ScoreState[ScoreT]:
        try:
            return self._scores[score_id]
        except KeyError as exc:
            raise ScoreNotFoundError(f"unknown score_id: {score_id}") from exc

    def _get_draft_state(self, draft_id: str) -> _DraftState[ScoreT]:
        try:
            return self._drafts[draft_id]
        except KeyError as exc:
            raise DraftNotFoundError(f"unknown draft_id: {draft_id}") from exc

    @staticmethod
    def _next_id(prefix: str, sequence: count) -> str:
        return f"{prefix}-{next(sequence)}"
