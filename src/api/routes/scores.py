from __future__ import annotations

import base64
from typing import Any, Callable, Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from src.api.canonical import CanonicalScore
from src.api.compose_service import ComposeServiceResult, export_score
from src.api.services import preview_window_inpaint
from src.api.store import (
    DraftNotFoundError,
    InMemoryScoreRepository,
    ScoreDraftRepository,
    ScoreNotFoundError,
    StaleRevisionError,
)

ComposeHandler = Callable[[BaseModel], ComposeServiceResult]


class ComposeRequest(BaseModel):
    prompt: str | None = None
    constraints: dict[str, Any] | None = None


class ComposeResponse(BaseModel):
    scoreId: str
    revision: int
    scoreXML: str
    measureMap: dict[str, str] | None = None
    eventHitMap: dict[str, str] | None = None
    midi: str | None = None


class InpaintConstraints(BaseModel):
    keepHarmony: bool | None = None
    keepRhythm: bool | None = None
    keepSoprano: bool | None = None
    fixedPitches: list[str] | None = None
    fixedOnsets: list[int] | None = None


class LockedRange(BaseModel):
    startTick: int
    endTick: int
    type: Literal["pitch", "onset", "all"]


class InpaintLocks(BaseModel):
    lockedEventIds: list[str] | None = None
    lockedRanges: list[LockedRange] | None = None


class InpaintPreviewRequest(BaseModel):
    scoreId: str
    measureId: str
    revision: int
    constraints: InpaintConstraints | None = None
    locks: InpaintLocks | None = None
    mode: Literal["window", "repair"] | None = None


class InpaintPreviewResponse(BaseModel):
    draftId: str
    scoreXML: str
    baseRevision: int
    highlightMeasureId: str | None = None
    measureMap: dict[str, str] | None = None
    eventHitMap: dict[str, str] | None = None
    lockedEventIds: list[str] | None = None
    changedMeasureIds: list[str] | None = None


class CommitDraftRequest(BaseModel):
    scoreId: str
    draftId: str


class CommitDraftResponse(BaseModel):
    scoreXML: str
    revision: int
    measureMap: dict[str, str] | None = None
    eventHitMap: dict[str, str] | None = None


class DiscardDraftRequest(BaseModel):
    scoreId: str
    draftId: str


class DiscardDraftResponse(BaseModel):
    ok: bool


def create_router(
    *,
    compose_service: ComposeHandler | None = None,
    repository: ScoreDraftRepository[CanonicalScore] | None = None,
) -> APIRouter:
    router = APIRouter()
    score_repository = repository or InMemoryScoreRepository[CanonicalScore]()

    @router.post("/compose", response_model=ComposeResponse)
    async def compose(request: ComposeRequest) -> ComposeResponse:
        if compose_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="compose service is not configured",
            )

        result = compose_service(request)
        stored_score = score_repository.create_score(result.score)
        return ComposeResponse(
            scoreId=stored_score.score_id,
            revision=stored_score.revision,
            scoreXML=result.score_xml,
            measureMap=result.measure_map,
            eventHitMap=result.event_hit_map,
            midi=base64.b64encode(result.midi).decode("ascii"),
        )

    @router.post("/inpaint_preview", response_model=InpaintPreviewResponse)
    async def inpaint_preview(request: InpaintPreviewRequest) -> InpaintPreviewResponse:
        if request.mode not in (None, "window"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"unsupported inpaint mode: {request.mode}",
            )

        try:
            result = preview_window_inpaint(
                score_repository,
                request.scoreId,
                revision=request.revision,
                measure_id=request.measureId,
                locked_event_ids=request.locks.lockedEventIds if request.locks else (),
            )
        except ScoreNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except StaleRevisionError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return InpaintPreviewResponse(
            draftId=result.draft_id,
            scoreXML=result.score_xml,
            baseRevision=result.base_revision,
            highlightMeasureId=result.highlight_measure_id,
            measureMap=result.measure_map,
            eventHitMap=result.event_hit_map,
            lockedEventIds=result.locked_event_ids,
            changedMeasureIds=result.changed_measure_ids,
        )

    @router.post("/commit_draft", response_model=CommitDraftResponse)
    async def commit_draft(request: CommitDraftRequest) -> CommitDraftResponse:
        _validate_draft_belongs_to_score(score_repository, request.scoreId, request.draftId)

        try:
            committed = score_repository.commit_draft(request.draftId)
        except StaleRevisionError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

        exported = export_score(committed.score)
        return CommitDraftResponse(
            scoreXML=exported.score_xml,
            revision=committed.revision,
            measureMap=exported.measure_map,
            eventHitMap=exported.event_hit_map,
        )

    @router.post("/discard_draft", response_model=DiscardDraftResponse)
    async def discard_draft(request: DiscardDraftRequest) -> DiscardDraftResponse:
        _validate_draft_belongs_to_score(score_repository, request.scoreId, request.draftId)
        score_repository.discard_draft(request.draftId)
        return DiscardDraftResponse(ok=True)

    return router


def _validate_draft_belongs_to_score(
    repository: ScoreDraftRepository[CanonicalScore],
    score_id: str,
    draft_id: str,
) -> None:
    try:
        draft = repository.get_draft(draft_id)
    except DraftNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    if draft.score_id != score_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"draft {draft_id!r} does not belong to score {score_id!r}",
        )
