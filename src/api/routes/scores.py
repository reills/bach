from __future__ import annotations

import base64
import random
from typing import Any, Callable, Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from src.api.canonical import (
    CanonicalScore,
    Event,
    EventNotFoundError,
    FingeringSelection,
    GuitarFingering,
    Measure,
    MeasureNotFoundError,
    Part,
    apply_fingering_selections,
    get_event_by_id,
    get_measure_by_id,
    splice_generated_measures,
)
from src.api.compose_service import ComposeServiceResult, export_score
from src.api.render import canonical_score_to_midi
from src.api.services import preview_window_inpaint
from src.api.store import (
    DraftNotFoundError,
    InMemoryScoreRepository,
    ScoreDraftRepository,
    ScoreNotFoundError,
    StaleRevisionError,
)
from src.api.store_serde import score_from_dict
from src.arrangers.guitar import GuitarArrangementSettings, convert_piano_score_to_guitar
from src.tabber import alternate_fingerings_for_event

ComposeHandler = Callable[[BaseModel], ComposeServiceResult]


class ComposeRequest(BaseModel):
    prompt: str | None = None
    constraints: dict[str, Any] | None = None
    name: str | None = None
    render_mode: Literal["guitar", "piano"] = "piano"


class ComposeResponse(BaseModel):
    scoreId: str
    revision: int
    document: dict[str, Any]
    scoreXML: str | None = None
    name: str
    createdAt: str
    updatedAt: str
    instrumentMode: Literal["guitar", "piano"]
    measureMap: dict[str, str] | None = None
    eventHitMap: dict[str, str] | None = None
    midi: str | None = None
    diagnostics: dict[str, Any] | None = None


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
    document: dict[str, Any]
    scoreXML: str | None = None
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
    document: dict[str, Any]
    scoreXML: str | None = None
    revision: int
    measureMap: dict[str, str] | None = None
    eventHitMap: dict[str, str] | None = None


class DiscardDraftRequest(BaseModel):
    scoreId: str
    draftId: str


class DiscardDraftResponse(BaseModel):
    ok: bool


class EventHitKeyRequest(BaseModel):
    barIndex: int
    voiceIndex: int | None = None
    beatIndex: int | None = None
    noteIndex: int | None = None


class AltPositionsRequest(BaseModel):
    scoreId: str
    measureId: str
    eventHitKey: EventHitKeyRequest | None = None


class AltPositionOption(BaseModel):
    stringIndex: int
    fret: int
    selected: bool


class AltPositionsResponse(BaseModel):
    eventId: str
    options: list[AltPositionOption]


class ApplyFingeringSelectionRequest(BaseModel):
    eventId: str
    stringIndex: int
    fret: int


class ApplyFingeringRequest(BaseModel):
    scoreId: str
    revision: int
    fingeringSelections: list[ApplyFingeringSelectionRequest]


class ApplyFingeringResponse(BaseModel):
    document: dict[str, Any]
    scoreXML: str | None = None
    revision: int


class AppendMeasuresRequest(BaseModel):
    scoreId: str
    revision: int
    count: int = 1


class AppendMeasuresResponse(BaseModel):
    document: dict[str, Any]
    scoreXML: str | None = None
    revision: int
    addedMeasureIds: list[str]


class GenerateMeasuresRequest(BaseModel):
    scoreId: str
    revision: int
    operation: Literal["prepend", "insert_before", "insert_after", "append", "replace"]
    count: int = 1
    measureId: str | None = None
    prompt: str | None = None
    constraints: dict[str, Any] | None = None
    render_mode: Literal["guitar", "piano"] | None = None


class GenerateMeasuresResponse(BaseModel):
    document: dict[str, Any]
    scoreXML: str | None = None
    revision: int
    insertedMeasureIds: list[str]
    replacedMeasureIds: list[str]
    changedMeasureIds: list[str]
    diagnostics: dict[str, Any] | None = None


class GuitarConversionSettingsRequest(BaseModel):
    difficulty: Literal["easy", "medium", "hard"] | None = None
    maxFret: int | None = None
    preferredPosition: int | None = None
    allowOctaveShift: bool | None = None
    octaveShiftPolicy: Literal["none", "below_range", "outside_range"] | None = None
    allowDropNotes: bool | None = None
    preserveMelody: bool | None = None
    preserveBass: bool | None = None
    maxHandSpanFrets: int | None = None
    maxNotesPerOnset: int | None = None
    tuning: list[int] | None = None


class ConvertToGuitarRequest(BaseModel):
    scoreId: str | None = None
    revision: int | None = None
    pianoScore: dict[str, Any] | None = None
    sourcePianoRevisionId: str | None = None
    settings: GuitarConversionSettingsRequest | None = None


class ConvertToGuitarResponse(BaseModel):
    scoreId: str
    revision: int
    branch: Literal["guitar"] = "guitar"
    instrumentMode: Literal["guitar"] = "guitar"
    document: dict[str, Any]
    scoreXML: str
    guitarMusicXml: str
    guitarTabXml: str | None = None
    midi: str
    sourcePianoRevisionId: str
    sourcePianoScoreId: str | None = None
    sourcePianoRevision: int | None = None
    conversionSettings: dict[str, Any]
    diagnostics: dict[str, Any]
    sourceMap: list[dict[str, Any]]


def create_router(
    *,
    compose_service: ComposeHandler | None = None,
    repository: ScoreDraftRepository[CanonicalScore] | None = None,
) -> APIRouter:
    router = APIRouter()
    score_repository = repository or InMemoryScoreRepository[CanonicalScore]()

    @router.post("/compose", response_model=ComposeResponse)
    def compose(request: ComposeRequest) -> ComposeResponse:
        if compose_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="compose service is not configured",
            )

        try:
            result = compose_service(request)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        stored_score = score_repository.create_score(result.score, name=request.name or "Untitled")
        return ComposeResponse(
            scoreId=stored_score.score_id,
            revision=stored_score.revision,
            document=_bundle_to_response_dict(result.document),
            scoreXML=result.document.score_xml,
            name=stored_score.name,
            createdAt=(stored_score.created_at.isoformat() if stored_score.created_at else ""),
            updatedAt=(stored_score.updated_at.isoformat() if stored_score.updated_at else ""),
            instrumentMode=result.document.instrument_mode,
            measureMap=result.document.measure_map,
            eventHitMap=result.document.event_hit_map,
            midi=base64.b64encode(result.midi).decode("ascii"),
            diagnostics=result.diagnostics,
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
        except MeasureNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return InpaintPreviewResponse(
            draftId=result.draft_id,
            document=_bundle_to_response_dict(result.document),
            scoreXML=result.document.score_xml,
            baseRevision=result.base_revision,
            highlightMeasureId=result.highlight_measure_id,
            measureMap=result.document.measure_map,
            eventHitMap=result.document.event_hit_map,
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
            document=_bundle_to_response_dict(exported),
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

    @router.post("/alt_positions", response_model=AltPositionsResponse)
    async def alt_positions(request: AltPositionsRequest) -> AltPositionsResponse:
        if request.eventHitKey is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="eventHitKey is required",
            )

        try:
            stored_score = score_repository.get_score(request.scoreId)
        except ScoreNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

        try:
            get_measure_by_id(stored_score.score, request.measureId)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

        exported = export_score(stored_score.score)
        event_view = _event_lookup_view(exported)
        resolved_measure_id = event_view.measure_map.get(str(request.eventHitKey.barIndex))
        if resolved_measure_id != request.measureId:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"event hit key bar {request.eventHitKey.barIndex} does not belong "
                    f"to measure {request.measureId!r}"
                ),
            )

        hit_key = _to_request_hit_key(request.eventHitKey)
        event_id = event_view.event_hit_map.get(hit_key)
        if event_id is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"unknown event hit key: {hit_key}",
            )

        try:
            part, event = _find_event_with_part(stored_score.score, event_id)
            alternates = alternate_fingerings_for_event(
                event,
                tuning=part.info.tuning,
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

        return AltPositionsResponse(
            eventId=event.id,
            options=[
                AltPositionOption(
                    stringIndex=option.string_index,
                    fret=option.fret,
                    selected=option == event.fingering,
                )
                for option in alternates
            ],
        )

    @router.post("/apply_fingering", response_model=ApplyFingeringResponse)
    async def apply_fingering(request: ApplyFingeringRequest) -> ApplyFingeringResponse:
        if not request.fingeringSelections:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="fingeringSelections must be non-empty",
            )

        try:
            draft = score_repository.create_draft(
                request.scoreId,
                base_revision=request.revision,
            )
        except ScoreNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except StaleRevisionError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

        try:
            updated_score = apply_fingering_selections(
                draft.score,
                _build_fingering_selections(draft.score, request.fingeringSelections),
            )
            score_repository.save_draft(draft.draft_id, updated_score)
            committed = score_repository.commit_draft(draft.draft_id)
        except EventNotFoundError as exc:
            score_repository.discard_draft(draft.draft_id)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except ValueError as exc:
            score_repository.discard_draft(draft.draft_id)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except StaleRevisionError as exc:
            score_repository.discard_draft(draft.draft_id)
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

        exported = export_score(committed.score)
        return ApplyFingeringResponse(
            document=_bundle_to_response_dict(exported),
            scoreXML=exported.score_xml,
            revision=committed.revision,
        )

    async def _generate_and_commit_measures(
        request: GenerateMeasuresRequest,
    ) -> GenerateMeasuresResponse:
        if compose_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="compose service is not configured",
            )
        if request.count <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="count must be positive",
            )
        if request.count > 32:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="count must be at most 32",
            )

        try:
            draft = score_repository.create_draft(
                request.scoreId,
                base_revision=request.revision,
            )
        except ScoreNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except StaleRevisionError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

        try:
            insert_index, replace_count = _resolve_measure_generation_window(
                draft.score,
                request,
            )
            context = _measure_context_summary(
                draft.score,
                insert_index=insert_index,
                replace_count=replace_count,
            )
            generation_request = ComposeRequest(
                prompt=request.prompt,
                constraints=_measure_generation_constraints(
                    request.constraints,
                    score=draft.score,
                    count=request.count,
                    context=context,
                ),
                render_mode=request.render_mode or _score_render_mode(draft.score),
            )
            generated = compose_service(generation_request)
            splice = splice_generated_measures(
                draft.score,
                generated.score,
                insert_index=insert_index,
                replace_count=replace_count,
                count=request.count,
            )
            updated_score = splice.score
            saved_draft = score_repository.save_draft(draft.draft_id, updated_score)
            committed = score_repository.commit_draft(saved_draft.draft_id)
        except MeasureNotFoundError as exc:
            score_repository.discard_draft(draft.draft_id)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except ValueError as exc:
            score_repository.discard_draft(draft.draft_id)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except StaleRevisionError as exc:
            score_repository.discard_draft(draft.draft_id)
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except Exception:
            score_repository.discard_draft(draft.draft_id)
            raise

        exported = export_score(committed.score)
        changed_measure_ids = [*splice.replaced_measure_ids, *splice.inserted_measure_ids]
        diagnostics: dict[str, Any] = {
            "operation": request.operation,
            "insertIndex": insert_index,
            "replaceCount": replace_count,
            "requestedCount": request.count,
            "insertedMeasureIds": splice.inserted_measure_ids,
            "replacedMeasureIds": splice.replaced_measure_ids,
            "changedMeasureIds": changed_measure_ids,
            "contextFitTransposition": splice.transposition,
            "context": context,
            "generation": generated.diagnostics,
        }
        return GenerateMeasuresResponse(
            document=_bundle_to_response_dict(exported),
            scoreXML=exported.score_xml,
            revision=committed.revision,
            insertedMeasureIds=splice.inserted_measure_ids,
            replacedMeasureIds=splice.replaced_measure_ids,
            changedMeasureIds=changed_measure_ids,
            diagnostics=diagnostics,
        )

    @router.post("/generate_measures", response_model=GenerateMeasuresResponse)
    async def generate_measures(request: GenerateMeasuresRequest) -> GenerateMeasuresResponse:
        return await _generate_and_commit_measures(request)

    @router.post("/append_measures", response_model=AppendMeasuresResponse)
    async def append_measures(request: AppendMeasuresRequest) -> AppendMeasuresResponse:
        response = await _generate_and_commit_measures(
            GenerateMeasuresRequest(
                scoreId=request.scoreId,
                revision=request.revision,
                operation="append",
                count=request.count,
            )
        )
        return AppendMeasuresResponse(
            document=response.document,
            scoreXML=response.scoreXML,
            revision=response.revision,
            addedMeasureIds=response.insertedMeasureIds,
        )

    @router.post("/api/convert-to-guitar", response_model=ConvertToGuitarResponse)
    @router.post("/convert-to-guitar", response_model=ConvertToGuitarResponse)
    async def convert_to_guitar(request: ConvertToGuitarRequest) -> ConvertToGuitarResponse:
        try:
            piano_score, source_score_id, source_revision, source_name = _resolve_piano_source(
                score_repository,
                request,
            )
            _validate_piano_source(piano_score)
            arrangement = convert_piano_score_to_guitar(
                piano_score,
                settings=_build_guitar_arrangement_settings(request.settings),
            )
            exported = export_score(arrangement.score)
            stored_guitar = score_repository.create_score(
                arrangement.score,
                name=f"Guitar arrangement of {source_name}",
            )
        except ScoreNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except StaleRevisionError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        score_view = exported.views["score"]
        tab_view = exported.views.get("tab")
        arrangement_payload = arrangement.to_dict()
        source_map_payload = arrangement.source_map.to_dict()
        return ConvertToGuitarResponse(
            scoreId=stored_guitar.score_id,
            revision=stored_guitar.revision,
            document=_bundle_to_response_dict(exported),
            scoreXML=score_view.xml,
            guitarMusicXml=score_view.xml,
            guitarTabXml=tab_view.xml if tab_view else None,
            midi=base64.b64encode(canonical_score_to_midi(arrangement.score)).decode("ascii"),
            sourcePianoRevisionId=_source_piano_revision_id(
                request,
                source_score_id=source_score_id,
                source_revision=source_revision,
            ),
            sourcePianoScoreId=source_score_id,
            sourcePianoRevision=source_revision,
            conversionSettings=arrangement_payload["settings"],  # type: ignore[index]
            diagnostics=arrangement_payload["diagnostics"],  # type: ignore[index]
            sourceMap=source_map_payload["notes"],  # type: ignore[index]
        )

    return router


def _resolve_piano_source(
    repository: ScoreDraftRepository[CanonicalScore],
    request: ConvertToGuitarRequest,
) -> tuple[CanonicalScore, str | None, int | None, str]:
    if request.scoreId and request.pianoScore is not None:
        raise ValueError("provide either scoreId or pianoScore, not both")
    if request.scoreId:
        if request.revision is None:
            raise ValueError("revision is required when scoreId is provided")
        stored_score = repository.get_score(request.scoreId)
        if stored_score.revision != request.revision:
            raise StaleRevisionError(
                f"score {request.scoreId!r} is at revision {stored_score.revision}, "
                f"not {request.revision}"
            )
        return stored_score.score, stored_score.score_id, stored_score.revision, stored_score.name
    if request.pianoScore is not None:
        return score_from_dict(request.pianoScore), None, request.revision, "inline piano score"
    raise ValueError("scoreId or pianoScore is required")


def _validate_piano_source(score: CanonicalScore) -> None:
    if len(score.parts) != 1:
        raise ValueError("piano-to-guitar conversion requires exactly one source part")
    instrument = score.parts[0].info.instrument
    if instrument != "piano":
        raise ValueError(f"piano-to-guitar conversion requires a piano source, got {instrument!r}")


def _build_guitar_arrangement_settings(
    request_settings: GuitarConversionSettingsRequest | None,
) -> GuitarArrangementSettings:
    if request_settings is None:
        return GuitarArrangementSettings()

    kwargs: dict[str, Any] = {}
    if request_settings.difficulty is not None:
        kwargs["difficulty"] = request_settings.difficulty
    if request_settings.maxFret is not None:
        kwargs["max_fret"] = request_settings.maxFret
    if request_settings.preferredPosition is not None:
        kwargs["preferred_position"] = request_settings.preferredPosition
    if request_settings.allowOctaveShift is not None:
        kwargs["octave_shift_policy"] = (
            "outside_range" if request_settings.allowOctaveShift else "none"
        )
    if request_settings.octaveShiftPolicy is not None:
        kwargs["octave_shift_policy"] = request_settings.octaveShiftPolicy
    if request_settings.allowDropNotes is not None:
        kwargs["allow_drop_notes"] = request_settings.allowDropNotes
    if request_settings.preserveMelody is not None:
        kwargs["preserve_melody"] = request_settings.preserveMelody
    if request_settings.preserveBass is not None:
        kwargs["preserve_bass"] = request_settings.preserveBass
    if request_settings.maxHandSpanFrets is not None:
        kwargs["max_hand_span_frets"] = request_settings.maxHandSpanFrets
    if request_settings.maxNotesPerOnset is not None:
        kwargs["max_notes_per_onset"] = request_settings.maxNotesPerOnset
    if request_settings.tuning is not None:
        kwargs["tuning"] = tuple(request_settings.tuning)
    return GuitarArrangementSettings(**kwargs)


def _source_piano_revision_id(
    request: ConvertToGuitarRequest,
    *,
    source_score_id: str | None,
    source_revision: int | None,
) -> str:
    if request.sourcePianoRevisionId:
        return request.sourcePianoRevisionId
    if source_score_id is not None and source_revision is not None:
        return f"{source_score_id}@{source_revision}"
    if source_revision is not None:
        return str(source_revision)
    return "inline"


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


def _find_event_with_part(score: CanonicalScore, event_id: str) -> tuple[Part, Event]:
    for part in score.parts:
        for event in part.events:
            if event.id == event_id:
                return part, event
    raise EventNotFoundError(f"unknown event id: {event_id}")


def _build_fingering_selections(
    score: CanonicalScore,
    selections: list[ApplyFingeringSelectionRequest],
) -> list[FingeringSelection]:
    canonical_selections: list[FingeringSelection] = []
    for selection in selections:
        event = get_event_by_id(score, selection.eventId)
        canonical_selections.append(
            FingeringSelection(
                event_id=selection.eventId,
                pitch_midi=event.pitch_midi,
                start_tick=event.start_tick,
                dur_tick=event.dur_tick,
                fingering=GuitarFingering(
                    string_index=selection.stringIndex,
                    fret=selection.fret,
                ),
            )
        )
    return canonical_selections


def _resolve_measure_generation_window(
    score: CanonicalScore,
    request: GenerateMeasuresRequest,
) -> tuple[int, int]:
    if request.operation == "append":
        return len(score.measures), 0
    if request.operation == "prepend":
        return 0, 0

    if request.measureId is None:
        raise ValueError(f"measureId is required for {request.operation}")

    measure = get_measure_by_id(score, request.measureId)
    if request.operation == "insert_before":
        return measure.index, 0
    if request.operation == "insert_after":
        return measure.index + 1, 0
    if request.operation == "replace":
        if measure.index + request.count > len(score.measures):
            raise ValueError("replace count extends beyond the score")
        return measure.index, request.count

    raise ValueError(f"unsupported measure operation: {request.operation}")


def _measure_generation_constraints(
    constraints: dict[str, Any] | None,
    *,
    score: CanonicalScore,
    count: int,
    context: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(constraints or {})
    voice_count = _score_voice_count(score)
    merged["measures"] = count
    merged["texture"] = voice_count
    merged["voices"] = voice_count
    seed = merged.get("seed", merged.get("randomSeed"))
    if seed is None:
        seed = random.SystemRandom().randint(1, 2_147_483_646)
    merged.setdefault("seed", seed)
    merged.setdefault("randomSeed", seed)
    merged["measureContext"] = context
    return merged


def _score_voice_count(score: CanonicalScore) -> int:
    voice_ids = {
        event.voice_id
        for part in score.parts
        for event in part.events
    }
    return max(1, max(voice_ids) + 1 if voice_ids else 1)


def _score_render_mode(score: CanonicalScore) -> Literal["guitar", "piano"]:
    instrument = score.parts[0].info.instrument if score.parts else ""
    return "piano" if instrument == "piano" else "guitar"


def _measure_context_summary(
    score: CanonicalScore,
    *,
    insert_index: int,
    replace_count: int,
    radius: int = 2,
) -> dict[str, Any]:
    after_index = insert_index + replace_count
    before = [
        _measure_summary(score, measure)
        for measure in score.measures[max(0, insert_index - radius) : insert_index]
    ]
    after = [
        _measure_summary(score, measure)
        for measure in score.measures[after_index : min(len(score.measures), after_index + radius)]
    ]
    return {
        "insertIndex": insert_index,
        "replaceCount": replace_count,
        "before": before,
        "after": after,
    }


def _measure_summary(score: CanonicalScore, measure: Measure) -> dict[str, Any]:
    voices: dict[str, list[int]] = {}
    for part in score.parts:
        for event in part.events:
            if not measure.start_tick <= event.start_tick < measure.end_tick:
                continue
            if event.pitch_midi is None:
                continue
            voices.setdefault(str(event.voice_id), []).append(event.pitch_midi)
    return {
        "id": measure.id,
        "index": measure.index,
        "startTick": measure.start_tick,
        "lengthTicks": measure.length_ticks,
        "voices": voices,
    }


def _to_request_hit_key(hit_key: EventHitKeyRequest) -> str:
    voice_index = -1 if hit_key.voiceIndex is None else hit_key.voiceIndex
    beat_index = -1 if hit_key.beatIndex is None else hit_key.beatIndex
    note_index = -1 if hit_key.noteIndex is None else hit_key.noteIndex
    return f"{hit_key.barIndex}|{voice_index}|{beat_index}|{note_index}"


def _bundle_to_response_dict(bundle: Any) -> dict[str, Any]:
    return {
        "instrumentMode": bundle.instrument_mode,
        "views": {
            name: {
                "xml": view.xml,
                "measureMap": view.measure_map,
                "eventHitMap": view.event_hit_map,
            }
            for name, view in bundle.views.items()
        },
    }


def _event_lookup_view(exported: Any) -> Any:
    return exported.views.get("tab") or exported.views["score"]
