from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Sequence

from src.api.canonical import (
    CanonicalScore,
    Event,
    Measure,
    Part,
    carry_in_events_for_measure,
    events_starting_in_measure,
    measure_by_id,
    replace_events_in_measure,
)
from src.api.compose_service import (
    ScoreDocumentBundleExport,
    _normalize_to_guitar_range,
    export_score,
)
from src.api.store import ScoreDraftRepository
from src.tabber import DEFAULT_MAX_FRET, tab_events

ReplacementPlanner = Callable[[Part, Measure, list[Event], list[Event]], list[Event]]


@dataclass(frozen=True)
class InpaintWindowResult:
    draft_id: str
    score: CanonicalScore
    document: ScoreDocumentBundleExport
    base_revision: int
    highlight_measure_id: str
    locked_event_ids: list[str]
    changed_measure_ids: list[str]

    @property
    def score_xml(self) -> str:
        return self.document.score_xml

    @property
    def measure_map(self) -> dict[str, str]:
        return self.document.measure_map

    @property
    def event_hit_map(self) -> dict[str, str]:
        return self.document.event_hit_map


def preview_window_inpaint(
    repository: ScoreDraftRepository[CanonicalScore],
    score_id: str,
    *,
    revision: int,
    measure_id: str,
    locked_event_ids: Sequence[str] = (),
    replacement_planner: ReplacementPlanner | None = None,
) -> InpaintWindowResult:
    draft = repository.create_draft(score_id, base_revision=revision)
    updated_score, effective_locked_ids, changed_measure_ids = _inpaint_score(
        draft.score,
        measure_id=measure_id,
        locked_event_ids=locked_event_ids,
        replacement_planner=replacement_planner,
    )
    updated_score = _retab_if_guitar(draft.score, updated_score)
    saved_draft = repository.save_draft(draft.draft_id, updated_score)
    exported = export_score(saved_draft.score)
    return InpaintWindowResult(
        draft_id=saved_draft.draft_id,
        score=saved_draft.score,
        document=exported,
        base_revision=saved_draft.base_revision,
        highlight_measure_id=measure_id,
        locked_event_ids=effective_locked_ids,
        changed_measure_ids=changed_measure_ids,
    )


def _retab_if_guitar(original_score: CanonicalScore, updated_score: CanonicalScore) -> CanonicalScore:
    """Re-tab guitar scores while preserving unchanged user-selected fingerings."""
    if len(updated_score.parts) != 1:
        return updated_score
    part = updated_score.parts[0]
    if not part.info.tuning:
        return updated_score

    normalized_score = _normalize_to_guitar_range(updated_score)
    normalized_part = normalized_score.parts[0]
    try:
        tabbed_events = tab_events(
            normalized_part.events,
            tuning=normalized_part.info.tuning,
            max_fret=DEFAULT_MAX_FRET,
        )
    except Exception as exc:
        raise ValueError(f"guitar inpaint retab failed: {exc}") from exc

    preserved_events = _preserve_unchanged_fingerings(
        original_score.parts[0].events,
        tabbed_events,
    )
    return replace(
        normalized_score,
        parts=[replace(normalized_part, events=preserved_events)],
    )


def _preserve_unchanged_fingerings(
    original_events: Sequence[Event],
    retabbed_events: Sequence[Event],
) -> list[Event]:
    original_by_id = {event.id: event for event in original_events}
    preserved: list[Event] = []
    for event in retabbed_events:
        original = original_by_id.get(event.id)
        if (
            original is not None
            and original.fingering is not None
            and original.start_tick == event.start_tick
            and original.dur_tick == event.dur_tick
            and original.voice_id == event.voice_id
            and original.pitch_midi == event.pitch_midi
        ):
            preserved.append(replace(event, fingering=original.fingering))
            continue
        preserved.append(event)
    return preserved


def _inpaint_score(
    score: CanonicalScore,
    *,
    measure_id: str,
    locked_event_ids: Sequence[str],
    replacement_planner: ReplacementPlanner | None,
) -> tuple[CanonicalScore, list[str], list[str]]:
    measure = measure_by_id(score, measure_id)
    planner = replacement_planner or _default_replacement_planner
    effective_locked_ids = list(locked_event_ids)
    changed_measure_ids = {measure.id}
    updated_parts: list[Part] = []

    for part in score.parts:
        carry_in_events = carry_in_events_for_measure(part, measure)
        effective_locked_ids.extend(event.id for event in carry_in_events)
        replacement_events = planner(
            part,
            measure,
            events_starting_in_measure(part, measure),
            carry_in_events,
        )
        changed_measure_ids.update(
            _changed_measure_ids_for_events(score, measure, replacement_events)
        )
        updated_parts.append(replace_events_in_measure(part, measure, replacement_events))

    return (
        replace(score, parts=updated_parts),
        _dedupe_ids(effective_locked_ids),
        [
            candidate.id
            for candidate in score.measures
            if candidate.id in changed_measure_ids
        ],
    )


def _default_replacement_planner(
    part: Part,
    measure: Measure,
    measure_events: list[Event],
    carry_in_events: list[Event],
) -> list[Event]:
    del carry_in_events
    return [
        replace(
            event,
            id=_replacement_event_id(part, measure, ordinal),
            pitch_midi=_regenerated_pitch(event.pitch_midi),
            fingering=None,
        )
        for ordinal, event in enumerate(measure_events)
    ]


def _replacement_event_id(part: Part, measure: Measure, ordinal: int) -> str:
    return f"{part.info.id}-{measure.id}-regen-{ordinal}"


def _regenerated_pitch(pitch_midi: int | None) -> int | None:
    if pitch_midi is None:
        return None
    if pitch_midi < 127:
        return pitch_midi + 1
    return pitch_midi - 1


def _dedupe_ids(event_ids: Sequence[str]) -> list[str]:
    ordered_ids: list[str] = []
    seen_ids: set[str] = set()
    for event_id in event_ids:
        if event_id in seen_ids:
            continue
        seen_ids.add(event_id)
        ordered_ids.append(event_id)
    return ordered_ids


def _changed_measure_ids_for_events(
    score: CanonicalScore,
    start_measure: Measure,
    replacement_events: Sequence[Event],
) -> set[str]:
    changed_measure_ids: set[str] = set()
    for event in replacement_events:
        if event.end_tick <= start_measure.end_tick:
            continue
        for measure in score.measures[start_measure.index + 1 :]:
            if event.start_tick < measure.end_tick and measure.start_tick < event.end_tick:
                changed_measure_ids.add(measure.id)
    return changed_measure_ids
