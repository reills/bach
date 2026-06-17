from dataclasses import dataclass, replace

from src.api.canonical.ops import EventNotFoundError
from src.api.canonical.types import CanonicalScore, Event, GuitarFingering, Part


@dataclass(frozen=True)
class FingeringSelection:
    event_id: str
    pitch_midi: int | None
    start_tick: int
    dur_tick: int
    fingering: GuitarFingering | None

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must be non-empty")


def apply_fingering_selections(
    score: CanonicalScore,
    selections: list[FingeringSelection],
) -> CanonicalScore:
    selection_by_event_id = _selection_map(selections)
    if not selection_by_event_id:
        return score

    updated_parts: list[Part] = []
    unmatched_event_ids = set(selection_by_event_id)

    for part in score.parts:
        updated_events: list[Event] = []
        for event in part.events:
            selection = selection_by_event_id.get(event.id)
            if selection is None:
                updated_events.append(event)
                continue

            _validate_selection(event, selection)
            updated_events.append(replace(event, fingering=selection.fingering))
            unmatched_event_ids.discard(event.id)

        updated_parts.append(Part(info=part.info, events=updated_events))

    if unmatched_event_ids:
        missing_event_id = sorted(unmatched_event_ids)[0]
        raise EventNotFoundError(f"unknown event id: {missing_event_id}")

    return CanonicalScore(
        header=score.header,
        measures=score.measures,
        parts=updated_parts,
    )


def _selection_map(selections: list[FingeringSelection]) -> dict[str, FingeringSelection]:
    selection_by_event_id: dict[str, FingeringSelection] = {}
    for selection in selections:
        if selection.event_id in selection_by_event_id:
            raise ValueError(f"duplicate fingering selection for event id: {selection.event_id}")
        selection_by_event_id[selection.event_id] = selection
    return selection_by_event_id


def _validate_selection(event: Event, selection: FingeringSelection) -> None:
    if event.pitch_midi != selection.pitch_midi:
        raise ValueError(f"fingering selections cannot change pitch for event {event.id}")
    if event.start_tick != selection.start_tick or event.dur_tick != selection.dur_tick:
        raise ValueError(f"fingering selections cannot change timing for event {event.id}")
