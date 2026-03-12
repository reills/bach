from src.api.canonical.types import CanonicalScore, Event, Measure, Part


class MeasureNotFoundError(ValueError):
    pass


class EventNotFoundError(ValueError):
    pass


def measure_by_id(score: CanonicalScore, measure_id: str) -> Measure:
    for measure in score.measures:
        if measure.id == measure_id:
            return measure
    raise MeasureNotFoundError(f"unknown measure id: {measure_id}")


def get_measure_by_id(score: CanonicalScore, measure_id: str) -> Measure:
    return measure_by_id(score, measure_id)


def event_by_id(score_or_part: CanonicalScore | Part, event_id: str) -> Event:
    parts = score_or_part.parts if isinstance(score_or_part, CanonicalScore) else [score_or_part]
    for part in parts:
        for event in part.events:
            if event.id == event_id:
                return event
    raise EventNotFoundError(f"unknown event id: {event_id}")


def get_event_by_id(score_or_part: CanonicalScore | Part, event_id: str) -> Event:
    return event_by_id(score_or_part, event_id)


def events_starting_in_measure(part: Part, measure: Measure) -> list[Event]:
    return [
        event
        for event in part.events
        if measure.start_tick <= event.start_tick < measure.end_tick
    ]


def carry_in_events_for_measure(part: Part, measure: Measure) -> list[Event]:
    return [
        event
        for event in part.events
        if event.start_tick < measure.start_tick < event.end_tick
    ]


def replace_events_in_measure(part: Part, measure: Measure, new_events: list[Event]) -> Part:
    _validate_replacement_events(measure, new_events)
    retained_before = [event for event in part.events if event.start_tick < measure.start_tick]
    retained_after = [event for event in part.events if event.start_tick >= measure.end_tick]
    return Part(info=part.info, events=[*retained_before, *new_events, *retained_after])


def _validate_replacement_events(measure: Measure, new_events: list[Event]) -> None:
    previous_start_tick = -1
    for event in new_events:
        if event.start_tick < previous_start_tick:
            raise ValueError("replacement events must be sorted by start_tick")
        if not measure.start_tick <= event.start_tick < measure.end_tick:
            raise ValueError("replacement events must start inside the target measure")
        previous_start_tick = event.start_tick
