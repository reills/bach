from dataclasses import dataclass, replace
from statistics import median

from src.api.canonical.types import CanonicalScore, Event, Measure, Part


class MeasureNotFoundError(ValueError):
    pass


class EventNotFoundError(ValueError):
    pass


@dataclass(frozen=True)
class MeasureSpliceResult:
    score: CanonicalScore
    inserted_measure_ids: list[str]
    replaced_measure_ids: list[str]
    transposition: int


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


def splice_generated_measures(
    score: CanonicalScore,
    generated_score: CanonicalScore,
    *,
    insert_index: int,
    replace_count: int,
    count: int,
    fit_to_context: bool = True,
) -> MeasureSpliceResult:
    _validate_splice_args(score, generated_score, insert_index, replace_count, count)

    splice_start = (
        score.measures[insert_index].start_tick
        if insert_index < len(score.measures)
        else score.total_ticks
    )
    splice_end = (
        score.measures[insert_index + replace_count - 1].end_tick
        if replace_count
        else splice_start
    )
    target_measure_length = _target_measure_length(score, insert_index)
    inserted_total_ticks = target_measure_length * count
    tick_delta = inserted_total_ticks - (splice_end - splice_start)
    replaced_measure_ids = [
        measure.id
        for measure in score.measures[insert_index : insert_index + replace_count]
    ]
    generated_measures = generated_score.measures[:count]
    transposition = (
        _context_transposition(
            score,
            generated_score,
            splice_start=splice_start,
            splice_end=splice_end,
            generated_measures=generated_measures,
        )
        if fit_to_context
        else 0
    )

    measures, inserted_measure_ids = _spliced_measures(
        score,
        insert_index=insert_index,
        replace_count=replace_count,
        target_measure_length=target_measure_length,
        count=count,
        existing_measure_ids={measure.id for measure in score.measures},
    )
    parts = _spliced_parts(
        score,
        generated_score,
        generated_measures=generated_measures,
        splice_start=splice_start,
        splice_end=splice_end,
        tick_delta=tick_delta,
        target_measure_length=target_measure_length,
        inserted_measure_ids=inserted_measure_ids,
        transposition=transposition,
    )
    header = replace(
        score.header,
        key_sig_map=_shift_tick_map(score.header.key_sig_map, splice_start, tick_delta),
        time_sig_map=_shift_tick_map(score.header.time_sig_map, splice_start, tick_delta),
        tempo_map=_shift_tick_map(score.header.tempo_map, splice_start, tick_delta),
    )
    return MeasureSpliceResult(
        score=CanonicalScore(header=header, measures=measures, parts=parts),
        inserted_measure_ids=inserted_measure_ids,
        replaced_measure_ids=replaced_measure_ids,
        transposition=transposition,
    )


def _validate_replacement_events(measure: Measure, new_events: list[Event]) -> None:
    previous_start_tick = -1
    for event in new_events:
        if event.start_tick < previous_start_tick:
            raise ValueError("replacement events must be sorted by start_tick")
        if not measure.start_tick <= event.start_tick < measure.end_tick:
            raise ValueError("replacement events must start inside the target measure")
        previous_start_tick = event.start_tick


def _part_voice_count(part: Part) -> int:
    if not part.events:
        return 1
    return max(event.voice_id for event in part.events) + 1


def _unique_id(base: str, existing_ids: set[str]) -> str:
    candidate = base
    suffix = 2
    while candidate in existing_ids:
        candidate = f"{base}-{suffix}"
        suffix += 1
    existing_ids.add(candidate)
    return candidate


def _validate_splice_args(
    score: CanonicalScore,
    generated_score: CanonicalScore,
    insert_index: int,
    replace_count: int,
    count: int,
) -> None:
    if not isinstance(insert_index, int) or isinstance(insert_index, bool):
        raise ValueError("insert_index must be an integer")
    if not isinstance(replace_count, int) or isinstance(replace_count, bool):
        raise ValueError("replace_count must be an integer")
    if not isinstance(count, int) or isinstance(count, bool):
        raise ValueError("count must be an integer")
    if insert_index < 0 or insert_index > len(score.measures):
        raise ValueError("insert_index must be within the score")
    if replace_count < 0:
        raise ValueError("replace_count must be non-negative")
    if insert_index + replace_count > len(score.measures):
        raise ValueError("replace_count extends beyond the score")
    if count <= 0:
        raise ValueError("count must be positive")
    if len(generated_score.measures) < count:
        raise ValueError("generated score does not contain enough measures")
    if len(score.parts) != len(generated_score.parts):
        raise ValueError("generated score must have the same part count")


def _target_measure_length(score: CanonicalScore, insert_index: int) -> int:
    if insert_index < len(score.measures):
        return score.measures[insert_index].length_ticks
    return score.measures[-1].length_ticks


def _spliced_measures(
    score: CanonicalScore,
    *,
    insert_index: int,
    replace_count: int,
    target_measure_length: int,
    count: int,
    existing_measure_ids: set[str],
) -> tuple[list[Measure], list[str]]:
    measures: list[Measure] = []
    inserted_measure_ids: list[str] = []
    current_tick = 0

    for measure in score.measures[:insert_index]:
        measures.append(
            replace(measure, index=len(measures), start_tick=current_tick)
        )
        current_tick += measure.length_ticks

    for _ in range(count):
        measure_id = _unique_id(f"m{len(measures)}", existing_measure_ids)
        measures.append(
            Measure(
                id=measure_id,
                index=len(measures),
                start_tick=current_tick,
                length_ticks=target_measure_length,
            )
        )
        inserted_measure_ids.append(measure_id)
        current_tick += target_measure_length

    for measure in score.measures[insert_index + replace_count :]:
        measures.append(
            replace(measure, index=len(measures), start_tick=current_tick)
        )
        current_tick += measure.length_ticks

    return measures, inserted_measure_ids


def _spliced_parts(
    score: CanonicalScore,
    generated_score: CanonicalScore,
    *,
    generated_measures: list[Measure],
    splice_start: int,
    splice_end: int,
    tick_delta: int,
    target_measure_length: int,
    inserted_measure_ids: list[str],
    transposition: int,
) -> list[Part]:
    parts: list[Part] = []
    for source_part, generated_part in zip(score.parts, generated_score.parts):
        retained_before, retained_after = _retained_source_events(
            source_part,
            splice_start=splice_start,
            splice_end=splice_end,
            tick_delta=tick_delta,
        )
        event_ids = {
            event.id
            for event in [*retained_before, *retained_after]
        }
        generated_events = _generated_splice_events(
            source_part,
            generated_part,
            generated_measures=generated_measures,
            splice_start=splice_start,
            target_measure_length=target_measure_length,
            inserted_measure_ids=inserted_measure_ids,
            event_ids=event_ids,
            transposition=transposition,
        )
        parts.append(
            Part(
                info=source_part.info,
                events=sorted(
                    [*retained_before, *generated_events, *retained_after],
                    key=lambda event: (event.start_tick, event.voice_id, event.id),
                ),
            )
        )
    return parts


def _retained_source_events(
    part: Part,
    *,
    splice_start: int,
    splice_end: int,
    tick_delta: int,
) -> tuple[list[Event], list[Event]]:
    retained_before: list[Event] = []
    retained_after: list[Event] = []
    for event in part.events:
        if event.end_tick <= splice_start:
            retained_before.append(event)
            continue
        if event.start_tick < splice_start < event.end_tick:
            retained_before.append(replace(event, dur_tick=splice_start - event.start_tick))
            continue
        if event.start_tick >= splice_end:
            retained_after.append(replace(event, start_tick=event.start_tick + tick_delta))
    return retained_before, retained_after


def _generated_splice_events(
    source_part: Part,
    generated_part: Part,
    *,
    generated_measures: list[Measure],
    splice_start: int,
    target_measure_length: int,
    inserted_measure_ids: list[str],
    event_ids: set[str],
    transposition: int,
) -> list[Event]:
    generated_events: list[Event] = []
    ordinal = 0
    for measure_offset, generated_measure in enumerate(generated_measures):
        target_measure_start = splice_start + (measure_offset * target_measure_length)
        target_measure_end = target_measure_start + target_measure_length
        target_measure_id = inserted_measure_ids[measure_offset]
        for event in generated_part.events:
            if event.start_tick >= generated_measure.end_tick or event.end_tick <= generated_measure.start_tick:
                continue
            local_start = max(event.start_tick, generated_measure.start_tick) - generated_measure.start_tick
            local_end = min(event.end_tick, generated_measure.end_tick) - generated_measure.start_tick
            scaled_start = _scale_tick(local_start, generated_measure.length_ticks, target_measure_length)
            scaled_end = _scale_tick(local_end, generated_measure.length_ticks, target_measure_length)
            start_tick = min(target_measure_end - 1, target_measure_start + scaled_start)
            end_tick = min(target_measure_end, target_measure_start + max(scaled_end, scaled_start + 1))
            if end_tick <= start_tick:
                continue
            pitch_midi = _transpose_pitch(event.pitch_midi, transposition)
            event_id = _unique_id(
                f"{source_part.info.id}-{target_measure_id}-gen-{event.voice_id}-{ordinal}",
                event_ids,
            )
            generated_events.append(
                Event(
                    id=event_id,
                    start_tick=start_tick,
                    dur_tick=end_tick - start_tick,
                    voice_id=event.voice_id,
                    pitch_midi=pitch_midi,
                    velocity=event.velocity,
                    fingering=event.fingering if transposition == 0 else None,
                )
            )
            ordinal += 1
    return generated_events


def _scale_tick(value: int, source_length: int, target_length: int) -> int:
    if source_length == target_length:
        return value
    return round((value / source_length) * target_length)


def _transpose_pitch(pitch_midi: int | None, semitones: int) -> int | None:
    if pitch_midi is None:
        return None
    return max(0, min(127, pitch_midi + semitones))


def _context_transposition(
    score: CanonicalScore,
    generated_score: CanonicalScore,
    *,
    splice_start: int,
    splice_end: int,
    generated_measures: list[Measure],
) -> int:
    shifts: list[int] = []
    for source_part, generated_part in zip(score.parts, generated_score.parts):
        voice_count = max(_part_voice_count(source_part), _part_voice_count(generated_part))
        for voice_id in range(voice_count):
            first_generated = _first_pitch_in_generated_measures(
                generated_part,
                generated_measures,
                voice_id=voice_id,
            )
            last_generated = _last_pitch_in_generated_measures(
                generated_part,
                generated_measures,
                voice_id=voice_id,
            )
            previous_pitch = _last_pitch_before(source_part, splice_start, voice_id=voice_id)
            next_pitch = _first_pitch_after(source_part, splice_end, voice_id=voice_id)
            if previous_pitch is not None and first_generated is not None:
                shifts.append(_nearby_register_shift(previous_pitch - first_generated))
            if next_pitch is not None and last_generated is not None:
                shifts.append(_nearby_register_shift(next_pitch - last_generated))
    if not shifts:
        return 0
    return int(round(median(shifts)))


def _nearby_register_shift(shift: int) -> int:
    while shift > 12:
        shift -= 12
    while shift < -12:
        shift += 12
    return shift


def _first_pitch_in_generated_measures(
    part: Part,
    measures: list[Measure],
    *,
    voice_id: int,
) -> int | None:
    measure_start = measures[0].start_tick
    measure_end = measures[-1].end_tick
    events = [
        event
        for event in part.events
        if event.voice_id == voice_id
        and event.pitch_midi is not None
        and measure_start <= event.start_tick < measure_end
    ]
    if not events:
        return None
    return min(events, key=lambda event: event.start_tick).pitch_midi


def _last_pitch_in_generated_measures(
    part: Part,
    measures: list[Measure],
    *,
    voice_id: int,
) -> int | None:
    measure_start = measures[0].start_tick
    measure_end = measures[-1].end_tick
    events = [
        event
        for event in part.events
        if event.voice_id == voice_id
        and event.pitch_midi is not None
        and measure_start <= event.start_tick < measure_end
    ]
    if not events:
        return None
    return max(events, key=lambda event: (event.start_tick, event.end_tick)).pitch_midi


def _last_pitch_before(part: Part, tick: int, *, voice_id: int) -> int | None:
    events = [
        event
        for event in part.events
        if event.voice_id == voice_id
        and event.pitch_midi is not None
        and event.start_tick < tick
    ]
    if not events:
        return None
    return max(events, key=lambda event: (event.start_tick, event.end_tick)).pitch_midi


def _first_pitch_after(part: Part, tick: int, *, voice_id: int) -> int | None:
    events = [
        event
        for event in part.events
        if event.voice_id == voice_id
        and event.pitch_midi is not None
        and event.start_tick >= tick
    ]
    if not events:
        return None
    return min(events, key=lambda event: event.start_tick).pitch_midi


def _shift_tick_map(tick_map: dict[int, object], splice_start: int, tick_delta: int) -> dict[int, object]:
    if tick_delta == 0:
        return dict(tick_map)
    shifted: dict[int, object] = {}
    for tick, value in tick_map.items():
        new_tick = tick + tick_delta if tick > splice_start else tick
        shifted[new_tick] = value
    return shifted
