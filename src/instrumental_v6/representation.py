from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import music21

from src.api.canonical import CanonicalScore, Event, Measure, Part, PartInfo, ScoreHeader
from src.emi.planner import PHRASE_ROLE_NAMES
from src.tokens import eventizer

TPQ = 24
GRID_TICKS = 6
DEFAULT_MAX_VOICES = 6
MAX_INTERVAL = 24
MAX_DURATION_STEPS = 32
MAX_VERTICAL_INTERVAL = 60
MAX_STEPS_PER_BAR = 48
MAX_BARS = 128
PHRASE_BARS = 8

STATE_REST = 0
STATE_HOLD = 1
STATE_NOTE = 2

FORM_NAMES = ["UNKNOWN", "INVENTION", "SINFONIA", "FUGUE", "SUITE", "PARTITA", "PRELUDE"]
FORM_TO_ID = {name: index for index, name in enumerate(FORM_NAMES)}
ROLE_TO_ID = {name: index for index, name in enumerate(PHRASE_ROLE_NAMES)}

DEVELOPMENT_NAMES = [
    "UNKNOWN",
    "OPENING_MOTIF",
    "SUBJECT",
    "ANSWER",
    "COUNTERSUBJECT",
    "EPISODE",
    "SEQUENCE_UP",
    "SEQUENCE_DOWN",
    "INVERSION",
    "STRETTO",
    "BINARY_A",
    "BINARY_B",
    "RECAP",
    "CADENCE",
]
DEVELOPMENT_TO_ID = {name: index for index, name in enumerate(DEVELOPMENT_NAMES)}

METER_NAMES = [
    "UNKNOWN",
    "2/4",
    "3/4",
    "3/8",
    "4/4",
    "6/8",
    "9/8",
    "12/8",
    "2/2",
    "3/2",
    "6/4",
    "8/8",
    "12/16",
    "6/16",
    "4/2",
    "9/16",
    "24/16",
    "4/8",
]
METER_TO_ID = {name: index for index, name in enumerate(METER_NAMES)}

MOTION_NAMES = ["UNKNOWN", "STATIC", "OBLIQUE", "CONTRARY", "SIMILAR", "PARALLEL"]
MOTION_TO_ID = {name: index for index, name in enumerate(MOTION_NAMES)}

GLOBAL_FIELD_NAMES = [
    "bar",
    "pos",
    "phrase_pos",
    "cadence_zone",
    "key_pc",
    "mode",
    "voice_count",
    "form",
    "meter",
    "section_role",
    "development",
    "entry_voice",
    "local_key_pc",
]
VOICE_FIELD_NAMES = ["state", "pitch", "mel", "dur", "tie", "degree"]
PAIR_FIELD_NAMES = [
    "interval",
    "interval_class",
    "consonance",
    "motion",
    "parallel_perfect",
    "direct_perfect",
    "crossing",
    "spacing_violation",
]

GLOBAL_FEATURE_SPECS = {
    "bar": MAX_BARS,
    "pos": MAX_STEPS_PER_BAR,
    "phrase_pos": PHRASE_BARS,
    "cadence_zone": 2,
    "key_pc": 13,
    "mode": 3,
    "voice_count": DEFAULT_MAX_VOICES + 1,
    "form": len(FORM_NAMES),
    "meter": len(METER_NAMES),
    "section_role": len(PHRASE_ROLE_NAMES),
    "development": len(DEVELOPMENT_NAMES),
    "entry_voice": DEFAULT_MAX_VOICES + 1,
    "local_key_pc": 13,
}
VOICE_FEATURE_SPECS = {
    "state": 3,
    "pitch": 129,
    "mel": (MAX_INTERVAL * 2) + 2,
    "dur": MAX_DURATION_STEPS + 1,
    "tie": 2,
    "degree": 13,
}
PAIR_FEATURE_SPECS = {
    "interval": MAX_VERTICAL_INTERVAL + 2,
    "interval_class": 13,
    "consonance": 4,
    "motion": len(MOTION_NAMES),
    "parallel_perfect": 2,
    "direct_perfect": 2,
    "crossing": 2,
    "spacing_violation": 2,
}

_MAJOR_SCALE = {0: 1, 2: 2, 4: 3, 5: 4, 7: 5, 9: 6, 11: 7}
_MINOR_SCALE = {0: 1, 2: 2, 3: 3, 5: 4, 7: 5, 8: 6, 10: 7, 11: 7}
_PERFECT_CLASSES = {0, 7}
_IMPERFECT_CLASSES = {3, 4, 8, 9}
_KNOWN_FUGUE_VOICE_COUNTS = {
    "BWV_0846": 4,
    "BWV_0848": 3,
    "BWV_0849": 5,
    "BWV_0850": 4,
    "BWV_0851": 3,
    "BWV_0853": 3,
    "BWV_0855": 2,
    "BWV_0867": 5,
    "BWV_0869": 4,
    "BWV_0891": 4,
    "BWV_0892": 4,
    "BWV_0893": 3,
}


@dataclass(frozen=True)
class DevelopmentStep:
    role: str
    operation: str
    entry_voice: int | None
    local_key_pc: int


@dataclass(frozen=True)
class InstrumentalV6Piece:
    piece_id: str
    source_path: str
    form: str
    movement_index: int
    tpq: int
    grid_ticks: int
    time_signature: str
    key: str | None
    key_pc: int
    mode: int
    voice_count: int
    max_voices: int
    bar_len_ticks: int
    steps_per_bar: int
    global_rows: list[list[int]]
    voice_rows: list[list[list[int]]]
    pair_rows: list[list[list[list[int]]]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InstrumentalV6Piece":
        return cls(
            piece_id=str(data["piece_id"]),
            source_path=str(data["source_path"]),
            form=str(data["form"]),
            movement_index=int(data["movement_index"]),
            tpq=int(data["tpq"]),
            grid_ticks=int(data["grid_ticks"]),
            time_signature=str(data["time_signature"]),
            key=data.get("key"),
            key_pc=int(data["key_pc"]),
            mode=int(data["mode"]),
            voice_count=int(data["voice_count"]),
            max_voices=int(data["max_voices"]),
            bar_len_ticks=int(data["bar_len_ticks"]),
            steps_per_bar=int(data["steps_per_bar"]),
            global_rows=[[int(value) for value in row] for row in data["global_rows"]],
            voice_rows=[
                [[int(value) for value in voice] for voice in row]
                for row in data["voice_rows"]
            ],
            pair_rows=[
                [
                    [[int(value) for value in pair] for pair in left]
                    for left in row
                ]
                for row in data["pair_rows"]
            ],
        )


def parse_musicxml_movements(
    path: str | Path,
    *,
    form: str,
    target_voices: int | None,
    max_voices: int = DEFAULT_MAX_VOICES,
    max_bars: int = 48,
    normalize_key: bool = True,
    max_movements: int = 0,
) -> list[InstrumentalV6Piece]:
    if target_voices is not None and not 2 <= target_voices <= max_voices:
        raise ValueError("target_voices must be between 2 and max_voices")
    if max_voices < 2:
        raise ValueError("max_voices must be at least 2")
    path = Path(path)
    score = music21.converter.parse(str(path))
    pieces: list[InstrumentalV6Piece] = []
    boundaries = _movement_boundaries(score)
    if form.upper() == "WTC" and max_movements > 0:
        movement_indices = list(range(min(len(boundaries), max_movements)))
    else:
        movement_indices = _select_movement_indices(len(boundaries), max_movements)
    for movement_index in movement_indices:
        boundary = boundaries[movement_index]
        movement_form = _movement_form(form, movement_index)
        try:
            piece = _build_piece(
                score,
                path=path,
                form=movement_form,
                target_voices=target_voices,
                max_voices=max_voices,
                movement_index=movement_index,
                start_ql=boundary[0],
                end_ql=boundary[1],
                time_signature=boundary[2],
                max_bars=max_bars,
                normalize_key=normalize_key,
            )
        except ValueError:
            continue
        if piece.global_rows:
            pieces.append(piece)
    return pieces


def _select_movement_indices(count: int, limit: int) -> list[int]:
    if count <= 0:
        return []
    if limit <= 0 or count <= limit:
        return list(range(count))
    if limit == 1:
        return [0]
    return sorted(
        {
            round(index * (count - 1) / (limit - 1))
            for index in range(limit)
        }
    )


def _movement_form(form: str, movement_index: int) -> str:
    normalized = form.upper()
    if normalized in {"WTC", "PRELUDE_FUGUE"}:
        return "PRELUDE" if movement_index % 2 == 0 else "FUGUE"
    return normalized


def build_development_plan(
    *,
    form: str,
    measures: int,
    voice_count: int,
    key_pc: int,
    mode: int,
) -> list[DevelopmentStep]:
    normalized = form.upper()
    if normalized == "SINFONIA":
        base = [
            ("SUBJECT_ENTRY", "SUBJECT", voice_count - 1),
            ("ANSWER_ENTRY", "ANSWER", max(0, voice_count - 2)),
            ("COUNTERSUBJECT", "COUNTERSUBJECT", 0),
            ("EPISODE", "EPISODE", None),
            ("SEQUENCE", "SEQUENCE_DOWN", None),
            ("SUBJECT_ENTRY", "INVERSION", 0),
            ("EPISODE", "SEQUENCE_UP", None),
            ("SUBJECT_ENTRY", "STRETTO", max(0, voice_count - 2)),
            ("CADENTIAL_PREP", "EPISODE", None),
            ("CADENCE", "CADENCE", None),
        ]
    elif normalized == "FUGUE":
        base: list[tuple[str, str, int | None]] = []
        for voice in range(voice_count):
            base.append(
                (
                    "SUBJECT_ENTRY" if voice % 2 == 0 else "ANSWER_ENTRY",
                    "SUBJECT" if voice % 2 == 0 else "ANSWER",
                    voice,
                )
            )
        base.extend(
            [
                ("COUNTERSUBJECT", "COUNTERSUBJECT", None),
                ("EPISODE", "EPISODE", None),
                ("SEQUENCE", "SEQUENCE_DOWN", None),
                ("SUBJECT_ENTRY", "STRETTO", None),
                ("CADENTIAL_PREP", "EPISODE", None),
                ("CADENCE", "CADENCE", None),
            ]
        )
    elif normalized in {"PARTITA", "SUITE"}:
        base = [
            ("OPENING", "OPENING_MOTIF", voice_count - 1),
            ("EPISODE", "BINARY_A", None),
            ("SEQUENCE", "SEQUENCE_UP", None),
            ("CADENTIAL_PREP", "BINARY_A", None),
            ("CADENCE", "CADENCE", None),
            ("EPISODE", "BINARY_B", None),
            ("SEQUENCE", "SEQUENCE_DOWN", None),
            ("EPISODE", "BINARY_B", None),
            ("SUBJECT_ENTRY", "RECAP", voice_count - 1),
            ("CADENCE", "CADENCE", None),
        ]
    else:
        base = [
            ("SUBJECT_ENTRY", "SUBJECT", min(1, voice_count - 1)),
            ("ANSWER_ENTRY", "ANSWER", 0),
            ("EPISODE", "EPISODE", None),
            ("SEQUENCE", "SEQUENCE_DOWN", None),
            ("SUBJECT_ENTRY", "RECAP", min(1, voice_count - 1)),
            ("CADENTIAL_PREP", "EPISODE", None),
            ("CADENCE", "CADENCE", None),
        ]
    plan: list[DevelopmentStep] = []
    for role, operation, entry_voice in _stretch_plan(base, measures):
        local_key = key_pc
        if key_pc < 12:
            if operation == "ANSWER":
                local_key = (key_pc + 7) % 12
            elif operation == "SEQUENCE_UP":
                local_key = (key_pc + 2) % 12
            elif operation == "SEQUENCE_DOWN":
                local_key = (key_pc + (10 if mode == 0 else 9)) % 12
        plan.append(DevelopmentStep(role, operation, entry_voice, local_key))
    return plan


def recompute_pair_rows(
    voice_row: list[list[int]],
    previous_voice_row: list[list[int]] | None,
    *,
    max_voices: int,
) -> list[list[list[int]]]:
    pair_row = [
        [[0] * len(PAIR_FIELD_NAMES) for _ in range(max_voices)]
        for _ in range(max_voices)
    ]
    active = [_active_pitch(voice_row[voice]) for voice in range(max_voices)]
    previous = [
        None if previous_voice_row is None else _active_pitch(previous_voice_row[voice])
        for voice in range(max_voices)
    ]
    for left in range(max_voices):
        for right in range(left + 1, max_voices):
            pair_row[left][right] = _pair_features(
                previous[left],
                previous[right],
                active[left],
                active[right],
                adjacent=right == left + 1,
            )
    return pair_row


def piece_to_canonical_score(piece: InstrumentalV6Piece) -> CanonicalScore:
    events: list[Event] = []
    active: list[tuple[int, int] | None] = [None] * piece.max_voices
    total_ticks = len(piece.global_rows) * piece.grid_ticks
    for row_index, voice_row in enumerate(piece.voice_rows):
        tick = row_index * piece.grid_ticks
        for voice in range(piece.max_voices):
            state, pitch = voice_row[voice][0], voice_row[voice][1]
            if state == STATE_NOTE and pitch > 0:
                if active[voice] is not None:
                    start, previous_pitch = active[voice]
                    if tick > start:
                        events.append(_event(start, tick - start, voice, previous_pitch, len(events)))
                active[voice] = (tick, pitch)
            elif state != STATE_HOLD and active[voice] is not None:
                start, previous_pitch = active[voice]
                if tick > start:
                    events.append(_event(start, tick - start, voice, previous_pitch, len(events)))
                active[voice] = None
    for voice, current in enumerate(active):
        if current is None:
            continue
        start, pitch = current
        if total_ticks > start:
            events.append(_event(start, total_ticks - start, voice, pitch, len(events)))
    events.sort(key=lambda event: (event.start_tick, event.voice_id, event.id))
    measure_count = max(1, (total_ticks + piece.bar_len_ticks - 1) // piece.bar_len_ticks)
    measures = [
        Measure(id=f"m{index}", index=index, start_tick=index * piece.bar_len_ticks, length_ticks=piece.bar_len_ticks)
        for index in range(measure_count)
    ]
    return CanonicalScore(
        header=ScoreHeader(
            tpq=piece.tpq,
            key_sig_map={0: piece.key} if piece.key else {},
            time_sig_map={0: piece.time_signature},
            tempo_map={0: 88},
        ),
        measures=measures,
        parts=[Part(PartInfo(id="P1", instrument="piano", midi_program=0), events=events)],
    )


def rows_to_piece(
    *,
    global_rows: list[list[int]],
    voice_rows: list[list[list[int]]],
    pair_rows: list[list[list[list[int]]]],
    template: InstrumentalV6Piece,
    piece_id: str,
) -> InstrumentalV6Piece:
    return InstrumentalV6Piece(
        piece_id=piece_id,
        source_path="generated",
        form=template.form,
        movement_index=template.movement_index,
        tpq=template.tpq,
        grid_ticks=template.grid_ticks,
        time_signature=template.time_signature,
        key=template.key,
        key_pc=template.key_pc,
        mode=template.mode,
        voice_count=template.voice_count,
        max_voices=template.max_voices,
        bar_len_ticks=template.bar_len_ticks,
        steps_per_bar=template.steps_per_bar,
        global_rows=global_rows,
        voice_rows=voice_rows,
        pair_rows=pair_rows,
    )


def decode_interval(value: int) -> int | None:
    if value <= 0:
        return None
    return max(-MAX_INTERVAL, min(MAX_INTERVAL, int(value) - MAX_INTERVAL - 1))


def meter_id(time_signature: str) -> int:
    return METER_TO_ID.get(time_signature, 0)


def scale_degree(pitch: int, key_pc: int, mode: int) -> int:
    if key_pc >= 12 or pitch <= 0:
        return 0
    relative = (pitch - key_pc) % 12
    return (_MINOR_SCALE if mode == 1 else _MAJOR_SCALE).get(relative, 8 + relative % 5)


def _build_piece(
    score: music21.stream.Score,
    *,
    path: Path,
    form: str,
    target_voices: int | None,
    max_voices: int,
    movement_index: int,
    start_ql: float,
    end_ql: float,
    time_signature: str,
    max_bars: int,
    normalize_key: bool,
) -> InstrumentalV6Piece:
    numerator, denominator = (int(value) for value in time_signature.split("/", 1))
    bar_len_ticks = int(round(numerator * (4.0 / denominator) * TPQ))
    steps_per_bar = bar_len_ticks // GRID_TICKS
    if bar_len_ticks <= 0 or steps_per_bar > MAX_STEPS_PER_BAR:
        raise ValueError("movement meter is outside v6 bounds")

    key = _key_at(score, start_ql)
    key_pc, mode = _key_context(key)
    transposition = 0
    if normalize_key and key_pc < 12:
        transposition = ((-key_pc + 6) % 12) - 6
        key_pc = 0
        key = "Cm" if mode == 1 else "C"

    start_tick = int(round(start_ql * TPQ))
    available_ticks = int(round((end_ql - start_ql) * TPQ))
    max_ticks = min(available_ticks, max_bars * bar_len_ticks)
    resolved_target_voices = target_voices
    if resolved_target_voices is None and form.upper() == "FUGUE":
        resolved_target_voices = _KNOWN_FUGUE_VOICE_COUNTS.get(path.stem.upper())
    assigned = _assign_movement_voices(
        score,
        start_tick=start_tick,
        max_ticks=max_ticks,
        target_voices=resolved_target_voices,
        max_voices=max_voices,
        transposition=transposition,
    )
    if len(assigned) < 2:
        raise ValueError("movement has fewer than two assigned voices")
    voice_count = min(max_voices, len(assigned))
    onset_maps = [
        {note_event.onset_tick: note_event for note_event in assigned.get(voice, [])}
        for voice in range(max_voices)
    ]
    last_tick = max(
        note_event.onset_tick + note_event.duration_tick
        for voice_events in assigned.values()
        for note_event in voice_events
    )
    total_ticks = min(
        max_ticks,
        max(bar_len_ticks, ((last_tick + bar_len_ticks - 1) // bar_len_ticks) * bar_len_ticks),
    )
    measures = max(1, total_ticks // bar_len_ticks)
    plan = build_development_plan(
        form=form,
        measures=measures,
        voice_count=voice_count,
        key_pc=key_pc,
        mode=mode,
    )
    previous_notes: list[int | None] = [None] * max_voices
    active_pitches: list[int | None] = [None] * max_voices
    active_until = [0] * max_voices
    global_rows: list[list[int]] = []
    voice_rows: list[list[list[int]]] = []
    pair_rows: list[list[list[list[int]]]] = []
    previous_voice_row: list[list[int]] | None = None

    for tick in range(0, total_ticks, GRID_TICKS):
        bar = min(MAX_BARS - 1, tick // bar_len_ticks)
        step = plan[min(bar, len(plan) - 1)]
        global_row = [
            bar,
            min(MAX_STEPS_PER_BAR - 1, (tick % bar_len_ticks) // GRID_TICKS),
            bar % PHRASE_BARS,
            int(step.role in {"CADENTIAL_PREP", "CADENCE"}),
            key_pc,
            mode,
            voice_count,
            FORM_TO_ID.get(form.upper(), 0),
            meter_id(time_signature),
            ROLE_TO_ID.get(step.role, 0),
            DEVELOPMENT_TO_ID.get(step.operation, 0),
            max_voices if step.entry_voice is None else min(step.entry_voice, max_voices - 1),
            step.local_key_pc,
        ]
        voice_row: list[list[int]] = []
        for voice in range(max_voices):
            if voice >= voice_count:
                voice_row.append([STATE_REST, 0, 0, 0, 0, 0])
                continue
            note_event = onset_maps[voice].get(tick)
            if note_event is not None:
                state = STATE_NOTE
                pitch = note_event.pitch
                melodic = _encode_interval(
                    None if previous_notes[voice] is None else pitch - previous_notes[voice]
                )
                duration = min(MAX_DURATION_STEPS, max(1, note_event.duration_tick // GRID_TICKS))
                tie = 0
                previous_notes[voice] = pitch
                active_pitches[voice] = pitch
                active_until[voice] = tick + note_event.duration_tick
            elif active_until[voice] > tick and active_pitches[voice] is not None:
                state = STATE_HOLD
                pitch = active_pitches[voice]
                melodic = 0
                duration = min(
                    MAX_DURATION_STEPS,
                    max(1, (active_until[voice] - tick) // GRID_TICKS),
                )
                tie = 1
            else:
                state = STATE_REST
                pitch = melodic = duration = tie = 0
                active_pitches[voice] = None
            voice_row.append([state, pitch, melodic, duration, tie, scale_degree(pitch, key_pc, mode)])
        pair_row = recompute_pair_rows(
            voice_row,
            previous_voice_row,
            max_voices=max_voices,
        )
        global_rows.append(global_row)
        voice_rows.append(voice_row)
        pair_rows.append(pair_row)
        previous_voice_row = voice_row

    return InstrumentalV6Piece(
        piece_id=f"{path.stem}_m{movement_index:02d}",
        source_path=str(path),
        form=form.upper(),
        movement_index=movement_index,
        tpq=TPQ,
        grid_ticks=GRID_TICKS,
        time_signature=time_signature,
        key=key,
        key_pc=key_pc,
        mode=mode,
        voice_count=voice_count,
        max_voices=max_voices,
        bar_len_ticks=bar_len_ticks,
        steps_per_bar=steps_per_bar,
        global_rows=global_rows,
        voice_rows=voice_rows,
        pair_rows=pair_rows,
    )


def _movement_boundaries(score: music21.stream.Score) -> list[tuple[float, float, str]]:
    if not score.parts:
        return []
    measures = list(score.parts[0].getElementsByClass(music21.stream.Measure))
    if not measures:
        return [(0.0, float(score.highestTime), _time_signature_at(score, 0.0))]
    boundaries: list[tuple[float, float, str]] = []
    start = float(measures[0].offset)
    meter = _measure_meter(measures[0]) or _time_signature_at(score, start)
    for index, measure in enumerate(measures):
        is_final = measure.rightBarline is not None and measure.rightBarline.type == "final"
        if not is_final and index != len(measures) - 1:
            continue
        end = float(measure.offset + measure.barDuration.quarterLength)
        if end > start:
            boundaries.append((start, end, meter))
        if index + 1 < len(measures):
            start = float(measures[index + 1].offset)
            meter = _measure_meter(measures[index + 1]) or _time_signature_at(score, start)
    return boundaries


def _assign_movement_voices(
    score: music21.stream.Score,
    *,
    start_tick: int,
    max_ticks: int,
    target_voices: int | None,
    max_voices: int,
    transposition: int,
) -> dict[int, list[eventizer.NoteEvent]]:
    part_events = [
        selected
        for part in score.parts
        if (
            selected := _movement_events(
                part,
                start_tick=start_tick,
                max_ticks=max_ticks,
                transposition=transposition,
            )
        )
    ]
    if target_voices is None:
        target_voices = _infer_voice_count(
            part_events,
            max_voices=max_voices,
            max_ticks=max_ticks,
        )
    if len(part_events) >= target_voices:
        return _rank_voice_lanes(part_events, target_voices=target_voices)

    if len(part_events) >= 2:
        base, remainder = divmod(target_voices, len(part_events))
        lanes: list[list[eventizer.NoteEvent]] = []
        for part_index, selected in enumerate(part_events):
            lane_count = base + int(part_index < remainder)
            if lane_count <= 1:
                lanes.append(selected)
                continue
            assigned = eventizer._assign_events_by_continuity(
                eventizer._collapse_unison_octaves(
                    selected,
                    max_octaves_per_pitch_class=lane_count,
                ),
                max_voices=lane_count,
            )
            lanes.extend(events for events in assigned.values() if events)
        if len(lanes) >= target_voices:
            return _rank_voice_lanes(lanes, target_voices=target_voices)

    raw_events = [
        note for selected in part_events for note in selected
    ]
    if not raw_events:
        return {}
    assigned = eventizer._assign_events_by_continuity(
        eventizer._collapse_unison_octaves(raw_events, max_octaves_per_pitch_class=2),
        max_voices=target_voices,
    )
    return _rank_voice_lanes(list(assigned.values()), target_voices=target_voices)


def _infer_voice_count(
    part_events: list[list[eventizer.NoteEvent]],
    *,
    max_voices: int,
    max_ticks: int,
) -> int:
    if len(part_events) >= 3:
        return min(max_voices, len(part_events))
    events = [note for selected in part_events for note in selected]
    if not events:
        return 2
    sounding_counts: list[int] = []
    for tick in range(0, max_ticks, GRID_TICKS):
        sounding = {
            note.pitch
            for note in events
            if note.onset_tick <= tick < note.onset_tick + note.duration_tick
        }
        if sounding:
            sounding_counts.append(len(sounding))
    if not sounding_counts:
        return min(max_voices, max(2, len(part_events)))
    sounding_counts.sort()
    percentile_index = round(0.9 * (len(sounding_counts) - 1))
    inferred = sounding_counts[percentile_index]
    return min(max_voices, max(2, inferred))


def _rank_voice_lanes(
    lanes: list[list[eventizer.NoteEvent]],
    *,
    target_voices: int,
) -> dict[int, list[eventizer.NoteEvent]]:
    ranked = sorted(
        (events for events in lanes if events),
        key=lambda events: (
            sum(note.pitch * note.duration_tick for note in events)
            / max(1, sum(note.duration_tick for note in events)),
            -sum(note.duration_tick for note in events),
        ),
    )
    if len(ranked) > target_voices:
        indices = eventizer._select_spread_indices(len(ranked), target_voices)
        ranked = [ranked[index] for index in indices]
    return {
        voice: _monophonize_lane(events, voice=voice, voice_count=len(ranked))
        for voice, events in enumerate(ranked)
    }


def _monophonize_lane(
    events: list[eventizer.NoteEvent],
    *,
    voice: int,
    voice_count: int,
) -> list[eventizer.NoteEvent]:
    by_onset: dict[int, list[eventizer.NoteEvent]] = {}
    for note in events:
        by_onset.setdefault(note.onset_tick, []).append(note)
    selected: list[eventizer.NoteEvent] = []
    for onset in sorted(by_onset):
        candidates = sorted(by_onset[onset], key=lambda note: note.pitch)
        if len(candidates) == 1:
            selected.append(candidates[0])
            continue
        position = 0.0 if voice_count <= 1 else voice / (voice_count - 1)
        index = round(position * (len(candidates) - 1))
        selected.append(candidates[index])
    return selected


def _movement_events(
    part: music21.stream.Part,
    *,
    start_tick: int,
    max_ticks: int,
    transposition: int,
) -> list[eventizer.NoteEvent]:
    selected: list[eventizer.NoteEvent] = []
    for note_event in eventizer._extract_events(part, TPQ):
        if not start_tick <= note_event.onset_tick < start_tick + max_ticks:
            continue
        selected.append(
            eventizer.NoteEvent(
                onset_tick=_snap_tick(note_event.onset_tick - start_tick),
                duration_tick=max(GRID_TICKS, _snap_tick(note_event.duration_tick)),
                pitch=max(1, min(128, note_event.pitch + transposition)),
            )
        )
    return selected


def _pair_features(
    previous_left: int | None,
    previous_right: int | None,
    left: int | None,
    right: int | None,
    *,
    adjacent: bool,
) -> list[int]:
    if left is None or right is None:
        return [0, 12, 0, 0, 0, 0, 0, 0]
    distance = abs(right - left)
    interval_class = distance % 12
    consonance = 1 if interval_class in _PERFECT_CLASSES else 2 if interval_class in _IMPERFECT_CLASSES else 3
    motion = "UNKNOWN"
    parallel = direct = 0
    if previous_left is not None and previous_right is not None:
        left_motion = left - previous_left
        right_motion = right - previous_right
        if left_motion == 0 and right_motion == 0:
            motion = "STATIC"
        elif left_motion == 0 or right_motion == 0:
            motion = "OBLIQUE"
        elif left_motion * right_motion < 0:
            motion = "CONTRARY"
        else:
            previous_class = abs(previous_right - previous_left) % 12
            motion = "PARALLEL" if previous_class == interval_class else "SIMILAR"
            parallel = int(
                previous_class in _PERFECT_CLASSES
                and interval_class == previous_class
                and left_motion != 0
                and right_motion != 0
            )
            direct = int(
                interval_class in _PERFECT_CLASSES
                and (abs(left_motion) > 2 or abs(right_motion) > 2)
            )
    return [
        min(MAX_VERTICAL_INTERVAL, distance) + 1,
        interval_class,
        consonance,
        MOTION_TO_ID[motion],
        parallel,
        direct,
        int(left >= right),
        int(adjacent and distance > 19),
    ]


def _active_pitch(voice: list[int]) -> int | None:
    return voice[1] if voice[0] in {STATE_NOTE, STATE_HOLD} and voice[1] > 0 else None


def _stretch_plan(
    base: Sequence[tuple[str, str, int | None]],
    measures: int,
) -> list[tuple[str, str, int | None]]:
    if measures <= 0:
        return []
    if measures <= len(base):
        out = list(base[:measures])
        out[-1] = ("CADENCE", "CADENCE", None)
        return out
    extension_length = measures - len(base)
    extension = [
        ("EPISODE", "EPISODE", None) if index % 2 == 0 else ("SEQUENCE", "SEQUENCE_DOWN", None)
        for index in range(extension_length)
    ]
    closing_start = max(0, len(base) - 2)
    for index in range(len(base) - 2, -1, -1):
        if base[index][1] in {"RECAP", "STRETTO"}:
            closing_start = index
            break
    return [*base[:closing_start], *extension, *base[closing_start:]]


def _measure_meter(measure: music21.stream.Measure) -> str | None:
    signatures = list(measure.getTimeSignatures(returnDefault=False))
    return signatures[0].ratioString if signatures else None


def _time_signature_at(score: music21.stream.Score, offset: float) -> str:
    signatures = [
        signature
        for signature in score.flatten().getElementsByClass(music21.meter.TimeSignature)
        if float(signature.offset) <= offset + 1e-6
    ]
    return signatures[-1].ratioString if signatures else "4/4"


def _key_at(score: music21.stream.Score, offset: float) -> str | None:
    keys = [
        key
        for key in score.flatten().getElementsByClass((music21.key.Key, music21.key.KeySignature))
        if float(key.offset) <= offset + 1e-6
    ]
    if not keys:
        detected = eventizer.detect_key_token(score, None)
        return None if detected is None else detected[4:]
    current = keys[-1]
    if isinstance(current, music21.key.KeySignature):
        current = current.asKey()
    tonic = current.tonic.name.replace("-", "b")
    return f"{tonic}m" if current.mode == "minor" else tonic


def _key_context(key: str | None) -> tuple[int, int]:
    if not key:
        return 12, 2
    mode = 1 if key.endswith("m") else 0
    tonic = key[:-1] if mode else key
    try:
        return int(music21.pitch.Pitch(tonic.replace("b", "-")).pitchClass), mode
    except Exception:
        return 12, 2


def _encode_interval(delta: int | None) -> int:
    if delta is None:
        return 0
    return max(-MAX_INTERVAL, min(MAX_INTERVAL, int(delta))) + MAX_INTERVAL + 1


def _snap_tick(value: int) -> int:
    return int(round(value / GRID_TICKS) * GRID_TICKS)


def _event(start: int, duration: int, voice: int, pitch: int, index: int) -> Event:
    velocity = min(92, 68 + voice * 4)
    return Event(
        id=f"v6-n{index}",
        start_tick=start,
        dur_tick=max(1, duration),
        voice_id=voice,
        pitch_midi=max(0, min(127, pitch)),
        velocity=velocity,
    )
