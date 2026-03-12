from dataclasses import dataclass, replace
from itertools import product
from typing import Sequence

from src.api.canonical.types import Event, GuitarFingering

STANDARD_GUITAR_TUNING = (40, 45, 50, 55, 59, 64)
DEFAULT_MAX_FRET = 20


@dataclass(frozen=True)
class TabNote:
    pitch_midi: int
    onset_tick: int
    dur_tick: int
    voice_id: int
    note_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.pitch_midi, int) or isinstance(self.pitch_midi, bool):
            raise ValueError("pitch_midi must be an integer")
        if not 0 <= self.pitch_midi <= 127:
            raise ValueError("pitch_midi must be a valid MIDI value")
        if not isinstance(self.onset_tick, int) or isinstance(self.onset_tick, bool):
            raise ValueError("onset_tick must be an integer")
        if self.onset_tick < 0:
            raise ValueError("onset_tick must be non-negative")
        if not isinstance(self.dur_tick, int) or isinstance(self.dur_tick, bool):
            raise ValueError("dur_tick must be an integer")
        if self.dur_tick <= 0:
            raise ValueError("dur_tick must be positive")
        if not isinstance(self.voice_id, int) or isinstance(self.voice_id, bool):
            raise ValueError("voice_id must be an integer")
        if self.voice_id < 0:
            raise ValueError("voice_id must be non-negative")


@dataclass(frozen=True)
class AssignedTabNote:
    pitch_midi: int
    onset_tick: int
    dur_tick: int
    voice_id: int
    fingering: GuitarFingering
    note_id: str | None = None


@dataclass(frozen=True)
class _IndexedNote:
    index: int
    pitch_midi: int
    onset_tick: int
    dur_tick: int
    voice_id: int


def tab_events(
    events: Sequence[Event],
    tuning: Sequence[int] = STANDARD_GUITAR_TUNING,
    max_fret: int = DEFAULT_MAX_FRET,
) -> list[Event]:
    normalized_tuning = _normalize_tuning(tuning)
    _validate_max_fret(max_fret)

    indexed_notes: list[_IndexedNote] = []
    tabbed_events: list[Event | None] = [None] * len(events)
    for index, event in enumerate(events):
        if event.pitch_midi is None:
            tabbed_events[index] = event
            continue
        indexed_notes.append(
            _IndexedNote(
                index=index,
                pitch_midi=event.pitch_midi,
                onset_tick=event.start_tick,
                dur_tick=event.dur_tick,
                voice_id=event.voice_id,
            )
        )

    assignments = _assign_fingerings(indexed_notes, normalized_tuning, max_fret)
    for indexed_note, fingering in assignments.items():
        tabbed_events[indexed_note.index] = replace(events[indexed_note.index], fingering=fingering)

    return [event for event in tabbed_events if event is not None]


def tab_notes(
    notes: Sequence[TabNote],
    tuning: Sequence[int] = STANDARD_GUITAR_TUNING,
    max_fret: int = DEFAULT_MAX_FRET,
) -> list[AssignedTabNote]:
    normalized_tuning = _normalize_tuning(tuning)
    _validate_max_fret(max_fret)

    indexed_notes = [
        _IndexedNote(
            index=index,
            pitch_midi=note.pitch_midi,
            onset_tick=note.onset_tick,
            dur_tick=note.dur_tick,
            voice_id=note.voice_id,
        )
        for index, note in enumerate(notes)
    ]
    assignments = _assign_fingerings(indexed_notes, normalized_tuning, max_fret)

    return [
        AssignedTabNote(
            pitch_midi=note.pitch_midi,
            onset_tick=note.onset_tick,
            dur_tick=note.dur_tick,
            voice_id=note.voice_id,
            note_id=note.note_id,
            fingering=assignments[_IndexedNote(index, note.pitch_midi, note.onset_tick, note.dur_tick, note.voice_id)],
        )
        for index, note in enumerate(notes)
    ]


def alternate_fingerings_for_event(
    event: Event,
    tuning: Sequence[int] = STANDARD_GUITAR_TUNING,
    max_fret: int = DEFAULT_MAX_FRET,
) -> list[GuitarFingering]:
    if event.pitch_midi is None:
        raise ValueError("alternate fingerings require a pitched event")

    normalized_tuning = _normalize_tuning(tuning)
    _validate_max_fret(max_fret)
    return _candidate_fingerings(event.pitch_midi, normalized_tuning, max_fret)


def _assign_fingerings(
    notes: Sequence[_IndexedNote],
    tuning: tuple[int, ...],
    max_fret: int,
) -> dict[_IndexedNote, GuitarFingering]:
    assignments: dict[_IndexedNote, GuitarFingering] = {}
    previous_by_voice: dict[int, GuitarFingering] = {}
    notes_by_onset: dict[int, list[_IndexedNote]] = {}

    for note in notes:
        notes_by_onset.setdefault(note.onset_tick, []).append(note)

    for onset_tick in sorted(notes_by_onset):
        onset_notes = notes_by_onset[onset_tick]
        if len(onset_notes) > len(tuning):
            raise ValueError(f"no playable guitar voicing at onset {onset_tick}: more notes than strings")

        candidate_lists = [_candidate_fingerings(note.pitch_midi, tuning, max_fret) for note in onset_notes]
        if any(not candidates for candidates in candidate_lists):
            raise ValueError(f"no playable guitar voicing at onset {onset_tick}: note is outside the fret range")

        best_combo: tuple[GuitarFingering, ...] | None = None
        best_cost: tuple[int, int, int, int, tuple[int, ...]] | None = None

        # On each onset, brute-force the small candidate grid and keep only unique-string voicings.
        for combo in product(*candidate_lists):
            strings = {fingering.string_index for fingering in combo}
            if len(strings) != len(combo):
                continue
            cost = _combo_cost(onset_notes, combo, previous_by_voice)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_combo = combo

        if best_combo is None:
            raise ValueError(f"no playable guitar voicing at onset {onset_tick}: duplicate string use required")

        for note, fingering in zip(onset_notes, best_combo):
            assignments[note] = fingering
            previous_by_voice[note.voice_id] = fingering

    return assignments


def _candidate_fingerings(pitch_midi: int, tuning: tuple[int, ...], max_fret: int) -> list[GuitarFingering]:
    candidates = [
        GuitarFingering(string_index=string_index, fret=pitch_midi - open_pitch)
        for string_index, open_pitch in enumerate(tuning)
        if 0 <= pitch_midi - open_pitch <= max_fret
    ]
    return sorted(candidates, key=lambda fingering: (fingering.fret, -fingering.string_index))


def _combo_cost(
    notes: Sequence[_IndexedNote],
    combo: Sequence[GuitarFingering],
    previous_by_voice: dict[int, GuitarFingering],
) -> tuple[int, int, int, int, tuple[int, ...]]:
    fret_movement = 0
    string_movement = 0
    total_fret = 0
    open_strings = 0

    for note, fingering in zip(notes, combo):
        total_fret += fingering.fret
        if fingering.fret == 0:
            open_strings += 1
        previous = previous_by_voice.get(note.voice_id)
        if previous is None:
            continue
        fret_movement += abs(previous.fret - fingering.fret)
        string_movement += abs(previous.string_index - fingering.string_index)

    return (
        fret_movement,
        string_movement,
        total_fret,
        -open_strings,
        tuple(fingering.string_index for fingering in combo),
    )


def _normalize_tuning(tuning: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(tuning)
    if not normalized:
        raise ValueError("tuning must define at least one string")
    for pitch in normalized:
        if not isinstance(pitch, int) or isinstance(pitch, bool):
            raise ValueError("tuning pitches must be integers")
        if not 0 <= pitch <= 127:
            raise ValueError("tuning pitches must be valid MIDI values")
    return normalized


def _validate_max_fret(max_fret: int) -> None:
    if not isinstance(max_fret, int) or isinstance(max_fret, bool):
        raise ValueError("max_fret must be an integer")
    if max_fret < 0:
        raise ValueError("max_fret must be non-negative")
