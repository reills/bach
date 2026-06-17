from collections import defaultdict
from collections.abc import Sequence

from src.api.canonical.types import Event
from src.tabber.heuristic import STANDARD_GUITAR_TUNING

_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def render_ascii_tab(
    events: Sequence[Event],
    tuning: Sequence[int] = STANDARD_GUITAR_TUNING,
) -> str:
    normalized_tuning = _normalize_tuning(tuning)
    string_labels = _string_labels(normalized_tuning)
    rows = {string_index: f"{label}|" for string_index, label in zip(reversed(range(len(normalized_tuning))), string_labels)}

    note_events = sorted(
        (event for event in events if event.pitch_midi is not None),
        key=lambda event: (event.start_tick, event.fingering.string_index if event.fingering is not None else -1, event.id),
    )
    onsets: dict[int, list[Event]] = defaultdict(list)
    for event in note_events:
        if event.fingering is None:
            raise ValueError("pitched events must include fingering data")
        if event.fingering.string_index >= len(normalized_tuning):
            raise ValueError("fingering string_index is outside the tuning")
        onsets[event.start_tick].append(event)

    for onset_tick in sorted(onsets):
        onset_events = onsets[onset_tick]
        by_string: dict[int, str] = {}
        for event in onset_events:
            string_index = event.fingering.string_index
            if string_index in by_string:
                raise ValueError(f"multiple notes share string {string_index} at onset {onset_tick}")
            by_string[string_index] = str(event.fingering.fret)

        slot_width = max((len(fret_text) for fret_text in by_string.values()), default=1)
        for string_index in reversed(range(len(normalized_tuning))):
            fret_text = by_string.get(string_index)
            if fret_text is None:
                rows[string_index] += "-" * slot_width
            else:
                rows[string_index] += fret_text.ljust(slot_width, "-")
            rows[string_index] += "-"

    return "\n".join(f"{rows[string_index]}|" for string_index in reversed(range(len(normalized_tuning))))


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


def _string_labels(tuning: tuple[int, ...]) -> list[str]:
    labels = [_NOTE_NAMES[pitch % 12] for pitch in reversed(tuning)]
    if len(labels) >= 2 and labels[0] == labels[-1]:
        labels[0] = labels[0].lower()
    return labels
