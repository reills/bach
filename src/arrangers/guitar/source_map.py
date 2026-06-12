from __future__ import annotations

from dataclasses import asdict, dataclass

from src.api.canonical import GuitarFingering


@dataclass(frozen=True)
class PianoToGuitarNoteMap:
    source_event_id: str
    target_event_id: str | None
    start_tick: int
    dur_tick: int
    voice_id: int
    source_pitch_midi: int
    target_pitch_midi: int | None
    semitone_shift: int
    dropped: bool = False
    fingering: GuitarFingering | None = None


@dataclass(frozen=True)
class PianoToGuitarSourceMap:
    notes: list[PianoToGuitarNoteMap]

    def to_dict(self) -> dict[str, object]:
        return {
            "notes": [asdict(note_map) for note_map in self.notes],
        }
